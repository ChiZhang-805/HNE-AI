"""Unified HNE temporal/reference/relational event model.

The development comparison motivates retaining ordered observations alongside
personal-reference summaries. The same shared GRUs, context modulation,
variable attention and invariant readout serve every dataset. No channel-ID
embedding or channel flattening enters the risk head.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .hne_set import _metrics, _person_weights, _scaler, _masked_pool


WIDTH = 32
MAX_EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 128
WEIGHT_DECAY = 0.001
LEARNING_RATES = (0.001, 0.0003)


def _device(value="auto"):
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available()
                        else "cpu" if value == "auto" else value)


def _scalar_scaler(values, origin_weights):
    # One common location/scale across state nodes, another across contexts.
    # No sensor index receives its own globally fitted normalization parameter.
    return _scaler(np.asarray(values).reshape(len(values), -1, 1), origin_weights)


class HNETemporal(nn.Module):
    def __init__(self, groups, history_steps):
        super().__init__()
        self.groups = np.asarray(groups).astype(str).tolist()
        self.history_steps = int(history_steps)
        self.patch_size = math.ceil(self.history_steps / 32)
        relation = np.flatnonzero(np.isin(self.groups, ["N", "NE"]))
        if len(relation) != 72:
            raise ValueError("Expected the unified 72-coordinate N/NE descriptor schema")
        self.register_buffer("relation_indices", torch.as_tensor(relation, dtype=torch.long))
        for prefix, size in (("node", 50), ("relation", 72), ("state", 1), ("context", 1)):
            self.register_buffer(prefix + "_mean", torch.zeros(size))
            self.register_buffer(prefix + "_scale", torch.ones(size))
            self.register_buffer(prefix + "_supported", torch.ones(size, dtype=torch.bool))
        self.register_buffer("prior_logit", torch.tensor(0.0))
        self.register_buffer("constant_logit", torch.tensor(float("nan")))
        self.state_patch = nn.Sequential(nn.Linear(2 * self.patch_size, WIDTH), nn.GELU())
        self.context_patch = nn.Sequential(nn.Linear(2 * self.patch_size, WIDTH), nn.GELU())
        self.state_gru = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.context_gru = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.node_reference = nn.Sequential(nn.Linear(100, WIDTH), nn.GELU(), nn.LayerNorm(WIDTH))
        self.node_fusion = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.GELU(), nn.LayerNorm(WIDTH))
        self.context_fusion = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.GELU(), nn.LayerNorm(WIDTH))
        self.modulate = nn.Linear(WIDTH, 2 * WIDTH)
        self.query = nn.Linear(WIDTH, WIDTH, bias=False)
        self.key = nn.Linear(WIDTH, WIDTH, bias=False)
        self.value = nn.Linear(WIDTH, WIDTH, bias=False)
        self.graph_norm = nn.LayerNorm(WIDTH)
        self.relation = nn.Sequential(nn.Linear(144, WIDTH), nn.GELU(), nn.LayerNorm(WIDTH))
        self.risk = nn.Sequential(nn.Linear(4 * WIDTH + 3, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))

    def fit_scalers(self, windows, nodes, pooled, origin_weights):
        p = nodes.shape[1]
        blocks = {"node": _scaler(nodes, origin_weights),
                  "relation": _scaler(pooled[:, self.relation_indices.cpu().numpy()], origin_weights),
                  "state": _scalar_scaler(windows[:, :, :p], origin_weights),
                  "context": _scalar_scaler(windows[:, :, p:], origin_weights)}
        for prefix, values in blocks.items():
            for suffix, value in zip(("mean", "scale", "supported"), values):
                target = getattr(self, prefix + "_" + suffix)
                target.copy_(torch.as_tensor(value, dtype=target.dtype, device=target.device))

    def _scaled(self, values, prefix):
        observed = torch.isfinite(values) & getattr(self, prefix + "_supported")
        mean = getattr(self, prefix + "_mean")
        filled = torch.where(observed, values, mean)
        scaled = ((filled - mean) / getattr(self, prefix + "_scale")).clamp(-8.0, 8.0)
        return scaled, observed

    def _temporal(self, values, prefix):
        scaled, observed = self._scaled(values, prefix)
        batch, steps, channels = scaled.shape
        if math.ceil(steps / 32) != self.patch_size:
            raise ValueError("A frozen model requires the same patch input width; retrain when changing sampling adapters")
        # Left padding adds unavailable history rather than a spurious future.
        pad = (-steps) % self.patch_size
        scaled = scaled.transpose(1, 2)
        mask = observed.transpose(1, 2).to(scaled.dtype)
        if pad:
            scaled, mask = F.pad(scaled, (pad, 0)), F.pad(mask, (pad, 0))
        patches = scaled.reshape(batch * channels, -1, self.patch_size)
        patch_masks = mask.reshape(batch * channels, -1, self.patch_size)
        token = getattr(self, prefix + "_patch")(torch.cat((patches, patch_masks), -1))
        _, hidden = getattr(self, prefix + "_gru")(token)
        return hidden[-1].reshape(batch, channels, WIDTH), observed

    def forward(self, windows, nodes, pooled, correct_prior=True, groups=None):
        if torch.isfinite(self.constant_logit):
            return self.constant_logit.expand(len(windows))
        p = nodes.shape[1]
        temporal, state_observed = self._temporal(windows[:, :, :p], "state")
        environment, context_observed = self._temporal(windows[:, :, p:], "context")
        reference, node_observed = self._scaled(nodes, "node")
        reference = self.node_reference(torch.cat((reference, node_observed.to(reference.dtype)), -1))
        state = self.node_fusion(torch.cat((temporal, reference), -1))
        node_present = state_observed.any(1) | node_observed[..., :48].any(-1)
        context_present = context_observed.any(1)
        context = self.context_fusion(_masked_pool(environment, context_present))
        gain, shift = self.modulate(context).chunk(2, -1)
        state = state * (2 * torch.sigmoid(gain[:, None])) + torch.tanh(shift[:, None])
        scores = self.query(state) @ self.key(state).transpose(1, 2) / math.sqrt(WIDTH)
        scores = scores.masked_fill(~node_present[:, None], -torch.inf)
        # A completely absent node set has no messages and no NaN softmax.
        scores = torch.where(node_present.any(1)[:, None, None], scores, torch.zeros_like(scores))
        attention = torch.softmax(scores, -1) * node_present[:, None].to(scores.dtype)
        state = self.graph_norm(state + attention @ self.value(state))
        if groups is None:
            relation_indices = self.relation_indices
        else:
            indices = np.flatnonzero(np.isin(np.asarray(groups).astype(str), ["N", "NE"]))
            if len(indices) != 72:
                raise ValueError("Transfer must preserve the unified 72-coordinate relation schema")
            relation_indices = torch.as_tensor(indices, dtype=torch.long, device=pooled.device)
        relation, relation_observed = self._scaled(pooled.index_select(1, relation_indices), "relation")
        relation = self.relation(torch.cat((relation, relation_observed.to(relation.dtype)), -1))
        support = torch.stack((state_observed.float().mean((1, 2)),
                               context_observed.float().mean((1, 2)),
                               relation_observed.float().mean(1)), 1)
        logit = self.risk(torch.cat((_masked_pool(state, node_present), context, relation, support), 1)).squeeze(-1)
        return logit + self.prior_logit if correct_prior else logit


def _arrays(windows, nodes, pooled, groups, name):
    windows, nodes, pooled = [np.ascontiguousarray(a, dtype=np.float32) for a in (windows, nodes, pooled)]
    if nodes.ndim != 3 or nodes.shape[-1] != 50 or nodes.shape[1] < 1:
        raise ValueError(f"{name}: nodes must be (origins, P, 50)")
    if windows.ndim != 3 or windows.shape[0] != len(nodes) or windows.shape[2] <= nodes.shape[1]:
        raise ValueError(f"{name}: windows must contain P state and Q>=1 context channels")
    if pooled.shape != (len(nodes), len(groups)):
        raise ValueError(f"{name}: pooled descriptor width does not match groups")
    return windows, nodes, pooled


def predict_proba(model, windows, nodes, pooled, groups=None, batch_size=256):
    groups = model.groups if groups is None else groups
    windows, nodes, pooled = _arrays(windows, nodes, pooled, groups, "prediction")
    if not len(windows):
        return np.empty(0, dtype=np.float64)
    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            tensors = [torch.as_tensor(a[start:start + batch_size], device=device)
                       for a in (windows, nodes, pooled)]
            result.append(torch.sigmoid(model(*tensors, groups=groups)).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def load_model(path, device="auto"):
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = HNETemporal(checkpoint["groups"], checkpoint["history_steps"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(_device(device)).eval()
    return model, checkpoint["metadata"]


def fit_predict(train, validation, test, groups, seed, learning_rate, output,
                device="auto", run_fingerprint=None):
    """Bundles contain windows/nodes/pooled; train/validation also y and person.

    The test bundle has no outcomes. This routine cannot select an epoch or a
    learning rate from a held-out label.
    """
    if learning_rate not in LEARNING_RATES:
        raise ValueError("Only the two prespecified learning rates are allowed")
    started = time.time()
    train_arrays = _arrays(*(train[k] for k in ("windows", "nodes", "pooled")), groups, "training")
    val_arrays = _arrays(*(validation[k] for k in ("windows", "nodes", "pooled")), groups, "validation")
    test_arrays = _arrays(*(test[k] for k in ("windows", "nodes", "pooled")), groups, "test")
    y = np.asarray(train["y"], dtype=np.float32)
    vy = np.asarray(validation["y"], dtype=np.float32)
    people, vpeople = np.asarray(train["person"]), np.asarray(validation["person"])
    for labels, persons, arrays, name in ((y, people, train_arrays, "training"),
                                         (vy, vpeople, val_arrays, "validation")):
        if not len(labels) or labels.shape != (len(arrays[0]),) or persons.shape != labels.shape:
            raise ValueError(f"{name}: known outcomes and people are required per origin")
        if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{name}: binary known outcomes only")
    torch.manual_seed(int(seed))
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = _device(device)
    base = _person_weights(people)
    val_weights = _person_weights(vpeople)
    prior = float(base @ y)
    correction = math.log(np.clip(prior, 1e-6, 1 - 1e-6) / np.clip(1 - prior, 1e-6, 1 - 1e-6))
    model = HNETemporal(groups, train_arrays[0].shape[1])
    model.fit_scalers(*train_arrays, base)
    model.prior_logit.fill_(correction)
    model.to(device)
    info = dict(model="HNE-temporal", width=WIDTH, patch_size=model.patch_size,
                history_steps=model.history_steps, state_nodes=train_arrays[1].shape[1],
                context_nodes=train_arrays[0].shape[2] - train_arrays[1].shape[1],
                parameters=sum(p.numel() for p in model.parameters()), seed=int(seed),
                learning_rate=float(learning_rate), weight_decay=WEIGHT_DECAY,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                device=str(device), train_origins=len(y), validation_origins=len(vy),
                test_origins=len(test_arrays[0]), train_people=len(np.unique(people)),
                validation_people=len(np.unique(vpeople)), training_person_weighted_prevalence=prior,
                logit_prior_correction=correction, run_fingerprint=run_fingerprint,
                selection_metric="validation_auc" if len(np.unique(vy)) == 2 else "unbalanced_validation_logloss",
                normalization="training-person-weighted shared node descriptors; one shared scale for state trajectories and one for context trajectories; separate masks",
                architecture="shared state/context GRU32; node-reference fusion; context FiLM; masked variable attention; invariant node/context pools; N/NE summary head; BCE",
                history=[])
    if len(np.unique(y)) == 1:
        model.constant_logit.fill_(correction)
        info.update(status="constant_single_training_class", best_epoch=0, epochs_run=0)
    else:
        class_mass = np.where(y > 0, prior, 1 - prior)
        weights = base / (2 * class_mass)
        weights *= len(y) / weights.sum()
        arrays_t = [torch.as_tensor(a, device=device) for a in train_arrays]
        y_t = torch.as_tensor(y, device=device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
        generator = torch.Generator().manual_seed(int(seed))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        best_score, best_state, best_epoch, stale = -math.inf, None, 0, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            permutation = torch.randperm(len(y), generator=generator)
            training_loss = 0.0
            for indices in permutation.split(BATCH_SIZE):
                indices = indices.to(device)
                optimizer.zero_grad(set_to_none=True)
                logit = model(*(a[indices] for a in arrays_t), correct_prior=False)
                loss = (F.binary_cross_entropy_with_logits(logit, y_t[indices], reduction="none") * weights_t[indices]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite HNE-temporal training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_loss += float(loss.detach()) * len(indices) / len(y)
            probability = predict_proba(model, *val_arrays)
            metrics = _metrics(vy, probability, val_weights)
            score = metrics["auc"] if metrics["auc"] is not None else -metrics["logloss"]
            info["history"].append(dict(epoch=epoch, balanced_training_loss=training_loss,
                                        validation_auc=metrics["auc"], validation_logloss=metrics["logloss"]))
            if score > best_score + 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= PATIENCE:
                break
        model.load_state_dict(best_state)
        info.update(status="trained", best_epoch=best_epoch, epochs_run=len(info["history"]))
    val_probability = predict_proba(model, *val_arrays)
    test_probability = predict_proba(model, *test_arrays)
    info["validation"] = _metrics(vy, val_probability, val_weights)
    info["fit_seconds"] = time.time() - started
    if output is not None:
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(format_version=1, groups=model.groups, history_steps=model.history_steps,
                          state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, metadata=info)
        torch.save(checkpoint, folder / "model.pt")
        np.savez_compressed(folder / "predictions.npz", val_probability=val_probability,
                            test_probability=test_probability)
        (folder / "metadata.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    return dict(val_probability=val_probability, test_probability=test_probability, metadata=info)
