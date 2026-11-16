"""Small permutation-invariant relational set head for prospective HNE risk.

The input contains shared node descriptors and already pooled relation changes;
this model does not implement per-edge message passing. All fitted descriptor
statistics use training people only. Class balancing is undone on the logit
scale before probability evaluation or prediction.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn
from torch.nn import functional as F


WIDTH = 16
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def _person_weights(person):
    person = np.asarray(person).astype(str)
    _, inverse, count = np.unique(person, return_inverse=True, return_counts=True)
    weights = 1.0 / count[inverse]
    return weights / weights.sum()


def _scaler(values, origin_weights):
    """One statistic per descriptor, shared across every node identity."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        weights = np.broadcast_to(
            origin_weights[:, None] / values.shape[1], values.shape[:2]
        ).reshape(-1)
        values = values.reshape(-1, values.shape[-1])
    else:
        weights = origin_weights
    observed = np.isfinite(values)
    mass = (observed * weights[:, None]).sum(0)
    safe = np.where(observed, values, 0.0)
    mean = (safe * weights[:, None]).sum(0) / np.maximum(mass, 1e-12)
    variance = (
        np.where(observed, (safe - mean) ** 2, 0.0) * weights[:, None]
    ).sum(0) / np.maximum(mass, 1e-12)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32), mass > 0


def _mlp(inputs, outputs=WIDTH):
    return nn.Sequential(nn.Linear(inputs, WIDTH), nn.GELU(), nn.Linear(WIDTH, outputs))


def _group_indices(groups):
    labels = np.asarray(groups).astype(str)
    relation = np.flatnonzero(np.isin(labels, ["N", "NE"]))
    context_group = np.flatnonzero(labels == "E")
    # 26 adjusted/effect node descriptors x 4 invariant summaries, followed
    # by 24 dynamic descriptors per observed-context variable.
    context = context_group[104:]
    if not len(relation) or not len(context) or len(context) % 24:
        raise ValueError("Expected N/NE and E=(104 node summaries + 24 per context)")
    return relation, context


def _masked_pool(encoded, present):
    count = present.sum(1, keepdim=True)
    mean = (encoded * present[..., None]).sum(1) / count.clamp_min(1)
    maximum = encoded.masked_fill(~present[..., None], -torch.inf).amax(1)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), 1)


class RelationalSetHead(nn.Module):
    """Shared node encoder plus separate relation and observed-context heads."""

    def __init__(self, groups, width=WIDTH):
        super().__init__()
        if width != WIDTH:
            raise ValueError("The prespecified relational set head has width 16")
        self.groups = np.asarray(groups).astype(str).tolist()
        relation, context = _group_indices(self.groups)
        self.register_buffer("relation_indices", torch.as_tensor(relation, dtype=torch.long))
        self.register_buffer("context_indices", torch.as_tensor(context, dtype=torch.long))
        for prefix, size in (("node", 50), ("relation", len(relation)), ("context", 24)):
            self.register_buffer(prefix + "_mean", torch.zeros(size))
            self.register_buffer(prefix + "_scale", torch.ones(size))
            self.register_buffer(prefix + "_supported", torch.ones(size, dtype=torch.bool))
        self.register_buffer("prior_logit", torch.tensor(0.0))
        self.register_buffer("constant_logit", torch.tensor(float("nan")))
        self.node_encoder = _mlp(100)
        self.relation_encoder = _mlp(2 * len(relation))
        self.context_encoder = _mlp(48)
        self.context_pool = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.GELU())
        # Node mean/max, relation/context embeddings, three support fractions.
        self.risk = _mlp(4 * WIDTH + 3, 1)

    def fit_scalers(self, nodes, pooled, weights):
        arrays = {
            "node": nodes,
            "relation": pooled[:, self.relation_indices.cpu().numpy()],
            "context": pooled[:, self.context_indices.cpu().numpy()].reshape(len(nodes), -1, 24),
        }
        for prefix, values in arrays.items():
            mean, scale, supported = _scaler(values, weights)
            for suffix, value in (("mean", mean), ("scale", scale), ("supported", supported)):
                buffer = getattr(self, prefix + "_" + suffix)
                buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device))

    def _encode_input(self, values, prefix):
        mask = torch.isfinite(values) & getattr(self, prefix + "_supported")
        centered = torch.where(mask, values, getattr(self, prefix + "_mean"))
        scaled = ((centered - getattr(self, prefix + "_mean")) /
                  getattr(self, prefix + "_scale")).clamp(-8.0, 8.0)
        return torch.cat((scaled, mask.to(scaled.dtype)), -1), mask

    def forward(self, nodes, pooled, correct_prior=True, groups=None):
        if torch.isfinite(self.constant_logit):
            return self.constant_logit.expand(nodes.shape[0])
        node_input, node_mask = self._encode_input(nodes, "node")
        # Reference context-response coefficients alone do not make a recently
        # unobserved/padded state node an observed history.
        node_present = node_mask[..., :48].any(-1)
        encoded = self.node_encoder(node_input)
        node_pool = _masked_pool(encoded, node_present)
        if groups is None:
            relation_indices, context_indices = self.relation_indices, self.context_indices
        else:
            relation, context = _group_indices(groups)
            if len(relation) != len(self.relation_indices):
                raise ValueError("Transfer must preserve the N/NE descriptor schema")
            relation_indices = torch.as_tensor(relation, dtype=torch.long, device=pooled.device)
            context_indices = torch.as_tensor(context, dtype=torch.long, device=pooled.device)
        relation_input, relation_mask = self._encode_input(
            pooled.index_select(1, relation_indices), "relation")
        context_input, context_mask = self._encode_input(
            pooled.index_select(1, context_indices).reshape(len(nodes), -1, 24), "context")
        context_pool = self.context_pool(_masked_pool(
            self.context_encoder(context_input), context_mask.any(-1)))
        support = torch.stack((node_present.float().mean(1),
                               relation_mask.float().mean(1),
                               context_mask.float().mean((1, 2))), 1)
        joined = torch.cat((node_pool, self.relation_encoder(relation_input),
                            context_pool, support), 1)
        logit = self.risk(joined).squeeze(-1)
        return logit + self.prior_logit if correct_prior else logit


def _arrays(nodes, pooled, groups, name):
    nodes = np.asarray(nodes, dtype=np.float32)
    pooled = np.asarray(pooled, dtype=np.float32)
    if nodes.ndim != 3 or nodes.shape[1] < 1 or nodes.shape[2] != 50:
        raise ValueError(f"{name}: expected nodes shaped (origins, variable nodes, 50)")
    if pooled.shape != (len(nodes), len(groups)):
        raise ValueError(f"{name}: pooled shape does not match origins/group descriptors")
    return np.ascontiguousarray(nodes), np.ascontiguousarray(pooled)


def _labels(y, person, count, name):
    y = np.asarray(y, dtype=np.float32)
    person = np.asarray(person)
    if y.shape != (count,) or person.shape != (count,):
        raise ValueError(f"{name}: labels and people must have one entry per origin")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name}: only known binary outcomes are allowed")
    return y, person


def predict_proba(model, nodes, pooled, batch_size=512, groups=None):
    """Frozen inference with variable state/context counts and their group map."""
    inference_groups = model.groups if groups is None else groups
    nodes, pooled = _arrays(nodes, pooled, inference_groups, "prediction")
    if not len(nodes):
        return np.empty(0, dtype=np.float64)
    device = next(model.parameters()).device
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(nodes), batch_size):
            n = torch.as_tensor(nodes[start:start + batch_size], device=device)
            p = torch.as_tensor(pooled[start:start + batch_size], device=device)
            output.append(torch.sigmoid(model(n, p, groups=inference_groups)).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def _metrics(y, probability, weights):
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    # Match the runner's selection metric; person-weighted diagnostics are
    # separate and do not silently change which epoch wins.
    loss = float(log_loss(y, probability, labels=[0, 1]))
    auc = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None
    return {"auc": auc, "logloss": loss,
            "person_weighted_logloss": float(log_loss(y, probability, sample_weight=weights, labels=[0, 1])),
            "person_weighted_auc": float(roc_auc_score(y, probability, sample_weight=weights))
            if len(np.unique(y)) == 2 else None}


def _write_output(output, model, metadata):
    if output is None:
        return
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "groups": model.groups,
        "width": WIDTH,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": metadata,
    }
    torch.save(checkpoint, folder / "model.pt")
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


def load_model(path, device="cpu"):
    """Load the complete source-trained model for frozen cross-domain inference."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = RelationalSetHead(checkpoint["groups"], checkpoint["width"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["metadata"]


def fit_predict(train_nodes, train_pooled, train_y, train_person,
                val_nodes, val_pooled, val_y, val_person,
                test_nodes, test_pooled, groups, seed, weight_decay, output):
    """Fit one fixed-budget candidate and return validation/test probabilities.

    Returns a dict with val_probability, test_probability, metadata, plus val/test
    aliases. ``output`` is a model directory, or None for in-memory evaluation.
    CPU is intentional: this small head usually costs less than GPU launch work.
    """
    groups = np.asarray(groups).astype(str)
    train_nodes, train_pooled = _arrays(train_nodes, train_pooled, groups, "training")
    val_nodes, val_pooled = _arrays(val_nodes, val_pooled, groups, "validation")
    test_nodes, test_pooled = _arrays(test_nodes, test_pooled, groups, "test")
    if not len(train_nodes) or not len(val_nodes):
        raise ValueError("Training and validation must contain eligible origins")
    train_y, train_person = _labels(train_y, train_person, len(train_nodes), "training")
    val_y, val_person = _labels(val_y, val_person, len(val_nodes), "validation")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    torch.manual_seed(int(seed))
    torch.set_num_threads(2)
    origin_weights = _person_weights(train_person)
    val_weights = _person_weights(val_person)
    prior = float(origin_weights @ train_y)
    model = RelationalSetHead(groups)
    model.fit_scalers(train_nodes, train_pooled, origin_weights)
    clipped_prior = min(max(prior, 1e-6), 1 - 1e-6)
    correction = math.log(clipped_prior / (1 - clipped_prior))
    model.prior_logit.fill_(correction)
    metadata = {
        "model": "HNE-relational-set-head",
        "seed": int(seed), "weight_decay": float(weight_decay), "width": WIDTH,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "device": "cpu",
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_origins": len(train_y), "validation_origins": len(val_y),
        "test_origins": len(test_nodes),
        "train_people": len(np.unique(train_person)),
        "validation_people": len(np.unique(val_person)),
        "training_person_weighted_prevalence": prior,
        "logit_prior_correction": correction,
        "training_weighting": "equal person followed by inverse class mass",
        "validation_weighting": "unweighted selection; additional equal-person diagnostics; no class balancing",
        "selection_metric": "auc" if len(np.unique(val_y)) == 2 else "negative_logloss",
        "node_scaler": "training-only, shared across nodes, observed entries only",
        "relation_descriptors": int(model.relation_indices.numel()),
        "context_descriptors_per_variable": 24,
        "training_context_variables": int(model.context_indices.numel() // 24),
        "history": [],
    }
    if len(np.unique(train_y)) == 1:
        model.constant_logit.fill_(correction)
        metadata.update(status="constant_single_training_class", best_epoch=0, epochs_run=0)
    else:
        # These weights sum to N. For a uniformly shuffled batch, mean(w * BCE)
        # estimates the full person- and class-balanced objective.
        class_mass = np.where(train_y > 0, prior, 1 - prior)
        fit_weights = origin_weights / (2 * class_mass)
        fit_weights = fit_weights / fit_weights.sum() * len(train_y)
        nodes_t = torch.from_numpy(train_nodes)
        pooled_t = torch.from_numpy(train_pooled)
        labels_t = torch.from_numpy(train_y)
        weights_t = torch.as_tensor(fit_weights, dtype=torch.float32)
        generator = torch.Generator().manual_seed(int(seed))
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                      weight_decay=float(weight_decay))
        best_score, stale, best_epoch, best_state = -math.inf, 0, 0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            permutation = torch.randperm(len(train_y), generator=generator)
            train_loss = 0.0
            for indices in permutation.split(BATCH_SIZE):
                optimizer.zero_grad(set_to_none=True)
                logits = model(nodes_t[indices], pooled_t[indices], correct_prior=False)
                loss = (F.binary_cross_entropy_with_logits(
                    logits, labels_t[indices], reduction="none") * weights_t[indices]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss in relational set head")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += float(loss.detach()) * len(indices) / len(train_y)
            probability = predict_proba(model, val_nodes, val_pooled)
            metrics = _metrics(val_y, probability, val_weights)
            score = metrics["auc"] if metrics["auc"] is not None else -metrics["logloss"]
            metadata["history"].append({"epoch": epoch, "balanced_training_loss": train_loss,
                                        "validation_auc": metrics["auc"],
                                        "validation_logloss": metrics["logloss"]})
            if score > best_score + 1e-6:
                best_score, stale, best_epoch = score, 0, epoch
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= PATIENCE:
                break
        model.load_state_dict(best_state)
        metadata.update(status="trained", best_epoch=best_epoch,
                        epochs_run=len(metadata["history"]))
    val_probability = predict_proba(model, val_nodes, val_pooled)
    test_probability = predict_proba(model, test_nodes, test_pooled)
    metadata["validation"] = _metrics(val_y, val_probability, val_weights)
    _write_output(output, model, metadata)
    return {"val_probability": val_probability, "test_probability": test_probability,
            "val": val_probability, "test": test_probability, "metadata": metadata}
