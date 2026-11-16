"""Training-only future supervision of the unchanged HNETemporal risk encoder.

The shared decoder reads a graph_norm hook. Its predictions never enter the
risk head; the inherited forward/predict interface remains the original one.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .hne_temporal import (HNETemporal, WIDTH, MAX_EPOCHS, PATIENCE, BATCH_SIZE,
                           WEIGHT_DECAY, _arrays, _device)
from .hne_set import _metrics, _person_weights

CONDITIONS = ("future-values", "future-moments")


class FutureHNE(HNETemporal):
    def __init__(self, groups, history_steps, condition):
        if condition not in CONDITIONS:
            raise ValueError("Unknown future objective")
        super().__init__(groups, history_steps)
        self.condition = condition
        self.aux_width = 7 if condition == "future-values" else 3
        self.decoder = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, self.aux_width))
        scale_width = 1 if condition == "future-values" else 3
        self.register_buffer("aux_mean", torch.zeros(scale_width))
        self.register_buffer("aux_scale", torch.ones(scale_width))
        self.register_buffer("aux_supported", torch.zeros(scale_width, dtype=torch.bool))
        self._node_latent = None
        self.graph_norm.register_forward_hook(self._capture_nodes)

    def _capture_nodes(self, module, inputs, output):
        self._node_latent = output

    def auxiliary(self):
        return self.decoder(self._node_latent)

    def fit_aux_scaler(self, target, base):
        values = np.asarray(target, dtype=np.float64)
        if self.condition == "future-values":
            values = values.reshape(len(values), -1, 1)
        observed = np.isfinite(values)
        count = observed.sum(1, keepdims=True)
        weights = base[:, None, None] / np.maximum(count, 1)
        mass = (weights * observed).sum((0, 1))
        safe = np.where(observed, values, 0.)
        mean = (weights * safe).sum((0, 1)) / np.maximum(mass, 1e-12)
        variance = (weights * np.where(observed, (safe-mean)**2, 0.)).sum((0, 1)) / np.maximum(mass, 1e-12)
        scale = np.sqrt(np.maximum(variance, 0.))
        scale = np.where(scale > 1e-6, scale, 1.)
        for name, value in (("mean", mean), ("scale", scale), ("supported", mass > 0)):
            buffer = getattr(self, "aux_"+name)
            buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device))

    def target_arrays(self, target):
        mean, scale = self.aux_mean.cpu().numpy(), self.aux_scale.cpu().numpy()
        mask = np.isfinite(target) & self.aux_supported.cpu().numpy()
        value = (np.where(mask, target, mean)-mean)/scale
        return value.astype(np.float32), mask


def origin_auxiliary(model, target, mask):
    loss = F.huber_loss(model.auxiliary(), target, reduction="none", delta=1.)
    return (loss * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)


def evaluate(model, arrays, target, mask, base):
    device = next(model.parameters()).device
    scores, losses = [], []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(arrays[0]), 256):
            sl = slice(first, first+256)
            inputs = [torch.as_tensor(a[sl], device=device) for a in arrays]
            scores.append(torch.sigmoid(model(*inputs)).cpu().numpy())
            losses.append(origin_auxiliary(model, torch.as_tensor(target[sl], device=device),
                                           torch.as_tensor(mask[sl], device=device)).cpu().numpy())
    return np.concatenate(scores).astype(np.float64), float(base @ np.concatenate(losses))


def fit_predict(train, validation, groups, condition, auxiliary_weight, seed, output,
                learning_rate=0.001, device="auto", run_fingerprint=None):
    if learning_rate != 0.001 or auxiliary_weight not in (.1, 1.):
        raise ValueError("Only the prespecified learning rate and auxiliary weights are supported")
    started = time.time()
    arrays = _arrays(*(train[k] for k in ("windows", "nodes", "pooled")), groups, "training")
    val_arrays = _arrays(*(validation[k] for k in ("windows", "nodes", "pooled")), groups, "validation")
    y, vy = np.asarray(train["y"], np.float32), np.asarray(validation["y"], np.float32)
    if len(np.unique(y)) != 2:
        raise ValueError("This auxiliary comparison requires both event classes in training")
    torch.manual_seed(int(seed)); torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    base, val_base = _person_weights(train["person"]), _person_weights(validation["person"])
    prior = float(base @ y)
    correction = math.log(np.clip(prior, 1e-6, 1-1e-6)/np.clip(1-prior, 1e-6, 1-1e-6))
    model = FutureHNE(groups, arrays[0].shape[1], condition)
    model.fit_scalers(*arrays, base)
    model.prior_logit.fill_(correction)
    model.fit_aux_scaler(train["auxiliary_target"], base)
    target, mask = model.target_arrays(train["auxiliary_target"])
    val_target, val_mask = model.target_arrays(validation["auxiliary_target"])
    device = _device(device); model.to(device)
    weights = base / (2*np.where(y > 0, prior, 1-prior))
    weights *= len(y)/weights.sum()
    arrays_t = [torch.as_tensor(a, device=device) for a in arrays]
    y_t, bce_weight = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in (y, weights)]
    target_t = torch.as_tensor(target, device=device)
    mask_t = torch.as_tensor(mask, device=device)
    aux_weight = torch.as_tensor(base*len(y), dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(int(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    folder = Path(output); folder.mkdir(parents=True, exist_ok=True)
    history, best_score, best_state, best_epoch, stale = [], -math.inf, None, 0, 0
    columns = ["epoch", "training_bce", "training_aux_huber", "training_total", "validation_auc", "validation_logloss", "validation_aux_huber"]
    with (folder / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for epoch in range(1, MAX_EPOCHS+1):
            model.train(); total_bce, total_aux = 0., 0.
            for indices in torch.randperm(len(y), generator=generator).split(BATCH_SIZE):
                indices = indices.to(device)
                optimizer.zero_grad(set_to_none=True)
                logit = model(*(a[indices] for a in arrays_t), correct_prior=False)
                bce = (F.binary_cross_entropy_with_logits(logit, y_t[indices], reduction="none")*bce_weight[indices]).mean()
                aux = (origin_auxiliary(model, target_t[indices], mask_t[indices])*aux_weight[indices]).mean()
                loss = bce + auxiliary_weight*aux
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite future-objective loss")
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_bce += float(bce.detach())*len(indices)/len(y)
                total_aux += float(aux.detach())*len(indices)/len(y)
            probability, val_aux = evaluate(model, val_arrays, val_target, val_mask, val_base)
            metrics = _metrics(vy, probability, val_base)
            score = metrics["auc"] if metrics["auc"] is not None else -metrics["logloss"]
            row = dict(epoch=epoch, training_bce=total_bce, training_aux_huber=total_aux,
                training_total=total_bce+auxiliary_weight*total_aux, validation_auc=metrics["auc"],
                validation_logloss=metrics["logloss"], validation_aux_huber=val_aux)
            history.append(row); writer.writerow(row); handle.flush()
            print(json.dumps(dict(condition=condition, auxiliary_weight=auxiliary_weight, seed=seed, **row)), flush=True)
            if score > best_score+1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    probability, val_aux = evaluate(model, val_arrays, val_target, val_mask, val_base)
    metadata = dict(model="HNE-temporal-"+condition, condition=condition, auxiliary_weight=auxiliary_weight,
        seed=int(seed), learning_rate=learning_rate, width=WIDTH, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY,
        train_origins=len(y), validation_origins=len(vy), test_origins=0,
        train_people=len(np.unique(train["person"])), validation_people=len(np.unique(validation["person"])),
        train_origins_without_aux=int((~mask.any((1, 2))).sum()),
        state_nodes=arrays[1].shape[1], context_nodes=arrays[0].shape[2]-arrays[1].shape[1],
        parameters=sum(p.numel() for p in model.parameters()),
        auxiliary_mean=model.aux_mean.cpu().tolist(), auxiliary_scale=model.aux_scale.cpu().tolist(),
        auxiliary_supported=model.aux_supported.cpu().tolist(),
        training_person_weighted_prevalence=prior, logit_prior_correction=correction,
        best_epoch=best_epoch, epochs_run=len(history), history=history,
        validation=_metrics(vy, probability, val_base), validation_aux_huber=val_aux,
        fit_seconds=time.time()-started, run_fingerprint=run_fingerprint, device=str(device))
    torch.save(dict(format_version=1, condition=condition, groups=model.groups,
                    history_steps=model.history_steps, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    metadata=metadata), folder / "model.pt")
    np.savez_compressed(folder / "predictions.npz", val_probability=probability)
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return dict(val_probability=probability, metadata=metadata)


def load_model(path, device="auto"):
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FutureHNE(checkpoint["groups"], checkpoint["history_steps"], checkpoint["condition"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(_device(device)).eval(), checkpoint["metadata"]
