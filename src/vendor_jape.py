"""Released JAPE modules with a daily interval-event endpoint adapter."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

VENDOR = Path(__file__).resolve().parents[1] / "literature/reference_code/JAPE"
_CLASSES = None


@contextmanager
def author_imports():
    owned = lambda name: name.split(".", 1)[0] in {"models", "layers", "utils", "exp", "data_provider"}
    previous = {k: v for k, v in sys.modules.items() if owned(k)}
    for name in previous:
        del sys.modules[name]
    old_path = list(sys.path)
    sys.path.insert(0, str(VENDOR))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(previous)
        sys.path[:] = old_path


def classes():
    global _CLASSES
    if _CLASSES is None:
        with author_imports():
            model = importlib.import_module("models.jape")
            anomaly = importlib.import_module("exp.anomaly")
            evaluation = importlib.import_module("utils.anomaly_eval")
            _CLASSES = (model.JAPEModel, anomaly.StructuralFeatureExtractor,
                        anomaly.DualViewAlertingHead, evaluation.BinaryFocalLossWithLogits,
                        evaluation.select_threshold)
    return _CLASSES


def architecture(cfg, channels, device):
    Model, Extractor, Head, _, _ = classes()
    forecaster = Model(SimpleNamespace(**cfg, enc_in=channels)).float().to(device)
    extractor = Extractor(tail_ratio=cfg["tail_ratio"]).to(device)
    slices = (cfg["seq_len"] - cfg["patch_len"]) // cfg["stride"] + 1
    head = Head(channels, 13, d_model=cfg["head_width"], n_heads=cfg["head_heads"],
                n_layers=cfg["head_layers"], dropout=cfg["forecast_dropout"],
                history_slice_count=slices).to(device)
    return forecaster, extractor, head


def transform(array, median, mean, scale):
    filled = np.where(np.isfinite(array), array, median)
    return ((filled - mean) / scale).astype(np.float32)


def checkpoint_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def forecast_loss(model, x, target, mask):
    main, auxiliary = model(x, return_auxiliary=True)
    count = mask.sum().clamp_min(1)
    return (((main-target).square() + (auxiliary-target).square()) * mask).sum() / count


@torch.no_grad()
def validate_forecast(model, x, target, mask, batch_size):
    model.eval()
    total, count = 0., 0
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        n = int(mask[start:end].sum())
        total += float(forecast_loss(model, x[start:end], target[start:end], mask[start:end])) * n
        count += n
    return total / count


@torch.no_grad()
def encode(model, extractor, x, batch_size):
    model.eval()
    values, tokens = [], []
    for start in range(0, len(x), batch_size):
        forecast, graphs = model(x[start:start+batch_size], return_graphs=True)
        graph = extractor(graphs["history_graph"], graphs["history_raw_graph"],
                          graphs["future_graph"], graphs["future_raw_graph"])
        values.append(forecast)
        tokens.append(graph)
    return torch.cat(values), torch.cat(tokens)


@torch.no_grad()
def head_probability(head, values, graphs, batch_size):
    head.eval()
    out = []
    for start in range(0, len(values), batch_size):
        logits = head(values[start:start+batch_size], graphs[start:start+batch_size])
        out.append(torch.sigmoid(logits[:, 1:].mean(1)).cpu().numpy())
    return np.concatenate(out)


def fit(train, future, labels, validation, validation_future, validation_labels,
        folder, cfg, seed, device="cuda"):
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    started = time.monotonic()
    median = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=(0, 1))
    median = np.nan_to_num(median, nan=0.).astype(np.float32)
    filled = np.where(np.isfinite(train), train, median)
    scaler = StandardScaler().fit(filled.reshape(-1, train.shape[-1]))
    mean, scale = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
    tensor = lambda array: torch.from_numpy(array).to(device)
    x, vx = [tensor(transform(a, median, mean, scale)) for a in (train, validation)]
    targets, masks = [], []
    for a in (future, validation_future):
        observed = np.isfinite(a)
        targets.append(tensor(np.where(observed, (a-mean)/scale, 0.).astype(np.float32)))
        masks.append(tensor(observed))
    y, vy = targets; mask, vmask = masks
    forecaster, extractor, head = architecture(cfg, train.shape[-1], device)
    batch = cfg["batch_size"]
    forecaster_path = folder / "forecaster.pt"
    if (folder / "forecaster.json").exists():
        forecast_summary = json.loads((folder / "forecaster.json").read_text())
        forecaster.load_state_dict(torch.load(forecaster_path, weights_only=True, map_location=device))
    else:
        optimizer = torch.optim.Adam(forecaster.parameters(), lr=cfg["learning_rate"])
        best, stale, rows = np.inf, 0, []
        for epoch in range(1, cfg["max_epochs"]+1):
            forecaster.train(); order = torch.randperm(len(x), device=device)
            total, count = 0., 0
            for indices in order.split(batch):
                optimizer.zero_grad()
                loss = forecast_loss(forecaster, x[indices], y[indices], mask[indices])
                loss.backward(); optimizer.step()
                n = int(mask[indices].sum()); total += float(loss.detach()) * n; count += n
            value = validate_forecast(forecaster, vx, vy, vmask, batch)
            rows.append({"epoch": epoch, "train_mse_sum": total/count, "validation_mse_sum": value})
            pd.DataFrame(rows).to_csv(folder / "forecast_history.csv", index=False)
            print(f"JAPE seed{seed} forecast {epoch}/{cfg['max_epochs']} val_mse={value:.7g}", flush=True)
            if not np.isfinite(value):
                raise FloatingPointError("JAPE forecasting produced nonfinite validation loss")
            if value < best:
                best, stale, best_epoch = value, 0, epoch
                torch.save(checkpoint_state(forecaster), forecaster_path)
            else:
                stale += 1
            if stale >= cfg["patience"]:
                break
        forecast_summary = {"epochs_run": epoch, "best_epoch": best_epoch, "validation_mse_sum": best}
        (folder / "forecaster.json").write_text(json.dumps(forecast_summary, indent=2))
        forecaster.load_state_dict(torch.load(forecaster_path, weights_only=True, map_location=device))
        del optimizer
    for parameter in forecaster.parameters():
        parameter.requires_grad_(False)
    f, g = encode(forecaster, extractor, x, batch)
    vf, vg = encode(forecaster, extractor, vx, batch)
    f_mean, f_std = f.mean((0, 1), keepdim=True), f.std((0, 1), keepdim=True).clamp_min(1e-6)
    g_mean, g_std = g.mean((0, 1), keepdim=True), g.std((0, 1), keepdim=True).clamp_min(1e-6)
    f, vf = (f-f_mean)/f_std, (vf-f_mean)/f_std
    g, vg = (g-g_mean)/g_std, (vg-g_mean)/g_std
    target = tensor(np.asarray(labels, dtype=np.float32))
    _, _, _, FocalLoss, select_threshold = classes()
    criterion = FocalLoss(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2, min_lr=1e-6)
    best, stale, rows = -np.inf, 0, []
    for epoch in range(1, cfg["max_epochs"]+1):
        head.train(); order = torch.randperm(len(f), device=device)
        total = 0.
        for indices in order.split(batch):
            optimizer.zero_grad()
            event_logit = head(f[indices], g[indices])[:, 1:].mean(1)
            loss = criterion(event_logit, target[indices])
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["head_gradient_clip"])
            optimizer.step(); total += float(loss.detach()) * len(indices)
        probability = head_probability(head, vf, vg, batch)
        threshold = select_threshold(probability, validation_labels)
        value = f1_score(validation_labels, probability >= threshold, zero_division=0)
        auroc = roc_auc_score(validation_labels, probability)
        scheduler.step(value)
        rows.append({"epoch": epoch, "train_focal_loss": total/len(f), "validation_f1": value,
                     "validation_auroc": auroc, "threshold": threshold, "learning_rate": optimizer.param_groups[0]["lr"]})
        pd.DataFrame(rows).to_csv(folder / "head_history.csv", index=False)
        print(f"JAPE seed{seed} head {epoch}/{cfg['max_epochs']} val_f1={value:.6f} auroc={auroc:.6f}", flush=True)
        if value > best:
            best, stale, best_epoch, best_threshold = value, 0, epoch, threshold
            torch.save(checkpoint_state(head), folder / "head.pt")
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    head.load_state_dict(torch.load(folder / "head.pt", weights_only=True, map_location=device))
    probability = head_probability(head, vf, vg, batch)
    preprocessing = {"median": median, "mean": mean, "scale": scale,
                     "f_mean": f_mean.cpu().numpy(), "f_std": f_std.cpu().numpy(),
                     "g_mean": g_mean.cpu().numpy(), "g_std": g_std.cpu().numpy()}
    np.savez_compressed(folder / "preprocessing.npz", **preprocessing)
    np.savez_compressed(folder / "validation_predictions.npz", probability=probability)
    metadata = {"configuration": cfg, "seed": seed, "channels": train.shape[-1], "forecast": forecast_summary,
                "head": {"epochs_run": epoch, "best_epoch": best_epoch, "validation_f1": best, "threshold": best_threshold},
                "parameters_forecast": sum(p.numel() for p in forecaster.parameters()),
                "parameters_head": sum(p.numel() for p in head.parameters()),
                "fit_seconds": time.monotonic()-started, "train_origins": len(x), "validation_origins": len(vx), "test_origins": 0}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return probability, metadata


def predict(windows, folder, device="cuda"):
    metadata = json.loads((folder / "metadata.json").read_text())
    cfg = metadata["configuration"]
    forecaster, extractor, head = architecture(cfg, metadata["channels"], device)
    forecaster.load_state_dict(torch.load(folder / "forecaster.pt", weights_only=True, map_location=device))
    head.load_state_dict(torch.load(folder / "head.pt", weights_only=True, map_location=device))
    with np.load(folder / "preprocessing.npz") as saved:
        norm = {name: saved[name] for name in saved.files}
    x = torch.from_numpy(transform(windows, norm["median"], norm["mean"], norm["scale"])).to(device)
    f, g = encode(forecaster, extractor, x, cfg["batch_size"])
    tensor = lambda name: torch.from_numpy(norm[name]).to(device)
    f, g = (f-tensor("f_mean"))/tensor("f_std"), (g-tensor("g_mean"))/tensor("g_std")
    return head_probability(head, f, g, cfg["batch_size"])
