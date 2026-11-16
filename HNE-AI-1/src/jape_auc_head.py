"""AUROC-selected alert head on the same frozen released JAPE forecaster."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from src import vendor_jape as native


def fit(train, y, validation, vy, source, output, seed, device="cuda"):
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((source / "metadata.json").read_text())
    cfg = info["configuration"]
    torch.set_num_threads(2); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    started = time.monotonic()
    forecaster, extractor, head = native.architecture(cfg, train.shape[-1], device)
    forecaster.load_state_dict(torch.load(source / "forecaster.pt", weights_only=True, map_location=device))
    for parameter in forecaster.parameters():
        parameter.requires_grad_(False)
    with np.load(source / "preprocessing.npz") as z:
        norms = {name: z[name] for name in z.files}
    def encoded(x):
        x = torch.from_numpy(native.transform(x, norms["median"], norms["mean"], norms["scale"])).to(device)
        f, g = native.encode(forecaster, extractor, x, cfg["batch_size"])
        tensor = lambda name: torch.from_numpy(norms[name]).to(device)
        return (f-tensor("f_mean"))/tensor("f_std"), (g-tensor("g_mean"))/tensor("g_std")
    f, g = encoded(train); vf, vg = encoded(validation)
    target = torch.from_numpy(np.asarray(y, np.float32)).to(device)
    Focal = native.classes()[3]
    criterion = Focal(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2, min_lr=1e-6)
    best, stale, history = -np.inf, 0, []
    for epoch in range(1, cfg["max_epochs"]+1):
        head.train(); order = torch.randperm(len(f), device=device); total = 0.
        for indices in order.split(cfg["batch_size"]):
            optimizer.zero_grad()
            loss = criterion(head(f[indices], g[indices])[:, 1:].mean(1), target[indices])
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["head_gradient_clip"])
            optimizer.step(); total += float(loss.detach())*len(indices)
        probability = native.head_probability(head, vf, vg, cfg["batch_size"])
        value = roc_auc_score(vy, probability)
        scheduler.step(value)
        history.append({"epoch": epoch, "training_focal_loss": total/len(f), "validation_auroc": value,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        pd.DataFrame(history).to_csv(output / "head_history.csv", index=False)
        print(f"JAPE-AUC seed{seed} head {epoch}/{cfg['max_epochs']} val_auroc={value:.6f}", flush=True)
        if value > best:
            best, stale, best_epoch = value, 0, epoch
            torch.save(native.checkpoint_state(head), output / "head.pt")
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    head.load_state_dict(torch.load(output / "head.pt", weights_only=True, map_location=device))
    probability = native.head_probability(head, vf, vg, cfg["batch_size"])
    np.savez_compressed(output / "validation_predictions.npz", probability=probability)
    metadata = {"seed": seed, "source": str(source), "configuration": cfg,
        "selection": "validation AUROC with unchanged20epoch limit,patience5; ReduceLROnPlateau also follows AUROC",
        "epochs_run": epoch, "best_epoch": best_epoch, "validation_auroc": best,
        "validation_ap": float(average_precision_score(vy, probability)), "seconds": time.monotonic()-started}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return probability, metadata


def predict(windows, source, output, device="cuda"):
    info = json.loads((source / "metadata.json").read_text())
    cfg = info["configuration"]
    model, extractor, head = native.architecture(cfg, windows.shape[-1], device)
    model.load_state_dict(torch.load(source / "forecaster.pt", weights_only=True, map_location=device))
    head.load_state_dict(torch.load(output / "head.pt", weights_only=True, map_location=device))
    with np.load(source / "preprocessing.npz") as z:
        norms = {name: z[name] for name in z.files}
    x = torch.from_numpy(native.transform(windows, norms["median"], norms["mean"], norms["scale"])).to(device)
    f, g = native.encode(model, extractor, x, cfg["batch_size"])
    tensor = lambda name: torch.from_numpy(norms[name]).to(device)
    return native.head_probability(head, (f-tensor("f_mean"))/tensor("f_std"),
                                   (g-tensor("g_mean"))/tensor("g_std"), cfg["batch_size"])
