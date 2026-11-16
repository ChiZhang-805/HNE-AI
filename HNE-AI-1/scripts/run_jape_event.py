"""Train the released JAPE modules on the shared College interval endpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import vendor_jape
from scripts.run_hne_external import score_description
from scripts.run_transfer import sha, write_json


def read_part(features, part):
    with np.load(features / f"{part}.npz", allow_pickle=False) as z:
        x, y = z["windows"], z["y"]
    meta = pd.read_csv(features / f"{part}.csv", dtype={"person_id": str})
    return x, y, meta


def targets(meta, daily, part, horizon):
    future = None
    for person, rows in meta.groupby("person_id", sort=False):
        with np.load(daily / part / f"{person}_daily.npz", allow_pickle=False) as z:
            values = np.concatenate([z["normalized"], z["normalized_contexts"]], axis=1)
        if future is None:
            future = np.full((len(meta), horizon, values.shape[1]), np.nan, np.float32)
        for index, row in rows.iterrows():
            d = int(row.origin_day)
            future[index] = values[d+1:d+horizon+1]
    return future


def write_scores(folder, part, meta, probability):
    table = meta.copy()
    table["score"] = probability
    table.to_csv(folder / f"{part}.csv", index=False)


def fit(args, cfg):
    train, y, train_meta = read_part(args.features, "train")
    validation, vy, val_meta = read_part(args.features, "validation")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"configuration": cfg,
        "features": {f"{part}.{suffix}": sha(args.features / f"{part}.{suffix}")
                     for part in ("train", "validation") for suffix in ("csv", "npz")},
        "daily": {f"{part}/{p}_daily.npz": sha(args.daily / part / f"{p}_daily.npz")
                  for part, meta in (("train", train_meta), ("validation", val_meta)) for p in meta.person_id.unique()},
        "source": {p.relative_to(ROOT).as_posix(): sha(p) for p in vendor_jape.VENDOR.rglob("*.py")},
        "adapter_sha256": sha(ROOT / "src/vendor_jape.py")}
    record = args.output / "fit_config.json"
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError("JAPE fit inputs or declared configuration changed")
    write_json(record, identity)
    if (args.output / "selection.json").exists():
        print("RETAIN completed JAPE event ensemble", flush=True)
        return
    future = targets(train_meta, args.daily, "train", cfg["pred_len"])
    val_future = targets(val_meta, args.daily, "validation", cfg["pred_len"])
    rows, probabilities, checkpoints = [], [], {}
    for seed in cfg["seeds"]:
        folder = args.output / f"seed{seed}"
        if (folder / "metadata.json").exists():
            metadata = json.loads((folder / "metadata.json").read_text())
            with np.load(folder / "validation_predictions.npz") as z:
                probability = z["probability"]
        else:
            print(f"START JAPE seed{seed} train={len(y)} validation={len(vy)}", flush=True)
            probability, metadata = vendor_jape.fit(train, future, y, validation, val_future, vy,
                                                    folder, cfg, seed, args.device)
        write_scores(folder, "validation", val_meta, probability)
        probabilities.append(probability)
        rows.append({"seed": seed, **score_description(vy, probability),
                     "forecast_epochs": metadata["forecast"]["epochs_run"], "head_epochs": metadata["head"]["epochs_run"]})
        for name in ("forecaster.pt", "head.pt", "preprocessing.npz"):
            checkpoints[f"seed{seed}/{name}"] = sha(folder / name)
        print(json.dumps(rows[-1]), flush=True)
    probability = np.mean(probabilities, axis=0)
    write_scores(args.output, "validation", val_meta, probability)
    np.savez_compressed(args.output / "validation_predictions.npz", seed_probability=np.stack(probabilities),
                        probability=probability, seeds=np.asarray(cfg["seeds"]))
    selection = {"identity": identity, "method": cfg["method"], "seeds": cfg["seeds"],
        "individual_seed": rows, "checkpoint_sha256": checkpoints, "validation": score_description(vy, probability),
        "threshold_val_negative_p90": float(np.quantile(probability[vy == 0], .9, method="linear")),
        "train_people": sorted(train_meta.person_id.unique().tolist()),
        "validation_people": sorted(val_meta.person_id.unique().tolist()),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "test_partition_read": False}
    write_json(args.output / "selection.json", selection)
    write_json(args.output / "freeze_manifest.json", {"selection_sha256": sha(args.output / "selection.json"),
        "checkpoint_sha256": checkpoints, "test_partition_read": False})
    print(json.dumps({"complete": "JAPE fit", "validation": selection["validation"]}), flush=True)


def predict(args, cfg):
    frozen = ROOT / "results/college_experience/frozen_models.json"
    if not frozen.exists():
        raise RuntimeError("Complete and freeze all College models before test prediction")
    manifest = json.loads((args.output / "freeze_manifest.json").read_text())
    selection = json.loads((args.output / "selection.json").read_text())
    if sha(args.output / "selection.json") != manifest["selection_sha256"]:
        raise ValueError("JAPE selection differs from its frozen version")
    for name, digest in manifest["checkpoint_sha256"].items():
        if sha(args.output / name) != digest:
            raise ValueError(f"Selected JAPE model changed: {name}")
    x, y, meta = read_part(args.features, "test")
    scores = []
    for seed in cfg["seeds"]:
        folder = args.output / f"seed{seed}"
        probability = vendor_jape.predict(x, folder, args.device)
        write_scores(folder, "test", meta, probability)
        scores.append(probability)
    probability = np.mean(scores, axis=0)
    write_scores(args.output, "test", meta, probability)
    np.savez_compressed(args.output / "test_predictions.npz", seed_probability=np.stack(scores), probability=probability)
    write_json(args.output / "test_summary.json", {"test_sha256": sha(args.features / "test.npz"),
        "selection_sha256": manifest["selection_sha256"], "metrics": score_description(y, probability),
        "threshold": selection["threshold_val_negative_p90"]})
    print(json.dumps({"complete": "JAPE predict", **score_description(y, probability)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["fit", "predict"], required=True)
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/college_experience")
    parser.add_argument("--daily", type=Path, default=ROOT / "data/college_study")
    parser.add_argument("--output", type=Path, default=ROOT / "results/college_jape")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/jape_event.json").read_text())
    with threadpool_limits(limits=2):
        (fit if args.phase == "fit" else predict)(args, cfg)
