"""Matched-primary-metric selection for JAPE's unchanged alerting architecture."""
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
from src import jape_auc_head
from scripts.run_jape_event import read_part, write_scores
from scripts.run_hne_external import score_description
from scripts.run_transfer import sha, write_json

SEEDS = (20261110, 20261111, 20261112)


def fit_group(train, y, validation, vy, val_meta, source, output):
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {f"seed{seed}/{name}": sha(source / f"seed{seed}" / name)
                     for seed in SEEDS for name in ("forecaster.pt", "preprocessing.npz")}
    identity = {"source": str(source.relative_to(ROOT)), "source_sha256": source_hashes,
                "head_adapter_sha256": sha(ROOT / "src/jape_auc_head.py"),
                "configuration": json.loads((ROOT / "configs/jape_auc_head.json").read_text())}
    config = output / "fit_config.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError("Existing JAPE AUROC condition differs")
    write_json(config, identity)
    if (output / "selection.json").exists():
        return
    scores, rows = [], []
    for seed in SEEDS:
        folder = output / f"seed{seed}"
        if (folder / "metadata.json").exists():
            metadata = json.loads((folder / "metadata.json").read_text())
            with np.load(folder / "validation_predictions.npz") as z:
                probability = z["probability"]
        else:
            probability, metadata = jape_auc_head.fit(train, y, validation, vy, source / f"seed{seed}", folder, seed)
        write_scores(folder, "validation", val_meta, probability)
        scores.append(probability)
        rows.append({"seed": seed, "epochs_run": metadata["epochs_run"], "best_epoch": metadata["best_epoch"],
                     **score_description(vy, probability)})
    probability = np.mean(scores, axis=0)
    write_scores(output, "validation", val_meta, probability)
    selection = {"identity": identity, "method": "JAPE-event-AUC-3seed", "seeds": list(SEEDS),
        "validation": score_description(vy, probability), "individual_seed": rows,
        "threshold_val_negative_p90": float(np.quantile(probability[vy == 0], .9)),
        "checkpoint_sha256": {f"seed{seed}/head.pt": sha(output / f"seed{seed}/head.pt") for seed in SEEDS},
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "test_partition_read": False}
    write_json(output / "selection.json", selection)
    write_json(output / "freeze_manifest.json", {"selection_sha256": sha(output / "selection.json"),
        "checkpoint_sha256": selection["checkpoint_sha256"], "source_sha256": source_hashes})
    print(json.dumps({"complete": str(output.relative_to(ROOT)), "validation": selection["validation"]}), flush=True)


def predict_group(x, meta, source, output, part):
    scores = []
    for seed in SEEDS:
        folder = output / f"seed{seed}"
        probability = jape_auc_head.predict(x, source / f"seed{seed}", folder)
        write_scores(folder, part, meta, probability)
        scores.append(probability)
    write_scores(output, part, meta, np.mean(scores, axis=0))


def main(phase):
    if phase == "development":
        for dataset in ("RESILIENT", "BrisT1D-Open"):
            with np.load(ROOT / f"data/features/unified_daily/{dataset}.npz") as z:
                x, y, folds = z["windows"], z["y"], z["fold"]
            meta = pd.read_csv(ROOT / f"data/features/unified_daily/{dataset}.csv", dtype={"person_id": str})
            for fold in range(5):
                tr = (folds != fold) & (folds != (fold+1)%5)
                va, he = folds == (fold+1)%5, folds == fold
                source = ROOT / f"results/jape_development/{dataset}/fold{fold}"
                output = ROOT / f"results/jape_auc_development/{dataset}/fold{fold}"
                fit_group(x[tr], y[tr], x[va], y[va], meta[va], source, output)
                predict_group(x[he], meta[he], source, output, "held_person")
    else:
        features = ROOT / "data/features/college_experience"
        source, output = ROOT / "results/college_jape", ROOT / "results/college_jape_auc"
        if phase == "fit":
            x, y, _ = read_part(features, "train")
            vx, vy, vm = read_part(features, "validation")
            fit_group(x, y, vx, vy, vm, source, output)
        else:
            if not (ROOT / "results/college_experience/frozen_models.json").exists():
                raise RuntimeError("Complete all model selections before College test prediction")
            x, y, meta = read_part(features, "test")
            predict_group(x, meta, source, output, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "fit", "predict"], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.phase)
