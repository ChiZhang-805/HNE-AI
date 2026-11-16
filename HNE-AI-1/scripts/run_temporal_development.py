"""Two-lr, three-seed development of the same HNE-temporal model in every domain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_additive import validation_score
from src.hne_temporal import LEARNING_RATES, fit_predict

SEEDS = (20261110, 20261111, 20261112)


def _metrics(y, probability):
    return dict(n=len(y), positives=int(y.sum()),
                auroc=float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
                ap=float(average_precision_score(y, probability)) if y.sum() else None,
                brier=float(brier_score_loss(y, probability)))


def _run_one(train, validation, test, groups, seed, lr, output, device, source_fingerprint, fit_function=fit_predict):
    fingerprint = hashlib.sha256(f"{source_fingerprint}:{seed}:{lr}".encode()).hexdigest()
    metadata_path, predictions_path = output / "metadata.json", output / "predictions.npz"
    if metadata_path.exists() and predictions_path.exists() and (output / "model.pt").exists():
        info = json.loads(metadata_path.read_text())
        if info.get("run_fingerprint") == fingerprint:
            prediction = np.load(predictions_path)
            return dict(val_probability=prediction["val_probability"],
                        test_probability=prediction["test_probability"], metadata=info)
    result = fit_function(train, validation, test, groups, seed, lr, output,
                          device=device, run_fingerprint=fingerprint)
    info = result["metadata"]
    print(json.dumps(dict(run=str(output.relative_to(ROOT)), seed=seed, lr=lr,
                          best_epoch=info["best_epoch"], epochs_run=info["epochs_run"],
                          validation=info["validation"], seconds=round(info["fit_seconds"], 2))), flush=True)
    return result


def run(feature_folder, output_folder, datasets, folds, device, *, fit_function=fit_predict,
        extra_core=(), model_name="HNE-temporal"):
    feature_folder, output_folder = feature_folder.resolve(), output_folder.resolve()
    source_hash = hashlib.sha256((ROOT / "src/hne_temporal.py").read_bytes()).hexdigest()
    extra_hashes = {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() for name in extra_core}
    for dataset in datasets:
        feature_path = feature_folder / f"{dataset}.npz"
        feature_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
        archive = np.load(feature_path)
        meta = pd.read_csv(feature_folder / f"{dataset}.csv", dtype={"person_id": str})
        y, partition, groups = archive["y"], archive["fold"], archive["groups"]
        persons = meta.person_id.to_numpy()
        if len(meta) != len(y) or not np.array_equal(meta.y.to_numpy(), y):
            raise ValueError("Feature rows do not match outcome metadata")
        for fold in folds:
            started = time.time()
            train_mask = ~np.isin(partition, [fold, (fold + 1) % 5])
            val_mask, test_mask = partition == (fold + 1) % 5, partition == fold
            train, validation, test = [{k: archive[k][mask] for k in ("windows", "nodes", "pooled")}
                                       for mask in (train_mask, val_mask, test_mask)]
            for bundle, mask in ((train, train_mask), (validation, val_mask)):
                bundle.update(y=y[mask], person=persons[mask])
            destination = output_folder / dataset / model_name / f"fold{fold}"
            fingerprint = f"{source_hash}:{feature_hash}:{fold}"
            if extra_hashes:
                fingerprint += ":" + json.dumps(extra_hashes, sort_keys=True)
            candidates, best, best_score, chosen_lr = [], None, -np.inf, None
            for lr in LEARNING_RATES:
                result = _run_one(train, validation, test, groups, SEEDS[0], lr,
                                  destination / f"lr{lr}" / f"seed{SEEDS[0]}", device, fingerprint, fit_function)
                score = validation_score(validation["y"], result["val_probability"])
                candidates.append(dict(learning_rate=lr, seed=SEEDS[0], validation_score=score,
                                       best_epoch=result["metadata"]["best_epoch"]))
                if score > best_score:
                    best_score, best, chosen_lr = score, result, lr
            repetitions = [best]
            for seed in SEEDS[1:]:
                repetitions.append(_run_one(train, validation, test, groups, seed, chosen_lr,
                                              destination / f"lr{chosen_lr}" / f"seed{seed}", device, fingerprint, fit_function))
            val_seeds = np.stack([r["val_probability"] for r in repetitions])
            test_seeds = np.stack([r["test_probability"] for r in repetitions])
            vp, tp = val_seeds.mean(0), test_seeds.mean(0)
            info = dict(dataset=dataset, model=model_name, fold=fold,
                        stage="development", feature_folder=str(feature_folder),
                        source_sha256=source_hash, feature_sha256=feature_hash,
                        chosen=dict(learning_rate=chosen_lr), trials=candidates, seeds=list(SEEDS),
                        primary_prediction="arithmetic mean of the three prespecified seed probabilities",
                        validation=_metrics(y[val_mask], vp), held_person=_metrics(y[test_mask], tp),
                        individual_seed=[dict(seed=seed, best_epoch=result["metadata"]["best_epoch"],
                                              validation=_metrics(y[val_mask], result["val_probability"]),
                                              held_person=_metrics(y[test_mask], result["test_probability"]))
                                         for seed, result in zip(SEEDS, repetitions)],
                        seconds=time.time() - started)
            if extra_hashes:
                info["extra_core_sha256"] = extra_hashes
            for mask, probability, split in ((val_mask, vp, "validation"), (test_mask, tp, "held_person")):
                frame = meta[mask].copy()
                frame["probability"] = probability
                frame.to_csv(destination / f"{split}.csv", index=False)
                if split == "held_person":
                    frame.to_csv(destination / "ensemble.csv", index=False)
            for seed, probability in zip(SEEDS, test_seeds):
                frame = meta[test_mask].copy()
                frame["probability"] = probability
                frame.to_csv(destination / f"seed{seed}_held_person.csv", index=False)
            np.savez_compressed(destination / "predictions.npz", val_probability=vp, test_probability=tp,
                                val_seed_probability=val_seeds, test_seed_probability=test_seeds)
            for name in ("metadata.json", "metrics.json"):
                (destination / name).write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(dict(dataset=dataset, fold=fold, complete=True,
                                  chosen_lr=chosen_lr, validation=info["validation"],
                                  held_person=info["held_person"], seconds=round(info["seconds"], 2))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/unified_hourly")
    parser.add_argument("--output", type=Path, default=ROOT / "results/unified_hourly")
    parser.add_argument("--datasets", nargs="+", default=["RESILIENT", "BrisT1D-Open"])
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.features, args.output, args.datasets, args.folds, args.device)
