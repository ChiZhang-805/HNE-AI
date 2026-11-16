"""Prespecified supplementary HNE-temporal fit and frozen external prediction.

Fit opens only College train/validation; the existing fit_predict receives an
empty, correctly shaped test bundle. The original temporal core is unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_additive import validation_score
from src.hne_temporal import (WIDTH, MAX_EPOCHS, PATIENCE, BATCH_SIZE, WEIGHT_DECAY,
                              LEARNING_RATES, fit_predict, load_model, predict_proba)
from scripts.run_hne_external import load_split, score_description, cluster_statistics, STATISTICS
from scripts.run_transfer import sha, write_json

SEEDS = (20261110, 20261111, 20261112)


def configuration():
    cohort = json.loads((ROOT / "configs/college_evaluation.json").read_text(encoding="utf-8"))
    return {"version": "college-temporal-external-v1-20261110", "dataset": cohort["dataset"],
        "method": "HNE-temporal", "role": "prespecified supplementary model; primary HNE-additive remains separate",
        **{key: cohort[key] for key in ("expected_nodes", "expected_window", "expected_pooled")},
        "learning_rates": list(LEARNING_RATES), "seeds": list(SEEDS), "width": WIDTH,
        "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "selection": "first seed selects learning rate by validation AUROC (negative logloss if single class); exact ties retain first candidate",
        "epoch_selection": "unchanged temporal core: validation AUROC with patience12; single-class negative logloss",
        "ensemble": "equal arithmetic mean of all3 fixed seed probabilities at the selected learning rate",
        "fit_partitions": ["train", "validation"], "refit_on_validation": False,
        "weighting": "training-person balanced then class balanced; prior-logit correction before prediction",
        "threshold": "linear90th percentile of ensemble validation negatives; alerts strictly greater",
        "test_bootstrap": {"replicates": 2000, "seed": SEEDS[0], "unit": "whole person"}}


def core_hashes(extra_core=()):
    return {name: sha(ROOT / "src" / name) for name in ["hne_temporal.py", "hne_set.py", *extra_core]}


def read_partition(features, part, cfg):
    data = load_split(features, part, cfg)
    with np.load(features / f"{part}.npz", allow_pickle=False) as archive:
        data["nodes"] = archive["nodes"]
    data["person"] = data["people"]
    return data


def save_scores(path, data, probability):
    frame = data["meta"].copy()
    frame["score"] = probability
    frame.to_csv(path, index=False)


def run_one(train, validation, empty, groups, seed, lr, folder, identity, device, fit_function=fit_predict):
    fingerprint = hashlib.sha256(json.dumps(
        {"identity": identity, "seed": seed, "learning_rate": lr}, sort_keys=True).encode()).hexdigest()
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Existing temporal fit differs: {folder}")
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            probability = saved["val_probability"]
        if not (folder / "model.pt").is_file():
            raise ValueError(f"Missing selected checkpoint: {folder}")
        print(f"REUSE seed={seed} lr={lr}", flush=True)
    else:
        print(f"START seed={seed} lr={lr} train={len(train['y'])} validation={len(validation['y'])}", flush=True)
        result = fit_function(train, validation, empty, groups, seed, lr, folder,
                              device=device, run_fingerprint=fingerprint)
        metadata, probability = result["metadata"], result["val_probability"]
        print(json.dumps({"seed": seed, "learning_rate": lr, "epochs_run": metadata["epochs_run"],
                          "best_epoch": metadata["best_epoch"], "validation": metadata["validation"],
                          "seconds": metadata["fit_seconds"]}), flush=True)
    save_scores(folder / "validation.csv", validation, probability)
    return probability, metadata


def fit(features, output, cfg, device, *, fit_function=fit_predict, extra_core=()):
    train = read_partition(features, "train", cfg)
    validation = read_partition(features, "validation", cfg)
    if set(train["people"]) & set(validation["people"]):
        raise ValueError("Training and validation people overlap")
    identity = {"configuration": cfg, "core": core_hashes(extra_core),
                "train": train["hashes"], "validation": validation["hashes"]}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "fit_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Existing temporal fit uses different data or configuration")
    if (output / "selection.json").exists():
        selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        if selection["identity"] != identity:
            raise ValueError("Existing temporal selection differs")
        for name, digest in selection["checkpoint_sha256"].items():
            if sha(output / name) != digest:
                raise ValueError(f"Frozen checkpoint changed: {name}")
        print("RETAIN completed temporal ensemble", flush=True)
        return
    write_json(config_path, identity)
    empty = {key: train[key][:0] for key in ("windows", "nodes", "pooled")}
    candidates, best, best_score, chosen_lr = [], None, -np.inf, None
    for lr in LEARNING_RATES:
        folder = output / f"lr{lr}" / f"seed{SEEDS[0]}"
        probability, metadata = run_one(train, validation, empty, train["groups"], SEEDS[0], lr,
                                        folder, identity, device, fit_function)
        value = validation_score(validation["y"], probability)
        candidates.append({"learning_rate": lr, "seed": SEEDS[0], "validation_selection_score": value,
                           "best_epoch": metadata["best_epoch"], **score_description(validation["y"], probability)})
        if value > best_score:
            best, best_score, chosen_lr = (probability, metadata), value, lr
    repeats = [best]
    for seed in SEEDS[1:]:
        repeats.append(run_one(train, validation, empty, train["groups"], seed, chosen_lr,
                               output / f"lr{chosen_lr}" / f"seed{seed}", identity, device, fit_function))
    probabilities = np.stack([value[0] for value in repeats])
    ensemble = probabilities.mean(0)
    save_scores(output / "validation.csv", validation, ensemble)
    np.savez_compressed(output / "validation_predictions.npz", seed_probability=probabilities,
                        probability=ensemble, seeds=np.asarray(SEEDS))
    negative = ensemble[validation["y"] == 0]
    threshold = float(np.quantile(negative, .9, method="linear")) if len(negative) else None
    checkpoints = [f"lr{chosen_lr}/seed{seed}/model.pt" for seed in SEEDS]
    selection = {"identity": identity, "method": cfg["method"], "selected_learning_rate": chosen_lr,
        "trials": candidates, "seeds": list(SEEDS), "weights": [1/3]*3,
        "checkpoint_sha256": {name: sha(output / name) for name in checkpoints},
        "validation": score_description(validation["y"], ensemble),
        "individual_seed": [{"seed": seed, "best_epoch": value[1]["best_epoch"],
                             "validation": score_description(validation["y"], value[0])}
                            for seed, value in zip(SEEDS, repeats)],
        "threshold_val_negative_p90": threshold, "validation_scores_sha256": sha(output / "validation.csv"),
        "train_people": sorted(set(train["people"])), "validation_people": sorted(set(validation["people"])),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "test_partition_read": False}
    write_json(output / "selection.json", selection)
    write_json(output / "freeze_manifest.json", {"selection_sha256": sha(output / "selection.json"),
        "checkpoint_sha256": selection["checkpoint_sha256"], "test_partition_read": False})
    print(json.dumps({"complete": "fit", "learning_rate": chosen_lr, "validation": selection["validation"]}), flush=True)


def predict(features, output, cfg, device, *, extra_core=()):
    freeze = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    if sha(output / "selection.json") != freeze["selection_sha256"]:
        raise ValueError("Temporal selection changed after freeze")
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if selection["identity"]["configuration"] != cfg or selection["identity"]["core"] != core_hashes(extra_core):
        raise ValueError("Temporal definition changed after freeze")
    for name, digest in freeze["checkpoint_sha256"].items():
        if sha(output / name) != digest:
            raise ValueError(f"Frozen checkpoint changed: {name}")
    test = read_partition(features, "test", cfg)
    if (set(selection["train_people"]) | set(selection["validation_people"])) & set(test["people"]):
        raise ValueError("Test people overlap earlier partitions")
    scores = []
    for seed in SEEDS:
        folder = output / f"lr{selection['selected_learning_rate']}" / f"seed{seed}"
        model, _ = load_model(folder, device=device)
        probability = predict_proba(model, test["windows"], test["nodes"], test["pooled"], test["groups"])
        save_scores(folder / "test.csv", test, probability)
        scores.append(probability)
        del model
    ensemble = np.mean(scores, axis=0)
    save_scores(output / "test.csv", test, ensemble)
    people = np.unique(test["people"])
    draws = np.random.default_rng(cfg["test_bootstrap"]["seed"]).multinomial(
        len(people), np.full(len(people), 1/len(people)), size=cfg["test_bootstrap"]["replicates"])
    point, boot, counts = cluster_statistics(test["y"], ensemble, test["people"],
        selection["threshold_val_negative_p90"], draws)
    metrics = {"method": cfg["method"], **counts}
    for index, name in enumerate(STATISTICS):
        finite = np.isfinite(boot[:, index])
        ci = np.quantile(boot[finite, index], [.025, .975]) if finite.any() else [None, None]
        metrics.update({name: point[index], name+"_ci_low": ci[0], name+"_ci_high": ci[1]})
    pd.DataFrame([metrics]).to_csv(output / "test_metrics.csv", index=False)
    np.savez_compressed(output / "test_predictions.npz", seed_probability=np.stack(scores), probability=ensemble,
                        bootstrap=boot, metric_names=np.asarray(STATISTICS))
    write_json(output / "test_summary.json", {"test_hashes": test["hashes"],
        "selection_sha256": freeze["selection_sha256"], "metrics": metrics})
    print(json.dumps({"complete": "predict", "metrics": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["fit", "predict"], default="fit")
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/college_experience")
    parser.add_argument("--output", type=Path, default=ROOT / "results/college_temporal")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--all-methods-frozen", action="store_true")
    args = parser.parse_args()
    if args.phase == "predict" and not args.all_methods_frozen:
        parser.error("predict requires --all-methods-frozen; test files have not been opened")
    features = args.features if args.features.is_absolute() else ROOT / args.features
    output = args.output if args.output.is_absolute() else ROOT / args.output
    with threadpool_limits(limits=2):
        (fit if args.phase == "fit" else predict)(features, output, configuration(), args.device)


if __name__ == "__main__":
    main()
