"""Freeze one HNE rule jointly trained on three labeled human domains.

This is an additional-source-data RQ3 condition. It is not a target-only,
equal-training-data main comparator. There is intentionally no test-data path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_additive import AdditiveWarning, validation_score, person_weights
from scripts.run_context_grid import load_unit, sources, label, FEATURES, GROUPS
from scripts.run_hne_external import score_description
from scripts.run_transfer import sha, write_json

RIDGES, CS = (.1, 1., 10.), (.01, .1, 1.)
TRAIN_UNITS = (("RESILIENT", "RESILIENT"), ("BrisT1D-Open", "BrisT1D-Open"),
               ("Human-Rhythms_train", "Human-Rhythms"))
VALIDATION = "Human-Rhythms_validation"


def unit_paths(unit, ridge):
    prefix = sources(unit)[1] if ridge == 10 else FEATURES / f"lambda{label(ridge)}" / unit
    return prefix.with_suffix(".csv"), prefix.with_suffix(".npz")


def load_datasets():
    datasets, hashes, baseline_keys = {}, {}, {}
    for ridge in RIDGES:
        parts, values = [], []
        for unit, domain in TRAIN_UNITS:
            meta, x, y = load_unit(unit, ridge)
            if unit == "Human-Rhythms_train" and set(meta.split) != {"train"}:
                raise ValueError("Human Rhythms training split changed")
            keys = meta[["person_id", "origin_day", "y"]].reset_index(drop=True)
            if unit in baseline_keys and not keys.equals(baseline_keys[unit]):
                raise ValueError("Context-ridge candidates use different training origins")
            baseline_keys[unit] = keys
            meta = meta.copy(); meta["domain"] = domain
            meta["person_key"] = domain + "::" + meta.person_id.astype(str)
            parts.append(meta); values.append(x)
            csv, npz = unit_paths(unit, ridge)
            hashes[f"lambda{label(ridge)}/{unit}"] = {"metadata": sha(csv), "features": sha(npz),
                                                    "metadata_path": str(csv), "features_path": str(npz)}
        train_meta, xtrain = pd.concat(parts, ignore_index=True), np.concatenate(values)
        val_meta, xval, yval = load_unit(VALIDATION, ridge)
        if set(val_meta.split) != {"validation"}:
            raise ValueError("Human Rhythms validation split changed")
        keys = val_meta[["person_id", "origin_day", "y"]].reset_index(drop=True)
        if VALIDATION in baseline_keys and not keys.equals(baseline_keys[VALIDATION]):
            raise ValueError("Context-ridge candidates use different validation origins")
        baseline_keys[VALIDATION] = keys
        val_meta = val_meta.copy(); val_meta["domain"] = "Human-Rhythms"
        val_meta["person_key"] = "Human-Rhythms::" + val_meta.person_id.astype(str)
        if set(train_meta.person_key) & set(val_meta.person_key):
            raise ValueError("A Human Rhythms person crosses train and validation")
        csv, npz = unit_paths(VALIDATION, ridge)
        hashes[f"lambda{label(ridge)}/{VALIDATION}"] = {"metadata": sha(csv), "features": sha(npz),
            "metadata_path": str(csv), "features_path": str(npz)}
        datasets[ridge] = (train_meta, xtrain, val_meta, xval)
    return datasets, hashes


def run(output):
    datasets, hashes = load_datasets()
    train_meta = datasets[10.][0]
    counts = {domain: {"origins": len(part), "people": part.person_key.nunique(), "positive": int(part.y.sum())}
              for domain, part in train_meta.groupby("domain")}
    expected = {"RESILIENT": 1990, "BrisT1D-Open": 1238, "Human-Rhythms": 9039}
    if {k: v["origins"] for k, v in counts.items()} != expected:
        raise ValueError("The prespecified joint training population changed")
    config = {"version": "joint-human-hne-v1-20261110", "comparison_role": "RQ3 additional labeled-source-data multidomain training",
        "formula": "unchanged AdditiveWarning, spline=True, all296 H96/N36/E128/NE36 coordinates",
        "context_ridge_candidates": list(RIDGES), "C_candidates": list(CS),
        "person_key": "domain::person_id", "weighting": "inverse person origin count; normalize each domain base mass to1; then global class balance and training-prior logit correction",
        "training_counts": counts, "training_origins": len(train_meta), "training_people": train_meta.person_key.nunique(),
        "validation": "Human Rhythms validation only; no previous-domain scores enter candidate selection",
        "selection": "maximum validation AUROC; single-class negative logloss; exact ties smaller context_ridge then smaller C",
        "refit": False, "lambda10": "original feature arrays reused", "test_partition_read": False,
        "input_hashes": hashes, "additive_implementation_sha256": sha(ROOT / "src/hne_additive.py")}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "fixed_grid.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8"))["configuration"] != config:
        raise ValueError("Previously recorded multidomain grid differs")
    if not config_path.exists():
        write_json(config_path, {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": config})
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if sha(output / "model.joblib") != selection["model_sha256"]:
            raise ValueError("Previously frozen joint model changed")
        print("REUSE frozen multidomain selection", flush=True)
        return
    rows = []
    for ridge in RIDGES:
        train_meta, xtrain, val_meta, xval = datasets[ridge]
        ytrain, yval = train_meta.y.to_numpy(np.int8), val_meta.y.to_numpy(np.int8)
        people, domains = train_meta.person_key.to_numpy(), train_meta.domain.to_numpy()
        for c in CS:
            candidate = output / "candidates" / f"lambda{label(ridge)}_C{label(c)}"
            candidate.mkdir(parents=True, exist_ok=True)
            if (candidate / "metadata.json").exists():
                row = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
                if sha(candidate / "model.joblib") != row["model_sha256"]:
                    raise ValueError("A stored candidate model changed")
                rows.append(row); continue
            model = AdditiveWarning(C=c, spline=True, groups=GROUPS)
            model.fit(xtrain, ytrain, people, domains=domains)
            probability = model.predict_proba(xval)
            if not np.isfinite(probability).all():
                raise FloatingPointError("Nonfinite joint-model validation probability")
            frame = val_meta.copy(); frame["probability"] = probability
            frame.to_csv(candidate / "validation.csv", index=False)
            joblib.dump({"model": model, "context_ridge": ridge, "C": c,
                "feature_spec": {"kind": "pooled", "columns": list(range(296))}}, candidate / "model.joblib")
            row = {"context_ridge": ridge, "C": c, "validation_selection_score": validation_score(yval, probability),
                **score_description(yval, probability), "model_sha256": sha(candidate / "model.joblib"),
                "positive_multiplier": float(model.positive_multiplier)}
            write_json(candidate / "metadata.json", row); rows.append(row)
            print(json.dumps(row), flush=True)
    chosen = max(rows, key=lambda r: (r["validation_selection_score"], -r["context_ridge"], -r["C"]))
    candidate = output / "candidates" / f"lambda{label(chosen['context_ridge'])}_C{label(chosen['C'])}"
    shutil.copyfile(candidate / "model.joblib", output / "model.joblib")
    shutil.copyfile(candidate / "validation.csv", output / "validation.csv")
    model = joblib.load(output / "model.joblib")["model"]
    train_meta, xtrain, val_meta, _ = datasets[chosen["context_ridge"]]
    frame = train_meta.copy(); frame["probability"] = model.predict_proba(xtrain)
    frame.to_csv(output / "train.csv", index=False)
    validation = pd.read_csv(output / "validation.csv")
    negatives = validation.loc[validation.y == 0, "probability"].to_numpy()
    threshold = float(np.quantile(negatives, .9, method="linear")) if len(negatives) else None
    # Record the actual order of balancing, including the post-class-weight mass.
    weight, multiplier = person_weights(train_meta.person_key.to_numpy(), train_meta.y.to_numpy(), train_meta.domain.to_numpy())
    post_class_mass = {d: float(weight[train_meta.domain.eq(d)].sum()/weight.sum()) for d in counts}
    selection = {"selected_context_ridge": chosen["context_ridge"], "selected_C": chosen["C"],
        "validation_selection_score": chosen["validation_selection_score"], "validation": score_description(validation.y.to_numpy(), validation.probability.to_numpy()),
        "threshold_val_negative_p90": threshold, "comparison_role": config["comparison_role"],
        "training_counts": counts, "training_origins": len(train_meta), "training_people": train_meta.person_key.nunique(),
        "validation_people": val_meta.person_key.nunique(), "domain_base_mass": {d: 1. for d in counts},
        "post_class_balance_domain_mass_fraction": post_class_mass, "positive_multiplier": float(multiplier),
        "model_sha256": sha(output / "model.joblib"), "config_sha256": sha(config_path),
        "train_scores_sha256": sha(output / "train.csv"), "validation_scores_sha256": sha(output / "validation.csv"),
        "candidate_models": rows, "fit": "selected candidate copied; no refit",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "test_partition_read": False}
    pd.DataFrame(rows).to_csv(output / "candidates.csv", index=False)
    write_json(selection_path, selection)
    print(json.dumps({"frozen": True, "lambda": chosen["context_ridge"], "C": chosen["C"], "validation": selection["validation"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/multidomain_hne/Human-Rhythms")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)


if __name__ == "__main__":
    main()
