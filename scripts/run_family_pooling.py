"""Finite family-pooling experiment on development and new-cohort train/val."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_context_grid as shared
from src.hne_family import pool_families, UNCHANGED_COLUMNS

FEATURES = ROOT / "data/features/family_pooling"
RESULTS = ROOT / "results/family_pooling"
RIDGES, CS, UNITS, DEVELOPMENT = shared.RIDGES, shared.CS, shared.UNITS, shared.DEVELOPMENT


def original_base(unit, ridge):
    return shared.sources(unit)[1] if ridge == 10 else shared.FEATURES / f"lambda{shared.label(ridge)}" / unit


def schema(unit):
    folder, baseline = shared.sources(unit)
    meta = pd.read_csv(baseline.with_suffix(".csv"), dtype={"person_id": str})
    result = None
    for person in meta.person_id.unique():
        with np.load(folder / f"{person}_daily.npz", allow_pickle=False) as z:
            current = {"state_names": z["state_names"].astype(str).tolist(),
                       "state_families": z["state_families"].astype(str).tolist()}
        if result is not None and current != result:
            raise ValueError(f"Schema changes between people in {unit}")
        result = current
    if len(result["state_names"]) != 10:
        raise ValueError("Unexpected state count")
    return result


def record_config():
    cfg = {
        "version": "schema-family-max-pooling-20261110-v1",
        "reason": "Test whether unequal numbers of related measurement fields dominate node pooling; shared family units match the already-fixed measurement schema.",
        "within_family": "finite signed maximum for each node descriptor; all missing stays NaN; no absolute-value transform",
        "between_family": "original pool_nodes: mean, population standard deviation, maximum, mean of largest two finite family values",
        "changed_coordinates": {"H": "0:96", "adjusted_state_E": "132:236"},
        "unchanged_coordinates": {"N": "96:132", "context_dynamics": "236:260", "NE": "260:296"},
        "families": "fixed source schema only, never learned or changed by reference support/outcomes",
        "context_ridge": list(RIDGES), "C": list(CS),
        "selection": "Validation AUROC or negative logloss if one class; exact ties smaller lambda then smaller C",
        "development": "Held fold k; validation fold (k+1)%5; train other three; held development scored only after joint lambda/C selection",
        "human_rhythms": "Only train and validation inputs; chosen train-fitted model, no refit; no test input or prediction mode",
        "schemas": {unit: schema(unit) for unit in UNITS},
        "input_hashes": {unit: {str(ridge): {"metadata": shared.sha(original_base(unit, ridge).with_suffix(".csv")),
                                             "features": shared.sha(original_base(unit, ridge).with_suffix(".npz"))}
                                for ridge in RIDGES} for unit in UNITS},
        "code_hashes": {name: shared.sha(ROOT / name) for name in
                        ("src/hne_family.py", "src/hne_features.py", "src/hne_additive.py", "scripts/run_context_grid.py")},
    }
    path = RESULTS / "fixed_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["configuration"] != cfg:
            raise ValueError("Family-pooling configuration or input changed")
    else:
        shared.write_json(path, {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": cfg})
    return cfg


def build_features(cfg):
    for unit in UNITS:
        families = cfg["schemas"][unit]["state_families"]
        for ridge in RIDGES:
            source = original_base(unit, ridge)
            destination = FEATURES / f"lambda{shared.label(ridge)}" / unit
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.with_suffix(".npz").exists():
                print(f"FEATURE_REUSE {unit} lambda={ridge}", flush=True)
                continue
            meta = pd.read_csv(source.with_suffix(".csv"), dtype={"person_id": str})
            with np.load(source.with_suffix(".npz"), allow_pickle=False) as z:
                nodes, old_pooled, groups, y, fold = z["nodes"], z["pooled"], z["groups"], z["y"], z["fold"]
            if not np.array_equal(groups, shared.GROUPS) or not np.array_equal(y, meta.y):
                raise ValueError("Source metadata or feature groups differ")
            pooled, family_nodes = pool_families(nodes, old_pooled, families)
            if not np.array_equal(pooled[:, UNCHANGED_COLUMNS], old_pooled[:, UNCHANGED_COLUMNS], equal_nan=True):
                raise ValueError("Family pooling unexpectedly altered relations or context dynamics")
            payload = dict(nodes=nodes, family_nodes=family_nodes, pooled=pooled, groups=groups, y=y, fold=fold,
                           state_families=np.asarray(families), family_names=np.unique(families), context_ridge=np.asarray(ridge))
            if "split" in meta:
                payload["split"] = meta.split.to_numpy(dtype=str)
            np.savez_compressed(destination.with_suffix(".npz"), **payload)
            meta.to_csv(destination.with_suffix(".csv"), index=False)
            info = {"unit": unit, "context_ridge": ridge, "origins": len(meta),
                    "family_names": np.unique(families).tolist(), "family_node_shape": list(family_nodes.shape),
                    "pooled_shape": list(pooled.shape), "N_context_NE_unchanged": True,
                    "source_path": str(source.with_suffix(".npz")),
                    "feature_sha256": shared.sha(destination.with_suffix(".npz"))}
            shared.write_json(destination.with_suffix(".json"), info)
            print(json.dumps({"feature_complete": info}), flush=True)


def load_unit(unit, ridge):
    path = FEATURES / f"lambda{shared.label(ridge)}" / unit
    meta = pd.read_csv(path.with_suffix(".csv"), dtype={"person_id": str})
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        x, y, groups = z["pooled"], z["y"], z["groups"]
    if x.shape != (len(meta), 296) or not np.array_equal(groups, shared.GROUPS) or not np.array_equal(meta.y, y):
        raise ValueError("Family feature schema disagrees")
    return meta, x, y


def select(name, datasets, train, val, output, cfg):
    best, selected = shared.candidates(name, datasets, train, val, output)
    bundle = joblib.load(output / "model.joblib")
    bundle["representation"] = "family_max_then_original_pool_nodes"
    bundle["family_rule"] = {key: cfg[key] for key in ("within_family", "between_family", "changed_coordinates", "unchanged_coordinates")}
    joblib.dump(bundle, output / "model.joblib")
    selected["representation"] = bundle["representation"]
    selected["model_sha256"] = shared.sha(output / "model.joblib")
    shared.write_json(output / "selection.json", selected)
    return best, selected


def fit_all(cfg):
    selections = []
    for unit in DEVELOPMENT:
        data = {r: load_unit(unit, r) for r in RIDGES}
        for fold in range(5):
            output = RESULTS / unit / f"fold{fold}"
            if (output / "metrics.json").exists():
                selections.append(json.loads((output / "metrics.json").read_text()))
                continue
            best, selected = select(f"{unit}/fold{fold}", data,
                lambda m: ~m.fold.isin([fold, (fold + 1) % 5]).to_numpy(),
                lambda m: m.fold.eq((fold + 1) % 5).to_numpy(), output, cfg)
            ridge, _, model, _ = best
            meta, x, y = data[ridge]; held = meta.fold.eq(fold).to_numpy()
            p = model.predict_proba(x[held])
            frame = meta[held].copy(); frame["probability"] = p
            frame.to_csv(output / "held_person.csv", index=False)
            selected.update(dataset=unit, fold=fold, held_development=shared.metrics(y[held], p))
            shared.write_json(output / "metrics.json", selected); selections.append(selected)
            print(json.dumps({"selected": unit, "fold": fold, "lambda": ridge, "C": selected["selected_C"],
                              "validation": selected["validation"], "held_development": selected["held_development"]}), flush=True)
    output = RESULTS / "Human-Rhythms"
    if not (output / "selection.json").exists():
        data = {}
        for ridge in RIDGES:
            tr, va = load_unit("Human-Rhythms_train", ridge), load_unit("Human-Rhythms_validation", ridge)
            data[ridge] = (pd.concat([tr[0], va[0]], ignore_index=True), np.concatenate([tr[1], va[1]]), np.concatenate([tr[2], va[2]]))
        _, selected = select("Human-Rhythms", data, lambda m: m.split.eq("train").to_numpy(),
                             lambda m: m.split.eq("validation").to_numpy(), output, cfg)
        shared.write_json(output / "freeze_manifest.json", {"model_sha256": selected["model_sha256"],
            "selection_sha256": shared.sha(output / "selection.json"), "configuration_sha256": shared.sha(RESULTS / "fixed_config.json"),
            "selected_context_ridge": selected["selected_context_ridge"], "selected_C": selected["selected_C"],
            "representation": selected["representation"], "feature_inputs": ["train", "validation"],
            "test_partition_read": False, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    summarize(selections)


def summarize(selections):
    rows = []
    for unit in DEVELOPMENT:
        scores = pd.concat([pd.read_csv(RESULTS / unit / f"fold{k}/held_person.csv", dtype={"person_id": str})
                            for k in range(5)], ignore_index=True)
        scores.to_csv(RESULTS / unit / "held_person.csv", index=False)
        common = pd.read_csv(ROOT / f"results/unified_daily/summary/{unit}_predictions.csv", dtype={"person_id": str},
                             usecols=["person_id", "origin_day", "primary_common"])
        scores = scores.merge(common, on=["person_id", "origin_day"], validate="one_to_one")
        for population in ("all", "common"):
            frame = scores if population == "all" else scores[scores.primary_common]
            aucs, weights = [], []
            for _, fold in frame.groupby("fold"):
                positive, negative = int(fold.y.sum()), int((fold.y == 0).sum())
                if positive and negative:
                    aucs.append(roc_auc_score(fold.y, fold.probability)); weights.append(positive * negative)
            rows.append({"dataset": unit, "population": population, "origins": len(frame), "positive": int(frame.y.sum()),
                         "fold_pair_weighted_auroc": float(np.average(aucs, weights=weights))})
    pd.DataFrame(rows).to_csv(RESULTS / "development_summary.csv", index=False)
    pd.DataFrame([{"dataset": s["dataset"], "fold": s["fold"], "context_ridge": s["selected_context_ridge"],
                   "C": s["selected_C"], "validation_auroc": s["validation"]["auroc"],
                   "held_development_auroc": s["held_development"]["auroc"]} for s in selections]).to_csv(
        RESULTS / "development_selections.csv", index=False)
    shared.write_json(RESULTS / "completion.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "grid": {"context_ridge": list(RIDGES), "C": list(CS)}, "development_summary": rows,
        "human_rhythms_selection": json.loads((RESULTS / "Human-Rhythms/selection.json").read_text()),
        "test_partition_read": False})
    print(json.dumps({"complete": rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["register", "features", "fit", "all"], default="all")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = record_config()
    with threadpool_limits(limits=1):
        if args.phase in {"features", "all"}:
            build_features(cfg)
        if args.phase in {"fit", "all"}:
            fit_all(cfg)
