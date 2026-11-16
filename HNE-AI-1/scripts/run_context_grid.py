"""Select reference-context shrinkage and the additive risk penalty on validation.

The fixed grid is recorded before feature generation. Human Rhythms has only
train/validation inputs here; this runner has no test-data path or scoring mode.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_additive import AdditiveWarning, validation_score
from src.hne_representation import Sampling, encode_person

FEATURES = ROOT / "data/features/context_grid"
RESULTS = ROOT / "results/context_grid"
RIDGES = (.1, 1., 10.)
CS = (.01, .1, 1.)
DEVELOPMENT = ("RESILIENT", "BrisT1D-Open")
UNITS = (*DEVELOPMENT, "Human-Rhythms_train", "Human-Rhythms_validation")
GROUPS = np.asarray(["H"] * 96 + ["N"] * 36 + ["E"] * 128 + ["NE"] * 36)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def label(value):
    return f"{value:g}".replace(".", "p")


def sources(unit):
    if unit in DEVELOPMENT:
        return ROOT / "data/daily_study" / unit, ROOT / "data/features/unified_daily" / unit
    split = unit.removeprefix("Human-Rhythms_")
    if split not in {"train", "validation"}:
        raise ValueError("Only Human Rhythms train/validation may be opened")
    return ROOT / "data/human_rhythms_study" / split, ROOT / "data/features/human_rhythms" / split


def record_grid():
    cfg = {
        "version": "context-reference-ridge-grid-20261110-v1",
        "context_ridge": list(RIDGES), "risk_C": list(CS),
        "reason": "Calibrate shrinkage of reference-only conditional context responses so expected environmental variation can be represented; unchanged mean-squared-error plus lambda times squared coefficient norm.",
        "representation": "same encode_person, daily 56-day reference, 8/16/32-day scales, 296 H/N/E/NE coordinates; original normalized arrays and all 10 state channels",
        "lambda10": "Reuse existing feature arrays without re-encoding",
        "selection": "Maximize validation AUROC, or negative logloss if one class; exact ties smaller context_ridge then smaller C",
        "development": "For each held fold k, validation=(k+1)%5; other three folds train. Score held development origins only after selecting lambda and C.",
        "new_cohort": "Train-fitted selected model; validation selects lambda and C; no train+validation refit; train/validation inputs only",
        "input_hashes": {unit: {"metadata": sha(sources(unit)[1].with_suffix(".csv")),
                               "baseline_features": sha(sources(unit)[1].with_suffix(".npz"))} for unit in UNITS},
        "code_hashes": {name: sha(ROOT / "src" / name) for name in
                        ("hne_representation.py", "hne_features.py", "hne_additive.py")},
    }
    path = RESULTS / "fixed_grid.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["configuration"] != cfg:
            raise ValueError("Existing fixed grid or inputs differ")
    else:
        write_json(path, {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                          "configuration": cfg})
    return cfg


def worker_init():
    global _thread_limit
    _thread_limit = threadpool_limits(limits=1)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def encode_one(task):
    unit, person, rows, cutoffs = task
    source, _ = sources(unit)
    start = time.monotonic()
    raw_path = source / f"{person}_daily.npz"
    with np.load(raw_path, allow_pickle=False) as z:
        states, contexts = z["normalized"], z["normalized_contexts"]
    if states.shape[1] != 10:
        raise ValueError(f"Unexpected state schema for {unit}/{person}: {states.shape}")
    for ridge in RIDGES[:-1]:
        target = FEATURES / f"lambda{label(ridge)}" / unit / "people" / f"{person}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with np.load(target, allow_pickle=False) as z:
                if not np.array_equal(rows, z["rows"]) or not np.array_equal(cutoffs, z["cutoffs"]):
                    raise ValueError("Cached person origins differ")
            continue
        nodes, pooled, groups = encode_person(states, contexts, cutoffs, Sampling.daily(), context_ridge=ridge)
        if not np.array_equal(groups, GROUPS):
            raise ValueError("Unexpected feature groups")
        np.savez_compressed(target, rows=rows, cutoffs=cutoffs, nodes=nodes, pooled=pooled,
                            groups=groups, source_sha256=np.asarray(sha(raw_path)))
    return {"unit": unit, "person": person, "origins": len(rows),
            "seconds": round(time.monotonic() - start, 2)}


def build_features(workers):
    for unit in UNITS:
        source, baseline = sources(unit)
        meta = pd.read_csv(baseline.with_suffix(".csv"), dtype={"person_id": str})
        if unit.startswith("Human-Rhythms_") and set(meta.split) != {unit.removeprefix("Human-Rhythms_")}:
            raise ValueError("Human Rhythms partition metadata mismatch")
        if all((FEATURES / f"lambda{label(r)}" / f"{unit}.npz").exists() for r in RIDGES[:-1]):
            print(f"FEATURE_REUSE {unit}", flush=True)
            continue
        tasks = [(unit, person, frame.index.to_numpy(), frame.origin_day.to_numpy() + 1)
                 for person, frame in meta.groupby("person_id", sort=False)]
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
            futures = [executor.submit(encode_one, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                print(json.dumps({"feature_person": index, "people": len(tasks), **future.result()}), flush=True)
        with np.load(baseline.with_suffix(".npz"), allow_pickle=False) as z:
            baseline_pooled, baseline_y = z["pooled"], z["y"]
            if not np.array_equal(z["groups"], GROUPS) or not np.array_equal(baseline_y, meta.y):
                raise ValueError("Existing feature metadata disagree")
        for ridge in RIDGES[:-1]:
            folder = FEATURES / f"lambda{label(ridge)}"
            pooled = np.full((len(meta), 296), np.nan, np.float32)
            nodes = np.full((len(meta), 10, 50), np.nan, np.float32)
            for _, person, rows, _ in tasks:
                with np.load(folder / unit / "people" / f"{person}.npz", allow_pickle=False) as z:
                    pooled[rows], nodes[rows] = z["pooled"], z["nodes"]
            raw = np.isin(GROUPS, ["H", "N"])
            if not np.array_equal(pooled[:, raw], baseline_pooled[:, raw], equal_nan=True):
                raise ValueError("Changing context ridge unexpectedly changed raw H/N coordinates")
            payload = dict(nodes=nodes, pooled=pooled, groups=GROUPS, y=baseline_y,
                           fold=meta.fold.to_numpy(), context_ridge=np.asarray(ridge))
            if "split" in meta:
                payload["split"] = meta.split.to_numpy(dtype=str)
            np.savez_compressed(folder / f"{unit}.npz", **payload)
            meta.to_csv(folder / f"{unit}.csv", index=False)
            changed = pooled[:, ~raw] - baseline_pooled[:, ~raw]
            write_json(folder / f"{unit}.json", {
                "unit": unit, "context_ridge": ridge, "origins": len(meta),
                "people": int(meta.person_id.nunique()), "pooled_shape": list(pooled.shape),
                "raw_H_N_identical_to_lambda10": True,
                "mean_absolute_E_NE_change_from_lambda10": float(np.nanmean(np.abs(changed))),
                "baseline_feature_path": str(baseline.with_suffix(".npz")),
                "features_sha256": sha(folder / f"{unit}.npz")})
        print(f"FEATURE_COMPLETE {unit} origins={len(meta)}", flush=True)


def load_unit(unit, ridge):
    baseline = sources(unit)[1]
    path = baseline if ridge == 10. else FEATURES / f"lambda{label(ridge)}" / unit
    meta = pd.read_csv(path.with_suffix(".csv"), dtype={"person_id": str})
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        pooled, y, groups = z["pooled"], z["y"], z["groups"]
    if pooled.shape != (len(meta), 296) or not np.array_equal(groups, GROUPS) or not np.array_equal(y, meta.y):
        raise ValueError(f"Feature schema mismatch for {unit}/{ridge}")
    return meta, pooled, y


def metrics(y, p):
    return {"origins": len(y), "positive": int(np.sum(y)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "ap": float(average_precision_score(y, p)) if np.any(y) else None}


def candidates(name, datasets, train_mask, val_mask, output):
    output.mkdir(parents=True, exist_ok=True)
    records, best, best_value = [], None, -np.inf
    for ridge in RIDGES:
        meta, x, y = datasets[ridge]
        train, val = train_mask(meta), val_mask(meta)
        if set(meta.loc[train, "person_id"]) & set(meta.loc[val, "person_id"]):
            raise ValueError("People cross train/validation")
        for c in CS:
            model = AdditiveWarning(C=c, spline=True, groups=GROUPS).fit(
                x[train], y[train], meta.loc[train, "person_id"].to_numpy())
            probability = model.predict_proba(x[val])
            score = validation_score(y[val], probability)
            row = {"unit": name, "context_ridge": ridge, "C": c,
                   "validation_selection_score": score, **metrics(y[val], probability)}
            records.append(row)
            candidate_path = output / f"lambda{label(ridge)}_C{label(c)}_validation.csv"
            frame = meta[val].copy(); frame["probability"] = probability
            frame.to_csv(candidate_path, index=False)
            if score > best_value:
                best_value = score
                best = (ridge, c, model, frame)
            print(json.dumps({"candidate": row}), flush=True)
    pd.DataFrame(records).to_csv(output / "candidates.csv", index=False)
    ridge, c, model, frame = best
    negative = frame.loc[frame.y.eq(0), "probability"].to_numpy()
    threshold = float(np.quantile(negative, .9, method="linear")) if len(negative) else None
    selection = {"unit": name, "selected_context_ridge": ridge, "selected_C": c,
                 "validation_selection_score": best_value, "candidates": records,
                 "threshold_val_negative_p90": threshold,
                 "fit": "training people only, selected by validation; no refit",
                 "validation": metrics(frame.y.to_numpy(), frame.probability.to_numpy()),
                 "selected_at_utc": datetime.now(timezone.utc).isoformat()}
    joblib.dump({"model": model, "context_ridge": ridge, "C": c,
                 "feature_spec": {"kind": "pooled", "columns": list(range(296))}}, output / "model.joblib")
    frame.to_csv(output / "validation.csv", index=False)
    selection["model_sha256"] = sha(output / "model.joblib")
    write_json(output / "selection.json", selection)
    return best, selection


def fit_all():
    selections = []
    for unit in DEVELOPMENT:
        datasets = {r: load_unit(unit, r) for r in RIDGES}
        for fold in range(5):
            output = RESULTS / unit / f"fold{fold}"
            if (output / "metrics.json").exists():
                selections.append(json.loads((output / "metrics.json").read_text()))
                continue
            best, selection = candidates(f"{unit}/fold{fold}", datasets,
                lambda m: ~m.fold.isin([fold, (fold + 1) % 5]).to_numpy(),
                lambda m: m.fold.eq((fold + 1) % 5).to_numpy(), output)
            ridge, _, model, _ = best
            meta, x, y = datasets[ridge]
            held = meta.fold.eq(fold).to_numpy()
            p = model.predict_proba(x[held])
            frame = meta[held].copy(); frame["probability"] = p
            frame.to_csv(output / "held_person.csv", index=False)
            selection["held_development"] = metrics(y[held], p)
            selection["dataset"], selection["fold"] = unit, fold
            write_json(output / "metrics.json", selection); selections.append(selection)
            print(json.dumps({"selected": unit, "fold": fold, "ridge": ridge,
                              "C": selection["selected_C"], "validation": selection["validation"],
                              "held_development": selection["held_development"]}), flush=True)
    output = RESULTS / "Human-Rhythms"
    if not (output / "selection.json").exists():
        datasets = {}
        for ridge in RIDGES:
            tr, va = load_unit("Human-Rhythms_train", ridge), load_unit("Human-Rhythms_validation", ridge)
            datasets[ridge] = (pd.concat([tr[0], va[0]], ignore_index=True),
                               np.concatenate([tr[1], va[1]]), np.concatenate([tr[2], va[2]]))
        _, selected = candidates("Human-Rhythms", datasets, lambda m: m.split.eq("train").to_numpy(),
                                  lambda m: m.split.eq("validation").to_numpy(), output)
        write_json(output / "freeze_manifest.json", {
            "selected_context_ridge": selected["selected_context_ridge"], "selected_C": selected["selected_C"],
            "selection_sha256": sha(output / "selection.json"), "model_sha256": selected["model_sha256"],
            "feature_inputs": ["train", "validation"], "test_partition_read": False,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    summarize(selections)


def summarize(selections):
    rows = []
    for unit in DEVELOPMENT:
        frames = [pd.read_csv(RESULTS / unit / f"fold{fold}/held_person.csv", dtype={"person_id": str})
                  for fold in range(5)]
        all_scores = pd.concat(frames, ignore_index=True)
        all_scores.to_csv(RESULTS / unit / "held_person.csv", index=False)
        common_path = ROOT / "results/unified_daily/summary" / f"{unit}_predictions.csv"
        common = pd.read_csv(common_path, dtype={"person_id": str},
                             usecols=["person_id", "origin_day", "primary_common"])
        all_scores = all_scores.merge(common, on=["person_id", "origin_day"], validate="one_to_one")
        for population in ("all", "common"):
            frame = all_scores if population == "all" else all_scores[all_scores.primary_common]
            aucs, weights = [], []
            for _, fold in frame.groupby("fold"):
                n_pos, n_neg = int(fold.y.sum()), int((fold.y == 0).sum())
                if n_pos and n_neg:
                    aucs.append(roc_auc_score(fold.y, fold.probability)); weights.append(n_pos * n_neg)
            rows.append({"dataset": unit, "population": population, "origins": len(frame),
                         "positive": int(frame.y.sum()), "fold_pair_weighted_auroc": float(np.average(aucs, weights=weights))})
    pd.DataFrame(rows).to_csv(RESULTS / "development_summary.csv", index=False)
    pd.DataFrame([{"dataset": s["dataset"], "fold": s["fold"], "context_ridge": s["selected_context_ridge"],
                   "C": s["selected_C"], "validation_auroc": s["validation"]["auroc"],
                   "held_development_auroc": s["held_development"]["auroc"]} for s in selections]).to_csv(
        RESULTS / "development_selections.csv", index=False)
    write_json(RESULTS / "completion.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "grid": {"context_ridge": list(RIDGES), "C": list(CS)}, "development_summary": rows,
        "human_rhythms_selection": json.loads((RESULTS / "Human-Rhythms/selection.json").read_text()),
        "test_partition_read": False})
    print(json.dumps({"complete": rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["register", "features", "fit", "all"], default="all")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    record_grid()
    with threadpool_limits(limits=1):
        if args.phase in {"features", "all"}:
            build_features(args.workers)
        if args.phase in {"fit", "all"}:
            fit_all()
