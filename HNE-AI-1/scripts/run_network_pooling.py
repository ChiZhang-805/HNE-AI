"""Finite network-aggregation comparison using existing daily inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_network_pooling import VARIANTS, NETWORK_COLUMNS, encode_networks
from src.hne_additive import AdditiveWarning, validation_score

FEATURES = ROOT / "data/features/network_pooling"
RESULTS = ROOT / "results/network_pooling"
DEVELOPMENT = ("RESILIENT", "BrisT1D-Open", "IFH-Affect")
DEFAULT_UNITS = (*DEVELOPMENT, "Human-Rhythms_train", "Human-Rhythms_validation")
GROUPS = np.array(["H"]*96 + ["N"]*36 + ["E"]*128 + ["NE"]*36)


def sources(unit):
    if unit in DEVELOPMENT[:2]:
        return ROOT / "data/daily_study" / unit, ROOT / "data/features/unified_daily" / unit
    if unit == "IFH-Affect":
        return ROOT / "data/expanded_development" / unit, ROOT / "data/features/expanded_development" / unit
    cohort, part = unit.rsplit("_", 1)
    if part not in ("train", "validation"):
        raise ValueError("Only train and validation are included in this development experiment")
    directory = {"Human-Rhythms": "human_rhythms", "College-Experience": "college"}[cohort]
    feature = "college_experience" if directory == "college" else directory
    return ROOT / f"data/{directory}_study" / part, ROOT / "data/features" / feature / part


def worker_init():
    global _threads
    _threads = threadpool_limits(limits=1)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def encode_one(task):
    unit, person, rows, cutoffs = task
    target = FEATURES / unit / "people" / f"{person}.npz"
    if target.exists():
        return
    source, _ = sources(unit)
    with np.load(source / f"{person}_daily.npz", allow_pickle=False) as data:
        networks = encode_networks(data["normalized"], data["normalized_contexts"],
                                   cutoffs, data["state_families"])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, rows=rows, **networks)


def build(units, workers):
    for unit in units:
        if all((FEATURES / name / f"{unit}.npz").exists() for name in VARIANTS):
            continue
        _, base = sources(unit)
        meta = pd.read_csv(base.with_suffix(".csv"), dtype={"person_id": str})
        tasks = [(unit, person, group.index.to_numpy(), group.origin_day.to_numpy()+1)
                 for person, group in meta.groupby("person_id", sort=False)]
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
            pending = [executor.submit(encode_one, task) for task in tasks]
            for count, future in enumerate(as_completed(pending), 1):
                future.result()
                if count % 20 == 0 or count == len(tasks):
                    print(f"NETWORK_FEATURES {unit} people={count}/{len(tasks)}", flush=True)
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as data:
            baseline, y = data["pooled"], data["y"]
        for name in VARIANTS:
            pooled = baseline.copy()
            for _, person, rows, _ in tasks:
                with np.load(FEATURES / unit / "people" / f"{person}.npz") as data:
                    pooled[np.ix_(rows, NETWORK_COLUMNS)] = data[name]
            destination = FEATURES / name
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / f"{unit}.npz", pooled=pooled, y=y, groups=GROUPS)
            meta.to_csv(destination / f"{unit}.csv", index=False)
        print(f"NETWORK_READY {unit} origins={len(meta)}", flush=True)


def load(unit, name):
    target = FEATURES / name / unit
    meta = pd.read_csv(target.with_suffix(".csv"), dtype={"person_id": str})
    with np.load(target.with_suffix(".npz")) as data:
        return meta, data["pooled"], data["y"]


def fit_one(meta, x, y, train, val, held, folder, cfg):
    if (folder / "selection.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    candidates, best_score, best, best_c = [], -np.inf, None, None
    for c in cfg["C_candidates"]:
        model = AdditiveWarning(C=c, spline=True, groups=GROUPS).fit(x[train], y[train],
            meta.loc[train, "person_id"].to_numpy())
        score = validation_score(y[val], model.predict_proba(x[val]))
        candidates.append({"C": c, "validation_selection_score": score})
        if score > best_score:
            best_score, best, best_c = score, model, c
    joblib.dump({"model": best, "feature_spec": {"kind": "pooled", "columns": list(range(296))},
                 "network_variant": folder.parent.name}, folder / "model.joblib")
    metrics = []
    for part, mask in (("validation", val), ("held_development", held)):
        if mask is None:
            continue
        frame = meta.loc[mask].copy()
        frame["probability"] = best.predict_proba(x[mask])
        frame.to_csv(folder / f"{part}.csv", index=False)
        metrics.append({"part": part, "origins": len(frame), "positive": int(frame.y.sum()),
                        "auroc": float(roc_auc_score(frame.y, frame.probability)),
                        "ap": float(average_precision_score(frame.y, frame.probability))})
    selection = {"C": best_c, "candidates": candidates, "metrics": metrics,
                 "selected_train_model": True, "context_ridge": 10}
    (folder / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({"fit": str(folder.relative_to(RESULTS)), **selection}), flush=True)


def fit(units, cfg):
    for name in VARIANTS:
        for unit in DEVELOPMENT:
            if unit not in units:
                continue
            meta, x, y = load(unit, name)
            for fold in range(5):
                fit_one(meta, x, y, ~meta.fold.isin([fold, (fold+1)%5]).to_numpy(),
                        meta.fold.eq((fold+1)%5).to_numpy(), meta.fold.eq(fold).to_numpy(),
                        RESULTS / unit / name / f"fold{fold}", cfg)
        for cohort in ("Human-Rhythms", "College-Experience"):
            if not all(f"{cohort}_{part}" in units for part in ("train", "validation")):
                continue
            parts = [load(f"{cohort}_{part}", name) for part in ("train", "validation")]
            meta = pd.concat([p[0] for p in parts], ignore_index=True)
            x, y = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
            fit_one(meta, x, y, meta.split.eq("train").to_numpy(), meta.split.eq("validation").to_numpy(),
                    None, RESULTS / cohort / name / "train", cfg)
    rows = []
    for unit in units:
        if unit not in DEVELOPMENT:
            continue
        for name in VARIANTS:
            parts = [pd.read_csv(RESULTS / unit / name / f"fold{k}" / "held_development.csv",
                                 dtype={"person_id": str}) for k in range(5)]
            frame = pd.concat(parts, ignore_index=True)
            frame.to_csv(RESULTS / unit / name / "held_development.csv", index=False)
            within = [roc_auc_score(g.y, g.probability) for _, g in frame.groupby("person_id") if g.y.nunique() == 2]
            fold_auc, fold_pairs = [], []
            for _, group in frame.groupby("fold"):
                if group.y.nunique() == 2:
                    fold_auc.append(roc_auc_score(group.y, group.probability))
                    fold_pairs.append(int(group.y.sum()) * int((group.y == 0).sum()))
            rows.append({"dataset": unit, "method": name, "origins": len(frame),
                "pooled_auroc": roc_auc_score(frame.y, frame.probability),
                "fold_pair_weighted_auroc": np.average(fold_auc, weights=fold_pairs),
                "within_person_auroc": np.mean(within),
                "ap": average_precision_score(frame.y, frame.probability)})
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS / "development_metrics.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", nargs="+", default=DEFAULT_UNITS)
    parser.add_argument("--phase", choices=("features", "fit", "all"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/network_pooling.json").read_text(encoding="utf-8"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "configuration.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if args.phase in ("features", "all"):
        build(args.units, args.workers)
    if args.phase in ("fit", "all"):
        with threadpool_limits(limits=2):
            fit(args.units, cfg)


if __name__ == "__main__":
    main()
