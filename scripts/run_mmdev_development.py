"""Cache each origin's past-only MM-DEV score once, then export all person folds.

MM-DEV-causal-median keeps the author's ISOMAP/S-map/eigenvalue chain with a
history-only median missing-data adapter. Native scores have a fixed positive
risk direction. The separately named LR row learns only a supervised scalar
readout, with C selected on validation people and no native score sign tuning.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mmdev import MODEL_NAME, MMDEVConfig, metadata, score_window, window_digest
from src.hne_additive import person_weights, validation_score

KEY = ["person_id", "origin_day", "issue_time_hours"]


def _worker(window):
    with threadpool_limits(limits=1):
        return score_window(window)


def _read_split(folder, fold, split):
    return pd.read_csv(folder / f"fold{fold}_{split}.csv", dtype={"person_id": str})


def _all_origins(folder):
    archive = np.load(folder / "fold0.npz")
    frames, windows = [], []
    for split in ("train", "validation", "test"):
        frame = _read_split(folder, 0, split)
        values = archive[f"{split}_x"]
        if len(frame) != len(values) or not np.array_equal(frame.y.to_numpy(), archive[f"{split}_y"]):
            raise ValueError("Window/label order differs from split metadata")
        frames.append(frame)
        windows.extend(values)
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(KEY).any():
        raise ValueError("Expected exactly one initial history per origin")
    frame["input_sha256"] = [window_digest(window) for window in windows]
    return frame, windows


def _cache(dataset, workers):
    source = ROOT / "data" / "development" / dataset
    destination = ROOT / "results" / "development" / dataset / MODEL_NAME
    destination.mkdir(parents=True, exist_ok=True)
    frame, windows = _all_origins(source)
    definition = metadata()
    code_hash = hashlib.sha256((ROOT / "src" / "mmdev.py").read_bytes()).hexdigest()
    fingerprint = hashlib.sha256((code_hash + "".join(frame.input_sha256) +
                                  json.dumps(asdict(MMDEVConfig()), sort_keys=True)).encode()).hexdigest()
    cache_path, info_path = destination / "origin_scores.csv", destination / "metadata.json"
    if cache_path.exists() and info_path.exists():
        existing = json.loads(info_path.read_text())
        if existing.get("cache_fingerprint") == fingerprint:
            cached = pd.read_csv(cache_path, dtype={"person_id": str})
            if len(cached) == len(frame):
                return source, destination, cached, existing
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(_worker, windows, chunksize=8), 1):
            results.append(result)
            if index % 100 == 0 or index == len(windows):
                print(json.dumps(dict(dataset=dataset, computed_origins=index, total_origins=len(windows),
                                      elapsed_seconds=round(time.time() - started, 1))), flush=True)
    cache = pd.concat((frame, pd.DataFrame(results)), axis=1)
    cache.to_csv(cache_path, index=False)
    info = dict(**definition, dataset=dataset, cache_fingerprint=fingerprint,
                source_code_sha256=code_hash, unique_origins=len(cache),
                finite_scores=int(np.isfinite(cache.score).sum()),
                reasons={str(k): int(v) for k, v in cache.reason.value_counts().items()},
                mean_imputed_fraction=float(cache.imputed_fraction.mean()),
                compute_seconds=time.time() - started, workers=workers)
    info_path.write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    return source, destination, cache, info


def _fold_frame(source, fold, split, cache):
    frame = _read_split(source, fold, split)
    # Ensure the shared cached history really is fold-independent; a fold-fitted
    # preprocessing difference must never be hidden by origin-level caching.
    values = np.load(source / f"fold{fold}.npz")[f"{split}_x"]
    frame["input_sha256"] = [window_digest(x) for x in values]
    detail = cache[KEY + [c for c in cache.columns if c not in frame.columns]]
    result = frame.merge(detail, on=KEY, how="left", validate="one_to_one")
    expected = frame[KEY + ["input_sha256"]].merge(
        cache[KEY + ["input_sha256"]], on=KEY, suffixes=("_fold", "_cache"), validate="one_to_one")
    if not expected.input_sha256_fold.eq(expected.input_sha256_cache).all():
        raise ValueError("Fold-specific histories cannot reuse this origin cache")
    if result.reason.isna().any():
        raise ValueError("Missing origin cache record")
    return result


def _metrics(frame, column, probability=False):
    good = np.isfinite(frame[column].to_numpy())
    y, score = frame.y.to_numpy()[good], frame[column].to_numpy()[good]
    result = dict(n_total=len(frame), n=int(good.sum()), positives=int(y.sum()),
                  auroc=float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
                  ap=float(average_precision_score(y, score)) if y.sum() else None)
    if probability:
        result["brier"] = float(brier_score_loss(y, score)) if len(y) else None
    return result


def _supervised(source, dataset, fold, frames):
    destination = ROOT / "results" / "development" / dataset / (MODEL_NAME + "-LR") / f"fold{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    train, validation, held = (frames[s] for s in ("train", "validation", "test"))
    fit_rows = np.isfinite(train.score.to_numpy())
    val_rows = np.isfinite(validation.score.to_numpy())
    y = train.y.to_numpy()[fit_rows]
    info = dict(model=MODEL_NAME + "-LR", dataset=dataset, fold=fold,
                input="One native MM-DEV-causal-median score only; no missingness/quality covariates",
                trials=[], train_scored_origins=int(fit_rows.sum()))
    predictions = {}
    fitted = None
    if not len(y):
        info.update(status="no_scored_training_origins", chosen=None)
        predictions = {split: np.full(len(frame), np.nan) for split, frame in frames.items()}
    elif len(np.unique(y)) == 1:
        prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        info.update(status="single_training_class", chosen=None, training_prior=prior)
        predictions = {split: np.where(np.isfinite(frame.score), prior, np.nan)
                       for split, frame in frames.items()}
        fitted = {"constant_probability": prior}
    else:
        scaler = StandardScaler().fit(train.loc[fit_rows, ["score"]])
        xtrain = scaler.transform(train.loc[fit_rows, ["score"]])
        weights, positive_multiplier = person_weights(train.person_id.to_numpy()[fit_rows], y)
        best_score, best = -np.inf, None
        for C in (0.01, 0.1, 1.0):
            model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            model.fit(xtrain, y, sample_weight=weights)
            if val_rows.any():
                p = expit(model.decision_function(scaler.transform(validation.loc[val_rows, ["score"]]))
                          - np.log(positive_multiplier))
                metric = validation_score(validation.y.to_numpy()[val_rows], p)
            else:
                metric = None
            info["trials"].append(dict(C=C, validation=metric))
            if best is None or (metric is not None and metric > best_score):
                best, best_C = model, C
                best_score = metric if metric is not None else -np.inf
        info.update(status="trained", chosen={"C": best_C},
                    positive_multiplier=float(positive_multiplier),
                    validation_selection="AUROC if two classes; unbalanced log loss otherwise; smallest C if validation has no scored origins")
        fitted = dict(model=best, scaler=scaler, positive_multiplier=float(positive_multiplier))
        for split, frame in frames.items():
            good = np.isfinite(frame.score.to_numpy())
            p = np.full(len(frame), np.nan)
            if good.any():
                p[good] = expit(best.decision_function(scaler.transform(frame.loc[good, ["score"]]))
                                - np.log(positive_multiplier))
            predictions[split] = p
    for split, frame in frames.items():
        result = frame.copy()
        result["probability"] = predictions[split]
        result.to_csv(destination / ("held_person.csv" if split == "test" else f"{split}.csv"), index=False)
    info["validation"] = _metrics(validation.assign(probability=predictions["validation"]), "probability", True)
    info["held_person"] = _metrics(held.assign(probability=predictions["test"]), "probability", True)
    (destination / "metrics.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    if fitted is not None:
        with (destination / "model.pkl").open("wb") as stream:
            pickle.dump(fitted, stream)


def run(datasets, workers, supervised=True):
    for dataset in datasets:
        source, destination, cache, info = _cache(dataset, workers)
        for fold in range(5):
            folder = destination / f"fold{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            frames = {split: _fold_frame(source, fold, split, cache)
                      for split in ("train", "validation", "test")}
            for split, frame in frames.items():
                frame.to_csv(folder / ("held_person.csv" if split == "test" else f"{split}.csv"), index=False)
            result = dict(model=MODEL_NAME, dataset=dataset, fold=fold, score_column="score",
                          native_direction="larger_is_higher_risk_fixed_before_evaluation",
                          validation=_metrics(frames["validation"], "score"),
                          held_person=_metrics(frames["test"], "score"),
                          cache_fingerprint=info["cache_fingerprint"])
            (folder / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            if supervised:
                _supervised(source, dataset, fold, frames)
        print(json.dumps(dict(dataset=dataset, complete=True, unique_origins=info["unique_origins"],
                              finite_scores=info["finite_scores"], reasons=info["reasons"],
                              compute_seconds=round(info["compute_seconds"], 2))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["RESILIENT", "BrisT1D-Open"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    with threadpool_limits(limits=1):
        run(args.datasets, args.workers, not args.native_only)
