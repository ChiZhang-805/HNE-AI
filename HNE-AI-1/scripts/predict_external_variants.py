"""Score the two frozen secondary models on the completed external origins.

No model fitting, parameter selection, threshold selection, or test metrics are
performed here. Family pooling is not part of this frozen evaluation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings
from zipfile import BadZipFile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_context_grid import GROUPS, sha, write_json
from src.hne_representation import Sampling, encode_person

BASELINE = ROOT / "data/features/human_rhythms/test"
SOURCE = ROOT / "data/human_rhythms_study/test"
FEATURE = ROOT / "data/features/context_grid/lambda0p1/Human-Rhythms_test"
RESULTS = ROOT / "results/human_rhythms"
MODELS = {
    "HNE-context-grid": ROOT / "results/context_grid/Human-Rhythms",
    "MultiDomain-HNE": ROOT / "results/multidomain_hne/Human-Rhythms",
}
RIDGE = .1
KEYS = ["person_id", "origin_day"]


def verify_freeze():
    manifest = json.loads((RESULTS / "frozen_models.json").read_text(encoding="utf-8"))

    def frozen(path):
        key = path.relative_to(ROOT).as_posix()
        if key not in manifest["files"] or sha(path) != manifest["files"][key]:
            raise ValueError(f"File not equal to the global frozen artifact: {key}")

    frozen(ROOT / "configs/external_evaluation.json")
    frozen(ROOT / "configs/daily_protocol.json")
    for name, folder in MODELS.items():
        frozen(folder / "model.joblib")
        selected = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        if selected["selected_context_ridge"] != RIDGE or selected["selected_C"] != .01:
            raise ValueError(f"Unexpected frozen candidate for {name}")
        if sha(folder / "model.joblib") != selected["model_sha256"]:
            raise ValueError("Frozen selection/model mismatch")
        if name == "HNE-context-grid":
            frozen(folder / "freeze_manifest.json")
            local = json.loads((folder / "freeze_manifest.json").read_text())
            if sha(folder / "selection.json") != local["selection_sha256"]:
                raise ValueError("Frozen context-grid selection changed")
        else:
            frozen(folder / "selection.json")
            if sha(folder / "validation.csv") != selected["validation_scores_sha256"]:
                raise ValueError("Frozen multidomain validation scores changed")
    grid = json.loads((ROOT / "results/context_grid/fixed_grid.json").read_text())["configuration"]
    for name, digest in grid["code_hashes"].items():
        if sha(ROOT / "src" / name) != digest:
            raise ValueError(f"Feature/risk implementation changed since training: {name}")
    return manifest


def wait_for_baseline(seconds):
    start = time.monotonic()
    while True:
        try:
            if not (SOURCE / "samples.csv").exists():
                raise FileNotFoundError("Main test sample construction is unfinished")
            meta = pd.read_csv(BASELINE.with_suffix(".csv"), dtype={"person_id": str})
            with np.load(BASELINE.with_suffix(".npz"), allow_pickle=False) as z:
                pooled, y, groups, fold = z["pooled"], z["y"], z["groups"], z["fold"]
            break
        except (FileNotFoundError, BadZipFile, EOFError, pd.errors.EmptyDataError):
            if time.monotonic() - start > seconds:
                raise TimeoutError("The main test feature artifact is not complete")
            print("WAIT main test metadata/features", flush=True)
            time.sleep(5)
    if meta.duplicated(KEYS).any() or set(meta.split) != {"test"}:
        raise ValueError("Duplicate origins or incorrect external partition")
    if pooled.shape != (len(meta), 296) or not np.array_equal(groups, GROUPS) or not np.array_equal(y, meta.y):
        raise ValueError("Main test metadata and features disagree")
    samples = pd.read_csv(SOURCE / "samples.csv", dtype={"person_id": str})
    if not meta[KEYS + ["y"]].equals(samples[KEYS + ["y"]]):
        raise ValueError("Main test samples and feature rows differ")
    return meta, pooled, y, fold


def worker_init():
    global _thread_limit
    _thread_limit = threadpool_limits(limits=1)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def encode_one(task):
    person, rows, cutoffs = task
    with np.load(SOURCE / f"{person}_daily.npz", allow_pickle=False) as z:
        states, contexts = z["normalized"], z["normalized_contexts"]
    if states.shape[1] != 10 or contexts.shape[1] != 5:
        raise ValueError("Unexpected frozen daily schema")
    nodes, pooled, groups = encode_person(states, contexts, cutoffs, Sampling.daily(), context_ridge=RIDGE)
    if not np.array_equal(groups, GROUPS):
        raise ValueError("Unexpected coordinate schema")
    return person, rows, nodes, pooled


def test_features(meta, baseline_pooled, y, fold, workers):
    if FEATURE.with_suffix(".npz").exists():
        saved_meta = pd.read_csv(FEATURE.with_suffix(".csv"), dtype={"person_id": str})
        with np.load(FEATURE.with_suffix(".npz"), allow_pickle=False) as z:
            pooled, saved_y, ridge = z["pooled"], z["y"], float(z["context_ridge"])
        if not saved_meta[KEYS].equals(meta[KEYS]) or not np.array_equal(y, saved_y) or ridge != RIDGE:
            raise ValueError("Existing frozen external feature origins differ")
        return pooled
    tasks = [(person, group.index.to_numpy(), group.origin_day.to_numpy() + 1)
             for person, group in meta.groupby("person_id", sort=False)]
    pooled = np.full((len(meta), 296), np.nan, np.float32)
    nodes = np.full((len(meta), 10, 50), np.nan, np.float32)
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        futures = [executor.submit(encode_one, task) for task in tasks]
        for future in as_completed(futures):
            person, rows, person_nodes, person_pooled = future.result()
            nodes[rows], pooled[rows] = person_nodes, person_pooled
            print(json.dumps({"features": person, "origins": len(rows)}), flush=True)
    raw = np.isin(GROUPS, ["H", "N"])
    if not np.array_equal(pooled[:, raw], baseline_pooled[:, raw], equal_nan=True):
        raise ValueError("Context ridge changed raw H/N coordinates")
    FEATURE.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(FEATURE.with_suffix(".csv"), index=False)
    np.savez_compressed(FEATURE.with_suffix(".npz"), nodes=nodes, pooled=pooled, groups=GROUPS,
                        y=y, fold=fold, split=meta.split.to_numpy(dtype=str), context_ridge=np.asarray(RIDGE))
    write_json(FEATURE.with_suffix(".json"), {"context_ridge": RIDGE, "origins": len(meta),
        "people": int(meta.person_id.nunique()), "main_feature_sha256": sha(BASELINE.with_suffix(".npz")),
        "main_metadata_sha256": sha(BASELINE.with_suffix(".csv")), "raw_H_N_identical": True,
        "labels": "copied unchanged from completed main test feature artifact",
        "feature_sha256": sha(FEATURE.with_suffix(".npz"))})
    return pooled


def predict(meta, pooled):
    align = meta.groupby("person_id", sort=False).size().rename("expected_origins").to_frame()
    for name, folder in MODELS.items():
        selected = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        bundle = joblib.load(folder / "model.joblib")
        if bundle["context_ridge"] != RIDGE or bundle["feature_spec"]["columns"] != list(range(296)):
            raise ValueError("Frozen model expects different coordinates")
        probability = bundle["model"].predict_proba(pooled)
        if not np.isfinite(probability).all():
            raise FloatingPointError("Nonfinite frozen-model predictions")
        out = RESULTS / name; out.mkdir(parents=True, exist_ok=True)
        frame = meta.copy(); frame["score"] = probability
        frame.to_csv(out / "test.csv", index=False)
        validation = pd.read_csv(folder / "validation.csv", dtype={"person_id": str})
        validation = validation.rename(columns={"probability": "score"})
        if set(validation.split) != {"validation"} or set(validation.person_id) & set(meta.person_id):
            raise ValueError("Validation and test identities overlap")
        validation.to_csv(out / "validation.csv", index=False)
        threshold = selected["threshold_val_negative_p90"]
        write_json(out / "selection.json", {"method": name,
            "selected_context_ridge": selected["selected_context_ridge"], "selected_C": selected["selected_C"],
            "threshold_val_negative_p90": threshold, "threshold_source": "unchanged frozen validation-negative P90",
            "source_model": str(folder / "model.joblib"), "source_model_sha256": selected["model_sha256"],
            "source_selection_sha256": sha(folder / "selection.json"),
            "source_validation_scores_sha256": sha(folder / "validation.csv"),
            "comparison_role": "parameter sensitivity" if name == "HNE-context-grid" else selected["comparison_role"],
            "fit_performed": False, "test_score_column": "score", "validation_score_column": "score"})
        if not frame[KEYS + ["y"]].equals(meta[KEYS + ["y"]]):
            raise ValueError("Prediction rows/labels changed")
        align[name + "_origins"] = frame.groupby("person_id").size()
        align[name + "_keys_and_labels_aligned"] = True
        print(json.dumps({"predicted": name, "people": int(frame.person_id.nunique()),
                          "origins": len(frame), "validation_origins": len(validation),
                          "frozen_val_p90": threshold}), flush=True)
    align.reset_index().to_csv(RESULTS / "external_variants_alignment.csv", index=False)
    write_json(RESULTS / "external_variants_completion.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "methods": list(MODELS),
        "context_ridge": RIDGE, "origins": len(meta), "people": int(meta.person_id.nunique()),
        "all_person_origin_keys_and_labels_aligned": True, "test_metrics_computed": False,
        "model_fitting_performed": False, "thresholds_retuned": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--wait-seconds", type=int, default=600)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    verify_freeze()
    meta, baseline, y, fold = wait_for_baseline(args.wait_seconds)
    with threadpool_limits(limits=1):
        pooled = test_features(meta, baseline, y, fold, args.workers)
        verify_freeze()
        predict(meta, pooled)
