"""Train/validation freeze and separately authorized Human Rhythms test scoring.

The fit phase opens only train and validation feature files. Run predict only
after the study coordinator has frozen every planned method and opened testing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_additive import AdditiveWarning, BoostedWarning, history_features, validation_score
from src.hne_coordinates import coordinate_mask
from src.dnb import score_window as dnb_score_window
from src.mmdev import MMDEVConfig, score_window as mmdev_score_window
from scripts.run_transfer import sha, write_json

LEARNED = ["HNE-additive", "HNE-history", "HNE-no-network", "HNE-no-environment",
           "HNE-context-append", "History-LR", "History-HGB", "HNE-HGB", "Missingness-LR"]
NATIVE = ["DNB", "MM-DEV-daily"]
ZERO_SHOT = ["ZeroShot-HNE-additive", "ZeroShot-HNE-history"]
METHODS = LEARNED + NATIVE + ZERO_SHOT
SCALES = (8, 16, 32)
MMDEV = MMDEVConfig(history_hours=32, smap_window=24, embedding_max=6)
STATISTICS = ("auroc", "within_person_auroc", "ap", "tpr_val_p90", "fpr_val_p90")


def configuration():
    return {"version": "human-rhythms-external-v1-20261110", "methods": METHODS,
        "C_candidates": [.01, .1, 1.], "seed": 20261110,
        "history_scales_days": list(SCALES), "expected_pooled": 296,
        "expected_nodes": [10, 50], "expected_window": [32, 15],
        "fit_partitions": ["train", "validation"], "selection": "maximum validation AUROC; single-class validation negative logloss; exact ties smaller C",
        "final_fit": "selected train-fitted model; no train+validation refit",
        "weighting": "equal person base weight then class balance; training-prior logit correction",
        "native_score_direction": "higher score is greater warning risk; never selected from validation or test",
        "dnb": {"history_days": 32, "minimum_joint_observations": 12, "input": "10 state coordinates"},
        "mmdev": {"parameters": asdict(MMDEV), "input": "all10 state and5 context coordinates",
            "adaptation": "past-only median-filled ISOMAP/S-map, daily32 history and24 local window, E1..6; retain undefined scores as NaN"},
        "threshold": "linear90th percentile of finite validation negative scores; test alerts use strictly greater than threshold",
        "zero_shot": "frozen earlier Humans source risk mappings; no new-cohort model training or C selection; P90 threshold calibration uses new-cohort validation negatives",
        "test_bootstrap": {"replicates": 2000, "seed": 20261110, "unit": "whole person", "interval": "percentile95"},
        "common_population": "intersection of finite scores across every method managed by this runner",
        "ap_definition": "average precision, not trapezoidal integration",
        "within_person_definition": "equal-person mean AUROC among people with both classes",
        "primary_comparison": "paired common-population cluster intervals; all-population metrics retain each method's actual coverage"}


def core_hashes():
    return {name: sha(ROOT / "src" / name) for name in
        ("hne_additive.py", "hne_features.py", "hne_coordinates.py", "dnb.py", "mmdev.py")}


def load_split(feature_dir, split, expected):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown partition")
    csv_path, array_path = feature_dir / f"{split}.csv", feature_dir / f"{split}.npz"
    meta = pd.read_csv(csv_path, dtype={"person_id": str})
    if "split" in meta and set(meta.split) != {split}:
        raise ValueError(f"Wrong split identities in {csv_path}")
    if "split" not in meta:
        meta["split"] = split
    if meta.duplicated(["person_id", "origin_day"]).any():
        raise ValueError("Repeated person/origin rows")
    with np.load(array_path, allow_pickle=False) as z:
        pooled, windows, y, groups = z["pooled"], z["windows"], z["y"], z["groups"]
        node_shape = z["nodes"].shape
    if pooled.shape != (len(meta), expected["expected_pooled"]):
        raise ValueError("Wrong pooled feature schema")
    if node_shape != (len(meta), *expected["expected_nodes"]) or windows.shape != (len(meta), *expected["expected_window"]):
        raise ValueError("Wrong history/node schema")
    expected_groups = np.asarray(["H"]*96+["N"]*36+["E"]*128+["NE"]*36)
    if not np.array_equal(groups, expected_groups) or not np.array_equal(meta.y.to_numpy(), y):
        raise ValueError("Feature groups or labels disagree with metadata")
    return {"meta": meta, "pooled": pooled, "windows": windows, "y": y.astype(np.int8),
            "groups": groups, "people": meta.person_id.to_numpy(), "state_dimension": node_shape[1],
            "hashes": {"metadata": sha(csv_path), "features": sha(array_path)}}


def feature_spec(method, groups):
    if method == "Missingness-LR":
        return {"kind": "missingness", "scales": list(SCALES)}
    if method.startswith("History"):
        return {"kind": "history", "scales": list(SCALES)}
    keep = coordinate_mask(method)
    if method == "HNE-history":
        keep = groups == "H"
    elif method == "HNE-no-network":
        keep = np.isin(groups, ["H", "E"])
    elif method == "HNE-no-environment":
        keep = np.isin(groups, ["H", "N"])
    return {"kind": "pooled", "columns": np.flatnonzero(keep).tolist()}


def make_features(data, spec):
    if spec["kind"] == "pooled":
        return data["pooled"][:, spec["columns"]]
    if spec["kind"] == "missingness":
        return np.concatenate([np.isfinite(data["windows"][:, -s:]).mean(1) for s in spec["scales"]], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return history_features(data["windows"], scales=spec["scales"])


def native_worker_init():
    global _native_thread_limit
    _native_thread_limit = threadpool_limits(limits=1)


def native_one(task):
    method, window, state_dimension = task
    if method == "DNB":
        return {"score": dnb_score_window(window[-32:, :state_dimension], minimum=12)}
    return mmdev_score_window(window, MMDEV)


def native_scores(data, method, workers):
    tasks = ((method, window, data["state_dimension"]) for window in data["windows"])
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=native_worker_init) as executor:
            details = list(executor.map(native_one, tasks, chunksize=32))
    else:
        details = list(map(native_one, tasks))
    return pd.DataFrame(details)


def score_description(y, score):
    valid = np.isfinite(score)
    y, score = np.asarray(y)[valid], np.asarray(score)[valid]
    return {"origins": len(y), "positive": int(y.sum()),
        "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else None,
        "ap": average_precision_score(y, score) if y.sum() else None}


def save_partition(folder, split, data, score, detail=None):
    frame = data["meta"].copy(); frame["score"] = score
    if detail is not None:
        for column in detail:
            if column != "score":
                frame[column] = detail[column].to_numpy()
    frame.to_csv(folder / f"{split}.csv", index=False)


def fit(feature_dir, output, cfg, workers):
    # These are the only feature partitions opened by this phase.
    train = load_split(feature_dir, "train", cfg)
    validation = load_split(feature_dir, "validation", cfg)
    temporal = cfg.get("partition_unit") == "time"
    if not temporal and set(train["people"]) & set(validation["people"]):
        raise ValueError("A person crosses train and validation")
    if temporal and train["meta"].future_end_date.max() >= validation["meta"].issue_time.min()[:10]:
        raise ValueError("Training outcomes reach the validation issue period")
    if len(np.unique(train["y"])) != 2:
        raise ValueError("Training population must contain both classes")
    identity = {"config": cfg, "train": train["hashes"], "validation": validation["hashes"], "core": core_hashes()}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "fit_config.json").exists() and json.loads((output / "fit_config.json").read_text(encoding="utf-8")) != identity:
        raise ValueError("Existing fit configuration differs; retain its frozen outputs")
    write_json(output / "fit_config.json", identity)
    records = {}
    cache = {}
    for method in METHODS:
        folder = output / method; folder.mkdir(parents=True, exist_ok=True)
        existing = folder / "selection.json"
        if existing.exists():
            saved = json.loads(existing.read_text(encoding="utf-8"))
            if saved["identity"] != identity:
                raise ValueError(f"Existing frozen method has different inputs/config: {method}")
            if any(not (folder / f"{s}.csv").exists() for s in ("train", "validation")):
                raise ValueError(f"Incomplete frozen outputs: {method}")
            if saved["model_sha256"] and sha(folder / "model.joblib") != saved["model_sha256"]:
                raise ValueError(f"Frozen model file changed: {method}")
            records[method] = saved
            print(f"REUSE {method}", flush=True)
            continue
        chosen, candidate_results, native_detail, source_info = None, [], {}, None
        if method in LEARNED:
            spec = feature_spec(method, train["groups"])
            cache_key = json.dumps(spec, sort_keys=True)
            if cache_key not in cache:
                cache[cache_key] = (make_features(train, spec), make_features(validation, spec))
            xtrain, xval = cache[cache_key]
            selected_groups = train["groups"][spec["columns"]] if spec["kind"] == "pooled" else None
            best, best_value = None, -np.inf
            for c in cfg["C_candidates"]:
                model = BoostedWarning(C=c, seed=cfg["seed"]) if method.endswith("HGB") else AdditiveWarning(
                    C=c, spline=method.startswith("HNE"), groups=selected_groups)
                model.fit(xtrain, train["y"], train["people"])
                score = model.predict_proba(xval)
                value = validation_score(validation["y"], score)
                candidate_results.append({"C": c, "validation_selection_score": value,
                    **score_description(validation["y"], score)})
                if value > best_value:
                    chosen, best, best_value = c, model, value
            scores = {"train": best.predict_proba(xtrain), "validation": best.predict_proba(xval)}
            joblib.dump({"model": best, "feature_spec": spec}, folder / "model.joblib")
        elif method in NATIVE:
            scores = {}
            for split, data in (("train", train), ("validation", validation)):
                native_detail[split] = native_scores(data, method, workers)
                scores[split] = native_detail[split].score.to_numpy()
        else:
            source_method = method.removeprefix("ZeroShot-")
            source_folder = ROOT / "results/transfer/source_models/Humans" / source_method
            source_path = source_folder / "model.joblib"
            source_config = json.loads((source_folder / "source_config.json").read_text(encoding="utf-8"))
            if sha(source_path) != source_config["model_sha256"]:
                raise ValueError("Earlier frozen source model has changed")
            source = joblib.load(source_path)
            spec = {"kind": "pooled", "columns": source["columns"].tolist()}
            scores = {split: source["model"].predict_proba(make_features(data, spec))
                      for split, data in (("train", train), ("validation", validation))}
            joblib.dump({"model": source["model"], "feature_spec": spec}, folder / "model.joblib")
            source_info = {"path": str(source_path), "sha256": sha(source_path), "source_selected_C": source_config["selected_C"],
                           "new_cohort_model_fit": False}
        if method not in NATIVE and any(not np.isfinite(p).all() for p in scores.values()):
            raise FloatingPointError(f"Nonfinite learned probability: {method}")
        for split, data in (("train", train), ("validation", validation)):
            save_partition(folder, split, data, scores[split], native_detail.get(split))
        negative = scores["validation"][(validation["y"] == 0) & np.isfinite(scores["validation"])]
        threshold = float(np.quantile(negative, .9, method="linear")) if len(negative) else None
        saved = {"identity": identity, "method": method, "selected_C": chosen, "candidates": candidate_results,
            "score_is_probability": method not in NATIVE,
            "score_semantics": "native uncalibrated dynamical score" if method in NATIVE else "source probability used as target risk score" if method in ZERO_SHOT else "train-fitted probability",
            "source_model": source_info, "threshold_val_negative_p90": threshold,
            "threshold_validation_negative_count": len(negative),
            "train": score_description(train["y"], scores["train"]),
            "validation": score_description(validation["y"], scores["validation"]),
            "train_people": sorted(set(train["people"])), "validation_people": sorted(set(validation["people"])),
            "model_sha256": sha(folder / "model.joblib") if method not in NATIVE else None,
            "train_scores_sha256": sha(folder / "train.csv"), "validation_scores_sha256": sha(folder / "validation.csv"),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(existing, saved); records[method] = saved
        print(json.dumps({"method": method, "C": chosen, "validation": saved["validation"], "val_negative_p90": threshold}), flush=True)
    write_json(output / "freeze_manifest.json", {"identity": identity, "managed_methods": METHODS,
        "method_selection_sha256": {m: sha(output / m / "selection.json") for m in METHODS},
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "test_partition_read": False})
    pd.DataFrame([{"method": m, "selected_C": r["selected_C"], **r["validation"],
                   "val_negative_p90": r["threshold_val_negative_p90"]} for m, r in records.items()]).to_csv(output / "validation_summary.csv", index=False)


def cluster_statistics(y, scores, people, threshold, draws):
    """Tie-exact weighted ROC/AP in bootstrap chunks; no quadratic origin matrix."""
    y, scores, people = np.asarray(y), np.asarray(scores), np.asarray(people)
    identities, inverse = np.unique(people, return_inverse=True)
    valid = np.isfinite(scores)
    _, ties = np.unique(scores[valid], return_inverse=True)
    positive = np.zeros((len(identities), int(ties.max()+1) if len(ties) else 0))
    negative = np.zeros_like(positive)
    np.add.at(positive, (inverse[valid & (y == 1)], ties[y[valid] == 1]), 1)
    np.add.at(negative, (inverse[valid & (y == 0)], ties[y[valid] == 0]), 1)
    within = np.full(len(identities), np.nan)
    pos_count, neg_count = positive.sum(1), negative.sum(1)
    for k in range(len(identities)):
        selected = valid & (inverse == k)
        if pos_count[k] and neg_count[k]:
            within[k] = roc_auc_score(y[selected], scores[selected])
    alert = scores > threshold if threshold is not None else np.zeros(len(y), dtype=bool)
    positive_alert = np.bincount(inverse[valid & (y == 1) & alert], minlength=len(identities))
    negative_alert = np.bincount(inverse[valid & (y == 0) & alert], minlength=len(identities))
    weights = np.vstack([np.ones((1, len(identities))), draws])
    output = np.full((len(weights), len(STATISTICS)), np.nan)
    within_valid = np.isfinite(within)
    for first in range(0, len(weights), 100):
        w = weights[first:first+100]
        p, n = w @ positive, w @ negative
        total_p, total_n = p.sum(1), n.sum(1)
        wins = (p * (np.cumsum(n, axis=1)-.5*n)).sum(1)
        output[first:first+len(w), 0] = np.divide(wins, total_p*total_n, out=np.full(len(w), np.nan), where=(total_p*total_n)>0)
        mass = w[:, within_valid].sum(1)
        output[first:first+len(w), 1] = np.divide(w[:, within_valid] @ within[within_valid], mass,
            out=np.full(len(w), np.nan), where=mass>0)
        descending_p, descending_n = p[:, ::-1], n[:, ::-1]
        cumulative_p, cumulative_n = np.cumsum(descending_p, axis=1), np.cumsum(descending_n, axis=1)
        precision = np.divide(cumulative_p, cumulative_p+cumulative_n,
            out=np.zeros_like(cumulative_p), where=(cumulative_p+cumulative_n)>0)
        output[first:first+len(w), 2] = np.divide((descending_p*precision).sum(1), total_p,
            out=np.full(len(w), np.nan), where=total_p>0)
        if threshold is not None:
            output[first:first+len(w), 3] = np.divide(w @ positive_alert, total_p, out=np.full(len(w), np.nan), where=total_p>0)
            output[first:first+len(w), 4] = np.divide(w @ negative_alert, total_n, out=np.full(len(w), np.nan), where=total_n>0)
    counts = {"origins": int(valid.sum()), "people": int(np.count_nonzero(pos_count+neg_count)),
              "positive": int(y[valid].sum()), "people_with_both_classes": int(within_valid.sum())}
    return output[0], output[1:], counts


def predict(feature_dir, output, cfg, workers, summarize=True):
    freeze = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    if freeze["managed_methods"] != METHODS or freeze["identity"]["config"] != cfg or freeze["identity"]["core"] != core_hashes():
        raise ValueError("Model/feature definitions changed since training freeze")
    selections = {}
    for method in METHODS:
        folder = output / method
        if sha(folder / "selection.json") != freeze["method_selection_sha256"][method]:
            raise ValueError(f"Frozen method selection changed: {method}")
        selected = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        if selected["model_sha256"] and sha(folder / "model.joblib") != selected["model_sha256"]:
            raise ValueError(f"Frozen model changed: {method}")
        selections[method] = selected
    # Intentional boundary: this is the sole test feature open in the program.
    test = load_split(feature_dir, "test", cfg)
    familiar = set(selections[METHODS[0]]["train_people"]) | set(selections[METHODS[0]]["validation_people"])
    if cfg.get("partition_unit") != "time" and familiar & set(test["people"]):
        raise ValueError("A test person occurred in train/validation")
    predictions = test["meta"].copy()
    for method in METHODS:
        folder = output / method
        detail = None
        if method in NATIVE:
            detail = native_scores(test, method, workers); score = detail.score.to_numpy()
        else:
            bundle = joblib.load(folder / "model.joblib")
            score = bundle["model"].predict_proba(make_features(test, bundle["feature_spec"]))
        save_partition(folder, "test", test, score, detail)
        predictions[method] = score
    predictions.to_csv(output / "test_predictions.csv", index=False)
    if not summarize:
        return
    common = np.isfinite(predictions[METHODS].to_numpy()).all(1)
    populations = {"all": np.ones(len(predictions), bool)}
    if common.any():
        populations["common"] = common
    rows, paired, bootstrap_arrays = [], [], {}
    for population, keep in populations.items():
        y, people = test["y"][keep], test["people"][keep]
        identities = np.unique(people)
        draws = np.random.default_rng(cfg["test_bootstrap"]["seed"]).multinomial(
            len(identities), np.full(len(identities), 1/len(identities)), size=cfg["test_bootstrap"]["replicates"])
        points, bootstraps = {}, {}
        for method in METHODS:
            threshold = selections[method]["threshold_val_negative_p90"]
            point, boot, counts = cluster_statistics(y, predictions.loc[keep, method].to_numpy(), people, threshold, draws)
            points[method], bootstraps[method] = point, boot
            row = {"population": population, "method": method, **counts, "val_negative_p90": threshold}
            for j, statistic in enumerate(STATISTICS):
                finite = np.isfinite(boot[:, j])
                ci = np.quantile(boot[finite, j], [.025, .975]) if finite.any() else [np.nan, np.nan]
                row.update({statistic: point[j], statistic+"_ci_low": ci[0], statistic+"_ci_high": ci[1],
                            statistic+"_bootstrap_valid": int(finite.sum())})
            rows.append(row); bootstrap_arrays[population+"__"+method] = boot
        if population == "common":
            for baseline in METHODS[1:]:
                for j, statistic in enumerate(STATISTICS):
                    delta = bootstraps["HNE-additive"][:, j]-bootstraps[baseline][:, j]
                    finite = np.isfinite(delta)
                    ci = np.quantile(delta[finite], [.025, .975]) if finite.any() else [np.nan, np.nan]
                    paired.append({"population": population, "model": "HNE-additive", "baseline": baseline,
                        "metric": statistic, "difference": points["HNE-additive"][j]-points[baseline][j],
                        "ci_low": ci[0], "ci_high": ci[1], "bootstrap_valid": int(finite.sum())})
    pd.DataFrame(rows).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "test_paired_differences.csv", index=False)
    np.savez_compressed(output / "test_bootstrap.npz", metric_names=np.asarray(STATISTICS), **bootstrap_arrays)
    write_json(output / "test_summary.json", {"test_hashes": test["hashes"], "freeze_manifest_sha256": sha(output / "freeze_manifest.json"),
        "origins": len(predictions), "people": len(np.unique(test["people"])), "positive": int(test["y"].sum()),
        "common_origins": int(common.sum()), "common_people": len(np.unique(test["people"][common])),
        "common_methods": METHODS, "prediction_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"complete": "test", "origins": len(predictions), "common_origins": int(common.sum())}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["fit", "predict"])
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/human_rhythms")
    parser.add_argument("--output", type=Path, default=ROOT / "results/human_rhythms")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.phase == "fit":
            fit(args.features, args.output, configuration(), args.workers)
        else:
            predict(args.features, args.output, configuration(), args.workers)


if __name__ == "__main__":
    main()
