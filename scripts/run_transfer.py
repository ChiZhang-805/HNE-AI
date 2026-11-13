"""Source-only model selection, frozen HNE transfer, and cluster uncertainty."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_additive import AdditiveWarning
from src.dnb import score_window

METRICS = ("global_auroc", "within_cluster_auroc", "auprc")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_human(dataset, config, need_windows=False):
    folder = ROOT / "data/features/unified_daily"
    metadata_path, feature_path = folder / f"{dataset}.csv", folder / f"{dataset}.npz"
    meta = pd.read_csv(metadata_path, dtype={"person_id": str})
    if dataset == "TOLIFE-CSA" and not set(meta.split).issubset(config["tolife_allowed_splits"]):
        raise ValueError("TOLIFE prepared file includes a forbidden test split; feature values were not opened")
    with np.load(feature_path, allow_pickle=False) as z:
        pooled, y, groups = z["pooled"], z["y"], z["groups"]
        folds = z["fold"]
        if not np.array_equal(meta.y.to_numpy(), y) or not np.array_equal(meta.fold.to_numpy(), folds):
            raise ValueError(f"Metadata/feature alignment changed: {dataset}")
        windows = z["windows"] if need_windows else None
        state_count = z["nodes"].shape[1] if need_windows else None
    if pooled.shape != (len(meta), 296):
        raise ValueError(f"Unexpected HNE representation: {dataset}")
    return {"pooled": pooled, "y": y.astype(np.int8), "groups": groups, "folds": folds,
            "meta": meta, "windows": windows, "state_count": state_count,
            "feature_sha256": sha(feature_path), "metadata_sha256": sha(metadata_path)}


def source_data(domains, config):
    arrays, metadata, domain_values, person_values, hashes = [], [], [], [], {}
    expected = np.asarray(["H"]*96 + ["N"]*36 + ["E"]*128 + ["NE"]*36)
    for domain in domains:
        data = load_human(domain, config)
        if not np.array_equal(data["groups"], expected):
            raise ValueError(f"Feature groups changed: {domain}")
        meta = data["meta"].copy()
        if set(meta.fold) != set(range(5)) or meta.groupby("person_id").fold.nunique().max() != 1:
            raise ValueError(f"Invalid original person folds: {domain}")
        arrays.append(data["pooled"]); metadata.append(meta)
        domain_values.extend([domain] * len(meta))
        person_values.extend(domain + "::" + meta.person_id)
        hashes[domain] = {"features": data["feature_sha256"], "metadata": data["metadata_sha256"]}
    return np.concatenate(arrays), pd.concat(metadata, ignore_index=True), expected, np.asarray(domain_values), np.asarray(person_values), hashes


def freeze_source(source_name, domains, config, output, config_hash):
    x, meta, groups, domain, person, hashes = source_data(domains, config)
    y, folds = meta.y.to_numpy(np.int8), meta.fold.to_numpy()
    frozen = {}
    for method in config["methods"]:
        folder = output / "source_models" / source_name / method
        folder.mkdir(parents=True, exist_ok=True)
        columns = np.flatnonzero(groups == "H") if method == "HNE-history" else np.arange(len(groups))
        identity = {"source_name": source_name, "domains": domains, "method": method,
                    "config_sha256": config_hash, "source_hashes": hashes,
                    "additive_implementation_sha256": sha(ROOT / "src/hne_additive.py")}
        existing = folder / "source_config.json"
        if existing.exists():
            saved = json.loads(existing.read_text(encoding="utf-8"))
            if saved["identity"] != identity or sha(folder / "model.joblib") != saved["model_sha256"]:
                raise ValueError(f"Frozen model differs from current source/config: {folder}")
            frozen[method] = saved
            print(f"REUSE frozen {source_name}/{method} C={saved['selected_C']}", flush=True)
            continue
        trials, candidates, all_oof = [], [], []
        for c in config["C_candidates"]:
            oof = np.full(len(meta), np.nan)
            for fold in range(5):
                train, val = folds != fold, folds == fold
                estimator = AdditiveWarning(C=c, spline=True, groups=groups[columns])
                estimator.fit(x[train][:, columns], y[train], person[train], domains=domain[train])
                oof[val] = estimator.predict_proba(x[val][:, columns])
                for name in domains:
                    keep = val & (domain == name)
                    auc = roc_auc_score(y[keep], oof[keep]) if len(np.unique(y[keep])) == 2 else np.nan
                    trials.append({"C": c, "fold": fold, "domain": name,
                        "train_samples": int(train.sum()), "validation_samples": int(keep.sum()),
                        "validation_people": len(np.unique(person[keep])), "validation_positive": int(y[keep].sum()),
                        "auroc": auc, "log_loss": log_loss(y[keep], oof[keep], labels=[0, 1])})
            trial_frame = pd.DataFrame([r for r in trials if r["C"] == c])
            by_domain = trial_frame.groupby("domain").auroc.mean()
            if by_domain.isna().any():
                raise ValueError("A source domain has no two-class validation fold")
            selection = float(by_domain.mean())
            candidates.append({"C": c, "selection_auroc": selection, "domain_mean_auroc": by_domain.to_dict(),
                               "valid_auc_folds": trial_frame.groupby("domain").auroc.count().to_dict()})
            prediction = meta[["dataset", "person_id", "fold", "origin_day", "y"]].copy()
            prediction["domain"], prediction["C"], prediction["probability"] = domain, c, oof
            all_oof.append(prediction)
            print(f"CV {source_name}/{method} C={c:g} source_AUROC={selection:.6f}", flush=True)
        selected = max(candidates, key=lambda r: (r["selection_auroc"], -r["C"]))
        pd.DataFrame(trials).to_csv(folder / "trials.csv", index=False)
        pd.concat(all_oof).to_csv(folder / "source_oof.csv", index=False)
        write_json(folder / "candidate_summary.json", candidates)
        fitted = AdditiveWarning(C=selected["C"], spline=True, groups=groups[columns])
        fitted.fit(x[:, columns], y, person, domains=domain)
        joblib.dump({"model": fitted, "columns": columns, "groups": groups[columns]}, folder / "model.joblib")
        saved = {"identity": identity, "selected_C": selected["C"], "selection_auroc": selected["selection_auroc"],
            "candidate_results": candidates, "fit_samples": len(y), "fit_people": len(np.unique(person)),
            "feature_count": len(columns), "positive_multiplier": float(fitted.positive_multiplier),
            "source_counts": {name: {"samples": int((domain == name).sum()),
                "people": len(np.unique(person[domain == name])), "positive": int(y[domain == name].sum())} for name in domains},
            "person_namespace": "domain::person_id", "weighting": config["weighting"],
            "model_sha256": sha(folder / "model.joblib"), "trials_sha256": sha(folder / "trials.csv"),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "target_data_used": False}
        write_json(existing, saved)
        frozen[method] = saved
        print(f"FROZEN {source_name}/{method} C={selected['C']:g}", flush=True)
    return frozen


def load_target(dataset, config):
    if dataset == "CARE-v6":
        folder = ROOT / "data/care_study"
        meta = pd.read_csv(folder / "samples.csv", dtype={"asset_group": str, "case_id": str})
        with np.load(folder / "features.npz", allow_pickle=False) as data:
            x, y, dnb = data["pooled"], data["y"], data["dnb_score"]
            if not np.array_equal(meta.y.to_numpy(), y) or not np.array_equal(meta.asset_group.to_numpy(), data["asset_group"]):
                raise ValueError("CARE metadata/features disagree")
        cluster = meta.asset_group.to_numpy()
        hashes = {"features": sha(folder / "features.npz"), "metadata": sha(folder / "samples.csv"),
                  "endpoint_config": sha(folder / "config.json")}
    else:
        data = load_human(dataset, config, need_windows=True)
        meta, x, y = data["meta"], data["pooled"], data["y"]
        dnb = np.asarray([score_window(w[-32:, :data["state_count"]], minimum=12) for w in data["windows"]])
        cluster = meta.person_id.to_numpy()
        hashes = {"features": data["feature_sha256"], "metadata": data["metadata_sha256"]}
    return meta, x, y, cluster, dnb, hashes


def cluster_metrics(y, score, clusters, draws):
    """Global origin metrics and equal-cluster within AUROC, with cluster resampling."""
    y, score, clusters = np.asarray(y), np.asarray(score), np.asarray(clusters)
    ids, inverse = np.unique(clusters, return_inverse=True)
    if draws.shape[1] != len(ids):
        raise ValueError("Bootstrap cluster order mismatch")
    valid = np.isfinite(score)
    within = np.full(len(ids), np.nan)
    for k in range(len(ids)):
        keep = valid & (inverse == k)
        if len(np.unique(y[keep])) == 2:
            within[k] = roc_auc_score(y[keep], score[keep])
    point = np.array([
        roc_auc_score(y[valid], score[valid]) if len(np.unique(y[valid])) == 2 else np.nan,
        np.nanmean(within) if np.isfinite(within).any() else np.nan,
        average_precision_score(y[valid], score[valid]) if np.any(y[valid] == 1) else np.nan])
    boot = np.full((len(draws), 3), np.nan)
    good_within = np.isfinite(within)
    for b, cluster_weight in enumerate(draws):
        weights = cluster_weight[inverse]
        keep = valid & (weights > 0)
        positive, negative = weights[keep & (y == 1)].sum(), weights[keep & (y == 0)].sum()
        if positive and negative:
            boot[b, 0] = roc_auc_score(y[keep], score[keep], sample_weight=weights[keep])
        mass = cluster_weight[good_within].sum()
        if mass:
            boot[b, 1] = np.dot(cluster_weight[good_within], within[good_within]) / mass
        if positive:
            boot[b, 2] = average_precision_score(y[keep], score[keep], sample_weight=weights[keep])
    details = {"origins": int(valid.sum()), "clusters": len(np.unique(clusters[valid])),
               "positive": int(y[valid].sum()), "clusters_with_both_classes": int(good_within.sum())}
    return point, boot, details


def evaluate_transfer(source, target, config, output):
    folder = output / f"{source}_to_{target}"
    folder.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for method in config["methods"]:
        source_folder = output / "source_models" / source / method
        saved = json.loads((source_folder / "source_config.json").read_text(encoding="utf-8"))
        if sha(source_folder / "model.joblib") != saved["model_sha256"]:
            raise ValueError("Frozen source model changed")
        frozen[method] = saved
    # No target values/labels are loaded until both frozen source files are present.
    meta, x, y, cluster, dnb, hashes = load_target(target, config)
    predictions = meta.copy()
    predictions["cluster_id"] = cluster
    model_hashes = {}
    for method in config["methods"]:
        model_path = output / "source_models" / source / method / "model.joblib"
        bundle = joblib.load(model_path)
        predictions[method] = bundle["model"].predict_proba(x[:, bundle["columns"]])
        model_hashes[method] = sha(model_path)
        if not np.isfinite(predictions[method]).all():
            raise FloatingPointError(f"Nonfinite source-frozen prediction: {source}/{target}/{method}")
    predictions["DNB"] = dnb
    predictions.to_csv(folder / "predictions.csv", index=False)
    write_json(folder / "target_config.json", {"source": source, "target": target,
        "endpoint": config["target_endpoints"][target], "cluster_unit": "asset" if target == "CARE-v6" else "person",
        "frozen_models": model_hashes, "source_frozen_at_utc": {m: f["frozen_at_utc"] for m, f in frozen.items()},
        "target_loaded_after_model_freeze": True, "target_hashes": hashes,
        "target_splits": sorted(meta.split.unique().tolist()) if "split" in meta else None,
        "target_fit_or_selection": False, "prediction_at_utc": datetime.now(timezone.utc).isoformat()})
    methods = config["methods"] + ["DNB"]
    common = np.isfinite(predictions[methods].to_numpy()).all(1)
    populations = {"all": np.ones(len(predictions), dtype=bool)}
    if common.any() and not common.all():
        populations["common"] = common
    rows, differences, arrays = [], [], {}
    for population, mask in populations.items():
        local_y, local_cluster = y[mask], cluster[mask]
        ids = np.unique(local_cluster)
        draws = np.random.default_rng(config["bootstrap_seed"]).multinomial(
            len(ids), np.full(len(ids), 1 / len(ids)), size=config["bootstrap_replicates"])
        estimates, bootstraps = {}, {}
        for method in methods:
            estimates[method], bootstraps[method], details = cluster_metrics(local_y,
                predictions.loc[mask, method].to_numpy(), local_cluster, draws)
            row = {"source": source, "target": target, "population": population, "method": method,
                   "endpoint": config["target_endpoints"][target], **details,
                   "cluster_unit": "asset" if target == "CARE-v6" else "person"}
            for j, metric in enumerate(METRICS):
                valid = np.isfinite(bootstraps[method][:, j])
                ci = np.quantile(bootstraps[method][valid, j], [.025, .975]) if valid.any() else [np.nan, np.nan]
                row.update({metric: estimates[method][j], metric + "_ci_low": ci[0], metric + "_ci_high": ci[1],
                            metric + "_bootstrap_valid": int(valid.sum())})
            rows.append(row)
            arrays[population + "__" + method] = bootstraps[method]
        if population == "common" or common.all():
            for baseline in ("HNE-history", "DNB"):
                for j, metric in enumerate(METRICS):
                    delta = bootstraps["HNE-additive"][:, j] - bootstraps[baseline][:, j]
                    finite = np.isfinite(delta)
                    ci = np.quantile(delta[finite], [.025, .975]) if finite.any() else [np.nan, np.nan]
                    differences.append({"source": source, "target": target, "population": population,
                        "model": "HNE-additive", "baseline": baseline, "metric": metric,
                        "difference": estimates["HNE-additive"][j] - estimates[baseline][j],
                        "ci_low": ci[0], "ci_high": ci[1], "bootstrap_valid": int(finite.sum())})
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(folder / "paired_differences.csv", index=False)
    np.savez_compressed(folder / "bootstrap.npz", metric_names=np.asarray(METRICS), **arrays)
    print(json.dumps({"transfer": f"{source}_to_{target}", "all": [
        {k: r[k] for k in ("method", "origins", "global_auroc", "within_cluster_auroc", "auprc")}
        for r in rows if r["population"] == "all"]}), flush=True)
    return rows, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transfer.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/transfer")
    parser.add_argument("--phase", choices=["all", "fit", "predict"], default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    config_hash = sha(args.config)
    write_json(args.output / "run_config.json", config)
    with threadpool_limits(limits=2):
        if args.phase in ("all", "fit"):
            source_config = {}
            for name, domains in config["source_groups"].items():
                source_config[name] = freeze_source(name, domains, config, args.output, config_hash)
            write_json(args.output / "source_freeze.json", source_config)
        if args.phase in ("all", "predict"):
            # Require the complete planned freeze before any target stage.
            if not (args.output / "source_freeze.json").exists():
                raise ValueError("Complete source-only fitting before target prediction")
            rows, differences = [], []
            for task in config["transfers"]:
                metrics, deltas = evaluate_transfer(task["source"], task["target"], config, args.output)
                rows.extend(metrics); differences.extend(deltas)
            pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
            pd.DataFrame(differences).to_csv(args.output / "paired_differences.csv", index=False)


if __name__ == "__main__":
    main()
