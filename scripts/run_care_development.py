"""Asset-separated CARE learning with the unchanged HNE coordinates and endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_additive import AdditiveWarning, BoostedWarning, validation_score
from src.hne_transfer import SourcePriorWarning
from scripts.run_transfer import cluster_metrics, sha, write_json
from scripts.summarize_development import fold_auc, bootstrap_aucs

METHODS = ["HNE-additive", "HNE-history", "HNE-no-network", "HNE-no-environment", "HNE-HGB", "DNB"]
SEED = 20261110


def load_data():
    folder = ROOT / "data/care_study"
    meta = pd.read_csv(folder / "samples.csv", dtype={"asset_group": str, "case_id": str})
    with np.load(folder / "features.npz", allow_pickle=False) as a:
        x, y, groups, dnb = a["pooled"], a["y"], a["groups"], a["dnb_score"]
        if not np.array_equal(y, meta.y) or not np.array_equal(a["asset_group"], meta.asset_group):
            raise ValueError("CARE metadata/feature alignment mismatch")
    return x, y, groups, dnb, meta


def initialize(output, meta):
    output.mkdir(parents=True, exist_ok=True)
    assets = sorted(meta.asset_group.unique(), key=lambda a: hashlib.sha256(f"{SEED}:CARE-v6:{a}".encode()).hexdigest())
    mapping = {a: i % 5 for i, a in enumerate(assets)}
    config = {"version": "care-asset-development-v1-20261110", "seed": SEED,
        "fold_assignment": "sort 36 asset_group by sha256('20261110:CARE-v6:'+asset_group); rank mod5",
        "partitions": "held fold f; validation (f+1)%5; training other three folds",
        "methods": METHODS, "C_candidates": [.01, .1, 1.],
        "selection": "validation AUROC; single-class validation uses negative logloss; tie smaller C",
        "endpoint": "unchanged CARE official fault in [issue+1d,issue+8d); distinct native industrial endpoint",
        "feature_sha256": sha(ROOT / "data/care_study/features.npz"),
        "endpoint_config_sha256": sha(ROOT / "data/care_study/config.json"),
        "feature_groups": "296=H96/N36/E128/NE36; reference-only max32 state selection already fixed",
        "weighting": "equal asset base weights, global class balance, training-prior logit correction",
        "primary_auroc": "positive-negative pair-weighted AUROC within each held fold; no ranking across separately fitted fold models",
        "additional_metrics": ["global_oof_auroc", "equal-asset within-asset AUROC", "raw-score average precision", "validation-rank average precision"],
        "bootstrap": "2000 resamples of complete assets; fixed seed20261110; paired draws across methods",
        "dnb": "fixed native dnb_score, higher means risk; no target fit or direction choice",
        "fewshot_plan": {"fractions": [.25, .5, 1.], "seeds": [20261110, 20261111, 20261112],
            "full_fraction_seeds": [20261110], "count": "ceil(fraction*number of training assets)",
            "strata": "train assets with any positive origin versus normal-only; proportional quotas, at least one per nonempty stratum",
            "sampling": "within strata sort sha256(seed:held_fold:asset_group); same subsets across all conditions",
            "conditions": ["target-only-basis", "source-basis-zero", "source-basis-prior"],
            "source": "frozen all-human HNE-additive model; unchanged source basis",
            "prior": "weighted target logistic loss plus squared distance to source raw-logit coefficients/(2C); unpenalized intercept; target-prior correction"}}
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing CARE development configuration differs")
    write_json(path, config)
    assignment = pd.DataFrame({"asset_group": assets, "fold": [mapping[a] for a in assets]})
    assignment.to_csv(output / "asset_folds.csv", index=False)
    cases = pd.read_csv(ROOT / "data/care_study/cases.csv")
    cases["fold"] = cases.asset_group.map(mapping)
    if cases.fold.isna().any() or cases.groupby("asset_group").fold.nunique().max() != 1:
        raise ValueError("Case/asset fold mapping failed")
    cases.to_csv(output / "case_folds.csv", index=False)
    meta = meta.copy(); meta["fold"] = meta.asset_group.map(mapping)
    return config, meta


def columns_for(method, groups):
    if method == "HNE-history":
        return np.flatnonzero(groups == "H")
    if method == "HNE-no-network":
        return np.flatnonzero(np.isin(groups, ["H", "E"]))
    if method == "HNE-no-environment":
        return np.flatnonzero(np.isin(groups, ["H", "N"]))
    return np.arange(len(groups))


def partitions(meta, held):
    folds = meta.fold.to_numpy()
    masks = {"train": (folds != held) & (folds != (held+1)%5),
             "validation": folds == (held+1)%5, "held_asset": folds == held}
    people = {k: set(meta.loc[m, "asset_group"]) for k, m in masks.items()}
    if any(people[a] & people[b] for a, b in [("train", "validation"), ("train", "held_asset"), ("validation", "held_asset")]):
        raise ValueError("An asset crosses split boundaries")
    return masks


def fit_full(output, x, y, groups, dnb, meta, config):
    for method in METHODS:
        columns = columns_for(method, groups)
        for held in range(5):
            folder = output / "full" / method / f"fold{held}"
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "metadata.json").exists():
                continue
            masks = partitions(meta, held)
            train, val = masks["train"], masks["validation"]
            trials, best, selected_c = [], None, None
            if method != "DNB":
                best_score = -np.inf
                for c in config["C_candidates"]:
                    model = BoostedWarning(C=c, seed=SEED) if method == "HNE-HGB" else AdditiveWarning(C=c, groups=groups[columns])
                    model.fit(x[train][:, columns], y[train], meta.loc[train, "asset_group"].to_numpy())
                    p = model.predict_proba(x[val][:, columns])
                    score = validation_score(y[val], p)
                    trials.append({"C": c, "validation_score": score})
                    if score > best_score:
                        best, best_score, selected_c = model, score, c
                joblib.dump({"model": best, "columns": columns}, folder / "model.joblib")
            pd.DataFrame(trials, columns=["C", "validation_score"]).to_csv(folder / "trials.csv", index=False)
            val_score = dnb[val] if method == "DNB" else best.predict_proba(x[val][:, columns])
            sorted_validation = np.sort(val_score[np.isfinite(val_score)])
            for part, mask in masks.items():
                frame = meta.loc[mask].copy()
                value = dnb[mask] if method == "DNB" else best.predict_proba(x[mask][:, columns])
                frame["score"] = value
                frame["validation_rank"] = np.where(np.isfinite(value),
                    np.searchsorted(sorted_validation, value, side="right")/len(sorted_validation), np.nan)
                frame.to_csv(folder / f"{part}.csv", index=False)
            write_json(folder / "metadata.json", {"method": method, "held_fold": held, "validation_fold": (held+1)%5,
                "C": selected_c, "input_features": len(columns), "score_is_probability": method != "DNB",
                "assets": {part: sorted(meta.loc[mask, "asset_group"].unique().tolist()) for part, mask in masks.items()},
                "rows": {part: int(mask.sum()) for part, mask in masks.items()},
                "config_sha256": sha(output / "config.json"), "model_sha256": sha(folder / "model.joblib") if method != "DNB" else None})
            print(f"FIT {method} fold{held} C={selected_c}", flush=True)


def summarize_full(output, meta):
    frame = meta.copy()
    keys = ["case_id", "origin_day"]
    for method in METHODS:
        table = pd.concat([pd.read_csv(output / "full" / method / f"fold{f}/held_asset.csv") for f in range(5)])
        table = table[keys + ["score", "validation_rank"]].rename(columns={"score": method, "validation_rank": method+"__rank"})
        frame = frame.merge(table, on=keys, validate="one_to_one")
    frame.to_csv(output / "full_predictions.csv", index=False)
    ids, inverse = np.unique(frame.asset_group, return_inverse=True)
    draws = np.random.default_rng(SEED).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=2000)
    rows, boots = [], {}
    for method in METHODS:
        point, other_boot, details = cluster_metrics(frame.y.to_numpy(), frame[method].to_numpy(), frame.asset_group.to_numpy(), draws)
        fold_boot = bootstrap_aucs(frame, method, inverse, draws)
        metric_points = {"auroc": fold_auc(frame, method), "global_auroc": point[0],
                         "within_asset_auroc": point[1], "auprc": point[2]}
        metric_boots = {"auroc": fold_boot, "global_auroc": other_boot[:, 0],
                        "within_asset_auroc": other_boot[:, 1], "auprc": other_boot[:, 2]}
        rank = frame[method+"__rank"].to_numpy(); y = frame.y.to_numpy()
        good = np.isfinite(rank)
        metric_points["ap_on_validation_rank"] = average_precision_score(y[good], rank[good])
        rank_boot = np.full(len(draws), np.nan)
        for k, draw in enumerate(draws):
            weight = draw[inverse]; keep = good & (weight > 0)
            if weight[keep & (y == 1)].sum():
                rank_boot[k] = average_precision_score(y[keep], rank[keep], sample_weight=weight[keep])
        metric_boots["ap_on_validation_rank"] = rank_boot
        row = {"method": method, **details}
        for metric, value in metric_points.items():
            boot = metric_boots[metric]; valid = np.isfinite(boot)
            ci = np.quantile(boot[valid], [.025, .975])
            row.update({metric: value, metric+"_ci_low": ci[0], metric+"_ci_high": ci[1], metric+"_bootstrap_valid": int(valid.sum())})
            boots[method+"__"+metric] = boot
        rows.append(row)
        print(json.dumps({"method": method, **metric_points}), flush=True)
    differences = []
    for baseline in METHODS[1:]:
        for metric in ("auroc", "global_auroc", "within_asset_auroc", "auprc", "ap_on_validation_rank"):
            delta = boots["HNE-additive__"+metric] - boots[baseline+"__"+metric]
            ci = np.nanquantile(delta, [.025, .975])
            estimate = rows[0][metric] - next(r[metric] for r in rows if r["method"] == baseline)
            differences.append({"model": "HNE-additive", "baseline": baseline, "metric": metric,
                "difference": estimate, "ci_low": ci[0], "ci_high": ci[1]})
    pd.DataFrame(rows).to_csv(output / "full_metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "full_paired_differences.csv", index=False)
    np.savez_compressed(output / "full_bootstrap.npz", **boots)


def choose_subset(meta, train, fraction, seed, held):
    labels = meta.loc[train].groupby("asset_group").y.max()
    positive = labels.index[labels > 0].tolist()
    normal = labels.index[labels == 0].tolist()
    key = lambda a: hashlib.sha256(f"{seed}:{held}:{a}".encode()).hexdigest()
    positive.sort(key=key); normal.sort(key=key)
    count = int(np.ceil(fraction*len(labels)))
    if fraction == 1:
        selected = positive + normal
    elif positive and normal:
        pcount = max(1, min(len(positive), count-1, int(np.floor(count*len(positive)/len(labels)+.5))))
        ncount = min(len(normal), count-pcount)
        pcount = count-ncount
        selected = positive[:pcount] + normal[:ncount]
    else:
        selected = (positive+normal)[:count]
    if len(selected) != count:
        raise ValueError("Stratified asset budget could not be satisfied")
    details = {"fraction": fraction, "seed": seed, "held_fold": held,
        "available_train_assets": len(labels), "selected_assets": sorted(selected),
        "selected_positive_assets": [a for a in selected if labels[a] > 0],
        "selected_normal_only_assets": [a for a in selected if labels[a] == 0],
        "available_positive_assets": len(positive), "available_normal_only_assets": len(normal),
        "positive_quota_rounding": "floor(k*n_positive/n_train+0.5), constrained by nonempty strata and capacity"}
    return train & meta.asset_group.isin(selected).to_numpy(), details


def fit_fewshot(output, x, y, groups, meta, config):
    if not (output / "full_metrics.csv").exists():
        raise ValueError("Complete the full target-domain experiment before few-shot fitting")
    source_path = ROOT / "results/transfer/source_models/Humans/HNE-additive/model.joblib"
    source = joblib.load(source_path)
    if not np.array_equal(source["columns"], np.arange(296)):
        raise ValueError("The frozen human source must use the complete shared296 representation")
    prior_config = {**config["fewshot_plan"], "C_candidates": config["C_candidates"],
        "source_model_sha256": sha(source_path), "source_model_path": str(source_path),
        "adapter_sha256": sha(ROOT / "src/hne_transfer.py"),
        "partition_config_sha256": sha(output / "config.json"),
        "objective": "sum normalized equal-asset class-balanced BCE + ||w-center||^2/(2C); intercept unpenalized",
        "source_coefficient": "raw balanced-logit coefficient; target positive multiplier removed from prediction logit",
        "selection": config["selection"], "fewshot_primary_summary": "mean of three subset-seed metrics, not a prediction ensemble; 100% one run"}
    cfg_path = output / "fewshot_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != prior_config:
        raise ValueError("Existing few-shot config differs")
    write_json(cfg_path, prior_config)
    conditions = config["fewshot_plan"]["conditions"]
    for fraction in config["fewshot_plan"]["fractions"]:
        seeds = [SEED] if fraction == 1 else config["fewshot_plan"]["seeds"]
        for seed in seeds:
            for held in range(5):
                masks = partitions(meta, held)
                train, subset = choose_subset(meta, masks["train"], fraction, seed, held)
                if len(np.unique(y[train])) != 2:
                    raise ValueError(f"Prespecified target subset lacks a class: {subset}")
                subset_folder = output / "fewshot" / f"budget{int(fraction*100)}" / f"seed{seed}" / f"fold{held}"
                subset_folder.mkdir(parents=True, exist_ok=True)
                write_json(subset_folder / "subset_assets.json", subset)
                val, test = masks["validation"], masks["held_asset"]
                for condition in conditions:
                    folder = subset_folder / condition
                    folder.mkdir(parents=True, exist_ok=True)
                    if (folder / "metadata.json").exists():
                        continue
                    best, best_score, chosen_c, trials = None, -np.inf, None, []
                    for c in config["C_candidates"]:
                        if condition == "target-only-basis":
                            model = AdditiveWarning(C=c, groups=groups)
                        else:
                            model = SourcePriorWarning(source["model"], C=c,
                                prior="source" if condition == "source-basis-prior" else "zero")
                        model.fit(x[train], y[train], meta.loc[train, "asset_group"].to_numpy())
                        p = model.predict_proba(x[val]); score = validation_score(y[val], p)
                        trials.append({"C": c, "validation_score": score,
                            "optimizer_success": getattr(model, "optimization_", {}).get("success", True),
                            "iterations": getattr(model, "optimization_", {}).get("iterations", None)})
                        if score > best_score:
                            best, best_score, chosen_c = model, score, c
                    pd.DataFrame(trials).to_csv(folder / "trials.csv", index=False)
                    joblib.dump(best, folder / "model.joblib")
                    sorted_validation = np.sort(best.predict_proba(x[val]))
                    for part, mask in (("train", train), ("validation", val), ("held_asset", test)):
                        table = meta.loc[mask].copy(); score = best.predict_proba(x[mask])
                        table["score"] = score
                        table["validation_rank"] = np.searchsorted(sorted_validation, score, side="right")/len(sorted_validation)
                        table.to_csv(folder / f"{part}.csv", index=False)
                    write_json(folder / "metadata.json", {"condition": condition, "fraction": fraction,
                        "seed": seed, "held_fold": held, "selected_C": chosen_c, "validation_score": best_score,
                        "train_assets": subset["selected_assets"], "train_samples": int(train.sum()),
                        "validation_assets": sorted(meta.loc[val, "asset_group"].unique().tolist()),
                        "held_assets": sorted(meta.loc[test, "asset_group"].unique().tolist()),
                        "source_model_sha256": prior_config["source_model_sha256"] if condition != "target-only-basis" else None,
                        "model_sha256": sha(folder / "model.joblib"), "config_sha256": sha(cfg_path),
                        "optimization": getattr(best, "optimization_", None)})
                    print(f"FEWSHOT budget{int(fraction*100)} seed{seed} fold{held} {condition} C={chosen_c}", flush=True)


def quick_auc_bootstrap(frame, draws, inverse):
    native_fold = bootstrap_aucs(frame, "score", inverse, draws)
    global_frame = frame.copy(); global_frame["fold"] = 0
    global_boot = bootstrap_aucs(global_frame, "score", inverse, draws)
    within = np.full(draws.shape[1], np.nan)
    for k in range(len(within)):
        rows = frame.iloc[np.flatnonzero(inverse == k)]
        if rows.y.nunique() == 2:
            within[k] = roc_auc_score(rows.y, rows.score)
    valid = np.isfinite(within); denominator = draws[:, valid].sum(1)
    within_boot = np.divide(draws[:, valid] @ within[valid], denominator,
        out=np.full(len(draws), np.nan), where=denominator > 0)
    points = np.array([fold_auc(frame, "score"), roc_auc_score(frame.y, frame.score), np.nanmean(within)])
    return points, np.column_stack([native_fold, global_boot, within_boot])


def summarize_fewshot(output, meta, config):
    ids, inverse = np.unique(meta.asset_group, return_inverse=True)
    draws = np.random.default_rng(SEED).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=2000)
    metric_names = ["auroc", "global_auroc", "within_asset_auroc"]
    seed_rows, summaries, paired, arrays = [], [], [], {}
    conditions = config["fewshot_plan"]["conditions"]
    for fraction in config["fewshot_plan"]["fractions"]:
        seeds = [SEED] if fraction == 1 else config["fewshot_plan"]["seeds"]
        averages, bootstrap_average = {}, {}
        for condition in conditions:
            points, bootstraps, aps = [], [], []
            for seed in seeds:
                parts = [pd.read_csv(output / "fewshot" / f"budget{int(fraction*100)}" / f"seed{seed}" /
                    f"fold{held}" / condition / "held_asset.csv") for held in range(5)]
                prediction = pd.concat(parts)[["case_id", "origin_day", "score", "validation_rank"]]
                frame = meta.merge(prediction, on=["case_id", "origin_day"], validate="one_to_one")
                if len(frame) != len(meta) or not np.array_equal(frame.asset_group, meta.asset_group):
                    raise ValueError("Few-shot held predictions are not aligned")
                point, boot = quick_auc_bootstrap(frame, draws, inverse)
                ap = average_precision_score(frame.y, frame.validation_rank)
                points.append(point); bootstraps.append(boot); aps.append(ap)
                seed_rows.append({"fraction": fraction, "condition": condition, "seed": seed,
                    **dict(zip(metric_names, point)), "ap_on_validation_rank": ap})
            points = np.asarray(points)
            averages[condition] = points.mean(0)
            bootstrap_average[condition] = np.nanmean(bootstraps, axis=0)
            arrays[f"budget{int(fraction*100)}__{condition}"] = bootstrap_average[condition]
            row = {"fraction": fraction, "condition": condition, "subset_seeds": len(seeds),
                "origins": len(meta), "assets": len(ids), "ap_on_validation_rank": float(np.mean(aps))}
            for j, metric in enumerate(metric_names):
                ci = np.nanquantile(bootstrap_average[condition][:, j], [.025, .975])
                row.update({metric: averages[condition][j], metric+"_seed_min": points[:, j].min(),
                    metric+"_seed_max": points[:, j].max(), metric+"_ci_low": ci[0], metric+"_ci_high": ci[1]})
            summaries.append(row)
            print(json.dumps(row), flush=True)
        for baseline in conditions[:2]:
            for j, metric in enumerate(metric_names):
                delta = bootstrap_average["source-basis-prior"][:, j] - bootstrap_average[baseline][:, j]
                ci = np.nanquantile(delta, [.025, .975])
                paired.append({"fraction": fraction, "model": "source-basis-prior", "baseline": baseline,
                    "metric": metric, "difference": averages["source-basis-prior"][j] - averages[baseline][j],
                    "ci_low": ci[0], "ci_high": ci[1]})
    pd.DataFrame(seed_rows).to_csv(output / "fewshot_seed_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "fewshot_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "fewshot_paired_differences.csv", index=False)
    np.savez_compressed(output / "fewshot_bootstrap.npz", metric_names=np.asarray(metric_names), **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/care_development")
    parser.add_argument("--stage", choices=["full", "fewshot", "all"], default="full")
    args = parser.parse_args()
    x, y, groups, dnb, meta = load_data()
    config, meta = initialize(args.output, meta)
    with threadpool_limits(limits=2):
        if args.stage in {"full", "all"}:
            fit_full(args.output, x, y, groups, dnb, meta, config)
            summarize_full(args.output, meta)
        if args.stage in {"fewshot", "all"}:
            fit_fewshot(args.output, x, y, groups, meta, config)
            summarize_fewshot(args.output, meta, config)


if __name__ == "__main__":
    main()
