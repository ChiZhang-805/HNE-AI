"""Fixed-ID sample efficiency on Human Rhythms; train/validation and test phases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from src.hne_additive import AdditiveWarning, validation_score
from src.hne_transfer import SourcePriorWarning
from scripts.run_hne_external import configuration as external_configuration, load_split, cluster_statistics, score_description, STATISTICS
from scripts.run_transfer import sha, write_json

CONDITIONS = ("target-only-basis", "source-basis-zero", "source-basis-prior")
SEEDS = (20261110, 20261111, 20261112)
FRACTIONS = (.25, .5, 1.)
SOURCE = ROOT / "results/transfer/source_models/Humans/HNE-additive"
BASELINE = ROOT / "results/human_rhythms/HNE-additive"


def configuration():
    return {"version": "human-rhythms-sample-efficiency-v1-20261110", "conditions": list(CONDITIONS),
        "C_candidates": [.01, .1, 1.], "fractions": list(FRACTIONS), "subset_seeds": list(SEEDS),
        "full_fraction_seeds": [SEEDS[0]], "expected_training_people": 85,
        "expected_subset_people": {"25": 22, "50": 43, "100": 85},
        "subset": "sort train person_id by sha256(f'{seed}:{person_id}'); take ceil(fraction*85); no label stratification, no resampling",
        "same_subset_across_conditions": True, "context_ridge": 10,
        "source": "earlier frozen Humans HNE-additive; no Human Rhythms source refit",
        "parameter_prior": "unchanged SourcePriorWarning; raw-logit source or zero coefficient center; intercept unpenalized; target-prior correction",
        "full_target_only": "reuse existing frozen Human Rhythms HNE-additive model and scores, without another fit",
        "selection": "maximum validation AUROC; single-class negative logloss; exact ties smaller C",
        "weighting": "equal target training-person base mass, then class balance",
        "threshold": "linear90th percentile of finite validation negative scores; alert strictly greater than threshold",
        "test_bootstrap": {"replicates": 2000, "seed": SEEDS[0], "unit": "whole person", "interval": "percentile95"},
        "seed_summary": "mean of three independently selected subset-seed metrics, not an ensemble; 100% one fit; fixed-subset-seed person bootstrap averages the same resampled-person metrics",
        "no_test_model_selection": True}


def implementation_hashes():
    return {name: sha(ROOT / "src" / name) for name in ("hne_additive.py", "hne_transfer.py")}


def subset_plan(people, expected_people=85):
    people = sorted(set(people))
    if len(people) != expected_people:
        raise ValueError(f"The fixed training population must contain {expected_people} people")
    plans = []
    for fraction in FRACTIONS:
        for seed in ([SEEDS[0]] if fraction == 1 else SEEDS):
            ranked = sorted(people, key=lambda person: hashlib.sha256(f"{seed}:{person}".encode()).hexdigest())
            count = int(np.ceil(fraction*len(people)))
            plans.append({"fraction": fraction, "budget": int(fraction*100), "seed": seed,
                "training_population": len(people), "selected_people": sorted(ranked[:count]),
                "selection_order": ranked[:count], "selected_person_count": count, "labels_used_for_selection": False})
    return plans


def destination(output, plan, condition=None):
    folder = output / f"budget{plan['budget']}" / f"seed{plan['seed']}"
    return folder / condition if condition else folder


def save_scores(folder, name, data, mask, scores):
    table = data["meta"].loc[mask].copy(); table["score"] = scores
    table.to_csv(folder / f"{name}.csv", index=False)


def fit(feature_dir, output, cfg=None, external_cfg=None, source_dir=None, baseline_dir=None):
    cfg = configuration() if cfg is None else cfg
    external_cfg = external_configuration() if external_cfg is None else external_cfg
    source_dir = SOURCE if source_dir is None else source_dir
    baseline_dir = BASELINE if baseline_dir is None else baseline_dir
    train = load_split(feature_dir, "train", external_cfg)
    validation = load_split(feature_dir, "validation", external_cfg)
    if set(train["people"]) & set(validation["people"]):
        raise ValueError("A person crosses train and validation")
    source_info = json.loads((source_dir / "source_config.json").read_text(encoding="utf-8"))
    baseline_info = json.loads((baseline_dir / "selection.json").read_text(encoding="utf-8"))
    if sha(source_dir / "model.joblib") != source_info["model_sha256"] or sha(baseline_dir / "model.joblib") != baseline_info["model_sha256"]:
        raise ValueError("A previously frozen model changed")
    if baseline_info["identity"]["train"] != train["hashes"] or baseline_info["identity"]["validation"] != validation["hashes"]:
        raise ValueError("Input features differ from the previously frozen default-ridge10 study")
    source = joblib.load(source_dir / "model.joblib")
    if not np.array_equal(source["columns"], np.arange(296)):
        raise ValueError("The human source must have the complete296 coordinate basis")
    identity = {"config": cfg, "train": train["hashes"], "validation": validation["hashes"],
        "core": implementation_hashes(), "source_model_sha256": sha(source_dir / "model.joblib"),
        "full_target_model_sha256": sha(baseline_dir / "model.joblib")}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "fit_config.json").exists() and json.loads((output / "fit_config.json").read_text()) != identity:
        raise ValueError("Existing sample-efficiency config differs")
    write_json(output / "fit_config.json", identity)
    # Every participant list is committed before any target-subset model is fitted.
    plans = subset_plan(train["people"], expected_people=cfg["expected_training_people"])
    for plan in plans:
        folder = destination(output, plan); folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "subset.json", plan)
    selections = []
    val_all = np.ones(len(validation["y"]), bool)
    for plan in plans:
        mask = np.isin(train["people"], plan["selected_people"])
        if len(np.unique(train["y"][mask])) != 2:
            raise ValueError("Prespecified pure-ID subset has only one label class; do not resample it")
        for condition in CONDITIONS:
            folder = destination(output, plan, condition); folder.mkdir(parents=True, exist_ok=True)
            selection_file = folder / "selection.json"
            if selection_file.exists():
                saved = json.loads(selection_file.read_text(encoding="utf-8"))
                if saved["identity"] != identity or saved["subset"] != plan or sha(folder / "model.joblib") != saved["model_sha256"]:
                    raise ValueError("Previously fitted subset/model differs")
                selections.append(saved); continue
            trials, reused = [], None
            if plan["fraction"] == 1 and condition == "target-only-basis":
                for name in ("model.joblib", "train.csv", "validation.csv"):
                    shutil.copyfile(baseline_dir / name, folder / name)
                chosen = baseline_info["selected_C"]
                trials = baseline_info["candidates"]
                vscore = pd.read_csv(folder / "validation.csv").score.to_numpy()
                tscore = pd.read_csv(folder / "train.csv").score.to_numpy()
                reused = str(baseline_dir)
                optimization = None
            else:
                best, best_value, chosen = None, -np.inf, None
                for c in cfg["C_candidates"]:
                    model = AdditiveWarning(C=c, groups=train["groups"]) if condition == "target-only-basis" else SourcePriorWarning(
                        source["model"], C=c, prior="source" if condition == "source-basis-prior" else "zero")
                    model.fit(train["pooled"][mask], train["y"][mask], train["people"][mask])
                    predicted = model.predict_proba(validation["pooled"])
                    value = validation_score(validation["y"], predicted)
                    trials.append({"C": c, "validation_selection_score": value,
                        **score_description(validation["y"], predicted), "optimization": getattr(model, "optimization_", None)})
                    if value > best_value:
                        chosen, best, best_value = c, model, value
                tscore, vscore = best.predict_proba(train["pooled"][mask]), best.predict_proba(validation["pooled"])
                if not np.isfinite(tscore).all() or not np.isfinite(vscore).all():
                    raise FloatingPointError("Nonfinite subset probability")
                joblib.dump({"model": best, "feature_spec": {"kind": "pooled", "columns": list(range(296))}}, folder / "model.joblib")
                save_scores(folder, "train", train, mask, tscore)
                save_scores(folder, "validation", validation, val_all, vscore)
                optimization = getattr(best, "optimization_", None)
            negative = vscore[validation["y"] == 0]
            threshold = baseline_info["threshold_val_negative_p90"] if reused else float(np.quantile(negative, .9, method="linear")) if len(negative) else None
            saved = {"identity": identity, "subset": plan, "condition": condition,
                "selected_C": chosen, "candidates": trials, "reused_from": reused,
                "train": baseline_info["train"] if reused else score_description(train["y"][mask], tscore),
                "validation": baseline_info["validation"] if reused else score_description(validation["y"], vscore),
                "validation_people": sorted(set(validation["people"])), "threshold_val_negative_p90": threshold,
                "optimization": optimization, "model_sha256": sha(folder / "model.joblib"),
                "train_scores_sha256": sha(folder / "train.csv"), "validation_scores_sha256": sha(folder / "validation.csv"),
                "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
            write_json(selection_file, saved); selections.append(saved)
            print(json.dumps({"budget": plan["budget"], "seed": plan["seed"], "condition": condition,
                "people": plan["selected_person_count"], "C": chosen, "validation": saved["validation"]}), flush=True)
    write_json(output / "freeze_manifest.json", {"identity": identity, "subsets": plans,
        "selections": {str(destination(output, s["subset"], s["condition"]).relative_to(output)):
            sha(destination(output, s["subset"], s["condition"]) / "selection.json") for s in selections},
        "training_people": sorted(set(train["people"])), "validation_people": sorted(set(validation["people"])),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "test_partition_read": False})
    table = pd.DataFrame([{"fraction": s["subset"]["fraction"], "budget": s["subset"]["budget"],
        "seed": s["subset"]["seed"], "people": s["subset"]["selected_person_count"], "condition": s["condition"],
        "selected_C": s["selected_C"], "training_origins": s["train"]["origins"],
        "training_positive": s["train"]["positive"], **s["validation"]} for s in selections])
    table.to_csv(output / "validation_seed_metrics.csv", index=False)
    table.groupby(["fraction", "budget", "people", "condition"], as_index=False).agg(
        subset_seeds=("seed", "size"), auroc=("auroc", "mean"), auroc_min=("auroc", "min"), auroc_max=("auroc", "max"),
        ap=("ap", "mean"), training_origins_mean=("training_origins", "mean")).to_csv(output / "validation_summary.csv", index=False)


def predict(feature_dir, output, cfg=None, external_cfg=None):
    cfg = configuration() if cfg is None else cfg
    external_cfg = external_configuration() if external_cfg is None else external_cfg
    freeze = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    if freeze["identity"]["config"] != cfg or freeze["identity"]["core"] != implementation_hashes():
        raise ValueError("Frozen transfer definition changed")
    selections = {}
    for relative, expected_hash in freeze["selections"].items():
        folder = output / relative
        if sha(folder / "selection.json") != expected_hash:
            raise ValueError("A frozen subset selection changed")
        selected = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        if sha(folder / "model.joblib") != selected["model_sha256"]:
            raise ValueError("A frozen subset model changed")
        selections[relative] = selected
    # This is the only test open; the fit phase never calls this function.
    test = load_split(feature_dir, "test", external_cfg)
    if (set(freeze["training_people"]) | set(freeze["validation_people"])) & set(test["people"]):
        raise ValueError("A test person was exposed during fitting or selection")
    people = np.unique(test["people"])
    draws = np.random.default_rng(cfg["test_bootstrap"]["seed"]).multinomial(len(people),
        np.full(len(people), 1/len(people)), size=cfg["test_bootstrap"]["replicates"])
    rows, summary, paired, bootstraps, estimates = [], [], [], {}, {}
    predictions = test["meta"].copy()
    for relative, selected in selections.items():
        folder = output / relative; model = joblib.load(folder / "model.joblib")["model"]
        score = model.predict_proba(test["pooled"])
        if not np.isfinite(score).all():
            raise FloatingPointError("Nonfinite held-out transfer prediction")
        save_scores(folder, "test", test, np.ones(len(score), bool), score)
        point, boot, counts = cluster_statistics(test["y"], score, test["people"], selected["threshold_val_negative_p90"], draws)
        plan, condition = selected["subset"], selected["condition"]
        key = (plan["budget"], condition, plan["seed"])
        estimates[key], bootstraps[key] = point, boot
        predictions[relative.replace("/", "__").replace("\\", "__")] = score
        rows.append({"fraction": plan["fraction"], "budget": plan["budget"], "seed": plan["seed"],
            "training_people": plan["selected_person_count"], "condition": condition,
            "selected_C": selected["selected_C"], **counts, **dict(zip(STATISTICS, point))})
    means, mean_boots = {}, {}
    for fraction in FRACTIONS:
        budget = int(fraction*100); seeds = [SEEDS[0]] if fraction == 1 else SEEDS
        for condition in CONDITIONS:
            points = np.asarray([estimates[(budget, condition, seed)] for seed in seeds])
            average_boot = np.nanmean([bootstraps[(budget, condition, seed)] for seed in seeds], axis=0)
            mean = points.mean(0); means[(budget, condition)] = mean; mean_boots[(budget, condition)] = average_boot
            row = {"fraction": fraction, "budget": budget, "condition": condition, "subset_seeds": len(seeds),
                "training_people": cfg["expected_subset_people"][str(budget)], "test_origins": len(test["y"]), "test_people": len(people)}
            for j, statistic in enumerate(STATISTICS):
                finite = np.isfinite(average_boot[:, j])
                ci = np.quantile(average_boot[finite, j], [.025, .975]) if finite.any() else [np.nan, np.nan]
                row.update({statistic: mean[j], statistic+"_seed_min": points[:, j].min(), statistic+"_seed_max": points[:, j].max(),
                    statistic+"_ci_low": ci[0], statistic+"_ci_high": ci[1], statistic+"_bootstrap_valid": int(finite.sum())})
            summary.append(row)
        for baseline in CONDITIONS[:2]:
            for j, statistic in enumerate(STATISTICS):
                delta = mean_boots[(budget, "source-basis-prior")][:, j] - mean_boots[(budget, baseline)][:, j]
                finite = np.isfinite(delta)
                ci = np.quantile(delta[finite], [.025, .975]) if finite.any() else [np.nan, np.nan]
                paired.append({"budget": budget, "model": "source-basis-prior", "baseline": baseline, "metric": statistic,
                    "difference": means[(budget, "source-basis-prior")][j]-means[(budget, baseline)][j],
                    "ci_low": ci[0], "ci_high": ci[1], "bootstrap_valid": int(finite.sum())})
    predictions.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "test_seed_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "test_paired_differences.csv", index=False)
    np.savez_compressed(output / "test_bootstrap.npz", metric_names=np.asarray(STATISTICS),
        **{f"budget{budget}__{condition}": boot for (budget, condition), boot in mean_boots.items()})
    write_json(output / "test_summary.json", {"test_hashes": test["hashes"], "origins": len(test["y"]),
        "people": len(people), "positive": int(test["y"].sum()), "freeze_manifest_sha256": sha(output / "freeze_manifest.json"),
        "seed_summary": cfg["seed_summary"], "prediction_at_utc": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["fit", "predict"], required=True)
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/human_rhythms")
    parser.add_argument("--output", type=Path, default=ROOT / "results/human_rhythms_transfer")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.phase == "fit":
            fit(args.features, args.output)
        else:
            predict(args.features, args.output)


if __name__ == "__main__":
    main()
