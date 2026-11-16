"""Build seven-day auxiliary targets, fit both fixed conditions, then freeze.

Only train/validation daily trajectories construct auxiliary targets. The risk
inputs and event origins stay identical to the existing temporal comparison.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_future import fit_predict, load_model
from src.hne_temporal import predict_proba
from src.hne_additive import validation_score
from scripts.run_temporal_external import read_partition, save_scores, configuration as temporal_configuration
from scripts.run_hne_external import score_description
from scripts.run_transfer import sha, write_json


def configuration():
    return json.loads((ROOT / "configs/future_objective.json").read_text(encoding="utf-8"))


def build_targets(features, daily, target_folder, cfg):
    target_folder.mkdir(parents=True, exist_ok=True)
    for split in cfg["fit_partitions"]:
        meta = pd.read_csv(features / f"{split}.csv", dtype={"person_id": str})
        source = {str(person): sha(daily / split / f"{person}_daily.npz") for person in meta.person_id.unique()}
        identity = {"config": cfg, "metadata_sha256": sha(features / f"{split}.csv"), "daily_sha256": source}
        description = target_folder / f"{split}.json"
        target_path = target_folder / f"{split}.npz"
        if description.exists():
            old = json.loads(description.read_text(encoding="utf-8"))
            if old["identity"] != identity or sha(target_path) != old["targets_sha256"]:
                raise ValueError("Existing auxiliary targets differ from the declared source")
            continue
        with np.load(features / f"{split}.npz", allow_pickle=False) as z:
            p = z["nodes"].shape[1]
        values = np.full((len(meta), p, 7), np.nan, np.float32)
        moments = np.full((len(meta), p, 3), np.nan, np.float32)
        valid_node = np.zeros((len(meta), p), bool)
        for person, rows in meta.groupby("person_id", sort=False):
            with np.load(daily / split / f"{person}_daily.npz", allow_pickle=False) as z:
                normalized = z["normalized"].astype(np.float64)
            if normalized.shape[1] != p:
                raise ValueError("Daily/feature state coordinate counts disagree")
            reference = normalized[:cfg["reference_days"]]
            supported = np.isfinite(reference).sum(0) >= cfg["minimum_reference_observations"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                threshold = np.maximum(np.nanquantile(np.abs(reference), .9, axis=0), .1)
            for index, row in rows.iterrows():
                d = int(row.origin_day)
                future = normalized[d+cfg["future_start_offset"]:d+cfg["future_end_offset"]+1]
                if len(future) != 7:
                    raise ValueError("An existing event origin lacks its seven calendar target days")
                observed = np.isfinite(future)
                count = observed.sum(0)
                good = supported & (count >= cfg["minimum_future_observations"])
                valid_node[index] = good
                values[index, good] = future[:, good].T
                for j in np.flatnonzero(good):
                    x = future[observed[:, j], j]
                    moments[index, j] = [x.mean(), np.log1p(x.var(ddof=0)), np.mean(np.abs(x) > threshold[j])]
        np.savez_compressed(target_path, values=values, moments=moments, valid_node=valid_node)
        record = {"identity": identity, "targets_sha256": sha(target_path), "origins": len(meta),
            "people": int(meta.person_id.nunique()), "valid_nodes": int(valid_node.sum()),
            "origins_without_aux": int((~valid_node.any(1)).sum()),
            "origins_not_full32_future_retained": int((~meta.future32_available.astype(bool)).sum()) if "future32_available" in meta else None,
            "values_shape": list(values.shape), "moments_shape": list(moments.shape)}
        write_json(description, record)
        print(json.dumps({"targets_ready": split, **{k:v for k,v in record.items() if k != "identity"}}), flush=True)


def run_one(train, validation, cfg, condition, weight, seed, folder, identity, device):
    fingerprint = hashlib.sha256(json.dumps({"identity": identity, "condition": condition,
        "auxiliary_weight": weight, "seed": seed}, sort_keys=True).encode()).hexdigest()
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["run_fingerprint"] != fingerprint:
            raise ValueError(f"Existing auxiliary fit differs: {folder}")
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            probability = saved["val_probability"]
    else:
        print(f"START {condition} lambda={weight} seed={seed}", flush=True)
        result = fit_predict(train, validation, train["groups"], condition, weight, seed, folder,
                             learning_rate=cfg["learning_rate"], device=device, run_fingerprint=fingerprint)
        probability, metadata = result["val_probability"], result["metadata"]
    save_scores(folder / "validation.csv", validation, probability)
    return probability, metadata


def fit(features, daily, targets, output, cfg, device):
    output.mkdir(parents=True, exist_ok=True)
    plan = output / "configuration.json"
    if plan.exists() and json.loads(plan.read_text(encoding="utf-8")) != cfg:
        raise ValueError("Existing auxiliary experiment configuration differs")
    write_json(plan, cfg)
    build_targets(features, daily, targets, cfg)
    expected = temporal_configuration()
    train, validation = [read_partition(features, part, expected) for part in cfg["fit_partitions"]]
    if set(train["person"]) & set(validation["person"]):
        raise ValueError("Training and validation people overlap")
    identity = {"config": cfg, "train": train["hashes"], "validation": validation["hashes"],
        "targets": {part: sha(targets / f"{part}.npz") for part in cfg["fit_partitions"]},
        "core": {name: sha(ROOT / "src" / name) for name in ["hne_future.py", "hne_temporal.py", "hne_set.py"]}}
    fit_config = output / "fit_config.json"
    if fit_config.exists() and json.loads(fit_config.read_text(encoding="utf-8")) != identity:
        raise ValueError("Existing auxiliary experiment inputs differ")
    write_json(fit_config, identity)
    selected, summary = {}, []
    for condition in cfg["conditions"]:
        folder = output / condition; folder.mkdir(exist_ok=True)
        selection_path = folder / "selection.json"
        if selection_path.exists():
            result = json.loads(selection_path.read_text(encoding="utf-8"))
            if result["identity"] != identity:
                raise ValueError("Existing auxiliary selection differs")
        else:
            for part, data in zip(cfg["fit_partitions"], [train, validation]):
                with np.load(targets / f"{part}.npz", allow_pickle=False) as z:
                    data["auxiliary_target"] = z["values" if condition == "future-values" else "moments"]
            trials, best, best_score, chosen = [], None, -np.inf, None
            for weight in cfg["auxiliary_weights"]:
                pred, metadata = run_one(train, validation, cfg, condition, weight, cfg["seeds"][0],
                    folder / f"lambda{weight}" / f"seed{cfg['seeds'][0]}", identity, device)
                score = validation_score(validation["y"], pred)
                trials.append({"auxiliary_weight": weight, "validation_selection_score": score,
                               "best_epoch": metadata["best_epoch"], **score_description(validation["y"], pred)})
                if score > best_score:
                    best, best_score, chosen = (pred, metadata), score, weight
            repeats = [best]
            for seed in cfg["seeds"][1:]:
                repeats.append(run_one(train, validation, cfg, condition, chosen, seed,
                    folder / f"lambda{chosen}" / f"seed{seed}", identity, device))
            seed_probability = np.stack([r[0] for r in repeats])
            probability = seed_probability.mean(0)
            save_scores(folder / "validation.csv", validation, probability)
            np.savez_compressed(folder / "validation_predictions.npz", seed_probability=seed_probability,
                                probability=probability, seeds=np.asarray(cfg["seeds"]))
            person_rows = [{"person_id": str(person), **score_description(validation["y"][ix], probability[ix])}
                           for person, ix in validation["meta"].groupby("person_id").groups.items()]
            pd.DataFrame(person_rows).to_csv(folder / "validation_people.csv", index=False)
            negative = probability[validation["y"] == 0]
            checkpoints = [f"{condition}/lambda{chosen}/seed{seed}/model.pt" for seed in cfg["seeds"]]
            result = {"identity": identity, "condition": condition, "selected_lambda": chosen,
                "trials": trials, "seeds": cfg["seeds"], "weights": [1/3]*3,
                "validation": score_description(validation["y"], probability),
                "individual_seed": [{"seed": seed, "best_epoch": rep[1]["best_epoch"],
                    "validation": score_description(validation["y"], rep[0])} for seed,rep in zip(cfg["seeds"], repeats)],
                "threshold_val_negative_p90": float(np.quantile(negative, .9, method="linear")) if len(negative) else None,
                "checkpoint_sha256": {name: sha(output / name) for name in checkpoints},
                "validation_path": f"{condition}/validation.csv", "validation_sha256": sha(folder / "validation.csv"),
                "train_people": sorted(set(train["person"])), "validation_people": sorted(set(validation["person"])),
                "test_partition_read": False, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
            write_json(selection_path, result)
        selected[condition] = {"selected_lambda": result["selected_lambda"], "seeds": result["seeds"],
            "checkpoint_sha256": result["checkpoint_sha256"], "validation_path": result["validation_path"],
            "validation_sha256": result["validation_sha256"],
            "selection_path": f"{condition}/selection.json", "selection_sha256": sha(selection_path)}
        summary.append({"condition": condition, "selected_lambda": result["selected_lambda"], **result["validation"]})
        print(json.dumps({"condition_complete": condition, "selected_lambda": result["selected_lambda"],
                          "validation": result["validation"]}), flush=True)
    pd.DataFrame(summary).to_csv(output / "validation_summary.csv", index=False)
    write_json(output / "freeze_manifest.json", {"identity": identity, "conditions": selected,
        "test_partition_read": False, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})


def predict(features, output, cfg, device):
    frozen = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    if frozen["identity"]["config"] != cfg:
        raise ValueError("Frozen auxiliary configuration changed")
    for name, digest in frozen["identity"]["core"].items():
        if sha(ROOT / "src" / name) != digest:
            raise ValueError("Frozen auxiliary core changed")
    for result in frozen["conditions"].values():
        if sha(output / result["selection_path"]) != result["selection_sha256"]:
            raise ValueError("Frozen auxiliary selection changed")
        for name, digest in result["checkpoint_sha256"].items():
            if sha(output / name) != digest:
                raise ValueError("Frozen auxiliary checkpoint changed")
    test = read_partition(features, "test", temporal_configuration())
    for condition, result in frozen["conditions"].items():
        selection = json.loads((output / result["selection_path"]).read_text(encoding="utf-8"))
        if (set(selection["train_people"]) | set(selection["validation_people"])) & set(test["people"]):
            raise ValueError("Test people overlap earlier partitions")
        scores = []
        for seed in cfg["seeds"]:
            folder = output / condition / f"lambda{result['selected_lambda']}" / f"seed{seed}"
            model, _ = load_model(folder, device=device)
            p = predict_proba(model, test["windows"], test["nodes"], test["pooled"], test["groups"])
            save_scores(folder / "test.csv", test, p); scores.append(p)
            del model
        save_scores(output / condition / "test.csv", test, np.mean(scores, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["fit", "predict"], default="fit")
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/college_experience")
    parser.add_argument("--daily", type=Path, default=ROOT / "data/college_study")
    parser.add_argument("--targets", type=Path, default=ROOT / "data/features/future_objective/College-Experience")
    parser.add_argument("--output", type=Path, default=ROOT / "results/college_future_objective")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--all-methods-frozen", action="store_true")
    args = parser.parse_args()
    if args.phase == "predict" and not args.all_methods_frozen:
        parser.error("predict requires --all-methods-frozen; test files have not been opened")
    with threadpool_limits(limits=2):
        if args.phase == "fit":
            fit(args.features, args.daily, args.targets, args.output, configuration(), args.device)
        else:
            predict(args.features, args.output, configuration(), args.device)


if __name__ == "__main__":
    main()
