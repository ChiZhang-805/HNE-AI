"""Daily FCM/CATCH runs, separate score-only LR, and three-seed ECDF ensembles."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def future_windows(meta, ids, dataset):
    """Read future trajectories only for this fold's train/validation people."""
    selected, future = [], []
    for person, rows in meta.loc[ids].groupby("person_id", sort=False):
        with np.load(ROOT / "data/daily_study" / dataset / f"{person}_daily.npz") as raw:
            trajectory = np.column_stack([raw["normalized"], raw["normalized_contexts"]])
        for index, row in rows.iterrows():
            start = int(row.origin_day) + 1
            if start + 32 <= len(trajectory):
                selected.append(index)
                future.append(trajectory[start:start + 32])
    if not selected:
        raise ValueError(f"No complete FCM future followup for {dataset}")
    return np.asarray(selected), np.asarray(future, dtype=np.float32)


def imputation(train):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=(0, 1))
    missing = np.flatnonzero(~np.isfinite(medians)).tolist()
    medians[~np.isfinite(medians)] = 0.
    return medians.astype(np.float32), missing


def fill(array, medians):
    return np.where(np.isfinite(array), array, medians).astype(np.float32)


def write_scores(folder, meta, partitions, scores):
    folder.mkdir(parents=True, exist_ok=True)
    for part, ids in partitions.items():
        table = meta.loc[ids].copy()
        table["score"] = scores[part]
        table.to_csv(folder / f"{part}.csv", index=False)


def calibrate(folder, meta, partitions, scores, seed):
    train_ids, val_ids = partitions["train"], partitions["validation"]
    train_y = meta.loc[train_ids, "y"].to_numpy()
    val_y = meta.loc[val_ids, "y"].to_numpy()
    candidates, best_loss, best_model, selected_c = [], np.inf, None, None
    for c in [.01, .1, 1.]:
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=c, random_state=seed, max_iter=1000))
        estimator.fit(scores["train"][:, None], train_y)
        loss = float(log_loss(val_y, estimator.predict_proba(scores["validation"][:, None])[:, 1], labels=[0, 1]))
        candidates.append(dict(C=c, validation_log_loss=loss))
        if loss < best_loss:
            best_loss, best_model, selected_c = loss, estimator, c
    probabilities = {part: best_model.predict_proba(value[:, None])[:, 1] for part, value in scores.items()}
    write_scores(folder, meta, partitions, probabilities)
    metadata = dict(method=folder.parent.name, source_method=folder.parent.name.removesuffix("-score-LR"),
                    fit="single raw anomaly score; training labels only", selection="minimum validation log-loss",
                    C=selected_c, candidates=candidates, seed=seed,
                    coefficient=float(best_model[-1].coef_[0, 0]), intercept=float(best_model[-1].intercept_[0]),
                    score_mean=float(best_model[0].mean_[0]), score_scale=float(best_model[0].scale_[0]),
                    train_windows=len(train_ids), validation_windows=len(val_ids))
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run(dataset, method, held_fold, batch_size, seed, output):
    feature_dir = ROOT / "data/features/unified_daily"
    with np.load(feature_dir / f"{dataset}.npz") as arrays:
        windows, folds, labels = arrays["windows"], arrays["fold"], arrays["y"]
    meta = pd.read_csv(feature_dir / f"{dataset}.csv", dtype={"person_id": str})
    if len(meta) != len(windows) or not np.array_equal(meta.y.to_numpy(), labels):
        raise ValueError("Feature and metadata rows disagree")
    validation_fold = (held_fold + 1) % 5
    partitions = dict(train=np.flatnonzero((folds != held_fold) & (folds != validation_fold)),
                      validation=np.flatnonzero(folds == validation_fold), held_person=np.flatnonzero(folds == held_fold))
    folder = output / dataset / method / f"fold{held_fold}"
    lr_folder = folder.parent.parent / f"{method}-score-LR" / folder.name
    complete = folder / "metadata.json"
    saved = json.loads(complete.read_text()) if complete.exists() else None
    if saved and saved["epochs_completed"] == 20:
        saved_seed = saved.get("seed", saved["config"].get("seed"))
        if saved_seed != seed or saved["config"]["batch_size"] != batch_size:
            raise ValueError(f"Existing completed fit has different seed/batch size: {folder}")
        scores = {part: pd.read_csv(folder / f"{part}.csv")["score"].to_numpy() for part in partitions}
        calibrate(lr_folder, meta, partitions, scores, seed)
        print(f"RESUME {dataset} {method} fold{held_fold}: completed20 epochs retained", flush=True)
        return
    train_ids, val_ids = partitions["train"], partitions["validation"]
    future = validation_future = None
    if method == "FCM":
        train_ids, future = future_windows(meta, train_ids, dataset)
        val_ids, validation_future = future_windows(meta, val_ids, dataset)
    medians, all_missing = imputation(windows[train_ids])
    train, validation = fill(windows[train_ids], medians), fill(windows[val_ids], medians)
    print(f"START {dataset} {method} fold{held_fold}: train={len(train_ids)} validation={len(val_ids)} held={len(partitions['held_person'])} batch={batch_size}", flush=True)
    if method == "FCM":
        from src.vendor_fcm import fit
        fitted = fit(train, fill(future, medians), validation, fill(validation_future, medians), folder,
                     batch_size=batch_size, seed=seed, epochs=20)
    else:
        from src.vendor_catch import fit
        fitted = fit(train, validation, folder, batch_size=batch_size, seed=seed, epochs=20)
    scores = {part: fitted.score(fill(windows[ids], medians)) for part, ids in partitions.items()}
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise FloatingPointError("Nonfinite public baseline scores")
    write_scores(folder, meta, partitions, scores)
    metadata = fitted.metadata
    metadata.update(dataset=dataset, held_fold=held_fold, validation_fold=validation_fold,
                    split_people={part: sorted(meta.loc[ids, "person_id"].unique().tolist()) for part, ids in partitions.items()},
                    scored_windows={part: len(ids) for part, ids in partitions.items()},
                    tail_without32day_future=dict(train=len(partitions["train"])-len(train_ids), validation=len(partitions["validation"])-len(val_ids)),
                    imputation=dict(rule="per-channel training-history median; no temporal padding", medians=medians.tolist(), all_missing_channels_zero=all_missing),
                    source=json.loads((ROOT / "third_party" / method / "SOURCE.json").read_text()),
                    task="same fixed unified daily future-disorder endpoint; no event labels in neural objectives",
                    input_shape=list(windows.shape[1:]), torch_version=torch.__version__, device_name=torch.cuda.get_device_name(0))
    complete.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    calibrate(lr_folder, meta, partitions, scores, seed)
    print(f"DONE {dataset} {method} fold{held_fold}: epochs20 selected{metadata['best_epoch']} seconds={metadata['training_seconds']:.1f}", flush=True)
    del fitted
    gc.collect(); torch.cuda.empty_cache()


def ensemble_repeats(output, original, datasets, methods, folds):
    """Equal-weight all seeds after a frozen validation-negative ECDF per fold."""
    seeds = [20261110, 20261111, 20261112]
    roots = {seed: original if seed == seeds[0] else output / f"seed{seed}" for seed in seeds}
    parts = ["train", "validation", "held_person"]
    keys = ["person_id", "origin_day"]
    per_fold, aggregate, training = [], [], []
    for dataset in datasets:
        for method in methods:
            collected = {(str(seed), part): [] for seed in seeds for part in parts}
            collected.update({("ensemble", part): [] for part in parts})
            for fold in folds:
                folders = {seed: root / dataset / method / f"fold{fold}" for seed, root in roots.items()}
                tables, references = {}, {}
                for seed, folder in folders.items():
                    metadata = json.loads((folder / "metadata.json").read_text())
                    actual_seed = metadata.get("seed", metadata["config"].get("seed"))
                    if metadata["epochs_completed"] != 20 or actual_seed != seed:
                        raise ValueError(f"Incomplete or wrong-seed fit: {folder}")
                    training.append(dict(dataset=dataset, method=method, seed=seed, fold=fold,
                                         epochs=metadata["epochs_completed"], best_epoch=metadata["best_epoch"],
                                         train_windows=metadata["train_windows"], validation_windows=metadata["validation_windows"],
                                         seconds=metadata["training_seconds"]))
                    tables[seed] = {part: pd.read_csv(folder / f"{part}.csv", dtype={"person_id": str}) for part in parts}
                    validation = tables[seed]["validation"]
                    references[seed] = np.sort(validation.loc[validation.y.eq(0), "score"].to_numpy())
                    if not len(references[seed]) or not np.isfinite(references[seed]).all():
                        raise ValueError(f"No finite validation-negative ECDF: {folder}")
                target = output / "ensemble" / dataset / method / f"fold{fold}"
                target.mkdir(parents=True, exist_ok=True)
                for part in parts:
                    combined = tables[seeds[0]][part].copy()
                    index = pd.MultiIndex.from_frame(combined[keys])
                    quantiles = []
                    for seed in seeds:
                        table = tables[seed][part].set_index(keys).reindex(index).reset_index()
                        if not np.array_equal(table.y, combined.y) or not np.isfinite(table.score).all():
                            raise ValueError(f"Seed rows disagree: {dataset}/{method}/fold{fold}/{part}")
                        raw = table.score.to_numpy()
                        quantile = np.searchsorted(references[seed], raw, side="right") / len(references[seed])
                        combined[f"seed{seed}_score"] = raw
                        combined[f"seed{seed}_quantile"] = quantile
                        quantiles.append(quantile)
                        table["validation_negative_quantile"] = quantile
                        table["evaluation_fold"] = fold
                        collected[(str(seed), part)].append(table)
                    combined["score"] = np.mean(quantiles, axis=0)
                    combined.to_csv(target / f"{part}.csv", index=False)
                    combined["validation_negative_quantile"] = combined.score
                    combined["evaluation_fold"] = fold
                    collected[("ensemble", part)].append(combined)
                info = dict(dataset=dataset, method=f"{method}-3seed-ECDF", held_fold=fold,
                            seeds=seeds, weights=[1/3]*3, selection="all three prespecified seeds; no seed selection",
                            transform="F_seed(s)=number(validation y=0 raw scores <= s)/number(validation y=0)",
                            aggregation="arithmetic mean of three validation-negative quantiles",
                            application="same frozen per-seed per-fold ECDF for train, validation and held_person",
                            validation_negative_counts={str(seed): len(value) for seed, value in references.items()},
                            sources={str(seed): str(folder) for seed, folder in folders.items()})
                (target / "metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            for (seed, part), frames in collected.items():
                frame = pd.concat(frames, ignore_index=True)
                numerator, pair_count = 0., 0
                for fold, group in frame.groupby("evaluation_fold"):
                    positives = int(group.y.sum())
                    pairs = positives * (len(group) - positives)
                    auc = float(roc_auc_score(group.y, group.score)) if pairs else np.nan
                    if pairs:
                        numerator += auc*pairs
                        pair_count += pairs
                    per_fold.append(dict(dataset=dataset, method=method, seed=seed, partition=part, fold=fold,
                                         origins=len(group), positives=positives, auroc=auc,
                                         ap=float(average_precision_score(group.y, group.score))))
                aggregate.append(dict(dataset=dataset, method=method, seed=seed, partition=part,
                                      origins=len(frame), people=frame.person_id.nunique(), positives=int(frame.y.sum()),
                                      auroc=numerator/pair_count if pair_count else np.nan,
                                      ap_on_validation_negative_quantile=float(average_precision_score(frame.y, frame.validation_negative_quantile))))
    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training).to_csv(summary / "training.csv", index=False)
    pd.DataFrame(per_fold).to_csv(summary / "per_fold.csv", index=False)
    metrics = pd.DataFrame(aggregate)
    metrics.to_csv(summary / "metrics.csv", index=False)
    variability = metrics[metrics.seed.ne("ensemble")].groupby(["dataset", "method", "partition"]).agg(
        seeds=("seed", "nunique"), auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auroc_min=("auroc", "min"), auroc_max=("auroc", "max"))
    variability.to_csv(summary / "seed_variation.csv")
    completion = dict(datasets=list(datasets), methods=list(methods), folds=list(folds),
                      seeds=seeds, new_fits=sum(row["seed"] != seeds[0] for row in training),
                      all_fits=len(training), new_epochs=sum(row["epochs"] for row in training if row["seed"] != seeds[0]),
                      all_epochs=sum(row["epochs"] for row in training), ensemble="all seeds with equal weights after validation-negative ECDF",
                      auroc="positive-negative-pair weighted AUROC within evaluation folds; no cross-model raw-score comparison",
                      training_rows="training samples occur in three folds, so train aggregate counts repeat origins")
    (summary / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(metrics[metrics.partition.eq("held_person")].to_string(index=False), flush=True)
    print(json.dumps(completion), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["RESILIENT", "BrisT1D-Open"], default=["RESILIENT", "BrisT1D-Open"])
    parser.add_argument("--methods", nargs="+", choices=["FCM", "CATCH"], default=["FCM", "CATCH"])
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--output", type=Path, default=ROOT / "results/unified_daily",
                        help="Training output root; for --ensemble-only, parent of seed20261111/seed20261112.")
    parser.add_argument("--ensemble-only", action="store_true", help="Assemble all three prespecified seeds without training.")
    parser.add_argument("--original-output", type=Path, default=ROOT / "results/unified_daily")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.ensemble_only:
        original = args.original_output if args.original_output.is_absolute() else ROOT / args.original_output
        ensemble_repeats(output, original, args.datasets, args.methods, args.folds)
        return
    for dataset in args.datasets:
        for method in args.methods:
            for fold in args.folds:
                run(dataset, method, fold, args.batch_size, args.seed, output)


if __name__ == "__main__":
    main()
