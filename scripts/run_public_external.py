"""Fixed daily FCM/CATCH training, then explicitly frozen test prediction.

fit opens train/validation only. predict needs --all-methods-frozen and opens
only test histories, never test targets or future trajectories. No tuning of
the neural settings or the daily event definition occurs in this entry point.
"""
from __future__ import annotations
import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = [20261110, 20261111, 20261112]
METHODS = ["FCM", "CATCH"]


def read_history(features, part, *, future=False):
    with np.load(features / f"{part}.npz") as data:
        windows = data["windows"].astype(np.float32)
        targets = data["future"].astype(np.float32) if future else None
        available = data["future_available"] if future else None
    # Event labels are read separately after the unsupervised network fit.
    meta = pd.read_csv(features / f"{part}.csv", dtype={"person_id": str}, usecols=lambda name: name != "y")
    if windows.ndim != 3 or windows.shape[1] != 32 or windows.shape[2] < 1 or len(meta) != len(windows):
        raise ValueError(f"Expected aligned {part} metadata and N x32 xC histories, C >0")
    if future:
        if targets.shape != windows.shape or available.shape != (len(windows),) or not np.isin(available, [0, 1]).all():
            raise ValueError(f"Invalid {part} future/future_available shapes or flags")
        available = available.astype(bool)
    return windows, targets, available, meta


def imputer(history):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(np.isfinite(history), history, np.nan), axis=(0, 1))
    missing = np.flatnonzero(~np.isfinite(median)).tolist()
    median[~np.isfinite(median)] = 0.
    return median.astype(np.float32), missing


def fill(array, median):
    return np.where(np.isfinite(array), array, median).astype(np.float32)


def write_scores(folder, part, meta, score, labels=None):
    folder.mkdir(parents=True, exist_ok=True)
    table = meta.copy()
    if labels is not None:
        table["y"] = labels
    table["score"] = score
    if not np.isfinite(table.score).all():
        raise FloatingPointError(f"Nonfinite scores: {folder}/{part}")
    table.to_csv(folder / f"{part}.csv", index=False)
    return table


def score_metrics(labels, score):
    return dict(origins=len(labels), positives=int(np.sum(labels)),
                auroc=float(roc_auc_score(labels, score)) if len(np.unique(labels)) == 2 else None,
                ap=float(average_precision_score(labels, score)))


def fit_score_lr(features, output, method, seed, train_score, val_score, train_meta, val_meta):
    # This is the only training-label read in the entire entry point.
    with np.load(features / "train.npz") as data:
        train_y = data["y"]
    with np.load(features / "validation.npz") as data:
        val_y = data["y"]
    if len(train_y) != len(train_score) or len(val_y) != len(val_score):
        raise ValueError("Score/label row counts disagree")
    best_loss, selected, best, trials = np.inf, None, None, []
    for c in [.01, .1, 1.]:
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=c, random_state=seed, max_iter=1000))
        estimator.fit(train_score[:, None], train_y)
        value = float(log_loss(val_y, estimator.predict_proba(val_score[:, None])[:, 1], labels=[0, 1]))
        trials.append(dict(C=c, validation_log_loss=value))
        if value < best_loss:
            best_loss, selected, best = value, c, estimator
    folder = output / f"{method}-score-LR" / f"seed{seed}"
    write_scores(folder, "train", train_meta, best.predict_proba(train_score[:, None])[:, 1], train_y)
    write_scores(folder, "validation", val_meta, best.predict_proba(val_score[:, None])[:, 1], val_y)
    selection = dict(method=f"{method}-score-LR", seed=seed, C=selected, candidates=trials,
                     fit="training labels on one frozen raw anomaly score only", selection="minimum validation log-loss",
                     score_mean=float(best[0].mean_[0]), score_scale=float(best[0].scale_[0]),
                     coefficient=float(best[-1].coef_[0, 0]), intercept=float(best[-1].intercept_[0]),
                     train_windows=len(train_score), validation_windows=len(val_score))
    (folder / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return val_y


def fit_one(features, output, method, seed, partition_unit="person"):
    folder = output / method / f"seed{seed}"
    if (folder / "selection.json").exists():
        print(f"RETAIN {method} seed{seed}: completed fixed fit", flush=True)
        return
    needs_future = method == "FCM"
    train, future, train_mask, train_meta = read_history(features, "train", future=needs_future)
    val, val_future, val_mask, val_meta = read_history(features, "validation", future=needs_future)
    datasets = train_meta["dataset"].dropna().unique()
    if len(datasets) != 1 or set(val_meta["dataset"]) != {datasets[0]}:
        raise ValueError("Training and validation must identify the same single dataset")
    dataset = str(datasets[0])
    if train.shape[1:] != val.shape[1:]:
        raise ValueError("Training and validation history dimensions differ")
    if partition_unit == "person" and set(train_meta.person_id) & set(val_meta.person_id):
        raise ValueError("Training and validation people overlap")
    if partition_unit == "time" and train_meta.future_end_date.max() >= val_meta.issue_time.min()[:10]:
        raise ValueError("Training outcomes reach the validation issue period")
    if not needs_future:
        train_mask, val_mask = np.ones(len(train), bool), np.ones(len(val), bool)
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"{method} has no eligible training or validation windows")
    train_meta["neural_fit_eligible"], val_meta["neural_fit_eligible"] = train_mask, val_mask
    median, missing = imputer(train[train_mask])
    print(f"START {dataset} {method} seed{seed}: train={train_mask.sum()} validation={val_mask.sum()} batch8 epochs20", flush=True)
    if needs_future:
        from src.vendor_fcm import fit
        fitted = fit(fill(train[train_mask], median), fill(future[train_mask], median),
                     fill(val[val_mask], median), fill(val_future[val_mask], median), folder,
                     seed=seed, epochs=20, batch_size=8)
    else:
        from src.vendor_catch import fit
        fitted = fit(fill(train, median), fill(val, median), folder, seed=seed, epochs=20, batch_size=8)
    train_score, val_score = fitted.score(fill(train, median)), fitted.score(fill(val, median))
    metadata = fitted.metadata
    metadata.update(dataset=dataset, seed=seed, input_shape=list(train.shape[1:]),
                    source=json.loads((ROOT / "third_party" / method / "SOURCE.json").read_text()),
                    input_splits=["train", "validation"],
                    split_people=dict(train=sorted(train_meta.person_id.unique().tolist()), validation=sorted(val_meta.person_id.unique().tolist())),
                    scored_windows=dict(train=len(train), validation=len(val)),
                    future_filter=dict(rule="complete32-day calendar followup indicated by future_available" if needs_future else "not required for reconstruction",
                                       train_selected=int(train_mask.sum()), validation_selected=int(val_mask.sum()),
                                       train_excluded=int((~train_mask).sum()), validation_excluded=int((~val_mask).sum())),
                    imputation=dict(rule="per-channel training-history median; no temporal padding", medians=median.tolist(), all_missing_channels_zero=missing))
    shape_text = f"{train.shape[1]}x{train.shape[2]}"
    metadata["adaptations"] = [item.replace("32x11", shape_text) for item in metadata["adaptations"]]
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_scores(folder, "train", train_meta, train_score)
    # Label-based operations are downstream of the fixed unsupervised fit.
    val_y = fit_score_lr(features, output, method, seed, train_score, val_score, train_meta, val_meta)
    write_scores(folder, "validation", val_meta, val_score, val_y)
    negative = np.sort(val_score[val_y == 0])
    if not len(negative):
        raise ValueError("Validation-negative ECDF needs at least one negative origin")
    selection = dict(method=method, seed=seed, epochs_completed=20, best_epoch=metadata["best_epoch"],
                     neural_selection=metadata["checkpoint_selection"], validation_metrics=score_metrics(val_y, val_score),
                     validation_negative_scores=negative.tolist(),
                     ecdf="number(validation-negative raw scores <= score)/number(validation-negative origins)",
                     imputation_medians=median.tolist(), checkpoint="checkpoint.pt")
    (folder / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"DONE {dataset} {method} seed{seed}:20 epochs", flush=True)
    del fitted
    gc.collect(); torch.cuda.empty_cache()


def restore(folder, method, seed):
    # Trusted local checkpoint created by the fixed training phase above.
    saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((folder / "metadata.json").read_text())
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = np.asarray(saved["scaler_mean"]), np.asarray(saved["scaler_scale"])
    scaler.var_, scaler.n_features_in_ = scaler.scale_ ** 2, len(scaler.mean_)
    if method == "FCM":
        from src import vendor_fcm
        with vendor_fcm.author_imports():
            model_class = importlib.import_module("models.FCM").Model
        model = model_class(SimpleNamespace(**saved["model_config"])).float().to("cuda")
        model.load_state_dict(saved["state_dict"])
        return vendor_fcm.FittedFCM(model, scaler, saved["config"], metadata)
    from src import vendor_catch
    config = vendor_catch.TransformerConfig(**saved["config"])
    model = vendor_catch.CATCHModel(config).to("cuda")
    model.load_state_dict(saved["state_dict"])
    return vendor_catch.FittedCATCH(model, scaler, config, metadata, "cuda", seed)


def predict(features, output, methods, seeds):
    # This function is entered only after the explicit all-methods freeze flag.
    for method in methods:
        for seed in seeds:
            for name in [method, f"{method}-score-LR"]:
                if not (output / name / f"seed{seed}" / "selection.json").is_file():
                    raise ValueError(f"Fit/selection is incomplete for {name} seed{seed}")
    windows, _, _, meta = read_history(features, "test")
    for method in methods:
        for seed in seeds:
            folder = output / method / f"seed{seed}"
            selection = json.loads((folder / "selection.json").read_text())
            fitted = restore(folder, method, seed)
            score = fitted.score(fill(windows, np.asarray(selection["imputation_medians"], dtype=np.float32)))
            write_scores(folder, "test", meta, score)
            lr_folder = output / f"{method}-score-LR" / f"seed{seed}"
            lr = json.loads((lr_folder / "selection.json").read_text())
            probability = expit((score-lr["score_mean"])/lr["score_scale"]*lr["coefficient"]+lr["intercept"])
            write_scores(lr_folder, "test", meta, probability)
            del fitted
            gc.collect(); torch.cuda.empty_cache()


def ensemble(output, method, parts):
    folders = {seed: output / method / f"seed{seed}" for seed in SEEDS}
    if any(not (folder / "selection.json").is_file() or any(not (folder / f"{part}.csv").is_file() for part in parts) for folder in folders.values()):
        print(f"ENSEMBLE {method}: waiting for all three fixed seeds", flush=True)
        return
    selections = {seed: json.loads((folder / "selection.json").read_text()) for seed, folder in folders.items()}
    target = output / method / "ensemble"
    target.mkdir(parents=True, exist_ok=True)
    validation_metrics = None
    for part in parts:
        tables = {seed: pd.read_csv(folder / f"{part}.csv", dtype={"person_id": str}) for seed, folder in folders.items()}
        table = tables[SEEDS[0]].copy()
        keys = ["person_id", "origin_day"]
        index = pd.MultiIndex.from_frame(table[keys])
        components = []
        for seed in SEEDS:
            aligned = tables[seed].set_index(keys).reindex(index)
            score = aligned.score.to_numpy()
            if not np.isfinite(score).all():
                raise ValueError(f"Missing/nonfinite seed scores in {method}/{part}")
            if "y" in table and not np.array_equal(aligned.y.to_numpy(), table.y.to_numpy()):
                raise ValueError(f"Seed validation labels disagree in {method}/{part}")
            negative = np.asarray(selections[seed]["validation_negative_scores"])
            quantile = np.searchsorted(negative, score, side="right") / len(negative)
            table[f"seed{seed}_score"], table[f"seed{seed}_quantile"] = score, quantile
            components.append(quantile)
        table["score"] = np.mean(components, axis=0)
        table.to_csv(target / f"{part}.csv", index=False)
        if part == "validation":
            validation_metrics = score_metrics(table.y.to_numpy(), table.score.to_numpy())
    # Keep the fitting-stage selection intact when later adding test scores.
    if validation_metrics is not None:
        frozen = dict(method=f"{method}-3seed-ECDF", seeds=SEEDS, weights=[1/3]*3,
                      transform="frozen per-seed validation-negative right-continuous ECDF", aggregation="arithmetic mean",
                      selection="all three prespecified seeds; no seed selection", validation_metrics=validation_metrics,
                      validation_negative_counts={str(seed):len(value["validation_negative_scores"]) for seed,value in selections.items()})
        (target / "selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["fit", "predict"], required=True)
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/human_rhythms")
    parser.add_argument("--output", type=Path, default=ROOT / "results/human_rhythms")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--all-methods-frozen", action="store_true", help="Explicit confirmation that all study methods are frozen before opening test histories.")
    args = parser.parse_args()
    features = args.features if args.features.is_absolute() else ROOT / args.features
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.phase == "predict":
        if not args.all_methods_frozen:
            parser.error("predict requires --all-methods-frozen; test files have not been opened")
        predict(features, output, args.methods, args.seeds)
        for method in args.methods:
            ensemble(output, method, ["test"])
        return
    for method in args.methods:
        for seed in args.seeds:
            fit_one(features, output, method, seed)
        ensemble(output, method, ["train", "validation"])


if __name__ == "__main__":
    main()
