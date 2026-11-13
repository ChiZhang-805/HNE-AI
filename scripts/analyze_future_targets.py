"""Measure learned future statistics on College validation people only."""
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_future import load_model
from scripts.run_temporal_external import configuration, read_partition

BASE = ROOT / "results/college_future_objective"
TARGETS = ROOT / "data/features/future_objective/College-Experience"
NAMES = ("signed_mean", "log1p_variance", "tail_fraction")


def statistics(values, threshold):
    observed = np.isfinite(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.stack((np.nanmean(values, axis=-1),
                           np.log1p(np.nanvar(values, axis=-1)),
                           ((np.abs(values) > threshold[..., None]) & observed).sum(-1)
                           / np.maximum(observed.sum(-1), 1)), axis=-1)
    result[observed.sum(-1) < 5] = np.nan
    return result


def decoder_predictions(data, condition):
    selection = json.loads((BASE / condition / "selection.json").read_text())
    outputs = []
    for seed in selection["seeds"]:
        folder = BASE / condition / f"lambda{selection['selected_lambda']}" / f"seed{seed}"
        model, _ = load_model(folder, device="cpu")
        rows = []
        with torch.inference_mode():
            for first in range(0, len(data["y"]), 256):
                inputs = [torch.as_tensor(data[key][first:first+256], dtype=torch.float32)
                          for key in ("windows", "nodes", "pooled")]
                model(*inputs)
                rows.append((model.auxiliary()*model.aux_scale+model.aux_mean).numpy())
        outputs.append(np.concatenate(rows))
    return np.mean(outputs, axis=0), model.aux_scale.numpy().copy()


def main():
    data = read_partition(ROOT / "data/features/college_experience", "validation", configuration())
    with np.load(TARGETS / "validation.npz") as saved:
        target = saved["moments"].astype(float)
    with np.load(TARGETS / "train.npz") as saved:
        constant = np.nanmedian(saved["moments"].reshape(-1, 3), axis=0)
    threshold = np.empty(target.shape[:2])
    for person, rows in data["meta"].groupby("person_id").groups.items():
        with np.load(ROOT / "data/college_study/validation" / f"{person}_daily.npz") as saved:
            reference = saved["normalized"][:56]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            threshold[rows] = np.maximum(np.nanquantile(np.abs(reference), .9, axis=0), .1)
    moment_prediction, scale = decoder_predictions(data, "future-moments")
    value_prediction, _ = decoder_predictions(data, "future-values")
    previous = data["windows"][:, -7:, :target.shape[1]].transpose(0, 2, 1)
    predictions = {
        "future-moments": moment_prediction,
        "future-values-statistics": statistics(value_prediction, threshold),
        "previous-seven-days": statistics(previous, threshold),
        "training-constant": np.broadcast_to(constant, target.shape)}
    common = np.isfinite(target).all(-1)
    for prediction in predictions.values():
        common &= np.isfinite(prediction).all(-1)
    origins = common.any(1)
    people = np.unique(data["people"][origins])
    rng = np.random.default_rng(20261110)
    draws = rng.multinomial(len(people), np.full(len(people), 1/len(people)), size=2000)
    rows, per_person, draw_losses = [], [], {}
    metric_names = ["standardized_huber"]+[name+"_mae" for name in NAMES]
    for method, prediction in predictions.items():
        absolute = np.abs(prediction-target)
        error = absolute/scale
        huber = np.where(error <= 1, .5*error**2, error-.5).mean(-1)
        coordinate_losses = np.concatenate((huber[..., None], absolute), axis=-1)
        origin_losses = np.where(common[..., None], coordinate_losses, 0).sum(1)/np.maximum(common.sum(1)[:, None], 1)
        values = []
        for person in people:
            mask = origins & (data["people"] == person)
            loss = origin_losses[mask].mean(0)
            values.append(loss)
            per_person.append({"method": method, "person_id": str(person), "origins": int(mask.sum()),
                               **dict(zip(metric_names, loss))})
        values = np.asarray(values)
        boot = draws @ values/len(people)
        draw_losses[method] = boot
        for j, name in enumerate(metric_names):
            low, high = np.quantile(boot[:, j], [.025, .975])
            rows.append({"method": method, "metric": name, "value": values[:, j].mean(),
                         "ci_low": low, "ci_high": high})
    comparisons = []
    for baseline in ("future-values-statistics", "previous-seven-days", "training-constant"):
        difference = draw_losses["future-moments"]-draw_losses[baseline]
        for j, name in enumerate(metric_names):
            low, high = np.quantile(difference[:, j], [.025, .975])
            point = next(row["value"] for row in rows if row["method"] == "future-moments" and row["metric"] == name)
            point -= next(row["value"] for row in rows if row["method"] == baseline and row["metric"] == name)
            comparisons.append({"model": "future-moments", "baseline": baseline,
                                "metric": name, "difference": point, "ci_low": low, "ci_high": high})
    output = BASE / "target_skill"
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(per_person).to_csv(output / "person_losses.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired_differences.csv", index=False)
    np.savez_compressed(output / "validation_predictions.npz", target=target, common=common,
                        threshold=threshold, **predictions)
    summary = {"partition": "validation", "people": len(people), "origins": int(origins.sum()),
               "common_nodes": int(common.sum()), "all_origins": len(target),
               "weighting": "valid nodes equally within origin; origins equally within person; people equally",
               "bootstrap": "2000 whole-person draws, seed20261110",
               "target_scale": scale.tolist(), "constant": constant.tolist(),
               "future_values": "statistics of the three-seed mean future trajectory",
               "future_moments": "three-seed mean decoder output; no post-hoc clipping",
               "past_baseline": "last7 observed input days ending d; at least5 days per node",
               "interpretation": "validation diagnostic after model selection, not independent confirmation"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
