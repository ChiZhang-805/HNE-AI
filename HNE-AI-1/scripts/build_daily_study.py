"""Adapt three cohorts and execute the one fixed daily future-block protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.daily_protocol import fit_reference, label_blocks, forecast_origins, robust_reference, normalize

CHANNELS = {
    "RESILIENT": ["sleep_mat_hr", "respiration", "sdnn", "watch_hr", "sleep_prop", "steps"],
    "BrisT1D-Open": ["bg", "hr", "steps", "dist", "cals", "carbs"],
}
FAMILIES = {
    "RESILIENT": ["heart", "respiration", "hrv", "heart", "sleep"],
    "BrisT1D-Open": ["glucose", "heart", "activity", "activity", "activity"],
}


def hourly_daily(path: Path, dataset: str, config: dict):
    with np.load(path, allow_pickle=False) as archive:
        values = archive["values"].astype(np.float64)
        clock = str(archive["clock"])
    if values.ndim != 2 or values.shape[1] != len(CHANNELS[dataset]):
        raise ValueError(f"Unexpected hourly schema: {path}")
    day_count = (len(values) + 23) // 24
    state = np.full((day_count, 10), np.nan)
    context = np.full((day_count, 1), np.nan)
    scount = np.zeros_like(state, dtype=np.int16)
    ccount = np.zeros_like(context, dtype=np.int16)
    for d in range(day_count):
        block = values[d * 24:(d + 1) * 24]
        for j in range(5):
            x = block[:, j]
            x = x[np.isfinite(x)]
            scount[d, 2 * j:2 * j + 2] = len(x)
            if len(x) >= config["hourly_minimum_state_observations_per_day"]:
                state[d, 2 * j] = np.mean(x)
                state[d, 2 * j + 1] = np.std(x, ddof=0)
        x = block[:, 5]
        x = x[np.isfinite(x)]
        ccount[d, 0] = len(x)
        if len(x) >= config["hourly_minimum_context_observations_per_day"]:
            context[d, 0] = np.mean(x)
    start = pd.Timestamp(clock)
    # BrisT1D source times are timezone-naive; retain its source clock rather than invent UTC.
    timezone_semantics = "UTC" if start.tzinfo is not None else "source_clock_timezone_unspecified"
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    dates = pd.date_range(start.normalize(), periods=day_count, freq="D").to_numpy(dtype="datetime64[D]")
    available = dates.astype("datetime64[s]") + np.timedelta64(1, "D")
    return {"states": state, "contexts": context, "state_observed_counts": scount,
            "context_observed_counts": ccount, "dates": dates, "available_at": available,
            "state_names": np.asarray([name + "__" + stat for name in CHANNELS[dataset][:-1] for stat in ["mean", "sd"]]),
            "context_names": np.asarray([CHANNELS[dataset][-1] + "__mean"]),
            "state_families": np.asarray([family for family in FAMILIES[dataset] for _ in range(2)]),
            "timezone_semantics": np.asarray(timezone_semantics)}


def tolife_daily(path: Path, mapping: dict, allowed_splits=("train", "validation")):
    frame = pd.read_csv(path)
    allowed = set(frame["split"])
    if not allowed <= set(allowed_splits):
        raise ValueError("TOLIFE daily file contains a different partition")
    state_cols, context_cols = mapping["state_columns"], mapping["context_columns"]
    families = []
    family_map = {"shri": "heart", "prvi": "heart", "bri": "respiration", "sqi": "sleep", "pei": "pulmonary"}
    for col in state_cols:
        families.append(family_map[mapping["features"][col]["period"]])
    states = frame[state_cols].to_numpy(np.float64)
    contexts = frame[context_cols].to_numpy(np.float64)
    dates = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[D]")
    available = pd.to_datetime(frame["available_at_utc"], utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    if len(dates) > 1 and not np.all(np.diff(dates) == np.timedelta64(1, "D")):
        raise ValueError(f"Daily grid is incomplete: {path}")
    return {"states": states, "contexts": contexts,
            "state_observed_counts": np.isfinite(states).astype(np.int16),
            "context_observed_counts": np.isfinite(contexts).astype(np.int16),
            "dates": dates, "available_at": available, "state_names": np.asarray(state_cols),
            "context_names": np.asarray(context_cols), "state_families": np.asarray(families),
            "timezone_semantics": np.asarray("UTC aggregation availability; site-local source alignment in schema_mapping")}


def write_person(folder: Path, dataset: str, person: str, split: str, fold: int,
                 arrays: dict, config: dict, protocol_sha: str):
    ref, normalized = fit_reference(arrays["states"], arrays["state_families"], config)
    blocks = label_blocks(normalized, ref, config)
    origins = forecast_origins(blocks, arrays["available_at"], arrays["dates"], config)
    context_center, context_scale, context_support = robust_reference(
        arrays["contexts"], config["reference_days"], config["minimum_reference_observations"], config["mad_multiplier"])
    payload = dict(arrays)
    payload.update({"normalized": normalized.astype(np.float32),
                    "normalized_contexts": normalize(arrays["contexts"], context_center, context_scale).astype(np.float32),
                    "reference_length": np.asarray(min(len(normalized), config["reference_days"])),
                    "protocol_reference_days": np.asarray(config["reference_days"]),
                    "reference_location": ref.location, "reference_scale": ref.scale,
                    "reference_observation_count": ref.observation_count,
                    "reference_block_count": ref.reference_block_count, "reference_eligible": ref.eligible,
                    "reference_amplitude_threshold": ref.amplitude_threshold,
                    "reference_bias_threshold": ref.bias_threshold,
                    "reference_absolute_point_threshold": ref.absolute_point_threshold,
                    "family_names": ref.family_names, "reference_eligible_families": ref.eligible_families,
                    "context_reference_location": context_center, "context_reference_scale": context_scale,
                    "context_reference_observation_count": context_support,
                    "person_id": np.asarray(person), "dataset": np.asarray(dataset),
                    "split": np.asarray(split), "fold": np.asarray(fold), "protocol_sha256": np.asarray(protocol_sha),
                    "future_labels": origins["future_labels"], "eligible_origin": origins["eligible_origin"],
                    "issue_time": origins["issue_time"]})
    payload.update({"block_" + name: value for name, value in blocks.items()})
    np.savez_compressed(folder / f"{person}_daily.npz", **payload)
    origin_rows = []
    for row in origins["origin_rows"]:
        d = row["origin_day"]
        origin_rows.append({"dataset": dataset, "person_id": person, "split": split, "fold": fold,
                            **row, "origin_date": str(arrays["dates"][d]),
                            "future_start_date": str(arrays["dates"][row["future_start_day"]]),
                            "future_end_date": str(arrays["dates"][row["future_end_day"]]),
                            "reference_eligible_variables": int(ref.eligible.sum()),
                            "reference_eligible_families": int(ref.eligible_families.sum())})
    accepted = [r for r in origin_rows if r["eligible_sample"]]
    person_summary = {"dataset": dataset, "person_id": person, "split": split, "fold": fold,
                      "days": len(normalized), "reference_length": min(len(normalized), config["reference_days"]),
                      "eligible_variables": int(ref.eligible.sum()), "eligible_families": int(ref.eligible_families.sum()),
                      "candidate_origins": len(origin_rows), "samples": len(accepted),
                      "positive": int(sum(r["y"] == 1 for r in accepted)),
                      "past_unknown": int(sum(not np.isfinite(r["past_block_label"]) for r in origin_rows)),
                      "past_abnormal": int(sum(r["past_block_label"] == 1 for r in origin_rows)),
                      "future_unknown": int(sum(not np.isfinite(r["y"]) for r in origin_rows))}
    return origin_rows, accepted, person_summary


def summarize_people(frame: pd.DataFrame):
    return {"people": len(frame), "people_with_samples": int((frame.samples > 0).sum()),
            "samples": int(frame.samples.sum()), "positive": int(frame.positive.sum()),
            "negative": int(frame.samples.sum() - frame.positive.sum()),
            "total_days": int(frame.days.sum()), "candidate_origins": int(frame.candidate_origins.sum())}


def build(root: Path, config_path: Path, output: Path, selected: list[str]):
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    protocol_sha = hashlib.sha256(config_bytes).hexdigest()
    summaries = {}
    for dataset in selected:
        folder = output / dataset
        folder.mkdir(parents=True, exist_ok=True)
        tasks = []
        if dataset in CHANNELS:
            source = root / "data/development" / dataset
            paths = sorted(source.glob("*_hourly.npz"))
            people = [p.name.removesuffix("_hourly.npz") for p in paths]
            ordered = sorted(people, key=lambda p: hashlib.sha256(
                f"{config['hourly_person_fold_seed']}:{dataset}:{p}".encode()).hexdigest())
            folds = {p: i % config["hourly_person_fold_count"] for i, p in enumerate(ordered)}
            # Verify ID/fold metadata against prior protocol without reading its labels.
            if (source / "samples.csv").exists():
                prior = pd.read_csv(source / "samples.csv", usecols=["person_id", "fold"]).drop_duplicates()
                if any(folds[r.person_id] != r.fold for r in prior.itertuples()):
                    raise ValueError(f"Original person fold mismatch in {dataset}")
            tasks = [(p, p.name.removesuffix("_hourly.npz"), "development", folds[p.name.removesuffix("_hourly.npz")]) for p in paths]
            mapping = None
        elif dataset == "TOLIFE-CSA":
            source = root / "data/new_cohorts/tolife-csa-2026"
            split = pd.read_csv(source / "subject_split.csv", usecols=["Kit-ID", "split"])
            # Discard held-out identities before opening any daily values.
            split = split[split["split"].isin(config["tolife_splits_allowed"])]
            tasks = [(source / "daily" / (str(row["Kit-ID"]) + ".csv"), str(row["Kit-ID"]), str(row["split"]), -1)
                     for _, row in split.iterrows()]
            mapping = json.loads((source / "daily/schema_mapping.json").read_text())
        else:
            raise ValueError(dataset)
        all_origins, all_samples, all_people = [], [], []
        for path, person, split, fold in tasks:
            arrays = hourly_daily(path, dataset, config) if mapping is None else tolife_daily(path, mapping)
            origins, samples, summary = write_person(folder, dataset, person, split, fold, arrays, config, protocol_sha)
            all_origins.extend(origins)
            all_samples.extend(samples)
            all_people.append(summary)
        people_frame = pd.DataFrame(all_people)
        people_frame.to_csv(folder / "people.csv", index=False)
        origin_frame = pd.DataFrame(all_origins)
        origin_frame.to_csv(folder / "origins.csv", index=False)
        sample_frame = pd.DataFrame(all_samples, columns=origin_frame.columns)
        sample_frame.to_csv(folder / "samples.csv", index=False)
        summary = summarize_people(people_frame)
        summary["splits"] = {s: summarize_people(part) for s, part in people_frame.groupby("split")}
        if dataset in CHANNELS:
            summary["development_cv"] = {
                str(f): {"train": summarize_people(people_frame[~people_frame.fold.isin([f, (f + 1) % 5])]),
                         "validation": summarize_people(people_frame[people_frame.fold == (f + 1) % 5]),
                         "exposed_evaluation": summarize_people(people_frame[people_frame.fold == f])}
                for f in range(5)}
        else:
            summary["heldout_test_people_read"] = 0
        summary["state_features"] = len(arrays["state_names"])
        summary["context_features"] = len(arrays["context_names"])
        summary["protocol_sha256"] = protocol_sha
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries[dataset] = summary
        print(dataset, json.dumps({k: v for k, v in summary.items() if k not in {"development_cv", "protocol_sha256"}}), flush=True)
    (output / "summary.json").write_text(json.dumps({"protocol": config, "protocol_sha256": protocol_sha,
                                                   "datasets": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/daily_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/daily_study")
    parser.add_argument("--datasets", nargs="+", default=["RESILIENT", "BrisT1D-Open", "TOLIFE-CSA"])
    args = parser.parse_args()
    build(args.root, args.config, args.output, args.datasets)
