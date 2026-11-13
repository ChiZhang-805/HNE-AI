"""Map College Experience to the unchanged protocol; test requires model freeze."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_daily_study import write_person, summarize_people
from src.hne_representation import Sampling, encode_person

DATASET = "College-Experience"
STATE_NAMES = ["sleep_duration", "still_duration", "vehicle_duration", "bike_duration",
               "foot_activity", "unlock_duration", "unlock_num", "distance", "visits"]
STATE_FAMILIES = ["device_rest", "activity", "activity", "activity", "activity",
                  "device_rest", "device_rest", "mobility", "mobility"]
CONTEXT_NAMES = ["home_duration", "study_duration", "social_duration", "max_campus_distance"]
QUALITY = ["quality_activity", "quality_audio", "quality_light", "quality_loc"]
SOURCE_FIELDS = ["is_ios", "sleep_duration", "act_still_ep_0", "act_in_vehicle_ep_0",
                 "act_on_bike_ep_0", "act_on_foot_ep_0", "act_walking_ep_0", "act_running_ep_0",
                 "unlock_duration_ep_0", "unlock_num_ep_0", "loc_dist_ep_0", "loc_visit_num_ep_0",
                 "loc_home_dur", "loc_study_dur", "loc_social_dur", "loc_max_dis_from_campus_ep_0",
                 *QUALITY]
SPLITS = ("train", "validation")


def worker_init():
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
    global THREAD_LIMIT
    THREAD_LIMIT = threadpool_limits(limits=1)


def read_allowed(source, assignment):
    """Filter identity before numeric conversion of the shared source CSV."""
    records = {r.person_id: [] for r in assignment.itertuples()}
    with (source / "Sensing/sensing.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            person = row["uid"]
            if person not in records:
                continue
            records[person].append([row["day"], *[float(row[c]) if row[c] else np.nan
                                                    for c in SOURCE_FIELDS]])
    return records


def map_daily(rows):
    frame = pd.DataFrame(rows, columns=["day", *SOURCE_FIELDS])
    frame.index = pd.to_datetime(frame.pop("day"), format="%Y%m%d")
    frame = frame.sort_index()
    dates = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    present = dates.isin(frame.index)
    frame = frame.reindex(dates)
    quality = frame[QUALITY].to_numpy(float)
    q_any = np.max(np.where(np.isfinite(quality), quality, -np.inf), axis=1)
    q_any[~np.isfinite(q_any)] = np.nan
    ios = frame.is_ios.to_numpy(float)
    foot = np.where(ios == 1, frame.act_walking_ep_0 + frame.act_running_ep_0,
                    np.where(ios == 0, frame.act_on_foot_ep_0, np.nan))
    states = np.column_stack([frame.sleep_duration, frame.act_still_ep_0,
        frame.act_in_vehicle_ep_0, frame.act_on_bike_ep_0, foot,
        frame.unlock_duration_ep_0, frame.unlock_num_ep_0, frame.loc_dist_ep_0,
        frame.loc_visit_num_ep_0]).astype(float)
    state_quality = np.column_stack([q_any, *[quality[:, 0]] * 4,
                                     q_any, q_any, quality[:, 3], quality[:, 3]])
    valid = np.isfinite(states) & (states >= 0) & (state_quality >= 4)
    valid[:, 0] &= states[:, 0] <= 24  # Zero is retained as a source-model estimate.
    states[~valid] = np.nan
    contexts = frame[["loc_home_dur", "loc_study_dur", "loc_social_dur",
                      "loc_max_dis_from_campus_ep_0"]].to_numpy(float, copy=True)
    context_valid = np.isfinite(contexts) & (contexts >= 0) & (quality[:, 3:4] >= 4)
    contexts[~context_valid] = np.nan
    calendar = dates.to_numpy(dtype="datetime64[D]")
    return dict(states=states, contexts=contexts, dates=calendar,
        available_at=calendar.astype("datetime64[s]") + np.timedelta64(1, "D"),
        state_names=np.asarray(STATE_NAMES), context_names=np.asarray(CONTEXT_NAMES),
        state_families=np.asarray(STATE_FAMILIES), state_valid=valid, context_valid=context_valid,
        state_observed_counts=valid.astype(np.int16),
        context_observed_counts=context_valid.astype(np.int16),
        reported_quality_hours=quality, quality_names=np.asarray(QUALITY),
        state_reported_quality_hours=state_quality, source_row_present=present,
        source_is_ios=ios, timezone_semantics=np.asarray(
            "source local calendar; timezone unspecified; assumed available next local midnight"))


def daily_worker(task):
    person, split, rows, output, config, digest = task
    arrays = map_daily(rows)
    return write_person(Path(output) / split, DATASET, person, split, 0, arrays, config, digest)


def feature_worker(task):
    person, folder, cutoffs, include_future = task
    with np.load(Path(folder) / f"{person}_daily.npz", allow_pickle=False) as archive:
        states = archive["normalized"]
        contexts = archive["normalized_contexts"]
    spec = Sampling.daily()
    nodes, pooled, groups = encode_person(states, contexts, cutoffs, spec)
    trajectory = np.column_stack((states, contexts)).astype(np.float32)
    length, channels = spec.scales[-1], trajectory.shape[1]
    windows = np.stack([trajectory[t-length:t] for t in cutoffs])
    payload = dict(nodes=nodes, pooled=pooled, groups=groups, windows=windows)
    if not include_future:
        return payload
    available = cutoffs + length <= len(trajectory)
    future = np.full((len(cutoffs), length, channels), np.nan, dtype=np.float32)
    for i, (cutoff, complete) in enumerate(zip(cutoffs, available)):
        if complete:
            future[i] = trajectory[cutoff:cutoff+length]
    payload.update(future=future, future_available=available)
    return payload


def update_coverage_summary(output, splits):
    """Describe already constructed, scorable people without rebuilding labels."""
    combined = {}
    for split in splits:
        folder = output / split
        people = pd.read_csv(folder / "people.csv", dtype={"person_id": str})
        measurements = []
        for person in people.loc[people.samples.gt(0), "person_id"]:
            with np.load(folder / f"{person}_daily.npz", allow_pickle=False) as data:
                states = np.isfinite(data["states"])
                contexts = np.isfinite(data["contexts"])
                dates = data["dates"]
                families = data["state_families"]
                family_valid = np.column_stack([states[:, families == name].any(1)
                                                for name in sorted(set(families))])
                row = dict(calendar_span_days=int((dates[-1] - dates[0]) / np.timedelta64(1, "D")) + 1,
                    valid_state_days_any=int(states.any(1).sum()),
                    valid_state_days_all=int(states.all(1).sum()),
                    valid_state_days_at_least_two_families=int((family_valid.sum(1) >= 2).sum()),
                    valid_context_days_any=int(contexts.any(1).sum()),
                    valid_context_days_all=int(contexts.all(1).sum()))
                row.update({"state__" + name: int(states[:, j].sum())
                            for j, name in enumerate(data["state_names"])})
                row.update({"context__" + name: int(contexts[:, j].sum())
                            for j, name in enumerate(data["context_names"])})
                measurements.append(row)
        values = pd.DataFrame(measurements)
        coverage = dict(people=len(values), population="people with at least one eligible forecast origin",
            calendar_definition="inclusive first-to-last stored source date, including missing dates",
            validity_definition="finite mapped value after fixed source-range and reported-quality>=4 rules, before reference normalization; retained zero counts; not complete continuous sensor days",
            statistics={name: {"median": float(values[name].median()), "min": int(values[name].min()),
                               "max": int(values[name].max())} for name in values})
        summary = json.loads((folder / "summary.json").read_text())
        summary["scorable_coverage"] = coverage
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        combined[split] = summary
        print("COVERAGE", split, json.dumps({k: v for k, v in coverage["statistics"].items()
                                             if not k.startswith(("state__", "context__"))}), flush=True)
    combined_path = output / "summary.json"
    previous = json.loads(combined_path.read_text()) if combined_path.exists() else {}
    previous.update(combined)
    combined_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")


def make_schema(config, digest):
    return dict(dataset=DATASET, source="College Experience Kaggle v5 Sensing/sensing.csv",
        source_url="https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset",
        state_names=STATE_NAMES, state_families=STATE_FAMILIES, context_names=CONTEXT_NAMES,
        state_units=["hours", "seconds", "seconds", "seconds", "seconds", "seconds",
                     "count", "metres", "count"],
        context_units=["hours", "hours", "hours", "metres"],
        state_source_fields=["sleep_duration", "act_still_ep_0", "act_in_vehicle_ep_0",
            "act_on_bike_ep_0", "is_ios==1:walking+running;is_ios==0:on_foot",
            "unlock_duration_ep_0", "unlock_num_ep_0", "loc_dist_ep_0", "loc_visit_num_ep_0"],
        context_source_fields=["loc_home_dur", "loc_study_dur", "loc_social_dur",
                               "loc_max_dis_from_campus_ep_0"],
        validity=dict(quality_units="reported hours; not necessarily continuous complete hours",
            quality_names=QUALITY, threshold_hours=4,
            device_rest="max of four finite quality fields >=4; sleep finite in[0,24];phone finite>=0",
            activity="quality_activity>=4; finite nonnegative source; both iOS foot components required",
            mobility_and_context="quality_loc>=4; finite nonnegative source",
            sleep_zero="retained source model estimate; not asserted to be actual sleeplessness",
            zero="retained when required source and quality are valid; never used to fill missingness",
            counts="state/context_observed_counts are binary valid-day indicators, not hourly counts"),
        calendar="First to last sensing.csv source date for each person, inclusive; missing dates remain NaN; steps.csv not part of selected mapping",
        availability="Every date is assumed available at next local midnight; no row-level release timestamp or timezone is supplied by source",
        timezone="Unspecified source local calendar retained; no UTC offset invented; availability is a study convention",
        family_note="sleep and phone grouped device_rest because sleep is inferred from phone/activity/environment; activity and mobility are other two families",
        missing="No interpolation, extrapolation, forward fill, duplicated mean/SD state or full-trajectory scaling",
        reference="Unchanged first56 calendar days via scripts.build_daily_study.write_person",
        future32="True only when32 future calendar days remain within stored source follow-up; missing values remain NaN; false rows carry NaN placeholders and must not enter FCM training",
        test_feature_policy="Only after model freeze; past windows/nodes/pooled plus independent evaluation y; no future or future_available arrays",
        supported_splits=["train", "validation", "test"], default_splits=list(SPLITS),
        construction_policy="Train/validation by default; test only after all models are frozen; actual construction status belongs to each split summary, not this fixed mapping",
        frozen_id_split="132train/44validation/44test; original220-ID assignment unchanged",
        event=config, protocol_sha256=digest,
        representation="src.hne_representation.encode_person with Sampling.daily; all9 states and4 contexts")


def main(args):
    if "test" in args.splits and not args.all_models_frozen:
        raise ValueError("Construct test only after all models are frozen; acknowledge with --all-models-frozen.")
    if args.phase == "summary":
        update_coverage_summary(args.output, args.splits)
        return
    config_bytes = (ROOT / "configs/daily_protocol.json").read_bytes()
    config = json.loads(config_bytes)
    digest = hashlib.sha256(config_bytes).hexdigest()
    assignment = pd.read_csv(args.source / "subject_split.csv", dtype={"person_id": str})
    assignment = assignment[assignment.split.isin(args.splits)]
    args.output.mkdir(parents=True, exist_ok=True)
    args.features.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        (args.output / split).mkdir(exist_ok=True)
    schema = make_schema(config, digest)
    if any(split != "test" for split in args.splits):
        (args.output / "schema_mapping.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        (args.features / "protocol.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    if "test" in args.splits:
        (args.output / "test/schema_mapping.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        (args.features / "test_protocol.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    if args.phase in ("all", "daily"):
        records = read_allowed(args.source, assignment)
        tasks = [(r.person_id, r.split, records[r.person_id], args.output, config, digest)
                 for r in assignment.itertuples()]
        aggregate = {s: {"origins": [], "samples": [], "people": []} for s in args.splits}
        with ProcessPoolExecutor(args.workers, initializer=worker_init) as pool:
            for task, result in zip(tasks, pool.map(daily_worker, tasks)):
                origins, samples, person = result
                dest = aggregate[task[1]]
                dest["origins"].extend(origins)
                dest["samples"].extend(samples)
                dest["people"].append(person)
                print("DAILY", task[1], task[0], "origins", len(samples), flush=True)
        for split, payload in aggregate.items():
            folder = args.output / split
            for name, rows in payload.items():
                pd.DataFrame(rows).to_csv(folder / f"{name}.csv", index=False)
            summary = summarize_people(pd.DataFrame(payload["people"]))
            summary.update(split=split, state_dimension=len(STATE_NAMES), context_dimension=len(CONTEXT_NAMES),
                           families=sorted(set(STATE_FAMILIES)), protocol_sha256=digest, test_created=split == "test")
            (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print("DAILY_COMPLETE", json.dumps(summary), flush=True)

    if args.phase in ("all", "features"):
        for split in args.splits:
            folder = args.output / split
            meta = pd.read_csv(folder / "samples.csv", dtype={"person_id": str})
            tasks = [(person, folder, part.origin_day.to_numpy(dtype=int) + 1, split != "test")
                     for person, part in meta.groupby("person_id", sort=False)]
            blocks = []
            with ProcessPoolExecutor(args.workers, initializer=worker_init) as pool:
                for task, result in zip(tasks, pool.map(feature_worker, tasks)):
                    blocks.append(result)
                    print("FEATURES", split, task[0], len(task[2]), flush=True)
            feature_keys = ["nodes", "pooled", "windows"]
            if split != "test":
                feature_keys.extend(["future", "future_available"])
            payload = {key: np.concatenate([b[key] for b in blocks]) for key in feature_keys}
            payload.update(groups=blocks[0]["groups"], y=meta.y.to_numpy(), fold=meta.fold.to_numpy(),
                           state_names=np.asarray(STATE_NAMES), context_names=np.asarray(CONTEXT_NAMES),
                           state_families=np.asarray(STATE_FAMILIES))
            if split != "test":
                meta["future32_available"] = payload["future_available"]
            np.savez_compressed(args.features / f"{split}.npz", **payload)
            meta.to_csv(args.features / f"{split}.csv", index=False)
            summary = json.loads((folder / "summary.json").read_text())
            summary.update(windows=list(payload["windows"].shape), nodes=list(payload["nodes"].shape),
                           pooled=list(payload["pooled"].shape))
            if split != "test":
                summary["complete32day_future"] = int(payload["future_available"].sum())
            (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print("FEATURES_COMPLETE", json.dumps(summary), flush=True)
    update_coverage_summary(args.output, args.splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/new_cohorts/college-experience")
    parser.add_argument("--output", type=Path, default=ROOT / "data/college_study")
    parser.add_argument("--features", type=Path, default=ROOT / "data/features/college_experience")
    parser.add_argument("--phase", choices=["all", "daily", "features", "summary"], default="all")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=list(SPLITS))
    parser.add_argument("--all-models-frozen", action="store_true",
                        help="Acknowledge that every method is frozen before constructing test data.")
    parser.add_argument("--workers", type=int, default=3)
    main(parser.parse_args())
