"""Download official daily records and inspect coverage without outcomes or scoring."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import date, datetime
import hashlib
import io
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "HAI-lab-UVA/Human-Rhythms-Dataset"
SEED = 20261110
SOURCES = {
    "heart": ("Cleaned_Raw_Data/Fitbit/Heart_Rate_Summary", "local_date_time", "heartrate_daily_restinghr"),
    "sleep": ("Cleaned_Raw_Data/Fitbit/Sleep_Summary", "local_date_time", "minutes_asleep"),
    "activity": ("Processed_Data/Behavioral/Fitbit/Steps_Summary", "local_segment_start_datetime", "fitbit_steps_summary_rapids_avgsumsteps"),
}


def get_bytes(url, limit=5 * 1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": "HNE-Research"})
    with urllib.request.urlopen(req, timeout=45) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"Not a small daily file: {url}")
    return content


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def download(root, sha, path):
    target = root / path
    if target.exists() and not target.read_bytes().startswith(b"version https://git-lfs"):
        content = target.read_bytes()
    else:
        last = None
        for attempt in range(3):
            for base in (f"https://media.githubusercontent.com/media/{REPOSITORY}/{sha}/",
                         f"https://raw.githubusercontent.com/{REPOSITORY}/{sha}/"):
                try:
                    content = get_bytes(base + path)
                    if content.startswith(b"version https://git-lfs"):
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    return dict(path=path, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    last = str(exc)
            time.sleep(attempt + 1)
        raise RuntimeError(f"{path}: {last}")
    return dict(path=path, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def longest_run(days):
    best = run = 0
    previous = None
    for day in sorted(days):
        run = run + 1 if previous is not None and day == previous + 1 else 1
        best = max(best, run)
        previous = day
    return best


def inspect(root, people):
    coverage, schema = [], {}
    for person in people:
        row = dict(person_id=person)
        date_sets = []
        for family, (folder, clock, field) in SOURCES.items():
            path = root / folder / (person + ".csv")
            rows = []
            if path.exists():
                with path.open(encoding="utf-8-sig", newline="") as file:
                    reader = csv.DictReader(file)
                    schema.setdefault(family, dict(folder=folder, fields=reader.fieldnames, clock=clock,
                        coverage_field=field, filtering="daily segment" if family == "activity" else
                        "main sleep only" if family == "sleep" else "all daily records"))
                    rows = list(reader)
            if family == "activity":
                rows = [r for r in rows if r.get("local_segment_label") == "daily"]
            if family == "sleep":
                rows = [r for r in rows if r.get("is_main_sleep", "").lower() in {"true", "1"}]
            calendar = {date.fromisoformat(r[clock][:10]).toordinal() for r in rows if r.get(clock)}
            observed = {date.fromisoformat(r[clock][:10]).toordinal() for r in rows
                        if r.get(clock) and finite(r.get(field))}
            row.update({family + "_calendar_days": len(calendar), family + "_observed_days": len(observed),
                        family + "_span_days": max(observed) - min(observed) + 1 if observed else 0,
                        family + "_longest_complete_run": longest_run(observed)})
            date_sets.append(observed)
        joint = set.intersection(*date_sets)
        union = set.union(*date_sets)
        row.update(joint_observed_days=len(joint), joint_span_days=max(joint)-min(joint)+1 if joint else 0,
                   joint_longest_complete_run=longest_run(joint),
                   any_state_span_days=max(union)-min(union)+1 if union else 0,
                   first_date=date.fromordinal(min(union)).isoformat() if union else "",
                   last_date=date.fromordinal(max(union)).isoformat() if union else "")
        coverage.append(row)
    write_csv(root / "participant_coverage.csv", coverage)
    write_json(root / "schema.json", schema)
    def median(values):
        values = sorted(values)
        n = len(values)
        return (values[(n-1)//2]+values[n//2])/2 if n else None
    summary = {"participants": len(people), "dates_only_and_finite_coverage": True,
               "future_labels_created": False, "model_scores_computed": False,
               "joint_definition": "finite resting HR, main-sleep minutes, and official daily mean steps on same date",
               "activity_caution": "Official daily summary includes padded calendar rows; only finite values count, but zero is retained as an observed value."}
    for column in ["any_state_span_days", "joint_span_days", "joint_observed_days", "joint_longest_complete_run"]:
        vals = [r[column] for r in coverage]
        summary[column] = dict(min=min(vals), median=median(vals), max=max(vals),
                               people_ge90=sum(v >= 90 for v in vals), people_ge96=sum(v >= 96 for v in vals))
    write_json(root / "coverage_summary.json", summary)
    print(json.dumps(summary), flush=True)


def source_day(value):
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return datetime.strptime(value, "%m/%d/%y").date()


def build_daily(root, splits, include_test=False):
    """Map the fixed native daily schema, without reference fitting or labels."""
    import numpy as np

    state_names = ["resting_heart_rate", "sleep_efficiency", "sleep_minutes_asleep",
        "sleep_minutes_awake", "sleep_minutes_in_bed", "sleep_minutes_to_fall_asleep",
        "sleep_minutes_after_wakeup", "daily_steps", "sleep_midpoint_sin", "sleep_midpoint_cos"]
    families = ["heart"] + ["sleep"] * 6 + ["activity", "sleep", "sleep"]
    context_names = ["temperature_2m_mean", "relative_humidity_2m_mean", "precipitation_sum",
                     "wind_speed_10m_mean", "daylight_duration"]
    units = ["beats/minute", "percent", "minutes", "minutes", "minutes", "minutes", "minutes",
             "steps", "unitless", "unitless"]
    output = root / "daily"
    output.mkdir(exist_ok=True)
    with (root / "Additional_Data/Weather/Daily_Weather.csv").open(encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        source_context = []
        for field in context_names:
            found = [c for c in reader.fieldnames if c.startswith(field + " (")]
            if len(found) != 1:
                raise ValueError(f"Weather schema changed: {field}")
            source_context.append(found[0])
        weather = {}
        for row in reader:
            d = source_day(row["time"])
            if d in weather:
                raise ValueError("Duplicate weather day")
            weather[d] = [float(row[c]) if finite(row[c]) else float("nan") for c in source_context]
    schema = dict(version="human-rhythms-native-daily-v1-20261110", state_names=state_names,
        state_families=families, state_units=units, context_names=context_names,
        source_weather_columns=source_context, context_units=["degF", "percent", "inch", "mph", "seconds"],
        state_mapping={state_names[0]: "Heart_Rate_Summary.heartrate_daily_restinghr",
            **{name: "Sleep_Summary." + field for name,field in zip(state_names[1:7],
                ["efficiency", "minutes_asleep", "minutes_awake", "minutes_in_bed", "minutes_to_fall_asleep", "minutes_after_wakeup"])},
            state_names[7]: "Steps_Summary.fitbit_steps_summary_rapids_avgsumsteps; local_segment_label=daily",
            state_names[8]: "sin(2*pi*local_clock_midpoint_hour/24)",
            state_names[9]: "cos(2*pi*local_clock_midpoint_hour/24)"},
        sleep="main sleep only; midpoint of actual local start/end datetimes; assigned to end date",
        activity="one official daily window, same-date 00:00:00 through 23:59:59; preserve native zero",
        calendar="first through last date with any finite selected state; no padding from weather or empty activity rows",
        availability="next local calendar midnight for every daily state and context",
        timezone_semantics="source local clock; no UTC conversion or invented timezone offset",
        context_semantics="shared study-location historical weather supplied by authors; not individual exposure",
        missing="NaN retained; no forward fill, interpolation, hourly expansion, or extrapolation",
        labels_created=False, reference_fitted=False, models_run=False,
        schema_chosen_before_reading_future_labels=True)
    write_json(output / "schema_mapping.json", schema)
    records = []
    for item in splits:
        if item["split"] == "test" and not include_test:
            continue
        person = item["person_id"]
        values = {}
        def put(d, indices, fields):
            vector = values.setdefault(d, np.full(10, np.nan, dtype=np.float64))
            vector[indices] = fields
        for family, (folder, clock, _) in SOURCES.items():
            path = root / folder / (person + ".csv")
            if not path.exists():
                continue
            seen = set()
            with path.open(encoding="utf-8-sig") as file:
                for row in csv.DictReader(file):
                    if family == "sleep" and row["is_main_sleep"].lower() not in {"true", "1"}:
                        continue
                    if family == "activity" and row["local_segment_label"] != "daily":
                        continue
                    d = source_day(row[clock])
                    if d in seen:
                        raise ValueError(f"Duplicate selected daily record: {person}/{family}/{d}")
                    seen.add(d)
                    if family == "heart":
                        value = row["heartrate_daily_restinghr"]
                        put(d, [0], [float(value) if finite(value) else np.nan])
                    elif family == "sleep":
                        columns = ["efficiency", "minutes_asleep", "minutes_awake", "minutes_in_bed",
                                   "minutes_to_fall_asleep", "minutes_after_wakeup"]
                        put(d, list(range(1,7)), [float(row[c]) if finite(row[c]) else np.nan for c in columns])
                        start = datetime.fromisoformat(row["local_start_date_time"])
                        end = datetime.fromisoformat(row["local_end_date_time"])
                        if end.date() != d or not (0 < (end-start).total_seconds() <= 86400):
                            raise ValueError(f"Invalid main-sleep interval: {person}/{d}")
                        midpoint = start + (end-start)/2
                        hour = midpoint.hour + midpoint.minute/60 + midpoint.second/3600 + midpoint.microsecond/3.6e9
                        angle = 2*np.pi*hour/24
                        put(d, [8,9], [np.sin(angle),np.cos(angle)])
                    else:
                        start, end = row["local_segment_start_datetime"], row["local_segment_end_datetime"]
                        if start[:10] != end[:10] or not start.endswith("00:00:00") or not end.endswith("23:59:59"):
                            raise ValueError(f"Non-daily activity window: {person}/{d}")
                        value = row["fitbit_steps_summary_rapids_avgsumsteps"]
                        put(d, [7], [float(value) if finite(value) else np.nan])
        observed_days = [d for d,v in values.items() if np.isfinite(v).any()]
        if not observed_days:
            raise ValueError(f"No observed state in {person}")
        first,last = min(observed_days),max(observed_days)
        dates = np.arange(np.datetime64(first), np.datetime64(last)+np.timedelta64(1,"D"), dtype="datetime64[D]")
        days = [date.fromisoformat(str(d)) for d in dates]
        states = np.array([values.get(d,np.full(10,np.nan)) for d in days],dtype=np.float32)
        contexts = np.array([weather.get(d,[np.nan]*5) for d in days],dtype=np.float32)
        available = dates.astype("datetime64[s]")+np.timedelta64(1,"D")
        if states.shape != (len(dates),10) or contexts.shape != (len(dates),5):
            raise ValueError("Daily dimension mismatch")
        np.savez_compressed(output / f"{person}.npz", states=states, contexts=contexts, dates=dates,
            available_at=available, state_names=np.asarray(state_names), context_names=np.asarray(context_names),
            state_families=np.asarray(families), split=np.asarray(item["split"]), person_id=np.asarray(person),
            dataset=np.asarray("Human-Rhythms"), timezone_semantics=np.asarray(schema["timezone_semantics"]),
            state_observed_counts=np.isfinite(states).astype(np.int16),
            context_observed_counts=np.isfinite(contexts).astype(np.int16))
        records.append(dict(person_id=person, split=item["split"], days=len(dates), first_date=str(first),
            last_date=str(last), days_with_state=int(np.isfinite(states).any(1).sum()),
            state_observed_fraction=float(np.isfinite(states).mean()),
            context_observed_fraction=float(np.isfinite(contexts).mean())))
    write_csv(output / "participants.csv", records)
    summary = dict(people=len(records), by_split={s:sum(r['split']==s for r in records) for s in ["train","validation","test"]},
        states=10, contexts=5, total_calendar_days=sum(r['days'] for r in records),
        include_test=include_test, labels_created=False, reference_fitted=False, models_run=False,
        source_sha256=hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest())
    write_json(output / "summary.json", summary)
    print(json.dumps(summary),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/new_cohorts/human-rhythms")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--build-daily", action="store_true", help="Map the agreed native daily fields without labels")
    parser.add_argument("--include-test", action="store_true", help="Explicitly include test IDs in daily conversion")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    tree_path = root / "repository_tree.json"
    if not tree_path.exists():
        tree_path.write_bytes(get_bytes(f"https://api.github.com/repos/{REPOSITORY}/git/trees/main?recursive=1"))
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    sha = tree["sha"]
    entries = [r for r in tree["tree"] if r["type"] == "blob"]
    people = sorted({Path(r["path"]).stem for r in entries if r["path"].startswith("Cleaned_Raw_Data/Fitbit/Steps/")})
    ranked = sorted(people, key=lambda p: hashlib.sha256(f"{SEED}:Human-Rhythms:{p}".encode()).hexdigest())
    n_train, n_validation = round(.6*len(ranked)), round(.2*len(ranked))
    splits = [dict(person_id=p, split="train" if rank<n_train else "validation" if rank<n_train+n_validation else "test",
                   hash_rank=rank, seed=SEED) for rank,p in enumerate(ranked)]
    split_path = root / "subject_split.csv"
    if split_path.exists():
        old = list(csv.DictReader(split_path.open(encoding="utf-8")))
        if [(r['person_id'],r['split']) for r in old] != [(r['person_id'],r['split']) for r in splits]:
            raise ValueError("Existing split differs")
    else:
        write_csv(split_path, splits)
    write_json(root / "split_definition.json", dict(seed=SEED, participants=len(people), train=n_train,
        validation=n_validation, test=len(people)-n_train-n_validation, selection="ID hash only; no outcome or coverage filtering",
        hash="sha256('20261110:Human-Rhythms:'+person_id); ascending; rounded 60%/20%, remainder test"))
    if args.build_daily:
        build_daily(root, splits, args.include_test)
        return
    if not args.inspect_only:
        paths = [r["path"] for r in entries if any(r["path"].startswith(v[0]+"/") for v in SOURCES.values())]
        paths += ["README.md", "Additional_Data/Weather/Daily_Weather.csv",
                  "Additional_Data/Weather/Weather_Descriptions.json", "Additional_Data/University_Events.csv"]
        downloaded, errors = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download,root,sha,p):p for p in paths}
            for i,future in enumerate(concurrent.futures.as_completed(futures),1):
                try: downloaded.append(future.result())
                except Exception as exc: errors.append(dict(path=futures[future],error=str(exc)))
                if i%50 == 0:print(f"Downloaded/checked {i}/{len(paths)}, failures {len(errors)}",flush=True)
        write_json(root / "source_manifest.json", dict(repository=f"https://github.com/{REPOSITORY}", commit=sha,
            paper="https://doi.org/10.1145/3770698", access="Public academic use as stated by repository README",
            files=sorted(downloaded,key=lambda r:r['path']), errors=errors, total_bytes=sum(r['bytes'] for r in downloaded),
            downloaded_content="official daily HR/sleep and official processed activity; no hourly construction",
            future_labels_created=False, model_scores_computed=False))
        if errors:raise RuntimeError(f"{len(errors)} download failures; retained completed files")
    inspect(root, people)


if __name__ == "__main__":
    main()
