"""Build the six hourly observations directly from either public source archive.

Requires only NumPy/Pandas, with no import from the previous research project.
Array row k covers (clock+k hours, clock+(k+1) hours]. Existing outputs are
recomputed and compared exactly; they are never overwritten. No labels, folds,
personal reference transforms, or models are constructed here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {
    "RESILIENT": ["sleep_mat_hr", "respiration", "sdnn", "watch_hr", "sleep_prop", "steps"],
    "BrisT1D-Open": ["bg", "hr", "steps", "dist", "cals", "carbs"],
}
RES_FILES = ("Sleep_physio.csv", "ScanWatch_HR.csv", "ScanWatch_Steps.csv", "Sleep_state.csv")


@contextmanager
def source_tables(dataset: str, source: Path):
    """Yield all raw participant IDs and a CSV reader, without extracting ZIPs."""
    if source.is_file():
        with ZipFile(source) as archive:
            entries = {}
            for member in archive.namelist():
                p = PurePosixPath(member)
                if any(s.startswith(".") or s == "__MACOSX" for s in p.parts):
                    continue
                if dataset == "RESILIENT":
                    if len(p.parts) >= 3 and p.parts[-3] == "Sleepmat_Watch_Data" and p.name in RES_FILES:
                        entries[(p.parent.name, p.name)] = member
                elif p.parent.name == "processed_state" and p.suffix == ".csv" and p.stem.startswith("P"):
                    entries[(p.stem, "observations.csv")] = member

            def read(person, filename):
                with archive.open(entries[(person, filename)]) as stream:
                    return pd.read_csv(stream, low_memory=False).drop_duplicates()

            yield sorted({p for p, _ in entries}), read
    elif source.is_dir():
        if dataset == "RESILIENT":
            people = sorted(p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith("."))
        else:
            people = sorted(p.stem for p in source.glob("P*.csv"))

        def read(person, filename):
            path = source / person / filename if dataset == "RESILIENT" else source / f"{person}.csv"
            return pd.read_csv(path, low_memory=False).drop_duplicates()

        yield people, read
    else:
        raise FileNotFoundError(source)


def hourly(times, values, total=False):
    # Remove exact CSV-row duplicates before this function. For multiple retained
    # values at one timestamp, take their mean before hourly aggregation.
    series = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=pd.DatetimeIndex(times))
    grouped = series.dropna().groupby(level=0).mean().resample("h", closed="right", label="right")
    return grouped.sum(min_count=1) if total else grouped.mean()


def load_resilient(person, read):
    physiology = read(person, "Sleep_physio.csv")
    times = pd.to_datetime(physiology.Timestamp, utc=True)
    data = {name: hourly(times, physiology[column]) for column, name in
            [("Heart Rate", "sleep_mat_hr"), ("Respiration Rate", "respiration"), ("SDNN_1", "sdnn")]}
    heart = read(person, "ScanWatch_HR.csv")
    data["watch_hr"] = hourly(pd.to_datetime(heart["HR Timestamp"], utc=True), heart["Heart Rate"])
    steps = read(person, "ScanWatch_Steps.csv")
    # Preserve the established availability convention: the hourly steps timestamp
    # does not establish availability within that hour, so delay it by one hour.
    data["steps"] = hourly(pd.to_datetime(steps["Steps Timestamp"], utc=True) + pd.Timedelta(hours=1),
                           steps.Steps, total=True)
    sleep = read(person, "Sleep_state.csv")
    start = pd.to_datetime(sleep["Start time"], utc=True)
    end = pd.to_datetime(sleep["End time"], utc=True)
    duration = (end - start).dt.total_seconds() / 3600
    valid = (duration > 0) & (duration <= 24)
    intervals = pd.DataFrame({"end": end[valid], "duration": duration[valid],
        "sleep": duration[valid] * sleep.loc[valid, "Sleep state"].isin(["light", "deep", "REM"]).astype(float)})
    # The entire observed interval becomes available at its end. This is a
    # duration-weighted state proportion, not reconstructed minute-level sleep.
    intervals = intervals.set_index("end").resample("h", closed="right", label="right").sum(min_count=1)
    data["sleep_prop"] = intervals.sleep / intervals.duration
    return pd.concat(data, axis=1, sort=True).sort_index()


def load_bris(person, read):
    observed = read(person, "observations.csv")
    # Public processed-state timestamps have no timezone: retain the source clock.
    times = pd.to_datetime(observed.timestamp)
    data = {name: hourly(times, observed[name], total=name not in {"bg", "hr"})
            for name in CHANNELS["BrisT1D-Open"]}
    return pd.concat(data, axis=1, sort=True).sort_index()


def to_grid(data, dataset):
    start = data.index.min().floor("D")
    stop = data.index.max().ceil("D")
    grid = pd.date_range(start, stop, freq="h", inclusive="left")
    values = data.reindex(grid + pd.Timedelta(hours=1))[CHANNELS[dataset]].to_numpy(np.float32)
    return values, str(start)


def prepare(dataset: str, source: Path, output: Path, verify_only=False):
    folder = output / dataset
    count = {"written": 0, "verified": 0, "hours": 0, "observed_values": 0}
    with source_tables(dataset, source) as (people, read):
        if not people:
            raise ValueError("No participant tables found; use the public ZIP or the documented participant directory")
        existing = {p.name.removesuffix("_hourly.npz") for p in folder.glob("*_hourly.npz")}
        if existing - set(people):
            raise ValueError("Existing hourly participant IDs are absent from the supplied source")
        if not verify_only:
            folder.mkdir(parents=True, exist_ok=True)
        for person in people:
            data = load_resilient(person, read) if dataset == "RESILIENT" else load_bris(person, read)
            values, clock = to_grid(data, dataset)
            path = folder / f"{person}_hourly.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if (saved["values"].dtype != values.dtype or str(saved["clock"]) != clock
                            or not np.array_equal(saved["values"], values, equal_nan=True)):
                        raise ValueError(f"Raw reconstruction differs from existing output: {path}")
                count["verified"] += 1
            elif verify_only:
                raise FileNotFoundError(path)
            else:
                np.savez_compressed(path, values=values, clock=clock)
                count["written"] += 1
            count["hours"] += len(values)
            count["observed_values"] += int(np.isfinite(values).sum())
    print(json.dumps({"dataset": dataset, "source": str(source.resolve()), "output": str(folder.resolve()),
                      "people": len(people), "channels": CHANNELS[dataset], **count}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=CHANNELS, required=True)
    parser.add_argument("--source", type=Path, required=True,
        help="Official ZIP; or extracted Sleepmat_Watch_Data (RES) / device_data/processed_state (Bris) directory")
    parser.add_argument("--output", type=Path, default=ROOT / "data/development")
    parser.add_argument("--verify-only", action="store_true", help="Recompute all raw inputs and compare without writing files")
    args = parser.parse_args()
    prepare(args.dataset, args.source, args.output, args.verify_only)


if __name__ == "__main__":
    main()
