"""Aggregate ESA-AD v2 channels and command logs for the common HNE protocol."""
import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/python"))
import zipfile_deflate64 as zipfile

BASE = ROOT / "data/new_cohorts/esa-ad-v2"


def commands(mission, spec, archive=None):
    folder = BASE / "metadata" / f"ESA-Mission{mission}"
    if archive is not None:
        folder.mkdir(parents=True, exist_ok=True)
        for name in ["channels.csv", "telecommands.csv", "labels.csv", "anomaly_types.csv"]:
            (folder / name).write_bytes(archive.read(f"ESA-Mission{mission}/{name}"))
    mapping = pd.read_csv(folder / "telecommands.csv")
    bounds = spec["split"][str(mission)]
    dates = pd.date_range(bounds["start"], pd.Timestamp(bounds["end_exclusive"])-pd.Timedelta(days=1), freq="D")
    counts = pd.DataFrame(0, index=dates, columns=[f"command_priority_{p}__count" for p in range(4)], dtype=np.int64)
    coverage = []
    for row in mapping.itertuples(index=False):
        member = f"ESA-Mission{mission}/telecommands/{row.Telecommand}.zip"
        source = io.BytesIO(archive.read(member)) if archive is not None else folder / "telecommands" / f"{row.Telecommand}.zip"
        frame = pd.read_pickle(source, compression="zip")
        by_day = pd.Series(1, index=frame.index).resample("D").sum() if len(frame) else pd.Series(dtype=np.int64)
        counts[f"command_priority_{int(row.Priority)}__count"] += by_day.reindex(dates, fill_value=0).astype(np.int64)
        coverage.append({"name": row.Telecommand, "priority": int(row.Priority), "count": len(frame),
                         "start": str(frame.index.min()) if len(frame) else "", "end": str(frame.index.max()) if len(frame) else ""})
    coverage = pd.DataFrame(coverage)
    if int(counts.to_numpy().sum()) != int(coverage["count"].sum()):
        raise ValueError("Declared mission dates do not contain all command executions")
    counts.rename_axis("date").to_csv(folder / "daily_commands.csv")
    coverage.to_csv(folder / "command_coverage.csv", index=False)
    print(json.dumps({"mission": mission, "commands": len(mapping), "executions": int(counts.to_numpy().sum()),
                      "days": len(counts), "counts_by_priority": counts.sum().to_dict()}), flush=True)
    return counts


def telemetry(mission, spec, protocol):
    cache = BASE / f"mission{mission}_daily_channels"
    cache.mkdir(exist_ok=True)
    folder = BASE / "metadata" / f"ESA-Mission{mission}"
    with zipfile.ZipFile(BASE / f"ESA-Mission{mission}.zip") as archive:
        command_frame = commands(mission, spec, archive)
        mapping = pd.read_csv(folder / "channels.csv")
        mapping = mapping[mapping.Categorical == "NO"]
        coverage = []
        for row in mapping.itertuples(index=False, name=None):
            name, subsystem, unit, group, target, categorical = row
            output = cache / f"{name}.csv"
            if output.exists():
                daily = pd.read_csv(output, index_col="date", parse_dates=True)
            else:
                member = f"ESA-Mission{mission}/channels/{name}.zip"
                frame = pd.read_pickle(io.BytesIO(archive.read(member)), compression="zip")
                series = frame[name].sort_index(kind="stable")
                series = series.groupby(level=0).mean()
                hourly = series.resample("h", closed="left", label="left").mean()
                grouped = hourly.resample("D", closed="left", label="left")
                daily = pd.DataFrame({"mean": grouped.mean(), "sd": grouped.std(ddof=0), "hours": grouped.count()})
                daily.rename_axis("date").to_csv(output)
                del frame, series, hourly
            coverage.append({"channel": name, "subsystem": subsystem, "target": target,
                             "first_date": str(daily.index.min()), "last_date": str(daily.index.max()),
                             "observed_days": int((daily.hours > 0).sum()), "four_hour_days": int((daily.hours >= 4).sum())})
            print(f"Mission{mission} {name}: {int((daily.hours > 0).sum())} observed days", flush=True)

    dates = command_frame.index
    states, contexts, scounts, ccounts, snames, cnames, families = [], [], [], [], [], [], []
    for row in mapping.itertuples(index=False, name=None):
        name, subsystem, unit, group, target, categorical = row
        daily = pd.read_csv(cache / f"{name}.csv", index_col="date", parse_dates=True).reindex(dates)
        hours = daily.hours.fillna(0).to_numpy(np.int16)
        if target == "YES":
            good = hours >= protocol["hourly_minimum_state_observations_per_day"]
            for statistic in ["mean", "sd"]:
                states.append(np.where(good, daily[statistic].to_numpy(), np.nan))
                scounts.append(hours)
                snames.append(f"{name}__{statistic}")
                families.append(subsystem)
        else:
            good = hours >= protocol["hourly_minimum_context_observations_per_day"]
            contexts.append(np.where(good, daily["mean"].to_numpy(), np.nan))
            ccounts.append(hours)
            cnames.append(f"{name}__mean")
    for name in command_frame.columns:
        contexts.append(command_frame[name].to_numpy())
        ccounts.append(np.ones(len(dates), dtype=np.int16))
        cnames.append(name)
    payload = {"states": np.column_stack(states).astype(np.float32),
               "contexts": np.column_stack(contexts).astype(np.float32),
               "state_observed_counts": np.column_stack(scounts), "context_observed_counts": np.column_stack(ccounts),
               "state_names": np.asarray(snames), "context_names": np.asarray(cnames), "state_families": np.asarray(families),
               "dates": dates.to_numpy(dtype="datetime64[D]"),
               "available_at": (dates + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[s]"),
               "timezone_semantics": np.asarray(spec["clock"])}
    np.savez_compressed(BASE / f"mission{mission}_daily.npz", **payload)
    pd.DataFrame(coverage).to_csv(folder / "channel_coverage.csv", index=False)
    schema = {"mission": mission, "state_names": snames, "context_names": cnames, "state_families": families,
              "days": len(dates), "state_shape": list(payload["states"].shape), "context_shape": list(payload["contexts"].shape),
              "mapping": spec}
    (folder / "daily_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(json.dumps({k: schema[k] for k in ["mission", "days", "state_shape", "context_shape"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["commands", "telemetry"], required=True)
    parser.add_argument("--missions", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    spec = json.loads((ROOT / "configs/esa_mapping.json").read_text())
    protocol = json.loads((ROOT / spec["daily_protocol"]).read_text())
    for mission in args.missions:
        if args.phase == "commands":
            commands(mission, spec)
        else:
            telemetry(mission, spec, protocol)
