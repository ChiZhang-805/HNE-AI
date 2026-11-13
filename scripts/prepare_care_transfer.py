"""Adapt official CARE v6 SCADA to causal hourly/daily, unlabeled HNE inputs.

No target model fitting, event-threshold calibration, interpolation, or status-based
filtering is performed. Case boundaries are retained because anonymization shifts
each case's year independently. Published event/status annotations live separately.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path("Q:/NAVA-Workspace/DNB-Research/Data/CARE-v6/CARE_To_Compare.zip")
EXPECTED_MD5 = "2547b58c21ac8c242d13232860cf500c"
EXPECTED_BYTES = 5_503_439_673
VERSION = "care-v6-causal-aggregate-v1-20261110"
# Explicit mapping from the publisher's feature_description.csv, never column order.
ENVIRONMENT = {
    "A": {"sensor_0": "Ambient temperature", "wind_speed_3": "Windspeed",
          "wind_speed_4": "Estimated windspeed"},
    "B": {"sensor_6": "Wind speed anemometer 2 calculated", "sensor_8": "Outside temperature",
          "wind_speed_59": "Wind speed 1", "wind_speed_60": "Wind speed 2", "wind_speed_61": "Wind speed"},
    "C": {"sensor_7": "Ambient temperature", "sensor_40": "Nacelle external air pressure",
          "sensor_41": "Nacelle outside temperature", "sensor_177": "Outside temperature",
          "wind_speed_235": "Wind speed 1", "wind_speed_236": "Wind speed 1+2",
          "wind_speed_237": "Wind speed 2"},
}
SOURCE_URLS = {
    "data": "https://zenodo.org/records/15846963",
    "paper": "https://doi.org/10.3390/data9120138",
    "faq": "https://aefdi.github.io/EnergyFaultDetector/care2compare_faq.html",
}


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def feature_schema(archive, farm, metadata_dir):
    base = f"CARE_To_Compare/Wind Farm {farm}/"
    raw = archive.read(base + "feature_description.csv")
    (metadata_dir / f"farm_{farm}_feature_description.csv").write_bytes(raw)
    description = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin1").fillna("")
    mapping = {row["sensor_name"]: row for row in description.to_dict("records")}
    for name, expected in ENVIRONMENT[farm].items():
        if mapping[name]["description"].strip() != expected:
            raise ValueError(f"Environment description changed: {farm}/{name}")
    names = sorted(n for n in archive.namelist() if n.startswith(base + "datasets/") and n.endswith(".csv"))
    with archive.open(names[0]) as stream:
        header = pd.read_csv(stream, sep=";", nrows=0).columns.tolist()
    states, contexts, excluded = [], [], []
    for col in header:
        if not col.endswith("_avg"):
            continue
        sensor = col[:-4]
        if sensor not in mapping:
            raise ValueError(f"Missing official description for {farm}/{sensor}")
        row = mapping[sensor]
        record = {"column": col, **row}
        if bool(row["is_angle"]) or bool(row["is_counter"]):
            record["reason"] = "angle_or_counter_requires_distinct_transform"
            excluded.append(record)
        elif sensor in ENVIRONMENT[farm]:
            contexts.append(record)
        else:
            states.append(record)
    if {r["sensor_name"] for r in contexts} != set(ENVIRONMENT[farm]):
        raise ValueError(f"Missing context averages for farm {farm}")
    schema = {"farm": farm, "source_columns": len(header), "source_average_columns": sum(c.endswith("_avg") for c in header),
              "states": states, "contexts": contexts, "excluded_average_columns": excluded,
              "state_selection": "all documented non-angle, non-counter averages except the explicit environment fields",
              "daily_state_aggregation": ["mean_of_hourly_means", "population_sd_of_hourly_means"],
              "daily_context_aggregation": "mean_of_hourly_means",
              "within_sensor_daily_family": "mean and sd share sensor_name; no component-level family taxonomy inferred"}
    dump(metadata_dir / f"farm_{farm}_schema.json", schema)
    return schema


def aggregate(frame, schema, min_daily_state, min_daily_context):
    state_cols = [r["column"] for r in schema["states"]]
    context_cols = [r["column"] for r in schema["contexts"]]
    cols = state_cols + context_cols
    # A source sample at h:50 is included only in the result available at h+1:00.
    # Timezone and whether the original 10-min timestamp denotes interval start/end
    # are unspecified, so use conservative end-of-container availability.
    data = frame.set_index("time_stamp")[cols].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    hourly_groups = data.resample("h", closed="left", label="left")
    means, counts = hourly_groups.mean(), hourly_groups.count()
    p = len(state_cols)
    common = {"state_names": np.asarray(state_cols), "context_names": np.asarray(context_cols),
              "state_families": np.asarray([r["sensor_name"] for r in schema["states"]]),
              "timezone_semantics": np.asarray("source_clock_timezone_unspecified; independent case year anonymization")}
    hourly = {**common, "states": means[state_cols].to_numpy(np.float32),
              "contexts": means[context_cols].to_numpy(np.float32),
              "state_observed_counts": counts[state_cols].to_numpy(np.int16),
              "context_observed_counts": counts[context_cols].to_numpy(np.int16),
              "timestamps": means.index.to_numpy(dtype="datetime64[s]"),
              "available_at": (means.index + pd.Timedelta(hours=1)).to_numpy(dtype="datetime64[s]")}
    by_day = means.resample("D", closed="left", label="left")
    dmean, dsd, dcount = by_day.mean(), by_day.std(ddof=0), by_day.count()
    good_state = dcount[state_cols].to_numpy() >= min_daily_state
    states = np.stack([dmean[state_cols].to_numpy(), dsd[state_cols].to_numpy()], axis=2)
    states[~good_state] = np.nan
    contexts = dmean[context_cols].to_numpy(copy=True)
    contexts[dcount[context_cols].to_numpy() < min_daily_context] = np.nan
    daily = {**common, "states": states.reshape(len(dmean), p * 2).astype(np.float32),
             "contexts": contexts.astype(np.float32),
             "state_observed_counts": np.repeat(dcount[state_cols].to_numpy(np.int16), 2, axis=1),
             "context_observed_counts": dcount[context_cols].to_numpy(np.int16),
             "state_names": np.asarray([c + "__" + stat for c in state_cols for stat in ("mean", "sd")]),
             "state_families": np.repeat(common["state_families"], 2),
             "dates": dmean.index.to_numpy(dtype="datetime64[D]"),
             "available_at": (dmean.index + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[s]")}
    return hourly, daily


def prepare_case(archive, farm, event, schema, output, min_state, min_context):
    event_id, asset = int(event["event_id"]), int(event["asset_id"])
    case = f"farm_{farm}_event_{event_id}"
    member = f"CARE_To_Compare/Wind Farm {farm}/datasets/{event_id}.csv"
    columns = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
    columns += [r["column"] for r in schema["states"] + schema["contexts"]]
    with archive.open(member) as stream:
        frame = pd.read_csv(stream, sep=";", usecols=columns)
    frame["time_stamp"] = pd.to_datetime(frame["time_stamp"], format="%Y-%m-%d %H:%M:%S")
    if set(frame.asset_id) != {asset}:
        raise ValueError(f"Asset mismatch: {case}")
    frame = frame.sort_values("time_stamp", kind="stable")
    if frame.time_stamp.duplicated().any():
        raise ValueError(f"Duplicate source timestamps: {case}; do not silently merge")
    delta = frame.time_stamp.diff().dropna().dt.total_seconds().to_numpy()
    if np.any(delta <= 0) or np.any(delta % 600 != 0):
        raise ValueError(f"Not on a 10-minute source grid: {case}")
    hourly, daily = aggregate(frame, schema, min_state, min_context)
    for frequency, payload in (("hourly", hourly), ("daily", daily)):
        payload.update({"case_id": np.asarray(case), "asset_group": np.asarray(f"farm_{farm}_asset_{asset}"),
                        "farm": np.asarray(farm), "source_member": np.asarray(member)})
        np.savez_compressed(output / frequency / f"{case}.npz", **payload)
    # This table is never joined into the input arrays, even for Farm A's
    # retrospectively backdated fault-status markers.
    annotations = frame[["time_stamp", "id", "train_test", "status_type_id"]].copy()
    annotations.insert(0, "asset_group", f"farm_{farm}_asset_{asset}")
    annotations.insert(0, "case_id", case)
    annotations.to_csv(output / "annotations" / f"{case}_timestamps.csv.gz", index=False)
    train = frame[frame.train_test == "train"]
    prediction = frame[frame.train_test == "prediction"]
    sensor = frame[columns[5:]].to_numpy(np.float64)
    span = lambda x: float((x.time_stamp.max() - x.time_stamp.min()).total_seconds() / 86400 + 1 / 144) if len(x) else 0.0
    item = {"case_id": case, "farm": farm, "asset_id": asset, "asset_group": f"farm_{farm}_asset_{asset}",
            "source_member": member, "source_crc32": f"{archive.getinfo(member).CRC:08x}",
            "source_csv_bytes": archive.getinfo(member).file_size,
            "source_rows": len(frame), "time_start": str(frame.time_stamp.min()), "time_end": str(frame.time_stamp.max()),
            "sampling_minutes": 10, "calendar_days": span(frame), "train_rows": len(train), "train_calendar_days": span(train),
            "prediction_rows": len(prediction), "prediction_calendar_days": span(prediction),
            "prediction_start": str(prediction.time_stamp.min()), "prediction_end": str(prediction.time_stamp.max()),
            "missing_timestamp_slots": int(np.sum(delta // 600 - 1)), "max_gap_minutes": float(max(delta) / 60),
            "nonfinite_measurement_fraction": float(1 - np.isfinite(sensor).mean()),
            "zero_measurement_fraction": float(np.mean(sensor == 0)),
            "hourly_rows": len(hourly["states"]), "daily_rows": len(daily["states"]),
            "hourly_state_count": hourly["states"].shape[1], "daily_state_count": daily["states"].shape[1],
            "context_count": hourly["contexts"].shape[1],
            "event_label": str(event["event_label"]), "event_start": str(event["event_start"]), "event_end": str(event["event_end"])}
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=ROOT / "data/care_transfer")
    parser.add_argument("--farms", nargs="+", choices=list("ABC"), default=list("ABC"))
    args = parser.parse_args()
    # Full local archive verification replaces downloading any new data.
    if args.archive.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Archive size differs from official v6")
    with args.archive.open("rb") as handle:
        md5 = hashlib.file_digest(handle, "md5").hexdigest()
    if md5 != EXPECTED_MD5:
        raise ValueError(f"Archive checksum differs from official v6: {md5}")
    config_path = ROOT / "configs/daily_protocol.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    min_state, min_context = (config["hourly_minimum_state_observations_per_day"],
                              config["hourly_minimum_context_observations_per_day"])
    for folder in ("hourly", "daily", "annotations", "source_metadata"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    records, all_events, schemas = [], [], {}
    with zipfile.ZipFile(args.archive) as archive:
        (args.output / "source_metadata/official_README.md").write_bytes(archive.read("CARE_To_Compare/README.md"))
        for farm in args.farms:
            schemas[farm] = schema = feature_schema(archive, farm, args.output / "source_metadata")
            raw = archive.read(f"CARE_To_Compare/Wind Farm {farm}/event_info.csv")
            (args.output / "source_metadata" / f"farm_{farm}_event_info.csv").write_bytes(raw)
            events = pd.read_csv(io.BytesIO(raw), sep=";").rename(columns={"asset": "asset_id"}).fillna("")
            events.insert(0, "farm", farm)
            events.insert(0, "case_id", [f"farm_{farm}_event_{v}" for v in events.event_id])
            all_events.append(events)
            for event in events.sort_values("event_id").to_dict("records"):
                item = prepare_case(archive, farm, event, schema, args.output, min_state, min_context)
                records.append(item)
                print(json.dumps({k: item[k] for k in ("case_id", "asset_id", "source_rows", "calendar_days", "hourly_state_count", "context_count")}), flush=True)
    cases = pd.DataFrame(records)
    cases.to_csv(args.output / "case_inventory.csv", index=False)
    assets = cases.groupby(["farm", "asset_id", "asset_group"], as_index=False).agg(
        case_count=("case_id", "size"), min_case_days=("calendar_days", "min"), max_case_days=("calendar_days", "max"),
        summed_case_days_not_unique=("calendar_days", "sum"), source_rows_across_cases=("source_rows", "sum"),
        min_prediction_days=("prediction_calendar_days", "min"), max_prediction_days=("prediction_calendar_days", "max"))
    assets.to_csv(args.output / "asset_inventory.csv", index=False)
    pd.concat(all_events, ignore_index=True).to_csv(args.output / "annotations/events.csv", index=False)
    manifest = {
        "version": VERSION, "source_archive": str(args.archive.resolve()), "archive_bytes": EXPECTED_BYTES,
        "archive_md5": md5, "source_urls": SOURCE_URLS, "license": "CC BY-SA 4.0",
        "case_count": len(cases), "asset_count": len(assets), "event_counts": cases.event_label.value_counts().to_dict(),
        "farm_schemas": {f: {"hourly_states": len(s["states"]), "daily_states": 2 * len(s["states"]), "contexts": len(s["contexts"])} for f, s in schemas.items()},
        "causal_aggregation": "10-minute values in [h,h+1h) mean become available at h+1h; daily means/SD of completed hourly means become available at next midnight",
        "hourly_minimum_observations": 1, "daily_minimum_observed_hours_state": min_state,
        "daily_minimum_observed_hours_context": min_context,
        "daily_config_source": str(config_path), "daily_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "input_values": "unstandardized physical values; no learned transformation, no imputation, no interpolation, no status/event mask",
        "missingness": "explicit NaN/absent timestamps remain missing; operator's undocumented zero-filled B/C values retained because true zero cannot be distinguished",
        "timestamp_semantics": "unknown timezone; each case independently year-shifted to start in 2022; never concatenate cases or sum durations as unique asset time",
        "asset_grouping": "same farm and asset_id identify the same turbine across cases; cluster evaluation at asset level",
        "official_event_semantics": "anomaly interval precedes a fault; event_end is fault/downtime boundary; normal cases have a designated normal interval; event_start is retrospectively estimated and not an observed warning target",
        "annotations": "separate annotations/events.csv and per-case 10-minute annotations; status/train_test/row ID never included in model inputs",
        "historical_use": "all 95 CARE v6 cases previously used by rhythm-dnb-care-v6-20261026 and adopted native CARE source; target-domain unseen-model evaluation, not an untouched dataset",
        "historical_source": "Q:/NAVA-Workspace/DNB-Research/Reference-Repositories/HNE/experiments/native/CARE/source.json",
        "hne_transfer_model_training_or_scores_read": False,
        "future_event_labels_created": False,
        "family_semantics": "state_families groups daily mean/SD by physical sensor name; component-level grouping and future target are not invented by this adapter",
    }
    dump(args.output / "manifest.json", manifest)
    print(json.dumps({"complete": True, "cases": len(cases), "assets": len(assets), "events": manifest["event_counts"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
