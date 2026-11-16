"""Map TOLIFE daily indices; the command-line default prepares train/validation.

The frozen-source evaluator requests test explicitly in a separate output folder.
Time assignment follows the release README; clinical estimates are not inputs.
"""
from __future__ import annotations

import argparse
import calendar
import csv
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data/new_cohorts/tolife-csa-2026"
INDICES = ("EI", "SHRI", "SII", "MI", "BRI", "SQI", "PRVI", "PEI")
STATE = {"SHRI", "BRI", "SQI", "PRVI", "PEI"}
ZONES = {"GERMANY": "Europe/Berlin", "SPAIN": "Europe/Madrid", "ITALY": "Europe/Rome"}
NAN = float("nan")


def european_local(ts: datetime, country: str) -> datetime:
    """Use IANA zones, or the exact shared EU rule for release years 2024–2026.

    Windows stdlib Python may lack tzdata. Germany, mainland Spain, and Italy
    use UTC+1 winter/UTC+2 summer in this release, changing at 01:00 UTC on the
    last Sundays of March/October. This fallback is restricted to those years.
    """
    try:
        return ts.astimezone(ZoneInfo(ZONES[country]))
    except ZoneInfoNotFoundError:
        if not 2024 <= ts.year <= 2026:
            raise ValueError("Install tzdata to process dates outside the released 2024–2026 period")
        def last_sunday(month: int) -> datetime:
            last = date(ts.year, month, calendar.monthrange(ts.year, month)[1])
            sunday = last - timedelta(days=(last.weekday() + 1) % 7)
            return datetime.combine(sunday, datetime.min.time(), timezone.utc) + timedelta(hours=1)
        summer = last_sunday(3) <= ts < last_sunday(10)
        return ts.astimezone(timezone(timedelta(hours=2 if summer else 1)))


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).astimezone(timezone.utc)


def assign_day(index: str, ts: datetime, country: str) -> tuple[date, str]:
    if index in {"SHRI", "BRI"}:
        return european_local(ts, country).date() + timedelta(days=1), index.lower()
    if index == "SQI":
        return ts.date(), index.lower()
    if index in {"EI", "SII"}:
        local = european_local(ts, country)
        night = local.hour >= 20
        return local.date() + timedelta(days=int(night)), index.lower() + ("_night" if night else "_day")
    # MI, PRVI and PEI use the 20:00 UTC aggregation-day boundary.
    return ts.date() + timedelta(days=int(ts.hour >= 20)), index.lower()


def number(text: str | None) -> float:
    if text is None or not text.strip():
        return NAN
    value = float(text)
    return value if math.isfinite(value) else NAN


def flag(text: str | None) -> int | None:
    if text is None or not text.strip():
        return None
    value = int(float(text))
    if value not in (0, 1):
        raise ValueError(f"Unexpected validity flag {value}")
    return value


def prepare(source: Path, splits=("train", "validation"), output=None) -> dict:
    with (source / "subject_split.csv").open(encoding="utf-8-sig", newline="") as f:
        subjects = list(csv.DictReader(f))
    # This filter runs before opening any participant directory or sensor file.
    development = [r for r in subjects if r["split"] in splits]
    reserved = {r["Kit-ID"] for r in subjects if r["split"] not in splits}
    source_schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    slots = [(index, slot) for index in INDICES for slot in
             ([index.lower() + "_night", index.lower() + "_day"] if index in {"EI", "SII"} else [index.lower()])]
    features = {}
    source_columns = {}
    metadata_columns = []
    for index, slot in slots:
        role = "state" if index in STATE else "context"
        source_columns[index] = [c for c in source_schema[index] if c not in {"Timestamp", "index_validity", "data_validity"}]
        for col in source_columns[index]:
            name = f"{role}__{slot}__{col}"
            features[name] = {"role": role, "source_file": index + ".csv", "source_column": col,
                              "period": slot, "valid_when": "index_validity == 1 and data_validity == 1"}
        metadata_columns.extend(f"meta__{slot}__{suffix}" for suffix in [
            "source_rows", "valid_rows", "source_timestamps", "source_index_validity", "source_data_validity",
            "selected_timestamp", "index_validity", "data_validity", "usable", "available_at_utc"])
    fields = ["participant", "split", "country", "date", "available_at_utc", *features, *metadata_columns]
    output = source / "daily" if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for person in development:
        pid, country = person["Kit-ID"], person["Country"]
        if pid in reserved:
            raise ValueError("Reserved participant reached preparation")
        grouped = {}
        actual_days = set()
        for index in INDICES:
            path = source / "index" / country / pid / (index + ".csv")
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != source_schema[index]:
                    raise ValueError(f"Schema changed: {pid}/{index}")
                for row in reader:
                    ts = parse_time(row["Timestamp"])
                    day, slot = assign_day(index, ts, country)
                    iv, dv = flag(row["index_validity"]), flag(row["data_validity"])
                    grouped.setdefault((day, slot), []).append((ts, iv, dv, row))
                    if dv == 1:
                        actual_days.add(day)
        if not actual_days:
            summary.append({"participant": pid, "split": person["split"], "rows": 0, "reason": "no source data_validity=1"})
            continue
        first, last = min(actual_days), max(actual_days)
        manifest_match = first.isoformat() == person["first_day"] and last.isoformat() == person["last_day"]
        n_rows = (last - first).days + 1
        duplicates = 0
        with (output / (pid + ".csv")).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for day_num in range(n_rows):
                day = first + timedelta(days=day_num)
                # Conservative release boundary: all index windows of D have ended by next UTC midnight.
                available = (datetime.combine(day + timedelta(days=1), datetime.min.time(), timezone.utc)
                             .isoformat().replace("+00:00", "Z"))
                result = {"participant": pid, "split": person["split"], "country": country,
                          "date": day.isoformat(), "available_at_utc": available}
                result.update({feature: NAN for feature in features})
                for index, slot in slots:
                    rows = sorted(grouped.get((day, slot), []), key=lambda r: r[0])
                    valid = [r for r in rows if r[1] == 1 and r[2] == 1]
                    duplicates += int(len(rows) > 1)
                    # Do not pool daily SD/range statistics. If multiple records map to a day,
                    # use the latest valid released daily record, retaining all source times/flags.
                    selected = valid[-1] if valid else rows[-1] if rows else None
                    prefix = f"meta__{slot}__"
                    result.update({prefix + "source_rows": len(rows), prefix + "valid_rows": len(valid),
                                   prefix + "source_timestamps": json.dumps([r[3]["Timestamp"] for r in rows]),
                                   prefix + "source_index_validity": json.dumps([r[1] for r in rows]),
                                   prefix + "source_data_validity": json.dumps([r[2] for r in rows]),
                                   prefix + "selected_timestamp": selected[3]["Timestamp"] if selected else "",
                                   prefix + "index_validity": selected[1] if selected else NAN,
                                   prefix + "data_validity": selected[2] if selected else NAN,
                                   prefix + "usable": int(bool(valid)),
                                   prefix + "available_at_utc": available if selected else ""})
                    if valid:
                        role = "state" if index in STATE else "context"
                        for col in source_columns[index]:
                            result[f"{role}__{slot}__{col}"] = number(selected[3][col])
                writer.writerow(result)
        summary.append({"participant": pid, "split": person["split"], "rows": n_rows,
                        "first_actual_day": first.isoformat(), "last_actual_day": last.isoformat(),
                        "source_data_days": len(actual_days), "manifest_bounds_match": manifest_match,
                        "multiple_records_in_slot_days": duplicates})
    mapping = {
        "source": "https://zenodo.org/records/23263785", "license": "CC-BY-4.0",
        "source_document": "../README.md", "output_frequency": "one calendar day",
        "splits_read": list(splits), "test_participants_read": sum(r['split']=='test' for r in development),
        "state_columns": [k for k, v in features.items() if v["role"] == "state"],
        "context_columns": [k for k, v in features.items() if v["role"] == "context"],
        "features": features,
        "validity": {"index_validity": "1=index computed valid;0=invalid",
                     "data_validity": "1=raw source data available;0=missing",
                     "use_rule": "both flags 1; otherwise all numeric feature values are NaN",
                     "absent_row": "source_rows=0;flags=NaN;usable=0;source timestamps=[]"},
        "date_assignment": {"SHRI/BRI": "UTC timestamp -> site local calendar date +1 day",
                            "SQI": "UTC calendar date",
                            "EI/SII": "site local timestamp;20:00 or later => next day/night;earlier => current day/day",
                            "MI/PRVI/PEI": "UTC timestamp;20:00 or later => next day;otherwise current day"},
        "site_timezones": ZONES,
        "availability": "Conservative next UTC midnight after aggregation day; source Timestamp is a period label, not always completion time",
        "grid": "complete calendar grid from first to last mapped day with any data_validity=1; gaps stay NaN",
        "multiple_records": "latest valid daily record per slot; no pooling of SD/range; every original timestamp and validity flag retained",
        "unused_files": ["CAT.csv", "CCQ.csv", "mMRC.csv", "clinical_indices.csv", "exacerbations.csv"],
        "endpoint": "none; no disorder labels or hourly interpolation generated"}
    (output / "schema_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    report = {"people_written": sum(r["rows"] > 0 for r in summary),
              "split_counts": {s: sum(r["split"] == s for r in summary) for s in splits},
              "test_participants_read": mapping['test_participants_read'], "state_features": len(mapping["state_columns"]),
              "context_features": len(mapping["context_columns"]), "daily_rows": sum(r["rows"] for r in summary),
              "manifest_bound_mismatches": [r["participant"] for r in summary if r.get("manifest_bounds_match") is False],
              "people": summary}
    (output / "preparation_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    args = parser.parse_args()
    report = prepare(args.source)
    print(json.dumps({k: v for k, v in report.items() if k != "people"}, indent=2))
