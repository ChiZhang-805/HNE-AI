"""Fetch pinned, open HNE cohorts and inventory schema without reading outcomes."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1] / "data" / "new_cohorts"
VERSION = "v1.3.0"
REPO = "digital-wellbeing/open-play"


def request(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "HNE-research-data-fetch/1.0"}), timeout=120)


def fetch_open_play() -> None:
    dest = ROOT / "open-play-2026"
    dest.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://api.github.com/repos/{REPO}/git/trees/{VERSION}?recursive=1"
    with request(tree_url) as r:
        tree = json.load(r)
    wanted = {"README.md", "LICENSE", "codebook.xlsx", "CITATION.cff", "data-process.qmd"}
    wanted.update(f"data/clean/{name}.csv.gz" for name in [
        "survey_daily", "survey_biweekly", "timeuse", "steam", "nintendo", "xbox", "ios", "android"
    ])
    entries = [x for x in tree["tree"] if x["path"] in wanted]
    if set(x["path"] for x in entries) != wanted:
        raise ValueError("Pinned source is missing expected files")
    if sum(x["size"] for x in entries) > 5_000_000_000:
        raise ValueError("Download exceeds agreed 5 GB limit")
    priority = {"data/clean/survey_daily.csv.gz": 0, "data/clean/timeuse.csv.gz": 1,
                "data/clean/survey_biweekly.csv.gz": 2}
    entries.sort(key=lambda x: (priority.get(x["path"], 3), x["path"]))
    def download(entry):
        relative = entry["path"]
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{REPO}/{VERSION}/{relative}"
        if not target.exists():
            with request(url) as r, target.with_suffix(target.suffix + ".part").open("wb") as out:
                while block := r.read(64 * 1024):
                    out.write(block)
            target.with_suffix(target.suffix + ".part").replace(target)
        content = target.read_bytes()
        blob_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if len(content) != entry["size"] or blob_sha != entry["sha"]:
            raise ValueError(f"Size/blob mismatch for {relative}")
        print(f"Verified {relative}: {len(content):,} bytes", flush=True)
        return {"path": relative, "url": url, "bytes": len(content),
                "github_blob_sha1": blob_sha, "sha256": hashlib.sha256(content).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download, entries))
    meta = {"dataset": "Open Play", "version": VERSION,
            "source": "https://zenodo.org/records/23419571",
            "repository": f"https://github.com/{REPO}/tree/{VERSION}",
            "license": "CC0-1.0 modified with no-reidentification clause; see LICENSE",
            "downloaded_utc_date": "2026-11-10", "tree_sha": tree["sha"],
            "scope": "official clean daily and biweekly surveys, time use and all five telemetry platforms; documentation",
            "files": manifest}
    (dest / "source_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    schema = {}
    for target in sorted((dest / "data" / "clean").glob("*.csv.gz")):
        with gzip.open(target, "rt", encoding="utf-8-sig", newline="") as f:
            schema[target.name] = next(csv.reader(f))
    (dest / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(schema, indent=2))


def reserve_open_play() -> None:
    """Read identifiers/waves/dates only; never calculate outcomes or label rates."""
    dest = ROOT / "open-play-2026"
    by_person = {}
    with gzip.open(dest / "data/clean/survey_daily.csv.gz", "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        schema = reader.fieldnames
        for row in reader:
            # Only these three metadata fields are accessed; all state values remain unused.
            pid, wave, date = row["pid"], int(row["wave"]), row["date"]
            item = by_person.setdefault(pid, {"waves": set(), "dates": set(), "rows": 0})
            item["waves"].add(wave)
            item["dates"].add(date[:10])
            item["rows"] += 1
    seed = "HNE-new-cohort-20261110-v1"
    ordered = sorted(by_person, key=lambda pid: hashlib.sha256(f"{seed}|{pid}".encode()).hexdigest())
    n = len(ordered)
    n_test = round(n * .2)
    n_validation = round(n * .2)
    records = []
    for i, pid in enumerate(ordered):
        split = "test" if i < n_test else "validation" if i < n_test + n_validation else "train"
        item = by_person[pid]
        waves = sorted(item["waves"])
        longest = run = 0
        previous = None
        for wave in waves:
            run = run + 1 if previous is not None and wave == previous + 1 else 1
            longest = max(longest, run)
            previous = wave
        records.append({"pid": pid, "split": split, "n_survey_rows": item["rows"],
                        "n_waves": len(waves), "n_calendar_dates_utc": len(item["dates"]),
                        "first_wave": min(waves), "last_wave": max(waves),
                        "longest_consecutive_waves": longest})
    with (dest / "subject_split.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: r["pid"]))
    counts = [r["n_waves"] for r in records]
    stats = {"rows": sum(r["n_survey_rows"] for r in records), "participants": n,
             "daily_columns": schema, "observed_waves_min_median_max": [min(counts), statistics.median(counts), max(counts)],
             "date_range_utc": [min(d for x in by_person.values() for d in x["dates"]),
                                max(d for x in by_person.values() for d in x["dates"])],
             "at_least_21_waves": sum(c >= 21 for c in counts),
             "at_least_21_consecutive_waves": sum(r["longest_consecutive_waves"] >= 21 for r in records),
             "split_method": f"sort SHA256({seed}|pid); first rounded 20% test, next rounded 20% validation, remainder train",
             "split_counts": {s: sum(r["split"] == s for r in records) for s in ["train", "validation", "test"]},
             "outcomes_inspected": False,
             "inspected_fields": ["pid", "wave", "date"],
             "purpose": "test people reserved before state, event or label inspection; train/validation available for development"}
    (dest / "cohort_metadata.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "daily_columns"}, indent=2))


def fetch_tolife() -> None:
    dest = ROOT / "tolife-csa-2026"
    dest.mkdir(parents=True, exist_ok=True)
    with request("https://zenodo.org/api/records/23263785") as r:
        record = json.load(r)
    manifest = []
    for entry in record["files"]:
        target = dest / entry["key"]
        if not target.exists():
            with request(entry["links"]["self"]) as r:
                target.write_bytes(r.read())
        content = target.read_bytes()
        expected = entry["checksum"].split(":", 1)[1]
        if len(content) != entry["size"] or hashlib.md5(content).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {target.name}")
        manifest.append({"file": entry["key"], "url": entry["links"]["self"],
                         "bytes": len(content), "md5": expected,
                         "sha256": hashlib.sha256(content).hexdigest()})
    with zipfile.ZipFile(dest / "index.zip") as z:
        if sum(i.file_size for i in z.infolist()) > 5_000_000_000:
            raise ValueError("Expanded cohort exceeds 5 GB limit")
        for item in z.infolist():
            output = (dest / item.filename).resolve()
            if not output.is_relative_to(dest.resolve()):
                raise ValueError("Archive member outside dataset directory")
        z.extractall(dest)
    (dest / "source_manifest.json").write_text(json.dumps({
        "dataset": "TOLIFE CSA", "version": "V2.0", "published": "2026-10-09",
        "source": "https://zenodo.org/records/23263785", "license": "CC-BY-4.0",
        "files": manifest, "outcomes_inspected": False}, indent=2) + "\n", encoding="utf-8")
    root = dest / "index"
    with (root / "kits.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        print("kits.csv fields:", reader.fieldnames)
        # In particular, n_exacerbations is deliberately never accessed.
        metadata = [{k: row[k] for k in ["Kit-ID", "Country", "first_day", "last_day", "n_days"]}
                    for row in reader]
    seed = "HNE-new-cohort-20261110-v1"
    ordered = sorted(metadata, key=lambda r: hashlib.sha256(f"{seed}|{r['Kit-ID']}".encode()).hexdigest())
    n = len(ordered)
    cutoff = round(n * .2)
    for i, item in enumerate(ordered):
        item["split"] = "test" if i < cutoff else "validation" if i < 2 * cutoff else "train"
    with (dest / "subject_split.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Kit-ID", "Country", "first_day", "last_day", "n_days", "split"])
        writer.writeheader()
        writer.writerows(sorted(ordered, key=lambda r: r["Kit-ID"]))
    schema = {}
    for name in ["EI", "SHRI", "SII", "MI", "BRI", "SQI", "PRVI", "PEI"]:
        path = next(root.glob(f"*/*/{name}.csv"))
        with path.open(encoding="utf-8-sig", newline="") as f:
            schema[name] = next(csv.reader(f))
    days = [int(row["n_days"]) for row in metadata]
    stats = {"participants": n, "valid_sensor_days_min_median_max": [min(days), statistics.median(days), max(days)],
             "participants_at_least_60_days": sum(x >= 60 for x in days),
             "participants_at_least_90_days": sum(x >= 90 for x in days),
             "total_sensor_person_days": sum(days),
             "first_date": min(row["first_day"] for row in metadata),
             "last_date": max(row["last_day"] for row in metadata),
             "split_method": f"sort SHA256({seed}|Kit-ID); first rounded 20% test, next rounded 20% validation, remainder train",
             "split_counts": {s: sum(row["split"] == s for row in metadata) for s in ["train", "validation", "test"]},
             "inspected_fields": ["Kit-ID", "Country", "first_day", "last_day", "n_days", "sensor CSV headers"],
             "outcomes_inspected": False,
             "excluded_from_state_inputs": ["CAT", "CCQ", "mMRC", "clinical_indices", "exacerbations"],
             "sampling": "Daily/nightly; EI and SII twice daily. Do not upsample to hourly or count interpolated rows.",
             "notes": "CAT/CCQ/mMRC are learned estimates, not participant-reported ground truth. Reserved test events have not been explored."}
    (dest / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    (dest / "cohort_metadata.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=["open-play", "tolife"])
    parser.add_argument("--reserve-only", action="store_true", help="Read ID/time metadata and freeze person splits")
    args = parser.parse_args()
    if args.cohort == "open-play":
        if args.reserve_only:
            reserve_open_play()
        else:
            fetch_open_play()
    elif args.cohort == "tolife":
        fetch_tolife()
