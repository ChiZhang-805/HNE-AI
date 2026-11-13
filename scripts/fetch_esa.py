"""Download the official ESA-AD v2 archives into the current project."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    1: (3776246073, "9770ad12ed730238f37c42d5c27ab436"),
    2: (None, "bfc72012691427d9327eb41f726ce45e"),
    3: (None, "d63943f09c81378acd9fc5e565ecc66e"),
}


def fetch(mission, output, workers=8):
    output.mkdir(parents=True, exist_ok=True)
    name = f"ESA-Mission{mission}.zip"
    target = output / name
    partial = output / (name + ".part")
    expected_size, expected_md5 = FILES[mission]
    url = f"https://zenodo.org/records/15237121/files/{name}?download=1"
    if target.exists():
        print(f"RETAIN {target}", flush=True)
        return
    start = partial.stat().st_size if partial.exists() else 0
    with requests.get(url, headers={"Range": "bytes=0-0"}, timeout=45) as response:
        response.raise_for_status()
        total = int(response.headers["Content-Range"].split("/")[-1])
    if expected_size is not None and total != expected_size:
        raise ValueError("Official archive length changed")
    chunks = output / f"mission{mission}_download_parts"
    chunks.mkdir(exist_ok=True)
    ranges = [(lo, min(lo+(4 << 20)-1, total-1)) for lo in range(start, total, 4 << 20)]
    print(f"DOWNLOAD mission{mission} offset={start} total={total} workers={workers}", flush=True)
    def piece(bounds):
        lo, hi = bounds
        path = chunks / f"{lo}-{hi}.part"
        expected = hi-lo+1
        written = path.stat().st_size if path.exists() else 0
        if written == expected:
            return path
        if written > expected:
            raise ValueError("Download segment is larger than its byte range")
        next_byte = lo+written
        with requests.get(url, headers={"Range": f"bytes={next_byte}-{hi}"}, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {next_byte}-{hi}/{total}":
                raise RuntimeError("Server returned a different byte range")
            with path.open("ab") as handle:
                for block in response.iter_content(1 << 20):
                    handle.write(block)
        if path.stat().st_size != expected:
            raise ValueError("Incomplete download segment")
        return path
    count, last, pending = start, time.monotonic(), ranges
    for attempt in range(3):
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(piece, bounds): bounds for bounds in pending}
            for completed in as_completed(futures):
                try:
                    count += completed.result().stat().st_size
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    failed.append(futures[completed])
                    print(f"mission{mission} retry segment={futures[completed][0]} reason={type(error).__name__}", flush=True)
                if time.monotonic()-last >= 30:
                    print(f"mission{mission} completed {count/1e9:.3f}/{total/1e9:.3f} GB", flush=True)
                    last = time.monotonic()
        pending = failed
        if not pending:
            break
        print(f"mission{mission} pass{attempt+1}: {len(pending)} incomplete segments remain", flush=True)
    if pending:
        raise RuntimeError(f"{len(pending)} download segments remain incomplete; completed segments are retained")
    with partial.open("ab") as handle:
        for lo, hi in ranges:
            with (chunks / f"{lo}-{hi}.part").open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    handle.write(block)
    if partial.stat().st_size != total:
        raise ValueError("Downloaded archive length differs from the server's length")
    with partial.open("rb") as handle:
        digest = hashlib.file_digest(handle, "md5").hexdigest()
    if digest != expected_md5:
        raise ValueError("Downloaded archive MD5 differs from the official record")
    partial.rename(target)
    for lo, hi in ranges:
        (chunks / f"{lo}-{hi}.part").unlink()
    chunks.rmdir()
    record = {"record": "https://zenodo.org/records/15237121", "version": "v2", "release_date": "2025-04-17",
              "url": url, "bytes": target.stat().st_size, "md5": digest}
    (output / f"mission{mission}_source.json").write_text(json.dumps(record, indent=2))
    print(f"COMPLETE mission{mission} bytes={target.stat().st_size} md5={digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, default=ROOT / "data/new_cohorts/esa-ad-v2")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for mission in args.missions:
        fetch(mission, args.output, args.workers)
