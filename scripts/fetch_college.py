"""Download the two public College Experience v5 tables and fix ID partitions."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/new_cohorts/college-experience"
API = "https://www.kaggle.com/api/v1/datasets/download/subigyanepal/college-experience-dataset/"


def main():
    folder = DESTINATION / "Sensing"
    folder.mkdir(parents=True, exist_ok=True)
    identities, records = set(), []
    for name, expected in (("sensing.csv", 521959313), ("steps.csv", 22812153)):
        target = folder / name
        url = API + "Sensing%2F" + name + "?datasetVersionNumber=5"
        if not target.exists():
            partial = target.with_suffix(".csv.part")
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(1 << 20):
                        handle.write(block)
            if partial.stat().st_size != expected:
                raise ValueError(f"Public v5 file size differs for {name}")
            partial.replace(target)
        for chunk in pd.read_csv(target, usecols=["uid"], dtype={"uid": str}, chunksize=100000):
            identities.update(chunk.uid.dropna())
        records.append({"name": "Sensing/" + name, "source": url,
                        "bytes": target.stat().st_size, "expected_bytes": expected})
        print(f"AVAILABLE {name} bytes={target.stat().st_size}", flush=True)
    ranked = sorted(identities, key=lambda p: hashlib.sha256(("20261110:College-Experience:"+p).encode()).hexdigest())
    n_train, n_validation = round(.6*len(ranked)), round(.2*len(ranked))
    split = pd.DataFrame({"person_id": ranked, "split": [
        "train" if i < n_train else "validation" if i < n_train+n_validation else "test"
        for i in range(len(ranked))]})
    path = DESTINATION / "subject_split.csv"
    if path.exists():
        existing = pd.read_csv(path, dtype={"person_id": str}).set_index("person_id").split
        if not split.set_index("person_id").split.sort_index().equals(existing.sort_index()):
            raise ValueError("Existing participant assignment differs from the fixed rule")
    else:
        split.to_csv(path, index=False)
    (DESTINATION / "download_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"FIXED_SPLIT people={len(split)} train={n_train} validation={n_validation} test={len(split)-n_train-n_validation}")


if __name__ == "__main__":
    main()
