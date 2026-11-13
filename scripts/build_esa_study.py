"""Create temporal ESA HNE train/validation or frozen-test inputs."""
import argparse
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
from scripts.build_daily_study import write_person
from src.hne_representation import Sampling, encode_person, select_reference_channels
from src.dnb import score_window


def build(mission, phase, mapping, protocol, digest):
    name = f"ESA-Mission{mission}"
    bounds = mapping["split"][str(mission)]
    output = ROOT / "data/esa_study" / name
    output.mkdir(parents=True, exist_ok=True)
    if phase == "test" and not (ROOT / "results/esa_study" / name / "frozen_models.json").exists():
        raise RuntimeError("Complete target training and source selection before creating test targets")
    with np.load(ROOT / "data/new_cohorts/esa-ad-v2" / f"mission{mission}_daily.npz") as z:
        arrays = {k: z[k] for k in z.files}
    keep = arrays["dates"] < np.datetime64(bounds["validation_end_exclusive"]) if phase == "fit" else np.ones(len(arrays["dates"]), dtype=bool)
    for key in ["states", "contexts", "state_observed_counts", "context_observed_counts", "dates", "available_at"]:
        arrays[key] = arrays[key][keep]
    origins, samples, summary = write_person(output, name, name, phase, -1, arrays, protocol, digest)
    meta = pd.DataFrame(samples)
    if meta.empty:
        (output / f"{phase}_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"mission": name, "phase": phase, **summary}), flush=True)
        return
    train_end = bounds["train_end_exclusive"]
    val_end = bounds["validation_end_exclusive"]
    meta["split"] = "boundary"
    meta.loc[meta.future_end_date < train_end, "split"] = "train"
    meta.loc[(meta.issue_time >= train_end) & (meta.future_end_date < val_end), "split"] = "validation"
    meta.loc[meta.issue_time >= val_end, "split"] = "test"
    wanted = ["train", "validation"] if phase == "fit" else ["test"]
    meta = meta[meta.split.isin(wanted)].reset_index(drop=True)
    with np.load(output / f"{name}_daily.npz") as z:
        selected = select_reference_channels(z["normalized"], z["state_names"])
        states, contexts = z["normalized"][:, selected], z["normalized_contexts"]
    spec = Sampling.daily()
    for split in wanted:
        part = meta[meta.split == split].reset_index(drop=True)
        if part.empty:
            print(f"{name}/{split}: zero scorable origins", flush=True)
            continue
        cutoffs = part.origin_day.to_numpy(int)+1
        nodes, pooled, groups = encode_person(states, contexts, cutoffs, spec)
        trajectory = np.column_stack((states, contexts))
        windows = np.stack([trajectory[t-32:t] for t in cutoffs])
        dnb = np.asarray([score_window(w[:, :len(selected)], minimum=12) for w in windows])
        payload = {"nodes": nodes, "pooled": pooled, "groups": groups, "windows": windows,
                   "y": part.y.to_numpy(), "dnb_score": dnb, "selected_states": selected}
        if split != "test":
            end_date = np.datetime64(train_end if split == "train" else val_end)
            bound = int(np.searchsorted(arrays["dates"], end_date))
            valid = cutoffs+32 <= bound
            future = np.full((len(part), 32, trajectory.shape[1]), np.nan, dtype=np.float32)
            for i in np.flatnonzero(valid):
                future[i] = trajectory[cutoffs[i]:cutoffs[i]+32]
            payload.update(future=future, future_available=valid)
        np.savez_compressed(output / f"{split}.npz", **payload)
        part.to_csv(output / f"{split}.csv", index=False)
        print(json.dumps({"mission": name, "split": split, "origins": len(part), "positive": int(part.y.sum()),
                          "states": states.shape[1], "contexts": contexts.shape[1]}), flush=True)
    summary["partitions"] = {s: {"origins": len(p), "positive": int(p.y.sum())} for s, p in meta.groupby("split")}
    summary["mapping"] = mapping
    (output / f"{phase}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--phase", choices=["fit", "test"], default="fit")
    args = parser.parse_args()
    mapping = json.loads((ROOT / "configs/esa_mapping.json").read_text())
    raw = (ROOT / mapping["daily_protocol"]).read_bytes()
    protocol, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with threadpool_limits(limits=2):
        for mission in args.missions:
            build(mission, args.phase, mapping, protocol, digest)
