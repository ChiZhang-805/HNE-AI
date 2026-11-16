"""Fit and evaluate the same family-event HNE graph on College Experience."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_family_graph import FamilyGraphWarning, family_unaries, family_edges
from src.hne_representation import Sampling, context_residuals

FEATURES = ROOT / "data/features/college_family_graph"
OUTPUT = ROOT / "results/college_family_graph"


def worker(task):
    split, person, indexes, days, names = task
    with threadpool_limits(limits=1):
        with np.load(ROOT / "data/college_study" / split / f"{person}_daily.npz") as z:
            states, context, families = z["normalized"], z["normalized_contexts"], z["state_families"]
            eligible = z["reference_eligible_families"].copy()
            targets = z["block_family_status"][days+8] if split != "test" else None
        spec = Sampling.daily()
        adjusted, _ = context_residuals(states, context, spec)
        edges = []
        for day in days:
            end = int(day)+1
            raw_parts, adjusted_parts = [], []
            for length in spec.scales:
                raw_parts.append(family_edges(states[end-length:end], states[:spec.reference], families, names))
                adjusted_parts.append(family_edges(adjusted[end-length:end], adjusted[:spec.reference], families, names))
            edges.append(np.column_stack(raw_parts+adjusted_parts))
        return indexes, np.asarray(edges, dtype=np.float32), eligible, targets


def build(split, workers):
    source = ROOT / "data/features/college_experience"
    meta = pd.read_csv(source / f"{split}.csv", dtype={"person_id": str})
    destination = FEATURES / f"{split}.npz"
    if destination.exists():
        return meta
    FEATURES.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "data/college_study" / split / f"{meta.person_id.iloc[0]}_daily.npz") as z:
        families, names = z["state_families"], z["family_names"]
    with np.load(source / f"{split}.npz") as z:
        unary = family_unaries(z["nodes"], z["pooled"], families, names)
    pair = np.full((len(meta), len(names)*(len(names)-1)//2, 72), np.nan, dtype=np.float32)
    eligible = np.zeros((len(meta), len(names)), dtype=bool)
    targets = np.full(eligible.shape, -1, dtype=np.int8)
    tasks = [(split, str(person), part.index.to_numpy(), part.origin_day.to_numpy(int), names)
             for person, part in meta.groupby("person_id", sort=False)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (indexes, edges, active, truth) in enumerate(pool.map(worker, tasks), 1):
            pair[indexes], eligible[indexes] = edges, active
            if truth is not None:
                targets[indexes] = truth
            if k % 10 == 0 or k == len(tasks):
                print(f"College family graph {split}: {k}/{len(tasks)} participants", flush=True)
    payload = dict(unary=unary, pair=pair, eligible=eligible, family_names=names)
    if split != "test":
        payload.update(family_y=targets, y=meta.y.to_numpy())
    np.savez_compressed(destination, **payload)
    return meta


def data(split):
    meta = pd.read_csv(ROOT / "data/features/college_experience" / f"{split}.csv", dtype={"person_id": str})
    with np.load(FEATURES / f"{split}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def save_predictions(folder, split, meta, model, arrays):
    probability, marginal = model.predict(arrays["unary"], arrays["pair"], arrays["eligible"])
    frame = meta.copy()
    frame["score"] = probability
    frame.to_csv(folder / f"{split}.csv", index=False)
    np.savez_compressed(folder / f"{split}_families.npz", probabilities=marginal, eligible=arrays["eligible"])
    return probability


def two_level(split, cfg):
    global_frame = pd.read_csv(ROOT / f"results/college_experience/HNE-additive/{split}.csv", dtype={"person_id": str})
    local_frame = pd.read_csv(OUTPUT / f"HNE-fg-conditional-graph/{split}.csv", dtype={"person_id": str})
    frame = global_frame.merge(local_frame[["person_id", "origin_day", "score"]].rename(columns={"score": "family_score"}),
                               on=["person_id", "origin_day"], validate="one_to_one")
    frame["score"] = cfg["two_level_weights"][0]*frame.score + cfg["two_level_weights"][1]*frame.family_score
    out = OUTPUT / "HNE-two-level"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / f"{split}.csv", index=False)
    return {"method": "HNE-two-level", "auroc": float(roc_auc_score(frame.y, frame.score)),
            "ap": float(average_precision_score(frame.y, frame.score)), "weights": cfg["two_level_weights"]}


def fit(cfg, workers):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "freeze_manifest.json").exists():
        print("REUSE completed College family graph fits", flush=True)
        return
    for split in ["train", "validation"]:
        build(split, workers)
    train_meta, train = data("train")
    val_meta, val = data("validation")
    selections = []
    for mode in cfg["methods"]:
        method = "HNE-fg-"+mode
        out = OUTPUT / method
        out.mkdir(exist_ok=True)
        if (out / "selection.json").exists():
            selections.append(json.loads((out / "selection.json").read_text()))
            continue
        trials, best, best_score, chosen = [], None, -np.inf, None
        for c in cfg["C"]:
            model = FamilyGraphWarning(c, mode, cfg["maximum_iterations"]).fit(
                train["unary"], train["pair"], train["family_y"], train["eligible"], train_meta.person_id.to_numpy())
            probability, _ = model.predict(val["unary"], val["pair"], val["eligible"])
            score = roc_auc_score(val_meta.y, probability)
            trials.append({"C": c, "validation_auroc": float(score), **model.optimization})
            print(f"College {method} C{c} validation_AUROC={score:.6f} iterations={model.optimization['iterations']}", flush=True)
            if score > best_score:
                best, best_score, chosen = model, score, c
        joblib.dump(best, out / "model.joblib")
        p = save_predictions(out, "validation", val_meta, best, val)
        selected = {"method": method, "C": chosen, "validation_auroc": float(best_score), "trials": trials,
                    "threshold": float(np.quantile(p[val_meta.y.to_numpy() == 0], .9))}
        (out / "selection.json").write_text(json.dumps(selected, indent=2))
        selections.append(selected)
    combined = two_level("validation", cfg)
    (OUTPUT / "HNE-two-level/selection.json").write_text(json.dumps(combined, indent=2))
    record = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": cfg,
              "selections": selections, "two_level": combined, "test_constructed": False}
    (OUTPUT / "freeze_manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(combined), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["fit", "predict"], default="fit")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/college_family_graph.json").read_text())
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with threadpool_limits(limits=2):
        if args.phase == "fit":
            fit(cfg, args.workers)
        else:
            if not (ROOT / "results/college_experience/frozen_models.json").exists():
                raise RuntimeError("Complete the College global freeze before test prediction")
            build("test", args.workers)
            meta, arrays = data("test")
            for mode in cfg["methods"]:
                out = OUTPUT / ("HNE-fg-"+mode)
                save_predictions(out, "test", meta, joblib.load(out / "model.joblib"), arrays)
            print(json.dumps(two_level("test", cfg)), flush=True)
