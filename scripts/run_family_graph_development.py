"""Shared future family graph, with independent and constant-coupling controls."""
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
from src.hne_family_graph import FamilyGraphWarning, family_edges
from src.hne_representation import Sampling, context_residuals
from scripts.run_cardinality_development import build as build_family, FEATURES as FAMILY_FEATURES

FEATURES = ROOT / "data/features/family_graph"
OUTPUT = ROOT / "results/family_graph_development"


def build(dataset):
    meta = build_family(dataset)
    target = FEATURES / f"{dataset}.npz"
    if target.exists():
        return meta
    FEATURES.mkdir(parents=True, exist_ok=True)
    with np.load(FAMILY_FEATURES / f"{dataset}.npz") as z:
        base = {key: z[key] for key in ["x", "family_y", "eligible", "y", "groups", "family_names"]}
    names = base["family_names"]
    pairs = len(names)*(len(names)-1)//2
    edges = np.full((len(meta), pairs, 72), np.nan, dtype=np.float32)
    spec = Sampling.daily()
    for person, part in meta.groupby("person_id", sort=False):
        with np.load(ROOT / "data/daily_study" / dataset / f"{person}_daily.npz") as z:
            states, context, families = z["normalized"], z["normalized_contexts"], z["state_families"]
        adjusted, _ = context_residuals(states, context, spec)
        for row in part.itertuples():
            end = int(row.origin_day)+1
            raw_parts, adjusted_parts = [], []
            for length in spec.scales:
                raw_parts.append(family_edges(states[end-length:end], states[:spec.reference], families, names))
                adjusted_parts.append(family_edges(adjusted[end-length:end], adjusted[:spec.reference], families, names))
            edges[row.Index] = np.column_stack(raw_parts+adjusted_parts)
    np.savez_compressed(target, unary=base["x"][:, :, np.r_[0:96,132:260]], pair=edges,
                        family_y=base["family_y"], eligible=base["eligible"], y=base["y"], family_names=names)
    print(json.dumps({"dataset": dataset, "edge_shape": list(edges.shape)}), flush=True)
    return meta


def run(dataset, cfg):
    meta = build(dataset)
    with np.load(FEATURES / f"{dataset}.npz") as z:
        unary, pair, yf, eligible, y, names = [z[k] for k in ["unary", "pair", "family_y", "eligible", "y", "family_names"]]
    records = []
    for mode in cfg["methods"]:
        method = "HNE-fg-"+mode
        for fold in range(5):
            out = OUTPUT / dataset / method / f"fold{fold}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "metrics.json").exists():
                records.append(json.loads((out / "metrics.json").read_text()))
                continue
            train = ~meta.fold.isin([fold, (fold+1)%5]).to_numpy()
            val = (meta.fold == (fold+1)%5).to_numpy()
            held = (meta.fold == fold).to_numpy()
            best_score, best, chosen, trials = -np.inf, None, None, []
            for c in cfg["C"]:
                model = FamilyGraphWarning(c, mode, cfg["maximum_iterations"]).fit(
                    unary[train], pair[train], yf[train], eligible[train], meta.person_id.to_numpy()[train])
                probability, _ = model.predict(unary[val], pair[val], eligible[val])
                score = roc_auc_score(y[val], probability)
                trials.append({"C": c, "validation_auroc": float(score), **model.optimization})
                print(f"{dataset} {mode} fold{fold} C{c} val={score:.6f} iterations={model.optimization['iterations']}", flush=True)
                if score > best_score:
                    best_score, best, chosen = score, model, c
            joblib.dump(best, out / "model.joblib")
            for split, mask in [("validation", val), ("held_person", held)]:
                probability, marginal = best.predict(unary[mask], pair[mask], eligible[mask])
                frame = meta.loc[mask].copy()
                frame["score"] = probability
                frame.to_csv(out / f"{split}.csv", index=False)
                np.savez_compressed(out / f"{split}_families.npz", predictions=marginal,
                                    targets=yf[mask], eligible=eligible[mask], family_names=names)
            record = {"dataset": dataset, "method": method, "fold": fold, "C": chosen,
                      "validation_auroc": float(best_score), "auroc": float(roc_auc_score(y[held], probability)),
                      "ap": float(average_precision_score(y[held], probability)), "origins": int(held.sum()),
                      "positive": int(y[held].sum()), "trials": trials}
            (out / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "trials"}), flush=True)
        parts = [pd.read_csv(OUTPUT / dataset / method / f"fold{f}/held_person.csv") for f in range(5)]
        pd.concat(parts, ignore_index=True).to_csv(OUTPUT / dataset / method / "held_person.csv", index=False)
    return records


if __name__ == "__main__":
    cfg = json.loads((ROOT / "configs/family_graph.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "configuration.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with threadpool_limits(limits=2):
        records = [record for dataset in cfg["datasets"] for record in run(dataset, cfg)]
    pd.DataFrame([{k: v for k, v in r.items() if k != "trials"} for r in records]).to_csv(OUTPUT / "metrics.csv", index=False)
