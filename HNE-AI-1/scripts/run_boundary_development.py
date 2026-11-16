"""Evaluate personal-boundary history coordinates under unchanged future labels."""
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_boundary import boundary_coordinates
from src.hne_features import pool_nodes
from src.hne_additive import AdditiveWarning
from src.hne_family_graph import FamilyGraphWarning

FEATURES = ROOT / "data/features/boundary_development"
OUTPUT = ROOT / "results/boundary_development"


def build(dataset, cfg):
    meta = pd.read_csv(ROOT / f"data/features/unified_daily/{dataset}.csv", dtype={"person_id": str})
    destination = FEATURES / f"{dataset}.npz"
    if destination.exists():
        return meta
    FEATURES.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / f"data/features/family_graph/{dataset}.npz") as z:
        names = z["family_names"]
    boundary = np.full((len(meta), len(names), 20), np.nan, dtype=np.float32)
    for person, part in meta.groupby("person_id", sort=False):
        with np.load(ROOT / f"data/daily_study/{dataset}/{person}_daily.npz") as z:
            boundary[part.index] = boundary_coordinates(z, part.origin_day.to_numpy(), cfg["past_block_end_lags"])
    pooled = np.asarray([pool_nodes(row) for row in boundary])
    np.savez_compressed(destination, family=boundary, pooled=pooled)
    print(json.dumps({"dataset": dataset, "boundary_shape": list(boundary.shape)}), flush=True)
    return meta


def run(dataset, cfg):
    meta = build(dataset, cfg)
    with np.load(FEATURES / f"{dataset}.npz") as z:
        boundary, boundary_pooled = z["family"], z["pooled"]
    with np.load(ROOT / f"data/features/unified_daily/{dataset}.npz") as z:
        pooled = np.column_stack((z["pooled"], boundary_pooled))
        groups = np.r_[z["groups"], np.asarray(["H"]*boundary_pooled.shape[1])]
    with np.load(ROOT / f"data/features/family_graph/{dataset}.npz") as z:
        unary = np.concatenate((z["unary"], boundary), axis=2)
        pair, yf, eligible = z["pair"], z["family_y"], z["eligible"]
    y, people = meta.y.to_numpy(), meta.person_id.to_numpy()
    records = []
    for method in cfg["methods"]:
        graph = method == "HNE-fg-boundary"
        columns = groups == "H" if method == "HNE-history-boundary" else np.ones(len(groups), dtype=bool)
        for fold in range(5):
            out = OUTPUT / dataset / method / f"fold{fold}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "selection.json").exists():
                records.append(json.loads((out / "selection.json").read_text()))
                continue
            train = ~meta.fold.isin([fold, (fold+1)%5]).to_numpy()
            val = (meta.fold == (fold+1)%5).to_numpy()
            held = (meta.fold == fold).to_numpy()
            best, best_score, chosen, trials = None, -np.inf, None, []
            for c in cfg["C"]:
                if graph:
                    model = FamilyGraphWarning(c, "conditional-graph", cfg["maximum_iterations"]).fit(unary[train], pair[train], yf[train], eligible[train], people[train])
                    p, _ = model.predict(unary[val], pair[val], eligible[val])
                else:
                    model = AdditiveWarning(c, groups=groups[columns]).fit(pooled[train][:, columns], y[train], people[train])
                    p = model.predict_proba(pooled[val][:, columns])
                score = roc_auc_score(y[val], p)
                trials.append({"C": c, "validation_auroc": float(score)})
                if score > best_score:
                    best, best_score, chosen = model, score, c
            joblib.dump({"model": best, "columns": columns, "graph": graph}, out / "model.joblib")
            for split, mask in [("validation", val), ("held_person", held)]:
                if graph:
                    p, _ = best.predict(unary[mask], pair[mask], eligible[mask])
                else:
                    p = best.predict_proba(pooled[mask][:, columns])
                frame = meta.loc[mask].copy()
                frame["score"] = p
                frame.to_csv(out / f"{split}.csv", index=False)
            record = {"dataset": dataset, "method": method, "fold": fold, "C": chosen,
                      "validation_auroc": float(best_score), "auroc": float(roc_auc_score(y[held], p)), "trials": trials}
            (out / "selection.json").write_text(json.dumps(record, indent=2))
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "trials"}), flush=True)
        pd.concat([pd.read_csv(OUTPUT / dataset / method / f"fold{f}/held_person.csv") for f in range(5)]).to_csv(OUTPUT / dataset / method / "held_person.csv", index=False)
    return records


if __name__ == "__main__":
    cfg = json.loads((ROOT / "configs/boundary_coordinates.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "configuration.json").write_text(json.dumps(cfg, indent=2))
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with threadpool_limits(limits=2):
        records = [r for dataset in cfg["datasets"] for r in run(dataset, cfg)]
    pd.DataFrame([{k: v for k, v in r.items() if k != "trials"} for r in records]).to_csv(OUTPUT / "metrics.csv", index=False)
