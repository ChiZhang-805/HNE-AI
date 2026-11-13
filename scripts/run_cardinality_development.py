"""Family-target supervision using existing HNE descriptors and participant folds."""
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
from src.hne_cardinality import CardinalityWarning
from src.hne_family import H_COLUMNS, E_COLUMNS
from src.hne_features import pool_nodes

OUTPUT = ROOT / "results/cardinality_development"
FEATURES = ROOT / "data/features/cardinality_development"


def build(dataset):
    source = ROOT / "data/features/unified_daily" / dataset
    meta = pd.read_csv(source.with_suffix(".csv"), dtype={"person_id": str})
    target = FEATURES / f"{dataset}.npz"
    if target.exists():
        return meta
    FEATURES.mkdir(parents=True, exist_ok=True)
    with np.load(source.with_suffix(".npz")) as z:
        nodes, pooled, groups = z["nodes"], z["pooled"], z["groups"]
    first = str(meta.person_id.iloc[0])
    with np.load(ROOT / "data/daily_study" / dataset / f"{first}_daily.npz") as z:
        state_families = z["state_families"].copy()
        names = z["family_names"].copy()
    x = np.repeat(pooled[:, None, :], len(names), axis=1)
    for f, name in enumerate(names):
        family_nodes = nodes[:, state_families == name]
        for i in range(len(meta)):
            x[i, f, :96] = pool_nodes(family_nodes[i][:, H_COLUMNS])
            x[i, f, 132:236] = pool_nodes(family_nodes[i][:, E_COLUMNS])
    family_y = np.full((len(meta), len(names)), -1, dtype=np.int8)
    eligible = np.zeros_like(family_y, dtype=bool)
    for person, part in meta.groupby("person_id", sort=False):
        with np.load(ROOT / "data/daily_study" / dataset / f"{person}_daily.npz") as z:
            if not np.array_equal(names, z["family_names"]) or not np.array_equal(state_families, z["state_families"]):
                raise ValueError("Family coordinates change within a dataset")
            family_y[part.index] = z["block_family_status"][part.future_end_day.to_numpy(int)]
            eligible[part.index] = z["reference_eligible_families"]
    np.savez_compressed(target, x=x, family_y=family_y, eligible=eligible,
                        y=meta.y.to_numpy(), groups=groups, family_names=names)
    print(json.dumps({"dataset": dataset, "family_features": list(x.shape), "family_names": names.tolist()}), flush=True)
    return meta


def run(dataset, config):
    meta = build(dataset)
    with np.load(FEATURES / f"{dataset}.npz") as z:
        x, yf, eligible, y, groups, names = [z[k] for k in ["x", "family_y", "eligible", "y", "groups", "family_names"]]
    records = []
    for global_weight in config["global_loss_weights"]:
        method = "HNE-family" if global_weight == 0 else "HNE-cardinality"
        for fold in range(5):
            out = OUTPUT / dataset / method / f"fold{fold}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "metrics.json").exists():
                records.append(json.loads((out / "metrics.json").read_text()))
                continue
            train = ~meta.fold.isin([fold, (fold+1)%5]).to_numpy()
            val = (meta.fold == (fold+1)%5).to_numpy()
            held = (meta.fold == fold).to_numpy()
            best_score, best, selected, trials = -np.inf, None, None, []
            for c in config["C"]:
                model = CardinalityWarning(c, global_weight, groups, config["maximum_iterations"]).fit(
                    x[train], yf[train], eligible[train], y[train], meta.person_id.to_numpy()[train])
                probability, _ = model.predict(x[val], eligible[val])
                score = roc_auc_score(y[val], probability)
                trials.append({"C": c, "validation_auroc": float(score), **model.optimization})
                print(f"{dataset} {method} fold{fold} C{c} val={score:.6f} iterations={model.optimization['iterations']}", flush=True)
                if score > best_score:
                    best_score, best, selected = score, model, c
            joblib.dump(best, out / "model.joblib")
            probability, families = best.predict(x[held], eligible[held])
            prediction = meta.loc[held].copy()
            prediction["score"] = probability
            prediction.to_csv(out / "held_person.csv", index=False)
            family_predictions = np.column_stack([yf[held].ravel(), families.ravel(), eligible[held].ravel()])
            np.savez_compressed(out / "family_predictions.npz", prediction=family_predictions,
                                family_names=names, origins=np.flatnonzero(held))
            record = {"dataset": dataset, "method": method, "fold": fold, "C": selected,
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
    config = json.loads((ROOT / "configs/family_cardinality.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with threadpool_limits(limits=2):
        records = [record for dataset in config["datasets"] for record in run(dataset, config)]
    pd.DataFrame([{k: v for k, v in r.items() if k != "trials"} for r in records]).to_csv(OUTPUT / "metrics.csv", index=False)
