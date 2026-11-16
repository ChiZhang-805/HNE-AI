"""Compare structured future-family experiments on the original common origins."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_development import fold_auc, bootstrap_aucs


def main():
    output = ROOT / "results/family_target_comparison"
    output.mkdir(exist_ok=True)
    rows, differences = [], []
    methods = {"HNE-family": "cardinality_development", "HNE-cardinality": "cardinality_development",
               **{"HNE-fg-"+m: "family_graph_development" for m in ["independent", "constant-coupling", "conditional-graph"]},
               **{m: "boundary_development" for m in ["HNE-boundary", "HNE-history-boundary", "HNE-fg-boundary"]},
               **{m: "rhythm_coordinates" for m in ["HNE-rhythm-H", "HNE-rhythm", "History-HGB-rhythm"]}}
    controls = ["DNB", "FCM-ensemble", "CATCH-ensemble", "History-LR", "History-HGB", "HNE-HGB", "JAPE-event", "JAPE-event-AUC"]
    for dataset in ["RESILIENT", "BrisT1D-Open"]:
        frame = pd.read_csv(ROOT / f"results/unified_daily/summary/{dataset}_predictions.csv", dtype={"person_id": str})
        frame = frame[frame.primary_common].copy()
        for method, folder in methods.items():
            prediction = pd.read_csv(ROOT / "results" / folder / dataset / method / "held_person.csv", dtype={"person_id": str})
            frame = frame.merge(prediction[["person_id", "origin_day", "score"]].rename(columns={"score": method}),
                                on=["person_id", "origin_day"], validate="one_to_one")
        frame = frame.copy()
        frame["HNE-two-level"] = .5*(frame["HNE-additive"]+frame["HNE-fg-conditional-graph"])
        names = [*methods, "HNE-two-level"]
        ids, inverse = np.unique(frame.person_id, return_inverse=True)
        weights = np.random.default_rng(20261110).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=2000)
        scores = {m: fold_auc(frame, m) for m in ["HNE-additive", *names, *controls]}
        boot = {m: bootstrap_aucs(frame, m, inverse, weights) for m in scores}
        best = max(controls, key=scores.get)
        strongest_boot = np.max(np.stack([boot[m] for m in controls]), axis=0)
        for method in ["HNE-additive", *names]:
            low, high = np.nanquantile(boot[method], [.025, .975])
            within = [roc_auc_score(g.y, g[method]) for _, g in frame.groupby("person_id") if g.y.nunique() == 2]
            rows.append({"dataset": dataset, "method": method, "origins": len(frame), "auroc": scores[method],
                         "ci_low": low, "ci_high": high, "within_person_auroc": float(np.mean(within))})
            targets = ["HNE-additive", "HNE-fg-independent", "HNE-fg-constant-coupling", "strongest-control"]
            if method == "HNE-rhythm":
                targets += ["HNE-rhythm-H", "History-HGB-rhythm"]
            for baseline in targets:
                if baseline == method:
                    continue
                comparator = strongest_boot if baseline == "strongest-control" else boot[baseline]
                point = scores[best] if baseline == "strongest-control" else scores[baseline]
                low, high = np.nanquantile(boot[method]-comparator, [.025, .975])
                differences.append({"dataset": dataset, "method": method, "baseline": baseline,
                                    "difference": scores[method]-point, "ci_low": low, "ci_high": high,
                                    "strongest_control": best if baseline == "strongest-control" else ""})
        frame.to_csv(output / f"{dataset}_predictions.csv", index=False)
        np.savez_compressed(output / f"{dataset}_bootstrap.npz", **boot)
        print(json.dumps({"dataset": dataset, "auroc": {m: scores[m] for m in ["HNE-additive", *names]}}), flush=True)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "paired_differences.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
