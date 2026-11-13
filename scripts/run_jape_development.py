"""Same released JAPE event adapter on the original two development cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import vendor_jape
from scripts.run_jape_event import targets, write_scores
from scripts.run_transfer import sha, write_json


def run(args):
    cfg = json.loads((ROOT / "configs/jape_event.json").read_text())
    cfg.update(version="jape-shared-development-event-v1-20261110",
               partition="same existing5 participant folds; next fold validation, remaining3 training; previously exposed development cohorts")
    for dataset in args.datasets:
        feature = ROOT / "data/features/unified_daily" / dataset
        with np.load(feature.with_suffix(".npz")) as z:
            windows, y, folds = z["windows"], z["y"], z["fold"]
        meta = pd.read_csv(feature.with_suffix(".csv"), dtype={"person_id": str})
        gathered = []
        for fold in range(5):
            ids = {"train": np.flatnonzero((folds != fold) & (folds != (fold+1)%5)),
                   "validation": np.flatnonzero(folds == (fold+1)%5),
                   "held_person": np.flatnonzero(folds == fold)}
            tr, va, he = (ids[name] for name in ("train", "validation", "held_person"))
            folder = args.output / dataset / f"fold{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            identity = {"configuration": cfg, "dataset": dataset, "fold": fold,
                "source_feature_sha256": sha(feature.with_suffix(".npz")),
                "adapter_sha256": sha(ROOT / "src/vendor_jape.py")}
            path = folder / "fit_config.json"
            if path.exists() and json.loads(path.read_text()) != identity:
                raise ValueError("JAPE development definition changed")
            write_json(path, identity)
            if (folder / "selection.json").exists():
                gathered.append(pd.read_csv(folder / "held_person.csv", dtype={"person_id": str}))
                continue
            train_meta = meta.iloc[tr].reset_index(drop=True)
            val_meta = meta.iloc[va].reset_index(drop=True)
            future = targets(train_meta, ROOT / "data/daily_study", dataset, cfg["pred_len"])
            vfuture = targets(val_meta, ROOT / "data/daily_study", dataset, cfg["pred_len"])
            val_scores, held_scores, counts = [], [], []
            for seed in cfg["seeds"]:
                dest = folder / f"seed{seed}"
                if (dest / "metadata.json").exists():
                    info = json.loads((dest / "metadata.json").read_text())
                    with np.load(dest / "validation_predictions.npz") as z:
                        probability = z["probability"]
                else:
                    print(f"START {dataset} fold{fold} JAPE seed{seed}", flush=True)
                    probability, info = vendor_jape.fit(windows[tr], future, y[tr], windows[va], vfuture, y[va],
                                                        dest, cfg, seed, args.device)
                val_scores.append(probability)
                write_scores(dest, "validation", val_meta, probability)
                held = vendor_jape.predict(windows[he], dest, args.device)
                held_scores.append(held)
                write_scores(dest, "held_person", meta.iloc[he], held)
                counts.append({"seed": seed, "forecast_epochs": info["forecast"]["epochs_run"],
                               "head_epochs": info["head"]["epochs_run"]})
            vp, hp = np.mean(val_scores, axis=0), np.mean(held_scores, axis=0)
            write_scores(folder, "validation", val_meta, vp)
            write_scores(folder, "held_person", meta.iloc[he], hp)
            selection = {"identity": identity, "seeds": cfg["seeds"], "training": counts,
                         "validation_auroc": float(roc_auc_score(y[va], vp)),
                         "held_person_auroc": float(roc_auc_score(y[he], hp)),
                         "threshold_val_negative_p90": float(np.quantile(vp[y[va] == 0], .9))}
            write_json(folder / "selection.json", selection)
            gathered.append(pd.read_csv(folder / "held_person.csv", dtype={"person_id": str}))
            print(json.dumps({"complete": dataset, "fold": fold, "held_person_auroc": selection["held_person_auroc"]}), flush=True)
        table = pd.concat(gathered, ignore_index=True)
        table.to_csv(args.output / dataset / "held_person.csv", index=False)
        pairs, numerator = 0, 0.
        for _, block in table.groupby("fold"):
            positives = int(block.y.sum()); pair_count = positives*(len(block)-positives)
            numerator += roc_auc_score(block.y, block.score)*pair_count
            pairs += pair_count
        summary = {"origins": len(table), "people": int(table.person_id.nunique()),
                   "positives": int(table.y.sum()), "fold_pair_auroc": float(numerator/pairs),
                   "pooled_auroc": float(roc_auc_score(table.y, table.score)),
                   "ap": float(average_precision_score(table.y, table.score))}
        write_json(args.output / dataset / "summary.json", summary)
        print(json.dumps({"dataset": dataset, **summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["RESILIENT", "BrisT1D-Open"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/jape_development")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)
