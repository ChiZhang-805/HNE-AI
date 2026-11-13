"""One fixed training-rank HNE variant across development and College."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_rank_basis import RankBasisWarning
from src.hne_additive import validation_score
from scripts.run_transfer import sha, write_json

OUTPUT = ROOT / "results/rank_basis"
COLLEGE = ROOT / "results/college_rank_basis"


def load(path):
    frame = pd.read_csv(path.with_suffix(".csv"), dtype={"person_id": str})
    with np.load(path.with_suffix(".npz")) as saved:
        return frame, saved["pooled"], saved["y"], saved["groups"]


def fit_one(train, validation, output, cfg, held=None):
    if (output / "selection.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    frame, x, y, groups = train
    vf, vx, vy, _ = validation
    best, best_score, selected = None, -np.inf, None
    trials = []
    for c in cfg["C_candidates"]:
        model = RankBasisWarning(C=c, groups=groups).fit(x, y, frame.person_id.to_numpy())
        value = validation_score(vy, model.predict_proba(vx))
        trials.append({"C": c, "validation_auroc": value})
        if value > best_score:
            best, best_score, selected = model, value, c
    joblib.dump({"model": best, "feature_spec": {"kind": "pooled", "columns": list(range(296))}}, output / "model.joblib")
    record = {"method": cfg["method"], "C": selected, "trials": trials,
              "config": cfg, "model_sha256": sha(output / "model.joblib"),
              "core": {name: sha(ROOT / "src" / name) for name in ("hne_rank_basis.py", "hne_additive.py")},
              "train_people": sorted(frame.person_id.unique()),
              "validation_people": sorted(vf.person_id.unique())}
    for part, data in (("validation", validation), ("held_development", held)):
        if data is None:
            continue
        table, features, labels, _ = data
        table = table.copy()
        table["score"] = best.predict_proba(features)
        table.to_csv(output / f"{part}.csv", index=False)
        record[part] = {"origins": len(labels), "auroc": roc_auc_score(labels, table.score),
                        "ap": average_precision_score(labels, table.score)}
    record["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "selection.json", record)
    write_json(output / "freeze_manifest.json", {"selection_sha256": sha(output / "selection.json"),
                                                 "model_sha256": record["model_sha256"]})
    print(json.dumps({"fit": str(output.relative_to(ROOT)), "C": selected,
                      "validation": record["validation"], "held_development": record.get("held_development")}), flush=True)


def fit(cfg):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "configuration.json", cfg)
    for dataset in cfg["development"]:
        parent = "expanded_development" if dataset == "IFH-Affect" else "unified_daily"
        data = load(ROOT / "data/features" / parent / dataset)
        frame, x, y, groups = data
        for fold in range(5):
            masks = (~frame.fold.isin([fold, (fold+1)%5]), frame.fold.eq((fold+1)%5), frame.fold.eq(fold))
            parts = [(frame.loc[m].copy(), x[m], y[m], groups) for m in masks]
            fit_one(parts[0], parts[1], OUTPUT / dataset / f"fold{fold}", cfg, held=parts[2])
    train, validation = [load(ROOT / "data/features/college_experience" / part) for part in ("train", "validation")]
    fit_one(train, validation, COLLEGE, cfg)
    summary = []
    for dataset in cfg["development"]:
        table = pd.concat([pd.read_csv(OUTPUT / dataset / f"fold{k}/held_development.csv", dtype={"person_id": str}) for k in range(5)], ignore_index=True)
        table.to_csv(OUTPUT / dataset / "held_development.csv", index=False)
        auc, pairs = [], []
        for _, group in table.groupby("fold"):
            if group.y.nunique() == 2:
                auc.append(roc_auc_score(group.y, group.score))
                pairs.append(int(group.y.sum())*int(group.y.eq(0).sum()))
        summary.append({"dataset": dataset, "origins": len(table),
                        "fold_pair_weighted_auroc": np.average(auc, weights=pairs),
                        "pooled_auroc": roc_auc_score(table.y, table.score)})
    pd.DataFrame(summary).to_csv(OUTPUT / "development_metrics.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def predict(cfg):
    json.loads((ROOT / "results/college_experience/frozen_models.json").read_text())
    selection = json.loads((COLLEGE / "selection.json").read_text())
    frozen = json.loads((COLLEGE / "freeze_manifest.json").read_text())
    if sha(COLLEGE / "selection.json") != frozen["selection_sha256"] or sha(COLLEGE / "model.joblib") != frozen["model_sha256"]:
        raise ValueError("Frozen rank model changed")
    if selection["config"] != cfg or any(sha(ROOT / "src" / name) != digest for name, digest in selection["core"].items()):
        raise ValueError("Rank model definition changed")
    frame, features, _, _ = load(ROOT / "data/features/college_experience/test")
    model = joblib.load(COLLEGE / "model.joblib")["model"]
    frame["score"] = model.predict_proba(features)
    frame.to_csv(COLLEGE / "test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "predict"), default="fit")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/rank_basis.json").read_text())
    with threadpool_limits(limits=2):
        (fit if args.phase == "fit" else predict)(cfg)
