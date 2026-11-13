"""College primary and prespecified auxiliary contrasts on the frozen test set."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_external import compare, read
from scripts.run_hne_external import METHODS


def main():
    base = ROOT / "results/college_experience"
    plan = json.loads((ROOT / "configs/college_extended_evaluation.json").read_text())
    folders = {name: base / name for name in METHODS}
    for method in ("FCM", "CATCH"):
        folders[f"{method}-3seed"] = base / method / "ensemble"
        for seed in (20261110, 20261111, 20261112):
            folders[f"{method}-s{seed}"] = base / method / f"seed{seed}"
            folders[f"{method}-score-LR-s{seed}"] = base / f"{method}-score-LR" / f"seed{seed}"
    folders["HNE-temporal"] = ROOT / "results/college_temporal"
    folders["HNE-reference-units"] = ROOT / "results/college_reference_units"
    folders["HNE-rank-basis"] = ROOT / "results/college_rank_basis"
    folders["JAPE-event-3seed"] = ROOT / "results/college_jape"
    folders["JAPE-event-AUC-3seed"] = ROOT / "results/college_jape_auc"
    for mode in ("independent", "constant-coupling", "conditional-graph"):
        name = "HNE-fg-"+mode
        folders[name] = ROOT / "results/college_family_graph" / name
    folders["HNE-two-level"] = ROOT / "results/college_family_graph/HNE-two-level"
    for name in ("HNE-rhythm-H", "HNE-rhythm", "History-HGB-rhythm"):
        folders[name] = ROOT / "results/college_rhythm" / name
    for seed in (20261110, 20261111, 20261112):
        folders[f"JAPE-event-s{seed}"] = ROOT / "results/college_jape" / f"seed{seed}"
    for name in ("future-values", "future-moments"):
        folders[f"HNE-{name}"] = ROOT / "results/college_future_objective" / name
    contrasts = [("HNE-future-moments", "HNE-future-values"),
                 ("HNE-future-moments", "HNE-temporal"),
                 ("HNE-future-values", "HNE-temporal"),
                 ("HNE-reference-units", "HNE-temporal"),
                 ("HNE-rank-basis", "HNE-additive"),
                 ("JAPE-event-AUC-3seed", "JAPE-event-3seed"),
                 ("HNE-two-level", "HNE-additive"),
                 ("HNE-fg-conditional-graph", "HNE-fg-independent"),
                 ("HNE-fg-conditional-graph", "HNE-fg-constant-coupling"),
                 ("HNE-rhythm", "HNE-additive"), ("HNE-rhythm", "HNE-rhythm-H"),
                 ("HNE-rhythm", "History-HGB-rhythm")]
    frame = read(ROOT / "data/features/college_experience/test.csv")
    compare(plan, folders, frame, base / "comparison", extra_pairs=contrasts)
    original = json.loads((ROOT / "configs/college_evaluation.json").read_text())
    output = base / "comparison"
    metrics = pd.read_csv(output / "metrics.csv").query("population == 'common'").set_index("method")
    controls = original["primary_comparators"]
    with np.load(output / "bootstrap.npz") as z:
        delta = z["common__" + original["primary_hne"]][:, 0] - np.max(
            np.stack([z["common__" + name][:, 0] for name in controls]), axis=0)
    best = metrics.loc[controls, "auroc"].idxmax()
    summary = {"controls": controls, "strongest_control": best,
        "difference": float(metrics.loc[original["primary_hne"], "auroc"] - metrics.loc[best, "auroc"]),
        "ci_low": float(np.nanquantile(delta, .025)), "ci_high": float(np.nanquantile(delta, .975)),
        "population": "same complete-scoring JAPE extension common origins"}
    (output / "original_controls.json").write_text(json.dumps(summary, indent=2))
    controls = plan["primary_comparators"]
    with np.load(output / "bootstrap.npz") as z:
        delta = z["common__HNE-two-level"][:, 0] - np.max(
            np.stack([z["common__" + name][:, 0] for name in controls]), axis=0)
    best = metrics.loc[controls, "auroc"].idxmax()
    result = {"model": "HNE-two-level", "comparison": "supplementary fixed condition", "strongest_control": best,
              "difference": float(metrics.loc["HNE-two-level", "auroc"]-metrics.loc[best, "auroc"]),
              "ci_low": float(np.nanquantile(delta, .025)), "ci_high": float(np.nanquantile(delta, .975))}
    (output / "two_level_controls.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
