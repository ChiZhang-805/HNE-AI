"""Compare every frozen Human Rhythms predictor on aligned test origins."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_hne_external import METHODS, STATISTICS, cluster_statistics

KEY = ["person_id", "origin_day"]
BASE = ROOT / "results/human_rhythms"
OUT = BASE / "comparison"


def read(path):
    return pd.read_csv(path, dtype={"person_id": str})


def main():
    plan = json.loads((ROOT / "configs/external_evaluation.json").read_text())
    folders = {name: BASE / name for name in METHODS}
    for method in ["FCM", "CATCH"]:
        folders[method + "-3seed"] = BASE / method / "ensemble"
        for seed in [20261110, 20261111, 20261112]:
            folders[f"{method}-s{seed}"] = BASE / method / f"seed{seed}"
            folders[f"{method}-score-LR-s{seed}"] = BASE / (method + "-score-LR") / f"seed{seed}"
    folders.update({name: BASE / name for name in ["HNE-context-grid", "MultiDomain-HNE"]})
    frame = read(ROOT / "data/features/human_rhythms/test.csv")
    compare(plan, folders, frame, OUT)


def compare(plan, folders, frame, output, extra_pairs=()):
    thresholds = {}
    for method, folder in folders.items():
        test, validation = read(folder / "test.csv"), read(folder / "validation.csv")
        frame = frame.merge(test[KEY + ["score"]].rename(columns={"score": method}),
                            on=KEY, how="left", validate="one_to_one")
        negative = validation.loc[validation.y.eq(0), "score"].to_numpy()
        thresholds[method] = float(np.quantile(negative[np.isfinite(negative)], .9))
    primary = [plan["primary_hne"]] + plan["primary_comparators"]
    common = np.isfinite(frame[primary].to_numpy()).all(1)
    frame["primary_common"] = common
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "predictions.csv", index=False)
    rows, differences, draws_saved = [], [], {}
    for population, selected in [("all", np.ones(len(frame), bool)), ("common", common)]:
        part = frame.loc[selected]
        people = part.person_id.to_numpy()
        identities = np.unique(people)
        draws = np.random.default_rng(20261110).multinomial(len(identities),
            np.full(len(identities), 1 / len(identities)), size=2000)
        points, boots = {}, {}
        for method in folders:
            point, boot, counts = cluster_statistics(part.y.to_numpy(), part[method].to_numpy(),
                                                     people, thresholds[method], draws)
            points[method], boots[method] = point, boot
            row = dict(population=population, method=method, **counts,
                       val_negative_p90=thresholds[method])
            for j, statistic in enumerate(STATISTICS):
                lo, hi = np.nanquantile(boot[:, j], [.025, .975])
                row.update({statistic: point[j], statistic + "_ci_low": lo,
                            statistic + "_ci_high": hi})
            rows.append(row)
            draws_saved[population + "__" + method] = boot
        if population == "common":
            for method in folders:
                if method == plan["primary_hne"]:
                    continue
                for j, statistic in enumerate(STATISTICS):
                    delta = boots[plan["primary_hne"]][:, j] - boots[method][:, j]
                    lo, hi = np.nanquantile(delta, [.025, .975])
                    differences.append(dict(model=plan["primary_hne"], baseline=method,
                        metric=statistic, difference=points[plan["primary_hne"]][j] - points[method][j],
                        ci_low=lo, ci_high=hi))
            controls = plan["primary_comparators"]
            point_best = max(points[m][0] for m in controls)
            boot_best = np.max(np.vstack([boots[m][:, 0] for m in controls]), axis=0)
            delta = boots[plan["primary_hne"]][:, 0] - boot_best
            lo, hi = np.nanquantile(delta, [.025, .975])
            differences.append(dict(model=plan["primary_hne"], baseline="strongest-predeclared-control",
                metric="auroc", difference=points[plan["primary_hne"]][0] - point_best, ci_low=lo, ci_high=hi))
            strongest = dict(method=max(controls, key=lambda m: points[m][0]), auroc=point_best,
                hne_auroc=points[plan["primary_hne"]][0], difference=points[plan["primary_hne"]][0] - point_best,
                ci_low=lo, ci_high=hi, bootstrap_rule="maximum across all predeclared controls in every participant-resampled draw")
            for model, baseline in extra_pairs:
                for j, statistic in enumerate(STATISTICS):
                    delta = boots[model][:, j] - boots[baseline][:, j]
                    lo, hi = np.nanquantile(delta, [.025, .975])
                    differences.append(dict(model=model, baseline=baseline, metric=statistic,
                        difference=points[model][j]-points[baseline][j], ci_low=lo, ci_high=hi))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "paired_differences.csv", index=False)
    np.savez_compressed(output / "bootstrap.npz", metric_names=np.asarray(STATISTICS), **draws_saved)
    summary = dict(total_origins=len(frame), common_origins=int(common.sum()),
        test_people=frame.person_id.nunique(), test_positive=int(frame.y.sum()),
        primary_methods=primary, strongest_control=strongest)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics[metrics.population.eq("common") & metrics.method.isin(primary)][
        ["method", "origins", "auroc", "auroc_ci_low", "auroc_ci_high", "within_person_auroc", "ap"]].to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
