"""Independent College reference-units fit using the original temporal loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_temporal_external as external
from scripts.run_transfer import write_json
from scripts.summarize_development import fold_auc
from src.hne_reference_units import fit_predict

EXTRA_CORE = ("hne_reference_units.py",)


def configuration():
    cfg = external.configuration()
    cfg.update(version="college-reference-units-v1-20261110", method="HNE-reference-units",
        role="prespecified supplementary scaling comparison; primary HNE-additive remains separate",
        trajectory_scaling="state/context already in personal reference units: mean0/scale1; original clip±8,mask; node50/relation72 scalers unchanged",
        motivation="training-only global SD compression diagnostic; no endpoint or origin changes")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["fit", "predict", "development"], default="fit")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--all-methods-frozen", action="store_true")
    args = parser.parse_args()
    if args.phase == "predict" and not args.all_methods_frozen:
        parser.error("predict requires --all-methods-frozen; test files have not been opened")
    is_development = args.phase == "development"
    features = args.features or ROOT / ("data/features/unified_daily" if is_development else "data/features/college_experience")
    output = args.output or ROOT / ("results/reference_units_development" if is_development else "results/college_reference_units")
    features = features if features.is_absolute() else ROOT / features
    output = output if output.is_absolute() else ROOT / output
    cfg = configuration()
    with threadpool_limits(limits=2):
        if is_development:
            from scripts.run_temporal_development import run
            datasets = ["RESILIENT", "BrisT1D-Open"]
            cfg = {key: value for key, value in cfg.items() if key not in ("expected_nodes", "expected_window", "dataset")}
            cfg.update(version="reference-units-development-v1-20261110", datasets=datasets,
                stage="previously_exposed_development", folds=list(range(5)),
                partition="heldf,validation(f+1)%5,trainingremaining3", features=str(features))
            output.mkdir(parents=True, exist_ok=True)
            plan = output / "configuration.json"
            if plan.exists() and json.loads(plan.read_text(encoding="utf-8")) != cfg:
                raise ValueError("Existing reference-units development configuration differs")
            write_json(plan, cfg)
            run(features, output, datasets, list(range(5)), args.device, fit_function=fit_predict,
                extra_core=EXTRA_CORE, model_name="HNE-reference-units")
            comparison = []
            for dataset in datasets:
                current = pd.concat([pd.read_csv(output / dataset / "HNE-reference-units" / f"fold{fold}/held_person.csv") for fold in range(5)])
                previous = pd.concat([pd.read_csv(ROOT / "results/unified_daily" / dataset / "HNE-temporal" / f"fold{fold}/held_person.csv") for fold in range(5)])
                keys = ["person_id", "origin_day", "fold", "y"]
                aligned = current.merge(previous[keys+["probability"]], on=keys, validate="one_to_one", suffixes=("_reference_units", "_original"))
                if len(aligned) != len(current) or len(aligned) != len(previous):
                    raise ValueError("Development populations differ between the two temporal models")
                current_auc, old_auc = [fold_auc(aligned, "probability_"+name) for name in ["reference_units", "original"]]
                comparison.append(dict(dataset=dataset, origins=len(aligned), people=aligned.person_id.nunique(),
                    positive=int(aligned.y.sum()), reference_units_auroc=current_auc,
                    original_temporal_auroc=old_auc, difference=current_auc-old_auc))
            pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
            print(pd.DataFrame(comparison).to_string(index=False), flush=True)
        elif args.phase == "fit":
            output.mkdir(parents=True, exist_ok=True)
            plan = output / "configuration.json"
            if plan.exists() and json.loads(plan.read_text(encoding="utf-8")) != cfg:
                raise ValueError("Existing reference-units configuration differs")
            write_json(plan, cfg)
            external.fit(features, output, cfg, args.device, fit_function=fit_predict, extra_core=EXTRA_CORE)
            selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
            first_checkpoint = next(iter(selection["checkpoint_sha256"]))
            metadata = json.loads((output / first_checkpoint).with_name("metadata.json").read_text(encoding="utf-8"))
            write_json(output / "input_scaling.json", metadata["input_scaling"])
        else:
            external.predict(features, output, cfg, args.device, extra_core=EXTRA_CORE)


if __name__ == "__main__":
    main()
