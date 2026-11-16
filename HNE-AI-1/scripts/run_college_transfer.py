"""Repeat the fixed source-prior sample-efficiency comparison on College."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_human_transfer as shared
from scripts.run_college_models import configuration as external_configuration


def configuration():
    cfg = shared.configuration()
    cfg.update(version="college-experience-sample-efficiency-v1-20261110",
        expected_training_people=126, expected_subset_people={"25":32,"50":63,"100":126},
        subset="sort train person_id by sha256(f'{seed}:{person_id}'); take ceil(fraction*126); no label stratification or resampling",
        source="same frozen RESILIENT/BrisT1D Humans HNE-additive source used for Human Rhythms; no new source fitting",
        full_target_only="reuse College HNE-additive train-fitted selected model and scores without another fit",
        purpose="RQ3: repeat shared-rule transfer across wearable physiological records and phone-based behavioral disturbance records",
        validation_people=44)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit","predict"), default="fit")
    parser.add_argument("--all-methods-frozen", action="store_true")
    args = parser.parse_args()
    features, output = ROOT / "data/features/college_experience", ROOT / "results/college_transfer"
    with threadpool_limits(limits=2):
        if args.phase == "fit":
            shared.fit(features, output, cfg=configuration(), external_cfg=external_configuration(),
                baseline_dir=ROOT / "results/college_experience/HNE-additive")
        else:
            if not args.all_methods_frozen:
                parser.error("predict requires --all-methods-frozen")
            shared.predict(features, output, cfg=configuration(), external_cfg=external_configuration())


if __name__ == "__main__":
    main()
