"""College Experience wrapper for the existing daily HNE and FCM/CATCH runners.

Default fit reads train/validation only. The coordinator opens predict after
all planned methods are frozen; no endpoint or model formula is changed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_hne_external as hne
from scripts import run_public_external as public


def configuration():
    cfg = hne.configuration()
    overrides = json.loads((ROOT / "configs/college_evaluation.json").read_text(encoding="utf-8"))
    # Preserve the native algorithm parameters from the existing runner; only
    # its cohort-facing input description changes.
    mmdev = {**cfg["mmdev"], **overrides.pop("mmdev")}
    cfg.update(overrides)
    cfg["mmdev"] = mmdev
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["fit", "predict"], default="fit")
    parser.add_argument("--suite", choices=["hne", "public", "all"], default="all")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--all-methods-frozen", action="store_true",
                        help="The coordinator has frozen every planned method before opening testing.")
    args = parser.parse_args()
    if args.phase == "predict" and not args.all_methods_frozen:
        parser.error("predict requires --all-methods-frozen; test files have not been opened")
    cfg = configuration()
    features = args.features or Path(cfg["feature_directory"])
    output = args.output or Path(cfg["output_directory"])
    features = features if features.is_absolute() else ROOT / features
    output = output if output.is_absolute() else ROOT / output
    with threadpool_limits(limits=2):
        if args.suite in ["hne", "all"]:
            function = hne.fit if args.phase == "fit" else hne.predict
            function(features, output, cfg, args.workers)
        if args.suite in ["public", "all"]:
            methods = cfg["public_baselines"]["methods"]
            seeds = cfg["public_baselines"]["seeds"]
            if args.phase == "fit":
                for method in methods:
                    for seed in seeds:
                        public.fit_one(features, output, method, seed)
                    public.ensemble(output, method, ["train", "validation"])
            else:
                public.predict(features, output, methods, seeds)
                for method in methods:
                    public.ensemble(output, method, ["test"])


if __name__ == "__main__":
    main()
