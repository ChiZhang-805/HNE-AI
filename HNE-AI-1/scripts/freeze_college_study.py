"""Save the completed College model choices before constructing test labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/college_experience/frozen_models.json"
FOLDERS = ("results/college_experience", "results/college_transfer",
           "results/college_temporal", "results/college_future_objective",
           "results/college_reference_units", "results/college_rank_basis", "results/college_jape",
           "results/college_jape_auc", "results/college_family_graph", "results/college_rhythm", "results/college_delayed")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    if OUTPUT.exists():
        print("College study choices are already frozen.")
        return
    required = [ROOT / folder / "freeze_manifest.json" for folder in FOLDERS]
    for method in ("FCM", "CATCH"):
        required.append(ROOT / f"results/college_experience/{method}/ensemble/selection.json")
        for seed in (20261110, 20261111, 20261112):
            for variant in (method, f"{method}-score-LR"):
                required.append(ROOT / f"results/college_experience/{variant}/seed{seed}/selection.json")
    for path in required:
        json.loads(path.read_text(encoding="utf-8"))
    if (ROOT / "data/features/college_experience/test.npz").exists():
        raise RuntimeError("College test features already exist before the global freeze")
    files = set(required)
    for folder in FOLDERS:
        files.update(path for path in (ROOT / folder).rglob("*")
                     if path.suffix in (".pt", ".joblib") or path.name in ("selection.json", "fit_config.json", "preprocessing.npz"))
    files.update(ROOT / path for path in (
        "configs/college_evaluation.json", "configs/college_extended_evaluation.json",
        "configs/jape_event.json", "configs/jape_auc_head.json", "configs/daily_protocol.json",
        "configs/family_graph.json", "configs/college_family_graph.json", "src/hne_family_graph.py",
        "configs/rhythm_coordinates.json", "src/hne_rhythm.py",
        "configs/college_delayed.json", "configs/delayed_adaptation.json", "src/hne_online.py",
        "data/new_cohorts/college-experience/subject_split.csv",
        "data/college_study/schema_mapping.json"))
    record = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_outcomes_constructed": False,
        "primary": json.loads((ROOT / "configs/college_evaluation.json").read_text()),
        "extended_controls": json.loads((ROOT / "configs/college_extended_evaluation.json").read_text()),
        "supplementary": ["HNE temporal three-seed head", "future-values and future-moments auxiliary objectives",
                          "personal-reference units versus additional global state/context scaling",
                          "unchanged additive HNE using training-rank coordinates",
                          "shared future-family HNE graph: independent, constant and dynamic coupling; fixed equal two-level mixture",
                          "actual hourly harmonics: history-only additions, full context-adjusted phase relations, and equally informed history HGB",
                          "separate prequential evaluation with matched8-day feedback delay,90-day memory,7-day updates and calibration controls",
                          "32/63/126-person target-only, source-basis-zero and source-basis-prior adaptation"],
        "files": {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(files)},
    }
    OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"College study frozen: {len(required)} completion records, {len(files)} selected/config/model files.")


if __name__ == "__main__":
    main()
