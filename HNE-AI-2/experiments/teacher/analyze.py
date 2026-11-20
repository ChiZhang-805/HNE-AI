"""Aggregate the private teacher audit without changing fitted-model selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[2]


def paired_interval(values, seed, repetitions=2000):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    def quantile(q):
        pos = q * (len(means) - 1); low = int(pos)
        return means[low] + (means[min(low + 1, len(means) - 1)] - means[low]) * (pos - low)
    return dict(mean=statistics.mean(values), ci95=[quantile(.025), quantile(.975)], people=len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-key', required=True)
    args = parser.parse_args()
    directory = ROOT / 'results/experiments' / args.run_key
    directory = directory.resolve()
    if not directory.is_relative_to((ROOT / 'results/experiments/teacher').resolve()):
        raise ValueError('Expected a teacher result directory')
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    with (directory / 'person_metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    calibration = {}
    matched_vs_older = {}
    for row in rows:
        group = '/'.join(row[k] for k in ('split', 'period', 'architecture', 'calibrated', 'mode', 'view'))
        calibration.setdefault(group, []).append(row)
        if row['architecture'] == 'residual' and row['view'] == 'HNE':
            key = '/'.join(row[k] for k in ('split', 'period', 'calibrated'))
            matched_vs_older.setdefault(key, {}).setdefault(row['person'], {})[row['mode']] = float(row['nll'])
    contrasts = {}
    for key, people in matched_vs_older.items():
        values = [p['older_conditions'] - p['aligned'] for p in people.values()]
        contrasts[key] = paired_interval(values, summary['config']['seed'])
    checks = [r for r in summary['conditional_gains'] if r['architecture'] == 'residual'
              and r['mode'] == 'aligned' and r['calibrated']]
    supported = bool(checks) and all(r['gain_NE']['interval'][0] > 0 for r in checks)
    result = dict(
        source_person_metrics_sha256=hashlib.sha256((directory / 'person_metrics.csv').read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        calibration={key: dict(people=len(part),
                              mean_coverage90=statistics.mean(float(r['coverage90']) for r in part),
                              mean_pit=statistics.mean(float(r['pit_mean']) for r in part))
                     for key, part in calibration.items()},
        matched_condition_advantage=contrasts,
        contrast_definition='HNE NLL with causally older conditions minus HNE NLL with aligned conditions; positive favors aligned',
        uncertainty='Person resampling conditional on fitted models and this development fold; does not include refitting/selection uncertainty',
        positive_conditional_gain_across_audited_windows=supported,
        interpretation=('Positive conditional gain on the audited windows; broader validation remains required.' if supported
                        else 'Stable positive conditional information across the audited windows is not established.'),
        negative_control_limit='Older-condition contrasts are diagnostic and are not causal identification.',
        model_selection_changed=False,
    )
    (directory / 'interpretation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(matched_condition_advantage=contrasts, model_selection_changed=False)))


if __name__ == '__main__':
    main()
