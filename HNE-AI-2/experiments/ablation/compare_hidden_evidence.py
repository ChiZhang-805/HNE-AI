"""Audit paired source predictions from the same frozen M04 anchor."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import sys


def read_run(root, key):
    data_root = (root / 'results/experiments').resolve(strict=True)
    run = (data_root / key).resolve(strict=True)
    if not run.is_relative_to(data_root):
        raise ValueError('Expected a local experiment run key')
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'completed' or summary['run_key'] != key:
        raise ValueError(f'Run is incomplete: {key}')
    methods = defaultdict(dict)
    with (run / 'predictions.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            origin = (row['person'], int(row['origin']))
            method = methods[row['method']]
            if row['split'] != 'validation' or origin in method:
                raise ValueError('Expected unique source-validation keys')
            label, logit, probability = map(float, (row['label'], row['logit'], row['probability']))
            if label not in (0, 1) or not math.isfinite(logit) or not 0 <= probability <= 1:
                raise ValueError('Invalid source forecast')
            method[origin] = (label, logit, probability)
    return summary, dict(methods), run


def paired_nll(left, right, seed, repetitions):
    if left.keys() != right.keys():
        raise ValueError('Unpaired origin keys')
    delta_by_person = defaultdict(list)
    for (person, origin), (label, _, left_probability) in left.items():
        other_label, _, right_probability = right[person, origin]
        if label != other_label:
            raise ValueError('Different outcome for a shared origin')
        epsilon = sys.float_info.epsilon
        p, q = (max(epsilon, min(1 - epsilon, value)) for value in (left_probability, right_probability))
        diff = math.log(q) - math.log(p) if label else math.log1p(-q) - math.log1p(-p)
        delta_by_person[person].append(diff)
    groups = [(math.fsum(delta_by_person[p]), len(delta_by_person[p])) for p in sorted(delta_by_person)]
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sample = rng.choices(groups, k=len(groups))
        draws.append(math.fsum(x[0] for x in sample) / sum(x[1] for x in sample))
    draws.sort()

    def percentile(q):
        index = q * (len(draws) - 1)
        low, high = math.floor(index), math.ceil(index)
        return draws[low] + (draws[high] - draws[low]) * (index - low)

    return dict(delta_nll=math.fsum(x[0] for x in groups) / len(left),
                ci95=[percentile(.025), percentile(.975)], people=len(groups),
                bootstrap_repetitions=repetitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--hidden-run', required=True)
    parser.add_argument('--output-run', required=True)
    parser.add_argument('--internal-run', help='Include the completed internal-guidance comparison when available')
    args = parser.parse_args()
    hidden, methods, destination = read_run(args.project_root, args.hidden_run)
    output, other, _ = read_run(args.project_root, args.output_run)
    base = 'M04_C2_LoRA'
    if hidden['anchor_inputs'] != output['anchor_inputs'] or methods[base] != other[base]:
        raise ValueError('Hidden and output fusion do not share the exact same M04 origins and evidence')
    summaries = dict(hidden=hidden, output=output)
    methods.update({name: rows for name, rows in other.items() if name != base})
    if args.internal_run:
        internal, also, _ = read_run(args.project_root, args.internal_run)
        if (internal['base_checkpoint_sha256'] != hidden['anchor_inputs']['baseline/candidate0.pt']
                or internal['evidence_receipts_sha256'] != hidden['anchor_inputs']['evidence_receipts.json']
                or also[base] != methods[base]):
            raise ValueError('Internal guidance does not share the exact same M04 origins and evidence')
        summaries['internal'] = internal
        methods.update({name: rows for name, rows in also.items() if name.startswith('M10_')})
    pairs = [(name, base) for name in methods if name != base]
    for policy in ('selected', 'always_on'):
        pairs.append((f'M12_full_{policy}', f'M12_history_only_{policy}'))
        if args.internal_run:
            pairs.extend([(f'M10_{policy}', f'M12_full_{policy}'), (f'M10_{policy}', f'M13_{policy}')])
    contrasts = {f'{left} vs {right}': paired_nll(methods[left], methods[right], 20261114, 2000)
                 for left, right in pairs}
    metrics = {name: entry for summary in summaries.values() for name, entry in summary['metrics'].items()
               if name in methods}
    result = dict(status='passed', scope='Source validation used for selection; exploratory only',
                  binary_event_metrics=metrics, matched_origin_count=len(methods[base]),
                  positive_people=len({p for (p, _), (label, _, _) in methods[base].items() if label}),
                  runs={name: summary['run_key'] for name, summary in summaries.items()},
                  delta_definition='Left minus right, negative favors left', paired_nll=contrasts,
                  bootstrap_scope='Fixed fitted models, resample people with all their origins')
    path = destination / 'paired_readout_comparison.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
