"""Paired participant intervals for prespecified prequential learning conditions."""
import json
from pathlib import Path
import sys
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_external import compare, read


def main():
    cfg = json.loads((ROOT / 'configs/college_delayed.json').read_text())
    folders, pairs = {}, []
    for method in cfg['source_methods']:
        conditions = cfg['common_conditions']+(['coefficients'] if method in cfg['coefficient_conditions'] else [])
        for condition in conditions:
            name = method+'__'+condition
            folders[name] = ROOT / 'results/college_delayed' / method / condition
            if condition != 'frozen':
                pairs.append((name, method+'__frozen'))
    controls = [name for name in folders if not name.startswith('HNE-additive__')]
    controls += ['HNE-additive__intercept', 'HNE-additive__affine']
    plan = {'primary_hne': 'HNE-additive__coefficients', 'primary_comparators': controls,
            'primary_metric': 'prequential AUROC', 'setting': cfg['test'],
            'interpretation': 'Source coefficient updates compared with every fixed calibration condition under identical matured feedback'}
    pairs += [('HNE-additive__coefficients', 'HNE-additive__intercept'),
              ('HNE-additive__coefficients', 'HNE-additive__affine'),
              ('HNE-additive__coefficients', 'HNE-history__coefficients')]
    meta = read(ROOT / 'data/features/college_experience/test.csv')
    compare(plan, folders, meta, ROOT / 'results/college_delayed/comparison', extra_pairs=pairs)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
