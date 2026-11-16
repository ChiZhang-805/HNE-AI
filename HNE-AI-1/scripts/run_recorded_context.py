"""Add an available timestamped context field while retaining the complete HNE core."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.daily_protocol import robust_reference, normalize
from src.hne_representation import Sampling, encode_person, select_reference_channels
from scripts.prepare_development_hourly import hourly
from scripts.run_development import run
from scripts.summarize_development import fold_auc, bootstrap_aucs

DATASET = 'BrisT1D-Open'
FEATURES = ROOT / 'data/features/recorded_context'
OUTPUT = ROOT / 'results/recorded_context'


def person_features(task):
    person, indices, days, source = task
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('ignore', RuntimeWarning)
        with np.load(ROOT / f'data/daily_study/{DATASET}/{person}_daily.npz') as data:
            states, contexts = data['normalized'], data['normalized_contexts']
            names, dates = data['state_names'], data['dates']
        observed = pd.read_csv(source / (person+'.csv'), low_memory=False).drop_duplicates()
        by_hour = hourly(pd.to_datetime(observed.timestamp), observed.insulin, total=True)
        clock = pd.date_range(pd.Timestamp(dates[0]), periods=len(dates)*24, freq='h')+pd.Timedelta(hours=1)
        values = by_hour.reindex(clock).to_numpy().reshape(len(dates), 24)
        count = np.isfinite(values).sum(1)
        daily = np.nanmean(values, axis=1)[:, None]
        center, scale, support = robust_reference(daily, 56, 28)
        added = normalize(daily, center, scale)
        contexts = np.column_stack((contexts, added)).astype(np.float32)
        selected = select_reference_channels(states, names, reference=56)
        states = states[:, selected]
        nodes, pooled, groups = encode_person(states, contexts, days+1, Sampling.daily())
        trajectory = np.column_stack((states, contexts))
        windows = np.stack([trajectory[d-31:d+1] for d in days])
        folder = ROOT / 'data/recorded_context'
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/(person+'.npz'), insulin_hourly_mean=daily,
                            insulin_hours=count, normalized_contexts=contexts, dates=dates)
        return indices, nodes, pooled, windows, groups, {
            'person_id': person, 'calendar_days': len(dates), 'observed_days': int((count>0).sum()),
            'reference_days_observed': int(support[0]), 'reference_scale': float(scale[0]),
            'origins_with_observed_insulin': int(np.isfinite(added[days]).sum())}


def build(source, workers):
    FEATURES.mkdir(parents=True, exist_ok=True)
    target = FEATURES / (DATASET+'.npz')
    if target.exists():
        return
    original = ROOT / 'data/features/unified_daily' / (DATASET+'.csv')
    meta = pd.read_csv(original, dtype={'person_id': str})
    tasks = [(str(person), part.index.to_numpy(), part.origin_day.to_numpy(), source)
             for person, part in meta.groupby('person_id', sort=False)]
    gathered, coverage = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(person_features, tasks):
            gathered.append(result)
            coverage.append(result[-1])
            print(json.dumps(result[-1]), flush=True)
    order = np.argsort(np.concatenate([r[0] for r in gathered]))
    payload = {name: np.concatenate([r[j] for r in gathered])[order]
               for j, name in enumerate(['indices', 'nodes', 'pooled', 'windows']) if j}
    payload.update(groups=gathered[0][4], y=meta.y.to_numpy(), fold=meta.fold.to_numpy())
    np.savez_compressed(target, **payload)
    meta.to_csv(target.with_suffix('.csv'), index=False)
    pd.DataFrame(coverage).to_csv(OUTPUT / 'coverage.csv', index=False)


def analyze(cfg):
    frame = pd.read_csv(ROOT / f'results/unified_daily/summary/{DATASET}_predictions.csv', dtype={'person_id': str})
    frame = frame[frame.primary_common].copy()
    names = []
    for model in cfg['models']:
        parts = [pd.read_csv(OUTPUT / DATASET / model / f'fold{k}/held_person.csv', dtype={'person_id': str}) for k in cfg['folds']]
        scored = pd.concat(parts, ignore_index=True)
        scored.to_csv(OUTPUT / DATASET / model / 'held_person.csv', index=False)
        name = model+'+insulin'
        names.append(name)
        frame = frame.merge(scored[['person_id', 'origin_day', 'probability']].rename(columns={'probability': name}),
                            on=['person_id', 'origin_day'], validate='one_to_one')
    originals = cfg['models']
    ids, inverse = np.unique(frame.person_id, return_inverse=True)
    weights = np.random.default_rng(20261110).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=2000)
    points = {name: fold_auc(frame, name) for name in originals+names}
    boots = {name: bootstrap_aucs(frame, name, inverse, weights) for name in points}
    rows, pairs = [], []
    for name in points:
        lo, hi = np.nanquantile(boots[name], [.025, .975])
        within = [roc_auc_score(p.y, p[name]) for _,p in frame.groupby('person_id') if p.y.nunique()==2]
        rows.append({'dataset':DATASET, 'method':name, 'origins':len(frame), 'auroc':points[name],
                     'ci_low':lo, 'ci_high':hi, 'within_person_auroc':float(np.mean(within))})
    for model in cfg['models']:
        lo, hi = np.nanquantile(boots[model+'+insulin']-boots[model], [.025,.975])
        pairs.append({'model':model+'+insulin', 'baseline':model,
                      'difference':points[model+'+insulin']-points[model], 'ci_low':lo, 'ci_high':hi})
    for baseline in ['HNE-context-append+insulin','History-HGB+insulin','History-LR+insulin']:
        lo, hi = np.nanquantile(boots['HNE-additive+insulin']-boots[baseline], [.025,.975])
        pairs.append({'model':'HNE-additive+insulin', 'baseline':baseline,
                      'difference':points['HNE-additive+insulin']-points[baseline], 'ci_low':lo, 'ci_high':hi})
    pd.DataFrame(rows).to_csv(OUTPUT/'metrics.csv', index=False)
    pd.DataFrame(pairs).to_csv(OUTPUT/'paired_differences.csv', index=False)
    frame.to_csv(OUTPUT/'predictions.csv', index=False)
    np.savez_compressed(OUTPUT/'bootstrap.npz', **boots)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/recorded_context.json').read_text())
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'configuration.json').write_text(json.dumps(cfg,indent=2))
    build(args.source,args.workers)
    with warnings.catch_warnings(),threadpool_limits(limits=2):
        warnings.simplefilter('ignore',RuntimeWarning)
        run(cfg['models'],cfg['folds'],FEATURES,OUTPUT,[DATASET],scales=(8,16,32))
        analyze(cfg)
