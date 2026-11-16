"""Compare first-order directed prediction gains on the fixed development folds."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_directed import GROUPS, VARIANTS, encode_networks, cnm_score
from src.hne_representation import select_reference_channels
from src.hne_additive import AdditiveWarning, validation_score

OUT = ROOT/'results/directed_network'
FEATURES = ROOT/'data/features/directed_network'
DATASETS = ('RESILIENT', 'BrisT1D-Open')


def init_worker():
    global _threads
    _threads = threadpool_limits(limits=1)
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def encode(task):
    dataset, person, rows, cutoffs = task
    path = FEATURES/dataset/'people'/f'{person}.npz'
    if path.exists():
        return
    with np.load(ROOT/f'data/daily_study/{dataset}/{person}_daily.npz') as data:
        selected = select_reference_channels(data['normalized'], data['state_names'])
        states = data['normalized'][:, selected]
        context = data['normalized_contexts']
        values = encode_networks(states, context, cutoffs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=rows, **values)


def build(workers):
    for dataset in DATASETS:
        target = FEATURES/dataset/'features.npz'
        if target.exists():
            continue
        base = ROOT/f'data/features/unified_daily/{dataset}'
        meta = pd.read_csv(base.with_suffix('.csv'), dtype={'person_id': str})
        tasks = [(dataset, person, g.index.to_numpy(), g.origin_day.to_numpy()+1)
                 for person, g in meta.groupby('person_id', sort=False)]
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            jobs = [pool.submit(encode, task) for task in tasks]
            for count, future in enumerate(as_completed(jobs), 1):
                future.result()
                if count % 10 == 0 or count == len(tasks):
                    print(dataset, 'people', count, '/', len(tasks), flush=True)
        with np.load(base.with_suffix('.npz')) as data:
            original = data['pooled']
            native = [cnm_score(w) for w in data['windows']]
        all_features = {}
        for name in VARIANTS:
            network = np.full((len(meta), 24), np.nan, dtype=np.float32)
            for _, person, rows, _ in tasks:
                with np.load(FEATURES/dataset/'people'/f'{person}.npz') as data:
                    network[rows] = data[name]
            all_features[name] = np.column_stack((original[:, :96], network[:, :12], original[:, 132:260], network[:, 12:]))
        np.savez_compressed(target, **all_features, groups=GROUPS)
        pd.concat([meta, pd.DataFrame(native)], axis=1).to_csv(FEATURES/dataset/'cnm_scores.csv', index=False)
        print(dataset, 'features complete', len(meta), 'CNM', sum(np.isfinite(v['score']) for v in native), flush=True)


def fit():
    cfg = json.loads((ROOT/'configs/directed_network.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'configuration.json').write_text(json.dumps(cfg, indent=2))
    for dataset in DATASETS:
        meta = pd.read_csv(FEATURES/dataset/'cnm_scores.csv', dtype={'person_id': str})
        values = dict(np.load(FEATURES/dataset/'features.npz'))
        values['CNM-GC-past32-LR'] = meta[['score']].to_numpy()
        for name in (*VARIANTS, 'CNM-GC-past32-LR'):
            x = values[name]
            finite = np.isfinite(meta.score).to_numpy() if name.startswith('CNM') else np.ones(len(meta), dtype=bool)
            for fold in range(5):
                folder = OUT/dataset/name/f'fold{fold}'
                if (folder/'selection.json').exists():
                    continue
                train = ~meta.fold.isin([fold, (fold+1)%5]).to_numpy() & finite
                val = meta.fold.eq((fold+1)%5).to_numpy() & finite
                held = meta.fold.eq(fold).to_numpy()
                candidates, best, best_value, best_c = [], None, -np.inf, None
                for c in cfg['C_candidates']:
                    model = AdditiveWarning(C=c, spline=name in VARIANTS,
                                            groups=GROUPS if name in VARIANTS else None)
                    model.fit(x[train], meta.y.to_numpy()[train], meta.person_id.to_numpy()[train])
                    value = validation_score(meta.y.to_numpy()[val], model.predict_proba(x[val]))
                    candidates.append(dict(C=c, validation=value))
                    if value > best_value:
                        best, best_value, best_c = model, value, c
                folder.mkdir(parents=True, exist_ok=True)
                joblib.dump(best, folder/'model.joblib')
                for part, mask in [('validation', val), ('held_person', held)]:
                    frame = meta.loc[mask].copy()
                    frame['probability'] = np.nan
                    good = finite[mask]
                    frame.loc[good, 'probability'] = best.predict_proba(x[mask][good])
                    frame.to_csv(folder/f'{part}.csv', index=False)
                (folder/'selection.json').write_text(json.dumps(dict(C=best_c, candidates=candidates), indent=2))
                print(dataset, name, fold, 'validation', round(best_value, 4), flush=True)
            parts = [pd.read_csv(OUT/dataset/name/f'fold{k}/held_person.csv', dtype={'person_id': str}) for k in range(5)]
            pd.concat(parts, ignore_index=True).to_csv(OUT/dataset/name/'held_person.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--phase', choices=['features', 'fit', 'all'], default='all')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    if args.phase in ['features', 'all']:
        build(args.workers)
    if args.phase in ['fit', 'all']:
        with threadpool_limits(limits=2):
            fit()


if __name__ == '__main__':
    main()
