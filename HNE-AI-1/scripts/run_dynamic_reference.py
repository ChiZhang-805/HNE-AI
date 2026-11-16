"""Fit the two fixed normal-dynamics references on existing development people."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_dynamic_reference import VARIANTS, encode_dynamic
from src.hne_representation import select_reference_channels
from src.hne_additive import AdditiveWarning, validation_score

OUT = ROOT/'results/dynamic_reference'
FEATURES = ROOT/'data/features/dynamic_reference'


def init_worker():
    global _threads
    _threads = threadpool_limits(limits=1)
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def encode(task):
    dataset, person, rows, cutoffs, config = task
    target = FEATURES/dataset/'people'/f'{person}.npz'
    if target.exists():
        return
    with np.load(ROOT/f'data/daily_study/{dataset}/{person}_daily.npz') as data:
        selected = select_reference_channels(data['normalized'], data['state_names'])
        values, details = encode_dynamic(data['normalized'][:, selected], data['normalized_contexts'], cutoffs, config)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, rows=rows, **values)
    target.with_suffix('.json').write_text(json.dumps(details, indent=2))


def build(config, workers):
    for dataset in config['datasets']:
        target = FEATURES/dataset/'features.npz'
        if target.exists():
            continue
        base = ROOT/f'data/features/unified_daily/{dataset}'
        meta = pd.read_csv(base.with_suffix('.csv'), dtype={'person_id': str})
        tasks = [(dataset, person, g.index.to_numpy(), g.origin_day.to_numpy()+1, config)
                 for person, g in meta.groupby('person_id', sort=False)]
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            futures = [pool.submit(encode, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 10 == 0 or count == len(tasks):
                    print(dataset, 'people', count, '/', len(tasks), flush=True)
        features = {name: np.full((len(meta), 296), np.nan, np.float32) for name in VARIANTS}
        diagnoses = []
        for _, person, rows, _, _ in tasks:
            with np.load(FEATURES/dataset/'people'/f'{person}.npz') as data:
                for name in VARIANTS:
                    features[name][rows] = data[name]
                full, own = data['reference_residuals']
                for node in range(full.shape[1]):
                    common = np.isfinite(full[:, node]) & np.isfinite(own[:, node])
                    if common.any():
                        diagnoses.append(dict(person_id=person, node=node, observations=int(common.sum()),
                            full_mse=float(np.mean(full[common, node]**2)), own_mse=float(np.mean(own[common, node]**2))))
        with np.load(base.with_suffix('.npz')) as original:
            for name in VARIANTS:
                if not np.allclose(features[name][:, :132], original['pooled'][:, :132], atol=1e-6, rtol=1e-6, equal_nan=True):
                    raise ValueError('Raw H/N coordinates changed')
                if not np.array_equal(features[name][:, 236:260], original['pooled'][:, 236:260], equal_nan=True):
                    raise ValueError('Context coordinates changed')
                # Reuse the unchanged coordinates exactly; per-person parallel
                # reductions can otherwise differ by one float32 ULP.
                features[name][:, :132] = original['pooled'][:, :132]
            np.savez_compressed(target, **features, groups=original['groups'])
        meta.to_csv(FEATURES/dataset/'samples.csv', index=False)
        pd.DataFrame(diagnoses).to_csv(FEATURES/dataset/'reference_prediction.csv', index=False)
        print(dataset, 'features complete', len(meta), flush=True)


def fit(config):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'configuration.json').write_text(json.dumps(config, indent=2))
    for dataset in config['datasets']:
        meta = pd.read_csv(FEATURES/dataset/'samples.csv', dtype={'person_id': str})
        values = dict(np.load(FEATURES/dataset/'features.npz'))
        for name in VARIANTS:
            x = values[name]
            for fold in range(5):
                folder = OUT/dataset/name/f'fold{fold}'
                if (folder/'selection.json').exists():
                    continue
                train = ~meta.fold.isin([fold, (fold+1)%5]).to_numpy()
                val = meta.fold.eq((fold+1)%5).to_numpy()
                held = meta.fold.eq(fold).to_numpy()
                candidates, best, best_value, best_c = [], None, -np.inf, None
                for c in config['C_candidates']:
                    model = AdditiveWarning(C=c, groups=values['groups'])
                    model.fit(x[train], meta.y.to_numpy()[train], meta.person_id.to_numpy()[train])
                    score = validation_score(meta.y.to_numpy()[val], model.predict_proba(x[val]))
                    candidates.append(dict(C=c, validation=score))
                    if score > best_value:
                        best, best_value, best_c = model, score, c
                folder.mkdir(parents=True, exist_ok=True)
                joblib.dump(best, folder/'model.joblib')
                for part, mask in [('validation', val), ('held_person', held)]:
                    frame = meta.loc[mask].copy()
                    frame['probability'] = best.predict_proba(x[mask])
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
    config = json.loads((ROOT/'configs/dynamic_reference.json').read_text())
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    if args.phase in ['features', 'all']:
        build(config, args.workers)
    if args.phase in ['fit', 'all']:
        with threadpool_limits(limits=2):
            fit(config)


if __name__ == '__main__':
    main()
