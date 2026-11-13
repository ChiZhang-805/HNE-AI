"""Population-to-person reference learning followed by the unchanged warning head."""
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
from src.hne_shared_reference import (WIDTH, VARIANTS, prepare_reference, shared_prior,
    reference_errors, normal_loss, encode_shared)
from src.hne_representation import select_reference_channels
from src.hne_additive import AdditiveWarning, validation_score

OUT = ROOT/'results/shared_reference'
FEATURES = ROOT/'data/features/shared_reference'


def init_worker():
    global _threads
    _threads = threadpool_limits(limits=1)
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_person(dataset, person):
    with np.load(ROOT/f'data/daily_study/{dataset}/{person}_daily.npz') as data:
        selected = select_reference_channels(data['normalized'], data['state_names'])
        return data['normalized'][:, selected], data['normalized_contexts']


def prepare_person(task):
    dataset, person, cfg = task
    target = FEATURES/dataset/'people'/f'{person}.npz'
    if target.exists():
        return
    states, context = load_person(dataset, person)
    bundle = prepare_reference(states, context, cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **bundle)


def load_bundle(dataset, person):
    with np.load(FEATURES/dataset/'people'/f'{person}.npz') as data:
        return {k: data[k] for k in data.files}


def encode_person_task(task):
    dataset, person, rows, cutoffs, prior, ridge, cfg, target = task
    target = Path(target)
    if target.exists():
        return
    states, context = load_person(dataset, person)
    bundle = load_bundle(dataset, person)
    pooled, error, coefficient = encode_shared(states, context, cutoffs, bundle, prior, ridge, cfg)
    reference, _ = reference_errors(bundle, prior, ridge, cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, rows=rows, pooled=pooled, standardized_error=error,
                        reference_residual=reference, personal_coefficients=coefficient, prior=prior)


def prepare(cfg, workers):
    for dataset in cfg['datasets']:
        meta = pd.read_csv(ROOT/f'data/features/unified_daily/{dataset}.csv', dtype={'person_id': str})
        tasks = [(dataset, p, cfg) for p in meta.person_id.unique()]
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            futures = [pool.submit(prepare_person, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 10 == 0 or count == len(tasks):
                    print(dataset, 'reference projections', count, '/', len(tasks), flush=True)


def fit(cfg, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'configuration.json').write_text(json.dumps(cfg, indent=2))
    for dataset in cfg['datasets']:
        meta = pd.read_csv(ROOT/f'data/features/unified_daily/{dataset}.csv', dtype={'person_id': str})
        with np.load(ROOT/f'data/features/unified_daily/{dataset}.npz') as data:
            original, groups = data['pooled'], data['groups']
        person_fold = meta.groupby('person_id').fold.first().to_dict()
        by_person = {person: g for person, g in meta.groupby('person_id', sort=False)}
        bundles = {p: load_bundle(dataset, p) for p in by_person}
        reference_records = []
        for fold in range(5):
            train_people = [p for p in by_person if person_fold[p] not in [fold, (fold+1)%5]]
            validation_people = [p for p in by_person if person_fold[p] == (fold+1)%5]
            gram = sum(bundles[p]['prior_gram'] for p in train_people)
            target = sum(bundles[p]['prior_target'] for p in train_people)
            mass = sum(float(bundles[p]['prior_mass']) for p in train_people)
            population = shared_prior(gram, target, mass, cfg['shared_ridge'])
            for name in VARIANTS:
                folder = OUT/dataset/name/f'fold{fold}'
                destination = FEATURES/dataset/name/f'fold{fold}'
                if (folder/'selection.json').exists():
                    continue
                shared = name == 'HNE-shared-reference'
                priors = {}
                for person in by_person:
                    if not shared:
                        priors[person] = np.zeros(WIDTH)
                    elif person in train_people:
                        b = bundles[person]
                        priors[person] = shared_prior(gram-b['prior_gram'], target-b['prior_target'],
                            mass-float(b['prior_mass']), cfg['shared_ridge'])
                    else:
                        priors[person] = population
                normal_candidates = []
                for ridge in cfg['personal_ridge_candidates']:
                    losses = [normal_loss(reference_errors(bundles[p], priors[p], ridge, cfg)[1]) for p in validation_people]
                    normal_candidates.append(dict(ridge=ridge, validation_reference_huber2=float(np.nanmean(losses))))
                ridge = min(normal_candidates, key=lambda row: row['validation_reference_huber2'])['ridge']
                tasks = [(dataset, p, g.index.to_numpy(), g.origin_day.to_numpy()+1, priors[p], ridge, cfg,
                          str(destination/'people'/f'{p}.npz')) for p, g in by_person.items()]
                with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
                    for future in as_completed([pool.submit(encode_person_task, task) for task in tasks]):
                        future.result()
                x = np.full_like(original, np.nan)
                for p, g in by_person.items():
                    with np.load(destination/'people'/f'{p}.npz') as data:
                        x[g.index] = data['pooled']
                        error = data['standardized_error']
                        split = 'held' if person_fold[p] == fold else 'validation' if person_fold[p] == (fold+1)%5 else 'train'
                        reference_records.append(dict(dataset=dataset, method=name, fold=fold, split=split, person_id=p,
                            ridge=ridge, normal_huber2=normal_loss(error),
                            normal_mse=float(np.nanmean(np.nanmean(error**2, axis=0))),
                            normal_mae=float(np.nanmean(np.nanmean(abs(error), axis=0)))))
                if not np.allclose(x[:, :132], original[:, :132], equal_nan=True, atol=1e-6, rtol=1e-6):
                    raise ValueError('Raw H/N changed')
                x[:, :132], x[:, 236:260] = original[:, :132], original[:, 236:260]
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination/'features.npz', pooled=x, groups=groups)
                train = meta.person_id.isin(train_people).to_numpy()
                val = meta.person_id.isin(validation_people).to_numpy()
                held = meta.fold.eq(fold).to_numpy()
                candidates, best, best_score, best_c = [], None, -np.inf, None
                with threadpool_limits(limits=2):
                    for c in cfg['C_candidates']:
                        model = AdditiveWarning(C=c, groups=groups)
                        model.fit(x[train], meta.y.to_numpy()[train], meta.person_id.to_numpy()[train])
                        score = validation_score(meta.y.to_numpy()[val], model.predict_proba(x[val]))
                        candidates.append(dict(C=c, validation=score))
                        if score > best_score:
                            best, best_score, best_c = model, score, c
                folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(reference_records[-len(by_person):]).to_csv(folder/'normal_reference.csv', index=False)
                joblib.dump(dict(risk_model=best, population_prior=population if shared else np.zeros(WIDTH),
                    personal_ridge=ridge, config=cfg), folder/'model.joblib')
                for part, mask in [('validation', val), ('held_person', held)]:
                    frame = meta.loc[mask].copy()
                    frame['probability'] = best.predict_proba(x[mask])
                    frame.to_csv(folder/f'{part}.csv', index=False)
                (folder/'selection.json').write_text(json.dumps(dict(C=best_c, candidates=candidates,
                    personal_ridge=ridge, reference_candidates=normal_candidates, training_people=train_people,
                    population_coefficients=population.tolist() if shared else [0.]*WIDTH), indent=2))
                print(dataset, name, 'fold', fold, 'ridge', ridge, 'validation AUC', round(best_score, 4), flush=True)
        for name in VARIANTS:
            parts = [pd.read_csv(OUT/dataset/name/f'fold{k}/held_person.csv', dtype={'person_id': str}) for k in range(5)]
            pd.concat(parts, ignore_index=True).to_csv(OUT/dataset/name/'held_person.csv', index=False)
        # Rebuild the reference comparison from the held-person predictions only.
        comparisons = []
        for person, fold in person_fold.items():
            errors = []
            for name in VARIANTS:
                with np.load(FEATURES/dataset/name/f'fold{fold}/people'/f'{person}.npz') as data:
                    errors.append(data['reference_residual'])
            full, local = errors
            for node in range(full.shape[1]):
                common = np.isfinite(full[:, node]) & np.isfinite(local[:, node])
                if common.any():
                    comparisons.append(dict(person_id=person, node=node, observations=int(common.sum()),
                        full_mse=float(np.mean(full[common, node]**2)), own_mse=float(np.mean(local[common, node]**2))))
        pd.DataFrame(comparisons).to_csv(FEATURES/dataset/'reference_prediction.csv', index=False)
        reference_parts = [pd.read_csv(OUT/dataset/name/f'fold{k}/normal_reference.csv', dtype={'person_id': str})
                           for name in VARIANTS for k in range(5)]
        pd.concat(reference_parts, ignore_index=True).to_csv(OUT/f'{dataset}_reference_predictions.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--phase', choices=['prepare', 'fit', 'all'], default='all')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    cfg = json.loads((ROOT/'configs/shared_reference.json').read_text())
    if args.phase in ['prepare', 'all']:
        prepare(cfg, args.workers)
    if args.phase in ['fit', 'all']:
        fit(cfg, args.workers)


if __name__ == '__main__':
    main()
