"""Compare within-day rhythm features with the unchanged daily HNE protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_rhythm import encode_rhythm
from src.hne_additive import AdditiveWarning, BoostedWarning, history_features
from scripts.summarize_development import fold_auc
from scripts.build_daily_study import FAMILIES

FEATURES = ROOT / 'data/features/rhythm_coordinates'
OUTPUT = ROOT / 'results/rhythm_coordinates'
COLLEGE = ROOT / 'results/college_rhythm'


def worker(task):
    dataset, split, person, indices, days, cfg = task
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('ignore', RuntimeWarning)
        if dataset == 'College-Experience':
            with np.load(ROOT / f'data/college_hourly/{split}/{person}.npz') as z:
                states, contexts, families = z['states'], z['contexts'], z['state_families']
        else:
            with np.load(ROOT / f'data/development/{dataset}/{person}_hourly.npz') as z:
                raw = z['values']
                states, contexts, families = raw[:, :5], raw[:, 5:], np.asarray(FAMILIES[dataset])
        x, groups, coverage = encode_rhythm(states, contexts, families, days+1, cfg)
        return indices, x, groups, coverage


def build(dataset, split, cfg, workers):
    source = ROOT / ('data/features/college_experience/'+split if dataset == 'College-Experience' else 'data/features/unified_daily/'+dataset)
    meta = pd.read_csv(source.with_suffix('.csv'), dtype={'person_id': str})
    name = dataset+'_'+split if split else dataset
    destination = FEATURES / (name+'.npz')
    if not destination.exists():
        FEATURES.mkdir(parents=True, exist_ok=True)
        x = np.full((len(meta), 228), np.nan, dtype=np.float32)
        coverage = np.zeros((len(meta), 6), dtype=np.float32)
        tasks = [(dataset, split, str(person), part.index.to_numpy(), part.origin_day.to_numpy(), cfg)
                 for person, part in meta.groupby('person_id', sort=False)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, (indices, features, groups, valid) in enumerate(pool.map(worker, tasks), 1):
                x[indices], coverage[indices] = features, valid
                if n % 20 == 0 or n == len(tasks):
                    print(f'Rhythm {name} {n}/{len(tasks)} participants', flush=True)
        np.savez_compressed(destination, features=x, groups=groups, valid_phase_fraction=coverage)
    with np.load(destination) as z:
        rhythm, rhythm_groups = z['features'], z['groups']
    with np.load(source.with_suffix('.npz')) as z:
        daily, daily_groups, windows = z['pooled'], z['groups'], z['windows']
    return meta, daily, daily_groups, windows, rhythm, rhythm_groups


def coordinates(data, method):
    meta, daily, groups, windows, rhythm, rhythm_groups = data
    raw = rhythm_groups == 'H'
    if method == 'History-HGB-rhythm':
        return np.column_stack((history_features(windows, scales=(8, 16, 32)), rhythm[:, raw])), None
    keep = raw if method == 'HNE-rhythm-H' else np.ones(len(rhythm_groups), dtype=bool)
    return np.column_stack((daily, rhythm[:, keep])), np.r_[groups, rhythm_groups[keep]]


def fit_one(meta, x, groups, masks, method, out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'selection.json').exists():
        return json.loads((out / 'selection.json').read_text())
    train, val, held = masks
    y, people = meta.y.to_numpy(), meta.person_id.to_numpy()
    best_score, trials = -np.inf, []
    for c in cfg['C']:
        model = BoostedWarning(c) if method == 'History-HGB-rhythm' else AdditiveWarning(c, groups=groups)
        model.fit(x[train], y[train], people[train])
        score = roc_auc_score(y[val], model.predict_proba(x[val]))
        trials.append({'C': c, 'validation_auroc': score})
        if score > best_score:
            best, best_score, chosen = model, score, c
    joblib.dump(best, out / 'model.joblib')
    record = {'method': method, 'C': chosen, 'trials': trials, 'validation_auroc': best_score}
    for name, mask in [('validation', val), ('held_person', held)]:
        if mask is None:
            continue
        frame = meta.loc[mask].copy()
        frame['score'] = best.predict_proba(x[mask])
        frame.to_csv(out / (name+'.csv'), index=False)
        record[name] = {'origins': len(frame), 'auroc': roc_auc_score(frame.y, frame.score),
                        'ap': average_precision_score(frame.y, frame.score)}
    (out / 'selection.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'output': str(out.relative_to(ROOT)), **{k:v for k,v in record.items() if k!='trials'}}), flush=True)
    return record


def development(cfg, workers):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'configuration.json').write_text(json.dumps(cfg, indent=2))
    metrics = []
    for dataset in cfg['development_datasets']:
        data = build(dataset, '', cfg, workers)
        meta = data[0]
        for method in cfg['methods']:
            x, groups = coordinates(data, method)
            for fold in range(5):
                masks = (~meta.fold.isin([fold, (fold+1)%5]), meta.fold.eq((fold+1)%5), meta.fold.eq(fold))
                fit_one(meta, x, groups, masks, method, OUTPUT / dataset / method / f'fold{fold}', cfg)
            frame = pd.concat([pd.read_csv(OUTPUT / dataset / method / f'fold{k}/held_person.csv') for k in range(5)])
            frame.to_csv(OUTPUT / dataset / method / 'held_person.csv', index=False)
            metrics.append({'dataset': dataset, 'method': method, 'fold_pair_auroc': fold_auc(frame, 'score')})
    pd.DataFrame(metrics).to_csv(OUTPUT / 'metrics.csv', index=False)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


def college_fit(cfg, workers):
    parts = [build('College-Experience', split, cfg, workers) for split in ['train', 'validation']]
    meta = pd.concat([p[0] for p in parts], ignore_index=True)
    train = np.arange(len(meta)) < len(parts[0][0])
    selections = []
    for method in cfg['methods']:
        choices = [coordinates(p, method) for p in parts]
        x = np.concatenate([p[0] for p in choices])
        selections.append(fit_one(meta, x, choices[0][1], (train, ~train, None), method, COLLEGE / method, cfg))
    (COLLEGE / 'freeze_manifest.json').write_text(json.dumps({'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                                                            'configuration': cfg, 'selections': selections}, indent=2))


def college_predict(cfg, workers):
    json.loads((ROOT / 'results/college_experience/frozen_models.json').read_text())
    data = build('College-Experience', 'test', cfg, workers)
    for method in cfg['methods']:
        x, _ = coordinates(data, method)
        model = joblib.load(COLLEGE / method / 'model.joblib')
        frame = data[0].copy()
        frame['score'] = model.predict_proba(x)
        frame.to_csv(COLLEGE / method / 'test.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['development', 'college-fit', 'college-predict'], default='development')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/rhythm_coordinates.json').read_text())
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    with threadpool_limits(limits=2):
        {'development': development, 'college-fit': college_fit, 'college-predict': college_predict}[args.phase](cfg, args.workers)
