"""Matched delayed-feedback learning with the existing HNE online algorithm."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_online import prequential
from scripts.run_delayed_adaptation import describe
from scripts.run_hne_external import make_features

OUTPUT = ROOT / 'results/college_delayed'
BASE = ROOT / 'results/college_experience'
KEYS = ['person_id', 'origin_day']


class Probabilities:
    def predict_proba(self, x):
        return x[:, 0]


def score_table(method, split):
    if method.startswith(('FCM-', 'CATCH-')):
        name = method.split('-')[0]
        paths = [BASE / (name+'-score-LR') / f'seed{seed}' / f'{split}.csv' for seed in [20261110, 20261111, 20261112]]
        if any(not p.exists() for p in paths):
            return None
        frames = [pd.read_csv(p, dtype={'person_id': str}).set_index(KEYS) for p in paths]
        frame = frames[0][['score']].copy()
        frame['score'] = np.mean([f.reindex(frame.index).score.to_numpy() for f in frames], axis=0)
        return frame.reset_index()
    if method == 'JAPE-event-AUC-3seed':
        path = ROOT / 'results/college_jape_auc' / f'{split}.csv'
    else:
        native = method.replace('MM-DEV-score-LR', 'MM-DEV-daily').removesuffix('-score-LR')
        path = BASE / native / f'{split}.csv'
    return pd.read_csv(path, dtype={'person_id': str}) if path.exists() else None


def input_data(method, split, cfg):
    meta = pd.read_csv(ROOT / f'data/features/college_experience/{split}.csv', dtype={'person_id': str})
    if method in ['HNE-additive', 'HNE-history', 'History-HGB', 'History-LR']:
        bundle = joblib.load(BASE / method / 'model.joblib')
        with np.load(ROOT / f'data/features/college_experience/{split}.npz') as z:
            arrays = {'pooled': z['pooled'], 'windows': z['windows']}
        return meta, bundle['model'], make_features(arrays, bundle['feature_spec'])
    scores = score_table(method, split)
    if scores is None:
        return None
    frame = meta.merge(scores[KEYS+['score']], on=KEYS, validate='one_to_one')
    frame = frame[np.isfinite(frame.score)].reset_index(drop=True)
    x = frame.score.to_numpy()[:, None]
    if method in ['DNB-score-LR', 'MM-DEV-score-LR']:
        folder = OUTPUT / method
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'readout.joblib'
        if path.exists():
            model = joblib.load(path)
        else:
            train = score_table(method, 'train')
            train = train[np.isfinite(train.score)]
            model = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['native_score_readout_C'], max_iter=1000))
            model.fit(train.score.to_numpy()[:, None], train.y.to_numpy())
            joblib.dump(model, path)
        x = model.predict_proba(x)[:, 1:2]
    return frame.drop(columns='score'), Probabilities(), x


def evaluate(meta, model, x, condition, c, core):
    p, count, latest = prequential(model, x, meta.y.to_numpy(), meta.person_id.to_numpy(),
                                   meta.origin_day.to_numpy(), condition, c, core)
    result = meta.copy()
    result['score'], result['mature_origins_used'], result['latest_feedback_origin'] = p, count, latest
    return result


def run(phase, cfg, core):
    OUTPUT.mkdir(exist_ok=True)
    if phase == 'predict':
        json.loads((ROOT / 'results/college_experience/frozen_models.json').read_text())
    split = 'validation' if phase == 'fit' else 'test'
    records, waiting = [], []
    for method in cfg['source_methods']:
        data = input_data(method, split, cfg)
        if data is None:
            waiting.append(method)
            continue
        meta, model, x = data
        conditions = cfg['common_conditions']+(['coefficients'] if method in cfg['coefficient_conditions'] else [])
        for condition in conditions:
            folder = OUTPUT / method / condition
            folder.mkdir(parents=True, exist_ok=True)
            if phase == 'fit':
                if (folder / 'selection.json').exists():
                    records.append(json.loads((folder / 'selection.json').read_text()))
                    continue
                best, trials = -np.inf, []
                for c in ([0.] if condition == 'frozen' else core['C_candidates']):
                    frame = evaluate(meta, model, x, condition, c, core)
                    metrics = describe(frame)
                    trials.append({'C': c, **metrics})
                    if metrics['auroc'] > best:
                        best, chosen, selected = metrics['auroc'], c, frame
                selected.to_csv(folder / 'validation.csv', index=False)
                record = {'source_method': method, 'condition': condition, 'C': chosen,
                          'trials': trials, 'validation': describe(selected)}
                (folder / 'selection.json').write_text(json.dumps(record, indent=2))
                records.append(record)
                print(json.dumps({k:v for k,v in record.items() if k!='trials'}), flush=True)
            else:
                record = json.loads((folder / 'selection.json').read_text())
                frame = evaluate(meta, model, x, condition, record['C'], core)
                frame.to_csv(folder / 'test.csv', index=False)
                records.append({'source_method': method, 'condition': condition, **describe(frame)})
    if phase == 'fit':
        pd.DataFrame([{'source_method': r['source_method'], 'condition': r['condition'], 'C': r['C'], **r['validation']} for r in records]).to_csv(OUTPUT / 'validation_metrics.csv', index=False)
        if not waiting:
            (OUTPUT / 'freeze_manifest.json').write_text(json.dumps({'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'configuration': cfg, 'core': core, 'selections': records}, indent=2))
    else:
        pd.DataFrame(records).to_csv(OUTPUT / 'test_metrics.csv', index=False)
    print(json.dumps({'phase': phase, 'complete_conditions': len(records), 'waiting_for': waiting}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['fit', 'predict'], default='fit')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/college_delayed.json').read_text())
    core = json.loads((ROOT / cfg['core_configuration']).read_text())
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    with threadpool_limits(limits=2):
        run(args.phase, cfg, core)
