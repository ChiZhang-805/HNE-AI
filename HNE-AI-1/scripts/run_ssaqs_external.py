"""Freeze source predictors, construct SSAQS once, and evaluate person-level transfer."""
from pathlib import Path
from datetime import datetime, timezone
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
from scripts.run_transfer import source_data, write_json, sha
from scripts.run_hne_external import native_scores, cluster_statistics
from src.hne_source_controls import SharedCoordinateBoosting


def fit(output, cfg):
    output.mkdir(parents=True, exist_ok=True)
    transfer = json.loads((ROOT/'configs/transfer.json').read_text())
    x, meta, groups, domains, people, hashes = source_data(cfg['sources'], transfer)
    y, folds = meta.y.to_numpy(), meta.fold.to_numpy()
    source = ROOT/'results/transfer/source_models/Humans'
    models = {}
    for method in ['HNE-additive', 'HNE-history']:
        models[method] = {'path': str(source/method/'model.joblib'), 'sha256': sha(source/method/'model.joblib')}
    for method in ['H-pooled-HGB', 'HNE-pooled-HGB']:
        folder = output/method
        folder.mkdir(exist_ok=True)
        columns = np.flatnonzero(groups == 'H') if method == 'H-pooled-HGB' else np.arange(len(groups))
        selected_path = folder/'selection.json'
        if not selected_path.exists():
            rows = []
            for c in [.01, .1, 1.]:
                for fold in range(5):
                    train, val = folds != fold, folds == fold
                    model = SharedCoordinateBoosting(c).fit(x[train][:, columns], y[train], people[train], domains[train])
                    scores = model.predict_proba(x[val][:, columns])
                    for domain in cfg['sources']:
                        use = domains[val] == domain
                        rows.append({'C': c, 'fold': fold, 'domain': domain, 'auroc': roc_auc_score(y[val][use], scores[use])})
            trials = pd.DataFrame(rows)
            quality = trials.groupby(['C', 'domain']).auroc.mean().groupby('C').mean()
            chosen = float(quality.idxmax())
            model = SharedCoordinateBoosting(chosen).fit(x[:, columns], y, people, domains)
            joblib.dump({'model': model, 'columns': columns}, folder/'model.joblib')
            trials.to_csv(folder/'source_cv.csv', index=False)
            write_json(selected_path, {'selected_C': chosen, 'source_auroc': float(quality.max()),
                                       'all_C': quality.to_dict(), 'source_people': len(np.unique(people))})
            print(json.dumps({'fit': method, 'selected_C': chosen, 'source_auroc': quality.max()}), flush=True)
        models[method] = {'path': str(folder/'model.joblib'), 'sha256': sha(folder/'model.joblib')}
    write_json(output/'frozen_models.json', {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg, 'source_features': hashes, 'models': models,
        'protocol_sha256': sha(ROOT/'configs/daily_protocol.json'),
        'adapter_sha256': sha(ROOT/'scripts/prepare_ssaqs.py')})


def evaluate(output, cfg):
    frozen = json.loads((output/'frozen_models.json').read_text())
    feature = ROOT/cfg.get('feature_directory', 'data/features/ssaqs')
    if not (feature/'external.csv').exists():
        print((ROOT/cfg.get('study_directory', 'data/ssaqs_study')/'summary.json').read_text())
        return
    frame = pd.read_csv(feature/'external.csv', dtype={'person_id': str})
    with np.load(feature/'external.npz') as z:
        pooled, windows = z['pooled'], z['windows']
    for method in cfg['methods']:
        if method in frozen['models']:
            bundle = joblib.load(frozen['models'][method]['path'])
            frame[method] = bundle['model'].predict_proba(pooled[:, bundle['columns']])
        elif method == 'CNM-GC-past32':
            from src.hne_directed import cnm_score
            frame[method] = [cnm_score(w)['score'] for w in windows]
        else:
            frame[method] = native_scores({'windows': windows, 'state_dimension': cfg['state_count']}, method, 3).score.to_numpy()
    frame.to_csv(output/'predictions.csv', index=False)
    rows, differences = [], []
    common = np.isfinite(frame[cfg['methods']]).all(axis=1)
    for population, mask in [('all', np.ones(len(frame), bool)), ('common', common)]:
        part = frame[mask]
        if not len(part):
            continue
        count = part.person_id.nunique()
        boot = cfg['bootstrap']
        draws = np.random.default_rng(boot['seed']).multinomial(count, np.full(count, 1/count), boot['replicates'])
        results = {}
        for method in cfg['methods']:
            point, sampling, counts = cluster_statistics(part.y, part[method], part.person_id, None, draws)
            results[method] = point, sampling
            row = {'population': population, 'method': method, **counts}
            for k, name in enumerate(['auroc', 'within_person_auroc', 'average_precision']):
                low, high = np.nanquantile(sampling[:, k], [.025, .975])
                row.update({name: point[k], name+'_ci_low': low, name+'_ci_high': high})
            rows.append(row)
        for comparator in cfg['comparators']+['strongest-control']:
            point, samples = results[cfg['primary']]
            if comparator == 'strongest-control':
                strongest = max(cfg['comparators'], key=lambda m: results[m][0][0])
                comparison = np.nanmax(np.stack([results[m][1][:, 0] for m in cfg['comparators']]), 0)
            else:
                strongest = comparator
                comparison = results[strongest][1][:, 0]
            low, high = np.nanquantile(samples[:, 0]-comparison, [.025, .975])
            differences.append({'population': population, 'target': cfg['primary'], 'comparator': comparator,
                                'strongest': strongest, 'delta': point[0]-results[strongest][0][0],
                                'ci_low': low, 'ci_high': high})
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(differences).to_csv(output/'paired_differences.csv', index=False)
    print(pd.DataFrame(rows)[['population', 'method', 'origins', 'positive', 'auroc', 'within_person_auroc']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['fit', 'evaluate'], required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT/'configs/ssaqs_external.json').read_text())
    output = ROOT/'results/ssaqs_external'
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    with threadpool_limits(limits=2):
        if args.phase == 'fit':
            fit(output, cfg)
        else:
            evaluate(output, cfg)
