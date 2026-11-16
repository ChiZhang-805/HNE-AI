"""Evaluate the original frozen source HNE on all reserved TOLIFE participants."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_tolife_daily import prepare
from scripts.build_daily_study import tolife_daily, write_person, summarize_people
from scripts.run_hne_external import native_scores, cluster_statistics
from scripts.run_transfer import sha, write_json
from src.hne_representation import Sampling, encode_person


def main():
    output = ROOT / 'results/tolife_external'
    output.mkdir(exist_ok=True)
    source = ROOT / 'data/new_cohorts/tolife-csa-2026'
    daily = source / 'daily_test'
    feature = ROOT / 'data/tolife_external'
    feature.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / 'configs/tolife_external.json').read_text())
    protocol = json.loads((ROOT / 'configs/daily_protocol.json').read_text())
    source_folder = ROOT / 'results/transfer/source_models/Humans'
    bundles = {m: joblib.load(source_folder / m / 'model.joblib') for m in cfg['methods'][:2]}
    freeze = output / 'frozen_models.json'
    if not freeze.exists():
        write_json(freeze, {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
                           'models': {m: sha(source_folder / m / 'model.joblib') for m in bundles},
                           'protocol_sha256': sha(ROOT / 'configs/daily_protocol.json'),
                           'test_constructed': False})
    prepare(source, splits=('test',), output=daily)
    mapping = json.loads((daily / 'schema_mapping.json').read_text())
    people = pd.read_csv(source / 'subject_split.csv').query("split == 'test'")
    samples, summaries, pooled, windows = [], [], [], []
    for person in people['Kit-ID']:
        arrays = tolife_daily(daily / f'{person}.csv', mapping, allowed_splits=('test',))
        _, accepted, summary = write_person(feature, 'TOLIFE-CSA', person, 'test', -1,
                                             arrays, protocol, sha(ROOT / 'configs/daily_protocol.json'))
        summaries.append(summary)
        if not accepted:
            continue
        with np.load(feature / f'{person}_daily.npz') as z:
            states, contexts = z['normalized'], z['normalized_contexts']
        cutoffs = np.asarray([r['origin_day']+1 for r in accepted])
        _, x, _ = encode_person(states, contexts, cutoffs, Sampling.daily())
        pooled.extend(x)
        trajectory = np.column_stack((states, contexts))
        windows.extend([trajectory[t-32:t] for t in cutoffs])
        samples.extend(accepted)
    summary = summarize_people(pd.DataFrame(summaries))
    pd.DataFrame(summaries).to_csv(feature / 'people.csv', index=False)
    write_json(feature / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    if not samples:
        return
    frame = pd.DataFrame(samples)
    x, windows = np.stack(pooled), np.stack(windows)
    frame.to_csv(feature / 'test.csv', index=False)
    np.savez_compressed(feature / 'test.npz', pooled=x, windows=windows, y=frame.y.to_numpy())
    calibration_meta = pd.read_csv(ROOT / 'data/features/unified_daily/TOLIFE-CSA.csv')
    with np.load(ROOT / 'data/features/unified_daily/TOLIFE-CSA.npz') as z:
        calibration = {'pooled': z['pooled'], 'windows': z['windows'], 'state_dimension': 32}
    thresholds = {}
    for method in cfg['methods']:
        if method in bundles:
            bundle = bundles[method]
            frame[method] = bundle['model'].predict_proba(x[:, bundle['columns']])
            calibration_score = bundle['model'].predict_proba(calibration['pooled'][:, bundle['columns']])
        else:
            frame[method] = native_scores({'windows': windows, 'state_dimension': 32}, method, 3).score.to_numpy()
            calibration_score = native_scores(calibration, method, 3).score.to_numpy()
        negatives = calibration_score[calibration_meta.y.eq(0) & np.isfinite(calibration_score)]
        thresholds[method] = float(np.quantile(negatives, .9)) if len(negatives) else None
    frame.to_csv(output / 'test_predictions.csv', index=False)
    write_json(output / 'thresholds.json', thresholds)
    rows, pairs = [], []
    common = np.isfinite(frame[cfg['methods']]).all(1)
    for population, keep in [('all', np.ones(len(frame), bool)), ('common', common)]:
        part = frame.loc[keep]
        if part.empty:
            continue
        n = part.person_id.nunique()
        draws = np.random.default_rng(cfg['bootstrap']['seed']).multinomial(n, np.full(n, 1/n), cfg['bootstrap']['replicates'])
        results = {}
        for method in cfg['methods']:
            point, boot, counts = cluster_statistics(part.y, part[method], part.person_id, thresholds[method], draws)
            results[method] = (point, boot)
            row = {'population': population, 'method': method, **counts}
            for k, metric in enumerate(['auroc', 'within_person_auroc', 'ap', 'tpr_training_p90', 'fpr_training_p90']):
                low, high = np.nanquantile(boot[:, k], [.025, .975])
                row.update({metric: point[k], metric+'_ci_low': low, metric+'_ci_high': high})
            rows.append(row)
        for control in cfg['primary_comparators']:
            point, boot = results['HNE-additive']
            base, base_boot = results[control]
            low, high = np.nanquantile(boot[:, 0]-base_boot[:, 0], [.025, .975])
            pairs.append({'population': population, 'model': 'HNE-additive', 'baseline': control,
                          'difference': point[0]-base[0], 'ci_low': low, 'ci_high': high})
    pd.DataFrame(rows).to_csv(output / 'test_metrics.csv', index=False)
    pd.DataFrame(pairs).to_csv(output / 'test_paired_differences.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    with threadpool_limits(limits=2):
        main()
