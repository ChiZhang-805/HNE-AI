"""Adapt previously exposed IFH daily observations and evaluate fixed HNE heads.

--phase build creates only IFH daily data/features. --phase evaluate runs the
six prespecified learned comparisons and native DNB; no public neural baseline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_daily_study import write_person, summarize_people
from scripts.run_development import run as run_development
from scripts.summarize_development import fold_auc, bootstrap_aucs
from src.hne_representation import Sampling, encode_person, select_reference_channels
from src.dnb import score_window

DATASET = 'IFH-Affect'
ZONE = 'America/Los_Angeles'
SEED = 20261110
STATES = ['hr_average', 'hr_lowest', 'rmssd', 'breath_average', 'total', 'efficiency',
          'onset_latency', 'deep', 'rem', 'temperature_delta']
FAMILIES = ['heart', 'heart', 'hrv', 'respiration', 'sleep', 'sleep', 'sleep', 'sleep', 'sleep', 'temperature']
CONTEXTS = ['steps', 'cal_active', 'average_met', 'inactive', 'medium', 'high']
UNITS = ['beats/min', 'beats/min', 'milliseconds', 'breaths/min', 'seconds', 'percent',
         'seconds', 'seconds', 'seconds', 'degrees Celsius deviation',
         'count', 'kcal', 'MET', 'minutes', 'minutes', 'minutes']
FIELD_MAPPING = {
    name: {'source_file': 'oura/sleep.csv' if i < len(STATES) else 'oura/activity.csv',
           'source_column': name, 'role': 'state' if i < len(STATES) else 'behavior_context',
           'unit': UNITS[i], **({'family': FAMILIES[i]} if i < len(STATES) else {})}
    for i, name in enumerate(STATES + CONTEXTS)
}
MODELS = ['HNE-additive', 'HNE-history', 'HNE-no-network', 'HNE-no-environment', 'History-HGB', 'History-LR']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_completed(path, modality):
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame['source_row'] = np.arange(len(frame))
    start = 'bedtime_start_timestamp' if modality == 'sleep' else 'day_start_timestamp'
    end = 'bedtime_end_timestamp' if modality == 'sleep' else 'day_end_timestamp'
    frame['start_utc'] = pd.to_datetime(pd.to_numeric(frame[start], errors='coerce'), unit='ms', utc=True)
    frame['end_utc'] = pd.to_datetime(pd.to_numeric(frame[end], errors='coerce'), unit='ms', utc=True)
    frame = frame[frame.start_utc.notna() & frame.end_utc.notna() & frame.end_utc.gt(frame.start_utc)].copy()
    frame['local_day'] = frame.end_utc.dt.tz_convert(ZONE).dt.tz_localize(None).dt.normalize()
    if modality == 'sleep':
        frame['total'] = pd.to_numeric(frame.total, errors='coerce')
        frame = frame[frame.total.gt(0)]
        order = ['local_day', 'total', 'end_utc', 'source_row']
    else:
        order = ['local_day', 'end_utc', 'start_utc', 'source_row']
    return frame.sort_values(order, kind='stable').drop_duplicates('local_day', keep='last').set_index('local_day')


def daily_arrays(source, person):
    frames = {name: read_completed(source/person/'oura'/f'{name}.csv', name) for name in ['sleep', 'activity']}
    all_days = [d for frame in frames.values() for d in frame.index]
    if not all_days:
        raise ValueError(f'No completed record for {person}')
    dates = pd.date_range(min(all_days), max(all_days), freq='D')
    next_midnight = dates + pd.DateOffset(days=1)
    availability_utc = next_midnight.tz_localize(ZONE).tz_convert('UTC').tz_localize(None).to_numpy(dtype='datetime64[s]')
    values = []
    source_end = np.full((len(dates), 2), np.datetime64('NaT', 'ms'), dtype='datetime64[ms]')
    selections = []
    for j, (name, columns) in enumerate([('sleep', STATES), ('activity', CONTEXTS)]):
        frame = frames[name]
        if frame.empty:
            values.append(np.full((len(dates), len(columns)), np.nan))
            continue
        x = frame[columns].apply(pd.to_numeric, errors='coerce').reindex(dates).to_numpy(float)
        x[~np.isfinite(x)] = np.nan
        values.append(x)
        end = frame.end_utc.reindex(dates).dt.tz_localize(None).to_numpy(dtype='datetime64[ms]')
        source_end[:, j] = end
        observed = ~np.isnat(end)
        if not np.all(end[observed] < availability_utc[observed]):
            raise ValueError('A source summary is unavailable at its assigned daily issue time')
        for day, row in frame.iterrows():
            at = (day + pd.DateOffset(days=1)).tz_localize(ZONE).tz_convert('UTC')
            selections.append({'person_id': person, 'modality': name, 'source_row': int(row.source_row),
                'source_date': str(row.date), 'assigned_local_date': str(day.date()),
                'start_utc': str(row.start_utc), 'end_utc': str(row.end_utc), 'available_at_utc': str(at)})
    state, context = values
    return {'states': state, 'contexts': context,
        'state_observed_counts': np.isfinite(state).astype(np.int16),
        'context_observed_counts': np.isfinite(context).astype(np.int16),
        'dates': dates.to_numpy(dtype='datetime64[D]'),
        # The shared calendar-day protocol compares clocks on one coordinate
        # system. Keep LA wall time here and retain the true UTC instant alongside.
        'available_at': next_midnight.to_numpy(dtype='datetime64[s]'),
        'available_at_utc': availability_utc, 'selected_source_end_utc': source_end,
        'state_names': np.asarray(STATES), 'context_names': np.asarray(CONTEXTS),
        'state_families': np.asarray(FAMILIES),
        'timezone_semantics': np.asarray('America/Los_Angeles local-calendar dates and wall-clock available_at/issue_time; separate *_utc stores actual instants')}, selections


def worker_init():
    global _limit
    _limit = threadpool_limits(limits=1)
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def build_person(task):
    source, output, person, fold, config, protocol_sha = task
    source, output = Path(source), Path(output)
    arrays, selections = daily_arrays(source, person)
    origins, samples, summary = write_person(output, DATASET, person, 'development', fold, arrays, config, protocol_sha)
    for row in origins:
        row['issue_time_utc'] = str(pd.Timestamp(row['issue_time']).tz_localize(ZONE).tz_convert('UTC'))
        row['future_start_utc'] = str(pd.Timestamp(row['future_start_date']).tz_localize(ZONE).tz_convert('UTC'))
        row['future_end_exclusive_utc'] = str((pd.Timestamp(row['future_end_date'])+pd.DateOffset(days=1)).tz_localize(ZONE).tz_convert('UTC'))
    # Add a UTC counterpart of the saved issue array without replacing its
    # local-calendar coordinate used by the unchanged endpoint implementation.
    path = output/f'{person}_daily.npz'
    with np.load(path, allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files}
    issue = pd.DatetimeIndex(payload['issue_time']).tz_localize(ZONE).tz_convert('UTC').tz_localize(None)
    payload['issue_time_utc'] = issue.to_numpy(dtype='datetime64[s]')
    np.savez_compressed(path, **payload)
    if not samples:
        return origins, samples, summary, selections, None
    states = payload['normalized']
    selected = select_reference_channels(states, payload['state_names'], reference=56)
    states, contexts = states[:, selected], payload['normalized_contexts']
    cutoffs = np.asarray([r['origin_day']+1 for r in samples])
    nodes, pooled, groups = encode_person(states, contexts, cutoffs, Sampling.daily())
    trajectory = np.column_stack([states, contexts]).astype(np.float32)
    windows = np.stack([trajectory[t-32:t] for t in cutoffs])
    return origins, samples, summary, selections, (nodes, pooled, groups, windows)


def build(source, output, features, workers):
    people = sorted(p.name for p in source.glob('par_*') if p.is_dir())
    if len(people) != 24:
        raise ValueError('The frozen raw roster must contain all24 archive IDs')
    ordered = sorted(people, key=lambda p: hashlib.sha256(f'{SEED}:{DATASET}:{p}'.encode()).hexdigest())
    folds = {p: i % 5 for i, p in enumerate(ordered)}
    protocol_path = ROOT/'configs/daily_protocol.json'
    config, protocol_sha = json.loads(protocol_path.read_text()), sha(protocol_path)
    output.mkdir(parents=True, exist_ok=True); features.mkdir(parents=True, exist_ok=True)
    schema = {'dataset': DATASET, 'stage': 'previously_exposed_development', 'states': STATES,
        'state_families': FAMILIES, 'contexts': CONTEXTS, 'EMA_used': False,
        'field_mapping': FIELD_MAPPING,
        'official_dataset': 'https://datadryad.org/dataset/doi:10.7280/D1WH6T',
        'official_article_doi': '10.1016/j.dib.2024.110228',
        'source_manifest': 'data/new_cohorts/ifh-affect/source_manifest.json',
        'calendar': ZONE, 'assignment': 'date of actual end epoch converted to America/Los_Angeles; source date labels are not used for alignment',
        'availability': 'next local midnight after the completed record; UTC counterpart preserves DST',
        'sleep_same_day': 'main sleep=max total sleep seconds; ties later end then later source row',
        'activity_same_day': 'latest completed summary by end then start; do not sum overlapping vendor-day summaries',
        'activity_period': 'raw Oura activity intervals generally span about04:00 to03:59 the following local day; assigned to natural day containing actual completion',
        'missing': 'complete local-date grid; no interpolation or fill; nonfinite numeric values stay NaN; true0 and signed temperature deviation retained',
        'endpoint_clock': 'existing daily_protocol runs in naive LA wall-clock calendar coordinates; original issue_time is local wall time and issue_time_utc is the actual instant',
        'gap': 'one complete local calendar day; future days index d+2 through d+8 inclusive',
        'folds': folds, 'fold_seed': SEED, 'fold_roster': 'all24 raw IDs, including short or unscoreable records',
        'fold_assignment': 'sort sha256(seed:IFH-Affect:person), rank modulo5; heldf/validation(f+1)%5/trainremaining3',
        'protocol_sha256': protocol_sha, 'feature_core': 'unchanged encode_person(Sampling.daily()), default context_ridge10',
        'source_files': {str(p.relative_to(source)): sha(p) for p in sorted(source.glob('par_*/oura/*.csv'))},
        'implementation_hashes': {n: sha(ROOT/n) for n in ['src/daily_protocol.py', 'src/hne_representation.py', 'src/hne_features.py']}}
    dump(output/'schema_mapping.json', schema)
    pd.DataFrame([{'person_id': p, 'split': 'development', 'fold': folds[p]} for p in people]).to_csv(output/'person_folds.csv', index=False)
    tasks = [(str(source), str(output), p, folds[p], config, protocol_sha) for p in people]
    origins, samples, summaries, chosen = [], [], [], []
    node, pooled, windows = [], [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for o, s, summary, selections, arrays in pool.map(build_person, tasks):
            origins.extend(o); samples.extend(s); summaries.append(summary); chosen.extend(selections)
            if arrays is not None:
                n, p, groups, w = arrays; node.extend(n); pooled.extend(p); windows.extend(w)
            print(json.dumps(summary), flush=True)
    meta = pd.DataFrame(samples); persons = pd.DataFrame(summaries)
    pd.DataFrame(origins).to_csv(output/'origins.csv', index=False)
    meta.to_csv(output/'samples.csv', index=False); persons.to_csv(output/'people.csv', index=False)
    pd.DataFrame(chosen).to_csv(output/'selected_observations.csv', index=False)
    if not len(meta):
        raise ValueError('No eligible IFH origins under the unchanged daily protocol')
    np.savez_compressed(features/f'{DATASET}.npz', nodes=np.stack(node), pooled=np.stack(pooled),
        windows=np.stack(windows), groups=groups, y=meta.y.to_numpy(np.int8), fold=meta.fold.to_numpy(np.int8))
    meta.to_csv(features/f'{DATASET}.csv', index=False)
    summary = {**summarize_people(persons), 'positive_rate': float(meta.y.mean()), 'state_dimension': 10,
        'context_dimension': 6, 'protocol_sha256': protocol_sha, 'schema_sha256': sha(output/'schema_mapping.json'),
        'feature_sha256': sha(features/f'{DATASET}.npz'), 'per_fold': {
            str(f): {'people': int(m.person_id.nunique()), 'origins': len(m), 'positive': int(m.y.sum())}
            for f, m in meta.groupby('fold')}, 'stage': 'previously_exposed_development'}
    dump(output/'summary.json', summary)
    print('READY', json.dumps(summary), flush=True)


def evaluate(features, results):
    run_development(MODELS, list(range(5)), features, results, [DATASET], scales=(8,16,32))
    with np.load(features/f'{DATASET}.npz', allow_pickle=False) as z:
        windows = z['windows']
    meta = pd.read_csv(features/f'{DATASET}.csv', dtype={'person_id': str})
    score = np.asarray([score_window(w[:, :10], minimum=12) for w in windows])
    for fold in range(5):
        folder = results/DATASET/'DNB'/f'fold{fold}'; folder.mkdir(parents=True, exist_ok=True)
        val, held = meta.fold.eq((fold+1)%5), meta.fold.eq(fold)
        for name, ix in [('train', ~(val|held)), ('validation', val), ('held_person', held)]:
            frame = meta[ix].copy(); frame['score'] = score[ix]
            frame.to_csv(folder/f'{name}.csv', index=False)
        dump(folder/'metadata.json', {'history_days': 32, 'minimum_joint_observations': 12,
            'direction': 'higher native DNB score indicates warning; no label-based selection', 'score_is_probability': False})
    summarize(meta, results)


def summarize(meta, results):
    folder = results/DATASET/'summary'; folder.mkdir(parents=True, exist_ok=True)
    merged = meta.copy(); keys = ['person_id', 'origin_day']; selections = []
    for name in [*MODELS, 'DNB']:
        parts = []
        for fold in range(5):
            base = results/DATASET/name/f'fold{fold}'
            h, v = pd.read_csv(base/'held_person.csv'), pd.read_csv(base/'validation.csv')
            col = 'score' if name == 'DNB' else 'probability'
            finite_val = np.sort(v[col].dropna())
            h[name] = h[col]
            h[name+'__rank'] = np.searchsorted(finite_val, h[col], side='right')/len(finite_val)
            parts.append(h[keys+[name, name+'__rank']])
            if name != 'DNB':
                info = json.loads((base/'metrics.json').read_text())
                selections.append({'method': name, 'fold': fold, 'C': info['chosen']['C'],
                    'validation_auroc': info['validation']['auroc'], 'held_auroc': info['held_person']['auroc']})
        merged = merged.merge(pd.concat(parts), on=keys, validate='one_to_one')
    common = np.isfinite(merged[[*MODELS, 'DNB']]).all(axis=1)
    merged['common'] = common; merged.to_csv(folder/'predictions.csv', index=False)
    rows, paired = [], []
    for population, selected in [('all', merged), ('common', merged[common])]:
        people, inverse = np.unique(selected.person_id, return_inverse=True)
        weights = np.random.default_rng(SEED).multinomial(len(people), np.full(len(people), 1/len(people)), size=2000)
        boot = {name: bootstrap_aucs(selected, name, inverse, weights) for name in [*MODELS, 'DNB']}
        for name in boot:
            valid = selected[np.isfinite(selected[name])]
            within = [roc_auc_score(g.y, g[name]) for _, g in valid.groupby('person_id') if g.y.nunique()==2]
            low, high = np.nanquantile(boot[name], [.025,.975])
            rows.append({'dataset': DATASET, 'population': population, 'method': name, 'origins': len(valid),
                'people': valid.person_id.nunique(), 'positive': int(valid.y.sum()), 'auroc': fold_auc(valid, name),
                'ci_low': low, 'ci_high': high, 'within_person_auroc': float(np.mean(within)) if within else None,
                'ap_on_validation_rank': average_precision_score(valid.y, valid[name+'__rank'])})
        if population == 'common':
            for baseline in [*MODELS[1:], 'DNB']:
                delta = boot['HNE-additive']-boot[baseline]
                low, high = np.nanquantile(delta, [.025,.975])
                paired.append({'model': 'HNE-additive', 'baseline': baseline,
                    'difference': fold_auc(selected, 'HNE-additive')-fold_auc(selected, baseline), 'ci_low': low, 'ci_high': high})
    pd.DataFrame(rows).to_csv(folder/'metrics.csv', index=False)
    pd.DataFrame(selections).to_csv(folder/'validation_selections.csv', index=False)
    pd.DataFrame(paired).to_csv(folder/'paired_differences.csv', index=False)
    print(pd.DataFrame(rows).query("population=='common'").to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['build', 'evaluate', 'all'], default='build')
    parser.add_argument('--source', type=Path, default=ROOT/'data/new_cohorts/ifh-affect/selected-files')
    parser.add_argument('--output', type=Path, default=ROOT/'data/expanded_development/IFH-Affect')
    parser.add_argument('--features', type=Path, default=ROOT/'data/features/expanded_development')
    parser.add_argument('--results', type=Path, default=ROOT/'results/expanded_development')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    with threadpool_limits(limits=2):
        if args.phase in ['build', 'all']:
            build(args.source, args.output, args.features, args.workers)
        if args.phase in ['evaluate', 'all']:
            evaluate(args.features, args.results)


if __name__ == '__main__':
    main()
