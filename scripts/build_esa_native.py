"""Future multivariate-subsequence warning from ESA's expert event annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.daily_protocol import robust_reference, normalize
from src.hne_representation import Sampling, encode_person, select_reference_channels


def event_table(mission, boundary):
    folder = ROOT / f'data/new_cohorts/esa-ad-v2/metadata/ESA-Mission{mission}'
    labels = pd.read_csv(folder / 'labels.csv')
    labels['start'] = pd.to_datetime(labels.StartTime, utc=True).dt.tz_localize(None)
    labels['end'] = pd.to_datetime(labels.EndTime, utc=True).dt.tz_localize(None)
    labels = labels[labels.start < pd.Timestamp(boundary)]
    events = labels.groupby('ID').agg(start=('start', 'min'), end=('end', 'max'), channels=('Channel', 'nunique')).reset_index()
    return events.merge(pd.read_csv(folder / 'anomaly_types.csv'), on='ID', validate='one_to_one')


def build(mission, phase, mapping, cfg):
    name = f'ESA-Mission{mission}'
    folder = ROOT / 'data/esa_native' / name
    folder.mkdir(parents=True, exist_ok=True)
    if phase == 'test':
        json.loads((ROOT / 'results/esa_native' / name / 'frozen_models.json').read_text())
    bounds = mapping['split'][str(mission)]
    stop = bounds['validation_end_exclusive'] if phase == 'fit' else bounds['end_exclusive']
    with np.load(ROOT / f'data/new_cohorts/esa-ad-v2/mission{mission}_daily.npz') as z:
        keep = z['dates'] < np.datetime64(stop)
        dates, states, contexts = z['dates'][keep], z['states'][keep], z['contexts'][keep]
        names = z['state_names']
    state_center, state_scale, _ = robust_reference(states, 56, 28)
    states = normalize(states, state_center, state_scale)
    selected = select_reference_channels(states, names)
    states = states[:, selected]
    context_center, context_scale, _ = robust_reference(contexts, 56, 28)
    contexts = normalize(contexts, context_center, context_scale)
    events = event_table(mission, stop)
    selected_events = events[(events.Category == 'Anomaly') & (events.Dimensionality == 'Multivariate') & (events.Length == 'Subsequence')]
    anomalies = events[events.Category == 'Anomaly']
    rare = events[events.Category == 'Rare Event']
    rows, excluded = [], {'past_anomaly': 0, 'gap_anomaly': 0, 'other_future_anomaly': 0}
    for day in range(cfg['first_origin_day'], len(dates)-8):
        issue = pd.Timestamp(dates[day])+pd.Timedelta(days=1)
        left, right = issue+pd.Timedelta(days=1), issue+pd.Timedelta(days=8)
        if ((anomalies.start < issue) & (anomalies.end >= issue-pd.Timedelta(days=7))).any():
            excluded['past_anomaly'] += 1
            continue
        if ((anomalies.start >= issue) & (anomalies.start < left)).any():
            excluded['gap_anomaly'] += 1
            continue
        positive = ((selected_events.start >= left) & (selected_events.start < right)).any()
        if not positive and ((anomalies.start >= left) & (anomalies.start < right)).any():
            excluded['other_future_anomaly'] += 1
            continue
        split = 'boundary'
        if right <= pd.Timestamp(bounds['train_end_exclusive']):
            split = 'train'
        elif issue >= pd.Timestamp(bounds['train_end_exclusive']) and right <= pd.Timestamp(bounds['validation_end_exclusive']):
            split = 'validation'
        elif issue >= pd.Timestamp(bounds['validation_end_exclusive']):
            split = 'test'
        if split not in (['train', 'validation'] if phase == 'fit' else ['test']):
            continue
        rows.append({'dataset': name, 'person_id': name, 'split': split, 'fold': -1, 'origin_day': day,
                     'issue_time': issue.isoformat(), 'origin_date': str(dates[day]),
                     'future_start_date': left.date().isoformat(),
                     'future_end_date': (right-pd.Timedelta(days=1)).date().isoformat(), 'y': int(positive),
                     'future_rare_event': int(((rare.start < right) & (rare.end >= left)).any())})
    meta = pd.DataFrame(rows)
    trajectory = np.column_stack((states, contexts)).astype(np.float32)
    counts = {}
    for split, frame in meta.groupby('split', sort=False):
        frame = frame.reset_index(drop=True)
        cutoffs = frame.origin_day.to_numpy()+1
        nodes, pooled, groups = encode_person(states, contexts, cutoffs, Sampling.daily())
        payload = {'nodes': nodes, 'pooled': pooled, 'groups': groups,
                   'windows': np.stack([trajectory[t-32:t] for t in cutoffs]),
                   'selected_states': selected, 'y': frame.y.to_numpy()}
        if split != 'test':
            boundary = int(np.searchsorted(dates, np.datetime64(bounds['train_end_exclusive'] if split == 'train' else bounds['validation_end_exclusive'])))
            available = cutoffs+32 <= boundary
            future = np.full((len(frame), 32, trajectory.shape[1]), np.nan, dtype=np.float32)
            for i in np.flatnonzero(available):
                future[i] = trajectory[cutoffs[i]:cutoffs[i]+32]
            payload.update(future=future, future_available=available)
        np.savez_compressed(folder / f'{split}.npz', **payload)
        frame.to_csv(folder / f'{split}.csv', index=False)
        counts[split] = {'origins': len(frame), 'positive': int(frame.y.sum()),
                         'rare_event_negative_origins': int(((frame.y == 0) & (frame.future_rare_event == 1)).sum())}
        print(json.dumps({'mission': mission, 'split': split, **counts[split], 'states': states.shape[1], 'contexts': contexts.shape[1]}), flush=True)
    record = {'mission': mission, 'phase': phase, 'partitions': counts, 'exclusions': excluded,
              'selected_event_count': len(selected_events), 'configuration': cfg,
              'selected_state_names': names[selected].tolist(), 'normalization': 'unchanged initial56-day reference'}
    (folder / f'{phase}_summary.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--missions', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--phase', choices=['fit', 'test'], default='fit')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/esa_native_event.json').read_text())
    mapping = json.loads((ROOT / 'configs/esa_mapping.json').read_text())
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    with threadpool_limits(limits=2):
        for mission in args.missions:
            build(mission, args.phase, mapping, cfg)
