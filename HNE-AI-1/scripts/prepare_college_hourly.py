"""Read actual College hourly columns for the same allowed participant splits."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/new_cohorts/college-experience'
OUTPUT = ROOT / 'data/college_hourly'
BASES = ['act_still', 'act_in_vehicle', 'act_on_bike', 'act_on_foot', 'act_walking',
         'act_running', 'unlock_duration', 'unlock_num', 'loc_dist', 'loc_visit_num',
         'loc_max_dis_from_campus']
FIELDS = [f'{base}_hr_{h}' for base in BASES for h in range(24)]
QUALITY = ['quality_activity', 'quality_audio', 'quality_light', 'quality_loc']
NAMES = ['still_duration', 'vehicle_duration', 'bike_duration', 'foot_activity',
         'unlock_duration', 'unlock_num', 'distance', 'visits']
FAMILIES = ['activity']*4+['device_rest']*2+['mobility']*2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation'])
    args = parser.parse_args()
    if 'test' in args.splits:
        json.loads((ROOT / 'results/college_experience/frozen_models.json').read_text())
    assignment = pd.read_csv(SOURCE / 'subject_split.csv', dtype={'person_id': str})
    assignment = assignment[assignment.split.isin(args.splits)]
    split_map = dict(zip(assignment.person_id, assignment.split))
    wanted = [p for p in split_map if not (OUTPUT / split_map[p] / f'{p}.npz').exists()]
    records = {p: [] for p in wanted}
    if not records:
        print('Requested College hourly files are already available.', flush=True)
        return
    with (SOURCE / 'Sensing/sensing.csv').open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            person = row['uid']
            if person in records:
                fields = ['is_ios', *QUALITY, *FIELDS]
                records[person].append([row['day'], *[float(row[c]) if row[c] else np.nan for c in fields]])
    coverage = []
    for person, rows in records.items():
        frame = pd.DataFrame(rows, columns=['day', 'is_ios', *QUALITY, *FIELDS])
        frame.index = pd.to_datetime(frame.pop('day'), format='%Y%m%d')
        dates = pd.date_range(frame.index.min(), frame.index.max(), freq='D')
        frame = frame.reindex(dates)
        raw = frame[FIELDS].to_numpy().reshape(len(frame), len(BASES), 24).transpose(0, 2, 1)
        ios = frame.is_ios.to_numpy()[:, None]
        foot = np.where(ios == 1, raw[:, :, 4]+raw[:, :, 5], np.where(ios == 0, raw[:, :, 3], np.nan))
        states = np.stack([raw[:, :, 0], raw[:, :, 1], raw[:, :, 2], foot,
                           raw[:, :, 6], raw[:, :, 7], raw[:, :, 8], raw[:, :, 9]], axis=-1)
        quality = frame[QUALITY].to_numpy()
        q_any = np.max(np.where(np.isfinite(quality), quality, -np.inf), axis=1)
        quality_by_state = np.column_stack([quality[:, 0]]*4+[q_any]*2+[quality[:, 3]]*2)
        states[(states < 0) | ~(quality_by_state[:, None, :] >= 4)] = np.nan
        context = raw[:, :, 10:11].copy()
        context[(context < 0) | ~(quality[:, None, 3:4] >= 4)] = np.nan
        folder = OUTPUT / split_map[person]
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f'{person}.npz', states=states.reshape(-1, len(NAMES)).astype(np.float32),
            contexts=context.reshape(-1, 1).astype(np.float32), state_names=NAMES, state_families=FAMILIES,
            context_names=['max_campus_distance'], dates=dates.to_numpy(dtype='datetime64[D]'))
        coverage.append({'person_id': person, 'split': split_map[person], 'days': len(dates),
                         'observed_state_values': int(np.isfinite(states).sum()),
                         'observed_context_values': int(np.isfinite(context).sum())})
    pd.DataFrame(coverage).to_csv(OUTPUT / ('_'.join(args.splits)+'_coverage.csv'), index=False)
    print(json.dumps({'prepared': len(coverage), 'state_channels': len(NAMES), 'context_channels': 1,
                      'splits': args.splits, 'observed_state_values': sum(r['observed_state_values'] for r in coverage)}), flush=True)


if __name__ == '__main__':
    main()
