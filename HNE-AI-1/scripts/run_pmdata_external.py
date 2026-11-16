"""Reuse the frozen source models on a further longitudinal cohort."""
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_daily_study import write_person, summarize_people
from scripts.run_transfer import sha, write_json
from scripts.run_ssaqs_external import evaluate
from src.hne_representation import Sampling, encode_person


def daily_json(path, transform):
    if not path.exists():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    rows = json.loads(path.read_text())
    values = [(pd.Timestamp(r['dateTime']).normalize(), transform(r['value'])) for r in rows]
    return pd.Series([v for _, v in values], index=[d for d, _ in values], dtype=float).groupby(level=0).mean()


def person_arrays(folder, cfg):
    sleep = json.loads((folder/'fitbit/sleep.json').read_text())
    rows = []
    for record in sleep:
        if not record['mainSleep']:
            continue
        start = pd.Timestamp(record['startTime'])
        hour = start.hour+start.minute/60+start.second/3600
        summary = record['levels']['summary']
        rows.append({'day': pd.Timestamp(record['dateOfSleep']), 'sleep_minutes': record['minutesAsleep'],
            'awake_minutes': record['minutesAwake'], 'deep_minutes': summary.get('deep', {}).get('minutes', np.nan),
            'rem_minutes': summary.get('rem', {}).get('minutes', np.nan),
            'sleep_start_sin': np.sin(2*np.pi*hour/24), 'sleep_start_cos': np.cos(2*np.pi*hour/24)})
    sleeping = pd.DataFrame(rows).groupby('day').mean()
    heart = daily_json(folder/'fitbit/resting_heart_rate.json', lambda v: float(v['value']))
    wellness = pd.read_csv(folder/'pmsys/wellness.csv')
    wellness['day'] = pd.to_datetime(wellness.effective_time_frame, utc=True).dt.tz_localize(None).dt.floor('D')
    report = wellness.groupby('day')[['fatigue', 'mood', 'stress', 'soreness']].mean()
    activities = {name: daily_json(folder/f'fitbit/{name}.json', float) for name in cfg['context_names']}
    tables = [sleeping, heart, report, *activities.values()]
    grid = pd.date_range(min(t.index.min() for t in tables if len(t)), max(t.index.max() for t in tables if len(t)), freq='D')
    states = pd.DataFrame(np.nan, index=grid, columns=cfg['state_names'])
    states[sleeping.columns] = sleeping.reindex(grid)
    states['resting_heart_rate'] = heart.reindex(grid)
    states[report.columns] = report.reindex(grid)
    contexts = pd.DataFrame({name: values.reindex(grid) for name, values in activities.items()}, index=grid)
    state, context = states.to_numpy(), contexts.to_numpy()
    dates = grid.to_numpy(dtype='datetime64[D]')
    return {'states': state, 'contexts': context, 'dates': dates,
            'available_at': dates.astype('datetime64[s]')+np.timedelta64(1, 'D'),
            'state_observed_counts': np.isfinite(state).astype(np.int16),
            'context_observed_counts': np.isfinite(context).astype(np.int16),
            'state_names': np.asarray(cfg['state_names']), 'context_names': np.asarray(cfg['context_names']),
            'state_families': np.asarray(cfg['state_families']), 'timezone_semantics': np.asarray(cfg['source_clock'])}


def run():
    cfg = json.loads((ROOT/'configs/pmdata_external.json').read_text())
    source = ROOT / cfg['raw_root']
    output = ROOT/'results/pmdata_external'
    output.mkdir(parents=True, exist_ok=True)
    source_frozen = ROOT/'results/ssaqs_external/frozen_models.json'
    frozen = json.loads(source_frozen.read_text())
    write_json(output/'frozen_models.json', {**frozen, 'config': cfg, 'parent_source_freeze': str(source_frozen),
                                           'parent_sha256': sha(source_frozen), 'adapter_sha256': sha(Path(__file__))})
    feature, study = ROOT/cfg['feature_directory'], ROOT/cfg['study_directory']
    feature.mkdir(parents=True, exist_ok=True)
    study.mkdir(parents=True, exist_ok=True)
    if not (feature/'external.npz').exists():
        protocol = json.loads((ROOT/'configs/daily_protocol.json').read_text())
        summaries, origins, nodes, pooled, windows = [], [], [], [], []
        for person in sorted(source.glob('p[0-9][0-9]')):
            arrays = person_arrays(person, cfg)
            _, accepted, summary = write_person(study, 'PMData', person.name, 'external', -1, arrays,
                                                protocol, sha(ROOT/'configs/daily_protocol.json'))
            summaries.append(summary)
            if accepted:
                with np.load(study/f'{person.name}_daily.npz') as data:
                    states, contexts = data['normalized'], data['normalized_contexts']
                cutoffs = np.asarray([r['origin_day']+1 for r in accepted])
                n, x, groups = encode_person(states, contexts, cutoffs, Sampling.daily())
                nodes.extend(n);pooled.extend(x);origins.extend(accepted)
                joined = np.column_stack((states, contexts))
                windows.extend([joined[t-32:t] for t in cutoffs])
            print(json.dumps(summary), flush=True)
        people = pd.DataFrame(summaries)
        people.to_csv(study/'people.csv', index=False)
        write_json(study/'summary.json', summarize_people(people))
        if origins:
            frame = pd.DataFrame(origins)
            frame.to_csv(feature/'external.csv', index=False)
            np.savez_compressed(feature/'external.npz', nodes=np.stack(nodes), pooled=np.stack(pooled),
                                windows=np.stack(windows), groups=groups, y=frame.y.to_numpy())
    evaluate(output, cfg)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    with threadpool_limits(limits=2):
        run()
