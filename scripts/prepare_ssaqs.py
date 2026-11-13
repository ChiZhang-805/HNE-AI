"""Apply the fixed daily protocol to the public SSAQS archive."""
from pathlib import Path
import json
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_daily_study import write_person, summarize_people
from scripts.run_transfer import sha, write_json
from src.hne_representation import Sampling, encode_person

STATE_NAMES = [f'{name}__{stat}' for name in ['rmssd', 'low_frequency', 'high_frequency'] for stat in ['mean', 'sd']]+['sleep_score', 'deep_sleep_minutes', 'reported_stress', 'reported_anxiety']
CONTEXT_NAMES = ['steps', 'sedentary_fraction', 'light_fraction', 'moderate_fraction', 'vigorous_fraction', 'registered_class_minutes']
LEVELS = ['SEDENTARY', 'LIGHTLY_ACTIVE', 'MODERATELY_ACTIVE', 'VERY_ACTIVE']


def source_time(series, offset):
    # An explicit timezone marker carries UTC semantics; unmarked HRV is local.
    text = series.astype(str)
    explicit = text.str.contains(r'(?:Z|[+-]\d\d:\d\d)$', regex=True)
    parsed = pd.to_datetime(text, format='mixed', utc=True, errors='coerce').dt.tz_localize(None)
    return parsed + pd.to_timedelta(explicit.astype(int)*offset, unit='s')


def person_arrays(archive, person, courses, schedules):
    names = set(archive.namelist())
    tables = {}
    for name in ['hrv', 'sleep', 'steps', 'activity_level', 'daily_questions']:
        path = f'{person}/{name}.csv'
        if path in names:
            with archive.open(path) as stream:
                tables[name] = pd.read_csv(stream).drop_duplicates()
    questions = tables['daily_questions']
    offset = int(questions.timeZoneOffset.iloc[0])
    dates = []
    for name, frame in tables.items():
        if name == 'daily_questions':
            completed = pd.to_numeric(frame.timeStampStop, errors='coerce')
            frame['clock'] = pd.to_datetime(completed+frame.timeZoneOffset, unit='s', errors='coerce')
        else:
            frame['clock'] = source_time(frame.timestamp, offset)
        frame['day'] = frame.clock.dt.floor('D')
        dates.extend(frame.day.dropna().tolist())
    grid = pd.date_range(min(dates), max(dates), freq='D')
    states = pd.DataFrame(np.nan, index=grid, columns=STATE_NAMES)
    contexts = pd.DataFrame(np.nan, index=grid, columns=CONTEXT_NAMES)
    if 'hrv' in tables:
        hrv = tables['hrv']
        hrv = hrv[hrv.coverage.gt(0)].groupby('clock')[['rmssd', 'low_frequency', 'high_frequency']].mean()
        hrv['day'] = hrv.index.floor('D')
        for name in ['rmssd', 'low_frequency', 'high_frequency']:
            grouped = hrv.groupby('day')[name]
            counts, mean, sd = grouped.count(), grouped.mean(), grouped.std(ddof=0)
            states[name+'__mean'] = mean.where(counts >= 4).reindex(grid)
            states[name+'__sd'] = sd.where(counts >= 4).reindex(grid)
    if 'sleep' in tables:
        sleep = tables['sleep'].groupby('clock')[['overall_score', 'deep_sleep_in_minutes']].mean()
        sleep['day'] = sleep.index.floor('D')
        daily = sleep.groupby('day')[['overall_score', 'deep_sleep_in_minutes']].mean()
        states[['sleep_score', 'deep_sleep_minutes']] = daily.reindex(grid).to_numpy()
    q = questions.dropna(subset=['clock']).groupby('day')[['stress', 'anxiety']].mean()
    states[['reported_stress', 'reported_anxiety']] = q.reindex(grid).to_numpy()
    if 'steps' in tables:
        steps = tables['steps'].groupby('clock').steps.mean()
        contexts['steps'] = steps.groupby(steps.index.floor('D')).sum(min_count=1).reindex(grid)
    if 'activity_level' in tables:
        activity = tables['activity_level'].dropna(subset=['clock']).drop_duplicates('clock')
        valid = activity.level.isin(LEVELS)
        denominator = activity[valid].groupby('day').size()
        for level, column in zip(LEVELS, CONTEXT_NAMES[1:5]):
            numerator = activity[valid & activity.level.eq(level)].groupby('day').size()
            contexts[column] = numerator.reindex(denominator.index, fill_value=0).div(denominator).reindex(grid)
    scheduled = np.zeros(len(grid), float)
    for course in courses.loc[courses.userid.astype(str).eq(person), 'course']:
        row = schedules.loc[schedules.idcourse.eq(course)].iloc[0]
        start, end = pd.Timestamp(row.startDate), pd.Timestamp(row.endDate)
        duration = (pd.to_timedelta(row.endTime+':00')-pd.to_timedelta(row.startTime+':00')).total_seconds()/60
        active_days = [i for i, name in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) if row[name] == 1]
        scheduled += duration*((grid >= start) & (grid <= end) & grid.dayofweek.isin(active_days))
    contexts['registered_class_minutes'] = scheduled
    state_values, context_values = states.to_numpy(), contexts.to_numpy()
    date_values = grid.to_numpy(dtype='datetime64[D]')
    return {'states': state_values, 'contexts': context_values,
            'state_observed_counts': np.isfinite(state_values).astype(np.int16),
            'context_observed_counts': np.isfinite(context_values).astype(np.int16),
            'dates': date_values, 'available_at': date_values.astype('datetime64[s]')+np.timedelta64(1, 'D'),
            'state_names': np.asarray(STATE_NAMES), 'context_names': np.asarray(CONTEXT_NAMES),
            'state_families': np.asarray(['hrv']*6+['sleep']*2+['self-report']*2),
            'timezone_semantics': np.asarray('explicit UTC converted to first participant offset; naive HRV source-local; questionnaire local completion day')}


def run():
    output = ROOT/'data/ssaqs_study'
    output.mkdir(parents=True, exist_ok=True)
    feature = ROOT/'data/features/ssaqs'
    feature.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT/'configs/daily_protocol.json').read_text())
    protocol_sha = sha(ROOT/'configs/daily_protocol.json')
    source = ROOT/'data/new_cohorts/ssaqs-2026/SSAQS dataset.zip'
    summaries, origins, nodes, pooled, windows = [], [], [], [], []
    with zipfile.ZipFile(source) as archive:
        people = sorted({n.split('/')[0] for n in archive.namelist() if n.split('/')[0].isdigit()}, key=int)
        with archive.open('users-courses.csv') as stream:
            courses = pd.read_csv(stream)
        with archive.open('course-details.csv') as stream:
            schedules = pd.read_csv(stream)
        for person in people:
            arrays = person_arrays(archive, person, courses, schedules)
            _, accepted, summary = write_person(output, 'SSAQS', person, 'external', -1, arrays, protocol, protocol_sha)
            summaries.append(summary)
            if accepted:
                with np.load(output/f'{person}_daily.npz') as z:
                    state, context = z['normalized'], z['normalized_contexts']
                cutoffs = np.asarray([row['origin_day']+1 for row in accepted])
                n, x, groups = encode_person(state, context, cutoffs, Sampling.daily())
                nodes.extend(n); pooled.extend(x)
                joined = np.column_stack((state, context))
                windows.extend([joined[t-32:t] for t in cutoffs])
                origins.extend(accepted)
            print(json.dumps({'person': person, 'days': summary['days'], 'samples': summary['samples']}), flush=True)
    people = pd.DataFrame(summaries)
    people.to_csv(output/'people.csv', index=False)
    summary = summarize_people(people)
    write_json(output/'summary.json', summary)
    if origins:
        frame = pd.DataFrame(origins)
        frame.to_csv(feature/'external.csv', index=False)
        np.savez_compressed(feature/'external.npz', pooled=np.stack(pooled), nodes=np.stack(nodes),
                            windows=np.stack(windows), groups=groups, y=frame.y.to_numpy())
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    with threadpool_limits(limits=2):
        run()
