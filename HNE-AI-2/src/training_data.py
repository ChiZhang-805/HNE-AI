"""Read-only hourly windows, fixed person folds, and prefix-only normalization."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import numpy as np
import pandas as pd
import torch

SOURCE = Path(os.environ.get('HNE_STUDY_ROOT', Path(__file__).resolve().parents[2] / 'HNE-AI-1')).expanduser().resolve()


def provenance_key(path, project):
    path = Path(path).resolve()
    if path.is_relative_to(project):
        return 'project/' + path.relative_to(project).as_posix()
    return 'inputs/' + path.relative_to(SOURCE).as_posix()
FAMILIES = {'RESILIENT': [0, 1, 2, 0, 3, 4], 'BrisT1D-Open': [0, 1, 2, 2, 2, 3]}


@dataclass
class Person:
    dataset: str
    person: str
    fold: int
    raw: np.ndarray
    clock: pd.Timestamp
    origins: np.ndarray
    labels: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    calibrated_input: np.ndarray
    source_files: tuple

    @property
    def state(self):
        return torch.tensor([[True] * 5 + [False]])

    @property
    def family(self):
        return torch.tensor([FAMILIES[self.dataset]])

    def small_window(self, start, end):
        assert 0 <= start < end <= len(self.raw) // 24
        x = self.raw[start * 24:end * 24].T.copy()
        mask = np.isfinite(x) & self.calibrated_input[:, None]
        x = (x - self.mean[:, None]) / self.scale[:, None]
        # The input transform has no clipping; finite raw extremes remain observable.
        x = np.where(mask, x, 0).astype('float32')
        p = x.shape[-1] // 16
        ticks = self.clock + pd.to_timedelta(np.arange(start * 24, end * 24, 16), unit='h')
        hour = np.asarray(ticks.hour) / 24 * 2 * np.pi
        week = (np.asarray(ticks.dayofweek) + np.asarray(ticks.hour) / 24) / 7 * 2 * np.pi
        cal = np.stack([np.sin(hour), np.cos(hour), np.sin(week), np.cos(week)], -1)
        return dict(values=torch.from_numpy(x.reshape(1, 6, p, 16)),
                    observed=torch.from_numpy(mask.reshape(1, 6, p, 16)),
                    calendar=torch.tensor(cal[None], dtype=torch.float32),
                    state=self.state, family=self.family)

    def large_window(self, origin):
        # origin d means observations through the end of d; issue at start d+1.
        stop = (origin + 1) * 24
        hours = np.r_[np.arange(56 * 24), np.arange(stop - 32 * 24, stop)]
        if hours[56 * 24] <= hours[56 * 24 - 1]:
            raise ValueError('Reference and recent windows must not overlap')
        x = self.raw[hours].T.copy().astype('float32')
        obs = np.isfinite(x)
        keep = obs.any(-1)
        state = self.state[:, keep]
        family = self.family[:, keep]
        if not (state & torch.from_numpy(obs[keep, -768:].any(-1))[None]).any():
            raise ValueError('No recent observed state channel')
        return dict(context=torch.from_numpy(x[keep]), observed=torch.from_numpy(obs[keep]),
                    hour_offsets=torch.tensor((hours - stop)[None].repeat(keep.sum(), 0)),
                    group_ids=torch.zeros(int(keep.sum()), dtype=torch.long)), state, family, keep


def load_people(dataset, fold, include_evaluation=False, only_split=None):
    result = {'train': [], 'validation': [], 'evaluation': []}
    for path in sorted((SOURCE / 'data/daily_study' / dataset).glob('*_daily.npz')):
        with np.load(path, allow_pickle=False) as day:
            f = int(day['fold'])
            split = 'evaluation' if f == fold else 'validation' if f == (fold + 1) % 5 else 'train'
            if only_split is not None and split != only_split:
                continue
            if split == 'evaluation' and not include_evaluation:
                continue  # Do not load evaluation outcomes during pipeline development.
            origins = np.flatnonzero(day['eligible_origin'] & np.isfinite(day['future_labels']))
            if not len(origins):
                continue
            labels = day['future_labels'][origins].astype('float32')
            person = str(day['person_id'])
        source = SOURCE / 'data/development' / dataset / path.name.replace('_daily', '_hourly')
        with np.load(source, allow_pickle=False) as hourly:
            raw = hourly['values'].copy().astype('float32')
            clock = pd.Timestamp(str(hourly['clock']))
        median, scales, usable = [], [], []
        for channel in raw[:28 * 24].T:
            v = channel[np.isfinite(channel)].astype('float64')
            good = len(v) >= 8
            m = np.median(v) if good else 0.
            # Constant/missing prefix channels cannot define personal deviations.
            sd = np.std(v) if good else 0.
            good = good and sd > 1e-6
            scale = max(1.4826 * np.median(np.abs(v - m)), .1 * sd, 1e-6) if good else 1.
            median.append(m); scales.append(scale); usable.append(good)
        result[split].append(Person(dataset, person, f, raw, clock, origins, labels,
            np.asarray(median), np.asarray(scales), np.asarray(usable), (str(path), str(source))))
    return result


def component_assignments(people, seed):
    def key(p):
        return int(hashlib.sha256(f'{p.dataset}|{p.person}|{seed}'.encode()).hexdigest(), 16)
    train = sorted(people['train'], key=key)
    result = {p.person: i % 3 for i, p in enumerate(train)}
    for split in ('validation', 'evaluation'):
        result.update({p.person: key(p) % 3 for p in people[split]})
    return result


def selected_origins(person, maximum=0):
    # Deterministic chronological coverage, selected without looking at labels.
    indexes = np.arange(len(person.origins))
    if maximum and len(indexes) > maximum:
        indexes = np.unique(np.linspace(0, len(indexes) - 1, maximum).round().astype(int))
    return [(int(person.origins[i]), float(person.labels[i])) for i in indexes]


def segment_weights(tokens=145):
    weights = torch.zeros(3, tokens)
    weights[0, :84] = 1
    weights[1, 84:132] = 1
    weights[2, 133:] = torch.tensor([0, .5] + [1] * 10)
    return weights
