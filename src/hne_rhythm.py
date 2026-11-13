"""Past-only, reference-relative harmonic coordinates shared across cohorts."""
import numpy as np

from .daily_protocol import robust_reference, normalize
from .hne_features import pool_nodes
from .hne_representation import Sampling, context_residuals


def prefix(x):
    return np.concatenate((np.zeros_like(x[:1]), np.cumsum(x, axis=0)), axis=0)


class HarmonicHistory:
    def __init__(self, hourly, cfg):
        self.cfg = cfg
        self.days = len(hourly)//24
        values = hourly[:self.days*24].reshape(self.days, 24, -1)
        observed = np.isfinite(values)
        count = observed.sum(1)
        valid_day = count >= cfg['minimum_hours_per_day']
        observed &= valid_day[:, None, :]
        weight = observed/np.maximum(count[:, None, :], 1)
        value = np.where(observed, values, 0.)
        angle = 2*np.pi*(np.arange(24)+.5)/cfg['period_recorded_hours']
        design = np.column_stack((np.ones(24), np.cos(angle), np.sin(angle)))
        self.xx = prefix(np.einsum('dhc,hj,hk->dcjk', weight, design, design))
        self.xy = prefix(np.einsum('dhc,dhc,hj->dcj', weight, value, design))
        self.yy = prefix((weight*value**2).sum(1))
        self.valid_days = prefix(valid_day.astype(float))
        self.hours = prefix(observed.sum(1).astype(float))
        self.clock = prefix(observed.astype(float))

    def fit_windows(self, starts, ends, reference=False):
        starts, ends = np.asarray(starts), np.asarray(ends)
        n = self.valid_days[ends]-self.valid_days[starts]
        denominator = np.maximum(n, 1.)
        a = (self.xx[ends]-self.xx[starts])/denominator[..., None, None]
        b = (self.xy[ends]-self.xy[starts])/denominator[..., None]
        second = (self.yy[ends]-self.yy[starts])/denominator
        penalty = np.diag([0., self.cfg['harmonic_ridge'], self.cfg['harmonic_ridge']])
        # Empty windows have no intercept information; their outputs are masked.
        regular = a+penalty
        regular[..., 0, 0] += (n == 0)
        beta = np.linalg.solve(regular, b[..., None])[..., 0]
        error = np.maximum(second-2*(beta*b).sum(-1)+np.einsum('...j,...jk,...k->...', beta, a, beta), 0.)
        variance = np.maximum(second-b[..., 0]**2, 0.)
        r2 = np.where(variance > 1e-12, np.clip(1-error/np.maximum(variance, 1e-12), 0., 1.), 0.)
        hours = self.hours[ends]-self.hours[starts]
        clock = self.clock[ends]-self.clock[starts]
        clock_hours = (clock > 0).sum(1)
        quarters = np.stack([(clock[:, k*6:(k+1)*6] > 0).any(1) for k in range(4)], axis=1).sum(1)
        minimum_days = self.cfg['minimum_reference_days'] if reference else self.cfg['minimum_fit_days']
        valid = (n >= minimum_days) & (hours >= self.cfg['minimum_fit_hours'])
        valid &= (clock_hours >= self.cfg['minimum_distinct_clock_hours']) & (quarters >= self.cfg['minimum_clock_quarters'])
        coverage = hours/(24*(ends-starts)[:, None])
        coefficient = beta[..., 1]+1j*beta[..., 2]
        amplitude = np.abs(coefficient)
        phasor = coefficient/np.maximum(amplitude, 1e-12)*np.sqrt(r2)
        return {name: np.where(valid, value, np.nan) for name, value in
                [('coefficient', coefficient), ('amplitude', amplitude), ('phasor', phasor),
                 ('r2', r2), ('error', error)]} | {'coverage': coverage, 'valid': valid}


def node_coordinates(current, reference):
    alignment = current['phasor']*np.conj(reference['phasor'])
    return np.stack((current['amplitude'],
        (current['amplitude']-reference['amplitude'])/(1+reference['amplitude']),
        alignment.real, alignment.imag,
        np.abs(current['coefficient']-reference['coefficient'])/(1+reference['amplitude']),
        current['r2'], (current['error']-reference['error'])/(1+reference['error']),
        current['coverage']), axis=-1)


def relation_coordinates(current, reference, families):
    pairs = [(i, j) for i in range(len(families)) for j in range(i+1, len(families)) if families[i] != families[j]]
    output = np.full((len(current['phasor']), 6), np.nan)
    if not pairs:
        output[:, -1] = 0.
        return output
    left, right = np.asarray(pairs).T
    local = current['phasor'][:, left]*np.conj(current['phasor'][:, right])
    original = reference['phasor'][:, left]*np.conj(reference['phasor'][:, right])
    delta = local-original
    absolute = np.abs(delta)
    output[:, 0] = np.nanmean(absolute, axis=1)
    output[:, 1] = np.nanstd(absolute, axis=1)
    output[:, 2] = np.nanmax(absolute, axis=1)
    output[:, 3] = np.nanmean(delta.real, axis=1)
    output[:, 4] = np.nanmean(np.abs(delta.imag), axis=1)
    output[:, 5] = np.isfinite(delta).mean(1)
    return output


def encode_rhythm(states, contexts, families, cutoff_days, cfg):
    reference = cfg['reference_days']*24
    center, scale, _ = robust_reference(states, reference, cfg['minimum_reference_days']*4)
    raw = np.clip(normalize(states, center, scale), -8, 8)
    center, scale, _ = robust_reference(contexts, reference, cfg['minimum_reference_days']*4)
    context = np.clip(normalize(contexts, center, scale), -8, 8)
    spec = Sampling(reference, tuple(w*24 for w in cfg['window_days']), tuple(cfg['context_lags_hours']),
                    (1, 24), 14*24, 24, cfg['minimum_reference_days']*4, 12)
    adjusted, _ = context_residuals(raw, context, spec, ridge=cfg['context_ridge'])
    cutoff_days = np.asarray(cutoff_days, dtype=int)
    all_features, groups, coverage = [], [], []
    for data, history_group, network_group in [(raw, 'H', 'N'), (adjusted, 'E', 'NE')]:
        harmonic = HarmonicHistory(data, cfg)
        ref = harmonic.fit_windows([0], [cfg['reference_days']], reference=True)
        nodes, relations = [], []
        for length in cfg['window_days']:
            current = harmonic.fit_windows(cutoff_days-length, cutoff_days)
            nodes.append(node_coordinates(current, ref))
            relations.append(relation_coordinates(current, ref, families))
            coverage.append(current['valid'].mean(1))
        pooled = np.asarray([pool_nodes(row) for row in np.concatenate(nodes, axis=-1)])
        relation = np.column_stack(relations)
        all_features.extend((pooled, relation))
        groups.extend([history_group]*pooled.shape[1]+[network_group]*relation.shape[1])
    return np.column_stack(all_features).astype(np.float32), np.asarray(groups), np.column_stack(coverage).astype(np.float32)
