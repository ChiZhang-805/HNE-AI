"""Shared normal dynamics with personal ridge adaptation in invariant coordinates."""
from __future__ import annotations

import numpy as np

from .hne_features import correlation_matrix
from .hne_representation import Sampling, encode_person, lag_context


WIDTH = 15
VARIANTS = ('HNE-shared-reference', 'HNE-compact-local')


def reference_scale(values, rows, minimum):
    reference = np.asarray(values[:len(rows)], float)[rows]
    count = np.isfinite(reference).sum(0)
    center = np.nanmean(reference, axis=0)
    scale = np.nanstd(reference, axis=0)
    supported = (count >= minimum) & (scale > 1e-8)
    center[~supported] = np.nan
    scale[~supported] = np.nan
    scaled = np.clip((values-center)/scale, -8., 8.)
    return scaled, center, scale


def normal_design(states, contexts, rows, cfg):
    """One reference-only projection, independent of channel identity or count."""
    r, p = cfg['reference_days'], states.shape[1]
    z, center, scale = reference_scale(states, rows, cfg['minimum_normalization_observations'])
    c, _, _ = reference_scale(contexts, rows, cfg['minimum_normalization_observations'])
    past = lag_context(z, cfg['state_lags'])
    environment = lag_context(c, cfg['context_lags'])
    x = np.zeros((len(states), p, WIDTH), dtype=np.float64)
    for k in range(2):
        values = past[:, k*p:(k+1)*p]
        x[:, :, k] = np.nan_to_num(values, nan=0.)
        x[:, :, 2+k] = np.isfinite(values)
    target = np.where(rows[:, None], z[:r], np.nan)
    for kind, streams, width, count, offset in (
            ('N', past, p, 2, 4), ('E', environment, contexts.shape[1], 3, 8)):
        for lag in range(count):
            values = streams[:, lag*width:(lag+1)*width]
            reference = np.where(rows[:, None], values[:r], np.nan)
            corr, reliability = correlation_matrix(target, reference,
                minimum=cfg['minimum_pair_observations'], shrink=cfg['edge_shrinkage'])
            edge = np.nan_to_num(corr*reliability, nan=0.)
            if kind == 'N':
                np.fill_diagonal(edge, 0.)
            denominator = np.maximum(abs(edge).sum(1), 1.)
            weights = edge/denominator[:, None]
            x[:, :, offset+2*lag] = np.nan_to_num(values, nan=0.)@weights.T
            x[:, :, offset+2*lag+1] = np.isfinite(values).astype(float)@abs(weights).T
    x[:, :, -1] = 1.
    available = np.any(x[:, :, [2, 3, 5, 7, 9, 11, 13]] > 0, axis=2)
    y = (states[:r]-center)/scale
    fit = np.isfinite(y) & available[:r] & rows[:, None]
    return x, y, center, scale, fit, available


def prepare_reference(states, contexts, cfg):
    states, contexts = np.asarray(states, float), np.asarray(contexts, float)
    r, p = cfg['reference_days'], states.shape[1]
    full = normal_design(states, contexts, np.ones(r, bool), cfg)
    output = dict(full_x=full[0].astype(np.float32), full_y=full[1], full_center=full[2],
                  full_scale=full[3], full_fit=full[4], full_available=full[5])
    blocks, cf_x, cf_y, cf_good = [], np.zeros((r, p, WIDTH)), np.zeros((r, p)), np.zeros((r, p), bool)
    day = np.arange(r)
    for start in range(0, r, cfg['reference_block_days']):
        stop = start+cfg['reference_block_days']
        held = (day >= start) & (day < stop)
        training = (day < start-cfg['reference_embargo_days']) | (day >= stop+cfg['reference_embargo_days'])
        design = normal_design(states[:r], contexts[:r], training, cfg)
        x, y, center, scale, fit, available = design
        good = np.isfinite(y) & available & held[:, None]
        cf_x[held], cf_y[held], cf_good[held] = x[held], y[held], good[held]
        blocks.append(dict(x=x, y=y, center=center, scale=scale, fit=fit, good=good))
    for key in ['x', 'y', 'center', 'scale', 'fit', 'good']:
        output['block_'+key] = np.stack([b[key] for b in blocks])
    # Every person contributes mass1, equally divided among supported nodes.
    node_count = cf_good.sum(0)
    weight = cf_good/np.maximum(node_count, 1)[None]
    weight /= max(np.count_nonzero(node_count), 1)
    usable_x, usable_y, usable_weight = cf_x[cf_good], np.clip(cf_y[cf_good], -cfg['population_target_clip'], cfg['population_target_clip']), weight[cf_good]
    output['prior_gram'] = (usable_x*usable_weight[:, None]).T@usable_x
    output['prior_target'] = usable_x.T@(usable_weight*usable_y)
    output['prior_mass'] = np.asarray(float(usable_weight.sum()))
    return output


def shared_prior(gram, target, mass, ridge):
    if mass <= 0:
        return np.zeros(WIDTH)
    penalty = np.eye(WIDTH)*ridge
    penalty[-1, -1] = 0.
    return np.linalg.solve(gram/mass+penalty, target/mass)


def local_coefficients(x, y, valid, prior, ridge, minimum):
    coefficient = np.broadcast_to(prior, (x.shape[1], WIDTH)).copy()
    for node in range(x.shape[1]):
        good = valid[:, node]
        if good.sum() < minimum:
            continue
        a, b = x[good, node], y[good, node]
        gram = a.T@a/len(a)
        coefficient[node] = np.linalg.solve(gram+ridge*np.eye(WIDTH), a.T@b/len(a)+ridge*prior)
    return coefficient


def reference_errors(bundle, prior, ridge, cfg):
    p, r = bundle['full_y'].shape[1], cfg['reference_days']
    residual, standardized = np.full((r, p), np.nan), np.full((r, p), np.nan)
    for k in range(len(bundle['block_x'])):
        x, y, good = bundle['block_x'][k], bundle['block_y'][k], bundle['block_good'][k]
        theta = local_coefficients(x, y, bundle['block_fit'][k], prior, ridge, cfg['minimum_local_targets'])
        error = y-np.einsum('tpd,pd->tp', x, theta)
        standardized[good] = error[good]
        original = error*bundle['block_scale'][k]
        residual[good] = original[good]
    return residual, standardized


def normal_loss(errors):
    absolute = abs(errors)
    node = np.nanmean(np.where(absolute <= 2., .5*errors**2, 2.*absolute-2.), axis=0)
    return float(np.nanmean(node))


def encode_shared(states, contexts, cutoffs, bundle, prior, ridge, cfg):
    r = cfg['reference_days']
    theta = local_coefficients(bundle['full_x'][:r], bundle['full_y'], bundle['full_fit'],
                               prior, ridge, cfg['minimum_local_targets'])
    predicted = np.einsum('tpd,pd->tp', bundle['full_x'], theta)
    residual = states-bundle['full_center']-bundle['full_scale']*predicted
    residual[~bundle['full_available']] = np.nan
    reference, error = reference_errors(bundle, prior, ridge, cfg)
    residual[:r] = reference
    effect = np.column_stack((np.linalg.norm(theta[:, [8, 10, 12]], axis=1)/np.sqrt(3),
                              bundle['full_fit'].mean(0))).astype(np.float32)
    _, pooled, groups = encode_person(states, contexts, cutoffs, Sampling.daily(), conditioned=(residual, effect))
    return pooled, error, theta
