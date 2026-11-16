"""Past-only first-order conditional prediction gain and CNM-GC adaptation."""
from __future__ import annotations

import numpy as np
from scipy.special import digamma

from .hne_representation import Sampling, context_residuals

VARIANTS = ("HNE-GC", "HNE-GC-null")
GROUPS = np.array(["H"]*96 + ["N"]*12 + ["E"]*128 + ["NE"]*12)


def prediction_gain(values, minimum=6):
    """Rows are targets, columns sources; all three values share observed dates."""
    values = np.asarray(values, dtype=np.float64)
    past, future = values[:-1], values[1:]
    width = values.shape[1]
    gain = np.full((width, width), np.nan)
    counts = np.zeros((width, width), dtype=int)
    for target in range(width):
        mask = (np.isfinite(past[:, target]) & np.isfinite(future[:, target]))[:, None] & np.isfinite(past)
        n = mask.sum(0)
        count = np.maximum(n, 1)
        x = np.where(mask, past[:, target, None], 0.)
        y = np.where(mask, future[:, target, None], 0.)
        z = np.where(mask, past, 0.)
        sx, sy, sz = x.sum(0), y.sum(0), z.sum(0)
        xx = (x*x).sum(0)-sx*sx/count
        yy = (y*y).sum(0)-sy*sy/count
        zz = (z*z).sum(0)-sz*sz/count
        xy = (x*y).sum(0)-sx*sy/count
        xz = (x*z).sum(0)-sx*sz/count
        yz = (y*z).sum(0)-sy*sz/count
        good = (n >= minimum) & (xx > 1e-12) & (yy > 1e-12) & (zz > 1e-12)
        safe_xx = np.where(good, xx, 1.)
        ry, rz = yy-xy*xy/safe_xx, zz-xz*xz/safe_xx
        good &= (ry > 1e-10*np.maximum(yy, 1e-12)) & (rz > 1e-10*np.maximum(zz, 1e-12))
        denominator = np.where(good, ry*rz, 1.)
        r2 = (yz-xy*xz/safe_xx)**2/denominator
        gain[target, good] = -np.log1p(-np.clip(r2[good], 0., 1.-1e-12))
        counts[target] = n
    np.fill_diagonal(gain, np.nan)
    return gain, counts


def centered_gain(gain, count):
    bias = digamma((np.maximum(count, 4)-2)/2)-digamma((np.maximum(count, 4)-3)/2)
    return gain-bias


def summarize(recent, reference, count):
    good = np.isfinite(recent) & np.isfinite(reference)
    if not good.any():
        return np.full(4, np.nan)
    weight = (count[good]-3)/(count[good]-3+24.)
    delta = weight*(recent[good]-reference[good])
    current = reference[good]+delta
    return np.array([np.average(abs(current), weights=weight), np.average(abs(delta), weights=weight),
                     np.max(abs(delta)), np.average(delta, weights=weight)])


def encode_networks(states, contexts, cutoffs):
    spec = Sampling.daily()
    states = np.asarray(states, dtype=float)
    adjusted, _ = context_residuals(states, np.asarray(contexts, dtype=float), spec)
    streams = (states, adjusted)
    references = [prediction_gain(stream[:spec.reference]) for stream in streams]
    output = {name: [] for name in VARIANTS}
    for cutoff in cutoffs:
        parts = {name: [] for name in VARIANTS}
        for stream, (reference, ref_count) in zip(streams, references):
            for length in spec.scales:
                recent, count = prediction_gain(stream[cutoff-length:cutoff])
                parts["HNE-GC"].extend(summarize(recent, reference, count))
                parts["HNE-GC-null"].extend(summarize(centered_gain(recent, count),
                                                              centered_gain(reference, ref_count), count))
        for name in VARIANTS:
            output[name].append(parts[name])
    return {name: np.asarray(value, dtype=np.float32) for name, value in output.items()}


def two_variance_groups(variance):
    """Globally optimal one-dimensional two-means, avoiding initialization noise."""
    order = np.argsort(variance, kind="stable")
    values = np.asarray(variance)[order]
    n = len(values)
    if n < 2 or values[-1] == values[0]:
        return np.array([], dtype=int), np.array([], dtype=int)
    s, ss = np.cumsum(values), np.cumsum(values**2)
    k = np.arange(1, n)
    objective = ss[k-1]-s[k-1]**2/k + ss[-1]-ss[k-1]-(s[-1]-s[k-1])**2/(n-k)
    objective[values[:-1] == values[1:]] = np.inf
    split = int(np.argmin(objective))+1
    return order[split:], order[:split]


def cnm_score(window, minimum=12):
    count = np.isfinite(window).sum(0)
    variance = np.nanvar(window, axis=0, ddof=1)
    eligible = (count >= minimum) & np.isfinite(variance) & (variance > 1e-12)
    selected = np.asarray(window)[:, eligible]
    dg, ndg = two_variance_groups(variance[eligible])
    result = dict(score=np.nan, channels=int(eligible.sum()), dominant=len(dg), nondominant=len(ndg), edges=0)
    if not len(dg) or not len(ndg):
        return result
    gain, support = prediction_gain(selected, minimum)
    directed = gain[np.ix_(ndg, dg)]
    result["edges"] = int(np.isfinite(directed).sum())
    if np.isfinite(directed).all():
        result["score"] = 1./max(float(directed.mean()), 1e-12)
    return result
