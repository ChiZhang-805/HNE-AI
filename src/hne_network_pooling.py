"""Family-balanced summaries of the same personal-reference lag graphs."""
from __future__ import annotations

import numpy as np

from .hne_features import correlation_matrix
from .hne_representation import Sampling, context_residuals

VARIANTS = ("HNE-block-network", "HNE-cross-network")
NETWORK_COLUMNS = np.r_[96:132, 260:296]


def edge_blocks(families, lag):
    """Unordered family pairs at lag zero; ordered pairs at positive lags."""
    _, labels = np.unique(np.asarray(families, dtype=str), return_inverse=True)
    a, b = np.broadcast_arrays(labels[:, None], labels[None, :])
    count = len(np.unique(labels))
    return (np.minimum(a, b) * count + np.maximum(a, b) if lag == 0
            else a * count + b), a != b


def summarize_edges(recent, baseline, reliability, families, lag, cross_only=False):
    valid = np.isfinite(recent) & np.isfinite(baseline)
    valid &= (np.triu(np.ones_like(valid), 1).astype(bool) if lag == 0
              else ~np.eye(len(valid), dtype=bool))
    blocks, cross = edge_blocks(families, lag)
    if cross_only:
        valid &= cross
    if not valid.any():
        return np.full(4, np.nan)
    weight = reliability[valid]
    delta = weight * (recent[valid] - baseline[valid])
    current = baseline[valid] + delta
    values = np.column_stack((np.abs(current), np.abs(delta), delta))
    block_ids = blocks[valid]
    block_means = []
    for block in np.unique(block_ids):
        keep = block_ids == block
        block_means.append(np.average(values[keep], axis=0, weights=weight[keep]))
    average = np.mean(block_means, axis=0)
    return np.array([average[0], average[1], np.max(np.abs(delta)), average[2]])


def encode_networks(states, contexts, cutoffs, families, context_ridge=10.):
    """Return both variants without recomputing or changing H/E descriptors."""
    spec = Sampling.daily()
    states = np.asarray(states, dtype=float)
    contexts = np.asarray(contexts, dtype=float)
    adjusted, _ = context_residuals(states, contexts, spec, ridge=context_ridge)
    streams = (states, adjusted)
    reference = []
    for stream in streams:
        prefix = stream[:spec.reference]
        reference.append({lag: correlation_matrix(
            prefix[lag:] if lag else prefix, prefix[:-lag] if lag else prefix,
            minimum=spec.pair_minimum)[0] for lag in spec.lags})
    result = {name: [] for name in VARIANTS}
    for cutoff in cutoffs:
        features = {name: [] for name in VARIANTS}
        for stream, ref in zip(streams, reference):
            for length in spec.scales:
                window = stream[cutoff-length:cutoff]
                for lag in spec.lags:
                    recent, reliability = correlation_matrix(
                        window[lag:] if lag else window,
                        window[:-lag] if lag else window,
                        minimum=spec.pair_minimum)
                    for name in VARIANTS:
                        features[name].extend(summarize_edges(recent, ref[lag], reliability,
                            families, lag, cross_only=name == "HNE-cross-network"))
        for name in VARIANTS:
            result[name].append(features[name])
    return {name: np.asarray(values, dtype=np.float32) for name, values in result.items()}
