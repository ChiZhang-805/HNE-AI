"""Schema-defined family pooling of HNE node descriptors.

This optional structural variant changes the node summaries only. Neither the
temporal descriptors nor raw/conditional relation summaries are recomputed.
"""
from __future__ import annotations

import numpy as np

from .hne_features import pool_nodes

H_COLUMNS = np.r_[0:8, 16:24, 32:40]
E_COLUMNS = np.r_[8:16, 24:32, 40:50]
UNCHANGED_COLUMNS = np.r_[96:132, 236:296]


def family_descriptors(nodes, state_families):
    """Finite signed maximum per descriptor within each fixed schema family.

    A family/descriptor with no finite member stays NaN. Families are not fitted
    or reweighted using a participant's reference, current history, or outcome.
    Input shape is (..., state variables, descriptors).
    """
    nodes = np.asarray(nodes)
    families = np.asarray(state_families, dtype=str)
    if nodes.ndim < 2 or families.ndim != 1 or nodes.shape[-2] != len(families):
        raise ValueError("Family labels must align with the node axis")
    if not len(families) or np.any(families == ""):
        raise ValueError("Every state needs a nonempty schema family")
    descriptors = []
    for family in np.unique(families):
        values = nodes[..., families == family, :]
        finite = np.isfinite(values)
        maximum = np.max(np.where(finite, values, -np.inf), axis=-2)
        descriptors.append(np.where(finite.any(axis=-2), maximum, np.nan))
    return np.stack(descriptors, axis=-2).astype(np.float32, copy=False)


def pool_families(nodes, pooled, state_families):
    """Replace H96 and adjusted-state E104 while retaining 296 coordinates."""
    nodes, pooled = np.asarray(nodes), np.asarray(pooled)
    if nodes.ndim != 3 or nodes.shape[2] != 50 or pooled.shape != (len(nodes), 296):
        raise ValueError("Expected sample x state x 50 nodes and sample x 296 pooled")
    family_nodes = family_descriptors(nodes, state_families)
    result = pooled.copy()
    for i, descriptors in enumerate(family_nodes):
        result[i, :96] = pool_nodes(descriptors[:, H_COLUMNS])
        result[i, 132:236] = pool_nodes(descriptors[:, E_COLUMNS])
    return result, family_nodes
