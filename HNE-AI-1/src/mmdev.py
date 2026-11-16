"""MM-DEV-causal-median: a temporally available adaptation, not paper replication.

The author's R scripts fit ISOMAP before retrospective rolling-window analysis.
Here ISOMAP, median filling and embedding selection use one past 168-hour window
only. Hour positions are preserved; missing values use that window's medians.
The original ISOMAP -> mixed lag embedding -> S-map Jacobian -> mean dominant
eigenvalue modulus chain is retained. 'Causal' means cutoff-respecting input,
not causal identification of sensor or environmental effects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np
from scipy.linalg import eigh, lstsq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist


MODEL_NAME = "MM-DEV-causal-median"
VERSION = "2026-11-10-v1"


@dataclass(frozen=True)
class MMDEVConfig:
    history_hours: int = 168
    smap_window: int = 70
    isomap_neighbors: int = 5
    isomap_components: int = 3
    embedding_max: int = 12
    tau: int = 1
    theta: float = 1.0


def metadata(config=MMDEVConfig()):
    return {
        "model": MODEL_NAME, "version": VERSION, "parameters": asdict(config),
        "paper": "https://doi.org/10.21203/rs.3.rs-7865028/v1",
        "author_scripts": ["Figure_4_2.R", "Figure_3_2.R", "Demo_check_embedding_dim.R"],
        "smap_reference": "https://ha0ye.github.io/rEDM/reference/block_lnlp.html",
        "adaptation": "Fit ISOMAP and select E separately inside each past 168-hour history; preserve hourly positions and fill each channel with its historical-window median.",
        "missingness": "All-missing columns use zero placeholders and are dropped with other constant columns from ISOMAP distances; no future observations or labels are used.",
        "parameter_origin": "k=5 and three leading manifold coordinates follow empirical author scripts; theta=1 and tau=1 follow the paper. The 70-point S-map window follows Figure_4_2.R; E=1..12 follows Demo_check_embedding_dim.R. These are fixed adaptation choices, not a universal author default.",
        "embedding_selection": "Maximum theta=0 leave-one-out one-step prediction correlation of the first ISOMAP coordinate within the already observed history; ties choose smaller E.",
        "jacobian": "For each source state with its one-hour successor in the final 70-hour window, exclude that source row, multiply both design and target by exp(-theta*distance/mean_distance), solve by SVD, and retain slope coefficients only.",
        "native_score": "Mean over local Jacobians of max(abs(eigenvalues)); larger is fixed as greater warning risk.",
        "disconnected_isomap": "Return NaN if the fixed 5-neighbor graph is disconnected; do not add unpublished graph-bridging edges.",
        "time_compression": False, "event_label_used_for_native_score": False,
        "single_origin_cache": True,
    }


def window_digest(window):
    x = np.array(window, dtype="<f4", copy=True, order="C")
    x[~np.isfinite(x)] = np.nan
    return hashlib.sha256(np.asarray(x.shape, dtype="<i8").tobytes() + x.tobytes()).hexdigest()


def _empty(reason, **fields):
    return dict(score=float("nan"), reason=reason, embedding_E=0,
                embedding_cv_rho=float("nan"), isomap_components=0,
                jacobian_count=0, **fields)


def _isomap(x, config):
    """Symmetric k-NN geodesics followed by classical metric MDS."""
    distance = cdist(x, x)
    search = distance.copy()
    np.fill_diagonal(search, np.inf)
    neighbor = np.argsort(search, axis=1, kind="stable")[:, :config.isomap_neighbors]
    adjacency = np.zeros(distance.shape, dtype=bool)
    adjacency[np.arange(len(x))[:, None], neighbor] = True
    adjacency |= adjacency.T
    components = connected_components(csr_matrix(adjacency), directed=False, return_labels=False)
    if components != 1:
        return None, int(components)
    # Explicit zero-length edges between duplicate points must remain edges in
    # sparse graph operations. A numerical epsilon does not bridge components.
    rows, columns = np.nonzero(adjacency)
    edge_distance = np.maximum(distance[rows, columns], np.finfo(float).eps)
    graph = csr_matrix((edge_distance, (rows, columns)), shape=distance.shape)
    geodesic = shortest_path(graph, directed=False, method="D")
    square = geodesic ** 2
    kernel = -.5 * (square - square.mean(0)[None] - square.mean(1)[:, None] + square.mean())
    size = min(config.isomap_components, len(x) - 1)
    eigenvalue, eigenvector = eigh(kernel, subset_by_index=(len(x) - size, len(x) - 1),
                                  check_finite=False, driver="evr")
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue, eigenvector = eigenvalue[order], eigenvector[:, order]
    positive = eigenvalue > max(1e-12, eigenvalue[0] * 1e-10)
    if not positive.any():
        return None, int(components)
    coordinate = eigenvector[:, positive] * np.sqrt(eigenvalue[positive])
    # Fix arbitrary eigenvector signs for deterministic cache contents. DEV is
    # unchanged by this coordinate sign convention.
    anchor = np.argmax(np.abs(coordinate), axis=0)
    signs = np.sign(coordinate[anchor, np.arange(coordinate.shape[1])])
    coordinate *= np.where(signs == 0, 1, signs)
    return coordinate, int(components)


def _standardize(x):
    scale = np.std(x, axis=0, ddof=1)
    if not np.isfinite(scale).all() or np.any(scale < 1e-12):
        return None
    return (x - x.mean(0)) / scale


def _embedding_dimension(first, config):
    """Theta-zero S-map is affine OLS; its LOO fit uses SVD hat diagonals."""
    best_E, best_rho = None, -np.inf
    for E in range(1, config.embedding_max + 1):
        start = (E - 1) * config.tau
        time = np.arange(start, len(first) - 1)
        if len(time) < E + 3:
            continue
        lagged = np.column_stack([first[time - k * config.tau] for k in range(E)])
        target = first[time + 1]
        design = np.column_stack((np.ones(len(time)), lagged))
        U, singular, Vt = np.linalg.svd(design, full_matrices=False)
        rank = singular > (max(design.shape) * np.finfo(float).eps * singular[0])
        fitted = U[:, rank] @ (U[:, rank].T @ target)
        leverage = np.square(U[:, rank]).sum(1)
        usable = (1 - leverage) > 1e-8
        if usable.sum() < E + 3:
            continue
        predicted = target[usable] - (target[usable] - fitted[usable]) / (1 - leverage[usable])
        actual = target[usable]
        if np.std(predicted) < 1e-12 or np.std(actual) < 1e-12:
            continue
        rho = float(np.corrcoef(predicted, actual)[0, 1])
        if np.isfinite(rho) and rho > best_rho + 1e-12:
            best_E, best_rho = E, rho
    return best_E, best_rho


def _mixed_embedding(coordinates, E, tau):
    m = coordinates.shape[1]
    if E <= m:
        return coordinates[:, :E]
    delay_count = E - m
    start = delay_count * tau
    time = np.arange(start, len(coordinates))
    return np.column_stack([coordinates[time]] +
                           [coordinates[time - k * tau, 0] for k in range(1, delay_count + 1)])


def _local_dominant(embedded, theta):
    """One-step vector S-map, equivalent to solving each target separately."""
    current, future = embedded[:-1], embedded[1:]
    design = np.column_stack((np.ones(len(current)), current))
    distance = cdist(current, current)
    dominant = []
    ranks = []
    for focal in range(len(current)):
        included = np.arange(len(current)) != focal
        d = distance[focal, included]
        mean_distance = float(d.mean())
        if mean_distance <= 1e-12:
            continue
        weight = np.exp(-theta * d / mean_distance)
        # Paper equations multiply BOTH sides by w, so the effective least
        # squares weights are w², not w. The intercept is not part of J.
        coefficients, _, rank, _ = lstsq(design[included] * weight[:, None],
                                         future[included] * weight[:, None],
                                         cond=None, check_finite=False, lapack_driver="gelsd")
        jacobian = coefficients[1:].T
        eigenvalues = np.linalg.eigvals(jacobian)
        if np.isfinite(eigenvalues).all():
            dominant.append(eigenvalues[np.argmax(np.abs(eigenvalues))])
            ranks.append(int(rank))
    return np.asarray(dominant, dtype=complex), ranks


def score_window(window, config=MMDEVConfig()):
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("Expected history-by-channel matrix")
    observed = np.isfinite(x)
    diagnostics = dict(imputed_fraction=float(1 - observed.mean()),
                       input_channels=int(x.shape[1]), graph_components=0)
    if len(x) != config.history_hours or len(x) < config.smap_window + config.embedding_max:
        return _empty("insufficient_history", **diagnostics)
    medians = np.asarray([np.median(column[np.isfinite(column)]) if np.isfinite(column).any()
                          else 0.0 for column in x.T])
    filled = np.where(observed, x, medians)
    scale = np.std(filled, axis=0, ddof=1)
    active = np.isfinite(scale) & (scale > 1e-12)
    diagnostics.update(active_channels=int(active.sum()),
                       dropped_constant_channels=int((~active).sum()))
    if not active.any():
        return _empty("no_nonconstant_channel", **diagnostics)
    standardized = (filled[:, active] - filled[:, active].mean(0)) / scale[active]
    try:
        coordinates, components = _isomap(standardized, config)
        diagnostics["graph_components"] = components
        if coordinates is None:
            reason = "disconnected_isomap_graph" if components != 1 else "no_positive_manifold_component"
            return _empty(reason, **diagnostics)
        first = _standardize(coordinates[:, :1])
        if first is None:
            return _empty("constant_first_manifold_component", **diagnostics)
        E, skill = _embedding_dimension(first[:, 0], config)
        if E is None:
            return _empty("embedding_prediction_unavailable", **diagnostics)
        embedded = _mixed_embedding(coordinates, E, config.tau)
        if len(embedded) < config.smap_window:
            return _empty("insufficient_embedded_history", **diagnostics)
        recent = _standardize(embedded[-config.smap_window:])
        if recent is None:
            return _empty("constant_recent_embedded_coordinate", **diagnostics)
        dominant, ranks = _local_dominant(recent, config.theta)
        if len(dominant) < E + 2:
            return _empty("insufficient_local_jacobians", **diagnostics)
        return dict(score=float(np.abs(dominant).mean()), reason="ok", embedding_E=int(E),
                    embedding_cv_rho=float(skill), isomap_components=int(coordinates.shape[1]),
                    jacobian_count=len(dominant), dominant_real_mean=float(dominant.real.mean()),
                    dominant_imaginary_absolute_mean=float(np.abs(dominant.imag).mean()),
                    minimum_regression_rank=min(ranks), **diagnostics)
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        return _empty("numerical_" + type(exc).__name__, **diagnostics)
