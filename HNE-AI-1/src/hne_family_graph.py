"""Conditional family-event graph with shared H/E unaries and N/NE edges."""
from itertools import combinations, product

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .hne_features import correlation_matrix


def family_unaries(nodes, pooled, families, names):
    """Same four node summaries, vectorized over origins within each family."""
    h_columns = np.r_[0:8,16:24,32:40]
    e_columns = np.r_[8:16,24:32,40:50]

    def pool(x):
        x = x.astype(np.float64)
        finite = np.isfinite(x)
        count = finite.sum(1)
        total = np.where(finite, x, 0).sum(1)
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        variance = np.divide(np.where(finite, (x-mean[:, None])**2, 0).sum(1), count,
                             out=np.full_like(mean, np.nan), where=count > 0)
        ordered = np.sort(np.where(finite, x, -np.inf), axis=1)
        maximum = np.where(count > 0, ordered[:, -1], np.nan)
        top = ordered[:, -min(2, x.shape[1]):]
        top_count = np.isfinite(top).sum(1)
        top_mean = np.divide(np.where(np.isfinite(top), top, 0).sum(1), top_count,
                             out=np.full_like(mean, np.nan), where=top_count > 0)
        return np.stack([mean, np.sqrt(variance), maximum, top_mean], axis=-1).reshape(len(x), -1)

    output = []
    for name in names:
        local = nodes[:, families == name]
        output.append(np.column_stack((pool(local[:, :, h_columns]), pool(local[:, :, e_columns]), pooled[:, 236:260])))
    return np.stack(output, axis=1).astype(np.float32)


def family_edges(window, reference, families, names, lags=(0, 1, 7), minimum=6):
    pairs = list(combinations(range(len(names)), 2))
    output = [[] for _ in pairs]
    for lag in lags:
        a, b = (window[lag:], window[:-lag]) if lag else (window, window)
        ar, br = (reference[lag:], reference[:-lag]) if lag else (reference, reference)
        recent, reliability = correlation_matrix(a, b, minimum=minimum)
        baseline, _ = correlation_matrix(ar, br, minimum=minimum)
        valid = np.isfinite(recent) & np.isfinite(baseline)
        if lag == 0:
            valid &= np.triu(np.ones_like(valid), 1).astype(bool)
        for k, (i, j) in enumerate(pairs):
            between = ((families[:, None] == names[i]) & (families[None, :] == names[j])) | ((families[:, None] == names[j]) & (families[None, :] == names[i]))
            keep = valid & between
            if not keep.any():
                output[k].extend([np.nan]*4)
                continue
            w = reliability[keep]
            delta = w*(recent[keep]-baseline[keep])
            current = baseline[keep]+delta
            output[k].extend([np.average(abs(current), weights=w), np.average(abs(delta), weights=w),
                              np.max(abs(delta)), np.average(delta, weights=w)])
    return np.asarray(output, dtype=np.float32)


class SplineCoordinates:
    def fit(self, x, groups):
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        z = np.clip(self.scaler.fit_transform(self.imputer.fit_transform(x)), -8, 8)
        self.basis = SplineTransformer(n_knots=3, degree=2, include_bias=False, extrapolation="linear")
        z = self.basis.fit_transform(z)
        self.basis_scaler = StandardScaler().fit(z)
        repeats = z.shape[1]//x.shape[1]
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        self.penalty = np.repeat(np.sqrt(counts[inverse]*repeats), repeats)
        return self

    def transform(self, x):
        z = np.clip(self.scaler.transform(self.imputer.transform(x)), -8, 8)
        return self.basis_scaler.transform(self.basis.transform(z))/self.penalty


class FamilyGraphWarning:
    def __init__(self, C=.1, mode="conditional-graph", max_iter=400):
        self.C, self.mode, self.max_iter = C, mode, max_iter

    @staticmethod
    def patterns(families):
        states = np.asarray(list(product([0., 1.], repeat=families)))
        edges = np.asarray([states[:, i]*states[:, j] for i, j in combinations(range(families), 2)]).T
        return states, edges

    def fit(self, unary, pair, family_y, eligible, person):
        n, families, dimensions = unary.shape
        pairs = pair.shape[1]
        pair_eligible = np.column_stack([eligible[:, i] & eligible[:, j] for i, j in combinations(range(families), 2)])
        unary_groups = np.asarray(["H"]*96+["E"]*128+["H"]*(dimensions-224))
        self.unary_transform = SplineCoordinates().fit(unary.reshape(-1, dimensions)[eligible.ravel()], unary_groups)
        u = self.unary_transform.transform(unary.reshape(-1, dimensions))
        if self.mode == "conditional-graph":
            self.pair_transform = SplineCoordinates().fit(pair.reshape(-1, 72)[pair_eligible.ravel()], np.asarray(["N"]*36+["NE"]*36))
            v = self.pair_transform.transform(pair.reshape(-1, 72))
        else:
            v = np.zeros((n*pairs, 0))
        self.unary_dimension, self.pair_dimension = u.shape[1], v.shape[1]
        states, edge_states = self.patterns(families)
        possible = ~np.any((states[None, :, :] > 0) & ~eligible[:, None, :], axis=2)
        known = (family_y >= 0) & eligible
        compatible = possible & ~np.any(known[:, None, :] & (states[None, :, :] != family_y[:, None, :]), axis=2)
        _, inverse, counts = np.unique(person, return_inverse=True, return_counts=True)
        weight = 1./counts[inverse]
        weight /= weight.sum()
        weight /= np.maximum(known.sum(1), 1)
        d, e = u.shape[1], v.shape[1]
        has_edge = self.mode != "independent"
        total = d+1+(e+1 if has_edge else 0)
        initial = np.zeros(total)
        prevalence = np.sum(weight[:, None]*known*np.maximum(family_y, 0))/np.sum(weight[:, None]*known)
        initial[d] = np.log(np.clip(prevalence, 1e-6, 1-1e-6)/np.clip(1-prevalence, 1e-6, 1))
        penalty = np.ones(total)/(self.C*n)
        penalty[d] = 0.
        if has_edge:
            penalty[-1] = 0.

        def objective(parameters):
            logits = (u@parameters[:d]+parameters[d]).reshape(n, families)
            energy = logits@states.T
            if has_edge:
                coupling = (v@parameters[d+1:d+1+e]+parameters[-1]).reshape(n, pairs)
                energy += coupling@edge_states.T
            energies = np.where(possible, energy, -np.inf)
            allowed = np.where(compatible, energy, -np.inf)
            normalizer = logsumexp(energies, axis=1)
            observed = logsumexp(allowed, axis=1)
            difference = (np.exp(energies-normalizer[:, None])-np.exp(allowed-observed[:, None]))*weight[:, None]
            unary_delta = (difference@states).ravel()
            gradient = np.zeros(total)
            gradient[:d] = u.T@unary_delta
            gradient[d] = unary_delta.sum()
            if has_edge:
                edge_delta = (difference@edge_states).ravel()
                gradient[d+1:d+1+e] = v.T@edge_delta
                gradient[-1] = edge_delta.sum()
            loss = np.dot(weight, normalizer-observed)+.5*np.sum(penalty*parameters**2)
            return loss, gradient+penalty*parameters

        result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                          options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-7})
        self.parameters = result.x
        self.optimization = {"success": bool(result.success), "message": str(result.message),
                             "iterations": int(result.nit), "objective": float(result.fun),
                             "maximum_gradient": float(np.max(abs(result.jac)))}
        return self

    def predict(self, unary, pair, eligible):
        n, families, dimensions = unary.shape
        u = self.unary_transform.transform(unary.reshape(-1, dimensions))
        d, e = self.unary_dimension, self.pair_dimension
        states, edge_states = self.patterns(families)
        logits = (u@self.parameters[:d]+self.parameters[d]).reshape(n, families)
        energy = logits@states.T
        if self.mode != "independent":
            v = self.pair_transform.transform(pair.reshape(-1, 72)) if e else np.zeros((n*pair.shape[1], 0))
            coupling = (v@self.parameters[d+1:d+1+e]+self.parameters[-1]).reshape(n, pair.shape[1])
            energy += coupling@edge_states.T
        possible = ~np.any((states[None, :, :] > 0) & ~eligible[:, None, :], axis=2)
        energy = np.where(possible, energy, -np.inf)
        posterior = np.exp(energy-logsumexp(energy, axis=1, keepdims=True))
        return posterior[:, states.sum(1) >= 2].sum(1), posterior@states
