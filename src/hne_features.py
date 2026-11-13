"""Personal-reference, context-residual and relation-change features for HNE."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCALES = (24, 72, 168)
LAGS = (0, 6, 24)
NODE_STATS = ("bias", "abs_bias", "amplitude", "sd", "tail", "drift", "memory1", "memory24")


def correlation(a, b, shrink=24, minimum=12):
    good = np.isfinite(a) & np.isfinite(b)
    n = int(good.sum())
    if n < minimum:
        return np.nan, 0.
    x, y = a[good], b[good]
    den = np.std(x) * np.std(y)
    if den < 1e-8:
        return np.nan, 0.
    rho = np.mean((x-x.mean()) * (y-y.mean())) / den
    reliability = (n-3) / (n-3+shrink)
    return float(np.clip(rho, -1., 1.)), reliability


def lagged_context(x, lags=LAGS):
    columns = []
    for lag in lags:
        a = np.full_like(x, np.nan)
        if lag:
            a[lag:] = x[:-lag]
        else:
            a[:] = x
        columns.append(a)
    return np.column_stack(columns)


def conditional_residuals(z, reference_hours=672, ridge=10.):
    """Estimate the person's normal context response using the calibration prefix."""
    n = z.shape[1] - 1
    context = lagged_context(z[:, -1])
    adjusted = z[:, :n].copy()
    coefficients = np.zeros((n, context.shape[1]+1), np.float32)
    context_support = np.zeros(n, np.float32)
    for j in range(n):
        ok = np.isfinite(context[:reference_hours]).all(1) & np.isfinite(z[:reference_hours, j])
        count = int(ok.sum())
        if count < 24:
            continue
        u = np.column_stack((np.ones(count), np.clip(context[:reference_hours][ok], -8, 8)))
        target = np.clip(z[:reference_hours, j][ok], -12, 12)
        penalty = np.eye(u.shape[1]) * ridge
        penalty[0, 0] = 0
        beta = np.linalg.solve(u.T @ u + penalty, u.T @ target)
        valid = np.isfinite(context).all(1)
        v = np.column_stack((np.ones(valid.sum()), np.clip(context[valid], -8, 8)))
        adjusted[valid, j] -= v @ beta
        # Out-of-block reference errors are comparable to monitoring errors.
        for start in range(0, reference_hours, 168):
            held = np.arange(reference_hours) // 168 == start // 168
            fit = ok & ~held
            target_rows = held & np.isfinite(context[:reference_hours]).all(1)
            if fit.sum() < 24:
                adjusted[:reference_hours, j][target_rows] = z[:reference_hours, j][target_rows]
                continue
            design = np.column_stack((np.ones(fit.sum()), np.clip(context[:reference_hours][fit], -8, 8)))
            response = np.clip(z[:reference_hours, j][fit], -12, 12)
            fold_beta = np.linalg.solve(design.T @ design + penalty, design.T @ response)
            held_design = np.column_stack((np.ones(target_rows.sum()), np.clip(context[:reference_hours][target_rows], -8, 8)))
            adjusted[:reference_hours, j][target_rows] = z[:reference_hours, j][target_rows] - held_design @ fold_beta
        coefficients[j] = beta
        context_support[j] = count/reference_hours
    return adjusted, coefficients, context_support


def series_stats(x, reference, memory_lags=(1,24), minimum=8, reference_minimum=24, pair_minimum=12):
    vals = x[np.isfinite(x)]
    ref = reference[np.isfinite(reference)]
    if len(vals) < minimum or len(ref) < reference_minimum:
        return np.full(len(NODE_STATS), np.nan)
    refsd = max(float(np.std(ref)), .1)
    refamp = max(float(np.quantile(ref, .9)-np.quantile(ref, .1)), .2)
    center = np.median(ref)
    mu = float(vals.mean()-center)/refsd
    amplitude = np.log1p((np.quantile(vals, .9)-np.quantile(vals, .1))/refamp)
    sd = np.log1p(np.std(vals)/refsd)
    tail = np.mean(np.abs(vals-center) > max(np.quantile(np.abs(ref-center), .9), .1))
    halves = np.array_split(x, 2)
    drift = (np.nanmean(halves[1])-np.nanmean(halves[0]))/refsd if all(np.isfinite(v).sum() >= max(2,minimum//2) for v in halves) else np.nan
    ac = []
    for lag in memory_lags:
        r, reliability = correlation(x[lag:], x[:-lag],minimum=pair_minimum) if len(x) > lag else (np.nan, 0)
        b, _ = correlation(reference[lag:], reference[:-lag],minimum=pair_minimum)
        ac.append(reliability*(r-b))
    return np.array([mu, abs(mu), amplitude, sd, tail, drift, *ac], np.float32)


def correlation_matrix(a,b,minimum=12,shrink=24):
    """All paired-observation correlations without a Python loop over edges."""
    ma=np.isfinite(a);mb=np.isfinite(b)
    za=np.where(ma,a,0.).astype(np.float64);zb=np.where(mb,b,0.).astype(np.float64)
    am=za.sum(0)/np.maximum(ma.sum(0),1);bm=zb.sum(0)/np.maximum(mb.sum(0),1)
    za=np.where(ma,za-am,0.);zb=np.where(mb,zb-bm,0.)
    ma=ma.astype(float);mb=mb.astype(float)
    count=ma.T@mb;n=np.maximum(count,1.)
    sa=za.T@mb;sb=ma.T@zb
    va=(za*za).T@mb-sa*sa/n;vb=ma.T@(zb*zb)-sb*sb/n
    denominator=np.sqrt(np.maximum(va,0.)*np.maximum(vb,0.))
    good=(count>=minimum)&(denominator/n>=1e-8)
    corr=np.divide(za.T@zb-sa*sb/n,denominator,out=np.full_like(n,np.nan),where=good)
    reliability=np.where(good,(n-3)/(n-3+shrink),0.)
    return np.clip(corr,-1.,1.),reliability


def graph_stats(window, reference, lags=LAGS, pair_minimum=12):
    """Directed lag graphs; reliability is determined by joint observed support."""
    features=[]
    for lag in lags:
        a,b=(window[lag:],window[:-lag]) if lag else (window,window)
        ar,br=(reference[lag:],reference[:-lag]) if lag else (reference,reference)
        recent,reliability=correlation_matrix(a,b,minimum=pair_minimum)
        baseline,_=correlation_matrix(ar,br,minimum=pair_minimum)
        valid=np.isfinite(recent)&np.isfinite(baseline)
        valid &= np.triu(np.ones_like(valid),1).astype(bool) if lag==0 else ~np.eye(len(valid),dtype=bool)
        if not valid.any():
            features.extend([np.nan]*4);continue
        w=reliability[valid];delta=w*(recent[valid]-baseline[valid]);current=baseline[valid]+delta
        features.extend([np.average(abs(current),weights=w),np.average(abs(delta),weights=w),
                         np.max(abs(delta)),np.average(delta,weights=w)])
    return features


def pool_nodes(nodes):
    """Permutation-invariant set summaries; no channel identity or fixed flattening."""
    out = []
    for values in nodes.T:
        valid = values[np.isfinite(values)]
        if not len(valid):
            out.extend([np.nan]*4)
        else:
            out.extend([np.mean(valid), np.std(valid), np.max(valid), np.mean(np.sort(valid)[-2:])])
    return np.asarray(out, np.float32)


def normalized_series(folder, person, ref):
    raw = np.load(folder / f"{person}_hourly.npz")["values"]
    profile = np.asarray(ref["profile"])
    scale = np.asarray(ref["scale"])
    states = (raw[:, :-1]-profile[np.arange(len(raw))%24])/scale
    cscale = ref["context_scale"]
    context = (raw[:, -1]-ref["context_center"])/(cscale if cscale > 1e-6 else 1)
    return np.column_stack((states, context)).astype(np.float32)


def extract_person(z, cutoffs):
    """Every feature at a cutoff uses only the reference prefix and past samples."""
    adjusted, beta, support = conditional_residuals(z)
    ref = z[:672, :-1]
    aref = adjusted[:672]
    nodes, pooled, groups = [], [], None
    for cutoff in cutoffs:
        node_parts, graph_raw, graph_adjusted, context_parts = [], [], [], []
        for hours in SCALES:
            raw = z[cutoff-hours:cutoff, :-1]
            adj = adjusted[cutoff-hours:cutoff]
            node_parts.append(np.stack([series_stats(raw[:, j], ref[:, j]) for j in range(raw.shape[1])]))
            node_parts.append(np.stack([series_stats(adj[:, j], aref[:, j]) for j in range(raw.shape[1])]))
            graph_raw.extend(graph_stats(raw, ref))
            graph_adjusted.extend(graph_stats(adj, aref))
            context_parts.extend(series_stats(z[cutoff-hours:cutoff, -1], z[:672, -1]))
        node = np.concatenate(node_parts, axis=1)
        # Context-effect changes are personal calibration features, not subject identifiers.
        effect = np.column_stack((np.linalg.norm(beta[:, 1:], axis=1), support))
        node = np.column_stack((node, effect))
        hp = pool_nodes(node[:, np.r_[0:8,16:24,32:40]])
        ep = np.r_[pool_nodes(node[:, np.r_[8:16,24:32,40:50]]), context_parts]
        nr, ne = np.asarray(graph_raw), np.asarray(graph_adjusted)
        feature = np.r_[hp, nr, ep, ne].astype(np.float32)
        group = ["H"]*len(hp) + ["N"]*len(nr) + ["E"]*len(ep) + ["NE"]*len(ne)
        nodes.append(node)
        pooled.append(feature)
        groups = group
    return np.stack(nodes), np.stack(pooled), groups


def build_features(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for folder in sorted(root.iterdir()):
        if not (folder / "samples.csv").exists():
            continue
        meta = pd.read_csv(folder / "samples.csv", dtype={"person_id": str})
        refs = json.loads((folder / "references.json").read_text())
        all_nodes, all_pooled, row_indices = [], [], []
        for person, frame in meta.groupby("person_id", sort=False):
            z = normalized_series(folder, person, refs[person])
            nodes, pooled, groups = extract_person(z, frame.issue_time_hours.to_numpy())
            all_nodes.extend(nodes)
            all_pooled.extend(pooled)
            row_indices.extend(frame.index)
        order = np.argsort(row_indices)
        np.savez_compressed(output/f"{folder.name}.npz", nodes=np.stack(all_nodes)[order],
                            pooled=np.stack(all_pooled)[order], groups=np.asarray(groups),
                            y=meta.y.to_numpy(), fold=meta.fold.to_numpy())
        meta.to_csv(output/f"{folder.name}.csv", index=False)
        print(json.dumps(dict(dataset=folder.name, samples=len(meta), nodes=np.stack(all_nodes).shape,
                              pooled=np.stack(all_pooled).shape)), flush=True)
