"""Personal lag dynamics and environmental context as a shared normal reference."""
from __future__ import annotations

import numpy as np

from .hne_representation import Sampling, encode_person, lag_context


VARIANTS = ("HNE-dynamic-reference", "HNE-own-history-reference")


def dynamic_reference(states, contexts, config, own_only=False):
    states, contexts = np.asarray(states, float), np.asarray(contexts, float)
    n, p = states.shape
    reference = config['reference_days']
    past = lag_context(states, config['state_lags'])
    environment = lag_context(contexts, config['context_lags'])
    residuals = np.full_like(states, np.nan)
    effect = np.zeros((p, 2), dtype=np.float32)
    diagnostics = []
    for node in range(p):
        own = np.array([k*p+node for k in range(len(config['state_lags']))])
        others = np.setdiff1d(np.arange(past.shape[1]), own)
        components = [past[:, own]] if own_only else [past[:, own], past[:, others], environment]
        design, supported, environment_columns = [], [], []
        offset = 0
        for group, values in enumerate(components):
            support = np.isfinite(values[:reference]).sum(0) >= config['minimum_predictor_observations']
            observed = np.isfinite(values) & support[None]
            selected = np.flatnonzero(support)
            width = len(selected)
            if not width:
                continue
            block = np.column_stack((np.where(observed[:, selected], np.clip(values[:, selected], -8, 8), 0.),
                                     observed[:, selected].astype(float)))
            block /= np.sqrt(width)
            design.append(block)
            supported.append(observed[:, selected].any(1))
            if group == 2 and not own_only:
                environment_columns.extend(range(offset, offset+width))
            offset += 2*width
        if not design:
            diagnostics.append(dict(node=node, supported_reference=0, calibrated_reference=0))
            continue
        x = np.column_stack(design)
        any_input = np.any(supported, axis=0)
        valid = np.isfinite(states[:reference, node]) & any_input[:reference]
        if valid.sum() < config['minimum_target_observations']:
            diagnostics.append(dict(node=node, supported_reference=int(valid.sum()), calibrated_reference=0))
            continue
        y = np.clip(states[:reference, node], -12, 12)

        def fit(mask):
            a, b = x[:reference][mask], y[mask]
            center, mean = a.mean(0), b.mean()
            centered, target = a-center, b-mean
            lam = len(a)*config['ridge_per_observation']
            if a.shape[1] > len(a):
                coefficient = centered.T @ np.linalg.solve(centered@centered.T+lam*np.eye(len(a)), target)
            else:
                coefficient = np.linalg.solve(centered.T@centered+lam*np.eye(a.shape[1]), centered.T@target)
            return coefficient, mean-center@coefficient

        beta, intercept = fit(valid)
        monitoring = np.arange(n) >= reference
        use = monitoring & any_input
        residuals[use, node] = states[use, node]-x[use]@beta-intercept
        # The reference is fully observed before any warning origin. Its
        # held-block prediction errors avoid fitting and scoring the same targets.
        residuals[:reference, node] = np.nan
        embargo = max(config['state_lags'])
        for start in range(0, reference, 14):
            held = (np.arange(reference) >= start) & (np.arange(reference) < start+14)
            excluded = (np.arange(reference) >= start-embargo) & (np.arange(reference) < start+14+embargo)
            training = valid & ~excluded
            if training.sum() < config['minimum_target_observations']:
                continue
            local_beta, local_intercept = fit(training)
            rows = np.flatnonzero(held & valid)
            residuals[rows, node] = states[rows, node]-x[rows]@local_beta-local_intercept
        calibrated = np.isfinite(residuals[:reference, node])
        ce = np.asarray(environment_columns, int)
        # Each environment value column was divided by sqrt(number of supported
        # environment values); convert back before reporting coefficient size.
        effect[node] = [np.linalg.norm(beta[ce])/max(len(ce), 1), valid.mean()]
        diagnostics.append(dict(node=node, supported_reference=int(valid.sum()),
            calibrated_reference=int(calibrated.sum()),
            reference_mse=float(np.mean(residuals[:reference, node][calibrated]**2)) if calibrated.any() else None,
            reference_state_mse=float(np.mean(states[:reference, node][calibrated]**2)) if calibrated.any() else None))
    return residuals, effect, diagnostics


def encode_dynamic(states, contexts, cutoffs, config):
    output, details, references = {}, [], []
    for name, own_only in zip(VARIANTS, [False, True]):
        residual, effect, diagnostics = dynamic_reference(states, contexts, config, own_only)
        nodes, pooled, groups = encode_person(states, contexts, cutoffs, Sampling.daily(), conditioned=(residual, effect))
        output[name] = pooled
        references.append(residual[:config['reference_days']])
        details.extend(dict(variant=name, **row) for row in diagnostics)
    output['reference_residuals'] = np.stack(references)
    return output, details
