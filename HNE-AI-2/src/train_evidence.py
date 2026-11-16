"""Event-label-free cross-fitting, personal adapters and held-out calibration."""
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from precursor_guidance import ConditionalPredictor, patch_nll, innovation_features


def family_mean(values, valid, family):
    """Equal target-family weight, equal fields within each family."""
    means = []
    for f in torch.unique(family):
        channels = []
        for c in torch.where(family == f)[0]:
            m = valid[c]
            if m.any():
                channels.append(values[c][m].mean())
        if channels:
            means.append(torch.stack(channels).mean())
    return torch.stack(means).mean() if means else values.sum() * 0


def predict(model, person, start, end, reference=None):
    window = person.small_window(start, end)
    if reference is None:
        ref = person.small_window(0, 28)
        reference = model.reference_summary(ref['values'], ref['observed'], ref['calendar'], ref['state'])
    return model(**window, reference=reference), window


def forecast_loss(prediction, window, burn=21):
    x, obs = window['values'], window['observed']
    state, family = window['state'][0], window['family'][0]
    losses = []
    for key in ('H', 'HN', 'HNE'):
        loss, valid = patch_nll(x, obs, *prediction['views'][key])
        valid = valid[0] & state[:, None]
        valid[:, :burn] = False
        losses.append(family_mean(loss[0], valid, family))
    mu, scale = prediction['views']['pair']
    target = x.permute(0, 2, 1, 3)[:, :, :, None].expand_as(mu)
    observed = obs.permute(0, 2, 1, 3)[:, :, :, None].expand_as(mu)
    pair, valid = patch_nll(target, observed, mu, scale)
    valid &= prediction['pair_allowed'] & state[None, None, :, None]
    valid[:, :burn] = False
    # Enumerate sources, averaging within each source family then across families.
    # This is the expectation of uniform one-source-per-family sampling.
    family_losses, family_valid = [], []
    for source_family in torch.unique(family):
        use = family == source_family
        mask = valid[0, :, :, use]
        mean = (pair[0, :, :, use] * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        family_losses.append(mean); family_valid.append(mask.any(-1))
    values = torch.stack(family_losses, -1)
    masks = torch.stack(family_valid, -1)
    mean = (values * masks).sum(-1) / masks.sum(-1).clamp_min(1)
    pair_loss = family_mean(mean.T, masks.any(-1).T, family)
    return torch.stack(losses).mean() + .25 * pair_loss


def fit_small_components(people, assignments, config, directory, log):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    models, report = {}, []
    for component in range(3):
        target = directory / f'component{component}.pt'
        if target.exists():
            saved = torch.load(target, map_location='cpu', weights_only=False)
            model = ConditionalPredictor(); model.load_state_dict(saved['state'])
            models[component] = model.eval().requires_grad_(False)
            report.append(saved['report']); continue
        pool = [p for p in people['train'] if assignments[p.person] != component]
        if len(pool) < 2:
            raise ValueError('At least two training people required outside each S fold')
        # Only this training-pool person's forecast NLL selects the epoch budget.
        fit, hold = pool[:-1], pool[-1:]
        seed = config['seed'] + 101 * component
        torch.manual_seed(seed)
        initial = ConditionalPredictor()
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['small_lr'], weight_decay=.0001)
        best, selected, stale, curve = float('inf'), 1, 0, []
        start = time.perf_counter()
        for epoch in range(1, config['small_epochs'] + 1):
            model.train(); epoch_loss = []
            order = np.random.default_rng(seed + epoch).permutation(len(fit))
            for i in order:
                optimizer.zero_grad()
                loss = sum(forecast_loss(*predict(model, fit[i], a, b)) for a, b in ((14, 42), (28, 56))) / 2
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite S loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step(); epoch_loss.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                val = float(sum(forecast_loss(*predict(model, hold[0], a, b)) for a, b in ((14, 42), (28, 56))) / 2)
            curve.append(dict(epoch=epoch, train=float(np.mean(epoch_loss)), held_out_forecast_nll=val))
            if val < best:
                best, selected, stale = val, epoch, 0
            else:
                stale += 1
            if stale >= config['small_patience']:
                break
        # Refit on the entire permissible pool for the internally selected epochs.
        model = initial
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['small_lr'], weight_decay=.0001)
        for epoch in range(selected):
            for i in np.random.default_rng(seed + epoch).permutation(len(pool)):
                optimizer.zero_grad()
                loss = sum(forecast_loss(*predict(model, pool[i], a, b)) for a, b in ((14, 42), (28, 56))) / 2
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        entry = dict(component=component, fit_people=[p.person for p in pool],
                     selection_holdout=[p.person for p in hold], selected_epochs=selected,
                     elapsed_seconds=time.perf_counter() - start, curve=curve,
                     event_labels_used=False)
        torch.save(dict(state=model.state_dict(), report=entry), target)
        models[component] = model.eval().requires_grad_(False); report.append(entry)
        log('small_component_complete', **{k: v for k, v in entry.items() if k != 'curve'})
    return models, report


def adapt_prediction(pred, bias, log_scale):
    result = {k: v for k, v in pred.items() if k != 'views'}
    result['views'] = {}
    for i, key in enumerate(('H', 'HN', 'HNE')):
        mu, sigma = pred['views'][key]
        result['views'][key] = (mu + bias[i][None, :, None, None],
                                 sigma * log_scale[i][None, :, None, None].exp())
    mu, sigma = pred['views']['pair']
    result['views']['pair'] = (mu + bias[0][None, None, :, None, None],
                               sigma * log_scale[0][None, None, :, None, None].exp())
    return result


def robust_stats(values, valid, axis):
    masked = np.where(np.expand_dims(valid, -1) if values.ndim == valid.ndim + 1 else valid, values, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(masked, axis=axis)
        mad = np.nanmedian(np.abs(masked - np.expand_dims(median, axis)), axis=axis)
    return np.nan_to_num(median), np.maximum(np.nan_to_num(mad) * 1.4826, .1)


def calibrate_person(model, person, steps=100):
    with torch.no_grad():
        ref = person.small_window(0, 28)
        reference = model.reference_summary(ref['values'], ref['observed'], ref['calendar'], ref['state'])
        pred, window = predict(model, person, 14, 42, reference)
    bias = torch.zeros(3, 6, requires_grad=True)
    scale = torch.zeros(3, 6, requires_grad=True)
    optimizer = torch.optim.Adam([bias, scale], lr=.03)
    for _ in range(steps):
        optimizer.zero_grad()
        adjusted = adapt_prediction(pred, bias, scale)
        # Personal fitting uses the three standard views; pair shares H's adapter.
        values = []
        for key in ('H', 'HN', 'HNE'):
            nll, valid = patch_nll(window['values'], window['observed'], *adjusted['views'][key])
            valid = valid[0] & person.state[0, :, None]; valid[:, :21] = False
            values.append(family_mean(nll[0], valid, person.family[0]))
        loss = torch.stack(values).mean() + .1 * (bias.square().mean() + scale.square().mean())
        loss.backward(); optimizer.step()
        with torch.no_grad():
            bias.clamp_(-2, 2); scale.clamp_(-math.log(2), math.log(2))
    with torch.no_grad():
        pred, window = predict(model, person, 28, 56, reference)
        node, valid, edge, edge_valid = innovation_features(adapt_prediction(pred, bias, scale), window['values'], window['observed'])
    node, valid = node[0, :, 21:].numpy(), valid[0, :, 21:].numpy() & person.state[0].numpy()[:, None]
    edge, edge_valid = edge[0, 21:].numpy(), edge_valid[0, 21:].numpy() & person.state[0].numpy()[None, :, None]
    nm, ns = robust_stats(node[..., :8], valid, axis=1)
    em, es = robust_stats(edge, edge_valid, axis=0)
    return dict(bias=bias.detach(), log_scale=scale.detach(), reference=reference.detach(),
                node_median=nm, node_scale=ns, node_available=valid.sum(1) >= 12,
                edge_median=em, edge_scale=es, edge_available=edge_valid.sum(0) >= 12,
                reference_coverage=window['observed'][0, :, 21:].float().mean((1, 2)).numpy())


def _family_average(x, mask, family):
    vals = []
    for f in np.unique(family):
        channels = []
        for c in np.flatnonzero(family == f):
            if mask[c].any():
                channels.append(float(np.mean(x[c][mask[c]])))
        if channels:
            vals.append(np.mean(channels))
    return float(np.mean(vals)) if vals else 0.


def _edge_average(x, mask, family, include_absent=False):
    """Equal source-family and target-family contributions to scalar summaries."""
    source_means, source_valid = [], []
    for f in np.unique(family):
        selected = family == f
        valid = mask[..., selected]
        if include_absent:
            source_means.append(x[..., selected].mean(-1))
            source_valid.append(np.ones_like(valid.any(-1)))
        else:
            source_means.append((x[..., selected] * valid).sum(-1) / valid.sum(-1).clip(1))
            source_valid.append(valid.any(-1))
    values, valid = np.stack(source_means, -1), np.stack(source_valid, -1)
    mean = (values * valid).sum(-1) / valid.sum(-1).clip(1)
    targets = valid.any(-1).T
    targets[-1] = False  # Observed context is never an event-state target.
    return _family_average(mean.T, targets, family)


@torch.no_grad()
def origin_evidence(model, person, calibration, origin):
    pred, window = predict(model, person, origin + 1 - 28, origin + 1, calibration['reference'])
    pred = adapt_prediction(pred, calibration['bias'], calibration['log_scale'])
    raw, valid, edge, edge_valid = innovation_features(pred, window['values'], window['observed'])
    raw = raw[0, :, 21:].numpy()
    nv = valid[0, :, 21:].numpy() & calibration['node_available'][:, None] & person.state[0].numpy()[:, None]
    ev = edge_valid[0, 21:].numpy() & calibration['edge_available'][None] & person.state[0].numpy()[None, :, None]
    node = raw.copy()
    node[..., :8] = np.clip(np.arcsinh((raw[..., :8] - calibration['node_median'][:, None]) / calibration['node_scale'][:, None]), -5, 5)
    node[~nv] = 0
    ef = np.zeros((21, 6, 6, 4), dtype='float32')
    ef[..., 0] = np.clip(np.arcsinh((edge[0, 21:].numpy() - calibration['edge_median']) / calibration['edge_scale']), -5, 5)
    ef[..., 1] = np.clip(np.arcsinh(calibration['edge_median'] / calibration['edge_scale']), -5, 5)
    coverage = window['observed'][0].float().mean(-1).numpy()
    ef[..., 2] = np.minimum(coverage[:, 21:].T[:, :, None], coverage[:, 20:41].T[:, None, :])
    ef[..., 3] = person.state[0].numpy()[None, None, :]
    ef[~ev] = 0
    fam = person.family[0].numpy(); state = person.state[0].numpy()
    all_state = np.broadcast_to(state[:, None], (6, 21))
    def avg(x, mask=nv):
        return _family_average(x, mask, fam)
    refcov = calibration['reference_coverage']
    summary = [avg(raw[..., 8], all_state), float(coverage[5, 21:].mean()),
               avg(nv.astype(float), all_state), _edge_average(ev.astype(float), ev, fam, include_absent=True),
               avg(np.broadcast_to(calibration['node_median'][:, 2, None], (6, 21))),
               avg(np.broadcast_to(np.abs(calibration['node_median'][:, 5, None]), (6, 21))),
               avg(np.abs(node[..., 2])), avg(np.abs(node[..., 5])), avg(node[..., 6]), avg(node[..., 7]),
               _edge_average(np.abs(ef[..., 0]), ev, fam),
               avg(raw[..., 8] - refcov[:, None], all_state)]
    raw_summary = np.asarray(summary, dtype='float32')
    summary = raw_summary.copy()
    # Reference NLL/residual quality can be arbitrarily large under a distribution
    # shift. Keep the raw diagnostics but bound the two MLP inputs identically in
    # every fusion/gating method; the transform never uses event labels.
    summary[4:6] = np.clip(np.arcsinh(summary[4:6]), -5, 5)
    if not all(np.isfinite(a).all() for a in (node, ef, summary)):
        raise FloatingPointError('Nonfinite calibrated evidence')
    return dict(node=torch.from_numpy(node), node_valid=torch.from_numpy(nv),
                edge=torch.from_numpy(ef), edge_valid=torch.from_numpy(ev), summary=torch.from_numpy(summary),
                raw_node=torch.from_numpy(raw), raw_edge=edge[0, 21:].clone(),
                raw_summary=torch.from_numpy(raw_summary))
