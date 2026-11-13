"""Fit the two fixed future-trajectory conditions on the original development folds."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_trajectory import TrajectoryHNE, joint_loss, simulated_event_probability
from src.hne_set import _person_weights
from src.hne_temporal import _device
from scripts.run_temporal_development import _metrics

REFERENCE = ['eligible', 'amplitude_threshold', 'bias_threshold', 'absolute_point_threshold']


def load_development(dataset):
    base = ROOT/'data/features/unified_daily'
    meta = pd.read_csv(base/f'{dataset}.csv', dtype={'person_id': str})
    with np.load(base/f'{dataset}.npz') as z:
        data = {k: z[k] for k in ('nodes', 'pooled', 'windows', 'groups', 'y', 'fold')}
    n, p = data['nodes'].shape[:2]
    data['target'] = np.full((n, p, 7), np.nan, np.float64)
    data['families'] = np.zeros((n, p), np.int64)
    for name in REFERENCE:
        data[name] = np.zeros((n, p), bool if name == 'eligible' else np.float64)
    for person, rows in meta.groupby('person_id', sort=False):
        with np.load(ROOT/f'data/daily_study/{dataset}/{person}_daily.npz') as z:
            normalized = (z['states']-z['reference_location'])/z['reference_scale']
            # Both original cohorts retain every state channel (10 < budget 32).
            if normalized.shape[1] != p:
                raise ValueError('Trajectory decoder and event variables must share the same coordinates')
            _, family = np.unique(z['state_families'], return_inverse=True)
            data['families'][rows.index] = family
            for name in REFERENCE:
                data[name][rows.index] = z['reference_'+name]
            for index, row in rows.iterrows():
                d = int(row.origin_day)
                data['target'][index] = normalized[d+2:d+9].T
    data['person'] = meta.person_id.to_numpy()
    return data, meta


def tensors(data, indices, device, names=('windows', 'nodes', 'pooled')):
    return [torch.as_tensor(data[k][indices], device=device) for k in names]


def evaluate_loss(model, data, indices, device, batch_size):
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start+batch_size]
            parts = joint_loss(model(*tensors(data, ix, device)),
                               torch.as_tensor(data['target'][ix], device=device))
            losses.append(torch.stack(parts, 1).cpu().numpy())
    losses = np.concatenate(losses)
    weights = _person_weights(data['person'][indices])
    return weights @ losses


def predict(model, data, indices, protocol, cfg, device):
    model.eval()
    result, availability = [], []
    generator = torch.Generator(device=device).manual_seed(cfg['monte_carlo_seed'])
    with torch.inference_mode():
        for start in range(0, len(indices), cfg['batch_size']):
            ix = indices[start:start+cfg['batch_size']]
            reference = {k: torch.as_tensor(data[k][ix], device=device) for k in REFERENCE}
            score, resolved = simulated_event_probability(
                model(*tensors(data, ix, device)), reference,
                torch.as_tensor(data['families'][ix], device=device), protocol,
                cfg['monte_carlo_draws'], generator)
            result.append(score); availability.append(resolved)
    return np.concatenate(result), np.concatenate(availability)


def fit_one(data, train, validation, held, rank, seed, cfg, protocol, output, device):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metadata.json').exists():
        with np.load(output/'predictions.npz') as z:
            return z['validation_probability'], z['held_probability']
    started = time.time()
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = TrajectoryHNE(data['groups'], data['windows'].shape[1], rank)
    weights = _person_weights(data['person'][train])
    model.fit_input_scalers(*(data[k][train] for k in ('windows', 'nodes', 'pooled')), weights)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(seed)
    train_weights = torch.as_tensor(weights*len(train), dtype=torch.float32, device=device)
    best, best_loss, best_epoch, stale, history = None, float('inf'), 0, 0, []
    for epoch in range(1, cfg['max_epochs']+1):
        model.train()
        order = rng.permutation(len(train))
        training_sum = 0.
        for start in range(0, len(train), cfg['batch_size']):
            local = order[start:start+cfg['batch_size']]
            ix = train[local]
            optimizer.zero_grad(set_to_none=True)
            per_origin = joint_loss(model(*tensors(data, ix, device)),
                                    torch.as_tensor(data['target'][ix], device=device))[0]
            loss = (per_origin*train_weights[local]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            training_sum += float(loss.detach())*len(ix)
        total, value, observation = evaluate_loss(model, data, validation, device, cfg['batch_size'])
        history.append(dict(epoch=epoch, train_loss=training_sum/len(train),
                            validation_loss=float(total), value_nll=float(value), observation_nll=float(observation)))
        if total < best_loss:
            best_loss, best_epoch, stale = float(total), epoch, 0
            best = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if stale >= cfg['patience']:
            break
    model.load_state_dict(best)
    torch.save({'state_dict': best, 'groups': data['groups'].tolist(),
                'history_steps': data['windows'].shape[1], 'rank': rank}, output/'model.pt')
    vp, vr = predict(model, data, validation, protocol, cfg, device)
    hp, hr = predict(model, data, held, protocol, cfg, device)
    np.savez_compressed(output/'predictions.npz', validation_probability=vp, held_probability=hp,
                        validation_resolved=vr, held_resolved=hr)
    info = dict(seed=seed, rank=rank, epochs=len(history), best_epoch=best_epoch,
                best_validation_loss=best_loss, validation=_metrics(data['y'][validation], vp),
                held_person=_metrics(data['y'][held], hp),
                held_resolved_probability=float(hr.mean()), seconds=time.time()-started)
    pd.DataFrame(history).to_csv(output/'learning_curve.csv', index=False)
    (output/'metadata.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps({'run': str(output.relative_to(ROOT)), **info}), flush=True)
    return vp, hp


def run(datasets, folds, device):
    cfg = json.loads((ROOT/'configs/joint_trajectory.json').read_text())
    protocol = json.loads((ROOT/'configs/daily_protocol.json').read_text())
    output = ROOT/'results/joint_trajectory'
    output.mkdir(parents=True, exist_ok=True)
    (output/'configuration.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    torch.set_num_threads(2)
    device = _device(device)
    for dataset in datasets:
        data, meta = load_development(dataset)
        for fold in folds:
            split = data['fold']
            train = np.flatnonzero(~np.isin(split, [fold, (fold+1)%5]))
            validation = np.flatnonzero(split == (fold+1)%5)
            held = np.flatnonzero(split == fold)
            for condition, rank in cfg['conditions'].items():
                folder = output/dataset/condition/f'fold{fold}'
                scores = [fit_one(data, train, validation, held, rank, seed, cfg, protocol,
                                  folder/f'seed{seed}', device) for seed in cfg['seeds']]
                for name, indices, probability in [('validation', validation, np.mean([x[0] for x in scores], 0)),
                                                    ('held_person', held, np.mean([x[1] for x in scores], 0))]:
                    frame = meta.iloc[indices].copy()
                    frame['probability'] = probability
                    frame.to_csv(folder/f'{name}.csv', index=False)
                print(json.dumps({'completed': f'{dataset}/{condition}/fold{fold}',
                                  'held_person': _metrics(data['y'][held], probability)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['RESILIENT', 'BrisT1D-Open'])
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.datasets, args.folds, args.device)
