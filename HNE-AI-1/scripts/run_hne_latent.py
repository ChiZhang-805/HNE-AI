"""Train continuous latent feedback and matched controls on identical person splits."""
from pathlib import Path
import argparse, gc, json, random, sys, time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from src.hne_latent import HNELatent, LatentData, latent_objective, compact_state
from src.hne_refined import StructuredEvidenceBasis
from run_hne_routed import person_weights


def metrics(y, p, people):
    return dict(auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                bce=float(log_loss(y, p, labels=[0, 1])),
                person_bce=float(log_loss(y, p, sample_weight=person_weights(people), labels=[0, 1])))


def make_fit(data, meta, numerical, train, config, variant, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    processor = StructuredEvidenceBasis().fit(numerical['pooled'][train], numerical['groups'])
    basis = [torch.as_tensor(x, device='cuda') for x in processor.transform(numerical['pooled'])]
    model = HNELatent(config, processor.widths, variant).cuda()
    prevalence = np.average(data.arrays['y'][train], weights=person_weights(meta.iloc[train].person_id.to_numpy()))
    class_weights = np.array([.5/(1-prevalence), .5/prevalence], dtype=np.float32)
    model.class_log_odds.fill_(float(np.log(class_weights[1]/class_weights[0])))
    target = data.arrays['future_targets'][train]
    model.future_mean.copy_(torch.as_tensor(np.nanmean(target, axis=(0, 1)), device='cuda'))
    model.future_scale.copy_(torch.as_tensor(np.nanstd(target, axis=(0, 1)).clip(.01), device='cuda'))
    with torch.no_grad(): model.risk.bias.zero_()
    groups = {'main': [], 'core': []}
    for name, p in model.named_parameters():
        if p.requires_grad: groups['core' if name.startswith('core.') else 'main'].append(p)
    optimizer = torch.optim.AdamW([dict(params=groups['main'], lr=config['learning_rate']),
                                  dict(params=groups['core'], lr=config['lora_learning_rate'])],
                                 weight_decay=config['weight_decay'])
    people = meta.person_id.to_numpy()
    return dict(model=model, optimizer=optimizer, basis=basis, processor=processor, train=train,
                by_person={p: train[people[train] == p] for p in np.unique(people[train])},
                rng=np.random.default_rng(seed), class_weights=class_weights,
                base_lrs=[g['lr'] for g in optimizer.param_groups])


def forward(state, data, ids, **kwargs):
    ix = torch.as_tensor(ids, device='cuda')
    return state['model'](*data.batch(ids), [x[ix] for x in state['basis']], **kwargs)


def train_epoch(state, data, config, lr_fraction):
    model = state['model']; model.train(); optimizer = state['optimizer']
    for group, base in zip(optimizer.param_groups, state['base_lrs']): group['lr'] = base*lr_fraction
    people = list(state['by_person']); rng = state['rng']; rows = []
    count = min(config['people_per_batch'], len(people)); per_person = config['batch_size']//count
    for _ in range(int(np.ceil(len(state['train'])/config['batch_size']))):
        chosen = rng.choice(people, count, replace=False)
        ids = np.concatenate([rng.choice(state['by_person'][p], per_person, replace=True) for p in chosen])
        optimizer.zero_grad(set_to_none=True)
        accumulated = {}
        for start in range(0, len(ids), config['micro_batch_size']):
            batch = ids[start:start+config['micro_batch_size']]; fraction = len(batch)/len(ids)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = forward(state, data, batch)
                loss, parts = latent_objective(model, out, torch.as_tensor(data.arrays['y'][batch], device='cuda'),
                    torch.as_tensor(state['class_weights'][data.arrays['y'][batch].astype(int)], device='cuda'),
                    torch.as_tensor(data.arrays['future_targets'][batch], device='cuda'), config)
            (loss*fraction).backward()
            for key, value in parts.items(): accumulated[key] = accumulated.get(key, 0.)+float(value.detach())*fraction
        gradient_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        rows.append({**accumulated, 'gradient_norm': float(gradient_norm)})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def predict(state, data, meta, ids, config, **kwargs):
    state['model'].eval(); probabilities = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(ids), config['evaluation_batch_size']):
            output = forward(state, data, ids[start:start+config['evaluation_batch_size']], **kwargs)
            probabilities.append(output['step_logits'].float().sigmoid().cpu().numpy())
    values = np.concatenate(probabilities); frame = meta.iloc[ids].copy()
    frame['probability'] = values[:, -1]
    for j in range(values.shape[1]): frame[f'step_{j+1}'] = values[:, j]
    return frame


def select_length(data, meta, numerical, train, config, variant, folder):
    path = folder/'selection.json'
    if path.exists(): return json.loads(path.read_text())
    states = []; fold = data.arrays['fold']
    for inner in np.unique(fold[train]):
        fit = train[fold[train] != inner]; validation = train[fold[train] == inner]
        state = make_fit(data, meta, numerical, fit, config, variant, config['seeds'][0]+int(inner))
        state.update(validation=validation, inner_fold=int(inner)); states.append(state)
    best = (-np.inf, -np.inf); since_best = plateau = 0; lr_fraction = 1.
    history = []; detailed = []; started = time.perf_counter()
    for epoch in range(1, config['maximum_epochs']+1):
        frames = []; objectives = []
        for state in states:
            parts = train_epoch(state, data, config, lr_fraction); objectives.append(parts['objective'])
            frame = predict(state, data, meta, state['validation'], config); frames.append(frame)
            tp = predict(state, data, meta, state['train'], config)
            detailed.append(dict(epoch=epoch, inner_fold=state['inner_fold'], lr_fraction=lr_fraction, **parts,
                **{f'train_{k}': v for k, v in metrics(tp.y, tp.probability, tp.person_id).items()},
                **{f'validation_{k}': v for k, v in metrics(frame.y, frame.probability, frame.person_id).items()}))
        weights = np.array([g.y.sum()*(len(g)-g.y.sum()) for g in frames])
        auc = float(np.average([roc_auc_score(g.y, g.probability) for g in frames], weights=weights))
        combined = pd.concat(frames, ignore_index=True)
        bce = float(log_loss(combined.y, combined.probability, labels=[0, 1]))
        record = dict(epoch=epoch, inner_auroc=auc, inner_bce=bce, lr_fraction=lr_fraction,
                      train_objective=float(np.mean(objectives)), seconds=time.perf_counter()-started)
        history.append(record)
        if (auc, -bce) > best:
            best = (auc, -bce); best_epoch = epoch; since_best = plateau = 0
            combined.to_csv(folder/'inner_validation.csv', index=False)
        else:
            since_best += 1; plateau += 1
        pd.DataFrame(detailed).to_csv(folder/'inner_curves.csv', index=False)
        print(json.dumps(dict(run=str(folder.relative_to(ROOT)), **record)), flush=True)
        if plateau >= config['plateau_patience']:
            lr_fraction = max(lr_fraction*.5, .125); plateau = 0
        if epoch >= config['minimum_epochs'] and since_best >= config['patience']: break
    selection = dict(config=config, selected_epoch=best_epoch, inner_auroc=best[0], inner_bce=-best[1],
                     lr_fractions=[row['lr_fraction'] for row in history[:best_epoch]], history=history,
                     training_people=sorted(meta.iloc[train].person_id.unique().tolist()),
                     inner_training_fits=len(states), seconds=time.perf_counter()-started)
    path.write_text(json.dumps(selection, indent=2), encoding='utf8')
    del state, states; gc.collect(); torch.cuda.empty_cache()
    return selection


def refit(data, meta, numerical, train, val, test, config, variant, seed, selection, folder):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'result.json').exists(): return
    state = make_fit(data, meta, numerical, train, config, variant, seed)
    history = []; started = time.perf_counter()
    for epoch, fraction in enumerate(selection['lr_fractions'], 1):
        parts = train_epoch(state, data, config, fraction)
        tp = predict(state, data, meta, train, config)
        history.append(dict(epoch=epoch, lr_fraction=fraction, **parts,
                            **{f'train_{k}': v for k, v in metrics(tp.y, tp.probability, tp.person_id).items()}))
    torch.save(compact_state(state['model']), folder/'model.pt')
    joblib.dump(state['processor'], folder/'basis.joblib')
    result = dict(version=config['version'], variant=variant, seed=seed, history=history,
                  execution_micro_batch=config['micro_batch_size'], effective_batch=config['batch_size'],
                  selected_epoch=selection['selected_epoch'], fit_seconds=time.perf_counter()-started,
                  trainable_parameters=sum(p.numel() for p in state['model'].parameters() if p.requires_grad),
                  feedback_gain=float(state['model'].feedback_gain.detach()))
    for part, ids in [('validation', val), ('held_person', test)]:
        frame = predict(state, data, meta, ids, config)
        frame.to_csv(folder/f'{part}.csv', index=False)
        result[part] = metrics(frame.y, frame.probability, frame.person_id)
    result['core_lora_changes'] = {n: float(p.detach().abs().max()) for n, p in state['model'].named_parameters()
                                  if n.startswith('core.') and n.endswith('.b')}
    (folder/'result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(dict(complete=str(folder.relative_to(ROOT)), **{k: result[k] for k in
                          ['selected_epoch', 'held_person', 'fit_seconds', 'feedback_gain']})), flush=True)
    del state; gc.collect(); torch.cuda.empty_cache()


def main(args):
    config = json.loads((ROOT/args.config).read_text()); torch.set_num_threads(2)
    for dataset in args.datasets or config['datasets']:
        data = LatentData(dataset)
        meta = pd.read_csv(ROOT/f'data/hne_routed/{dataset}/samples.csv', dtype={'person_id': str})
        numerical = np.load(ROOT/f'data/features/unified_daily/{dataset}.npz'); fold = data.arrays['fold']
        for k in args.folds or config['folds']:
            train = np.flatnonzero(~np.isin(fold, [k, (k+1)%5]))
            val = np.flatnonzero(fold == (k+1)%5); test = np.flatnonzero(fold == k)
            for variant in args.variants or config['variants']:
                folder = ROOT/args.output/dataset/variant/f'fold{k}'; folder.mkdir(parents=True, exist_ok=True)
                selection = select_length(data, meta, numerical, train, config, variant, folder)
                if args.selection_only:
                    continue
                seeds = config['seeds'][:args.seeds]
                for seed in seeds:
                    refit(data, meta, numerical, train, val, test, config, variant, seed, selection, folder/f'seed{seed}')
                for part in ['validation', 'held_person']:
                    frames = [pd.read_csv(folder/f'seed{s}/{part}.csv', dtype={'person_id': str}) for s in seeds]
                    frame = frames[0].copy()
                    for col in ['probability']+[c for c in frame if c.startswith('step_')]:
                        frame[col] = np.mean([f[col].to_numpy() for f in frames], axis=0)
                    frame.to_csv(folder/f'{part}.csv', index=False)
        del data; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/hne_latent.json')
    parser.add_argument('--output', default='results/hne_latent')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--folds', nargs='+', type=int)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--selection-only', action='store_true', help='Use training-person inner folds only; do not score outer folds.')
    main(parser.parse_args())
