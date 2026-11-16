"""Intervene on continuous states and reuse source weights across the two cohorts."""
from pathlib import Path
import argparse, gc, json, sys
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from src.hne_latent import HNELatent, LatentData
from run_hne_latent import predict, forward, metrics


def load_state(checkpoint, numerical, config, variant):
    processor = joblib.load(checkpoint/'basis.joblib')
    model = HNELatent(config, processor.widths, variant).cuda().eval()
    model.load_state_dict(torch.load(checkpoint/'model.pt', map_location='cuda', weights_only=True), strict=False)
    return dict(model=model, basis=[torch.as_tensor(x, device='cuda') for x in processor.transform(numerical)])


def trajectories(state, data, meta, ids, config):
    rows = []; gradients = []
    for start in range(0, len(ids), config['micro_batch_size']):
        batch = ids[start:start+config['micro_batch_size']]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = forward(state, data, batch, return_states=True)
            loss = F.binary_cross_entropy_with_logits(output['logits'].float(),
                         torch.as_tensor(data.arrays['y'][batch], device='cuda'))
        gs = torch.autograd.grad(loss, output['state_list'], allow_unused=True)
        states = output['states'].float().detach()
        for j in range(states.shape[1]):
            change = (states[:, j]-states[:, j-1]).square().mean((1, 2)).sqrt() if j else torch.zeros(len(batch), device='cuda')
            gradient = gs[j].float().flatten(1).norm(dim=1) if gs[j] is not None else torch.zeros(len(batch), device='cuda')
            f = meta.iloc[batch][['person_id', 'origin_day', 'fold', 'y']].copy()
            f['step'] = j+1; f['state_rms_change'] = change.cpu().numpy()
            f['final_event_gradient_norm'] = gradient.detach().cpu().numpy()
            f['probability'] = output['step_logits'][:, j].float().detach().sigmoid().cpu().numpy()
            rows.append(f)
        del output, loss, gs
    return pd.concat(rows, ignore_index=True)


def interventions(args, config):
    for dataset in args.datasets or config['datasets']:
        data = LatentData(dataset)
        meta = pd.read_csv(ROOT/f'data/hne_routed/{dataset}/samples.csv', dtype={'person_id': str})
        numerical = np.load(ROOT/f'data/features/unified_daily/{dataset}.npz')['pooled']
        for k in args.folds or config['folds']:
            ids = np.flatnonzero(data.arrays['fold'] == k)
            root = ROOT/'results/hne_latent'/dataset/'recurrent'/f'fold{k}'
            available = sorted(root.glob('seed*/model.pt'))
            if not available: continue
            summaries = []
            for weights in available:
                checkpoint = weights.parent; output = checkpoint/'state_interventions'; output.mkdir(exist_ok=True)
                if (output/'metrics.json').exists(): continue
                state = load_state(checkpoint, numerical, config, 'recurrent')
                # One fixed permutation makes state mixing independent of participant/file ordering.
                shuffled_ids = np.random.default_rng(20261114).permutation(ids)
                frame = meta.iloc[ids].copy()
                for label, options in [('depth1', dict(steps=1)), ('depth2', dict(steps=2)),
                    ('depth4', dict(steps=4)), ('depth8', dict(steps=8)),
                    ('reset', dict(intervention='reset')), ('shuffle', dict(intervention='shuffle'))]:
                    order = shuffled_ids if label == 'shuffle' else ids
                    f = predict(state, data, meta, order, config, **options)
                    f = f.set_index(['person_id', 'origin_day']).reindex(pd.MultiIndex.from_frame(frame[['person_id', 'origin_day']]))
                    frame[label] = f.probability.to_numpy()
                frame.to_csv(output/'predictions.csv', index=False)
                trajectories(state, data, meta, ids, config).to_csv(output/'trajectories.csv', index=False)
                summary = {label: metrics(frame.y, frame[label], frame.person_id) for label in
                           ['depth1', 'depth2', 'depth4', 'depth8', 'reset', 'shuffle']}
                (output/'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
                print(json.dumps(dict(interventions=str(checkpoint.relative_to(ROOT)), metrics=summary)), flush=True)
                del state; gc.collect(); torch.cuda.empty_cache()
        del data; gc.collect(); torch.cuda.empty_cache()


def transfer(args, config):
    for source in args.datasets or config['datasets']:
        target = next(ds for ds in config['datasets'] if ds != source)
        data = LatentData(target)
        meta = pd.read_csv(ROOT/f'data/hne_routed/{target}/samples.csv', dtype={'person_id': str})
        numerical = np.load(ROOT/f'data/features/unified_daily/{target}.npz')['pooled']; ids = np.arange(len(meta))
        for variant in config['variants']:
            weights = sorted((ROOT/'results/hne_latent'/source/variant).glob('fold*/seed*/model.pt'))
            if not weights: continue
            predictions = []
            folder = ROOT/f'results/hne_latent/transfer/{source}_to_{target}/{variant}'
            folder.mkdir(parents=True, exist_ok=True)
            for path in weights:
                name = path.parent.parent.name+'_'+path.parent.name
                output = folder/f'{name}.csv'
                if output.exists(): f = pd.read_csv(output, dtype={'person_id': str})
                else:
                    state = load_state(path.parent, numerical, config, variant)
                    f = predict(state, data, meta, ids, config); f.to_csv(output, index=False)
                    del state; gc.collect(); torch.cuda.empty_cache()
                predictions.append(f.probability.to_numpy())
            frame = meta.copy(); frame['probability'] = np.mean(predictions, axis=0)
            frame.to_csv(folder/'predictions.csv', index=False)
            result = dict(source=source, target=target, method=variant, source_models=len(weights),
                          complete_source_ensemble=len(weights)==len(config['folds'])*len(config['seeds']),
                          target_training_labels=0, **metrics(frame.y, frame.probability, frame.person_id))
            (folder/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf8')
            print(json.dumps(result), flush=True)
        del data; gc.collect(); torch.cuda.empty_cache()


def input_interventions(args, config):
    from chronos import Chronos2Pipeline
    for dataset in args.datasets or config['datasets']:
        data = LatentData(dataset)
        meta = pd.read_csv(ROOT/f'data/hne_routed/{dataset}/samples.csv', dtype={'person_id': str})
        for condition in ['within_day_shuffle', 'day_order_shuffle', 'channel_day_shift']:
            input_folder = ROOT/'data/hne_interventions'/dataset/condition
            raw = np.load(input_folder/'history.npy', mmap_mode='r')
            numerical = np.load(input_folder/'pooled.npy')
            backbone = Chronos2Pipeline.from_pretrained(str(ROOT/'tools/models/chronos-2'), device_map='cuda').model.eval()
            captured = {}
            def capture(module, values, kwargs):
                captured['x'] = values[0]
                captured['mask'] = kwargs['attention_mask'][:, 0, 0] == 0
            handle = backbone.encoder.block[-2].register_forward_pre_hook(capture, with_kwargs=True)
            encoded = []; masks = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for start in range(0, len(raw), 16):
                    values = torch.tensor(np.asarray(raw[start:start+16]).copy().reshape(-1, 256), device='cuda')
                    n = len(values)//6
                    backbone.encode(values, group_ids=torch.arange(n, device='cuda').repeat_interleave(6))
                    encoded.append(captured['x'].half().reshape(n, 6, -1, 768))
                    masks.append(captured['mask'].reshape(n, 6, -1))
            data.history = torch.cat(encoded); data.mask = torch.cat(masks)
            handle.remove(); del backbone, encoded, masks; gc.collect(); torch.cuda.empty_cache()
            for variant in config['variants']:
                for k in args.folds or config['folds']:
                    ids = np.flatnonzero(data.arrays['fold']==k)
                    root = ROOT/'results/hne_latent'/dataset/variant/f'fold{k}'
                    output = ROOT/'results/hne_latent/input_interventions'/dataset/condition/variant/f'fold{k}'
                    weights = sorted(root.glob('seed*/model.pt'))
                    if not weights: continue
                    output.mkdir(parents=True, exist_ok=True); all_predictions = []
                    for path in weights:
                        state = load_state(path.parent, numerical, config, variant)
                        frame = predict(state, data, meta, ids, config)
                        frame.to_csv(output/(path.parent.name+'.csv'), index=False)
                        all_predictions.append(frame.probability.to_numpy())
                        del state; gc.collect(); torch.cuda.empty_cache()
                    frame['probability'] = np.mean(all_predictions, axis=0)
                    frame.to_csv(output/'held_person.csv', index=False)
            print(json.dumps(dict(dataset=dataset, input_intervention=condition)), flush=True)
        del data; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['interventions', 'transfer', 'input_interventions'], default='interventions')
    parser.add_argument('--datasets', nargs='+'); parser.add_argument('--folds', type=int, nargs='+')
    args = parser.parse_args(); torch.set_num_threads(2)
    config = json.loads((ROOT/'configs/hne_latent.json').read_text())
    dict(interventions=interventions, transfer=transfer, input_interventions=input_interventions)[args.mode](args, config)
