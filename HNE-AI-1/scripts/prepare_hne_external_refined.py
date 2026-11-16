"""Preserve existing endpoints while preparing variable-channel source transfer."""
from pathlib import Path
import argparse, gc, json, sys, warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.daily_protocol import robust_reference, normalize


def normalized_grid(values):
    center, scale, _ = robust_reference(values, 448, 32)
    return np.clip(normalize(values, center, scale), -12, 12).astype(np.float32)


def prepare(dataset):
    out = ROOT/'data/hne_external_refined'/dataset
    out.mkdir(parents=True, exist_ok=True)
    if (out/'mapping.json').exists(): return
    grids = {}; references = []
    if dataset == 'College-Behavior':
        source = ROOT/'data/features/college_experience'
        meta = pd.read_csv(source/'test.csv', dtype={'person_id': str})
        numerical = np.load(source/'test.npz')
        people = sorted(meta.person_id.unique())
        for person in people:
            daily = np.load(ROOT/f'data/college_study/test/{person}_daily.npz')
            hourly = np.load(ROOT/f'data/college_hourly/test/{person}.npz')
            values = np.full((len(daily['dates']), 24, 9), np.nan, np.float32)
            index = pd.Index(daily['dates']).get_indexer(hourly['dates'])
            recorded = np.column_stack([hourly['states'], hourly['contexts']]).reshape(-1, 24, 9)
            values[index[index >= 0]] = recorded[index >= 0]
            grid = normalized_grid(np.nanmean(values.reshape(-1, 3, 9), axis=1))
            grids[person] = grid; references.append(grid[:448].T)
        state_count, context_count = 8, 1
        resolution = 'Actual hourly behavior records aggregated to observed three-hour bins; calendar aligned to the original daily endpoint.'
        endpoint = 'Original persistent multivariate fluctuation endpoint; same 41 scorable participants among the fixed 44 test IDs.'
    else:
        mission = int(dataset[-1]); source = ROOT/'data/esa_native'/dataset
        meta = pd.read_csv(source/'test.csv', dtype={'person_id': str})
        numerical = np.load(source/'test.npz')
        raw = np.load(ROOT/f'data/new_cohorts/esa-ad-v2/mission{mission}_daily.npz')
        selected = numerical['selected_states']
        recorded = np.column_stack([raw['states'][:, selected], raw['contexts']])
        values = np.full((len(recorded)*8, recorded.shape[1]), np.nan, np.float32)
        values[7::8] = recorded
        grid = normalized_grid(values)
        people = [dataset]; grids[dataset] = grid; references.append(grid[:448].T)
        state_count, context_count = len(selected), raw['contexts'].shape[1]
        resolution = 'Observed daily telemetry/commands placed at their end-of-day availability on a three-hour published-clock grid; other bins remain missing, with no interpolation.'
        endpoint = 'Existing expert multivariate subsequence onset endpoint; original fixed future-fluctuation endpoint has no eligible origins here and remains separate.'
    assert np.array_equal(numerical['y'], meta.y.to_numpy())
    history = np.stack([grids[r.person_id][(int(r.origin_day)+1)*8-256:(int(r.origin_day)+1)*8].T for r in meta.itertuples()])
    stats = np.nan_to_num(np.stack([np.nanmean(history, axis=2), np.log1p(np.nanstd(history, axis=2)), np.isfinite(history).mean(2)], axis=-1)).astype(np.float32)
    np.savez_compressed(out/'arrays.npz', history=history, reference=np.asarray(references), stats=stats,
        person_index=np.asarray([people.index(p) for p in meta.person_id]), pooled=numerical['pooled'])
    meta.to_csv(out/'samples.csv', index=False)
    mapping = dict(dataset=dataset, n=len(meta), people=len(people), state_channels=state_count,
        context_channels=context_count, temporal_resolution=resolution, endpoint=endpoint,
        reference='Same first56-day personal prefix and 32-day observed history; source model weights/transforms fixed.',
        target_training_labels=0, prior_use='These cohorts were evaluated with the earlier additive HNE; this is a new frozen-model evaluation, not a never-inspected cohort.',
        neural_input_shape=list(history.shape), numerical_input_shape=list(numerical['pooled'].shape),
        observed_history_fraction=float(np.isfinite(history).mean()))
    (out/'mapping.json').write_text(json.dumps(mapping, indent=2), encoding='utf8')
    print(json.dumps(mapping), flush=True)


def encode(datasets):
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(2)
    model = Chronos2Pipeline.from_pretrained(str(ROOT/'tools/models/chronos-2'), device_map='cuda').model.eval()
    captured = {}
    def hook(module, args, kwargs):
        captured['x'] = args[0].detach()
        captured['mask'] = kwargs['attention_mask'][:, 0, 0, :] == 0
    handle = model.encoder.block[-1].register_forward_pre_hook(hook, with_kwargs=True)
    for dataset in datasets:
        folder = ROOT/'data/hne_external_refined'/dataset; raw = np.load(folder/'arrays.npz')
        # References are small and reused across origins. Histories are encoded
        # once per batch while all source models are resident during evaluation.
        for name in ['reference']:
            if (folder/f'{name}_mask.npy').exists(): continue
            x = raw[name]; channels, length = x.shape[1:]; tokens = length//16+2
            cache = np.lib.format.open_memmap(folder/f'{name}_prefix.npy', mode='w+', dtype=np.float16, shape=(len(x), channels, tokens, 768))
            masks = []; batch = max(1, 96//channels)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for start in range(0, len(x), batch):
                    values = torch.tensor(x[start:start+batch].reshape(-1, length), device='cuda')
                    groups = torch.arange(len(values)//channels, device='cuda').repeat_interleave(channels)
                    model.encode(values, group_ids=groups)
                    cache[start:start+batch] = captured['x'].float().cpu().numpy().reshape(-1, channels, tokens, 768)
                    masks.append(captured['mask'].cpu().numpy().reshape(-1, channels, tokens))
                    if start % (batch*100) == 0: print(dataset, name, start, '/', len(x), flush=True)
            cache.flush(); np.save(folder/f'{name}_mask.npy', np.concatenate(masks))
        print(dataset, 'frozen multivariate encoding complete', flush=True)
    handle.remove(); del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['College-Behavior', 'ESA-Mission1', 'ESA-Mission2'])
    parser.add_argument('--encode', action='store_true')
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for dataset in args.datasets: prepare(dataset)
    if args.encode: encode(args.datasets)
