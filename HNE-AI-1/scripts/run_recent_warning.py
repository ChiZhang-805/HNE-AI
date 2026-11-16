"""Evaluate frozen time-series models with a participant-held event head."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache/mantis'
MODEL_REVISION = 'bc7d5ab40c02133386a28e2c127f35c17c86901d'
TIMESFM_REVISION = '1d952420fba87f3c6dee4f240de0f1a0fbc790e3'
REGULARIZATION = (0.001, 0.01, 0.1, 1.0)
KEY = ['person_id', 'origin_day', 'fold', 'y']


@torch.inference_mode()
def mantis_features(network, history, device):
    n, _, length = history.shape
    grid = np.arange(length)
    embeddings = []
    # Each channel is encoded independently, as in MantisTrainer.transform.
    for batch in np.array_split(history, (n + 15) // 16):
        x = batch.reshape(-1, length).copy()
        for row in x:
            observed = np.isfinite(row)
            if not observed.all():
                row[:] = np.interp(grid, grid[observed], row[observed]) if observed.any() else 0.
        x = torch.from_numpy(x).to(device).unsqueeze(1)
        x = F.interpolate(x, size=512, mode='linear', align_corners=False)
        embeddings.append(network(x).reshape(len(batch), -1).cpu().numpy())
    return np.concatenate(embeddings)


@torch.inference_mode()
def moirai_features(model, history, device):
    n, channels, length = history.shape
    patch = model.patch_size
    tokens = length // patch
    channel_ids = torch.arange(channels, device=device).repeat_interleave(tokens)
    features = []
    for batch in np.array_split(history, (n + 31) // 32):
        b = len(batch)
        x = torch.from_numpy(batch).to(device).reshape(b, channels * tokens, patch)
        observed = x.isfinite()
        active = observed.reshape(b, channels, tokens, patch).any(dim=(1, 3))
        times = (active.cummax(1).values.long().cumsum(1) - 1).clamp_min(0).repeat(1, channels)
        preds = model(x.nan_to_num(), observed, torch.ones_like(times), times,
                      channel_ids.expand(b, -1), torch.zeros_like(times, dtype=torch.bool),
                      training_mode=False)
        # Official structure_multi_predict takes all four forecast patches from
        # each channel's last context token. No target observation is passed in.
        q = preds[:, tokens-1::tokens].reshape(b, channels, 4, 9, patch)
        q = q.transpose(3, 4).reshape(b, channels, 8, 8, 9)[:, :, 1:]
        means = q[..., [0, 4, 8]].mean(dim=3)
        spread = q[..., 4].std(dim=3, unbiased=False)
        features.append(torch.cat((means, spread.unsqueeze(-1)), -1).reshape(b, -1).cpu().numpy())
    return np.concatenate(features)


@torch.inference_mode()
def timesfm_features(model, history, device):
    n, channels, length = history.shape
    inputs = [row if np.isfinite(row).any() else np.zeros(length, dtype=np.float32)
              for row in history.reshape(-1, length)]
    median, quantiles = model.forecast(horizon=64, inputs=inputs)
    days = quantiles.reshape(n, channels, 8, 8, 3)[:, :, 1:]
    spread = median.reshape(n, channels, 8, 8)[:, :, 1:].std(axis=3)
    return np.concatenate([days.mean(axis=3), spread[..., None]], axis=-1).reshape(n, -1)


def fit_cohort(name, method, network, device):
    folder = ROOT / 'data/hne_routed' / name
    out = ROOT / 'results' / f'{method}_warning' / name
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(folder / 'samples.csv', dtype={'person_id': str})[KEY]
    with np.load(folder / 'arrays.npz') as arrays:
        encode = {'mantis': mantis_features, 'moirai2': moirai_features, 'timesfm': timesfm_features}[method]
        z = encode(network, arrays['history'], device)
    folds = meta.fold.to_numpy()
    labels = meta.y.to_numpy()
    scores, selections = np.empty(len(meta)), []
    for fold in range(5):
        valid = folds == (fold + 1) % 5
        held = folds == fold
        train = ~(valid | held)
        candidates = []
        for c in REGULARIZATION:
            head = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            head.fit(z[train], labels[train])
            p = head.predict_proba(z[valid])[:, 1]
            candidates.append((roc_auc_score(labels[valid], p), -log_loss(labels[valid], p), c, head))
        auc, neg_loss, c, head = max(candidates, key=lambda item: item[:2])
        scores[held] = head.predict_proba(z[held])[:, 1]
        joblib.dump(head, out / f'head_fold{fold}.joblib')
        selections.append(dict(fold=fold, C=c, val_auroc=auc, val_log_loss=-neg_loss))
        print(name, fold, 'held origins', held.sum(), 'selected C', c, flush=True)
    meta.assign(score=scores).to_csv(out / 'held_person.csv', index=False)
    (out / 'selection.json').write_text(json.dumps(selections, indent=2))


def load_mantis():
    sys.path.insert(0, str(CACHE / 'author/src'))
    from mantis.architecture import MantisV1
    checkpoint = CACHE / 'Mantis-8M'
    if not checkpoint.is_dir():
        checkpoint = Path(snapshot_download('paris-noah/Mantis-8M', revision=MODEL_REVISION, local_dir=checkpoint,
                          allow_patterns=['model.safetensors', 'config.json']))
    config = json.loads((checkpoint / 'config.json').read_text())
    config.pop('debugging', None)
    network = MantisV1(**config)
    state = load_file(str(checkpoint / 'model.safetensors'))
    # V1 stores the original scalar grid as a buffer; the released weights omit these constants.
    for i, encoder in enumerate(network.tokgen_unit.scalar_encoders):
        state.setdefault(f'tokgen_unit.scalar_encoders.{i}.scales', encoder.scales)
    network.load_state_dict(state, strict=True)
    return network


def load_moirai():
    cache = ROOT / 'cache/moirai2'
    source = cache / 'author/src'
    sys.path[:0] = [str(source), str(cache / 'deps')]
    spec = importlib.util.spec_from_file_location('moirai2_core', source / 'uni2ts/model/moirai2/module.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checkpoint = cache / 'small'
    if not checkpoint.is_dir():
        checkpoint = Path(snapshot_download('Salesforce/moirai-2.0-R-small', local_dir=checkpoint,
                          revision='30f43ff08c8494f4943ae1521e9d4e94a0fbb389',
                          allow_patterns=['model.safetensors', 'config.json']))
    network = module.Moirai2Module(**json.loads((checkpoint / 'config.json').read_text()))
    network.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=True)
    return network


def load_timesfm(device):
    cache = ROOT / 'cache/timesfm'
    sys.path.insert(0, str(cache / 'author/src'))
    import timesfm
    checkpoint = cache / '2p5'
    if not (checkpoint / 'model.safetensors').is_file():
        snapshot_download('google/timesfm-2.5-200m-pytorch', revision=TIMESFM_REVISION,
                          local_dir=checkpoint, allow_patterns=['model.safetensors', 'config.json'])
    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
    model.model.device = torch.device(device)
    model.model.device_count = 1
    model.load_checkpoint(str(checkpoint))
    model.compile(timesfm.ForecastConfig(max_context=256, max_horizon=64, per_core_batch_size=32,
                  normalize_inputs=True, use_continuous_quantile_head=True, fix_quantile_crossing=True,
                  infer_is_positive=False, output_quantiles=[0.1, 0.5, 0.9]))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['mantis', 'moirai2', 'timesfm'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.method == 'timesfm':
        network = load_timesfm(args.device)
    else:
        network = load_mantis() if args.method == 'mantis' else load_moirai()
        network.to(args.device).eval()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for cohort in ('RESILIENT', 'BrisT1D-Open'):
            fit_cohort(cohort, args.method, network, args.device)
