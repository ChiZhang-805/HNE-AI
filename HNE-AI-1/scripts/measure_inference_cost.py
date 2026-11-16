"""Measure CPU warning latency from normalized histories with loaded models."""
import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_public_external import restore, fill, SEEDS
from scripts.run_hne_external import make_features, native_one
from src.hne_representation import encode_person, Sampling


def main():
    cfg = json.loads((ROOT / 'configs/inference_cost.json').read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    output = ROOT / 'results/inference_cost'
    output.mkdir(exist_ok=True)
    folder = ROOT / 'results/human_rhythms'
    features = ROOT / 'data/features/human_rhythms'
    meta = pd.read_csv(features / 'test.csv', dtype={'person_id': str})
    indices = np.random.default_rng(20261110).choice(len(meta), cfg['origins'], replace=False)
    meta = meta.iloc[indices].reset_index(drop=True)
    meta[['person_id', 'origin_day']].to_csv(output / 'origins.csv', index=False)
    with np.load(features / 'test.npz') as z:
        windows, pooled = z['windows'][indices], z['pooled'][indices]
    prefixes = []
    for r in meta.itertuples():
        with np.load(ROOT / f'data/human_rhythms_study/test/{r.person_id}_daily.npz') as z:
            prefixes.append((z['normalized'][:r.origin_day+1], z['normalized_contexts'][:r.origin_day+1]))
    hne = joblib.load(folder / 'HNE-additive/model.joblib')
    hgb = joblib.load(folder / 'History-HGB/model.joblib')
    functions = {}
    parameters = {}
    checkpoint_bytes = {}
    def full_hne():
        result=[]
        for start in range(0,len(prefixes),cfg['batch_size']):
            batch=[]
            for states, contexts in prefixes[start:start+cfg['batch_size']]:
                _, x, _ = encode_person(states, contexts, [len(states)], Sampling.daily())
                batch.append(x)
            result.extend(hne['model'].predict_proba(np.concatenate(batch)))
        return np.asarray(result)
    functions['HNE-from-history'] = full_hne
    functions['HNE-cached-head'] = lambda: np.concatenate([
        hne['model'].predict_proba(pooled[i:i+cfg['batch_size']]) for i in range(0,len(pooled),cfg['batch_size'])])
    parameters['HNE-from-history'] = parameters['HNE-cached-head'] = int(hne['model'].model.coef_.size+1)
    checkpoint_bytes['HNE-from-history'] = checkpoint_bytes['HNE-cached-head'] = (folder / 'HNE-additive/model.joblib').stat().st_size
    def history_hgb():
        return np.concatenate([hgb['model'].predict_proba(make_features({'windows': windows[i:i+cfg['batch_size']]}, hgb['feature_spec']))
                               for i in range(0,len(windows),cfg['batch_size'])])
    functions['History-HGB'] = history_hgb
    checkpoint_bytes['History-HGB'] = (folder / 'History-HGB/model.joblib').stat().st_size
    for method in ['DNB','MM-DEV-daily']:
        functions[method] = lambda m=method: np.asarray([native_one((m,w,10))['score'] for w in windows])
        parameters[method] = 0
        checkpoint_bytes[method] = 0
    loaded = []
    for method in ['FCM','CATCH']:
        components=[]
        for seed in SEEDS:
            source = folder / method / f'seed{seed}'
            fitted = restore(source,method,seed)
            fitted.model.to('cpu')
            if method=='FCM':fitted.config['device']='cpu'
            else:fitted.device='cpu'
            selection=json.loads((source/'selection.json').read_text())
            components.append((fitted,np.asarray(selection['imputation_medians'],np.float32),
                               np.asarray(selection['validation_negative_scores'])))
            loaded.append(fitted)
        def ensemble(parts=components):
            quantiles=[]
            for fitted,median,negative in parts:
                score=fitted.score(fill(windows,median))
                quantiles.append(np.searchsorted(negative,score,side='right')/len(negative))
            return np.mean(quantiles,axis=0)
        name=method+'-3seed'
        functions[name]=ensemble
        parameters[name]=sum(sum(p.numel() for p in fitted.model.parameters()) for fitted,_,_ in components)
        checkpoint_bytes[name]=sum((folder/method/f'seed{seed}/checkpoint.pt').stat().st_size for seed in SEEDS)
    rows=[]
    # The first complete call initializes runtime kernels; five measured calls follow.
    for method,function in functions.items():
        function()
        durations=[]
        for repeat in range(cfg['repetitions']):
            begin=time.perf_counter()
            score=function()
            seconds=time.perf_counter()-begin
            durations.append(seconds)
            print(f'{method} repeat{repeat+1}: {1000*seconds/len(meta):.3f}ms/origin',flush=True)
        row={'method':method,'device':'CPU','threads':cfg['cpu_threads'],'origins':len(meta),
             'median_ms_per_origin':1000*np.median(durations)/len(meta),
             'q25_ms_per_origin':1000*np.quantile(durations,.25)/len(meta),
             'q75_ms_per_origin':1000*np.quantile(durations,.75)/len(meta),
             'parameters':parameters.get(method),'checkpoint_bytes':checkpoint_bytes[method],
             'scored_origins':int(np.isfinite(score).sum())}
        rows.append(row)
        pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
        (output/f'{method}_durations.json').write_text(json.dumps(durations))
    cfg['cpu_model']='Intel Core i9-13900HX'
    cfg['python']=sys.version
    cfg['torch']=torch.__version__
    (output/'configuration.json').write_text(json.dumps(cfg,indent=2))


if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    with threadpool_limits(limits=1):
        main()
