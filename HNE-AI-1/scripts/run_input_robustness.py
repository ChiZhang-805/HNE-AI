"""Frozen-model robustness to lost observations and measurement noise."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.hne_representation import Sampling,encode_person
from src.hne_additive import history_features
from src.dnb import score_window
from scripts.summarize_development import fold_auc

CONDITIONS={'missing10':('missing',.10),'missing25':('missing',.25),'noise025':('noise',.25)}
SEEDS=[20261110,20261111,20261112]


def main():
    out=ROOT/'results/robustness';out.mkdir(parents=True,exist_ok=True)
    (out/'configuration.json').write_text(json.dumps(dict(conditions=CONDITIONS,seeds=SEEDS,
        reference='Initial56 days unchanged; perturb only later observed history',
        frozen='Daily HNE-additive and History-HGB selected fivefold models; no retraining or threshold choice',
        labels='Original future labels and eligibility unchanged; noise never changes outcomes',
        noise_unit='Gaussian standard deviation in each personal-reference normalized coordinate',
        missing='Additional independent removal of observed state and context entries'),indent=2))
    rows=[]
    spec=Sampling.daily()
    for dataset in ['RESILIENT','BrisT1D-Open']:
        data=np.load(ROOT/f'data/features/unified_daily/{dataset}.npz')
        meta=pd.read_csv(ROOT/f'data/features/unified_daily/{dataset}.csv',dtype={'person_id':str})
        common=pd.read_csv(ROOT/f'results/unified_daily/summary/{dataset}_predictions.csv',dtype={'person_id':str})
        common=meta[['person_id','origin_day']].merge(common[['person_id','origin_day','primary_common']],on=['person_id','origin_day'],validate='one_to_one').primary_common.to_numpy()
        fitted={}
        for fold in range(5):
            for name in ['HNE-additive','History-HGB']:
                with (ROOT/f'results/unified_daily/{dataset}/{name}/fold{fold}/model.pkl').open('rb') as file:
                    fitted[fold,name]=pickle.load(file)
        for condition,(mode,amount) in CONDITIONS.items():
            for seed in SEEDS:
                folder=out/dataset/condition;folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'seed{seed}.csv'
                if path.exists():
                    result=pd.read_csv(path,dtype={'person_id':str})
                else:
                    features=np.empty_like(data['pooled']);windows=np.empty_like(data['windows'])
                    for person,frame in meta.groupby('person_id',sort=False):
                        source=np.load(ROOT/f'data/daily_study/{dataset}/{person}_daily.npz')
                        p=source['normalized'].shape[1]
                        values=np.column_stack((source['normalized'],source['normalized_contexts'])).copy()
                        identity=int.from_bytes(hashlib.sha256(f'{dataset}|{person}'.encode()).digest()[:4],'little')
                        rng=np.random.default_rng(seed+identity)
                        post=values[spec.reference:]
                        if mode=='missing':post[rng.random(post.shape)<amount]=np.nan
                        else:post+=rng.normal(0,amount,post.shape)
                        cutoffs=frame.origin_day.to_numpy()+1
                        _,features[frame.index],_=encode_person(values[:,:p],values[:,p:],cutoffs,spec)
                        windows[frame.index]=np.stack([values[t-32:t] for t in cutoffs])
                    result=meta.copy()
                    result['DNB']=[score_window(w[:,:data['nodes'].shape[1]]) for w in windows]
                    history=history_features(windows,scales=(8,16,32))
                    for fold in range(5):
                        ix=meta.fold.eq(fold).to_numpy()
                        result.loc[ix,'HNE-additive']=fitted[fold,'HNE-additive'].predict_proba(features[ix])
                        result.loc[ix,'History-HGB']=fitted[fold,'History-HGB'].predict_proba(history[ix])
                    result.to_csv(path,index=False)
                valid=common&np.isfinite(result['DNB'])&np.isfinite(result['HNE-additive'])&np.isfinite(result['History-HGB'])
                for name in ['HNE-additive','History-HGB','DNB']:
                    rows.append(dict(dataset=dataset,condition=condition,seed=seed,model=name,
                        n=int(valid.sum()),positives=int(result.loc[valid,'y'].sum()),
                        auroc=fold_auc(result.loc[valid],name)))
                print(dataset,condition,seed,'complete',flush=True)
                pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False)
    table=pd.DataFrame(rows)
    table.groupby(['dataset','condition','model']).agg(auroc=('auroc','mean'),sd=('auroc','std'),
        minimum=('auroc','min'),maximum=('auroc','max'),n_min=('n','min'),n_max=('n','max')).reset_index().to_csv(out/'summary.csv',index=False)


if __name__=='__main__':
    warnings.filterwarnings('ignore',message='All-NaN slice encountered')
    warnings.filterwarnings('ignore',message='Mean of empty slice')
    warnings.filterwarnings('ignore',message='Degrees of freedom <= 0 for slice')
    with threadpool_limits(limits=1):main()
