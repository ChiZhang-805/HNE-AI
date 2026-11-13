"""Finite timing-oriented head experiment; identical HNE coordinates and labels."""
from pathlib import Path
import json
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.hne_pairwise import PairwiseWarning


def main():
    definition=dict(C=[.01,.1,1.],alpha=[.25,1.],
        reason='Separate within-person advance warning from between-person susceptibility.',
        pair_sampling='Every positive-negative pair within each training person; equal mass per person with both classes.',
        selection='Mean of validation aggregate AUROC and mean within-person AUROC; aggregate only when no validation person has both classes.',
        shared='Same296 HNE coordinates, reference, core modules, spline bases, prior correction, origin labels and person folds.')
    (ROOT/'configs/pairwise_head.json').write_text(json.dumps(definition,indent=2))
    for dataset in ['RESILIENT','BrisT1D-Open']:
        z=np.load(ROOT/f'data/features/unified_daily/{dataset}.npz')
        meta=pd.read_csv(ROOT/f'data/features/unified_daily/{dataset}.csv',dtype={'person_id':str})
        for fold in range(5):
            out=ROOT/f'results/unified_daily/{dataset}/HNE-pairwise/fold{fold}';out.mkdir(parents=True,exist_ok=True)
            if (out/'metrics.json').exists():continue
            held=z['fold']==fold;val=z['fold']==(fold+1)%5;train=~(held|val)
            trials=[];best=None;best_value=-np.inf
            for C in definition['C']:
                for alpha in definition['alpha']:
                    model=PairwiseWarning(C=C,alpha=alpha,groups=z['groups']).fit(z['pooled'][train],z['y'][train],meta.person_id[train])
                    vp=model.predict_proba(z['pooled'][val]);vm=meta[val].copy();vm['p']=vp
                    aggregate=roc_auc_score(vm.y,vm.p)
                    within=[roc_auc_score(g.y,g.p) for _,g in vm.groupby('person_id') if g.y.nunique()==2]
                    value=(aggregate+np.mean(within))/2 if within else aggregate
                    trial=dict(C=C,alpha=alpha,validation=float(value),aggregate=float(aggregate),
                        within=float(np.mean(within)),pairs=model.pair_count,**model.optimization)
                    trials.append(trial)
                    if value>best_value:best_value=value;best=model;chosen=trial
            with (out/'model.pkl').open('wb') as file:pickle.dump(best,file)
            for split,mask in [('validation',val),('held_person',held)]:
                frame=meta[mask].copy();frame['probability']=best.predict_proba(z['pooled'][mask]);frame.to_csv(out/f'{split}.csv',index=False)
            (out/'metrics.json').write_text(json.dumps(dict(dataset=dataset,fold=fold,chosen=chosen,trials=trials),indent=2))
            print(dataset,fold,chosen,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
