from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT/'data/legacy'


def fold_auc(frame, column, weights=None):
    total, mass=0.,0.
    weights=np.ones(len(frame)) if weights is None else weights
    for fold in sorted(frame.fold.unique()):
        ix=(frame.fold.to_numpy()==fold) & np.isfinite(frame[column].to_numpy()) & (weights>0)
        y=frame.y.to_numpy()[ix];p=frame[column].to_numpy()[ix];w=weights[ix]
        pairs=w[y==0].sum()*w[y==1].sum()
        if pairs:
            total+=roc_auc_score(y,p,sample_weight=w)*pairs;mass+=pairs
    return total/mass if mass else np.nan


def bootstrap_aucs(frame, column, inverse, bootstrap_weights):
    k=bootstrap_weights.shape[1]
    wins=np.zeros((k,k));pairs=np.zeros((k,k))
    y=frame.y.to_numpy();score=frame[column].to_numpy();folds=frame.fold.to_numpy()
    for fold in np.unique(folds):
        pos=np.flatnonzero((folds==fold)&(y==1)&np.isfinite(score))
        neg=np.flatnonzero((folds==fold)&(y==0)&np.isfinite(score))
        if not len(pos) or not len(neg):continue
        comparison=(score[pos,None]>score[None,neg])+.5*(score[pos,None]==score[None,neg])
        ids=(np.repeat(inverse[pos],len(neg)),np.tile(inverse[neg],len(pos)))
        np.add.at(wins,ids,comparison.ravel());np.add.at(pairs,ids,1.)
    numerator=np.einsum('bi,ij,bj->b',bootstrap_weights,wins,bootstrap_weights,optimize=True)
    denominator=np.einsum('bi,ij,bj->b',bootstrap_weights,pairs,bootstrap_weights,optimize=True)
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)


def main():
    out=ROOT/'results/development/summary';out.mkdir(parents=True,exist_ok=True)
    legacy=pd.read_parquet(LEGACY/'analysis/origin_predictions.parquet')
    summary=[];comparisons=[]
    for dataset in ['RESILIENT','BrisT1D-Open']:
        meta=pd.read_csv(ROOT/f'data/features/{dataset}.csv',dtype={'person_id':str})
        key=['person_id','origin_day']
        frame=meta.copy()
        previous=legacy[legacy.dataset.eq(dataset)].copy()
        previous.person_id=previous.person_id.astype(str)
        columns={'HNE-full__score':'previous-HNE','DNB__score':'DNB','FCM__score':'FCM','CATCH__score':'CATCH',
                 'History-LR__score':'previous-History-LR','Missingness-LR__score':'Missingness-LR'}
        frame=frame.merge(previous[key+list(columns)+['primary_common_subset']].rename(columns=columns),on=key,validate='one_to_one')
        names=list(columns.values())
        for model in ['HNE-additive','HNE-set','History-LR','History-HGB','HNE-HGB','HNE-no-network','HNE-no-environment','HNE-history']:
            folder=ROOT/f'results/development/{dataset}/{model}'
            paths=list(folder.glob('fold*/held_person.csv'))
            if len(paths)!=5:
                continue
            result=pd.concat([pd.read_csv(p,dtype={'person_id':str}) for p in paths])
            frame=frame.merge(result[key+['probability']].rename(columns={'probability':model}),on=key,validate='one_to_one')
            names.append(model)
            if model=='HNE-set':
                paths=list(folder.glob('fold*/ensemble.csv'))
                if len(paths)==5:
                    result=pd.concat([pd.read_csv(p,dtype={'person_id':str}) for p in paths])
                    frame=frame.merge(result[key+['probability']].rename(columns={'probability':'HNE-set-ensemble'}),on=key,validate='one_to_one')
                    names.append('HNE-set-ensemble')
        frame.to_csv(out/f'{dataset}_predictions.csv',index=False)
        for sample in ['all','common']:
            sub=frame.copy() if sample=='all' else frame[frame.primary_common_subset].copy()
            for name in names:
                valid=sub[np.isfinite(sub[name])]
                summary.append(dict(dataset=dataset,sample=sample,model=name,n=len(valid),positives=int(valid.y.sum()),
                                    auroc=fold_auc(valid,name)))
            targets=[name for name in ['HNE-additive','HNE-set-ensemble'] if name in names]
            person,inv=np.unique(sub.person_id,return_inverse=True)
            rng=np.random.default_rng(20261110)
            boot_weights=rng.multinomial(len(person),np.full(len(person),1/len(person)),size=1000)
            boot_scores={name:bootstrap_aucs(sub,name,inv,boot_weights) for name in names}
            for target in targets:
                for baseline in ['DNB','History-LR','previous-History-LR','History-HGB','HNE-HGB','previous-HNE']:
                    if baseline not in names or (sample=='all' and baseline=='DNB'):
                        continue
                    delta=fold_auc(sub,target)-fold_auc(sub,baseline)
                    boot=boot_scores[target]-boot_scores[baseline]
                    boot=boot[np.isfinite(boot)]
                    low,high=np.quantile(boot,[.025,.975])
                    comparisons.append(dict(dataset=dataset,sample=sample,model=target,baseline=baseline,
                                            delta=delta,ci_low=low,ci_high=high,bootstrap_replicates=len(boot)))
        print(dataset,[(r['model'],round(r['auroc'],4)) for r in summary if r['dataset']==dataset and r['sample']=='common'],flush=True)
    pd.DataFrame(summary).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'paired_differences.csv',index=False)


if __name__=='__main__':
    main()
