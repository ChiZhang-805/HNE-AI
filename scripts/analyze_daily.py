from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_development import fold_auc,bootstrap_aucs

METHODS=['DNB','DNB-tuned','FCM','CATCH','FCM-score-LR','CATCH-score-LR',
         'History-LR','History-HGB','HNE-HGB','HNE-additive','HNE-set','HNE-temporal',
         'HNE-no-network','HNE-no-environment','HNE-history','Missingness-LR',
         'HNE-no-amplitude','HNE-no-location','HNE-no-memory','HNE-no-tail',
         'HNE-scale8','HNE-scale16','HNE-scale32','HNE-context-append',
         'MM-DEV-daily','MM-DEV-daily-score-LR','HNE-pairwise','FCM-ensemble','CATCH-ensemble','JAPE-event','JAPE-event-AUC']
PRIMARY=['DNB','FCM','CATCH','History-LR','History-HGB','HNE-HGB','HNE-additive']


def main():
    out=ROOT/'results/unified_daily/summary';out.mkdir(parents=True,exist_ok=True)
    rows=[];differences=[];coverage=[]
    for dataset in ['RESILIENT','BrisT1D-Open']:
        meta=pd.read_csv(ROOT/f'data/features/unified_daily/{dataset}.csv',dtype={'person_id':str})
        frame=meta.copy();key=['person_id','origin_day'];complete=[]
        for method in METHODS:
            base=ROOT/f'results/unified_daily/{dataset}/{method}'
            if method.endswith('-ensemble'):
                base=ROOT/f'results/public_daily_repeats/ensemble/{dataset}/{method.removesuffix("-ensemble")}'
            if method == 'JAPE-event':
                base=ROOT/f'results/jape_development/{dataset}'
            if method == 'JAPE-event-AUC':
                base=ROOT/f'results/jape_auc_development/{dataset}'
            paths=list(base.glob('fold*/held_person.csv'))
            if len(paths)!=5:continue
            parts=[]
            for path in sorted(paths):
                part=pd.read_csv(path,dtype={'person_id':str});validation=pd.read_csv(path.parent/'validation.csv')
                col='probability' if 'probability' in part else 'score'
                val_score=validation[col].to_numpy();val_score=val_score[np.isfinite(val_score)]
                negative=validation.loc[validation.y.eq(0),col].dropna().to_numpy()
                cutoff=np.quantile(negative,.9) if len(negative) else np.nan
                part[method]=part[col]
                part[method+'__rank']=np.searchsorted(np.sort(val_score),part[col],side='right')/len(val_score)
                part[method+'__alert']=np.where(np.isfinite(part[col]),part[col]>cutoff,np.nan)
                parts.append(part[key+[method,method+'__rank',method+'__alert']])
            predictions=pd.concat(parts)
            frame=frame.merge(predictions,on=key,validate='one_to_one');complete.append(method)
        common=np.ones(len(frame),bool)
        for method in PRIMARY:
            if method in complete:common &= np.isfinite(frame[method])
        frame['primary_common']=common;frame.to_csv(out/f'{dataset}_predictions.csv',index=False)
        for population in ['all','common']:
            sub=frame if population=='all' else frame[common]
            people,inverse=np.unique(sub.person_id,return_inverse=True)
            weights=np.random.default_rng(20261110).multinomial(len(people),np.full(len(people),1/len(people)),size=2000)
            boot={m:bootstrap_aucs(sub,m,inverse,weights) for m in complete}
            for method in complete:
                valid=sub[np.isfinite(sub[method])].copy();interval=np.nanquantile(boot[method],[.025,.975])
                within=[roc_auc_score(g.y,g[method]) for _,g in valid.groupby('person_id') if g.y.nunique()==2]
                row=dict(dataset=dataset,population=population,method=method,origins=len(valid),people=valid.person_id.nunique(),
                    positives=int(valid.y.sum()),auroc=fold_auc(valid,method),ci_low=interval[0],ci_high=interval[1],
                    within_person_auroc=np.mean(within) if within else np.nan,people_with_both_classes=len(within),
                    ap_on_validation_rank=average_precision_score(valid.y,valid[method+'__rank']),
                    sensitivity=valid.loc[valid.y.eq(1),method+'__alert'].mean(),
                    false_positive_rate=valid.loc[valid.y.eq(0),method+'__alert'].mean())
                rows.append(row)
            if 'HNE-additive' in complete:
                controls=['DNB','FCM-ensemble','CATCH-ensemble','History-LR','History-HGB','HNE-HGB','JAPE-event','JAPE-event-AUC']
                if population=='common' and all(m in complete for m in controls):
                    best=max(controls,key=lambda m:fold_auc(sub,m))
                    delta=boot['HNE-additive']-np.max(np.stack([boot[m] for m in controls]),axis=0)
                    low,high=np.nanquantile(delta,[.025,.975])
                    differences.append(dict(dataset=dataset,population=population,model='HNE-additive',
                        baseline='strongest-extended-control',strongest=best,origins=len(sub),
                        delta=fold_auc(sub,'HNE-additive')-fold_auc(sub,best),ci_low=low,ci_high=high))
                for baseline in complete:
                    if baseline=='HNE-additive':continue
                    paired=sub[np.isfinite(sub[baseline]) & np.isfinite(sub['HNE-additive'])]
                    ids,inv=np.unique(paired.person_id,return_inverse=True)
                    bw=np.random.default_rng(20261110).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
                    delta=bootstrap_aucs(paired,'HNE-additive',inv,bw)-bootstrap_aucs(paired,baseline,inv,bw)
                    low,high=np.nanquantile(delta,[.025,.975])
                    differences.append(dict(dataset=dataset,population=population,model='HNE-additive',baseline=baseline,
                        origins=len(paired),delta=fold_auc(paired,'HNE-additive')-fold_auc(paired,baseline),ci_low=low,ci_high=high))
        coverage.append(dict(dataset=dataset,complete_methods=complete,main_methods_pending=[m for m in PRIMARY if m not in complete],
                             total_origins=len(frame),common_origins=int(common.sum())))
        print(dataset,[(r['method'],round(r['auroc'],4)) for r in rows if r['dataset']==dataset and r['population']=='common'],flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
    (out/'completion.json').write_text(json.dumps(coverage,indent=2))


if __name__=='__main__':main()
