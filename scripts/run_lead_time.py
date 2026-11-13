"""Evaluate frozen warning scores at three future gaps on matched origins."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_development import fold_auc,bootstrap_aucs


def main():
    out=ROOT/'results/lead_time';out.mkdir(parents=True,exist_ok=True)
    gaps=[1,3,7];methods=['DNB','FCM-ensemble','CATCH-ensemble','HNE-additive']
    (out/'configuration.json').write_text(json.dumps(dict(gap_days=gaps,future_block_days=7,
        target_index='calendar days [d+gap+1,d+gap+7], origin d observed through next00:00',
        models='All trained for original gap1 target; frozen weights and source scores; no gap-specific fitting',
        population='Original common evaluation origins with known future seven-day blocks at all three gaps',
        uncertainty='2000 participant cluster bootstrap; identical resamples across methods and gaps'),indent=2))
    rows=[]
    for dataset in ['RESILIENT','BrisT1D-Open']:
        columns=['person_id','origin_day','fold','y','primary_common']+methods
        frame=pd.read_csv(ROOT/f'results/unified_daily/summary/{dataset}_predictions.csv',dtype={'person_id':str},usecols=columns)
        for gap in gaps:frame[f'y_gap{gap}']=np.nan
        for person,group in frame.groupby('person_id'):
            labels=np.load(ROOT/f'data/daily_study/{dataset}/{person}_daily.npz')['block_label']
            for gap in gaps:
                target=group.origin_day.to_numpy()+gap+7
                known=target<len(labels)
                frame.loc[group.index[known],f'y_gap{gap}']=labels[target[known]]
        use=frame.primary_common.to_numpy()&frame[[f'y_gap{g}' for g in gaps]].notna().all(axis=1).to_numpy()
        frame=frame[use].copy();frame.to_csv(out/f'{dataset}_origins.csv',index=False)
        people,inverse=np.unique(frame.person_id,return_inverse=True)
        weights=np.random.default_rng(20261110).multinomial(len(people),np.full(len(people),1/len(people)),size=2000)
        for gap in gaps:
            scored=frame.copy();scored['y']=scored[f'y_gap{gap}']
            for method in methods:
                boot=bootstrap_aucs(scored,method,inverse,weights);low,high=np.nanquantile(boot,[.025,.975])
                within=[roc_auc_score(part.y,part[method]) for _,part in scored.groupby('person_id') if part.y.nunique()==2]
                rows.append(dict(dataset=dataset,gap=gap,model=method,n=len(scored),positives=int(scored.y.sum()),
                    people=len(people),auroc=fold_auc(scored,method),ci_low=low,ci_high=high,
                    within_person_auroc=float(np.mean(within)),within_person_count=len(within)))
        print(dataset,[(r['gap'],r['model'],round(r['auroc'],3)) for r in rows if r['dataset']==dataset],flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)


if __name__=='__main__':main()
