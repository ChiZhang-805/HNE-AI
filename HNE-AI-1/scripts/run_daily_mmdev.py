"""Past-only MM-DEV adaptation to the common 32-day histories."""
from pathlib import Path
from dataclasses import asdict
import json
import sys
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.mmdev import MMDEVConfig,score_window
from src.hne_additive import AdditiveWarning,validation_score


def main():
    config=MMDEVConfig(history_hours=32,smap_window=24,embedding_max=6)
    for dataset in ['RESILIENT','BrisT1D-Open']:
        data=np.load(ROOT/f'data/features/unified_daily/{dataset}.npz')
        meta=pd.read_csv(ROOT/f'data/features/unified_daily/{dataset}.csv',dtype={'person_id':str})
        name='MM-DEV-daily'
        out=ROOT/f'results/unified_daily/{dataset}/{name}';out.mkdir(parents=True,exist_ok=True)
        if (out/'origin_scores.csv').exists():
            frame=pd.read_csv(out/'origin_scores.csv',dtype={'person_id':str})
        else:
            detail=[]
            for index,window in enumerate(data['windows']):
                detail.append(score_window(window,config))
                if index%500==0:print(dataset,index,flush=True)
            frame=pd.concat([meta,pd.DataFrame(detail)],axis=1)
            frame.to_csv(out/'origin_scores.csv',index=False)
        (out/'configuration.json').write_text(json.dumps(dict(parameters=asdict(config),
            adaptation='Past-only ISOMAP and S-map; 32 days; 24-day local Jacobian window; E=1..6 chosen by past one-step LOO prediction.',
            input='All 11 observed state and context coordinates, identical input to public frameworks.',
            missing='Within-history median; no time compression; undefined disconnected k=5 graph scores retained as missing.'),indent=2))
        for fold in range(5):
            val=frame.fold.eq((fold+1)%5).to_numpy();held=frame.fold.eq(fold).to_numpy();train=~(val|held)
            native=out/f'fold{fold}';native.mkdir(exist_ok=True)
            finite=np.isfinite(frame.score)
            fitted=[]
            for C in (.01,.1,1.):
                ix=train&finite
                model=AdditiveWarning(C=C,spline=False).fit(frame.loc[ix,['score']],frame.y.to_numpy()[ix],frame.person_id.to_numpy()[ix])
                ix=val&finite
                value=validation_score(frame.y.to_numpy()[ix],model.predict_proba(frame.loc[ix,['score']]))
                fitted.append((value,C,model))
            best=max(fitted,key=lambda row:row[0]);model=best[2]
            lr=ROOT/f'results/unified_daily/{dataset}/{name}-score-LR/fold{fold}';lr.mkdir(parents=True,exist_ok=True)
            with (lr/'model.pkl').open('wb') as file:pickle.dump(model,file)
            (lr/'selection.json').write_text(json.dumps(dict(C=best[1],trials=[dict(C=c,validation=v) for v,c,_ in fitted]),indent=2))
            for split,ix in [('train',train),('validation',val),('held_person',held)]:
                part=frame[ix].copy();part.to_csv(native/f'{split}.csv',index=False)
                good=np.isfinite(part.score);part['probability']=np.nan
                part.loc[good,'probability']=model.predict_proba(part.loc[good,['score']])
                part.to_csv(lr/f'{split}.csv',index=False)
        print(dataset,'complete',int(finite.sum()),len(frame),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
