"""Development comparison of delayed personal feedback against calibration controls."""
from pathlib import Path
import json
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.hne_online import prequential
from scripts.summarize_development import fold_auc


def run_split(model,arrays,frame,condition,c,cfg):
    p,count,latest=prequential(model,arrays['pooled'],arrays['y'],frame.person_id,
        frame.origin_day,condition,c,cfg)
    result=frame.copy();result['score']=p;result['mature_origins_used']=count;result['latest_feedback_origin']=latest
    return result


def describe(frame):
    y=frame.y.to_numpy();p=frame.score.to_numpy()
    within=[roc_auc_score(g.y,g.score) for _,g in frame.groupby('person_id') if g.y.nunique()==2]
    return dict(auroc=float(roc_auc_score(y,p)),within_person_auroc=float(np.mean(within)),
        person_nll=float(np.mean([log_loss(g.y,g.score,labels=[0,1]) for _,g in frame.groupby('person_id')])),
        person_brier=float(np.mean([np.mean((g.y-g.score)**2) for _,g in frame.groupby('person_id')])),
        adapted_fraction=float((frame.mature_origins_used>0).mean()))


def main():
    cfg=json.loads((ROOT/'configs/delayed_adaptation.json').read_text())
    output=ROOT/'results/delayed_adaptation';summary=[]
    for dataset in ['RESILIENT','BrisT1D-Open','Human-Rhythms']:
        if dataset=='Human-Rhythms':
            z=np.load(ROOT/'data/features/human_rhythms/validation.npz')
            frame=pd.read_csv(ROOT/'data/features/human_rhythms/validation.csv',dtype={'person_id':str})
            model=joblib.load(ROOT/'results/human_rhythms/HNE-additive/model.joblib')['model']
            units=[(None,model,{'validation':({'pooled':z['pooled'],'y':z['y']},frame)})]
        else:
            z=np.load(ROOT/f'data/features/unified_daily/{dataset}.npz')
            frame=pd.read_csv(ROOT/f'data/features/unified_daily/{dataset}.csv',dtype={'person_id':str})
            units=[]
            for fold in range(5):
                with open(ROOT/f'results/unified_daily/{dataset}/HNE-additive/fold{fold}/model.pkl','rb') as f:model=pickle.load(f)
                parts={}
                for split,k in [('validation',(fold+1)%5),('development',fold)]:
                    m=z['fold']==k
                    parts[split]=({'pooled':z['pooled'][m],'y':z['y'][m]},frame[m].copy())
                units.append((fold,model,parts))
        for fold,model,parts in units:
            for condition in cfg['conditions']:
                folder=output/dataset/('train-validation' if fold is None else f'fold{fold}')/condition
                folder.mkdir(parents=True,exist_ok=True)
                if (folder/'selection.json').exists():
                    summary.append(json.loads((folder/'selection.json').read_text()));continue
                trials=[];best=-np.inf
                for c in ([0.] if condition=='frozen' else cfg['C_candidates']):
                    val=run_split(model,*parts['validation'],condition,c,cfg)
                    metrics=describe(val);trials.append(dict(C=c,**metrics))
                    if metrics['auroc']>best:chosen,cached,best=c,val,metrics['auroc']
                cached.to_csv(folder/'validation.csv',index=False)
                row=dict(dataset=dataset,fold=fold,condition=condition,C=chosen,
                    candidates=trials,validation_auroc=best)
                if 'development' in parts:
                    held=run_split(model,*parts['development'],condition,chosen,cfg)
                    held.to_csv(folder/'development.csv',index=False)
                    row.update({'development_'+k:v for k,v in describe(held).items()})
                (folder/'selection.json').write_text(json.dumps(row,indent=2))
                summary.append(row);print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!='candidates'} for r in summary]).to_csv(output/'summary.csv',index=False)
    rows=[]
    for dataset in ['RESILIENT','BrisT1D-Open']:
        for condition in cfg['conditions']:
            frame=pd.concat([pd.read_csv(p,dtype={'person_id':str}) for p in
                sorted((output/dataset).glob(f'fold*/{condition}/development.csv'))])
            rows.append(dict(dataset=dataset,condition=condition,fold_pair_auroc=fold_auc(frame,'score'),**describe(frame)))
    pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
