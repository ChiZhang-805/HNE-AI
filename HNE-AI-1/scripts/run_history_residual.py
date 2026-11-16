"""History-anchored HNE development; Human Rhythms test is never opened."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_additive import AdditiveWarning, validation_score
from src.hne_history_residual import HistoryResidualWarning
from scripts.summarize_development import fold_auc


def summarize(output):
    rows=[]
    for dataset in ['RESILIENT','BrisT1D-Open']:
        common=pd.read_csv(ROOT/f'results/unified_daily/summary/{dataset}_predictions.csv',
            dtype={'person_id':str},usecols=['person_id','origin_day','primary_common'])
        for condition in ['crossfit','in-sample']:
            frame=pd.concat([pd.read_csv(p,dtype={'person_id':str})
                for p in sorted((output/dataset).glob(f'fold*/{condition}/development.csv'))])
            frame=frame.merge(common,on=['person_id','origin_day'],validate='one_to_one')
            for population in ['all','common']:
                sub=frame if population=='all' else frame[frame.primary_common]
                rows.append(dict(dataset=dataset,condition=condition,population=population,
                    origins=len(sub),auroc=fold_auc(sub,'score')))
    pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def main():
    config = json.loads((ROOT/'configs/history_residual.json').read_text())
    output = ROOT/'results/history_residual'
    summary = []
    for dataset in ['RESILIENT', 'BrisT1D-Open', 'Human-Rhythms']:
        if dataset == 'Human-Rhythms':
            root = ROOT/'data/features/human_rhythms'
            arrays = {s:np.load(root/f'{s}.npz') for s in ['train','validation']}
            meta = {s:pd.read_csv(root/f'{s}.csv', dtype={'person_id':str}) for s in arrays}
            units = [(None, arrays, meta)]
        else:
            root = ROOT/'data/features/unified_daily'
            z = np.load(root/f'{dataset}.npz')
            all_meta = pd.read_csv(root/f'{dataset}.csv', dtype={'person_id':str})
            units = []
            for fold in range(5):
                masks = {'train':~np.isin(z['fold'],[fold,(fold+1)%5]),
                         'validation':z['fold']==(fold+1)%5, 'development':z['fold']==fold}
                arrays = {s:{'pooled':z['pooled'][m], 'y':z['y'][m], 'groups':z['groups']} for s,m in masks.items()}
                meta = {s:all_meta.loc[m].copy() for s,m in masks.items()}
                units.append((fold, arrays, meta))
        for fold, arrays, meta in units:
            train, val = arrays['train'], arrays['validation']
            groups = train['groups']; h = groups=='H'
            destination = output/dataset/('train-validation' if fold is None else f'fold{fold}')
            destination.mkdir(parents=True, exist_ok=True)
            complete=[destination/c/'selection.json' for c in ['crossfit','in-sample']]
            if all(p.exists() for p in complete):
                summary.extend(json.loads(p.read_text()) for p in complete)
                continue
            history_trials=[]
            for c in config['history_C']:
                history=AdditiveWarning(C=c,groups=groups[h]).fit(train['pooled'][:,h],train['y'],meta['train'].person_id)
                score=validation_score(val['y'],history.predict_proba(val['pooled'][:,h]))
                history_trials.append(dict(C=c,validation_auroc=score))
            hc=max(history_trials,key=lambda r:r['validation_auroc'])['C']
            for crossfit in [True,False]:
                folder=destination/('crossfit' if crossfit else 'in-sample')
                folder.mkdir(exist_ok=True)
                if (folder/'selection.json').exists():
                    summary.append(json.loads((folder/'selection.json').read_text())); continue
                candidates=[]; best=None; best_value=-np.inf
                for c in config['correction_C']:
                    model=HistoryResidualWarning(hc,c,crossfit,config['history_crossfit_seed']).fit(
                        train['pooled'],train['y'],meta['train'].person_id,groups)
                    value=validation_score(val['y'],model.predict_proba(val['pooled']))
                    candidates.append(dict(correction_C=c,validation_auroc=value,optimization=model.optimization))
                    if value>best_value:best,best_value,chosen=model,value,c
                joblib.dump(best,folder/'model.joblib')
                record=dict(dataset=dataset,fold=fold,crossfit=crossfit,history_C=hc,correction_C=chosen,
                    history_candidates=history_trials,candidates=candidates,validation_auroc=best_value)
                for split in ['validation','development']:
                    if split not in arrays:continue
                    data=arrays[split]; frame=meta[split].copy()
                    frame['score']=best.predict_proba(data['pooled'])
                    frame.to_csv(folder/f'{split}.csv',index=False)
                    record[split+'_auroc']=float(roc_auc_score(data['y'],frame.score))
                (folder/'selection.json').write_text(json.dumps(record,indent=2))
                summary.append(record)
                print(json.dumps({k:v for k,v in record.items() if k not in ['history_candidates','candidates']}),flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['history_candidates','candidates']} for r in summary]).to_csv(output/'summary.csv',index=False)
    summarize(output)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
