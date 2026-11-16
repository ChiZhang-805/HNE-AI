from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.hne_set import fit_predict
from scripts.run_development import metrics

config=json.loads((ROOT/'configs/development.json').read_text())
for dataset in config['datasets']:
    z=np.load(ROOT/f'data/features/{dataset}.npz')
    meta=pd.read_csv(ROOT/f'data/features/{dataset}.csv',dtype={'person_id':str})
    for fold in config['development_folds']:
        folder=ROOT/f'results/development/{dataset}/HNE-set/fold{fold}'
        chosen=json.loads((folder/'metrics.json').read_text())['chosen']
        split=z['fold']; train=~np.isin(split,[fold,(fold+1)%5]);val=split==(fold+1)%5;test=split==fold
        ensemble=[pd.read_csv(folder/'held_person.csv').probability.to_numpy()]
        for seed in config['seeds'][1:]:
            out=folder/f'seed{seed}'
            if (out/'held_person.csv').exists():
                ensemble.append(pd.read_csv(out/'held_person.csv').probability.to_numpy())
                continue
            result=fit_predict(z['nodes'][train],z['pooled'][train],z['y'][train],meta.person_id[train].to_numpy(),
                z['nodes'][val],z['pooled'][val],z['y'][val],meta.person_id[val].to_numpy(),
                z['nodes'][test],z['pooled'][test],z['groups'],seed=seed,weight_decay=chosen['weight_decay'],output=out)
            frame=meta[test].copy();frame['probability']=result['test_probability'];frame.to_csv(out/'held_person.csv',index=False)
            vframe=meta[val].copy();vframe['probability']=result['val_probability'];vframe.to_csv(out/'validation.csv',index=False)
            info=dict(dataset=dataset,fold=fold,seed=seed,chosen=chosen,validation=metrics(z['y'][val],result['val_probability']),
                      held_person=metrics(z['y'][test],result['test_probability']))
            (out/'metrics.json').write_text(json.dumps(info,indent=2))
            ensemble.append(result['test_probability'])
            print(json.dumps(info),flush=True)
        frame=meta[test].copy();frame['probability']=np.mean(ensemble,axis=0);frame.to_csv(folder/'ensemble.csv',index=False)
