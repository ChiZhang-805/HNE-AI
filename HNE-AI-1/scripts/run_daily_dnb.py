from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dnb import score_window

for dataset in ['RESILIENT','BrisT1D-Open']:
    z=np.load(ROOT/f'data/features/unified_daily/{dataset}.npz')
    meta=pd.read_csv(ROOT/f'data/features/unified_daily/{dataset}.csv',dtype={'person_id':str})
    p=z['nodes'].shape[1]
    scores=np.array([[score_window(w[-length:,:p]) for w in z['windows']] for length in (16,24,32)])
    for fold in range(5):
        val=meta.fold.to_numpy()==(fold+1)%5
        test=meta.fold.to_numpy()==fold
        train=~(val|test)
        val_scores=[]
        for s in scores:
            good=val&np.isfinite(s)
            score=roc_auc_score(z['y'][good],s[good]) if len(np.unique(z['y'][good]))==2 else .5
            val_scores.append(score)
        chosen=int(np.argmax(val_scores))
        for name,index in [('DNB',2),('DNB-tuned',chosen)]:
            out=ROOT/f'results/unified_daily/{dataset}/{name}/fold{fold}';out.mkdir(parents=True,exist_ok=True)
            for split,mask in [('train',train),('validation',val),('held_person',test)]:
                frame=meta[mask].copy();frame['score']=scores[index][mask];frame.to_csv(out/f'{split}.csv',index=False)
            (out/'metadata.json').write_text(json.dumps(dict(history_days=(16,24,32)[index],minimum_joint_observations=12,
                candidate_channel_cap=10,selected_set_size='largest feasible 3 to 5',module_sizes=[2,3],
                window_choice='fixed32' if name=='DNB' else 'validation_AUROC',score_is_probability=False),indent=2))
    print(json.dumps(dict(dataset=dataset,origins=len(meta),finite_longest=int(np.isfinite(scores[2]).sum()))),flush=True)
