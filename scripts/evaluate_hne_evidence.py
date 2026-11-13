"""Frozen-weight interventions on personal memory, channel exchange and context."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import make_hne,RoutedData
from run_hne_routed import predict,metrics
from analyze_hne_routed import fold_auc


def run():
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    variant=json.loads((ROOT/'results/hne_routed/architecture_selection.json').read_text())['primary_variant']
    root=ROOT/'results/hne_routed/evidence';root.mkdir(parents=True,exist_ok=True);totals=[]
    for ds in config['datasets']:
        data=RoutedData(ds);meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
        for condition in ['original','reference_zero','network_cut','context_zero']:
            frames=[]
            for k in config['folds']:
                folder=root/ds/condition/f'fold{k}';folder.mkdir(parents=True,exist_ok=True)
                path=folder/'predictions.csv'
                if path.exists():frames.append(pd.read_csv(path,dtype={'person_id':str}));continue
                selected=json.loads((ROOT/f'results/hne_routed/{ds}/{variant}/fold{k}/selection.json').read_text())
                ids=np.flatnonzero(meta.fold.eq(k));probabilities=[]
                for checkpoint in selected['runs']:
                    model=make_hne(config,variant).cuda();model.load_state_dict(torch.load(ROOT/checkpoint/'model.pt',map_location='cuda',weights_only=True))
                    model.evidence_intervention=None if condition=='original' else condition
                    p,_,_=predict(model,data,ids,config['batch_size']);probabilities.append(p)
                    del model;torch.cuda.empty_cache()
                frame=meta.iloc[ids].copy();frame['probability']=np.mean(probabilities,axis=0);frame.to_csv(path,index=False);frames.append(frame)
            full=pd.concat(frames,ignore_index=True);full.to_csv(root/ds/f'{condition}.csv',index=False)
            totals.append(dict(dataset=ds,method=variant,condition=condition,n=len(full),auroc=fold_auc(full,'probability')))
            print(json.dumps(totals[-1]),flush=True)
    pd.DataFrame(totals).to_csv(root/'metrics.csv',index=False)


if __name__=='__main__':run()
