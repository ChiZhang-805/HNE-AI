"""Intervene on the learned path while keeping each model's weights fixed."""
from pathlib import Path
import argparse,itertools,json,sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,log_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import HNERouted,RoutedData
from run_hne_routed import predict


def run(args):
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    rows=[]
    for ds in args.datasets:
        data=RoutedData(ds);meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
        for k in args.folds:
            folder=ROOT/f'results/hne_routed/{ds}/{args.variant}/fold{k}'
            if not (folder/'selection.json').exists():continue
            selection=json.loads((folder/'selection.json').read_text())
            ids=np.flatnonzero(data.arrays['fold']==k);y=data.arrays['y'][ids]
            paths=list(itertools.product(range(3),repeat=3));values=[]
            source=ROOT/selection['runs'][0]
            if args.variant=='path_policy':
                cache=np.load(source/'rollouts.npz')
                index={int(v):i for i,v in enumerate(cache['ids'])}
                candidates=cache['probabilities'][[index[int(v)] for v in ids]]
                frame=pd.read_csv(source/'held_person.csv',dtype={'person_id':str})
                prob=frame.probability.to_numpy();routes=frame[['route_1','route_2','route_3']].to_numpy()
            else:
                model=HNERouted(config).cuda();model.load_state_dict(torch.load(source/'model.pt',weights_only=True,map_location='cuda'))
                prob,routes,_=predict(model,data,ids,config['batch_size'])
                for path in paths:
                    p,_,_=predict(model,data,ids,config['batch_size'],force_path=path);values.append(p)
                candidates=np.stack(values,1)
                del model;torch.cuda.empty_cache()
            losses=-(y[:,None]*np.log(np.clip(candidates,1e-7,1-1e-7))+(1-y[:,None])*np.log(np.clip(1-candidates,1e-7,1-1e-7)))
            learned_loss=-(y*np.log(np.clip(prob,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-prob,1e-7,1-1e-7)))
            indices=routes[:,0]*9+routes[:,1]*3+routes[:,2]
            ranks=(losses<learned_loss[:,None]-1e-6).sum(1)+1
            trace=meta.iloc[ids].copy();trace['probability']=prob
            trace['path']=[''.join('HNE'[j] for j in path) for path in routes]
            trace['path_loss_rank']=ranks;trace['oracle_loss']=losses.min(1)
            trace['learned_path_loss']=learned_loss;trace['mean_random_path_loss']=losses.mean(1)
            dest=ROOT/'results/hne_routed/path_interventions'/args.variant/ds/f'fold{k}';dest.mkdir(parents=True,exist_ok=True)
            trace.to_csv(dest/'traces.csv',index=False)
            np.savez_compressed(dest/'all_paths.npz',probability=candidates,paths=np.asarray(paths),y=y)
            row=dict(dataset=ds,method=args.variant,fold=k,seed=int(source.name.removeprefix('seed')),n=len(ids),unique_paths=len(np.unique(indices)),
                     learned_auroc=float(roc_auc_score(y,prob)),fixed_HNE_auroc=float(roc_auc_score(y,candidates[:,paths.index((0,1,2))])),
                     learned_bce=float(learned_loss.mean()),mean_random_path_bce=float(losses.mean()),
                     oracle_bce=float(losses.min(1).mean()),mean_path_rank=float(ranks.mean()),
                     best_path_fraction=float((ranks==1).mean()),path_effect_mean=float(np.ptp(candidates,axis=1).mean()))
            rows.append(row);print(json.dumps(row),flush=True)
    out=ROOT/'results/hne_routed/path_interventions'/args.variant;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['RESILIENT','BrisT1D-Open']);p.add_argument('--folds',type=int,nargs='+',default=list(range(5)));p.add_argument('--variant',choices=['learned','path_policy'],default='path_policy')
    run(p.parse_args())
