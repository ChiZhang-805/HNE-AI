"""Fixed-weight order interventions; oracle losses are post-prediction diagnostics."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import HNEContextPaths
from train_hne_context_paths import predict,PATHS,path_distribution


def run():
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text());rows=[]
    out=ROOT/'results/hne_routed/context_path_interventions';out.mkdir(parents=True,exist_ok=True)
    for ds in config['datasets']:
        meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
        for k in config['folds']:
            selection=json.loads((ROOT/f'results/hne_routed/{ds}/context_path/fold{k}/selection.json').read_text())
            ids=np.flatnonzero(meta.fold.eq(k));y=meta.y.to_numpy()[ids]
            for checkpoint in selection['runs']:
                seed=Path(checkpoint).name;cache=np.load(ROOT/f'results/hne_routed/{ds}/context_cache/fold{k}/{seed}/states.npz')
                model=HNEContextPaths(config);model.load_state_dict(torch.load(ROOT/checkpoint/'model.pt',weights_only=True,map_location='cpu'))
                nodes=torch.tensor(cache['nodes'][ids]);readout=torch.tensor(cache['readout'][ids])
                p,r,all_p=predict(model,nodes,readout)
                with torch.no_grad():policy=path_distribution(model,nodes).numpy()
                prob=np.clip(all_p,1e-7,1-1e-7);loss=-(y[:,None]*np.log(prob)+(1-y[:,None])*np.log1p(-prob))
                probability=np.clip(p,1e-7,1-1e-7);selected_loss=-(y*np.log(probability)+(1-y)*np.log1p(-probability))
                rank=(loss<selected_loss[:,None]-1e-6).sum(1)+1
                frame=meta.iloc[ids].copy();frame['probability']=p;frame['path']=[''.join('HNE'[j] for j in path) for path in r]
                frame['path_loss_rank']=rank;frame['learned_bce']=selected_loss;frame['random_path_expected_bce']=loss.mean(1);frame['oracle_bce']=loss.min(1)
                folder=out/ds/f'fold{k}'/seed;folder.mkdir(parents=True,exist_ok=True);frame.to_csv(folder/'traces.csv',index=False)
                np.savez_compressed(folder/'path_predictions.npz',paths=PATHS,probabilities=all_p,policy=policy,y=y)
                rows.append(dict(dataset=ds,fold=k,seed=int(seed.removeprefix('seed')),n=len(ids),
                    auroc=roc_auc_score(y,p),fixed_HNE_auroc=roc_auc_score(y,all_p[:,0]),
                    learned_bce=float(selected_loss.mean()),random_path_expected_bce=float(loss.mean()),oracle_bce=float(loss.min(1).mean()),
                    mean_path_rank=float(rank.mean()),best_path_fraction=float((rank==1).mean()),
                    unique_paths=frame.path.nunique(),path_effect_mean=float(np.ptp(all_p,axis=1).mean())))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);print(pd.DataFrame(rows).groupby('dataset')[['auroc','mean_path_rank','unique_paths','path_effect_mean']].mean().to_string())


if __name__=='__main__':run()
