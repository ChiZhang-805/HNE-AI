"""Fixed-predictor path training with per-example standardized loss targets."""
from pathlib import Path
import copy,json,sys,time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import HNEContextFree
from train_hne_free_paths import cached,scores,path_distribution,predict,PATHS
from run_hne_routed import metrics,person_weights


def run():
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    for ds in config['datasets']:
        meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str});y=torch.tensor(meta.y.to_numpy(),dtype=torch.float32)
        for k in config['folds']:
            train=np.flatnonzero(~meta.fold.isin([k,(k+1)%5]));val=np.flatnonzero(meta.fold.eq((k+1)%5));held=np.flatnonzero(meta.fold.eq(k))
            selected=json.loads((ROOT/f'results/hne_routed/{ds}/context_free/fold{k}/selection.json').read_text());folders=[]
            for checkpoint in selected['runs']:
                seed=int(Path(checkpoint).name.removeprefix('seed'));folder=ROOT/f'results/hne_routed/{ds}/context_free_rank/fold{k}/seed{seed}'
                folders.append(folder);folder.mkdir(parents=True,exist_ok=True)
                if (folder/'result.json').exists():continue
                started=time.time();nodes,risk=cached(ds,k,seed);torch.manual_seed(seed)
                model=HNEContextFree(config);model.load_state_dict(torch.load(ROOT/checkpoint/'model.pt',weights_only=True,map_location='cpu'));model.eval()
                with torch.no_grad():
                    candidates=scores(model,nodes[train],risk[train]);loss=F.binary_cross_entropy_with_logits(candidates,y[train,None].expand_as(candidates),reduction='none')
                    scale=loss.std(1,keepdim=True,unbiased=False).clamp_min(1e-4)
                    q=(-(loss-loss.mean(1,keepdim=True))/scale).softmax(1)
                for p in model.parameters():p.requires_grad_(False)
                for p in model.router.parameters():p.requires_grad_(True)
                weights=person_weights(meta.person_id.to_numpy()[train]);prior=np.average(y[train].numpy(),weights=weights)
                weights*=np.where(y[train].numpy()>0,.5/max(prior,1e-5),.5/max(1-prior,1e-5));weights/=weights.mean();weights=torch.tensor(weights,dtype=torch.float32)
                p,_,_=predict(model,nodes[val],risk[val]);m=metrics(y[val].numpy(),p,meta.person_id.to_numpy()[val]);best=(m['person_auroc'],-m['bce'])
                best_epoch=0;state=copy.deepcopy(model.router.state_dict());history=[dict(epoch=0,**m)]
                optimizer=torch.optim.AdamW(model.router.parameters(),lr=config['path_policy_learning_rate'],weight_decay=config['weight_decay'])
                for epoch in range(config['path_policy_epochs']):
                    model.router.train();order=torch.randperm(len(train));losses=[]
                    for start in range(0,len(train),config['path_policy_batch_size']):
                        ix=order[start:start+config['path_policy_batch_size']];optimizer.zero_grad(set_to_none=True)
                        policy=path_distribution(model,nodes[train[ix.numpy()]])
                        loss=(-(q[ix]*policy.clamp_min(1e-8).log()).sum(1)*weights[ix]).mean();loss.backward();optimizer.step();losses.append(float(loss.detach()))
                    p,_,_=predict(model,nodes[val],risk[val]);m=metrics(y[val].numpy(),p,meta.person_id.to_numpy()[val]);key=(m['person_auroc'],-m['bce'])
                    history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**m))
                    if key>best:best=key;best_epoch=epoch+1;state=copy.deepcopy(model.router.state_dict())
                model.router.load_state_dict(state);torch.save(model.state_dict(),folder/'model.pt')
                original=json.loads((ROOT/checkpoint/'result.json').read_text());constant=original['constant_path_index']
                result=dict(source_checkpoint=checkpoint,seed=seed,best_epoch=best_epoch,history=history,fit_seconds=time.time()-started,
                    active_trainable_parameters=sum(p.numel() for p in model.router.parameters()),loss_scale_floor=1e-4,
                    training='router only; standardized per-example final-path losses; source predictor and readout fixed',constant_path_index=constant)
                for part,ids in [('validation',val),('held_person',held)]:
                    p,r,all_p=predict(model,nodes[ids],risk[ids]);frame=meta.iloc[ids].copy();frame['probability']=p
                    frame['constant_path_probability']=all_p[:,constant]
                    for j in range(3):frame[f'route_{j+1}']=r[:,j]
                    frame.to_csv(folder/f'{part}.csv',index=False);result[part]=metrics(y[ids].numpy(),p,frame.person_id.to_numpy())
                (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
                print(json.dumps({'ranking_fit':str(folder.relative_to(ROOT)),'best_epoch':best_epoch,'held_person':result['held_person']}),flush=True)
            destination=folders[0].parent
            for part in ['validation','held_person']:
                frames=[pd.read_csv(p/f'{part}.csv',dtype={'person_id':str}) for p in folders];frame=frames[0].copy()
                for col in ['probability','constant_path_probability']:frame[col]=np.mean([f[col] for f in frames],axis=0)
                frame.to_csv(destination/f'{part}.csv',index=False)
            (destination/'selection.json').write_text(json.dumps({'config':config,'runs':[str(f.relative_to(ROOT)) for f in folders]},indent=2))


if __name__=='__main__':run()
