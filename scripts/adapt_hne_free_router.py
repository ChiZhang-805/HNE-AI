"""Cross-domain adaptation of the 27-path controller, with source weights fixed."""
from pathlib import Path
import copy,json,sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import HNEContextFree,RoutedData
from train_hne_path_policy import enumerate_paths
from train_hne_free_paths import path_distribution,choose_path,PATHS
from run_hne_routed import metrics,person_weights


def routed(model,nodes,probabilities):
    model.eval()
    with torch.no_grad():indices,routes=choose_path(model,nodes)
    return probabilities[np.arange(len(nodes)),indices.numpy()],routes.numpy()


def run():
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    for source,target in [('RESILIENT','BrisT1D-Open'),('BrisT1D-Open','RESILIENT')]:
        selected=json.loads((ROOT/f'results/hne_routed/{source}/context_free/fold0/selection.json').read_text())
        data=RoutedData(target);meta=pd.read_csv(ROOT/f'data/hne_routed/{target}/samples.csv',dtype={'person_id':str})
        people=meta.person_id.to_numpy();fold=meta.fold.to_numpy();y=torch.tensor(meta.y.to_numpy(),dtype=torch.float32)
        root=ROOT/f'results/hne_routed/router_transfer_free/{source}_to_{target}';root.mkdir(parents=True,exist_ok=True)
        for checkpoint in selected['runs']:
            seed=int(Path(checkpoint).name.removeprefix('seed'));cache_path=root/f'source_seed{seed}_rollouts.npz'
            model=HNEContextFree(config);model.load_state_dict(torch.load(ROOT/checkpoint/'model.pt',weights_only=True,map_location='cpu'))
            if not cache_path.exists():
                model.cuda();probabilities,features=enumerate_paths(model,data,np.arange(len(meta)),config['batch_size'])
                np.savez_compressed(cache_path,probabilities=probabilities,features=features)
                model.cpu();torch.cuda.empty_cache();print('Cached',source,target,seed,flush=True)
            arrays=np.load(cache_path);nodes=torch.tensor(arrays['features']);probabilities=arrays['probabilities']
            source_router=copy.deepcopy(model.router.state_dict());zero,_=routed(model,nodes,probabilities)
            candidate=torch.logit(torch.tensor(probabilities).clamp(1e-6,1-1e-6))
            for parameter in model.parameters():parameter.requires_grad_(False)
            for parameter in model.router.parameters():parameter.requires_grad_(True)
            for k in config['folds']:
                destination=root/f'fold{k}/seed{seed}';destination.mkdir(parents=True,exist_ok=True)
                if (destination/'result.json').exists():continue
                pool=np.unique(people[~np.isin(fold,[k,(k+1)%5])]);rng=np.random.default_rng(20261113+k)
                chosen=rng.permutation(pool)[:2];train=np.flatnonzero(np.isin(people,chosen));val=np.flatnonzero(fold==(k+1)%5);held=np.flatnonzero(fold==k)
                weights=person_weights(people[train]);prior=np.average(y[train].numpy(),weights=weights)
                weights*=np.where(y[train].numpy()>0,.5/max(prior,1e-5),.5/max(1-prior,1e-5));weights/=weights.mean();weights=torch.tensor(weights,dtype=torch.float32)
                loss_by_path=F.binary_cross_entropy_with_logits(candidate[train],y[train,None].expand(-1,27),reduction='none')
                model.router.load_state_dict(source_router);initial,_=routed(model,nodes[val],probabilities[val])
                m=metrics(y[val].numpy(),initial,people[val]);best=(m['person_auroc'],-m['bce']);best_epoch=0;state=copy.deepcopy(source_router)
                history=[dict(epoch=0,**m)];torch.manual_seed(seed)
                optimizer=torch.optim.AdamW(model.router.parameters(),lr=config['path_policy_learning_rate'],weight_decay=config['weight_decay'])
                for epoch in range(config['path_policy_epochs']):
                    model.router.train();order=torch.randperm(len(train));losses=[]
                    for start in range(0,len(train),config['path_policy_batch_size']):
                        ix=order[start:start+config['path_policy_batch_size']];optimizer.zero_grad(set_to_none=True)
                        policy=path_distribution(model,nodes[train[ix.numpy()]])
                        loss=((policy*loss_by_path[ix]).sum(1)+.001*(policy*policy.clamp_min(1e-8).log()).sum(1))
                        loss=(loss*weights[ix]).mean();loss.backward();optimizer.step();losses.append(float(loss.detach()))
                    p,_=routed(model,nodes[val],probabilities[val]);m=metrics(y[val].numpy(),p,people[val]);key=(m['person_auroc'],-m['bce'])
                    history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**m))
                    if key>best:best=key;best_epoch=epoch+1;state=copy.deepcopy(model.router.state_dict())
                model.router.load_state_dict(state);torch.save(state,destination/'router.pt')
                constant=int(np.argmax([metrics(y[val].numpy(),probabilities[val,j],people[val])['person_auroc'] for j in range(27)]))
                result=dict(source_checkpoint=checkpoint,seed=seed,best_epoch=best_epoch,history=history,
                    target_train_people=chosen.tolist(),target_train_origins=len(train),target_train_positives=int(y[train].sum()),
                    target_validation_people=np.unique(people[val]).tolist(),target_validation_origins=len(val),
                    constant_path=PATHS[constant].tolist(),updated_parameters='router only; source computation and readout fixed')
                for part,ids in [('validation',val),('held_person',held)]:
                    p,r=routed(model,nodes[ids],probabilities[ids]);frame=meta.iloc[ids].copy()
                    frame['probability']=p;frame['source_probability']=zero[ids];frame['constant_path_probability']=probabilities[ids,constant]
                    for j in range(3):frame[f'route_{j+1}']=r[:,j]
                    frame.to_csv(destination/f'{part}.csv',index=False);result[part]=metrics(y[ids].numpy(),p,people[ids])
                (destination/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
                print(json.dumps({'free_router_transfer':str(destination.relative_to(ROOT)),'held_person':result['held_person']}),flush=True)
        for k in config['folds']:
            destination=root/f'fold{k}'
            for part in ['validation','held_person']:
                frames=[pd.read_csv(p/f'{part}.csv',dtype={'person_id':str}) for p in sorted(destination.glob('seed*'))];frame=frames[0].copy()
                for col in ['probability','source_probability','constant_path_probability']:frame[col]=np.mean([f[col] for f in frames],axis=0)
                frame.to_csv(destination/f'{part}.csv',index=False)


if __name__=='__main__':run()
