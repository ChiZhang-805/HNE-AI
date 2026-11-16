"""Adapt only the source router using two target training participants per fold."""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import HNERouted,RoutedData
from train_hne_path_policy import enumerate_paths,teacher,route_predictions,PATHS
from run_hne_routed import metrics,person_weights


def run(args):
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    for source,target in [('RESILIENT','BrisT1D-Open'),('BrisT1D-Open','RESILIENT')]:
        selection=json.loads((ROOT/f'results/hne_routed/{source}/path_policy/fold0/selection.json').read_text())
        data=RoutedData(target);meta=pd.read_csv(ROOT/f'data/hne_routed/{target}/samples.csv',dtype={'person_id':str})
        y=data.arrays['y'];fold=data.arrays['fold'];people=meta.person_id.to_numpy();ids=np.arange(len(meta))
        root=ROOT/'results/hne_routed/router_transfer'/f'{source}_to_{target}';root.mkdir(parents=True,exist_ok=True)
        for checkpoint in selection['runs']:
            seed=int(Path(checkpoint).name.removeprefix('seed'));cache_path=root/f'source_seed{seed}_rollouts.npz'
            model=HNERouted(config).cuda();model.load_state_dict(torch.load(ROOT/checkpoint/'model.pt',weights_only=True,map_location='cuda'))
            if cache_path.exists():
                cache=np.load(cache_path);probabilities,features=cache['probabilities'],cache['features']
            else:
                probabilities,features=enumerate_paths(model,data,ids,config['batch_size'])
                np.savez_compressed(cache_path,probabilities=probabilities,features=features)
            source_router=copy.deepcopy(model.router.state_dict())
            zero,_=route_predictions(model.router,features,probabilities)
            for k in range(5):
                output=root/f'fold{k}'/f'seed{seed}';output.mkdir(parents=True,exist_ok=True)
                if (output/'result.json').exists():continue
                pool=np.unique(people[~np.isin(fold,[k,(k+1)%5])])
                rng=np.random.default_rng(20261113+k);chosen=rng.permutation(pool)[:args.people]
                train=np.flatnonzero(np.isin(people,chosen));val=np.flatnonzero(fold==(k+1)%5);held=np.flatnonzero(fold==k)
                q,mass=teacher(probabilities[train],y[train],config['path_policy_temperature'])
                weights=person_weights(people[train]);py=np.average(y[train],weights=weights)
                weights*=np.where(y[train]>0,.5/max(py,1e-5),.5/max(1-py,1e-5))
                x=torch.tensor(features[train].reshape(-1,288),device='cuda');targets=torch.tensor(q.reshape(-1,3),device='cuda')
                w=torch.tensor((mass*weights[:,None]).reshape(-1),device='cuda');w/=w.mean()
                model.router.load_state_dict(source_router)
                for p in model.parameters():p.requires_grad_(False)
                for p in model.router.parameters():p.requires_grad_(True)
                torch.manual_seed(seed);optimizer=torch.optim.AdamW(model.router.parameters(),lr=config['path_policy_learning_rate'],weight_decay=config['weight_decay'])
                best=(-np.inf,-np.inf);history=[]
                for epoch in range(config['path_policy_epochs']):
                    model.router.train();order=torch.randperm(len(x),device='cuda');losses=[]
                    for start in range(0,len(x),config['path_policy_batch_size']):
                        ix=order[start:start+config['path_policy_batch_size']];optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            loss=(-(targets[ix]*model.router(x[ix]).float().log_softmax(-1)).sum(-1)*w[ix]).mean()
                        loss.backward();optimizer.step();losses.append(float(loss.detach()))
                    p,_=route_predictions(model.router,features[val],probabilities[val]);m=metrics(y[val],p,people[val]);key=(m['person_auroc'],-m['bce'])
                    history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**m))
                    if key>best:best=key;best_epoch=epoch+1;best_router=copy.deepcopy(model.router.state_dict())
                model.router.load_state_dict(best_router);torch.save(best_router,output/'router.pt')
                constant=int(np.argmax([metrics(y[val],probabilities[val,j],people[val])['person_auroc'] for j in range(27)]))
                result=dict(source=source,target=target,source_checkpoint=checkpoint,seed=seed,best_epoch=best_epoch,history=history,
                    target_train_people=chosen.tolist(),target_train_origins=len(train),target_train_positives=int(y[train].sum()),
                    target_validation_people=np.unique(people[val]).tolist(),target_validation_origins=len(val),
                    source_weights='fixed; target updates router only',validation_constant_path=PATHS[constant].tolist())
                for part,ix in [('validation',val),('held_person',held)]:
                    p,r=route_predictions(model.router,features[ix],probabilities[ix]);frame=meta.iloc[ix].copy()
                    frame['probability']=p;frame['source_probability']=zero[ix];frame['constant_path_probability']=probabilities[ix,constant]
                    for j in range(3):frame[f'route_{j+1}']=r[:,j]
                    frame.to_csv(output/f'{part}.csv',index=False);result[part]=metrics(y[ix],p,people[ix])
                (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
                print(json.dumps({'router_transfer':str(output.relative_to(ROOT)),'held_person':result['held_person']}),flush=True)
                del optimizer,x,targets,w
            del model;torch.cuda.empty_cache()
        for k in range(5):
            output=root/f'fold{k}'
            for part in ['validation','held_person']:
                frames=[pd.read_csv(p/part_file,dtype={'person_id':str}) for p in sorted(output.glob('seed*')) for part_file in [f'{part}.csv']]
                frame=frames[0].copy()
                for col in ['probability','source_probability','constant_path_probability']:frame[col]=np.mean([f[col] for f in frames],axis=0)
                frame.to_csv(output/f'{part}.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--people',type=int,default=2);run(p.parse_args())
