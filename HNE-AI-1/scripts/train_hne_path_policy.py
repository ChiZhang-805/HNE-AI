"""Learn full-path choices under fixed model weights from 27 training rollouts.

Only training outcomes create action targets. A validation-selected constant
path is a comparator; held-person labels never participate in routing.
"""
from pathlib import Path
import argparse,itertools,json,sys,copy,time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import HNERouted,RoutedData
from run_hne_routed import metrics,person_weights
PATHS=np.array(list(itertools.product(range(3),repeat=3)))
# Nodes 0: root; 1..3: first action; 4..12: first two actions.


def enumerate_paths(model,data,ids,bs):
    probabilities=[];features=np.empty((len(ids),13,288),np.float32)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for path in PATHS:
            scores=[]
            for start in range(0,len(ids),bs):
                out=model(*data.batch(ids[start:start+bs]),force_path=path)
                scores.append(out['logits'].float().sigmoid().cpu().numpy())
                f=out['route_features'].float().cpu().numpy()
                features[start:start+bs,0]=f[:,0]
                features[start:start+bs,1+path[0]]=f[:,1]
                features[start:start+bs,4+path[0]*3+path[1]]=f[:,2]
            probabilities.append(np.concatenate(scores))
    return np.stack(probabilities,1),features


def teacher(probability,y,temperature=.1):
    p=np.clip(probability,1e-6,1-1e-6)
    losses=-(y[:,None]*np.log(p)+(1-y[:,None])*np.log1p(-p))
    logq=-losses/temperature;logq-=logq.max(1,keepdims=True)
    q=np.exp(logq);q/=q.sum(1,keepdims=True)
    targets=np.zeros((len(y),13,3),np.float32);mass=np.zeros((len(y),13),np.float32)
    for i,path in enumerate(PATHS):
        for node,action in [(0,path[0]),(1+path[0],path[1]),(4+path[0]*3+path[1],path[2])]:
            targets[:,node,action]+=q[:,i];mass[:,node]+=q[:,i]
    targets/=np.maximum(mass[:,:,None],1e-10)
    return targets,mass


def route_predictions(router,features,probabilities):
    router.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        f=torch.tensor(features,device='cuda');idx=torch.arange(len(f),device='cuda')
        a=router(f[:,0]).argmax(-1)
        b=router(f[idx,1+a]).argmax(-1)
        c=router(f[idx,4+3*a+b]).argmax(-1)
        choices=(9*a+3*b+c).cpu().numpy();routes=torch.stack([a,b,c],1).cpu().numpy()
    return probabilities[np.arange(len(probabilities)),choices],routes


def run(args):
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    for ds in args.datasets:
        data=RoutedData(ds);meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
        folds=data.arrays['fold'];y=data.arrays['y'];person=meta.person_id.to_numpy()
        for k in args.folds:
            selected=json.loads((ROOT/f'results/hne_routed/{ds}/learned/fold{k}/selection.json').read_text())
            train=np.flatnonzero(~np.isin(folds,[k,(k+1)%5]));val=np.flatnonzero(folds==(k+1)%5);test=np.flatnonzero(folds==k)
            all_ids=np.r_[train,val,test];parts={'validation':(val,slice(len(train),len(train)+len(val))),
                                            'held_person':(test,slice(len(train)+len(val),None))}
            completed=[]
            for source in selected['runs'][:args.seeds]:
                seed=int(Path(source).name.removeprefix('seed'))
                folder=ROOT/f'results/hne_routed/{ds}/path_policy/fold{k}'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
                completed.append(folder)
                if (folder/'result.json').exists():
                    old=json.loads((folder/'result.json').read_text())
                    if old.get('router_inference_precision')=='bfloat16':continue
                    (folder/'float32_router_result.json').write_text(json.dumps(old,indent=2))
                    (folder/'float32_router_model.pt').write_bytes((folder/'model.pt').read_bytes())
                started=time.time();torch.manual_seed(seed);np.random.seed(seed)
                model=HNERouted(config).cuda();model.load_state_dict(torch.load(ROOT/source/'model.pt',weights_only=True,map_location='cuda'))
                cache=folder/'rollouts.npz'
                if cache.exists():
                    saved=np.load(cache);probabilities,features=saved['probabilities'],saved['features']
                else:
                    probabilities,features=enumerate_paths(model,data,all_ids,config['batch_size'])
                    np.savez_compressed(cache,probabilities=probabilities,features=features,ids=all_ids)
                q,mass=teacher(probabilities[:len(train)],y[train],config['path_policy_temperature'])
                weights=person_weights(person[train]);py=np.average(y[train],weights=weights)
                weights*=np.where(y[train]>0,.5/max(py,1e-5),.5/max(1-py,1e-5))
                x=torch.tensor(features[:len(train)].reshape(-1,288),device='cuda')
                targets=torch.tensor(q.reshape(-1,3),device='cuda')
                w=torch.tensor((mass*weights[:,None]).reshape(-1),device='cuda');w/=w.mean()
                for p in model.parameters():p.requires_grad_(False)
                for p in model.router.parameters():p.requires_grad_(True)
                optimizer=torch.optim.AdamW(model.router.parameters(),lr=config['path_policy_learning_rate'],weight_decay=config['weight_decay'])
                vs=parts['validation'][1];best=(-np.inf,-np.inf);history=[]
                for epoch in range(config['path_policy_epochs']):
                    model.router.train();order=torch.randperm(len(x),device='cuda');losses=[]
                    for start in range(0,len(x),config['path_policy_batch_size']):
                        ix=order[start:start+config['path_policy_batch_size']];optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            logits=model.router(x[ix]).float()
                            loss=(-(targets[ix]*logits.log_softmax(-1)).sum(-1)*w[ix]).mean()
                        loss.backward();optimizer.step();losses.append(float(loss.detach()))
                    p,_=route_predictions(model.router,features[vs],probabilities[vs]);m=metrics(y[val],p,person[val])
                    key=(m['person_auroc'],-m['bce'])
                    history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),**m})
                    if key>best:best=key;best_epoch=epoch+1;best_router=copy.deepcopy(model.router.state_dict())
                model.router.load_state_dict(best_router)
                torch.save(model.state_dict(),folder/'model.pt')
                validation_path_scores=[metrics(y[val],probabilities[vs,i],person[val])['person_auroc'] for i in range(27)]
                constant=int(np.argmax(validation_path_scores))
                result={'source_checkpoint':source,'seed':seed,'best_epoch':best_epoch,'history':history,
                        'path_temperature':config['path_policy_temperature'],'train_oracle_paths':27,'constant_path_from_validation':PATHS[constant].tolist(),
                        'fit_seconds':time.time()-started,'core_weights':'fixed; only router updated in stage two',
                        'router_inference_precision':'bfloat16',
                        'active_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
                for part,(ids,sl) in parts.items():
                    p,r=route_predictions(model.router,features[sl],probabilities[sl])
                    frame=meta.iloc[ids].copy();frame['probability']=p;frame['constant_path_probability']=probabilities[sl,constant]
                    for j in range(3):frame[f'route_{j+1}']=r[:,j]
                    frame.to_csv(folder/f'{part}.csv',index=False);result[part]=metrics(y[ids],p,person[ids])
                (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
                print(json.dumps({'path_policy':str(folder.relative_to(ROOT)),'validation':result['validation'],'held_person':result['held_person'],'seconds':result['fit_seconds']}),flush=True)
                del model,optimizer,x,targets,w;torch.cuda.empty_cache()
            destination=completed[0].parent
            for part in parts:
                frames=[pd.read_csv(f/f'{part}.csv',dtype={'person_id':str}) for f in completed]
                frame=frames[0].copy()
                for col in ['probability','constant_path_probability']:frame[col]=np.mean([f[col] for f in frames],axis=0)
                frame.to_csv(destination/f'{part}.csv',index=False)
            (destination/'selection.json').write_text(json.dumps({'config':config,'runs':[str(f.relative_to(ROOT)) for f in completed]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['RESILIENT','BrisT1D-Open']);p.add_argument('--folds',type=int,nargs='+',default=list(range(5)));p.add_argument('--seeds',type=int,default=3)
    run(p.parse_args())
