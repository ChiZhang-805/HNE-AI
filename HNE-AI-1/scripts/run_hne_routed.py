"""Train the same pretrained HNE model and computation paths in both cohorts."""
from pathlib import Path
import argparse, json, sys, time, random, gc
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.hne_routed import HNERouted, RoutedData, objective


def person_weights(persons):
    _,inverse,counts=np.unique(persons,return_inverse=True,return_counts=True)
    w=1/counts[inverse]
    return w/w.mean()


def metrics(y,p,persons):
    # person_auroc is pooled AUROC with inverse participant sample counts;
    # it still includes pairs from different participants.
    w=person_weights(persons)
    return {'auroc':float(roc_auc_score(y,p)), 'person_auroc':float(roc_auc_score(y,p,sample_weight=w)),
            'ap':float(average_precision_score(y,p)), 'bce':float(log_loss(y,p,labels=[0,1]))}


def predict(model,data,ids,batch_size,force_path=None):
    model.eval(); probabilities=[];routes=[];futures=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(ids),batch_size):
            out=model(*data.batch(ids[start:start+batch_size]),force_path=force_path)
            probabilities.append(out['logits'].float().sigmoid().cpu().numpy())
            futures.append(out['future'].float().cpu().numpy())
            if 'routes' in out:routes.append(out['routes'].cpu().numpy())
    return np.concatenate(probabilities),np.concatenate(routes) if routes else None,np.concatenate(futures)


def fit(data,meta,train_ids,val_ids,test_ids,config,variant,seed,lr,folder,initial=None):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=HNERouted(config,variant).cuda()
    if initial is not None:model.load_state_dict(torch.load(initial,weights_only=True,map_location='cuda'))
    initial_router=model.router[3].weight.detach().cpu().clone()
    y=data.arrays['y'];persons=meta.person_id.to_numpy()
    weights=person_weights(persons[train_ids]); py=np.average(y[train_ids],weights=weights)
    weights=weights*np.where(y[train_ids]>0,.5/max(py,1e-5),.5/max(1-py,1e-5))
    weights/=weights.mean()
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=lr,weight_decay=config['weight_decay'])
    start=time.perf_counter();best=(-np.inf,-np.inf);history=[]
    for epoch in range(config['epochs']):
        model.train();order=np.random.permutation(len(train_ids));losses=[]
        for offset in range(0,len(order),config['batch_size']):
            local=order[offset:offset+config['batch_size']];ids=train_ids[local]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(*data.batch(ids),epoch=epoch)
                loss=objective(output,torch.tensor(y[ids],device='cuda'),
                    torch.tensor(weights[local],device='cuda',dtype=torch.float32),
                    torch.tensor(data.arrays['future_targets'][ids],device='cuda'),config,variant)
            if not torch.isfinite(loss):raise FloatingPointError(f'{folder} epoch{epoch}: nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.)
            optimizer.step();losses.append(float(loss.detach()))
        vp,vr,_=predict(model,data,val_ids,config['batch_size'])
        m=metrics(y[val_ids],vp,persons[val_ids]);key=(m['person_auroc'],-m['bce'])
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),**m,'seconds':time.perf_counter()-start}
        if vr is not None:record['route_counts']=np.bincount(vr.flatten(),minlength=3).tolist()
        history.append(record)
        if key>best:
            best=key;best_epoch=epoch+1
            torch.save(model.state_dict(),folder/'model.pt')
        if epoch%5==0 or epoch==config['epochs']-1:
            print(json.dumps({'run':str(folder.relative_to(ROOT)),**record}),flush=True)
    model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cuda'))
    result={'variant':variant,'seed':seed,'learning_rate':lr,'best_epoch':best_epoch,'epochs':config['epochs'],
            'trainable_parameters':sum(p.numel() for p in trainable),'fit_seconds':time.perf_counter()-start,'history':history}
    for part,ids in [('validation',val_ids),('held_person',test_ids)]:
        p,r,f=predict(model,data,ids,config['batch_size'])
        frame=meta.iloc[ids].copy();frame['probability']=p
        if r is not None:
            for step in range(r.shape[1]):frame[f'route_{step+1}']=r[:,step]
        frame.to_csv(folder/f'{part}.csv',index=False)
        np.save(folder/f'{part}_future.npy',f)
        result[part]=metrics(y[ids],p,persons[ids])
    result['lora_max_changes_from_zero']={name:float(p.detach().abs().max()) for name,p in model.named_parameters() if name.endswith('.b')}
    result['router_max_weight_change']=float((model.router[3].weight.detach().cpu()-initial_router).abs().max())
    result['active_trainable_parameters']=sum(p.numel() for p in trainable if p.grad is not None)
    (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'complete':str(folder.relative_to(ROOT)),'validation':result['validation'],'held_person':result['held_person'],'seconds':result['fit_seconds']}),flush=True)
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return result


def run(args):
    config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    if args.epochs:config['epochs']=args.epochs
    torch.set_num_threads(2)
    for ds in args.datasets:
        data=RoutedData(ds);meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
        fold=data.arrays['fold']
        for variant in args.variants:
            data=RoutedData(ds,native=variant=='native')
            for k in args.folds:
                train=np.flatnonzero(~np.isin(fold,[k,(k+1)%5]));val=np.flatnonzero(fold==(k+1)%5);test=np.flatnonzero(fold==k)
                dest=ROOT/'results/hne_routed'/ds/variant/f'fold{k}'
                lrs=config['learning_rates'] if variant in ['learned','plain','fixed','native'] else [config['learning_rates'][0]]
                runs=[]
                for lr in lrs:
                    path=dest/f'lr{lr}'/f'seed{config["seeds"][0]}'
                    r=fit(data,meta,train,val,test,config,variant,config['seeds'][0],lr,path)
                    runs.append((r['validation']['person_auroc'],-r['validation']['bce'],lr,path))
                _,_,chosen,selected=max(runs,key=lambda x:x[:2]);selected_paths=[selected]
                for seed in config['seeds'][1:args.seeds]:
                    path=dest/f'lr{chosen}'/f'seed{seed}'
                    fit(data,meta,train,val,test,config,variant,seed,chosen,path);selected_paths.append(path)
                for part in ['validation','held_person']:
                    frames=[pd.read_csv(p/f'{part}.csv',dtype={'person_id':str}) for p in selected_paths]
                    frame=frames[0].copy();frame['probability']=np.mean([f.probability.to_numpy() for f in frames],axis=0)
                    frame.to_csv(dest/f'{part}.csv',index=False)
                (dest/'selection.json').write_text(json.dumps({'config':config,'lr':chosen,'runs':[str(p.relative_to(ROOT)) for p in selected_paths]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=['RESILIENT','BrisT1D-Open'])
    p.add_argument('--variants',nargs='+',default=['learned','plain','fixed'])
    p.add_argument('--folds',nargs='+',type=int,default=list(range(5)))
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--epochs',type=int)
    run(p.parse_args())
