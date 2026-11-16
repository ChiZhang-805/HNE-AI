"""Train context readout and computation policy over all 27 cached H/N/E paths."""
from pathlib import Path
import argparse,json,sys,time,itertools
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import HNEContextFree
from run_hne_routed import metrics,person_weights
PATHS=np.array(list(itertools.product(range(3),repeat=3)))
VARIANTS=['context_free','context_free_fixed','context_free_plain']
FIXED=5  # H, N, E in lexicographic path enumeration.


def cached(ds,k,seed):
    folder=ROOT/f'results/hne_routed/{ds}/path_policy/fold{k}/seed{seed}'
    arrays=np.load(folder/'rollouts.npz');order=np.argsort(arrays['ids'])
    p=torch.tensor(arrays['probabilities'][order]).clamp(1e-6,1-1e-6)
    return torch.tensor(arrays['features'][order]),torch.logit(p)


def path_distribution(model,nodes):
    logp=model.router(nodes).log_softmax(-1)
    return torch.stack([logp[:,0,p[0]]+logp[:,1+p[0],p[1]]+
                        logp[:,4+3*p[0]+p[1],p[2]] for p in PATHS],1).exp()


def choose_path(model,nodes):
    logits=model.router(nodes);ix=torch.arange(len(nodes))
    a=logits[:,0].argmax(-1);b=logits[ix,1+a].argmax(-1);c=logits[ix,4+3*a+b].argmax(-1)
    return 9*a+3*b+c,torch.stack([a,b,c],1)


def scores(model,nodes,risk):
    root=model.context_head(nodes[:,0]).squeeze(-1)
    return root[:,None].expand_as(risk) if model.context_variant=='context_free_plain' else root[:,None]+model.risk_scale*risk


def predict(model,nodes,risk):
    model.eval()
    with torch.no_grad():
        candidate=scores(model,nodes,risk)
        if model.context_variant=='context_free':chosen,routes=choose_path(model,nodes)
        else:
            chosen=torch.full((len(nodes),),FIXED,dtype=torch.long)
            routes=None if model.context_variant=='context_free_plain' else torch.tensor(PATHS[FIXED]).expand(len(nodes),-1)
        p=candidate[torch.arange(len(nodes)),chosen].sigmoid().numpy()
    return p,routes.numpy() if routes is not None else None,candidate.sigmoid().numpy()


def train(ds,k,source,nodes,risk,variant,config):
    seed=int(Path(source).name.removeprefix('seed'));folder=ROOT/f'results/hne_routed/{ds}/{variant}/fold{k}/seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():return
    meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
    train_ids=np.flatnonzero(~meta.fold.isin([k,(k+1)%5]));val=np.flatnonzero(meta.fold.eq((k+1)%5))
    y=torch.tensor(meta.y.to_numpy(),dtype=torch.float32)
    weights=person_weights(meta.person_id.to_numpy()[train_ids]);prior=np.average(y[train_ids].numpy(),weights=weights)
    weights*=np.where(y[train_ids].numpy()>0,.5/max(prior,1e-5),.5/max(1-prior,1e-5))
    weights/=weights.mean();weights=torch.tensor(weights,dtype=torch.float32)
    torch.manual_seed(seed);model=HNEContextFree(config,variant)
    model.load_state_dict(torch.load(ROOT/source/'model.pt',weights_only=True,map_location='cpu'),strict=False)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in model.context_head.parameters():parameter.requires_grad_(True)
    if variant!='context_free_plain':model.risk_scale.requires_grad_(True)
    if variant=='context_free':
        for parameter in model.router.parameters():parameter.requires_grad_(True)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=config['path_policy_learning_rate'],weight_decay=config['weight_decay'])
    best=(-np.inf,-np.inf);history=[];started=time.time()
    for epoch in range(config['path_policy_epochs']):
        model.train();order=torch.randperm(len(train_ids));losses=[]
        for start in range(0,len(order),config['path_policy_batch_size']):
            ix=order[start:start+config['path_policy_batch_size']];ids=train_ids[ix.numpy()];optimizer.zero_grad(set_to_none=True)
            candidate=scores(model,nodes[ids],risk[ids]);target=y[ids,None].expand_as(candidate)
            loss_by_path=F.binary_cross_entropy_with_logits(candidate,target,reduction='none')
            if variant=='context_free':
                policy=path_distribution(model,nodes[ids]);loss=(policy*loss_by_path).sum(1)
                loss=loss+0.001*(policy*policy.clamp_min(1e-8).log()).sum(1)
            else:loss=loss_by_path[:,FIXED]
            loss=(loss*weights[ix]).mean();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        p,_,_=predict(model,nodes[val],risk[val]);m=metrics(y[val].numpy(),p,meta.person_id.to_numpy()[val]);key=(m['person_auroc'],-m['bce'])
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**m))
        if key>best:best=key;best_epoch=epoch+1;torch.save(model.state_dict(),folder/'model.pt')
    model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cpu'))
    p,r,all_p=predict(model,nodes[val],risk[val]);frame=meta.iloc[val].copy();frame['probability']=p
    constant=int(np.argmax([metrics(y[val].numpy(),all_p[:,j],meta.person_id.to_numpy()[val])['person_auroc'] for j in range(27)]))
    if r is not None:
        for j in range(3):frame[f'route_{j+1}']=r[:,j]
        frame['constant_path_probability']=all_p[:,constant]
    frame.to_csv(folder/'validation.csv',index=False)
    result=dict(source_checkpoint=source,variant=variant,seed=seed,best_epoch=best_epoch,history=history,
        validation=metrics(y[val].numpy(),p,frame.person_id.to_numpy()),fit_seconds=time.time()-started,
        active_trainable_parameters=sum(p.numel() for p in parameters),constant_path_index=constant,
        risk_scale=float(model.risk_scale.detach()),training='fixed stage-one weights; context head and exact expected 27-path risk',
        path_orders=PATHS.tolist(),entropy_weight=.001)
    (folder/'fit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'free_fit':str(folder.relative_to(ROOT)),'validation':result['validation']}),flush=True)


def freeze_and_evaluate(config):
    selection_path=ROOT/'results/hne_routed/architecture_selection.json'
    previous=json.loads(selection_path.read_text());means=dict(previous['validation_means']);by_dataset=dict(previous['validation_by_dataset'])
    personal=dict(previous['participant_balanced_validation_means']);counts=dict(previous['selection_counts'])
    for variant in VARIANTS:
        values=[]
        for ds in config['datasets']:
            for k in config['folds']:
                for folder in sorted((ROOT/f'results/hne_routed/{ds}/{variant}/fold{k}').glob('seed*')):
                    fit=json.loads((folder/'fit.json').read_text());v=pd.read_csv(folder/'validation.csv')
                    values.append(dict(ds=ds,auroc=fit['validation']['auroc'],person=fit['validation']['person_auroc'],pairs=v.y.sum()*(len(v)-v.y.sum())))
        by_dataset[variant]={ds:sum(x['auroc']*x['pairs'] for x in values if x['ds']==ds)/sum(x['pairs'] for x in values if x['ds']==ds) for ds in config['datasets']}
        means[variant]=float(np.mean(list(by_dataset[variant].values())));personal[variant]=float(np.mean([v['person'] for v in values]));counts[variant]=len(values)
    choice={**previous,'primary_variant':max(['path_policy','context_path','context_free'],key=means.get),
            'validation_means':means,'validation_by_dataset':by_dataset,'participant_balanced_validation_means':personal,'selection_counts':counts}
    selection_path.write_text(json.dumps(choice,indent=2),encoding='utf8');print(json.dumps(choice),flush=True)
    for ds in config['datasets']:
        meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
        for k in config['folds']:
            held=np.flatnonzero(meta.fold.eq(k))
            for variant in VARIANTS:
                destination=ROOT/f'results/hne_routed/{ds}/{variant}/fold{k}';folders=sorted(destination.glob('seed*'))
                for folder in folders:
                    result=json.loads((folder/'fit.json').read_text());nodes,risk=cached(ds,k,result['seed'])
                    model=HNEContextFree(config,variant);model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cpu'))
                    p,r,all_p=predict(model,nodes[held],risk[held]);frame=meta.iloc[held].copy();frame['probability']=p
                    if r is not None:
                        for j in range(3):frame[f'route_{j+1}']=r[:,j]
                        frame['constant_path_probability']=all_p[:,result['constant_path_index']]
                    frame.to_csv(folder/'held_person.csv',index=False);result['held_person']=metrics(frame.y.to_numpy(),p,frame.person_id.to_numpy())
                    (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
                for part in ['validation','held_person']:
                    frames=[pd.read_csv(folder/f'{part}.csv',dtype={'person_id':str}) for folder in folders];frame=frames[0].copy()
                    for column in ['probability','constant_path_probability']:
                        if column in frame:frame[column]=np.mean([f[column] for f in frames],axis=0)
                    frame.to_csv(destination/f'{part}.csv',index=False)
                (destination/'selection.json').write_text(json.dumps({'config':config,'runs':[str(f.relative_to(ROOT)) for f in folders]},indent=2))


def run():
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    for ds in config['datasets']:
        for k in config['folds']:
            selected=json.loads((ROOT/f'results/hne_routed/{ds}/learned/fold{k}/selection.json').read_text())
            for source in selected['runs']:
                seed=int(Path(source).name.removeprefix('seed'));nodes,risk=cached(ds,k,seed)
                for variant in VARIANTS:train(ds,k,source,nodes,risk,variant,config)
    freeze_and_evaluate(config)


if __name__=='__main__':run()
