"""Context-preserving readout and exact expected loss over six H/N/E orders."""
from pathlib import Path
import argparse,copy,itertools,json,sys,time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import HNERouted,HNEContextPaths,RoutedData
from run_hne_routed import metrics,person_weights
PATHS=np.array(list(itertools.permutations(range(3))))
VARIANTS=['context_path','context_fixed','context_plain']


def prepare(ds,k,source,config,cache_group='context_cache'):
    seed=Path(source).name;folder=ROOT/f'results/hne_routed/{ds}/{cache_group}/fold{k}'/seed
    folder.mkdir(parents=True,exist_ok=True);path=folder/'states.npz'
    if path.exists():return path
    data=RoutedData(ds);n=len(data.arrays['y']);d=config['hidden_width'];bs=config['batch_size']
    model=HNERouted(config).cuda().eval();model.load_state_dict(torch.load(ROOT/source/'model.pt',map_location='cuda',weights_only=True),strict=False)
    nodes=np.zeros((n,13,3*d),np.float32);readout=np.zeros((n,6,2*d),np.float32)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for pi,p in enumerate(PATHS):
            for start in range(0,n,bs):
                ids=np.arange(start,min(start+bs,n));out=model(*data.batch(ids),force_path=p)
                f=out['route_features'].float().cpu().numpy()
                nodes[ids,0]=f[:,0];nodes[ids,1+p[0]]=f[:,1];nodes[ids,4+3*p[0]+p[1]]=f[:,2]
                readout[ids,pi]=out['readout_features'].float().cpu().numpy()
    np.savez_compressed(path,nodes=nodes,readout=readout,paths=PATHS)
    (folder/'source.json').write_text(json.dumps({'checkpoint':source,'paths':PATHS.tolist()},indent=2))
    del model;torch.cuda.empty_cache();print('Prepared',ds,k,seed,flush=True)
    return path


def path_distribution(model,nodes):
    logits=model.router(nodes);parts=[]
    for p in PATHS:
        root=logits[:,0].log_softmax(-1)[:,p[0]]
        middle=logits[:,1+p[0]].clone();middle[:,p[0]]=-1e4
        parts.append(root+middle.log_softmax(-1)[:,p[1]])
    return torch.stack(parts,1).exp()


def choose_path(model,nodes):
    logits=model.router(nodes);indices=torch.arange(len(nodes),device=nodes.device)
    first=logits[:,0].argmax(-1);second_logits=logits[indices,1+first].clone()
    second_logits.scatter_(-1,first[:,None],-1e4);second=second_logits.argmax(-1);third=3-first-second
    route=torch.stack([first,second,third],1)
    lookup=torch.full((27,),-1,device=nodes.device,dtype=torch.long)
    for i,p in enumerate(PATHS):lookup[9*p[0]+3*p[1]+p[2]]=i
    return lookup[9*first+3*second+third],route


def scores(model,nodes,readout):
    base=model.context_head(nodes[:,0]).squeeze(-1)
    if model.context_variant=='context_plain':return base[:,None].expand(-1,6)
    return base[:,None]+model.context_correction(readout).squeeze(-1)


def predict(model,nodes,readout):
    model.eval()
    with torch.no_grad():
        candidates=scores(model,nodes,readout)
        if model.context_variant=='context_path':chosen,routes=choose_path(model,nodes)
        else:
            chosen=torch.zeros(len(nodes),dtype=torch.long)
            routes=None if model.context_variant=='context_plain' else torch.tensor(PATHS[0]).expand(len(nodes),-1)
        p=candidates[torch.arange(len(nodes)),chosen].sigmoid().numpy()
    return p,routes.numpy() if routes is not None else None,candidates.sigmoid().numpy()


def train(ds,k,source,cache,variant,config):
    seed=int(Path(source).name.removeprefix('seed'));folder=ROOT/f'results/hne_routed/{ds}/{variant}/fold{k}'/f'seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():return
    meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
    train_ids=np.flatnonzero(~meta.fold.isin([k,(k+1)%5]));val=np.flatnonzero(meta.fold.eq((k+1)%5))
    values=np.load(cache);nodes=torch.tensor(values['nodes']);readout=torch.tensor(values['readout']);y=torch.tensor(meta.y.to_numpy(),dtype=torch.float32)
    weights=person_weights(meta.person_id.to_numpy()[train_ids]);prior=np.average(y[train_ids].numpy(),weights=weights)
    weights*=np.where(y[train_ids].numpy()>0,.5/max(prior,1e-5),.5/max(1-prior,1e-5));weights/=weights.mean();weights=torch.tensor(weights,dtype=torch.float32)
    torch.manual_seed(seed);model=HNEContextPaths(config,variant)
    model.load_state_dict(torch.load(ROOT/source/'model.pt',weights_only=True,map_location='cpu'),strict=False)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in model.context_head.parameters():parameter.requires_grad_(True)
    if variant!='context_plain':
        for parameter in model.context_correction.parameters():parameter.requires_grad_(True)
    if variant=='context_path':
        for parameter in model.router.parameters():parameter.requires_grad_(True)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=config['path_policy_learning_rate'],weight_decay=config['weight_decay'])
    best=(-np.inf,-np.inf);history=[];started=time.time()
    for epoch in range(config['path_policy_epochs']):
        model.train();order=torch.randperm(len(train_ids));losses=[]
        for start in range(0,len(order),config['path_policy_batch_size']):
            ix=order[start:start+config['path_policy_batch_size']];ids=train_ids[ix.numpy()];optimizer.zero_grad(set_to_none=True)
            candidate=scores(model,nodes[ids],readout[ids]);target=y[ids,None].expand_as(candidate)
            losses_by_path=F.binary_cross_entropy_with_logits(candidate,target,reduction='none')
            if variant=='context_path':
                policy=path_distribution(model,nodes[ids]);loss=(policy*losses_by_path).sum(1)
                loss=loss+0.001*(policy*policy.clamp_min(1e-8).log()).sum(1)
            else:loss=losses_by_path[:,0]
            loss=(loss*weights[ix]).mean();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        p,_,_=predict(model,nodes[val],readout[val]);m=metrics(y[val].numpy(),p,meta.person_id.to_numpy()[val]);key=(m['person_auroc'],-m['bce'])
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**m))
        if key>best:
            best=key;best_epoch=epoch+1;torch.save(model.state_dict(),folder/'model.pt')
    model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cpu'))
    p,r,all_p=predict(model,nodes[val],readout[val]);validation=meta.iloc[val].copy();validation['probability']=p
    if r is not None:
        for j in range(3):validation[f'route_{j+1}']=r[:,j]
    constant=int(np.argmax([metrics(y[val].numpy(),all_p[:,j],meta.person_id.to_numpy()[val])['person_auroc'] for j in range(6)]))
    if variant!='context_plain':validation['constant_path_probability']=all_p[:,constant]
    validation.to_csv(folder/'validation.csv',index=False)
    result=dict(source_checkpoint=source,variant=variant,seed=seed,best_epoch=best_epoch,history=history,
        validation=metrics(y[val].numpy(),p,meta.person_id.to_numpy()[val]),fit_seconds=time.time()-started,
        active_trainable_parameters=sum(p.numel() for p in parameters),constant_path_index=constant,
        training='fixed stage-one computational weights; context readout and exact expected path risk',
        path_orders=PATHS.tolist(),entropy_weight=.001)
    (folder/'fit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'context_fit':str(folder.relative_to(ROOT)),'validation':result['validation']}),flush=True)


def freeze_and_evaluate(config):
    # One architecture is selected for BOTH datasets, using validation scores only.
    values={variant:[] for variant in ['path_policy']+VARIANTS}
    for ds in config['datasets']:
        for k in config['folds']:
            for variant in values:
                for p in sorted((ROOT/f'results/hne_routed/{ds}/{variant}/fold{k}').glob('seed*')):
                    record=json.loads((p/('result.json' if variant=='path_policy' else 'fit.json')).read_text())
                    validation=pd.read_csv(p/'validation.csv')
                    pairs=float(validation.y.sum()*(len(validation)-validation.y.sum()))
                    values[variant].append({'dataset':ds,'auroc':record['validation']['auroc'],
                        'pairs':pairs,'person_auroc':record['validation']['person_auroc']})
    per_dataset={name:{ds:sum(r['auroc']*r['pairs'] for r in rows if r['dataset']==ds)/sum(r['pairs'] for r in rows if r['dataset']==ds)
        for ds in config['datasets']} for name,rows in values.items()}
    means={name:float(np.mean(list(v.values()))) for name,v in per_dataset.items()}
    primary=max(['path_policy','context_path'],key=means.get)
    selection={'criterion':'primary-metric validation AUROC: pair-weighted within each dataset, then equal-weight mean of both datasets',
               'primary_variant':primary,'validation_means':means,'validation_by_dataset':per_dataset,
               'participant_balanced_validation_means':{name:float(np.mean([r['person_auroc'] for r in rows])) for name,rows in values.items()},
               'selection_counts':{k:len(v) for k,v in values.items()}}
    (ROOT/'results/hne_routed/architecture_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
    print(json.dumps(selection),flush=True)
    for ds in config['datasets']:
        meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
        for k in config['folds']:
            held=np.flatnonzero(meta.fold.eq(k))
            for variant in VARIANTS:
                destination=ROOT/f'results/hne_routed/{ds}/{variant}/fold{k}';folders=sorted(destination.glob('seed*'))
                for folder in folders:
                    result=json.loads((folder/'fit.json').read_text());cache=ROOT/f'results/hne_routed/{ds}/context_cache/fold{k}'/folder.name/'states.npz'
                    arrays=np.load(cache);model=HNEContextPaths(config,variant)
                    model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cpu'))
                    p,r,all_p=predict(model,torch.tensor(arrays['nodes'][held]),torch.tensor(arrays['readout'][held]))
                    frame=meta.iloc[held].copy();frame['probability']=p
                    if variant!='context_plain':frame['constant_path_probability']=all_p[:,result['constant_path_index']]
                    if r is not None:
                        for j in range(3):frame[f'route_{j+1}']=r[:,j]
                    frame.to_csv(folder/'held_person.csv',index=False)
                    result['held_person']=metrics(frame.y.to_numpy(),p,frame.person_id.to_numpy())
                    (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
                for part in ['validation','held_person']:
                    frames=[pd.read_csv(folder/f'{part}.csv',dtype={'person_id':str}) for folder in folders];frame=frames[0].copy()
                    for col in ['probability','constant_path_probability']:
                        if col in frame:frame[col]=np.mean([f[col] for f in frames],axis=0)
                    frame.to_csv(destination/f'{part}.csv',index=False)
                (destination/'selection.json').write_text(json.dumps({'config':config,'runs':[str(f.relative_to(ROOT)) for f in folders]},indent=2))


def run(args):
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    for ds in config['datasets']:
        for k in config['folds']:
            selected=json.loads((ROOT/f'results/hne_routed/{ds}/learned/fold{k}/selection.json').read_text())
            for source in selected['runs']:
                cache=prepare(ds,k,source,config)
                for variant in VARIANTS:train(ds,k,source,cache,variant,config)
    freeze_and_evaluate(config)


if __name__=='__main__':
    p=argparse.ArgumentParser();run(p.parse_args())
