"""Participant-separated main and RQ experiments for jointly trained HNE."""
from pathlib import Path
import argparse,json,sys,time,random,gc
import joblib
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.hne_routed import RoutedData
from src.hne_joint import HNEJoint,EvidenceBasis,joint_objective
from run_hne_routed import person_weights,metrics


class JointData(RoutedData):
    """Keep frozen prefix tokens on the GPU throughout the actual fits."""
    def __init__(self,dataset):
        super().__init__(dataset)
        self.cached_history=torch.tensor(np.asarray(self.history),device='cuda')
        self.cached_reference=torch.tensor(np.asarray(self.reference),device='cuda')
        self.cached_mask=torch.tensor(self.mask,device='cuda')
        self.cached_rmask=torch.tensor(self.rmask,device='cuda')
        self.cached_people=torch.tensor(self.arrays['person_index'],device='cuda',dtype=torch.long)
        self.cached_stats=torch.tensor(self.arrays['stats'],device='cuda')
    def batch(self,ids,device='cuda'):
        index=torch.as_tensor(ids,device='cuda',dtype=torch.long);people=self.cached_people[index]
        return [self.cached_history[index].float(),self.cached_mask[index],
            self.cached_reference[people].float(),self.cached_rmask[people],self.cached_stats[index]]


def basis_batch(basis,ids):
    return [torch.as_tensor(x[ids],device='cuda') for x in basis]


def forward_batch(model,data,basis,ids):
    evidence=basis_batch(basis,ids)
    if model.variant=='coordinates_only':
        return model(torch.empty(len(ids),device='cuda'),None,None,None,None,evidence)
    return model(*data.batch(ids),evidence)


def predict(model,data,basis,ids,batch_size):
    model.eval();all_p=[];selected=[];policies=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(ids),batch_size):
            ix=ids[start:start+batch_size]
            out=forward_batch(model,data,basis,ix)
            all_p.append(out['all_logits'].float().sigmoid().cpu().numpy())
            selected.append(out['selected'].cpu().numpy())
            policies.append(out['policy'].float().cpu().numpy())
    probabilities=np.concatenate(all_p);choice=np.concatenate(selected)
    return probabilities[np.arange(len(ids)),choice],probabilities,choice,np.concatenate(policies)


def fit(data,meta,basis,processor,train,val,test,config,variant,seed,folder):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=HNEJoint(config,processor.widths,variant).cuda()
    if variant=='coordinates_only':
        for module in [model.tail,model.prompt_up,model.readout]:
            for p in module.parameters():p.requires_grad_(False)
        model.prompt_types.requires_grad_(False)
    if variant=='fixed':
        for p in model.router.parameters():p.requires_grad_(False)
    trainable=[p for p in model.parameters() if p.requires_grad]
    groups={'main':[],'lora':[],'evidence':[]}
    for name,p in model.named_parameters():
        if p.requires_grad:groups['lora' if name.startswith('tail.') else 'evidence' if name.startswith('evidence_coefficients.') else 'main'].append(p)
    optimizer=torch.optim.AdamW([
        {'params':groups['main'],'lr':config['learning_rate']},
        {'params':groups['lora'],'lr':config['lora_learning_rate']},
        {'params':groups['evidence'],'lr':config['evidence_learning_rate'],'weight_decay':0.}],weight_decay=config['weight_decay'])
    y=data.arrays['y'];persons=meta.person_id.to_numpy();person_code=pd.factorize(persons)[0]
    weights=person_weights(persons[train]);prevalence=np.average(y[train],weights=weights)
    weights*=np.where(y[train]>0,.5/prevalence,.5/(1-prevalence));weights/=weights.mean()
    best=(-np.inf,-np.inf);history=[];started=time.perf_counter()
    for epoch in range(config['epochs']):
        model.train();order=np.random.permutation(len(train));losses=[]
        for offset in range(0,len(order),config['batch_size']):
            local=order[offset:offset+config['batch_size']];ix=train[local]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=forward_batch(model,data,basis,ix)
                loss=joint_objective(model,output,torch.tensor(y[ix],device='cuda'),
                    torch.tensor(weights[local],device='cuda',dtype=torch.float32),
                    torch.tensor(person_code[ix],device='cuda'),
                    torch.tensor(data.arrays['future_targets'][ix],device='cuda'),config)
            loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step()
            losses.append(loss.detach())
        vp,_,_,_=predict(model,data,basis,val,config['batch_size'])
        m=metrics(y[val],vp,persons[val]);key=(m['auroc'],-m['bce'])
        record={'epoch':epoch+1,'loss':float(torch.stack(losses).mean()),**m,'seconds':time.perf_counter()-started};history.append(record)
        if key>best:
            best=key;best_epoch=epoch+1;torch.save(model.state_dict(),folder/'model.pt')
        if epoch%5==0 or epoch==config['epochs']-1:
            print(json.dumps({'run':str(folder.relative_to(ROOT)),**record}),flush=True)
    model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cuda'))
    result={'variant':variant,'seed':seed,'best_epoch':best_epoch,'epochs':config['epochs'],
        'trainable_parameters':sum(p.numel() for p in trainable),'fit_seconds':time.perf_counter()-started,'history':history}
    for part,ids in [('validation',val),('held_person',test)]:
        p,all_p,chosen,policies=predict(model,data,basis,ids,config['batch_size'])
        frame=meta.iloc[ids].copy();frame['probability']=p;frame['selected_path']=chosen
        for j in range(6):frame[f'path_{j}']=all_p[:,j];frame[f'policy_{j}']=policies[:,j]
        frame['uniform_probability']=all_p.mean(1);frame.to_csv(folder/f'{part}.csv',index=False)
        result[part]=metrics(y[ids],p,persons[ids])
        result[part]['mean_path_span']=float(np.mean(all_p.max(1)-all_p.min(1)))
        result[part]['path_counts']=np.bincount(chosen,minlength=6).tolist()
    result['lora_max_changes_from_zero']={n:float(p.detach().abs().max()) for n,p in model.named_parameters() if n.startswith('tail.') and n.endswith('.b')}
    result['evidence_l2_norms']=[float(v.detach().norm()) for v in model.evidence_coefficients]
    result['gate_weight_l2_norms']=[float(g.weight.detach().norm()) for g in model.gates]
    (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'complete':str(folder.relative_to(ROOT)),'validation':result['validation'],'held_person':result['held_person'],'seconds':result['fit_seconds']}),flush=True)
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return result


def ensemble(dest,paths,config):
    frames={}
    columns=['probability','uniform_probability']+[f'{key}_{j}' for key in ['path','policy'] for j in range(6)]
    for part in ['validation','held_person']:
        parts=[pd.read_csv(p/f'{part}.csv',dtype={'person_id':str}) for p in paths]
        frame=parts[0].drop(columns='selected_path').copy()
        for column in columns:frame[column]=np.mean([f[column].to_numpy() for f in parts],axis=0)
        frames[part]=frame
    val=frames['validation'];constant=max(range(6),key=lambda j:metrics(val.y,val[f'path_{j}'],val.person_id)['auroc'])
    for part,frame in frames.items():
        frame['constant_probability']=frame[f'path_{constant}'];frame.to_csv(dest/f'{part}.csv',index=False)
    (dest/'selection.json').write_text(json.dumps({'config':config,'constant_path':constant,'runs':[str(p.relative_to(ROOT)) for p in paths]},indent=2),encoding='utf8')


def run(args):
    config=json.loads((ROOT/'configs/hne_joint.json').read_text(encoding='utf8'));torch.set_num_threads(2)
    for ds in args.datasets:
        data=JointData(ds);meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
        numerical=np.load(ROOT/f'data/features/unified_daily/{ds}.npz')
        fold=data.arrays['fold'];x=numerical['pooled'];groups=numerical['groups']
        for k in args.folds:
            train=np.flatnonzero(~np.isin(fold,[k,(k+1)%5]));val=np.flatnonzero(fold==(k+1)%5);test=np.flatnonzero(fold==k)
            processor=EvidenceBasis().fit(x[train],groups);basis=processor.transform(x)
            cache=ROOT/f'data/hne_joint/{ds}';cache.mkdir(parents=True,exist_ok=True)
            joblib.dump(processor,cache/f'fold{k}_basis.joblib')
            for variant in args.variants:
                dest=ROOT/'results/hne_joint'/ds/variant/f'fold{k}';paths=[]
                for seed in config['seeds'][:args.seeds]:
                    folder=dest/f'seed{seed}';fit(data,meta,basis,processor,train,val,test,config,variant,seed,folder);paths.append(folder)
                ensemble(dest,paths,config)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['RESILIENT','BrisT1D-Open'])
    p.add_argument('--variants',nargs='+',default=['joint','fixed','coordinates_only','no_coordinates'])
    p.add_argument('--folds',nargs='+',type=int,default=list(range(5)));p.add_argument('--seeds',type=int,default=3)
    run(p.parse_args())
