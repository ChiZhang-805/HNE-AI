"""Fit the HNE recurrent evidence reader and transfer each frozen fit to the other cohort."""
import argparse
import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_evidence_recursion import EvidenceRecursion
from src.hne_refined import StructuredEvidenceBasis
from src.hne_additive import person_weights

OUT = ROOT/'results/evidence_recursion'
KEY = ['person_id','origin_day','fold','y']


def pool_prefix(folder, masks, part):
    hidden = np.load(folder/f'{part}_prefix.npy', mmap_mode='r')
    valid = np.load(masks/f'{part}_mask.npy')[:, :, :-2]
    pooled = []
    for first in range(0,len(hidden),64):
        x = hidden[first:first+64, :, :-2].astype(np.float32)
        m = valid[first:first+64].astype(np.float32)
        streams = []
        for channels in (slice(None,-1),slice(-1,None)):
            weight = m[:,channels]
            count = weight.sum((1,2))
            streams.append((x[:,channels]*weight[...,None]).sum((1,2))/np.maximum(count[:,None],1))
        pooled.append(np.stack(streams,1))
    return np.concatenate(pooled)


def cohort(name, device):
    root = ROOT/'data/hne_routed'/name
    raw = np.load(root/'arrays.npz')
    prefix = ROOT/'data/hne_latent'/name
    values = pool_prefix(prefix,root,'history') - pool_prefix(prefix,root,'reference')[raw['person_index']]
    meta = pd.read_csv(root/'samples.csv', dtype={'person_id':str})
    numerical = ROOT/'data/features/unified_daily'
    key = pd.read_csv(numerical/f'{name}.csv', dtype={'person_id':str})[KEY]
    if not meta[KEY].equals(key) or not np.array_equal(raw['fold'],meta.fold):
        raise ValueError(f'{name}: participant or forecast origin mismatch')
    packed = np.load(numerical/f'{name}.npz')
    return dict(name=name, meta=meta, prefix=torch.as_tensor(values,device=device),
                y=torch.as_tensor(meta.y.to_numpy(copy=True),device=device,dtype=torch.float32),
                x=packed['pooled'], groups=packed['groups'])


def tensors(processor, data, device):
    return [torch.as_tensor(x,device=device) for x in processor.transform(data['x'])]


def predict(model, data, features, indices, variant, omit=None):
    model.eval()
    with torch.no_grad():
        scores=model([x[indices] for x in features],data['prefix'][indices],variant,omit)
    return scores[:,-1].sigmoid().cpu().numpy()


def fit(config, source, target, fold, seed, variant, processor, features, target_features, device, anchor):
    meta=source['meta']
    train=np.flatnonzero(~meta.fold.isin([fold,(fold+1)%5]))
    val=np.flatnonzero(meta.fold==(fold+1)%5)
    test=np.flatnonzero(meta.fold==fold)
    if not set(meta.iloc[train].person_id).isdisjoint(meta.iloc[test].person_id):
        raise ValueError('Participant overlap')
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    residual_scale=config['residual_logit_scale'] if anchor is not None else None
    model=EvidenceRecursion(processor.widths,width=config['width'],depth=config['depth'],residual_scale=residual_scale).to(device)
    if anchor is not None:
        model.anchor_weights.copy_(torch.as_tensor(anchor[0],device=device))
        model.anchor_bias.fill_(anchor[1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    frequency=meta.iloc[train].person_id.value_counts()
    weights=(1/meta.person_id.map(frequency)).to_numpy()
    weights=torch.tensor(weights/np.nanmean(weights[train]),dtype=torch.float32,device=device)
    root=OUT/'anchored' if anchor is not None else OUT
    folder=root/source['name']/variant/f'fold{fold}'/f'seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    best=(-np.inf,-np.inf)
    history=[]
    if anchor is not None:
        p=predict(model,source,features,val,variant)
        y=meta.y.to_numpy()[val]
        best=(roc_auc_score(y,p),-log_loss(y,p,labels=[0,1]))
        best_epoch=0
        checkpoint=copy.deepcopy(model.state_dict())
        history.append((0,np.nan,best[0],-best[1]))
    for epoch in range(1,config['max_epochs']+1):
        model.train()
        objective=0.
        for ix in np.array_split(rng.permutation(train),int(np.ceil(len(train)/config['batch_size']))):
            optimizer.zero_grad(set_to_none=True)
            logits=model([x[ix] for x in features],source['prefix'][ix],variant)
            labels=source['y'][ix,None].expand_as(logits)
            loss=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
            per_row=loss[:,-1]
            if loss.shape[1]>1:
                w=config['intermediate_loss_weight']
                per_row=(per_row+w*loss[:,:-1].mean(1))/(1+w)
            value=(weights[ix]*per_row).mean()
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            objective+=float(value.detach())*len(ix)
        p=predict(model,source,features,val,variant)
        y=meta.y.to_numpy()[val]
        score=(roc_auc_score(y,p),-log_loss(y,p,labels=[0,1]))
        history.append((epoch,objective/len(train),score[0],-score[1]))
        if score>best:
            best=score
            best_epoch=epoch
            checkpoint=copy.deepcopy(model.state_dict())
        elif epoch-best_epoch>=config['patience']:
            break
    model.load_state_dict(checkpoint)
    torch.save({'state':model.cpu().state_dict(),'widths':processor.widths,'width':config['width'],
                'depth':config['depth'],'variant':variant,'epoch':best_epoch,'residual_scale':residual_scale},folder/'model.pt')
    model.to(device)
    pd.DataFrame(history,columns=['epoch','train_loss','validation_auroc','validation_log_loss']).to_csv(folder/'curves.csv',index=False)
    for name,data,basis,ix in [('held',source,features,test),
                                ('transfer',target,target_features,np.arange(len(target['meta'])))]:
        frame=data['meta'].iloc[ix][KEY].copy()
        frame['risk']=predict(model,data,basis,ix,variant)
        if anchor is not None and variant=='feedback':
            frame['anchor']=model.anchor([f[ix] for f in basis]).sigmoid().detach().cpu().numpy()
        if name=='held' and variant=='feedback':
            for j,letter in enumerate('HNE'):
                frame[f'omit_{letter}']=predict(model,data,basis,ix,variant,omit=j)
        frame.to_csv(folder/f'{name}.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--anchored',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    cfg=json.loads((ROOT/'configs/hne_evidence_recursion.json').read_text())
    cohorts=[cohort(name,args.device) for name in cfg['datasets']]
    for source,target in (cohorts,cohorts[::-1]):
        for fold in range(5):
            train=~source['meta'].fold.isin([fold,(fold+1)%5])
            processor=StructuredEvidenceBasis().fit(source['x'][train],source['groups'])
            if not np.array_equal(source['groups'],target['groups']):
                raise ValueError('Transfer evidence coordinates differ')
            folder=OUT/source['name']/f'fold{fold}'
            folder.mkdir(parents=True,exist_ok=True)
            if not args.anchored or not (folder/'basis.joblib').exists():
                joblib.dump(processor,folder/'basis.joblib')
            features=tensors(processor,source,args.device)
            foreign=tensors(processor,target,args.device)
            anchor=None
            if args.anchored:
                y=source['meta'].loc[train,'y'].to_numpy()
                weights,pos_multiplier=person_weights(source['meta'].loc[train,'person_id'],y)
                rule=LogisticRegression(C=cfg['anchor_C'],max_iter=2000)
                rule.fit(np.concatenate(processor.transform(source['x'][train]),axis=1),y,sample_weight=weights)
                anchor=(rule.coef_[0],rule.intercept_[0]-np.log(pos_multiplier))
            for variant in cfg['variants']:
                for seed in cfg['seeds']:
                    fit(cfg,source,target,fold,seed,variant,processor,features,foreign,args.device,anchor)
            print(f"{source['name']} fold {fold} completed",flush=True)


if __name__=='__main__':
    main()
