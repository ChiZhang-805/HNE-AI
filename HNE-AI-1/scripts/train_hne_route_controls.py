"""Matched-label direct-head and fixed-weight random-path controls."""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import mlp
from run_hne_routed import metrics,person_weights


def run(args):
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text());device=args.device
    for ds in args.datasets:
        meta=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str})
        for k in args.folds:
            selection=json.loads((ROOT/f'results/hne_routed/{ds}/path_policy/fold{k}/selection.json').read_text())
            destination=ROOT/f'results/hne_routed/{ds}/route_controls/fold{k}';folders=[]
            for source in selection['runs']:
                seed=int(Path(source).name.removeprefix('seed'));folder=destination/f'seed{seed}';folders.append(folder)
                if (folder/'result.json').exists():continue
                folder.mkdir(parents=True,exist_ok=True);cache=np.load(ROOT/source/'rollouts.npz')
                frame=meta.iloc[cache['ids']].reset_index(drop=True)
                masks={'train':~frame.fold.isin([k,(k+1)%5]).to_numpy(),
                       'validation':frame.fold.eq((k+1)%5).to_numpy(),'held_person':frame.fold.eq(k).to_numpy()}
                x=torch.tensor(cache['features'][:,0],device=device);y=torch.tensor(frame.y.to_numpy(),device=device,dtype=torch.float32)
                weights=person_weights(frame.loc[masks['train'],'person_id'].to_numpy())
                py=np.average(frame.loc[masks['train'],'y'],weights=weights)
                weights*=np.where(frame.loc[masks['train'],'y']>0,.5/py,.5/(1-py));weights/=weights.mean()
                w=torch.tensor(weights,device=device,dtype=torch.float32)
                ti=torch.tensor(np.flatnonzero(masks['train']),device=device);vi=np.flatnonzero(masks['validation'])
                torch.manual_seed(seed);head=mlp(288,1,96).to(device)
                optimizer=torch.optim.AdamW(head.parameters(),lr=config['path_policy_learning_rate'],weight_decay=config['weight_decay'])
                best=(-np.inf,-np.inf);history=[]
                for epoch in range(config['path_policy_epochs']):
                    head.train();order=torch.randperm(len(ti),device=device);losses=[]
                    for start in range(0,len(ti),config['path_policy_batch_size']):
                        ix=order[start:start+config['path_policy_batch_size']];optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                            logits=head(x[ti[ix]]).squeeze(-1).float()
                            loss=(F.binary_cross_entropy_with_logits(logits,y[ti[ix]],reduction='none')*w[ix]).mean()
                        loss.backward();optimizer.step();losses.append(float(loss.detach()))
                    head.eval()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):p=head(x[vi]).squeeze(-1).float().sigmoid().cpu().numpy()
                    v=frame.iloc[vi];m=metrics(v.y.to_numpy(),p,v.person_id.to_numpy());key=(m['person_auroc'],-m['bce'])
                    history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**m))
                    if key>best:best=key;best_epoch=epoch+1;state=copy.deepcopy(head.state_dict())
                head.load_state_dict(state);head.eval();torch.save(state,folder/'model.pt')
                rng=np.random.default_rng(seed+31);random_path=rng.integers(27,size=len(frame))
                result={'source_checkpoint':source,'best_epoch':best_epoch,'history':history,
                    'trainable_parameters':sum(p.numel() for p in head.parameters()),
                    'training_device':device,'precision':'bfloat16' if device.startswith('cuda') else 'float32',
                    'control':'direct binary head over the same root routing features; source computation weights fixed'}
                for part in ['validation','held_person']:
                    ix=np.flatnonzero(masks[part]);f=frame.iloc[ix].copy()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):p=head(x[ix]).squeeze(-1).float().sigmoid().cpu().numpy()
                    f['probability']=p;f['random_path_probability']=cache['probabilities'][ix,random_path[ix]]
                    f['uniform_path_probability']=cache['probabilities'][ix].mean(1)
                    f.to_csv(folder/f'{part}.csv',index=False)
                    result[part]=metrics(f.y.to_numpy(),p,f.person_id.to_numpy())
                (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
                print(json.dumps({'direct_head':str(folder.relative_to(ROOT)),'held_person':result['held_person']}),flush=True)
                del head,optimizer,x,y,w;torch.cuda.empty_cache()
            for part in ['validation','held_person']:
                frames=[pd.read_csv(f/f'{part}.csv',dtype={'person_id':str}) for f in folders];frame=frames[0].copy()
                for col in ['probability','random_path_probability','uniform_path_probability']:frame[col]=np.mean([f[col] for f in frames],axis=0)
                frame.to_csv(destination/f'{part}.csv',index=False)
            (destination/'selection.json').write_text(json.dumps({'runs':[str(f.relative_to(ROOT)) for f in folders]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['RESILIENT','BrisT1D-Open']);p.add_argument('--folds',type=int,nargs='+',default=list(range(5)));p.add_argument('--device',default='cpu');run(p.parse_args())
