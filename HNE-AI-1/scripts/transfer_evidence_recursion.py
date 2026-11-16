"""Evaluate frozen source models on public College Experience behavioral sensing."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from run_evidence_recursion import ROOT, OUT, KEY, EvidenceRecursion


def encode(histories, states, backbone, captured, device):
    channels, length=histories.shape[1:]
    pooled=[]
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,enabled=device.type=='cuda'):
        for start in range(0,len(histories),24):
            values=torch.as_tensor(histories[start:start+24],device=device)
            groups=torch.arange(len(values),device=device).repeat_interleave(channels)
            backbone.encode(values.reshape(-1,length),group_ids=groups)
            hidden=captured['hidden'].reshape(len(values),channels,-1,768)[:,:,:-2].float()
            valid=captured['mask'].reshape(len(values),channels,-1)[:,:,:-2]
            streams=[]
            for subset in (slice(None,states),slice(states,None)):
                mask=valid[:,subset]
                streams.append((hidden[:,subset]*mask[...,None]).sum((1,2))/mask.sum((1,2)).clamp_min(1)[:,None])
            pooled.append(torch.stack(streams,1).cpu())
    return torch.cat(pooled)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--anchored',action='store_true')
    args=parser.parse_args()
    device=torch.device(args.device)
    output=OUT/'anchored' if args.anchored else OUT
    torch.set_num_threads(2)
    dataset='College-Behavior'
    folder=ROOT/'data/hne_external_refined'/dataset
    data=np.load(folder/'arrays.npz')
    mapping=json.loads((folder/'mapping.json').read_text())
    spec=json.loads((ROOT/'configs/hne_evidence_recursion.json').read_text())
    frame=pd.read_csv(folder/'samples.csv',dtype={'person_id':str})[KEY].copy()
    raw=ROOT/'data/features/college_experience'
    native=np.load(raw/'test.npz')
    source_groups=np.load(ROOT/f"data/features/unified_daily/{spec['datasets'][0]}.npz")['groups']
    if not np.array_equal(native['groups'],source_groups) or not frame.equals(pd.read_csv(raw/'test.csv',dtype={'person_id':str})[KEY]):
        raise ValueError('External coordinates or origin IDs do not match')
    backbone=Chronos2Pipeline.from_pretrained(str(ROOT/'tools/models/chronos-2'),device_map=str(device)).model.eval()
    captured={}
    def capture(module,args,kwargs):
        captured['hidden']=args[0].detach()
        captured['mask']=kwargs['attention_mask'][:,0,0,:]==0
    hook=backbone.encoder.block[-2].register_forward_pre_hook(capture,with_kwargs=True)
    reference=encode(data['reference'],mapping['state_channels'],backbone,captured,device)
    history=encode(data['history'],mapping['state_channels'],backbone,captured,device)
    prefix=(history-reference[data['person_index']]).to(device)
    hook.remove()
    del captured, backbone
    x=data['pooled']
    with torch.no_grad():
        for source in spec['datasets']:
            for fold in range(5):
                proc=joblib.load(OUT/source/f'fold{fold}'/'basis.joblib')
                basis=[torch.from_numpy(a).to(device) for a in proc.transform(x)]
                for variant in ('feedback','fixed_query','one_step'):
                    probabilities=[]
                    for seed in spec['seeds']:
                        ck=torch.load(output/source/variant/f'fold{fold}'/f'seed{seed}'/'model.pt',map_location=device,weights_only=True)
                        model=EvidenceRecursion(ck['widths'],width=ck['width'],depth=ck['depth'],residual_scale=ck.get('residual_scale')).to(device)
                        model.load_state_dict(ck['state'])
                        model.eval()
                        parts=[]
                        for i in range(0,len(x),1024):
                            logits=model([f[i:i+1024] for f in basis],prefix[i:i+1024],variant)
                            parts.append(logits[:,-1].sigmoid().cpu().numpy())
                        probabilities.append(np.concatenate(parts))
                    frame[f'{source}/{variant}/fold{fold}']=np.mean(probabilities,axis=0)
    frame.to_csv(output/'college_transfer.csv.gz',index=False)
    print(dataset,len(frame),frame.person_id.nunique(),int(frame.y.sum()),flush=True)


if __name__=='__main__':
    main()
