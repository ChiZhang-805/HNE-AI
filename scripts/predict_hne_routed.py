"""Run an HNE checkpoint from raw hourly history and personal reference.

Input NPZ: history [768,C], reference [1344,C], last channel is observed context.
No future observations or labels are accepted by this inference entry point.
"""
from pathlib import Path
import argparse,json,sys,warnings
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.daily_protocol import robust_reference,normalize
from src.hne_routed import make_hne


def inference(history,reference,checkpoint,config,variant='path_policy'):
    from chronos import Chronos2Pipeline
    channels=history.shape[-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        h=np.nanmean(history.reshape(-1,3,channels),axis=1)
        r=np.nanmean(reference.reshape(-1,3,channels),axis=1)
        center,scale,_=robust_reference(r,len(r),32)
        h=np.clip(normalize(h,center,scale),-12,12).T.astype(np.float32)
        r=np.clip(normalize(r,center,scale),-12,12).T.astype(np.float32)
        stats=np.stack([np.nanmean(h,axis=1),np.log1p(np.nanstd(h,axis=1)),np.isfinite(h).mean(1)],axis=-1)
    backbone=Chronos2Pipeline.from_pretrained(str(ROOT/'tools/models/chronos-2'),device_map='cuda').model.eval()
    saved={}
    def hook(module,args,kwargs):
        saved['x']=args[0].detach().float()
        saved['mask']=kwargs['attention_mask'][:,0,0,:]==0
    handle=backbone.encoder.block[-1].register_forward_pre_hook(hook,with_kwargs=True)
    inputs=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for array in [h,r]:
            backbone.encode(torch.tensor(array,device='cuda'),group_ids=torch.arange(channels,device='cuda'))
            inputs.append((saved['x'].unsqueeze(0),saved['mask'].unsqueeze(0)))
    handle.remove();del backbone;torch.cuda.empty_cache()
    model=make_hne(config,variant).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(inputs[0][0],inputs[0][1],inputs[1][0],inputs[1][1],torch.tensor(np.nan_to_num(stats)[None],device='cuda',dtype=torch.float32))
    return {'risk_score':float(out['logits'].float().sigmoid()[0]),
            'path':['HNE'[i] for i in out['routes'][0].tolist()],
            'future_standardized_mean_and_log_sd':out['future'][0].float().cpu().tolist()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--variant',default=None)
    args=p.parse_args();torch.set_num_threads(2)
    bundle=np.load(args.input);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    if args.variant is None:args.variant=json.loads((ROOT/'results/hne_routed/architecture_selection.json').read_text())['primary_variant']
    print(json.dumps(inference(bundle['history'],bundle['reference'],args.checkpoint,config,args.variant),indent=2))
