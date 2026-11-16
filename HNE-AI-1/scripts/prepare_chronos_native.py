"""Original multivariate grouping for the direct Chronos-2 fine-tuning baseline."""
from pathlib import Path
import numpy as np
import torch
from chronos import Chronos2Pipeline

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
model=Chronos2Pipeline.from_pretrained(str(ROOT/'tools/models/chronos-2'),device_map='cuda').model.eval()
saved={}
def hook(module,args,kwargs):saved['x']=args[0].detach()
handle=model.encoder.block[-1].register_forward_pre_hook(hook,with_kwargs=True)
for dataset in ['RESILIENT','BrisT1D-Open']:
    folder=ROOT/'data/hne_routed'/dataset
    a=np.load(folder/'arrays.npz');x=a['history']
    out=folder/'history_native_prefix.npy'
    if out.exists():continue
    cache=np.lib.format.open_memmap(out,mode='w+',dtype=np.float16,shape=(len(x),6,18,768))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(x),16):
            v=torch.tensor(x[start:start+16].reshape(-1,256),device='cuda')
            group=torch.arange(len(v)//6,device='cuda').repeat_interleave(6)
            model.encode(v,group_ids=group)
            cache[start:start+16]=saved['x'].float().cpu().numpy().reshape(-1,6,18,768)
    cache.flush();print(dataset,'native prefix complete',flush=True)
handle.remove()
