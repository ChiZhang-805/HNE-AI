"""Three-hour histories and frozen Chronos-2 prefix tokens for routed HNE.

Only the first 11 pretrained blocks are cached. The last pretrained block and
the HNE paths are trained together later. Labels retain the existing endpoint.
"""
from pathlib import Path
import sys, json, warnings
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.daily_protocol import robust_reference, normalize


def make_arrays(dataset):
    folder = ROOT / 'data/hne_routed' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / f'data/features/unified_daily/{dataset}.csv', dtype={'person_id':str})
    persons = sorted(meta.person_id.unique())
    history, references, stats, targets, futures, pids = [], [], [], [], [], []
    grids = {}
    for person in persons:
        raw = np.load(ROOT / f'data/development/{dataset}/{person}_hourly.npz')['values']
        raw = raw[:len(raw)//3*3].reshape(-1,3,6)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            grid = np.nanmean(raw,axis=1)
        center, scale, _ = robust_reference(grid, 56*8, 32)
        grid = np.clip(normalize(grid, center, scale), -12, 12).astype(np.float32)
        grids[person] = grid
        references.append(grid[:56*8].T)
    for row in meta.itertuples():
        grid = grids[row.person_id]
        end = (int(row.origin_day)+1)*8
        window = grid[end-32*8:end].T
        history.append(window)
        pids.append(persons.index(row.person_id))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            stats.append(np.stack([np.nanmean(window,axis=1), np.log1p(np.nanstd(window,axis=1)), np.isfinite(window).mean(1)],axis=-1))
            future = grid[(int(row.origin_day)+2)*8:(int(row.origin_day)+9)*8,:5]
            targets.append(np.stack([np.nanmean(future,axis=0), np.log1p(np.nanstd(future,axis=0))],axis=-1))
        f = np.full((256,6),np.nan,np.float32)
        future32 = grid[end:end+256]
        if len(future32)==256: f[:]=future32
        futures.append(f)
    values = dict(history=np.asarray(history,np.float32), reference=np.asarray(references,np.float32),
                  stats=np.nan_to_num(stats).astype(np.float32), future_targets=np.asarray(targets,np.float32),
                  future32=np.asarray(futures,np.float32), person_index=np.asarray(pids), persons=np.asarray(persons),
                  y=meta.y.to_numpy(np.float32), fold=meta.fold.to_numpy(int))
    np.savez_compressed(folder/'arrays.npz',**values)
    meta.to_csv(folder/'samples.csv',index=False)
    print(dataset, len(meta),len(persons),'history',values['history'].shape,flush=True)
    return values, folder


def cache_prefix(values,folder,model):
    captured={}
    def capture(module,args,kwargs):
        captured['hidden']=args[0].detach()
        captured['mask']=(kwargs['attention_mask'][:,0,0,:]==0).detach()
    handle=model.encoder.block[-1].register_forward_pre_hook(capture,with_kwargs=True)
    for name in ['history','reference']:
        output=folder/f'{name}_prefix.npy'
        if output.exists() and (folder/f'{name}_mask.npy').exists(): continue
        x=values[name]
        tokens=x.shape[-1]//16+2
        cache=np.lib.format.open_memmap(output,mode='w+',dtype=np.float16,shape=(len(x),6,tokens,768))
        masks=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(x),16):
                context=torch.tensor(x[start:start+16].reshape(-1,x.shape[-1]),device='cuda')
                model.encode(context,group_ids=torch.arange(len(context),device='cuda'))
                cache[start:start+16]=captured['hidden'].float().cpu().numpy().reshape(-1,6,tokens,768)
                masks.append(captured['mask'].cpu().numpy().reshape(-1,6,tokens))
                if start%256==0: print(folder.name,name,start,'/',len(x),flush=True)
        cache.flush()
        np.save(folder/f'{name}_mask.npy',np.concatenate(masks))
    handle.remove()


if __name__=='__main__':
    from chronos import Chronos2Pipeline
    from huggingface_hub import snapshot_download
    torch.set_num_threads(2)
    config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    model_dir=ROOT/'tools/models/chronos-2'
    if not (model_dir/'model.safetensors').exists():
        snapshot_download(config['backbone'],revision=config['backbone_revision'],local_dir=model_dir,
            cache_dir=ROOT.parent/'Caches/huggingface',allow_patterns=['config.json','model.safetensors','README.md'])
    model=Chronos2Pipeline.from_pretrained(str(model_dir),device_map='cuda').model.eval()
    for p in model.parameters():p.requires_grad_(False)
    torch.save({'config':model.config.to_dict(),'block':model.encoder.block[-1].state_dict(),
                'norm':model.encoder.final_layer_norm.state_dict()},ROOT/'tools/models/chronos-2/tail.pt')
    for ds in ['RESILIENT','BrisT1D-Open']:
        arrays,folder=make_arrays(ds)
        cache_prefix(arrays,folder,model)
