"""Temporal-order interventions with distributions and endpoint labels fixed."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import make_hne,RoutedData
from run_hne_routed import predict,metrics


def prepare(dataset):
    folder=ROOT/f'data/hne_routed/{dataset}'
    if all((folder/f'{kind}_mask.npy').exists() for kind in ['within_day_shuffle','day_order_shuffle']):return
    from chronos import Chronos2Pipeline
    a=np.load(folder/'arrays.npz');original=a['history']
    model=Chronos2Pipeline.from_pretrained(str(ROOT/'tools/models/chronos-2'),device_map='cuda').model.eval()
    saved={}
    def hook(module,args,kwargs):
        saved['x']=args[0].detach();saved['mask']=kwargs['attention_mask'][:,0,0,:]==0
    handle=model.encoder.block[-1].register_forward_pre_hook(hook,with_kwargs=True)
    for kind in ['within_day_shuffle','day_order_shuffle']:
        if (folder/f'{kind}_mask.npy').exists():continue
        rng=np.random.default_rng(20261113);x=original.copy().reshape(-1,6,32,8)
        for i in range(len(x)):
            if kind=='within_day_shuffle':
                for day in range(32):x[i,:,day,:]=x[i,:,day,:][:,rng.permutation(8)]
            else:x[i]=x[i][:,rng.permutation(32),:]
        x=x.reshape(original.shape)
        cache=np.lib.format.open_memmap(folder/f'{kind}_prefix.npy',mode='w+',dtype=np.float16,shape=(len(x),6,18,768));masks=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(x),16):
                context=torch.tensor(x[start:start+16].reshape(-1,256),device='cuda')
                model.encode(context,group_ids=torch.arange(len(context),device='cuda'))
                cache[start:start+16]=saved['x'].float().cpu().numpy().reshape(-1,6,18,768)
                masks.append(saved['mask'].cpu().numpy().reshape(-1,6,18))
        cache.flush();np.save(folder/f'{kind}_mask.npy',np.concatenate(masks));print(dataset,kind,'encoded',flush=True)
    handle.remove();del model;torch.cuda.empty_cache()


def evaluate(dataset,variants):
    config=json.loads((ROOT/'configs/hne_routed.json').read_text());meta=pd.read_csv(ROOT/f'data/hne_routed/{dataset}/samples.csv',dtype={'person_id':str})
    output=ROOT/f'results/hne_routed/temporal/{dataset}';output.mkdir(parents=True,exist_ok=True);rows=[]
    for variant in variants:
        for kind in ['original','within_day_shuffle','day_order_shuffle']:
            data=RoutedData(dataset)
            if kind!='original':
                data.history=np.load(ROOT/f'data/hne_routed/{dataset}/{kind}_prefix.npy',mmap_mode='r')
                data.mask=np.load(ROOT/f'data/hne_routed/{dataset}/{kind}_mask.npy')
            frames=[]
            for k in range(5):
                selection=json.loads((ROOT/f'results/hne_routed/{dataset}/{variant}/fold{k}/selection.json').read_text())
                ids=np.flatnonzero(data.arrays['fold']==k);predictions=[]
                for source in selection['runs']:
                    model=make_hne(config,variant).cuda();model.load_state_dict(torch.load(ROOT/source/'model.pt',weights_only=True,map_location='cuda'))
                    p,_,_=predict(model,data,ids,config['batch_size']);predictions.append(p)
                    del model;torch.cuda.empty_cache()
                frame=meta.iloc[ids].copy();frame['probability']=np.mean(predictions,axis=0);frames.append(frame)
                rows.append({'method':variant,'condition':kind,'fold':k,**metrics(frame.y.to_numpy(),frame.probability.to_numpy(),frame.person_id.to_numpy())})
            pd.concat(frames).to_csv(output/f'{variant}_{kind}.csv',index=False)
    destination=output/'fold_metrics.csv';table=pd.DataFrame(rows)
    if destination.exists():table=pd.concat([pd.read_csv(destination),table]).drop_duplicates(['method','condition','fold'],keep='last')
    table.to_csv(destination,index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--variants',nargs='+',default=['path_policy','context_path','context_plain','context_free','context_free_plain']);args=p.parse_args()
    torch.set_num_threads(2)
    for ds in ['RESILIENT','BrisT1D-Open']:
        prepare(ds)
        if not args.prepare_only:evaluate(ds,args.variants)
