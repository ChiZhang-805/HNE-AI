"""Bidirectional zero-shot transfer with the exact same HNE model weights."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_routed import make_hne,RoutedData
from run_hne_routed import predict


def run(args):
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text());metrics=[]
    for source,target in [('RESILIENT','BrisT1D-Open'),('BrisT1D-Open','RESILIENT')]:
        data=RoutedData(target);meta=pd.read_csv(ROOT/f'data/hne_routed/{target}/samples.csv',dtype={'person_id':str});ids=np.arange(len(meta))
        for variant in args.variants:
            data=RoutedData(target,native=variant=='native')
            predictions=[]
            for k in range(5):
                f=ROOT/f'results/hne_routed/{source}/{variant}/fold{k}/selection.json'
                selection=json.loads(f.read_text())
                for checkpoint in selection['runs']:
                    model=make_hne(config,variant).cuda()
                    model.load_state_dict(torch.load(ROOT/checkpoint/'model.pt',weights_only=True,map_location='cuda'))
                    p,r,_=predict(model,data,ids,config['batch_size']);predictions.append(p)
                    del model;torch.cuda.empty_cache()
            frame=meta.copy();frame['probability']=np.mean(predictions,axis=0)
            folder=ROOT/'results/hne_routed/transfer'/f'{source}_to_{target}'/variant;folder.mkdir(parents=True,exist_ok=True)
            frame.to_csv(folder/'predictions.csv',index=False)
            metrics.append(dict(source=source,target=target,method=variant,source_models=len(predictions),
                                target_training_labels=0,n=len(frame),auroc=float(roc_auc_score(frame.y,frame.probability)),
                                ap=float(average_precision_score(frame.y,frame.probability))))
            print(json.dumps(metrics[-1]),flush=True)
    destination=ROOT/'results/hne_routed/transfer/metrics.csv';table=pd.DataFrame(metrics)
    if destination.exists():table=pd.concat([pd.read_csv(destination),table]).drop_duplicates(['source','target','method'],keep='last')
    table.to_csv(destination,index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',default=['path_policy','learned','context_path','context_fixed','context_plain','native','context_free','context_free_fixed','context_free_plain'])
    run(p.parse_args())
