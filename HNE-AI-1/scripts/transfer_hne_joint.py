"""Freeze source weights and numerical transforms for bidirectional transfer."""
from pathlib import Path
import argparse,json,sys,gc
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.hne_joint import HNEJoint
from src.hne_routed import RoutedData
from run_hne_joint import predict,JointData
from analyze_hne_routed import clustered_auc_bootstrap


def run(args):
    config=json.loads((ROOT/'configs/hne_joint.json').read_text(encoding='utf8'));torch.set_num_threads(2)
    out=ROOT/'results/hne_joint/transfer';out.mkdir(parents=True,exist_ok=True);metrics=[]
    for source in args.sources:
        target='BrisT1D-Open' if source=='RESILIENT' else 'RESILIENT'
        data=JointData(target);meta=pd.read_csv(ROOT/f'data/hne_routed/{target}/samples.csv',dtype={'person_id':str})
        x=np.load(ROOT/f'data/features/unified_daily/{target}.npz')['pooled'];ids=np.arange(len(meta));comparison=meta.copy()
        for variant in args.variants:
            folder=out/f'{source}_to_{target}'/variant;folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'predictions.csv').exists():
                predictions=[];uniform=[];constants=[]
                for k in range(5):
                    processor=joblib.load(ROOT/f'data/hne_joint/{source}/fold{k}_basis.joblib');basis=processor.transform(x)
                    selected=json.loads((ROOT/f'results/hne_joint/{source}/{variant}/fold{k}/selection.json').read_text(encoding='utf8'))
                    for checkpoint in selected['runs']:
                        model=HNEJoint(config,processor.widths,variant).cuda().eval()
                        model.load_state_dict(torch.load(ROOT/checkpoint/'model.pt',map_location='cuda',weights_only=True))
                        p,all_p,choice,policy=predict(model,data,basis,ids,config['batch_size'])
                        predictions.append(p);uniform.append(all_p.mean(1));constants.append(all_p[:,selected['constant_path']])
                        del model;gc.collect();torch.cuda.empty_cache()
                frame=meta.copy();frame['probability']=np.mean(predictions,0)
                frame['uniform_probability']=np.mean(uniform,0);frame['constant_probability']=np.mean(constants,0)
                frame.to_csv(folder/'predictions.csv',index=False)
            frame=pd.read_csv(folder/'predictions.csv',dtype={'person_id':str});comparison[variant]=frame.probability
            names={variant:'probability'}
            if variant=='joint':names.update(best_constant='constant_probability',uniform='uniform_probability')
            for method,column in names.items():
                comparison[method]=frame[column]
                metrics.append(dict(source=source,target=target,method=method,source_models=15,target_training_labels=0,
                    n=len(frame),auroc=roc_auc_score(frame.y,frame[column]),ap=average_precision_score(frame.y,frame[column])))
                print(json.dumps(metrics[-1]),flush=True)
        # All target participants are predicted by the same source ensemble.
        comparison['fold']=0;methods=[r['method'] for r in metrics if r['source']==source]
        boot=clustered_auc_bootstrap(comparison,methods,2000);primary=methods.index('joint') if 'joint' in methods else None
        for i,method in enumerate(methods):
            row=next(r for r in metrics if r['source']==source and r['method']==method)
            row['auroc_low'],row['auroc_high']=np.nanquantile(boot[:,i],[.025,.975])
            if primary is not None:
                row['joint_delta_low'],row['joint_delta_high']=np.nanquantile(boot[:,primary]-boot[:,i],[.025,.975])
        comparison.to_csv(out/f'{source}_to_{target}/paired_predictions.csv',index=False)
    dest=out/'metrics.csv';table=pd.DataFrame(metrics)
    if dest.exists():table=pd.concat([pd.read_csv(dest),table]).drop_duplicates(['source','target','method'],keep='last')
    table.to_csv(dest,index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',default=['RESILIENT','BrisT1D-Open'])
    p.add_argument('--variants',nargs='+',default=['joint','fixed','coordinates_only','no_coordinates']);run(p.parse_args())
