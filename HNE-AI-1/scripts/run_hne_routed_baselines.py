"""DNB and released FCM/CATCH on the same 256x6 histories as routed HNE."""
from pathlib import Path
import argparse,json,sys,gc,warnings
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dnb import score_window


def run(args):
    config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    for dataset in args.datasets:
        data=np.load(ROOT/f'data/hne_routed/{dataset}/arrays.npz')
        meta=pd.read_csv(ROOT/f'data/hne_routed/{dataset}/samples.csv',dtype={'person_id':str})
        windows=data['history'].transpose(0,2,1);folds=data['fold'];y=data['y']
        dnb=None
        for method in args.methods:
            if method=='DNB':
                path=ROOT/f'results/hne_routed/{dataset}/DNB/raw.npy';path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():dnb=np.load(path)
                else:
                    dnb=np.array([score_window(x,minimum=32) for x in windows]);np.save(path,dnb)
                print(dataset,'DNB finite',np.isfinite(dnb).sum(),'/',len(dnb),flush=True)
            for k in args.folds:
                ids={'train':np.flatnonzero(~np.isin(folds,[k,(k+1)%5])),
                     'validation':np.flatnonzero(folds==(k+1)%5),'held_person':np.flatnonzero(folds==k)}
                seeds=config['seeds'][:args.seeds] if method!='DNB' else config['seeds'][:1]
                dest=ROOT/'results/hne_routed'/dataset/method/f'fold{k}'
                for seed in seeds:
                    folder=dest/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'result.json').exists():continue
                    train_ids,val_ids=ids['train'],ids['validation']
                    if method=='FCM':
                        complete=np.isfinite(data['future32']).any((1,2))
                        train_ids=train_ids[complete[train_ids]];val_ids=val_ids[complete[val_ids]]
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning)
                        med=np.nanmedian(windows[train_ids],axis=(0,1))
                    med=np.nan_to_num(med)
                    fill=lambda x:np.where(np.isfinite(x),x,med).astype(np.float32)
                    if method=='DNB':
                        scores={part:dnb[ix] for part,ix in ids.items()}
                        info={'method':'DNB','training':'none','score':'original SD*abs(PCCin)/abs(PCCout)'}
                    elif method=='FCM':
                        from src.vendor_fcm import fit
                        fitted=fit(fill(windows[train_ids]),fill(data['future32'][train_ids]),fill(windows[val_ids]),
                                   fill(data['future32'][val_ids]),folder,batch_size=config['baseline_batch_size'],seed=seed,epochs=config['baseline_epochs'])
                        scores={part:fitted.score(fill(windows[ix])) for part,ix in ids.items()};info=fitted.metadata
                    else:
                        from src.vendor_catch import fit
                        fitted=fit(fill(windows[train_ids]),fill(windows[val_ids]),folder,
                                   batch_size=config['baseline_batch_size'],seed=seed,epochs=config['baseline_epochs'])
                        scores={part:fitted.score(fill(windows[ix])) for part,ix in ids.items()};info=fitted.metadata
                    # The raw score and a label-calibrated score are both retained.
                    tm=np.isfinite(scores['train']);vm=np.isfinite(scores['validation'])
                    choices=[]
                    for c in [.01,.1,1.]:
                        estimator=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000))
                        estimator.fit(scores['train'][tm,None],y[ids['train']][tm])
                        loss=log_loss(y[ids['validation']][vm],estimator.predict_proba(scores['validation'][vm,None])[:,1],labels=[0,1])
                        choices.append((loss,c,estimator))
                    _,c,estimator=min(choices,key=lambda r:r[0])
                    reference=np.sort(scores['validation'][vm & (y[ids['validation']]==0)])
                    for part,ix in ids.items():
                        frame=meta.iloc[ix].copy();s=scores[part];valid=np.isfinite(s)
                        frame['raw_score']=s;frame['probability']=np.nan;frame['calibrated_probability']=np.nan
                        frame.loc[valid,'probability']=np.searchsorted(reference,s[valid],side='right')/len(reference)
                        frame.loc[valid,'calibrated_probability']=estimator.predict_proba(s[valid,None])[:,1]
                        frame.to_csv(folder/f'{part}.csv',index=False)
                    info.update(input_shape=list(windows.shape[1:]),input_interval_hours=3,history_days=32,
                        train_windows=len(train_ids),validation_windows=len(val_ids),seed=seed,
                        imputation_medians=med.tolist(),score_calibration_C=c,
                        adaptations=['three-hour 256x6 observations; original modules and objectives','training-only imputation and scaling','held-person scoring uses history only'])
                    (folder/'result.json').write_text(json.dumps(info,indent=2,default=str),encoding='utf8')
                    if method!='DNB':del fitted;gc.collect();torch.cuda.empty_cache()
                for part in ['validation','held_person']:
                    frames=[pd.read_csv(dest/f'seed{s}'/f'{part}.csv',dtype={'person_id':str}) for s in seeds]
                    frame=frames[0].copy()
                    for col in ['raw_score','probability','calibrated_probability']:
                        frame[col]=np.mean([f[col].to_numpy() for f in frames],axis=0)
                    frame['rank_probability']=frame['probability']
                    frame['probability']=frame['raw_score']
                    frame.to_csv(dest/f'{part}.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['RESILIENT','BrisT1D-Open'])
    p.add_argument('--methods',nargs='+',default=['DNB','FCM','CATCH']);p.add_argument('--folds',nargs='+',type=int,default=list(range(5)))
    p.add_argument('--seeds',type=int,default=3);run(p.parse_args())
