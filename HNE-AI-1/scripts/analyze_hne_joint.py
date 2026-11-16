"""Main comparisons, paired participant intervals, and routing evidence."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from analyze_hne_routed import fold_auc,clustered_auc_bootstrap

OUT=ROOT/'results/hne_joint/summary'
KEY=['person_id','origin_day','fold','y']
VARIANTS=['joint','fixed','coordinates_only','no_coordinates']


def clustered_ap_bootstrap(frame,methods,repeats=2000):
    people=np.sort(frame.person_id.unique());index={p:i for i,p in enumerate(people)}
    rng=np.random.default_rng(20261113)
    draws=np.stack([np.bincount(rng.integers(len(people),size=len(people)),minlength=len(people)) for _ in range(repeats)]).astype(float)
    numerator=np.zeros((repeats,len(methods)));denominator=np.zeros(repeats)
    for _,group in frame.groupby('fold'):
        denominator+=draws[:,group.person_id.map(index)].sum(1)
        for j,method in enumerate(methods):
            g=group.sort_values(method,ascending=False);score=g[method].to_numpy();y=g.y.to_numpy()
            weight=draws[:,g.person_id.map(index)];total=weight.sum(1)
            ends=np.flatnonzero(np.r_[score[:-1]!=score[1:],True])
            tp=np.cumsum(weight*y[None],axis=1)[:,ends];count=np.cumsum(weight,axis=1)[:,ends]
            precision=np.divide(tp,count,out=np.zeros_like(tp),where=count>0)
            delta=np.diff(tp,axis=1,prepend=0.)
            ap=np.divide((delta*precision).sum(1),tp[:,-1],out=np.zeros(repeats),where=tp[:,-1]>0)
            numerator[:,j]+=ap*total
    return numerator/denominator[:,None]


def load_folds(folder,column=None):
    if len(list(folder.glob('fold*/held_person.csv')))!=5:return None
    held=[];validation={}
    for k in range(5):
        p=pd.read_csv(folder/f'fold{k}/held_person.csv',dtype={'person_id':str})
        v=pd.read_csv(folder/f'fold{k}/validation.csv',dtype={'person_id':str})
        col=column or ('probability' if 'probability' in p else 'score')
        p['value']=p[col];v['value']=v[col];held.append(p);validation[k]=v
    return pd.concat(held,ignore_index=True),validation


def dataset_sources(ds):
    sources={}
    for method in ['DNB','DNB-tuned','FCM-score-LR','CATCH-score-LR','HNE-additive']:
        sources[method+' daily' if method!='HNE-additive' else 'previous_HNE']=(ROOT/f'results/unified_daily/{ds}/{method}',None)
    for method in ['FCM','CATCH']:
        sources[method+' daily']=(ROOT/f'results/public_daily_repeats/ensemble/{ds}/{method}',None)
    for method in ['DNB','FCM','CATCH']:
        sources[method+' 3h']=(ROOT/f'results/hne_routed/{ds}/{method}',None)
        sources[method+' 3h calibrated']=(ROOT/f'results/hne_routed/{ds}/{method}','calibrated_probability')
    sources['Chronos-2 FT']=(ROOT/f'results/hne_routed/{ds}/native',None)
    for variant in VARIANTS:sources[variant]=(ROOT/f'results/hne_joint/{ds}/{variant}',None)
    for method,column in [('best_constant','constant_probability'),('uniform','uniform_probability')]:
        sources[method]=(ROOT/f'results/hne_joint/{ds}/joint',column)
    return sources


def main():
    OUT.mkdir(parents=True,exist_ok=True);totals=[];differences=[];intervals=[];folds=[];route_rows=[];seed_rows=[];within_differences=[];ap_differences=[]
    for ds in ['RESILIENT','BrisT1D-Open']:
        sources={m:(f,c) for m,(f,c) in dataset_sources(ds).items() if len(list(f.glob('fold*/held_person.csv')))==5}
        loaded={m:load_folds(f,c) for m,(f,c) in sources.items()}
        if 'joint' not in loaded:continue
        merged=None
        for method,(frame,val) in loaded.items():
            z=frame[KEY+['value']].rename(columns={'value':method})
            merged=z if merged is None else merged.merge(z,on=KEY,how='inner',validate='one_to_one')
        merged=merged.dropna(subset=list(loaded));merged.to_csv(OUT/f'{ds}_paired_predictions.csv',index=False)
        methods=list(loaded);samples=clustered_auc_bootstrap(merged,methods,2000)
        bootstrap={m:samples[:,i] for i,m in enumerate(methods)}
        apsamples=clustered_ap_bootstrap(merged,methods,2000)
        ap_boot={m:apsamples[:,i] for i,m in enumerate(methods)}
        personal=pd.DataFrame([{**{'person_id':person},**{m:roc_auc_score(g.y,g[m]) for m in methods}}
            for person,g in merged.groupby('person_id') if g.y.nunique()==2])
        personal.to_csv(OUT/f'{ds}_within_person_auroc.csv',index=False)
        rng=np.random.default_rng(20261113);draw=rng.integers(len(personal),size=(2000,len(personal)))
        within_boot={m:personal[m].to_numpy()[draw].mean(1) for m in methods}
        for method in methods:
            truth=[];alarm=[];aps=[]
            available=loaded[method][0].dropna(subset='value')
            for k,g in merged.groupby('fold'):
                v=loaded[method][1][int(k)].dropna(subset='value')
                threshold=float(np.quantile(v.loc[v.y.eq(0),'value'],.9,method='higher'))
                truth.extend(g.y);alarm.extend(g[method]>threshold)
                ap=average_precision_score(g.y,g[method]);aps.append((len(g),ap))
                folds.append(dict(dataset=ds,method=method,fold=int(k),n=len(g),positives=int(g.y.sum()),
                    auroc=roc_auc_score(g.y,g[method]),ap=ap,threshold=threshold))
            truth=np.asarray(truth);alarm=np.asarray(alarm)
            within=[roc_auc_score(g.y,g[method]) for _,g in merged.groupby('person_id') if g.y.nunique()==2]
            lo,hi=np.nanquantile(bootstrap[method],[.025,.975])
            wlo,whi=np.quantile(within_boot[method],[.025,.975])
            alo,ahi=np.quantile(ap_boot[method],[.025,.975])
            totals.append(dict(dataset=ds,method=method,n=len(merged),people=merged.person_id.nunique(),
                available_n=len(available),cohort_n=len(loaded[method][0]),all_available_auroc=fold_auc(available,'value'),
                auroc=fold_auc(merged,method),ap=sum(n*a for n,a in aps)/len(merged),within_person_auroc=np.mean(within),
                people_with_both_classes=len(within),balanced_accuracy=balanced_accuracy_score(truth,alarm),
                sensitivity_at_validation_10pct_FPR=alarm[truth==1].mean(),held_false_positive_rate=alarm[truth==0].mean(),
                auroc_low=lo,auroc_high=hi,ap_low=alo,ap_high=ahi,within_person_auroc_low=wlo,within_person_auroc_high=whi))
            intervals.append(dict(dataset=ds,method=method,auroc_low=lo,auroc_high=hi))
            if method!='joint':
                low,high=np.nanquantile(bootstrap['joint']-bootstrap[method],[.025,.975])
                differences.append(dict(dataset=ds,comparison=method,delta=fold_auc(merged,'joint')-fold_auc(merged,method),low=low,high=high))
                low,high=np.quantile(within_boot['joint']-within_boot[method],[.025,.975])
                within_differences.append(dict(dataset=ds,comparison=method,people=len(personal),
                    delta=personal['joint'].mean()-personal[method].mean(),low=low,high=high))
                low,high=np.quantile(ap_boot['joint']-ap_boot[method],[.025,.975])
                joint_ap=sum(len(g)*average_precision_score(g.y,g['joint']) for _,g in merged.groupby('fold'))/len(merged)
                ap_differences.append(dict(dataset=ds,comparison=method,delta=joint_ap-sum(n*a for n,a in aps)/len(merged),low=low,high=high))
        external=[m for m in methods if m.startswith(('DNB','FCM','CATCH','Chronos'))]
        best=max(external,key=lambda m:fold_auc(merged,m))
        low,high=np.nanquantile(bootstrap['joint']-np.stack([bootstrap[m] for m in external],1).max(1),[.025,.975])
        differences.append(dict(dataset=ds,comparison='strongest_external',strongest_observed=best,
            delta=fold_auc(merged,'joint')-fold_auc(merged,best),low=low,high=high))
        for variant in VARIANTS:
            folder=ROOT/f'results/hne_joint/{ds}/{variant}'
            for seed in [20261113,20261114,20261115]:
                paths=sorted(folder.glob(f'fold*/seed{seed}/held_person.csv'))
                if len(paths)!=5:continue
                frame=pd.concat([pd.read_csv(p,dtype={'person_id':str}) for p in paths],ignore_index=True)
                frame=frame.merge(merged[KEY],on=KEY,how='inner',validate='one_to_one')
                seed_rows.append(dict(dataset=ds,variant=variant,seed=seed,auroc=fold_auc(frame,'probability'),n=len(frame)))
                if variant!='joint':continue
                p=frame[[f'path_{j}' for j in range(6)]].to_numpy().clip(1e-7,1-1e-7)
                y=frame.y.to_numpy()[:,None];loss=-(y*np.log(p)+(1-y)*np.log(1-p));choice=frame.selected_path.to_numpy(int)
                selected_loss=loss[np.arange(len(p)),choice]
                ranks=1+(loss<selected_loss[:,None]).sum(1)+.5*((loss==selected_loss[:,None]).sum(1)-1)
                policy=frame[[f'policy_{j}' for j in range(6)]].to_numpy()
                route_rows.append(dict(dataset=ds,seed=seed,n=len(frame),mean_path_span=np.mean(p.max(1)-p.min(1)),
                    selected_loss_rank=np.mean(ranks),selected_bce=selected_loss.mean(),uniform_path_bce=loss.mean(),
                    retrospective_best_path_bce=loss.min(1).mean(),policy_sum_max_error=np.max(abs(policy.sum(1)-1)),
                    **{f'path{j}_fraction':np.mean(choice==j) for j in range(6)}))
    for name,rows in [('metrics',totals),('paired_intervals',differences),('fold_metrics',folds),('auroc_intervals',intervals),
                      ('routing',route_rows),('seed_metrics',seed_rows),('within_person_intervals',within_differences),
                      ('ap_intervals',ap_differences)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    table=pd.DataFrame(totals)
    if len(table):print(table[['dataset','method','n','auroc','within_person_auroc']].to_string(index=False))


if __name__=='__main__':main()
