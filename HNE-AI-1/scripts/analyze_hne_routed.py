"""Paired cohort comparisons and numerical figures from saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/hne_routed'
METHODS=['path_policy','learned','context_path','context_fixed','context_plain','context_free','context_free_fixed','context_free_plain','context_free_rank','plain','native','fixed','route_controls','random','no_history','no_network','no_environment','no_path_loss','DNB','FCM','CATCH','previous_HNE']
LABELS={'path_policy':'HNE','learned':'HNE joint','plain':'Chronos-2 CI-FT','native':'Chronos-2 FT','fixed':'HNE fixed','random':'HNE random',
        'DNB calibrated':'DNB+Cal','FCM calibrated':'FCM+Cal','CATCH calibrated':'CATCH+Cal'}


def fold_auc(frame,column,weights=None):
    total=0.;pairs=0.
    for _,g in frame.groupby('fold'):
        w=np.ones(len(g)) if weights is None else g.person_id.map(weights).to_numpy()
        pos=w[g.y.eq(1)].sum();neg=w[g.y.eq(0)].sum();pair=pos*neg
        if pair:
            total+=roc_auc_score(g.y,g[column],sample_weight=w)*pair;pairs+=pair
    return total/pairs if pairs else np.nan


def clustered_auc_bootstrap(frame,methods,repeats=1000):
    people=np.sort(frame.person_id.unique());count=len(people)
    indices={p:i for i,p in enumerate(people)}
    credits=np.zeros((len(methods),count,count));pairs=np.zeros((count,count))
    for _,group in frame.groupby('fold'):
        positive=group[group.y.eq(1)];negative=group[group.y.eq(0)]
        pi=positive.person_id.map(indices).to_numpy();ni=negative.person_id.map(indices).to_numpy()
        cell=(pi[:,None]*count+ni[None,:]).ravel()
        pairs+=np.bincount(cell,minlength=count*count).reshape(count,count)
        for j,method in enumerate(methods):
            a=positive[method].to_numpy()[:,None];b=negative[method].to_numpy()[None,:]
            score=(a>b).astype(float)+.5*(a==b)
            credits[j]+=np.bincount(cell,weights=score.ravel(),minlength=count*count).reshape(count,count)
    rng=np.random.default_rng(20261113)
    weights=np.stack([np.bincount(rng.integers(count,size=count),minlength=count) for _ in range(repeats)]).astype(float)
    numerator=np.einsum('bi,mij,bj->bm',weights,credits,weights,optimize=True)
    denominator=np.einsum('bi,ij,bj->b',weights,pairs,weights,optimize=True)
    return numerator/denominator[:,None]


def analyze():
    out=RESULT/'summary';out.mkdir(parents=True,exist_ok=True)
    totals=[];perfold=[];comparisons=[];intervals=[]
    choice_file=RESULT/'architecture_selection.json'
    selected_primary=json.loads(choice_file.read_text())['primary_variant'] if choice_file.exists() else 'path_policy'
    for ds in ['RESILIENT','BrisT1D-Open']:
        frames={};validation={}
        for method in METHODS:
            method_root=ROOT/'results/unified_daily'/ds/'HNE-additive' if method=='previous_HNE' else RESULT/ds/method
            paths=list(method_root.glob('fold*/held_person.csv'))
            if len(paths)!=5:continue
            frame=pd.concat([pd.read_csv(p,dtype={'person_id':str}) for p in sorted(paths)],ignore_index=True)
            frame['method']=method;frames[method]=frame
            validation[method]={k:pd.read_csv(method_root/f'fold{k}/validation.csv') for k in range(5)}
            if 'calibrated_probability' in frame:
                name=method+' calibrated';frames[name]=frame.assign(probability=frame.calibrated_probability)
                validation[name]={k:v.assign(probability=v.calibrated_probability) for k,v in validation[method].items()}
            if 'constant_path_probability' in frame:
                name=method+'_constant';frames[name]=frame.assign(probability=frame.constant_path_probability)
                validation[name]={k:v.assign(probability=v.constant_path_probability) for k,v in validation[method].items()}
            for name,column in [('random_path','random_path_probability'),('uniform_paths','uniform_path_probability')]:
                if column in frame:
                    frames[name]=frame.assign(probability=frame[column])
                    validation[name]={k:v.assign(probability=v[column]) for k,v in validation[method].items()}
        if not frames:continue
        merged=None
        for method,frame in frames.items():
            z=frame[['person_id','origin_day','fold','y','probability']].rename(columns={'probability':method})
            merged=z if merged is None else merged.merge(z,on=['person_id','origin_day','fold','y'],how='inner',validate='one_to_one')
        merged=merged.dropna(subset=list(frames))
        merged.to_csv(out/f'{ds}_paired_predictions.csv',index=False)
        for method in frames:
            prediction=[];truth=[];sens=[]
            for k,g in merged.groupby('fold'):
                v=validation[method][int(k)].dropna(subset=['probability'])
                neg=v.loc[v.y.eq(0),'probability'];threshold=float(np.quantile(neg,.9,method='higher'))
                prediction.extend(g[method]>=threshold);truth.extend(g.y)
                positive=g.y.eq(1)
                sens.append((int(((g[method]>=threshold)&positive).sum()),int(positive.sum())))
                perfold.append(dict(dataset=ds,method=method,fold=int(k),n=len(g),positives=int(g.y.sum()),
                                    auroc=float(roc_auc_score(g.y,g[method])),ap=float(average_precision_score(g.y,g[method])),threshold=threshold))
            fold_ap=sum(len(g)*average_precision_score(g.y,g[method]) for _,g in merged.groupby('fold'))/len(merged)
            truth_array=np.asarray(truth);alarm=np.asarray(prediction)
            person_scores=[roc_auc_score(g.y,g[method]) for _,g in merged.groupby('person_id') if g.y.nunique()==2]
            totals.append(dict(dataset=ds,method=method,people=merged.person_id.nunique(),n=len(merged),
                               auroc=fold_auc(merged,method),ap=float(fold_ap),
                               within_person_auroc=float(np.mean(person_scores)),people_with_both_classes=len(person_scores),
                               balanced_accuracy=float(balanced_accuracy_score(truth,prediction)),
                               sensitivity_at_validation_10pct_FPR=sum(a for a,b in sens)/sum(b for a,b in sens),
                               held_false_positive_rate=float(alarm[truth_array==0].mean()),
                               origin_alarm_fraction=float(alarm.mean()),
                               brier=float(brier_score_loss(merged.y,merged[method])) if method not in ['DNB','FCM','CATCH'] else np.nan))
        primary=selected_primary if selected_primary in frames else 'learned'
        if primary in frames:
            opponents=[m for m in frames if m!=primary];delta={m:[] for m in opponents}
            external=[m for m in frames if m in ['native','DNB','FCM','CATCH','DNB calibrated','FCM calibrated','CATCH calibrated']]
            methods=list(frames);samples=clustered_auc_bootstrap(merged,methods)
            bootstrap={m:samples[:,i] for i,m in enumerate(methods)}
            delta={m:bootstrap[primary]-bootstrap[m] for m in opponents}
            best_comparison=bootstrap[primary]-np.max(np.stack([bootstrap[m] for m in external],1),axis=1) if external else []
            for m,values in delta.items():
                lo,hi=np.nanquantile(values,[.025,.975])
                comparisons.append(dict(dataset=ds,primary=primary,comparison=m,delta=fold_auc(merged,primary)-fold_auc(merged,m),low=lo,high=hi))
            if external:
                lo,hi=np.nanquantile(best_comparison,[.025,.975]);best=max(external,key=lambda m:fold_auc(merged,m))
                comparisons.append(dict(dataset=ds,primary=primary,comparison='strongest_external',strongest_observed=best,
                    delta=fold_auc(merged,primary)-fold_auc(merged,best),low=lo,high=hi))
            for m,values in bootstrap.items():
                lo,hi=np.nanquantile(values,[.025,.975]);intervals.append(dict(dataset=ds,method=m,auroc_low=lo,auroc_high=hi))
    table=pd.DataFrame(totals);table.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(perfold).to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'auroc_intervals.csv',index=False)
    if len(table):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(1,2,figsize=(13,4.1),layout='constrained')
        for ax,ds in zip(axes,['RESILIENT','BrisT1D-Open']):
            order=['DNB','DNB calibrated','FCM','FCM calibrated','CATCH','CATCH calibrated','native',selected_primary]
            t=table[table.dataset.eq(ds)&table.method.isin(order)]
            t=t.set_index('method').reindex([x for x in order if x in t.method.to_list()])
            bars=ax.bar(np.arange(len(t)),t.auroc,color=['black' if x==selected_primary else '.72' for x in t.index],width=.65)
            ax.bar_label(bars,labels=[f'{x:.3f}' for x in t.auroc],padding=4)
            ax.set_xticks(np.arange(len(t)),['HNE' if x==selected_primary else LABELS.get(x,x) for x in t.index],rotation=30,ha='right');ax.set_ylim(0,1);ax.set_title(ds);ax.set_ylabel('AUROC')
        fig.savefig(out/'main_auroc.png',dpi=250);fig.savefig(out/'main_auroc.pdf');plt.close(fig)
    print(table.to_string(index=False))


if __name__=='__main__':analyze()
