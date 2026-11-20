"""Analyze completed CPU studies without treating conditional heads as full CV.

Person bootstrap is stratified by original evaluation fold, paired across seeds
and methods, with fixed trained models. It does not include retraining uncertainty.
"""
import json
import os
from pathlib import Path
import sys
import hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'cache/matplotlib'))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
OUT=ROOT/'results/local_experiments'
METRICS=('auroc','auprc','brier','nll','accuracy','sensitivity')
SEEDS=(20261113,20261114,20261115)
REPETITIONS=2000


def weighted_metrics(y,p,w):
    """Vectorized exact weighted ranking metrics, including tied probabilities."""
    p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=float);w=np.asarray(w,dtype=float)
    order=np.argsort(-p,kind='stable');s=p[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    positive=np.add.reduceat(w[:,order]*y[order],starts,axis=1)
    negative=np.add.reduceat(w[:,order]*(1-y[order]),starts,axis=1)
    tp=positive.cumsum(1);fp=negative.cumsum(1)
    npos=tp[:,-1];nneg=fp[:,-1];n=npos+nneg
    with np.errstate(divide='ignore',invalid='ignore'):
        auc=(positive*(nneg[:,None]-fp+.5*negative)).sum(1)/(npos*nneg)
        ap=np.divide(positive*tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0).sum(1)/npos
        ap=np.where((npos>0)&(nneg>0),ap,np.nan)
        brier=(w*(p-y)**2).sum(1)/n
        eps=np.finfo(np.float64).eps;q=p.clip(eps,1-eps)
        nll=(w*(-y*np.log(q)-(1-y)*np.log1p(-q))).sum(1)/n
        acc=(w*((p>=.5)==y)).sum(1)/n
        sens=(w*(y*(p>=.5))).sum(1)/npos
    return dict(auroc=auc,auprc=ap,brier=brier,nll=nll,accuracy=acc,sensitivity=sens),dict(auroc=npos*nneg,auprc=npos,brier=n,nll=n,accuracy=n,sensitivity=npos)


def person_weights(frame,seed):
    people,person_index=np.unique(frame.person,return_inverse=True)
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(people),(REPETITIONS,len(people)))
    counts=np.zeros((REPETITIONS,len(people)),dtype=np.int16)
    for i,row in enumerate(samples):counts[i]=np.bincount(row,minlength=len(people))
    return np.vstack((np.ones((1,len(frame))),counts[:,person_index]))


def check_metrics():
    rng=np.random.default_rng(18);y=rng.integers(0,2,47);p=np.round(rng.random(47),1)
    w=rng.integers(0,4,(5,47));m,_=weighted_metrics(y,p,w)
    for i in range(5):
        assert np.isclose(m['auroc'][i],roc_auc_score(y,p,sample_weight=w[i]),atol=1e-12)
        assert np.isclose(m['auprc'][i],average_precision_score(y,p,sample_weight=w[i]),atol=1e-12)
        assert np.isclose(m['nll'][i],log_loss(y,p,sample_weight=w[i],labels=[0,1]),atol=1e-12)
        assert np.isclose(m['brier'][i],brier_score_loss(y,p,sample_weight=w[i]),atol=1e-12)
    # All-positive and all-negative draws remain unscorable ranking observations.
    for yy in (np.zeros(5),np.ones(5)):
        a,_=weighted_metrics(yy,np.linspace(0,1,5),np.ones((1,5)))
        assert np.isnan(a['auroc'][0]) and np.isnan(a['auprc'][0])


def aggregate(folds):
    out={}
    for metric in METRICS:
        values=np.stack([x[0][metric] for x in folds]);weights=np.stack([x[1][metric] for x in folds])
        valid=np.isfinite(values)&(weights>0)
        numerator=np.where(valid,values*weights,0).sum(0);denominator=np.where(valid,weights,0).sum(0)
        out[metric]=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    return out


def interval(values):
    good=values[1:][np.isfinite(values[1:])]
    return dict(value=float(values[0]),ci_low=float(np.quantile(good,.025)) if len(good) else None,
                ci_high=float(np.quantile(good,.975)) if len(good) else None,valid_bootstraps=int(len(good)))


def baseline_analysis():
    registry=pd.read_csv(ROOT/'results/local_baselines/run_registry.csv')
    if len(registry)!=150 or not registry.status.eq('completed').all():raise RuntimeError('CPU baseline study is incomplete')
    frames=[pd.read_csv(p,dtype={'person':str}) for p in sorted((ROOT/'results/local_baselines').glob('*/*/*/*/predictions.csv'))]
    predictions=pd.concat(frames,ignore_index=True);predictions=predictions[predictions.split=='evaluation']
    priors=[]
    for (dataset,fold,seed),part in predictions[predictions.method=='M01_HGB'].groupby(['dataset','fold','seed']):
        training=pd.read_csv(ROOT/'artifacts/local_baselines'/dataset/f'fold{fold}'/f'seed{seed}'/'M01_HGB/training_origins.csv',dtype={'person':str})
        prior=training.groupby('person').label.mean().mean()
        priors.append(part.assign(method='Training prior',probability=prior))
    predictions=pd.concat([predictions,*priors],ignore_index=True)
    summary=[];seed_summary=[];coverage=[];calibration=[];prior_contrasts=[]
    for dataset,data in predictions.groupby('dataset'):
        methods=data.method.unique();weights={};indexes={};method_boot={}
        for fold in range(5):
            reference=data[(data.fold==fold)&(data.seed==SEEDS[0])&(data.method==methods[0])].sort_values(['person','origin'])
            indexes[fold]=reference.set_index(['person','origin']).label
            weights[fold]=person_weights(reference,20261120+fold)
            coverage.append(dict(dataset=dataset,fold=fold,origins=len(reference),people=reference.person.nunique(),positives=int(reference.label.sum()),positive_people=reference[reference.label==1].person.nunique()))
        for method in methods:
            method_boot[method]={}
            results=[]
            for seed in SEEDS:
                folds=[]
                for fold in range(5):
                    df=data[(data.fold==fold)&(data.seed==seed)&(data.method==method)].sort_values(['person','origin'])
                    if not df.set_index(['person','origin']).label.equals(indexes[fold]):raise AssertionError('Unpaired public predictions')
                    folds.append(weighted_metrics(df.label.to_numpy(),df.probability.to_numpy(),weights[fold]))
                seed_result=aggregate(folds);results.append(seed_result)
                seed_summary.append(dict(dataset=dataset,method=method,seed=seed,**{k:float(v[0]) for k,v in seed_result.items()}))
            for metric in METRICS:
                arrays=np.stack([x[metric] for x in results])
                method_boot[method][metric]=np.mean(arrays,axis=0)
                summary.append(dict(dataset=dataset,method=method,metric=metric,**interval(np.mean(arrays,axis=0)),seed_sd=float(np.std(arrays[:,0],ddof=1))))
            df=data[(data.seed==20261114)&(data.method==method)]
            bins=np.minimum((df.probability.to_numpy()*10).astype(int),9)
            for b in range(10):
                part=df[bins==b]
                calibration.append(dict(dataset=dataset,method=method,bin=b,left=b/10,right=(b+1)/10,n=len(part),mean_probability=part.probability.mean(),event_rate=part.label.mean()))
        for method in methods:
            if method=='Training prior':continue
            for metric in METRICS:
                prior_contrasts.append(dict(dataset=dataset,method=method,reference='Training prior',metric=metric,**interval(method_boot[method][metric]-method_boot['Training prior'][metric])))
    pd.DataFrame(summary).to_csv(OUT/'baseline_summary.csv',index=False)
    pd.DataFrame(seed_summary).to_csv(OUT/'baseline_seed_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'fold_coverage.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'calibration_bins.csv',index=False)
    pd.DataFrame(prior_contrasts).to_csv(OUT/'baseline_prior_contrasts.csv',index=False)
    return pd.DataFrame(summary),pd.DataFrame(calibration)


def evidence_analysis():
    registry=pd.read_csv(ROOT/'results/cached_evidence/run_registry.csv')
    if len(registry)!=45:raise RuntimeError('Conditional evidence study is incomplete')
    df=pd.concat([pd.read_csv(p,dtype={'person':str}) for p in sorted((ROOT/'results/cached_evidence').glob('*_predictions.csv'))])
    null_registry=pd.read_csv(ROOT/'results/cached_null_controls/run_registry.csv');assert len(null_registry)==6
    selection=pd.concat([registry,null_registry.assign(view='no_evidence')],ignore_index=True)
    selection.assign(selected_epoch_zero=selection.selected_epoch.eq(0).astype(int)).groupby(['mode','view']).agg(
        runs=('head_seed','size'),selected_epoch_zero=('selected_epoch_zero','sum'),minimum_selected_epoch=('selected_epoch','min'),
        maximum_selected_epoch=('selected_epoch','max'),median_seconds=('seconds','median'),maximum_seconds=('seconds','max')).to_csv(OUT/'head_selection.csv')
    null_df=pd.concat([pd.read_csv(p,dtype={'person':str}) for p in sorted((ROOT/'results/cached_null_controls').glob('*_predictions.csv'))])
    df=pd.concat([df,null_df],ignore_index=True)
    b=pd.read_csv(ROOT/'results/key_comparison/cpu_pilot/BrisT1D-Open/fold0_seed20261114_predictions.csv',dtype={'person':str})
    baseline={'validation':b[(b.method=='M04_C2_LoRA')&(b.split=='validation')][['person','origin','label','probability']],
              'transfer':pd.read_csv(ROOT/'results/guidance_transfer/RESILIENT_predictions.csv',dtype={'person':str}).rename(columns={'baseline':'probability'})[['person','origin','label','probability']]}
    records=[];deltas=[];bootstrap_values={}
    for split,data in df.groupby('split'):
        ref=baseline[split].sort_values(['person','origin']);w=person_weights(ref,20261120)
        base,_=weighted_metrics(ref.label.to_numpy(),ref.probability.to_numpy(),w)
        for (mode,view,policy),group in data.groupby(['mode','view','policy']):
            results=[]
            for seed in SEEDS:
                part=group[group.head_seed==seed].sort_values(['person','origin'])
                assert list(zip(part.person,part.origin))==list(zip(ref.person,ref.origin))
                assert np.array_equal(part.label,ref.label)
                result,_=weighted_metrics(part.label.to_numpy(),part.probability.to_numpy(),w);results.append(result)
            for metric in METRICS:
                values=np.stack([x[metric] for x in results]).mean(0)
                bootstrap_values[(split,mode,view,policy,metric)]=values
                records.append(dict(split=split,mode=mode,view=view,policy=policy,metric=metric,**interval(values)))
                deltas.append(dict(split=split,mode=mode,view=view,policy=policy,metric=metric,**interval(values-base[metric]),people=ref.person.nunique()))
    result=pd.DataFrame(records);delta=pd.DataFrame(deltas)
    result.to_csv(OUT/'evidence_summary.csv',index=False);delta.to_csv(OUT/'evidence_paired_differences.csv',index=False)
    contrasts=[]
    for split in baseline:
        for mode,control in [('hidden','hidden_only'),('logit','logit_calibration')]:
            for policy in ('selected','always_on'):
                for metric in METRICS:
                    contrast=bootstrap_values[(split,mode,'aligned',policy,metric)]-bootstrap_values[(split,control,'no_evidence',policy,metric)]
                    contrasts.append(dict(split=split,mode=mode,control=control,policy=policy,metric=metric,**interval(contrast)))
    pd.DataFrame(contrasts).to_csv(OUT/'evidence_vs_null.csv',index=False)
    return result,delta


def within_person_analysis():
    frames=[pd.read_csv(p,dtype={'person':str}) for p in sorted((ROOT/'results/local_baselines').glob('*/*/*/*/predictions.csv'))]
    data=pd.concat(frames,ignore_index=True);data=data[data.split=='evaluation'];rows=[]
    for (dataset,method,seed,fold,person),part in data.groupby(['dataset','method','seed','fold','person']):
        positive=int(part.label.sum());negative=len(part)-positive
        rows.append(dict(dataset=dataset,method=method,seed=seed,fold=fold,person=person,origins=len(part),positives=positive,
                         pairs=positive*negative,auroc=float(roc_auc_score(part.label,part.probability)) if positive and negative else np.nan))
    people=pd.DataFrame(rows);people.to_csv(OUT/'within_person_metrics.csv',index=False)
    summary=[];seed_rows=[]
    for dataset,part in people.groupby('dataset'):
        reference=part[(part.method=='M01_HGB')&(part.seed==SEEDS[0])].sort_values(['fold','person']);weights=[]
        for fold,group in reference.groupby('fold'):weights.append(person_weights(group,20261120+fold))
        w=np.concatenate(weights,axis=1);pairs=reference.pairs.to_numpy();denominator=(w*pairs).sum(1)
        for method,g in part.groupby('method'):
            values=[]
            for seed in SEEDS:
                p=g[g.seed==seed].sort_values(['fold','person']);assert list(zip(p.fold,p.person))==list(zip(reference.fold,reference.person))
                assert np.array_equal(p.pairs,pairs)
                numerator=(w*pairs*p.auroc.fillna(0).to_numpy()).sum(1)
                estimate=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0);values.append(estimate)
                seed_rows.append(dict(dataset=dataset,method=method,seed=seed,auroc=float(estimate[0])))
            summary.append(dict(dataset=dataset,method=method,metric='within_person_auroc',**interval(np.mean(values,axis=0)),
                                evaluable_people=int((pairs>0).sum()),total_people=len(pairs),positive_negative_pairs=int(pairs.sum())))
    pd.DataFrame(summary).to_csv(OUT/'within_person_summary.csv',index=False);pd.DataFrame(seed_rows).to_csv(OUT/'within_person_seed_metrics.csv',index=False)


def alarm_analysis():
    registry=pd.read_csv(ROOT/'results/local_alarms/run_registry.csv')
    assert len(registry)==150 and registry.status.eq('completed').all()
    rows=[]
    for (dataset,method,seed),g in registry.groupby(['dataset','method','seed']):
        total=g[['candidate_issue_days','scorable_issue_days','unknown_days','alerts','false_alerts','alerts_unknown','alerts_positive','timely_hits','events']].sum()
        rows.append(dict(dataset=dataset,method=method,seed=seed,**total.to_dict(),
                         recall=total.timely_hits/total.events if total.events else np.nan,
                         false_per30=30*total.false_alerts/total.scorable_issue_days))
    per_seed=pd.DataFrame(rows);per_seed.to_csv(OUT/'alarm_seed_metrics.csv',index=False)
    per_seed.groupby(['dataset','method']).mean(numeric_only=True).drop(columns='seed').to_csv(OUT/'alarm_summary.csv')
    checks=[]
    for path in (ROOT/'results/local_alarms').glob('*/*/*/*/report.json'):
        report=json.loads(path.read_text());assert report['validation']['false_per_30_scorable_issue_days']<=1+1e-12
        checks.extend(report['known_prediction_max_error'].values())
    (OUT/'alarm_validation.json').write_text(json.dumps(dict(replays=150,known_forecast_max_abs_error=max(checks),validation_threshold_budgets_passed=True,
                                                          scope='Evaluation false-alarm rates need not meet the validation budget; unknown alerts retain their chronological silence effect'),indent=2)+'\n')


def figures(baselines,calibration,deltas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    names={'M01_HGB':'HGB','Random Forest':'Random Forest','XGBoost':'XGBoost','LightGBM':'LightGBM','HYDRA':'HYDRA'}
    styles={'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'}
    with plt.rc_context(styles):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        order=['M01_HGB','Random Forest','XGBoost','LightGBM','HYDRA']
        for ax,(dataset,part) in zip(axes,baselines[baselines.metric=='auroc'].groupby('dataset')):
            part=part.set_index('method').loc[order]
            ax.hlines(np.arange(5),part.ci_low,part.ci_high,color='#0072B2',lw=1)
            ax.scatter(part.value,np.arange(5),marker='o',color='#0072B2',s=22)
            ax.axvline(.5,color='.4',ls='--',lw=.7);ax.set(yticks=np.arange(5),yticklabels=[names[m] for m in order],xlim=(0,1),xlabel='Within-fold weighted AUROC',title=dataset)
            ax.invert_yaxis()
        fig.suptitle('Three pipeline seeds; 95% person-bootstrap intervals with fixed models',fontsize=9)
        for ext in ('pdf','png'):fig.savefig(OUT/f'baseline_discrimination.{ext}',dpi=300)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        views=['aligned','absolute_only','coverage_only','no_score_calibration','node_edge_only']
        labels=['Aligned','Raw HNE errors','Coverage only','No score calibration','Node / edge only']
        for ax,split in zip(axes,['validation','transfer']):
            part=deltas[(deltas.split==split)&(deltas['mode']=='logit')&(deltas.policy=='always_on')&(deltas.metric=='nll')].set_index('view').loc[views]
            ax.hlines(np.arange(5),part.ci_low*1e4,part.ci_high*1e4,color='#D55E00',lw=1)
            ax.scatter(part.value*1e4,np.arange(5),marker='s',color='#D55E00',s=22)
            ax.axvline(0,color='.4',ls='--',lw=.7);ax.set(yticks=np.arange(5),yticklabels=labels,xlabel=r'$\Delta$NLL ($10^{-4}$ units)',title='Bris dev. (3 people)' if split=='validation' else 'RES transfer (9 people)');ax.invert_yaxis()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        fig.suptitle('Three head seeds, one pilot S/L; 95% person-bootstrap intervals',fontsize=9)
        for ext in ('pdf','png'):fig.savefig(OUT/f'evidence_output_ablation.{ext}',dpi=300)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        markers=['o','s','^','v','D'];colors=['#0072B2','#D55E00','#009E73','#7A5195','#555555']
        for ax,(dataset,part) in zip(axes,calibration.groupby('dataset')):
            for method,marker,color in zip(order,markers,colors):
                bins=part[part.method==method].sort_values('bin')
                ax.plot(bins.mean_probability,bins.event_rate,marker=marker,color=color,label=names[method],lw=.8,ms=4)
            ax.plot([0,1],[0,1],'--',color='.6',lw=.7);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed event fraction',title=dataset)
        axes[1].legend(fontsize=7,loc='upper left')
        fig.suptitle('Seed 20261114; ten fixed probability bins; empty bins remain gaps',fontsize=9)
        for ext in ('pdf','png'):fig.savefig(OUT/f'baseline_reliability.{ext}',dpi=300)
        plt.close(fig)
        teacher_frames=[]
        for label,directory in [('Pilot','small_predictor'),('Selected','teacher_selection')]:
            part=pd.read_csv(ROOT/f'results/{directory}/paired_gains.csv')
            teacher_frames.append(part[part.period=='reference_score'].assign(teacher=label))
        teacher=pd.concat(teacher_frames,ignore_index=True);teacher.to_csv(OUT/'teacher_quality_source.csv',index=False)
        teacher_limits=(min(float(teacher[['gain_N','gain_E']].min().min())*1.2,-.1),max(float(teacher[['gain_N','gain_E']].max().max())*1.2,.1))
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        for ax,(dataset,part) in zip(axes,teacher.groupby('dataset')):
            for x,(label,gain) in enumerate([('Pilot','gain_N'),('Selected','gain_N'),('Pilot','gain_E'),('Selected','gain_E')]):
                values=part.loc[part.teacher==label,gain].to_numpy()
                offsets=np.linspace(-.12,.12,len(values))
                ax.scatter(x+offsets,values,color='#777777' if label=='Pilot' else '#0072B2',marker='o' if gain=='gain_N' else 's',s=18,alpha=.8)
                ax.plot([x-.2,x+.2],[np.median(values)]*2,color='black',lw=1.4)
            ax.axhline(0,color='.4',ls='--',lw=.7)
            ax.set_yscale('symlog',linthresh=.1)
            ax.set(xticks=np.arange(4),xticklabels=['Pilot N','Selected N','Pilot E','Selected E'],ylabel='Conditional NLL gain (symlog)',title=f'{dataset} ({part.person.nunique()} people)',ylim=teacher_limits)
            ax.tick_params(axis='x',labelsize=8)
        fig.suptitle('Reference-score periods; dots are people; black bars are medians',fontsize=9)
        for ext in ('pdf','png'):fig.savefig(OUT/f'teacher_quality.{ext}',dpi=300)
        plt.close(fig)
        coverage=pd.read_csv(OUT/'fold_coverage.csv')
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        for ax,(dataset,part) in zip(axes,coverage.groupby('dataset')):
            ax.bar(part.fold,part.positive_people,label='People with positive origins',color='#0072B2',edgecolor='black',lw=.4)
            ax.bar(part.fold,part.people-part.positive_people,bottom=part.positive_people,label='Other evaluable people',color='white',edgecolor='black',hatch='//',lw=.4)
            ax.set(xticks=range(5),xlabel='Evaluation fold',ylabel='People',title=dataset,ylim=(0,17))
        axes[1].legend(fontsize=7,loc='upper right')
        for ext in ('pdf','png'):fig.savefig(OUT/f'fold_support.{ext}',dpi=300)
        plt.close(fig)


def main():
    import psutil
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    OUT.mkdir(parents=True,exist_ok=True);check_metrics()
    b,c=baseline_analysis();e,d=evidence_analysis();within_person_analysis();alarm_analysis();figures(b,c,d)
    manifest=dict(bootstrap_repetitions=REPETITIONS,seed=20261120,
                  estimator='Within-fold AUROC weighted by positive-negative pairs; AUPRC by positives; NLL/Brier/accuracy by origins; sensitivity by positives; mean of per-seed metrics',
                  uncertainty='95% percentile person bootstrap within original folds; paired across methods/seeds; trained models fixed; no retraining uncertainty',
                  conditional_heads='One pilot S/L, three head seeds, source validation selected; exploratory; not complete-pipeline CV or confirmation',
                  reliability='Seed 20261114, ten fixed equal-width probability bins; empty bins retained as NaN; no fitted calibration or smoothing',
                  teacher_quality='Reference-score period; equal-family person averages; raw person points and medians, deterministic horizontal offsets; symlog y with linear threshold 0.1; positive gain means conditional prediction improves; pilot and source-selected teachers share persons',
                  figures='Provisional general research figures; no venue-specific compliance claim; 7.2 x 3.2 inches; 300 dpi PNG and vector PDF',
                  comparison='Exploratory summaries, not the six preregistered M10 contrasts; no confirmatory p-values',
                  baseline_units=150,conditional_units=45)
    manifest['null_controls']='Six exploratory head runs, 36 candidates, same pilot and source selection; positive affine logit calibration (2 parameters), hidden-only residual refit (793 parameters); neither receives S evidence; not formal matched CV'
    manifest['training_prior']='Unfitted reference: each outer fold uses mean of per-training-person event prevalence; same scalar for its evaluation origins; no validation or evaluation labels used for the scalar'
    manifest['within_person']='AUROC compares positive and negative origins only within the same person, weighted by their pair counts; single-class people contribute zero pairs; original fold-stratified person bootstrap and per-seed metric averaging retained; descriptive diagnostic, not a new primary metric'
    manifest['figures_alt']={
        'baseline_discrimination':'AUROC estimates and fixed-model person-bootstrap intervals for five CPU baselines in each development cohort; dashed line denotes chance ranking.',
        'evidence_output_ablation':'Change in NLL from the fixed pilot baseline, displayed in 1e-4 units, for five always-on full-output evidence views; source validation is selection-exposed; transfer heads use no target selection.',
        'baseline_reliability':'Observed fraction versus average predicted probability in ten fixed bins for one prespecified seed; empty bins break curves; bin counts are in calibration_bins.csv.',
        'teacher_quality':'Person-level conditional prediction gains before and after source-only teacher search; markers identify N/E, black bars are medians; positive values denote improved predictive NLL.',
        'fold_support':'Each bar counts evaluable people in an outer fold, with positive-origin people solid and others hatched; Bris fold 1 contains only one positive person.'}
    manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'results/local_baselines/run_registry.csv',ROOT/'results/cached_evidence/run_registry.csv',ROOT/'results/cached_null_controls/run_registry.csv',ROOT/'results/local_alarms/run_registry.csv',ROOT/'results/small_predictor/person_metrics.csv',ROOT/'results/teacher_selection/person_metrics.csv']}
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(b[b.metric=='auroc'].to_string(index=False))


if __name__=='__main__':main()
