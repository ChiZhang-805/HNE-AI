"""Publication tables and monochrome bar figures from saved measurements."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper'
TABLES=OUT/'tables';FIGURES=OUT/'figures'
TABLES.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})


def ranked_rows(rows, columns, across=False):
    """Mark the two highest distinct scores at the table's displayed precision."""
    styled=[list(map(str,row)) for row in rows]
    groups=([[(i,j) for j in columns] for i in range(len(rows))] if across else
            [[(i,j) for i in range(len(rows))] for j in columns])
    for cells in groups:
        cells=[(i,j) for i,j in cells if styled[i][j]!='--']
        scores={cell:float(styled[cell[0]][cell[1]]) for cell in cells}
        levels=sorted(set(scores.values()),reverse=True)[:2]
        for (i,j),value in scores.items():
            if value in levels:
                command=['textbf','underline'][levels.index(value)]
                styled[i][j]='\\'+command+'{'+styled[i][j]+'}'
    return styled


def table(path,columns,rows,alignment,group_header=None,*,rank_columns=(),rank_across=False,highlight=None,shade_column=None):
    rows=ranked_rows(rows,rank_columns,rank_across)
    lines=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}'+alignment+r'@{}}',r'\toprule']
    if group_header:lines += [group_header+r' \\',r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}']
    lines += [' & '.join(columns)+r' \\',r'\midrule']
    for row in rows:
        if row[0]==highlight:row[0]=r'\cellcolor{hnegray}'+row[0]
        if shade_column is not None:row[shade_column]=r'\cellcolor{hnegray}'+row[shade_column]
        lines.append(' & '.join(row)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular*}']
    (TABLES/path).write_text('\n'.join(lines),encoding='utf-8')


def grouped_table(path, columns, groups, alignment):
    """Keep cohort labels in row headers so detailed appendix tables fit a column."""
    lines=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}'+alignment+r'@{}}',r'\toprule',
           ' & '.join(columns)+r' \\']
    for label,rows in groups:
        lines += [r'\midrule',rf'\multicolumn{{{len(columns)}}}{{l}}{{{label}}} \\',r'\midrule']
        lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular*}']
    (TABLES/path).write_text('\n'.join(lines),encoding='utf-8')


def compact_comparison(path, rows):
    """Keep uncertainty and alarm rates beside each method in a compact column."""
    rows=ranked_rows(rows,[1,3,4])
    lines=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',
           r'Method & \shortstack{AUROC\\{}[95\% CI]} & Within & AP & \shortstack{TPR/FPR\\(\%)} \\',r'\midrule']
    for name,auc,ci,within,ap,tpr,fpr in rows:
        name=name.replace(' (3 seeds)','')
        prefix=r'\cellcolor{hnegray}' if name=='HNE' else ''
        lines += [prefix+name+' & '+r'\shortstack[r]{'+auc+r' \\ {}'+ci+'} & '+within+' & '+ap+' & '+tpr+'/'+fpr+r' \\']
    lines += [r'\bottomrule',r'\end{tabular*}']
    (TABLES/path).write_text('\n'.join(lines),encoding='utf-8')


def save(fig,name):
    if name in {'main_auroc','human_adaptation','robustness','ssaqs_transfer'}:
        for index,ax in enumerate(fig.axes):
            ax.set_title('')
            ax.tick_params(labelsize=9.5)
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
            if ax.get_legend() is not None:
                for label in ax.get_legend().get_texts():label.set_fontsize(8)
            segments=[segment for collection in ax.collections
                      if hasattr(collection,'get_segments') for segment in collection.get_segments()]
            for label in ax.texts:
                if not label.get_text().replace('.','',1).isdigit():continue
                x,_=label.get_position()
                bar=min(ax.patches,key=lambda p:abs(p.get_x()+p.get_width()/2-x))
                upper=max([bar.get_height()]+[float(segment[:,1].max()) for segment in segments
                          if len(segment) and abs(float(segment[:,0].mean())-x)<1e-6])
                label.set_position((x,upper+.018))
                label.set_va('bottom');label.set_color('black');label.set_fontsize(9.5)
                label.set_bbox(None)
                if name=='human_adaptation':label.set_fontsize(9)
            if len(fig.axes)>1:
                ax.text(.01,1.04,f'({chr(97+index)})',transform=ax.transAxes,fontsize=10,va='bottom')
    fig.savefig(FIGURES/f'{name}.pdf',bbox_inches='tight',pad_inches=.03)
    fig.savefig(FIGURES/f'{name}.png',bbox_inches='tight',pad_inches=.03)
    plt.close(fig)


def directed_assets():
    path=ROOT/'results/directed_network/metrics.csv'
    if not path.exists():return
    values=pd.read_csv(path).query("population == 'cnm-common'").set_index(['dataset','method'])
    names=[('CNM-GC-past32','CNM-GC (past 32 days)'),('CNM-GC-past32-LR','CNM-GC + scalar LR'),
           ('CATCH-ensemble','CATCH (3 seeds)'),('HNE-GC','HNE: prediction-gain network'),
           ('HNE-GC-null','HNE: null-centered gain'),('HNE-additive','HNE')]
    rows=[]
    for method,label in names:
        row=[label]
        for dataset in ['RESILIENT','BrisT1D-Open']:
            r=values.loc[(dataset,method)]
            row += [f'{r.auroc:.3f}',f'[{r.ci_low:.3f}, {r.ci_high:.3f}]',f'{r.within_person_auroc:.3f}']
        rows.append(row)
    table('directed_network.tex',['Method','RES AUC',r'95\% CI','Within','Bris AUC',r'95\% CI','Within'],rows,'lrrrrrr')


def ssaqs_assets():
    path=ROOT/'results/ssaqs_external/metrics.csv'
    if not path.exists():return
    frames={'SSAQS':pd.read_csv(path).set_index(['population','method'])}
    pm=ROOT/'results/pmdata_external/metrics.csv'
    if pm.exists():frames['PMData']=pd.read_csv(pm).set_index(['population','method'])
    names=[('DNB','DNB'),('MM-DEV-daily','MM-DEV'),('CNM-GC-past32','CNM-GC'),
           ('H-pooled-HGB',r'$H$ trees'),('HNE-pooled-HGB','All trees'),('HNE-history',r'$H$ only'),('HNE-additive','HNE')]
    rows=[]
    for method,label in names:
        row=[label]
        for cohort,metrics in frames.items():
            if ('common',method) not in metrics.index:
                row += ['--','--','--'];continue
            r=metrics.loc[('common',method)]
            row += [f'{r.auroc:.3f}',f'[{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}]',f'{r.within_person_auroc:.3f}']
        rows.append(row)
    columns=['Method']
    for cohort in frames:columns += [cohort+' AUC',r'95\% CI','Within']
    grouped_table('ssaqs_transfer.tex',['Method','AUROC',r'95\% CI','Within'],
                  [(cohort,[[r[0]]+r[1+3*i:4+3*i] for r in rows]) for i,cohort in enumerate(frames)],'lrrr')
    fig,axes=plt.subplots(len(frames),1,figsize=(3.45,8.1),layout='constrained',squeeze=False)
    for ax,(cohort,metrics) in zip(axes.ravel(),frames.items()):
        selected=[(m,l) for m,l in names if ('common',m) in metrics.index]
        values=np.asarray([metrics.loc[('common',m),'auroc'] for m,_ in selected])
        interval=np.asarray([[metrics.loc[('common',m),'auroc_ci_low'],metrics.loc[('common',m),'auroc_ci_high']] for m,_ in selected])
        colors=[str(v) for v in np.linspace(.85,.35,len(selected)-1)]+['black']
        bars=ax.bar(np.arange(len(selected)),values,width=.68,color=colors,
                    yerr=np.maximum(np.stack([values-interval[:,0],interval[:,1]-values]),0),
                    error_kw={'elinewidth':.7,'capsize':2})
        for i,(bar,value) in enumerate(zip(bars,values)):
            ax.text(bar.get_x()+bar.get_width()/2,value*.45,f'{value:.3f}',ha='center',va='center',
                    fontsize=7,color='white' if i>=len(selected)//2 else 'black',
                    bbox={'facecolor':colors[i],'edgecolor':'none','pad':.25})
        tick_labels=[l.replace('MM-DEV','MM-\nDEV').replace('CNM-GC','CNM-\nGC').replace(' trees','\ntrees').replace(' only','\nonly') for _,l in selected]
        ax.set(ylim=(0,1.10),xticks=np.arange(len(selected)),xticklabels=tick_labels,title=cohort,ylabel='AUROC')
        ax.set_yticks([0,.25,.5,.75,1]);ax.tick_params(axis='x',length=0,labelsize=6.5)
    save(fig,'ssaqs_transfer')


def college_development_assets():
    methods=[('results/college_experience/HNE-additive','HNE additive'),
             ('results/college_experience/History-HGB','History HGB'),
             ('results/college_temporal','Temporal BCE'),
             ('results/college_future_objective/future-values','Future values'),
             ('results/college_future_objective/future-moments','Future moments'),
             ('results/college_reference_units','Personal units'),
             ('results/college_rank_basis','Rank basis')]
    rows=[]
    for folder,label in methods:
        result=json.loads((ROOT/folder/'selection.json').read_text())
        record=result.get('validation',result.get('validation_metrics'))
        if record is None:
            frame=pd.read_csv(ROOT/folder/'validation.csv')
            from sklearn.metrics import roc_auc_score,average_precision_score
            record={'auroc':roc_auc_score(frame.y,frame.score),'ap':average_precision_score(frame.y,frame.score)}
        rows.append([label,f'{record["auroc"]:.3f}',f'{record["ap"]:.3f}'])
    table('college_learning.tex',['Model / objective','AUROC','AP'],rows,'lrr')
    values=pd.read_csv(ROOT/'results/college_future_objective/target_skill/metrics.csv')
    rows=[]
    names=[('training-constant','Training median'),('previous-seven-days','Previous seven days'),
           ('future-values-statistics','Future values'),('future-moments','Future moments')]
    for name,label in names:
        part=values[values.method.eq(name)].set_index('metric')
        rows.append([label]+[f'{part.loc[m,"value"]:.3f}' for m in
                    ['signed_mean_mae','log1p_variance_mae','tail_fraction_mae','standardized_huber']])
    table('future_targets.tex',['Target predictor','Mean','Log-var.','Tail','Huber'],rows,'lrrrr')
    repeated=pd.read_csv(ROOT/'results/reference_units_development/comparison.csv')
    table('reference_units.tex',['Cohort','Global scale','Personal units',r'$\Delta$'],
          [[r.dataset.replace('BrisT1D-Open','BrisT1D'),f'{r.original_temporal_auroc:.3f}',
            f'{r.reference_units_auroc:.3f}',f'{r.difference:+.3f}'] for r in repeated.itertuples()],'lrrr')


def structured_development_assets():
    metrics=pd.read_csv(ROOT/'results/family_target_comparison/metrics.csv').set_index(['dataset','method'])
    methods=[('HNE-additive','HNE'),('HNE-fg-independent','Independent families'),
             ('HNE-fg-constant-coupling','Constant family edges'),('HNE-fg-conditional-graph','Conditional family edges'),
             ('HNE-two-level','Equal two-level mixture'),('HNE-rhythm-H','HNE + hourly history'),
             ('HNE-rhythm','HNE + all hourly coordinates'),('History-HGB-rhythm','History HGB + hourly history')]
    rows=[]
    for method,label in methods:
        row=[label]
        for dataset in ['RESILIENT','BrisT1D-Open']:
            r=metrics.loc[(dataset,method)]
            row.extend([f'{r.auroc:.3f}',f'{r.within_person_auroc:.3f}'])
        rows.append(row)
    table('structured_development.tex',['Model','RES AUC','Within','Bris AUC','Within'],rows,'lrrrr')


def college_test_assets():
    path=ROOT/'results/college_experience/comparison/metrics.csv'
    if not path.exists():return
    metrics=pd.read_csv(path).query("population=='common'").set_index('method')
    summary=json.loads((path.parent/'summary.json').read_text())
    best=summary['strongest_control']
    macros={'CollegeN':summary['common_origins'],'CollegePeople':summary['test_people'],
            'CollegeAllN':summary['total_origins'],'CollegePositive':summary['test_positive'],
            'CollegeHNE':f'{metrics.loc["HNE-additive","auroc"]:.3f}',
            'CollegeFCM':f'{metrics.loc["FCM-3seed","auroc"]:.3f}',
            'CollegeCATCH':f'{metrics.loc["CATCH-3seed","auroc"]:.3f}',
            'CollegeBestName':best['method'].replace('-', ' '),
            'CollegeBestAUC':f'{best["auroc"]:.3f}',
            'CollegeBestDelta':f'{best["difference"]:+.3f}',
            'CollegeBestLow':f'{best["ci_low"]:.3f}',
            'CollegeBestHigh':f'{best["ci_high"]:.3f}'}
    primary=[('DNB','DNB'),('MM-DEV-daily','MM-DEV'),('FCM-3seed','FCM'),('CATCH-3seed','CATCH'),
             ('JAPE-event-3seed','JAPE (F1)'),('JAPE-event-AUC-3seed','JAPE (AUROC)'),
             ('History-LR','History LR'),('History-HGB','History HGB'),('HNE-additive','HNE')]
    rows=[]
    for method,label in primary:
        r=metrics.loc[method]
        rows.append([label,int(r.origins),f'{r.auroc:.3f}',f'[{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}]',
                     f'{r.within_person_auroc:.3f}',f'{r.ap:.3f}',f'{100*r.tpr_val_p90:.1f}',f'{100*r.fpr_val_p90:.1f}'])
    compact_comparison('college_test.tex',[[r[0]]+r[2:] for r in rows])
    variants=[('HNE-additive','HNE'),('HNE-history',r'$H$ only'),('HNE-no-network',r'No $N/NE$'),
              ('HNE-no-environment',r'No $E/NE$'),('HNE-context-append','Context appended'),
              ('HNE-temporal','Temporal BCE'),('HNE-reference-units','Personal units'),('HNE-rank-basis','Rank basis'),
              ('HNE-future-values','Future values'),('HNE-future-moments','Future moments'),
              ('HNE-fg-independent','Independent families'),('HNE-fg-constant-coupling','Constant family edges'),
              ('HNE-fg-conditional-graph','Conditional family edges'),('HNE-two-level','Equal two-level mixture'),
              ('HNE-rhythm-H','HNE + hourly history'),('HNE-rhythm','HNE + all hourly coordinates'),
              ('History-HGB-rhythm','History HGB + hourly history'),
              ('ZeroShot-HNE-additive','Frozen source HNE'),('ZeroShot-HNE-history',r'Frozen source $H$')]
    rows=[]
    for method,label in variants:
        r=metrics.loc[method]
        rows.append([label,f'{r.auroc:.3f}',f'[{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}]',
                     f'{r.within_person_auroc:.3f}',f'{r.ap:.3f}'])
    table('college_test_variants.tex',['Model','AUROC',r'95\% CI','Within','AP'],rows,'lrrrr')
    fig,ax=plt.subplots(figsize=(7.05,2.8),layout='constrained')
    values=np.array([metrics.loc[m,'auroc'] for m,_ in primary])
    interval=np.array([[metrics.loc[m,'auroc_ci_low'],metrics.loc[m,'auroc_ci_high']] for m,_ in primary])
    bars=ax.bar(np.arange(len(primary)),values,color=['.72']*8+['black'],width=.68)
    ax.errorbar(np.arange(len(primary)),values,yerr=np.maximum(np.array([values-interval[:,0],interval[:,1]-values]),0),
                fmt='none',color='black',capsize=3,linewidth=.8)
    for bar,value,upper in zip(bars,values,interval[:,1]):
        ax.text(bar.get_x()+bar.get_width()/2,upper+.025,f'{value:.3f}',ha='center',fontsize=9)
    ax.set(xticks=np.arange(len(primary)),xticklabels=[label.replace(' ','\n',1) for _,label in primary],
           ylim=(0,1),ylabel='AUROC',title='College Experience')
    ax.tick_params(axis='x',labelsize=8)
    save(fig,'college_auroc')
    transfer=ROOT/'results/college_transfer/test_metrics.csv'
    if transfer.exists():
        data=pd.read_csv(transfer)
        rows=[]
        for n,part in data.groupby('training_people',sort=True):
            by=part.set_index('condition')
            rows.append([int(n)]+[f'{by.loc[c,"auroc"]:.3f}' for c in
                        ['target-only-basis','source-basis-zero','source-basis-prior']])
        table('college_adaptation.tex',['People','Target','Source basis','Source prior'],rows,'lrrr')
    delayed=ROOT/'results/college_delayed/comparison/metrics.csv'
    if delayed.exists():
        data=pd.read_csv(delayed).query("population=='common'").set_index('method')
        cfg=json.loads((ROOT/'configs/college_delayed.json').read_text())
        rows=[]
        for method in cfg['source_methods']:
            label=method.replace('HNE-additive','HNE').replace('HNE-history',r'$H$ only').replace('-score-LR-3seed',' score LR').replace('-score-LR',' score LR').replace('-3seed','')
            row=[label]
            for condition in ['frozen','intercept','affine','coefficients']:
                key=method+'__'+condition
                row.append(f'{data.loc[key,"auroc"]:.3f}' if key in data.index else '--')
            rows.append(row)
        table('college_delayed.tex',['Source model','Frozen','Intercept','Affine','Coefficients'],rows,'lrrrr')
        for condition,suffix in [('frozen','Frozen'),('intercept','Intercept'),('affine','Affine'),('coefficients','Coefficients')]:
            macros['CollegeDelayed'+suffix]=f'{data.loc["HNE-additive__"+condition,"auroc"]:.3f}'
        best_name=data.auroc.idxmax()
        macros['CollegeDelayedBestName']=best_name.replace('__', ' / ').replace('-', ' ')
        macros['CollegeDelayedBestAUC']=f'{data.loc[best_name,"auroc"]:.3f}'
    (TABLES/'college_numbers.tex').write_text('\n'.join(
        '\\newcommand{\\'+key+'}{'+str(value)+'}' for key,value in macros.items()),encoding='utf-8')


def inference_assets():
    path=ROOT/'results/inference_cost/metrics.csv'
    if not path.exists():return
    metrics=pd.read_csv(path).set_index('method')
    methods=[('DNB','DNB'),('MM-DEV-daily','MM-DEV'),('FCM-3seed','FCM (3 seeds)'),
             ('CATCH-3seed','CATCH (3 seeds)'),('History-HGB','History HGB'),
             ('HNE-from-history','HNE from history'),('HNE-cached-head','HNE cached head')]
    rows=[]
    for method,label in methods:
        r=metrics.loc[method]
        rows.append([label,f'{r.median_ms_per_origin:.2f}',
                     f'[{r.q25_ms_per_origin:.2f}, {r.q75_ms_per_origin:.2f}]',
                     f'{r.checkpoint_bytes/1024:.1f}'])
    table('inference_cost.tex',['Method','ms/origin','IQR','KiB'],rows,'lrrr')
    fig,ax=plt.subplots(figsize=(7.05,2.6),layout='constrained')
    values=np.array([metrics.loc[m,'median_ms_per_origin'] for m,_ in methods])
    bars=ax.bar(np.arange(len(methods)),values,width=.65,color=['.7']*5+['black','.3'])
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.5,f'{value:.2f}',ha='center',fontsize=9)
    ax.set(xticks=np.arange(len(methods)),xticklabels=['DNB','MM-DEV','FCM ×3','CATCH ×3','History HGB','HNE','HNE head'],
           ylabel='CPU ms / origin',ylim=(0,max(values)*1.18))
    save(fig,'inference_cost')


def esa_assets():
    completed=[]
    source_rows, target_rows, event_rows=[],[],[]
    targets={}
    target_methods=[('DNB','DNB'),('MM-DEV-daily','MM-DEV'),('FCM-3seed','FCM'),
                    ('CATCH-3seed','CATCH'),('History-LR','History LR'),('History-HGB','History HGB'),
                    ('HNE-additive','HNE'),('HNE-history',r'$H$ only'),
                    ('HNE-no-network',r'No $N/NE$'),('HNE-no-environment',r'No $E/NE$')]
    for mission in [1,2]:
        folder=ROOT/f'results/esa_native/ESA-Mission{mission}'
        path=folder/'test_metrics.csv'
        if not path.exists():continue
        completed.append((mission,folder))
        metrics=pd.read_csv(path)
        source=metrics[metrics.population.eq('primary_common')].set_index('method')
        for method,label in [('DNB','DNB'),('ZeroShot-HNE-history',r'Source $H$'),('ZeroShot-HNE-additive','Source HNE')]:
            r=source.loc[method]
            source_rows.append([mission,label,int(r.origins),int(r.positive),f'{r.auroc:.3f}',
                                f'[{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}]',f'{r.ap:.3f}',
                                f'{100*r.tpr_val_p90:.1f}',f'{100*r.fpr_val_p90:.1f}'])
        target=metrics[metrics.population.eq('target_common')].set_index('method')
        targets[mission]=target
        for method,label in target_methods:
            r=target.loc[method]
            target_rows.append([mission,label,int(r.origins),f'{r.auroc:.3f}',
                                f'[{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}]',f'{r.ap:.3f}'])
        rare=pd.read_csv(folder/'benign_event_false_alarms.csv')
        rare=rare[rare.population.eq('primary_common')].set_index(['method','subgroup'])
        for method,label in [('ZeroShot-HNE-history',r'Source $H$'),('ZeroShot-HNE-additive','Source HNE')]:
            r=rare.loc[(method,'benign_rare')];o=rare.loc[(method,'ordinary')]
            event_rows.append([mission,label,int(r.origins),f'{100*r.fpr:.1f}',
                               f'[{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]',int(o.origins),
                               f'{100*o.fpr:.1f}' if o.origins else '--'])
    if not completed:return
    source_groups=[];target_groups=[];event_groups=[]
    for mission in targets:
        part=[r for r in source_rows if r[0]==mission]
        source_groups.append((f'Mission {mission}: $n={part[0][2]}$, $n_+={part[0][3]}$',
            [[r[1],r'\shortstack[r]{'+r[4]+r' \\ {}'+r[5]+'}',r[6],r[7],r[8]] for r in part]))
        part=[r for r in target_rows if r[0]==mission]
        target_groups.append((f'Mission {mission}: $n={part[0][2]}$',[[r[1]]+r[3:] for r in part]))
        part=[r for r in event_rows if r[0]==mission]
        event_groups.append((f'Mission {mission}: rare $n={part[0][2]}$, ordinary $n={part[0][5]}$',
            [[r[1],r[3],r[4],r[6]] for r in part]))
    grouped_table('esa_source.tex',['Model',r'\shortstack{AUROC\\{}[95\% CI]}','AP',r'TPR (\%)',r'FPR (\%)'],source_groups,'lrrrr')
    grouped_table('esa_target.tex',['Model','AUROC',r'95\% CI','AP'],target_groups,'lrrr')
    grouped_table('esa_benign.tex',['Model',r'\shortstack{Rare\\FPR (\%)}',r'95\% CI',r'\shortstack{Ordinary\\FPR (\%)}'],event_groups,'lrrr')
    fig,axes=plt.subplots(1,len(completed),figsize=(3.5*len(completed),2.7),layout='constrained',squeeze=False)
    for ax,(mission,folder) in zip(axes.ravel(),completed):
        frame=pd.read_csv(folder/'test_metrics.csv').query("population=='target_common'").set_index('method')
        selected=target_methods[:7]
        values=np.array([frame.loc[m,'auroc'] for m,_ in selected])
        interval=np.array([[frame.loc[m,'auroc_ci_low'],frame.loc[m,'auroc_ci_high']] for m,_ in selected])
        bars=ax.bar(np.arange(len(selected)),values,color=['.72']*6+['black'],width=.7)
        ax.errorbar(np.arange(len(selected)),values,yerr=np.maximum(np.array([values-interval[:,0],interval[:,1]-values]),0),
                    fmt='none',color='black',capsize=2,linewidth=.6)
        for index,(bar,value) in enumerate(zip(bars,values)):
            ax.text(bar.get_x()+bar.get_width()/2,value*.5,f'{value:.3f}',ha='center',va='center',
                    color='white' if index==6 else 'black',fontsize=8)
        ax.set(xticks=np.arange(len(selected)),xticklabels=[label.replace('History ','H-') for _,label in selected],
               ylim=(0,1),ylabel='AUROC',title=f'ESA Mission {mission}')
        ax.tick_params(axis='x',rotation=45,labelsize=7)
    save(fig,'esa_auroc')


def external_assets():
    metrics=pd.read_csv(ROOT/'results/human_rhythms/comparison/metrics.csv')
    common=metrics[metrics.population.eq('common')].set_index('method')
    primary=[('DNB','DNB'),('MM-DEV-daily','MM-DEV'),('FCM-3seed','FCM (3 seeds)'),
             ('CATCH-3seed','CATCH (3 seeds)'),('History-LR','History LR'),
             ('History-HGB','History HGB'),('HNE-additive','HNE')]
    rows=[]
    for method,label in primary:
        r=common.loc[method]
        rows.append([label,f'{r.auroc:.3f}',f'[{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}]',
            f'{r.within_person_auroc:.3f}',f'{r.ap:.3f}',f'{100*r.tpr_val_p90:.1f}',f'{100*r.fpr_val_p90:.1f}'])
    compact_comparison('external_main.tex',rows)
    variants=[('HNE-additive','HNE'),('HNE-history','$H$ only'),('HNE-no-network','No $N$ / $NE$'),
        ('HNE-no-environment','No $E$ / $NE$'),('HNE-context-append','Context appended'),
        ('HNE-context-grid','Context ridge selected'),('ZeroShot-HNE-additive','Frozen source HNE'),
        ('ZeroShot-HNE-history','Frozen source $H$'),('MultiDomain-HNE','Pooled-domain HNE')]
    table('external_variants.tex',['Model','AUROC','Within','AP'],
        [[label,f'{common.loc[m,"auroc"]:.3f}',f'{common.loc[m,"within_person_auroc"]:.3f}',f'{common.loc[m,"ap"]:.3f}']
         for m,label in variants],'lrrr')
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.65),layout='constrained')
    panels=[(['DNB','FCM-3seed','CATCH-3seed','History-HGB','HNE-additive'],
             ['DNB','FCM','CATCH','HGB','HNE'],'Target-only training'),
            (['HNE-additive','HNE-context-grid','ZeroShot-HNE-additive','MultiDomain-HNE'],
             ['Target','Ridge','Source','Pooled'],'Human Rhythms')]
    for ax,(methods,labels,title) in zip(axes,panels):
        part=common.loc[methods];v=part.auroc.to_numpy()
        colors=[str(c) for c in np.linspace(.82,0,len(v))]
        bars=ax.bar(np.arange(len(v)),v,color=colors,width=.62,
            yerr=np.vstack([v-part.auroc_ci_low,part.auroc_ci_high-v]),
            error_kw={'elinewidth':.8,'capsize':2})
        for i,(bar,value) in enumerate(zip(bars,v)):
            ax.text(bar.get_x()+bar.get_width()/2,value*.5,f'{value:.3f}',ha='center',va='center',
                color='black' if i<2 else 'white',fontsize=8)
        ax.set(ylim=(0,1),ylabel='AUROC',title=title,xticks=np.arange(len(v)),xticklabels=labels)
        ax.set_yticks([0,.25,.5,.75,1]);ax.tick_params(axis='x',length=0)
    save(fig,'external_auroc')
    transfer=pd.read_csv(ROOT/'results/human_rhythms_transfer/test_metrics.csv')
    names=[('target-only-basis','Target'),('source-basis-zero','Source basis'),('source-basis-prior','Source prior')]
    rows=[]
    for budget,part in transfer.groupby('budget'):
        idx=part.set_index('condition')
        rows.append([int(part.training_people.iloc[0])]+[f'{idx.loc[m,"auroc"]:.3f}' for m,_ in names])
    table('human_adaptation.tex',['People','Target','Source basis','Source prior'],rows,'lrrr')
    fig,ax=plt.subplots(figsize=(3.45,2.6),layout='constrained')
    x=np.arange(3)
    for j,(method,label) in enumerate(names):
        p=transfer[transfer.condition.eq(method)].sort_values('budget');v=p.auroc.to_numpy()
        bars=ax.bar(x+(j-1)*.28,v,width=.25,color=['.82','.45','black'][j],label=label,
            yerr=np.vstack([v-p.auroc_ci_low,p.auroc_ci_high-v]),error_kw={'elinewidth':.6,'capsize':1.5})
        for bar,value in zip(bars,v):
            ax.text(bar.get_x()+bar.get_width()/2,value*.5,f'{value:.2f}',ha='center',va='center',fontsize=7,
                color='black' if j==0 else 'white')
    ax.set(ylim=(0,1),ylabel='AUROC',xlabel='Target training participants',xticks=x,xticklabels=['22','43','85'])
    ax.set_yticks([0,.25,.5,.75,1]);ax.tick_params(axis='x',length=0)
    ax.legend(frameon=False,ncol=3,fontsize=6.5,loc='upper center')
    save(fig,'human_adaptation')
    rows=[]
    for model in ['FCM','CATCH']:
        for seed in [20261110,20261111,20261112]:
            native=common.loc[f'{model}-s{seed}']
            readout=common.loc[f'{model}-score-LR-s{seed}']
            rows.append([f'{model} {seed}']+[f'{native[m]:.3f}/{readout[m]:.3f}'
                for m in ['auroc','within_person_auroc','ap']])
    table('external_seeds.tex',['Model / seed','AUROC','Within','AP'],rows,'lrrr')
    lead=pd.read_csv(ROOT/'results/lead_time/metrics.csv');rows=[]
    for (dataset,gap),part in lead.groupby(['dataset','gap'],sort=False):
        idx=part.set_index('model');h=idx.loc['HNE-additive']
        rows.append([dataset,gap,int(h.n),f'{idx.loc["DNB","auroc"]:.3f}',f'{idx.loc["FCM-ensemble","auroc"]:.3f}',
            f'{idx.loc["CATCH-ensemble","auroc"]:.3f}',f'{h.auroc:.3f}',f'{h.within_person_auroc:.3f}'])
    grouped_table('lead_time.tex',['Gap','DNB','FCM','CATCH','HNE','Within'],
        [(f'{dataset}: $n={next(r[2] for r in rows if r[0]==dataset)}$',
          [[r[1]]+r[3:] for r in rows if r[0]==dataset]) for dataset in dict.fromkeys(r[0] for r in rows)],'rrrrrr')
    cohorts=pd.read_csv(ROOT/'results/cohort_characteristics.csv')
    cohorts=cohorts[cohorts.row_type.eq('cohort_summary')];rows=[]
    for r in cohorts.itertuples():
        label={'RESILIENT':'RESILIENT','BrisT1D-Open':'BrisT1D','Human-Rhythms':'Human Rhythms','CARE':'CARE'}.get(r.dataset,r.dataset)
        if r.dataset=='Human-Rhythms':label+=' ('+r.split+')'
        def dimension(a,b):return str(int(a)) if a==b else f'{int(a)}--{int(b)}'
        rows.append([label,f'{int(r.original_entities)}/{int(r.scorable_entities)}',
            dimension(r.daily_state_dim_min,r.daily_state_dim_max),dimension(r.context_dim_min,r.context_dim_max),
            f'{r.calendar_span_days_median:g}',f'{r.state_observed_days_median:g}',int(r.scorable_origins),int(r.positive_origins)])
    for split in ['train','validation','test']:
        folder=ROOT/'data/college_study'/split
        summary_path=folder/'summary.json'
        if not summary_path.exists():continue
        summary=json.loads(summary_path.read_text())
        spans,observed=[],[]
        for path in folder.glob('*_daily.npz'):
            with np.load(path) as z:
                spans.append(len(z['dates']))
                observed.append(np.isfinite(z['states']).any(1).sum())
        rows.append([f'College ({split})',f'{summary["people"]}/{summary["people_with_samples"]}',9,4,
                     f'{np.median(spans):g}',f'{np.median(observed):g}',summary['samples'],summary['positive']])
    table('cohorts.tex',['Cohort / split','Units','$P$','$Q$','Span','Observed','$n$','$n^+$'],rows,'lrrrrrrr')


def compact_assets():
    development = pd.read_csv(ROOT/'results/unified_daily/summary/metrics.csv')
    development = development[development.population.eq('common')].set_index(['dataset','method'])
    external = {}
    for cohort, folder in [('Human', 'human_rhythms'), ('College', 'college_experience')]:
        values = pd.read_csv(ROOT/f'results/{folder}/comparison/metrics.csv')
        external[cohort] = values[values.population.eq('common')].set_index('method')
    names = [('DNB','DNB'), ('MM-DEV-daily','MM-DEV'), ('FCM-3seed','FCM'),
             ('CATCH-3seed','CATCH'), ('JAPE-event-3seed','JAPE (F1)'),
             ('JAPE-event-AUC-3seed','JAPE (AUROC)'), ('History-HGB','History HGB'), ('HNE-additive','HNE')]
    table('heldout.tex', ['Method','Human Rhythms','College'],
          [[label]+[f'{frame.loc[method,"auroc"]:.3f}' if method in frame.index else '--'
                    for frame in external.values()] for method,label in names], 'lrr',rank_columns=[1,2],highlight='HNE')
    groups = [('HNE-additive','HNE'), ('HNE-history','$H$ only'),
              ('HNE-no-network','No $N/NE$'), ('HNE-no-environment','No $E/NE$'),
              ('HNE-context-append','Context appended')]
    rows = []
    for method,label in groups:
        rows.append([label]+[f'{development.loc[(cohort,method),"auroc"]:.3f}'
                            for cohort in ['RESILIENT','BrisT1D-Open']]
                    +[f'{frame.loc[method,"auroc"]:.3f}' for frame in external.values()])
    table('modules.tex', ['Model','RES','Bris','Human','College'],rows,'lrrrr',rank_columns=[1,2,3,4],highlight='HNE')
    table('college_core_variants.tex', ['Model','AUROC','Within','AP'],
          [[label]+[f'{external["College"].loc[method,metric]:.3f}'
                    for metric in ['auroc','within_person_auroc','ap']] for method,label in groups],'lrrr')

    transfer = pd.read_csv(ROOT/'results/transfer/metrics.csv')
    transfer = transfer[transfer.population.eq('common') | transfer.target.eq('CARE-v6')]
    rows=[]
    short={'RESILIENT':'RES','BrisT1D-Open':'Bris','RESILIENT+BrisT1D-Open':'RES+Bris',
           'Humans':'RES+Bris','CARE-v6':'CARE'}
    for (source,target),part in transfer.groupby(['source','target'],sort=False):
        frame=part.set_index('method');h=frame.loc['HNE-additive']
        rows.append([short.get(source,source),short.get(target,target),int(h.origins)]
                    +[f'{frame.loc[m,"global_auroc"]:.3f}' for m in ['DNB','HNE-history','HNE-additive']]
                    +[f'[{h.global_auroc_ci_low:.3f}, {h.global_auroc_ci_high:.3f}]'])
    for mission in [1,2]:
        values=pd.read_csv(ROOT/f'results/esa_native/ESA-Mission{mission}/test_metrics.csv')
        frame=values[values.population.eq('primary_common')].set_index('method')
        h=frame.loc['ZeroShot-HNE-additive']
        rows.append(['RES+Bris',f'ESA M{mission}',int(h.origins)]
                    +[f'{frame.loc[m,"auroc"]:.3f}' for m in ['DNB','ZeroShot-HNE-history','ZeroShot-HNE-additive']]
                    +[f'[{h.auroc_ci_low:.3f}, {h.auroc_ci_high:.3f}]'])
    table('crossdomain.tex',['Source','Target','$n$','DNB','Source $H$','Source HNE',r'HNE 95\% CI'],rows,'llrrrrr',rank_columns=[3,4,5],rank_across=True,shade_column=5)


def main():
    metrics=pd.read_csv(ROOT/'results/unified_daily/summary/metrics.csv')
    metrics=metrics[metrics.population.eq('common')]
    datasets=['RESILIENT','BrisT1D-Open']
    lookup={(r.dataset,r.method):r for r in metrics.itertuples()}
    main_methods=['DNB','FCM-ensemble','CATCH-ensemble','JAPE-event','JAPE-event-AUC','History-HGB','HNE-additive']
    pretty={'HNE-additive':'HNE','History-LR':'History LR','History-HGB':'History HGB',
            'FCM-ensemble':'FCM (3 seeds)','CATCH-ensemble':'CATCH (3 seeds)',
            'JAPE-event':'JAPE (F1)','JAPE-event-AUC':'JAPE (AUROC)'}
    rows=[]
    for method in main_methods:
        row=[pretty.get(method,method)]
        for dataset in datasets:
            r=lookup[dataset,method]
            row += [f'{r.auroc:.3f}',f'[{r.ci_low:.3f}, {r.ci_high:.3f}]',
                    f'{r.ap_on_validation_rank:.3f}',f'{r.within_person_auroc:.3f}']
        rows.append(row)
    table('main.tex',['Method','AUROC',r'95\% CI','AP','Within','AUROC',r'95\% CI','AP','Within'],rows,'l'+'r'*8,
        r'& \multicolumn{4}{c}{RESILIENT} & \multicolumn{4}{c}{BrisT1D-Open}',rank_columns=[1,3,4,5,7,8],highlight='HNE')

    fig,axes=plt.subplots(1,2,figsize=(7.05,2.55),layout='constrained')
    plotted=['DNB','FCM-ensemble','CATCH-ensemble','JAPE-event-AUC','History-HGB','HNE-additive']
    for ax,dataset in zip(axes,datasets):
        rs=[lookup[dataset,m] for m in plotted];values=np.array([r.auroc for r in rs])
        lower=np.maximum(0,values-np.array([r.ci_low for r in rs]));upper=np.maximum(0,np.array([r.ci_high for r in rs])-values)
        bars=ax.bar(np.arange(len(rs)),values,color=['.85','.72','.58','.43','.25','black'],width=.62,
            yerr=np.vstack((lower,upper)),error_kw={'elinewidth':.8,'capsize':2,'capthick':.8})
        for index,(bar,value) in enumerate(zip(bars,values)):
            ax.text(bar.get_x()+bar.get_width()/2,value*.5,f'{value:.3f}',ha='center',va='center',
                color='black' if index<2 else 'white',fontsize=8)
        ax.set(ylim=(0,1),ylabel='AUROC',title=dataset,xticks=np.arange(len(rs)),xticklabels=['DNB','FCM','CATCH','JAPE','HGB','HNE'])
        ax.set_yticks([0,.25,.5,.75,1]);ax.tick_params(axis='x',length=0)
    save(fig,'main_auroc')

    groups=[('HNE-additive','HNE'),('HNE-history','$H$ only'),('HNE-no-network','No $N$ / $NE$'),
        ('HNE-no-environment','No $E$ / $NE$'),('HNE-context-append','Context appended'),
        ('HNE-no-amplitude','No amplitude / SD'),('HNE-no-location','No location / drift'),
        ('HNE-no-memory','No temporal memory'),('HNE-no-tail','No tail frequency'),
        ('HNE-scale8','8 days'),('HNE-scale16','16 days'),('HNE-scale32','32 days')]
    rows=[[label]+[f'{lookup[d,m].auroc:.3f}' for d in datasets] for m,label in groups[:5] if (datasets[0],m) in lookup]
    table('ablations.tex',['Representation','RESILIENT','BrisT1D'],rows,'lrr')
    table('temporal_ablations.tex',['Temporal descriptors','RESILIENT','BrisT1D'],
        [[label]+[f'{lookup[d,m].auroc:.3f}' for d in datasets] for m,label in groups
         if m in ['HNE-additive','HNE-no-amplitude','HNE-no-location','HNE-no-memory','HNE-no-tail','HNE-scale8','HNE-scale16','HNE-scale32']],'lrr',rank_columns=[1,2],highlight='HNE')

    headnames=[('HNE-additive','Additive'),('HNE-HGB','Boosted'),('HNE-set','Set encoder'),('HNE-temporal','Temporal encoder'),('HNE-pairwise','Within-person ranking')]
    rows=[[label]+[f'{lookup[d,m].auroc:.3f}' for d in datasets] for m,label in headnames if (datasets[0],m) in lookup]
    table('heads.tex',['Risk head','RESILIENT','BrisT1D'],rows,'lrr',rank_columns=[1,2],highlight='Additive')

    rows=[]
    for dataset in datasets:
        for m in ['DNB','FCM','CATCH','HNE-additive']:
            r=lookup[dataset,m]
            rows.append([dataset if m=='DNB' else '',pretty.get(m,m),f'{100*r.sensitivity:.1f}',f'{100*r.false_positive_rate:.1f}'])
    table('operating_points.tex',['Cohort','Method',r'TPR (\%)',r'FPR (\%)'],rows,'llrr')

    rows=[]
    for dataset in datasets:
        for method in ['DNB-tuned','MM-DEV-daily','MM-DEV-daily-score-LR','Missingness-LR']:
            if (dataset,method) not in lookup:continue
            r=lookup[dataset,method]
            rows.append([dataset,method.replace('MM-DEV-daily','MM-DEV'),r.origins,f'{r.auroc:.3f}',f'[{r.ci_low:.3f}, {r.ci_high:.3f}]'])
    grouped_table('additional_baselines.tex',['Method','$n$','AUROC',r'95\% CI'],
        [(dataset,[r[1:] for r in rows if r[0]==dataset]) for dataset in datasets],'lrrr')

    transfer_path=ROOT/'results/transfer/metrics.csv'
    if transfer_path.exists():
        transfer=pd.read_csv(transfer_path)
        transfer.to_csv(TABLES/'transfer_values.csv',index=False)
        rows=[]
        selected=transfer.population.eq('common') | (transfer.source.eq('Humans') & transfer.target.eq('CARE-v6'))
        for (source,target),part in transfer[selected].groupby(['source','target'],sort=False):
            mapping=part.set_index('method')
            labels={'RESILIENT':'RES','BrisT1D-Open':'Bris','Humans':'RES + Bris','CARE-v6':'CARE','TOLIFE-CSA':'TOLIFE'}
            row=[labels.get(source,source),labels.get(target,target),int(part.iloc[0].origins)]
            for method in ['DNB','HNE-history','HNE-additive']:
                r=mapping.loc[method]
                row += [f'{r.global_auroc:.3f}',f'[{r.global_auroc_ci_low:.3f}, {r.global_auroc_ci_high:.3f}]']
            rows.append(row)
        table('transfer.tex',['Source','Target','$n$','DNB',r'95\% CI','$H$ only',r'95\% CI','HNE',r'95\% CI'],rows,'llrrrrrrr')
    care_path=ROOT/'results/care_development/fewshot_metrics.csv'
    if care_path.exists():
        care=pd.read_csv(care_path);rows=[]
        conditions=['target-only-basis','source-basis-zero','source-basis-prior']
        for fraction,part in care.groupby('fraction'):
            mapping=part.set_index('condition')
            rows.append([f'{100*fraction:.0f}\\%']+[f"{mapping.loc[c,'auroc']:.3f}" for c in conditions])
        table('care_adaptation.tex',['Assets','Target','Source basis','Source prior'],rows,'lrrr')
    robust_path=ROOT/'results/robustness/summary.csv'
    if robust_path.exists():
        robust=pd.read_csv(robust_path);rows=[]
        for dataset in datasets:
            for condition in ['missing10','missing25','noise025']:
                subset=robust[robust.dataset.eq(dataset)&robust.condition.eq(condition)].set_index('model')
                rows.append([dataset,condition]+[f"{subset.loc[m,'auroc']:.3f} $\\pm$ {subset.loc[m,'sd']:.3f}" for m in ['DNB','History-HGB','HNE-additive']])
        condition_names={'missing10':r'10\% missing','missing25':r'25\% missing','noise025':r'Noise $0.25$'}
        grouped_table('robustness.tex',['Perturbation','DNB','History HGB','HNE'],
            [(dataset,[[condition_names[r[1]]]+r[2:] for r in rows if r[0]==dataset]) for dataset in datasets],'lrrr')
        fig,axes=plt.subplots(1,2,figsize=(7.05,2.6),layout='constrained')
        for ax,dataset in zip(axes,datasets):
            x=np.arange(3);sub=robust[robust.dataset.eq(dataset)]
            for j,(method,color,label) in enumerate([('DNB','.8','DNB'),('History-HGB','.5','History HGB'),('HNE-additive','black','HNE')]):
                selection=sub[sub.model.eq(method)].set_index('condition').loc[['missing10','missing25','noise025']]
                values=selection.auroc.to_numpy();bars=ax.bar(x+(j-1)*.24,values,width=.22,color=color,
                    yerr=selection.sd.to_numpy(),error_kw={'elinewidth':.6,'capsize':2},label=label)
                for bar,value in zip(bars,values):
                    ax.text(bar.get_x()+bar.get_width()/2,value*.5,f'{value:.2f}',ha='center',va='center',fontsize=7,
                        color='white' if j==2 else 'black')
            ax.set(ylim=(0,1),xticks=x,xticklabels=['10% missing','25% missing',r'Noise $\\sigma=.25$'.replace('\\\\','\\')],ylabel='AUROC',title=dataset)
            ax.set_yticks([0,.25,.5,.75,1]);ax.tick_params(axis='x',length=0)
        axes[0].legend(frameon=False,fontsize=7,ncol=3,loc='upper center')
        save(fig,'robustness')
    external_assets()
    esa_assets()
    inference_assets()
    college_test_assets()
    ssaqs_assets()
    compact_assets()
    print(json.dumps(dict(tables=[p.name for p in TABLES.glob('*.tex')],figures=[p.name for p in FIGURES.glob('*.pdf')]),indent=2))


if __name__=='__main__':main()
