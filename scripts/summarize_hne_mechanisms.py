"""Summarize routing and temporal/transfer experiments from saved predictions."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_hne_routed import fold_auc
RESULT=ROOT/'results/hne_routed'


def summarize():
    rows=[];counts=[];training=[];temporal=[]
    for ds in ['RESILIENT','BrisT1D-Open']:
        for variant in ['learned','path_policy','context_path','context_fixed','context_plain','context_free','context_free_fixed','context_free_plain','context_free_rank','plain','native','fixed']:
            for k in range(5):
                selection=RESULT/ds/variant/f'fold{k}'/'selection.json'
                if not selection.exists():continue
                for source in json.loads(selection.read_text())['runs']:
                    folder=ROOT/source;seed=int(folder.name.removeprefix('seed'))
                    result=json.loads((folder/'result.json').read_text())
                    training.append(dict(dataset=ds,method=variant,fold=k,seed=seed,
                        validation_auroc=result['validation']['auroc'],
                        held_auroc=result['held_person']['auroc'],best_epoch=result['best_epoch'],
                        seconds=result['fit_seconds'],trainable_parameters=result.get('active_trainable_parameters')))
                    frame=pd.read_csv(folder/'held_person.csv')
                    if 'route_1' not in frame:continue
                    paths=[''.join('HNE'[int(i)] for i in path) for path in frame[['route_1','route_2','route_3']].to_numpy()]
                    freq=pd.Series(paths).value_counts();proportion=freq/len(frame)
                    rows.append(dict(dataset=ds,method=variant,fold=k,seed=seed,n=len(frame),unique_paths=len(freq),
                        path_entropy=float(-(proportion*np.log(proportion)).sum()),
                        dominant_path=freq.index[0],dominant_fraction=float(proportion.iloc[0])))
                    for name,count in freq.items():counts.append(dict(dataset=ds,method=variant,fold=k,seed=seed,path=name,count=count))
        for p in (RESULT/'temporal'/ds).glob('*.csv'):
            if p.name=='fold_metrics.csv':continue
            frame=pd.read_csv(p,dtype={'person_id':str})
            method=next(m for m in ['context_free_plain','context_free','path_policy','context_path','context_plain','plain'] if p.stem.startswith(m+'_'))
            condition=p.stem[len(method)+1:]
            temporal.append(dict(dataset=ds,method=method,condition=condition,n=len(frame),
                auroc=fold_auc(frame,'probability'),
                ap=sum(len(g)*average_precision_score(g.y,g.probability) for _,g in frame.groupby('fold'))/len(frame)))
    out=RESULT/'summary';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'routing_distribution.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'path_counts.csv',index=False)
    pd.DataFrame(training).to_csv(out/'selected_seed_results.csv',index=False)
    pd.DataFrame(temporal).to_csv(out/'temporal_metrics.csv',index=False)
    if temporal:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        figure,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
        table=pd.DataFrame(temporal)
        primary=json.loads((RESULT/'architecture_selection.json').read_text())['primary_variant']
        order=['original','within_day_shuffle','day_order_shuffle'];labels=['Original','Within-day shuffle','Day-order shuffle']
        for axis,ds in zip(axes,['RESILIENT','BrisT1D-Open']):
            control='context_free_plain' if primary=='context_free' else 'context_plain'
            for offset,method,color,label in [(-.18,control,'.72','Context head'),(.18,primary,'black','HNE')]:
                selected=table[table.dataset.eq(ds)&table.method.eq(method)].set_index('condition').reindex(order)
                bars=axis.bar(np.arange(3)+offset,selected.auroc,width=.34,color=color,label=label)
                axis.bar_label(bars,labels=[f'{x:.3f}' for x in selected.auroc],padding=3,fontsize=8)
            axis.set_xticks(np.arange(3),labels,rotation=12);axis.set_ylim(0,1);axis.set_ylabel('AUROC');axis.set_title(ds);axis.legend(frameon=False)
        figure.savefig(out/'temporal_auroc.png',dpi=250);figure.savefig(out/'temporal_auroc.pdf');plt.close(figure)
    print(pd.DataFrame(training).groupby(['dataset','method']).held_auroc.agg(['size','mean','std']).to_string())


if __name__=='__main__':summarize()
