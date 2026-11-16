"""Numerical bars and a LaTeX table for the completed routed-HNE experiments."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_hne_routed import fold_auc
RESULT=ROOT/'results/hne_routed';OUT=RESULT/'summary'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def bar_panels(tables,orders,labels,titles,name,black):
    figure,axes=plt.subplots(1,2,figsize=(10.5,3.9),layout='constrained')
    for axis,table,order,title in zip(axes,tables,orders,titles):
        values=table.set_index('method').reindex(order).auroc
        bars=axis.bar(np.arange(len(values)),values,width=.64,color=['black' if m in black else '.72' for m in order])
        axis.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=4)
        axis.set_xticks(np.arange(len(values)),[labels.get(m,m) for m in order],fontsize=9)
        axis.set_ylim(0,1);axis.set_ylabel('AUROC');axis.set_title(title)
    figure.savefig(OUT/f'{name}.png',dpi=250);figure.savefig(OUT/f'{name}.pdf');plt.close(figure)


def build():
    metrics=pd.read_csv(OUT/'metrics.csv');cohorts=['RESILIENT','BrisT1D-Open']
    primary=json.loads((RESULT/'architecture_selection.json').read_text())['primary_variant']
    order=['context_plain','context_fixed','context_path_constant','context_path'] if primary=='context_path' else ['learned','path_policy_constant','route_controls','path_policy']
    if primary=='context_free':order=['context_free_plain','context_free_fixed','context_free_constant','context_free']
    if primary=='context_free' and 'context_free_rank' in metrics.method.values:order.append('context_free_rank')
    labels={'learned':'Joint routing','fixed':'Fixed H-N-E','best_constant_path':'Best constant path',
            'route_controls':'Direct head','path_policy':'HNE','path_policy_constant':'Best constant path',
            'context_plain':'Matched context head','context_fixed':'Fixed H-N-E','context_path_constant':'Best constant path','context_path':'HNE',
            'context_free_plain':'Matched\nhead','context_free_fixed':'Fixed\nH-N-E','context_free_constant':'Best constant\npath','context_free':'HNE','context_free_rank':'Path\nranking'}
    bar_panels([metrics[metrics.dataset.eq(ds)] for ds in cohorts],[order,order],labels,cohorts,'path_auroc',{primary})
    transfer_path=RESULT/'transfer/metrics.csv'
    if transfer_path.exists():
        table=pd.read_csv(transfer_path);directions=[('RESILIENT','BrisT1D-Open'),('BrisT1D-Open','RESILIENT')]
        order=['native','context_plain','context_fixed','path_policy','context_path']
        names={'native':'Chronos-2 FT','context_plain':'Context head','context_fixed':'Fixed H-N-E',
               'path_policy':'27-path HNE','context_path':'Context-path HNE'}
        if primary=='context_free':
            order=['native','context_free_plain','context_free_fixed','context_path','context_free']
            names.update(native='Chronos-2\nFT',context_free_plain='Context\nhead',context_free_fixed='Fixed\nH-N-E',context_path='6-path\nHNE',context_free='HNE')
        bar_panels([table[table.source.eq(a)&table.target.eq(b)] for a,b in directions],[order,order],names,
            [f'{a} → {b}' for a,b in directions],'transfer_auroc',{primary})
    adaptation=[]
    adaptation_root=RESULT/'router_transfer_context' if (RESULT/'router_transfer_context').exists() else RESULT/'router_transfer'
    if primary=='context_free':adaptation_root=RESULT/'router_transfer_free'
    for direction in sorted(adaptation_root.glob('*_to_*')):
        files=sorted(direction.glob('fold*/held_person.csv'))
        if len(files)!=5:continue
        frame=pd.concat([pd.read_csv(p,dtype={'person_id':str}) for p in files],ignore_index=True)
        source,target=direction.name.split('_to_')
        for method,column in [('source','source_probability'),('constant','constant_path_probability'),('adapted','probability')]:
            adaptation.append(dict(source=source,target=target,method=method,auroc=fold_auc(frame,column),n=len(frame)))
    if adaptation:
        table=pd.DataFrame(adaptation);table.to_csv(OUT/'router_transfer_metrics.csv',index=False)
        directions=[('RESILIENT','BrisT1D-Open'),('BrisT1D-Open','RESILIENT')];order=['source','constant','adapted']
        bar_panels([table[table.source.eq(a)&table.target.eq(b)] for a,b in directions],[order,order],
            {'source':'Frozen source','constant':'Target constant\npath','adapted':'Adapted router'},
            [f'{a} → {b}' for a,b in directions],'router_transfer_auroc',{'adapted'})
    evidence_path=RESULT/'evidence/metrics.csv'
    if evidence_path.exists():
        evidence=pd.read_csv(evidence_path).rename(columns={'method':'variant','condition':'method'})
        order=['original','reference_zero','network_cut','context_zero']
        bar_panels([evidence[evidence.dataset.eq(ds)] for ds in cohorts],[order,order],
            {'original':'HNE','reference_zero':'Reference\nzero','network_cut':'Network\ncut','context_zero':'Context\nzero'},
            cohorts,'evidence_auroc',{'original'})
    rows=['DNB','DNB calibrated','FCM','FCM calibrated','CATCH','CATCH calibrated','native',primary]
    names={'native':'Chronos-2 FT',primary:'HNE','DNB calibrated':'DNB + calibration',
           'FCM calibrated':'FCM + calibration','CATCH calibrated':'CATCH + calibration'}
    ordered=metrics[metrics.method.isin(rows)].pivot(index='method',columns='dataset',values='auroc').reindex(rows)
    ranks=ordered.rank(ascending=False,method='min')
    lines=[r'\begin{tabular}{lcc}',r'\toprule',r'Method & RESILIENT & BrisT1D-Open \\',r'\midrule']
    for method in rows:
        cells=[]
        for ds in cohorts:
            value=f'{ordered.loc[method,ds]:.3f}'
            if ranks.loc[method,ds]==1:value=r'\textbf{'+value+'}'
            elif ranks.loc[method,ds]==2:value=r'\underline{'+value+'}'
            cells.append(value)
        lines.append(names.get(method,method)+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (OUT/'main_table.tex').write_text('\n'.join(lines),encoding='utf8')
    print('Saved main table, path and transfer bars to',OUT)


if __name__=='__main__':build()
