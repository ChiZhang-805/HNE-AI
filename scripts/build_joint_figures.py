"""Black-and-white numerical figures and the result report for joint HNE."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/hne_joint/summary'
COHORTS=['RESILIENT','BrisT1D-Open']
LABELS={'joint':'HNE','fixed':'Fixed\nH-N-E','coordinates_only':'No\nbackbone','no_coordinates':'No numerical\nevidence',
        'best_constant':'Best constant\npath','uniform':'Uniform\npaths','previous_HNE':'Previous\nHNE','Chronos-2 FT':'Chronos-2\nFT',
        'DNB daily':'DNB\n(daily)','FCM daily':'FCM\n(daily)','CATCH daily':'CATCH\n(daily)',
        'DNB 3h':'DNB\n(3h)','FCM 3h':'FCM\n(3h)','CATCH 3h':'CATCH\n(3h)'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def bars(tables,order,titles,filename,metric='auroc',ylabel='AUROC'):
    figure,axes=plt.subplots(1,2,figsize=(11.8,4.2),layout='constrained')
    for axis,table,title in zip(axes,tables,titles):
        names=[m for m in order if m in table.method.values];frame=table.set_index('method').loc[names]
        values=frame[metric].to_numpy();rects=axis.bar(np.arange(len(values)),values,width=.63,
            color=['black' if m=='joint' else '.74' for m in names])
        axis.bar_label(rects,labels=[f'{v:.3f}' for v in values],padding=4,fontsize=10)
        axis.set_xticks(np.arange(len(values)),[LABELS.get(m,m) for m in names],fontsize=9)
        axis.set_ylim(0,1);axis.set_yticks([0,.25,.5,.75,1]);axis.set_ylabel(ylabel);axis.set_title(title)
    figure.savefig(OUT/f'{filename}.png',dpi=260);figure.savefig(OUT/f'{filename}.pdf');plt.close(figure)


def build():
    metrics=pd.read_csv(OUT/'metrics.csv');paired=pd.read_csv(OUT/'paired_intervals.csv')
    tables=[metrics[metrics.dataset.eq(ds)] for ds in COHORTS]
    bars(tables,['DNB daily','FCM daily','CATCH daily','Chronos-2 FT','previous_HNE','joint'],COHORTS,'main_auroc')
    bars(tables,['DNB daily','FCM daily','CATCH daily','Chronos-2 FT','previous_HNE','joint'],COHORTS,'main_ap',metric='ap',ylabel='Average precision')
    bars(tables,['DNB 3h','FCM 3h','CATCH 3h','Chronos-2 FT','joint'],COHORTS,'sequence_comparison')
    bars(tables,['DNB daily','DNB 3h','FCM daily','CATCH daily','previous_HNE','joint'],COHORTS,
        'within_person_auroc',metric='within_person_auroc',ylabel='Mean within-participant AUROC')
    bars(tables,['no_coordinates','coordinates_only','joint'],COHORTS,'evidence_auroc')
    bars(tables,['fixed','best_constant','uniform','joint'],COHORTS,'path_auroc')
    transfer_file=ROOT/'results/hne_joint/transfer/metrics.csv'
    transfer=pd.read_csv(transfer_file) if transfer_file.exists() else pd.DataFrame()
    if len(transfer):
        bars([transfer[transfer.source.eq(ds)] for ds in COHORTS],['no_coordinates','coordinates_only','fixed','joint'],
            ['RESILIENT → BrisT1D-Open','BrisT1D-Open → RESILIENT'],'transfer_auroc')
    order=['DNB daily','DNB-tuned daily','FCM daily','CATCH daily','FCM-score-LR daily','CATCH-score-LR daily',
        'DNB 3h','FCM 3h','CATCH 3h','DNB 3h calibrated','FCM 3h calibrated','CATCH 3h calibrated',
        'Chronos-2 FT','previous_HNE','joint']
    table=metrics[metrics.method.isin(order)].pivot(index='method',columns='dataset',values='auroc').reindex(order)
    ranks=table.rank(ascending=False,method='min');lines=[r'\begin{tabular}{lcc}',r'\toprule',r'Method & RESILIENT & BrisT1D-Open \\',r'\midrule']
    for method in order:
        cells=[]
        for ds in COHORTS:
            v=f'{table.loc[method,ds]:.3f}'
            if ranks.loc[method,ds]==1:v=r'\textbf{'+v+'}'
            elif ranks.loc[method,ds]==2:v=r'\underline{'+v+'}'
            cells.append(v)
        lines.append(LABELS.get(method,method).replace('\n',' ')+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(OUT/'main_table.tex').write_text('\n'.join(lines),encoding='utf8')
    records=[json.loads(p.read_text(encoding='utf8')) for p in (ROOT/'results/hne_joint').glob('*/*/fold*/seed*/result.json')]
    report=['# HNE 联合训练：实验结果','',
        f'已完成 {len(records)} 次完整拟合；每次 30 epochs。两库共用 Chronos-2、H/N/E 数值证据、三步递推、路由目标和同一配置。',
        '','## 主实验','',
        '主指标为参与者隔离五折中的正负样本对加权 AUROC，三个种子平均预测。所有方法在共同有效样本上比较。公开框架同时列出日级与三小时输入版本。','',
        '| 方法 | RESILIENT | BrisT1D-Open |','| --- | ---: | ---: |']
    for m in order:report.append(f'| {LABELS.get(m,m).replace(chr(10)," ")} | {table.loc[m,COHORTS[0]]:.4f} | {table.loc[m,COHORTS[1]]:.4f} |')
    report.extend(['','预设次指标：平均精度 AP 与验证集 10% 误报预算下的告警表现。','',
        '| 数据集 | 方法 | AP | 灵敏度 | 实际误报率 |','| --- | --- | ---: | ---: | ---: |'])
    for ds in COHORTS:
        for m in ['FCM daily','CATCH daily','previous_HNE','joint']:
            r=metrics[metrics.dataset.eq(ds)&metrics.method.eq(m)].iloc[0]
            report.append(f'| {ds} | {LABELS[m].replace(chr(10)," ")} | {r.ap:.4f} | {r.sensitivity_at_validation_10pct_FPR:.4f} | {r.held_false_positive_rate:.4f} |')
    report.extend(['','## RQ1：模型能否学习多指标节律紊乱的前兆？',''])
    for ds in COHORTS:
        t=metrics[metrics.dataset.eq(ds)].set_index('method');r=paired[paired.dataset.eq(ds)&paired.comparison.eq('strongest_external')].iloc[0]
        report.append(f'{ds}：HNE {t.loc["joint","auroc"]:.4f}；最高外部对照 {r.strongest_observed} {t.loc[r.strongest_observed,"auroc"]:.4f}；差值 {r.delta:+.4f}，参与者配对 95% 区间 [{r.low:+.4f}, {r.high:+.4f}]。共同样本 {int(t.loc["joint","n"])} 个，{int(t.loc["joint","people"])} 人。')
        report.append(f'去基模 {t.loc["coordinates_only","auroc"]:.4f}；去数值证据 {t.loc["no_coordinates","auroc"]:.4f}；原 HNE {t.loc["previous_HNE","auroc"]:.4f}。\n')
        report.append(f'预设次指标：同人 AUROC 为 {t.loc["joint","within_person_auroc"]:.4f}，原 HNE 为 {t.loc["previous_HNE","within_person_auroc"]:.4f}；使用同时具有正常与紊乱评价时点的 {int(t.loc["joint","people_with_both_classes"])} 人。逐人配对区间见 within_person_intervals.csv。\n')
    report.extend(['## RQ2：学习推理路径能否提升紊乱预警？','','| 方法 | RESILIENT | BrisT1D-Open |','| --- | ---: | ---: |'])
    for m in ['fixed','best_constant','uniform','joint']:
        t=metrics[metrics.method.eq(m)].set_index('dataset');report.append(f'| {LABELS[m].replace(chr(10)," ")} | {t.loc[COHORTS[0],"auroc"]:.4f} | {t.loc[COHORTS[1],"auroc"]:.4f} |')
    routing=pd.read_csv(OUT/'routing.csv')
    for ds in COHORTS:
        r=routing[routing.dataset.eq(ds)].mean(numeric_only=True)
        report.append(f'\n{ds}：六路径平均概率跨度 {r.mean_path_span:.4f}；动态选择路径的真实标签损失平均排名 {r.selected_loss_rank:.3f}/6。真实标签仅用于事后诊断，不进入推理。')
    report.extend(['','## RQ3：预警机制能否迁移到不同领域的紊乱？','',
        '源域输入变换与模型全部冻结，目标标签不参与训练，源域 15 个模型平均。两个领域均已参与此前方法开发；这里衡量冻结参数的跨域迁移。','',
        '| 源 → 目标 | 方法 | AUROC |','| --- | --- | ---: |'])
    if len(transfer):
        for r in transfer.itertuples():report.append(f'| {r.source} → {r.target} | {LABELS.get(r.method,r.method).replace(chr(10)," ")} | {r.auroc:.4f} |')
    report.extend(['','## 文件','',
        '模型：src/hne_joint.py；训练：scripts/run_hne_joint.py；原始观测推理：scripts/predict_hne_joint.py；迁移：scripts/transfer_hne_joint.py。',
        '完整数值：metrics.csv；配对差值：paired_intervals.csv；分折/种子：fold_metrics.csv、seed_metrics.csv；路径：routing.csv。',
        '图形均为黑白柱状图，PNG 与 PDF 同时输出。旧论文 PDF 的数值未自动替换为本轮结果。'])
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf8')
    print('Saved numerical figures, LaTeX table, and result report:',OUT)


if __name__=='__main__':build()
