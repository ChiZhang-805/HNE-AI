"""Write the current research result from completed numerical outputs."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];RESULT=ROOT/'results/hne_routed';OUT=RESULT/'summary'
DATASETS=['RESILIENT','BrisT1D-Open']


def table(frame,columns,labels):
    lines=['| '+' | '.join(labels)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        cells=[f'{row[c]:.4f}' if isinstance(row[c],float) else str(row[c]) for c in columns]
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def run():
    metrics=pd.read_csv(OUT/'metrics.csv');intervals=pd.read_csv(OUT/'auroc_intervals.csv')
    paired=pd.read_csv(OUT/'paired_intervals.csv');choice=json.loads((RESULT/'architecture_selection.json').read_text())
    primary=choice['primary_variant'];methods=['DNB','DNB calibrated','FCM','FCM calibrated','CATCH','CATCH calibrated','native',primary]
    values=metrics.pivot(index='method',columns='dataset',values='auroc').loc[methods].reset_index()
    values['method']=values.method.replace({'native':'Chronos-2 FT',primary:'HNE'})
    lines=['# HNE：预训练模型内部的权重与路径学习',
        '本轮完成统一模型实现、两库训练、作者框架对照、路径训练比较、信息与节律干预、双向冻结迁移和少参与者路径适配。',
        '## 方法与训练',
        '基模为 Chronos-2 120M，前 11 层冻结，末层 Q/V 使用 rank-8 LoRA。在末层之前插入共享的个人历史 H、指标网络 N、情境 E 运算，由可训练控制器选择三步路径。两个数据集共用同一实现、目标与选参规则。',
        '第一阶段联合训练 553,735 个参数；第二阶段固定运算权重，训练参照读出、路径系数与控制器，共 55,877 个参数。第二阶段精确计算 27 条路径的期望事件损失，推理逐步选择动作。',
        '五折按参与者划分，三次随机重复；输入过去 32 天三小时序列与个人最初 56 天参考，预测至少一天间隔后的七天持续异常波动/偏离。详细公式、代码来源、训练目标和 RQ 判据见 [设计文档]('+ (ROOT/'literature/hne_model_design.md').as_posix()+')。',
        '## 主比较',
        '下表为折内正负样本对加权 AUROC，使用三种子均值与各方法共同有效时点。DNB、FCM、CATCH 的原始分数和训练标签校准结果分别列出，所有数值均来自本轮统一任务重跑。',
        table(values,['method']+DATASETS,['方法']+DATASETS)]
    for ds in DATASETS:
        h=metrics[metrics.dataset.eq(ds)&metrics.method.eq(primary)].iloc[0]
        ci=intervals[intervals.dataset.eq(ds)&intervals.method.eq(primary)].iloc[0]
        comparison=paired[paired.dataset.eq(ds)&paired.comparison.eq('strongest_external')].iloc[0]
        old=metrics[metrics.dataset.eq(ds)&metrics.method.eq('previous_HNE')].iloc[0]
        opponent='Chronos-2 FT' if comparison.strongest_observed=='native' else comparison.strongest_observed
        lines.append(f"{ds}：{int(h.people)} 人、{int(h.n)} 个共同有效时点；HNE AUROC {h.auroc:.4f}，95% 区间 [{ci.auroc_low:.4f}, {ci.auroc_high:.4f}]；最强对照为 {opponent}，差值 {comparison.delta:+.4f}，配对区间 [{comparison.low:.4f}, {comparison.high:.4f}]。原 HNE 在相同时点为 {old.auroc:.4f}。")
    lines+=['## RQ1：模型能否学习多指标节律紊乱的前兆？',
        '以下干预保留全部可评价时点，模型权重固定。日内/跨日打乱保留数值分布；个人记忆置零、通道交换切断、情境置零分别改变对应信息来源。',
        table(pd.read_csv(OUT/'temporal_metrics.csv').query('method == @primary'),['dataset','condition','auroc'],['数据集','时间条件','AUROC']),
        table(pd.read_csv(RESULT/'evidence/metrics.csv'),['dataset','condition','auroc'],['数据集','信息条件','AUROC']),
        '保留时间顺序及个人参考均带来更高分数。网络交换的影响很小，情境置零后的表现更高。参与者内 AUROC 单独列入 metrics.csv，以区分个体差异与同一个人的时间变化。',
        '## RQ2：学习推理路径能否提升紊乱预警？']
    path_methods=['context_free_plain','context_free_fixed','context_free_constant','context_free','context_free_rank']
    path_table=metrics[metrics.method.isin(path_methods)].pivot(index='method',columns='dataset',values='auroc').reindex(path_methods).reset_index()
    lines+=[table(path_table,['method']+DATASETS,['路径/读出训练']+DATASETS),
        '`context_free_plain` 为参数量相近的直接读出；`context_free_fixed` 单独训练固定 H→N→E 的读出；`context_free_constant` 使用 HNE 相同权重，在验证集选定恒定路径；`context_free_rank` 固定预测权重，用标准化路径损失重新训练控制器。',
        '当前动态路径没有稳定超过最佳恒定路径。原始同权重路径诊断如下；损失最优路径由评价答案事后计算，仅用于诊断。']
    paths=pd.read_csv(RESULT/'free_path_interventions/metrics.csv').groupby('dataset')[['mean_path_rank','path_effect_mean','unique_paths']].mean().reset_index()
    lines.append(table(paths,['dataset','mean_path_rank','path_effect_mean','unique_paths'],['数据集','所选路径损失排名/27','路径风险跨度','每模型使用路径数']))
    lines+=['## RQ3：预警机制能否迁移到不同领域的紊乱？',
        '冻结迁移采用 15 个源模型集成；目标标签不更新源权重。模型架构验证选择使用了两个开发队列。',
        table(pd.read_csv(RESULT/'transfer/metrics.csv').query("method in ['native','context_free_plain','context_free_fixed','context_free']"),['source','target','method','auroc'],['来源','目标','方法','AUROC']),
        '路径适配固定源第 0 折的三种子模型，目标每折仅用两个训练参与者更新路由器，另用该折验证参与者选训练轮次。下面的冻结源、恒定路径与适配路由使用相同源权重和目标评价时点。',
        table(pd.read_csv(OUT/'router_transfer_metrics.csv'),['source','target','method','auroc'],['来源','目标','路径规则','AUROC']),
        '## 实现、权重与材料']
    checks=pd.read_csv(OUT/'training_summary.csv');raw=json.loads((OUT/'raw_inference.json').read_text())
    lines.append(f"30 个最终模型的第二阶段计算权重均与来源 checkpoint 完全一致：{bool(checks.computation_weights_identical.all())}。第一阶段 LoRA 更新、第二阶段路由器更新及路径系数见 training_summary.csv。")
    for x in raw:
        lines.append(f"{x['dataset']} 原始小时输入已执行完整模型：风险 {x['risk_score']:.6f}，路径 {'→'.join(x['path'])}；与缓存评价差值 {x['absolute_difference']:.6f}，路径一致 {x['path_matches']}。")
    counts=[]
    for ds in DATASETS:
        for method in ['learned','native','path_policy','context_path','context_fixed','context_plain','context_free','context_free_fixed','context_free_plain','context_free_rank','route_controls','DNB','FCM','CATCH']:
            n=len(list((RESULT/ds/method).glob('**/result.json')))
            counts.append(dict(dataset=ds,method=method,completed_fits=n))
    pd.DataFrame(counts).to_csv(OUT/'fit_counts.csv',index=False)
    lines+=['复现：在项目目录运行 `./scripts/reproduce_routed.ps1 -Stage all`；已有结果汇总运行 `-Stage analysis`。原始输入执行示例见 `scripts/run_hne_raw_examples.py`。',
        '结果包含逐样本预测、逐种子权重、训练曲线、聚类统计区间、数值表和六组黑白柱状图。原始 HNE 的 paper/main.pdf 保留此前方法与结果，本轮没有把旧分数标成新模型成绩。',
        '## 本轮判断',
        '预训练模型内部的权重学习、运算路径学习和跨域执行已经实现并完成实验。个人参照与时间顺序提供了有效信息；动态选路和情境分支的稳定增益没有得到支持。新模型尚未超过原 HNE，两项结果不能合并为一条已成功的创新结论。']
    (OUT/'report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
    print('Wrote',OUT/'report.md')


if __name__=='__main__':run()
