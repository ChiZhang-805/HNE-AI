# HNE-AI-1 · 在模型内部重复读取证据，提前预警多变量紊乱

[完整论文实验设计](paper/experiment_plan.md)：24组研究实验，正文4表4图、附录12表8图；统一HNE核心、未来事件协议、公开对照、连续推理机制和跨域迁移。[机器可读清单](configs/paper_experiments.csv)中的状态为设计完成，执行结果按对应模型和协议另行汇总。

以个人前 56 天为参考，用截至预测日的 32 天历史判断第 2–8 天是否会出现持续的跨指标偏移或波动放大。论文主架构把个人历史、指标关系和环境编码为记忆，将上一轮隐状态写入下一连续位置，反复通过 Chronos-2 的两个共享 Transformer 块，训练其中的 LoRA 及证据投影。

2026-11-20：对照 Coconut、Huginn、LoopFormer、CODI 与 SIM-CoT，将核心前向过程整理为证据记忆、步条件、可见位置、核心更新、结果读出与状态回填。[代码阅读顺序与借鉴位置](paper/continuous_warning_framework.md#对照作者代码怎样读懂-hne)逐项对应函数。两库真实输入上的输出、混合精度训练损失与梯度和整理前一致，原模型文件继续适用。

| 内容 | 入口 |
| --- | --- |
| 当前研究 | [三个研究问题、模型与实验解释](paper/continuous_warning_framework.md) · [连续推理文献](literature/continuous_reasoning_methods.md) |
| 模型与训练 | [Transformer 内部反馈](src/hne_latent.py) · [配置](configs/hne_latent.json) · [训练入口](scripts/run_hne_latent.py) |
| 实验与复现 | [运行说明](experiments/README.md) · [数据来源](data/README.md) · [2025–2026 文献与外部对照](literature/recent_public_warning_comparators.md) |
| Transformer 内部反馈结果 | [五折成绩](results/hne_latent/summary/metrics.csv) · [反馈对照区间](results/hne_latent/summary/rq2_simultaneous_intervals.csv) |
| GRU 证据读取实验 | [模型](src/hne_evidence_recursion.py) · [两表 PDF](results/evidence_recursion/HNE_AI_1_tables.pdf) · [数值](results/evidence_recursion/table1.csv) · [配对区间](results/evidence_recursion/recent_baselines_paired.csv) |
| 此前的独立实验 | [加性模型论文 PDF](paper/main.pdf) · [LaTeX](paper/main.tex) · [原汇报](paper/briefing.pptx) · [历史结果导航](results/README.md) |

在共同预测时点上，RESILIENT 有 55 人、1,938 个时点，BrisT1D-Open 有 19 人、1,196 个时点。Transformer 内部反馈模型的 AUROC 为 0.7697 和 0.6475，对应汇总中的 recurrent 行。其相对单步模型的 AUROC 差为 −0.0190 和 +0.0703，同时置信区间均跨零。

现有两表的 GRU 证据读取实现取得 AUROC 0.7796 和 0.6950，八个外部对照中 CATCH 在两库均最高，分别为 0.7651 和 0.6510。相对 CATCH 的两项配对区间均跨零。该 GRU 实现相对同深度固定查询提高 0.0044 和 0.0049。以上两库用于开发；GRU 实现的冻结迁移另见[迁移与机制差值](results/evidence_recursion/table2.csv)。

论文 PDF 和两页汇报记录早期方法。架构说明逐项对应 Transformer 内部反馈与 GRU 证据读取的模型、训练方式及实验成绩。

2026-11-19 更新：完成 2025—2026 年内部推理文献与作者源码核对，主清单 15 篇，另补读轨迹评估工作。[引用与出处](literature/internal_reasoning_2025_2026.json) · [调研与 HNE 优化](literature/continuous_reasoning_methods.md)。

新配置加入证据、当前状态和循环时间调节的 Transformer 内部更新，以及逐步未来动态监督。[统一配置](configs/hne_latent_grounded.json)。两库四种条件完成 24 次训练人员内部拟合，完整条件相对无调节的内部 AUROC 差为 +0.0037／−0.0030；开发比较不替换上面的五折主结果。[汇总 JSON](results/hne_latent_grounded/summary/development_summary.json) · [数值表](results/hne_latent_grounded/summary/development_metrics.csv) · [损失曲线数据](results/hne_latent_grounded/summary/development_curves.csv)。
