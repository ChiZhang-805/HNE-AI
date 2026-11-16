# HNE-AI-1 · 在模型内部重复读取证据，提前预警多变量紊乱

以个人前 56 天为参考，用截至预测日的 32 天历史判断第 2–8 天是否会出现持续的跨指标偏移或波动放大。HNE 结合个人历史、指标关系和环境，连续四次读取证据。上一轮隐状态决定下一轮查询。

| 内容 | 入口 |
| --- | --- |
| 当前研究 | [三个研究问题、模型与实验解释](paper/continuous_warning_framework.md) · [连续推理文献](literature/continuous_reasoning_methods.md) |
| 模型与训练 | [模型实现](src/hne_evidence_recursion.py) · [配置](configs/hne_evidence_recursion.json) · [训练入口](scripts/run_evidence_recursion.py) |
| 实验与复现 | [运行说明](experiments/README.md) · [数据来源](data/README.md) · [2025–2026 文献与外部对照](literature/recent_public_warning_comparators.md) |
| 当前结果 | [主实验和消融 PDF](results/evidence_recursion/HNE_AI_1_tables.pdf) · [主表 CSV](results/evidence_recursion/table1.csv) · [配对区间](results/evidence_recursion/recent_baselines_paired.csv) |
| 此前的独立实验 | [加性模型论文 PDF](paper/main.pdf) · [LaTeX](paper/main.tex) · [原汇报](paper/briefing.pptx) · [历史结果导航](results/README.md) |

在共同预测时点上，RESILIENT 有 55 人、1,938 个时点，BrisT1D-Open 有 19 人、1,196 个时点。完整 HNE 的 AUROC 为 0.7796 和 0.6950，八个外部对照中 CATCH 在两库均最高，分别为 0.7651 和 0.6510。相对 CATCH 的两项配对区间均跨零。完整四步模型相对同深度固定查询提高 0.0044 和 0.0049。以上两库用于开发；冻结迁移另见[迁移与机制差值](results/evidence_recursion/table2.csv)。

论文 PDF 和两页汇报记录早期方法。当前连续推理的模型、成绩和两张表使用上表“当前研究”“当前结果”的入口。
