# RESILIENT 与 BrisT1D-Open 的近期研究和外部对照

截至 2026-11-16，按数据集名称、作者、数据 DOI 及论文前向引用检索，并核对原文方法和数据发布页。下列研究在方法、数据表或作者发布记录中明确标识队列。

| 原队列 | 研究 | 实际使用与结局 |
| --- | --- | --- |
| RESILIENT | Fletcher-Lloyd et al., 2026，npj Digital Medicine | Pilot High-risk cohort 来自 RESILIENT。研究用床垫传感器的夜间活动和生理特征估计睡眠年龄，并考察与认知风险的关系。作者方法稿报告该队列分析了 50 人、470 个样本，其中包括 10 位照护伙伴。 [正式论文](https://doi.org/10.1038/s41746-026-02964-0) · [队列来源与回归模型，§§2、4](https://www.medrxiv.org/content/10.1101/2025.08.29.25334735v2.full.pdf) |
| BrisT1D-Open | Lu et al., 2026，GlucoFM-Bench | 表 1 列出 Bristol 的 20 人、848,574 个五分钟测量；作者数据卡链接到 Bristol 的 Open 版本。原文报告 Bris-T1D 分项结果，比较 Chronos-2、Moirai 2.0、Timer 和 TimesFM 2.5 等模型。目标是血糖值，预测时距为 15、30、60 和 90 分钟。 [方法与分队列表](https://arxiv.org/html/2606.06881v1) · [作者数据卡](https://huggingface.co/datasets/glucofmbench/GlucoFM-Bench) |
| RESILIENT | Wu et al., 2026，StepFM，预印本 | 表 6 使用小时步数的宏观分支，对比 SSCP。8 项健康结局平均 AUROC 为 0.6819 / 0.6673，单独的睡眠紊乱任务为 0.5603 / 0.6243。该任务把入睡后清醒时间至少 90 分钟作为标签。[原文表 6 和标签](https://arxiv.org/html/2607.06954v1) |
| BrisT1D-Open | Maqsood et al., 2026，FedPhysio-Adapter | 原文点名 Open 版，按人留出，联邦适配器预测血糖，对低血糖风险高且预测区间宽的窗口延迟给出预测。[正式论文](https://doi.org/10.1007/978-3-032-35387-0_13) |
| BrisT1D-Open | Wolff et al., 2026，MetaboNet | 数据整合论文在表 1 列明纳入 19 人，并给出 Open 版来源。可用于追溯队列，不提供与本文相同的七日预警成绩。[正式论文](https://doi.org/10.1177/19322968261441637) · [表 1](https://arxiv.org/html/2601.11505v1) |

同源竞赛研究另有 Gordon James（2026）的博士论文。作者分析 BrisT1D 竞赛优胜模型，报告加入手表数据后，一小时血糖预测的 RMSE 约下降 3%。[布里斯托大学学位论文](https://research-information.bris.ac.uk/en/studentTheses/the-potential-of-smartwatches-in-artificial-intelligence-enabled-/)

Naskinova et al.（2025）评估了生理模型加 LightGBM 的餐后血糖预测，原文明确使用 Kaggle BRIST1D。AutoBiGluNet（2026）在摘要中报告了 BrisT1D 血糖回归成绩；可核实的摘要未区分长期公开版和竞赛版。这两项成绩都没有用作 HNE 的未来事件分类分数。[Naskinova et al.](https://doi.org/10.3390/a18100623) · [AutoBiGluNet](https://pubmed.ncbi.nlm.nih.gov/42395262/)

## 四项近期实测对照

| 方法 | 上游任务 | 本研究的统一预警适配 |
| --- | --- | --- |
| Moirai 2.0（Liu et al., 2025） | 多变量概率预测，GlucoFM-Bench 用过的冻结骨干 | 作者代码和 R-small 权重对六条三小时序列联合预测接下来 64 步。丢弃第一天，仅在预测的第 2–8 天提取每日三个分位数（0.1、0.5、0.9）的均值及预测中位线的日内标准差。用逻辑回归输出七日事件风险。 [方法](https://arxiv.org/abs/2511.11698) · [模型和源码](https://huggingface.co/Salesforce/moirai-2.0-R-small) |
| JAPE（Wei et al., 2026，预印本） | 结合多变量预测和指标依赖图预测未来异常 | 使用作者的预测器、依赖图特征和双视角报警头。按原有事件适配，将未来第 2–8 天的报警 logits 取平均再计算事件概率。复用两库已有五折、每折三种子的留出人员预测，重新配对计分。 [论文](https://arxiv.org/abs/2608.11801) · [作者代码](https://anonymous.4open.science/r/JAPE-B74E/) · [实验配置](../configs/jape_event.json) |
| Mantis（Feofanov et al., ICML 2026） | 冻结的时序分类编码器 | 使用作者 Mantis-8M 权重。六条历史序列独立编码，拼接表示，按作者示例标准化后训练逻辑回归。历史内缺口插值，整条历史通道缺测时取个人参考水平；只在截至预测日的 32 天内操作。 [正式论文](https://proceedings.mlr.press/v306/feofanov26a.html) · [作者代码](https://github.com/vfeofanov/mantis) |
| TimesFM 2.5（Google Research, 2025） | GlucoFM-Bench 在 Bris-T1D 评价过的概率预测器 | 下载作者 PyTorch 200M 权重，逐通道预测 64 个三小时点。与 Moirai 使用相同日区间、三个分位数、日内波动特征和验证选参的逻辑回归。事件头在本研究数据上训练。[作者代码](https://github.com/google-research/timesfm) · [权重](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |

Mantis、Moirai 2.0 和 TimesFM 2.5 都使用 HNE 相同的按人五折。每次三折训练、一折验证、一折评估，独立拟合每折的缩放器和确定性 L2 逻辑回归。三个冻结基模均按验证 AUROC 从 C = 0.001、0.01、0.1、1 中选择，同分取较低验证对数损失。JAPE 使用已有三种子概率平均。四组结果都按 person_id、origin_day、fold 和真实事件标签对齐到原主表，RESILIENT 保留 55 人、1,938 个时点，BrisT1D-Open 保留 19 人、1,196 个时点。主表为各自的绝对 AUROC、AP；配对差的区间按参与者重采样 2,000 次。两个主库均为已经用于方法开发的队列。[主结果](../results/evidence_recursion/table1.csv) · [HNE 相对四个对照的配对区间](../results/evidence_recursion/recent_baselines_paired.csv)

八项外部方法中，CATCH 的 AUROC 在两库均最高。HNE 相对 CATCH 的差为 +0.0145（95% 配对区间 −0.0538 至 +0.0924）和 +0.0441（−0.0598 至 +0.1700），分别对应 RESILIENT 和 BrisT1D-Open。计算使用与四项近期对照相同的逐人重采样及折内配对权重。

本轮 TimesFM 2.5 + LR 的 AUROC / AP 分别为 0.5625 / 0.2316 和 0.5191 / 0.1735；HNE 的配对 AUROC 差为 +0.2171（95% 区间 0.1042 至 0.3057）和 +0.1760（−0.0201 至 0.3282）。

## 复现

在本项目已有公开数据和 HNE 环境中执行。Mantis 与 Moirai 作者源码分别位于 `cache/mantis/author`（vfeofanov/mantis 提交 `23ecfe493830db177546ef159947de7d79a573bd`）和 `cache/moirai2/author`（SalesforceAIResearch/uni2ts 提交 `cfd46d4510ed8896f263116f32928eede05b0a75`）。新检出项目时，将这两个上游仓库按相应提交检出到上述位置；Moirai 的核心推理依赖 jaxtyping，本次用 0.3.11。TimesFM 使用 google-research/timesfm 提交 `918df02c9d366b2145bc5876135fa2131d3ed2a6`，检出到 `cache/timesfm/author`。三套模型权重的 Hugging Face 版本已固定在运行脚本内。Mantis 严格加载全部学习参数；原权重文件省略的两个标量网格常量来自作者的模型配置。

```shell
python scripts/run_recent_warning.py mantis
python scripts/run_recent_warning.py moirai2
python scripts/run_recent_warning.py timesfm
python scripts/run_jape_development.py
python scripts/report_evidence_recursion.py
```

前三条命令分别生成每折分类头、正则参数选择结果和逐时点留出分数。第四条使用已有 JAPE 训练产物，缺失时按原配置训练。最后一条在相同预测时点重建两张表。
