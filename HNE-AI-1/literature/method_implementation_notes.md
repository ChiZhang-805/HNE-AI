# 可实现近邻：TipPFN、DEV与MM-DEV

核对日期：2026-11-10。这里记录官方方法、代码入口和本项目任务所需的适配，不报告尚未运行的实验结果。

## TipPFN（M12）

论文为[arXiv:2605.12308v1](https://arxiv.org/html/2605.12308v1)。[NeurIPS2026官方页面](https://neurips.cc/virtual/2026/poster/155891)标题明确为Poster，结构化元数据`creditText=NeurIPS 2026`，指向[OpenReview24hX13I66G](https://openreview.net/forum?id=24hX13I66G)。方法与数值已读公开arXiv主文及附录；尚未核实proceedings最终版本。官方poster页面原始HTML已保存`fulltext/neurips.html`。

**代码/权重。** 公开原文没有官方仓库或checkpoint链接；[作者Tobias Weber主页](https://2bys.github.io/publications/)该条只有Paper，而其其他项目另有Code链接；[kausable官网](https://kausable.ai/)介绍TipPFN但没有模型下载。本轮GitHub仓库API按`TipPFN`搜索返回0条、HuggingFace模型API同名返回空数组；[HuggingFace论文条目](https://huggingface.co/papers/2605.12308)只是论文页。结论是**本轮未找到可下载的官方代码/权重**，不是断言作者从未发布。没有联系作者或调用其服务。

**输入。** query为1–16个观测特征的历史序列；训练窗长可变，实证使用16–128步。模型还可接收0–3个已完成context episodes，测试包括4个上下文；context包含观测轨迹和RDTC目标。每个episode有时间/episode标识、观测列、目标列和mask；query的全部RDTC被mask，query只能注意过去，context内部可全注意。归一化只用已观察值。若将环境作为额外特征，仍需计入输入维数，并遵循所有方法相同的截止时点。

**输出与连续分数。** 模型预测RDTC的99个分位数，合成监督为`Λ=1−归一化控制参数`，目标变换为`tanh(5Λ)`；辅助观测轨迹损失权重0.2。原文主连续分数是预测末端`1−median(Λ)`，附录也比较`P(Λ<0.05)`等CDF分数。原文窗口W对应未来`192−W`步。可以用这些连续分数计算事件二分类AUROC，但其含义是近临界程度，不自动等于HNE未来持续异常概率。若改变目标/进行事件校准，需称为适配并在开发数据完成。若context RDTC由真实事件生成，只能取预测时已结束的开发episode，不能把query未来onset构造成输入。

**预训练代价。** Appendix B.8给最终checkpoint训练4×H200 NVL（各141GB）约47小时，即约189 H200 GPU小时；约100万context-query对来自约12万合成系统，生成约7400 CPU核小时。最终评价约150 H200 GPU小时；含失败试验的整个项目约2150 GPU小时，不能混同为单次推断成本。未提供足够单窗延时测量，无法据此准确预测本机推断耗时。缺checkpoint时从头训练不是轻量对照。

**结果边界。** Table1两context平均AUROC0.859、SWEC0.816、Daphnia0.833、TAC0.519；附录患者级TabPFN中位较高、某些少发作患者无效、古气候zero-shot过晚。它是必须引用的直接近邻，但当前不能列入“已实际运行”的对照表，也不能用小型自写Transformer沿用TipPFN名称。

## DEV（M09）

[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9821862/)的官方[Zenodo代码/数据记录](https://zenodo.org/records/7379358)提供`DEV_Bifurcation-main.zip`，约1.0MB，页面MD5为`0e7977de28d1f22f33da38325a8c8a43`。本轮直接文件下载返回403，未取得文件，未把记录页面当作下载成功。论文Methods写明R与rEDM实现；不需要预训练权重。

输入为单变量滑动历史窗，经延迟嵌入得局部状态。S-map加权局部线性回归的系数形成Jacobian，再求dominant eigenvalue；窗口内平均后得到DEV，模可作为连续失稳score，实/虚部用于解释类型。原文将score时间戳放在窗口**最后一个点**。E与lag扫描1–12，非线性θ在0–2.5内按预测能力选，窗长根据预测能力平台选；HNE应把这些选择放在开发段。多变量若先池化或每变量计算后聚合，则是明确适配，不能称原文多变量DEV。

Zenodo页面对其中**经验数据**另有非营利用途及发表前联系数据提供者的条款；这不妨碍本项目用自己的数据实现/比较DEV算法。此处没有额外下载并复用这些经验数据。

## MM-DEV（M10）

[论文公开PDF](https://doi.org/10.21203/rs.3.rs-7865028/v1)列出[官方GitFront仓库](https://gitfront.io/r/koushikgarain/yKJvnhYx1mvc/MM-DEV/)。本轮实际下载并阅读的原脚本保存在`reference_code/mmdev/`：README、`Demo_check_embedding_dim.R`、`Figure_2.R`、`Figure_3_1.R`、`Figure_3_2.R`、`Figure_3_3.R`、`Figure_4_1.R`、`Figure_4_2.R`、`Figure_4_3.R`、`Figure_S_8_10_11.R`。未运行原作者实验，也未修改这些脚本。

**核心。** 多变量序列先scale、欧氏距离、ISOMAP，再选前若干分量并补第一分量的滞后；各输出变量分别S-map回归，系数拼成局部Jacobian，按特征值模取dominant eigenvalue。窗口内分别平均其模、实部和虚部。官方实现把输出放在窗口末端j。`Figure_3_2.R`提供化学与微生物方法，`Figure_4_2.R`提供湖泊方法，`Figure_2.R`含模拟分析。主要依赖R4.3.1、rEDM1.2.3、vegan；绘图/文件另需dplyr、deSolve、readxl、R.matlab、openxlsx等。无预训练checkpoint。

**时间边界。** `Figure_3_2.R`先对完整110日化学数据scale+ISOMAP，再取1:90进入滑窗；`Figure_4_2.R`先对全1978–2019月级记录scale+ISOMAP，再滚动DEV。直接运行得到的是回顾性流形表示。连续预警必须在当前历史窗或固定开发数据拟合scale/ISOMAP并明确样本外映射；不能将完整测试序列先做流形后声称提前可用。官方选窗demo也遍历整条输入序列，因此本项目只给它开发段。

建议可运行优先级是DEV与因果MM-DEV适配；保留相同未来持续事件标签、输入截止和评价指标。MM-DEV在短窗的embedding维数必须受有效样本数约束；原文反应器窗25日，湖泊50–100个月，不能按维数盲目搬用12阶嵌入。TipPFN在权重可得前作为直接文献近邻和方法边界，不假报实验结果。

## JAPE（2026）：实际原模型与区间事件适配

原文：[arXiv:2608.11801v2](https://arxiv.org/html/2608.11801v2)，作者代码：[JAPE-B74E](https://anonymous.4open.science/r/JAPE-B74E/)。2026-11-10通过该站公开`/api/repo/JAPE-B74E/files?path=...`和`/file/...`端点下载原始modules；文件位于`reference_code/JAPE`，未改作者源文件。实际导入`models.jape.JAPEModel`、`exp.anomaly.StructuralFeatureExtractor`、`DualViewAlertingHead`和原focal loss；DSTR交替时序与有向滞后图、detach后的未来图分支、13维图token、FiLM与交叉注意力均保留。

使用Appendix C参数：预测骨干3层、128维、8头、FFN256、topk5、lag3；风险头3层、64维、4头；batch128、lr1e−4、最多20轮、patience5。预测阶段Adam、主/辅助MSE等权；风险阶段冻结预测器，AdamW、weight decay1e−4、focal alpha0.25/gamma2、梯度裁剪1。原代码风险选择按30个0.05–0.95阈值上的验证F1，并采用ReduceLROnPlateau；保持此规则，没有改为挑最高测试AUROC。

本项目的明确适配：日级32步历史、patch4/stride2、连续预测d+1至d+8；预警头d+2至d+8七个logit的均值监督同一个未来区间标签，d+1只作为间隔日数值预测。未来缺测位置不进入MSE，保留全部事件origins；输入按同一公开基线训练历史中位数填充与StandardScaler处理。原逐时刻标签不适用于本任务，因此名称写为`JAPE-event`，不声称复现其原benchmark分数。使用本研究固定的参与者划分，不调用原作者在测试时间线内交错构造监督子集的入口。NPE源码保留，当前比较衡量事件风险，不报告尚未评价的变量解释分数。
