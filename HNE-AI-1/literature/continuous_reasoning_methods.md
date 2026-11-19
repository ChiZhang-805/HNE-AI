# 2025—2026 模型内部推理：论文、代码与 HNE 优化

更新：2026-11-19。本轮核对 22 项检索记录，选出 15 篇具有 2025／2026 正式会议出处及公开实现、涉及内部连续推理／循环计算／表示轨迹的论文。另列辅助模型路线和理论工作。下文“借鉴”是我们据此提出的 HNE 设计，效果由本项目实验检验。

引用数取 OpenAlex 的标题匹配记录，截至 2026-11-19；不同版本不相加，不用 GitHub 星数代替引用数。按正式发表年份筛选：Coconut 首次预印本在 2024 年，正式发表于 COLM 2025。2026 年论文单列，不把近期发表等同于高引用。[引用、会议、源码版本记录](internal_reasoning_2025_2026.json)

## 2025 年主干论文

| 论文 | 正式发表 | 引用 | 内部机制与 HNE 的借鉴位置 |
| --- | --- | ---: | --- |
| [Coconut](https://openreview.net/forum?id=Itxz7S4Ip3) | COLM 2025 | 958 | 最后隐状态回填下一连续位置；HNE 保留四步反馈及跨步梯度 |
| [Huginn](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html) | NeurIPS 2025 | 558 | 共享核心反复更新；HNE 循环读取历史记忆，用共享时序核心继续计算 |
| [CODI](https://aclanthology.org/2025.emnlp-main.36/) | EMNLP 2025 | 354 | 显式／隐式分支的指定 token 表示蒸馏；启发为隐状态增加可检验目标 |
| [MoR](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8b08bbf8b420faa6eeb4020720582ec7-Abstract-Conference.html) | NeurIPS 2025 | 191 | 学习逐 token 递归深度；用于后续按输入分配计算预算 |
| [Soft Thinking](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f7396d1c54d51416958d63e285377103-Abstract-Conference.html) | NeurIPS 2025 | 156 | 词分布加权为连续概念；启发保留多条证据的软组合 |
| [Reasoning by Superposition](https://proceedings.neurips.cc/paper_files/paper/2025/hash/72c363c2a573ca2128bd176d3317696b-Abstract-Conference.html) | NeurIPS 2025 | 96 | 连续状态并行表示图搜索分支；用于设计可控的多证据机制实验 |
| [CTM](https://papers.nips.cc/paper_files/paper/2025/hash/1f1628d502c62ef3725fb3b0b8eb4219-Abstract-Conference.html) | NeurIPS 2025 | 24 | 神经元内部活动历史的同步量生成查询；启发分析状态怎样改变证据读取 |
| [TMLT](https://proceedings.mlr.press/v267/xu25x.html) | ICML 2025 | 21 | 循环步编码调节共享层；直接对应 HNE 核心内部的条件计算 |

引用规模和可迁移性分开判断。CODI 影响较大，但原训练依赖文字推理教师；TMLT 的步条件机制直接对应我们的共享时序核心。CTM 的神经元同步发生在模型内部，HNE 的 N 表示观测指标关系，不能混同。Superposition 的图可达性证明也不是生理因果机制的证明。

## 2026 年的新机制

| 论文 | 正式发表 | 引用 | 原论文解决的问题与 HNE 取舍 |
| --- | --- | ---: | --- |
| [SIM-CoT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d087955ee13fe9a7402eedec879b9c3-Abstract-Conference.html) | ICLR 2026 | 74 | 增加隐步骤时表示趋同；训练辅助解码器提供逐步语义监督。HNE 每步预测未来波动摘要 |
| [LoopFormer](https://loopformer.github.io/) | ICLR 2026 | 71 | 固定深度难适应不同预算；用内部时间、步长调节核心并训练长短轨迹。HNE 先检验四步内的调节 |
| [CoT²](https://proceedings.iclr.cc/paper_files/paper/2026/hash/85ec26ef94c3acb4c195e905df1ff4f7-Abstract-Conference.html) | ICLR 2026 | 34 | 连续分布承载并行探索；启发检验多证据保留，不把生理观测写成已知搜索树 |
| [LaDiR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/843e1d8ab4819c8c8a4221fce2c8d455-Abstract-Conference.html) | ICLR 2026 | 25 | 潜空间扩散反复修订推理表示；借鉴状态修订问题，暂不引入 VAE／扩散全套训练 |
| [DIS](https://proceedings.mlr.press/v306/asadulaev26a.html) | ICML 2026 | 10 | 递归中存在无效计算；用渐进改进监督提高每步作用。HNE 检查每步预测误差 |
| [∇-Reasoner](https://openreview.net/pdf?id=pEJAja73dk) | ICLR 2026 | 7 | 推理时对潜在轨迹求梯度；借鉴状态干预，未来结局不用于在线奖励 |
| [DynaSteer](https://proceedings.mlr.press/v306/wang26hz.html) | ICML 2026 | 4 | 动态表示编辑纠正轨迹；借鉴干预分析，并计入额外前向成本 |

LoopFormer 的完整方法还包括长短轨迹的 shortcut consistency。Huginn 的消融指出，显式步条件也可能影响路径独立性及深度外推。因此 HNE 当前先固定四步，检验条件调节是否有益；可变深度要另做训练与比较。

## 已读取的作者代码

已下载论文 PDF 和关键源码，未克隆训练数据和模型权重。缓存位于 `cache/Literature/latent-reasoning`。原有源码的 blob、新增仓库的 tree SHA 记录在文献 JSON；下面是阅读入口。

| 论文 | 作者代码 | 核对内容 |
| --- | --- | --- |
| Coconut | [coconut.py](https://github.com/facebookresearch/coconut/blob/HEAD/coconut.py) | 分段前向、最后隐状态回填、梯度链 |
| Huginn | [model_dynamic.py](https://github.com/seal-rg/recurrent-pretraining/blob/HEAD/recpre/model_dynamic.py) | 输入注入、共享核心、深度展开 |
| CODI | [model.py](https://github.com/zhenyi4/codi/blob/HEAD/src/model.py) | 教师／学生前向、表示蒸馏与反馈投影 |
| MoR | [expert_choice_router.py](https://github.com/raymin0223/mixture_of_recursions/blob/HEAD/model/mor_model/expert_choice_router.py) | 路由容量、递归深度及计算对照 |
| Soft Thinking | [sampler.py](https://github.com/eric-ai-lab/Soft-Thinking/blob/HEAD/sglang_soft_thinking_pkg/python/sglang/srt/layers/sampler.py) | 概率归一化、软嵌入、熵 |
| Superposition | [coconut.py](https://github.com/Ber666/reasoning-by-superposition/blob/72d22af7ec1533aef6bd208e3f08a819d3e1b47f/coconut.py) | 连续反馈与图搜索任务 |
| CTM | [ctm.py](https://github.com/SakanaAI/continuous-thought-machines/blob/HEAD/models/ctm.py) | 神经元历史、同步量、查询 |
| TMLT | [model.py](https://github.com/kevin671/tmlt/blob/72b322843bc029f032fd001b6d6d71b83c758a4d/model.py) | 步编码和层调节 |
| SIM-CoT | [coconut.py](https://github.com/InternLM/SIM-CoT/blob/HEAD/Coconut/coconut.py) | thought index、解释解码器及逐步监督 |
| LoopFormer | [model.py](https://github.com/armenjeddi/loopformer/blob/59a8ae85bf72dd60f722871788faf51d009b920a/model.py) · [train.py](https://github.com/armenjeddi/loopformer/blob/59a8ae85bf72dd60f722871788faf51d009b920a/train.py) | t、Δt 调节 attention／MLP，长短轨迹训练 |
| CoT² | [continuous_train.py](https://github.com/alperengozeten/CoT2/blob/HEAD/train/continuous_train.py) | 连续分布目标及软步骤 |
| LaDiR | [仓库](https://github.com/mk322/LaDiR/tree/5eaa5968393e5e93262bb53b867683704dd1da3a) | `ladir_modeling_flow.py`、`sft_trainer.py` 的去噪和训练 |
| DIS | [仓库](https://github.com/machinestein/Deep-Improvement-Supervision/tree/5de79e3376f27767d456d6b80df0973b73dfe2e8) | TRM 答案扰动、逐步恢复目标和训练入口 |
| ∇-Reasoner | [optimize.py](https://github.com/VITA-Group/Nabla-Reasoner/blob/46a15711cc2a31be62dc24f07cc357e39aa35249/optimize.py) | 潜在 logits 梯度、奖励及候选接受 |
| DynaSteer | [仓库](https://github.com/tianlwang/DynaSteer/tree/acd1a917e01d7c6f4301a11a1e6dfa36bc143db0) | Qwen3 内部编辑和动态 steering |

## 从发展脉络到我们的研究缺口

**推理表示从文字转为连续状态。** Coconut 打通隐状态回填；Soft Thinking 保留多个候选的软表示；CoT² 和 Superposition 研究连续表示如何承载并行探索。HNE 继承“状态在模型内部继续计算”，用多变量时序作为输入。

**增加步骤后，如何保证计算有效？** CODI 用教师表示提供信号，SIM-CoT 将信号细化到步骤，DIS 检查无效递归。我们的训练数据没有逐步题解，但有未来指标的实际观测。因此让各步解码未来窗口的均值和波动幅度，再检验新增计算是否改善最终预警。DIS 原实现中的渐进恢复目标来自正确答案被扰动后的离散网格，不能直接当作人的真实状态轨迹。

**共享权重怎样形成输入相关的计算路径？** Huginn 使用共享递归核心；TMLT 和 LoopFormer 加入循环时间条件；MoR 学习分配深度。HNE 把个人证据、上一隐状态和内部步数同时用于共享 Transformer 的调节，让输入和当前状态决定更新方式。

由此得到具体缺口：**缺少文字推理链的多变量预警任务，怎样让循环核心依据个人历史、指标关系与情境，逐步形成有预测价值的隐状态？**

## HNE 的实现与实验

本轮在 Chronos-2 前十层冻结、最后两层共享循环、Q/V LoRA 和四步反馈上增加：

1. **证据与状态调节核心。** 上一隐状态、H/N/E 投影、循环时间共同生成调节量，作用于时间注意力输入、时间注意力残差、通道注意力残差和前馈残差。调节头零初始化，再学习输入相关的更新。
2. **逐步预测未来动态。** 所有步骤用同一个预测头解码未来窗口均值和 log 标准差；辅助误差按步骤平均。未来值只作为训练目标。这里各步预测同一个未来窗口，没有人为指定“第一步看历史、第二步看网络”的顺序。

[核心实现](../src/hne_latent.py) · [统一配置](../configs/hne_latent_grounded.json) · [训练入口](../scripts/run_hne_latent.py) · [架构说明](../paper/continuous_warning_framework.md)

| RQ | 核心比较 | 需要的证据 |
| --- | --- | --- |
| RQ1：多指标节律证据能否改善紊乱提前预警？ | HNE 对 DNB、FCM、CATCH 及近期公开方法；H/N/E 屏蔽 | 同一未来事件协议的 AUROC、AP；固定误报预算的灵敏度和提前量 |
| RQ2：证据驱动的内部状态细化是否改善预警？ | 完整模型、无调节、同参数仅步条件调节、仅终步辅助监督、同深度反馈重置、单步 | 配对性能差、个体区间、每步未来摘要误差、状态干预及实测成本 |
| RQ3：内部更新规则能否跨领域迁移？ | 两库双向冻结迁移和外部队列；完整反馈对无调节、重置、单步 | 固定来源域权重及变换后的目标域成绩与机制差值 |

两库使用相同核心、目标、人员划分、选轮规则和辅助损失系数。新重置对照同时移除显式反馈与前面潜在位置的 KV 读取。当前比较先在原训练人员的三折内部验证完成，不以外层人员的评估分数选择方案。配置锁定后运行五折、三个种子的主表和消融表。逐步监督学习同一个未来摘要，其作用是强化预测训练；它不自动保证不同步骤具有不同语义或误差单调下降，这两点需要实测。

当前配置固定四步，没有加入自适应退出和长短轨迹一致性。后续若研究计算预算，应同时改变训练预算，报告 1／2／4／8 步的性能—耗时曲线，不能把增加前向次数直接称为机制增益。

## 另列的路线与筛选结果

- **SoftCoT，ACL 2025，210 引用。** [论文](https://aclanthology.org/2025.acl-long.1137/) · [代码](https://github.com/xuyige/SoftCoT)。辅助模型生成软思维再投影到基模，主要服务第二篇“小模型指导大模型”。
- **Rethinking LLM Reasoning: From Explicit Trajectories to Latent Representations，ICLR 2026。** [论文](https://openreview.net/pdf/81998f90e3cff2c52213db93dfa73043f4a4396f.pdf) · [LRT 代码](https://github.com/MobiusDai/LRT)。OpenAlex 本轮未精确匹配，引用数不填；辅助推理网络归第二篇参考。
- **Reasoning with Latent Thoughts，ICLR 2025，242 引用。** [论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/2676109d49d1eb26d6bc584a8f556305-Paper-Conference.pdf)。作为共享深度与参数量的理论依据；未核实作者代码，不计入 15 篇实现清单。
- **What Makes Effective Supervision in Latent Chain-of-Thought，ICML 2026，2 引用。** [论文](https://proceedings.mlr.press/v306/chen26bi.html) · [仓库](https://github.com/EIT-NLP/Supervision-in-Latent-CoT)。仓库本次仅有代码即将发布的说明，作为监督机制理论参考。
- **HRM、TRM。** 检索记录分别为 285、266 引用；未确认正式顶会／顶刊出处，不凭引用数列入已核实会议清单。
- **iCoT、SoftCoT++、Ouro。** 保留既有阅读缓存。iCoT 在时间范围外，其余未在本轮补齐正式发表资格。LatentSeek 检索返回不同标题，未借用该论文引用数。

H/N/E 信息的有效性、内部更新的有效性、跨领域迁移的有效性分别由三个 RQ 检验。第一篇的方法主张落在证据条件化的内部更新与未来动态监督；数学题上的推理成绩用于理解机制，不作为时序预警增益的证据。

## 轨迹评估的补充阅读

- **Latent Thinking Optimization（ICLR 2026）。** [论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a4b1a25e1b37a17733c1fea50d0b7446-Abstract-Conference.html) · [作者代码](https://github.com/ninglab/LTO)。已读论文第 4 节和作者的接受—拒绝采样实现：从训练样本的内部状态学习轨迹质量评分，按评分选择候选轨迹。HNE 可借鉴它构造“状态质量是否可预测”的机制实验，评分器只用训练人员的结果学习。其轨迹选择与 HNE 的核心更新训练分别评估。
- **Tracing the Traces（ICLR 2026）。** [论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/03e81b3bdb6c0a61093defd319d12203-Abstract-Conference.html) · [作者代码](https://github.com/martinagvilas/LT-signals)。已读第 3 节及表示变化度量源码。净位移、累计位移与更新方向一致性可用于描述 HNE 的四步轨迹，并与各步预测误差配对分析；“轨迹有变化”本身不等于“预警更准确”。
- **HyRea（ICLR 2026）。** [论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2cb880950081ceb85100951b0ad0d542-Abstract-Conference.html) · [作者仓库](https://github.com/zhaoyiran924/HyRea)。显式／隐式切换及 GRPO 提供另一种策略训练思路，但作者当前尚未发布实现，不计入代码清单。

这两篇轨迹评估论文和 HyRea 的引用数本轮未取得，未填入高引用排名。主清单的 15 篇与上述补充材料分别标记。
