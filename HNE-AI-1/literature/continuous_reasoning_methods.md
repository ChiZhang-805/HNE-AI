# 连续推理如何用于多维紊乱提前预警

本次下载并阅读 15 篇方法论文。14 篇核对了作者的公开实现，SoftCoT 和 SoftCoT++ 使用同一仓库的不同训练入口。论文 PDF 与本次实际读取的源文件放在项目缓存 `cache/Literature/latent-reasoning`，该目录的 `sources.json` 对照论文、源码链接和已取得的 Git blob。下文的实验对象来自原论文，对 HNE 的取舍来自本研究。

从 iCoT 到 Coconut，变化发生在推理时传递的对象：前者让模型逐渐省略显式步骤，后者把隐状态直接再作为输入。CODI、SIM-CoT 随后追问这种隐表示靠什么监督。Huginn、HRM、TRM 将问题转到共享计算深度，CTM 把内部状态接回输入读取。对未来紊乱预警，最能区分这些解释的实验，是在相同历史、参数量、步数和监督下，只改变下一查询是否依赖上一状态。

## iCoT

乘法、GSM8K。把显式步骤按课程逐渐移除，比较最终答案正确率。 `scheduled_to_remove` 决定删去多少解释词，作者还改变移除位置与扰动。 减少监督并不自动产生可反复更新的状态。HNE 的单步模型也必须接受相同的预警监督。 [原文](https://arxiv.org/abs/2405.14838) · [作者实现](https://github.com/da03/Internalize_CoT_Step_by_Step/blob/HEAD/src/train.py)

## Coconut

在 GSM8K、ProntoQA 和 ProsQA 上，把最后隐状态作为下一连续位置的输入，比较显式推理、直接答题和潜空间推理。 `Coconut.forward` 按潜位置分段前向，把上一步隐状态写回输入嵌入。 沿训练图传递状态，并在相同步数下切断这条通路。传感数据没有语言推理链，不能照搬其分阶段去词训练。 [原文](https://arxiv.org/abs/2412.06769) · [作者实现](https://github.com/facebookresearch/coconut/blob/HEAD/coconut.py)

## CODI

显式与隐式分支共享模型，在答案附近蒸馏隐表示；另做域外算术测试。 `CODI.forward` 展开隐步骤并计算教师、学生和对齐损失。 中间表示需要学习信号。HNE 在各步读出同一未来结局，并分别比较单步和多步。 [原文](https://arxiv.org/abs/2502.21074) · [作者实现](https://github.com/zhenyi4/codi/blob/HEAD/src/model.py)

## SoftCoT

辅助语言模型产生一串连续向量，由投影层接入基模；在 GSM8K 和 StrategyQA 等任务上评估。 `get_inputs_embeds_for_base_model` 将辅助表示投影到基模嵌入空间。 借鉴异构证据的维度投影。小模型指导大模型另归第二篇。 [原文](https://arxiv.org/abs/2502.12134) · [作者实现](https://github.com/xuyige/SoftCoT/blob/HEAD/llm_model.py)

## SoftCoT++

在多条软思维路径之间增加差异并作推理时扩展，与自一致采样比较。 `train_softcotpp.py` 进入 scaling-nce 分支；同仓 `llm_model.py` 实现多条软提示及对比损失。 随机种子的模型集成和单次前向中的推理深度应分别记账。 [原文](https://arxiv.org/abs/2505.11484) · [作者实现](https://github.com/xuyige/SoftCoT/blob/HEAD/train_softcotpp.py)

## SoftThinking

以词分布的加权嵌入表示概念，报告数学和编程任务的正确率、生成长度。 采样器保留 top-k 概率并归一化，还计算熵。 可借鉴连续加权读取。这里的加权对象是 H/N/E 观测证据，权重本身不能证明生理机制。 [原文](https://arxiv.org/abs/2505.15778) · [作者实现](https://github.com/eric-ai-lab/Soft-Thinking/blob/HEAD/sglang_soft_thinking_pkg/python/sglang/srt/layers/sampler.py)

## SIM-CoT

训练用解释解码器把每个隐步骤对齐到题解步骤，在 GSM8k-Aug 及外部算术题上检查隐状态是否退化。 作者的 Coconut 分支按 thought index 构造解释输入，训练时监督相应隐状态。 同一步反复输出相近风险也可能只是重复。需在等深度下固定查询，检验回读是否有贡献。 [原文](https://arxiv.org/abs/2509.20317) · [作者实现](https://github.com/InternLM/SIM-CoT/blob/HEAD/Coconut/coconut.py)

## CoT2

在可枚举中间分支的算术与逻辑任务中研究连续分布能否同时表示多个可能的下一步。 `train_soft_steps_batch` 用下一词分布与目标分布学习软步骤，数据生成器可指定不同搜索分支。 节律数据缺少可验证的搜索树。三类证据能共同进入下一状态，并不足以声称模型进行了树搜索。 [原文](https://arxiv.org/abs/2505.23648) · [作者实现](https://github.com/alperengozeten/CoT2/blob/HEAD/train/continuous_train.py)

## LoopedThoughts

用加法、p-hop 检索和 i-GSM 分开测试参数共享、展开深度和计算量。 本文作机制与对照设计依据，本次未核实作者实现。 单步回答“多算是否有用”，四步固定查询回答“同样计算量下，按隐状态回读是否有用”。 [原文](https://arxiv.org/abs/2502.17416)

## Huginn

预训练共享循环块，在 GSM8K、MATH 等任务上改变推理深度并检查收益。 `model_dynamic.py` 将输入处理、循环核心和输出处理分开。 固定预训练时间序列前缀，循环内共享更新权重。增加步数的效果要由对照估计。 [原文](https://arxiv.org/abs/2502.05171) · [作者实现](https://github.com/seal-rg/recurrent-pretraining/blob/HEAD/recpre/model_dynamic.py)

## HRM

高低两级隐状态以不同频率更新，在 Sudoku、迷宫及 ARC 上做计算深度实验。 `HierarchicalReasoningModel_ACTV1_Inner` 分别维护高低状态，外层负责跨步携带状态。 观察的物理时间与内部更新序号要分清，同一条未来窗口的预测可包含多次隐更新。 [原文](https://arxiv.org/abs/2506.21734) · [作者实现](https://github.com/sapientinc/HRM/blob/HEAD/models/hrm/hrm_act_v1.py)

## TRM

用更小的共享网络反复更新潜变量与答案，在 Sudoku-Extreme 等任务上消融有效深度和监督方式。 `trm.py` 分别维护潜状态和答案表示，并在递归迭代后读出。 把可读的基础预警分数和反复更新的修正量显式分开，可与基础分数直接比较。 [原文](https://arxiv.org/abs/2510.04871) · [作者实现](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/HEAD/models/recursive_reasoning/trm.py)

## CTM

令模型拥有独立于输入时钟的内部时间，用神经元活动同步产生查询；在迷宫及图像任务上分析迭代过程。 `compute_synchronisation` 和查询投影把内部动态连接到下一轮输入读取。 上一状态生成新查询，同一输入在不同步可有不同的证据权重。HNE 实现了回读，并未复刻神经元同步。 [原文](https://arxiv.org/abs/2505.05522) · [作者实现](https://github.com/SakanaAI/continuous-thought-machines/blob/HEAD/models/ctm.py)

## MoR

逐词分配递归深度，比较路由方式、共享权重和 FLOPs。 专家选择路由按容量在 token 上取 top-k，再执行递归块。 不同深度可能只反映算力差异。本轮固定四步并保留等计算量对照。 [原文](https://arxiv.org/abs/2507.10524) · [作者实现](https://github.com/raymin0223/mixture_of_recursions/blob/HEAD/model/mor_model/expert_choice_router.py)

## Ouro

把共享层循环和深度分配纳入预训练，比较不同退出方式和展开深度。 作者模型的 `early_exit_gate` 从隐状态产生退出信号，在循环期间记录门值。 固定深度收益尚需证据，当前版本没有自适应停止，亦不声称节省推理算力。 [原文](https://arxiv.org/abs/2510.25741) · [作者实现](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)

## 时序任务与原论文之间的差异

数学题可以为每个中间步骤提供参考答案。可穿戴记录只有未来窗口的传感器结局，本研究只用同一未来事件标签监督中间读出。留出人员、固定证据查询、重置隐状态、删除证据通路和冻结来源模型的跨域迁移，分别检验预警、内部计算和外推。
