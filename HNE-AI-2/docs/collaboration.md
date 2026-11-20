# 两人协作与实验交付

适用范围是 **HNE-AI-2：小模型指导大模型的紊乱预警研究**。目录按实验问题组织，A/B表示职责，不建立按人划分的代码目录。

- **A：本侧研究者及本聊天的 Codex**，负责小模型、共享大模型训练、内部指导、同证据融合、折外门控及需要重新训练全流程的实验。
- **B：另一位研究者及其 Codex**，负责公开算法基线、替代证据和检索、机制与扰动、独立评价及论文图表。
- A/B不是假定的GitHub账户，未配置自动指派或CODEOWNERS。任务说明不会自动启动另一人的Codex。

公开代码已补齐第二篇独立输入入口、完整证据接口及相关训练/基线读取修正。数据与配对M04/S通过私有交接包提供，配置和验收见[交接说明](private_handoff.md)。已有公开结果保持原样，本次没有新增公开逐人数据或模型。仍缺少的辅助入口列在各模块`pending_local_entry_points`；正式全折调度和公平候选搜索仍需继续完成。

## 实验模块与责任

| 目录 | 主责 | 交付范围 | 实验包 |
| --- | --- | --- | --- |
| [teacher](../experiments/teacher/) | A | 教师条件预测与证据 | SENS |
| [baselines](../experiments/baselines/) | B | 公开基线与冻结探针 | EXT |
| [core](../experiments/core/) | A | 共享基线与内部及融合主对照 | CORE |
| [ablation](../experiments/ablation/) | A | 重训组件消融 | ABL |
| [evidence_controls](../experiments/evidence_controls/) | B | 统计结构检索与错配证据 | EVID |
| [utility](../experiments/utility/) | A | 折外效用与条件干预 | GATE |
| [transfer](../experiments/transfer/) | A | 跨人跨域迁移与适配边界 | TRANSFER, TSUP |
| [robustness](../experiments/robustness/) | B | 缺失漂移与表示稳健性 | STRESS, REPRESENT |
| [mechanisms](../experiments/mechanisms/) | B | 合成机制与冻结干预 | MECH, SYN |
| [horizons](../experiments/horizons/) | A | 预测间隔与预警距离 | LEAD |
| [efficiency](../experiments/efficiency/) | A | 数据效率与全流程成本 | DATA, BACKBONE, ONLINE |
| [evaluation](../experiments/evaluation/) | B | 独立评价统计与论文图表 | CAL, CASE, INVAR |

共12个目录，覆盖19个实验包和13个规划方法。方法逐项归属见[机器索引](../experiments/registry.json)。规划核心比较为两库×5折×3种子×13方法，共390个评价单元；完整初始规划990个单元包含其他方法。它们是任务规模，不是已完成数量，也不等于训练次数。

**共同基线只训练和登记一套。** A维护共享M04及主比较调度，B提供其负责方法的适配器。所有依赖方法引用同一折、种子的M04和S证据，不能两人各训练一套后直接拼表。公开方法配置已对齐13方法标识；A仍须完成正式全折调度和公平候选搜索后再按计划执行。

## 两边现在分别做什么

**A先做教师与共同接口。** 在源内层人员划分上比较既有共享教师、纯H教师、可关闭N/E条件残差；只按源预测损失选择设置。交付条件增益、校准的评价入口与排除审计；核对共享训练代码、M04、内部指导和完整证据融合的接口。先阅读已有代码与原论文，不根据目标集预警成绩反复选框架。

**B先做独立基线与评价。** 核对仓库已有基线和统计公式；实现同一持续紊乱终点的PatchTST-event、CATCH-event和A2P-event适配器，并准备合成机制生成器、扰动器和独立评价器。原论文其他任务的数字不能直接填入本项目主表。无需等待A即可准备这些模块；完整指导/检索比较需等共同S/M04交付。

**第一次交接：** A交付有协议、输入、检查点和代码hash的证据包及共同M04；B交付基线适配器、共同起点和统计检查。在相同小批历史窗口上验证键、mask、概率、时刻边界、显式关闭恒等与梯度隔离。训练权重、人员名单和逐起点记录通过双方授权的私有环境交接，不放公共GitHub。

## 实验推进顺序

1. 双方核对数据资格、既有人员折、协议和资源；运行目录校验。A对齐执行配置，B核对评估合同。
2. A完成教师比较及共同基线；B完成公共基线、评价器、机制生成器与扰动器。
3. 同证据接口锁定后，A执行主方法、无指导/输入/隐藏/输出融合与组件重训；B执行普通统计、结构、检索、错配和无证据对照。保持起点、参数及选型预算可比。
4. A构建全流程人员OOF和效用门控，再做迁移、预测间隔、效率；B独立审计排除条件并评估冻结检查点的稳健性和机制。OOF须覆盖S、LoRA、事件头、G及各自选型。
5. B统一统计、图表和实验文字；A撰写方法、训练、迁移与成本；双方核对结论和失败记录。负结果不删除，不能以改终点或只留最好种子制造优势。

## 文件边界与合并

- 旧`src/`、`scripts/`、`configs/`和历史结果路径保留。新实验专用脚本和配置进入`experiments/<topic>/`；多实验复用代码放`src/`。不使用v1/v2/new/final后缀。
- A整合共享训练：`src/precursor_guidance.py`、`src/train_evidence.py`、`src/train_backbone.py`、`src/chronos_bridge.py`、`src/event_readout.py`、`src/pack_guidance.py`、`src/fusion/`、`src/train_comparison.py`及训练配置。
- B整合公共基线、评价、报警、统计和LaTeX导出；新增对照适配器优先放自己的实验模块，需要共享训练接口时给A提供具体输入/输出样例。
- 数据、标签、人员折和选型是共同合同：A整合变更，B核验成对一致性。不能局部改标签或重分折；保留旧实验对应的协议来源。
- 分支按任务命名，如`exp/teacher-condition`、`exp/baseline-patchtst`、`exp/evaluation-alarm`。合并前拉取最新远端，明确暂存文件，不force push/reset/clean清理别人工作。

## 运行登记、产物与资源

一运行键只允许一个执行者。启动前在对应`runs/`提交脱敏认领收据并确认远端没有重复；本地文件锁不能协调两台机器。示例见[收据模板](../experiments/run_receipt.example.json)。计划状态不代表认领，认领不代表运行完成。

新私有输出使用`artifacts/experiments/<topic>/<dataset>/fold<k>/seed<s>/<method>/<config_digest>/`及对应`results/experiments/.../`。`config_digest`区分设置，不是方案版本号。旧产物留原位置并在manifest中引用。公共仓库只提交审核过的代码、计划、脱敏收据及获授权公开的汇总；不提交逐人数据、预测、个人校准、权重、密钥或完整私有manifest。

状态为`planned → claimed → running → completed`，另有`failed`和`waiting_dependency`。完成必须有真实日志、成本、来源和可追溯结果；pilot、缓存头或短检查不能冒充正式多折实验。

核心实验无需API。先完成可独立开展的CPU准备和短检查；大规模拟合先核对可用设备、数据和共同预算。不得占用或中断第一篇实验，不自动租GPU，不借用其他项目的API账户。

## 论文图表

B负责[表格](../paper/tables/)和[图片](../paper/figures/)索引；A交付训练设置、成本及来源。表一为同终点公共方法比较，Method有原论文引用，Ours最后且不加外部引用；表二为消融。所有表使用英文、居中标题、LaTeX三线表、以数字为主且无表下注释。缺失结果写`--`。保留既有导出入口和结果，不制作框架图。

## 可直接交给另一侧Codex的任务

你负责HNE-AI-2的B侧。先读`AGENTS.md`、本文件、`docs/interfaces.md`、`experiments/baselines/plan.json`和`experiments/evaluation/plan.json`，执行`python scripts/validate_collaboration.py`。从公开基线适配器和独立评价器开始，机制生成器/扰动器可以并行准备。区分已有公开入口与尚待交付的本地脚本；数据和原模型按`docs/reproduction.md`在授权私有环境准备。待A交付共同M04和S后执行依赖比较，避免重训第二套锚点。保留所有有用代码和负结果，不影响第一篇，不自行调用其他项目密钥。

本侧A从`experiments/teacher/plan.json`和`experiments/core/plan.json`开始。目录校验只检查职责、依赖、路径和固定协议，不训练模型、不读取人员结局、不连接网络。
