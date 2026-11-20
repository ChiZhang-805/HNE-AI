# HNE-AI-2 协作约定

- 保留所有有用代码、历史结果和来源；新内容整合到对应稳定名称，不添加v1/v2/new/final后缀。
- 先检查Git状态，再读`docs/collaboration.md`、`docs/interfaces.md`、`experiments/registry.json`及目标模块`plan.json`。
- 按实验组织目录；A负责方法/共享训练，B负责对照/独立评价。不要建立按人划分的代码树，不重复训练两套共同M04。
- 新实验专用实现放`experiments/<topic>/`，可复用实现放`src/`；保留旧入口路径。
- 规划目录与训练实现区分：`pending_local_entry_points`列出尚未公开交付的辅助入口。先核验代码/配置，不把任务计划当作实验已完成。
- A整合共享训练和协议变更，B核对数据与评估一致性。保持固定标签、既有人员折、时间边界及公平选型预算。
- 运行前推送唯一键的脱敏认领收据；真实完成必须有日志、成本和来源。不能用本地锁代替跨机器协调。
- 人员数据、逐起点预测、个人校准、权重和密钥不进公共GitHub；新原始产物放Git忽略的`artifacts/experiments/`和`results/experiments/`。
- 第一篇的目录、数据、权重和实验保持只读；不得占用或中断它的任务。核心实验无需API，不自动租GPU，不跨项目回退密钥。
- 论文表格使用英文居中标题和LaTeX三线表，以数字为主、无表注。表一公共算法引用原文，Ours最后；表二为消融。暂不制作框架图。
- 修改任务索引后运行`python scripts/validate_collaboration.py`。只暂存明确文件；不得force push/reset/clean覆盖他人工作。

- 输入统一经`src/study_inputs.py`解析，默认第二篇项目根目录；不回退到第一篇。私有交接按`docs/private_handoff.md`验收，历史manifest保留原记录。
- 私有交接ZIP、模型、证据和预测不得进入Git；`packages/`、`artifacts/`和相关数据目录保持忽略。
