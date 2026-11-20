# 缺失漂移与表示稳健性

主责：**B**；交叉核验：**A**。状态：`planned`。

指导在缺失、噪声、参考污染、单位和通道变化下何时失效？

## 任务

- `robustness-1`：实现可复现的一因素扰动器：块缺失、情境缺失、参考污染、噪声、时钟延迟和人员错配
- `robustness-2`：完成单位/重复字段/通道表示变化检查；同步变换参考元数据
- `robustness-3`：对A冻结的模型做成对评分；输出损失增加、覆盖、关闭和失败分层

## 验收

- 保留干净标签和起点，不因扰动重新筛选有利子集
- 缺失不当零观测；单因素与组合实验分开
- 数值接口恒等与经验稳健性是不同结论

## 已有公开入口

- [src/protocol.py](../../src/protocol.py)
- [scripts/verify_design_core.py](../../scripts/verify_design_core.py)

待交付或对齐的辅助入口：`configs/paper_experiments.json`、`scripts/verify_complete_interfaces.py`。这些文件目前不在公开仓库；由主责者核验现有私有实现后单独交付，不能当作已同步代码。

完整运行前置模块：core, evidence_controls。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：STRESS, REPRESENT。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
