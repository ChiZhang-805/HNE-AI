# 折外效用与条件干预

主责：**A**；交叉核验：**B**。状态：`planned`。

能否在结局未知时预测开启指导的效用并降低有害干预？

## 任务

- `utility-1`：构建整条S/L/事件头/G及其选择均排除被评分人员的嵌套OOF
- `utility-2`：以成熟标签的BCE_off−BCE_on训练效用回归器，源验证锁阈值
- `utility-3`：比较always-off/on、质量门控、学习门控与不可部署oracle；保存所有起点

## 验收

- 禁止使用旧pilot的样本内L缓存充当OOF
- 部署决策不得读取label或真实utility
- 报告全体风险、保留率、有害比例和失败，不只评接受子集

## 已有公开入口

- [src/precursor_guidance.py](../../src/precursor_guidance.py)

待交付或对齐的辅助入口：`configs/paper_experiments.json`。这些文件目前尚未公开交付，不应视为可直接运行。

完整运行前置模块：core, evaluation。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：GATE。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
