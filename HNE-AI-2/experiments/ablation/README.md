# 重训组件消融

主责：**A**；交叉核验：**B**。状态：`planned`。

H/N/E、个人适配、分数校准与时间/通道指导各贡献什么？

## 任务

- `ablation-1`：按ABL定义重训H-only、去N/去E、个人/分数校准及单源损失控制
- `ablation-2`：完成绝对误差、time-only、group-only和recent-only成对锚点
- `ablation-3`：各条件保留同预算、同起点和未选检查点；交付给B统一评分

## 验收

- 受影响S/G必须重训
- 每次仅改变预定因素；依赖证据重新构建
- recent-only必须有自己的无指导锚点

## 已有公开入口

- [src/train_evidence.py](../../src/train_evidence.py)
- [src/train_comparison.py](../../src/train_comparison.py)

待交付或对齐的辅助入口：`configs/paper_experiments.json`。这些文件目前尚未公开交付，不应视为可直接运行。

完整运行前置模块：core。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：ABL。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
