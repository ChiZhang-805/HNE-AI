# 跨人跨域迁移与适配边界

主责：**A**；交叉核验：**B**。状态：`planned`。

重学当地参考规律后，固定指导规则能否跨域复用；需要多少目标监督？

## 任务

- `transfer-1`：完成Bris↔RES及人体→Air的冻结T0和无标签参考重学T1
- `transfer-2`：单列T2/T3目标监督适配预算及失败边界
- `transfer-3`：交付锁定源模型、阈值、适配记录；由B独立回放和评分

## 验收

- 目标监督与无标签适配不能混称
- 外部保留结局在框架/选择锁定后评分
- 所有迁移比较引用相同源检查点与目标合法起点

## 已有公开入口

- [scripts/run_guidance_transfer.py](../../scripts/run_guidance_transfer.py)
- [configs/guidance_transfer.json](../../configs/guidance_transfer.json)

待交付或对齐的辅助入口：`scripts/prepare_air_quality.py`。这些文件目前不在公开仓库；由主责者核验现有私有实现后单独交付，不能当作已同步代码。

完整运行前置模块：core。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：TRANSFER, TSUP。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
