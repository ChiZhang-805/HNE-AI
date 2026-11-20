# 预测间隔与预警距离

主责：**A**；交叉核验：**B**。状态：`planned`。

七日持续终点保持不变时，提前间隔拉长会损失多少预测能力？

## 任务

- `horizons-1`：沿用1/3/7/14日可评分性审计，生成各任务独立配置而不覆写主协议
- `horizons-2`：为不同gap重新训练所有指定成对方法并锁定源阈值
- `horizons-3`：交付完整资格日历与未知结局，由B做报警及共同起点比较

## 验收

- 不把可评分性计数当效果
- 终点长度固定七日，主gap1继续复用原协议
- 报告各gap人数、阳性、未知及共同起点；不把操作性一天lead称临床提前量

## 已有公开入口

- [configs/daily_protocol.json](../../configs/daily_protocol.json)
- [scripts/audit_forecast_horizons.py](../../scripts/audit_forecast_horizons.py)

完整运行前置模块：core。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：LEAD。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
