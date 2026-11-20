# 公开基线与冻结探针

主责：**B**；交叉核验：**A**。状态：`planned`。

在同一终点、起点、信息预算和选择规则下，强公开方法能达到什么水平？

## 任务

- `baselines-1`：核对既有HGB/RF/XGBoost/LightGBM/HYDRA入口、评价规则及候选拟合预算
- `baselines-2`：实现PatchTST-event、CATCH-event、A2P-event和冻结C2 probe的任务适配
- `baselines-3`：接入统一六候选选择预算、全合法起点、3种子与人员折；交付预测适配器

## 验收

- 不填入原论文其他任务成绩
- 训练/验证/评估人员分离且全方法键一致
- 无GPU也能运行短检查；正式资源和输入单独登记

## 已有公开入口

- [scripts/run_public_baselines.py](../../scripts/run_public_baselines.py)
- [scripts/fit_public_trees.py](../../scripts/fit_public_trees.py)

待交付或对齐的辅助入口：`scripts/run_local_baselines.py`。这些文件目前不在公开仓库；由主责者核验现有私有实现后单独交付，不能当作已同步代码。

完整运行前置模块：无，可独立准备。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：EXT。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
