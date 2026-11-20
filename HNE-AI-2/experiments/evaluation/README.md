# 独立评价统计与论文图表

主责：**B**；交叉核验：**A**。状态：`planned`。

所有方法是否在同一估计对象上被公平评分，增益和不确定性是否支持论文主张？

## 任务

- `evaluation-1`：固化键/来源校验、折内与人员内指标、配对bootstrap、六项主检验Holm和未知结局回放
- `evaluation-2`：统一校准、风险/保留率、预定病例选择；验收工程合同而非复制实现测试
- `evaluation-3`：由同一结果注册生成主文/附录全部图表与源CSV；A复核机制和方法文字

## 验收

- 各种子先算指标再平均；不把bootstrap或种子当独立新样本
- 阈值只看源验证，未知报警保留静默影响；确认性与探索性分析分开
- 英文居中标题、三线表、数字为主、主表公开算法有引用且Ours最后，无表注、无框架图

## 已有公开入口

- [scripts/export_guidance_tables.py](../../scripts/export_guidance_tables.py)

待交付或对齐的辅助入口：`scripts/analyze_local_experiments.py`、`scripts/replay_local_baselines.py`、`scripts/export_local_experiments.py`、`scripts/review_local_outputs.py`。这些文件目前不在公开仓库；由主责者核验现有私有实现后单独交付，不能当作已同步代码。

完整运行前置模块：无，可独立准备。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：CAL, CASE, INVAR。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
