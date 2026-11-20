# 数据效率与全流程成本

主责：**A**；交叉核验：**B**。状态：`planned`。

性能需要多少人员和计算；扩展骨干或在线更新是否改变结论？

## 任务

- `efficiency-1`：执行标签盲选的嵌套25/50/75/100%人员设置，记录内层可行性
- `efficiency-2`：测完整S/L/G、搜索、OOF、个人适配、缓存与部署的真实成本；交给B汇总
- `efficiency-3`：仅在需要骨干普适性主张时扩展骨干；ONLINE为主结果之外的可选成熟标签更新

## 验收

- 100%与干净设置复用；记录失败及全部候选成本
- 无独占GPU实测不能由CPU小试推算GPU总工时
- 第二骨干和在线扩展不能顶替主对照或混入冻结主结果

## 已有公开入口

- [scripts/benchmark_key_comparison.py](../../scripts/benchmark_key_comparison.py)

待交付或对齐的辅助入口：`configs/paper_experiments.json`。这些文件目前不在公开仓库；由主责者核验现有私有实现后单独交付，不能当作已同步代码。

完整运行前置模块：core。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：DATA, BACKBONE, ONLINE。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
