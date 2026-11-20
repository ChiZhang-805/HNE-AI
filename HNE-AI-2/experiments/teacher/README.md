# 教师条件预测与证据

主责：**A**；交叉核验：**B**。状态：`planned`。

个人参考规律中的N/E条件信息是否在人员外样本上改善预测，而不是仅产生非零失配？

## 任务

- `teacher-1`：比较既有共享教师、纯H和可关闭N/E条件残差；源内层预测损失选型
- `teacher-2`：复核H/HN/HNE和单源目标的校准、条件增益、参考长度与教师集成
- `teacher-3`：交付带排除审计和hash的固定证据包，保持旧教师与旧缓存可追溯

## 验收

- 每个被评分人员被排除于对应共享教师拟合和选择
- 参考28/14/14时间边界通过
- 同时报告条件增益、NLL、覆盖/PIT与失败；不用目标预警成绩选教师

## 已有公开入口

- [src/precursor_guidance.py](../../src/precursor_guidance.py)
- [src/train_evidence.py](../../src/train_evidence.py)
- [scripts/select_small_predictor.py](../../scripts/select_small_predictor.py)

待交付或对齐的辅助入口：`scripts/audit_small_predictor.py`。这些文件目前尚未公开交付，不应视为可直接运行。

完整运行前置模块：无，可独立准备。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：SENS。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
