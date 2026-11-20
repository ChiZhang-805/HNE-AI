# 教师条件预测与证据

主责：**A**；交叉核验：**B**。状态：`running`。

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

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。模块任务仍未全部完成；实际进展以 runs 中的单项收据为准。

## 当前开发入口

`python experiments/teacher/run.py --code-commit <source-commit>`。配置见 `config.json`，设计和选型边界见 `design.md`。首个源折的27个超参数试验已完成，使用3个交叉拟合教师；不读取事件标签，不代表全折SENS完成。

`audit_calibration.py --core-run-key <key>`审计独立个人校准是否改变H/HN/HNE对比。`shared_calibration.py --teacher-run-key <key> --code-commit <commit>`固定已选教师，让各分支共用H的个人校准；首折控制已完成，不改变正在运行的核心证据或模型选择。稳定的条件正增益仍未建立，全折结论尚待检验。
