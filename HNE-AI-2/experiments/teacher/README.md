# 教师条件预测与证据

负责人 **A**；另一侧负责交付复核。任务事实以 [plan.json](plan.json) 为准。

研究问题：个人参考规律中的N/E条件信息是否在人员外样本上改善预测，而不是仅产生非零失配？

当前证据：已完成单源折3学习率×3组件检查；条件增益仍弱；结构比较和全折选择待完成。

## 本模块任务

1. 比较既有共享教师、纯H和可关闭N/E条件残差；源内层预测损失选型

2. 复核H/HN/HNE和单源目标的校准、条件增益、参考长度与教师集成

3. 交付带排除审计和hash的固定证据包，保持旧教师与旧缓存可追溯

## 入口与边界

- [scripts/select_small_predictor.py](../../scripts/select_small_predictor.py)

- [scripts/audit_small_predictor.py](../../scripts/audit_small_predictor.py)

- [src/precursor_guidance.py](../../src/precursor_guidance.py)

- [src/train_evidence.py](../../src/train_evidence.py)

完整运行前置模块：无；可直接准备实现与检查。准备代码、合成生成器和接口测试不必等待完整模型。

旧入口和历史结果不移动；新实验专用脚本/配置放本目录。共享训练代码由A整合，评分和图表入口由B整合。

## 验收

- 每个被评分人员被排除于对应共享教师拟合和选择

- 参考28/14/14时间边界通过

- 同时报告条件增益、NLL、覆盖/PIT与失败；不用目标预警成绩选教师

认领/完成收据放[runs/](runs/)。原始记录、逐起点预测和权重进入Git忽略的私有实验输出路径。

[两人协作规则](../../docs/collaboration.md) · [共享接口](../../docs/interfaces.md) · [完整实验矩阵](../../experiment_matrix.md)

## 当前开发入口

`python experiments/teacher/run.py --code-commit <source-commit>`。配置见 `config.json`，设计和选型边界见 `design.md`。首个源折的27个超参数试验已完成，使用3个交叉拟合教师；不读取事件标签，不代表全折SENS完成。
