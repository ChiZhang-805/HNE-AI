# 按实验组织的协作入口

目录按研究问题划分；A/B只表示主责。旧代码和已有结果保留原位。

| 实验 | 主责 | 前置完整运行 |
| --- | --- | --- |
| [教师条件预测与证据](teacher/) | A | 无 |
| [公开基线与冻结探针](baselines/) | B | 无 |
| [共享基线与内部及融合主对照](core/) | A | teacher |
| [重训组件消融](ablation/) | A | core |
| [统计结构检索与错配证据](evidence_controls/) | B | core |
| [折外效用与条件干预](utility/) | A | core, evaluation |
| [跨人跨域迁移与适配边界](transfer/) | A | core |
| [缺失漂移与表示稳健性](robustness/) | B | core, evidence_controls |
| [合成机制与冻结干预](mechanisms/) | B | core |
| [预测间隔与预警距离](horizons/) | A | core |
| [数据效率与全流程成本](efficiency/) | A | core |
| [独立评价统计与论文图表](evaluation/) | B | 无 |

[两侧完整任务说明](../docs/collaboration.md) · [接口合同](../docs/interfaces.md) · [机器索引](registry.json)

先运行`python scripts/validate_collaboration.py`。该检查不启动训练；计划状态不等于任务已认领。
