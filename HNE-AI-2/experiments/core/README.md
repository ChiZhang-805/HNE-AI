# 共享基线与内部及融合主对照

负责人 **A**；另一侧负责交付复核。任务事实以 [plan.json](plan.json) 为准。

研究问题：内部注入能否超越同一冻结基线上的完整输入、输出和隐藏证据融合？

当前证据：8组真实骨干短训练和24项接口检查已完成；主方法全折效果尚未完成。

## 本模块任务

1. 交付统一M04及S/证据hash，补齐全折和公平六候选编排

2. 完成A负责方法的5折3种子；接入B的M03/M07/M08/M09/M11输出

3. 记录真实attention/hidden/logit变化、always-on/selected及显式关闭恒等

## 入口与边界

- [scripts/run_key_comparison.py](../../scripts/run_key_comparison.py)

- [src/train_comparison.py](../../src/train_comparison.py)

- [src/chronos_bridge.py](../../src/chronos_bridge.py)

- [src/fusion/evidence.py](../../src/fusion/evidence.py)

- [scripts/verify_complete_interfaces.py](../../scripts/verify_complete_interfaces.py)

完整运行前置模块：teacher。准备代码、合成生成器和接口测试不必等待完整模型。

旧入口和历史结果不移动；新实验专用脚本/配置放本目录。共享训练代码由A整合，评分和图表入口由B整合。

## 验收

- 一个fold/seed共享唯一匹配的M04；特殊锚点单独登记

- 至少训练6轮与从第1轮选型分开，残差含epoch0

- 输入残差经过完整骨干；指导阶段冻结L/事件头；不把工程检查抵扣390评价

认领/完成收据放[runs/](runs/)。原始记录、逐起点预测和权重进入Git忽略的私有实验输出路径。

[两人协作规则](../../docs/collaboration.md) · [共享接口](../../docs/interfaces.md) · [完整实验矩阵](../../experiment_matrix.md)

## 当前开发入口

`python experiments/core/run.py --code-commit <source-commit>`。配置见 `config.json`。先从已完成教师的 selection.json 读取固定S，再重训共同M04。M10与M12分别采用3个学习率乘2个weight decay，共6候选，每个训练6轮；第1轮起选型，另含epoch0。M12采用node/edge-only输入，额外12维摘要置零。保留selected与always-on，按源验证NLL选型。仅复用经逐份精确重算确认的原骨干前缀，不导入旧M04。单源折、抽样训练起点、完整源验证，不能当作正式外层测试。所有原始产物留私有忽略目录。
