# 共享基线与内部及融合主对照

主责：**A**；交叉核验：**B**。状态：`planned`。

内部注入能否超越同一冻结基线上的完整输入、输出和隐藏证据融合？

## 任务

- `core-config-sync`：先核对公开运行配置与13方法规划，补齐M13及一致预算；保留旧配置来源，不把规划当作已交付训练实现。
- `core-1`：交付统一M04及S/证据hash，补齐全折和公平六候选编排
- `core-2`：完成A负责方法的5折3种子；接入B的M03/M07/M08/M09/M11输出
- `core-3`：记录真实attention/hidden/logit变化、always-on/selected及显式关闭恒等

## 验收

- 一个fold/seed共享唯一匹配的M04；特殊锚点单独登记
- 至少训练6轮与从第1轮选型分开，残差含epoch0
- 输入残差经过完整骨干；指导阶段冻结L/事件头；不把工程检查抵扣390评价

## 已有公开入口

- [scripts/run_key_comparison.py](../../scripts/run_key_comparison.py)
- [src/train_comparison.py](../../src/train_comparison.py)
- [src/chronos_bridge.py](../../src/chronos_bridge.py)
- [src/fusion/evidence.py](../../src/fusion/evidence.py)
- [scripts/verify_complete_interfaces.py](../../scripts/verify_complete_interfaces.py)

完整运行前置模块：teacher。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：CORE。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
