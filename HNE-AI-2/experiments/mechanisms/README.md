# 合成机制与冻结干预

主责：**B**；交叉核验：**A**。状态：`planned`。

证据与风险是否响应可控前兆，并能区分良性变化和无前兆冲击？

## 任务

- `mechanisms-1`：先完成按完整系统划分的配对合成生成器和固定终点检查
- `mechanisms-2`：比较耦合、相位、幅度/方差、良性情境及无前兆冲击
- `mechanisms-3`：冻结后对内部/融合同时施加错序、错边、错人等干预并评分真实变点

## 验收

- 共享轨迹/噪声，按完整系统分割，不按重叠窗口随机分割
- 变点真值仅用于评分，不能进入推理输入
- 良性或突发单类保留NA；不改终点强造阳性

## 已有公开入口

- [src/evidence_alignment.py](../../src/evidence_alignment.py)

待交付或对齐的辅助入口：`configs/paper_experiments.json`。这些文件目前尚未公开交付，不应视为可直接运行。

完整运行前置模块：core。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：MECH, SYN。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
