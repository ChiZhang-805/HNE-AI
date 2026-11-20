# 统计结构检索与错配证据

主责：**B**；交叉核验：**A**。状态：`planned`。

收益来自条件失配、一般结构、历史检索、缺失模式，还是额外读出参数？

## 任务

- `evidence_controls-1`：在实验目录实现M07统计、M08结构、M09时间合格检索和M11打乱证据适配器
- `evidence_controls-2`：维持相同S/L冻结接口，补全coverage-only、raw-error、node/edge-only与无证据控制
- `evidence_controls-3`：区分重训对照和冻结推理错配；记录证据可用性与时间资格

## 验收

- 依赖方法共用A交付的M04，不能私自另训基线
- 检索库与邻居选择只见合法历史；未来值仅可用作训练目标
- 完整融合保留的信息与额外摘要必须显式对齐

## 已有公开入口

- [src/evidence_alignment.py](../../src/evidence_alignment.py)
- [src/pack_guidance.py](../../src/pack_guidance.py)

待交付或对齐的辅助入口：`scripts/run_cached_evidence.py`、`scripts/run_cached_null_controls.py`。这些文件目前尚未公开交付，不应视为可直接运行。

完整运行前置模块：core。前置只约束完整运行，不阻止接口、适配器及合成检查。

实验包：EVID。资源：CPU for preparation and bounded checks; full refits require measured capacity, dedicated GPU recommended where appropriate; no mandatory API。

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。本次仅建立计划，不新增实验完成声明。
