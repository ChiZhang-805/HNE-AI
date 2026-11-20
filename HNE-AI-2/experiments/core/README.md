# 共享基线与内部及融合主对照

主责：**A**；交叉核验：**B**。状态：`running`。

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

详见[机器计划](plan.json)、[两人分工](../../docs/collaboration.md)、[接口合同](../../docs/interfaces.md)。认领记录放`runs/`；私有产物路径见`plan.json`。模块任务仍未全部完成；实际进展以 runs 中的单项收据为准。

## 当前开发入口

`python experiments/core/run.py --code-commit <source-commit>`。配置见 `config.json`。先从已完成教师的 selection.json 读取固定S，再重训共同M04。M10与M12分别采用3个学习率乘2个weight decay，共6候选，每个训练6轮；第1轮起选型，另含epoch0。M12采用node/edge-only输入，额外12维摘要置零。保留selected与always-on，按源验证NLL选型。仅复用经逐份精确重算确认的原骨干前缀，不导入旧M04。单源折、抽样训练起点、完整源验证，不能当作正式外层测试。所有原始产物留私有忽略目录。

`verify_batching.py --run-key <private-run-key>`???????????????????????`batching.py`????????????????????????????????

Windows长任务可用 `start_local.ps1 -PythonExecutable <authorized-python.exe> -RunKey <claimed-key> -CodeCommit <runtime-source-commit>`启动隐藏后台进程。它核对配置、检查重复进程、固定低优先级单线程CPU，并把启动收据及标准输出保存在私有运行目录。必须同时核对实际工作进程和检查点更新时间，不能只相信status.json的running字样。首次日志字段冲突修复通过 `continue_run.py` 保留原manifest、源码、第一轮模型和优化器；剔除日志调用后的训练函数AST一致。后续会话中断的未记录计算成本保留为未知，不冒充完整总耗时。

新开发运行可以在独立配置中设置 `include_logit_fusion: true`，让 M13 使用共同 M04 的 logit 和与 M10/M12 相同的节点、边和掩码。M13 使用同一组六候选、六轮预算和验证 NLL 选择规则，同时保留 selected 和 always-on。每种方法显式关闭时直接返回共同 M04 的冻结输出。冻结前后独立前向的最大 logit 误差另写入验证收据。

在空闲设备上先运行 `python experiments/core/verify_stages.py --device cuda`，用公开原骨干和合成窗口验证四个阶段及检查点重载。CPU 检查已通过，CUDA 实际数值和耗时待验证。新 CUDA 开发运行用独立配置设置 `device: "cuda"`，沿用源侧分折和一条 CPU 线程，并在认领后执行 `run.py --config <cuda-config.json> --code-commit <source-commit>`。设备与配置进入来源记录，CUDA 运行重新训练其共同 M04，不能跨设备混用旧运行锚点。原有 CPU 运行按照其不可变源码和配置继续。
