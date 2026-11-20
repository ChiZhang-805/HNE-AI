# 第二篇私有交接：配置、校验与运行

本说明配合`HNE-AI-2-private-handoff.zip`（输入和旧M04/S联调）及`HNE-AI-2-baseline-handoff.zip`（历史基线补充）使用。代码在公共GitHub；数据、模型、个人证据和历史预测只通过私有渠道交接。无需第一篇工程，也无需API。

## 包中有什么

- BrisT1D-Open与RESILIENT的处理后小时观测、日级标签、固定人员折和配套清单。
- 原始公开Chronos-2配置和预训练权重。
- 两套相互独立的已有M04/S配对：`cpu_pilot`为单折开发试跑，`contract_check`为短流程工程检查；均是BrisT1D-Open、fold0、seed20261114。
- 每套自己的M04、三个S组件、个人校准、缓存证据、原始manifest、源码快照、状态和耗时记录。
- 原始预测CSV用于复算对照。CSV中有部分本包未携带的历史方法，不要据此认为其全部检查点已经交付。
- 私有总清单中逐文件记录大小和SHA-256，并绑定已验证的公开代码提交。

两套产物不能混用。它们不是两库×五折×三种子的正式共同模型。包内不含第一篇训练模型、不含其他教师候选、不含优化器状态。体积较大的前缀缓存省略；校验脚本从原始模型重新计算一个真实窗口。

## 1. 准备独立代码目录

如果已经在开发，先检查Git状态，保留自己的分支和修改。不要覆盖脏工作区。下面示例用于新的工作目录：

```powershell
git clone https://github.com/ChiZhang-805/HNE-AI.git
cd HNE-AI
```

私有包说明首行和`artifacts/private_handoff/manifest.json`中给出固定提交。用该提交建立自己的任务分支，确保初次复算代码一致：

```powershell
git switch -c exp/b-handoff <说明首行的完整提交哈希>
```

先核对收到的ZIP：

```powershell
Get-FileHash '你的接收目录/HNE-AI-2-private-handoff.zip' -Algorithm SHA256
```

结果应与同目录`SHA256SUMS.txt`一致。ZIP内部已有顶层`HNE-AI-2/`，所以解压到仓库根目录`HNE-AI/`，不要再解压进`HNE-AI-2/`导致重复嵌套。首次解压到新代码目录，已有同名私有文件时先核对哈希，勿强制覆盖。

```powershell
Expand-Archive -LiteralPath '你的接收目录/HNE-AI-2-private-handoff.zip' -DestinationPath .
cd HNE-AI-2
$env:HNE_STUDY_ROOT = (Get-Location).Path
```

Linux/macOS使用相应解压工具，并在`HNE-AI-2/`中执行`export HNE_STUDY_ROOT="$PWD"`。变量只指向你本机第二篇的输入根目录，不指向A的盘符或第一篇项目。

## 2. 先做无依赖完整性校验

```powershell
python scripts/validate_collaboration.py
python scripts/verify_private_handoff.py
```

第二条只使用Python标准库：检查全部包内文件、协议、原始模型、历史源码快照及配对训练数据，并核验当前关键代码与固定提交的文件哈希。必须看到`status: passed`。失败时保留错误，不能通过改manifest或关闭校验继续。

## 3. 创建本项目的Python环境

实测环境为Python 3.13、PyTorch 2.8和`requirements.txt`中的固定依赖。Windows示例：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS用`python3.13 -m venv .venv`及`.venv/bin/python`。依赖安装需要网络；后续校验从本地读取权重，不需要模型下载、API或GPU。不同设备可安装对应的PyTorch 2.8构建，但应记录环境差异并通过复算检查，不擅自改变其他依赖来声称复现成功。

## 4. CPU真实数据与模型联调

```powershell
.\.venv\Scripts\python.exe scripts/verify_private_handoff.py --smoke --profile cpu_pilot
.\.venv\Scripts\python.exe scripts/verify_private_handoff.py --smoke --profile contract_check
```

每条只复算一个验证人员的历史窗口：核对两库固定折，加载对应S，检查该人员未进入该S组件拟合，重建证据并与缓存比较，再加载对应M04，重新编码输入并与原始预测比较。它不训练、不加载评估折结局、不占GPU、不调用API。

结果写入`artifacts/handoff_validation/`。应看到`status: passed`、证据一致性通过和M04概率误差不超过`1e-5`。耗时与机器有关，以实际输出为准；这只是联调，不是新的预测效果实验。

## 5. B侧接着做什么

按`docs/collaboration.md`和自己的模块`plan.json`继续公开基线、评价器、机制生成器和扰动器。数据检查通过后可以开展不依赖正式M04/S的工作。正式指导/检索比较仍须等待A交付对应折和种子的共同产物。

新代码通过`study_inputs.SOURCE`或`training_data.SOURCE`读取输入。数据加载接口为`load_people(dataset, fold, include_evaluation=False)`；每个`Person`包含固定fold及本项目输入文件路径。验证后再显式进入获分配的正式评价流程，不把开发联调当作测试集选择依据。

不要为了测试安装而直接执行完整训练或整批基线脚本。运行前先登记唯一任务键，避免重复已有实验。旧目录中保留的是历史完成记录；不同代码/路径的运行不能覆盖它们或静默恢复。首个输入包没有完整基线模型集合；下述补充包交付已有历史基线，不需要全部重跑。

`run_guidance_transfer.py`是历史单折迁移入口，会产生新的逐人预测，并非本次验收必需命令。它还依赖本包没有交付的指导/融合检查点；首次验收使用上面的专用校验脚本。

## 历史基线补充与当前缺口

2026-11-20交接范围如下。B截图所述工作尚未取得代码与原始收据，属于B自报状态，不写作A已复核的完成结果。

| 材料 | 可交付状态 | 用途 |
| --- | --- | --- |
| 两库处理后输入、固定人员折、原始公开Chronos-2 | 首个输入包已具备 | 本机独立配置与联调 |
| HGB、RF、XGBoost、LightGBM、HYDRA | 150组模型/训练收据、预测、训练起点 | 历史基线独立复核 |
| 上述模型的报警回放 | 150组收据、issue_days、alerts、events | 固定验证阈值、未知结局日与静默期复核 |
| XGBoost恢复 | 30个UBJSON模型及恢复收据，原joblib保留 | 按原选中参数恢复；原预测误差小于1e-6 |
| 统计汇总 | 历史baseline/alarm汇总与分析、输出审查来源 | 私有复核；公开前由B按固定统计合同审阅 |
| 正式共同M04/S、合法起点/掩码、排除与来源包 | 两库×五折×三种子共30组均未交付 | 不得用旧pilot或工程检查抵扣 |
| 正式M10/M12/M13 | 未交付 | 等待对应共同M04/S；M13不在当前开发运行中 |
| 新教师与M04/M10/M12开发流程 | 仅Bris fold0 seed20261114，源侧开发运行 | 不是30组正式共同产物，不供正式外层拼表 |

补充包按原实验路径组织，完整保留历史manifest、源码快照和故障恢复材料。`artifacts/baseline_handoff/audit.json`给出逐组文件索引与独立输入hash映射；`formal_availability.csv`逐项列出30组缺口；`B_CODEX_PROMPT.txt`是可转交的任务说明。**补充包包含历史外层标签与预测**；只用作已有开发队列的复核，不据此调节新模型，也不能声称这些队列首次开盲。

### 配置与验收

使用Python 3.11或以上标准库即可验收，无需torch、GPU或API。`--study-root`指向已验收的第二篇输入根目录（其下应有`data/`、`configs/daily_protocol.json`），不得指向第一篇。

```powershell
# 从已发布的 exp/teacher-condition 分支取得该脚本；在 HNE-AI-2 中运行。
Get-FileHash '接收目录/HNE-AI-2-baseline-handoff.zip' -Algorithm SHA256
python scripts/baseline_handoff.py verify --archive '接收目录/HNE-AI-2-baseline-handoff.zip' --study-root $env:HNE_STUDY_ROOT
```

先与`BASELINE_SHA256SUMS.txt`比较压缩包SHA-256。验证器直接读ZIP，不解压、不加载pickle、不重训、不产生新外层评分；核对全部文件hash、150+150登记、模型/恢复来源、原人员折排除、跨方法/种子起点一致、已存预测一致性、记录的验证报警预算，以及186份输入的第二篇副本。通过会显示`status: passed`。这不是重新执行模型推理或独立重算全部统计量。

需使用文件时解压到本机授权项目下**尚不存在的接收目录**，得到`HNE-AI-2/`；不要覆盖B已有代码或实验文件。B评估适配器显式区分输入根目录与接收到的历史产物根目录。不建议直接重跑旧的全批训练/回放入口，它们保留了历史环境与恢复假设。

独立复核按以下文件关联：

1. `results/local_baselines/<dataset>/fold<k>/seed<s>/<method>/predictions.csv`含validation/evaluation的已知结局预测；`artifacts/local_baselines/.../training_origins.csv`是训练起点与人员排除依据。方法目录中的空格替换为下划线。
2. `results/local_alarms/.../issue_days.csv`保留全部合格发报日，包括未知结局日。不能删除未知日再回放；它们仍可发报并触发七日静默。`report.json`保存源验证选择的阈值与来源，`alerts.csv`/`events.csv`用于逐条核对。
3. 汇总定义见`results/local_experiments/analysis_manifest.json`：AUROC在折内按正负对数加权；AUPRC按阳性数；NLL/Brier/accuracy按起点数；先逐种子计算再取均值。人员bootstrap在原折内重采样，跨方法/种子配对，固定训练模型，不含重训不确定性。保留单类别折/人员不可计分情况。
4. XGBoost确需推理时，采用`artifacts/local_model_recovery/.../estimator.ubj`及配套settings、恢复收据。旧joblib仅作为历史证据保留；不要重复触发已记录的本机原生反序列化崩溃。其他模型的依赖与源码以历史来源快照为准。

现有150组基线使用全部合法训练起点、人员等权和六个固定候选，验证NLL选型。B自报CATCH/PatchTST各两轮、每人最多四窗口属于开发预算。可先复核完整性和评分器，但在统一并锁定训练/选型预算前，不把它们与历史全起点基线合为公平主表，不以外层成绩调参。

### B代码交回：403的可行处理

2026-11-20只读核实：`Angelinalyt`对`ChiZhang-805/HNE-AI`仅有read权限。尚未授予写权限。可以使用fork+PR；也可以先私有交回Git bundle，由A审查后推送脱敏运行登记。**在运行登记可远端核对且无重复认领之前，B保持真实外层评估未启动。**

以下命令在B的仓库根目录运行。先检查工作区并保存其自己的修改，不reset/clean，也不覆盖分支。截图提交是`902e675`，若已有后续提交，应如实交付新HEAD及差异，不回退旧提交。

```powershell
git status --short
git rev-parse exp/evaluation-baselines
git log --oneline 8b3a5e892d2e28ee4c2d2a8fd12e539e6e264286..exp/evaluation-baselines
git diff --name-status 8b3a5e892d2e28ee4c2d2a8fd12e539e6e264286 exp/evaluation-baselines
git merge-base --is-ancestor 8b3a5e892d2e28ee4c2d2a8fd12e539e6e264286 exp/evaluation-baselines
# 上一条必须成功；否则保留输出，交回真实分叉点，不自行改写历史。
New-Item -ItemType Directory -Force 'HNE-AI-2/packages/private-handoff'
git bundle create 'HNE-AI-2/packages/private-handoff/HNE-AI-2-evaluation-code.bundle' exp/evaluation-baselines '^8b3a5e892d2e28ee4c2d2a8fd12e539e6e264286'
git bundle verify 'HNE-AI-2/packages/private-handoff/HNE-AI-2-evaluation-code.bundle'
Get-FileHash 'HNE-AI-2/packages/private-handoff/HNE-AI-2-evaluation-code.bundle' -Algorithm SHA256
```

**创建bundle前，逐提交审阅上述提交范围和文件清单**，确保其中没有数据、权重、个人预测、校准、完整私有manifest或密钥；仅审查最终diff不足以排除历史中曾加入又删除的私有文件。若发现，停止公共推送，保留原分支，私下交接问题清单后准备只含安全代码的提交。bundle也只通过私有渠道传给A。若已存在同名bundle，先比较HEAD与SHA；同内容不重复生成，不覆盖尚未确认交付的旧文件。

B同时交回：完整提交hash、脱敏的60条运行键/状态与检查点hash登记、51项测试报告、合成流程检查报告、固定配置与验证阈值选择规则；60个检查点/阈值张量及完整来源另做私有包。说明实际已写预测/已读标签状态。A收到后先在隔离工作树审查、验证，再整合登记；不覆盖双方运行中的源文件。

## 私有文件与历史来源

`data/`、`tools/models/`、`artifacts/`、私有`results/`、`packages/`和`.venv/`均应保持Git忽略。提交时只暂存明确的代码文件，不上传数据、模型、预测、校准或交接ZIP。

原始manifest里的第一篇路径是历史来源记录，必须原样保留；验证器依据`/data/`后的相对路径和内容哈希核对第二篇副本，不会访问第一篇目录。原始数据复制并不改变既有开发暴露身份，不能声明为全新独立测试集。第一篇的文件和运行任务保持不变。
