# 第二篇私有交接：配置、校验与运行

本说明配合`HNE-AI-2-private-handoff.zip`使用。代码在公共GitHub；数据、模型、个人证据和历史预测只通过私有渠道交接。无需第一篇工程，也无需API。

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

不要为了测试安装而直接执行完整训练或整批基线脚本。运行前先登记唯一任务键，避免重复已有实验。旧目录中保留的是历史完成记录；不同代码/路径的运行不能覆盖它们或静默恢复。私有包没有交付完整基线模型集合，不意味着已有基线工作应全部重跑。

`run_guidance_transfer.py`是历史单折迁移入口，会产生新的逐人预测，并非本次验收必需命令。它还依赖本包没有交付的指导/融合检查点；首次验收使用上面的专用校验脚本。

## 私有文件与历史来源

`data/`、`tools/models/`、`artifacts/`、私有`results/`、`packages/`和`.venv/`均应保持Git忽略。提交时只暂存明确的代码文件，不上传数据、模型、预测、校准或交接ZIP。

原始manifest里的第一篇路径是历史来源记录，必须原样保留；验证器依据`/data/`后的相对路径和内容哈希核对第二篇副本，不会访问第一篇目录。原始数据复制并不改变既有开发暴露身份，不能声明为全新独立测试集。第一篇的文件和运行任务保持不变。
