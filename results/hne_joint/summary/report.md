# HNE 联合训练：实验结果

已完成 120 次完整拟合；每次 30 epochs。两库共用 Chronos-2、H/N/E 数值证据、三步递推、路由目标和同一配置。

## 主实验

主指标为参与者隔离五折中的正负样本对加权 AUROC，三个种子平均预测。所有方法在共同有效样本上比较。公开框架同时列出日级与三小时输入版本。

| 方法 | RESILIENT | BrisT1D-Open |
| --- | ---: | ---: |
| DNB (daily) | 0.4723 | 0.4874 |
| DNB-tuned daily | 0.4731 | 0.5263 |
| FCM (daily) | 0.7494 | 0.5974 |
| CATCH (daily) | 0.7651 | 0.6510 |
| FCM-score-LR daily | 0.7165 | 0.5650 |
| CATCH-score-LR daily | 0.7609 | 0.6368 |
| DNB (3h) | 0.5287 | 0.6327 |
| FCM (3h) | 0.5874 | 0.5284 |
| CATCH (3h) | 0.5212 | 0.5075 |
| DNB 3h calibrated | 0.4550 | 0.4393 |
| FCM 3h calibrated | 0.5397 | 0.4623 |
| CATCH 3h calibrated | 0.4614 | 0.5175 |
| Chronos-2 FT | 0.5580 | 0.6498 |
| Previous HNE | 0.8006 | 0.7083 |
| HNE | 0.7815 | 0.6539 |

预设次指标：平均精度 AP 与验证集 10% 误报预算下的告警表现。

| 数据集 | 方法 | AP | 灵敏度 | 实际误报率 |
| --- | --- | ---: | ---: | ---: |
| RESILIENT | FCM (daily) | 0.3575 | 0.3173 | 0.1119 |
| RESILIENT | CATCH (daily) | 0.3629 | 0.3269 | 0.1218 |
| RESILIENT | Previous HNE | 0.4348 | 0.4583 | 0.1248 |
| RESILIENT | HNE | 0.3944 | 0.4679 | 0.1593 |
| BrisT1D-Open | FCM (daily) | 0.1903 | 0.2384 | 0.1282 |
| BrisT1D-Open | CATCH (daily) | 0.2787 | 0.2583 | 0.1215 |
| BrisT1D-Open | Previous HNE | 0.3368 | 0.3311 | 0.1158 |
| BrisT1D-Open | HNE | 0.2657 | 0.3510 | 0.1493 |

## RQ1：模型能否学习多指标节律紊乱的前兆？

RESILIENT：HNE 0.7815；最高外部对照 CATCH daily 0.7651；差值 +0.0164，参与者配对 95% 区间 [-0.0623, +0.0909]。共同样本 1938 个，55 人。
去基模 0.7747；去数值证据 0.5716；原 HNE 0.8006。

预设次指标：同人 AUROC 为 0.5916，原 HNE 为 0.5373；使用同时具有正常与紊乱评价时点的 36 人。逐人配对区间见 within_person_intervals.csv。

BrisT1D-Open：HNE 0.6539；最高外部对照 CATCH daily 0.6510；差值 +0.0029，参与者配对 95% 区间 [-0.2079, +0.0770]。共同样本 1196 个，19 人。
去基模 0.6726；去数值证据 0.6542；原 HNE 0.7083。

预设次指标：同人 AUROC 为 0.6239，原 HNE 为 0.6232；使用同时具有正常与紊乱评价时点的 15 人。逐人配对区间见 within_person_intervals.csv。

## RQ2：学习推理路径能否提升紊乱预警？

| 方法 | RESILIENT | BrisT1D-Open |
| --- | ---: | ---: |
| Fixed H-N-E | 0.7802 | 0.6951 |
| Best constant path | 0.7818 | 0.6540 |
| Uniform paths | 0.7813 | 0.6563 |
| HNE | 0.7815 | 0.6539 |

RESILIENT：六路径平均概率跨度 0.0264；动态选择路径的真实标签损失平均排名 3.261/6。真实标签仅用于事后诊断，不进入推理。

BrisT1D-Open：六路径平均概率跨度 0.0236；动态选择路径的真实标签损失平均排名 3.491/6。真实标签仅用于事后诊断，不进入推理。

## RQ3：预警机制能否迁移到不同领域的紊乱？

源域输入变换与模型全部冻结，目标标签不参与训练，源域 15 个模型平均。两个领域均已参与此前方法开发；这里衡量冻结参数的跨域迁移。

| 源 → 目标 | 方法 | AUROC |
| --- | --- | ---: |
| RESILIENT → BrisT1D-Open | HNE | 0.7396 |
| RESILIENT → BrisT1D-Open | Best constant path | 0.7409 |
| RESILIENT → BrisT1D-Open | Uniform paths | 0.7410 |
| RESILIENT → BrisT1D-Open | Fixed H-N-E | 0.7400 |
| RESILIENT → BrisT1D-Open | No backbone | 0.7354 |
| RESILIENT → BrisT1D-Open | No numerical evidence | 0.6062 |
| BrisT1D-Open → RESILIENT | HNE | 0.7927 |
| BrisT1D-Open → RESILIENT | Best constant path | 0.7937 |
| BrisT1D-Open → RESILIENT | Uniform paths | 0.7942 |
| BrisT1D-Open → RESILIENT | Fixed H-N-E | 0.7953 |
| BrisT1D-Open → RESILIENT | No backbone | 0.8137 |
| BrisT1D-Open → RESILIENT | No numerical evidence | 0.5652 |

## 文件

模型：src/hne_joint.py；训练：scripts/run_hne_joint.py；原始观测推理：scripts/predict_hne_joint.py；迁移：scripts/transfer_hne_joint.py。
完整数值：metrics.csv；配对差值：paired_intervals.csv；分折/种子：fold_metrics.csv、seed_metrics.csv；路径：routing.csv。
图形均为黑白柱状图，PNG 与 PDF 同时输出。旧论文 PDF 的数值未自动替换为本轮结果。