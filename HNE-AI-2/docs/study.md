# Study design

The forecast is issued after a participant's recent seven-day block is known to be normal. Observations end on day (d). Day (d+1) is a gap. The target is positive when at least two eligible measurement families are abnormal during days (d+2) through (d+8). Family status uses sustained deviations from the individual's 56-day reference. Missing or indeterminate outcomes are excluded from origin-level classification. The rules and thresholds are defined in the [protocol](../configs/daily_protocol.json) and [label implementation](../src/protocol.py).

## From personal history to evidence

The causal predictor learns to forecast the next 16-hour segment from the same channel's history, the other state channels and observed behavioral context. Nested input views expose what each additional source contributes to prediction. The global predictor is cross-fitted by participant. The first 28 personal reference days establish normalization, days 29–42 fit personal adjustments, and days 43–56 calibrate prediction errors. Event labels supervise the downstream task. [Predictor](../src/precursor_guidance.py) · [Calibration and training](../src/train_evidence.py) · [Model configuration](../configs/study.json)

During a 32-day monitoring history, recent deviations generate node and directed-edge evidence. The guidance module converts it into bounded attention biases in the final four Chronos-2 blocks. The unguided model retains the same backbone and event head. The fusion control uses the same evidence at the readout. The time-reversal ablation reverses valid evidence segments at inference while preserving masks and the trained weights. [Evidence alignment](../src/evidence_alignment.py) · [Evidence packing](../src/pack_guidance.py) · [Full-evidence fusion](../src/fusion/evidence.py)

## Completed experiments

| Experiment | Evaluation | Aggregate record |
| --- | --- | --- |
| Source training | 12 BrisT1D-Open participants, 91 selected training origins. Three validation participants, 185 origins and three positives. Checkpoint selection used this validation fold. | [Trainable parameters and scores](../results/reported/source_training.csv) |
| Frozen transfer and ablations | Bris-trained shared weights; prefix-only personal calibration on RESILIENT. Nine target participants, 263 validation origins and 30 positives. Comparators use the same origins. | [Paired contrasts and participant bootstrap intervals](../results/reported/frozen_transfer.json) · [Pair-weighted within-person AUROC](../results/reported/within_person.json) |
| External algorithms | Random Forest, XGBoost, LightGBM and the authors' multivariate HYDRA implementation were fitted on the same source training origins. Table 1 reports locally measured scores. Accuracy and sensitivity use a probability cutoff of 0.5 for the tree and guidance models, and the ridge decision boundary for HYDRA. | [Confusion counts and ranking metrics](../results/reported/public_baselines.json) |
| Alarm replay | Each method's threshold was chosen on BrisT1D-Open with a budget of one false alert per 30 scorable issue days. The threshold and seven-day silence were carried to RESILIENT. | [Alarm counts, unknown outcomes and event recall](../results/reported/alarm_replay.json) |

The datasets capture different signals. Public baselines use six common daily features constructed from personal deviations, family coverage and context. Trees summarize these features over recent windows. HYDRA receives their ordered daily trajectories. AUPRC in Table 1 is average precision. The [implementation](../scripts/run_public_baselines.py) records the tree settings, HYDRA scaling, participant-grouped ridge selection and decision rules.

Source validation was also used for model selection. The current measurements support comparisons within this development fold. The [experiment matrix](../configs/experiment_matrix.json) specifies the planned multi-fold and multi-seed evaluation.
