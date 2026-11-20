# SAGE-TS · Small-model guidance for early warning

## Experiment-based collaboration

Two researchers share one codebase organized by experiment. Start with the [assignment and handoff guide](docs/collaboration.md), [experiment directories](experiments/README.md), and [shared interface contract](docs/interfaces.md). A owns method/training; B owns baselines/evaluation. Run `python scripts/validate_collaboration.py` before claiming work.

The runtime now defaults to Paper 2 inputs and includes the complete-evidence interface corrections. Follow the [private handoff guide](docs/private_handoff.md) for data/model setup, file verification and CPU replay. Private inputs and weights are distributed separately. Full matched multi-fold experiments remain pending; missing helper entries are listed in each `plan.json`.


Can a small model learn an individual's usual sensor relationships and use deviations to improve a time-series foundation model's warning of a sustained abnormality? SAGE-TS uses these deviations to bias attention in the last four blocks of Chronos-2. This directory contains the study design, implementation and single-fold development results for the second HNE-AI project.

| Material | Entry |
| --- | --- |
| Study and evaluation design | [Research question, cohort and controls](docs/study.md) · [Fixed outcome protocol](configs/daily_protocol.json) · [Planned experiment matrix](configs/experiment_matrix.json) |
| Model and experiments | [Small predictor and guidance](src/precursor_guidance.py) · [Chronos attention interface](src/chronos_bridge.py) · [Source training](scripts/run_key_comparison.py) · [Frozen transfer](scripts/run_guidance_transfer.py) |
| Table 1 · Published algorithms and SAGE-TS | [PDF](results/tables/guidance_results.pdf) · [LaTeX](results/tables/guidance_results.tex) · [Metrics and confusion counts](results/reported/public_baselines.json) · [Baseline citations](literature/public_baselines.bib) |
| Table 2 · Inference ablations | [PDF](results/tables/guidance_ablations.pdf) · [LaTeX](results/tables/guidance_ablations.tex) · [Paired contrasts](results/reported/frozen_transfer.json) |
| Reproduction | [Environment, CPU checks, table export and training commands](docs/reproduction.md) |

On 185 BrisT1D-Open validation origins, SAGE-TS had AUROC 0.8187, the same as the unguided backbone. With shared weights frozen, its AUROC on 263 RESILIENT origins was 0.6009, versus 0.6010 without guidance. At the fixed 0.5 threshold, SAGE-TS detected 0 of 3 and 0 of 30 positive origins, respectively. The [alert replay](results/reported/alarm_replay.json) evaluates a separate operating point selected on the source cohort.

The source comparison used a training budget of 91 origins. The [training summary](results/reported/source_training.csv) and [timing breakdown](results/reported/source_timing.json) cover that experiment and the additional same-evidence fusion control. The [short pipeline check](results/reported/pipeline_check.json) is a separate engineering run.
