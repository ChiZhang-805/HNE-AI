# Reproduction

If you received the private handoff, start with [its setup and CPU replay instructions](private_handoff.md). The runtime defaults to the HNE-AI-2 root and never requires the first-paper project. Private data/models remain outside Git. The commands further below describe historical experiments and may require additional artifacts not included in the handoff.

Run commands from `HNE-AI-2/` in a clone of this repository. The development runs used Python 3.13 and float32 CPU inference. The attention adapter calls Chronos-2 internal methods, so use the [measured dependency versions](../requirements.txt). PyTorch 2.8.0 was installed from the CUDA 12.8 wheel distribution on the measurement host. A compatible CPU installation can run the synthetic checks.

```sh
python -m pip install -r requirements.txt
python scripts/verify_design_core.py
```

The check constructs random tensors and a small Chronos model. It exercises prediction causality, timestep and participant isolation, observation masks, the zero-guidance identity and gradient flow. The detailed result is written under `artifacts/verification/`.

## Regenerate the tables from public aggregates

The [reported results](../results/reported/) contain aggregate values for both tables. Put the Tectonic executable on PATH and run the exporter. It produces standalone LaTeX and PDF files in [results/tables](../results/tables/). For direct compilation, run `tectonic guidance_results.tex` or `tectonic guidance_ablations.tex` in that directory.

```sh
python scripts/export_guidance_tables.py
```

The main table sorts external methods by BrisT1D-Open accuracy, using source AUROC to break ties. The ablation table computes changes relative to aligned internal guidance. The absolute probability change and NLL difference are expressed in units of 10⁻⁵.

## Prepare inputs for a new training run

Obtain the BrisT1D-Open processed device data and the RESILIENT archive from the dataset owners under their respective access conditions. Individual records, forecast outputs and fitted weights are kept with the researcher running the experiments. [BrisT1D-Open repository](https://data.bris.ac.uk/data/dataset/33z5jc8fa6tob21ptrugzqog08) · [RESILIENT data record](https://doi.org/10.5281/zenodo.16755408) · [RESILIENT data descriptor](https://www.nature.com/articles/s41597-025-05958-x)

The commands below use a POSIX shell. Set `HNE_STUDY_ROOT` to an absolute directory for the prepared data and pretrained weights. Set `BRIS_ARCHIVE` and `RESILIENT_ARCHIVE` to the corresponding downloaded ZIP or extracted dataset directory. The input layout has `data/development/<dataset>/*_hourly.npz`, `data/daily_study/<dataset>/*_daily.npz` and `tools/models/chronos-2/` underneath the study root. On PowerShell, environment variables use `$env:HNE_STUDY_ROOT`, `$env:BRIS_ARCHIVE` and `$env:RESILIENT_ARCHIVE`.

The pinned public preprocessing checkout below contains both cohort readers and the daily label builder used by the shared protocol. Set the three environment variables before running the commands. [Hourly reader](https://github.com/ChiZhang-805/HNE-AI/blob/cab41206546a0a039de100316e817cf81e836026/scripts/prepare_development_hourly.py) · [Daily builder](https://github.com/ChiZhang-805/HNE-AI/blob/cab41206546a0a039de100316e817cf81e836026/scripts/build_daily_study.py)

```sh
: "${HNE_STUDY_ROOT:?Set an absolute input directory}"
: "${BRIS_ARCHIVE:?Set the BrisT1D-Open dataset path}"
: "${RESILIENT_ARCHIVE:?Set the RESILIENT dataset path}"
mkdir -p "$HNE_STUDY_ROOT"
git -C .. worktree add --detach "$HNE_STUDY_ROOT/preprocessing" cab41206546a0a039de100316e817cf81e836026
prep="$HNE_STUDY_ROOT/preprocessing/scripts"
python "$prep/prepare_development_hourly.py" --dataset BrisT1D-Open --source "$BRIS_ARCHIVE" --output "$HNE_STUDY_ROOT/data/development"
python "$prep/prepare_development_hourly.py" --dataset RESILIENT --source "$RESILIENT_ARCHIVE" --output "$HNE_STUDY_ROOT/data/development"
python "$prep/build_daily_study.py" --root "$HNE_STUDY_ROOT" --config configs/daily_protocol.json --output "$HNE_STUDY_ROOT/data/daily_study" --datasets BrisT1D-Open RESILIENT
hf download amazon/chronos-2 config.json model.safetensors --revision 29ec3766d36d6f73f0696f85560a422f50e8498c --local-dir "$HNE_STUDY_ROOT/tools/models/chronos-2"
```

The pretrained weight SHA-256 in the source run was `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`. The fixed daily protocol SHA-256 was `3d58b3089a3e7d25a442ec8aa78638c509fc03fcf2af901e1cfbe2b5ab17607c`. [Pretrained Chronos-2](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c)

## Run the measured development protocol

```sh
python scripts/verify_training_pipeline.py
python scripts/run_key_comparison.py --profile cpu_pilot --dataset BrisT1D-Open --fold 0 --seed 20261114 --device cpu
python scripts/train_evidence_fusion.py --profile cpu_pilot --dataset BrisT1D-Open --fold 0 --seed 20261114
python scripts/run_guidance_transfer.py
python scripts/fetch_hydra.py
python scripts/run_public_baselines.py
python scripts/replay_guidance_alarms.py
python scripts/export_guidance_tables.py --from-run
```

The source fit produces the unguided, internal-guidance, output-correction and summary-fusion checkpoints. The second command after training fits the full-evidence fusion control used by the transfer runner. Shuffling the order of those steps would leave a required checkpoint unavailable. The transfer run writes paired origin scores for BrisT1D-Open and RESILIENT. The external-baseline runner checks the participant, origin and label keys against those predictions. The HYDRA fetcher verifies the original source and license at the recorded upstream commit. [HYDRA implementation](https://github.com/angus924/hydra/tree/144bb7aa3186654042f00f36b078370c620acb06)

The standalone engineering profile is `python scripts/run_key_comparison.py --profile pipeline_check --device cpu`. For a completed source pilot, run `python scripts/benchmark_key_comparison.py` followed by `python scripts/summarize_key_comparison.py` to produce the deployment timing and local source report. Generated participant-level predictions, local manifests and checkpoints live in Git-ignored run directories. Published summaries remain in `results/reported/`.
