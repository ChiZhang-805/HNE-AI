# Independent inputs for Paper 2

All new runs resolve inputs through `src/study_inputs.py`. The default is this HNE-AI-2 directory. `HNE_STUDY_ROOT` may explicitly name another absolute, authorized prepared-input directory; it never defaults to the first-paper directory.

The root contains `data/development/<dataset>/*_hourly.npz`, `data/daily_study/<dataset>/*_daily.npz`, the fixed-fold metadata, and `tools/models/chronos-2/{config.json,model.safetensors}`. No private files or trained checkpoints are distributed through this public repository.

For an existing private handoff, follow [setup, integrity verification and CPU replay](private_handoff.md). It includes matched historical M04/S pairs for development and engineering checks, not the full formal study. The first paper is not required to run these checks.

For a separately authorized prepared-input source, `python scripts/prepare_local_inputs.py --source-root ABSOLUTE_SOURCE` copies only observations, fixed daily records and original public pretrained weights into this project. It verifies every file and refuses different existing destinations. It does not move source files or import another paper's trained models. Source receipts stay private under `artifacts/input_handoff/`.

Historical run manifests retain the original paths and hashes. Never rewrite them to make a new run look like a continuation. The handoff verifier maps historical data suffixes to delivered files and verifies their content. Copied development data do not become a new independent test cohort.
