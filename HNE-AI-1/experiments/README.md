# Reproduce the warning experiments

Run commands from `HNE-AI-1/`. The published tables use participant-disjoint five-fold development evaluation. Fold k is held out, fold (k+1) modulo five selects settings, and the other three train the model. The HNE ensemble averages three seeds within fold. The table reports within-fold pair-weighted AUROC and sample-weighted fold AP.

## Render the two published tables

Install the analysis dependencies in [requirements.txt](../requirements.txt) and Tectonic on PATH, or set `TECTONIC` to its executable. This entry uses only the two released aggregate CSVs.

```shell
python scripts/report_evidence_recursion.py --layout-only
```

The output is [HNE_AI_1_tables.pdf](../results/evidence_recursion/HNE_AI_1_tables.pdf). The [row layout](../paper/tables/evidence_recursion_layout.tex) and [rendering code](../scripts/report_evidence_recursion.py) are editable.

## Prepare data and train HNE

Use the [authors’ archives](../data/README.md) and a CUDA environment with [training dependencies](../requirements-routed.txt). Replace the archive placeholders with paths to your downloads. The data adapter builds both datasets and downloads the pinned Chronos-2 weights. The next command caches the ten-block prefix. The following commands train both readouts, with and without the frozen base-risk anchor.

```shell
python scripts/prepare_public_main.py --resilient-archive PATH_TO_SLEEP_ZIP --brist1d-archive PATH_TO_BRISTOL_ZIP
python scripts/prepare_hne_latent.py
python scripts/run_evidence_recursion.py --anchored
python scripts/run_evidence_recursion.py
```

## Fit external comparators and score the same origins

```powershell
python scripts/fetch_author_code.py
python scripts/run_daily_dnb.py
python scripts/run_hne_routed_baselines.py --methods DNB
foreach ($seed in 20261110,20261111,20261112) {
    $dest = if ($seed -eq 20261110) { 'results/unified_daily' } else { "results/public_daily_repeats/seed$seed" }
    python scripts/run_public_daily.py --seed $seed --output $dest
}
python scripts/run_public_daily.py --ensemble-only --output results/public_daily_repeats --original-output results/unified_daily
python scripts/prepare_chronos_native.py
python scripts/run_hne_routed.py --variants native
```

For the three frozen time-series models, get the author implementations at the revisions documented in the [comparator notes](../literature/recent_public_warning_comparators.md). Install the libraries required by the author repositories, including `jaxtyping` for Moirai, and make the original [JAPE sources](https://anonymous.4open.science/r/JAPE-B74E/) available under `literature/reference_code/JAPE/` (the author's `models/`, `layers/` and `utils/` directories). The adapter downloads the pinned model weights on first use.

```shell
python scripts/run_recent_warning.py mantis
python scripts/run_recent_warning.py moirai2
python scripts/run_recent_warning.py timesfm
python scripts/run_jape_development.py
```

For the external College cohort, use the original [College data experiment](../scripts/reproduce.ps1) (college steps fit, freeze and test-data on prepared College inputs), [hourly preparation](../scripts/prepare_college_hourly.py) and then the frozen transfer:

```shell
python scripts/prepare_college_hourly.py --splits test
python scripts/prepare_hne_external_refined.py --datasets College-Behavior --encode
python scripts/transfer_evidence_recursion.py --anchored
python scripts/report_evidence_recursion.py
```

The full report takes the intersection of scoreable origins, pairs by participant, origin day, fold and event, and resamples participants 2,000 times. DNB's three-hour and tuned daily eligibility masks retain the original comparison population. All per-origin scores, fitted heads, checkpoints and representations are produced locally.
