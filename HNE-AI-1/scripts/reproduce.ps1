<#
.SYNOPSIS
Run one real stage of the HNE study from the project root.
.DESCRIPTION
Clean-clone order (uses the existing Python environment; never installs one):
  .\scripts\reproduce.ps1 -Stage data
  .\scripts\reproduce.ps1 -Stage features
  .\scripts\reproduce.ps1 -Stage development
  .\scripts\reproduce.ps1 -Stage external -ExternalStep fit
  .\scripts\reproduce.ps1 -Stage external -ExternalStep freeze
  .\scripts\reproduce.ps1 -Stage external -ExternalStep test-data
  .\scripts\reproduce.ps1 -Stage external -ExternalStep predict
  .\scripts\reproduce.ps1 -Stage analysis
  .\scripts\reproduce.ps1 -Stage paper

Development really trains models. For a complete independent retraining use a
new clone/clean project directory, with the same official raw archives. Completed
study directories should use external/predict, analysis, or paper. There is no
all-stages default. -Plan prints the commands without running or downloading.

Data downloads missing RES/Bris/CARE archives to -RawRoot (CARE is 5.5 GB), pins
the Human Rhythms repository tree, and uses the existing TOLIFE download adapter.
Only external/test-data constructs Human Rhythms test outcomes, after freezing.
TOLIFE reserved records are evaluated separately by the tolife stage using frozen source models.

NumPy/Pandas/SciPy/scikit-learn/joblib/threadpoolctl are needed for CPU stages;
public FCM/CATCH training additionally requires PyTorch CUDA and einops, and the
vendored third_party/FCM and third_party/CATCH trees. See requirements.txt for the
measured versions. Paper output additionally requires matplotlib and Tectonic.
Optional -Supplementary adds the set/temporal/pairwise/family head experiments.
.EXAMPLE
.\scripts\reproduce.ps1 -Stage external -ExternalStep predict
.EXAMPLE
.\scripts\reproduce.ps1 -Stage development -Plan
.EXAMPLE
.\scripts\reproduce.ps1 -Stage data -RawRoot ./data/raw
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('data', 'features', 'development', 'external', 'college', 'esa', 'tolife', 'trajectory', 'directed', 'dynamic-reference', 'shared-reference', 'ssaqs', 'pmdata', 'cost', 'analysis', 'paper')]
    [string]$Stage,
    [ValidateSet('fit', 'freeze', 'test-data', 'predict')]
    [string]$ExternalStep = 'predict',
    [ValidateSet('data', 'fit', 'freeze', 'test-data', 'predict', 'analysis')]
    [string]$CollegeStep = 'fit',
    [ValidateSet('data', 'fit', 'freeze', 'test-data', 'predict', 'analysis')]
    [string]$ESAStep = 'fit',
    [ValidateSet('download', 'fit', 'data', 'evaluate')]
    [string]$SSAQSPhase = 'evaluate',
    [ValidateSet(1, 2)]
    [int[]]$Missions = @(1, 2),
    [string]$Python = 'python',
    [string]$RawRoot = './data/raw',
    [string]$Tectonic = 'tectonic',
    [switch]$Supplementary,
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$humanRoot = Join-Path $projectRoot 'data/new_cohorts/human-rhythms'
$humanTree = '3fe83300ca9726a1ff6755d235347bc63a976022'

function Invoke-PythonScript {
    param([string]$Script, [string[]]$Arguments = @())
    $scriptPath = Join-Path $PSScriptRoot $Script
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) { throw "Missing script: $scriptPath" }
    $display = (@($Python, $scriptPath) + $Arguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ' '
    Write-Host "& $display"
    if (-not $Plan) {
        & $Python $scriptPath @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Script exited with code $LASTEXITCODE" }
    }
}

function Get-OfficialFile {
    param([string]$Url, [string]$Destination)
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        Write-Host "Using existing source: $Destination"
        return
    }
    Write-Host "Download $Url -> $Destination"
    if (-not $Plan) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
    }
}

function Invoke-IfMissing {
    param([string]$Artifact, [string]$Script, [string[]]$Arguments = @())
    if (-not $Plan -and (Test-Path -LiteralPath (Join-Path $projectRoot $Artifact))) {
        Write-Host "Using existing artifact: $Artifact"
    } else {
        Invoke-PythonScript $Script $Arguments
    }
}

Push-Location $projectRoot
try {
    if (-not $Plan) {
        if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { throw "Python not found: $Python; pass -Python with your existing environment." }
        if (($Stage -eq 'development' -or ($Stage -eq 'external' -and $ExternalStep -eq 'fit')) -and
            (Test-Path -LiteralPath 'results/human_rhythms/frozen_models.json')) {
            throw 'This study is frozen. Run predict/analysis/paper here; use a clean clone for independent retraining.'
        }
        if ($Stage -eq 'development' -and (Test-Path -LiteralPath 'results/unified_daily/RESILIENT/DNB/fold0/held_person.csv')) {
            throw 'Development training is complete in this directory. Use analysis to recompute its saved-score metrics, or retrain in a clean clone.'
        }
    }

    switch ($Stage) {
        'data' {
            $resArchive = Join-Path $RawRoot 'resilient-v3/Sleepmat_Watch_Data.zip'
            $brisArchive = Join-Path $RawRoot 'brist1d-open/33z5jc8fa6tob21ptrugzqog08.zip'
            $careArchive = Join-Path $RawRoot 'CARE-v6/CARE_To_Compare.zip'
            Get-OfficialFile 'https://zenodo.org/records/16755408/files/Sleepmat_Watch_Data.zip?download=1' $resArchive
            Get-OfficialFile 'https://data.bris.ac.uk/datasets/33z5jc8fa6tob21ptrugzqog08/33z5jc8fa6tob21ptrugzqog08.zip' $brisArchive
            Get-OfficialFile 'https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1' $careArchive
            Invoke-PythonScript 'prepare_development_hourly.py' @('--dataset', 'RESILIENT', '--source', $resArchive)
            Invoke-PythonScript 'prepare_development_hourly.py' @('--dataset', 'BrisT1D-Open', '--source', $brisArchive)
            Get-OfficialFile "https://api.github.com/repos/HAI-lab-UVA/Human-Rhythms-Dataset/git/trees/${humanTree}?recursive=1" (Join-Path $humanRoot 'repository_tree.json')
            Invoke-IfMissing 'data/new_cohorts/human-rhythms/source_manifest.json' 'prepare_human_rhythms.py'
            Invoke-PythonScript 'prepare_human_rhythms.py' @('--build-daily')
            Invoke-IfMissing 'data/new_cohorts/tolife-csa-2026/subject_split.csv' 'fetch_hne_cohort.py' @('tolife')
            Invoke-IfMissing 'data/new_cohorts/tolife-csa-2026/daily/schema_mapping.json' 'prepare_tolife_daily.py'
            Invoke-IfMissing 'data/care_transfer/manifest.json' 'prepare_care_transfer.py' @('--archive', $careArchive)
        }
        'features' {
            Invoke-IfMissing 'data/daily_study/summary.json' 'build_daily_study.py' @('--datasets', 'RESILIENT', 'BrisT1D-Open', 'TOLIFE-CSA')
            Invoke-PythonScript 'build_unified_features.py' @('--frequency', 'daily')
            Invoke-PythonScript 'build_human_rhythms_study.py' @('--splits', 'train', 'validation')
            Invoke-IfMissing 'data/care_study/features.npz' 'build_care_study.py' @('--workers', '2')
        }
        'development' {
            $models = @('HNE-additive', 'HNE-history', 'HNE-no-network', 'HNE-no-environment', 'HNE-context-append',
                'History-LR', 'History-HGB', 'HNE-HGB', 'Missingness-LR', 'HNE-no-amplitude', 'HNE-no-location',
                'HNE-no-memory', 'HNE-no-tail', 'HNE-scale8', 'HNE-scale16', 'HNE-scale32')
            if ($Supplementary) { $models += 'HNE-set' }
            Invoke-PythonScript 'run_development.py' (@('--features', 'data/features/unified_daily', '--output', 'results/unified_daily',
                '--datasets', 'RESILIENT', 'BrisT1D-Open', '--scales', '8', '16', '32', '--models') + $models)
            Invoke-PythonScript 'run_daily_dnb.py'
            Invoke-PythonScript 'run_daily_mmdev.py'
            Invoke-PythonScript 'run_public_daily.py' @('--seed', '20261110', '--output', 'results/unified_daily')
            foreach ($seed in @('20261111', '20261112')) {
                Invoke-PythonScript 'run_public_daily.py' @('--seed', $seed, '--output', "results/public_daily_repeats/seed$seed")
            }
            Invoke-PythonScript 'run_public_daily.py' @('--ensemble-only', '--output', 'results/public_daily_repeats', '--original-output', 'results/unified_daily')
            Invoke-PythonScript 'run_jape_development.py'
            Invoke-PythonScript 'run_jape_auc.py' @('--phase', 'development')
            if ($Supplementary) {
                Invoke-PythonScript 'run_temporal_development.py' @('--features', 'data/features/unified_daily', '--output', 'results/unified_daily')
                Invoke-PythonScript 'run_pairwise_development.py'
            }
            Invoke-PythonScript 'analyze_daily.py'
            if ($Supplementary) {
                # Family pooling requires the context-grid features and the
                # already-established common development population.
                Invoke-PythonScript 'run_context_grid.py' @('--phase', 'features', '--workers', '3')
                Invoke-PythonScript 'run_family_pooling.py' @('--phase', 'all')
                Invoke-PythonScript 'run_cardinality_development.py'
                Invoke-PythonScript 'run_family_graph_development.py'
                Invoke-PythonScript 'run_boundary_development.py'
                Invoke-PythonScript 'run_rhythm_coordinates.py' @('--phase', 'development', '--workers', '3')
            }
            Invoke-PythonScript 'run_transfer.py' @('--phase', 'fit')
            Invoke-PythonScript 'run_transfer.py' @('--phase', 'predict')
            Invoke-PythonScript 'run_care_development.py' @('--stage', 'all')
        }
        'external' {
            switch ($ExternalStep) {
                'fit' {
                    # Development summaries and all source-only models must exist.
                    Invoke-PythonScript 'run_context_grid.py' @('--phase', 'all', '--workers', '3')
                    Invoke-PythonScript 'run_hne_external.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_public_external.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_human_transfer.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_multidomain_hne.py'
                }
                'freeze' { Invoke-PythonScript 'freeze_external_study.py' }
                'test-data' {
                    if (-not $Plan -and -not (Test-Path -LiteralPath 'results/human_rhythms/frozen_models.json')) {
                        throw 'Run external/freeze before external/test-data.'
                    }
                    Invoke-PythonScript 'prepare_human_rhythms.py' @('--build-daily', '--include-test')
                    Invoke-PythonScript 'build_human_rhythms_study.py' @('--splits', 'test')
                }
                'predict' {
                    if (-not $Plan -and -not (Test-Path -LiteralPath 'results/human_rhythms/frozen_models.json')) {
                        throw 'Run external/freeze and external/test-data before external/predict.'
                    }
                    Invoke-PythonScript 'run_hne_external.py' @('--phase', 'predict')
                    Invoke-PythonScript 'run_public_external.py' @('--phase', 'predict', '--all-methods-frozen')
                    Invoke-PythonScript 'run_human_transfer.py' @('--phase', 'predict')
                    Invoke-PythonScript 'predict_external_variants.py' @('--workers', '3', '--wait-seconds', '0')
                }
            }
        }
        'college' {
            switch ($CollegeStep) {
                'data' {
                    Invoke-PythonScript 'fetch_college.py'
                    Invoke-IfMissing 'data/features/college_experience/validation.npz' 'prepare_college_study.py' @('--workers', '3')
                }
                'fit' {
                    Invoke-PythonScript 'run_college_models.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_college_transfer.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_temporal_external.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_future_objective.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_reference_units.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_reference_units.py' @('--phase', 'development')
                    Invoke-PythonScript 'run_rank_basis.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_jape_event.py' @('--phase', 'fit')
                    if ($Supplementary) { Invoke-PythonScript 'analyze_family_targets.py' }
                    Invoke-PythonScript 'run_jape_auc.py' @('--phase', 'fit')
                    Invoke-PythonScript 'run_family_graph_college.py' @('--phase', 'fit', '--workers', '3')
                    Invoke-PythonScript 'prepare_college_hourly.py' @('--splits', 'train', 'validation')
                    Invoke-PythonScript 'run_rhythm_coordinates.py' @('--phase', 'college-fit', '--workers', '3')
                    Invoke-PythonScript 'run_college_delayed.py' @('--phase', 'fit')
                }
                'freeze' { Invoke-PythonScript 'freeze_college_study.py' }
                'test-data' {
                    if (-not $Plan -and -not (Test-Path -LiteralPath 'results/college_experience/frozen_models.json')) {
                        throw 'Run college/freeze before college/test-data.'
                    }
                    Invoke-PythonScript 'prepare_college_study.py' @('--splits', 'test', '--all-models-frozen', '--workers', '3')
                }
                'predict' {
                    if (-not $Plan -and -not (Test-Path -LiteralPath 'results/college_experience/frozen_models.json')) {
                        throw 'Run college/freeze and college/test-data before college/predict.'
                    }
                    Invoke-PythonScript 'run_college_models.py' @('--phase', 'predict', '--all-methods-frozen')
                    Invoke-PythonScript 'run_college_transfer.py' @('--phase', 'predict', '--all-methods-frozen')
                    Invoke-PythonScript 'run_temporal_external.py' @('--phase', 'predict', '--all-methods-frozen')
                    Invoke-PythonScript 'run_future_objective.py' @('--phase', 'predict', '--all-methods-frozen')
                    Invoke-PythonScript 'run_reference_units.py' @('--phase', 'predict', '--all-methods-frozen')
                    Invoke-PythonScript 'run_rank_basis.py' @('--phase', 'predict')
                    Invoke-PythonScript 'run_jape_event.py' @('--phase', 'predict')
                    Invoke-PythonScript 'run_jape_auc.py' @('--phase', 'predict')
                    Invoke-PythonScript 'run_family_graph_college.py' @('--phase', 'predict', '--workers', '3')
                    Invoke-PythonScript 'prepare_college_hourly.py' @('--splits', 'test')
                    Invoke-PythonScript 'run_rhythm_coordinates.py' @('--phase', 'college-predict', '--workers', '3')
                    Invoke-PythonScript 'run_college_delayed.py' @('--phase', 'predict')
                }
                'analysis' {
                    Invoke-PythonScript 'analyze_college.py'
                    Invoke-PythonScript 'analyze_future_targets.py'
                    Invoke-PythonScript 'analyze_college_delayed.py'
                }
            }
        }
        'esa' {
            $missionArguments = @('--missions') + @($Missions | ForEach-Object { [string]$_ })
            switch ($ESAStep) {
                'data' {
                    Invoke-PythonScript 'fetch_esa.py' ($missionArguments + @('--workers', '16'))
                    Invoke-PythonScript 'prepare_esa.py' ($missionArguments + @('--phase', 'telemetry'))
                    Invoke-PythonScript 'build_esa_study.py' ($missionArguments + @('--phase', 'fit'))
                    Invoke-PythonScript 'build_esa_native.py' ($missionArguments + @('--phase', 'fit'))
                }
                'fit' {
                    Invoke-PythonScript 'run_esa_models.py' ($missionArguments + @('--phase', 'fit', '--workers', '3'))
                    Invoke-PythonScript 'run_esa_models.py' ($missionArguments + @('--phase', 'public-fit'))
                }
                'freeze' { Invoke-PythonScript 'run_esa_models.py' ($missionArguments + @('--phase', 'freeze')) }
                'test-data' { Invoke-PythonScript 'build_esa_native.py' ($missionArguments + @('--phase', 'test')) }
                'predict' { Invoke-PythonScript 'run_esa_models.py' ($missionArguments + @('--phase', 'predict', '--workers', '3')) }
                'analysis' { Invoke-PythonScript 'run_esa_models.py' ($missionArguments + @('--phase', 'analysis')) }
            }
        }
        'tolife' { Invoke-PythonScript 'run_tolife_external.py' }
        'trajectory' {
            Invoke-PythonScript 'run_joint_trajectory.py'
            Invoke-PythonScript 'analyze_joint_trajectory.py'
        }
        'directed' {
            Invoke-PythonScript 'run_directed_network.py'
            Invoke-PythonScript 'analyze_directed_network.py'
        }
        'dynamic-reference' {
            Invoke-PythonScript 'run_dynamic_reference.py'
            Invoke-PythonScript 'analyze_dynamic_reference.py'
        }
        'shared-reference' {
            Invoke-PythonScript 'run_shared_reference.py'
            Invoke-PythonScript 'analyze_dynamic_reference.py' @('--experiment', 'shared_reference')
        }
        'ssaqs' {
            switch ($SSAQSPhase) {
                'download' { Invoke-PythonScript 'fetch_ssaqs.py' }
                'fit' { Invoke-PythonScript 'run_ssaqs_external.py' @('--phase', 'fit') }
                'data' { Invoke-PythonScript 'prepare_ssaqs.py' }
                'evaluate' { Invoke-PythonScript 'run_ssaqs_external.py' @('--phase', 'evaluate') }
            }
        }
        'pmdata' { Invoke-PythonScript 'run_pmdata_external.py' }
        'cost' { Invoke-PythonScript 'measure_inference_cost.py' }
        'analysis' {
            Invoke-PythonScript 'analyze_daily.py'
            Invoke-PythonScript 'run_input_robustness.py'
            Invoke-PythonScript 'run_lead_time.py'
            Invoke-PythonScript 'analyze_external.py'
        }
        'paper' {
            Invoke-PythonScript 'build_paper_assets.py'
            Write-Host "In paper/: & '$Tectonic' main.tex"
            if (-not $Plan) {
                if (-not (Get-Command $Tectonic -ErrorAction SilentlyContinue)) { throw "Tectonic not found: $Tectonic; pass -Tectonic." }
                Push-Location (Join-Path $projectRoot 'paper')
                try {
                    & $Tectonic 'main.tex'
                    if ($LASTEXITCODE -ne 0) { throw "Tectonic exited with code $LASTEXITCODE" }
                } finally { Pop-Location }
            }
        }
    }
} finally { Pop-Location }
