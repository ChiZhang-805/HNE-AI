param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    foreach ($dataset in @('RESILIENT', 'BrisT1D-Open')) {
        & $Python -u scripts/run_hne_joint.py --datasets $dataset
        if ($LASTEXITCODE -ne 0) { throw "Training exited for $dataset" }
    }
    & $Python -u scripts/transfer_hne_joint.py
    if ($LASTEXITCODE -ne 0) { throw 'Transfer exited' }
    & $Python scripts/analyze_hne_joint.py
    if ($LASTEXITCODE -ne 0) { throw 'Analysis exited' }
    & $Python scripts/build_joint_figures.py
} finally { Pop-Location }
