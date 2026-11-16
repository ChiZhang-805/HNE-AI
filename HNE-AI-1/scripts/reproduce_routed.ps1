param(
    [ValidateSet('data','train','baselines','policy','context','free','ranking','controls','paths','evidence','temporal','transfer','adapt','inference','analysis','all')]
    [string]$Stage='all',
    [string]$Python='python'
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $steps=@{
        data='scripts/prepare_hne_routed.py'
        train='scripts/run_hne_routed.py'
        baselines='scripts/run_hne_routed_baselines.py'
        policy='scripts/train_hne_path_policy.py'
        context='scripts/train_hne_context_paths.py'
        free='scripts/train_hne_free_paths.py'
        ranking='scripts/train_hne_path_ranking.py'
        controls='scripts/train_hne_route_controls.py'
        paths='scripts/evaluate_hne_paths.py'
        evidence='scripts/evaluate_hne_evidence.py'
        temporal='scripts/evaluate_hne_temporal.py'
        transfer='scripts/run_hne_routed_transfer.py'
        adapt='scripts/adapt_hne_context_router.py'
        inference='scripts/run_hne_raw_examples.py'
        analysis='scripts/aggregate_hne_baselines.py'
    }
    $selected=if($Stage -eq 'all'){@('data','train','baselines','policy','context','free','ranking','controls','paths','evidence','temporal','transfer','adapt','inference','analysis')}else{@($Stage)}
    foreach($name in $selected){
        if($name -eq 'train'){& $Python $steps[$name] '--variants' 'learned'}else{& $Python $steps[$name]}
        if($LASTEXITCODE -ne 0){throw "HNE stage failed: $name"}
        if($name -eq 'data'){
            & $Python 'scripts/prepare_chronos_native.py'
            if($LASTEXITCODE -ne 0){throw 'Native Chronos prefix failed'}
        }
        if($name -eq 'train'){
            & $Python 'scripts/run_hne_routed.py' '--variants' 'native'
            if($LASTEXITCODE -ne 0){throw 'Native Chronos baseline failed'}
        }
        if($name -eq 'analysis'){
            & $Python 'scripts/analyze_hne_routed.py'
            if($LASTEXITCODE -ne 0){throw 'Paired result analysis failed'}
            & $Python 'scripts/summarize_hne_mechanisms.py'
            if($LASTEXITCODE -ne 0){throw 'Mechanism summaries failed'}
            & $Python 'scripts/summarize_hne_training.py'
            if($LASTEXITCODE -ne 0){throw 'Training summaries failed'}
            & $Python 'scripts/build_routed_figures.py'
            if($LASTEXITCODE -ne 0){throw 'Numerical figures failed'}
            & $Python 'scripts/write_hne_routed_report.py'
            if($LASTEXITCODE -ne 0){throw 'Research report failed'}
        }
        if($name -eq 'paths'){
            & $Python 'scripts/evaluate_hne_context_paths.py'
            if($LASTEXITCODE -ne 0){throw 'Context path analysis failed'}
            & $Python 'scripts/evaluate_hne_free_paths.py'
            if($LASTEXITCODE -ne 0){throw 'Free path analysis failed'}
        }
        if($name -eq 'adapt'){
            & $Python 'scripts/adapt_hne_free_router.py'
            if($LASTEXITCODE -ne 0){throw 'Free router adaptation failed'}
        }
    }
} finally {Pop-Location}
