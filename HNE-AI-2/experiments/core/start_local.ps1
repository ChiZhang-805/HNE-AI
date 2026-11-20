param(
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][string]$RunKey,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$CodeCommit
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$artifactRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'artifacts/experiments/core'))
if ($RunKey -notmatch '^core/[A-Za-z0-9_.-]+/fold[0-9]+/seed[0-9]+/M04_M10_M12/[0-9a-f]{16}$') {
    throw 'Unexpected core run key'
}
$runDirectory = [IO.Path]::GetFullPath((Join-Path $projectRoot ('artifacts/experiments/' + $RunKey)))
if (-not $runDirectory.StartsWith($artifactRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Run directory escaped the project artifact root'
}
$configurationPath = Join-Path $projectRoot 'experiments/core/config.json'
$configuration = Get-Content -LiteralPath $configurationPath -Raw -Encoding UTF8 | ConvertFrom-Json
$digest = (Get-FileHash -LiteralPath $configurationPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($configuration.device -ne 'cpu' -or $configuration.threads -ne 1 -or $configuration.include_evaluation) {
    throw 'This local launcher only permits the registered single-thread source-only CPU study'
}
$expectedKey = 'core/{0}/fold{1}/seed{2}/M04_M10_M12/{3}' -f $configuration.dataset, $configuration.fold, $configuration.seed, $digest.Substring(0, 16)
if ($RunKey -ne $expectedKey) { throw 'Run key does not match the current configuration' }
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw 'Python executable not found' }
if (-not (Test-Path -LiteralPath (Join-Path $runDirectory 'manifest.json') -PathType Leaf)) { throw 'Claimed private run manifest is required' }
$statusPath = Join-Path $runDirectory 'status.json'
$previousStatus = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($previousStatus.status -eq 'completed') {
    @{ status = 'completed'; action = 'preserved_existing_run' } | ConvertTo-Json -Compress
    return
}
$lockPath = Join-Path $runDirectory 'launcher.lock'
$launchLock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    $inspectCode = @'
import json,sys,psutil
from pathlib import Path
root=Path(sys.argv[1]).resolve()
targets={str(root/'experiments/core/run.py').replace('\\','/').lower(),str(root/'experiments/core/continue_run.py').replace('\\','/').lower()}
matches=[]
for p in psutil.process_iter(['pid','name','cmdline']):
    try:
        if 'python' not in (p.info['name'] or '').lower(): continue
        args=[x.replace('\\','/').lower() for x in p.info['cmdline'] or []]
        match=any(x in targets for x in args)
        relative=any(x in ('experiments/core/run.py','experiments/core/continue_run.py') for x in args)
        if match or (relative and Path(p.cwd()).resolve()==root): matches.append(p.pid)
    except psutil.NoSuchProcess: pass
print(json.dumps(matches))
'@
    $inspection = & $PythonExecutable -c $inspectCode $projectRoot
    if ($LASTEXITCODE -ne 0) { throw 'Process inspection failed; refusing an unverified duplicate launch' }
    $existing = @($inspection | ConvertFrom-Json)
    if ($existing.Count -gt 0) { throw ('A core process is already running: ' + ($existing -join ', ')) }
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $logDirectory = Join-Path $runDirectory ('process_logs/' + $stamp)
    New-Item -ItemType Directory -Path $logDirectory | Out-Null
    $standardOutput = Join-Path $logDirectory 'stdout.log'
    $standardError = Join-Path $logDirectory 'stderr.log'
    $entryPoint = Join-Path $projectRoot 'experiments/core/run.py'
    $process = Start-Process -FilePath $PythonExecutable -ArgumentList @('-u', ('"' + $entryPoint + '"'), '--code-commit', $CodeCommit) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $standardOutput -RedirectStandardError $standardError -PassThru
    $process.PriorityClass = [Diagnostics.ProcessPriorityClass]::BelowNormal
    $receipt = [ordered]@{ event = 'launched'; utc = [DateTime]::UtcNow.ToString('o'); pid = $process.Id; run_key = $RunKey; code_commit = $CodeCommit; launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant(); prior_status = $previousStatus; stdout = $standardOutput; stderr = $standardError; priority = 'BelowNormal'; device = 'cpu'; threads = 1 }
    $json = $receipt | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText((Join-Path $runDirectory 'launch.json'), $json, [Text.UTF8Encoding]::new($false))
    Add-Content -LiteralPath (Join-Path $runDirectory 'process_events.jsonl') -Value ($receipt | ConvertTo-Json -Depth 8 -Compress) -Encoding UTF8
    Start-Sleep -Seconds 3
    $process.Refresh()
    if ($process.HasExited) { throw ('Training process exited during startup; inspect ' + $standardError) }
    $receipt | ConvertTo-Json -Depth 8 -Compress
}
finally { $launchLock.Dispose() }
