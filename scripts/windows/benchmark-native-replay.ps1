[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Module,
    [Parameter(Mandatory=$true)][string]$Candidate,
    [Parameter(Mandatory=$true)][string]$FixtureDirectory,
    [string]$HipRoot, [string]$Probe, [string]$OutputDirectory,
    [ValidateRange(1,10000)][int]$Pairs=32,
    [ValidateRange(1,256)][int[]]$Batches=@(1,64),
    [ValidateSet('events','dispatch')][string[]]$Timings=@('events','dispatch'))
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$HipRoot) { $HipRoot=Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$Probe) { $Probe=Join-Path $repo 'build/windows-rdna4-therock/d4r_native_replay_probe.exe' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/native-benchmark-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
& python.exe (Join-Path $repo 'tools/windows/benchmark_native_replay.py') --probe $Probe --hip-root $HipRoot `
    --module $Module --candidate $Candidate --fixture-dir $FixtureDirectory --output-dir $OutputDirectory `
    --pairs $Pairs --batches @Batches --timings @Timings
if ($LASTEXITCODE) { throw "Native benchmark failed. Logs: $OutputDirectory" }
