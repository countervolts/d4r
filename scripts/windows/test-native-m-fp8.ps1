[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$CaptureDirectory,
    [Parameter(Mandatory=$true)][string]$BaselineValidation,
    [string]$ModuleDirectory, [string]$HipRoot, [string]$OutputDirectory, [switch]$Benchmark)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$ModuleDirectory) { $ModuleDirectory=Join-Path $repo 'build/native-m-fp8-experimental' }
if (!$HipRoot) { $HipRoot=Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo 'test-results/m-native-fp8-replay' }
$previousPython=$env:PYTHONPATH
try {
    $env:PYTHONPATH="$(Join-Path $repo '.tools/python/vendor');$previousPython"
    $metadata=Get-Content -LiteralPath (Join-Path $ModuleDirectory 'build-info.json') -Raw | ConvertFrom-Json
    if ($metadata.hardwarePacking) {
        & python -u (Join-Path $repo 'tools/windows/fp8_pack_validate.py') `
            --probe (Join-Path $repo 'dist/windows-rdna4-command-list/bin/d4r_native_replay_probe.exe') `
            --module (Join-Path $ModuleDirectory 'packing-probe/fp8_pack_probe.hsaco') `
            --hip-root $HipRoot --output-dir (Join-Path $OutputDirectory 'packing-check')
        if ($LASTEXITCODE) { throw 'Hardware packing does not preserve every finite FP8 encoding; stopping before network replay.' }
    }
    $benchmarkArguments=@()
    if ($Benchmark) { $benchmarkArguments=@('--benchmark-control-dir',(Join-Path $repo 'build/native-m-gfx1201')) }
    & python -u (Join-Path $repo 'tools/windows/m_native_replay_compare.py') `
        --capture-dir $CaptureDirectory --module-dir $ModuleDirectory --hip-root $HipRoot `
        --probe (Join-Path $repo 'dist/windows-rdna4-command-list/bin/d4r_native_replay_probe.exe') `
        --baseline-validation $BaselineValidation --output-dir $OutputDirectory @benchmarkArguments
    if ($LASTEXITCODE) { throw "Experimental FP8 validation failed; baseline unchanged. Logs: $OutputDirectory" }
} finally { $env:PYTHONPATH=$previousPython }
