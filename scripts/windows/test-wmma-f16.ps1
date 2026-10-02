[CmdletBinding()]
param([string]$HipRoot, [string]$PackageRoot, [string]$OutputDirectory)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$HipRoot) { $HipRoot=Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$PackageRoot) { $PackageRoot=Join-Path $repo 'dist/windows-rdna4-command-list' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/wmma-f16-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$compiler=Join-Path $HipRoot 'lib/llvm/bin/clang++.exe'
if (!(Test-Path -LiteralPath $compiler)) { $compiler=Join-Path $HipRoot 'bin/clang++.exe' }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$module=Join-Path $OutputDirectory 'wmma_f16_compare.hsaco'
& $compiler -x hip -std=c++17 --offload-arch=gfx1201 --offload-device-only --no-gpu-bundle-output `
    -mno-wavefrontsize64 -nogpuinc -nogpulib -O3 (Join-Path $repo 'tools/windows/wmma_f16_compare.hip') -o $module
if ($LASTEXITCODE) { throw 'WMMA arithmetic probe compilation failed.' }
$previousPython=$env:PYTHONPATH
try {
    $env:PYTHONPATH="$(Join-Path $repo '.tools/python/vendor');$previousPython"
    & python.exe (Join-Path $repo 'tools/windows/wmma_f16_compare.py') `
        --probe (Join-Path $PackageRoot 'bin/d4r_native_replay_probe.exe') --module $module `
        --hip-root $HipRoot --output-dir $OutputDirectory
    if ($LASTEXITCODE) { throw "WMMA arithmetic diagnostic failed; logs: $OutputDirectory" }
} finally { $env:PYTHONPATH=$previousPython }
# A successful diagnostic is not acceptance of native FP16 accumulators for K.
# Read baselineEquivalent/gameOptimizationAccepted in validation.json.
