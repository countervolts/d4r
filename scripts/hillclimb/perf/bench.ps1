[CmdletBinding()]
param(
    [string]$Control = 'C:/d4r-hc/rt-ctl',     # portable d4r folder of the last kept build
    [string]$Candidate = 'C:/d4r-hc/rt-cand',  # the tree under test (filled by -Build)
    [string]$Cases = '.',                      # regex on matrix.json case ids
    [int]$Rounds = 0,                          # default: matrix.json's
    [string]$Out,
    [switch]$Build                             # build the shim and copy it into -Candidate first
)
# The hill climb's single entry point (docs/hillclimb/performance.md): optionally builds the current tree's shim
# into the candidate runtime folder, then runs bench.py's interleaved control/candidate matrix. The runtime folders
# are portable d4r folders whose engine\ is the shared model directory (C:\d4r-hc\models, HIP layers included).
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
if ($Build) {
    & bash (Join-Path $repo 'scripts/build_d4r_nvngx_shim.sh')
    if ($LASTEXITCODE) { throw 'shim build failed' }
    Copy-Item (Join-Path $repo 'build/d4r_nvngx.dll') (Join-Path $Candidate 'nvngx.dll') -Force
    Copy-Item (Join-Path $repo 'build/d4r_hip_net.dll') $Candidate -Force
}
$bench = @((Join-Path $PSScriptRoot 'bench.py'), '--control', $Control, '--candidate', $Candidate, '--cases', $Cases)
if ($Rounds) { $bench += @('--rounds', $Rounds) }
if ($Out) { $bench += @('--out', $Out) }
& python @bench
if ($LASTEXITCODE) { throw 'bench failed' }
