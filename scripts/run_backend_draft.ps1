param(
    [string]$Arch = 'gfx1101',
    [string]$Python = '.tools/backend-venv/Scripts/python.exe',
    [string]$Cxx = '.tools/llvm-mingw-20261020-ucrt-x86_64/bin/clang++.exe',
    [string]$Dxc = '.tools/dxc/bin/x64/dxc.exe',
    [string]$Work = 'build/backend-draft/recheck',
    [int]$Trials = 5
)
# Portable Windows diagnostic build. All outputs stay local; no game install,
# configuration changes, commits or driver installation.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    foreach ($tool in @($Python, $Cxx, $Dxc)) {
        if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) {
            throw "Tool not found: $tool. See docs/backend-draft.md for setup."
        }
    }
    New-Item -ItemType Directory -Force $Work | Out-Null
    $runner = Join-Path $Work 'd4r-k-layer.exe'
    & $Cxx -std=c++17 -O2 -static -DNOMINMAX -DWIN32_LEAN_AND_MEAN `
        engine/d3d12/replay_layer.cpp -ld3d12 -ldxgi -ld3dcompiler -o $runner
    if ($LASTEXITCODE) { throw 'D3D12 draft build failed' }
    & $Python engine/d3d12/bench_backends.py $Work --runner $runner --dxc $Dxc --arch $Arch --trials $Trials
    if ($LASTEXITCODE) { throw 'Backend comparison failed; inspect comparison.json and trial logs' }
    Write-Output "Validated results: $Work/comparison.json"
} finally {
    Pop-Location
}
