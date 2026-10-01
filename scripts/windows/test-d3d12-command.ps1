param(
    [int]$Iterations = 32,
    [switch]$NoHooks,
    [switch]$SmallSeed,
    [switch]$DebugLayer,
    [switch]$Warp,
    [switch]$NoHip,
    [ValidateSet('basic','indirect','indirect-root-reset')][string]$Mode = 'indirect',
    [string]$OutputDirectory = 'test-results/d3d12-command-development'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
Push-Location $repo
$names = @('D4R_COMMAND_PROBE_NO_HOOKS','D4R_COMMAND_PROBE_SMALL_SEED','D4R_COMMAND_DEBUG_SDK','D4R_DIAG_DIR','D4R_COMMAND_PROBE_WARP','D4R_COMMAND_PROBE_NO_HIP')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
try {
    if ($Iterations -lt 1 -or $Iterations -gt 1024) { throw 'Iterations must be 1..1024' }
    $cmake = "$repo/.tools/python/cmake/data/bin/cmake.exe"
    $build = "$repo/build/windows-rdna4-therock"
    $hip = "$repo/.tools/therock-10.2.0a20260929/_rocm_sdk_core"
    $output = [IO.Path]::GetFullPath((Join-Path $repo $OutputDirectory))
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $target = 'd4r_d3d12_command_probe'
    if ($DebugLayer) {
        $sdk = "$repo/.tools/agility-1.619.5"
        if (!(Test-Path "$sdk/build/native/bin/x64/d3d12SDKLayers.dll")) { throw 'Unpack Microsoft.Direct3D.D3D12 1.619.5 into .tools/agility-1.619.5 for this optional debug probe' }
        & $cmake -S . -B $build "-DD4R_AGILITY_ROOT=$sdk" > "$output/configure.log"
        if ($LASTEXITCODE) { throw 'Debug probe configuration failed' }
        $target = 'd4r_d3d12_command_debug_probe'
    }
    & $cmake --build $build --target d4r_nvngx_windows $target > "$output/build.log"
    if ($LASTEXITCODE) { throw "Build failed: $output/build.log" }
    [Environment]::SetEnvironmentVariable($names[0], $(if ($NoHooks) { '1' } else { $null }), 'Process')
    [Environment]::SetEnvironmentVariable($names[1], $(if ($SmallSeed) { '1' } else { $null }), 'Process')
    [Environment]::SetEnvironmentVariable($names[2], $(if ($DebugLayer) { '1' } else { $null }), 'Process')
    [Environment]::SetEnvironmentVariable($names[3], $output, 'Process')
    [Environment]::SetEnvironmentVariable($names[4], $(if ($Warp) { '1' } else { $null }), 'Process')
    [Environment]::SetEnvironmentVariable($names[5], $(if ($NoHip -or $Warp) { '1' } else { $null }), 'Process')
    # PowerShell 5 treats native stderr as ErrorRecord: retain it without
    # converting it into a terminating exception, then check the process code.
    $ErrorActionPreference = 'Continue'
    & "$build/$target.exe" --hip-root $hip --module "$build/d4r_nvngx.dll" --interop-mode $Mode --iterations $Iterations > "$output/stdout.log" 2> "$output/stderr.log"
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Get-Content "$output/stdout.log", "$output/stderr.log" | Select-String 'D3D12_VALIDATION|DRED|DEVICE_REMOVED|PROBE_GPU|DIFFERENT|FAIL|PASS|installed'
    Write-Host "D3D12 command diagnostic logs: $output"
    if ($code) { throw "D3D12 probe exit=$code" }
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$saved[$name],'Process') }
    Pop-Location
}
