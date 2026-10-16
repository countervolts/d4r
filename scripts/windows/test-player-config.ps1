# Drag-in player mode checks. Builds with the vendored llvm-mingw; needs no
# NVIDIA DLLs and launches no GPU kernels. The loader case enumerates HIP
# devices and is skipped (reported, not passed) without a supported AMD GPU.
[CmdletBinding()]
param(
    [string]$CompilerRoot, [string]$HipRoot, [string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$CompilerRoot) { $CompilerRoot=Join-Path $repo '.tools/llvm-mingw-20260922-ucrt-x86_64' }
if (!$HipRoot) { $HipRoot=Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/player-config/'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$cxx=Join-Path $CompilerRoot 'bin/x86_64-w64-mingw32-clang++.exe'
if (!(Test-Path -LiteralPath $cxx)) { throw "llvm-mingw not found: $cxx" }
if (!(Test-Path -LiteralPath (Join-Path $HipRoot 'include/hip/hip_runtime_api.h'))) { throw "HIP headers not found under $HipRoot" }

# Same generated target table as CMakeLists.txt.
$targets=(Get-Content -LiteralPath (Join-Path $repo 'tools/windows/gpu-targets.json') -Raw | ConvertFrom-Json).targets
$generated=Join-Path $OutputDirectory 'generated'
New-Item -ItemType Directory -Force $generated | Out-Null
$entries=($targets | ForEach-Object { "  GpuTarget{`"$($_.architecture)`", $($_.elfMachine), $($_.isaMajor), `"$($_.family)`"}," }) -join "`n"
@('#pragma once','#include <array>','namespace d4r::diag {',
  'struct GpuTarget { const char* architecture; unsigned elf_machine; unsigned isa_major; const char* family; };',
  "inline constexpr std::array<GpuTarget, $($targets.Count)> gpu_targets = {{",$entries,'}};','}') |
    Set-Content -LiteralPath (Join-Path $generated 'gpu_targets.generated.h') -Encoding ASCII
$common=@('-std=c++17','-O1','-static','-DWIN32_LEAN_AND_MEAN','-DNOMINMAX','-D__HIP_PLATFORM_AMD__','-DD4R_TARGET_ARCH="gfx1201"',
    '-I',$generated,'-I',(Join-Path $repo 'tools/windows'),'-I',(Join-Path $repo 'tools'),'-I',(Join-Path $HipRoot 'include'))
# Windows PowerShell turns native stderr into terminating errors under Stop;
# judge native tools by exit code and keep their output in the log.
function Invoke-Native([string]$Log,[string]$Exe,[string[]]$Arguments) {
    $ErrorActionPreference='Continue'
    & $Exe @Arguments *> $Log
    return $LASTEXITCODE
}
function Build([string]$Name,[string[]]$Arguments) {
    $log=Join-Path $OutputDirectory "build-$Name.log"
    # PowerShell 5 / Legacy native passing strips embedded quotes. PowerShell
    # 7's Standard/Windows passing preserves them and must not double-escape.
    if ($PSVersionTable.PSVersion.Major -le 5 -or $PSNativeCommandArgumentPassing -eq 'Legacy') {
        $Arguments=@($Arguments | ForEach-Object { $_.Replace('"','\"') })
    }
    if (Invoke-Native $log $cxx $Arguments) { throw "Build failed: $Name; see $log" }
}
$results=@()

# 1. d4r.ini validation and DLL pin decisions (the shim's own header).
$probe=Join-Path $OutputDirectory 'player_config_probe.exe'
Build 'player-config' ($common+@('-municode',(Join-Path $repo 'tools/windows/player_config_probe.cpp'),'-o',$probe,'-lversion','-ldbghelp','-lbcrypt'))
if (Invoke-Native (Join-Path $OutputDirectory 'player-config.log') $probe @((Join-Path $OutputDirectory 'pins-fixture'))) { throw "player_config_probe failed; see $(Join-Path $OutputDirectory 'player-config.log')" }
$results+=@{case='d4r-ini-and-dll-pins'; passed=$true}

# 2. Loader: GPU-free marker, LUID-based target selection, forwarding, and a
#    clear failure when the folder has no build for the detected GPU.
$folder=Join-Path $OutputDirectory 'game/d4r'
New-Item -ItemType Directory -Force (Join-Path $folder 'hip/bin') | Out-Null
Build 'loader' ($common+@('-shared',(Join-Path $repo 'tools/windows/d4r_nvngx_loader.cpp'),'-o',(Join-Path $folder '_nvngx.dll'),'-lversion','-ldbghelp'))
$harness=Join-Path $OutputDirectory 'loader_selection_probe.exe'
Build 'harness' @('-municode','-static',(Join-Path $repo 'tools/windows/loader_selection_probe.cpp'),'-o',$harness,'-ldxgi','-ldxguid')
$stub=Join-Path $OutputDirectory 'stub.cpp'
@('#define API extern "C" __declspec(dllexport)',
  'struct Req { unsigned FeatureSupported, MinHWArchitecture; char MinOSVersion[255]; };',
  'API unsigned NVSDK_NGX_D3D12_GetFeatureRequirements(void*, const void*, Req* r) { r->FeatureSupported = 0; r->MinHWArchitecture = 0x5101; return 1; }') |
    Set-Content -LiteralPath $stub -Encoding ASCII
$stubDll=Join-Path $OutputDirectory 'stub.dll'
Build 'stub' @('-shared','-static',$stub,'-o',$stubDll)
foreach ($name in @('amdhip64_7.dll','amd_comgr.dll','rocm_kpack.dll')) {
    $source=Join-Path $HipRoot "bin/$name"
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination (Join-Path $folder 'hip/bin') }
}
$log=Join-Path $OutputDirectory 'loader-without-builds.log'
[void](Invoke-Native $log $harness @((Join-Path $folder '_nvngx.dll')))
$text=Get-Content -LiteralPath $log -Raw
if ($text -notmatch 'MARKER version=1 hip_loaded=0') { throw "Discovery marker loaded HIP or failed; see $log" }
if ($text -match 'supported=0') { throw "An adapter was reported supported without any target build; see $log" }
$gpuLog=Join-Path $folder 'd4r_nvngx.log'
$detected=$null
if (Test-Path -LiteralPath $gpuLog) { $detected=[regex]::Match((Get-Content -LiteralPath $gpuLog -Raw),'no build for your GPU \((gfx[0-9a-f]+)\)').Groups[1].Value }
if (!$detected) {
    $results+=@{case='loader-selection'; passed=$null; skipped='no supported AMD HIP device detected'}
    Write-Host 'SKIP loader selection: no supported AMD HIP device detected on this machine.' -ForegroundColor Yellow
} else {
    if ((Get-Content -LiteralPath $gpuLog -Raw) -notmatch 'It supports: none \(the d4r folder is incomplete') { throw "Unclear unsupported-GPU message; see $gpuLog" }
    Copy-Item -LiteralPath $stubDll -Destination (Join-Path $folder "_nvngx_$detected.dll")
    $log=Join-Path $OutputDirectory 'loader-with-build.log'
    [void](Invoke-Native $log $harness @((Join-Path $folder '_nvngx.dll')))
    if ((Get-Content -LiteralPath $log -Raw) -notmatch 'vendor=0x1002 status=0x1 supported=0 marker=0x5101') { throw "AMD adapter was not forwarded to _nvngx_$detected.dll; see $log" }
    $results+=@{case='loader-selection'; passed=$true; architecture=$detected; gpuKernelsExecuted=$false}
}
$results | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
Write-Host "PASS PLAYER_CONFIG cases=$(@($results | Where-Object { $_.passed }).Count) skipped=$(@($results | Where-Object { $_.skipped }).Count) Report: $OutputDirectory"
