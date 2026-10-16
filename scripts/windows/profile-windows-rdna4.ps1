[CmdletBinding()]
param(
    [ValidateSet(5,11,12,13)][int]$Preset = 11,
    [string]$ZludaRoot, [string]$PackageRoot, [string]$OutputDirectory,
    [string]$NativeKernelDirectory,
    [int]$Iterations = 4,
    [switch]$Translated, [switch]$NativeFp8
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$ZludaRoot) { $ZludaRoot=Join-Path $repo 'dist/zluda-windows-final' }
if (!$PackageRoot) { $PackageRoot=Join-Path $repo 'dist/windows-rdna4-command-list' }
$GpuArch=Get-D4RGpuTarget $PackageRoot $null
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/profile-' + $Preset + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$nativeDirectory=if ($NativeFp8) {
    if ($GpuArch -notmatch '^gfx12' -or $Preset -notin @(12,13)) { throw '-NativeFp8 requires gfx12 and preset L/M.' }
    if (!$NativeKernelDirectory) { $NativeKernelDirectory=Join-Path $repo 'build/native-m-fp8-token-lane-hwpack' }
    $info=Get-Content -LiteralPath (Join-Path $NativeKernelDirectory 'build-info.json') -Raw | ConvertFrom-Json
    if ($info.architecture -ne $GpuArch -or !$info.nativeFp8 -or !$info.tokenLanes) { throw 'NativeFp8 needs the matching gfx12 token-lane M kernel bundle.' }
    [IO.Path]::GetFullPath($NativeKernelDirectory)
} elseif ($Translated -or $Preset -eq 5) { $null }
elseif ($Preset -eq 11) { Join-Path $repo "build/native-k-$GpuArch" }
else { Join-Path $repo "build/native-m-$GpuArch" }
$settings=@{D4R_ZLUDA_PROFILE='1'; D4R_PROFILE_STAGES='1'; D4R_VALIDATE_OUTPUT='1';
    D4R_ZLUDA_PROFILE_DEFERRED=$null; D4R_ZLUDA_PROFILE_ALLOW_LEGACY=$null; D4R_ZLUDA_PROFILE_EVERY=$null;
    D4R_ZLUDA_NATIVE_DIR=$nativeDirectory}
$settings['D4R_ZLUDA_WMMA_FP8_NATIVE']=if ($NativeFp8) { '1' } else { $null }
$old=@{}
try {
    foreach ($key in $settings.Keys) { $old[$key]=[Environment]::GetEnvironmentVariable($key,'Process'); [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process') }
    $arguments=@{RuntimeProfile='therock'; ZludaRoot=$ZludaRoot; PackageRoot=$PackageRoot; OutputDirectory=$OutputDirectory;
        NgxCore=(Join-Path $repo '_nvngx.dll'); DlssDll=(Join-Path $repo 'nvngx_dlss.dll'); NgxMode='d3d12';
        NgxAbi='project-legacy'; Preset=$Preset; NgxOnly=$true; RequireNativeNetwork=(!$Translated -and $Preset -ne 5); NativeFp8=$NativeFp8; CommandListBackend=$true;
        CaptureExceptions=$true; EarlyIndirectProbe=$true; PixelProfile='depth-stencil'; BarrierMode='inherited-legacy';
        Iterations=$Iterations; TimeoutSeconds=300; OptiScalerDll=(Join-Path $repo 'dist/optiscaler-windows-d4r/OptiScaler.dll')}
    & (Join-Path $PSScriptRoot 'test-windows-rdna4.ps1') @arguments
    if ($LASTEXITCODE) { throw 'Profiling workload failed; see diagnostic bundle.' }
    $python=(Get-Command python.exe -ErrorAction Stop).Source
    & $python (Join-Path $repo 'tools/windows/profile_report.py') $OutputDirectory --output (Join-Path $OutputDirectory 'profile.json')
    if ($LASTEXITCODE) { throw 'No usable profile records.' }
} finally {
    foreach ($key in $old.Keys) { [Environment]::SetEnvironmentVariable($key,$old[$key],'Process') }
}
