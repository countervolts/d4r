[CmdletBinding()]
param(
    [ValidateSet(11,13)][int]$Preset = 11,
    [string]$ZludaRoot, [string]$PackageRoot, [string]$OutputDirectory,
    [int]$Iterations = 4,
    [switch]$Translated
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$ZludaRoot) { $ZludaRoot=Join-Path $repo 'dist/zluda-windows-final' }
if (!$PackageRoot) { $PackageRoot=Join-Path $repo 'dist/windows-rdna4-command-list' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/profile-' + $Preset + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$settings=@{D4R_ZLUDA_PROFILE='1'; D4R_PROFILE_STAGES='1'; D4R_VALIDATE_OUTPUT='1';
    D4R_ZLUDA_PROFILE_DEFERRED=$null; D4R_ZLUDA_PROFILE_ALLOW_LEGACY=$null; D4R_ZLUDA_PROFILE_EVERY=$null;
    D4R_ZLUDA_NATIVE_DIR=$(if ($Translated) { $null } elseif ($Preset -eq 11) { Join-Path $repo 'build/native-k-gfx1201' } else { Join-Path $repo 'build/native-m-gfx1201' })}
$old=@{}
try {
    foreach ($key in $settings.Keys) { $old[$key]=[Environment]::GetEnvironmentVariable($key,'Process'); [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process') }
    $arguments=@{RuntimeProfile='therock'; ZludaRoot=$ZludaRoot; PackageRoot=$PackageRoot; OutputDirectory=$OutputDirectory;
        NgxCore=(Join-Path $repo '_nvngx.dll'); DlssDll=(Join-Path $repo 'nvngx_dlss.dll'); NgxMode='d3d12';
        NgxAbi='project-legacy'; Preset=$Preset; NgxOnly=$true; RequireNativeNetwork=(!$Translated); CommandListBackend=$true;
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
