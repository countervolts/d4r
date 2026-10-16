[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$DlssDll,
    [Parameter(Mandatory=$true)][string]$NativeDirectory,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$HipRoot,
    [string]$EmitterRoot,
    [string]$GpuArch='gfx1101',
    [string]$Kernel='hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel'
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
[void](Get-D4RGpuTargetInfo $GpuArch)
$DlssDll=(Get-Item -LiteralPath $DlssDll).FullName
$NativeDirectory=(Get-Item -LiteralPath $NativeDirectory).FullName
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
$buildRoot=[IO.Path]::GetFullPath((Join-Path $repo 'build'))+'\'
if(!$OutputDirectory.StartsWith($buildRoot,[StringComparison]::OrdinalIgnoreCase)){
    throw 'Private generated kernels must stay under ignored build/.'
}
if(Test-Path -LiteralPath $OutputDirectory){throw 'Use a new private cache directory.'}
$expected='3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983'
if((Get-FileHash -LiteralPath $DlssDll).Hash.ToLowerInvariant() -ne $expected){throw 'Expected pinned DLSS SR 310.9.1.'}
if(!$HipRoot){$HipRoot=Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core'}
if(!$EmitterRoot){$EmitterRoot=Join-Path $repo 'dist/zluda-windows-emitter'}
$objects=@(Get-ChildItem -LiteralPath $NativeDirectory -Filter '*.hsaco' -File)
if($objects.Count -ne 16){throw 'NativeDirectory must contain the baseline 11 K and 5 M objects.'}
foreach($family in @('k','m')){
    foreach($source in Get-ChildItem -LiteralPath (Join-Path $repo "kernels/$family") -Filter '*.hip' -File){
        if(!(Test-Path -LiteralPath (Join-Path $NativeDirectory ($source.BaseName+'.hsaco')))){
            throw "Baseline native directory is incomplete: $($source.BaseName)"
        }
    }
}
foreach($object in $objects){Assert-D4RCodeObjectTarget $object.FullName $GpuArch}
& (Join-Path $PSScriptRoot 'build-native-texture.ps1') -DlssDll $DlssDll -Kernel $Kernel `
    -HipRoot $HipRoot -ZludaRoot $EmitterRoot -GpuArch $GpuArch -OutputDirectory (Join-Path $OutputDirectory 'texture')
if($LASTEXITCODE){throw 'Private texture build failed.'}
$native=Join-Path $OutputDirectory 'native'
New-Item -ItemType Directory -Path $native | Out-Null
foreach($object in $objects){Copy-Item -LiteralPath $object.FullName -Destination $native}
foreach($marker in @('d4r-native-fp8.txt')){
    if(Test-Path -LiteralPath (Join-Path $NativeDirectory $marker)){
        Copy-Item -LiteralPath (Join-Path $NativeDirectory $marker) -Destination $native
    }
}
Copy-Item -LiteralPath (Join-Path $OutputDirectory "texture/$Kernel.hsaco") -Destination $native
& python (Join-Path $repo 'kernels/tools/kernel_manifest.py') $native $DlssDll
if($LASTEXITCODE){throw 'Private native manifest generation failed.'}
@{schema=1; architecture=$GpuArch; privateNvidiaDerivedCode=$true; nativeDirectory=$native;
    dlssSha256=$expected; kernel=$Kernel; validation='built only; validate complete frames before game use';
    files=@(Get-ChildItem -LiteralPath $native -File | ForEach-Object {
        @{name=$_.Name; sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
    })} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'cache.json') -Encoding UTF8
Write-Host 'Private cache built; no game or existing configuration changed.'
Write-Host "After validation, set [Paths] NativeDir = $native"
