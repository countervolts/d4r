[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$DlssDll,
    [string]$Kernel = 'hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel',
    [string]$HipRoot = 'C:\Program Files\AMD\ROCm\7.2',
    [string]$ZludaRoot,
    [ValidateSet('cu','wgp')][string]$ShaderMode='cu',
    [string]$GpuArch='gfx1201',
    [string]$OutputDirectory
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
[void](Get-D4RGpuTargetInfo $GpuArch)
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$ZludaRoot) { $ZludaRoot=Join-Path $repo 'dist/zluda-windows-final' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo "build/private-textures-$GpuArch" }
& python (Join-Path $repo 'tools/windows/build_native_texture.py') --dlss-dll $DlssDll --hip-root $HipRoot `
    --zluda-root $ZludaRoot --kernel $Kernel --output-directory $OutputDirectory --shader-mode $ShaderMode --gpu-arch $GpuArch
if ($LASTEXITCODE) { throw "Native texture build failed ($LASTEXITCODE); inspect the output work directory." }
