[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$DlssDll,
    [string]$Kernel = 'hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel',
    [string]$HipRoot = 'C:\Program Files\AMD\ROCm\7.2',
    [string]$ZludaRoot,
    [string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$ZludaRoot) { $ZludaRoot=Join-Path $repo 'dist/zluda-windows-profile' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo 'build/private-textures-gfx1201' }
& python (Join-Path $repo 'tools/windows/build_native_texture.py') --dlss-dll $DlssDll --hip-root $HipRoot `
    --zluda-root $ZludaRoot --kernel $Kernel --output-directory $OutputDirectory
if ($LASTEXITCODE) { throw "Native texture build failed ($LASTEXITCODE); inspect the output work directory." }
