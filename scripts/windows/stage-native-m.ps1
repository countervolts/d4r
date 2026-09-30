[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$DlssDll, [string]$PackageRoot, [string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$PackageRoot) { $PackageRoot = Join-Path $repo 'dist/windows-rdna4-therock' }
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo 'build/native-m-gfx1201' }
if (![IO.Path]::IsPathRooted($DlssDll) -or !(Test-Path -LiteralPath $DlssDll)) {
    throw 'Supply the absolute path to your local nvngx_dlss.dll.'
}
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
foreach ($layer in @('enc1','enc2','enc3_tube','dec2','dec1')) {
    $name = "rrlite_${layer}_4x4"
    $source = Join-Path $PackageRoot "experimental/m/${name}_gfx1201.hsaco"
    if (!(Test-Path -LiteralPath $source)) { throw "Native M module missing: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $OutputDirectory "$name.hsaco") -Force
}
& python (Join-Path $repo 'kernels/tools/kernel_manifest.py') $OutputDirectory $DlssDll
if ($LASTEXITCODE) { throw "Native manifest generation failed ($LASTEXITCODE)" }
@{architecture='gfx1201'; family='M'; numerics='SWIN_EXACT; SWIN_EXACT_PV; FP16 widening; native FP8 disabled';
    dlssSha256=(Get-FileHash -LiteralPath $DlssDll -Algorithm SHA256).Hash;
    validation='synthetic layers passed; actual DLSS capture/reference validation required';
    modules=@(Get-ChildItem -LiteralPath $OutputDirectory -Filter '*.hsaco' | Get-FileHash -Algorithm SHA256)} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'build-info.json') -Encoding UTF8
Write-Host "Experimental native M directory: $OutputDirectory"
