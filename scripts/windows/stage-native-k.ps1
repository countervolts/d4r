[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$DlssDll, [string]$PackageRoot, [string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$PackageRoot) { $PackageRoot = Join-Path $repo 'dist/windows-rdna4-therock' }
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo 'build/native-k-gfx1201' }
if (![IO.Path]::IsPathRooted($DlssDll) -or !(Test-Path -LiteralPath $DlssDll)) {
    throw 'Supply the absolute path to your local nvngx_dlss.dll.'
}
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$layers = @('enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0')
foreach ($layer in $layers) {
    $name = "dltss_pwin_${layer}_layer"
    $source = Join-Path $PackageRoot "experimental/k/${name}_gfx1201.hsaco"
    if (!(Test-Path -LiteralPath $source)) { throw "Native K module missing: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $OutputDirectory "$name.hsaco") -Force
}
& python (Join-Path $repo 'kernels/tools/kernel_manifest.py') $OutputDirectory $DlssDll
if ($LASTEXITCODE) { throw "Native manifest generation failed ($LASTEXITCODE)" }
@{architecture='gfx1201'; family='K'; dlssSha256=(Get-FileHash -LiteralPath $DlssDll -Algorithm SHA256).Hash;
    validation='synthetic layers passed; actual DLSS capture/reference validation required';
    modules=@(Get-ChildItem -LiteralPath $OutputDirectory -Filter '*.hsaco' | Get-FileHash -Algorithm SHA256)} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'build-info.json') -Encoding UTF8
Write-Host "Experimental native K directory: $OutputDirectory"
