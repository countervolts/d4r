# Offline installer fixture. Uses locally supplied NVIDIA DLLs and never writes to a game folder.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$NgxCore,
    [Parameter(Mandatory=$true)][string]$DlssDll,
    [Parameter(Mandatory=$true)][string]$DlssdDll,
    [string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/windows-dlss-installer/'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $OutputDirectory) { throw "Use a new output directory: $OutputDirectory" }
$NgxCore=(Get-Item -LiteralPath $NgxCore -ErrorAction Stop).FullName
$DlssDll=(Get-Item -LiteralPath $DlssDll -ErrorAction Stop).FullName
$DlssdDll=(Get-Item -LiteralPath $DlssdDll -ErrorAction Stop).FullName
$expected=@{
    '_nvngx.dll'='66767018c36b3bab46398dade3adf173daa3730fda75965689ea848c9bc4e79b'
    'nvngx_dlss.dll'='3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983'
    'nvngx_dlssd.dll'='4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80'
}
$inputs=@{'_nvngx.dll'=$NgxCore;'nvngx_dlss.dll'=$DlssDll;'nvngx_dlssd.dll'=$DlssdDll}
foreach($name in $inputs.Keys) {
    if ((Get-FileHash -LiteralPath $inputs[$name] -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$name]) {
        throw "Wrong local NVIDIA DLL for installer fixture: $name"
    }
}
$package=Join-Path $OutputDirectory 'package'
$runtime=Join-Path $package 'd4r'
$cache=Join-Path $OutputDirectory 'cache'
$backupRoot=Join-Path $OutputDirectory 'backups'
$nvidiaCache=Join-Path $cache 'nvidia'
New-Item -ItemType Directory -Force (Join-Path $runtime 'ngx'),(Join-Path $runtime 'dlss'),(Join-Path $runtime 'rr'),$nvidiaCache | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'packaging/windows/d4r.ini') -Destination (Join-Path $runtime 'd4r.ini')
Copy-Item -LiteralPath (Join-Path $repo 'tools/windows/dll-pins.txt') -Destination (Join-Path $runtime 'dll-pins.txt')
$targets=@(
    @{source=$DlssDll; name='nvngx_dlss.dll'; relative='dlss/nvngx_dlss.dll'},
    @{source=$DlssdDll; name='nvngx_dlssd.dll'; relative='rr/nvngx_dlssd.dll'},
    @{source=$NgxCore; name='_nvngx.dll'; relative='ngx/_nvngx.dll'}
)
$originals=@{}
foreach($target in $targets) {
    $path=Join-Path $runtime $target.relative
    $original="fixture original $($target.name)"
    Set-Content -LiteralPath $path $original -Encoding ASCII
    $originals[$target.relative]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    Copy-Item -LiteralPath $target.source -Destination (Join-Path $nvidiaCache $target.name)
}

$installer=Join-Path $PSScriptRoot 'install-dlss.ps1'
$powershell=Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
function Invoke-Installer([string]$name) {
    $log=Join-Path $OutputDirectory "$name.log"
    & $powershell -NoProfile -ExecutionPolicy Bypass -File $installer -PackageRoot $package -CacheRoot $cache -BackupRoot $backupRoot -Yes *> $log
    if ($LASTEXITCODE) { throw "Installer fixture failed ($name); see $log" }
    return Get-Content -LiteralPath $log -Raw
}
$first=Invoke-Installer 'install'
foreach($target in $targets) {
    $path=Join-Path $runtime $target.relative
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$target.name]) {
        throw "Installed hash mismatch: $($target.relative)"
    }
}
$manifests=@(Get-ChildItem -LiteralPath $backupRoot -Recurse -Filter 'backup-manifest.json' -File)
if ($manifests.Count -ne 1) { throw "Expected one outside backup manifest, found $($manifests.Count)" }
$manifest=Get-Content -LiteralPath $manifests[0].FullName -Raw | ConvertFrom-Json
if (@($manifest.files).Count -ne 3) { throw 'Expected three backed-up originals.' }
foreach($entry in $manifest.files) {
    if (!$entry.backup -or !(Test-Path -LiteralPath $entry.backup) -or $entry.sha256 -ne $originals[$entry.path.Substring($runtime.Length+1).Replace('\','/')]) {
        throw "Backup path or original hash mismatch: $($entry.path)"
    }
    if ((Get-FileHash -LiteralPath $entry.backup -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Backup bytes changed: $($entry.backup)" }
}
$second=Invoke-Installer 'rerun'
if ($second -notmatch 'All three NVIDIA DLLs already match their pinned hashes\.' -or
    @(Get-ChildItem -LiteralPath $backupRoot -Recurse -Filter 'backup-manifest.json' -File).Count -ne 1) {
    throw 'Installer rerun was not idempotent.'
}
@{passed=$true;files=3;outsideBackups=3;idempotent=$true;archiveDownloaded=$false;driverRun=$false} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
Write-Host "PASS WINDOWS_DLSS_INSTALLER files=3 backups=3 idempotent=1 Report: $OutputDirectory"
