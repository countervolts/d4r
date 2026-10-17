[CmdletBinding()]
param(
    [string[]]$GpuArchitectures,
    [ValidateSet('stable','therock')][string]$RuntimeProfile='therock',
    [string]$ZludaRoot, [string]$DlssDll,
    [switch]$PackageGames,
    [switch]$KernelsOnly,
    [string]$BuildRoot, [string]$OutputRoot, [string]$ArchiveDirectory
)
$ErrorActionPreference='Stop'
# Load the hash function in this scope before invoking child build scripts (PS 5.1).
Import-Module Microsoft.PowerShell.Utility
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$GpuArchitectures) { $GpuArchitectures=@(Get-D4RGpuTargets | ForEach-Object { $_.architecture }) }
foreach ($arch in $GpuArchitectures) { [void](Get-D4RGpuTargetInfo $arch) }
if (@($GpuArchitectures | Select-Object -Unique).Count -ne $GpuArchitectures.Count) { throw 'Duplicate GPU target.' }
if (!$BuildRoot) { $BuildRoot=Join-Path $repo $(if ($KernelsOnly) { 'build/windows-kernels' } else { 'build/windows-gpu-coverage' }) }
if (!$OutputRoot) { $OutputRoot=Join-Path $repo $(if ($KernelsOnly) { 'dist/windows-kernels' } else { 'dist/windows-gpu-coverage' }) }
if (!$ZludaRoot) { $ZludaRoot=Join-Path $repo 'dist/zluda-windows-final' }
if ($KernelsOnly -and $PackageGames) { throw 'KernelsOnly cannot package games; game packages also require host DLLs.' }
if ($ArchiveDirectory -and !$PackageGames) { throw 'ArchiveDirectory requires PackageGames.' }
if ($PackageGames -and (!$DlssDll -or ![IO.Path]::IsPathRooted($DlssDll) -or !(Test-Path -LiteralPath $DlssDll))) {
    throw 'PackageGames requires an absolute path to your local nvngx_dlss.dll; it will not be redistributed.'
}
New-Item -ItemType Directory -Force $BuildRoot | Out-Null
if ($ArchiveDirectory) { New-Item -ItemType Directory -Force $ArchiveDirectory | Out-Null }
$results=[Collections.Generic.List[object]]::new()
foreach ($arch in $GpuArchitectures) {
    $build="$BuildRoot-$arch"; $diag="$OutputRoot-$arch"
    $log=Join-Path $BuildRoot "$arch-build.log"
    Write-Host "Building $arch (compile/package only; no GPU workloads). Log: $log"
    & (Join-Path $PSScriptRoot 'build-windows-rdna4.ps1') -RuntimeProfile $RuntimeProfile -GpuArch $arch `
        -ZludaRoot $ZludaRoot -BuildDirectory $build -InstallDirectory $diag -KernelsOnly:$KernelsOnly *> $log
    $objects=@(Get-ChildItem -LiteralPath $diag -Recurse -Filter '*.hsaco')
    if ($objects.Count -ne 19) { throw "Expected 19 code objects for $arch, found $($objects.Count). See $log" }
    foreach ($object in $objects) { Assert-D4RCodeObjectTarget $object.FullName $arch }
    $row=[ordered]@{architecture=$arch; compiled=$true; kernelsOnly=[bool]$KernelsOnly; codeObjectCount=$objects.Count;
        hardwareTestExecuted=$false; buildDirectory=$build; diagnosticRoot=$diag; log=$log;
        objects=@($objects | ForEach-Object { @{path=$_.FullName; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} })}
    if ($PackageGames) {
        $k="$BuildRoot-native-k-$arch"; $m="$BuildRoot-native-m-$arch"; $game="$OutputRoot-game-$arch"
        & (Join-Path $PSScriptRoot 'stage-native-k.ps1') -DlssDll $DlssDll -PackageRoot $diag -GpuArch $arch -OutputDirectory $k
        & (Join-Path $PSScriptRoot 'stage-native-m.ps1') -DlssDll $DlssDll -PackageRoot $diag -GpuArch $arch -OutputDirectory $m
        $packageArgs=@{DiagnosticRoot=$diag; GpuArch=$arch; PackageRoot=$game; ZludaRoot=$ZludaRoot; KernelKRoot=$k; KernelMRoot=$m}
        if ($ArchiveDirectory) { $packageArgs.ArchivePath=Join-Path $ArchiveDirectory "d4r-windows-$arch.zip" }
        & (Join-Path $PSScriptRoot 'package-windows-game.ps1') @packageArgs
        $row.gamePackage=$game
    }
    $results.Add($row)
    $results | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $BuildRoot 'matrix-build.json') -Encoding UTF8
    Write-Host "Compiled $arch; hardware validation remains separate."
}
