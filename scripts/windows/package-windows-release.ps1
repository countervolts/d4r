# Builds the Windows drag-in release: extract it into the game folder, copy
# the two NVIDIA DLLs, edit d4r\d4r.ini. Like the Linux release, no installer
# or launcher is involved. Inputs are the per-target game packages that
# package-windows-game.ps1 produces (the same inputs as the quick test).
[CmdletBinding()]
param(
    [string]$GamePackagePrefix,
    [Parameter(Mandatory=$true)][string]$LoaderDll,
    [Parameter(Mandatory=$true)][string]$PackageRoot,
    [string]$ArchivePath
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$GamePackagePrefix) { $GamePackagePrefix=Join-Path $repo 'dist/windows-gpu-coverage-game' }
$PackageRoot=[IO.Path]::GetFullPath($PackageRoot)
if (Test-Path -LiteralPath $PackageRoot) { throw 'Use a new output directory to avoid mixing releases.' }
$git=Get-Command git -ErrorAction SilentlyContinue
$commit=if ($git) { (& git -C $repo rev-parse HEAD).Trim() } else { $null }
if ($ArchivePath -and (!$git -or [bool](& git -C $repo status --porcelain))) { throw 'A public ZIP needs git and committed sources.' }
New-Item -ItemType Directory -Force $PackageRoot | Out-Null
$files=[Collections.Generic.List[object]]::new()
function Stage([string]$Source,[string]$Relative) {
    $destination=Join-Path $PackageRoot $Relative
    if (Test-Path -LiteralPath $destination) { throw "Duplicate release path: $Relative" }
    New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $destination
    $item=Get-Item -LiteralPath $destination
    $files.Add(@{path=$Relative.Replace('\','/'); sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash; version=$item.VersionInfo.FileVersion})
}
$targets=[ordered]@{}; $baseline=$null; $commonStaged=$false
foreach ($gpu in Get-D4RGpuTargets) {
    $arch=$gpu.architecture; $source="$GamePackagePrefix-$arch"
    $metadata=Get-Content -LiteralPath (Join-Path $source 'package.json') -Raw | ConvertFrom-Json
    if ($metadata.architecture -ne $arch -or $metadata.d4rWorkingTreeDirty) { throw "Uncommitted or wrong-target input: $source" }
    if (!$baseline) { $baseline=$metadata }
    if ($metadata.dlssSha256 -ne $baseline.dlssSha256 -or $metadata.zludaBuild.sourceCommit -ne $baseline.zludaBuild.sourceCommit) { throw 'Input packages do not use the same validated dependencies.' }
    $objects=0
    foreach ($file in $metadata.files) {
        $relative=$file.path.Replace('\','/'); $path=Join-Path $source $file.path
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $file.sha256) { throw "Input file changed: $source/$relative" }
        if ($relative.EndsWith('.hsaco')) { Assert-D4RCodeObjectTarget $path $arch; $objects++ }
        # Per-target files get target-specific names so one folder serves every GPU.
        if ($relative -eq 'd4r/_nvngx.dll') { Stage $path "d4r/_nvngx_$arch.dll" }
        elseif ($relative -eq 'd4r/nvapi/nvapi64.dll') { Stage $path "d4r/nvapi/$arch/nvapi64.dll" }
        elseif ($relative -like 'd4r/pixel_convert_*.hsaco') { Stage $path $relative }
        elseif ($relative.StartsWith('d4r/native/')) { Stage $path ("d4r/native/$arch/" + $relative.Substring('d4r/native/'.Length)) }
        elseif ($relative -eq 'OptiScaler.dll' -or $relative.StartsWith('licenses/') -or
                ($relative.StartsWith('d4r/hip/') -or $relative.StartsWith('d4r/zluda/'))) {
            $public=if ($relative -eq 'OptiScaler.dll') { 'dxgi.dll' } elseif ($relative.StartsWith('licenses/')) { "d4r/$relative" } else { $relative }
            if (!$commonStaged) { Stage $path $public }
            elseif ((Get-FileHash -LiteralPath (Join-Path $PackageRoot $public) -Algorithm SHA256).Hash -ne $file.sha256) { throw "Shared runtime differs in $arch : $relative" }
        }
        # Installer-only files (launcher, inventory, PowerShell, docs) are not part of a drag-in install.
    }
    if ($objects -ne 17) { throw "Expected 17 code objects for $arch" }
    $commonStaged=$true
    $targets[$arch]=@{binarySourceCommit=$metadata.d4rCommit; hardwareValidation=$metadata.hardwareValidation}
}
$loader=(Get-Item -LiteralPath $LoaderDll).FullName
if ([IO.Path]::GetFileName($loader) -ne 'd4r_nvngx_loader.dll') { throw 'Pass the d4r_nvngx_loader.dll built by CMake.' }
Stage $loader 'd4r/_nvngx.dll'
Stage (Join-Path $repo 'packaging/windows/OptiScaler.ini') 'OptiScaler.ini'
Stage (Join-Path $repo 'packaging/windows/d4r.ini') 'd4r/d4r.ini'
Stage (Join-Path $repo 'tools/windows/dll-pins.txt') 'd4r/dll-pins.txt'
foreach ($optional in @(@{source='packaging/windows/WINDOWS_README.txt'; path='WINDOWS_README.txt'},
                         @{source='scripts/windows/d4r-check.ps1'; path='d4r/d4r-check.ps1'})) {
    $path=Join-Path $repo $optional.source
    if (Test-Path -LiteralPath $path) { Stage $path $optional.path }
}
# Shows players where the NGX core goes; a ZIP cannot carry an empty folder.
$placeholder=Join-Path $PackageRoot 'd4r/ngx/PUT _nvngx.dll HERE.txt'
New-Item -ItemType Directory -Force (Split-Path $placeholder) | Out-Null
"Copy NVIDIA's _nvngx.dll into this folder. See WINDOWS_README.txt." | Set-Content -LiteralPath $placeholder -Encoding ASCII
$files.Add(@{path='d4r/ngx/PUT _nvngx.dll HERE.txt'; sha256=(Get-FileHash -LiteralPath $placeholder -Algorithm SHA256).Hash; version=$null})
if (Get-ChildItem -LiteralPath $PackageRoot -Recurse -File | Where-Object { $_.Name -eq 'nvngx_dlss.dll' -or ($_.Name -eq '_nvngx.dll' -and $_.DirectoryName -ne (Join-Path $PackageRoot 'd4r')) }) {
    throw 'Proprietary NVIDIA binaries must not be in the release.'
}
# Read by d4r-check.ps1: every shipped file's hash, so a damaged or mixed install is reported.
$manifest=[ordered]@{schema=1; kind='windows-drag-in'; packageCommit=$commit; binarySourceCommit=$baseline.d4rCommit;
    dependencies=@{optiScalerSourceCommit=$baseline.optiScaler.sourceCommit; zludaSourceCommit=$baseline.zludaBuild.sourceCommit; hipRuntime=$baseline.hipRuntime};
    targets=$targets; files=$files}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PackageRoot 'd4r/package.json') -Encoding UTF8
if ($ArchivePath) {
    $ArchivePath=[IO.Path]::GetFullPath($ArchivePath)
    if ($ArchivePath.StartsWith($PackageRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Place the ZIP outside the output directory.' }
    New-Item -ItemType Directory -Force (Split-Path $ArchivePath) | Out-Null
    Add-Type -AssemblyName System.IO.Compression,System.IO.Compression.FileSystem
    $stream=[IO.File]::Open($ArchivePath,[IO.FileMode]::Create,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    $archive=$null
    try {
        $archive=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create)
        foreach ($relative in @($files.path)+@('d4r/package.json')) {
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,(Join-Path $PackageRoot $relative),$relative,[IO.Compression.CompressionLevel]::Optimal)
        }
    } finally { if ($archive) { $archive.Dispose() }; $stream.Dispose() }
}
Write-Host "Windows drag-in release: $PackageRoot ($($targets.Count) GPU targets, no NVIDIA DLLs)"
