[CmdletBinding()]
param(
    [string]$GamePackagePrefix, [Parameter(Mandatory=$true)][string]$PackageRoot,
    [string]$ArchivePath
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$GamePackagePrefix) { $GamePackagePrefix=Join-Path $repo 'dist/windows-gpu-coverage-game' }
$PackageRoot=[IO.Path]::GetFullPath($PackageRoot)
if (Test-Path -LiteralPath $PackageRoot) { throw 'Use a new output directory to avoid mixing releases.' }
$commit=(& git -C $repo rev-parse HEAD).Trim()
if ($ArchivePath -and [bool](& git -C $repo status --porcelain)) { throw 'Commit source changes before creating a public ZIP.' }
New-Item -ItemType Directory -Force $PackageRoot | Out-Null
$files=[Collections.Generic.List[object]]::new()
function Stage([string]$Source,[string]$Relative) {
    $destination=Join-Path $PackageRoot $Relative
    New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $destination
    $item=Get-Item -LiteralPath $destination
    $entry=@{path=$Relative.Replace('\','/'); sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash; version=$item.VersionInfo.FileVersion}
    $files.Add($entry)
    return $entry
}
$targets=[ordered]@{}; $common=@(); $baseline=$null
foreach ($gpu in Get-D4RGpuTargets) {
    $arch=$gpu.architecture; $source="$GamePackagePrefix-$arch"
    $metadata=Get-Content -LiteralPath (Join-Path $source 'package.json') -Raw | ConvertFrom-Json
    if ($metadata.architecture -ne $arch -or $metadata.d4rWorkingTreeDirty) { throw "Uncommitted or wrong-target input: $source" }
    $targetFiles=@()
    foreach ($file in $metadata.files) {
        $relative=$file.path.Replace('\','/')
        if ((Get-FileHash -LiteralPath (Join-Path $source $file.path) -Algorithm SHA256).Hash -ne $file.sha256) { throw "Input file changed: $source/$relative" }
        if ($relative.EndsWith('.hsaco')) { Assert-D4RCodeObjectTarget (Join-Path $source $file.path) $arch }
        if ($relative -eq 'gpu-target.json' -or $relative -eq 'd4r/_nvngx.dll' -or $relative -eq 'd4r/d4r_gpu_inventory.exe' -or $relative -eq 'd4r/d4r_debug_launcher.exe' -or $relative -eq 'd4r/nvapi/nvapi64.dll' -or
            $relative.StartsWith('d4r/native/') -or $relative -like 'd4r/pixel_convert_*.hsaco') {
            $entry=Stage (Join-Path $source $file.path) "files/targets/$arch/$relative"
            $targetFiles+=@{path=$relative; sha256=$entry.sha256; version=$entry.version}
        } elseif ($relative -eq 'OptiScaler.dll' -or $relative.StartsWith('d4r/') -or $relative.StartsWith('licenses/')) {
            $publicPath=if ($relative.StartsWith('licenses/')) { $relative } else { "files/common/$relative" }
            if (!$baseline) {
                $entry=Stage (Join-Path $source $file.path) $publicPath
                if (!$relative.StartsWith('licenses/')) { $common+=@{path=$relative; sha256=$entry.sha256; version=$entry.version} }
            } elseif ((Get-FileHash -LiteralPath (Join-Path $PackageRoot $publicPath) -Algorithm SHA256).Hash -ne $file.sha256) {
                throw "Shared runtime differs in $arch : $relative"
            }
        }
    }
    if (@($targetFiles | Where-Object { $_.path.EndsWith('.hsaco') }).Count -ne 17) { throw "Expected 17 code objects for $arch" }
    $targets[$arch]=@{binarySourceCommit=$metadata.d4rCommit; hardwareValidation=$metadata.hardwareValidation; dlssSha256=$metadata.dlssSha256; files=$targetFiles}
    if (!$baseline) { $baseline=$metadata }
    if ($metadata.dlssSha256 -ne $baseline.dlssSha256 -or $metadata.zludaBuild.sourceCommit -ne $baseline.zludaBuild.sourceCommit) { throw 'Input packages do not use the same validated dependencies.' }
}
foreach ($name in @('windows-game.ps1','game-summary.ps1','gpu-target.ps1','gpu-preflight.ps1')) {
    $entry=Stage (Join-Path $PSScriptRoot $name) "files/common/$name"
    $common+=@{path=$name; sha256=$entry.sha256; version=$entry.version}
}
$entry=Stage (Join-Path $repo 'tools/windows/gpu-targets.json') 'files/common/gpu-targets.json'
$common+=@{path='gpu-targets.json'; sha256=$entry.sha256; version=$entry.version}
[void](Stage (Join-Path $PSScriptRoot 'quick-test.ps1') 'files/quick-test.ps1')
[void](Stage (Join-Path $repo 'docs/windows-quick-test.txt') 'READ-ME-FIRST.txt')
foreach ($launch in @(@{name='START-K.cmd'; arguments='-Preset 11'},@{name='START-M.cmd'; arguments='-Preset 13'},@{name='RESTORE-GAME.cmd'; arguments='-Action restore'})) {
    $path=Join-Path $PackageRoot $launch.name
    @('@echo off','cd /d "%~dp0"',('"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0files\quick-test.ps1" '+$launch.arguments),'pause') |
        Set-Content -LiteralPath $path -Encoding ASCII
    $files.Add(@{path=$launch.name; sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash; version=$null})
}
# Publicly supplied original identity, never the binary itself.
$metadata=[ordered]@{schema=1; packageCommit=$commit; binarySourceCommit=$baseline.d4rCommit;
    ngxCoreSha256='66767018C36B3BAB46398DADE3ADF173DAA3730FDA75965689EA848C9BC4E79B';
    dependencies=@{optiScalerSourceCommit=$baseline.optiScaler.sourceCommit; zludaSourceCommit=$baseline.zludaBuild.sourceCommit;
        llvmSource=$baseline.zludaBuild.llvmSource; hipRuntime=$baseline.hipRuntime};
    commonFiles=$common; targets=$targets; files=$files}
$metadata | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $PackageRoot 'files/package.json') -Encoding UTF8
if ($ArchivePath) {
    $ArchivePath=[IO.Path]::GetFullPath($ArchivePath)
    if ($ArchivePath.StartsWith($PackageRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Place the ZIP outside the output directory.' }
    New-Item -ItemType Directory -Force (Split-Path $ArchivePath) | Out-Null
    Add-Type -AssemblyName System.IO.Compression,System.IO.Compression.FileSystem
    $stream=[IO.File]::Open($ArchivePath,[IO.FileMode]::Create,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    $archive=$null
    try {
        $archive=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create)
        foreach ($relative in @($files.path)+@('files/package.json')) {
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,(Join-Path $PackageRoot $relative),$relative,[IO.Compression.CompressionLevel]::Optimal)
        }
    } finally { if ($archive) { $archive.Dispose() }; $stream.Dispose() }
}
Write-Host "Quick-test package: $PackageRoot (11 targets, automatic GPU selection, no NVIDIA DLLs)"
