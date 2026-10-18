# Builds the Windows drag-in release: extract it into the game folder and run
# install.bat for the pinned local NVIDIA dependencies. No game launcher is
# required. Inputs are the per-target game packages that
# package-windows-game.ps1 produces (the same inputs as the quick test).
[CmdletBinding()]
param(
    [string]$GamePackagePrefix,
    [Parameter(Mandatory=$true)][string]$LoaderDll,
    [Parameter(Mandatory=$true)][string]$PackageRoot,
    [string[]]$Architectures,   # default: every target in gpu-targets.json; a subset makes a smaller release
    [string]$ArchivePath,
    [string]$RrToolsArchive,
    [string]$FsrPayload,        # build_fsr_payload.py output: vkd3d-proton core + amdxc64 shim for the experimental FSR RR backend
    [switch]$LocalDevelopment
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$RrToolsArchive) { $RrToolsArchive=Join-Path $repo 'build/rr-portable-tools-v2.zip' }
if (!(Test-Path -LiteralPath $RrToolsArchive -PathType Leaf) -or
    (Get-FileHash -LiteralPath $RrToolsArchive -Algorithm SHA256).Hash -ne '297289a57a82142eb952479c99112b1c352a6753eb15bc6b2431bf686aed9b22') {
    throw 'Pass -RrToolsArchive with the verified clean portable-tools v2 ZIP. Never package a used tool directory.'
}
if (!$GamePackagePrefix) { $GamePackagePrefix=Join-Path $repo 'dist/windows-gpu-coverage-game' }
$PackageRoot=[IO.Path]::GetFullPath($PackageRoot)
if($LocalDevelopment){
    $buildRoot=[IO.Path]::GetFullPath((Join-Path $repo 'build'))+'\'
    foreach($candidate in @($PackageRoot,$ArchivePath) | Where-Object { $_ }){
        if(!([IO.Path]::GetFullPath($candidate)).StartsWith($buildRoot,[StringComparison]::OrdinalIgnoreCase)){
            throw 'Local development packages and ZIPs must stay under ignored build/.'
        }
    }
}
if (Test-Path -LiteralPath $PackageRoot) { throw 'Use a new output directory to avoid mixing releases.' }
$git=Get-Command git -ErrorAction SilentlyContinue
$workingTreeDirty=(!$git -or [bool](& git -C $repo status --porcelain))
$commit=if ($git) { (& git -C $repo rev-parse HEAD).Trim() } else { $null }
if ($ArchivePath -and !$LocalDevelopment -and (!$git -or [bool](& git -C $repo status --porcelain))) { throw 'A public ZIP needs git and committed sources.' }
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
$targets=[ordered]@{}; $baseline=$null; $commonStaged=$false; $inventoryStaged=$false
$selected=@(Get-D4RGpuTargets | Where-Object { !$Architectures -or $Architectures -contains $_.architecture })
if (!$selected.Count -or ($Architectures -and $selected.Count -ne @($Architectures | Select-Object -Unique).Count)) { throw "Unknown GPU target in -Architectures: $($Architectures -join ', ')" }
foreach ($gpu in $selected) {
    $arch=$gpu.architecture; $source="$GamePackagePrefix-$arch"
    $metadata=Get-Content -LiteralPath (Join-Path $source 'package.json') -Raw | ConvertFrom-Json
    if ($metadata.architecture -ne $arch -or ($metadata.d4rWorkingTreeDirty -and !$LocalDevelopment)) { throw "Uncommitted or wrong-target input: $source" }
    if (!$baseline) { $baseline=$metadata }
    if ($metadata.dlssSha256 -ne $baseline.dlssSha256 -or $metadata.zludaBuild.sourceCommit -ne $baseline.zludaBuild.sourceCommit) { throw 'Input packages do not use the same validated dependencies.' }
    $objects=0
    foreach ($file in $metadata.files) {
        $relative=$file.path.Replace('\','/'); $path=Join-Path $source $file.path
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $file.sha256) { throw "Input file changed: $source/$relative" }
        if ($relative.EndsWith('.hsaco')) { Assert-D4RCodeObjectTarget $path $arch; $objects++ }
        # Per-target files get target-specific names so one folder serves every GPU.
        # One GPU-query tool serves every target: it reports each device's real architecture.
        if ($relative -eq 'd4r/d4r_gpu_inventory.exe') { if (!$inventoryStaged) { Stage $path 'd4r/d4r_gpu_inventory.exe'; $inventoryStaged=$true } }
        elseif ($relative -eq 'd4r/_nvngx.dll') { Stage $path "d4r/_nvngx_$arch.dll" }
        elseif ($relative -eq 'd4r/nvapi/nvapi64.dll') { Assert-D4RNvapiTarget $path $arch; Stage $path "d4r/nvapi/$arch/nvapi64.dll" }
        # The copy beside nvcuda.dll is one GPU's build: it is never shared. Each target's own build ships in
        # d4r/nvapi/<arch>/ (above); the old shared copy made every GPU load the first target's NVAPI.
        elseif ($relative -eq 'd4r/zluda/nvapi64.dll') { }
        elseif ($relative -like 'd4r/pixel_convert_*.hsaco') { Stage $path $relative }
        elseif ($relative.StartsWith('d4r/native/')) { Stage $path ("d4r/native/$arch/" + $relative.Substring('d4r/native/'.Length)) }
        elseif ($relative -eq 'd4r/nvapi/backend/nvapi64.dll' -or $relative -eq 'OptiScaler.dll' -or $relative.StartsWith('licenses/') -or
                ($relative.StartsWith('d4r/hip/') -or $relative.StartsWith('d4r/zluda/'))) {
            $public=if ($relative -eq 'OptiScaler.dll') { 'dxgi.dll' } elseif ($relative.StartsWith('licenses/')) { "d4r/$relative" } else { $relative }
            if (!$commonStaged) { Stage $path $public }
            elseif ((Get-FileHash -LiteralPath (Join-Path $PackageRoot $public) -Algorithm SHA256).Hash -ne $file.sha256) { throw "Shared runtime differs in $arch : $relative" }
        }
        # Installer-only files (launcher, PowerShell installer, docs) are not part of a drag-in install.
    }
    if ($objects -ne 17) { throw "Expected 17 code objects for $arch" }
    if (!(Test-Path -LiteralPath (Join-Path $PackageRoot "d4r/nvapi/$arch/nvapi64.dll") -PathType Leaf)) { throw "No d4r/nvapi/nvapi64.dll in the $arch input; the release needs one NVAPI build per target." }
    $commonStaged=$true
    $targets[$arch]=@{binarySourceCommit=$metadata.d4rCommit; hardwareValidation=$metadata.hardwareValidation;
        nativeFp8M=[bool]$metadata.nativeFp8M; tokenLanesM=[bool]$metadata.tokenLanesM}
}
$loader=(Get-Item -LiteralPath $LoaderDll).FullName
if ([IO.Path]::GetFileName($loader) -ne 'd4r_nvngx_loader.dll') { throw 'Pass the d4r_nvngx_loader.dll built by CMake.' }
Stage $loader 'd4r/_nvngx.dll'
Stage (Join-Path $repo 'packaging/windows/OptiScaler.ini') 'OptiScaler.ini'
Stage (Join-Path $repo 'packaging/windows/d4r.ini') 'd4r/d4r.ini'
Stage (Join-Path $repo 'tools/windows/dll-pins.txt') 'd4r/dll-pins.txt'
Stage (Join-Path $repo 'tools/windows/gpu-targets.json') 'd4r/gpu-targets.json'
if (!$inventoryStaged) { throw 'No d4r/d4r_gpu_inventory.exe in the inputs; d4r-check.ps1 needs it to query the GPU.' }
if ($FsrPayload) {
    # Experimental FSR RR backend: d4r-built vkd3d-proton (LGPL, source offer in SOURCE.txt) and d4r's amdxc64 shim.
    # AMD's denoiser is never bundled; install.bat downloads the hash-pinned file when FSR is selected.
    $FsrPayload=[IO.Path]::GetFullPath($FsrPayload)
    $payloadManifest=Get-Content -LiteralPath (Join-Path $FsrPayload 'fsr-payload.json') -Raw | ConvertFrom-Json
    if ($payloadManifest.kind -ne 'd4r-fsr-payload' -or $payloadManifest.amdDenoiserBundled) { throw 'Unsupported FSR payload manifest.' }
    foreach ($property in $payloadManifest.files.PSObject.Properties) {
        $file=Join-Path $FsrPayload $property.Name
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $property.Value) { throw "FSR payload file changed: $($property.Name)" }  # -ne ignores case
    }
    foreach ($name in @('d4r_vkd3d_core.dll','amdxc64.dll')) { Stage (Join-Path $FsrPayload "fsr/$name") "d4r/fsr/$name" }
    Stage (Join-Path $FsrPayload 'SOURCE.txt') 'd4r/fsr/SOURCE.txt'
    foreach ($license in Get-ChildItem -LiteralPath (Join-Path $FsrPayload 'licenses') -File) { Stage $license.FullName "d4r/licenses/$($license.Name)" }
    foreach ($patch in Get-ChildItem -LiteralPath (Join-Path $repo 'patches/vkd3d-proton-windows') -Filter '*.patch' -File) { Stage $patch.FullName "d4r/fsr/patches/$($patch.Name)" }
}
foreach ($optional in @(@{source='packaging/windows/WINDOWS_README.txt'; path='WINDOWS_README.txt'},
                         @{source='packaging/windows/install.bat'; path='install.bat'},
                         @{source='scripts/windows/install-dlss.ps1'; path='install-dlss.ps1'},
                         @{source='tools/windows/install_dependencies.py'; path='install-dependencies.py'},
                         @{source='scripts/windows/d4r-check.ps1'; path='d4r/d4r-check.ps1'})) {
    $path=Join-Path $repo $optional.source
    if (Test-Path -LiteralPath $path) { Stage $path $optional.path }
}
Add-Type -AssemblyName System.IO.Compression,System.IO.Compression.FileSystem
$toolZip=[IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($RrToolsArchive))
try {
    $reader=[IO.StreamReader]::new($toolZip.GetEntry('rr-tools.json').Open())
    try { $toolManifest=$reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
    foreach ($entry in $toolZip.Entries) {
        if ($entry.FullName.EndsWith('/')) { continue }
        $name=$entry.FullName
        if ($name -match '(^/|\\|:|(^|/)\.\.(/|$))') { throw "Unsafe portable-tools entry: $name" }
        $relative='d4r/model-tools/'+$name
        $destination=Join-Path $PackageRoot $relative
        New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
        $inputStream=$entry.Open(); $outputStream=[IO.File]::Open($destination,[IO.FileMode]::CreateNew)
        try { $inputStream.CopyTo($outputStream) } finally { $inputStream.Dispose(); $outputStream.Dispose() }
        $digest=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        if ($name -ne 'rr-tools.json' -and $digest -ne $toolManifest.files.PSObject.Properties[$name].Value) {
            throw "Portable-tools hash mismatch: $name"
        }
        $files.Add(@{path=$relative; sha256=$digest; version=$null})
    }
} finally { $toolZip.Dispose() }
# Shows players where the NGX core goes; a ZIP cannot carry an empty folder.
$placeholder=Join-Path $PackageRoot 'd4r/ngx/PUT _nvngx.dll HERE.txt'
New-Item -ItemType Directory -Force (Split-Path $placeholder) | Out-Null
"Copy NVIDIA's _nvngx.dll into this folder. See WINDOWS_README.txt." | Set-Content -LiteralPath $placeholder -Encoding ASCII
$files.Add(@{path='d4r/ngx/PUT _nvngx.dll HERE.txt'; sha256=(Get-FileHash -LiteralPath $placeholder -Algorithm SHA256).Hash; version=$null})
if (Get-ChildItem -LiteralPath $PackageRoot -Recurse -File | Where-Object { $_.Name -in @('nvngx_dlss.dll','nvngx_dlssd.dll','amd_fidelityfx_denoiser_dx12.dll') -or ($_.Name -eq '_nvngx.dll' -and $_.DirectoryName -ne (Join-Path $PackageRoot 'd4r')) }) {
    throw 'Proprietary NVIDIA binaries must not be in the release.'
}
# Read by d4r-check.ps1: every shipped file's hash, so a damaged or mixed install is reported.
$manifest=[ordered]@{schema=1; kind='windows-drag-in'; localDevelopment=[bool]$LocalDevelopment;
    packageWorkingTreeDirty=[bool]$workingTreeDirty; packageCommit=$commit; binarySourceCommit=$baseline.d4rCommit;
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
Write-Host "Windows drag-in release: $PackageRoot ($($targets.Count) GPU target(s): $($targets.Keys -join ', '); no NVIDIA DLLs)"
