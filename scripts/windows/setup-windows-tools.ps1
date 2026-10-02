[CmdletBinding()]
param([ValidateSet('stable', 'therock')][string]$RuntimeProfile = 'stable')
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$localTools = Join-Path $repo '.tools'
$downloadDirectory = Join-Path $localTools 'downloads'
New-Item -ItemType Directory -Force $downloadDirectory | Out-Null
function Get-VerifiedArchive([string]$Url, [string]$Name, [string]$Hash, [string]$Destination, [string]$Marker) {
    if (Test-Path -LiteralPath $Marker) { Write-Host "Already present: $Marker"; return }
    $archive = Join-Path $downloadDirectory $Name
    if (!(Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Hash) {
        throw "SHA256 mismatch: $archive (remove this download and retry)"
    }
    if ([IO.Path]::GetExtension($archive) -eq '.whl') {
        python -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' $archive $Destination
        if ($LASTEXITCODE) { throw "Wheel extraction exit=$LASTEXITCODE" }
    } else { Expand-Archive -LiteralPath $archive -DestinationPath $Destination -Force }
    if (!(Test-Path -LiteralPath $Marker)) { throw "Archive layout changed: $Marker" }
}
Get-VerifiedArchive 'https://github.com/mstorsjo/llvm-mingw/releases/download/20260922/llvm-mingw-20260922-ucrt-x86_64.zip' `
    'llvm-mingw.zip' 'e3ad77d117a4bea19a7a3b333341824d79a5a371004a10e25b8504e7b3047666' $localTools `
    (Join-Path $localTools 'llvm-mingw-20260922-ucrt-x86_64/bin/x86_64-w64-mingw32-clang++.exe')
Get-VerifiedArchive 'https://github.com/vosen/ZLUDA/releases/download/v7-preview.11/zluda-windows-ee2f25a.zip' `
    'zluda.zip' '7788c1ed43385e62f0cc5f4d1aead24c92b671c02f1d685f63182540dea16d74' (Join-Path $localTools 'zluda') `
    (Join-Path $localTools 'zluda/zluda/nvcuda.dll')
if ($RuntimeProfile -eq 'therock') {
    Get-VerifiedArchive 'https://nightly.repo.amd.com/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-10.2.0a20260929-py3-none-win_amd64.whl' `
        'rocm_sdk_core-10.2.0a20260929-py3-none-win_amd64.whl' `
        '9623b97ca511eaa176905075a504393eca6fde594222af10b55e4fb1edfe5a25' `
        (Join-Path $localTools 'therock-10.2.0a20260929') `
        (Join-Path $localTools 'therock-10.2.0a20260929/_rocm_sdk_core/bin/amdhip64_7.dll')
}
if (!(Test-Path (Join-Path $localTools 'python/cmake/data/bin/cmake.exe')) -or
    !(Test-Path (Join-Path $localTools 'python/bin/ninja.exe'))) {
    python -m pip install --target (Join-Path $localTools 'python') cmake==3.31.6 ninja==1.11.1.4
    if ($LASTEXITCODE) { throw "pip install exit=$LASTEXITCODE" }
}
if (!(Test-Path (Join-Path $localTools 'python/vendor/numpy/__init__.py'))) {
    python -m pip install --disable-pip-version-check --only-binary=:all: `
        --target (Join-Path $localTools 'python/vendor') numpy==2.4.6
    if ($LASTEXITCODE) { throw "NumPy reference install exit=$LASTEXITCODE" }
}
Write-Host 'Tools ready. Run scripts/windows/build-windows-rdna4.ps1.'
