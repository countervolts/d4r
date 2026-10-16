[CmdletBinding()]
param([string]$SourceRoot)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$SourceRoot) { $SourceRoot = Join-Path $repo 'external/ZLUDA' }
$base = 'ee2f25a180099fa42f36b2346732e1f2470a03ad'
$relative = 'build/zluda-patch-check-' + [Guid]::NewGuid().ToString('N')
$snapshot = Join-Path $repo $relative
New-Item -ItemType Directory -Force $snapshot | Out-Null
$archive = "$snapshot.tar"
& git -C $SourceRoot archive --format=tar "--output=$archive" $base
if ($LASTEXITCODE) { throw 'Could not archive the pinned ZLUDA base.' }
& tar.exe -xf $archive -C $snapshot
if ($LASTEXITCODE) { throw 'Could not extract the ZLUDA patch-check snapshot.' }
Push-Location $repo
try {
    foreach ($patch in (Get-ChildItem -LiteralPath (Join-Path $repo 'patches/zluda') -Filter '*.patch' | Sort-Object Name)) {
        & git apply "--directory=$relative" --check $patch.FullName
        if ($LASTEXITCODE) { throw "Fresh-source patch check failed: $($patch.Name)" }
        & git apply "--directory=$relative" --whitespace=nowarn $patch.FullName
        if ($LASTEXITCODE) { throw "Fresh-source patch apply failed: $($patch.Name)" }
        Write-Host "PASS patch $($patch.Name)"
    }
    Write-Host "Pinned-source patch snapshot: $snapshot"
} finally { Pop-Location }
