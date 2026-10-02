[CmdletBinding()]
param([string]$SourceRoot)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$SourceRoot) { $SourceRoot = Join-Path $repo 'external/ZLUDA' }
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$base = 'ee2f25a180099fa42f36b2346732e1f2470a03ad'
if (!(Test-Path (Join-Path $SourceRoot 'Cargo.toml'))) { throw 'Clone upstream ZLUDA into external/ZLUDA first.' }
Push-Location $SourceRoot
try {
    $revision = (& git rev-parse HEAD).Trim()
    if ($revision -ne $base) { throw "Expected ZLUDA $base, found $revision; review patches against the new revision." }
    $dirty = @(& git status --porcelain --untracked-files=no --ignore-submodules=all)
    if ($dirty.Count) { throw 'ZLUDA source has tracked modifications. Keep this prepared checkout; do not reapply patches.' }
    & git switch -c windows-rdna4
    if ($LASTEXITCODE) { throw 'Could not create the local ZLUDA branch.' }
    foreach ($patch in (Get-ChildItem -LiteralPath (Join-Path $repo 'patches/zluda') -Filter '*.patch' | Sort-Object Name)) {
        & git apply --check $patch.FullName
        if ($LASTEXITCODE) { throw "Patch check failed: $($patch.Name)" }
        & git apply --whitespace=nowarn $patch.FullName
        if ($LASTEXITCODE) { throw "Patch failed: $($patch.Name)" }
        Write-Host "Applied $($patch.Name)"
    }
    Write-Host "Prepared Windows ZLUDA source at $SourceRoot (base $base)"
} finally { Pop-Location }
