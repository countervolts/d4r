[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$LlvmSourceRoot)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$expected='ff4dc1f7c9e1c64d4d69e40f4ed30c2280a96dfd'
$actual=(& git -C $LlvmSourceRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -or $actual -ne $expected) { throw "Expected audited ZLUDA LLVM source $expected, found $actual; review the patch against this revision." }
$patches=@(Get-ChildItem (Join-Path $repo 'patches/zluda-llvm') -Filter *.patch | Sort-Object Name)
foreach ($patch in $patches) {
    $oldPreference=$ErrorActionPreference
    try {
        $ErrorActionPreference='Continue'
        & git -C $LlvmSourceRoot apply --reverse --check $patch.FullName 2>$null
        $applied=($LASTEXITCODE -eq 0)
    } finally { $ErrorActionPreference=$oldPreference }
    if (!$applied) {
        & git -C $LlvmSourceRoot apply --check $patch.FullName
        if ($LASTEXITCODE) { throw "LLVM source patch conflicts: $($patch.Name)" }
        & git -C $LlvmSourceRoot apply $patch.FullName
        if ($LASTEXITCODE) { throw "LLVM source patch failed: $($patch.Name)" }
    }
    Write-Host "LLVM source patch present: $($patch.Name)"
}
# The pinned backport is the entire intended LLVM delta.
if ($patches.Count -ne 1) { throw 'Review combined LLVM source provenance before adding more patches.' }
$audit=Join-Path $repo ('build/llvm-source-audit-'+[Guid]::NewGuid().ToString('N')+'.patch')
New-Item -ItemType Directory -Force (Split-Path $audit) | Out-Null
try {
    & git -C $LlvmSourceRoot diff --binary --output=$audit
    if ($LASTEXITCODE) { throw 'Cannot audit LLVM source diff.' }
    $actualDiff=[IO.File]::ReadAllText($audit).Replace("`r`n","`n").TrimEnd()
    $expectedDiff=[IO.File]::ReadAllText($patches[0].FullName).Replace("`r`n","`n").TrimEnd()
    if ($actualDiff -cne $expectedDiff) { throw 'LLVM source contains changes outside the exported audited patch.' }
} finally { if (Test-Path -LiteralPath $audit) { Remove-Item -LiteralPath $audit } }
