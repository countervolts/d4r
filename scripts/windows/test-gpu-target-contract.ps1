[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$PackageRoot, [Parameter(Mandatory=$true)][string]$CodeObject)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$target=Get-D4RGpuTarget $PackageRoot $null
$other=if ($target -eq 'gfx1201') { 'gfx1200' } else { 'gfx1201' }
function Expect-Rejection([scriptblock]$Action) {
    $rejected=$false
    try { & $Action | Out-Null } catch { $rejected=$true }
    if (!$rejected) { throw 'GPU target contract accepted incompatible input.' }
}
if ((Get-D4RGpuTarget $PackageRoot $target) -ne $target) { throw 'Incorrect package target.' }
Assert-D4RCodeObjectTarget $CodeObject $target
Expect-Rejection { Get-D4RGpuTarget $PackageRoot $other }
Expect-Rejection { Get-D4RGpuTarget $PackageRoot 'gfx1250' }
Expect-Rejection { Assert-D4RCodeObjectTarget $CodeObject $other }
Expect-Rejection { Assert-D4RCodeObjectTarget (Join-Path $PackageRoot 'gpu-target.json') $target }
# Old packages had no target metadata and were gfx1201 only.
$legacy=Join-Path $PackageRoot '__no-target-metadata__'
if (Test-Path -LiteralPath (Join-Path $legacy 'gpu-target.json')) { throw 'Unexpected legacy test fixture.' }
if ((Get-D4RGpuTarget $legacy $null) -ne 'gfx1201') { throw 'Legacy target changed.' }
Expect-Rejection { Get-D4RGpuTarget $legacy 'gfx1200' }
Write-Host "PASS GPU_TARGET_SCRIPT target=$target rejected=5 hardware_executed=0"
