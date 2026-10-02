# Isolated installer checks; never starts a real game or launches GPU kernels.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PackageRoot,
    [Parameter(Mandatory=$true)][string]$NgxCore,
    [Parameter(Mandatory=$true)][string]$DlssDll,
    [string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$PackageRoot=(Get-Item -LiteralPath $PackageRoot).FullName
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/windows-quick-test/'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
function Invoke-Child([string]$Name,[string[]]$Arguments,[int]$ExpectedExit=0) {
    $log=Join-Path $OutputDirectory "$Name.log"
    & "$env:SystemRoot/System32/WindowsPowerShell/v1.0/powershell.exe" -NoProfile -STA -ExecutionPolicy Bypass `
        -File (Join-Path $PackageRoot 'files/quick-test.ps1') @Arguments *> $log
    if ($LASTEXITCODE -ne $ExpectedExit) { throw "Unexpected exit $LASTEXITCODE for $Name; see $log" }
}
function Fixture([string]$Name) {
    $dir=Join-Path $OutputDirectory $Name
    New-Item -ItemType Directory -Force $dir | Out-Null
    $exe=Join-Path $dir 'Fixture.exe'
    Set-Content -LiteralPath $exe 'installer fixture; do not execute' -Encoding ASCII
    return $exe
}
$results=@()
$exe=Fixture ('space GPU '+[char]0x03a9+' test')
$game=Split-Path $exe
$before=@{}
foreach ($name in @('dxgi.dll','OptiScaler.ini')) {
    Set-Content -LiteralPath (Join-Path $game $name) "original $name" -Encoding ASCII
    $before[$name]=(Get-FileHash -LiteralPath (Join-Path $game $name)).Hash
}
Invoke-Child 'install' @('-Action','install','-GameExe',$exe,'-NgxCore',$NgxCore,'-DlssDll',$DlssDll)
$saved=Get-Content -LiteralPath (Join-Path $PackageRoot 'work/settings.json') -Raw | ConvertFrom-Json
if ($saved.game -ne $exe -or $saved.architecture -ne 'gfx1201') { throw 'GPU detection or remembered selection failed on this RX9070XT fixture.' }
# Restore uses remembered game selection and needs neither HIP nor NVIDIA DLLs.
Invoke-Child 'restore' @('-Action','restore')
foreach ($name in $before.Keys) { if ((Get-FileHash -LiteralPath (Join-Path $game $name)).Hash -ne $before[$name]) { throw 'Original game bytes not restored.' } }
if (Test-Path -LiteralPath (Join-Path $game 'd4r')) { throw 'Runtime directory remains after restore.' }
$results+=@{case='auto-gfx1201-install-remember-restore'; passed=$true; gpuKernelsExecuted=$false}
foreach ($case in @('wrong-core','wrong-dlss','vkd3d')) {
    $exe=Fixture $case; $game=Split-Path $exe
    $core=if ($case -eq 'wrong-core') { Join-Path $PackageRoot 'files/targets/gfx1201/d4r/_nvngx.dll' } else { $NgxCore }
    $dlss=if ($case -eq 'wrong-dlss') { $NgxCore } else { $DlssDll }
    if ($case -eq 'vkd3d') { Set-Content -LiteralPath (Join-Path $game 'd3d12.dll') 'vkd3d-proton fixture; do not load' -Encoding ASCII }
    Invoke-Child $case @('-Action','install','-GameExe',$exe,'-NgxCore',$core,'-DlssDll',$dlss) 1
    if ((Test-Path -LiteralPath (Join-Path $game '.d4r-backup')) -or (Test-Path -LiteralPath (Join-Path $game 'dxgi.dll'))) { throw "$case changed game files before rejection." }
    $message=Get-Content -LiteralPath (Join-Path $OutputDirectory "$case.log") -Raw
    $match=[regex]::Match($message,'SEND THIS ONE FILE: (.+\.zip)')
    if (!$match.Success -or !(Test-Path -LiteralPath $match.Groups[1].Value.Trim())) { throw "$case did not produce its single report ZIP." }
    $results+=@{case=$case; passed=$true; rejectedBeforeGameChanges=$true; gpuKernelsExecuted=$false}
}
$readme=Join-Path $PackageRoot 'READ-ME-FIRST.txt'
$original=[IO.File]::ReadAllBytes($readme)
try {
    Add-Content -LiteralPath $readme 'tampered fixture'
    $exe=Fixture 'tampered-package'
    Invoke-Child 'tampered-package' @('-Action','install','-GameExe',$exe,'-NgxCore',$NgxCore,'-DlssDll',$DlssDll) 1
    $game=Split-Path $exe
    if ((Test-Path -LiteralPath (Join-Path $game '.d4r-backup')) -or (Test-Path -LiteralPath (Join-Path $game 'dxgi.dll'))) { throw 'Tampered package changed game files.' }
    $results+=@{case='tampered-package'; passed=$true; rejectedBeforeGameChanges=$true; gpuKernelsExecuted=$false}
} finally { [IO.File]::WriteAllBytes($readme,$original) }
$results | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
Write-Host "PASS QUICK_TEST_FIXTURES positive=1 negative=4 real_game_untouched=1 Report: $OutputDirectory"
