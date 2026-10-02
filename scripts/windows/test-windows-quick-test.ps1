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
foreach ($name in @('dxgi.dll','OptiScaler.ini','nvngx_dlss.dll')) {
    Set-Content -LiteralPath (Join-Path $game $name) "original $name" -Encoding ASCII
    $before[$name]=(Get-FileHash -LiteralPath (Join-Path $game $name)).Hash
}
Invoke-Child 'install' @('-Action','install','-GameExe',$exe,'-NgxCore',$NgxCore,'-DlssDll',$DlssDll)
if ((Get-FileHash -LiteralPath (Join-Path $game 'd4r/nvngx_dlss.dll')).Hash -ne (Get-FileHash -LiteralPath $DlssDll).Hash) {
    throw 'Numeric NGX driver feature DLL was not staged beside the shim.'
}
if ((Get-FileHash -LiteralPath (Join-Path $game 'nvngx_dlss.dll')).Hash -ne $before['nvngx_dlss.dll']) { throw 'Original application DLSS DLL was changed.' }
$saved=Get-Content -LiteralPath (Join-Path $PackageRoot 'work/settings.json') -Raw | ConvertFrom-Json
if ($saved.game -ne $exe -or $saved.architecture -ne 'gfx1201') { throw 'GPU detection or remembered selection failed on this RX9070XT fixture.' }
# Restore uses remembered game selection and needs neither HIP nor NVIDIA DLLs.
Invoke-Child 'restore' @('-Action','restore')
foreach ($name in $before.Keys) { if ((Get-FileHash -LiteralPath (Join-Path $game $name)).Hash -ne $before[$name]) { throw 'Original game bytes not restored.' } }
if (Test-Path -LiteralPath (Join-Path $game 'd4r')) { throw 'Runtime directory remains after restore.' }
$results+=@{case='auto-gfx1201-install-remember-restore'; passed=$true; gpuKernelsExecuted=$false}
# A harmless child refuses attached debuggers and produces an empty stderr.
# The default launcher must allow it to exit and must report no DLSS session.
$exe=Join-Path (Join-Path $OutputDirectory 'launcher-fixture') 'Fixture.exe'
New-Item -ItemType Directory -Force (Split-Path $exe) | Out-Null
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class D4RLauncherFixture {
    [DllImport("kernel32.dll")] private static extern bool IsDebuggerPresent();
    public static int Main(string[] args) { return IsDebuggerPresent() ? 91 : 0; }
}
'@ -OutputAssembly $exe -OutputType ConsoleApplication
Invoke-Child 'run-no-debug' @('-Action','run','-GameExe',$exe,'-NgxCore',$NgxCore,'-DlssDll',$DlssDll) 1
$report=Get-ChildItem -LiteralPath (Join-Path $PackageRoot 'results') -Directory | Sort-Object Name -Descending | Select-Object -First 1
$status=Get-Content -LiteralPath (Join-Path $report.FullName 'quick-test.json') -Raw | ConvertFrom-Json
$environment=Get-Content -LiteralPath (Join-Path $report.FullName 'environment.json') -Raw | ConvertFrom-Json
if ($environment.attachedDebugger -or $status.hardwareValidationCompleted -or
    $status.status -ne 'inconclusive-process-exited-without-dlss-frames' -or $status.gameSummary.exitCode -ne '0x0' -or
    (Get-Item -LiteralPath (Join-Path $report.FullName 'd4r.stderr.log')).Length -ne 0) {
    throw 'Default launch attaches a debugger, mishandles empty stderr or falsely validates a launcher exit.'
}
Invoke-Child 'restore-launcher-fixture' @('-Action','restore','-GameExe',$exe)
$results+=@{case='run-no-debug-empty-stderr-no-false-pass'; passed=$true; gpuKernelsExecuted=$false}
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
Write-Host "PASS QUICK_TEST_FIXTURES cases=$($results.Count) real_game_untouched=1 Report: $OutputDirectory"
