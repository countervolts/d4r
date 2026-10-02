[CmdletBinding()]
param(
    [int]$TargetProcessId,
    [string]$ProcessName = 'SHProto-Win64-Shipping',
    [ValidateRange(5,300)][int]$Seconds = 30,
    [ValidateRange(0,300)][int]$DelaySeconds = 0,
    [string]$PresentMonPath,
    [string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$PresentMonPath) {
    $PresentMonPath=Join-Path $repo '.tools/presentmon-2.6.0/PresentMon.exe'
    if (!(Test-Path -LiteralPath $PresentMonPath)) {
        New-Item -ItemType Directory -Force (Split-Path $PresentMonPath) | Out-Null
        Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/GameTechDev/PresentMon/releases/download/v2.6.0/PresentMon-2.6.0-x64.exe' -OutFile $PresentMonPath
    }
}
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/presentmon-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$PresentMonPath=(Get-Item -LiteralPath $PresentMonPath -ErrorAction Stop).FullName
if ((Get-FileHash -LiteralPath $PresentMonPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
    'b2a706bc6ad475749e3b7e3409263aa1e6906d45bdcf993f6dbc0f660188f1af') {
    throw 'Expected the official PresentMon 2.6.0 x64 CLI. See docs/windows-performance.md.'
}
if ($TargetProcessId) { $target=Get-Process -Id $TargetProcessId -ErrorAction Stop }
else {
    $targets=@(Get-Process -Name $ProcessName -ErrorAction Stop)
    if ($targets.Count -ne 1) { throw 'Select one running game process using -TargetProcessId.' }
    $target=$targets[0]
}
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$csv=Join-Path $OutputDirectory 'presentmon.csv'
if (Test-Path -LiteralPath $csv) { throw 'Preserve the existing capture; choose another OutputDirectory.' }
@{version='2.6.0'; executable=$PresentMonPath; sha256=(Get-FileHash -LiteralPath $PresentMonPath -Algorithm SHA256).Hash;
    targetProcessId=$target.Id; processName=$target.ProcessName; seconds=$Seconds; delaySeconds=$DelaySeconds;
    hwsRegistry=$(Get-ItemPropertyValue -LiteralPath 'HKLM:/SYSTEM/CurrentControlSet/Control/GraphicsDrivers' -Name HwSchMode -ErrorAction SilentlyContinue);
    driver=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)} |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'presentmon-environment.json') -Encoding UTF8
# Unique ETW session: never stop or replace an existing user's trace. The CLI
# terminates itself at the bound/process exit, without a service installation.
$session='D4R_' + [Guid]::NewGuid().ToString('N')
& $PresentMonPath --process_id $target.Id --timed $Seconds --delay $DelaySeconds --terminate_after_timed --terminate_on_proc_exit --session_name $session --no_console_stats --no_track_input --qpc_time_ms --output_file $csv 2>&1 |
    Tee-Object -FilePath (Join-Path $OutputDirectory 'presentmon-console.log')
if ($LASTEXITCODE) { throw "PresentMon failed with exit code $LASTEXITCODE; see presentmon-console.log." }
if (!(Test-Path -LiteralPath $csv) -or (Get-Item -LiteralPath $csv).Length -eq 0) { throw 'PresentMon captured no frames.' }
Write-Host "PresentMon capture: $csv"
