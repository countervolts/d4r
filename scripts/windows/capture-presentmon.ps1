[CmdletBinding()]
param(
    [int]$TargetProcessId,
    [string]$ProcessName = 'SHProto-Win64-Shipping',
    [ValidateRange(5,300)][int]$Seconds = 30,
    [ValidateRange(0,300)][int]$DelaySeconds = 0,
    [string]$PresentMonPath,
    [switch]$GpuTelemetry,
    [string]$TelemetryPath,
    [string]$GpuName = 'RX 9070 XT',
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
if ($TelemetryPath) { $GpuTelemetry=$true }
if ($GpuTelemetry) {
    if (!$TelemetryPath) { $TelemetryPath=Join-Path $repo 'build/gpu-telemetry/d4r_gpu_telemetry.exe' }
    $TelemetryPath=(Get-Item -LiteralPath $TelemetryPath -ErrorAction Stop).FullName
    if ($GpuName.Contains('"')) { throw 'GpuName cannot contain a double quote.' }
    foreach ($file in @('gpu-telemetry.csv','gpu-telemetry.stderr.log','gpu-telemetry-environment.json')) {
        if (Test-Path -LiteralPath (Join-Path $OutputDirectory $file)) { throw "Preserve existing $file; choose another OutputDirectory." }
    }
}
@{version='2.6.0'; executable=$PresentMonPath; sha256=(Get-FileHash -LiteralPath $PresentMonPath -Algorithm SHA256).Hash;
    targetProcessId=$target.Id; processName=$target.ProcessName; seconds=$Seconds; delaySeconds=$DelaySeconds;
    gpuTelemetryRequested=[bool]$GpuTelemetry;
    hwsRegistry=$(Get-ItemPropertyValue -LiteralPath 'HKLM:/SYSTEM/CurrentControlSet/Control/GraphicsDrivers' -Name HwSchMode -ErrorAction SilentlyContinue);
    driver=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)} |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'presentmon-environment.json') -Encoding UTF8
# Unique ETW session: never stop or replace an existing user's trace. The CLI
# terminates itself at the bound/process exit, without a service installation.
$session='D4R_' + [Guid]::NewGuid().ToString('N')
$telemetry=$null; $telemetryStarted=$false; $outStream=$null; $errStream=$null; $outTask=$null; $errTask=$null
try {
    if ($GpuTelemetry) {
        $info=[Diagnostics.ProcessStartInfo]::new()
        $info.FileName=$TelemetryPath; $info.UseShellExecute=$false; $info.CreateNoWindow=$true
        $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
        $info.Arguments="--seconds $Seconds --delay-ms $($DelaySeconds*1000) --interval-ms 500 --target-pid $($target.Id) --gpu-name `"$GpuName`""
        $outStream=[IO.File]::Open((Join-Path $OutputDirectory 'gpu-telemetry.csv'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        $errStream=[IO.File]::Open((Join-Path $OutputDirectory 'gpu-telemetry.stderr.log'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        $telemetry=[Diagnostics.Process]::new(); $telemetry.StartInfo=$info
        if (!$telemetry.Start()) { throw 'GPU telemetry process failed to start.' }
        $telemetryStarted=$true
        $outTask=$telemetry.StandardOutput.BaseStream.CopyToAsync($outStream)
        $errTask=$telemetry.StandardError.BaseStream.CopyToAsync($errStream)
    }
    & $PresentMonPath --process_id $target.Id --timed $Seconds --delay $DelaySeconds --terminate_after_timed --terminate_on_proc_exit --session_name $session --no_console_stats --no_track_input --qpc_time_ms --output_file $csv 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputDirectory 'presentmon-console.log')
    if ($LASTEXITCODE) { throw "PresentMon failed with exit code $LASTEXITCODE; see presentmon-console.log." }
} finally {
    try { if ($telemetryStarted) {
        $timedOut=$false
        if (!$telemetry.HasExited -and !$telemetry.WaitForExit(30000)) { $timedOut=$true; $telemetry.Kill(); $telemetry.WaitForExit() }
        if ($outTask) { [void]$outTask.GetAwaiter().GetResult() }
        if ($errTask) { [void]$errTask.GetAwaiter().GetResult() }
        $driverDll=Join-Path $env:WINDIR 'System32/amdadlx64.dll'
        $driverInfo=$null
        if (Test-Path -LiteralPath $driverDll) {
            $driverInfo=@{path=$driverDll; version=(Get-Item -LiteralPath $driverDll).VersionInfo.FileVersion;
                sha256=(Get-FileHash -LiteralPath $driverDll -Algorithm SHA256).Hash}
        }
        @{executable=$TelemetryPath; sha256=(Get-FileHash -LiteralPath $TelemetryPath -Algorithm SHA256).Hash;
            exitCode=$telemetry.ExitCode; timeout=$timedOut; targetProcessId=$target.Id; gpuName=$GpuName;
            intervalMs=500; driverDll=$driverInfo; readOnly=$true;
            buildInfo=$(if (Test-Path -LiteralPath (Join-Path (Split-Path $TelemetryPath) 'build-info.json')) {
                Get-Content -LiteralPath (Join-Path (Split-Path $TelemetryPath) 'build-info.json') -Raw | ConvertFrom-Json
            })} |
            ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'gpu-telemetry-environment.json') -Encoding UTF8
        if ($telemetry.ExitCode -ne 0) { Write-Warning 'GPU telemetry failed; retain PresentMon capture and see gpu-telemetry.stderr.log.' }
    } } finally {
        if ($telemetry) { $telemetry.Dispose() }
        if ($outStream) { $outStream.Dispose() }
        if ($errStream) { $errStream.Dispose() }
    }
}
if (!(Test-Path -LiteralPath $csv) -or (Get-Item -LiteralPath $csv).Length -eq 0) { throw 'PresentMon captured no frames.' }
Write-Host "PresentMon capture: $csv"
