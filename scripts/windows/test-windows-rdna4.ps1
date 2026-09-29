[CmdletBinding()]
param(
    [string]$HipRoot = $env:HIP_PATH,
    [string]$ZludaRoot,
    [string]$PackageRoot,
    [string]$OutputDirectory,
    [int]$Iterations = 32,
    [int]$TimeoutSeconds = 180
)
$ErrorActionPreference = 'Stop'
if (!$PackageRoot) {
    if (Test-Path (Join-Path $PSScriptRoot 'bin/d4r_hip_gfx1201_probe.exe')) { $PackageRoot = $PSScriptRoot }
    else { $PackageRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../dist/windows-rdna4-diagnostics')) }
}
$PackageRoot = [IO.Path]::GetFullPath($PackageRoot)
if (!$HipRoot) { $HipRoot = 'C:\Program Files\AMD\ROCm\7.2' }
$HipRoot = [IO.Path]::GetFullPath($HipRoot)
if (!$ZludaRoot) {
    $localZluda = [IO.Path]::GetFullPath((Join-Path $PackageRoot '../../.tools/zluda/zluda'))
    if (Test-Path (Join-Path $localZluda 'nvcuda.dll')) { $ZludaRoot = $localZluda }
    else { throw 'Pass -ZludaRoot with the directory containing the Windows ZLUDA nvcuda.dll.' }
}
$ZludaRoot = [IO.Path]::GetFullPath($ZludaRoot)
if (!$OutputDirectory) { $OutputDirectory = Join-Path $PackageRoot ('results/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$originalPath = $env:PATH
$originalHip = $env:HIP_PATH
$originalDiag = $env:D4R_DIAG_DIR
$originalLog = $env:ZLUDA_LOG_DIR
$originalCuda = $env:ZLUDA_CUDA_LIB
$exitStatus = 1
$summary = [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); hipRoot=$HipRoot; zludaRoot=$ZludaRoot; tests=@()}
function Quote-Argument([string]$Value) {
    # All generated paths are absolute file/directory paths, with no trailing backslash.
    if ($Value.Contains('"')) { throw 'Double quotes are not allowed in arguments' }
    return '"' + $Value.TrimEnd('\') + '"'
}
function Invoke-Probe([string]$Name, [string]$Exe, [string[]]$Arguments) {
    if (!(Test-Path -LiteralPath $Exe)) { throw "Executable missing: $Exe" }
    $stdout = Join-Path $OutputDirectory "$Name.stdout.log"
    $stderr = Join-Path $OutputDirectory "$Name.stderr.log"
    $argumentLine = ($Arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Exe
    $info.Arguments = $argumentLine
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    if (!$process.Start()) { throw "Process start failed: $Exe" }
    # Consume both streams concurrently; a verbose JIT trace must not block its child.
    $outTask = $process.StandardOutput.ReadToEndAsync()
    $errTask = $process.StandardError.ReadToEndAsync()
    $finished = $process.WaitForExit($TimeoutSeconds * 1000)
    if (!$finished) { $process.Kill(); $process.WaitForExit() }
    $code = $process.ExitCode
    [IO.File]::WriteAllText($stdout, $outTask.GetAwaiter().GetResult())
    [IO.File]::WriteAllText($stderr, $errTask.GetAwaiter().GetResult())
    $process.Dispose()
    $codeHex = if ($null -eq $code) { 'unknown' } else {
        '0x{0:x8}' -f [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$code), 0)
    }
    $pass = $finished -and $code -eq 0 -and (Select-String -LiteralPath $stdout -Pattern '^PASS ' -Quiet)
    $summary.tests += [ordered]@{name=$Name; passed=$pass; timedOut=(!$finished); exitCode=$code; exceptionOrExitHex=$codeHex}
    Write-Host "$Name passed=$pass exit=$codeHex timeout=$(!$finished)"
    return $pass
}
try {
    if ($Iterations -lt 1 -or $Iterations -gt 10000) { throw 'Iterations must be 1..10000' }
    if ($TimeoutSeconds -lt 1) { throw 'TimeoutSeconds must be positive' }
    $env:HIP_PATH = $HipRoot
    $env:PATH = "$(Join-Path $HipRoot 'bin');$ZludaRoot;$originalPath"
    $env:D4R_DIAG_DIR = $OutputDirectory
    $env:ZLUDA_LOG_DIR = Join-Path $OutputDirectory 'zluda-trace'
    $env:ZLUDA_CUDA_LIB = Join-Path $ZludaRoot 'nvcuda.dll'
    $files = @((Join-Path $HipRoot 'bin/amdhip64_7.dll'), (Join-Path $HipRoot 'bin/amd_comgr_3.dll'),
        (Join-Path $ZludaRoot 'nvcuda.dll'), (Join-Path $PackageRoot 'bin/probe_gfx1201.hsaco'))
    $summary.files = @($files | ForEach-Object {
        if (!(Test-Path -LiteralPath $_)) { throw "Required file missing: $_" }
        $file = Get-Item -LiteralPath $_
        [ordered]@{path=$file.FullName; version=$file.VersionInfo.FileVersion;
            sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash}
    })
    try {
        $summary.displayDrivers = @(Get-CimInstance Win32_VideoController |
            Select-Object Name,DriverVersion,DriverDate,PNPDeviceID)
        $summary.os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber
    } catch { $summary.inventoryError = $_.Exception.Message }
    $bin = Join-Path $PackageRoot 'bin'
    $hipOk = Invoke-Probe 'hip' (Join-Path $bin 'd4r_hip_gfx1201_probe.exe') @(
        '--hip-root', $HipRoot, '--module', (Join-Path $bin 'probe_gfx1201.hsaco'), '--iterations', "$Iterations")
    if ($hipOk) {
        $cudaOk = $true
        foreach ($context in @('primary', 'created')) {
            $ok = Invoke-Probe "cuda-$context" (Join-Path $bin 'd4r_cuda_driver_probe.exe') @(
                '--hip-root', $HipRoot, '--cuda-dll', (Join-Path $ZludaRoot 'nvcuda.dll'),
                '--context', $context, '--iterations', "$Iterations")
            $cudaOk = $cudaOk -and $ok
            # On failure collect the upstream trace automatically if available.
            if (!$ok -and (Test-Path (Join-Path $ZludaRoot 'zluda.exe'))) {
                Invoke-Probe "cuda-$context-trace" (Join-Path $ZludaRoot 'zluda.exe') @(
                    '--zluda-trace', '--', (Join-Path $bin 'd4r_cuda_driver_probe.exe'),
                    '--hip-root', $HipRoot, '--cuda-dll', (Join-Path $ZludaRoot 'nvcuda.dll'),
                    '--context', $context, '--iterations', '1') | Out-Null
            }
        }
        if ($cudaOk) { $exitStatus = 0 }
    }
} catch {
    $summary.error = $_.Exception.Message
    Write-Warning $summary.error
} finally {
    $summary.passed = $exitStatus -eq 0
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
    $env:PATH = $originalPath
    $env:HIP_PATH = $originalHip
    $env:D4R_DIAG_DIR = $originalDiag
    $env:ZLUDA_LOG_DIR = $originalLog
    $env:ZLUDA_CUDA_LIB = $originalCuda
    $archive = "$OutputDirectory.zip"
    Compress-Archive -LiteralPath $OutputDirectory -DestinationPath $archive -Force
    Write-Host "Diagnostic bundle: $archive"
}
exit $exitStatus
