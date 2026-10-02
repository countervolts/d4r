[CmdletBinding()]
param([string]$ZludaRoot, [string]$PackageRoot, [string]$HipRoot, [string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$ZludaRoot) { $ZludaRoot = Join-Path $repo 'dist/zluda-windows-deferred-profile' }
if (!$PackageRoot) { $PackageRoot = Join-Path $repo 'dist/windows-rdna4-command-list' }
if (!$HipRoot) { $HipRoot = Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo ('test-results/deferred-profile-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$settings = @{
    D4R_ZLUDA_PROFILE=$null; D4R_ZLUDA_PROFILE_DEFERRED='1'; D4R_ZLUDA_PROFILE_EVERY='1';
    D4R_ZLUDA_PROFILE_ALLOW_LEGACY=$null;
    D4R_DIAG_NONBLOCKING_STREAM='1'; D4R_DIAG_QUEUE_PTX=$null;
    D4R_DIAG_DIR=$OutputDirectory;
    ZLUDA_CACHE_DIR=(Join-Path $repo 'build/zluda-cache-windows');
    PATH=((Join-Path $HipRoot 'bin') + ';' + $ZludaRoot + ';' + $env:PATH)
}
$old=@{}; $cases=@()
function RunProbe([string[]]$Arguments, [string]$Stdout, [string]$Stderr) {
    $info=[Diagnostics.ProcessStartInfo]::new()
    $info.FileName=Join-Path $PackageRoot 'bin/d4r_cuda_driver_probe.exe'
    $info.Arguments=$Arguments -join ' '
    $info.UseShellExecute=$false; $info.CreateNoWindow=$true
    $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
    $p=[Diagnostics.Process]::new(); $p.StartInfo=$info
    $outFile=$null; $errFile=$null; $outTask=$null; $errTask=$null
    try {
        if (!$p.Start()) { throw 'Could not launch the CUDA diagnostic.' }
        $outFile=[IO.File]::Open($Stdout,[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        $errFile=[IO.File]::Open($Stderr,[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        $outTask=$p.StandardOutput.BaseStream.CopyToAsync($outFile)
        $errTask=$p.StandardError.BaseStream.CopyToAsync($errFile)
        $timer=[Diagnostics.Stopwatch]::StartNew()
        while (!$p.WaitForExit(1000)) {
            if ($timer.Elapsed.TotalSeconds -ge 120) { throw "CUDA diagnostic timed out; logs: $OutputDirectory" }
        }
        $p.WaitForExit()
        [void]$outTask.GetAwaiter().GetResult(); [void]$errTask.GetAwaiter().GetResult()
        return $p.ExitCode
    } finally {
        if (!$p.HasExited) { $p.Kill(); $p.WaitForExit() }
        if ($outTask) { try { [void]$outTask.GetAwaiter().GetResult() } catch {} }
        if ($errTask) { try { [void]$errTask.GetAwaiter().GetResult() } catch {} }
        if ($outFile) { $outFile.Dispose() }; if ($errFile) { $errFile.Dispose() }
        $p.Dispose()
    }
}
try {
    foreach ($key in $settings.Keys) {
        $old[$key]=[Environment]::GetEnvironmentVariable($key,'Process')
        [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process')
    }
    foreach ($context in @('primary','created')) {
        foreach ($queued in @($false,$true)) {
            $name="$context-$(if ($queued) { 'queued' } else { 'single' })"
            [Environment]::SetEnvironmentVariable('D4R_DIAG_QUEUE_PTX', $(if ($queued) { '1' } else { $null }), 'Process')
            $iterations=129
            $stdout=Join-Path $OutputDirectory "$name.stdout.log"
            $stderr=Join-Path $OutputDirectory "$name.stderr.log"
            # Native Windows argument parsing: quote absolute paths explicitly.
            $arguments=@('--hip-root', ('"' + $HipRoot + '"'), '--cuda-dll', ('"' + (Join-Path $ZludaRoot 'nvcuda.dll') + '"'),
                         '--context', $context, '--iterations', "$iterations")
            $exitCode=RunProbe $arguments $stdout $stderr
            if ($exitCode) { throw "$name failed ($exitCode); logs: $OutputDirectory" }
            $out=Get-Content -LiteralPath $stdout -Raw
            $err=Get-Content -LiteralPath $stderr -Raw
            if ($out -notmatch 'PASS CUDA architecture=gfx1201' -or $out -notmatch 'guard_verified=1') { throw "$name did not verify PTX output" }
            $records=[regex]::Matches($err, '(?m)^D4R_KERNEL_PROFILE kernel="d4r_ptx_pattern" [^\r\n]*serializing=0 deferred=1 [^\r\n]+')
            $invalid=[regex]::Matches($err, '(?m)^D4R_KERNEL_PROFILE_INVALID kernel="d4r_ptx_pattern" [^\r\n]*reason=invalid_elapsed_time serializing=0 deferred=1 [^\r\n]+')
            if ($records.Count + $invalid.Count -ne $iterations -or $err -match 'serializing=1|D4R_KERNEL_PROFILE_SKIPPED') {
                throw "$name expected $iterations collected deferred records, found $($records.Count) valid / $($invalid.Count) invalid"
            }
            foreach ($record in $records) {
                if ($record.Value -notmatch 'gpu_ms=([0-9.]+)') {
                    throw "$name contains an invalid GPU time"
                }
                $gpuMs=[double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture)
                if ([double]::IsNaN($gpuMs) -or [double]::IsInfinity($gpuMs)) { throw "$name contains an invalid GPU time" }
            }
            $cases += @{context=$context; queued=$queued; launches=$iterations; collected=($records.Count+$invalid.Count);
                validTimings=$records.Count; invalidTimings=$invalid.Count; exitCode=$exitCode}
            Write-Host "PASS ${name}: $($records.Count) valid / $($invalid.Count) invalid deferred timings; exact PTX output and guards"
        }
    }
    # Legacy default-stream timing markers would add stream dependencies.
    [Environment]::SetEnvironmentVariable('D4R_DIAG_NONBLOCKING_STREAM',$null,'Process')
    [Environment]::SetEnvironmentVariable('D4R_DIAG_QUEUE_PTX',$null,'Process')
    $stdout=Join-Path $OutputDirectory 'default-stream.stdout.log'; $stderr=Join-Path $OutputDirectory 'default-stream.stderr.log'
    $arguments=@('--hip-root', ('"' + $HipRoot + '"'), '--cuda-dll', ('"' + (Join-Path $ZludaRoot 'nvcuda.dll') + '"'), '--iterations', '3')
    $exitCode=RunProbe $arguments $stdout $stderr
    $out=Get-Content -LiteralPath $stdout -Raw; $err=Get-Content -LiteralPath $stderr -Raw
    if ($exitCode -or $out -notmatch 'PASS CUDA architecture=gfx1201' -or
        $err -notmatch 'reason=legacy_default_stream' -or $err -match '(?m)^D4R_KERNEL_PROFILE ') {
        throw 'Default-stream diagnostics did not safely skip timing events.'
    }
    @{passed=$true; architecture='gfx1201'; cases=$cases; defaultStreamSkipped=$true; addedCompletionWaits=0} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'validation.json') -Encoding UTF8
    Write-Host "PASS deferred kernel profiling: $OutputDirectory"
} finally {
    foreach ($key in $old.Keys) { [Environment]::SetEnvironmentVariable($key,$old[$key],'Process') }
}
