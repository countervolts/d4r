# Parse completed diagnostic records. Empty streams are valid, including games
# that refuse a debugger before initializing the upscaler.
function Get-D4RGameSummary {
    param([AllowEmptyString()][string]$Stdout='', [AllowEmptyString()][string]$Stderr='',
        [AllowEmptyString()][string]$Debugger='', [int]$Preset=11, [double]$RunSeconds=0,
        [bool]$ValidateOutput=$false, $GameCrash=$null)
    $gameExit=[regex]::Match($Debugger, 'EXIT code=0x([0-9a-f]+)')
    $exitCode=if ($gameExit.Success) { '0x'+$gameExit.Groups[1].Value } else { 'diagnostic_timeout' }
    $frames=[regex]::Matches($Stdout, '(?m)^D4R_FRAME cpu_frame_copies=([0-9]+) frame_age=([0-9]+) interop_ngx_ms=([0-9.]+)\r?$')
    $checks=[regex]::Matches($Stdout, '(?m)^D4R_OUTPUT_VALIDATION elements=([0-9]+) nan=([0-9]+) inf=([0-9]+) diagnostics_cpu_bytes=8\r?$')
    $launches=[regex]::Matches($Stderr, '\[d4r-launch\] kernel="([^"]+)" backend=(native|translated)')
    $kernels=@($launches | ForEach-Object { $_.Groups[1].Value+' '+$_.Groups[2].Value } | Group-Object | ForEach-Object {
        $parts=$_.Name.Split(' '); @{kernel=$parts[0]; backend=$parts[1]; launches=$_.Count}
    })
    $capabilities=@([regex]::Matches($Stdout, '(?m)^D4R_CAPABILITIES available=(-?[0-9]+) feature_init=(0x[0-9a-f]+) needs_driver=(-?[0-9]+)\r?$') |
        ForEach-Object { $_.Groups[2].Value } | Select-Object -Unique)
    $session=if ($frames.Count) { 'completed-dlss-frames' }
        elseif ($capabilities -contains '0xbad00004') { 'ngx-feature-not-found' }
        elseif ($capabilities.Count -and $capabilities -ne '0x00000001') { 'ngx-capability-failure' }
        elseif ($exitCode -eq '0x0') { 'process-exited-without-dlss-frames' }
        elseif ($exitCode -eq 'diagnostic_timeout') { 'diagnostic-timeout-without-dlss-frames' }
        else { 'process-failed-before-dlss-frames' }
    return [ordered]@{exitCode=$exitCode; session=$session; featureInitResults=$capabilities;
        runSeconds=$RunSeconds; preset=$Preset; gameCrash=$GameCrash;
        nativeLaunches=([regex]::Matches($Stderr, '\[d4r-launch\].*backend=native')).Count;
        completedFrames=$frames.Count; outputValidationRequested=$ValidateOutput;
        outputGpuChecks=$checks.Count; kernels=$kernels;
        nonfiniteOutputs=@($checks | Where-Object { $_.Groups[2].Value -ne '0' -or $_.Groups[3].Value -ne '0' }).Count;
        previousFrameOutputs=@($frames | Where-Object { $_.Groups[2].Value -ne '0' }).Count;
        cpuImageCopyFrames=@($frames | Where-Object { $_.Groups[1].Value -ne '0' }).Count;
        failures=([regex]::Matches($Stderr, 'D4R_[A-Z0-9_]*FAILURE\b')).Count}
}
