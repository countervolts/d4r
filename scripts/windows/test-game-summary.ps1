[CmdletBinding()]
param([string]$LogRoot, [string]$OutputDirectory)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'game-summary.ps1')
$cases=@(
    @{name='empty-streams-debugger-refused'; preset=11; stdout=''; stderr=''; debugger='EXIT code=0xdeadc0de'; session='process-failed-before-dlss-frames'},
    @{name='launcher-exit'; preset=11; stdout=''; stderr=''; debugger='EXIT code=0x0'; session='process-exited-without-dlss-frames'},
    @{name='feature-not-found'; preset=11; stdout='D4R_CAPABILITIES available=0 feature_init=0xbad00004 needs_driver=0'; stderr='D4R_WINDOWS_FAILURE unavailable'; debugger='EXIT code=0xc0000409'; session='ngx-feature-not-found'},
    @{name='complete-frame-k'; preset=11; native=1; kernel='dltss_pwin_enc0_layer'; stdout="D4R_FRAME cpu_frame_copies=0 frame_age=0 interop_ngx_ms=12.4`nD4R_OUTPUT_VALIDATION elements=3 nan=0 inf=0 diagnostics_cpu_bytes=8";
        stderr='[d4r-launch] kernel="dltss_pwin_enc0_layer" backend=native'; debugger='EXIT code=0x0'; session='completed-dlss-frames'},
    @{name='complete-frame-e'; preset=5; native=0; kernel=$null; stdout="D4R_FRAME cpu_frame_copies=0 frame_age=0 interop_ngx_ms=14.2`nD4R_OUTPUT_VALIDATION elements=3 nan=0 inf=0 diagnostics_cpu_bytes=8";
        stderr=''; debugger='EXIT code=0x0'; session='completed-dlss-frames'},
    @{name='complete-frame-l'; preset=12; native=1; kernel='rrlite_enc1_4x4'; stdout="D4R_FRAME cpu_frame_copies=0 frame_age=0 interop_ngx_ms=18.4`nD4R_OUTPUT_VALIDATION elements=3 nan=0 inf=0 diagnostics_cpu_bytes=8";
        stderr='[d4r-launch] kernel="rrlite_enc1_4x4" backend=native'; debugger='EXIT code=0x0'; session='completed-dlss-frames'},
    @{name='truncated-frame'; preset=13; stdout='D4R_FRAME cpu_frame_copies=0 frame_age=0 interop_ngx_ms='; stderr=''; debugger=''; session='diagnostic-timeout-without-dlss-frames'}
)
$results=@()
foreach ($case in $cases) {
    $summary=Get-D4RGameSummary -Stdout $case.stdout -Stderr $case.stderr -Debugger $case.debugger -Preset $case.preset
    if ($summary.session -ne $case.session) { throw "Wrong session classification: $($case.name)" }
    if ($case.name -like 'complete-frame-*') {
        $kernelCount=if ($case.kernel) { @($summary.kernels | Where-Object kernel -eq $case.kernel).Count } else { 0 }
        if ($summary.completedFrames -ne 1 -or $summary.outputGpuChecks -ne 1 -or $summary.nativeLaunches -ne $case.native -or
            $kernelCount -ne $case.native -or $summary.preset -ne $case.preset -or
            $summary.nonfiniteOutputs -or $summary.previousFrameOutputs -or $summary.cpuImageCopyFrames) { throw 'Complete frame counters are wrong.' }
    } elseif ($summary.completedFrames -ne 0) { throw 'A failed or incomplete run was counted as a frame.' }
    $results+=@{case=$case.name; passed=$true; summary=$summary}
}
if ($LogRoot) {
    foreach ($entry in @{ 'acbf-k'='process-failed-before-dlss-frames'; 'cyberpunk-k'='ngx-feature-not-found';
            'cyberpunk-m'='ngx-feature-not-found'; 'sm2-k'='process-exited-without-dlss-frames' }.GetEnumerator()) {
        $dir=Get-ChildItem -LiteralPath (Join-Path $LogRoot $entry.Key) -Directory | Select-Object -First 1
        $summary=Get-D4RGameSummary -Stdout ([IO.File]::ReadAllText((Join-Path $dir.FullName 'd4r.stdout.log'))) `
            -Stderr ([IO.File]::ReadAllText((Join-Path $dir.FullName 'd4r.stderr.log'))) `
            -Debugger ([IO.File]::ReadAllText((Join-Path $dir.FullName 'debugger.log')))
        if ($summary.session -ne $entry.Value -or $summary.completedFrames -ne 0 -or $summary.nativeLaunches -ne 0) { throw "Unexpected classification for actual report $($entry.Key)" }
        $results+=@{case='issue-10-'+$entry.Key; passed=$true; summary=$summary}
    }
}
if ($OutputDirectory) {
    New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
    $results | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary-parser.json') -Encoding UTF8
}
Write-Host "PASS GAME_SUMMARY cases=$($results.Count)"
