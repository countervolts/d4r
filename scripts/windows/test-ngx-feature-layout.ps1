# Reproduce numeric NGX feature discovery without launching or modifying a game.
# The private NVIDIA DLL staging directory is deleted; reports contain logs only.
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$DiagnosticRoot,
    [Parameter(Mandatory=$true)][string]$HipRoot,
    [Parameter(Mandatory=$true)][string]$ZludaRoot,
    [Parameter(Mandatory=$true)][string]$NativeRoot,
    [Parameter(Mandatory=$true)][string]$OptiScalerDll,
    [Parameter(Mandatory=$true)][string]$NgxCore,
    [Parameter(Mandatory=$true)][string]$DlssDll,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [UInt64]$ApplicationId=100152211)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
foreach ($name in @('DiagnosticRoot','HipRoot','ZludaRoot','NativeRoot','OptiScalerDll','NgxCore','DlssDll')) {
    Set-Variable -Name $name -Value ((Get-Item -LiteralPath (Get-Variable -Name $name -ValueOnly)).FullName)
}
$arch=Get-D4RGpuTarget $DiagnosticRoot
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$runtime=Join-Path ([IO.Path]::GetTempPath()) ('d4r-feature-layout-'+[Guid]::NewGuid().ToString('N'))
$settings=@{HIP_PATH=$HipRoot; PATH="$HipRoot\bin;$ZludaRoot;$env:PATH";
    D4R_NVAPI_BACKEND=(Join-Path $ZludaRoot 'nvapi64.dll'); ZLUDA_CUDA_LIB=(Join-Path $ZludaRoot 'nvcuda.dll');
    D4R_ZLUDA_NATIVE_DIR=$NativeRoot; D4R_ZLUDA_VERBOSE='1';
    D4R_ZLUDA_WMMA='1'; D4R_ZLUDA_WMMA_F16_REFERENCE='1'; D4R_ZLUDA_WMMA_FP8='1'; D4R_ZLUDA_WMMA_FP8_NATIVE='0';
    D4R_VALIDATE_OUTPUT='1'; D4R_DIAG_DIR=$null; ZLUDA_LOG_DIR=$null; D4R_FORMAT_MODULE=$null;
    D4R_ASYNC_INTEROP=$null; D4R_BATCH_INPUT_COPIES=$null; D4R_D3D12_COMMAND_BACKEND=$null}
$old=@{}; $results=@()
try {
    foreach ($key in $settings.Keys) { $old[$key]=[Environment]::GetEnvironmentVariable($key,'Process'); [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process') }
    foreach ($dir in @('app','d4r','d4r/vendor')) { New-Item -ItemType Directory -Force (Join-Path $runtime $dir) | Out-Null }
    $exe=Join-Path $runtime 'app/d4r_d3d12_dlss_probe.exe'
    $shim=Join-Path $runtime 'd4r/_nvngx.dll'
    $core=Join-Path $runtime 'd4r/vendor/_nvngx.dll'
    $feature=Join-Path $runtime 'd4r/vendor/nvngx_dlss.dll'
    Copy-Item -LiteralPath (Join-Path $DiagnosticRoot 'bin/d4r_d3d12_dlss_probe.exe') -Destination $exe
    Copy-Item -LiteralPath (Join-Path $DiagnosticRoot 'bin/d4r_nvngx.dll') -Destination $shim
    Copy-Item -LiteralPath (Join-Path $DiagnosticRoot "bin/pixel_convert_${arch}.hsaco") -Destination (Split-Path $shim)
    Copy-Item -LiteralPath $NgxCore -Destination $core
    Copy-Item -LiteralPath $DlssDll -Destination $feature
    $opti=Join-Path $runtime 'app/OptiScaler.dll'
    Copy-Item -LiteralPath $OptiScalerDll -Destination $opti
    $cases=@(@{name='vendor-only-negative'; preset=11; frontend='d4r'; expectedExit=4},
        @{name='optiscaler-vendor-only-negative'; preset=11; frontend='optiscaler'; expectedExit=4},
        @{name='caller-dll-k'; preset=11; frontend='d4r'; expectedExit=0},
        @{name='optiscaler-caller-k'; preset=11; frontend='optiscaler'; expectedExit=0},
        @{name='optiscaler-caller-m'; preset=13; frontend='optiscaler'; expectedExit=0})
    foreach ($case in $cases) {
        $out=Join-Path $OutputDirectory $case.name
        New-Item -ItemType Directory -Force $out | Out-Null
        if ($case.expectedExit -eq 0) {
            $feature=Join-Path $runtime 'd4r/nvngx_dlss.dll'
            if (!(Test-Path -LiteralPath $feature)) { Copy-Item -LiteralPath $DlssDll -Destination $feature }
        }
        $env:D4R_DIAG_DIR=$out; $env:ZLUDA_LOG_DIR=Join-Path $out 'zluda-trace'
        $env:D4R_FORMAT_MODULE=Join-Path $runtime "d4r/pixel_convert_${arch}.hsaco"
        $module=if ($case.frontend -eq 'optiscaler') { $opti } else { $shim }
        if ($case.frontend -eq 'optiscaler') {
            @"
[Upscalers]
Dx12Upscaler=dlss
[FrameGen]
Enabled=false
FGInput=nofg
FGOutput=nofg
[DLSS]
Enabled=true
AllowExternalBackend=true
RenderPresetOverride=true
RenderPresetForAll=$($case.preset)
UseGenericAppIdWithDlss=false
[Libraries]
NvngxPath=$(Split-Path $shim)
NvngxDlssPath=$(Split-Path $feature)
NvngxFeaturePath=$(Split-Path $feature)
[Hotfix]
RestoreComputeSignature=false
RestoreGraphicSignature=false
ExtendedStateRestore=false
[Log]
LogToFile=true
LogLevel=0
[Menu]
DisableSplash=true
"@ | Set-Content -LiteralPath (Join-Path $runtime 'app/OptiScaler.ini') -Encoding UTF8
        }
        $arguments=@('--hip-root',$HipRoot,'--cuda-dll',(Join-Path $ZludaRoot 'nvcuda.dll'),
            '--module',$module,'--ngx-core',$core,'--dlss-dll',$feature,
            '--nvapi-dll',(Join-Path $DiagnosticRoot 'nvapi-compat/nvapi64.dll'),
            '--ngx-app-id',"$ApplicationId",'--ngx-abi','driver','--ngx-frontend',$case.frontend,
            '--interop-mode','command-list','--iterations','3','--preset',"$($case.preset)")
        $info=New-Object Diagnostics.ProcessStartInfo
        # Capture harness faults outside OptiScaler's loader hooks. Games use
        # the no-debug launcher by default; this developer reproducer is separate.
        $arguments=@('--output-directory',$out,'--',$exe)+$arguments
        $info.FileName=Join-Path $DiagnosticRoot 'bin/d4r_debug_launcher.exe'; $info.WorkingDirectory=Split-Path $exe
        $info.Arguments=(@($arguments | ForEach-Object { if ($_.Contains('"')) { throw 'Unsupported quote in test path.' }; '"'+$_+'"' }) -join ' ')
        $info.UseShellExecute=$false; $info.CreateNoWindow=$true; $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
        foreach ($key in $settings.Keys) { if (![Environment]::GetEnvironmentVariable($key,'Process')) { $info.EnvironmentVariables.Remove($key) } }
        $process=New-Object Diagnostics.Process; $process.StartInfo=$info
        try {
            [void]$process.Start(); $stdout=$process.StandardOutput.ReadToEndAsync(); $stderr=$process.StandardError.ReadToEndAsync()
            if (!$process.WaitForExit(240000)) { $process.Kill(); $process.WaitForExit(); throw "NGX layout test timed out: $($case.name)" }
            $log=$stdout.GetAwaiter().GetResult(); $errorLog=$stderr.GetAwaiter().GetResult()
            [IO.File]::WriteAllText((Join-Path $out 'stdout.log'),$log)
            [IO.File]::WriteAllText((Join-Path $out 'stderr.log'),$errorLog)
            if ($process.ExitCode -ne $case.expectedExit) { throw "Unexpected exit $($process.ExitCode): $($case.name), see $out" }
            if ($case.expectedExit) {
                if ($log -notmatch 'D4R_CAPABILITIES available=0 feature_init=0xbad00004' -or $log -match 'D4R_FRAME ') { throw 'Old vendor-only layout did not reproduce FeatureNotFound.' }
                if ($case.frontend -eq 'optiscaler' -and $errorLog -notmatch 'FAIL D3D12_DLSS D3D12 Init NGX=') { throw 'OptiScaler swallowed the numeric backend initialization failure.' }
            } else {
                if ($log -notmatch 'D4R_CAPABILITIES available=1 feature_init=0x00000001 needs_driver=0' -or
                    ([regex]::Matches($log,'D3D12_OUTPUT preset=\d+ frame=\d+ finite=1')).Count -ne 3 -or
                    ([regex]::Matches($log,'D4R_OUTPUT_VALIDATION elements=\d+ nan=0 inf=0 diagnostics_cpu_bytes=8')).Count -ne 3 -or
                    $log -notmatch 'PASS D3D12_DLSS .*fast_path_cpu_copies=0 frame_age=0' -or $errorLog -match 'D4R_WINDOWS_FAILURE') { throw 'Corrected feature layout did not complete finite zero-copy frames.' }
                $required=@(Get-Content -LiteralPath (Join-Path $NativeRoot 'd4r-kernels.txt') | Where-Object {
                    if ($case.preset -eq 11) { $_ -match '^dltss_pwin_' } else { $_ -match '^rrlite_' }
                } | ForEach-Object { $_.Split(' ')[0] })
                if ($required.Count -ne $(if ($case.preset -eq 11) { 11 } else { 5 })) { throw 'Incomplete native manifest.' }
                foreach ($kernel in $required) {
                    if ($errorLog -notmatch ('\[d4r-launch\] kernel="'+[regex]::Escape($kernel)+'" backend=native') -or
                        $errorLog -match ('\[d4r-launch\] kernel="'+[regex]::Escape($kernel)+'" backend=translated')) { throw "Missing native network execution: $kernel" }
                }
                if ($case.frontend -eq 'd4r' -and $log -notmatch 'D3D12_REQUIREMENTS warp_supported=0 enumeration_success=1') { throw 'WARP adapter enumeration regression failed.' }
            }
            $results+=@{case=$case.name; passed=$true; expectedExit=$case.expectedExit; application=$ApplicationId; preset=$case.preset; architecture=$arch}
            Write-Host "PASS NGX_FEATURE_LAYOUT $($case.name)"
        } finally {
            if ($case.frontend -eq 'optiscaler') {
                Get-ChildItem -LiteralPath (Split-Path $exe) -Filter '*.log' -File | ForEach-Object {
                    Copy-Item -LiteralPath $_.FullName -Destination $out -Force
                }
            }
            $process.Dispose()
        }
    }
    for ($frame=0; $frame -lt 3; $frame++) {
        $direct=[IO.File]::ReadAllBytes((Join-Path $OutputDirectory "caller-dll-k/output-$frame.rgba16f"))
        $frontend=[IO.File]::ReadAllBytes((Join-Path $OutputDirectory "optiscaler-caller-k/output-$frame.rgba16f"))
        if ($direct.Length -ne $frontend.Length -or $direct.Length -ne 512*288*8) { throw 'RGB comparison dimensions differ.' }
        for ($pixel=0; $pixel -lt $direct.Length; $pixel+=8) {
            if (([BitConverter]::ToUInt64($direct,$pixel) -band [UInt64]0xffffffffffff) -ne
                ([BitConverter]::ToUInt64($frontend,$pixel) -band [UInt64]0xffffffffffff)) { throw "K frontend RGB mismatch: frame=$frame offset=$pixel" }
        }
    }
    $results+=@{case='k-direct-optiscaler-rgb'; passed=$true; frames=3; rgbByteExact=$true; maxAbsoluteError=0; maxRelativeError=0}
    Write-Host 'PASS NGX_FEATURE_LAYOUT k_direct_optiscaler_rgb exact=1 max_abs=0 max_rel=0'
} finally {
    foreach ($key in $old.Keys) { [Environment]::SetEnvironmentVariable($key,$old[$key],'Process') }
    $results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'feature-layout.json') -Encoding UTF8
    $resolved=[IO.Path]::GetFullPath($runtime)
    $temp=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')+'\'
    if (!$resolved.StartsWith($temp,[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName($resolved) -notmatch '^d4r-feature-layout-[0-9a-f]{32}$') { throw 'Refusing cleanup outside private NGX staging.' }
    if (Test-Path -LiteralPath $resolved) {
        foreach ($attempt in 1..10) {
            try { Remove-Item -LiteralPath $resolved -Recurse -Force; break }
            catch { if ($attempt -eq 10) { throw }; Start-Sleep -Milliseconds 300 }
        }
    }
}
