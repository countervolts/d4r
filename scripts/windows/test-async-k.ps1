[CmdletBinding()]
param(
    [string]$NgxCore, [string]$DlssDll, [string]$NativeRoot,
    [string]$ZludaRoot, [string]$PackageRoot,
    [string]$Resolution = '3840x2160',
    [switch]$ProfileGpuBoundary,
    [switch]$ProfileKernelsDeferred,
    [switch]$BatchInputCopies,
    [ValidateSet('baseline','packed','unorm','depth-stencil')][string]$PixelProfile = 'depth-stencil',
    [string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$NgxCore) { $NgxCore = Join-Path $repo '_nvngx.dll' }
if (!$DlssDll) { $DlssDll = Join-Path $repo 'nvngx_dlss.dll' }
if (!$NativeRoot) { $NativeRoot = Join-Path $repo 'build/native-k-gfx1201' }
if (!$ZludaRoot) { $ZludaRoot = Join-Path $repo 'dist/zluda-windows-api-profile' }
if (!$PackageRoot) { $PackageRoot = Join-Path $repo 'dist/windows-rdna4-command-list' }
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo ('test-results/async-k-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
if ($Resolution -notmatch '^([0-9]+)x([0-9]+)$') { throw 'Resolution must be WIDTHxHEIGHT' }
$width=$Matches[1]; $height=$Matches[2]
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$settings = @{D4R_DIAG_BURST='1'; D4R_DIAG_RECREATE=$null; D4R_ASYNC_INTEROP=$null;
    D4R_BATCH_INPUT_COPIES=$null;
    D4R_ZLUDA_NATIVE_DIR=$NativeRoot; D4R_QUIET_API='1'; D4R_VALIDATE_OUTPUT='1';
    D4R_PROFILE_STAGES='1'; D4R_ZLUDA_PROFILE=$null; D4R_ZLUDA_PROFILE_API=$null;
    D4R_ZLUDA_PROFILE_DEFERRED=$(if ($ProfileKernelsDeferred) { '1' } else { $null });
    D4R_ZLUDA_PROFILE_ALLOW_LEGACY=$(if ($ProfileKernelsDeferred) { '1' } else { $null });
    D4R_ZLUDA_PROFILE_EVERY=$(if ($ProfileKernelsDeferred) { '1' } else { $null });
    D4R_PROFILE_GPU_BOUNDARY=$(if ($ProfileGpuBoundary) { '1' } else { $null });
    ZLUDA_CACHE_DIR=(Join-Path $repo 'build/zluda-cache-windows');
    PYTHONPATH=(Join-Path $repo '.tools/python/vendor')}
$old=@{}
try {
    foreach ($key in $settings.Keys) {
        $old[$key]=[Environment]::GetEnvironmentVariable($key,'Process')
        [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process')
    }
    foreach ($mode in @('burst','recreate')) {
        [Environment]::SetEnvironmentVariable('D4R_DIAG_RECREATE', $(if ($mode -eq 'recreate') { '1' } else { $null }), 'Process')
        foreach ($variant in @('sync','async')) {
            [Environment]::SetEnvironmentVariable('D4R_ASYNC_INTEROP', $(if ($variant -eq 'async') { '1' } else { $null }), 'Process')
            [Environment]::SetEnvironmentVariable('D4R_BATCH_INPUT_COPIES', $(if ($BatchInputCopies -and $variant -eq 'async') { '1' } else { $null }), 'Process')
            $output=Join-Path $OutputDirectory "$variant-$mode"
            $arguments=@{RuntimeProfile='therock'; ZludaRoot=$ZludaRoot; PackageRoot=$PackageRoot;
                NgxCore=$NgxCore; DlssDll=$DlssDll; NgxOnly=$true; NgxMode='d3d12'; NgxAbi='project-legacy';
                Preset=11; NgxCreateFlags=11; NgxOutputResolution=$Resolution; RequireNativeNetwork=$true;
                CommandListBackend=$true; EarlyIndirectProbe=$true; PixelProfile=$PixelProfile;
                BarrierMode='inherited-legacy'; Iterations=3; CaptureExceptions=$true; TimeoutSeconds=600;
                OptiScalerDll=(Join-Path $repo 'dist/optiscaler-windows-d4r/OptiScaler.dll'); OutputDirectory=$output}
            & (Join-Path $PSScriptRoot 'test-windows-rdna4.ps1') @arguments
            if ($LASTEXITCODE) { throw "Failed $variant-$mode; logs: $output" }
            if ($ProfileKernelsDeferred) {
                $stderr=Get-Content -LiteralPath (Join-Path $output 'd3d12-evaluate-preset-11.stderr.log') -Raw
                if ($stderr -match '(?m)^D4R_KERNEL_PROFILE .*serializing=1') { throw 'Unexpected serializing kernel profiler.' }
                foreach ($layer in @('enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0')) {
                    $pattern='(?m)^D4R_KERNEL_PROFILE(?:_INVALID)? kernel="dltss_pwin_' + $layer + '_layer" backend=native phase=main '
                    $events=[regex]::Matches($stderr,$pattern)
                    if ($events.Count -ne 3) { throw "Expected three collected $layer events for $variant-$mode, found $($events.Count)." }
                }
            }
            if ($ProfileGpuBoundary -and $variant -eq 'async') {
                $stdout=Get-Content -LiteralPath (Join-Path $output 'd3d12-evaluate-preset-11.stdout.log') -Raw
                $timings=[regex]::Matches($stdout, '(?m)^D4R_GPU_BOUNDARY [^\r\n]*diagnostics_cpu_bytes=32 serializing=0\r?$')
                if ($timings.Count -ne 3) { throw "Expected three completed GPU timing records for $variant-$mode, found $($timings.Count)." }
            }
        }
        & python.exe (Join-Path $repo 'tools/windows/frame_compare.py') --reference (Join-Path $OutputDirectory "sync-$mode") --actual (Join-Path $OutputDirectory "async-$mode") --width $width --height $height --exact |
            Tee-Object -FilePath (Join-Path $OutputDirectory "$mode-comparison.log")
        if ($LASTEXITCODE) { throw "Async $mode output mismatch" }
    }
    @{passed=$true; architecture='gfx1201'; preset=11; resolution=$Resolution;
        modes=@('burst','recreate'); framesPerMode=3; exactRgb=$true; gpuBoundaryTiming=[bool]$ProfileGpuBoundary;
        deferredKernelTiming=[bool]$ProfileKernelsDeferred; batchInputs=[bool]$BatchInputCopies; pixelProfile=$PixelProfile} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'validation.json') -Encoding UTF8
    Write-Host "PASS asynchronous K burst/recreate: $OutputDirectory"
} finally {
    foreach ($key in $old.Keys) { [Environment]::SetEnvironmentVariable($key,$old[$key],'Process') }
}
