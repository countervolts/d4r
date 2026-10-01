[CmdletBinding()]
param(
    [string]$HipRoot = $env:HIP_PATH,
    [ValidateSet('stable', 'therock')][string]$RuntimeProfile = 'stable',
    [string]$ZludaRoot,
    [string]$PackageRoot,
    [string]$OutputDirectory,
    [string]$NgxCore = $env:D4R_NGX_CORE,
    [string]$DlssDll = $env:D4R_DLSS_DLL,
    [ValidateSet('init', 'evaluate', 'd3d12')][string]$NgxMode = 'init',
    [ValidateSet('driver','sdk','project')][string]$NgxAbi = 'driver',
    [string]$OptiScalerDll,
    [ValidateSet(5, 11, 13)][int]$Preset = 11,
    [switch]$NgxOnly,
    [switch]$Trace,
    [switch]$RequireNativeNetwork,
    [switch]$CommandListBackend,
    [switch]$CaptureExceptions,
    [switch]$EarlyIndirectProbe,
    [ValidateSet('baseline','packed','unorm')][string]$PixelProfile = 'baseline',
    [ValidateSet('legacy','enhanced','inherited-legacy','inherited-enhanced')][string]$BarrierMode = 'legacy',
    [int]$Iterations = 32,
    [int]$TimeoutSeconds = 180
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if ($RuntimeProfile -eq 'therock') {
    # Works both from scripts/windows and the installed dist/<profile> directory.
    if (!$PSBoundParameters.ContainsKey('HipRoot')) { $HipRoot = Join-Path $repoRoot '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
    if (!$PackageRoot) {
        if (Test-Path (Join-Path $PSScriptRoot 'bin/d4r_hip_gfx1201_probe.exe')) { $PackageRoot = $PSScriptRoot }
        else { $PackageRoot = Join-Path $repoRoot 'dist/windows-rdna4-therock' }
    }
}
if (!$PackageRoot) {
    if (Test-Path (Join-Path $PSScriptRoot 'bin/d4r_hip_gfx1201_probe.exe')) { $PackageRoot = $PSScriptRoot }
    else { $PackageRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../dist/windows-rdna4-diagnostics')) }
}
$PackageRoot = [IO.Path]::GetFullPath($PackageRoot)
if (!$HipRoot) { $HipRoot = 'C:\Program Files\AMD\ROCm\7.2' }
$HipRoot = [IO.Path]::GetFullPath($HipRoot)
if (!$ZludaRoot) {
    $builtZluda = Join-Path $repoRoot 'dist/zluda-windows-native'
    $localZluda = [IO.Path]::GetFullPath((Join-Path $PackageRoot '../../.tools/zluda/zluda'))
    if (Test-Path (Join-Path $builtZluda 'build-info.json')) { $ZludaRoot = $builtZluda }
    elseif (Test-Path (Join-Path $localZluda 'nvcuda.dll')) { $ZludaRoot = $localZluda }
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
$originalNvapi = $env:D4R_NVAPI_BACKEND
$originalPythonPath = $env:PYTHONPATH
$originalVerbose = $env:D4R_ZLUDA_VERBOSE
$originalCache = $env:ZLUDA_CACHE_DIR
$originalCodegen = @{}
foreach ($setting in @('D4R_ZLUDA_WMMA','D4R_ZLUDA_WMMA_FP8','D4R_ZLUDA_WMMA_FP8_NATIVE','D4R_ZLUDA_WMMA_F16_REFERENCE')) {
    $originalCodegen[$setting] = [Environment]::GetEnvironmentVariable($setting,'Process')
}
$exitStatus = 1
$ngxRuntimeDirectory = $null
$summary = [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); profile=$RuntimeProfile; hipRoot=$HipRoot; zludaRoot=$ZludaRoot; tests=@()}
function Quote-Argument([string]$Value) {
    # All generated paths are absolute file/directory paths, with no trailing backslash.
    if ($Value.Contains('"')) { throw 'Double quotes are not allowed in arguments' }
    return '"' + $Value.TrimEnd('\') + '"'
}
function Invoke-Probe([string]$Name, [string]$Exe, [string[]]$Arguments) {
    if (!(Test-Path -LiteralPath $Exe)) { throw "Executable missing: $Exe" }
    if ($CaptureExceptions) {
        $Arguments = @('--output-directory', $OutputDirectory, '--', $Exe) + $Arguments
        $Exe = Join-Path $PackageRoot 'bin/d4r_debug_launcher.exe'
        if (!(Test-Path -LiteralPath $Exe)) { throw "Debugger launcher missing: $Exe" }
    }
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
    $outFile = [IO.File]::Open($stdout, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $errFile = [IO.File]::Open($stderr, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $outTask = $process.StandardOutput.BaseStream.CopyToAsync($outFile)
    $errTask = $process.StandardError.BaseStream.CopyToAsync($errFile)
    $finished = $process.WaitForExit($TimeoutSeconds * 1000)
    if (!$finished) { $process.Kill(); $process.WaitForExit() }
    $code = $process.ExitCode
    try {
        $null = $outTask.GetAwaiter().GetResult()
        $null = $errTask.GetAwaiter().GetResult()
    } finally { $outFile.Dispose(); $errFile.Dispose() }
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
    if (!$env:ZLUDA_CACHE_DIR) { $env:ZLUDA_CACHE_DIR = Join-Path $repoRoot 'build/zluda-cache-windows' }
    $summary.zludaCacheDirectory = $env:ZLUDA_CACHE_DIR
    foreach ($setting in @('D4R_ZLUDA_WMMA','D4R_ZLUDA_WMMA_FP8','D4R_ZLUDA_WMMA_FP8_NATIVE','D4R_ZLUDA_WMMA_F16_REFERENCE')) {
        if ($null -eq $originalCodegen[$setting]) {
            [Environment]::SetEnvironmentVariable($setting, $(if ($setting -eq 'D4R_ZLUDA_WMMA_FP8_NATIVE') { '0' } else { '1' }), 'Process')
        }
    }
    $summary.codegen = @{}
    foreach ($setting in $originalCodegen.Keys) { $summary.codegen[$setting] = [Environment]::GetEnvironmentVariable($setting,'Process') }
    Write-Host "ZLUDA codegen: WMMA=$env:D4R_ZLUDA_WMMA FP8 widening=$env:D4R_ZLUDA_WMMA_FP8 native FP8=$env:D4R_ZLUDA_WMMA_FP8_NATIVE f16 reference=$env:D4R_ZLUDA_WMMA_F16_REFERENCE"
    if ($NgxOnly -and (!$NgxCore -or !$DlssDll)) { throw '-NgxOnly requires both locally supplied NVIDIA DLL paths.' }
    if ($RequireNativeNetwork) {
        if ($NgxMode -notin @('evaluate','d3d12') -or $Preset -notin @(11,13) -or !$env:D4R_ZLUDA_NATIVE_DIR) {
            throw '-RequireNativeNetwork requires K/M Evaluate and D4R_ZLUDA_NATIVE_DIR.'
        }
        $env:D4R_ZLUDA_VERBOSE = '1'
    }
    $summary.foundationDiagnosticsPerformed = !$NgxOnly
    if ($Iterations -lt 1 -or $Iterations -gt 10000) { throw 'Iterations must be 1..10000' }
    if ($TimeoutSeconds -lt 1) { throw 'TimeoutSeconds must be positive' }
    $env:HIP_PATH = $HipRoot
    $env:PATH = "$(Join-Path $HipRoot 'bin');$ZludaRoot;$originalPath"
    $env:D4R_DIAG_DIR = $OutputDirectory
    $env:ZLUDA_LOG_DIR = Join-Path $OutputDirectory 'zluda-trace'
    $env:ZLUDA_CUDA_LIB = Join-Path $ZludaRoot 'nvcuda.dll'
    $env:D4R_NVAPI_BACKEND = Join-Path $ZludaRoot 'nvapi64.dll'
    $localPythonVendor = Join-Path $repoRoot '.tools/python/vendor'
    if (Test-Path (Join-Path $localPythonVendor 'numpy/__init__.py')) {
        $env:PYTHONPATH = "$localPythonVendor;$originalPythonPath"
    }
    $comgr = Join-Path $HipRoot 'bin/amd_comgr_3.dll'
    if (!(Test-Path -LiteralPath $comgr)) { $comgr = Join-Path $HipRoot 'bin/amd_comgr.dll' }
    $files = @((Join-Path $HipRoot 'bin/amdhip64_7.dll'), $comgr,
        (Join-Path $ZludaRoot 'nvcuda.dll'), (Join-Path $PackageRoot 'bin/probe_gfx1201.hsaco'),
        (Join-Path $PackageRoot 'bin/wmma_gfx1201.hsaco'),
        (Join-Path $PackageRoot 'experimental/k/dltss_pwin_enc1_layer_gfx1201.hsaco'),
        (Join-Path $PackageRoot 'experimental/k/dltss_pwin_enc2_layer_gfx1201.hsaco'))
    if ($OptiScalerDll) {
        if ($NgxMode -ne 'd3d12' -or !$CommandListBackend) { throw '-OptiScalerDll requires D3D12 and -CommandListBackend' }
        if (![IO.Path]::IsPathRooted($OptiScalerDll)) { throw '-OptiScalerDll must be absolute' }
        $files += $OptiScalerDll
    }
    if ($NgxCore -or $DlssDll) {
        if (!$NgxCore -or !$DlssDll) { throw 'Supply both -NgxCore and -DlssDll absolute paths.' }
        if (![IO.Path]::IsPathRooted($NgxCore) -or ![IO.Path]::IsPathRooted($DlssDll)) {
            throw 'NVIDIA DLL paths must be absolute.'
        }
        $files += @($NgxCore, $DlssDll)
    }
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
    $hipOk = $true
    if (!$NgxOnly) { $hipOk = Invoke-Probe 'hip' (Join-Path $bin 'd4r_hip_gfx1201_probe.exe') @(
        '--hip-root', $HipRoot, '--module', (Join-Path $bin 'probe_gfx1201.hsaco'), '--iterations', "$Iterations")
    }
    if ($hipOk) {
        if (!$NgxOnly) {
        $wmmaOk = Invoke-Probe 'gfx12-wmma' (Join-Path $bin 'd4r_gfx12_wmma_probe.exe') @(
            '--hip-root', $HipRoot, '--module', (Join-Path $bin 'wmma_gfx1201.hsaco'), '--iterations', "$Iterations")
        if (!$wmmaOk) { throw 'gfx12 WMMA layout validation failed; refusing to mark K/M readiness.' }
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (!$pythonCommand) { throw 'Python 3.11+ with NumPy 2.4.6 is needed for K reference checks; run setup-windows-tools.ps1.' }
        foreach ($layer in @('enc1', 'enc2')) {
            $kFixture = Join-Path $OutputDirectory "k-$layer-fixture"
            $kModuleOk = Invoke-Probe "k-$layer-gpu" (Join-Path $bin 'd4r_k_module_probe.exe') @(
                '--hip-root', $HipRoot,
                '--module', (Join-Path $PackageRoot "experimental/k/dltss_pwin_${layer}_layer_gfx1201.hsaco"),
                '--kernel-name', $layer, '--fixture-dir', $kFixture, '--iterations', "$Iterations")
            if (!$kModuleOk) { throw "K $layer GPU execution failed; see the k-$layer-gpu logs." }
            $kReferenceOk = Invoke-Probe "k-$layer-reference" $pythonCommand.Source @(
                (Join-Path $PackageRoot 'k_layer_validate.py'), '--kernel-name', $layer, '--fixture-dir', $kFixture)
            if (!$kReferenceOk) { throw "K $layer output differs from pwin_model.py; see the k-$layer-reference logs." }
        }
        }
        $cudaOk = $true
        if (!$NgxOnly) {
        foreach ($context in @('primary', 'created')) {
            $ok = Invoke-Probe "cuda-$context" (Join-Path $bin 'd4r_cuda_driver_probe.exe') @(
                '--hip-root', $HipRoot, '--cuda-dll', (Join-Path $ZludaRoot 'nvcuda.dll'),
                '--context', $context, '--iterations', "$Iterations")
            $cudaOk = $cudaOk -and $ok
            # On failure collect the upstream trace automatically if available.
            if (!$ok -and (Test-Path (Join-Path $ZludaRoot 'trace/nvcuda.dll'))) {
                Invoke-Probe "cuda-$context-trace" (Join-Path $bin 'd4r_cuda_driver_probe.exe') @(
                    '--hip-root', $HipRoot, '--cuda-dll', (Join-Path $ZludaRoot 'trace/nvcuda.dll'),
                    '--context', $context, '--iterations', '1') | Out-Null
            }
        }
        if ($cudaOk -and ($NgxMode -eq 'evaluate' -or (Test-Path (Join-Path $ZludaRoot 'build-info.json')))) {
            $cudaOk = Invoke-Probe 'cuda-images' (Join-Path $bin 'd4r_cuda_image_probe.exe') @(
                '--hip-root', $HipRoot, '--cuda-dll', (Join-Path $ZludaRoot 'nvcuda.dll'), '--iterations', "$Iterations")
            if ($cudaOk) {
                $previousF16 = $env:D4R_ZLUDA_WMMA_F16_REFERENCE
                try {
                    $env:D4R_ZLUDA_WMMA_F16_REFERENCE = '1'
                    $cudaOk = Invoke-Probe 'cuda-f16-mma-reference' (Join-Path $bin 'd4r_cuda_mma_probe.exe') @(
                        '--hip-root', $HipRoot, '--cuda-dll', (Join-Path $ZludaRoot 'nvcuda.dll'))
                } finally { $env:D4R_ZLUDA_WMMA_F16_REFERENCE = $previousF16 }
            }
        }
        }
        if ($cudaOk) {
            $mapOk = $interopOk = $true
            if (!$NgxOnly) {
            $commandOk = Invoke-Probe 'd3d12-command-backend' (Join-Path $bin 'd4r_d3d12_command_probe.exe') @(
                '--hip-root', $HipRoot, '--module', (Join-Path $bin 'd4r_nvngx.dll'), '--interop-mode', 'indirect', '--iterations', "$Iterations")
            if (!$commandOk) { throw 'D3D12 command-list order/state/lifetime validation failed.' }
            $pixelsOk = Invoke-Probe 'd3d12-gpu-pixel-formats' (Join-Path $bin 'd4r_d3d12_pixel_probe.exe') @(
                '--hip-root', $HipRoot, '--module', (Join-Path $bin 'pixel_convert_gfx1201.hsaco'), '--iterations', '2')
            if (!$pixelsOk) { throw 'D3D12 typed SRV/UAV format conversion validation failed.' }
            $mapOk = Invoke-Probe 'interop-map-lifetime' (Join-Path $bin 'd4r_d3d12_hip_interop_probe.exe') @(
                '--hip-root', $HipRoot, '--interop-mode', 'map', '--iterations', '64')
            $interopOk = Invoke-Probe 'interop-roundtrip' (Join-Path $bin 'd4r_d3d12_hip_interop_probe.exe') @(
                '--hip-root', $HipRoot, '--module', (Join-Path $bin 'probe_gfx1201.hsaco'), '--iterations', "$Iterations")
            if (!$mapOk -or !$interopOk) {
                foreach ($mode in @('resource', 'import')) {
                    Invoke-Probe "interop-$mode-lifetime" (Join-Path $bin 'd4r_d3d12_hip_interop_probe.exe') @(
                        '--hip-root', $HipRoot, '--interop-mode', $mode, '--iterations', '32') | Out-Null
                }
                Invoke-Probe 'hip-stream-lifetime' (Join-Path $bin 'd4r_hip_stream_lifecycle_probe.exe') @(
                    '--hip-root', $HipRoot, '--module', (Join-Path $bin 'probe_gfx1201.hsaco'), '--iterations', '32') | Out-Null
            }
            }
            if ($mapOk -and $interopOk) { $exitStatus = 0 }
            if ($exitStatus -eq 0 -and $NgxCore) {
                # The driver Init ABI searches the executable directory. Keep the
                # supplied binaries in a private temporary runtime, outside dist/ZIP.
                $ngxRuntimeDirectory = Join-Path ([IO.Path]::GetTempPath()) ('d4r-ngx-' + [Guid]::NewGuid().ToString('N'))
                New-Item -ItemType Directory -Path $ngxRuntimeDirectory | Out-Null
                $ngxExe = Join-Path $ngxRuntimeDirectory 'd4r_ngx_cuda_init_probe.exe'
                $localCore = Join-Path $ngxRuntimeDirectory '_nvngx.dll'
                $localDlss = Join-Path $ngxRuntimeDirectory 'nvngx_dlss.dll'
                $probeName = if ($NgxMode -eq 'd3d12') { 'd4r_d3d12_dlss_probe.exe' } else { 'd4r_ngx_cuda_init_probe.exe' }
                Copy-Item -LiteralPath (Join-Path $bin $probeName) -Destination $ngxExe
                Copy-Item -LiteralPath $NgxCore -Destination $localCore
                Copy-Item -LiteralPath $DlssDll -Destination $localDlss
                $selectedCuda = if ($Trace) { Join-Path $ZludaRoot 'trace/nvcuda.dll' } else { Join-Path $ZludaRoot 'nvcuda.dll' }
                if (!(Test-Path -LiteralPath $selectedCuda)) { throw "CUDA runtime missing: $selectedCuda" }
                $summary.traceRequested = [bool]$Trace
                $ngxArguments = @('--hip-root', $HipRoot, '--cuda-dll', $selectedCuda,
                    '--ngx-core', $localCore, '--dlss-dll', $localDlss,
                    '--nvapi-dll', (Join-Path $PackageRoot 'nvapi-compat/nvapi64.dll'),
                    '--ngx-mode', $NgxMode, '--ngx-abi', $NgxAbi, '--preset', "$Preset", '--iterations', "$Iterations")
                if ($NgxMode -eq 'd3d12') {
                    # The driver NGX core resolves feature DLLs relative to its
                    # caller module as well as the executable. Keep the shim
                    # alongside the private user-supplied DLLs for this test.
                    $localShim = Join-Path $ngxRuntimeDirectory 'd4r_nvngx.dll'
                    Copy-Item -LiteralPath (Join-Path $bin 'd4r_nvngx.dll') -Destination $localShim
                    Copy-Item -LiteralPath (Join-Path $bin 'pixel_convert_gfx1201.hsaco') -Destination $ngxRuntimeDirectory
                    if ($OptiScalerDll) {
                        $bridgeDirectory = Join-Path $ngxRuntimeDirectory 'd4r'
                        New-Item -ItemType Directory -Path $bridgeDirectory | Out-Null
                        Copy-Item -LiteralPath $localShim -Destination (Join-Path $bridgeDirectory '_nvngx.dll')
                        Copy-Item -LiteralPath (Join-Path $bin 'pixel_convert_gfx1201.hsaco') -Destination $bridgeDirectory
                        $localShim = Join-Path $ngxRuntimeDirectory 'OptiScaler.dll'
                        Copy-Item -LiteralPath $OptiScalerDll -Destination $localShim
                        $ini = @"
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
RenderPresetForAll=$Preset
UseGenericAppIdWithDlss=false
[Libraries]
NvngxPath=$bridgeDirectory
NvngxDlssPath=$localDlss
NvngxFeaturePath=$ngxRuntimeDirectory
[Hotfix]
RestoreComputeSignature=false
RestoreGraphicSignature=false
ExtendedStateRestore=false
[Log]
LogToFile=true
LogLevel=0
[Menu]
DisableSplash=true
"@
                        $ini | Set-Content -LiteralPath (Join-Path $ngxRuntimeDirectory 'OptiScaler.ini') -Encoding UTF8
                        $summary.optiScaler = @{path=$OptiScalerDll; mode='standalone'; preset=$Preset}
                        $ngxArguments += @('--ngx-frontend', 'optiscaler')
                    }
                    $ngxArguments += @('--module', $localShim, '--pixel-profile', $PixelProfile, '--barrier-mode', $BarrierMode)
                    if ($EarlyIndirectProbe) {
                        if (!$CommandListBackend) { throw '-EarlyIndirectProbe requires -CommandListBackend' }
                        $ngxArguments += @('--early-indirect', '1')
                    }
                    if ($CommandListBackend) { $ngxArguments += @('--interop-mode', 'command-list') }
                }
                $ngxName = if ($NgxMode -eq 'evaluate') { "ngx-evaluate-preset-$Preset" } elseif ($NgxMode -eq 'd3d12') { "d3d12-evaluate-preset-$Preset" } else { 'ngx-init' }
                $ngxOk = Invoke-Probe $ngxName $ngxExe $ngxArguments
                if ($OptiScalerDll) {
                    Get-ChildItem -LiteralPath $ngxRuntimeDirectory -Filter '*.log' -File |
                        ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $OutputDirectory ('optiscaler-' + $_.Name)) }
                    Copy-Item -LiteralPath (Join-Path $ngxRuntimeDirectory 'OptiScaler.ini') -Destination (Join-Path $OutputDirectory 'OptiScaler.ini')
                }
                if ($ngxOk -and $RequireNativeNetwork) {
                    $hits = @()
                    $nativeLayers = if ($Preset -eq 11) { @('enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0') }
                                    else { @('enc1','enc2','enc3_tube','dec2','dec1') }
                    foreach ($layer in $nativeLayers) {
                        $name = if ($Preset -eq 11) { "dltss_pwin_${layer}_layer" } else { "rrlite_${layer}_4x4" }
                        $pattern = '\[d4r-launch\] kernel="' + $name + '" backend=native'
                        $count = @(Select-String -LiteralPath (Join-Path $OutputDirectory "$ngxName.stderr.log") -Pattern $pattern).Count
                        $hits += @{layer=$layer; nativeLaunches=$count; expectedFrames=$Iterations}
                        if ($count -lt $Iterations) { $ngxOk=$false }
                    }
                    $summary.nativeTransformer = $hits
                    $summary.nativeTransformerPassed = $ngxOk
                    Write-Host "Native preset $Preset transformer launches validated=$ngxOk"
                }
                if (!$ngxOk) {
                    $exitStatus = 1
                    if (!$Trace -and (Test-Path (Join-Path $ZludaRoot 'trace/nvcuda.dll'))) {
                        $traceArguments = @(
                            '--hip-root', $HipRoot, '--cuda-dll', (Join-Path $ZludaRoot 'trace/nvcuda.dll'),
                            '--ngx-core', $localCore, '--dlss-dll', $localDlss,
                            '--nvapi-dll', (Join-Path $PackageRoot 'nvapi-compat/nvapi64.dll'),
                            '--ngx-mode', $NgxMode, '--preset', "$Preset", '--iterations', '1')
                        if ($NgxMode -eq 'd3d12') { $traceArguments += @('--module', $localShim) }
                        Invoke-Probe "$ngxName-trace" $ngxExe $traceArguments | Out-Null
                    }
                }
            }
        }
    }
} catch {
    $summary.error = $_.Exception.Message
    Write-Warning $summary.error
} finally {
    if ($ngxRuntimeDirectory -and (Test-Path -LiteralPath $ngxRuntimeDirectory)) {
        $resolvedRuntime = [IO.Path]::GetFullPath($ngxRuntimeDirectory)
        $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
        if (!$resolvedRuntime.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or
            [IO.Path]::GetFileName($resolvedRuntime) -notmatch '^d4r-ngx-[0-9a-f]{32}$') {
            throw 'Refusing to remove a path outside the private NGX temporary runtime.'
        }
        Remove-Item -LiteralPath $resolvedRuntime -Recurse -Force
    }
    $summary.passed = $exitStatus -eq 0
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
    $env:PATH = $originalPath
    $env:HIP_PATH = $originalHip
    $env:D4R_DIAG_DIR = $originalDiag
    $env:ZLUDA_LOG_DIR = $originalLog
    $env:ZLUDA_CUDA_LIB = $originalCuda
    $env:D4R_NVAPI_BACKEND = $originalNvapi
    $env:PYTHONPATH = $originalPythonPath
    $env:D4R_ZLUDA_VERBOSE = $originalVerbose
    $env:ZLUDA_CACHE_DIR = $originalCache
    foreach ($setting in $originalCodegen.Keys) { [Environment]::SetEnvironmentVariable($setting, $originalCodegen[$setting], 'Process') }
    $archive = "$OutputDirectory.zip"
    Compress-Archive -LiteralPath $OutputDirectory -DestinationPath $archive -Force
    Write-Host "Diagnostic bundle: $archive"
}
exit $exitStatus
