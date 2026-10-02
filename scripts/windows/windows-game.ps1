[CmdletBinding()]
param(
    [ValidateSet('install','restore','run')][string]$Action = 'run',
    [Parameter(Mandatory=$true)][string]$GameExe,
    [string]$NgxCore, [string]$DlssDll,
    [ValidateSet(11,13)][int]$Preset = 11,
    [int]$RunSeconds = 0,
    [string]$DiagnosticResolution,
    [string]$CacheDirectory = $env:ZLUDA_CACHE_DIR,
    [string]$LocalTextureKernels,
    [switch]$ValidateOutput,
    [switch]$ProfileStages,
    [switch]$ProfileKernels,
    [switch]$ProfileKernelsDeferred,
    [switch]$ProfileLegacyStream,
    [ValidateRange(1,1000000)][int]$KernelProfileEvery = 17,
    [switch]$ProfileCommandHooks,
    [switch]$ProfileCudaApi,
    [switch]$ProfileGpuBoundary,
    [switch]$UncachedInteropLists,
    [switch]$AsyncInterop,
    [switch]$BatchInputCopies,
    [switch]$VerboseRuntime,
    [switch]$CaptureExceptions,
    [string[]]$GameArguments = @('-dx12'),
    [string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
. (Join-Path $PSScriptRoot 'gpu-preflight.ps1')
if ($ProfileGpuBoundary -and !$AsyncInterop) { throw '-ProfileGpuBoundary requires -AsyncInterop.' }
if ($ProfileKernels -and $ProfileKernelsDeferred) { throw 'Choose either serializing or deferred kernel profiling.' }
if ($ProfileLegacyStream -and !$ProfileKernelsDeferred) { throw '-ProfileLegacyStream requires -ProfileKernelsDeferred.' }
$package = [IO.Path]::GetFullPath($PSScriptRoot)
$GameExe = (Get-Item -LiteralPath $GameExe -ErrorAction Stop).FullName
$game = Split-Path $GameExe
$backup = Join-Path $game '.d4r-backup'
$manifestPath = Join-Path $backup 'manifest.json'
function GamePath([string]$relative) {
    $path = [IO.Path]::GetFullPath((Join-Path $game $relative))
    if (!$path.StartsWith($game + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Path escapes game directory: $relative" }
    return $path
}
if (Get-Process -Name ([IO.Path]::GetFileNameWithoutExtension($GameExe)) -ErrorAction SilentlyContinue) { throw 'Close the game before installing, restoring or running this diagnostic.' }
if ($Action -eq 'restore') {
    if (!(Test-Path -LiteralPath $manifestPath)) { throw 'No d4r installation manifest exists.' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    foreach ($entry in $manifest.files) {
        $path = GamePath $entry.path
        if ((Test-Path -LiteralPath $path) -and (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.installedHash) { throw "File changed since installation; preserved: $path" }
    }
    foreach ($entry in $manifest.files) {
        $path = GamePath $entry.path
        if ($entry.existed) { Copy-Item -LiteralPath (Join-Path $backup ('original/' + $entry.path)) -Destination $path -Force }
        elseif (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    }
    # Remove only empty directories; preserve user files and diagnostic logs.
    if (Test-Path -LiteralPath (GamePath 'd4r')) {
        Get-ChildItem -LiteralPath (GamePath 'd4r') -Directory -Recurse | Sort-Object { $_.FullName.Length } -Descending | ForEach-Object {
            if (!(Get-ChildItem -LiteralPath $_.FullName -Force)) { Remove-Item -LiteralPath $_.FullName }
        }
        if (!(Get-ChildItem -LiteralPath (GamePath 'd4r') -Force)) { Remove-Item -LiteralPath (GamePath 'd4r') }
    }
    Move-Item -LiteralPath $manifestPath -Destination (Join-Path $backup ('restored-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.json'))
    Write-Host "Original game files restored. Backups retained: $backup"
    return
}
if (!$NgxCore -or !$DlssDll) { throw 'Pass both local NVIDIA DLL paths with -NgxCore and -DlssDll.' }
$NgxCore = (Get-Item -LiteralPath $NgxCore).FullName
$DlssDll = (Get-Item -LiteralPath $DlssDll).FullName
$metadata = Get-Content -LiteralPath (Join-Path $package 'package.json') -Raw | ConvertFrom-Json
$GpuArch=Get-D4RGpuTarget $package $metadata.architecture
foreach ($file in $metadata.files) {
    if ($file.path.EndsWith('.hsaco')) { Assert-D4RCodeObjectTarget (Join-Path $package $file.path) $GpuArch }
    if ((Get-FileHash -LiteralPath (Join-Path $package $file.path) -Algorithm SHA256).Hash -ne $file.sha256) { throw "Package file changed; rebuild package: $($file.path)" }
}
if ((Get-FileHash -LiteralPath $DlssDll -Algorithm SHA256).Hash -ne $metadata.dlssSha256) { throw 'DLSS DLL does not match the validated 310.9.1 native manifest. Rebuild and validate native manifests for this DLL first.' }
Invoke-D4RGpuPreflight $package $game $GpuArch (Join-Path $package ('results/preflight-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')))
$textureFiles=@(); $textureManifest=@()
if ($LocalTextureKernels) {
    $LocalTextureKernels=(Get-Item -LiteralPath $LocalTextureKernels -ErrorAction Stop).FullName
    $validation=Get-Content -LiteralPath (Join-Path $LocalTextureKernels 'validation.json') -Raw | ConvertFrom-Json
    if (!$validation.passed -or !$validation.strictRgb -or $validation.architecture -ne $GpuArch -or
        $validation.source.dlssSha256 -ne $metadata.dlssSha256 -or !$validation.source.accuracy) {
        throw 'Local texture kernels must pass test-native-texture.ps1 with the exact packaged DLSS identity.'
    }
    foreach ($object in $validation.source.objects.PSObject.Properties) {
        if ($object.Name -notin @('hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel.hsaco','hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel.hsaco')) {
            throw "Unvalidated local texture variant: $($object.Name)"
        }
        $path=Join-Path $LocalTextureKernels $object.Name
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $object.Value) { throw 'Local texture object changed after validation.' }
        Assert-D4RCodeObjectTarget $path $GpuArch
        $textureFiles+=Get-Item -LiteralPath $path
    }
    if ($textureFiles.Count -ne 2) { throw 'Expected both validated local K output variants.' }
    # Use the manifest covered by validation, rather than a mutable extra file.
    foreach ($line in Get-Content -LiteralPath (Join-Path $LocalTextureKernels 'd4r-kernels.txt')) {
        if (!$line.Trim() -or $line.StartsWith('#')) { continue }
        if ($line -notmatch '^([A-Za-z0-9_]+) ([0-9a-f]{16})$' -or ($Matches[1]+'.hsaco') -notin @($textureFiles.Name)) { throw 'Invalid local texture manifest.' }
        $textureManifest+=$line
    }
    if ($textureManifest.Count -ne 2) { throw 'Expected two local texture manifest identities.' }
    if ((Get-FileHash -LiteralPath (Join-Path $LocalTextureKernels 'd4r-kernels.txt') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $validation.manifestSha256) { throw 'Local texture manifest changed after validation.' }
}
$manifest = if (Test-Path -LiteralPath $manifestPath) { Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json } else { [pscustomobject]@{game=$GameExe; files=@()} }
if ($manifest.game -ne $GameExe) { throw 'Backup manifest belongs to another executable.' }
if (!(Test-Path -LiteralPath $manifestPath) -and (Test-Path -LiteralPath (GamePath 'd4r'))) { throw 'An existing d4r directory has no installation manifest; preserved.' }
New-Item -ItemType Directory -Force $backup | Out-Null
function InstallFile([string]$source, [string]$relative) {
    $relative=$relative.Replace('/','\')
    $path = GamePath $relative
    $entry = @($manifest.files | Where-Object { $_.path.Replace('/','\') -eq $relative }) | Select-Object -First 1
    if ($entry) {
        if ((Test-Path -LiteralPath $path) -and (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.installedHash) { throw "Installed file changed; preserved: $path" }
    } else {
        $existed = Test-Path -LiteralPath $path
        if ($existed) {
            $original = Join-Path $backup ('original/' + $relative)
            New-Item -ItemType Directory -Force (Split-Path $original) | Out-Null
            Copy-Item -LiteralPath $path -Destination $original
        }
        $entry = [pscustomobject]@{path=$relative; existed=[bool]$existed; installedHash=''}
        $manifest.files = @($manifest.files) + @($entry)
    }
    New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
    Copy-Item -LiteralPath $source -Destination $path -Force
    $entry.installedHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
}
InstallFile (Join-Path $package 'OptiScaler.dll') 'dxgi.dll'
foreach ($file in $metadata.files) {
    $relative=$file.path.Replace('/','\')
    if ($relative.StartsWith('d4r\',[StringComparison]::OrdinalIgnoreCase)) {
        InstallFile (Join-Path $package $relative) $relative
    }
}
if ($LocalTextureKernels) {
    foreach ($file in $textureFiles) { InstallFile $file.FullName ('d4r/native/' + $file.Name) }
    $combinedManifest=Join-Path $backup 'generated-native-manifest.txt'
    @(Get-Content -LiteralPath (Join-Path $package 'd4r/native/d4r-kernels.txt')) + $textureManifest |
        Set-Content -LiteralPath $combinedManifest -Encoding ASCII
    InstallFile $combinedManifest 'd4r/native/d4r-kernels.txt'
}
# Private local test copies only; these are never added to the public package.
InstallFile $NgxCore 'd4r/vendor/_nvngx.dll'
InstallFile $DlssDll 'd4r/vendor/nvngx_dlss.dll'
$bridge = GamePath 'd4r'
$vendor = GamePath 'd4r/vendor'
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
NvngxPath=$bridge
NvngxDlssPath=$vendor\nvngx_dlss.dll
NvngxFeaturePath=$vendor
[Hotfix]
RestoreComputeSignature=false
RestoreGraphicSignature=false
ExtendedStateRestore=false
[Log]
LogToFile=true
LogLevel=$(if ($VerboseRuntime) { 0 } else { 2 })
[Menu]
DisableSplash=true
"@
$tempIni = Join-Path $backup 'generated-OptiScaler.ini'
$ini | Set-Content -LiteralPath $tempIni -Encoding UTF8
InstallFile $tempIni 'OptiScaler.ini'
if ($Action -eq 'install') { Write-Host "Installed development backend for preset $Preset. Use -Action restore to revert."; return }
if (!$OutputDirectory) { $OutputDirectory = Join-Path $package ('results/game-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
if (!$CacheDirectory) { $CacheDirectory = Join-Path $package 'cache' }
$CacheDirectory = [IO.Path]::GetFullPath($CacheDirectory)
$settings = @{
    D4R_HIP_ROOT=(GamePath 'd4r/hip'); HIP_PATH=(GamePath 'd4r/hip');
    D4R_NVCUDA_DLL=(GamePath 'd4r/zluda/nvcuda.dll'); ZLUDA_CUDA_LIB=(GamePath 'd4r/zluda/nvcuda.dll');
    D4R_NVAPI_DLL=(GamePath 'd4r/nvapi/nvapi64.dll'); D4R_NVAPI_BACKEND=(GamePath 'd4r/zluda/nvapi64.dll');
    D4R_NGX_CORE=(GamePath 'd4r/vendor/_nvngx.dll'); D4R_DLSS_DLL=(GamePath 'd4r/vendor/nvngx_dlss.dll');
    D4R_FORMAT_MODULE=(GamePath "d4r/pixel_convert_${GpuArch}.hsaco");
    D4R_ZLUDA_NATIVE_DIR=(GamePath 'd4r/native'); D4R_D3D12_COMMAND_BACKEND='1'; D4R_ZLUDA_VERBOSE='1';
    D4R_ZLUDA_WMMA='1'; D4R_ZLUDA_WMMA_FP8='1'; D4R_ZLUDA_WMMA_FP8_NATIVE='0'; D4R_ZLUDA_WMMA_F16_REFERENCE='1';
    # Do not inherit relaxation flags from unrelated experiments. Unset and
    # false have the same numerical effect; unset reuses the validated cache.
    D4R_ZLUDA_IGNORE_DENORMAL=$null; D4R_ZLUDA_FAST_MATH=$null;
    D4R_ZLUDA_WMMA_F32ACC=$null; D4R_ZLUDA_WAVE64=$null;
    D4R_ZLUDA_WMMA_LAYOUT=$null;
    D4R_QUIET_API=$(if ($VerboseRuntime) { $null } else { '1' });
    D4R_VALIDATE_OUTPUT=$(if ($ValidateOutput) { '1' } else { $null });
    D4R_PROFILE_STAGES=$(if ($ProfileStages) { '1' } else { $null });
    D4R_ZLUDA_PROFILE=$(if ($ProfileKernels) { '1' } else { $null });
    D4R_ZLUDA_PROFILE_DEFERRED=$(if ($ProfileKernelsDeferred) { '1' } else { $null });
    D4R_ZLUDA_PROFILE_ALLOW_LEGACY=$(if ($ProfileLegacyStream) { '1' } else { $null });
    D4R_ZLUDA_PROFILE_EVERY=$(if ($ProfileKernelsDeferred) { "$KernelProfileEvery" } else { $null });
    D4R_PROFILE_COMMAND_HOOKS=$(if ($ProfileCommandHooks) { '1' } else { $null });
    D4R_ZLUDA_PROFILE_API=$(if ($ProfileCudaApi) { '1' } else { $null });
    D4R_PROFILE_GPU_BOUNDARY=$(if ($ProfileGpuBoundary) { '1' } else { $null });
    D4R_DISABLE_INTEROP_LIST_CACHE=$(if ($UncachedInteropLists) { '1' } else { $null });
    D4R_ASYNC_INTEROP=$(if ($AsyncInterop) { '1' } else { $null });
    D4R_BATCH_INPUT_COPIES=$(if ($BatchInputCopies) { '1' } else { $null });
    D4R_DIAG_DIR=$OutputDirectory; ZLUDA_LOG_DIR=(Join-Path $OutputDirectory 'zluda-trace');
    ZLUDA_CACHE_DIR=$CacheDirectory; PATH=((GamePath 'd4r/hip/bin') + ';' + (GamePath 'd4r/zluda') + ';' + $env:PATH)
}
$old = @{}; $process = $null; $originalSettings = @()
$outFile = $null; $errFile = $null; $outTask = $null; $errTask = $null
try {
    if ($DiagnosticResolution) {
        if ([IO.Path]::GetFileName($GameExe) -ne 'SHProto-Win64-Shipping.exe' -or $DiagnosticResolution -notmatch '^([0-9]{3,4})x([0-9]{3,4})$') { throw 'DiagnosticResolution currently supports Silent Hill 2 only; use e.g. 1280x720.' }
        $renderWidth=[int]$Matches[1]; $renderHeight=[int]$Matches[2]
        if ($renderWidth -gt 4096 -or $renderHeight -gt 2160) { throw 'Diagnostic resolution is too large.' }
        $userRoot=Join-Path $env:LOCALAPPDATA 'SilentHill2/Saved'
        foreach ($relative in @('Config/Windows/GameUserSettings.ini','SaveGames/GFXSettings.SHProto-Win64-Shipping.xml')) {
            $path=Join-Path $userRoot $relative
            $saved=Join-Path $OutputDirectory ('original-settings/' + $relative)
            New-Item -ItemType Directory -Force (Split-Path $saved) | Out-Null
            Copy-Item -LiteralPath $path -Destination $saved
            $originalSettings += @{path=$path; saved=$saved}
            if ($relative.EndsWith('.ini')) {
                $text=[IO.File]::ReadAllText($path)
                $text=$text -replace 'Resolution=\(X=[0-9]+,Y=[0-9]+\)', "Resolution=(X=$renderWidth,Y=$renderHeight)"
                $text=$text -replace 'ScreenMode=Borderless', 'ScreenMode=Windowed'
                $text=$text -replace '(?m)^ResolutionSizeX=[0-9]+', "ResolutionSizeX=$renderWidth"
                $text=$text -replace '(?m)^ResolutionSizeY=[0-9]+', "ResolutionSizeY=$renderHeight"
                $text=$text -replace '(?m)^FullscreenMode=[0-9]+', 'FullscreenMode=2'
                [IO.File]::WriteAllText($path,$text,[Text.UTF8Encoding]::new($false))
            } else {
                $xml=[xml](Get-Content -LiteralPath $path -Raw -Encoding Unicode)
                $xml.GSA_SDK.GAMESETTINGS.RESOLUTION.Width="$renderWidth"
                $xml.GSA_SDK.GAMESETTINGS.RESOLUTION.Height="$renderHeight"
                $xml.Save($path)
            }
        }
        Write-Host "Temporary diagnostic resolution: $DiagnosticResolution; original settings restore at exit."
    }
    foreach ($key in $settings.Keys) { $old[$key] = [Environment]::GetEnvironmentVariable($key,'Process'); [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process') }
    @{preset=$Preset; game=$GameExe; package=$metadata; attachedDebugger=[bool]$CaptureExceptions;
      diagnostics=@{asyncInterop=[bool]$AsyncInterop; batchInputCopies=[bool]$BatchInputCopies; validateOutput=[bool]$ValidateOutput;
        profileGpuBoundary=[bool]$ProfileGpuBoundary; profileStages=[bool]$ProfileStages; profileCudaApi=[bool]$ProfileCudaApi;
        profileKernels=[bool]$ProfileKernels; profileKernelsDeferred=[bool]$ProfileKernelsDeferred;
        profileLegacyStream=[bool]$ProfileLegacyStream; kernelProfileEvery=$KernelProfileEvery};
      driver=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate); arguments=$GameArguments} |
        ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'environment.json') -Encoding UTF8
    function Quote([string]$value) { if ($value.Contains('"')) { throw 'A command argument contains an unsupported quote.' }; return '"' + $value + '"' }
    $arguments = @('--output-directory', (Quote $OutputDirectory))
    if (!$CaptureExceptions) { $arguments += '--no-debug' }
    $arguments += @('--', (Quote $GameExe)) + @($GameArguments | ForEach-Object { Quote $_ })
    # Drain raw bytes concurrently. PowerShell's line-oriented redirection can
    # throttle a verbose game and distort the very timings we are collecting.
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = GamePath 'd4r/d4r_debug_launcher.exe'
    $info.Arguments = $arguments -join ' '
    $info.WorkingDirectory = $game
    $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    # Empty task switches mean unset. Some PowerShell/.NET hosts preserve empty
    # values, which would enable Rust's presence-based diagnostic flags.
    foreach ($key in @($info.EnvironmentVariables.Keys)) {
        if ($key.StartsWith('D4R_') -and [string]::IsNullOrWhiteSpace($info.EnvironmentVariables[$key])) {
            $info.EnvironmentVariables.Remove($key)
        }
    }
    if ([Environment]::GetEnvironmentVariable('D4R_ZLUDA_WGP','Process') -ne '1') {
        $info.EnvironmentVariables.Remove('D4R_ZLUDA_WGP')
    }
    $process = [Diagnostics.Process]::new(); $process.StartInfo = $info
    if (!$process.Start()) { throw 'Game diagnostic launch failed.' }
    $outFile = [IO.File]::Open((Join-Path $OutputDirectory 'd4r.stdout.log'), [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $errFile = [IO.File]::Open((Join-Path $OutputDirectory 'd4r.stderr.log'), [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $outTask = $process.StandardOutput.BaseStream.CopyToAsync($outFile)
    $errTask = $process.StandardError.BaseStream.CopyToAsync($errFile)
    Write-Host "Game diagnostic started (preset $Preset). Logs: $OutputDirectory"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while (!$process.WaitForExit(1000)) {
        if ($RunSeconds -gt 0 -and $timer.Elapsed.TotalSeconds -ge $RunSeconds) { $process.Kill(); $process.WaitForExit(); break }
    }
    $process.WaitForExit()
    [void]$outTask.GetAwaiter().GetResult(); [void]$errTask.GetAwaiter().GetResult()
    $outFile.Dispose(); $outFile = $null; $errFile.Dispose(); $errFile = $null
    foreach ($file in Get-ChildItem -LiteralPath $game -Filter 'OptiScaler*.log' -File) { Copy-Item -LiteralPath $file.FullName -Destination $OutputDirectory -Force }
    $stderr = Get-Content -LiteralPath (Join-Path $OutputDirectory 'd4r.stderr.log') -Raw
    $stdout = Get-Content -LiteralPath (Join-Path $OutputDirectory 'd4r.stdout.log') -Raw
    $debugger = Get-Content -LiteralPath (Join-Path $OutputDirectory 'debugger.log') -Raw
    $gameExit = [regex]::Match($debugger, 'EXIT code=0x([0-9a-f]+)')
    $gameCrash = $null
    if ([IO.Path]::GetFileName($GameExe) -eq 'SHProto-Win64-Shipping.exe' -and $gameExit.Success -and $gameExit.Groups[1].Value -ne '0') {
        $crashRoot = Join-Path $env:LOCALAPPDATA 'SilentHill2/Saved/Crashes'
        $latest = Get-ChildItem -LiteralPath $crashRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.LastWriteTime -ge (Get-Date).AddSeconds(-$timer.Elapsed.TotalSeconds-5) } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
            Copy-Item -LiteralPath $latest.FullName -Destination (Join-Path $OutputDirectory 'ue-crash') -Recurse
            [xml]$context = Get-Content -LiteralPath (Join-Path $latest.FullName 'CrashContext.runtime-xml') -Raw
            $gameCrash = @{type=[string]$context.FGenericCrashContext.RuntimeProperties.CrashType;
                reason=[string]$context.FGenericCrashContext.RuntimeProperties.ErrorMessage}
        }
    }
    # A bounded stop can interrupt the last printf; count only complete records.
    $frames = [regex]::Matches($stdout, '(?m)^D4R_FRAME cpu_frame_copies=([0-9]+) frame_age=([0-9]+) interop_ngx_ms=([0-9.]+)\r?$')
    $checks = [regex]::Matches($stdout, '(?m)^D4R_OUTPUT_VALIDATION elements=([0-9]+) nan=([0-9]+) inf=([0-9]+) diagnostics_cpu_bytes=8\r?$')
    $nonfinite = @($checks | Where-Object { $_.Groups[2].Value -ne '0' -or $_.Groups[3].Value -ne '0' }).Count
    $launches = [regex]::Matches($stderr, '\[d4r-launch\] kernel="([^"]+)" backend=(native|translated)')
    $kernels = @($launches | ForEach-Object { $_.Groups[1].Value + ' ' + $_.Groups[2].Value } | Group-Object | ForEach-Object {
        $parts=$_.Name.Split(' '); @{kernel=$parts[0]; backend=$parts[1]; launches=$_.Count}
    })
    @{exitCode=$(if ($gameExit.Success) { '0x' + $gameExit.Groups[1].Value } else { 'diagnostic_timeout' }); runSeconds=$timer.Elapsed.TotalSeconds; preset=$Preset; gameCrash=$gameCrash;
        nativeLaunches=([regex]::Matches($stderr, '\[d4r-launch\].*backend=native')).Count;
        completedFrames=$frames.Count; outputValidationRequested=[bool]$ValidateOutput;
        outputGpuChecks=$checks.Count; nonfiniteOutputs=$nonfinite; kernels=$kernels;
        previousFrameOutputs=@($frames | Where-Object { $_.Groups[2].Value -ne '0' }).Count;
        cpuImageCopyFrames=@($frames | Where-Object { $_.Groups[1].Value -ne '0' }).Count;
        failures=([regex]::Matches($stderr, 'D4R_[A-Z0-9_]*FAILURE\b')).Count} |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
    Compress-Archive -LiteralPath $OutputDirectory -DestinationPath ($OutputDirectory + '.zip') -Force
    Write-Host "Game diagnostic bundle: $OutputDirectory.zip"
} finally {
    if ($process -and !$process.HasExited) { $process.Kill(); $process.WaitForExit() }
    if ($outTask) { try { [void]$outTask.GetAwaiter().GetResult() } catch {} }
    if ($errTask) { try { [void]$errTask.GetAwaiter().GetResult() } catch {} }
    if ($outFile) { $outFile.Dispose() }; if ($errFile) { $errFile.Dispose() }
    if ($process) { $process.Dispose() }
    foreach ($key in $old.Keys) { [Environment]::SetEnvironmentVariable($key,$old[$key],'Process') }
    foreach ($entry in $originalSettings) { Copy-Item -LiteralPath $entry.saved -Destination $entry.path -Force }
}
