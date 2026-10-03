[CmdletBinding()]
param(
    [ValidateSet('run','install','restore')][string]$Action='run',
    [ValidateSet(11,13)][int]$Preset=11,
    [string]$GameExe, [string]$NgxCore, [string]$DlssDll,
    [switch]$CaptureExceptions
)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$work=Join-Path $root 'work'
$settingsPath=Join-Path $work 'settings.json'
New-Item -ItemType Directory -Force $work | Out-Null
$saved=$null
if (Test-Path -LiteralPath $settingsPath) {
    try { $saved=Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json }
    catch { Write-Host 'Saved choices could not be read. Select the files again.' }
}
function Select-LocalFile([string]$Title, [string]$FileName, [string]$Previous) {
    if ($Previous -and (Test-Path -LiteralPath $Previous -PathType Leaf)) { return (Get-Item -LiteralPath $Previous).FullName }
    Add-Type -AssemblyName System.Windows.Forms
    $dialog=New-Object Windows.Forms.OpenFileDialog
    try {
        $dialog.Title=$Title; $dialog.FileName=$FileName
        $dialog.Filter=if ($FileName -eq '*.exe') { 'Game executable (*.exe)|*.exe' } else { "$FileName|$FileName" }
        if ($dialog.ShowDialog() -ne [Windows.Forms.DialogResult]::OK) { throw 'File selection cancelled. No test was launched.' }
        return $dialog.FileName
    } finally { $dialog.Dispose() }
}
function Copy-Payload([string]$Source, [string]$Destination, [string]$Relative) {
    $path=[IO.Path]::GetFullPath((Join-Path $Destination $Relative))
    if (!$path.StartsWith($Destination+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid package path.' }
    New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
    Copy-Item -LiteralPath (Join-Path $Source $Relative) -Destination $path -Force
}
function Get-BundledNvidiaFile([string]$Relative) {
    $directory=Join-Path $PSScriptRoot 'nvidia'
    $path=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Relative))
    if (!$path.StartsWith($directory+'\',[StringComparison]::OrdinalIgnoreCase) -or
        !(Test-Path -LiteralPath $path -PathType Leaf)) { throw 'Invalid bundled NVIDIA file. Extract the complete package again.' }
    return $path
}
if ($Action -eq 'restore') {
    $GameExe=Select-LocalFile 'Select the SAME game executable used for the test' '*.exe' $(if ($GameExe) { $GameExe } elseif ($saved) { $saved.game })
    & (Join-Path $PSScriptRoot 'common/windows-game.ps1') -Action restore -GameExe $GameExe
    Write-Host 'RESTORED. Launch the game normally again.' -ForegroundColor Green
    return
}
$output=Join-Path $root ('results/test-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Force $output | Out-Null
$status=[ordered]@{action=$Action; preset=$Preset; status='starting'; hardwareValidationCompleted=$false}
$selectedPackage=$null
try {
    Write-Host 'd4r Windows quick test - keep this window open until you EXIT the game.'
    Write-Host 'Checking package and detecting the GPU...'
    $manifest=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'package.json') -Raw | ConvertFrom-Json
    foreach ($file in $manifest.files) {
        $path=[IO.Path]::GetFullPath((Join-Path $root $file.path))
        if (!$path.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase) -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $file.sha256) { throw "Package is incomplete or changed. Extract the ZIP again. File: $($file.path)" }
    }
    $status.packageCommit=$manifest.packageCommit
    $inventoryDir=Join-Path $output 'gpu'
    New-Item -ItemType Directory -Force $inventoryDir | Out-Null
    $info=New-Object Diagnostics.ProcessStartInfo
    $info.FileName=Join-Path $PSScriptRoot 'targets/gfx1201/d4r/d4r_gpu_inventory.exe'
    $info.Arguments='--hip-root "'+(Join-Path $PSScriptRoot 'common/d4r/hip')+'" --output-dir "'+$inventoryDir+'"'
    $info.UseShellExecute=$false; $info.CreateNoWindow=$true
    $info.EnvironmentVariables['D4R_DIAG_DIR']=$inventoryDir
    $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
    $process=New-Object Diagnostics.Process; $process.StartInfo=$info
    try {
        if (!$process.Start()) { throw 'GPU detection could not start.' }
        $stdout=$process.StandardOutput.ReadToEndAsync(); $stderr=$process.StandardError.ReadToEndAsync()
        $timedOut=!$process.WaitForExit(30000)
        if ($timedOut) { $process.Kill(); $process.WaitForExit() }
        $stdout.GetAwaiter().GetResult() | Set-Content (Join-Path $inventoryDir 'stdout.log') -Encoding UTF8
        $stderr.GetAwaiter().GetResult() | Set-Content (Join-Path $inventoryDir 'stderr.log') -Encoding UTF8
        $status.inventoryExit='0x{0:X8}' -f [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$process.ExitCode),0)
        if ($timedOut -or $process.ExitCode) { throw 'HIP could not detect your GPU. The diagnostic ZIP contains the details.' }
    } finally { $process.Dispose() }
    $inventory=Get-Content -LiteralPath (Join-Path $inventoryDir 'gpu-inventory.json') -Raw | ConvertFrom-Json
    $devices=@($inventory.devices | Where-Object { $_.buildEligible -and $_.wavefrontSize -eq 32 -and $manifest.targets.PSObject.Properties.Name -contains $_.architecture } | Sort-Object vramBytes -Descending)
    if (!$devices.Count) { throw 'No supported RDNA3/RDNA4 HIP device was detected. See the GPU inventory in the diagnostic ZIP.' }
    $device=$devices[0]
    if (@($devices.architecture | Select-Object -Unique).Count -gt 1) {
        for ($i=0; $i -lt $devices.Count; $i++) { Write-Host "$($i+1): $($devices[$i].name) ($($devices[$i].architecture))" }
        $choice=Read-Host 'Select the GPU used by your game (press Enter for 1)'
        if ($choice) {
            $index=0
            if (![int]::TryParse($choice,[ref]$index) -or $index -lt 1 -or $index -gt $devices.Count) { throw 'Invalid GPU selection.' }
            $device=$devices[$index-1]
        }
    }
    $arch=$device.architecture; $target=$manifest.targets.$arch
    $status.gpu=$device; $status.binarySourceCommit=$target.binarySourceCommit
    Write-Host "Detected: $($device.name) / $arch" -ForegroundColor Cyan
    Write-Host 'Experimental build. This run is a correctness test, not an FPS benchmark.'
    $previousGame=if ($saved) { $saved.game } else { $null }
    if (!$GameExe -and $previousGame -and (Test-Path -LiteralPath $previousGame)) {
        Write-Host "Remembered game: $previousGame"
        $reuse=Read-Host 'Press Enter to use this game, or type C to choose another (restore the old game first)'
        if ($reuse -match '^(?i)c$') { $previousGame=$null }
        elseif ($reuse) { throw 'Invalid choice. Press Enter or type C on the next run.' }
    }
    $GameExe=Select-LocalFile 'Select the actual D3D12 game .exe (not Steam or a launcher)' '*.exe' $(if ($GameExe) { $GameExe } else { $previousGame })
    $bundledCore=$null; $bundledDlss=$null
    if ($manifest.bundledNvidia) {
        $bundledCore=Get-BundledNvidiaFile $manifest.bundledNvidia.ngxCore
        $bundledDlss=Get-BundledNvidiaFile $manifest.bundledNvidia.dlssDll
    }
    $status.nvidiaDllSource=@{core=$(if ($NgxCore) { 'explicit' } elseif ($bundledCore) { 'bundled' } else { 'user-supplied' });
        dlss=$(if ($DlssDll) { 'explicit' } elseif ($bundledDlss) { 'bundled' } else { 'user-supplied' })}
    if ($bundledCore -and !$NgxCore -and !$DlssDll) { Write-Host 'Using the validated NVIDIA DLL pair included by this package distributor.' }
    $NgxCore=Select-LocalFile 'Select your ORIGINAL NVIDIA _nvngx.dll (32.0.16.1714)' '_nvngx.dll' $(if ($NgxCore) { $NgxCore } elseif ($bundledCore) { $bundledCore } elseif ($saved) { $saved.ngxCore })
    $DlssDll=Select-LocalFile 'Select your NVIDIA nvngx_dlss.dll (310.9.1)' 'nvngx_dlss.dll' $(if ($DlssDll) { $DlssDll } elseif ($bundledDlss) { $bundledDlss } elseif ($saved) { $saved.dlssDll })
    $status.game=$GameExe
    $status.localDlls=@($NgxCore,$DlssDll | ForEach-Object { $item=Get-Item -LiteralPath $_; @{path=$item.FullName; version=$item.VersionInfo.FileVersion; sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash} })
    if ((Get-FileHash -LiteralPath $NgxCore -Algorithm SHA256).Hash -ne $manifest.ngxCoreSha256) {
        throw 'Wrong _nvngx.dll. This test needs the original NVIDIA 32.0.16.1714 DLL, not the d4r shim. Select the correct file on the next run.'
    }
    if ((Get-FileHash -LiteralPath $DlssDll -Algorithm SHA256).Hash -ne $target.dlssSha256) {
        throw 'Wrong nvngx_dlss.dll. This test needs exactly NVIDIA DLSS 310.9.1. Select the correct file on the next run.'
    }
    if ($GameExe.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Select the installed GAME executable, outside this extracted package.' }
    # Save only validated choices. Wrong DLLs are never remembered.
    @{game=$GameExe; ngxCore=$NgxCore; dlssDll=$DlssDll; architecture=$arch} | ConvertTo-Json | Set-Content -LiteralPath $settingsPath -Encoding UTF8
    $selectedPackage=Join-Path $work "package-$arch"
    New-Item -ItemType Directory -Force $selectedPackage | Out-Null
    $selectedFiles=@()
    foreach ($file in $manifest.commonFiles) {
        Copy-Payload (Join-Path $PSScriptRoot 'common') $selectedPackage $file.path
        $selectedFiles+=$file
    }
    foreach ($file in $target.files) {
        Copy-Payload (Join-Path $PSScriptRoot "targets/$arch") $selectedPackage $file.path
        $selectedFiles+=$file
    }
    @{architecture=$arch; d4rCommit=$target.binarySourceCommit; packageCommit=$manifest.packageCommit;
      dlssSha256=$target.dlssSha256; hardwareValidation=$target.hardwareValidation; files=$selectedFiles} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $selectedPackage 'package.json') -Encoding UTF8
    Write-Host 'Original game files will be backed up. RESTORE-GAME.cmd reverses the installation.'
    Write-Host 'In the game: select DLSS if available; otherwise select FSR/XeSS to let OptiScaler intercept it.'
    Write-Host 'The first launch compiles shaders and can take several minutes. Then play for 30 seconds and EXIT the game normally.'
    & (Join-Path $selectedPackage 'windows-game.ps1') -Action $Action -GameExe $GameExe -NgxCore $NgxCore -DlssDll $DlssDll `
        -Preset $Preset -ValidateOutput -CaptureExceptions:$CaptureExceptions -OutputDirectory $output
    if ($Action -eq 'install') { $status.status='installed-only' }
    else {
        $summary=Get-Content -LiteralPath (Join-Path $output 'summary.json') -Raw | ConvertFrom-Json
        $status.gameSummary=$summary
        $required=Get-Content -LiteralPath (Join-Path $selectedPackage 'd4r/native/d4r-kernels.txt') | Where-Object {
            if ($Preset -eq 11) { $_ -match '^dltss_pwin_' } else { $_ -match '^rrlite_' }
        } | ForEach-Object { $_.Split(' ')[0] }
        $hits=@($summary.kernels | Where-Object { $_.backend -eq 'native' -and $_.launches -gt 0 } | ForEach-Object { $_.kernel })
        $status.missingNativeKernels=@($required | Where-Object { $hits -notcontains $_ })
        $status.translatedNativeKernels=@($summary.kernels | Where-Object { $_.backend -eq 'translated' -and $required -contains $_.kernel } | ForEach-Object { $_.kernel })
        $passed=$summary.exitCode -eq '0x0' -and $summary.completedFrames -gt 0 -and $summary.nativeLaunches -gt 0 -and
            $summary.outputGpuChecks -gt 0 -and $summary.failures -eq 0 -and $summary.nonfiniteOutputs -eq 0 -and
            $summary.previousFrameOutputs -eq 0 -and $summary.cpuImageCopyFrames -eq 0 -and
            @($required).Count -eq $(if ($Preset -eq 11) { 11 } else { 5 }) -and
            $status.missingNativeKernels.Count -eq 0 -and $status.translatedNativeKernels.Count -eq 0
        $status.hardwareValidationCompleted=[bool]$passed
        $status.status=if ($passed) { 'backend-checks-passed-image-still-needs-user-review' }
            elseif ($summary.completedFrames -eq 0 -and $summary.exitCode -eq '0x0' -and $summary.failures -eq 0) { 'inconclusive-process-exited-without-dlss-frames' }
            else { 'backend-checks-failed-or-no-dlss-frames' }
        $status.session=$summary.session
        if ($status.status -eq 'inconclusive-process-exited-without-dlss-frames') {
            Write-Host 'The launched process exited without a DLSS frame. A Steam/launcher restart is not tracked by this test. Keep Steam open and select the actual game executable. Send the ZIP if it restarts again.' -ForegroundColor Yellow
        }
        Write-Host $(if ($passed) { 'Backend checks PASSED. Please also report whether the image looked correct.' } else { 'Test did not pass. Send the diagnostic ZIP; do not guess from the FPS.' }) -ForegroundColor $(if ($passed) { 'Green' } else { 'Yellow' })
    }
} catch {
    $status.status='failed'; $status.error=$_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $output 'error.txt') -Encoding UTF8
    Write-Host "TEST STOPPED: $($status.error)" -ForegroundColor Red
} finally {
    if ($selectedPackage) {
        $preflights=Join-Path $selectedPackage 'results'
        $latest=Get-ChildItem -LiteralPath $preflights -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($latest) { Copy-Item -LiteralPath $latest.FullName -Destination (Join-Path $output 'preflight') -Recurse -Force }
    }
    try { Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'driver.json') -Encoding UTF8 } catch {}
    $status | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $output 'quick-test.json') -Encoding UTF8
    Compress-Archive -LiteralPath $output -DestinationPath ($output+'.zip') -Force
    Write-Host "SEND THIS ONE FILE: $output.zip" -ForegroundColor Cyan
    Write-Host 'Also tell us: GPU model, game name, K or M, and whether the image was correct.'
    Write-Host 'Close the game before running RESTORE-GAME.cmd.'
}
if ($status.status -eq 'failed' -or $status.status -eq 'backend-checks-failed-or-no-dlss-frames' -or $status.status.StartsWith('inconclusive-')) { exit 1 }
