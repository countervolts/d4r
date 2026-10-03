# Checks a d4r install on Windows. It only reads files and prints what it finds.
# Run it from the game folder (the one holding the game's .exe and d4r\), or give that folder:
#   powershell -ExecutionPolicy Bypass -File d4r\d4r-check.ps1 [GAME_FOLDER]
# Add -Report to also write one ZIP (logs, this output, GPU and driver) to send with a bug report.
[CmdletBinding()]
param(
    [string]$GameFolder,
    [switch]$Report,
    [string]$InventoryFile,   # tests: a gpu-inventory.json to use instead of running the GPU query
    [string]$ReportDirectory  # tests: where -Report writes its ZIP (default: your Desktop)
)
$ErrorActionPreference = 'Stop'
if (!$GameFolder) {
    $GameFolder = if ((Split-Path $PSScriptRoot -Leaf) -eq 'd4r') { Split-Path $PSScriptRoot -Parent } else { (Get-Location).Path }
}
$game = [IO.Path]::GetFullPath($GameFolder).TrimEnd('\')
$d4r = Join-Path $game 'd4r'
$problems = 0
$lines = [Collections.Generic.List[string]]::new()
function Say([string]$text) { $lines.Add($text); Write-Host $text }
function Ok([string]$text) { Say ('  ok       ' + $text) }
function Bad([string]$text) { Say ('  MISSING  ' + $text); $script:problems++ }
function Note([string]$text) { Say ('  note     ' + $text) }
function Hash([string]$path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
function Version([string]$path) {
    try { $v = (Get-Item -LiteralPath $path).VersionInfo; if ($v.FileVersion) { return ($v.FileVersion -replace ',\s*', '.') } } catch {}
    return 'unknown'
}
function Read-TextFile([string]$path) { [IO.File]::ReadAllText($path) }

Say "d4r install in $game"
if (!(Test-Path -LiteralPath $d4r -PathType Container)) {
    Say 'no d4r folder here; run this from the folder with the game .exe (extract the d4r ZIP there first)'
    exit 1
}
if (@(Get-ChildItem -LiteralPath $game -Filter '*.exe' -File -ErrorAction SilentlyContinue).Count) { Ok 'game executable next to d4r\' }
else { Note 'no .exe next to d4r\ (the d4r files must sit in the folder of the game''s main .exe; for Unreal Engine games that is <game>\<Project>\Binaries\Win64)' }

# --- the install's own file list ---------------------------------------------------------------
$manifest = $null
$manifestPath = Join-Path $d4r 'package.json'
if (Test-Path -LiteralPath $manifestPath) {
    try { $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json } catch { $manifest = $null }
}
if (!$manifest) { Bad 'd4r\package.json is missing or damaged (extract the d4r ZIP again)' }

# --- files --------------------------------------------------------------------------------------
$required = @('dxgi.dll', 'OptiScaler.ini', 'd4r\_nvngx.dll', 'd4r\d4r.ini', 'd4r\dll-pins.txt',
    'd4r\hip\bin\amdhip64_7.dll', 'd4r\zluda\nvcuda.dll', 'd4r\zluda\nvapi64.dll')
foreach ($file in $required) {
    if (Test-Path -LiteralPath (Join-Path $game $file) -PathType Leaf) { Ok $file }
    else { Bad "$file (extract the d4r ZIP again over the game folder)" }
}
if ($manifest) {
    $edited = @('OptiScaler.ini', 'd4r/d4r.ini')   # players are meant to edit these
    $damaged = 0
    foreach ($entry in $manifest.files) {
        $path = Join-Path $game ($entry.path -replace '/', '\')
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
            if ($required -notcontains ($entry.path -replace '/', '\')) { Bad "$($entry.path) (extract the d4r ZIP again)" }
            continue
        }
        if ((Hash $path) -ne $entry.sha256) {
            if ($entry.path -eq 'dxgi.dll') { Note 'dxgi.dll is not the one that came with d4r (another OptiScaler or ReShade?). Move it away and extract the d4r ZIP again' }
            elseif ($edited -contains $entry.path) { Note "$($entry.path) has been edited (fine if you did it)" }
            else { Bad "$($entry.path) differs from the release (damaged or replaced; extract the d4r ZIP again)"; $damaged++ }
        }
    }
    if (!$damaged) { Ok 'every other d4r file matches the release' }
}

# --- OptiScaler.ini ---------------------------------------------------------------------------
$optiIni = Join-Path $game 'OptiScaler.ini'
if (Test-Path -LiteralPath $optiIni) {
    $text = Read-TextFile $optiIni
    if ($text -notmatch '(?im)^\s*AllowExternalBackend\s*=\s*true') { Bad 'OptiScaler.ini: [DLSS] AllowExternalBackend must be true (this is not the d4r OptiScaler.ini)' }
    if ($text -notmatch '(?im)^\s*Dx12Upscaler\s*=\s*dlss') { Note 'OptiScaler.ini: [Upscalers] Dx12Upscaler is not dlss; the game will not use d4r' }
    $m = [regex]::Match($text, '(?im)^\s*NvngxPath\s*=\s*(\S+)')
    if (!$m.Success) { Bad 'OptiScaler.ini: [Libraries] NvngxPath is missing (it must name the d4r folder)' }
    else {
        $folder = $m.Groups[1].Value
        $resolved = if ([IO.Path]::IsPathRooted($folder)) { $folder } else { Join-Path $game $folder }
        if (Test-Path -LiteralPath (Join-Path $resolved '_nvngx.dll')) { Ok "OptiScaler.ini NvngxPath = $folder" }
        else { Bad "OptiScaler.ini NvngxPath = $folder does not hold _nvngx.dll (use NvngxPath=d4r, or the full path of the d4r folder)" }
    }
}

# --- NVIDIA DLLs (never shipped; the exact versions come from dll-pins.txt) -----------------------
$pins = @()
$pinsPath = Join-Path $d4r 'dll-pins.txt'
if (Test-Path -LiteralPath $pinsPath) {
    foreach ($line in Get-Content -LiteralPath $pinsPath) {
        $line = $line.Trim()
        if (!$line -or $line.StartsWith('#')) { continue }
        $f = $line -split '\s+'
        if ($f.Count -ge 4 -and $f[1].Length -eq 64) { $pins += [pscustomobject]@{ kind = $f[0]; sha = $f[1].ToUpperInvariant(); version = $f[2]; status = $f[3] } }
    }
}
function Check-Nvidia([string]$kind, [string]$relative, [string]$label, [string]$source) {
    $path = Join-Path $game $relative
    $wanted = @($pins | Where-Object { $_.kind -eq $kind } | ForEach-Object { $_.version }) -join ' or '
    if (!$wanted) { Note "$label cannot be checked (d4r\dll-pins.txt is missing or damaged)"; return }
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { Bad "${relative}: copy NVIDIA's $label here (version $wanted; $source)"; return }
    $hash = Hash $path
    $pin = @($pins | Where-Object { $_.kind -eq $kind -and $_.sha -eq $hash }) | Select-Object -First 1
    if (!$pin) { Bad "$relative is version $(Version $path), which d4r does not accept. d4r on Windows needs exactly $wanted ($source)"; return }
    if ($pin.status -eq 'validated') { Ok "$relative ($label $($pin.version), tested on Windows)" }
    else { Ok "$relative ($label $($pin.version))"; Note "$label $($pin.version) is accepted but has had less Windows testing" }
}
Check-Nvidia 'dlss' 'd4r\nvngx_dlss.dll' 'nvngx_dlss.dll' 'many games ship one in their folder; see WINDOWS_README.txt'
Check-Nvidia 'ngx' 'd4r\ngx\_nvngx.dll' '_nvngx.dll' 'it is part of NVIDIA''s display driver package; see WINDOWS_README.txt'

# --- GPU and the build for it --------------------------------------------------------------------
$registryPath = Join-Path $d4r 'gpu-targets.json'
$registry = @()
if (Test-Path -LiteralPath $registryPath) { $registry = @((Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json).targets) }
$inventory = $null
if ($InventoryFile) { $inventory = Get-Content -LiteralPath $InventoryFile -Raw | ConvertFrom-Json }
elseif (Test-Path -LiteralPath (Join-Path $d4r 'd4r_gpu_inventory.exe')) {
    $work = Join-Path ([IO.Path]::GetTempPath()) ('d4r-check-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $work | Out-Null
    try {
        $info = New-Object Diagnostics.ProcessStartInfo
        $info.FileName = Join-Path $d4r 'd4r_gpu_inventory.exe'
        $info.Arguments = '--hip-root "' + (Join-Path $d4r 'hip') + '" --output-dir "' + $work + '"'
        $info.UseShellExecute = $false; $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
        $info.EnvironmentVariables['D4R_DIAG_DIR'] = $work
        $process = New-Object Diagnostics.Process; $process.StartInfo = $info
        [void]$process.Start()
        $null = $process.StandardOutput.ReadToEndAsync(); $null = $process.StandardError.ReadToEndAsync()
        if (!$process.WaitForExit(30000)) { $process.Kill(); $process.WaitForExit() }
        if (Test-Path -LiteralPath (Join-Path $work 'gpu-inventory.json')) { $inventory = Get-Content -LiteralPath (Join-Path $work 'gpu-inventory.json') -Raw | ConvertFrom-Json }
    } catch {} finally {
        $process.Dispose()
        Get-ChildItem -LiteralPath $work -File -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item -LiteralPath $_.FullName }
        Remove-Item -LiteralPath $work -ErrorAction SilentlyContinue
    }
}
$arch = $null
if (!$inventory) { Note 'could not ask the AMD runtime about your GPU (d4r_gpu_inventory.exe is missing, or the AMD graphics driver is not installed or is too old)' }
else {
    $supported = @($registry | ForEach-Object { $_.architecture })
    $devices = @($inventory.devices | Where-Object { $_.wavefrontSize -eq 32 -and ($supported -contains $_.architecture -or !$supported.Count) })
    if (!$devices.Count) {
        $seen = @($inventory.devices | ForEach-Object { "$($_.name) ($($_.architecture))" }) -join ', '
        Bad ("no supported RDNA3/RDNA4 GPU found by the AMD runtime. It sees: " + $(if ($seen) { $seen } else { 'nothing' }) + '. Update the AMD Adrenalin driver; RDNA2 and older are not supported')
    } else {
        foreach ($device in $devices) {
            $a = $device.architecture
            $missing = @()
            foreach ($rel in @("d4r\_nvngx_$a.dll", "d4r\pixel_convert_$a.hsaco", "d4r\nvapi\$a\nvapi64.dll", "d4r\native\$a\d4r-kernels.txt")) {
                if (!(Test-Path -LiteralPath (Join-Path $game $rel))) { $missing += $rel }
            }
            $native = @(Get-ChildItem -LiteralPath (Join-Path $d4r "native\$a") -Filter '*.hsaco' -ErrorAction SilentlyContinue).Count
            if ($missing.Count) { Bad "GPU $($device.name) ($a): this release has no complete build for it (missing: $($missing -join ', '))" }
            else {
                Ok "GPU $($device.name) ($a): build present, $native native kernels"
                if (!$arch) { $arch = $a }
            }
            if ($a -ne 'gfx1201') { Note "only the RX 9070 XT (gfx1201) has been tested on real hardware; $a is compile-tested" }
        }
    }
    if (@($inventory.devices | Where-Object { $_.wavefrontSize -eq 32 }).Count -gt 1) { Note 'several GPUs found; d4r uses the one the game renders on' }
}
if ($arch -and $registry.Count) {
    $info = @($registry | Where-Object { $_.architecture -eq $arch }) | Select-Object -First 1
    $bad = 0
    $objects = @(Get-Item -LiteralPath (Join-Path $d4r "pixel_convert_$arch.hsaco") -ErrorAction SilentlyContinue) +
        @(Get-ChildItem -LiteralPath (Join-Path $d4r "native\$arch") -Filter '*.hsaco' -ErrorAction SilentlyContinue)
    foreach ($object in $objects) {
        $stream = [IO.File]::OpenRead($object.FullName)
        try { $header = New-Object byte[] 64; [void]$stream.Read($header, 0, 64) } finally { $stream.Dispose() }
        if ($header[0] -ne 127 -or $header[1] -ne 69 -or $header[2] -ne 76 -or $header[3] -ne 70 -or $header[48] -ne $info.elfMachine) {
            Bad "$($object.Name) is not built for $arch (do not rename or mix files from other GPUs; extract the d4r ZIP again)"; $bad++
        }
    }
    if ($objects.Count -and !$bad) { Ok "all $($objects.Count) GPU code objects are built for $arch" }
}

# --- Linux / Wine leftovers ------------------------------------------------------------------------
$proxyFound = $false
foreach ($name in @('d3d12.dll', 'd3d12core.dll')) {
    $proxy = Join-Path $game $name
    if (Test-Path -LiteralPath $proxy -PathType Leaf) {
        $item = Get-Item -LiteralPath $proxy
        if ($item.Length -le 64MB -and [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($proxy)) -match '(?i)vkd3d|wined3d') {
            Bad "$name in the game folder is vkd3d-proton/Wine (left over from a Linux install?). Move it out; d4r on Windows uses native D3D12"
            $proxyFound = $true
        }
    }
}
if (!$proxyFound) { Ok 'no vkd3d/Wine d3d12.dll in the game folder' }

# --- d4r.ini ---------------------------------------------------------------------------------------
$iniPath = Join-Path $d4r 'd4r.ini'
$model = 'K'
if (Test-Path -LiteralPath $iniPath) {
    $section = ''; $iniProblems = 0
    $known = @{ 'dlss.model' = 1; 'dlss.showwatermark' = 1; 'interop.asyncinterop' = 1; 'debug.validateoutput' = 1; 'debug.log' = 1; 'paths.cachedir' = 1 }
    $boolean = @('dlss.showwatermark', 'interop.asyncinterop', 'debug.validateoutput')
    foreach ($raw in (Read-TextFile $iniPath).TrimStart([char]0xFEFF) -split "`n") {
        $line = $raw.Trim()
        if (!$line -or $line.StartsWith(';') -or $line.StartsWith('#')) { continue }
        if ($line.StartsWith('[')) { $section = $line.Trim('[', ']').Trim(); continue }
        $eq = $line.IndexOf('='); if ($eq -lt 0) { continue }
        $key = $line.Substring(0, $eq).Trim()
        $value = ($line.Substring($eq + 1) -replace '\s[;#].*$', '').Trim()
        $name = ("$section.$key").ToLowerInvariant()
        $v = $value.ToLowerInvariant()
        if (!$known.ContainsKey($name)) { Note "d4r.ini: [$section] $key is not a Windows setting; ignored"; $iniProblems++ }
        elseif ($name -eq 'dlss.model') {
            if (@('', 'auto', 'k', 'dlss4', '11') -contains $v) { $model = 'K' }
            elseif (@('m', 'dlss4.5', '13') -contains $v) { $model = 'M' }
            else { Note "d4r.ini: [DLSS] Model '$value' is not available on Windows (use K or M); K will be used"; $iniProblems++ }
        }
        elseif ($boolean -contains $name) {
            if (@('', 'auto', '1', 'true', 'yes', 'on', '0', 'false', 'no', 'off') -notcontains $v) { Note "d4r.ini: [$section] $key must be true or false, not '$value'; the default will be used"; $iniProblems++ }
        }
        elseif ($name -eq 'debug.log' -and @('', 'auto', 'normal', 'verbose') -notcontains $v) { Note "d4r.ini: [Debug] Log must be normal or verbose, not '$value'; normal will be used"; $iniProblems++ }
    }
    if (!$iniProblems) { Ok "d4r.ini is valid (model $model)" }
}

# --- what the last game start said -----------------------------------------------------------------
$log = Join-Path $d4r 'd4r_nvngx.log'
if (Test-Path -LiteralPath $log) {
    $messages = @(Get-Content -LiteralPath $log | Where-Object { $_ -match '^d4r: ' })
    $failed = @($messages | Where-Object { $_ -match 'DLSS disabled' })
    if ($failed.Count) { foreach ($m in $failed | Select-Object -First 3) { Bad ('last game start: ' + $m.Substring(5)) } }
    elseif ($messages.Count) { Ok ('last game start: ' + $messages[0].Substring(5)) }
    else { Note 'd4r\d4r_nvngx.log has no d4r summary line yet (the game may not have chosen DLSS)' }
    foreach ($m in $messages | Where-Object { $_ -notmatch 'DLSS disabled|player mode' } | Select-Object -First 5) { Note ('last game start: ' + $m.Substring(5)) }
} else { Note 'no d4r\d4r_nvngx.log yet: start the game once and choose DLSS (or FSR/XeSS) in its graphics settings' }
if (Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($game + '\', [StringComparison]::OrdinalIgnoreCase) -and $_.ProcessName -notmatch '^(powershell|pwsh)$' }) {
    Note 'the game is running; its log is rewritten at every start'
}

Say ''
if ($problems -eq 0) { Say 'Everything d4r needs is in place.' } else { Say "$problems thing(s) to fix above." }

# --- bug-report ZIP ------------------------------------------------------------------------------
if ($Report) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $destination = if ($ReportDirectory) { $ReportDirectory } else { [Environment]::GetFolderPath('Desktop') }
    $staging = Join-Path ([IO.Path]::GetTempPath()) ("d4r-report-$stamp")
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    try {
        $lines | Set-Content -LiteralPath (Join-Path $staging 'check.txt') -Encoding UTF8
        foreach ($file in @('d4r\d4r_nvngx.log', 'd4r\d4r.ini', 'OptiScaler.ini', 'OptiScaler.log')) {
            $source = Join-Path $game $file
            if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination (Join-Path $staging (Split-Path $file -Leaf)) }
        }
        Get-ChildItem -LiteralPath $game -Filter 'OptiScaler*.log' -File | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $staging -Force }
        try { Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $staging 'driver.json') -Encoding UTF8 } catch {}
        if ($inventory) { $inventory | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $staging 'gpu-inventory.json') -Encoding UTF8 }
        Get-ChildItem -LiteralPath $game -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName.StartsWith($d4r + '\') -or $_.Name -in @('dxgi.dll', 'OptiScaler.ini') } |
            ForEach-Object { '{0}  {1,12}  {2}' -f (Hash $_.FullName), $_.Length, $_.FullName.Substring($game.Length + 1) } |
            Set-Content -LiteralPath (Join-Path $staging 'files.sha256.txt') -Encoding UTF8
        $zip = Join-Path $destination "d4r-report-$stamp.zip"
        Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zip -Force
        Write-Host ''
        Write-Host "Bug report written: $zip" -ForegroundColor Cyan
        Write-Host 'Also tell us: GPU, game, K or M, and what the image looked like. The logs can contain local folder names; look before posting publicly.'
    } finally {
        Get-ChildItem -LiteralPath $staging -File -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item -LiteralPath $_.FullName }
        Remove-Item -LiteralPath $staging -ErrorAction SilentlyContinue
    }
}
if ($problems) { exit 1 }
