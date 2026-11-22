# Checks a native Windows d4r install (the Windows counterpart of d4r-check.sh): run d4r\d4r-check.cmd, or
#   powershell -ExecutionPolicy Bypass -File d4r\d4r-check.ps1 [GAME_FOLDER]
# from the game folder (the one holding the game's .exe and d4r\). It only reads files and prints what it finds.
param([string]$Game)
# Windows PowerShell 5.1 leaves $PSScriptRoot empty in parameter defaults
if (-not $Game) { $Game = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$Game = (Resolve-Path -LiteralPath $Game).Path
$D4r = Join-Path $Game 'd4r'
$Ini = Join-Path $Game 'OptiScaler.ini'
$script:problems = 0
function Ok($text) { Write-Host "  ok       $text" }
function Bad($text) { Write-Host "  MISSING  $text"; $script:problems++ }
function Note($text) { Write-Host "  note     $text" }
function Misconfigured($text) { Write-Host "  CONFIG   $text"; $script:problems++ }
function Present($path, $what, $fix) {
    if (Test-Path -LiteralPath (Join-Path $Game $path)) { Ok $what } else { Bad "$what ($fix)" }
}
# KEY's value in [SECTION] of OptiScaler.ini, only when it is set exactly once
function IniValue($section, $key) {
    $active = $false; $values = @()
    foreach ($line in Get-Content -LiteralPath $Ini) {
        $line = $line.Trim()
        if ($line.StartsWith('[')) { $active = $line -ieq "[$section]"; continue }
        if ($active -and $line -match "^(?i)$key\s*=\s*(.*)$") { $values += $Matches[1].Trim() }
    }
    if ($values.Count -eq 1) { return $values[0] }
    return $null
}
function FileVersion($path) {
    $info = (Get-Item -LiteralPath $path).VersionInfo
    return '{0}.{1}.{2}' -f $info.FileMajorPart, $info.FileMinorPart, $info.FileBuildPart
}

Write-Host "d4r install in $Game"
if (-not (Test-Path -LiteralPath $D4r)) { Write-Host 'no d4r folder here; run this from the folder with the game .exe'; exit 1 }
if (Get-ChildItem -LiteralPath $Game -Filter '*.exe' -File) { Ok 'game executable next to d4r\' }
else { Note 'no .exe next to d4r\ (it must be the game''s main executable folder)' }
Present 'dxgi.dll' 'dxgi.dll (OptiScaler)' 're-extract the d4r zip'
if (Test-Path -LiteralPath $Ini) {
    Ok "OptiScaler config: $Ini"
    # File presence alone does not establish NGX routing: OptiDllPath=OptiScaler can still pick a system _nvngx.dll
    $path = IniValue 'Libraries' 'NvngxPath'
    $absolute = (Join-Path $D4r 'nvngx.dll').ToLowerInvariant()
    $normalized = if ($path) { $path.Replace('/', '\').ToLowerInvariant() } else { '' }
    if ($normalized -in @('d4r\nvngx.dll', '.\d4r\nvngx.dll', $absolute)) {
        Ok "OptiScaler NGX routing: NvngxPath=$path (d4r shim)"
        if ($normalized -ne $absolute) { Note 'a relative NvngxPath needs the game''s working directory to be this folder' }
    } else {
        Misconfigured 'OptiScaler NGX routing: set [Libraries] NvngxPath=d4r\nvngx.dll (exactly once)'
        Note 'd4r\ngx\_nvngx.dll is the NVIDIA core for the shim, not the OptiScaler target'
    }
} else { Bad "OptiScaler config: $Ini (re-extract the d4r zip)" }
foreach ($f in 'nvngx.dll', 'nvcuda.dll', 'zluda\nvcuda.dll', 'nvapi\nvapi64.dll', 'd4r.ini') {
    Present "d4r\$f" "d4r\$f" 're-extract the d4r zip'
}

# NVIDIA's files, which install.cmd downloads
$dlss = Join-Path $D4r 'nvngx_dlss.dll'
if (Test-Path -LiteralPath $dlss) {
    $v = FileVersion $dlss
    if ($v -like '310.7.*' -or $v -like '310.9.*') { Ok "d4r\nvngx_dlss.dll (DLSS $v; the engine models and kernels are built for 310.7)" }
    else { Note "d4r\nvngx_dlss.dll is DLSS ${v}: run install.cmd for the version the engine models are built from" }
} else { Bad 'd4r\nvngx_dlss.dll (run install.cmd)' }
$dlssd = Join-Path $D4r 'nvngx_dlssd.dll'
if (Test-Path -LiteralPath $dlssd) { Ok "d4r\nvngx_dlssd.dll (DLSS Ray Reconstruction $(FileVersion $dlssd))" }
else { Bad 'd4r\nvngx_dlssd.dll (Ray Reconstruction; run install.cmd)' }
Present 'd4r\ngx\_nvngx.dll' 'd4r\ngx\_nvngx.dll (NVIDIA NGX runtime)' 'run install.cmd'
# An _nvngx.dll beside the game's .exe (left by another DLSS mod or an NVIDIA install copy) is loaded instead of d4r's
# NGX core and breaks DLSS (Palworld)
$stray = Join-Path $Game '_nvngx.dll'
if (Test-Path -LiteralPath $stray) {
    $version = (Get-Item -LiteralPath $stray).VersionInfo.FileVersion
    Misconfigured "stray _nvngx.dll in the game folder$(if ($version) { " ($version)" }): move it out; d4r uses d4r\ngx\_nvngx.dll"
} else { Ok 'no stray _nvngx.dll in the game folder' }

# The CUDA path (ZLUDA) for what the engine does not cover runs on HIP, which AMD's driver installs
$hip = Join-Path $env:SystemRoot 'System32\amdhip64_7.dll'
$script:gpuArchs = @()
if (Test-Path -LiteralPath $hip) { Ok "AMD HIP runtime ($hip, $((Get-Item -LiteralPath $hip).VersionInfo.FileVersion))" }
else { Bad 'AMD HIP runtime (amdhip64_7.dll): install AMD''s current Adrenalin driver' }
if (Test-Path -LiteralPath $hip) {
    # hipDeviceProp_tR0600 (HIP 6.0+): gcnArchName at byte 1160 of 1472 (tools/wine_nvcuda_bridge.c reads the same)
    Add-Type -Namespace D4r -Name Hip -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("amdhip64_7.dll")] public static extern int hipGetDeviceCount(out int count);
[System.Runtime.InteropServices.DllImport("amdhip64_7.dll")] public static extern int hipGetDevicePropertiesR0600(byte[] props, int device);
'@
    $count = 0
    if ([D4r.Hip]::hipGetDeviceCount([ref]$count) -eq 0) {
        for ($i = 0; $i -lt $count; $i++) {
            $props = New-Object byte[] 2048
            if ([D4r.Hip]::hipGetDevicePropertiesR0600($props, $i) -ne 0) { continue }
            $name = [Text.Encoding]::ASCII.GetString($props, 0, 256).Split([char]0)[0]
            $arch = [Text.Encoding]::ASCII.GetString($props, 1160, 256).Split([char]0)[0].Split(':')[0]
            $script:gpuArchs += $arch
            # RDNA4 also has <target>-fp8, which d4r.ini's [Kernels] NativeFp8 (on by default) selects
            $folders = @("kernels\$arch", "kernels\$arch-fp8") | Where-Object { Test-Path -LiteralPath (Join-Path $D4r $_) }
            if ($folders) { Ok "GPU $name ($arch): native kernels for the CUDA path present ($($folders -join ', '))" }
            else { Note "GPU $name ($arch): no native kernels for it in this release (the CUDA path runs, slower)" }
        }
    } else { Note 'HIP lists no GPU' }
}
# RDNA2 (gfx103x): the native engine needs RDNA 3 or later, so no models are built and DLSS runs on ZLUDA
$rdna2Only = $script:gpuArchs.Count -gt 0 -and -not ($script:gpuArchs | Where-Object { $_ -notlike 'gfx103*' })

# The engine's models, which install.cmd builds
$engine = Join-Path $D4r 'engine'
$models = @('rr-d', 'rr-e', 'rr-f', 'k', 'k-ldr', 'm', 'm-ldr', 'l', 'l-ldr')
$have = @($models | Where-Object { Test-Path -LiteralPath (Join-Path $engine $_) })
if ($rdna2Only) { Ok 'engine models: none needed (RDNA2 runs DLSS through ZLUDA and the native kernels)' }
elseif ($have.Count -eq $models.Count) { Ok "engine models: $($have -join ', ')" }
elseif ($have.Count) {
    Bad "engine models: only $($have -join ', ') (run install.cmd -EngineOnly; the rest run on ZLUDA, much slower)"
} else { Bad 'engine models in d4r\engine (run install.cmd -EngineOnly; until then DLSS runs on ZLUDA, much slower)' }

# d4r.ini [DLSS] Model on RDNA2: M and L would run on translated kernels at seconds per frame; d4r runs K instead
$d4rIni = Join-Path $D4r 'd4r.ini'
if ($rdna2Only -and (Test-Path -LiteralPath $d4rIni)) {
    $section = ''; $model = ''
    foreach ($line in Get-Content -LiteralPath $d4rIni) {
        $line = ($line -replace '[;#].*$', '').Trim()
        if ($line.StartsWith('[')) { $section = $line; continue }
        if ($section -ieq '[DLSS]' -and $line -match '^(?i)Model\s*=\s*(.*)$') { $model = $Matches[1].Trim().ToUpperInvariant() }
    }
    if ($model -in @('M', 'L', '12', '13', 'DLSS4.5')) {
        Misconfigured "d4r.ini [DLSS] Model = $model is not supported on RDNA2 (seconds per frame, then a crash); d4r runs K instead. Set Model = K"
    }
}

if ($script:problems -eq 0) {
    Write-Host "`nFiles and OptiScaler NGX routing checks passed (runtime initialization is not tested)."
    exit 0
}
Write-Host "`n$($script:problems) thing(s) to fix above."
exit 1
