# NVIDIA installer for native Windows (the Windows counterpart of install.sh): beside the release's d4r\ or inside it.
# Installs NVIDIA's DLLs and then builds the native engine's models from them on this machine. Both are kept in
# %LOCALAPPDATA%\d4r, so the next game's install copies them instead of downloading or building again.
# Run install.cmd (double-click), or: powershell -ExecutionPolicy Bypass -File install.ps1 [options] [folder]
[CmdletBinding()]
param(
    [switch]$Yes,            # accept every default (also the behaviour without a console)
    [switch]$NoEngine,       # only the NVIDIA DLLs, no engine models
    [switch]$EngineOnly,     # only the engine models (the DLLs are already installed)
    [switch]$RebuildEngine,  # build the models again even when a cached build fits
    [switch]$NoFrameGeneration,  # leave DLSS Frame Generation (RDNA3 / RDNA4) as it is
    [string]$Folder = $PSScriptRoot
)
# Windows PowerShell 5.1 (what install.cmd runs) leaves $PSScriptRoot empty in parameter defaults
if (-not $Folder) { $Folder = Split-Path -Parent $MyInvocation.MyCommand.Path }
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is many times slower with its progress bar

# The same pins as install.sh
$DlssUrl = 'https://raw.githubusercontent.com/NVIDIA/DLSS/a291cc7d2cc642a51566f3dfd5376f635cd1b284/lib/Windows_x86_64/rel/nvngx_dlss.dll'
$DlssSha256 = 'be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e'
$DlssdUrl = 'https://raw.githubusercontent.com/NVIDIA/DLSS/374959484e79a640feaba44c93ac8cfb0a03f5b5/lib/Windows_x86_64/rel/nvngx_dlssd.dll'
$DlssdSha256 = '4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80'
$DriverUrl = 'https://us.download.nvidia.com/Windows/596.36/596.36-desktop-win10-win11-64bit-international-dch-whql.exe'
$DriverSha256 = '4a1793d2bc7792f43270fe8286ec381ae24409fe29049dbe1eb8530ed89e19cf'
$NgxSha256 = '0e3423b3b7afa1019c7ecac053a8408df8e5d9d9d904434705c6aaeacec02fef'
# The model builder's Python: python.org's embeddable build and numpy's wheel from PyPI (both pinned)
$PythonVersion = '3.14.8'
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$PythonSha256 = 'a93abe456ab01bd96d7a085b3cdb6566b3063f4241360d114142fbdb07f0a310'
$NumpyUrl = 'https://files.pythonhosted.org/packages/52/9a/e1e512ebc948d5b9dd33b08736760f0ebbed2848fd4eda1f553088a6dcee/numpy-2.5.4-cp314-cp314-win_amd64.whl'
$NumpySha256 = '2ca144f15135b6212a5c47b1e2aeca6e412f102f95a2d5d88d8aec77eb255de3'
# DLSS Frame Generation (d4r\fg\dlssg): DLSS-G 310.7.0 from the DLSS repository (the full model) and Streamline
# 2.14.16's DLSS-G 310.9.16 (FG Lite) with its DLLs for OptiScaler's DLSS-G output, as install.sh pins them
$DlssgUrl = 'https://raw.githubusercontent.com/NVIDIA/DLSS/a291cc7d2cc642a51566f3dfd5376f635cd1b284/lib/Windows_x86_64/rel/nvngx_dlssg.dll'
$DlssgSha256 = '135eaf0733c1e37381a8c28abcf7a862404a54132b81787c04e35d09efc5e36f'
$SlUrl = 'https://github.com/NVIDIA-RTX/Streamline/releases/download/v2.14.16/streamline-sdk-v2.14.16.zip'
$SlSha256 = '32041e51db0cab709a271b72a352281b690c2abbd669a827809401a5dd23956b'
$SlFiles = [ordered]@{
    'nvngx_dlssg.dll' = '8d59d49b36410dd6b62fef2503a5871d347415bd85d943fa0848563a95dd50c7'
    'sl.interposer.dll' = 'b1d7395aebd7b9722841d64dcf7433680020b6a8e421f851c16d5ba0c2182edf'
    'sl.common.dll' = '38eea3b4209afc455636ea372b5933d65fcc13becec6c84d101dcfc4d8648f75'
    'sl.dlss_g.dll' = '3ed9538df724993bea8a050195a6a78b9c8c8a566d5b05ade477aab3c390e90f'
    'sl.reflex.dll' = '443577e869352436ff7ec8f4d9c90418d3e95368436ce219261e0f4f07a38122'
    'sl.pcl.dll' = '5698c38e722318681f215c9fe68e45b9bedd51378acc699d8eee26e961e57770'
}

function Fail([string]$Message) { Write-Host "install.ps1: $Message" -ForegroundColor Red; exit 1 }
function Sha256([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Valid([string]$Path, [string]$Hash) { (Test-Path -LiteralPath $Path -PathType Leaf) -and (Sha256 $Path) -eq $Hash }
function Interactive { -not $Yes -and [Environment]::UserInteractive -and -not [Console]::IsInputRedirected }
function Ask([string]$Question, [bool]$Default) {
    if (-not (Interactive)) { return $Default }
    while ($true) {
        $answer = (Read-Host "$Question [$(if ($Default) { 'Y/n' } else { 'y/N' })]").Trim().ToLowerInvariant()
        if ($answer -eq '') { return $Default }
        if ($answer.StartsWith('y')) { return $true }
        if ($answer.StartsWith('n')) { return $false }
        Write-Host '  Please answer y or n (or press Enter for the default).'
    }
}
function Download([string]$Url, [string]$Path) {
    try { Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing }
    catch { Fail "Download failed: $Url ($($_.Exception.Message)). Run the installer again to retry." }
}
function Verify([string]$Path, [string]$Hash) { if (-not (Valid $Path $Hash)) { Fail "SHA-256 mismatch for $Path; nothing installed." } }
# Runs a native program; its exit code ($LASTEXITCODE) is what the caller checks. Windows PowerShell 5.1 (what
# install.cmd runs) turns a native program's redirected stderr into a terminating error under 'Stop'.
function Native([scriptblock]$Block) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Block } finally { $ErrorActionPreference = $saved }
}

$Root = (Resolve-Path -LiteralPath $Folder).Path
$Runtime = Join-Path $Root 'd4r'
if (Test-Path -LiteralPath (Join-Path $Root 'd4r.ini')) { $Runtime = $Root }
if (-not (Test-Path (Join-Path $Runtime 'nvngx.dll')) -or -not (Test-Path (Join-Path $Runtime 'd4r.ini'))) {
    Fail "Extract the release ZIP first; no d4r runtime found in $Root"
}
# What games need here to offer DLSS, which an OptiScaler.ini from an earlier Windows release may lack: without an
# NVAPI (AMD's driver has no nvapi64.dll) games such as Cyberpunk 2077 offer FSR, and OptiScaler's fakenvapi answers
# them; OptiScaler treats non-NVIDIA GPUs as not DLSS capable unless told otherwise (ForceNvidia, d4r's patch).
$optiIni = Join-Path $Root 'OptiScaler.ini'
if ((Test-Path -LiteralPath $optiIni) -and -not (Test-Path -LiteralPath "$env:SystemRoot\System32\nvapi64.dll")) {
    $lines = [Collections.Generic.List[string]]([IO.File]::ReadAllText($optiIni) -split "`n")
    $changed = @()
    foreach ($want in @(@('fakenvapi', 'UseFakenvapi'), @('Spoofing', 'ForceNvidia'))) {
        $section = ''; $header = -1; $found = $false
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $line = $lines[$i].Trim()
            if ($line -match '^\[(.+)\]') { $section = $Matches[1].Trim(); if ($section -eq $want[0]) { $header = $i } }
            elseif ($section -eq $want[0] -and $line -match "^$($want[1])\s*=\s*(\S*)") {
                $found = $true
                if ($Matches[1] -notmatch '^(true|1|yes|on)$') {
                    $lines[$i] = "$($want[1])=true" + $(if ($lines[$i].EndsWith("`r")) { "`r" } else { '' })
                    $changed += "[$($want[0])] $($want[1])=true"
                }
                break
            }
        }
        if (-not $found -and $header -ge 0) {
            $lines.Insert($header + 1, "$($want[1])=true" + $(if ($lines[$header].EndsWith("`r")) { "`r" } else { '' }))
            $changed += "[$($want[0])] $($want[1])=true"
        }
    }
    if ($changed) {
        [IO.File]::WriteAllText($optiIni, ($lines -join "`n"), (New-Object Text.UTF8Encoding $false))
        Write-Host "OptiScaler.ini: set $($changed -join ', '), which games need to offer DLSS here."
    }
}

$Cache = if ($env:D4R_CACHE_DIR) { $env:D4R_CACHE_DIR } else { Join-Path $env:LOCALAPPDATA 'd4r' }
$Nvidia = Join-Path $Cache 'nvidia'
$Tmp = Join-Path ([IO.Path]::GetTempPath()) ("d4r-install-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Tmp, $Nvidia | Out-Null

# Replace DST with SRC through a staged copy beside it; a differing previous file is kept as a backup.
function Place([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Destination) | Out-Null
    $staged = "$Destination.install.$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
    Copy-Item -LiteralPath $Source -Destination $staged
    if (Test-Path -LiteralPath $Destination) {
        $backup = "$Destination.backup.$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
        Copy-Item -LiteralPath $Destination -Destination $backup
        Write-Host "Saved previous DLL: $backup"
    }
    Move-Item -LiteralPath $staged -Destination $Destination -Force
    Write-Host "Installed: $Destination"
}
function Cached([string]$Name, [string]$Hash) { $f = Join-Path $Nvidia $Name; if (Valid $f $Hash) { $f } else { $null } }
function Keep([string]$Path, [string]$Name) {
    Copy-Item -LiteralPath $Path -Destination (Join-Path $Nvidia "$Name.tmp") -Force
    Move-Item -LiteralPath (Join-Path $Nvidia "$Name.tmp") -Destination (Join-Path $Nvidia $Name) -Force
}

function Install-Dlls {
    $dlss = Join-Path $Runtime 'nvngx_dlss.dll'
    $dlssd = Join-Path $Runtime 'nvngx_dlssd.dll'
    $ngx = Join-Path $Runtime 'ngx\_nvngx.dll'
    $needDlss = -not (Valid $dlss $DlssSha256)
    $needDlssd = -not (Valid $dlssd $DlssdSha256)
    $needNgx = -not (Valid $ngx $NgxSha256)
    if (-not $needDlss -and -not $needDlssd -and -not $needNgx) { Write-Host 'NVIDIA DLLs are already installed and verified.'; return }
    $dlssSource = if ($needDlss) { Cached 'nvngx_dlss.dll' $DlssSha256 } else { $null }
    $dlssdSource = if ($needDlssd) { Cached 'nvngx_dlssd.dll' $DlssdSha256 } else { $null }
    $ngxSource = if ($needNgx) { Cached '_nvngx.dll' $NgxSha256 } else { $null }
    $size = 0
    if ($needDlss -and -not $dlssSource) { $size += 57 }
    if ($needDlssd -and -not $dlssdSource) { $size += 47 }
    if ($needNgx -and -not $ngxSource) { $size += 915 }
    if ($size -gt 0) {
        Write-Host "d4r needs NVIDIA's DLSS files, downloaded from NVIDIA under NVIDIA's license terms:"
        Write-Host '  DLSS: https://github.com/NVIDIA/DLSS/blob/v310.9.1/LICENSE.txt'
        Write-Host '  Driver: https://www.nvidia.com/en-us/drivers/nvidia-license/'
        if (-not (Ask "Download them now (~$size MiB, once: later installs use $Nvidia)?" $true)) {
            Fail 'Not downloaded. d4r needs these files; run the installer again when you want to install them.'
        }
    }
    if ($needDlss -and -not $dlssSource) {
        Write-Host 'Downloading DLSS 310.7.0...'
        $dlssSource = Join-Path $Tmp 'nvngx_dlss.dll'
        Download $DlssUrl $dlssSource
        Verify $dlssSource $DlssSha256
        Keep $dlssSource 'nvngx_dlss.dll'
    }
    if ($needDlssd -and -not $dlssdSource) {
        Write-Host 'Downloading DLSS Ray Reconstruction 310.9.1...'
        $dlssdSource = Join-Path $Tmp 'nvngx_dlssd.dll'
        Download $DlssdUrl $dlssdSource
        Verify $dlssdSource $DlssdSha256
        Keep $dlssdSource 'nvngx_dlssd.dll'
    }
    if ($needNgx -and -not $ngxSource) {
        Write-Host 'Downloading NVIDIA driver 596.36 (~915 MiB); extracting NGX without running the installer...'
        $driver = Join-Path $Tmp 'driver.exe'
        Download $DriverUrl $driver
        Verify $driver $DriverSha256
        # The driver package is a 7-Zip self-extractor: Windows' own tar (libarchive) reads it; 7-Zip if installed
        $out = Join-Path $Tmp 'ngx'
        New-Item -ItemType Directory -Force -Path $out | Out-Null
        Native { & "$env:SystemRoot\System32\tar.exe" -x -f $driver -C $out 'Display.Driver/_nvngx.dll' 2>$null }
        $ngxSource = Join-Path $out 'Display.Driver\_nvngx.dll'
        if (-not (Test-Path $ngxSource)) {
            $sevenZip = @("$env:ProgramFiles\7-Zip\7z.exe", "${env:ProgramFiles(x86)}\7-Zip\7z.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
            if ($sevenZip) { Native { & $sevenZip e -y -bso0 -bsp0 "-o$out" $driver 'Display.Driver/_nvngx.dll' | Out-Null }; $ngxSource = Join-Path $out '_nvngx.dll' }
        }
        Remove-Item -LiteralPath $driver -Force
        if (-not (Test-Path $ngxSource)) { Fail 'Could not extract NGX from the driver package; nothing installed.' }
        Verify $ngxSource $NgxSha256
        Keep $ngxSource '_nvngx.dll'
    }
    # Everything is verified before any destination changes.
    if ($needDlss) { Place $dlssSource $dlss }
    if ($needDlssd) { Place $dlssdSource $dlssd }
    if ($needNgx) { Place $ngxSource $ngx }
    Write-Host 'NVIDIA DLLs installed and verified.'
}

# A Python with numpy and a zstd decoder (3.14's own, or the zstandard module: NVIDIA's DLLs hold their kernels
# zstd-compressed) for the model builder: the system's if it has both, else a private one in the cache
function Get-Python {
    foreach ($candidate in @(@('py', '-3'), @('python'))) {
        $exe = $candidate[0]
        $arguments = @($candidate | Select-Object -Skip 1)
        if (Get-Command $exe -ErrorAction SilentlyContinue) {
            $found = Native { & $exe @arguments -c 'import sys, numpy, importlib.util as u; assert sys.version_info >= (3, 11); assert u.find_spec(''compression'') or u.find_spec(''zstandard''); print(sys.executable)' 2>$null } |
                Select-Object -Last 1
            if ($LASTEXITCODE -eq 0 -and $found -and (Test-Path $found)) { return $found }
        }
    }
    $dir = Join-Path $Cache "python-$PythonVersion"
    $python = Join-Path $dir 'python.exe'
    if (Test-Path $python) {
        Native { & $python -c 'import numpy; from compression import zstd' 2>$null }
        if ($LASTEXITCODE -eq 0) { return $python }
    }
    Write-Host "Setting up Python $PythonVersion with numpy for the model builder (~25 MiB, once: $dir)..."
    $zip = Join-Path $Tmp 'python.zip'
    $wheel = Join-Path $Tmp 'numpy.zip'
    Download $PythonUrl $zip; Verify $zip $PythonSha256
    Download $NumpyUrl $wheel; Verify $wheel $NumpySha256
    Remove-Item -Recurse -Force -LiteralPath $dir -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $zip -DestinationPath $dir
    $site = Join-Path $dir 'Lib\site-packages'
    Expand-Archive -LiteralPath $wheel -DestinationPath $site
    # without its ._pth file the embeddable build is a normal Python: the builder's scripts import the modules
    # beside them, which the ._pth's isolated mode leaves off sys.path; site adds Lib\site-packages
    Get-ChildItem -LiteralPath $dir -Filter 'python*._pth' | Remove-Item -Force
    Native { & $python -c 'import numpy; from compression import zstd' 2>$null }
    if ($LASTEXITCODE -ne 0) { Fail "the private Python in $dir cannot import numpy and zstd" }
    return $python
}

function Install-Engine {
    $builder = Join-Path $Runtime 'tools\build-engine-models.py'
    if (-not (Test-Path $builder)) {
        if ($EngineOnly) { Fail "missing $builder (this release has no engine-model builder)" }
        return
    }
    if ($NoEngine) { Write-Host 'No engine models: run the installer again to build them later.'; return }
    $python = Get-Python
    $env:PYTHONUNBUFFERED = '1'   # the builder's progress in order with its tools' errors
    # what the native D3D12 engine runs: Ray Reconstruction D/E/F and Super Resolution K, M and L
    $rrModels = @('rr-d', 'rr-e', 'rr-f')
    $srModels = @('k', 'k-ldr', 'm', 'm-ldr', 'l', 'l-ldr')
    $models = $rrModels + $srModels
    # One build serves every game with the same DLLs, builder and models: the key covers them.
    $parts = @("models=$($models -join ',')", (Sha256 (Join-Path $Runtime 'nvngx_dlss.dll')),
        (Sha256 (Join-Path $Runtime 'nvngx_dlssd.dll')))
    $tools = Join-Path $Runtime 'tools'
    Get-ChildItem -LiteralPath $tools -Recurse -File | Where-Object { $_.Extension -ne '.pyc' } |
        Sort-Object { $_.FullName.Substring($tools.Length) } | ForEach-Object { $parts += (Sha256 $_.FullName) }
    $sha = [Security.Cryptography.SHA256]::Create()
    $key = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($parts -join "`n"))) -replace '-', '').Substring(0, 16).ToLowerInvariant()
    $engine = Join-Path $Runtime 'engine'
    $stamp = Join-Path $engine '.d4r-models'
    $modelTools = Join-Path $tools 'models\model_tools.py'
    $have = { param($dir) Native { & $python $modelTools validate $dir @models 2>$null }; $LASTEXITCODE -eq 0 }
    if (-not $RebuildEngine -and (Test-Path $stamp) -and (Get-Content $stamp) -eq $key -and (& $have $engine)) {
        Write-Host 'The engine models in this folder are up to date.'; return
    }
    $shared = Join-Path $Cache "engine-models\$key"
    if (-not $RebuildEngine -and (Test-Path $shared) -and (& $have $shared)) {
        Write-Host "Using the engine models built earlier ($shared)."
        foreach ($m in $models) {
            Remove-Item -Recurse -Force -LiteralPath (Join-Path $engine $m) -ErrorAction SilentlyContinue
            New-Item -ItemType Directory -Force -Path $engine | Out-Null
            Copy-Item -Recurse -LiteralPath (Join-Path $shared $m) -Destination (Join-Path $engine $m)
        }
    } else {
        Write-Host "Building the engine's models from NVIDIA's DLLs on this machine: a few minutes. On RDNA2 the first"
        Write-Host "Super Resolution build can take about half an hour while ZLUDA compiles DLSS's kernels."
        # separately, as install.sh does: one set failing still builds the other
        $failed = @()
        Native { & $python $builder --models ($rrModels -join ',') $Runtime }
        if ($LASTEXITCODE -ne 0) { $failed += 'Ray Reconstruction' }
        Native { & $python $builder --models ($srModels -join ',') $Runtime }
        if ($LASTEXITCODE -ne 0) { $failed += 'Super Resolution' }
        if ($failed) {
            Fail "the engine models for $($failed -join '/') were not built (reason above). d4r runs those on its normal path; fix the cause and run the installer again."
        }
        Remove-Item -Recurse -Force -LiteralPath "$shared.tmp" -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force -Path "$shared.tmp" | Out-Null
        foreach ($m in $models) { Copy-Item -Recurse -LiteralPath (Join-Path $engine $m) -Destination "$shared.tmp\" }
        Remove-Item -Recurse -Force -LiteralPath $shared -ErrorAction SilentlyContinue
        Move-Item -LiteralPath "$shared.tmp" -Destination $shared
    }
    Set-Content -LiteralPath $stamp -Value $key -NoNewline
    Write-Host "The engine models are in $engine."
}

# DLSS Frame Generation (d4r\fg\dlssg) on RDNA3 / RDNA4: NVIDIA's DLSS-G DLLs (cached in %LOCALAPPDATA%\d4r\nvidia),
# their kernels translated for this GPU, and OptiScaler's NGX route through the DLSS-G host (d4r\tools\fg_install.py).
# A failure here leaves the upscaling set up; it is reported, not fatal.
function Install-FrameGeneration {
    $setup = Join-Path $Runtime 'tools\fg_install.py'
    if ($NoFrameGeneration -or -not (Test-Path $setup) -or -not (Test-Path (Join-Path $Runtime 'fg\dlssg\nvngx.dll_dlssg_host.dll'))) { return }
    if (-not (Test-Path (Join-Path $Root 'OptiScaler.ini'))) { return }
    $python = Get-Python
    Native { & $python $setup $Runtime --check }
    if ($LASTEXITCODE -ne 0) { return }
    $dir = Join-Path $Nvidia 'dlssg'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $full = Join-Path $Nvidia 'nvngx_dlssg-310.7.0.dll'
    $missing = @($SlFiles.Keys | Where-Object { -not (Valid (Join-Path $dir $_) $SlFiles[$_]) })
    $needFull = -not (Valid $full $DlssgSha256)
    if ($missing -or $needFull) {
        Write-Host "DLSS Frame Generation uses NVIDIA's DLSS-G 310.7.0 and the Streamline SDK 2.14.16, under NVIDIA's license terms:"
        Write-Host '  https://github.com/NVIDIA/DLSS/blob/v310.9.1/LICENSE.txt'
        Write-Host '  https://github.com/NVIDIA-RTX/Streamline/blob/main/license.txt'
        $size = $(if ($missing) { 263 } else { 0 }) + $(if ($needFull) { 8 } else { 0 })
        if (-not (Ask "Download them now (~$size MiB, once: later installs use $Nvidia)?" $true)) {
            Write-Host 'DLSS Frame Generation: not set up (not downloaded). Run the installer again to set it up.'; return
        }
    }
    if ($needFull) {
        Write-Host 'Downloading DLSS-G 310.7.0...'
        $file = Join-Path $Tmp 'nvngx_dlssg.dll'
        Download $DlssgUrl $file; Verify $file $DlssgSha256; Keep $file 'nvngx_dlssg-310.7.0.dll'
    }
    if ($missing) {
        Write-Host 'Downloading NVIDIA Streamline SDK 2.14.16 (~263 MiB)...'
        $zip = Join-Path $Tmp 'streamline.zip'
        Download $SlUrl $zip; Verify $zip $SlSha256
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [IO.Compression.ZipFile]::OpenRead($zip)
        try {
            foreach ($name in $SlFiles.Keys) {
                $entry = $archive.GetEntry("bin/x64/$name")
                if (-not $entry) { Fail "Streamline SDK archive has no bin/x64/$name" }
                $out = Join-Path $Tmp $name
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $out, $true)
                Verify $out $SlFiles[$name]
            }
        } finally { $archive.Dispose() }
        Remove-Item -LiteralPath $zip -Force
        foreach ($name in $SlFiles.Keys) {
            Copy-Item -LiteralPath (Join-Path $Tmp $name) -Destination (Join-Path $dir "$name.tmp") -Force
            Move-Item -LiteralPath (Join-Path $dir "$name.tmp") -Destination (Join-Path $dir $name) -Force
        }
    }
    Native { & $python $setup $Runtime --full $full --lite (Join-Path $dir 'nvngx_dlssg.dll') --streamline $dir }
    if ($LASTEXITCODE -ne 0) { Write-Host 'DLSS Frame Generation: not set up (reason above). Upscaling is unaffected.' -ForegroundColor Yellow }
}

try {
    if ($EngineOnly) { $NoEngine = $false } else { Install-Dlls }
    Install-Engine
    Install-FrameGeneration
    Write-Host 'Done.'
} finally {
    Remove-Item -Recurse -Force -LiteralPath $Tmp -ErrorAction SilentlyContinue
}
