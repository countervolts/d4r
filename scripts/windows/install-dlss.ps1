[CmdletBinding()]
param(
    [string]$PackageRoot = $PSScriptRoot,
    [string]$CacheRoot = (Join-Path $env:LOCALAPPDATA 'd4r/cache'),
    [string]$BackupRoot = (Join-Path $env:LOCALAPPDATA 'd4r/backups'),
    [switch]$Yes
)
$ErrorActionPreference = 'Stop'

$source = [IO.Path]::GetFullPath($PackageRoot)
$runtime = if (Test-Path -LiteralPath (Join-Path $source 'd4r/d4r.ini')) {
    Join-Path $source 'd4r'
} elseif (Test-Path -LiteralPath (Join-Path $source 'd4r.ini')) {
    $source
} else {
    throw "No d4r runtime found under $source. Extract the Windows release first."
}
$runtime = [IO.Path]::GetFullPath($runtime)
$CacheRoot = [IO.Path]::GetFullPath($CacheRoot)
$BackupRoot = [IO.Path]::GetFullPath($BackupRoot)
$gameRoot = if ($runtime -eq $source) { Split-Path -Parent $source } else { $source }
$gameRoot = [IO.Path]::GetFullPath($gameRoot)
$backupRoot = Join-Path $BackupRoot ('dlss-install-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N'))
foreach ($path in @($CacheRoot, $BackupRoot)) {
    if ($path.Equals($gameRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $path.StartsWith($gameRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cache and backups must stay outside the game directory: $path"
    }
}

$dlssUrl = 'https://raw.githubusercontent.com/NVIDIA/DLSS/374959484e79a640feaba44c93ac8cfb0a03f5b5/lib/Windows_x86_64/rel/nvngx_dlss.dll'
$dlssHash = '3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983'
$dlssdUrl = 'https://raw.githubusercontent.com/NVIDIA/DLSS/374959484e79a640feaba44c93ac8cfb0a03f5b5/lib/Windows_x86_64/rel/nvngx_dlssd.dll'
$dlssdHash = '4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80'
$driverUrl = 'https://us.download.nvidia.com/Windows/617.14/617.14-desktop-win10-win11-64bit-international-dch-whql.exe'
$driverHash = '63d3da5ed52547bf3fd86204bb9ea1455947b24e2aa2519f359126f9a6efe13f'
$ngxHash = '66767018c36b3bab46398dade3adf173daa3730fda75965689ea848c9bc4e79b'
$sevenZipUrl = 'https://github.com/ip7z/7zip/releases/download/26.04/7zr.exe'
$sevenZipHash = '256feca8e274e5da655e2a284fabafd9f554365eb164862089dacd4e8276d282'

function Get-Sha256([string]$Path) {
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $sha.Dispose() }
}

$pinPath = Join-Path $runtime 'dll-pins.txt'
if (!(Test-Path -LiteralPath $pinPath -PathType Leaf)) { throw "Missing $pinPath. Use a current d4r Windows release." }
$pins = Get-Content -LiteralPath $pinPath -Raw
foreach ($required in @(@{kind='dlss'; hash=$dlssHash}, @{kind='dlssd'; hash=$dlssdHash}, @{kind='ngx'; hash=$ngxHash})) {
    if ($pins -notmatch ('(?im)^\s*' + [regex]::Escape($required.kind) + '\s+' + [regex]::Escape($required.hash) + '\s+\S+\s+\S+\s*$')) {
        throw "This release does not recognize NVIDIA $($required.kind) hash $($required.hash). Update the d4r package first."
    }
}

if (!$Yes) {
    $answer = Read-Host 'Download DLSS 310.9.1, Ray Reconstruction 310.9.1, and NVIDIA NGX core from driver 617.14 (~945 MiB)? [Y/n]'
    if ($answer -match '^(?i)n') { Write-Host 'No files changed.'; exit 0 }
}

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$work = Join-Path $tempRoot ('d4r-dlss-install-' + [Guid]::NewGuid().ToString('N'))
$work = [IO.Path]::GetFullPath($work)
if (!$work.StartsWith($tempRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or
    (Split-Path -Leaf $work) -notmatch '^d4r-dlss-install-[a-f0-9]{32}$') {
    throw "Temporary work path escaped the expected temp folder: $work"
}
New-Item -ItemType Directory -Force -Path $work | Out-Null
$cacheNvidia = Join-Path $CacheRoot 'nvidia'
$cacheTools = Join-Path $CacheRoot 'tools'
New-Item -ItemType Directory -Force -Path $cacheNvidia,$cacheTools | Out-Null

function Get-VerifiedFile([string]$Name, [string]$Url, [string]$ExpectedHash, [string]$CacheDirectory, [string]$WorkDirectory) {
    $cached = Join-Path $CacheDirectory $Name
    if (Test-Path -LiteralPath $cached -PathType Leaf) {
        $hash = Get-Sha256 $cached
        if ($hash -eq $ExpectedHash) { return $cached }
        $quarantine = "$cached.bad-$([DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss'))"
        Move-Item -LiteralPath $cached -Destination $quarantine
        Write-Warning "Moved invalid cache file to $quarantine"
    }
    $download = Join-Path $WorkDirectory $Name
    Write-Host "Downloading $Name from its pinned official source..."
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $download
    $hash = Get-Sha256 $download
    if ($hash -ne $ExpectedHash) { throw "SHA256 mismatch for $Name; no runtime files changed." }
    $staged = "$cached.tmp-$([Guid]::NewGuid().ToString('N'))"
    Copy-Item -LiteralPath $download -Destination $staged
    Move-Item -LiteralPath $staged -Destination $cached
    return $cached
}

$changed = [Collections.Generic.List[object]]::new()
try {
    $dlssCached = Get-VerifiedFile 'nvngx_dlss.dll' $dlssUrl $dlssHash $cacheNvidia $work
    $dlssdCached = Get-VerifiedFile 'nvngx_dlssd.dll' $dlssdUrl $dlssdHash $cacheNvidia $work
    $ngxCache = Join-Path $cacheNvidia '_nvngx.dll'
    $validCoreCache = (Test-Path -LiteralPath $ngxCache -PathType Leaf) -and
        (Get-Sha256 $ngxCache) -eq $ngxHash
    if ($validCoreCache -and (Get-Item -LiteralPath $ngxCache).VersionInfo.FileVersion -ne '32.0.16.1714') {
        $validCoreCache = $false
    }
    if (!$validCoreCache) {
        if (Test-Path -LiteralPath $ngxCache -PathType Leaf) {
            $bad = "$ngxCache.bad-$([DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss'))"
            Move-Item -LiteralPath $ngxCache -Destination $bad
            Write-Warning "Moved invalid cache file to $bad"
        }
        $driverPath = Join-Path $work 'nvidia-driver-617.14.exe'
        Write-Host 'Downloading NVIDIA driver 617.14 (~945 MiB); installer will not run.'
        Invoke-WebRequest -UseBasicParsing -Uri $driverUrl -OutFile $driverPath
        $driverActual = Get-Sha256 $driverPath
        if ($driverActual -ne $driverHash) { throw 'NVIDIA driver archive SHA256 mismatch; no runtime files changed.' }
        $sevenZip = Get-VerifiedFile '7zr-26.04.exe' $sevenZipUrl $sevenZipHash $cacheTools $work
        $extract = Join-Path $work 'ngx'
        New-Item -ItemType Directory -Force -Path $extract | Out-Null
        & $sevenZip e -y "-o$extract" $driverPath 'Display.Driver/_nvngx.dll' | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "7-Zip extraction failed: exit=$LASTEXITCODE" }
        $extractedNgx = Join-Path $extract '_nvngx.dll'
        if (!(Test-Path -LiteralPath $extractedNgx -PathType Leaf)) { throw 'NVIDIA driver archive has no Display.Driver/_nvngx.dll.' }
        $actual = Get-Sha256 $extractedNgx
        if ($actual -ne $ngxHash) { throw "Extracted NGX core SHA256 mismatch: $actual" }
        $version = (Get-Item -LiteralPath $extractedNgx).VersionInfo.FileVersion
        if ($version -ne '32.0.16.1714') { throw "Unexpected NGX core version $version; expected 32.0.16.1714." }
        $stagedNgx = "$ngxCache.tmp-$([Guid]::NewGuid().ToString('N'))"
        Copy-Item -LiteralPath $extractedNgx -Destination $stagedNgx
        Move-Item -LiteralPath $stagedNgx -Destination $ngxCache
    }

    $assets = @(
        @{ source=$dlssCached; relative='dlss/nvngx_dlss.dll'; hash=$dlssHash },
        @{ source=$dlssdCached; relative='rr/nvngx_dlssd.dll'; hash=$dlssdHash },
        @{ source=$ngxCache; relative='ngx/_nvngx.dll'; hash=$ngxHash }
    )
    $originals = [Collections.Generic.List[object]]::new()
    foreach ($asset in $assets) {
        $destination = Join-Path $runtime $asset.relative
        if (Test-Path -LiteralPath $destination -PathType Container) { throw "DLL destination is a directory: $destination" }
        $present = Test-Path -LiteralPath $destination -PathType Leaf
        $previousHash = if ($present) { Get-Sha256 $destination } else { $null }
        if ($previousHash -eq $asset.hash) { continue }
        $backup = $null
        if ($present) {
            $backup = Join-Path $backupRoot $asset.relative
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backup) | Out-Null
            Copy-Item -LiteralPath $destination -Destination $backup
            $backupHash = Get-Sha256 $backup
            if ($backupHash -ne $previousHash) { throw "Backup hash mismatch for $destination; no runtime files changed." }
        }
        $originals.Add(@{path=$destination; existed=$present; sha256=$previousHash; backup=$backup; installHash=$asset.hash})
    }
    if ($originals.Count) {
        New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
        $changed = $originals
        @{ createdUtc=[DateTime]::UtcNow.ToString('o'); files=$originals } |
            ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $backupRoot 'backup-manifest.json') -Encoding UTF8
        foreach ($entry in $originals) {
            $asset = $assets | Where-Object { (Join-Path $runtime $_.relative) -eq $entry.path } | Select-Object -First 1
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $entry.path) | Out-Null
            $stillPresent = Test-Path -LiteralPath $entry.path -PathType Leaf
            if ($entry.existed) {
                if (!$stillPresent -or (Get-Sha256 $entry.path) -ne $entry.sha256) {
                    throw "Destination changed after backup; refusing to replace $($entry.path)"
                }
            } elseif (Test-Path -LiteralPath $entry.path) {
                throw "Destination appeared after validation; refusing to replace $($entry.path)"
            }
            $staged = "$($entry.path).install-$([Guid]::NewGuid().ToString('N'))"
            Copy-Item -LiteralPath $asset.source -Destination $staged
            Move-Item -LiteralPath $staged -Destination $entry.path -Force
        }
        Write-Host "Backups and original hashes: $backupRoot"
    } else {
        Write-Host 'All three NVIDIA DLLs already match their pinned hashes.'
    }
    Write-Host 'Installed NVIDIA DLSS 310.9.1, Ray Reconstruction 310.9.1, and NGX 32.0.16.1714.'
    Write-Host 'Files remain local under the applicable NVIDIA licenses; no binaries are bundled into d4r releases.'
    Write-Host 'This installer does not build local K/M/L or Ray Reconstruction model data.'
} catch {
    for ($i = $changed.Count - 1; $i -ge 0; $i--) {
        $entry = $changed[$i]
        if (!(Test-Path -LiteralPath $entry.path)) { continue }
        if ($entry.existed -and $entry.backup) {
            $currentHash = Get-Sha256 $entry.path
            if ($currentHash -ne $entry.installHash -and $currentHash -ne $entry.sha256) {
                Write-Warning "Rollback left a file changed by another process: $($entry.path)"
                continue
            }
            $rollback = "$($entry.path).rollback-$([Guid]::NewGuid().ToString('N'))"
            Copy-Item -LiteralPath $entry.backup -Destination $rollback
            Move-Item -LiteralPath $rollback -Destination $entry.path -Force
        } elseif ((Get-Sha256 $entry.path) -eq $entry.installHash) {
            Remove-Item -LiteralPath $entry.path
        }
    }
    throw
} finally {
    $resolvedWork = [IO.Path]::GetFullPath($work)
    if (!$resolvedWork.StartsWith($tempRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path -Leaf $resolvedWork) -notmatch '^d4r-dlss-install-[a-f0-9]{32}$') {
        throw "Refusing to remove a path outside the installer temp folder: $resolvedWork"
    }
    Remove-Item -LiteralPath $resolvedWork -Recurse -Force -ErrorAction SilentlyContinue
}
