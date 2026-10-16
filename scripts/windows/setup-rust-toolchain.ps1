[CmdletBinding()]
param([string]$ToolRoot, [string]$DownloadRoot)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$ToolRoot) { $ToolRoot = Join-Path $repo '.tools/rust-1.98.1-gnu' }
if (!$DownloadRoot) { $DownloadRoot = Join-Path $repo '.tools/downloads' }
$ToolRoot = [IO.Path]::GetFullPath($ToolRoot)
$DownloadRoot = [IO.Path]::GetFullPath($DownloadRoot)
New-Item -ItemType Directory -Force $ToolRoot,$DownloadRoot | Out-Null
# Pinned official Rust 2026-09-03 manifest. The toolchain is local to this
# workspace; rustup, PATH changes and machine-wide installation are unnecessary.
$components = @(
    @{Name='rustc'; Hash='f0e8e33973771acc4d2f87891b19d4f3f2d3827e2e7848c091d7c070bd63479c'},
    @{Name='cargo'; Hash='4bd77f16bd2a26db6eacf9320414d3a792d9998cb5e9ac122280ba56126f2a44'},
    @{Name='rust-std'; Hash='5bb599a541fcb9c0edc00e512570f60d2262623f1e2a19a44cce3a7a97208788'},
    @{Name='rust-mingw'; Hash='75d898804789c12ca969365f0a86d85c3bca1fdb070f0be615ca513a986b2674'}
)
foreach ($component in $components) {
    $package = "$($component.Name)-1.98.1-x86_64-pc-windows-gnu"
    $archive = Join-Path $DownloadRoot "$package.tar.xz"
    if (!(Test-Path $archive) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $component.Hash) {
        Write-Host "Downloading official Rust component $($component.Name)"
        & curl.exe --fail --location --silent --show-error --connect-timeout 15 --max-time 300 --retry 2 `
            "https://static.rust-lang.org/dist/2026-09-03/$package.tar.xz" --output $archive
        if ($LASTEXITCODE) { throw "Download failed for $package (exit=$LASTEXITCODE)" }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $component.Hash) {
        throw "Rust component SHA256 mismatch: $archive"
    }
    $extract = Join-Path $DownloadRoot $package
    if (!(Test-Path (Join-Path $extract 'components'))) {
        & tar.exe -xf $archive -C $DownloadRoot
        if ($LASTEXITCODE) { throw "Extract failed: $archive" }
    }
    foreach ($entry in (Get-Content -LiteralPath (Join-Path $extract 'components'))) {
        $payload = Join-Path $extract $entry
        foreach ($directory in @('bin','lib','share','etc')) {
            $source = Join-Path $payload $directory
            if (Test-Path $source) { Copy-Item -LiteralPath $source -Destination $ToolRoot -Recurse -Force }
        }
    }
    Write-Host "Verified and installed $package"
}
& (Join-Path $ToolRoot 'bin/rustc.exe') --version
if ($LASTEXITCODE) { throw 'Rust compiler failed to start' }
& (Join-Path $ToolRoot 'bin/cargo.exe') --version
if ($LASTEXITCODE) { throw 'Cargo failed to start' }
Write-Host "Local Rust toolchain: $ToolRoot"
