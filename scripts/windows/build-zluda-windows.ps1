[CmdletBinding()]
param([string]$SourceRoot, [string]$LlvmBuildRoot, [string]$InstallRoot, [switch]$FetchOnly, [switch]$CacheTestsOnly, [int]$Jobs = 4)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$SourceRoot) { $SourceRoot = Join-Path $repo 'external/ZLUDA' }
if (!$LlvmBuildRoot) { $LlvmBuildRoot = Join-Path $repo 'build/zluda-llvm-windows-gnu' }
if (!$InstallRoot) { $InstallRoot = Join-Path $repo 'dist/zluda-windows-native' }
$rust = Join-Path $repo '.tools/rust-1.98.1-gnu'
$toolchain = Join-Path $repo '.tools/llvm-mingw-20260922-ucrt-x86_64'
$target = Join-Path $repo 'build/zluda-rust-windows-gnu'
New-Item -ItemType Directory -Force $target,$InstallRoot | Out-Null
$settings = @{
    PATH="$(Join-Path $rust 'bin');$(Join-Path $toolchain 'bin');$(Join-Path $repo '.tools/python/cmake/data/bin');$(Join-Path $repo '.tools/python/bin');C:\Program Files\Git\usr\bin;$env:PATH"
    CARGO_HOME=(Join-Path $repo '.tools/cargo-home')
    CARGO_TARGET_DIR=$target
    CARGO_BUILD_JOBS="$Jobs"
    RUSTC=(Join-Path $rust 'bin/rustc.exe')
    CC=(Join-Path $toolchain 'bin/x86_64-w64-mingw32-clang.exe')
    CXX=(Join-Path $toolchain 'bin/x86_64-w64-mingw32-clang++.exe')
    AR=(Join-Path $toolchain 'bin/llvm-ar.exe')
    CXXSTDLIB='c++'
    LIBRARY_PATH=(Join-Path $rust 'lib/rustlib/x86_64-pc-windows-gnu/lib/self-contained')
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=(Join-Path $toolchain 'bin/x86_64-w64-mingw32-clang.exe')
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS='-C link-arg=-static'
    ZLUDA_LLVM_BUILD_DIR=$LlvmBuildRoot
}
$previous = @{}
foreach ($name in $settings.Keys) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name,'Process')
    [Environment]::SetEnvironmentVariable($name,$settings[$name],'Process')
}
Push-Location $SourceRoot
try {
    $cargo = Join-Path $rust 'bin/cargo.exe'
    if (!(Test-Path $cargo)) { throw 'Run setup-rust-toolchain.ps1 first.' }
    if ($CacheTestsOnly) {
        $ErrorActionPreference = 'Continue'
        & $cargo test --locked --release --target x86_64-pc-windows-gnu -p zluda_cache `
            2>&1 | Tee-Object (Join-Path $target 'cache-tests.log')
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE) { throw "ZLUDA cache regression tests failed ($LASTEXITCODE)" }
        return
    }
    if ($FetchOnly) {
        $ErrorActionPreference = 'Continue'
        & $cargo fetch --locked --target x86_64-pc-windows-gnu 2>&1 | Tee-Object (Join-Path $target 'fetch.log')
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE) { throw "Cargo fetch failed ($LASTEXITCODE)" }
        return
    }
    if (!(Test-Path (Join-Path $LlvmBuildRoot 'bin/llvm-config.exe'))) { throw 'Run build-zluda-llvm.ps1 first.' }
    foreach ($variant in @('','_constrained','_w64','_constrained_w64')) {
        if (!(Test-Path "ptx/lib/zluda_ptx_impl$variant.bc")) { throw 'Run build-zluda-helpers.ps1 first.' }
    }
    $ErrorActionPreference = 'Continue'
    & $cargo build --locked --release --target x86_64-pc-windows-gnu -p zluda -p nvapi -p zluda_trace `
        2>&1 | Tee-Object (Join-Path $target 'build.log')
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE) { throw "ZLUDA build failed ($LASTEXITCODE)" }
    $ErrorActionPreference = 'Continue'
    & $cargo build --locked --release --target x86_64-pc-windows-gnu -p ptx --example d4r_emit `
        2>&1 | Tee-Object (Join-Path $target 'emit-build.log')
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE) { throw "d4r_emit build failed ($LASTEXITCODE)" }
    $release = Join-Path $target 'x86_64-pc-windows-gnu/release'
    Copy-Item -LiteralPath (Join-Path $release 'nvcuda.dll') -Destination $InstallRoot -Force
    Copy-Item -LiteralPath (Join-Path $release 'nvapi64.dll') -Destination $InstallRoot -Force
    # The C++ dependencies select libc++ dynamically even when Rust requests
    # static GNU support libraries. Keep these portable runtime DLLs local.
    foreach ($runtime in @('libc++.dll', 'libunwind.dll')) {
        Copy-Item -LiteralPath (Join-Path $toolchain "bin/$runtime") -Destination $InstallRoot -Force
    }
    New-Item -ItemType Directory -Force (Join-Path $InstallRoot 'trace') | Out-Null
    Copy-Item -LiteralPath (Join-Path $release 'zluda_trace.dll') -Destination (Join-Path $InstallRoot 'trace/nvcuda.dll') -Force
    Copy-Item -LiteralPath (Join-Path $release 'examples/d4r_emit.exe') -Destination $InstallRoot -Force
    @{base='ee2f25a180099fa42f36b2346732e1f2470a03ad'; target='x86_64-pc-windows-gnu';
        sourceCommit=(& git rev-parse HEAD); workingTreeDirty=[bool](& git status --porcelain);
        rust='1.98.1'; llvm=(& (Join-Path $LlvmBuildRoot 'bin/llvm-config.exe') --version);
        files=@(Get-ChildItem -LiteralPath $InstallRoot -Filter '*.dll' | Get-FileHash -Algorithm SHA256)} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $InstallRoot 'build-info.json') -Encoding UTF8
    Write-Host "Native Windows d4r ZLUDA runtime: $InstallRoot"
} finally {
    Pop-Location
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
}
