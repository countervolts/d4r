[CmdletBinding()]
param([string]$SourceRoot, [string]$BuildRoot, [int]$Jobs = 4)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$SourceRoot) { $SourceRoot = Join-Path $repo 'external/ZLUDA' }
if (!$BuildRoot) { $BuildRoot = Join-Path $repo 'build/zluda-llvm-windows-gnu' }
$toolchain = Join-Path $repo '.tools/llvm-mingw-20260922-ucrt-x86_64'
$cmake = Join-Path $repo '.tools/python/cmake/data/bin/cmake.exe'
$ninja = Join-Path $repo '.tools/python/bin/ninja.exe'
if (!(Test-Path $ninja)) { $ninja = Join-Path $repo '.tools/python/ninja/data/bin/ninja.exe' }
if (!(Test-Path (Join-Path $SourceRoot 'ext/llvm-project/llvm/CMakeLists.txt'))) {
    throw 'Run git submodule update --init --depth 1 ext/llvm-project in the ZLUDA checkout.'
}
New-Item -ItemType Directory -Force $BuildRoot | Out-Null
& (Join-Path $PSScriptRoot 'prepare-zluda-llvm-source.ps1') -LlvmSourceRoot (Join-Path $SourceRoot 'ext/llvm-project')
$oldPath = $env:PATH
try {
    $env:PATH = "$(Join-Path $toolchain 'bin');$oldPath"
    $arguments = @('-S', (Join-Path $SourceRoot 'ext/llvm-project/llvm'), '-B', $BuildRoot,
        '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja", '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_C_COMPILER=$(Join-Path $toolchain 'bin/x86_64-w64-mingw32-clang.exe')",
        "-DCMAKE_CXX_COMPILER=$(Join-Path $toolchain 'bin/x86_64-w64-mingw32-clang++.exe')",
        '-DLLVM_TARGETS_TO_BUILD=AMDGPU', '-DLLVM_ENABLE_PROJECTS=llvm;lld',
        '-DLLVM_ENABLE_WARNINGS=OFF', '-DLLVM_ENABLE_TERMINFO=OFF', '-DLLVM_ENABLE_LIBXML2=OFF',
        '-DLLVM_ENABLE_LIBEDIT=OFF', '-DLLVM_ENABLE_LIBPFM=OFF', '-DLLVM_ENABLE_ZLIB=OFF',
        '-DLLVM_ENABLE_ZSTD=OFF', '-DLLVM_INCLUDE_BENCHMARKS=OFF', '-DLLVM_INCLUDE_EXAMPLES=OFF',
        '-DLLVM_INCLUDE_TESTS=OFF', '-DLLVM_BUILD_TESTS=OFF', '-DLLVM_BUILD_TOOLS=OFF',
        '-DLLVM_OPTIMIZED_TABLEGEN=ON', '-DLLVM_PARALLEL_LINK_JOBS=1')
    $ErrorActionPreference = 'Continue'
    & $cmake @arguments 2>&1 | Tee-Object (Join-Path $BuildRoot 'configure.log')
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE) { throw "LLVM configure failed ($LASTEXITCODE)" }
    $ErrorActionPreference = 'Continue'
    & $cmake --build $BuildRoot --parallel $Jobs --target llvm-config llvm-libraries lldCommon lldELF `
        2>&1 | Tee-Object (Join-Path $BuildRoot 'build.log')
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE) { throw "LLVM build failed ($LASTEXITCODE)" }
    & (Join-Path $BuildRoot 'bin/llvm-config.exe') --version
    if ($LASTEXITCODE) { throw 'llvm-config verification failed' }
    @{sourceBase=(& git -C (Join-Path $SourceRoot 'ext/llvm-project') rev-parse HEAD).Trim(); fullSourceDeltaVerified=$true;
      sourcePatches=@(Get-ChildItem (Join-Path $repo 'patches/zluda-llvm') -Filter *.patch | ForEach-Object {
        @{name=$_.Name; sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})} |
        ConvertTo-Json -Depth 5 | Set-Content (Join-Path $BuildRoot 'd4r-llvm-build-info.json') -Encoding UTF8
    Write-Host "Native Windows LLVM: $BuildRoot"
} finally { $env:PATH = $oldPath }
