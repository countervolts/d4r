[CmdletBinding()]
param(
    [string]$HipRoot = $env:HIP_PATH,
    [ValidateSet('gfx1100','gfx1101','gfx1102','gfx1103','gfx1150','gfx1151','gfx1152','gfx1153','gfx1154','gfx1200','gfx1201')]
    [string]$GpuArch = 'gfx1101',
    [int]$Parallel = [Math]::Max(1, [Environment]::ProcessorCount - 1)
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$HipRoot) { throw 'Pass -HipRoot or set HIP_PATH to the Windows HIP SDK root.' }
$HipRoot = [IO.Path]::GetFullPath($HipRoot)
if (!(Test-Path -LiteralPath (Join-Path $HipRoot 'include/hip/hip_runtime_api.h'))) {
    throw "HIP headers not found under $HipRoot"
}
& (Join-Path $PSScriptRoot 'setup-native-engine.ps1')
if ($LASTEXITCODE) { throw "Native engine dependency setup failed: $LASTEXITCODE" }

$tools = Join-Path $repo '.tools'
$cmake = Join-Path $tools 'python/cmake/data/bin/cmake.exe'
$ninja = Join-Path $tools 'python/ninja/data/bin/ninja.exe'
if (!(Test-Path -LiteralPath $cmake)) { $cmake = Join-Path $tools 'python/bin/cmake.exe' }
if (!(Test-Path -LiteralPath $ninja)) { $ninja = Join-Path $tools 'python/bin/ninja.exe' }
if (!(Test-Path -LiteralPath $cmake) -or !(Test-Path -LiteralPath $ninja)) {
    throw 'Pinned CMake/Ninja missing. Run scripts/windows/setup-windows-tools.ps1 first.'
}
$toolchain = Join-Path $tools 'llvm-mingw-20260922-ucrt-x86_64/bin'
$cCompiler = Join-Path $toolchain 'x86_64-w64-mingw32-clang.exe'
$cxxCompiler = Join-Path $toolchain 'x86_64-w64-mingw32-clang++.exe'
if (!(Test-Path -LiteralPath $cCompiler) -or !(Test-Path -LiteralPath $cxxCompiler)) {
    throw 'Pinned llvm-mingw compiler missing. Run scripts/windows/setup-windows-tools.ps1 first.'
}
$oldPath = $env:PATH
$env:PATH = "$toolchain;$oldPath"
try {
    $build = Join-Path $repo "build/native-engine-windows-$GpuArch"
    $headers = Join-Path $tools 'Vulkan-Headers-v1.4.341'
    $volk = Join-Path $tools 'volk'
    & $cmake --fresh -S $repo -B $build -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" `
        "-DCMAKE_C_COMPILER=$cCompiler" "-DCMAKE_CXX_COMPILER=$cxxCompiler" `
        "-DD4R_HIP_ROOT=$HipRoot" "-DD4R_GPU_ARCH=$GpuArch" `
        '-DD4R_BUILD_NATIVE_ENGINE=ON' "-DD4R_VULKAN_HEADERS_ROOT=$headers" "-DD4R_VOLK_ROOT=$volk" `
        "-DD4R_MINHOOK_ROOT=$(Join-Path $repo 'external/MinHook')"
    if ($LASTEXITCODE) { throw "Native engine configure failed: $LASTEXITCODE" }
    & $cmake --build $build --target d4r_engine_k d4r_engine_m d4r_engine_rr --parallel $Parallel
    if ($LASTEXITCODE) { throw "Native engine build failed: $LASTEXITCODE" }
} finally { $env:PATH = $oldPath }
Write-Host "Built native runners in ${build}: d4r-k.exe, d4r-m.exe, d4r-rr.exe."
