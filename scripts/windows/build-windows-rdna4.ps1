[CmdletBinding()]
param(
    [string]$HipRoot = $env:HIP_PATH,
    [ValidateSet('stable', 'therock')][string]$RuntimeProfile = 'stable',
    [string]$GpuArch = 'gfx1201',
    [string]$ZludaRoot,
    [string]$ToolchainRoot,
    [string]$BuildDirectory,
    [string]$InstallDirectory,
    [switch]$KernelsOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
[void](Get-D4RGpuTargetInfo $GpuArch)
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if ($RuntimeProfile -eq 'therock') {
    if (!$PSBoundParameters.ContainsKey('HipRoot')) { $HipRoot = Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
    if (!$BuildDirectory) { $BuildDirectory = Join-Path $repo 'build/windows-rdna4-therock' }
    if (!$InstallDirectory) { $InstallDirectory = Join-Path $repo 'dist/windows-rdna4-therock' }
}
if (!$HipRoot) { $HipRoot = 'C:\Program Files\AMD\ROCm\7.2' }
if (!$BuildDirectory) { $BuildDirectory = Join-Path $repo 'build/windows-rdna4' }
if (!$InstallDirectory) { $InstallDirectory = Join-Path $repo 'dist/windows-rdna4-diagnostics' }
if ($GpuArch -ne 'gfx1201') {
    if (!$PSBoundParameters.ContainsKey('BuildDirectory')) { $BuildDirectory += "-$GpuArch" }
    if (!$PSBoundParameters.ContainsKey('InstallDirectory')) { $InstallDirectory += "-$GpuArch" }
}
if ($KernelsOnly) {
    if (!$PSBoundParameters.ContainsKey('BuildDirectory')) { $BuildDirectory=Join-Path $repo "build/windows-kernels-$GpuArch" }
    if (!$PSBoundParameters.ContainsKey('InstallDirectory')) { $InstallDirectory=Join-Path $repo "dist/windows-kernels-$GpuArch" }
}
$HipRoot = [IO.Path]::GetFullPath($HipRoot).TrimEnd('\', '/')
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory).TrimEnd('\', '/')
$InstallDirectory = [IO.Path]::GetFullPath($InstallDirectory).TrimEnd('\', '/')
if (!$ZludaRoot -and (Test-Path (Join-Path $repo '.tools/zluda/zluda/nvcuda.dll'))) {
    $ZludaRoot = Join-Path $repo '.tools/zluda/zluda'
}
if (!$ToolchainRoot -and (Test-Path (Join-Path $repo '.tools/llvm-mingw-20260922-ucrt-x86_64'))) {
    $ToolchainRoot = Join-Path $repo '.tools/llvm-mingw-20260922-ucrt-x86_64'
}
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
$cmake = if ($cmakeCommand) { $cmakeCommand.Source } else { Join-Path $repo '.tools/python/cmake/data/bin/cmake.exe' }
$ninjaCommand = Get-Command ninja -ErrorAction SilentlyContinue
$ninja = if ($ninjaCommand) { $ninjaCommand.Source } else { Join-Path $repo '.tools/python/ninja/data/bin/ninja.exe' }
if (!(Test-Path $ninja)) { $ninja = Join-Path $repo '.tools/python/bin/ninja.exe' }
if (!(Test-Path $cmake) -or !(Test-Path $ninja)) {
    throw 'CMake >=3.24 and Ninja required. See docs/windows-rdna4-port.md for local tool setup.'
}
New-Item -ItemType Directory -Force $BuildDirectory | Out-Null
$minhook = Join-Path $repo 'external/MinHook'
if (!$KernelsOnly) {
    if (!(Test-Path (Join-Path $minhook 'include/MinHook.h'))) {
        & git clone --depth 1 --branch v1.3.4 https://github.com/TsudaKageyu/minhook.git $minhook
        if ($LASTEXITCODE) { throw 'MinHook 1.3.4 source download failed.' }
    }
    $minhookCommit = (& git -C $minhook rev-parse HEAD).Trim()
    if ($minhookCommit -ne 'c3fcafdc10146beb5919319d0683e44e3c30d537') {
        throw "Expected audited MinHook 1.3.4 source, found $minhookCommit"
    }
}
$originalPath = $env:PATH
try {
    $configure = @('--fresh', '-S', $repo, '-B', $BuildDirectory, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
        "-DCMAKE_MAKE_PROGRAM=$ninja", "-DD4R_HIP_ROOT=$HipRoot", "-DD4R_GPU_ARCH=$GpuArch", "-DCMAKE_INSTALL_PREFIX=$InstallDirectory")
    $configure += "-DD4R_KERNELS_ONLY=$([int][bool]$KernelsOnly)"
    if ($ToolchainRoot) {
        $compiler = Join-Path $ToolchainRoot 'bin/x86_64-w64-mingw32-clang++.exe'
        if (!(Test-Path $compiler)) { throw "Compiler missing: $compiler" }
        $env:PATH = "$(Join-Path $ToolchainRoot 'bin');$originalPath"
        $configure += "-DCMAKE_CXX_COMPILER=$compiler"
        $configure += "-DCMAKE_C_COMPILER=$(Join-Path $ToolchainRoot 'bin/x86_64-w64-mingw32-clang.exe')"
    }
    if ($ZludaRoot) { $configure += "-DD4R_ZLUDA_ROOT=$ZludaRoot" }
    $ErrorActionPreference = 'Continue'
    & $cmake @configure 2>&1 | Tee-Object (Join-Path $BuildDirectory 'configure.log')
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE) { throw "CMake configure exit=$LASTEXITCODE" }
    $ErrorActionPreference = 'Continue'
    $buildArgs=@('--build', $BuildDirectory, '--parallel', '4')
    if ($KernelsOnly) { $buildArgs+=@('--target', 'd4r_native_compile', 'd4r_probe_code_object', 'd4r_pixel_code_object', 'd4r_wmma_code_object') }
    & $cmake @buildArgs 2>&1 | Tee-Object (Join-Path $BuildDirectory 'build.log')
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE) { throw "Build exit=$LASTEXITCODE" }
    $ErrorActionPreference = 'Continue'
    $installArgs=@('--install', $BuildDirectory)
    if ($KernelsOnly) { $installArgs+=@('--component', 'kernels') }
    & $cmake @installArgs 2>&1 | Tee-Object (Join-Path $BuildDirectory 'install.log')
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE) { throw "Install exit=$LASTEXITCODE" }
    foreach ($object in Get-ChildItem -LiteralPath $BuildDirectory -Filter '*.hsaco') {
        Assert-D4RCodeObjectTarget $object.FullName $GpuArch
    }
    Write-Host "Built $GpuArch artifacts: $InstallDirectory"
} finally { $env:PATH = $originalPath }
