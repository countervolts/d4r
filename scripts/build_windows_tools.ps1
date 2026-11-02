param(
    [string]$MinGW = 'C:/msys64/mingw64',   # MSYS2 MinGW-w64 (pacman -S mingw-w64-x86_64-gcc)
    [string]$Rocm = $env:D4R_ROCM_DIR,      # AMD HIP SDK; default: the newest C:/ROCm/<version>
    [string]$NgxInclude = $env:NVSDK_NGX_INCLUDE,
    [string]$Out = 'build',
    [switch]$Kernels,                       # also kernels/build.sh k for this GPU (needs Git Bash)
    [string]$Arch = ''                      # GPU target for -Kernels, default: the installed GPU
)
# Native Windows build of the analysis tools: tools/ with MinGW-w64 (the toolchain the Linux scripts
# cross-compile with; the harnesses need its headers) and the engine runners with Visual Studio.
# Outputs: $Out/tools (replays, benchmarks, probes, harnesses, kvk, evk) and $Out/engine-win/Release
# (d4r-k, d4r-m, d4r-rr, d4r-e, d4r-k-d3d12). See docs/windows-tools.md.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (-not (Test-Path -LiteralPath "$MinGW/bin/g++.exe")) {
        throw "MinGW-w64 g++ not found in $MinGW/bin. Install MSYS2 and 'pacman -S mingw-w64-x86_64-gcc', or pass -MinGW."
    }
    if (-not $Rocm) {
        $Rocm = Get-ChildItem -Directory 'C:/ROCm' -ErrorAction SilentlyContinue |
            Where-Object { Test-Path "$($_.FullName)/include/hip/hip_runtime_api.h" } |
            Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName
    }
    $hip = @()
    if ($Rocm) { $hip = @("-DD4R_ROCM_DIR=$Rocm") } else { Write-Warning 'No HIP SDK found: HIP tools are skipped' }
    $ngx = @()
    if ($NgxInclude) { $ngx = @("-DNVSDK_NGX_INCLUDE=$NgxInclude") }

    # MinGW's compilers need their own DLLs on PATH while they run
    $env:PATH = "$MinGW/bin;$env:PATH"
    cmake -S tools -B "$Out/tools" -G Ninja "-DCMAKE_C_COMPILER=$MinGW/bin/gcc.exe" `
        "-DCMAKE_CXX_COMPILER=$MinGW/bin/g++.exe" -DCMAKE_BUILD_TYPE=Release @hip @ngx
    if ($LASTEXITCODE) { throw 'tools: configure failed' }
    cmake --build "$Out/tools"
    if ($LASTEXITCODE) { throw 'tools: build failed' }

    cmake -S engine -B "$Out/engine-win" @hip
    if ($LASTEXITCODE) { throw 'engine: configure failed (needs Visual Studio and the Vulkan SDK)' }
    cmake --build "$Out/engine-win" --config Release
    if ($LASTEXITCODE) { throw 'engine: build failed' }

    if ($Kernels) {
        if (-not $Rocm) { throw '-Kernels needs the HIP SDK (-Rocm)' }
        if (-not $Arch) { $Arch = (& "$Rocm/lib/llvm/bin/amdgpu-arch.exe" | Select-Object -First 1) }
        $bash = (Get-Command bash -ErrorAction SilentlyContinue).Source
        if (-not $bash) { throw '-Kernels needs Git Bash (bash on PATH)' }
        $env:D4R_ROCM_DIR = $Rocm
        $env:D4R_GPU_ARCH = $Arch
        & $bash kernels/build.sh k "$Out/kernels-$Arch"
        if ($LASTEXITCODE) { throw 'kernels/build.sh k failed' }
    }
    Write-Output "Tools: $Out/tools; engine runners: $Out/engine-win/Release"
} finally {
    Pop-Location
}
