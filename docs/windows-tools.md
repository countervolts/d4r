# Analysis tools on native Windows

The replays, benchmarks, probes and harnesses in `tools/`, the engine runners and the native kernel builds
run on Windows as well as Linux. On Linux the `scripts/build_*.sh` / `run_*.sh` pairs still apply. On Windows
two scripts replace them: one builds everything, and the other runs a built tool with ZLUDA.

## Prerequisites

| Component | Used for |
|---|---|
| [MSYS2](https://www.msys2.org) with `pacman -S mingw-w64-x86_64-gcc` (`C:/msys64/mingw64`) | `tools/`. The same MinGW-w64 toolchain the Linux scripts cross-compile with; the D3D12/Vulkan harnesses need its headers |
| Visual Studio 2022 (or 18) with the C++ workload | the engine runners (`engine/CMakeLists.txt`); its MSVC 2022 toolset is also what the HIP SDK's clang compiles against |
| Vulkan SDK | Vulkan probes, `kvk`, `evk`, the engine, `glslangValidator` |
| AMD HIP SDK (`C:/ROCm/<version>`, found automatically or with `D4R_ROCM_DIR`) | HIP tools, MicroCUDA, the engine's HIP network, `kernels/build.sh` |
| CMake, Ninja, Python 3, Git Bash | builds; Git Bash runs the kernel build scripts |
| ZLUDA for Windows (`nvcuda.dll`) | at run time, for the CUDA-driver tools |
| NVIDIA's `nvsdk_ngx*.h` (`NVSDK_NGX_INCLUDE`) | optional: the `ngx_*_probe` tools |

## Build

```powershell
scripts/build_windows_tools.ps1            # build/tools and build/engine-win/Release
scripts/build_windows_tools.ps1 -Kernels   # also kernels/build.sh k for the installed GPU (build/kernels-<arch>)
```

The configure step prints which tools were built and why any were skipped. The same builds by hand:

```powershell
cmake -S tools -B build/tools -G Ninja -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe
cmake --build build/tools                  # with C:/msys64/mingw64/bin on PATH
cmake -S engine -B build/engine-win && cmake --build build/engine-win --config Release
```

A Visual Studio configure of `tools/` also works. It skips `replay_dlss_fp8_ops` (it uses `_Float16`) and the
programs written for MinGW's headers (`d3d12_dlss_harness`, `vulkan_dlss_harness`, `rr_backend_check`,
`fsr_rr_probe`, `d3d12_hip_interop_probe`, the `ngx_*_probe` tools).

From Git Bash, `kernels/build.sh`, `kernels/native/build_native.sh`, `scripts/build_d4r_motion_kernels.sh` and
`scripts/build_zluda_ptx_helpers.sh` run unchanged with `D4R_ROCM_DIR=/c/ROCm/<version>` and
`D4R_GPU_ARCH=gfx1201` (or your GPU's target).

## Run

```powershell
scripts/run_windows_tool.ps1 replay_dlss_kernel -Zluda C:/zluda/target/release CAPTURE.fatbin KERNEL
scripts/run_windows_tool.ps1 bench_dlss_kernel -Zluda C:/zluda/target/release FATBIN REPLAY_DIR 100
scripts/run_windows_tool.ps1 hip_vk_interop_probe
scripts/run_windows_tool.ps1 d4r-k MODEL_DIR frames.case - 20
```

`-Zluda` (or `ZLUDA_ROOT` / `D4R_ZLUDA_DIR`) puts ZLUDA's directory first on `PATH`; HIP comes from the AMD
driver. The Python drivers take the runner as an argument (`kvk`, `evk`, `--runner`) or find
`build/tools/<tool>.exe` themselves (`engine/rr/zluda_replay.py`; there `--zluda` names ZLUDA's directory
on both systems).

## What differs from Linux

- **CUDA driver library.** The tools still call `dlopen("libcuda.so")`. On Windows, `tools/d4r_dlfcn.h` maps
  that to `LoadLibrary("nvcuda.dll")`, and `libamdhip64.so*` to `amdhip64_7.dll`. A path is used as given, so
  `D4R_BENCH_LIBCUDA=build/tools/d4r_microcuda.dll` selects MicroCUDA.
- **Vulkan memory shared with HIP.** Linux exports an opaque file descriptor; Windows exports an opaque NT
  handle (`VK_KHR_external_memory_win32`). This covers `hip_vk_interop_probe`, `hip_vk_texture_stale_repro`
  (`tools/d4r_vk_hip_memory.h`) and the engine's HIP network in `d4r-k` / `d4r-m` (`engine/vk_external.h`).
- **hipcc targets.** On Windows, hipcc does not detect the GPU. Without `--offload-arch`, kernel launches fail
  and a texture sample reads zeros. The CMake build passes `amdgpu-arch`'s answer (`D4R_GPU_ARCH` overrides
  it). `hip_vk_texture_stale_repro` now checks the launch, so a missing code object fails instead of
  reporting `MISMATCH`.
- **HIP SDK and Visual Studio 18.** ROCm 7.14's clang rejects Visual Studio 18's `<cmath>`. The kernel scripts
  (through `scripts/hip_windows_env.sh`) and the CMake build (`D4R_HIP_VCTOOLS`) point it at the Visual Studio
  2022 toolset unless `VCToolsInstallDir` is already set.
- **Device libraries.** The HIP SDK keeps them in `lib/llvm/amdgcn/bitcode`. The kernel scripts use that path
  when `amdgcn/bitcode` is absent.
- **Texture kernels and LLVM versions.** ROCm 7.14's clang is LLVM 23; ZLUDA's `d4r_emit` reads LLVM 22
  bitcode (`Unknown attribute kind (106)`). `kernels/tex/build_tex.sh` rewrites the three LLVM 23 constructs its
  IR uses and assembles it with `D4R_LLVM_AS`, the `llvm-as.exe` of ZLUDA's own LLVM build
  (`target/release/build/llvm_zluda-*/out/build/bin`). `scripts/package_release_windows.sh` finds it itself.
- **GPU target of the CUDA bridge.** Windows has no KFD topology: the native Windows `nvcuda.dll` reads each
  GPU's target from HIP (`hipGetDevicePropertiesR0600`) to pick its `d4r\kernels` folder.
- **MicroCUDA** builds as `d4r_microcuda.dll` and hashes packs with BCrypt instead of OpenSSL.
- **Staging shaders.** `scripts/build_vulkan_staging.py` compiles them (the `.sh` now calls it).

## Linux only

These have no Windows counterpart:

- the Wine/Proton launchers and runners (`run_*_proton.sh`, `d4r_play.sh`, `run_ngx_*_probe.sh`);
- the Wine CUDA bridge, vkd3d-proton and GLIBC release builds;
- `tools/microcuda/arch_spoof.c` (an `LD_PRELOAD` shim);
- the tests that run Windows programs under Wine (`test_asi_setup`, part of `test_features`; they skip).

## Tests

The `tests/` suite runs from PowerShell or Git Bash with MSYS2's MinGW compilers first on `PATH` (the tests
compile C/C++ snippets with `cc` / `c++`) and a Python that has numpy. `tests/platform_support.py` makes the
shell steps run in Git Bash, not WSL's `bash`. The opt-in GPU tests need the runners and kernels built first
(`scripts/build_windows_tools.ps1`, `scripts/build_d4r_motion_kernels.sh`, `scripts/build_vulkan_staging.py`):

```sh
export PATH="/c/msys64/mingw64/bin:$PATH"
export D4R_RUN_GPU_TESTS=1 D4R_ROCM_DIR='C:\ROCm\7.14' D4R_GPU_ARCH=gfx1201 \
       D4R_DUMP_RUNNER="$PWD/build/tools/dump_runner.exe" D4R_EVK="$PWD/build/tools/evk.exe" \
       D4R_WIRE_TEST_KERNEL="$PWD/build/wine-nvcuda/generated/gfx1201.hsaco"
python -m unittest discover -s tests -v
```

Three tests fail on Windows. Each failure is real; none comes from the test harness:

- `test_latency_optimizations` (native callback cancellation). MSYS2's winpthreads rejects
  `pthread_condattr_setclock(CLOCK_MONOTONIC)` (EINVAL), so `d4rNgxTimelineCreate` in the native Windows
  CUDA bridge always fails. The NGX timeline needs a Windows wait path before it works there.
- `test_full_and_clean_release_archives`. The release packager needs `zip`, which Git Bash lacks. The
  packager builds the Linux release in any case.
- `test_routing_separate_from_file_presence`, the Wine `Z:` path case. Wine's drive mapping only exists
  under Proton.

## Verified

Checked on Windows 11, RX 9070 XT (gfx1201), AMD HIP SDK 7.14 and MSYS2 GCC 16.2:

- Every `tools/` target builds; the `ngx_*_probe` tools need NVIDIA's headers, so they were not built.
- These pass: `vulkan_staging_probe`, `vulkan_output_conversion_probe`, `vulkan_depth_copy_probe`,
  `hip_vk_interop_probe` and `hip_vk_texture_stale_repro` (image and buffer). Texture sampling follows
  Vulkan's writes.
- `bench_cuda_copies` runs through Windows ZLUDA.
- `kernels/build.sh k` builds the eleven K layers, and `build_d4r_motion_kernels.sh` builds every target.
- `d4r-k` (Visual Studio, HIP-enabled build) runs synthetic 1080p→4K frames on the Vulkan network. The
  `hip` path was checked through device creation and memory export only: a HIP model (`compile_k.py --hip`)
  needs a real capture.
