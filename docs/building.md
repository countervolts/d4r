<!-- Modified in this fork for CUDA Ray Reconstruction support and validation (2026). -->
# Building d4r

These steps produce the pieces the launcher needs: a patched ZLUDA, a patched vkd3d-proton, the NGX shim and CUDA bridge, a staged runtime directory with your NVIDIA files, and the native kernels. The versions below are the ones that were tested; others may need changes.

| Component | Tested version |
|---|---|
| ZLUDA | `ee2f25a` (upstream), plus `patches/zluda/0002` through `0014` in order |
| vkd3d-proton | `3dfc6f07` (the base GE-Proton11-3 ships), plus `patches/vkd3d-proton/0001` through `0005` in order |
| ROCm | 7.2 (HIP runtime, clang, device libraries) |
| Proton | GE-Proton11-3 (with its OptiScaler integration) |
| DLSS | `nvngx_dlss.dll` 310.7.0 |

## 1. ZLUDA

```sh
git clone https://github.com/vosen/ZLUDA zluda && cd zluda
git checkout ee2f25a
git submodule update --init --recursive
git lfs pull
for p in 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012 0013 0014; do git apply /path/to/d4r/patches/zluda/$p-*.patch; done
# rebuild the device helpers the patches changed (ptx/lib/zluda_ptx_impl*.bc)
ZLUDA_SOURCE_ROOT=$PWD ROCM_ROOT=/opt/rocm /path/to/d4r/scripts/build_zluda_ptx_helpers.sh
LIBRARY_PATH=/opt/rocm/lib cargo build --release -p zluda
LIBRARY_PATH=/opt/rocm/lib cargo build --release -p ptx --example d4r_emit
mkdir -p ~/.cache/d4r-zluda-current
cp target/release/libnvcuda.so ~/.cache/d4r-zluda-current/
ln -sf libnvcuda.so ~/.cache/d4r-zluda-current/libcuda.so
```

Build the release ZLUDA library on a system or in a container with GLIBC 2.41 or older. A binary
linked on a newer system can require newer GLIBC symbols even when the ZLUDA source does not need new
GLIBC features. Before packaging, check it with
`scripts/check_glibc_compat.sh 2.41 /path/to/libnvcuda.so`. The release packager runs the same
check on ZLUDA, the Wine CUDA bridge, and every bundled ROCm library. Test the rebuilt library
with the D3D12 harness and in a game before publishing a release.

The launcher looks for ZLUDA in `D4R_ZLUDA_DIR` (default `~/.cache/d4r-zluda-current`, or `ZludaDir` in d4r.ini).

What the patches add:

- **0002**: the CUDA driver API surface NGX and DLSS use (arrays, textures, surfaces, 1010102 formats, launch parameter buffers), the PTX features DLSS kernels need, and module dumps for debugging (`D4R_ZLUDA_DUMP_DIR`).
- **0003**: an implicit 256-thread launch bound for kernels without PTX bounds (removes massive register spilling, `D4R_ZLUDA_IMPLICIT_MAX_BLOCK`) and f16 tensor-core MMA on RDNA3 WMMA (`D4R_ZLUDA_WMMA`).
- **0004**: the native kernel override hook (`D4R_ZLUDA_NATIVE_DIR`, see [native-kernels.md](native-kernels.md)), FP8 MMA on WMMA (`D4R_ZLUDA_WMMA_FP8`), optional elision of per-instruction denormal mode switches (`D4R_ZLUDA_IGNORE_DENORMAL`), inlined image helpers, and linking extra bitcode into a PTX module (`D4R_ZLUDA_EXTRA_BC`, used by the texture-kernel build).
- **0005**: a null texture object (CUDA handle 0) reads as zeros, as on NVIDIA GPUs, instead of faulting the GPU. DLSS samples absent optional inputs that way in some configurations (low-resolution motion vectors without HDR, as in Ghost of Tsushima).
- **0006**: `m16n8k8` f16 MMAs (DLSS 3 CNN, presets E/F) on RDNA3 WMMA like the k16 ones (`D4R_ZLUDA_WMMA_K8=0` disables it); weight-image slots for native prep kernels (`d4r_prep_key_at`, `d4r_prep_key_slots`); and wave64 compilation for offline texture-kernel builds (`D4R_ZLUDA_WAVE64=1`, with wave64 builds of the helper bitcode). `scripts/build_zluda_ptx_helpers.sh` now writes the `_w64` helper variants too.
- **0007**: gfx12 WMMA layout lowering, optional native e4m3 FP8 WMMA (`D4R_ZLUDA_WMMA_FP8_NATIVE=1`), and an architecture argument for `d4r_emit`. `D4R_ZLUDA_WMMA_LAYOUT=12` on gfx11 is a validation shim, not a release setting.
- **0008**: post-phase kernels for native overrides (`NAME_post1` … `NAME_post8`, see [native-kernels.md](native-kernels.md)). Preset K's dec5 layer runs as four phase kernels, so the K set needs this patch: an older `libnvcuda.so` would skip the phases.
- **0009**: CUDA denoiser PTX support: four-half surface stores, floating-coordinate half texture sampling, streaming-cache store spellings, byte-vector trapping stores (including low-byte truncation from wider registers), finite-saturating half conversion, correctly typed half exponentiation, half unordered comparisons, and atomic reductions. CUDA block barriers use a workgroup release fence before the hardware barrier and an acquire fence afterward. It also fixes strict floating-point division operand selection in the LLVM AMDGPU backend and Cargo build-profile detection. Two LLVM sinking fixes remove repeated instruction-order rebuilding and avoid alias scans when no sink target exists; write tracking and memory-dependency checks remain intact. Initialize the LLVM submodule before applying this patch; rebuild device helpers and ZLUDA afterward.
- **0010**: gfx12 native-FP8 half-to-e4m3 requantization, including the ReLU variant. Hardware packed FP8 conversion replaces the software rounding path only when `D4R_ZLUDA_WMMA_FP8_NATIVE=1` selects gfx12 native FP8. Explicit saturation and sign restoration preserve finite overflow, infinities, signed zero and NaNs. Rebuild both the device helpers and ZLUDA.
- **0011**: gfx12 native FP8 operands stay packed through the NVIDIA-to-WMMA lane gathers, removing fake-half splitting and repacking. Paired and unpaired operations preserve the previous K permutation, accumulation order and FP32-shadow scaling. Unpaired gathers execute in every source lane before zero-padding; masking EXEC around them loses valid source values. Rebuild both the device helpers and ZLUDA.
- **0012**: coalesce complete accumulator-register overwrites into one shadow-layout conversion. Intervening reads, rewritten shadow accesses and basic-block boundaries prevent coalescing; partial writes retain read/modify/write behavior. Gfx12 native-FP8 functions request eight waves per SIMD to limit register pressure. Rebuild ZLUDA and `d4r_emit`.
- **0013**: preserve signs through the native FP8 conversion instead of reconstructing them afterward. Explicitly clamp finite overflow and infinities, and map NaNs to signed infinity before conversion: native infinity produces the required `0x7f`/`0xff`, whereas native NaN always produces `0xff`. Plain and ReLU paths remain exhaustive-table identical to software. Rebuild device helpers and ZLUDA.
- **0014**: a module whose every visible kernel has a native replacement in `D4R_ZLUDA_NATIVE_DIR` (and that declares no globals) is not compiled when DLSS loads it; its PTX is kept and compiled only if a function without a native replacement is requested. DLSS loads every module at feature creation, and on RDNA2 each transformer layer (emulated MMA) took minutes to compile for code that never ran, so the first K frame took about half an hour on a Steam Deck. Rebuild ZLUDA.

When linking on a system with ROCm libraries outside the default search path, include their library directory in `LIBRARY_PATH`. The build also needs the appropriate ROCm link libraries. `CARGO_BUILD_JOBS=8` caps parallel Rust compilation if memory is limited.

## 2. vkd3d-proton

```sh
scripts/build_vkd3d_proton_d4r.sh ~/.cache/d4r-vkd3d-d4r
```

Patches `0001` and `0002` provide same-frame command-list splitting and resource-lifetime tracking; both are required by the shim, which rejects feature creation without the lifetime interface, even when CPU staging or previous-frame output is selected. Patch `0003` adds experimental exportable linear textures, enabled only by `VKD3D_D4R_LINEAR_TEXTURES=1`; patch `0004` (`VKD3D_D4R_LINEAR_RGBA8=WxH`) extends that to the RGBA8 output the shim writes in place (`D4R_SHIM_INPLACE=1`). Patch `0005` enables `VK_EXT_pipeline_robustness`, so the engine's pipelines opt out of buffer robustness (`D4R_ENGINE_BUFFER_ROBUSTNESS=1` keeps it). Point `D4R_VKD3D_DIR` (or `VkD3DDir` in d4r.ini) at the output and update `d3d12.dll` and `d3d12core.dll` together.

Game-local `d3d12.dll` and `d3d12core.dll` override the prefix's copies. Update both if the game already contains an older d4r runtime; changing only `D4R_VKD3D_DIR` does not replace those local overrides.

## 3. Shim, bridge and NVAPI identity

```sh
scripts/build_d4r_nvngx_shim.sh      # build/d4r_nvngx.dll   (needs x86_64-w64-mingw32-g++ and clang-cl)
scripts/build_wine_nvcuda_bridge.sh  # build/wine-nvcuda/    (needs winegcc and winebuild)
```

The launcher builds the small NVAPI identity bridge (`scripts/build_d4r_nvapi_identity.sh`) itself.

## 4. Runtime directory

```sh
scripts/install_d4r_runtime.sh /path/to/_nvngx.dll /path/to/nvngx_dlss.dll
# With the CUDA-capable Ray Reconstruction feature library:
scripts/install_d4r_runtime.sh /path/to/_nvngx.dll /path/to/nvngx_dlss.dll /path/to/nvngx_dlssd.dll
```

This stages the shim, the bridge, the NGX core and the DLSS feature libraries in `D4R_RUNTIME_DIR` (default `~/.local/share/d4r-dlss`, or `RuntimeDir` in d4r.ini). The NVIDIA files come from your own sources, for example an NVIDIA driver package (NGX core) and a game or the DLSS SDK (feature libraries). Ray Reconstruction requires a CUDA-capable `nvngx_dlssd.dll`, such as 310.7; the older D3D12-only denoiser is not a replacement. The installer also places the denoiser beside the calling shim, where the NGX core discovers it.

RR D/E temporal evaluation and separate alpha output have been verified on gfx1201 with native FP8 WMMA (`[Kernels] NativeFp8 = true`). For developer launches without an INI, set `D4R_ZLUDA_WMMA=1`, `D4R_ZLUDA_WMMA_FP8=1` and `D4R_ZLUDA_WMMA_FP8_NATIVE=1`. The scalar FP8 fallback produced nonfinite temporal RR output and is not a validated RR path. Other GPU targets and image-quality parity with RTX hardware remain unverified.

## 5. Native kernels

```sh
D4R_ROCM_DIR=/opt/rocm \
D4R_DLSS_DLL=/path/to/nvngx_dlss.dll \
D4R_ZLUDA_EMIT=/path/to/zluda/target/release/examples/d4r_emit \
kernels/build.sh all kernels/out/native
```

`kernels/out/native` then holds one code object per replaced DLSS kernel; set `NativeKernelDirFast` in d4r.ini (or `D4R_ZLUDA_NATIVE_DIR`) to it. `kernels/build.sh k` or `m` builds only the network layers and needs neither the DLL nor ZLUDA. `kernels/build.sh l` builds L's shared network layers and unfolded texture variants; it needs the DLL and emitter. The texture kernels (`tex`, including L) extract PTX from your DLL into `kernels/extracted/` and compile it offline for `D4R_GPU_ARCH` through `D4R_ZLUDA_EMIT`; that directory and the build output are git-ignored and must not be redistributed. On gfx12, build a second variant with `D4R_NATIVE_FP8=1` and place it in an `<arch>-fp8` folder.

To add the accuracy option to a developer set, repeat the build with `D4R_PREFER_ACCURACY=1` and output `kernels/out/native/accuracy` (multi-target builds: `kernels/accuracy/<arch>` and `<arch>-fp8`). Use a separate empty directory; successfully built accuracy sets receive `d4r-accuracy.txt`. Enable `[Kernels] PreferAccuracy = true` and restart. The option defaults to false and never substitutes the fast set when accuracy binaries are absent.

## 6. Configure and play

The first launch copies `config/d4r.ini.default` to `~/.config/d4r/d4r.ini`. Set the `[Launch]` paths (Proton, the game's compatdata prefix and the game executable), the `[Paths]` above, and the model in `[DLSS]`, then run:

```sh
scripts/d4r_play.sh
```

In the game, pick DLSS as the upscaler (OptiScaler intercepts it). Every setting in the file can also be given as an environment variable, which takes precedence; `D4R_NO_CONFIG=1` ignores the file.

### Cyberpunk 2077 RR launch notes

The optimized RR route was exercised in a loaded Cyberpunk 2077 2.31 save on RX 9070 XT (gfx1201), with preset E, path tracing and Ultra Performance (640×360 → 1920×1080). The user also verified gameplay. This is a functional check, not a comparative FPS benchmark.

On the tested GE-Proton11-3 setup, Wine's builtin ICU aborted on `icuuc.dll.u_setMemoryFunctions_65`. Use the game's native ICU DLLs for that setup:

```sh
WINEDLLOVERRIDES='icuuc=n,b;icuin=n,b;icudt=n,b;d3d12=n,b;d3d12core=n,b' \
  D4R_RR_ENABLE=1 D4R_RR_PRESET=5 \
  D4R_SHIM_VRAM_INTEROP=1 D4R_SHIM_SPLIT_FRAME=1 D4R_SHIM_LINEAR_INPUTS=1 \
  D4R_ZLUDA_WMMA=1 D4R_ZLUDA_WMMA_FP8=1 D4R_ZLUDA_WMMA_FP8_NATIVE=1 \
  scripts/d4r_play.sh
```

Merge these overrides with any existing `WINEDLLOVERRIDES`; do not discard overrides needed by other mods. Keep the patched vkd3d DLLs current, including game-local copies as described above. Select DLSS, Ray Reconstruction and path tracing in the game. `D4R_SHIM_RR_VRAM_GUIDES` defaults to `1` with VRAM interop; `0` selects host guide staging for A/B diagnostics.

Leave `D4R_SHIM_VRAM_VERIFY`, `D4R_SHIM_ALPHA_VERIFY` and output capture disabled for normal play or benchmarking. The in-game diagnostic run enabled verification and reported zero differing bytes for all six supplied RR guides: normals, diffuse/specular albedo, specular hit distance, pre-particle color and the subsurface-scattering guide. Verification itself downloads GPU data.


## 7. Package a release

```sh
scripts/fetch_rocm_runtime.sh         # -> ~/.cache/d4r-rocm-runtime (AMD's ROCm 7.2.4 runtime, checksummed)
D4R_OPTISCALER=/path/to/OptiScaler_0.9.4.7z \
D4R_DLSS_DLLS=/path/to/310.7/nvngx_dlss.dll:/path/to/310.9/nvngx_dlss.dll \
D4R_ZLUDA_DIR=~/.cache/d4r-zluda-current D4R_VKD3D_DIR=~/.cache/d4r-vkd3d-d4r D4R_ROCM_DIR=/opt/rocm \
scripts/package_release.sh            # -> dist/d4r-<version>.zip
```

Set `D4R_INSTALL_MODE=asi` to build `d4r-<version>-asi.zip` for an existing
game-local OptiScaler installation. Its only top-level entries are `d4r.asi`
and `d4r/`, ready to extract into the configured plugin folder. It omits
OptiScaler's DLL/INI and does not require `D4R_OPTISCALER`. The runtime and
`install.sh` are inside `d4r/`; the NVIDIA installer also accepts its original
proxy-package layout. The native `d4r/d4r-setup.exe` helper installs the supplied
D3D12 pair beside the game EXE and merges the NGX route into the active loader's
INI (including GE-Proton's prefix-local INI) with backups, after
waiting for the invoking game process to exit. Subsequent unchanged launches do
nothing. `InitializeASI` launches setup without GPU work under the loader lock.

The build writes `build/d4r.asi` and `build/d4r-setup.exe`.
See [ASI_README.txt](../packaging/ASI_README.txt).

The script:
- builds the shim, the bridge and both fast and accuracy network-layer kernels for gfx1100–gfx1103 and gfx1200–gfx1201 by default (`D4R_GPU_ARCHS` can select fewer targets); each gfx12 target also gets an `<arch>-fp8` variant;
- writes the kernel manifest from the DLLs you list (it records hashes of their PTX, nothing else);
- stages OptiScaler as `dxgi.dll` with the settings in `packaging/optiscaler.settings`, applied by `scripts/configure_optiscaler.py`; both full and clean ZIPs set `NvngxPath=d4r\nvngx.dll` explicitly, retaining `OptiDllPath=d4r` for other libraries;
- includes `install.sh` instead of NVIDIA's two DLLs by default, retaining any supplied texture kernels (built from NVIDIA's PTX by `kernels/build.sh tex`); builds accuracy texture sets with `D4R_ZLUDA_EMIT` unless marked prebuilt sets are supplied in `D4R_BUNDLE_TEX/accuracy/<target>`; the full variant adds the optional `nvngx_dlssd.dll` supplied through `D4R_BUNDLE_DLSSD`;
- with `D4R_BUNDLE_ENGINE`, adds the native engine's shaders as `d4r/engine/k`; the model's weights and tables are never packaged;
- zips the result together with the ZLUDA and vkd3d-proton builds, the ROCm runtime (as `d4r/rocm`), `packaging/d4r.ini`, the licenses and the patches.

`D4R_BUNDLE_NVIDIA=installer` is the default: the ZIP includes `install.sh`, and users run `bash install.sh` after extraction to fetch the pinned official NVIDIA DLLs. This mode requires `D4R_BUNDLE_DLSS` to match the DLSS 310.7.0 SHA-256 pinned in the installer, so bundled texture kernels match the downloaded library; `D4R_BUNDLE_NGX` is only needed with `D4R_BUNDLE_NVIDIA=1`, which retains the old DLL-bundling mode.

The NVIDIA files are not covered by d4r's license; redistributing them is up to whoever publishes the zip. `D4R_BUNDLE_NVIDIA=0` builds `d4r-<version>-nonvidia.zip` without them and without the texture kernels. [architecture.md](architecture.md#portable-installs-the-release-zip) describes how the installed files work together.

### GLIBC 2.41 release builds

Build the pinned environment with:

```sh
docker build -f packaging/build/Dockerfile.glibc241 -t d4r-build:glibc241 packaging/build
```

Inside that image, `scripts/build_release_glibc241.sh` rebuilds the GPU device helpers, ZLUDA and its LLVM, `d4r_emit`, the NGX shim, Wine CUDA bridge, patched vkd3d-proton, and performance/accuracy texture sets for every release target. Packaging then builds both network sets, checks all bundled Linux libraries against 2.41, and records the toolchain in `d4r/source/BUILD_INFO.txt`.

Mount the repository at `/work`, ROCm's compiler/headers/device libraries at `/opt/rocm`, and an official Rust toolchain (1.98.0 used for 0.1.3) at `/opt/rust`, with `/opt/rust/bin` on `PATH`. Keep toolchain and third-party runtime inputs read-only. Supply the normal package inputs listed above and these build settings:

| Variable | Input |
|---|---|
| `D4R_RELEASE_BUILD_ROOT` | writable directory under `/work/build/` |
| `D4R_ZLUDA_SRC` | isolated checkout inside that directory, with both pinned LLVM and HiGHS submodules, the real OCKL LFS payload, and patches `0002`–`0014` applied |
| `D4R_VKD3D_SRC` | isolated vkd3d-proton checkout inside that directory, with its submodules and patches `0001`–`0005` applied |
| `CARGO_HOME` | writable build-local Cargo cache; prefetch the locked dependencies for an offline build |
| `D4R_ROCM_DIR` | `/opt/rocm` |
| `D4R_ROCM_LINK_STUBS` | optional directory for additional ROCm link libraries |

Run `bash scripts/build_release_glibc241.sh` inside the container. The script checks that the build system actually reports `glibc 2.41`, rejects external source-cache checkouts, builds with the locked Cargo dependencies offline, and runs the configuration tests before packaging. The ZIP and checksum go in `$D4R_RELEASE_BUILD_ROOT/dist/`; logs and `toolchain.txt` remain in the build directory. AMD's bundled ROCm runtime, OptiScaler and NVIDIA's libraries remain supplied binaries; the Linux ABI check covers the bundled ROCm libraries too.

`D4R_BUNDLE_TEX` accepts the older flat directory for gfx1101, or a directory with per-target subdirectories (`gfx1100/`–`gfx1103/`, `gfx1200/`, `gfx1201/`, plus `gfx1200-fp8/` and `gfx1201-fp8/`). Build each texture set with `D4R_GPU_ARCH` and `D4R_ZLUDA_EMIT`; the `-fp8` sets also need `D4R_NATIVE_FP8=1`. `d4r_emit` targets those GPUs offline. Missing texture kernels fall back to ZLUDA on that target; the network-layer kernels are still included. The published Super Resolution benchmarks use RX 7700 XT; separate native-FP8 Ray Reconstruction D/E temporal validation and Cyberpunk gameplay use RX 9070 XT (gfx1201). This does not validate every packaged native-kernel target.

## Checks

- `python3 -m unittest discover -s tests -v` includes OptiScaler INI generation, full/clean package staging and NGX-routing checker regressions. Package tests use fixture binaries and GPU-build stand-ins; real DLL loading still needs the OptiScaler/Proton harness.

Full proxy packages also require `fakenvapi.dll`, `libxell.dll`, `libxess_fg.dll`,
`amd_fidelityfx_framegeneration_dx12.dll` and `Licenses/XeSS_LICENSE.txt` from the
same OptiScaler release. These are staged in `d4r/`; the NVAPI router is built
alongside the shim. ASI packages retain the existing OptiScaler dependencies.
See [opt-in FG and latency setup](frame-generation.md) for configuration and rollback.
For the offscreen OptiScaler SR harness, `D4R_OPTISCALER_FEATURE_RUNTIME=/path/to/d4r`
enables the latency router against the supplied real dependencies while leaving FG off.
- `scripts/check_environment.sh` lists the tools, GPU and Proton builds it finds.
- The D3D12 harness (`scripts/run_d3d12_dlss_harness_proton.sh`) drives DLSS outside a game.
  - `D4R_HARNESS_MOTION_SCENE=1 D4R_HARNESS_SAVE_FRAMES=1` saves a temporal sequence for byte comparisons. Compare the regular output path against `D4R_SHIM_OUTPUT_PACKED=1` and/or `D4R_SHIM_INPLACE=1 VKD3D_D4R_LINEAR_TEXTURES=1`, with VRAM interop, split frames and direct output enabled; `D4R_HARNESS_OUTPUT_RGB10A2=1` selects packed output.
  - `D4R_HARNESS_RECREATE_OUTPUT=1` releases and recreates the game's output texture between completed frames; compare its copy and in-place sequences to catch imported-memory handle reuse.
  - `D4R_HARNESS_RECREATE=4 D4R_HARNESS_VERIFY_RECREATION=1` checks repeated feature/quality recreation. `D4R_HARNESS_SKIP_RECORDED_INPUT=1` checks discarded command-list input readiness without injecting stale history.
- Native kernels have their own validation path; see [native-kernels.md](native-kernels.md).

### Ray Reconstruction VRAM comparison

On the validated gfx1201/native-FP8 runtime, build the harness and shim, then compare host and GPU guide transport:

```sh
scripts/build_d3d12_dlss_harness.sh
D4R_SHIM_SPLIT_FRAME=1 D4R_SHIM_LINEAR_INPUTS=1 \
  D4R_SHIM_VRAM_VERIFY=1 D4R_SHIM_ALPHA_VERIFY=1 \
  D4R_RR_ALPHA=1 D4R_RR_GUIDE_PROBE=1 \
  D4R_HARNESS_TEMPORAL=1 \
  D4R_HARNESS_SIZE='193 109 288 162' \
  scripts/run_rr_guide_compare.sh /tmp/rr-compare \
    rgba16f,rgba16f,rgba8unorm,rgba8unorm \
    rgba8snorm,rgba16unorm,bgra8unorm,rgb10a2unorm \
    rgba32f,r32f,rgb11b10float,rgba8unorm \
    rgba16f,r32uint,bgra8unorm,rgba16f
```

The script stages the matching vkd3d DLLs, real DXVK-NVAPI, shim, and feature libraries together. It defaults to RR preset E and enables the required native-FP8 flags. Override `D4R_RUNTIME_DIR`, `D4R_PROTON_DIR`, or `D4R_VKD3D_DIR` for an isolated runtime. Neither feature discovery nor failed evaluation is bypassed.

Every saved RGB and alpha frame is compared. The reset guide probe requires an image response, not spatial confinement: transformer attention may legitimately change both halves. `D4R_HARNESS_RR_ALPHA_FORMAT` selects `r32f`, `r16f`, `r8unorm`, or `r16unorm`; alpha captures are always decoded float32, with format-appropriate outside-subrect sentinels.

Verification intentionally downloads GPU data. Unset `D4R_SHIM_VRAM_VERIFY` and `D4R_SHIM_ALPHA_VERIFY` for performance measurements and games. For same-frame conversion checks, combine alpha verification with `D4R_SHIM_OUTPUT_DUMP_DIR` and the existing dump frame selectors; compare each saved alpha region with the corresponding presented alpha capture, allowing one destination quantization step for blit rounding.

`python3 -m unittest discover -s tests -v` includes output-comparator regressions for channel swaps, nonfinite pixels, missing/truncated captures, alpha frame mismatches, and writes outside the output subrect.
