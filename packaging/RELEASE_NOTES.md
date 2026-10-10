# Unreleased

- `scripts/check_redistributable.py` refuses NVIDIA's binaries, NGX SDK headers, PTX and code objects, d4r's PTX translations and the engine's model data. `scripts/package_release.sh` runs it on every release before zipping, and `git config core.hooksPath scripts/git-hooks` runs it before each commit.
- Native engine fixes: the K adapter now passes the game's exposure scale to the exposure shader (it was silently ignored); the HIP network's shared buffers are reused across feature recreation instead of leaking their ROCm mapping (21 → 6 imports over seven K features in the harness); `hipnet.bin` must belong to the preset being created; an M-only game no longer loads the K model; `scripts/install_d4r_runtime.sh` stages `engine/k-ldr` and each model's `hip/` folder.
- Native engine, preset K: output is now byte-identical to the CUDA reference over an eleven-frame sequence at Quality (1706×960 → 2560×1440, native HIP network, RX 9070 XT), where later frames previously fell to 47–51 dB. The engine now selects NGX's reconstruction table by upscaling ratio instead of using one table for every ratio, and two rounding-order differences in the input and output stages are gone. K models must be rebuilt (`prepare_k.py`, `compile_k.py`); older packages still load and behave as before.
- Native engine, wider game coverage (K packages must be rebuilt with the current `compile_k.py`, marker `D4RO0003`; older packages keep refusing the new cases with a logged reason):
  - Preset K accepts display-resolution motion vectors (MVLowRes clear; output-sized, in output pixels). Byte-identical to the phase-capable CUDA reference over eleven frames at 1706×960 → 2560×1440 (HDR, create flags 0x49) and 1712×960 → 2560×1440 (LDR, 0x68), and over three frames at 1280×720 → 2560×1440. The reference must use a ZLUDA with patch 0008; the old bundled library omits K's bottleneck post phases and is itself off by ~33 dB. Preset M still uses the CUDA backend for display-resolution vectors.
  - Preset K accepts a game-supplied exposure texture (AutoExposure off) and any positive exposure scale and pre-exposure in both the automatic and the game path. With the native HIP network at 1706×960 → 2560×1440, HDR K is byte-identical to the CUDA backend over eleven frames for automatic exposure with scale and pre-exposure 1/0.5/0.7/2/3 (and combinations) and for game textures 0.4–32, except an effective exposure of exactly 2 or 4 (first differing frame 104.7 dB / 112.8 dB, then ≈49–63 dB); LDR ignores the exposure and is byte-identical for any value. Preset M accepts the AutoExposure flag, an exposure texture, pre-exposure and exposure scale, which NGX ignores for M (CUDA and engine output identical for each).
  - Presets K and M accept the DLSS indicator's inverted-axis parameters, which only place NGX's debug indicator (NGX and engine output byte-identical with and without them), regular (non-inverted) depth (K), and output subrectangles (not yet run in the harness).
- Merge the RDNA4 token-lane Swin and hardware-FP8 texture-tail optimizations, preserving MicroCUDA's allocation-identity tube weight cache.
- Add opt-in packed R10G10B10A2 and in-place DLSS output, with vkd3d-proton patches `0001`–`0003`. In-place mappings retain their source textures until process exit; output recreation can increase retained VRAM.
- Fix deferred input-marker ordering: wait for the game's GPU copies before CUDA uploads or diagnostic readbacks, not just before NGX evaluation.
- Prevent stale in-place output mappings when games destroy and recreate output textures. The harness now supports `D4R_HARNESS_RECREATE_OUTPUT=1` for this regression.
- Integration validation on RX 9070 XT: moving 12-frame RGBA16F/R10G10B10A2 copy, packed and in-place sequences matched byte-for-byte; fast and accuracy paths, token-disabled reference layers, MicroCUDA tube-cache reuse, discarded input recordings and four feature-recreation cycles were exercised. gfx1101 and gfx1200 compatibility builds also passed; those targets were not runtime-tested.

# d4r 0.1.3

This release adds an optional accuracy mode and fixes the GLIBC compatibility failure, DLSS artifacts, and stale output-buffer redirects reported after 0.1.2. Accuracy mode is **off by default**.

## Added

- **Prefer accuracy over performance.** Set `[Kernels] PreferAccuracy = true` in `d4r/d4r.ini` and restart the game. The release includes separate accuracy variants of every native network and texture kernel for all supported RDNA3 and RDNA4 targets, including RDNA4's native FP8 variants.
  - K restores FP16 accumulator rounding after each matrix step. M also restores intermediate FP8 quantization.
  - The accuracy path retains denormal handling, disables relaxed accumulation/fast-math switches, and keeps NGX's synchronization calls. It uses a separate compilation cache.
  - Missing accuracy variants use translated NVIDIA kernels and are logged; the fast native set is never silently substituted.
  - This aims to preserve NVIDIA's arithmetic. It is slower, its FPS cost has not been benchmarked, and **1:1 image quality against RTX DLSS is not proven**.
- **Reproducible GLIBC 2.41 build environment.** A pinned Debian build image and release-build script rebuild ZLUDA/LLVM, the Wine bridge, the Windows DLLs, and both kernel sets. The ZIP includes toolchain details in `d4r/source/BUILD_INFO.txt`.

## Fixed

- **GLIBC loader compatibility ([#6](https://github.com/countervolts/d4r/issues/6)).** The previous release required `GLIBC_2.44` from `libm.so.6`, preventing ZLUDA from loading on systems such as the reporter's Bazzite installation. This release is built in a GLIBC 2.41 environment; the package check rejects any bundled Linux library requiring a newer ABI.
- **Cyberpunk 2077 preset-K black artifacts ([#5](https://github.com/countervolts/d4r/issues/5)).** Preserve FP16/FP64 denormal requirements while limiting the performance optimization to FP32. The fix covers both translated kernels and the rebuilt native texture kernels. Captured-scene replays showed substantially fewer sky artifacts; this is not a guarantee that every scene is artifact-free.
- **Stale direct-output redirects after resource destruction ([#8](https://github.com/countervolts/d4r/issues/8)).** Reused CUDA array handles no longer inherit an earlier feature's output buffer. This addresses the quality-switch corruption reported with preset M in Spider-Man 2.

## Improved

- **Synchronization diagnostics and an optional blocking output wait.** `D4R_SHIM_BLOCKING_SYNC=1` yields the CPU while waiting for DLSS output, with a fallback to context synchronization. It is an opt-in mitigation for [#7](https://github.com/countervolts/d4r/issues/7); reduced CPU usage on the reporter's RDNA4 setup still needs confirmation.
- **More useful failure logs.** Rejected input/output formats and dimensions are now recorded even after startup, with repeated errors rate-limited. Output-wait CPU and wall time are reported separately, and GPU timing queries are deferred until asynchronous work finishes.
- **More conservative accuracy texture builds.** Strict surface conversions use separate helper functions, avoiding a compiler failure when the original denormal handling is retained.
- **Accuracy build safeguards.** The builder refuses to label a directory containing older fast binaries as an accuracy set. Configuration tests cover every release target, missing-set fallback, invalid markers, environment precedence, and the default-off behavior.

## Also included from 0.1.2

These capabilities remain in the release; they are not newly validated hardware results:

- Experimental RDNA4 builds for gfx1200/gfx1201, with `NativeFp8` enabled by default, alongside gfx1100–gfx1103 RDNA3 builds.
- Kernel selection uses the GPU with the most SIMDs when integrated and discrete GPUs coexist; `D4R_GPU_ARCH` overrides it.
- DLSS 3 matrix-kernel acceleration; M's cached weight preparation, packed FP8 encoding, improved patch-merge/weight loads, and direct output; K's CU-mode layers; wave64 performance texture variants.
- Native texture/output variants for the game's motion-vector resolution, HDR/LDR input and depth direction, compiled offline per GPU target.
- The bundled ROCm 7.2.4 runtime: no separate ROCm installation is required.

The 0.1.2 performance measurements remain historical results. They are not benchmarks of this rebuild or the new accuracy mode.

## Install or upgrade

1. Back up your `d4r/d4r.ini` and `OptiScaler.ini` if you customized them.
2. Extract `d4r-0.1.3.zip` into the folder holding the game's main `.exe` (`<Project>/Binaries/Win64` for Unreal Engine games), keeping the ZIP's layout.
3. Restore your settings if needed. Older INIs omit `PreferAccuracy`, which defaults to off; add it under `[Kernels]` to enable it.
4. Select GE-Proton 11 in Steam (tested integration: GE-Proton11-3), with:
   ```
   PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %command%
   ```
5. Select DLSS in the game and restart after changing the accuracy option.

The rebuilt runtime may compile a new kernel cache on first start. `sh d4r/d4r-check.sh` checks the install. [SUPPORTED_GAMES.md](https://github.com/countervolts/d4r/blob/main/SUPPORTED_GAMES.md) lists game-specific setup and known issues.

## Validation and limits

- All bundled Linux ELF libraries are checked against the GLIBC 2.41 ABI limit; release loading is checked inside that environment.
- The release contains performance and accuracy kernels for all eight target variants. Configuration tests and sampled CPU-emulated kernel/reference checks cover the new accuracy path; these do not establish full-game or RTX parity.
- RDNA4 remains experimental: emulator/compiler coverage does not prove hardware speed, image quality, or driver stability. The external RX 7900 XTX report is RDNA3, not RDNA4 validation.
- Super Resolution only: Frame Generation, Ray Reconstruction, and DLSS 5 Neural Rendering are unsupported. Keep Ray Reconstruction off.

## Licenses

d4r is GNU General Public License, version 3 only (GPL-3.0-only). ZLUDA, LLVM, vkd3d-proton, OptiScaler, and the bundled ROCm libraries retain their respective licenses in `d4r/licenses`; sources and patches are listed in `d4r/source`. NVIDIA's libraries and texture kernels compiled from its code are not covered by those licenses. d4r is not affiliated with NVIDIA or AMD.

The attached `.sha256` file contains the ZIP's checksum.
