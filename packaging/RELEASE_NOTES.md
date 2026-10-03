<!-- Modified in this fork for CUDA Ray Reconstruction support and validation (2026). -->
# Unreleased

## Ray Reconstruction

- Run the signed CUDA-capable 3.10.7 DLSS-Denoiser through the NGX core's feature-13 lifecycle.
  Presets D and E select distinct network weights independently of the Super Resolution preset.
- Preserve per-frame guide resources, camera matrices, subrects, and stable CUDA texture handles.
  Support normalized 8-bit albedo guides with the CUDA array-format ABI and preserve full guide rows.
- Support pre-Init feature discovery through the denoiser's requirements export without initializing
  NGX with a provisional application identity. Initialized capability checks remain authoritative.
- Keep separate alpha and RGB outputs paired, including padded output subrects and current-frame
  split presentation. Separate alpha no longer forces the whole feature through host staging.
  R32F alpha stays in VRAM; half/UNORM alpha uses GPU conversion with hardware-dependent rounding.
  Raw integer alpha retains its existing host conversion without disabling RGB VRAM interop.
  Converted alpha can differ from CPU quantization by one destination step.
  Normals, roughness, albedo, and other compatible RR guides use shared VRAM buffers; BGRA and
  packed RGB guides convert on the GPU. Raw integer guides retain their host conversion.
  Add actual-CUDA-array verification, per-frame rendered RGB/alpha comparisons, and source-alpha
  capture for checking conversion against the exact frame being presented.
- Bound R8_UINT alpha stores to one byte and balance guide/alpha references when evaluation
  cannot place its frame marker. Publish cached denoiser capabilities as a coherent snapshot.
- Extend the ZLUDA patch stack for real denoiser PTX, LLVM lowering, half texture/store operations,
  and release/acquire ordering around workgroup barriers.
- Verified D/E temporal evaluation and separate alpha on RX 9070 XT (gfx1201) with native FP8 WMMA.
  Scalar FP8 temporal RR produced nonfinite output and is not validated. Cold compilation can stall
  initial rendering; other GPU targets and image-quality parity with RTX hardware remain unverified.
- Cyberpunk 2077 2.31 rendered loaded local saves with path tracing and RR preset D enabled at
  1920×1080 and 3840×2160 Ultra Performance on gfx1201. The actual overlay reported DLSSD 310.7.0,
  and CUDA feature-13 evaluations completed. The observed ~53 FPS at 1080p and ~20 FPS at 4K are
  snapshots, not comparative benchmarks. Newer DLSS 4.5 RR preset F is not implemented or validated.
- Rechecked the optimized VRAM guide path in a loaded Cyberpunk 2077 2.31 save with path tracing,
  RR preset E and 1080p Ultra Performance on gfx1201; the user verified gameplay.
  All six supplied guide arrays matched the host reference byte-for-byte in diagnostic checks.
  The tested GE-Proton11-3 setup required native ICU overrides
  (`icuuc=n,b;icuin=n,b;icudt=n,b`) to avoid Wine's unimplemented `u_setMemoryFunctions_65`.
  Keep guide/alpha verification off for normal play; it intentionally adds GPU readbacks.
- Rebased RR onto the latest `linux` branch, preserving per-slot input markers and ordered
  frame retirement. Rebuilt shim/bridge and rechecked a loaded Cyberpunk scene with camera
  motion, RR E and path tracing at 1080p Ultra Performance on gfx1201.
  Removed brittle source-text/wording tests; configuration validation remains covered.

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

d4r is Apache 2.0. ZLUDA, LLVM, vkd3d-proton, OptiScaler, and the bundled ROCm libraries retain their respective licenses in `d4r/licenses`; sources and patches are listed in `d4r/source`. NVIDIA's libraries and texture kernels compiled from its code are not covered by those licenses. d4r is not affiliated with NVIDIA or AMD.

The attached `.sha256` file contains the ZIP's checksum.
