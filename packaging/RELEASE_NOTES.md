<!-- Modified in this fork for CUDA Ray Reconstruction support and validation (2026). -->
# 0.1.5

- Switch the DLSS model in game from OptiScaler's menu: DLSS Settings > Render Presets Override, pick E, K, L or M, Apply Changes. A preset chosen there (or `RenderPresetOverride` in OptiScaler.ini) now wins over `[DLSS] Model`, which still fills quality modes left at Default and applies again when the override is unticked (`[OptiScaler] FollowPresets`, default true). The shim now also uses the preset of the feature's own quality mode instead of always the Quality slot.
- Switch to FSR from the same menu (Upscaler (DX12) > FSR 3.X/4): the zip ships AMD's FidelityFX loader and FSR 4.1.1 upscaler from the OptiScaler release, and the shim lets OptiScaler and AMD's FidelityFX modules read the GPU's real PCI ids (from sysfs) while the game, vkd3d-proton and NVIDIA's libraries keep the NVIDIA identity (`[OptiScaler] Fsr4`, default true). Under the NVIDIA identity OptiScaler saw no RDNA GPU and FSR 4.1.1 produced FSR 3.1's output byte for byte; now RDNA3 and RDNA4 get FSR 4.1 (through GE-Proton's FSR 4 library; the harness output is byte-identical to FSR 4 without the spoof) and other GPUs FSR 3.1. DLSS output is unchanged.
- RDNA2 (gfx1030–gfx1036: RX 6000 series, Steam Deck, RDNA2 integrated GPUs): native DLSS 4 (K) layers. RDNA2 has no WMMA, so `kernels/common/wmma_layout.h` does each 16×16×16 product with `v_dot2c_f32_f16` and a `row_share` DPP broadcast folded into the instruction, in the gfx11 register layout. On an RX 7700 XT this path is bit-identical to the WMMA instruction (random chains and K's whole network over 24 frames) and runs K in 3.6 ms per 1080p frame (WMMA: 1.48 ms; fully translated without WMMA, which RDNA2 got before: 46.8 ms; RDNA2 also keeps K's output tail translated, about +0.7 ms on RDNA3). E runs translated, as on every GPU; M and L are translated only on RDNA2. Not yet run on RDNA2 hardware.

- RDNA2 (Steam Deck): preset M runs on the engine without cooperative matrices or FP8 (`layer_m_dot2.comp`: v_dot2, with the MLP, patch merges and attention output projection in int8), about 14.4 ms at 640x360 -> 1280x720 on the Deck, and `install.sh` now builds `engine/m` and `engine/m-ldr` there (L, which has no such path, is still left out). The model builder replaces M's and L's network kernels in their test frame with empty ones of the same parameters (`model_tools.py stub-ptx`, through the bridge's `D4R_CUDA_REPLACE_DIR`): compile_m only reads their launches, and ZLUDA would compile each with emulated matrix products for minutes. A cold Super Resolution build on the Deck takes about 150 s with M; M's frame alone took over 10 minutes before.
- Native engine, preset M: display-resolution motion vectors (games that clear MVLowRes, such as Subnautica 2), byte-identical to the CUDA backend with NVIDIA's translated kernels in the harness; such games ran M on the CUDA backend before (on the Steam Deck: minutes of kernel compilation, then a slow path). M packages must be rebuilt (`D4RM0005`; `install.sh` does it, the changed builder invalidates cached ones); older packages keep these games on the CUDA backend. L still uses the CUDA backend for them.
- Fix: with `[Interop] ElideNgxSync` (the default), NGX's weight uploads could arrive incomplete. NGX reuses one host buffer for every upload and synchronises before each reuse; HIP's asynchronous copies from ordinary host memory may still read the buffer after returning, which CUDA's do not, so with that synchronisation elided some tensors arrived as zeros whenever the uploads ran back to back (on the Deck: preset M's input-stage weights, and the last head tensor of E's and F's reconstruction kernel in every capture). The bridge now copies such sources itself and frees each copy once the upload has completed. Rebuild the E and F engine models (`install.sh --engine-only --rebuild-engine`; the changed builder also invalidates the cached ones).
- The model builder's test frames poll their inputs on the CPU (`D4R_SHIM_GPU_INPUT_WAIT=0`): with the GPU-side input wait, the replay dump's per-launch synchronisation left NGX's output empty and the capture check failed.
- `install.sh` sets up the native engine: after the NVIDIA DLLs (now also DLSS Ray Reconstruction 310.9.1) it offers to build the engine's models on the user's machine (`d4r/tools/build-engine-models.sh`): Ray Reconstruction D, E and F straight from `nvngx_dlssd.dll`, Super Resolution K, M and L (and their LDR variants) from one test frame per model rendered through d4r under GE-Proton. The verified DLLs and the built models are kept in `~/.cache/d4r`, so the next game's install neither downloads nor builds again. Questions replace the old command-line choices (`--help` lists options for scripted installs).
- `[Engine] Enabled = auto` is the new default: the engine runs whenever `d4r/engine` holds its models, with the fastest model builds; it is faster on RDNA4. Anything it cannot run still uses the CUDA backend.
- Not in this build: the experimental AMD FSR Ray Regeneration backend for Ray Reconstruction (`[RayReconstruction] Backend = fsr`) that an earlier nightly carried. Its code is not part of d4r's sources; Ray Reconstruction runs on NVIDIA's DLSS denoiser (engine or CUDA backend).

- ASI setup ignores Proton accessibility helpers and other background utilities, and serializes setup per plugin installation. Its D3D12 configuration uses regular Windows library hooks and disables Agility SDK upgrades. Game-specific startup remains a separate runtime check.

- ASI setup now finds the INI beside the active OptiScaler loader, including GE-Proton's `system32/umu` installation. Missing-file errors identify the actual missing path, and INI updates are staged beside their destination even when the game and prefix are on different filesystems.
- Optional OptiScaler ASI add-on (`D4R_INSTALL_MODE=asi`): drop `d4r.asi` and its companion `d4r/` folder into the existing plugin directory, run its `install.sh` once for NVIDIA's DLLs, then launch. The ASI installs the supplied D3D12 wrapper/private core beside the EXE and configures NGX after the game closes, with backups. The runtime stays in the plugin folder; no Python is needed. Game rendering still requires validation.
- Native engine, preset L (DLSS 4.5, render preset 12): the engine now runs L ([Engine] Enabled, models in `engine/l` and `engine/l-ldr`, built by `engine/compile_m.py` from a capture of preset L; `scripts/install_d4r_runtime.sh` stages `D4R_ENGINE_MODEL_L` / `D4R_ENGINE_MODEL_L_LDR`). Every stage, `d4r-m` on captured frames and the D3D12 harness are byte-identical to the CUDA backend running NVIDIA's translated kernels, apart from isolated single-frame knife edges (96-122 dB). 1280x720 -> 3840x2160 takes 3.8 ms on an RX 9070 XT. Display-resolution motion vectors keep the CUDA backend, as for M. Details: `engine/L_NOTES.md`.
- Native engine fixes found with L, which also apply to M (rebuild M packages with the current `compile_m.py`): the encoders' patch merge is clamped to ±2π as NVIDIA's kernels do; the automatic exposure is measured on NGX's grid at Ultra Performance (half the render size, at least 256, when the output is more than twice the render size; half the output was used, which changed every value downstream); the downsample's filter weights are exact (glslang rounded them toward zero: 92% of the output values were identical, now all); and the network uses the summation order of NVIDIA's kernels as ZLUDA runs them. M is now byte-identical to that CUDA backend in the harness too, where it measured 36-41 dB.
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
- Restore behavioral preset-override coverage through a shared, allocation-free policy used by
  feature creation: SR/RR isolation, every quality mode, unset/empty overrides, and explicit
  default-preset selection. Tests compile the helper directly without extracting shim source
  or matching diagnostic wording; polling configuration validation and precedence remain covered.

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
- **Preset M network dispatch barriers.** The ten layer dispatches no longer flush global memory between layers; an empty pipeline barrier keeps the serialization the layers actually need. Captured-frame replays stayed byte-identical at every stage. On an RX 9070 XT at Balanced (1488×837 → 2560×1440) the network median dropped 1863.0 → 1765.6 µs and the whole frame 2515.0 → 2404.6 µs; Quality and Performance improved 36.6 and 62.8 µs with byte-identical outputs.

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
