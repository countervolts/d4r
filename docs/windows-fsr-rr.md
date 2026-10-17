# Native Windows FSR Ray Regeneration

## Status (2026-10-17): runs on RDNA3 through vkd3d-proton; synthetic validation only

`[RayReconstruction] Backend = fsr` answers a game's DLSS RR request (NGX feature 13) with AMD's FSR Ray
Regeneration 1.2 denoiser, then upscales with an internal Super Sampling feature. Selection, `install.bat` [F]
and `d4r\fsr` payload are described in `WINDOWS_README.txt`. **No game exposes RR to test with, so everything
below is a synthetic scene; game compatibility and image quality against DLSS RR are unverified.**

Why the earlier "zero providers" result was correct: AMD's denoiser ships only `wmma_fp8` shaders, issued
through AMD's driver extension (`AmdExtD3DShaderIntrinsicsUAV`), and RDNA3 has no FP8 WMMA hardware. Native
Windows drivers therefore offer no provider. Linux gets one through vkd3d-proton with dxil-spirv's scalar FP8
fallback and a Proton `amdxc64.dll` shim. The Windows route reproduces that:

1. **vkd3d-proton for Windows** (`tools/windows/build_fsr_payload.py`): vkd3d-proton `3dfc6f07` plus dxil-spirv
   PR 307 (`f9f6294f`) and `-Denable_extended_emulation=true`, built with llvm-mingw clang; two libc++ include
   patches in `patches/vkd3d-proton-windows/`. The shipped `d4r\fsr\d4r_vkd3d_core.dll` is `d3d12core.dll`
   renamed and loaded by path; `IVKD3DCoreInterface::CreateDevice` is called directly (the IDL has no base
   interface, so it is vtable slot 0). `DXIL_SPIRV_CONFIG=wmma_rdna3_workaround` is set in-process.
2. **amdxc64 shim** (`tools/windows/amdxc64_shim`): recovered from the denoiser's call sites (interface IDs
   `{014937ec-…}` factory, `{f714e11a-…}` wave-matrix properties with a 16x16x16 type-0xb entry,
   `{ba019d53-…}` intrinsics enable, `{b58d6601-…}` device info). It advertises the FP8 shape the shaders use;
   the arithmetic is emulated by dxil-spirv. The device-info values are placeholders. It never forwards to
   AMD's driver.
3. **Lookup redirect.** The denoiser finds `amdxc64.dll` with `GetModuleHandle`. In a game process AMD's own
   D3D12 driver has already loaded the real `amdxc64.dll`, which crashes on a vkd3d device. The loaded copy of
   AMD's DLL gets its `GetModuleHandleA/W` imports redirected in memory to the shim. The DLL on disk is not
   modified.

Data path (`tools/windows/fsr_rr_backend.cpp`): the game queue copies the guides into the same shared D3D12
packet the native DLSS backend uses; the packet is imported into vkd3d's `VkDevice`; `fsr_rr_prepare` and
`fsr_rr_compose` (Vulkan, recorded through vkd3d's DXVK interop) convert DLSS-RR guides to FSR inputs and back
around `ffxDispatch`; the composed colour returns through the packet to a game-device texture, and the internal
SR feature upscales it into the game's output. Every hand-off is CPU-synchronized; no vkd3d patch is needed.

Evidence (all synthetic, RX 7700 XT, static camera, zero jitter):
- Probe (`fsr_rr_probe`, 960x540, 4 signals): provider found, context created, 48 frames finite, denoised, reset
  handled; GPU median 20.3 ms (`build/fsr-vkd3d-probe-v1/out3`).
- NGX feature-13 path (`tools/windows/test_fsr_rr_ngx.py`, noisy scene with a known noise-free answer):
  640x360 render=output without SR, RMSE vs clean 0.488 -> 0.060 (`build/fsr-rr-shim-run-v9`); 640x360 ->
  1280x720 with SR from the extracted v9 package, RMSE 0.289 against 0.581 for the noisy input and 0.314 for the
  noise-free render-size image nearest-upsampled (`build/fsr-rr-ngx-upscale-v9-v1`). About 10-11 ms per frame
  (convert+denoise+compose ~7 ms, SR submit ~2.6 ms); first RR start compiles AMD's shaders (~65 s cold).
- Native DLSS RR regression with the same shim: `PASS NGX RR ... standalone_exact=1`.

Guide conversion check: a diagnostic (`D4R_FSR_RR_DUMP_DIR`, first `D4R_FSR_RR_DUMP_FRAMES` frames) dumps the
converted linear depth, motion and normals. On the moving fixture the converted motion equals the ground-truth
`PreviousUV - CurrentUV` (FFX's contract) to 1.5e-6, the view-z delta is 0 for lateral motion, and linear depth is
reconstructed to 5e-7 (`build/fsr-rr-dump-moving-v2`).

Motion vectors matter, measured (FSR alone, same-size, moving camera, RMSE vs noise-free): with AMD's default
stability bias the ordering was misleading (correct 0.107, flipped 0.066, zeroed 0.158, because default bias
over-weights history and rejection scored better); at stability bias 0 it is physically ordered (correct 0.033,
flipped 0.042, zeroed 0.061). `FsrStabilityBias` (`D4R_FSR_RR_STABILITY_BIAS`), `FsrDetail` (`D4R_FSR_RR_DETAIL`),
`D4R_FSR_RR_KEY_<n>`, `D4R_FSR_RR_DEMOD_FLOOR` and `D4R_FSR_RR_HALO_LIMIT` follow the Linux adapter. Through the
full pipeline (FSR then SR): same size, bias 0 gives 0.036 against 0.107 for auto (3x); with a 2x SR upscale auto
gives 0.214 against 0.248 for 0 (auto 14% better). The shipped default is therefore `auto`.

Timing (RX 7700 XT, synthetic, two direct signals, SR included, per frame, median): 1080p Quality
(1280x720 -> 1920x1080) 29.8 ms total, 1440p Quality (1707x960 -> 2560x1440) 50.6 ms total. GPU timestamps
(`D4R_FSR_RR_LOG`) at 1280x720 render: prepare 0.5 ms, **AMD's denoiser dispatch 21 ms**, compose+copy 0.2 ms: the
denoiser itself dominates. Linux reports about 8 ms for AMD's denoiser at 1280x720 on gfx1101 under RADV; the same
vkd3d-proton/dxil-spirv stack is about 2.6x slower here, on AMD's Windows Vulkan driver. Optimization parity with
Linux is NOT achieved, and the cause (driver code generation for the emulated FP8 shaders, cooperative-matrix use,
a missing vkd3d/dxil-spirv option) has not been identified. The 3-4 ms outside the denoiser (packet copies and
the CPU-synchronized hand-offs) is also unoptimized.

Limits: output may exceed render size only through the SR stage (without SR, output must equal render size);
dynamic render sizes must stay within creation dimensions and the SR library's queried dynamic range;
unsupported sizes fail before submission (see the newer dynamic-resolution evidence below);
diffuse is a direct signal (no diffuse hit distance); the exposure texture only goes to the SR stage; HDR +
render-resolution unjittered motion only; RGBA16F/RGBA32F output without SR; mapped float guide formats and depth
planes have focused synthetic coverage, while additional guide encodings remain unsupported; alpha output is not produced;
first RR start compiles AMD's shaders (~65 s, cached per game exe in
`%LOCALAPPDATA%\d4r\cache\vkd3d`). Each evaluation blocks the CPU until its GPU work completes (as the
native DLSS RR backend does), so it is not asynchronous. No game evidence exists for FSR RR.

## Dynamic resolution: SR range identified and enforced (2026-10-17)

The earlier `0xBAD00005` was caused by an invalid SR render size, not a blanket lack of dynamic-resolution support.
NVIDIA's own log reports Quality's minimum as half the output dimensions: the previous 640x360 -> 1280x720
fixture tried 576x324, below the 640x360 minimum. Plain SR reproduces that rejection with flags 0 and 11,
including reset-every-frame (`build/sr-dynamic-isolation-v1/`). Valid plain-SR subrects pass 24 synthetic frames
with creation 426x256, output 640x384 and smaller render 384x224 (`build/sr-dynamic-isolation-v2/summary.json`).

The internal SR feature now queries `DLSSOptimalSettingsCallback` using
[NVIDIA's helper contract](https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx_helpers.h).
FSR intersects the returned range with its existing creation-size bounds and rejects invalid sizes synchronously
before recording a command boundary. A missing or invalid query retains the previous creation-size-only range.
No ratio is hardcoded in production. The plain-SR probe validates/logs its requested subrect; the FSR wrapper
checks every reported extent and SR submission count. `--sr-bounds` tests each axis below the pinned Quality
minimum, restores valid dimensions, and completes subsequent evaluations on the same command list.

Moving-camera FSR+SR: creation 640x360, output 960x540, render sizes 640x360 / 576x324 / 480x270 and back,
24 frames per run. Both retained creation-sized textures and resized textures pass, without feature recreation.
Queried SR range is 480x270..960x540; creation dimensions still cap this FSR feature at 640x360.
Evidence: `build/fsr-rr-valid-dynamic-maxres-bounds-v1/summary.json` and
`build/fsr-rr-valid-dynamic-resized-bounds-v1/summary.json`. Both below-minimum axes reject before submission;
all 24 subsequent SR outputs are finite and nonuniform. Tail RMSE is 0.07665 versus noisy 0.48826 and ideal
nearest-resampled clean input 0.03894. This is synthetic denoising evidence, not exact/reference-quality parity:
maximum error within the sky mask reaches 0.78223. Alpha, other encodings, broader motion and games remain open.

## Format coverage and denoiser optimization trials (2026-10-17, later pass)

FSR-only output now accepts RGBA32F, including temporal switching with RGBA16F. Composition still runs at
half precision. A checked Vulkan nearest blit widens the completed half image into a lazily allocated float
image, then copies it through the existing shared packet. No CPU tensor conversion is added, and no new alpha
or FP32-denoiser claim is made. The ordinary FSR+SR path continues to hand SR a half image.

Synthetic moving-camera tests at 640x360 pass 24 frames each: canonical half output; alternating half/float
output plus alternating canonical/float guides; and RGBA32F output with typeless D32S8 depth and unchanged
stencil. `--verify-output-widening` compares every final output component against a diagnostic readback of
the same frame's composed half image, widening the independent reference where needed. All components match
exactly. Converted linear depth, motion and normals also match canonical dumps byte-for-byte across all 24
guide-switch frames. Evidence: `build/fsr-formats-{half,switch,d32s8}-composed-v1/`.
`test_fsr_rr_ngx.py --without-sr` uses the existing diagnostic environment contract because player startup
requires SR dependencies. It is not an installer/player-without-dependencies compatibility claim.

Three denoiser/compiler trials produced no reliable gain and are not promoted: cooperative conversion-loop
unrolling, forcing the minimum subgroup size, and retaining the RDNA3 layout workaround without global FP16
RTZ. AMD's denoiser and all quality settings/signals remain unchanged. At 1280x720, these fixtures start near
21.5 ms and reach about 31-32 ms in later frames, so a single whole-run median can mislead. Late-frame timings
stay within the control spread. Static and moving noisy fixtures retain the existing finite/nonuniform and
RMSE < 0.5 * noisy-input gate. No lower-resolution or skipped-signal shortcut is substituted.

Cross-process FSR output is not bit-exact even for repeated unchanged controls: in the 720p moving fixture,
frame 0 has 2,087,117 differing components and maximum absolute difference 0.1435546875 between two controls.
Their tail RMSE remains about 0.0612 versus noisy 0.4872. This precludes attributing inter-process image
differences solely to a candidate. The same-frame format-conversion gate avoids that ambiguity; K keeps its
strict full-frame exact gate. The FSR numerical variability is recorded, not fixed or reclassified as exact.
Evidence: `build/fsr-perf-720-{baseline-v2,baseline-repeat-v1,unroll-v2,wave32-v1,rte-v1}/` and
`build/fsr-perf-static-720-{baseline,unroll}-v1/`. Experimental binaries remain private under build/;
the shipped payload stays `build/fsr-payload-v1`. The two third-party source files edited for experiments
were restored; pre-existing libc++ include fixes remain. Linux denoiser performance parity is still open.

## History: the investigation before the vkd3d route

## Corrected provider descriptor recheck

The query now passes `FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER` in
`ffxQueryDescGetVersions.createDescType`, rather than its effect ID. AMD's
[API contract](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/Kits/FidelityFX/api/include/ffx_api.h)
defines this field as the creation description type. A bounded query against
the unchanged pinned AMD DLL and real RX 7700 XT still returns OK with zero
providers. The adapter reports shader model 6.6. No context is created or
denoiser dispatch attempted; FSR remains unavailable. Evidence:
`build/fsr-rr-correct-descriptor-v1/validation.json`. The earlier probe binary is
preserved in `build/fsr-rr-descriptor-baseline-v1/`. No driver, identity,
security, GPU setting or game changes occurred.

The native D3D12 provider check currently fails on RX 7700 XT. This does not
implement a working Windows FSR RR backend or DLSS RR replacement.

AMD SDK 2.3's unmodified denoiser 1.2.0.2740 loads successfully and accepts the
provider query, but returns an empty provider list. The installed driver reports
shader model 6.6. No context creation or denoiser dispatch follows this result.
The DLL's Authenticode signature was valid, signed by Advanced Micro Devices;
its SHA256 matches the Linux reference's tested binary:
`48f1e5888ba6a0a3d59a98b9751e37c392b0f7b8c223d0082d5c1f40642879d3`.

Linux's RDNA3 route uses vkd3d-proton, dxil-spirv scalar FP8 emulation, and
`wmma_rdna3_workaround`. Those are not native Windows driver features. Native
parity requires a compatible shader execution route as well as the DLSS guide
conversion and internal SR adapter. Merely importing the Linux backend or
selecting `Backend=fsr` would not supply those pieces.

The DLSS native D/E path already uses Linux's gfx11 shaders and lowering. A
read-only comparison against `34e6b0fe71d770729000a553f79b784e9b2dc279` verifies
25 shader/runtime/packer sources after CRLF/LF normalization. Both installer-built
D/E models verify all 66 products, 11 native 32-wide stages and the translated
64-wide output stage. Evidence: `build/rr-rdna3-source-parity-v1.json`.
This establishes source/recipe parity for native DLSS RR only. It provides no
FSR dispatch, FSR optimization parity or measured GPU-time parity. FSR's missing
native execution route remains a separate prerequisite.

## Reproduce without a game

```powershell
./scripts/windows/setup-fsr-rr.ps1
cmake --build build/native-engine-windows-gfx1101 --target d4r_fsr_rr_probe
./scripts/windows/test-fsr-rr.ps1 `
  -Probe build/native-engine-windows-gfx1101/d4r_fsr_rr_probe.exe `
  -DenoiserDll .tools/fsr-rr-1.2/amd_fidelityfx_denoiser_dx12.dll `
  -OutputDirectory build/fsr-rr-new-query -QueryOnly
```

Use a new output folder under ignored `build/`. The wrapper pins the DLL hash,
captures stdout/stderr, limits execution to 120 seconds, and propagates failure.
Without `-QueryOnly`, the harness proceeds only if a provider exists: it measures
48 synthetic frames with static/moving cameras and a reset against noise-free
signals. A successful query alone is not execution or image-quality evidence.
The Windows adapter helper queries the device's DXGI identity. It does not load
Proton driver shims or modify drivers, GPU settings, or security policy.

## Source provenance

`tools/fsr_rr_probe.cpp`, `tools/fsr_rr_scene.h`, `tools/d4r_fsr_rr.h`, and the
MIT-licensed FidelityFX API headers were imported from Linux reference
`34e6b0fe71d770729000a553f79b784e9b2dc279`. The probe includes the native Windows
identity helper instead of Linux's sysfs/vkd3d identity helper. AMD's DLL stays
under ignored `.tools/` and is not included in source or release packages.

Evidence: `build/fsr-rr-native-wrapper-query/probe.json`. No game was touched.
