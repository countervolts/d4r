# FSR Ray Regeneration as the Ray Reconstruction backend (experimental)

`[RayReconstruction] Backend = fsr` answers a game's DLSS Ray Reconstruction request (NGX feature 13)
with AMD's FSR Ray Regeneration 1.2 instead of NVIDIA's denoiser. `Backend = dlss` (the default) keeps
d4r's existing DLSS Ray Reconstruction. Changing the backend needs a game restart, or the game
recreating its Ray Reconstruction feature.

Status: works on an RX 7700 XT (gfx1101) under Proton with the synthetic test scene described below.
Alan Wake 2 logs from 2026-10-14 confirm denoiser dispatch and internal DLSS K upscaling, but the
game test reports excessive noise and artifacts. **Game image quality is not validated.**
AMD supports FSR Ray Regeneration only on RX 9000 / Windows 11. This
route is Linux/Proton only, and nothing here is validated on native Windows or for native Vulkan games.

## What runs

In the game's own command list and frame:

1. `kernels/support/fsr_rr_prepare.comp` (Vulkan, through vkd3d-proton's interop) turns the DLSS-RR
   inputs into the denoiser's inputs:
   - signed linear depth (from hardware or linear depth and the game's ViewToClip);
   - previous-minus-current UV motion with the camera's view-z delta;
   - octahedral normals plus roughness;
   - linear albedos.

   It splits the noisy colour into a diffuse and a specular signal by albedo and demodulates both
   (after Zach Hembree's and burak113's GPL OptiScaler adapter). What cannot be demodulated, the sky,
   and particles (when `DLSSD.ColorBeforeParticles` is given) go to a residual that bypasses the
   denoiser.
2. AMD's `amd_fidelityfx_denoiser_dx12.dll`, unmodified, runs through `ffxDispatch`. Each lobe uses
   the indirect path only when the game supplies that lobe's hit distance; otherwise it uses the
   direct path.
3. `fsr_rr_compose.comp` remodulates the denoised signals and adds the residual.
4. An internal Super Sampling feature, configured like the user's own (`[DLSS] Model`, engine or
   CUDA), upscales to the output size the game asked for.

`nvngx_dlssd.dll` is not used. When a frame cannot be denoised (missing guides or matrices, jittered
motion vectors, `DLSSD.OutputAlpha` requested), the evaluation fails and the reason is logged. It is
never upscaled noisy as if it had been denoised.

## Requirements

- AMD's `amd_fidelityfx_denoiser_dx12.dll` from FidelityFX SDK 2.3. Tested with SHA-256
  `48f1e5888ba6a0a3d59a98b9751e37c392b0f7b8c223d0082d5c1f40642879d3`, file version 1.2.0.2740.
  Put it in the d4r folder, or set `[RayReconstruction] FsrDenoiser`. It is AMD's binary under
  AMD's terms, not part of this repository.
- d4r's vkd3d-proton built with `D4R_VKD3D_FSR_RR=1 scripts/build_vkd3d_proton_d4r.sh OUT`. This build
  adds upstream dxil-spirv's accepted scalar FP8 fallback (PR 307) and extended emulation.
- `DXIL_SPIRV_CONFIG=wmma_rdna3_workaround` in the game's environment.
- Proton's `amdxc64.dll`. vkd3d-proton loads it only when the GPU is reported as AMD, so under
  `PROTON_FORCE_NVAPI` the shim loads it after confirming through sysfs that the GPU is AMD's.
  Without it, AMD's runtime offers no provider.

## Tuning

Two `[RayReconstruction]` keys adjust the denoiser. Both are off by default.

- `FsrStabilityBias` (`D4R_FSR_RR_STABILITY_BIAS`): AMD's temporal stability bias. `auto` keeps AMD's
  default of 1. With that default, reflections and fine detail come out blurred. On Alan Wake 2's water,
  the result had about half the detail of DLSS Ray Reconstruction. `0` brought the replayed water capture
  from 29.5 to 36.6 dB PSNR against DLSS RR's output. On the synthetic scene it roughly halved the error
  against ground truth, whether the camera was still, moving or just after a cut. GPU time was unchanged.
- `FsrDetail` (`D4R_FSR_RR_DETAIL`, 0–1): the share of the colour that skips the denoiser and goes
  through the residual. It keeps detail the denoiser removes as noise: hair, water ripples and foliage.
  The cost is some grain. With `0.25` on top of stability bias 0, PSNR against DLSS Ray Reconstruction's
  output went up by 0.4 dB on the water capture, 1.4 dB on the street capture and 0.8 dB on a walking
  capture in town. Frame-to-frame flicker off the water went from 0.0069 to 0.0073 (DLSS RR: 0.0060).

Two guards are on by default, because a game's single colour has light in it that the albedos did not
shape, such as fog, sky light, or the background behind a thin object's edge:

- No albedo below 0.1 divides a signal (`D4R_FSR_RR_DEMOD_FLOOR`). Dividing fog by dark foliage
  overflowed, and the fog lost its colour and haze.
- Each denoised lobe is capped at the brightest noisy input of that lobe in its 3×3 neighbourhood
  (`D4R_FSR_RR_HALO_LIMIT`, 0 turns it off). Without the cap, the denoiser leaks a thin object's large
  demodulated signal across its edge, which shows as bright or blue fringes around poles, wires, hair
  and the edges of water.

`D4R_FSR_RR_KEY_<n>=<value>` overrides any of AMD's six configuration keys, for diagnostics. The log
lists AMD's defaults when the denoiser context is created.

## Known limitations

- One noisy colour cannot be split exactly into diffuse and specular. On the test scene this route
  flattens some shading gradients compared with DLSS Ray Reconstruction.
- Normals are assumed to be in world space. The depth delta comes from the camera's motion only, so
  moving objects get no depth-change term.
- Guide subrect origins, jittered motion vectors and alpha output are not supported.
- Cost scales with render resolution. AMD's denoiser alone, on gfx1101:
  - about 4 ms at 960×540 (2 signals);
  - about 8 ms at 1280×720;
  - about 26 ms at 1920×1080 (4 signals).

## Checks

- `tools/fsr_rr_probe.cpp` (`scripts/run_fsr_rr_probe_proton.sh`) tests AMD's runtime alone: provider,
  context, multi-frame denoising measured against ground truth, and reset.
- `tools/rr_backend_check.cpp` (`scripts/run_rr_backend_check_proton.sh`) drives a feature-13 request
  through the shim and compares the output with the scene rendered noise-free at output resolution.
  `--feature 1` gives the Super-Sampling-only baseline. With `D4R_PROFILE=1`, the log reports GPU time
  per stage.
- `--hw-depth --separate-eval-params` checks callers that put depth/roughness modes only on the
  creation parameter object. The FSR adapter uses the resolved creation/per-frame settings; it
  must not silently interpret hardware depth as linear when a frame omits the creation setting.
