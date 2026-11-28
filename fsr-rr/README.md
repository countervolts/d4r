# FSR Ray Regeneration on AMD RDNA 3 (Linux / Proton)

AMD's **FSR Ray Regeneration 1.2** denoiser, which AMD ships only for RX 9000 on Windows 11, running on an RDNA 3 card (tested on an RX 7700 XT, gfx1101).
d4r answers a game's DLSS Ray Reconstruction request (NGX feature 13) with AMD's own, unmodified `amd_fidelityfx_denoiser_dx12.dll`
instead of NVIDIA's denoiser. Select it with `[RayReconstruction] Backend = fsr`; `Backend = dlss` (the default) keeps d4r's DLSS Ray Reconstruction.

Status: works under Proton with the synthetic test scene. **Image quality in real games is not validated.** This route is Linux/Proton only.
Full write-up (what runs, requirements, tuning keys, limitations, measured cost): [docs/fsr-ray-regeneration.md](../docs/fsr-ray-regeneration.md).

## How it works
1. `fsr_rr_prepare.comp` turns the DLSS-RR inputs into the denoiser's inputs (linear depth, motion, octahedral normals, albedo) and splits the noisy colour into demodulated diffuse and specular.
2. AMD's denoiser runs through `ffxDispatch`.
3. `fsr_rr_compose.comp` remodulates and adds the residual (sky, particles).
4. d4r's own Super Sampling feature upscales to the output size.

## What makes it work on RDNA 3
AMD's denoiser uses FP8 WMMA, which RDNA 3 does not have. The patched vkd3d-proton emulates it
([patches/vkd3d-proton-fsr-rr/](../patches/vkd3d-proton-fsr-rr/)) and the game runs with `DXIL_SPIRV_CONFIG=wmma_rdna3_workaround`.
Also needed: AMD's denoiser DLL (not in this repository) and Proton's `amdxc64.dll`.

## Where the code is
| What | Where |
|---|---|
| Backend (Linux/Proton) | [tools/d4r_fsr_rr_backend.inl](../tools/d4r_fsr_rr_backend.inl), `tools/d4r_fsr_rr*.h` |
| Guide conversion shaders | [kernels/support/](../kernels/support/) `fsr_rr_prepare.comp`, `fsr_rr_compose.comp` |
| vkd3d-proton patches (FP8 emulation, built-in shader replacements) | [patches/vkd3d-proton-fsr-rr/](../patches/vkd3d-proton-fsr-rr/) |
| Test scene and probe | [tools/fsr_rr_probe.cpp](../tools/fsr_rr_probe.cpp), [tools/fsr_rr_scene.h](../tools/fsr_rr_scene.h) |
| Probe scripts | [scripts/build_fsr_rr_probe.sh](../scripts/build_fsr_rr_probe.sh), [scripts/run_fsr_rr_probe_proton.sh](../scripts/run_fsr_rr_probe_proton.sh) |
| Native Windows route (RDNA 4, original D3D12 denoiser) | [docs/windows-fsr-rr.md](../docs/windows-fsr-rr.md), [tools/windows/fsr_rr_backend.cpp](../tools/windows/fsr_rr_backend.cpp) |
| DLSS Ray Reconstruction backend (the default) | [engine/rr/](../engine/rr/README.md) |

The code stays in these shared locations because the main d4r build uses it; this folder is the entry point.
