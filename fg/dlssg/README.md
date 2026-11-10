# DLSS Frame Generation (fg/dlssg)

NVIDIA DLSS Frame Generation (`nvngx_dlssg.dll`, 2x/3x/4x) on RDNA3 and RDNA4 Radeon GPUs under Proton, alongside
d4r's DLSS SR/RR. The game's own DLSS-G DLL runs unmodified. Its CUDA kernels are translated to SPIR-V on your
machine and run through a Vulkan layer that provides `VK_NVX_binary_import` on RADV.

Ported from [itchyfeetleech/dlssg-amd](https://github.com/itchyfeetleech/dlssg-amd) (commit 784abb8,
[d4r issue #56](https://github.com/countervolts/d4r/issues/56)). GPL-3.0-only like the rest of d4r; LICENSE and
NOTICE here are the upstream project's.

```
game - Streamline - OptiScaler (NvngxPath) - NGX host (nvngx.dll_dlssg_host.dll) - feature 11: nvngx_dlssg.dll
                                                                                  - other features: d4r's shim
nvngx_dlssg.dll - vkd3d-proton (cubin launches, d4r patch 0006) - nvxrt layer - RADV
```

- `host/`: NGX core for feature 11. Tells DLSS-G it runs on Blackwell, so multi-frame generation and the PTX kernels
  are used: through `Context.GPU.ArchID` and, for DLSS-G 310.7, which asks NVAPI instead, by wrapping
  `NvAPI_GPU_GetArchInfo` for that DLL alone (the rest of the process keeps Ada for DLSS SR's weights). Everything
  else goes to d4r's shim (`d4r\d4r_dlssg_next.dll`).
- `nvxrt/`: Vulkan layer implementing `VK_NVX_binary_import` / `VK_NVX_image_view_handle`. It stays inert unless the
  GPU is an RDNA3/RDNA4 RADV device and the kernel cache was built for that generation.
- `translator/`: PTX to SPIR-V. `PTXC_LAYOUT=gfx11` builds the WMMA operands in RDNA3's layout (A/B: 16 K elements
  per lane, lanes 16-31 repeating 0-15; verified against RADV and bit-identical between the shuffle and LDS paths).
  Doubles are kept as bits (loads and `cvt.f32.f64`, no Float64 needed); generic loads/stores are translated as global
  only when the pointer provably comes from kernel parameters (also through a local array).
- `t1/`: hand-written kernels for the network layers (DLSS-G 310.7.0 builds), bit-identical to the translation, on
  RDNA4 and RDNA3 (`_chain.py`'s streamed B operands in gfx11's layout; a shorter prefetch keeps the chained conv2
  groups co-resident on RDNA3). Persistent grids were sized for 56 CUs; `NVXRT_CUS` (read from KFD) rescales them.
  `pipe_bcf_3107.py` gives 310.7's BlendCandidatesFused the xor shuffles of `pipe_bcf.py`.
- The vkd3d-proton fix (cubin launches end the render pass and re-emit state) is d4r's
  `patches/vkd3d-proton/0006-cubin-launch-state.patch`, so d4r's one `d3d12.dll` serves both.

No NVIDIA code or data is in this folder. The kernel cache is derived from your DLL and must not be redistributed.

## Use (d4r release)

`install.sh` does this when the game ships `nvngx_dlssg.dll` (and again on every run, a no-op unless the DLL changed). By
hand, with the game closed, from the folder with the game's exe:

```sh
python3 d4r/d4r-features.py . --fg dlssg            # --exact, --lite, --dll PATH; --fg off undoes it
```

It translates the kernels of the game's `nvngx_dlssg.dll` (needs `python3` and `glslangValidator`; the installer
release bundles the latter) into `~/.cache/nvxrt` (RDNA4) or `~/.cache/nvxrt-gfx11` (RDNA3), routes OptiScaler's
NGX path through the host, sets OptiScaler's `Dxgi` and `SpoofHAGS`, `HwSchMode=2` in the game's Proton prefix, and
`d4r.ini [FrameGeneration] Backend = dlssg`. At launch `d4r_boot.dll` enables the layer, so no Steam launch options
are needed. Then enable DLSS Frame Generation in the game. `d4r.ini [FrameGeneration] Model = lite | full` picks
NVIDIA's FG Lite or the full model at the next launch (it overrides `dlssg_host.ini` Lite). Logs: `d4r/d4r_boot.log`,
`d4r/fg/dlssg/dlssg_host.log`, `d4r/fg/dlssg/nvxrt.log`. Run it again after a game update replaces the DLL.

- Tested: DLSS-G 310.7.0 (two builds). All DLSS-G 310.x DLLs on hand translate, including Streamline 2.14.16's
  (October 2026, DLSS-G 310.9.16, needed for FG Lite). DLSS-G 3.x keeps most of its PTX encrypted and is
  refused; `d4r-features.py` stops without changing anything if a kernel cannot be translated.
- Mesa 26.2.5 or newer (26.2.2 miscompiles the convolution kernels on RDNA4).
- Don't use it in online games with anti-cheat.

## Measured

Upstream, The Witcher 3, RX 9070, 2560x1440, RT on: off 108.8 fps; 2x 84.5 real / 169 shown (2.16 ms per real frame);
4x 70.8 / 283 (4.69 ms); 4x Lite 84.7 / 339.

d4r on RX 7700 XT (gfx1101), harness at 2560x1440, DLSS-G 310.7.0, fast mode, GPU ms per real frame:
2x 5.4, 4x 11.7, 6x 18.0 (translated kernels only: 9.5 / 22.2 / -). In Ghost of Tsushima at 4x: 10.1 ms
(4.75 + 2.68 + 2.65), 59-66 real fps.

FG Lite (`--lite`, DLSS-G 310.9.16 from Streamline 2.14.16), same setup: 2x 2.86, 4x 3.83, 6x 4.81 ms per real
frame (each further generated frame ~0.49 ms); Ghost of Tsushima 4x: 4.2-4.5 ms, 76-82 real fps. Details of each step: private/notes/dlssg-rdna3-opt.md (not published).

## Build and test

`scripts/build_dlssg.sh` builds the host and the layer; `scripts/package_release.sh` stages them into
`d4r/fg/dlssg`. `harness/run.sh` runs a DLSS-G DLL on a synthetic scene without a game
(`NVXRT_ARCH=gfx11 VKD3D_DIR=<d4r vkd3d build> harness/run.sh nvngx_dlssg.dll outdir --mfc 3`).
More detail: [docs/architecture.md](docs/architecture.md).
