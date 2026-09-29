# d4r (dlss 4 radeon)

d4r runs NVIDIA's official DLSS Super Resolution library (`nvngx_dlss.dll`) in Windows games on AMD Radeon GPUs under Linux and Proton. The game asks for DLSS as usual; the DLSS network runs on the AMD GPU through [ZLUDA](https://github.com/vosen/ZLUDA) (CUDA on ROCm/HIP), with the heaviest DLSS kernels replaced by hand-written RDNA3 code.

**Supported DLSS models:** DLSS 3 CNN (E), DLSS 4 transformer (K, default), and DLSS 4.5 transformer (M). DLSS 5 is not supported.

It is an experiment. It has been tested on one GPU in a handful of games, it depends on unreleased patches to ZLUDA and vkd3d-proton.

**Windows / RDNA4 development:** the `windows-rdna4` branch adds native Windows
gfx1201 HIP and ZLUDA diagnostics. Build commands, hardware results and remaining
porting gates are in [docs/windows-rdna4-port.md](docs/windows-rdna4-port.md).
This branch does not yet provide a Windows DLSS runtime.

See [supported games](SUPPORTED_GAMES.md) for the tested games and DLSS models.

> **Not affiliated with NVIDIA or AMD.**

## Results

[DLSS Ultra Performance demo video](https://cdn.ayois.gay/dlss).

Radeon RX 7700 XT (RDNA3, gfx1101), SILENT HILL Townfall at 2560×1440, every DLSS result is presented in the frame it belongs to (no added latency).

| Mode | DLSS 3 CNN (preset E) | DLSS 4 (preset K) | DLSS 4.5 (preset M) | FSR 4 |
|---|---|---|---|---|
| Quality (1705×960) | **72.4** fps | 69.4 | 51.5 | 76.0 |
| Balanced (1488×837) | **80.5** | 76.6 | 59.6 | 84.5 |
| Performance (1280×720) | **87.8** | 84.0 | 69.9 | 94.0 |
| Ultra Performance (853×480) | 88.9 | **94.5** | 90.3 | 107.3 |

For reference, native 2560×1440 without upscaling (the game's TSR at 100%) runs at 49.1 fps on the same walk.

On the same GPU DLSS starts at a disadvantage: its networks were designed for NVIDIA's tensor cores, and parts of them still run as translated NVIDIA code. How the numbers were measured, and what each optimisation contributed, is in [docs/performance.md](docs/performance.md).

## Known issues

- In some games native upscaling can show visible artifacting.

## GPU support

d4r needs RDNA3 (Radeon RX 7000 series), but full speed depends on the chip: the release ships its native kernels only for **gfx1101 (Navi 32)**, and on other GPUs DLSS 4 and 4.5 fall back to the much slower translated NVIDIA code.

| GPU | Chip | DLSS 3 CNN (E) | DLSS 4 (K) | DLSS 4.5 (M) |
|---|---|---|---|---|
| RX 7700 XT (tested), RX 7800 XT, RX 7700, Radeon PRO W7700 | gfx1101 | full speed | full speed | full speed |
| RX 7900 GRE / XT / XTX, Radeon PRO W7800 / W7900 | gfx1100 | full speed | runs without its transformer (worse image, slower) | about half speed |
| RX 7600 / 7600 XT, RX 7650 GRE | gfx1102 | full speed | runs without its transformer (worse image, slower) | about half speed |
| RDNA4, RDNA2 and older | | not supported | not supported | not supported |

Only the RX 7700 XT has been tested; the other gfx1101 cards use the same chip. Without the native kernels, NVIDIA's translated DLSS 4 layers produce invalid values and DLSS discards the network's output, so K loses the transformer's image quality. Preset E is unaffected on any RDNA3 GPU. The kernels can be built for gfx1100 and gfx1102 (`D4R_GPU_ARCHS` in `scripts/package_release.sh`), but those builds are untested.

## How it works

```
game (D3D12) ──► OptiScaler (DLSS inputs) ──► d4r_nvngx.dll  (NGX D3D12 API, tools/d4r_nvngx_shim.cpp)
                                                   │  inputs/output stay in VRAM (vkd3d-proton Vulkan interop,
                                                   │  command list split around DLSS for same-frame results)
                                                   ▼
                          official NGX core + nvngx_dlss.dll  (their CUDA path)
                                                   ▼
                          nvcuda.dll  (Wine CUDA bridge, tools/wine_nvcuda_bridge.c)
                                                   ▼
                          ZLUDA  (patches/zluda: PTX → AMDGPU, WMMA, native kernel overrides)
                                                   ▼
                          native RDNA3 kernels  (kernels/: DLSS 4 and 4.5 network layers)
```

- **The shim** (`d4r_nvngx.dll`) implements the D3D12 NGX entry points OptiScaler calls, copies the game's colour, depth and motion vectors into buffers shared with HIP, evaluates DLSS through the CUDA version of NGX, and writes the result back into the game's output texture.
- **The bridge** is a Wine builtin `nvcuda.dll` that forwards the CUDA driver API to ZLUDA on the Linux side, plus a few helpers the shim needs (Vulkan memory import, asynchronous array copies, GPU-side waits).
- **ZLUDA** compiles NVIDIA's PTX for the AMD GPU. The patches add what DLSS needs (textures, surfaces, FP8 and tensor-core MMA on RDNA3 WMMA) and a hook that serves hand-written kernels in place of selected PTX kernels.
- **Native kernels** reimplement the DLSS 4 and 4.5 network layers for RDNA3 and replace parts of a few texture-heavy kernels. They are 2–3× faster than the translated versions and are checked against numpy reference models of the layers.

Details: [docs/architecture.md](docs/architecture.md) and [docs/native-kernels.md](docs/native-kernels.md).

## Install the release

The release zip works like an OptiScaler release: its contents go into the folder that holds the game's main `.exe`.

1. Extract `d4r-<version>.zip` there. It contains:
   - OptiScaler 0.9.4 as `dxgi.dll`, with an `OptiScaler.ini` set up for d4r;
   - the d4r-patched vkd3d-proton (`d3d12.dll`, `d3d12core.dll`);
   - an `d4r` folder with the shim, the CUDA bridge, ZLUDA, the ROCm 7.2.4 runtime, the native kernels, this game's `d4r.ini`, and NVIDIA's `nvngx_dlss.dll` (310.7) and `_nvngx.dll`.
2. In Steam, select GE-Proton 11 for the game and set these launch options: `PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %command%`.

ROCm does not need to be installed: the zip includes its runtime (from AMD's Ubuntu 22.04 packages, which run under Steam's container runtime on any distribution). The zip's NVIDIA files and the kernels built from NVIDIA's code are not covered by this repository's license (see [NOTICE](NOTICE)). The Proton prefix and the system are not changed. [packaging/D4R_README.txt](packaging/D4R_README.txt) is the full guide that ships in the zip; `scripts/package_release.sh` builds the zip (see [docs/building.md](docs/building.md#7-package-a-release)).

## Requirements (building from source)

- Linux with an AMD RDNA3 GPU. The native kernels use gfx11 WMMA; they are built for gfx1101 by default (`D4R_GPU_ARCH` for other RDNA3 chips, untested). Without them DLSS runs through ZLUDA alone: much more slowly, and DLSS 4 (K) without its transformer (see [GPU support](#gpu-support)).
- ROCm with HIP and its clang (tested with ROCm 7.2).
- GE-Proton with OptiScaler integration (tested with GE-Proton11-3).
- Build tools: a Rust toolchain and git-lfs (ZLUDA), meson and ninja (vkd3d-proton), `winegcc`/`winebuild` (bridge), `x86_64-w64-mingw32-g++` and `clang-cl` (shim), Python 3.
- NVIDIA's `nvngx_dlss.dll` 310.7 and a matching NGX core `_nvngx.dll`.

## Build and run

The full sequence is in [docs/building.md](docs/building.md). In short:

1. Build ZLUDA `ee2f25a` with `patches/zluda/0002`–`0005` applied.
2. Build the patched vkd3d-proton: `scripts/build_vkd3d_proton_d4r.sh OUT_DIR`.
3. Build the shim and bridge: `scripts/build_d4r_nvngx_shim.sh`, `scripts/build_wine_nvcuda_bridge.sh`.
4. Stage the runtime with your NVIDIA files: `scripts/install_d4r_runtime.sh _nvngx.dll nvngx_dlss.dll`.
5. Build the native kernels: `D4R_ROCM_DIR=… D4R_DLSS_DLL=… D4R_ZLUDA_BUILD=… kernels/build.sh`.
6. Fill in `~/.config/d4r/d4r.ini` (created from `config/d4r.ini.default` on first launch) and start the game with `scripts/d4r_play.sh`.

The launcher backs up and restores everything it touches in the Proton prefix: OptiScaler.ini, the prefix's `d3d12.dll`/`d3d12core.dll`, and the game's Engine.ini when cvars are set.

## Repository layout

| Path | Contents |
|---|---|
| `tools/` | the NGX shim, the Wine CUDA bridge, a D3D12 DLSS harness, probes and kernel replay tools |
| `kernels/` | native RDNA3 kernels (`k/` DLSS 4, `m/` DLSS 4.5, `tex/` texture-kernel parts), their build script, validation tools and numpy reference models |
| `patches/` | ZLUDA and vkd3d-proton patches |
| `scripts/` | build, install, launch and probe scripts |
| `config/` | the default `d4r.ini` for the developer launcher |
| `packaging/` | the release's `d4r.ini`, OptiScaler settings, user guide and install check |
| `docs/` | architecture, build, native kernel and performance notes |

## License

Apache License 2.0 (see [LICENSE](LICENSE)). The patches in `patches/` are offered under the licenses of the projects they modify: ZLUDA (Apache-2.0 or MIT) and vkd3d-proton (LGPL-2.1). See [NOTICE](NOTICE).
