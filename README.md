# d4r (dlss 4 radeon)

d4r runs NVIDIA's official DLSS Super Resolution library (`nvngx_dlss.dll`) in Windows games on AMD Radeon GPUs under Linux and Proton. The game asks for DLSS as usual; the DLSS network runs on the AMD GPU through [ZLUDA](https://github.com/vosen/ZLUDA) (CUDA on ROCm/HIP), with the heaviest DLSS kernels replaced by hand-written RDNA3 and RDNA4 code.

**Supported DLSS models:** DLSS 3 CNN (E), DLSS 4 transformer (K, default), and DLSS 4.5 transformer (M, plus experimental L). DLSS 5 is PURPOSELY not supported.

Preset L is NVIDIA's DLSS 4.5 model for Ultra Performance, especially at 4K. Set `[DLSS] Model = L` in `d4r.ini` and select Ultra Performance in game or OptiScaler; selecting L alone does not change the input resolution. L shares M's native Swin layers and has its own texture-kernel build. Its unfolded input/output arithmetic remains translated, so performance and image quality need game testing; the M benchmarks below do not apply to L.

**Proof of concept:** d4r shows that DLSS can run on an AMD GPU, but it is not really that practical for everyday use yet. It has been tested on one GPU in a handful of games and depends on unreleased patches to ZLUDA and vkd3d-proton.

See [supported games](SUPPORTED_GAMES.md) for the tested games and DLSS models.

The development shim also has [experimental native Vulkan support](docs/vulkan.md).

A [standalone native inference engine](engine/README.md) runs presets K and M from weights extracted from your own
`nvngx_dlss.dll`, with d4r's own Vulkan input, network and reconstruction shaders (optionally the native HIP network
layers) and GPU-resident temporal history. The game shim uses the CUDA/ZLUDA path described below by default;
`[Engine] Enabled = true` (`D4R_ENGINE=1`) records supported K and M evaluations with the native engine into the
game's own command list instead and hands anything else to the CUDA path (see the engine's README). Presets E and L
remain CUDA-only.

> **Not affiliated with NVIDIA or AMD.**


To contact me my discord is `._ayo`. or join the [d4r discord server](https://discord.gg/mgTWpSq8f5)

## Results

[DLSS Ultra Performance demo video](https://cdn.ayois.gay/dlss).

Radeon RX 7700 XT (RDNA3, gfx1101), SILENT HILL Townfall at 2560×1440, every DLSS result is presented in the frame it belongs to (no added latency).

| Mode | DLSS 3 CNN (preset E) | DLSS 4 (preset K) | DLSS 4.5 (preset M) | FSR 4\* |
|---|---|---|---|---|
| Quality (1705×960) | **71.7** fps | 67.9 | 49.3 | 76.0 |
| Balanced (1488×837) | **80.8** | 75.9 | 58.0 | 84.5 |
| Performance (1280×720) | **89.7** | 84.1 | 69.0 | 94.0 |
| Ultra Performance (853×480) | 89.1 | **95.5** | 92.2 | 107.3 |

\* FSR 4 was measured in an earlier session (2026-09-27); the DLSS columns on 2026-09-29, when the same machine ran about 2–4% slower overall. Run back to back on the same day, this release is faster than 0.1.1 at Quality with every model (E 71.2 → 71.9, K 67.9 → 68.1, M 48.4 → 49.3 fps), and more in games whose DLSS settings differ from Townfall's, whose kernel variants now run natively too.

For reference, native 2560×1440 without upscaling (the game's TSR at 100%) runs at 49.1 fps on the same walk.

On the same GPU DLSS starts at a disadvantage: its networks were designed for NVIDIA's tensor cores, and parts of them still run as translated NVIDIA code. How the numbers were measured, and what each optimisation contributed, is in [docs/performance.md](docs/performance.md).

## FSR 4.1.1 vs. DLSS 4.5 comparison

- [Side-by-side comparison: FSR 4.1.1 (left) and DLSS 4.5 preset M (right)](https://cdn.ayois.gay/sk-ant-api03-4Dr0yN5YAzsuAOcyp7KEfeRyB0pbh2iAv2vfywa9QITHIbQScnVQnef9jsxJzDcvWMD13r8C1qGAoQQevjLqzGPT0bAQ9AA)
- [FSR 4.1.1 standalone video](https://cdn.ayois.gay/sk-ant-api03-YZpWqINTI2HBt6KWJTeyY2b9JJbxUTZdBVRR06KZD7b1x27QLjvQrZZX8Um5sZqkquuOlzJTHUh84FOOLsetNy-PAIiWxAA)
- [DLSS 4.5 preset M standalone video](https://cdn.ayois.gay/sk-ant-api03-0GLTr6qEYGfUKWYoZeseLdTzPslh1AvlVLdlBEUvV893bi19b7qha5wKnuQXN2tl6l4ffaRKj5warfAFDDjOlYJNYWVQ3AA)

DLSS 4.5 was using the following settings,

```
PreferAccuracy = true

NativeSwinEncoders = false

MotionVectorDilation = 2
```

## Known issues

- In some games using some models native upscaling can show visible artifacting.

For users who prioritize fidelity over speed, set `[Kernels] PreferAccuracy = true` in `d4r/d4r.ini` and restart the game. It defaults to `false`. This selects accuracy variants of every native kernel and restores conservative translation settings. It aims to match NVIDIA's arithmetic; 1:1 image quality against RTX DLSS is not yet proven. See [native kernel numerics](docs/native-kernels.md#numerics).

## GPU support

d4r builds for RDNA3 and RDNA4. A newly built release compiles native DLSS 4 and 4.5 network kernels for the targets below and selects the matching set at runtime. Only the RX 7700 XT has been tested by this project on a real GPU; an external video reports DLSS 4.5 running through d4r 0.1.2 on an RX 7900 XTX. RDNA4 runtime and performance remain unverified on hardware.

!WE ARE LOOKING FOR VIDEOS OF IT WORKING, IF YOU HAVE A MENTIONED HERE AND DOESNT HAVE A VIDEO PLEASE TAKE ONE!

| GPU | Chip | FP8 math | Runtime testing |
|---|---|---|---|
| RX 7700 XT, RX 7800 XT, RX 7700, Radeon PRO W7700 | gfx1101 | widened to f16 | RX 7700 XT only |
| RX 7900 GRE / XT / XTX, Radeon PRO W7800 / W7900 | gfx1100 | widened to f16 | RX 7900 XTX: [external video](https://www.youtube.com/watch?v=_GLjJ2Dn5pU) (DLSS 4.5); other cards untested |
| RX 7600 / 7600 XT, RX 7650 GRE | gfx1102 | widened to f16 | untested |
| RDNA3 integrated GPUs | gfx1103 | widened to f16 | untested |
| RX 9070 XT / 9070 / 9070 GRE, Radeon AI PRO R9700 | gfx1201 | native FP8 (`NativeFp8`, default on) | RX 9070 XT: [external video](https://www.youtube.com/watch?v=lZ0BLYqAtCs), other cards untested |
| RX 9060 XT / 9060 | gfx1200 | native FP8 (`NativeFp8`, default on) | compile only |
| RDNA2 and older | | | unsupported |

Native network kernels are built for all listed gfx11/gfx12 targets. Texture kernels are compiled for each target by `d4r_emit` when supplied to the package script; RDNA4 also has a `-fp8` variant. The bridge selects the KFD GPU with the most SIMDs, avoiding an integrated GPU when a discrete GPU is present; `D4R_GPU_ARCH` overrides that choice. Missing native kernels fall back to ZLUDA and can be much slower. Earlier translated K layers produced invalid values; preserving FP16 denormal handling fixes that failure in recorded captures, but RTX image-quality parity remains unverified. Preset E does not use the native network kernels.

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
                          native RDNA3 and RDNA4 kernels  (kernels/: DLSS 4 and 4.5 network layers)
```

- **The shim** (`d4r_nvngx.dll`) implements the D3D12 NGX entry points OptiScaler calls, copies the game's colour, depth and motion vectors into buffers shared with HIP, evaluates DLSS through the CUDA version of NGX, and writes the result back into the game's output texture.
- **The bridge** is a Wine builtin `nvcuda.dll` that forwards the CUDA driver API to ZLUDA on the Linux side, plus a few helpers the shim needs (Vulkan memory import, asynchronous array copies, GPU-side waits).
- **ZLUDA** compiles NVIDIA's PTX for the AMD GPU. The patches add what DLSS needs (textures, surfaces, FP8 and tensor-core MMA on RDNA3/RDNA4 WMMA) and a hook that serves hand-written kernels in place of selected PTX kernels.
- **Native kernels** reimplement the DLSS 4 and 4.5 network layers for RDNA3 and RDNA4 and replace parts of a few texture-heavy kernels. The 2–3× speedup was measured on the RX 7700 XT; RDNA4 speed is unverified.

Details: [docs/architecture.md](docs/architecture.md) and [docs/native-kernels.md](docs/native-kernels.md).

## Install the release

The release zip works like an OptiScaler release: its contents go into the folder that holds the game's main `.exe`.

1. Extract `d4r-<version>.zip` there. It contains:
   - OptiScaler 0.9.4 as `dxgi.dll`, with an `OptiScaler.ini` set up for d4r;
   - the d4r-patched vkd3d-proton (`d3d12.dll`, `d3d12core.dll`);
   - a `d4r` folder with the shim, the CUDA bridge, ZLUDA, the ROCm 7.2.4 runtime, the native kernels, this game's `d4r.ini`, and an `install.sh` beside the `d4r` folder.
2. Run `bash install.sh` in the extracted release folder. It downloads NVIDIA DLSS 310.7.0 and NGX from official NVIDIA sources, verifies SHA-256 hashes, and places them in `d4r/nvngx_dlss.dll` and `d4r/ngx/_nvngx.dll`. Requires Bash, curl, sha256sum and 7-Zip (`7zz`, `7z` or `7za`); no sudo. The NGX download is a roughly 915 MiB driver archive which is extracted without executing it. Matching files are skipped and differing files are backed up.
3. In Steam, select GE-Proton 11 for the game and set these launch options: `PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %command%`.

When updating, replace `d3d12.dll` and `d3d12core.dll` together with `d4r/nvngx.dll`. The shim requires their matching resource-lifetime extension.

ROCm does not need to be installed: the zip includes its runtime (from AMD's Ubuntu 22.04 packages, which run under Steam's container runtime on any distribution). The downloaded NVIDIA files and the bundled kernels built from NVIDIA's code are not covered by this repository's license (see [NOTICE](NOTICE)). The Proton prefix and the system are not changed. [packaging/D4R_README.txt](packaging/D4R_README.txt) is the full guide that ships in the zip; `scripts/package_release.sh` builds the zip (see [docs/building.md](docs/building.md#7-package-a-release)).

## Requirements (building from source)

- Linux with an AMD RDNA3 or RDNA4 GPU. The native kernels use gfx11 or gfx12 WMMA. `scripts/package_release.sh` builds network kernels for gfx1100–gfx1103 and gfx1200–gfx1201 by default; `D4R_GPU_ARCH` selects one target for a standalone `kernels/build.sh` invocation. Runtime support outside the RX 7700 XT remains unverified on hardware (see [GPU support](#gpu-support)).
- ROCm with HIP and its clang (tested with ROCm 7.2).
- GE-Proton with OptiScaler integration (tested with GE-Proton11-3).
- Build tools: a Rust toolchain and git-lfs (ZLUDA), meson and ninja (vkd3d-proton), `winegcc`/`winebuild` (bridge), `x86_64-w64-mingw32-g++` and `clang-cl` (shim), Python 3.
- NVIDIA's `nvngx_dlss.dll` 310.7 and a matching NGX core `_nvngx.dll`.

## Build and run

The full sequence is in [docs/building.md](docs/building.md). In short:

1. Build ZLUDA `ee2f25a` with `patches/zluda/0002`–`0008` applied, including its `d4r_emit` example for offline texture builds.
2. Build the patched vkd3d-proton: `scripts/build_vkd3d_proton_d4r.sh OUT_DIR`.
3. Build the shim and bridge: `scripts/build_d4r_nvngx_shim.sh`, `scripts/build_wine_nvcuda_bridge.sh`.
4. Stage the runtime with your NVIDIA files: `scripts/install_d4r_runtime.sh _nvngx.dll nvngx_dlss.dll`.
5. Build the native kernels: `D4R_ROCM_DIR=… D4R_DLSS_DLL=… D4R_ZLUDA_EMIT=… kernels/build.sh`.
6. Fill in `~/.config/d4r/d4r.ini` (created from `config/d4r.ini.default` on first launch) and start the game with `scripts/d4r_play.sh`.

The launcher backs up and restores everything it touches in the Proton prefix: OptiScaler.ini, the prefix's `d3d12.dll`/`d3d12core.dll`, and the game's Engine.ini when cvars are set.

## Preset videos

Gameplay recordings of each DLSS model at the Quality, Performance and Ultra Performance presets.

**Ready or Not**

| Model | Quality | Performance |
|---|---|---|
| DLSS 4.5 | [video](https://cdn.ayois.gay/sk-ant-api03-OPG1vqqsmDkNduG5fXJBelDI-1APyL9_ytJrnOSekZxoh-2NL8NqNuqL3pCI24kJQsudaB896GYpu5TFv93tabAgK1H2sAA) | [video](https://cdn.ayois.gay/sk-ant-api03-d-4XS1it5mtMkm-QRz0Qnkfu9FAjz_dFJURvsh_3gPZTnC1sljpWc28F6fyjUWIC6xlZm-FckWGjaP-KuIipYmroa3dQ8AA) |
| DLSS 4 | [video](https://cdn.ayois.gay/sk-ant-api03-uXTWm4hN5jZksiSm4fvgIOr0dw43hGJ0TpZLe2ZghIbGb7aR9CGGlTV-1paBYO7G6lzIf83tOrCdaKdJerJ4re6epRWDEAA) | [video](https://cdn.ayois.gay/sk-ant-api03-fcZCkctdFOv513z1u6rPBrRseY47Qnw43Gnv5mZf_GekHw-yUoWYcBixjp2qqqbBIf7sVmht0Y676U96n2cwmFIsRgWkHAA) |
| DLSS 3 | [video](https://cdn.ayois.gay/sk-ant-api03-_vuywLcsMzZiJxcNgucBeeXFzWi44mes5A1777bg_4XQ2X0MH22J9M7JFpTS0_q6yaGWMXf3FvL8MDiv_dc7UANxrCmlGAA) | [video](https://cdn.ayois.gay/sk-ant-api03-WzaIy34WIesQeOOTVQNXZwpPPE8khmbUynvMw9B-Luj8GFcZ_sJvGXfg6aW00TD0Rr3nWu43ELJGx2K6qs_HL6FkLdmb1AA) |

**Townfall**

| Model | Quality | Ultra Performance |
|---|---|---|
| DLSS 4.5 | [video](https://cdn.ayois.gay/sk-ant-api03-p8uL0hOUklvA-BSeEafS8LQU-gayzbqAE3NacWAmckLoXOSf2sXGf7OeG1LmvOqeGFRh_ceCWi4pLBl9yxJlvYwQWH3Y1AA) | [video](https://cdn.ayois.gay/sk-ant-api03-Mz0BzYTrLjaYSuQXtOEHTw4ydYVbPvHmMjflm8UhFp976ftXqMExD8BCMYlo93JLEGwVL3rMWoZR0CH_deJJU7jX8_s3GAA) |
| DLSS 4 | [video](https://cdn.ayois.gay/sk-ant-api03-WgNoUPTwH0vt55Ag7RaWprXXINU5OOOEtuDXOvIjHgfAScSiIPIJGVaERoF-EA3JO7LS8xkohUlb6AFsW8AVZ2vMB14JwAA) | [video](https://cdn.ayois.gay/sk-ant-api03-skXZVD6G4DUdAfQw0RLz_UJW8GOfBlpZx1N4ezaF_UIXm7OIY1ciQysfRu4zRsxSXA8plZJi4NSzkXkV7Oxmg9ytg9zaaAA) |

## Repository layout

| Path | Contents |
|---|---|
| `tools/` | the NGX shim, the Wine CUDA bridge, a D3D12 DLSS harness, probes and kernel replay tools |
| `kernels/` | native RDNA3/RDNA4 kernels (`k/` DLSS 4, `m/` DLSS 4.5, `tex/` texture-kernel parts, `common/` WMMA layouts), their build script, validation tools and numpy reference models |
| `engine/` | standalone weight-based preset K and M inference runtime (Vulkan shaders, optional native HIP network), model packaging and full-frame validation |
| `patches/` | ZLUDA and vkd3d-proton patches |
| `scripts/` | build, install, launch and probe scripts |
| `config/` | the default `d4r.ini` for the developer launcher |
| `packaging/` | the release's `d4r.ini`, OptiScaler settings, user guide and install check |
| `docs/` | architecture, build, native kernel and performance notes |

## Star History

<a href="https://www.star-history.com/?type=date&repos=countervolts%2Fd4r">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=countervolts/d4r&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=countervolts/d4r&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=countervolts/d4r&type=date&legend=top-left" />
 </picture>
</a>

## License

GNU General Public License, version 3 only (GPL-3.0-only; see [LICENSE](LICENSE)). The patches in `patches/` are offered under the licenses of the projects they modify: ZLUDA (Apache-2.0 or MIT) and vkd3d-proton (LGPL-2.1). See [NOTICE](NOTICE).
