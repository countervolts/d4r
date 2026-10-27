# d4r (dlss 4 radeon)

want to run xess (xmx) on radeon? check out [x4r](https://github.com/countervolts/x4r)!

d4r runs NVIDIA's official DLSS Super Resolution (`nvngx_dlss.dll`) and experimental Ray Reconstruction (`nvngx_dlssd.dll`) in Windows games on AMD Radeon GPUs under Linux and Proton. The game asks for DLSS as usual; the DLSS network runs on the AMD GPU through [ZLUDA](https://github.com/vosen/ZLUDA) (CUDA on ROCm/HIP), with the heaviest kernels replaced by hand-written RDNA2, RDNA3 and RDNA4 code.

**Supported DLSS models:** DLSS 3 CNN (E), DLSS 4 transformer (K, default), and DLSS 4.5 transformer (M, plus L for Ultra Performance). Ray Reconstruction presets D, E and F are experimental ([building](docs/building.md)). DLSS 5 is PURPOSELY not supported.

**Proof of concept:** d4r shows that dlss can be ran on radeon gpus, See [supported games](SUPPORTED_GAMES.md) for the tested games and models. As well check the charts out below.

> **Not affiliated with NVIDIA or AMD.**

To contact me my discord is `._ayo`. or join the [d4r discord server](https://discord.gg/mgTWpSq8f5)

## Results

[DLSS Ultra Performance demo video](https://cdn.ayois.gay/dlss) · [more videos](docs/videos.md)

All tests are using DLSS K and FSR 4.1.1 and at 1440p (unless specified otherwise).

Radeon RX 7700 XT, SILENT HILL Townfall.

| Mode | DLSS 3 CNN (E)\* | DLSS 4 (K) | DLSS 4.5 (M)\* | FSR FPS | DLSS K upscale time | FSR upscale time |
|---|---|---|---|---|---|---|
| Quality | 71.7 fps | **75.7** | 49.3 | 73.3 | **2.05 ms** | 2.42 ms |
| Balanced | 80.8 | **82.5** | 58.0 | 80.2 | **2.01 ms** | 2.39 ms |
| Ultra Performance | 89.1 | **100.6** | 92.2 | 96.9 | **1.99 ms** | 2.26 ms |

DLSS 4 (K) FPS was measured on 2026-10-21, Using no approximations. 
\* Presets E and M are from 2026-09-29, on an older d4r build, so compare them with each other rather than with the newer columns.

Radeon RX 7700 XT, Ghost of Tsushima.

| Mode | DLSS 4 (K) | FSR | DLSS upscale time | FSR upscale time |
|---|---|---|---|---|
| Quality | **90.9** | 90.2 | **1.99 ms** | 2.50 ms |
| Balanced | **93.8** | 93.2 | **1.96 ms** | 2.47 ms |
| Ultra Performance | 101.7 | **102.5** | **1.91 ms** | 2.36 ms |

Measured on 2026-10-21, Using no approximations. 
K and FSR 4.1.1 are within 1% across all three modes, with K slightly ahead at Quality and Balanced and FSR slightly ahead at Ultra Performance.

Radeon RX 7700 XT, Cyberpunk 2077.

| Mode | DLSS 4 (K) | FSR | DLSS upscale time | FSR upscale time |
|---|---|---|---|---|
| Quality | **87.6** | 86.4 | **2.08 ms** | 2.28 ms |
| Balanced | **99.5** | 98.0 | **2.04 ms** | 2.20 ms |
| Ultra Performance | **134.0** | 132.5 | **1.98 ms** | 2.08 ms |

Measured on 2026-10-21, Using no approximations. 
DLSS averages 1.1–1.5% higher FPS than FSR across these runs, with upscale times 0.10–0.20 ms lower.

## Switching models and FSR in game

Open OptiScaler's menu (Insert):

- **DLSS model:** under DLSS Settings, tick *Render Presets Override*, pick the preset (E, K, L or M) and press *Apply Changes*. Unticking it returns to `[DLSS] Model` in `d4r/d4r.ini`.
- **FSR:** set *Upscaler (DX12)* to *FSR 3.X/4*. RDNA3 and RDNA4 get FSR 4.1 (needs GE-Proton, which supplies AMD's FSR 4 library); other GPUs get FSR 3.1.

Both are on by default; `[OptiScaler] FollowPresets` and `Fsr4` in `d4r.ini` turn them off.

Change OptiScaler settings and use **Save INI** normally. Steam launches use the
saved INI; d4r's ASI setup also preserves your choices after initial installation.

The [vkhip backend](docs/experimental-vkhip.md) runs preset K on RDNA3 and is on by default.

Experimental [XeFG/FSR frame generation and XeLL/Anti-Lag latency conversion](docs/frame-generation.md)
are bundled in the `OptiScaler/` folder. Enable FG in OptiScaler’s Insert menu or use the
per-game setup command with INI backups. Fresh installs keep FG off. The pinned v10
build is a prerelease; game testing is still required.

## Known issues

- In some games using some models native upscaling can show visible artifacting. For fidelity over speed, set `[Kernels] PreferAccuracy = true` in `d4r/d4r.ini` ([native kernel numerics](docs/native-kernels.md#numerics)).

## GPU support

!WE ARE LOOKING FOR VIDEOS OF IT WORKING, IF YOU HAVE A MENTIONED HERE AND DOESNT HAVE A VIDEO PLEASE TAKE ONE!

| GPU | Chip | Native kernels | Runtime testing |
|---|---|---|---|
| RX 7700 XT, RX 7800 XT, RX 7700, Radeon PRO W7700 | gfx1101 | K, M, L (WMMA) | RX 7700 XT |
| RX 7900 GRE / XT / XTX, Radeon PRO W7800 / W7900 | gfx1100 | K, M, L (WMMA) | RX 7900 XTX: [external video](https://www.youtube.com/watch?v=_GLjJ2Dn5pU) |
| RX 7600 / 7600 XT, RX 7650 GRE, RDNA3 integrated GPUs | gfx1102, gfx1103 | K, M, L (WMMA) | untested |
| RX 9070 XT / 9070 / 9070 GRE, Radeon AI PRO R9700 | gfx1201 | K, M, L (WMMA, native FP8) | RX 9070 XT: [external video](https://www.youtube.com/watch?v=lZ0BLYqAtCs) |
| RX 9060 XT / 9060 | gfx1200 | K, M, L (WMMA, native FP8) | compile only |
| RX 6000 series, Steam Deck, RDNA2 integrated GPUs | gfx1030–gfx1036 | K (software matrix products) | validated on RDNA3 only |
| RDNA1 and older | | | unsupported |

RDNA2 has no WMMA instructions: its K kernels do the same matrix products with `v_dot2` and DPP broadcasts, bit-identical to the WMMA build. Running that RDNA2 path on an RX 7700 XT for testing, K took 3.6 ms per 1080p frame, against 46.8 ms fully translated (what RDNA2 had before); RDNA3 itself keeps WMMA (1.48 ms). E and K are the models to use there; M and L run fully translated. Kernels missing for a GPU fall back to ZLUDA and can be much slower. Details: [docs/native-kernels.md](docs/native-kernels.md).

## How it works

```
game (D3D12) ──► OptiScaler ──► d4r nvngx.dll (NGX D3D12 API) ──► NVIDIA's NGX core + nvngx_dlss.dll (CUDA path)
                                                                          ▼
           native RDNA kernels ◄── ZLUDA (PTX → AMDGPU) ◄── nvcuda.dll (Wine CUDA bridge)
```

The shim keeps the game's inputs and output in VRAM through vkd3d-proton's Vulkan interop; ZLUDA compiles NVIDIA's PTX and serves hand-written kernels in place of the heaviest ones. An optional [native inference engine](engine/README.md) (`[Engine]`) runs K, L and M with d4r's own Vulkan shaders. Details: [docs/architecture.md](docs/architecture.md).

## Install the release

1. Extract `d4r-<version>.zip` into the folder that holds the game's main `.exe` (OptiScaler 10.0.0-pre1 as `dxgi.dll`, its `OptiScaler/` dependencies, the d4r-patched vkd3d-proton, AMD's FSR files and the `d4r` folder).
2. Run `bash install.sh` there. It downloads NVIDIA's DLSS and NGX files from NVIDIA, checks their SHA-256 hashes and puts them in `d4r/`. Needs Bash, curl, sha256sum and 7-Zip; no sudo, ROCm not required.
3. In Steam, select a Proton version for the game (most work; tested: GE-Proton11-3) and set the launch options `PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %command%`.

When updating, replace `d3d12.dll`, `d3d12core.dll` and `d4r/nvngx.dll` together. Already using OptiScaler? The `-asi` zip adds d4r as a plugin ([ASI guide](packaging/ASI_README.txt)). The full guide ships in the zip as [D4R_README.txt](packaging/D4R_README.txt).

Northlight games (Control, Alan Wake 2) keep their original motion vectors by default. Experimental display-resolution motion-vector reconstruction requires `[Interop] NativeMotionVectors = true` in `d4r.ini` and a matching patched driver. It can cause motion artifacts; restart the game after changing it.

## Building from source

Linux with an RDNA2, RDNA3 or RDNA4 GPU, ROCm 7.2 with HIP and its clang, GE-Proton 11, a Rust toolchain, meson/ninja, `winegcc`, `x86_64-w64-mingw32-g++`, `clang-cl` and Python 3. The full sequence is in [docs/building.md](docs/building.md); `scripts/package_release.sh` builds the release zip.

| Path | Contents |
|---|---|
| `tools/` | the NGX shim, the Wine CUDA bridge, a D3D12 DLSS harness, probes |
| `kernels/` | native kernels (`k/` DLSS 4, `m/` DLSS 4.5, `tex/` texture tails, `common/` matrix layouts) and their tools |
| `engine/` | the native K/L/M inference engine |
| `patches/` | ZLUDA, vkd3d-proton and OptiScaler patches |
| `scripts/`, `config/`, `packaging/`, `docs/` | build/launch scripts, the developer `d4r.ini`, release files, documentation |

## Star History

<a href="https://www.star-history.com/?type=date&repos=countervolts%2Fd4r">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=countervolts/d4r&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=countervolts/d4r&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=countervolts/d4r&type=date&legend=top-left" />
 </picture>
</a>

## License

GNU General Public License, version 3 only (GPL-3.0-only; see [LICENSE](LICENSE)). The patches in `patches/` are offered under the licenses of the projects they modify: ZLUDA (Apache-2.0 or MIT) and vkd3d-proton (LGPL-2.1). See [NOTICE](NOTICE). The downloaded NVIDIA files and the kernels built from NVIDIA's code are not covered by this license.
