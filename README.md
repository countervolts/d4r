# d4r (dlss 4 radeon)

want to run xess (xmx) on radeon? check out [x4r](https://github.com/countervolts/x4r)!

nightly releases for d4r can be found [here](https://github.com/countervolts/d4r-nightly)

d4r runs NVIDIA's official DLSS Super Resolution (`nvngx_dlss.dll`) and experimental Ray Reconstruction (`nvngx_dlssd.dll`) in DirectX 12 games on AMD Radeon GPUs, on Linux (Proton) and natively on Windows. The game asks for DLSS as usual; OptiScaler hands the call to d4r, which runs the network on the AMD GPU: on d4r's own native engine where it covers the model, otherwise through [ZLUDA](https://github.com/vosen/ZLUDA) (CUDA on ROCm/HIP) with the heaviest kernels replaced by hand-written RDNA2, RDNA3 and RDNA4 code.

**Supported DLSS models:** DLSS 3 CNN (E), DLSS 4 transformer (K, default), and DLSS 4.5 transformer (M, plus L for Ultra Performance). Ray Reconstruction presets D, E and F are experimental. DLSS 5 is PURPOSELY not supported.

**Proof of concept:** d4r shows that dlss can be ran on radeon gpus. See [supported games](SUPPORTED_GAMES.md) for the tested games and models.

> **Not affiliated with NVIDIA or AMD.**

To contact me my discord is `._ayo`. or join the [d4r discord server](https://discord.gg/mgTWpSq8f5)

## Install

The release zips do not contain NVIDIA's files. Their installers download the pinned official DLSS, Ray Reconstruction and NGX files from NVIDIA, check their SHA-256 hashes and keep a verified copy in a per-user cache.

### Linux (Proton)

1. Extract `d4r-<version>.zip` into the folder that holds the game's main `.exe` (OptiScaler as `dxgi.dll`, the d4r-patched vkd3d-proton, AMD's FSR files and the `d4r` folder).
2. Run `bash install.sh` there. It installs NVIDIA's DLLs into `d4r/` and builds the native engine's models from them. Needs Bash, curl, sha256sum, 7-Zip and python3 with numpy; no sudo, ROCm not required.
3. In Steam, select a Proton version for the game (tested: GE-Proton11-3). No launch options are needed.

`d4r_boot.dll`, which d4r's `dxgi.dll` loads first, shows the game an NVIDIA GPU (what `PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100` did before; those options still work and can be removed). When updating, replace `dxgi.dll`, `d4r_boot.dll`, `d3d12.dll`, `d3d12core.dll` and `d4r/nvngx.dll` together.

The full guide ships in the zip as [D4R_README.txt](packaging/D4R_README.txt). Already using OptiScaler? The `-asi` zip adds d4r as a plugin ([ASI guide](packaging/ASI_README.txt)).

DLSS Frame Generation (NVIDIA's own, 2x-6x, and FG Lite) runs on RDNA3 and RDNA4: `install.sh` sets it up (it downloads DLSS-G 310.9.16 from NVIDIA's Streamline SDK and translates it on your machine), also for games without DLSS-G of their own through OptiScaler; `d4r.ini [FrameGeneration] Model = auto` uses FG Lite on RDNA3 and the full model on RDNA4 ([docs/frame-generation.md](docs/frame-generation.md)).

Northlight games (Control, Alan Wake 2) keep their original motion vectors by default. Experimental display-resolution motion-vector reconstruction requires `[Interop] NativeMotionVectors = true` in `d4r.ini` and a matching patched driver. It can cause motion artifacts; restart the game after changing it.

### Windows

1. Extract the Windows release zip into the folder that holds the game's main `.exe`. If the folder already has a `dxgi.dll` or `OptiScaler.ini` from another OptiScaler install, move those out of the way first.
2. Double-click `install.cmd` (`install.bat` in the HIP/ZLUDA build). It downloads NVIDIA's files and prepares the native models; Python and the other tools it needs are included or set up privately.
3. Launch the game normally and select DLSS.

Needs Windows 10 or 11 with a current AMD Adrenalin driver; ROCm and the HIP SDK are not needed. Full guides: [native D3D12 release](packaging/WINDOWS_README.txt), [HIP/ZLUDA release](packaging/windows/WINDOWS_README.txt). Avoid games with anti-cheat: OptiScaler injects a DLL.

## Switching models and FSR in game

Open OptiScaler's menu (Insert):

- **DLSS model:** under DLSS Settings, tick *Render Presets Override*, pick the preset (E, K, L or M) and press *Apply Changes*. Unticking it returns to `[DLSS] Model` in `d4r/d4r.ini`.
- **FSR:** set *Upscaler (DX12)* to *FSR 3.X/4*.

In some games some models can show visible artifacting with native upscaling. For fidelity over speed, set `[Kernels] PreferAccuracy = true` in `d4r/d4r.ini` ([native kernel numerics](docs/native-kernels.md#numerics)).

## Results

RDNA3, DLSS 4 (K), 2560x1440 output.

**Windows, native D3D12**

| Mode | Input | FPS | K GPU mean | K GPU median |
|---|---|---:|---:|---:|
| Quality | 1712x960 | 98.2 | 2.221 ms | 2.111 ms |
| Balanced | 1512x848 | 105.4 | 2.224 ms | 2.223 ms |
| Performance | 1280x720 | 114.0 | 2.201 ms | 2.208 ms |
| Ultra Performance | 856x480 | 125.6 | 2.120 ms | 2.124 ms |

Presets M and L, complete-frame replay GPU median (in game is a little lower):

| Preset | Mode | Input | GPU time |
|---|---|---|---:|
| M | Performance | 1280x720 | 7.00 ms |
| L | Ultra Performance | 856x480 | 5.44 ms |

**Linux, RX 7700 XT, Ghost of Tsushima**

| Mode | FPS | K GPU time |
|---|---:|---:|
| Quality | 90.9 | 1.99 ms |
| Balanced | 93.8 | 1.96 ms |
| Ultra Performance | 101.7 | 1.91 ms |

The Windows and Linux runs used different benchmark runs and probably different settings, so compare within a table rather than across them. More measurements: [performance](docs/performance.md), [videos](docs/videos.md).

## GPU support

| GPU | Chip | Native kernels |
|---|---|---|
| RX 7700 XT, RX 7800 XT, RX 7700, Radeon PRO W7700 | gfx1101 | K, M, L (WMMA) |
| RX 7900 GRE / XT / XTX, Radeon PRO W7800 / W7900 | gfx1100 | K, M, L (WMMA) |
| RX 7600 / 7600 XT, RX 7650 GRE, RDNA3 integrated GPUs | gfx1102, gfx1103 | K, M, L (WMMA) |
| RDNA 3.5 integrated GPUs | gfx1150–gfx1154 | K, M (Windows HIP/ZLUDA build) |
| RX 9070 XT / 9070 / 9070 GRE, Radeon AI PRO R9700 | gfx1201 | K, M, L (WMMA, native FP8) |
| RX 9060 XT / 9060 | gfx1200 | K, M, L (WMMA, native FP8) |
| RX 6000 series, Steam Deck, RDNA2 integrated GPUs | gfx1030–gfx1036 | K (software matrix products) |
| RDNA1 and older | | unsupported |

Kernels missing for a GPU fall back to ZLUDA and can be much slower. Details: [native kernels](docs/native-kernels.md), [Windows GPU coverage](docs/windows-gpu-support.md).

## How it works

On Linux the shim keeps the game's inputs and output in VRAM through vkd3d-proton's Vulkan interop, and the [native inference engine](engine/README.md) records K, M, L and Ray Reconstruction into the game's frame. On Windows the [D3D12 engine](engine/d3d12/README.md) records them directly into the game's D3D12 command list. Anything the engines do not cover runs NVIDIA's DLSS through ZLUDA with native kernels in place of the heaviest ones. Details: [architecture](docs/architecture.md).

## Building from source

- **Linux:** an RDNA2, RDNA3 or RDNA4 GPU, ROCm 7.2 with HIP and its clang, GE-Proton 11, a Rust toolchain, meson/ninja, `winegcc`, `x86_64-w64-mingw32-g++`, `clang-cl` and Python 3. See [building](docs/building.md); `scripts/package_release.sh` builds the release zip.
- **Windows:** `scripts/package_release_windows.sh` builds the native D3D12 release. For the HIP/ZLUDA build, run `build-kernels.bat` to compile the K/M kernels for every RDNA 3, 3.5 and 4 target (first run `scripts\windows\setup-windows-tools.ps1 -RuntimeProfile therock`); see [Windows GPU coverage and release builds](docs/windows-gpu-support.md).

| Path | Contents |
|---|---|
| `tools/` | the NGX shims (Linux and Windows), the Wine CUDA bridge, DLSS harnesses, probes |
| `kernels/` | native kernels (`k/` DLSS 4, `m/` DLSS 4.5, `tex/` texture tails, `common/` matrix layouts) and their tools |
| `engine/` | the native K/L/M and Ray Reconstruction engines (Vulkan; `d3d12/` for Windows) |
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
