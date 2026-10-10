# Architecture

d4r keeps NVIDIA's DLSS library unmodified and gives it what it expects: an NGX D3D12 or Vulkan entry point in the game process, a CUDA driver underneath, and CUDA kernels that run. Everything between those points is translation.

The [experimental native Vulkan frontend](vulkan.md) reuses the CUDA backend with
host staging and command-buffer event synchronization; it does not use vkd3d-proton.

## Components

| Piece | Where | Role |
|---|---|---|
| OptiScaler (in GE-Proton) | external | Hooks the game's upscaler call and forwards DLSS requests to an NGX DLL (`NvngxPath`) |
| NGX shim `d4r_nvngx.dll` | `tools/d4r_nvngx_shim.cpp` | Implements the NGX D3D12 API. Moves inputs and output between the game's D3D12 resources and CUDA, and drives the official NGX core through its CUDA API |
| NGX core + `nvngx_dlss.dll` | NVIDIA, supplied by you | The real DLSS: parameter handling, network selection, the CUDA kernels |
| CUDA bridge `nvcuda.dll` | `tools/wine_nvcuda_bridge.c` | A Wine builtin whose Windows exports forward to ZLUDA's Linux `libcuda.so`, plus d4r helpers (Vulkan memory import, asynchronous 2D copies, GPU-side waits, output redirect, linear textures) |
| ZLUDA | `patches/zluda` | CUDA driver API on HIP; compiles NVIDIA's PTX to AMDGPU code; serves native kernels in place of PTX kernels |
| Native kernels | `kernels/` | RDNA3 implementations of the DLSS network layers and parts of texture kernels |
| vkd3d-proton patch | `patches/vkd3d-proton` | Splits a D3D12 command list around DLSS and retains external resources through command allocator completion |
| NVAPI identity | `tools/d4r_nvapi_identity.c` | Reports an Ada (sm_89) GPU so NGX selects the network weights matching the PTX ZLUDA compiles |

## One frame

With VRAM interop and split frames (the defaults when the patched vkd3d-proton is present):

1. **The game records DLSS.** Its `EvaluateFeature` call reaches the shim with the game's command list. Into that list the shim records:
   - Vulkan commands (vkd3d-proton interop) that copy colour, depth, motion vectors and optional exposure into buffers exported to HIP. Linear R11G11B10, RGB10A2, and RGBA8/BGRA8 colour inputs are blitted to RGBA16F; multichannel float exposure is blitted to R32F using its first component when the device supports the conversion.
   - A frame marker.
   - A split. vkd3d-proton submits everything the game records after the DLSS call separately, gated on a timeline semaphore reaching this frame's number.
   - At the start of that second part, a copy of the DLSS result into the game's output texture. Linear R11G11B10, RGB10A2, and RGBA8/BGRA8 outputs are converted from RGBA16F on the GPU. sRGB formats continue through host staging to preserve their existing byte interpretation.
2. **The CUDA worker queues DLSS.** The prep thread checks the GPU-written marker for that input slot with short CPU sleeps. A later frame's marker cannot make a discarded frame's stale inputs appear ready. It drops recorded-but-unsubmitted frames when the queue skips them, they time out or their feature retires, before any DLSS work is queued. The CUDA worker then queues:
   - the input copies into CUDA arrays, or none when NGX samples the buffers in place (`LinearInputs`);
   - NGX's evaluation (NGX's own CPU synchronisations are elided with `ElideNgxSync`);
   - the result, written straight into the output buffer by the native output kernel (`DirectOutput`, preset K) or copied there.
3. **Release.** When the GPU finishes, the worker signals the timeline semaphore and the rest of the game's frame runs, now with this frame's result.

The watchdog reports delayed output but never signals an unfinished producer as ready. Cancellation and completion can occur out of order across the prep and CUDA threads, so the output timeline advances only through a completed prefix of admitted frames. Cancelling a later frame cannot release a game copy while an earlier producer is still writing. Retiring features remain registered until their CPU/CUDA jobs drain. The lifetime-aware vkd3d extension retains their staging buffers, imported buffers, conversion images and split semaphore through command-allocator reset or destruction after GPU completion. Discarded recordings are covered too; cleanup no longer relies on a 50 ms delay.

The shim requires the matching lifetime-aware d3d12.dll and d3d12core.dll, even for host staging. It rejects older runtimes rather than recording references whose lifetime it cannot protect. FrameAge > 0 shows the newest finished result; without VRAM interop, inputs and output are staged through host memory.

## Why each piece exists

- **CUDA NGX instead of D3D12 NGX.** NVIDIA's D3D12 path needs NVIDIA's driver. The CUDA path only needs a CUDA driver API, which ZLUDA provides.
- **The Ada identity.** NGX picks network weights by GPU architecture. The weights have to match the PTX that ZLUDA compiles, which is sm_89 for DLSS 310.
- **Native kernels.** Translating NVIDIA's warp-level matrix code is slow (register pressure, lane shuffles, per-instruction mode switches). Earlier translated K layers also produced non-finite outputs that the final kernel masked. Native layers avoid that path; the compiler now preserves FP16 denormal requirements too, which fixes the translated failure in recorded captures. Numerical and image-quality parity with RTX hardware remains unverified.
- **Sleeping readiness checks.** Waiting for input submission before queueing DLSS avoids an uncancellable GPU wait. Explicit output event queries with short sleeps avoid HIP event/context waits that can consume a CPU core. Submillisecond waits use NtDelayExecution because MinGW std::sleep_for can truncate them to zero. Linear inputs and direct output remain available.

## Configuration

`config/d4r.ini.default` documents every setting. `scripts/d4r_config.py` turns the file into environment variables for `scripts/run_game_d4r_optiscaler_proton.sh`, and an environment variable that is already set wins. The most useful switches:

| d4r.ini | Environment | Effect |
|---|---|---|
| `[DLSS] Model` | `D4R_DLSS_PRESET` | DLSS network: `E` (CNN), `K` (DLSS 4), `M` (DLSS 4.5) |
| `[Latency] FrameAge` | `D4R_SHIM_SPLIT_FRAME`, `D4R_SHIM_MAX_IN_FLIGHT` | 0 = same-frame results; higher values pipeline frames |
| `[Kernels] NativeKernels` | `D4R_ZLUDA_NATIVE_DIR` | native kernel directory (`fast` / `exact` / `off`) |
| `[Kernels] PreferAccuracy` | `D4R_PREFER_ACCURACY` | default off; accuracy variants for every native family and conservative translation/synchronization settings |
| `[Interop] VramInterop` | `D4R_SHIM_VRAM_INTEROP` | keep inputs and output in VRAM |
| `[Interop] LinearInputs` | `D4R_SHIM_LINEAR_INPUTS` | NGX samples the input buffers in place |
| `[Interop] DirectOutput` | `D4R_SHIM_OUTPUT_DIRECT` | the output kernel writes the game-side buffer directly (preset K) |
| `[Interop] ElideNgxSync` | `D4R_ELIDE_NGX_SYNC` | skip NGX's internal CPU waits |
| `[Paths] JitCacheDir` | `XDG_CACHE_HOME` | where ZLUDA caches compiled kernels |

## Portable installs (the release zip)

The release zip (`scripts/package_release.sh`) is unpacked into the folder that holds a game's main `.exe`, like an OptiScaler release. Nothing in the Proton prefix or the system is changed.

| File | Role |
|---|---|
| `dxgi.dll`, `OptiScaler.ini` | OptiScaler 0.9.4. Its INI sets DLSS as the DX12 upscaler, `NvngxPath=d4r\nvngx.dll` to select the shim explicitly, and `OptiDllPath=d4r` for the other libraries. |
| `d3d12.dll`, `d3d12core.dll` | the d4r-patched vkd3d-proton. Proton loads DLLs from the game folder before `system32`. |
| `d4r\nvngx.dll`, `d4r\nvcuda.dll` | the shim and the CUDA bridge |
| `d4r\zluda\libcuda.so` | ZLUDA |
| `d4r\kernels\<gfx target>\` | native kernels and their manifest `d4r-kernels.txt` |
| `d4r\d4r.ini` | this game's settings |
| `d4r\nvngx_dlss.dll`, `d4r\ngx\_nvngx.dll` | NVIDIA's files, downloaded by `install.sh` (`D4R_BUNDLE_NVIDIA=installer`, the default), bundled (`=1`) or added by the user (`=0`) |

**NGX routing.** OptiScaler tries `_nvngx.dll` before `nvngx.dll`, with system fallback for each name. `OptiDllPath=d4r` alone can therefore select Proton's system `_nvngx.dll` before reaching the shim. A file-valued `NvngxPath` takes priority on the first probe, regardless of the probe's name. It must select `d4r\nvngx.dll`, never `d4r\ngx\_nvngx.dll`: the latter is NVIDIA's core, which the shim loads internally. See OptiScaler 0.9.4's [NGX loader](https://github.com/optiscaler/OptiScaler/blob/v0.9.4/OptiScaler/proxies/NVNGX_Proxy.h) and [override handling](https://github.com/optiscaler/OptiScaler/blob/v0.9.4/OptiScaler/Util.cpp).

OptiScaler 0.9.4 checks a relative `NvngxPath` against the process working directory; it does not anchor that override to the executable like it does `OptiDllPath`. The portable INI assumes the game's working directory is the executable folder. Launchers that use another working directory need an absolute Windows filename for the shim. The Proton developer game launcher and OptiScaler harness already generate that form for `bin/d4r_nvngx.dll`.

`sh d4r/d4r-check.sh` checks file presence and the explicit NGX route separately; it does not prove runtime initialization. OptiScaler's startup `nvngx.dll not found!` warning checks the executable folder and already loaded modules, not the override. It can appear before the shim loads from `d4r/`; verify the later `NVNGXProxy::InitNVNGX Loaded from ...\d4r\nvngx.dll` line and `d4r/d4r_nvngx.log` instead. Keep the shim in `d4r/`.

**Settings.** When a `d4r.ini` sits next to the shim, the shim turns it into the same environment variables the developer launcher sets, with the release's defaults. A variable that is already set, for example in the launch options, wins. Settings for the Linux side (ZLUDA's and the bridge's) are handed to the bridge through `d4rSetEnv` right after it is loaded, before its first CUDA call.

**Loading ROCm.** Steam runs GE-Proton 11 inside its container runtime (SteamLinuxRuntime_4), whose library search path has no ROCm. The bridge therefore:
- loads HIP, HSA and comgr by path, from `RocmDir` (default: the release's bundled `d4r/rocm`, else `/opt/rocm`);
- looks up any dependency they lack in the host's library directories, which the container mounts under `/run/host`;
- sets `XDG_CACHE_HOME` only while ZLUDA initialises, so ZLUDA's kernel cache goes to `~/.cache/d4r` without affecting the rest of the game.

**Native kernel checks.** For each kernel, `d4r-kernels.txt` lists the hash of the DLSS PTX module it was written for. ZLUDA is pointed at a per-process directory, and the bridge adds a kernel to it only when NGX loads a module with a matching hash. A DLSS version whose kernel changed therefore gets ZLUDA's compile of that kernel: slower, but correct. The bridge picks the `gfx` folder from the GPU's KFD topology entry. `DirectOutput` takes effect only while the bridge reports that NGX uses a native `hiluma_engine_output_*` kernel. That kernel is one of the texture kernels built from NVIDIA's PTX, and it is worth about 3 fps for DLSS 4 at Quality.

**NGX identity and watermark.** A game that initialises NGX with a project ID (Unreal's DLSS plugin does) is passed to the NGX core's `NVSDK_NGX_CUDA_Init_ProjectID` with that ID. In this setup the DLSS library still stamps "DLSS SDK - DO NOT DISTRIBUTE" over its output. The shim sets the SDK's `Disable.Watermark` evaluation parameter unless `[DLSS] ShowWatermark = true` (`D4R_SHIM_WATERMARK=1`).

**NVAPI.** The launch option `PROTON_FORCE_NVAPI=1` makes Proton enable dxvk-nvapi on the AMD driver and set `WINE_HIDE_AMD_GPU`, so Windows sees an NVIDIA PCI ID. OptiScaler's NVIDIA check accepts that, so the NVAPI identity bridge is only needed by the developer launcher. `DXVK_NVAPI_GPU_ARCH=AD100` is still required, because NGX picks its weights by GPU architecture.

## Logs

The launcher writes each run's logs to `~/.cache/d4r-dlss-captures/game-<exe>-<time>/`. `d4r_nvngx.log` holds the shim's log. With `[Debug] Profile = true` it also records one `D4R_PROFILE` line per frame, with the stage timings.

Output completion uses event queries with 200 us NtDelayExecution sleeps by default. `D4R_SHIM_BLOCKING_SYNC = 0`
under `[Env]` restores context synchronization for an A/B check. The profile records
`output_sync_wall`, `output_sync_cpu`, and `output_sync_method=event_query_sleep` for the new path.
An unavailable or failed event falls back to context synchronization. The old `GpuWait` setting
is no longer used: input readiness is checked on the CPU before CUDA work is queued.

The `d2h` field on this path includes the output synchronization wait, even when direct output
skips the array-to-buffer copy.
