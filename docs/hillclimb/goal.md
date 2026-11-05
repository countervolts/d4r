Make d4r's native Windows upscaling on this RX 9070 XT (gfx1201) match NVIDIA's DLSS speed in D3D12 and Vulkan
games, for SR K, M, L and RR F.

Targets: GPU time per evaluate, end-to-end through d4r_nvngx.dll, at 720p→1440p / 1080p→4K:

| Preset | 720p→1440p | 1080p→4K |
|---|---:|---:|
| M | ≤ 1.081 ms | ≤ 2.553 ms |
| L | ≤ 1.413 ms | ≤ 3.161 ms |
| RR F | ≤ 2.100 ms | ≤ 4.658 ms |
| K | ≤ 0.649 ms | ≤ 1.532 ms |

Don't stop until every target is met in both d3d12_dlss_harness and vulkan_dlss_harness and confirmed in game, or
until profiling shows nothing large left to try.

Build your own benchmark. Interleave control and candidate, and measure the noise before you trust a gain. There is
no reference DLSS output; NVIDIA's own output differs between its architectures. So a change must keep output
byte-identical to the previous build on this GPU, unless it deliberately changes the numerics in the fast set only.
Such a change must still look right (finite, no edge bars, follows motion, settles when still) and be checked in
Cyberpunk 2077.

Tools: use whatever helps you see where the time goes, and install anything else you need. The ones already here:
- **Radeon Developer Tool Suite**, unpacked at `C:\Users\Wouter\Desktop\RadeonDeveloperToolSuite-2026-02-02-1757\`.
  - **RGP:** per-dispatch timing, wavefront occupancy, cache counters and instruction timing. Capture with
    `RadeonDeveloperPanel.exe`, or headless with `RadeonDeveloperServiceCLI.exe`.
  - **RGA (`rga.exe`):** offline ISA, register and LDS usage for DXIL, SPIR-V and HIP code objects.
  - **RMV:** VRAM. **RRA:** ray tracing.
- **The repo's RGP helpers:** `tools/rgp_clip.c`, `tools/rgp_instruction_mix.py`, `engine/d3d12/rgp_counters.py`,
  `tools/radv_isa_mix.py`. Earlier captures are under `nightly-work/rgp` if present.
- **RenderDoc** (`C:\Program Files\RenderDoc\`, `renderdoccmd.exe`): inspect a frame's resources and passes in the
  harness or a game.
- **Windows Performance Recorder/xperf** (`wpr`, `xperf`): CPU-side stalls, queue waits and fence waits; GPUView-style
  analysis of cross-queue handoffs.
- **ROCm 7.14** (`C:\ROCm\7.14\lib\llvm\bin\llvm-objdump.exe`, `llvm-readelf.exe`, `hipInfo.exe`): disassemble the
  hsaco code objects and check registers and scratch.
- **d4r's own profiling:** `D4R_ENGINE_PROFILE`, `D4R_REPLAY_PHASES`, `D4R_REPLAY_HIP_PHASES`, `D4R_VKHIP_STAMPS`,
  `windows-game.ps1 -ProfileStages`.

You may install more developer tools, for example PIX (`winget install Microsoft.PIX`), the Windows SDK's
GPUView, Python packages, or newer builds of the Radeon tools. Use official sources and user-level installs, and
note each one in journal.md. Profiling tools only: no drivers, no driver updates, no system-wide settings changes.

Hard rules:
- Only inject into Cyberpunk 2077, Ghost of Tsushima and Portal RTX. Never touch a game with anti-cheat. Always
  restore game folders with windows-game.ps1.
- Never commit NVIDIA DLLs, weights, captures or derived kernels. RGP and RenderDoc captures of NVIDIA-derived
  kernels stay in ignored folders too.
- No driver, clock, power, TDR or HAGS changes.
- Don't change what gfx11/gfx10 or Linux builds compile. The kernels/k flag lines are shared with Linux.
- Commit each win on branch hillclimb/perf. Never push. Keep a journal.md of what you tried, including what didn't
  work.

Traps:
- The harness pauses 200 ms between frames by default, and the GPU then clocks down about 15%. Time with frames
  0–4 ms apart.
- Profilers change timing. RGP's instruction tracing and serialized kernel profiling distort timings, so take
  timings from runs without the profiler attached, and use the profiler only to find where the time goes.
- Check in the log that the route you meant to time actually ran. vkhip in the Vulkan harness needs
  D4R_HARNESS_ENGINE_DEVICE=1.
- Moving work outside the timed bracket, or caching output across frames, is not a speedup.
- OptiScaler's "DLSS frame time" is the whole frame. Its upscaler GPU time is what d4r costs.

Background (known numbers, leads, dead ends): docs/hillclimb/performance.md, docs/performance.md,
docs/experimental-vkhip.md, docs/windows-k-d3dhip.md.
