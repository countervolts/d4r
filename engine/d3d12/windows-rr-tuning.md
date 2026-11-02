# Native Windows RR D/E/F

RR presets D, E and F run directly on the game's D3D12 command list on gfx12 AMD
hardware. No Vulkan interop, CUDA worker or NGX runtime is needed for this path.
`runtime_rr_d3d12` implements D/E; `runtime_rr2_d3d12.inc` implements F's exposure,
six encoders, ten decoder passes, tiled reconstruction and temporal histories.
`game_rr_d3d12` adapts game resource states. Geometry and camera conventions
follow the corresponding Vulkan runtimes.

Build from a Visual Studio developer shell with CMake and the Vulkan SDK:

```powershell
cmake -S engine -B build/engine-win
cmake --build build/engine-win --config Release --target d4r-rr-d3d12 d4r-rr-zluda
python engine/compile_rr_dxil.py MODEL_RR_D --output build/rr-d --dxc "$env:VULKAN_SDK/Bin/dxc.exe"
python engine/compile_rr_dxil.py MODEL_RR_E --output build/rr-e --dxc "$env:VULKAN_SDK/Bin/dxc.exe"
python engine/compile_rr_dxil.py MODEL_RR_F --output build/rr-f --dxc "$env:VULKAN_SDK/Bin/dxc.exe"
```

D/E input models come from `engine/rr/build_native_model.sh`, using native gfx12
relaxed arithmetic and presets 4 (D) or 5 (E). F uses a `D4RR0002` model from
`engine/compile_rr2.py` (preset 6). A Linux handoff containing
`*.preprocessed.comp` works directly. Otherwise preprocessing needs
`glslangValidator` and the model's generated `native-work` headers and sources.
DXC uses SM 6.6, native 16-bit types and wave32 for network stages. The builder
adds DXIL, a `rr-d3d12.txt` sidecar and transposed weight tiles; Vulkan models
remain untouched when `--output` names another directory. Generated model files
contain private data; keep them outside source control.

Build the frontend using `scripts/build_d4r_nvngx_shim.sh`. Set `D4R_ENGINE=1`,
`D4R_RR_ENABLE=1`, `D4R_RR_BACKEND=dlss`, and select D/E/F through the game's NGX
preset hints or `D4R_RR_PRESET=4` / `5` / `6`. Model lookup uses `engine/rr-d`,
`engine/rr-e` and `engine/rr-f` beside the shim, overridden by
`D4R_ENGINE_MODEL_RR_D_DIR`, `D4R_ENGINE_MODEL_RR_E_DIR` and
`D4R_ENGINE_MODEL_RR_F_DIR`. Explicit D/E hints require their matching packages.
When F is installed, other hints (including Default and vendor-flagged defaults)
select F, matching DLSSD 310.9. RR-only installations can initialize without SR
packages or NVIDIA runtime DLLs.

The public adapter supports zero origins, exact render-size input textures,
RGBA16F UAV output, HDR and unjittered render-resolution motion. Pass both
camera matrices and the four required material guides. Hit distance,
disocclusion, subsurface and particles are optional. Resize recreates the engine
and resets its history. Separate output alpha is explicitly refused. Shader
model 6.6/native-half support is checked when loading; gfx11 is unsupported.
Every recording must be followed by `submitted(fence,value)` or `discarded()`;
submission tracking in the shim retains resources until their fence retires.

F runs the split input gather when present and the four-phase native kernels
for small window grids, like Vulkan. Packed FP8 arena matrices stay packed;
the gather's reversed workgroup order and the phases' expanded grids use their
actual dispatch dimensions. Phase records occupy parameter byte 104; the
rolled-loop zero word follows at byte 108. Per-frame parameter blocks are copied
in one upload, whose storage is retained until submission retires or recording
is discarded. Adjacent reconstruction tile groups overlap their post and down
passes on alternating arena slots.

F follows packed-roughness mode even when a separate roughness texture is bound;
the direct engine also accepts an unbound roughness texture in that mode. The
frontend preserves creation-time roughness/depth modes when evaluation uses a
fresh parameter object, and exposes the native RR optimal-settings callback.
Network timings retain the D/E replay format: decoder intervals combine both
kernels; `hkpn` means F's complete post/down/pooled reconstruction interval.

## Preset F: throughput-first results

F batches FP8 encodes and binary16 RNE conversions, removes redundant codec
round trips, and combines adjacent halfword accesses without increasing their
alignment requirement. Pixel gathers retain software FP8 encoding: AGS
conversions in divergent flow are unsafe on this driver. The hybrid decoder
uses native float `tanh` instead of the FP64 OCML polynomial.
[D3D texture loads return zero outside the texture](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-load),
so F's zero-border fetches need no extra dimension queries or bounds branches.

RX 9070 XT, driver 32.0.32015.7029, three interleaved trials; 120 repeats per
input, retaining the final 60. Times are GPU medians, including F's exposure,
network and reconstruction, not CPU submission or input/readback transfers.
The first functional F shaders and final shaders use the same final runtime:

| F case | First functional D3D12 | Optimized D3D12 | Linux RADV handoff |
|---|---:|---:|---:|
| 720p -> 1440p | 5.779 ms | 4.377 ms | 3.131 ms |
| 1080p -> 4K | 12.267 ms | 9.152 ms | 6.660 ms |

Linux latency parity is **not reached**. These are measured Windows throughput
improvements, not a parity claim. ADL's active-load median across the final 4K
comparison was 2713 MHz (memory 2505 MHz); the Linux handoff reported about
3150 MHz. Clocks and desktop workloads were not controlled across OSes.

Occupancy pass (`inf_select`, `residual_pack`, `branch_pin`; method under
Diagnostics): 4K went from 9.22 to 8.80 ms (three interleaved trials against
the build above, `build/rr-f-work/bench-opt4-4k`). Per-stage changes in us:
dec0 2201 -> 2028, dec1 1094 -> 998, dec2 651 -> 610, enc1 607 -> 567; the
other stages are unchanged. At 1440p the stage changes (-10 to -33 us each)
are within that run's trial-to-trial noise, and the totals were 4.7-4.9 ms for
both builds. The 4K replay output is byte-identical to the build above.

Actual-GPU sweeps rejected head/MLP-branch unrolling, four-wave hot kernels,
wave32 reconstruction, native-half rounding and packed-half activation
variants. FP8 range clamps remain enabled: the faster unclamped experiment
can turn out-of-range values into NaNs.

Secondary quality measurements use the same CUDA reference and metric below;
F's build selection prioritizes throughput rather than a PSNR-loss gate:

| F case | Linux PSNR | D3D12 PSNR |
|---|---:|---:|
| temporal, 32 frames | 42.4506 dB | 41.8127 dB |
| 1440p, 4 frames | 48.6634 dB | 48.2791 dB |
| 4K, 4 frames | 48.6705 dB | 48.4182 dB |

Verification: all 43 DXIL shaders built from generated model sources without
handoff preprocessed files; D3D12-debug temporal/high-resolution replays;
127x93 -> 255x187 with resets, varying pre-exposure, packed roughness, hardware
and inverted depth, hit distance, disocclusion, subsurface and particles;
unbound roughness in packed mode; discard/re-record with all 32 outputs
unchanged. The rebuilt NGX frontend passed 12 moving-scene/camera-cut frames
with separate evaluation parameters and missing CUDA/NGX DLLs. D/E debug
replays and the shared attention-bias bit-layout regressions also passed.
These remain synthetic replay/harness checks, not real-game validation.

Private evidence: `build/rr-f-work/bench-final-{4k,1440p}/result.json`,
`quality-final-*.json`, `frontend-final-model.log`, `final-clocks.json` and the
RGP captures under `build/rr-f-work/`. The generated runtime model is
`build/rr-f-d3d12-final`; none of the private model data belongs in a release.

## Reference and quality

The independent `d4r-rr-zluda` executable runs original NVIDIA CUDA kernels
synchronously through Windows ZLUDA. It uploads each frame's raw inputs,
evaluates, synchronizes and downloads that exact output. It has no presentation
ring or native-kernel replacements. Supply your own licensed core/feature DLLs.
Build the CUDA bridge and NVAPI shim with
`scripts/build_windows_nvcuda_bridge.sh` and `scripts/build_windows_nvapi.sh`
(MSYS2 MinGW); ZLUDA must support gfx12 FP8 WMMA.

```powershell
python scripts/rr_reference_windows.py --case CASE --output build/reference-d `
  --preset 4 --core PATH/_nvngx.dll --dll PATH/nvngx_dlssd.dll `
  --sr-dll PATH/nvngx_dlss.dll --zluda PATH/ZLUDA `
  --runner build/engine-win/Release/d4r-rr-zluda.exe
build/engine-win/Release/d4r-rr-d3d12.exe build/rr-d CASE build/dx-d
python engine/d3d12/compare_rr.py --case CASE --reference build/reference-d/reference `
  --linux LINUX_OUTPUTS --d3d12 build/dx-d --output build/quality-d.json --max-extra-loss 0.25
```

Repeat with preset 5/E or preset 6/F. The launcher creates a private app folder,
checks frame sizes and records DLL hashes. ZLUDA uses its normal Windows cache
under `%LOCALAPPDATA%/zluda`; a cold reference can take minutes to compile.
`engine/rr/shim_case.py` converts Linux raw cases to harness replays and imports
compatible captured game planes without guessing camera matrices or guides.

Reference: DLSSD 310.9.1, DLSS 310.7.0. This measures each backend against the
same NVIDIA reference, rather than requiring the backends to match each other.
The metric pools MSE over all frames before converting to dB, uses fixed peak 1
and unclipped linear RGB, and excludes alpha. Nonfinite RGB and missing or wrong
size frames fail. These are synthetic replay results, not real-game validation.

RX 9070 XT (driver 32.0.32015.7029), pooled PSNR against ZLUDA; `temporal` is the
32-frame 128x96 -> 384x288 handoff, `1440p` its four 1280x720 -> 2560x1440 frames:

| Preset | Case | Linux | Previous D3D12 build | Current D3D12 build |
|---|---|---:|---:|---:|
| D | temporal | 36.2311 dB | 20.6648 dB | 36.0553 dB |
| D | 1440p | 47.1007 dB | 33.0089 dB | 46.3025 dB |
| E | temporal | 33.4768 dB | 20.8988 dB | 33.7249 dB |
| E | 1440p | 36.2753 dB | 27.5317 dB | 36.3057 dB |

The previous build is the 2026-10-24 "speed" translation, whose PSNR gate had
been deferred; its integer FP8 profile broke the image (below).

## Timing

The replay idles while it loads each input, and the GPU takes about half a
second of load to reach its boost clock (ADL PMLog: 1.5 -> 2.4 -> 2.8 GHz).
Ten repeats per input therefore time ramping clocks: the earlier tables in this
file (13-27 ms at 4K) overstated D3D12 latency about 1.7x, and the Windows
Vulkan engine reads equally slow that way. `bench_rr.py` now defaults to 120
repeats per input and keeps the second half of each.

RX 9070 XT, normal clocks, three interleaved trials, medians (ms):

| Preset | Case | Previous D3D12 build | Current D3D12 build | Linux RADV (handoff) |
|---|---|---:|---:|---:|
| D | 1080p -> 4K | 10.633 | 8.709 | 7.333 |
| D | 720p -> 1440p | 4.969 | 4.109 | 3.373 |
| E | 1080p -> 4K | 10.634 | 8.790 | 7.309 |
| E | 720p -> 1440p | 4.967 | 4.280 | 3.358 |

E at 1440p varied between trials (4.15-4.65 ms). The last build with reference
quality before this work (`translate(speed=False)`) took 14.07 ms for D at 4K.
D at 4K, current build, per stage (us): exposure 124, enc0 2322, enc1 689,
enc2 374, enc3 179, enc4 133, enc5 75, dec4 136, dec3 315, dec2 538, dec1 906,
dec0 2163, hkpn 728.

Remaining difference to Linux:

- Clocks: under this replay the Windows driver holds about 2.8 GHz, the Linux
  handoff ran at a 3.15 GHz median. Scaled by clock alone [inference], D at 4K
  would be about 7.8 ms against Linux's 7.33.
- The Windows Vulkan driver on Linux's own SPIR-V is no faster: 8.62 ms for the
  network (best of 120 repeats) against D3D12's 7.83 ms (median). The rest of
  the gap to RADV is the Windows compiler's code, shared by both APIs.
- The parameter blocks are written with one `WriteBufferImmediate` entry per
  dword (about 750 per frame): 41 us of the 124 us exposure interval at 4K.
  The exposure dispatches themselves take 83 us (Linux 77).

## Implementation

Packed FP8 operands use two registers and AGS matrix operations, with direct raw
loads, appended transposed weight copies for B operands and the attention-bias
tiles (each lane's eight halves contiguous). Short static matrix loops are
unrolled; the head (`hd`) and MLP-branch (`br`) loops stay rolled behind a
runtime zero (`loopzero` manifest entries), which unrolled cost 0.2 ms at 4K in
instruction-cache misses. Two deep decoders exceed D3D12's 32 KiB shared-memory
limit, so their pool uses a separate UAV slice per window with device-memory
synchronization.

Arithmetic (`rr_hlsl.relaxed_math`, `rr_wmma.hlsli`):

- FP8 decoding is the AGS conversion of the code's register (exact, one
  instruction per pair). The bit-arithmetic decoder was about ten instructions
  per code: 1.0 ms at 4K.
- FP8 encoding of single values is branch-free. The early-return form compiled
  to divergent exec masks with a hazard wait per test (0.45 ms, mostly in
  encoder 0's translated gather).
- Activation tiles are quantized with an f32 clamp to +-448 and the plain
  conversion. Values beyond 448 saturate as RADV's conversion does; a NaN
  becomes -448.
- Conversions to binary16 round to nearest even through the AGS matrix
  conversion. AMD's D3D12 compiler converts casts and `f32tof16` toward zero
  (a GPU probe: half of all random values differ from RNE), which cost E 1.7 dB;
  the bit-arithmetic RNE emulation costs 1.1 ms at 4K, the AGS conversion
  nothing. It flushes a few binary16 subnormal results to zero (29 of 65536
  random tiny values); keeping them exact costs 0.8 ms.
- `precise` stays in the network stages (free in time, 1.2 dB for E) and is
  dropped in the output stage (hkpn), where it costs 0.2 ms and no quality.
- The attention numerator reads the cubic table: computing it was 0.33 ms
  slower.

The removed integer FP8 profile kept scalar FP8 values as codes and cleared NaN
codes after every conversion (13% of the network time); it lost 12-15 dB.

A driver hazard: an AGS conversion whose result is used only inside a divergent
branch is moved into that branch, and its lowering then reads inactive lanes.
The stages store re-packed token words under bounds conditions; with plain
words there, encoder 2 onwards turned NaN. The decode/encode round trip that
used to stand there hid this. `rrPin` now keeps those conversions in uniform
flow with an unconditional shared-memory store.

Tried, no gain: unrolled head/MLP loops (above); `[flatten]` on the translated
gather's conditions (no change); the AGS conversion for single-value FP8
encoding (2% of encoder 0, but it would run in divergent code).

## Diagnostics

The runner prints GPU timestamps for exposure and every stage. Repeat counts
exercise temporal history, matching the Vulkan replay runner; use one repeat for
quality comparison. `D4R_RR_DUMP=1` saves intermediate surfaces and histories.
`D4R_RR_TEST_DISCARD=1` discards and re-records every frame to exercise history
rollback. `D4R_D3D12_DEBUG=1` enables the debug layer and fails on stored errors.
The current D and E builds pass the 32-frame debug-layer replay, and their
discard replays match the normal replay byte for byte.

`profile_rr.py` captures one stage with RDP CLI, using separate stage
submissions and a file gate before the selected stage on the third replay:

```powershell
python engine/d3d12/profile_rr.py --panel PATH/RadeonDeveloperPanelCLI.exe `
  --runner build/rr-win/d4r-rr-d3d12.exe --model build/rr-d --case CASE `
  --stage enc0 --output build/rr-d-enc0.rgp
```

The underlying replay controls are `D4R_RR_PROFILE_SPLIT=1`,
`D4R_RR_PROFILE_GATE=PATH`, and `D4R_RR_PROFILE_BOUNDARY=1..12` (the boundary
before enc0 through hkpn). Split timestamps contain CPU waits and must not be
used as throughput measurements. The capture's code objects hold the driver's
ISA for every pipeline (`llvm-objdump --mcpu=gfx1201` from the RDTS `utils/lc`
toolchain decodes all gfx12 instructions; LLVM 19's does not).

### Finding a stage's bottleneck with RGP and RGA

Neither tool needs the GUI; both ship in the Radeon Developer Tool Suite
(`RadeonDeveloperPanelCLI.exe`, `rga.exe`).

1. Capture the stage with `profile_rr.py` (above). It enables instruction
   tracing and counter collection, so the `.rgp` holds the streaming (SPM)
   counters RGP's Counters view shows.
2. Read the counters: `python engine/d3d12/rgp_counters.py CAPTURE.rgp` prints
   a timeline. Each dispatch is a plateau, and the dip between two plateaus is
   the barrier. `--window START END` totals one plateau, and `--csv` dumps every
   sample. Use it to classify the stage:
   - Memory unit busy near 100% with a high L0 hit rate and little fetch
     traffic means vector memory latency. More waves help; fewer ALU
     instructions do not.
   - High LDS bank conflict, instruction cache misses or fetch size point
     elsewhere.
3. Check occupancy with RGA. It compiles the built DXIL with the installed
   driver, the same compiler as the runtime:

   ```powershell
   rga -s dx12 -c gfx1201 --cs-blob build/rr-f/dec0_0_m.dxil `
     --isa OUT/x.isa --livereg OUT/x.lr -a OUT/x.csv
   ```

   `--isa` is required; without it RGA writes nothing. The end of
   `gfx1201_x_comp.lr` reports `Maximum # VGPR used N, VGPRs allocated by HW: M`.
   gfx1201 has 1536 wave32 VGPRs per SIMD with a 24-register granule, so the
   waves per SIMD are 1536 // M:

   | allocated | 120 | 168 | 192 | 216 | 240 | 264 |
   |---|---|---|---|---|---|---|
   | waves/SIMD | 12 | 9 | 8 | 7 | 6 | 5 |

   LDS can limit occupancy further: a one-wave workgroup gets 64 KiB / LDS
   per CU. Per line, the `.lr` file shows the live count `Rn` and which
   registers are live. A register defined before a loop, unused inside it and
   read after it is the usual waste.
4. Compare with RADV. Dump RADV's ISA (`RADV_DEBUG=asm,shaderstats` with the
   noop DRM shim) and count its registers the same way.
5. Rebuild the one stage, check the allocation with RGA, and time it before
   doing a full build.

Example: dec0_0 showed 100% memory-unit busy. RGA reported 255 requested and
264 allocated (5 waves) against RADV's 216 (7 waves); the cause was the
residual sunk past the head loop (`residual_pack`). The fix gave 216 VGPRs and
took the stage from 1218 to 1053 us.

The `SqttData` chunks hold the instruction trace that RGP's Instruction Timing
view uses. rocprof-trace-decoder parses the gfx12 token stream, but the
timings it produced from RGP captures were not usable; use the RGP GUI for
per-instruction timing.

Benchmark:

```powershell
python engine/compile_rr_dxil.py MODEL_RR_D --output build/rr-d
python engine/d3d12/bench_rr.py --runner build/rr-win/d4r-rr-d3d12.exe `
  --baseline BASELINE_MODEL --candidate build/rr-d --case CASE `
  --output build/rr-timing --trials 3
```

The benchmark rotates run order and rejects missing, duplicate or malformed
timestamp samples. Performance repeats advance history, so these outputs are
separate from single-repeat quality replays. It does not change GPU clocks.
Scratch scripts, logs, ISA and measurements of this pass are under
`build/rr-cmp/` and `build/rr-exp/`.
