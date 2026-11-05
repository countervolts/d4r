# D3D12 inference backend and diagnostic tools

[Native RR D/E/F](windows-rr-tuning.md) is built with `compile_rr_dxil.py` and
replayed with `d4r-rr-d3d12`. `d4r-rr-zluda` and
`scripts/rr_reference_windows.py` generate synchronous Windows ZLUDA references.

K's [full-frame native D3D12 backend](windows-k-tuning.md) is packaged by
`compile_k.py --dxil yes` alongside Vulkan's kernels and replayed with
`d4r-k-d3d12`. It includes all encoder/decoder layers and reconstruction.

`--d3d12-target vkd3d` builds the package for vkd3d-proton on RADV, which needs
the patched vkd3d-proton (`patches/vkd3d-proton` 0006 and 0007). Both targets
filter every subpixel exactly; the centre Gaussian estimate the Windows target
used to share lost thin detail (17-19 dB against the ZLUDA reference on the
harness's reset frames, 48-51 dB without it). With Shader Model 6.9 stages and fused
precise `mad` it runs within 0.5% (4K) and 1.5% (1440p) of native Vulkan on an
RX 9070 XT; `compile_k.py`'s `build_dxil` records the measurements.

The standalone block experiments documented below are diagnostic tools for
the original HLSL migration. Their dot2/WMMA variants do not select packaged
full-frame kernel implementations. Game integration remains unverified.

The milestone 1 [host contract](host-contract.md) (`host.h`, `d4r-host-test`)
and the D3D12->Vulkan interop measurement (`d4r-interop-bench`) live here too.

The shader performs normalization, Q/K/V projections, window attention, output
projection, both residual additions and the chunked MLP in a single dispatch.
Two implementations share the existing 448-byte `<4i108I` parameter layout and
the f16 activation/rounding helpers in `k_math.hlsli`:

- `layer_k.hlsl`: the original dot2 correctness baseline, using pair-interleaved
  weights from `test_k.dot2_weights`.
- `layer_k_wmma.hlsl`: accelerated AMD WMMA, using the existing row-major model
  weights directly, 16x16x16 f16 products, and f32 accumulation. This is selected
  explicitly with `--backend wmma`; dot2 remains the default baseline.

No new weight format or extraction process is needed. Both paths currently
support plain non-positional blocks only, not encoder/decoder stages.

The original dot2 shader's default C=160 layout needs more than 32 KiB of shared
memory. This port uses two waves/heads per round to fit D3D12's limit. That raises
register pressure, and that baseline is **not performance competitive**. The
WMMA implementation instead uses four waves, the Vulkan matrix shader's padded
shared layouts, and 31 KiB of shared memory at C=160. It keeps normalized tokens
and residual accumulators in registers across heads. The baseline has now been
checked on native Windows and Linux/vkd3d on RX 9070 XT. Its performance ranking
reverses between those systems; see the [Linux report](linux-radv-benchmark.md).
The newer tuning variants below have only been tested on Windows so far. Game
integration remains unverified.

## Synchronization and scheduling ablations

The measured baseline remains the default. `test_layer.py --backend wmma` also
accepts `--wmma-variant`:

| Variant | Wave-local LDS fences | Paired weight loads |
|---|---|---|
| `baseline` | Group memory + group execution barrier | Original scheduling |
| `wave-sync` | Memory fence without group execution barrier | Original scheduling |
| `prefetch` | Original group barriers | Two K steps / output tiles at a time |
| `combined` | Memory fence without group execution barrier | Paired loads |

`wave-sync` changes only regions owned by a single fixed wave32: normalized
input rows, score/softmax rows, per-wave attention output, residual rows and MLP
hidden rows. All calls are converged. It retains `GroupMemoryBarrier()` for LDS
ordering. The initialization of the wave counter and both cross-wave K/V
boundaries per attention head keep `GroupMemoryBarrierWithGroupSync()`. For
C=160/NH=8 that changes 86 group execution barriers into 17 group execution
barriers plus 69 memory-only fences per workgroup. It is a tested optimization
candidate for this fixed-wave AMD extension path, not a general instruction to
remove shared-memory synchronization from other shaders.

`prefetch` follows `layer.comp` more closely: request four weight tiles for two
K steps before using them in Q/K/V and fc1, load output-projection tiles in
pairs, and request the first fc2 tiles before GELU. Each accumulator still sees
the same arithmetic order. Larger live operand sets can increase register
pressure, so measure this separately from the synchronization change.

Five rotating/interleaved trials on native Windows / RX 9070 XT, synthetic
C=160/NH=8/32x32/seed 42, 100 ordered dispatches per batch, debug layer disabled:

| Variant | Median of trial medians | Trial range |
|---|---:|---:|
| Baseline | 121.877 us | 119.388–123.478 us |
| Wave-local fences | 116.307 us | 115.838–122.502 us |
| Prefetch | 74.587 us | 73.331–76.829 us |
| Combined | 71.949 us | 69.483–75.538 us |

Every variant produced bit-identical outputs across all trials, including the
separate debug validation runs, and all matched baseline bit-for-bit. The
combined median is 41% lower than baseline on this Windows run. A larger
128x64 grid (three rotating trials) measured 138.188 us baseline versus
94.878 us combined; output remained bit-identical. Clocks were not pinned;
these are exploratory shader-latency results, not full-frame inference or proof
that the variants improve RADV.

`bench_wmma.py` generates one common fixture set, compiles each variant once,
checks every timed result against the CPU oracle, requires stable output within
each kernel, and rotates the first kernel every round. Separate debug runs are
optional; timed runs always omit the debug layer. It saves commands, raw trials,
hashes, compiler versions, DXIL listings, static barrier-mode counts and
`ablation-result.json`. Dispatches continue to have write-ordering barriers;
the benchmark does not use unsafe overlapping writes to the same output.

For a Linux/RADV comparison using the environment already prepared in the
[Linux report](linux-radv-benchmark.md):

```sh
python3 engine/d3d12/bench_wmma.py build/wmma-ablation \
    --runner build/d4r-k-layer.exe --vulkan-runner build/d4r-vulkan-layer \
    --dxc /path/to/dxc --trials 7 --repeats 100 --launch-prefix wine
```

The Vulkan runner stays native; only the D3D12 runner receives the Wine prefix.
Use `--vulkan-launch-prefix` separately if explicitly testing a Windows Vulkan
runner. The same independent prefix behavior is now supported by `test_layer.py`
and `test_matrix.py`, fixing the wrapper limitation in the original Linux
report. Keep the reported driver/layer/cache settings and repeat with a distinct
work directory plus `--width 128 --height 64` to inspect grid scaling. The
original CMake-built `dec5-wmma.dxil` remains the baseline; the wrapper and
ablation script compile the selected variants explicitly.

## AMD WMMA result and reproduction

The shader uses the AGS reserved register space `0x7fff0ade`, matching the public
[vkd3d-proton WMMA tests](https://github.com/HansKristian-Work/vkd3d-proton/blob/master/tests/d3d12_sm_advanced.c)
and [dxil-spirv's protocol reference](https://github.com/HansKristian-Work/dxil-spirv/blob/master/shaders/ags/wmma_ags.h).
`ags_wmma.hlsli` implements the small f16/f32 subset needed here, including buffer
and LDS loads/stores, conversion and matrix multiply-add. Matrix fragment layout
stays opaque. The runner adds the reserved root UAV and uses raw buffer views in
WMMA mode. **No AGS library or AGS-created device was needed** in native Windows
tests. This is a vendor extension protocol, not the SM6.10 Linear Algebra API;
no preview Agility SDK or driver installation was required.

The extension is experimental and currently tested only on RX 9070 XT / AMD
26.9.1, plus the Linux baseline configuration recorded in the linked report.
Do not infer support on other architectures from the regular SM6.6
feature checks. The explicit WMMA test validates actual output rather than
silently falling back to dot2 on an unsupported implementation.

```powershell
cmake -S engine/d3d12 -B build/d3d12 -A x64 -DD4R_D3D12_WMMA_TESTS=ON -DD4R_D3D12_VULKAN_REFERENCE=ON
cmake --build build/d3d12 --config Release
ctest --test-dir build/d3d12 -C Release --output-on-failure
python engine/d3d12/test_layer.py build/d3d12/wmma-speed --runner build/d3d12/Release/d4r-k-layer.exe --synthetic 160 --heads 8 --width 32 --height 32 --backend wmma --repeats 100 --vulkan-runner build/d3d12/Release/d4r-vulkan-layer.exe --vulkan-time
```

Both paths on native Windows, GPU timestamps, no debug layer, synthetic weights,
C=160 / 8 heads / 32x32 tokens / shift (0,0):

| Kernel | GPU time per dispatch |
|---|---:|
| Initial D3D12 dot2 baseline, earlier 20-repeat run | 2000.104 us |
| D3D12 WMMA, 100 repeats | 119.140 us |
| Existing Vulkan matrix shader, same 100-repeat comparison | 323.107 us |

In this run WMMA is 2.71x faster than the existing Vulkan implementation. This
is a comparison of these kernels on this Windows driver, not evidence that the
D3D12 API is intrinsically faster or a comparison against Linux/RADV/HIP. Output
agreement is 88.43 dB against Vulkan and 88.39 dB against the independent NumPy
oracle (maximum absolute error 0.001953125 for both). The shifted 11x9 / C=64 /
2-head case also passes, at 90.00 dB against Vulkan. All cases use synthetic data;
captured weights and game quality still need checking.

`test_matrix.py` separately proves the matrix protocol before the full block:

```powershell
python engine/d3d12/test_matrix.py build/d3d12/wmma-tiles --runner build/d3d12/Release/d4r-k-layer.exe --vulkan-runner build/d3d12/Release/d4r-vulkan-layer.exe --tiles 128 --steps 128 --repeats 100
```

Each independent tile computes `C + steps * A@B` using unique random f16 A/B and
nonzero f32 C. The reference uses float64. Tests reject nonfinite results, wrong
sizes and writes beyond the output tile array. At 128 tiles / 128 dependent
products, D3D12 measured 2.294 us versus Vulkan's 2.606 us per dispatch, with
**bit-identical f32 outputs**. These are tile microbenchmarks, not layer or frame
timings. The test suite covers one tile, 128 independent tiles, and 128 tiles
with 128 dependent accumulations.

Add `-DD4R_D3D12_DEBUG_TESTS=ON` for WMMA accuracy tests with the D3D12 debug
layer. Omit it for machines without Windows Graphics Tools. The speed comparison
test intentionally leaves the debug layer off.

## Build on Windows

Requires a C++17 compiler and Windows SDK, CMake, and a DXC executable on PATH.
The shader needs Shader Model 6.6, wave32, and native 16-bit operations. No CUDA,
HIP, ROCm, NVIDIA DLLs or downloaded weights are needed for synthetic testing.
Python 3 and NumPy are needed for the tests; `-DBUILD_TESTING=OFF` builds without
Python. The optional Vulkan comparison also needs a Vulkan SDK and glslang.

From the repository root, in a developer shell (CMake can also discover an
installed Visual Studio compiler automatically):

```powershell
cmake -S engine/d3d12 -B build/d3d12 -A x64 -DD4R_D3D12_GPU_TESTS=ON
cmake --build build/d3d12 --config Release
ctest --test-dir build/d3d12 -C Release --output-on-failure
```

GPU tests are opt-in. CPU tests check the existing parameter ABI, dot-product
orientation, residual rounding, shifted-window coverage and failure handling.
GPU tests use deterministic nonzero synthetic weights and an independent NumPy
oracle. They cover the 160-channel/eight-head shape and shifted, partial windows.
Every output must be finite and meet the accuracy threshold (60 dB by default).

For the debug layer, enable Windows' Graphics Tools optional feature if it is
not already installed. `--debug` fails on validation errors as well as accuracy
failures. It is optional; it is available on the machine used for initial tests.

```powershell
python engine/d3d12/test_layer.py build/d3d12/check --runner build/d3d12/Release/d4r-k-layer.exe --synthetic 160 --heads 8 --debug
```

The test saves parameters, packed weights, input, GPU output, CPU reference and
`result.json` under the requested work directory. Shader defines and the replay
data must agree. The low-level executable intentionally consumes prepared binary
data; use the Python wrapper to validate model offsets and compile matching
shader dimensions.

## Compare with the existing Vulkan matrix shader

```powershell
cmake -S engine/d3d12 -B build/d3d12 -DD4R_D3D12_VULKAN_REFERENCE=ON
cmake --build build/d3d12 --config Release
python engine/d3d12/test_layer.py build/d3d12/compare --runner build/d3d12/Release/d4r-k-layer.exe --synthetic 160 --heads 8 --debug --vulkan-runner build/d3d12/Release/d4r-vulkan-layer.exe
```

This compiles the existing `engine/layer.comp` without editing it and compares
its output on the same inputs/row-major weights. The Vulkan adapter must support
the shader's f16 cooperative matrix shape and features. This is an output
comparison; the old Vulkan runner's wall-clock benchmark is not used to claim a
GPU speed comparison unless the opt-in timestamp mode below is selected.

Add `--vulkan-time --repeats 20` and omit `--debug` to compare GPU timings.
`D4R_EVK_GPU_TIMING=1` enables the reference runner's GPU queries and orders its
repeated storage-buffer writes. Its default timing mode remains wall clock.
Both paths exclude uploads/readback and include ordering between dispatches.
The runners use different batch/warmup counts, so these are exploratory medians,
not tightly controlled performance claims.

Historical dot2 baseline: RX 9070 XT / AMD 26.9.1, both paths on native Windows, synthetic C=160 with eight
heads, shift (0,0), 20 repeats, debug layer off:

| Token grid | D3D12 dot2 prototype | Existing Vulkan matrix shader | Prototype / Vulkan |
|---|---:|---:|---:|
| 8x8 | 2028.546 us | 321.340 us | 6.31x |
| 32x32 | 2000.104 us | 331.130 us | 6.04x |

The 32x32 outputs agree at 78.51 dB. This compares the initial implementations:
dot2 versus accelerated matrices. It does **not** isolate API overhead, establish
the performance of the separate WMMA backend, or measure Linux or full-frame
game performance. The dot2 baseline does not meet the performance gate.

To reproduce the second case, append `--width 32 --height 32 --vulkan-time
--repeats 20` to the comparison command above and remove `--debug`. Full inputs,
outputs, kernel choices and timings are saved in the work directory's
`result.json`.

Native Windows RX 9070 XT, AMD 26.9.1, initial synthetic checks:

| Case | Against NumPy | Against existing Vulkan matrix shader |
|---|---:|---:|
| C=160, 8 heads, 8x8 tokens | 77.40 dB | 77.44 dB |
| C=64, 2 heads, 11x9 tokens, shift (4,3) | 76.79 dB | not initially measured |

These establish mathematical agreement on synthetic data, not game quality or
parity against NVIDIA. The initial 8x8 C=160 case took approximately 2 ms per
dispatch with the debug layer; it is a correctness baseline, not a useful game
performance result. The runner reports median GPU timestamp timings over five
batches after warmup, excluding uploads/readback and including UAV ordering
between repeated writes. Run without `--debug` for performance investigations.

## Replay an extracted model

```powershell
python engine/d3d12/test_layer.py build/d3d12/captured --runner build/d3d12/Release/d4r-k-layer.exe --model path/to/k --layer dec5 --input path/to/dec5.in --width 16 --height 16 --shift 0 2 --reference path/to/dec5-vulkan.f16
```

`--model` is the prefix of the existing `extract_k.py` `.json` and `.bin` files.
`--input` is the plain f16 `[height][width][C]` token grid, as exported for a plain
layer by `test_k.py`. Supply that launch's actual token dimensions and window
shift. Only plain, non-positional layers with folded gains are accepted; encoder
and decoder layers are rejected. The NumPy oracle also runs on these weights.
Add `--backend wmma` for the accelerated path (C >= 64); it uses row-major weights
and requires no dot2 repacking. Native replay with actual captured K data has not
yet been tested on the development machine.
`--reference` optionally checks an existing result; `--vulkan-runner` can instead
produce the comparison directly. Large CPU reference grids can take time.

## d3dhip: the native HIP network between the D3D12 stages

`d3d12/hip_net_d3d12.h` (`d4r::dx::HipNetworkDx`) runs the model's native HIP layer kernels (`engine/hip_net.h`) between
the D3D12 input and expansion stages. The token and head buffers are committed D3D12 buffers
(`D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`, `D3D12_RESOURCE_STATE_COMMON`, never transitioned), shared with HIP
through `ID3D12Device::CreateSharedHandle` and imported by hip_net's Windows path with
`hipExternalMemoryHandleTypeOpaqueWin32`; construction checks that HIP device 0 is the D3D12 adapter — its PCI identity
is AMD (SetupAPI) and it is the card behind the device's LUID (PCI vendor/device/subsystem; ROCm 7.14 does not report
the LUID, so it comes from DXGI's enumeration). A description OptiScaler spoofed (an NVIDIA card for the game) is
accepted only on a machine with a single AMD display adapter; otherwise construction throws with the reason.

Synchronization is CPU-serialized on purpose: record the front (exposure and the input stage, which writes the tokens)
and wait for a D3D12 fence on the CPU, call `HipNetworkDx::run()` (one `hipGraphLaunch` plus `hipStreamSynchronize`), then
record the back (expansion, reconstruction, downsample) and submit. Zero-copy shared fences are not attempted. The M/L
engine (`MEngine`) and the K engine (`Engine`) gained `ExternalNetworkDx` constructors and `recordFront`/`recordBack` whose
dispatch sets equal `recordFrame`'s; M needs the plane-layout `enc0_m_planes`/`dec0_m_planes` DXIL (`compile_m.py --dxil
yes`), K's token/head layout already matches the native K plan.

```
python engine/compile_m.py CAPTURE MODEL --dxil yes --hip KERNEL_DIR   # KERNEL_DIR: D4R_VKHIP_LAYERS=1 kernels/build.sh m DIR
d4r-m-d3d12 MODEL 1280 720 2560 1440 OUT [repeats] hip -- COLOR MOTION DEPTH JX JY ...
d4r-k-d3d12  MODEL CASE.txt OUT [repeats] hip
```

A model with `hipnet.bin` and `hip/` takes the route by default (the runner logs
`network: hip (default; D4R_DX_NET=dxil to force the DXIL network)`); `D4R_DX_NET=dxil` keeps the DXIL network byte
for byte. A missing HIP network, plane DXIL or HIP import logs
`network: hip unavailable (<reason>); using the DXIL network` and runs the DXIL path.

Measured 2026-10-30, RX 9070 XT (gfx1201, driver 32.0.32015.7029), `d4r-m-d3d12` on the sr-m capture's 1280x720 ->
2560x1440 synthetic frames, 12 timed frames: whole frame 1856 us GPU median, front 142 us (exposure 18 + input 124),
network 0.95 ms (HIP events), expansion 89, reconstruction 357, downsample 83; ~1.4 ms of CPU serialization per frame.
The hip output is finite and 54.7-59.2 dB RGB against the DXIL network on those frames (the same frames put the Vulkan
`d4r-m ... hip` route at 50.4-51.9 dB against DXIL, and our output 50.6-53.3 dB against that route: the remaining
distance is the D3D12/Vulkan stage arithmetic). K with a working package (`build/win-vs-linux-20261026/models/sr-k`)
runs the same split: 62.4-63.8 dB against the DXIL network. The DXIL route without `hip`/`D4R_DX_NET=dxil` is
byte-identical to the pre-d3dhip runner on every tested frame.

### In a game (2026-10-30)

`tools/d4r_nvngx_shim.cpp` runs the route for preset M on native D3D12 when the package has `hipnet.bin`, `hip/` and the
plane DXIL (HDR or LDR: `compile_m.py --hip ... --dxil yes [--ldr]`). The game's command list is split at the network
(`tools/d4r_d3d12_split.h`): the front is recorded into the game's list, `begin()` sends later recording into d4r's second
list where the back is recorded, and the queue executes the first part, calls the split's `beforeSecond` hook (wait for
the front's fence on the CPU, `HipNetworkDx::run()`), then the second part. The hook owns the CPU round trip; with a hook
the split satisfies its own fence, so a failing network cannot hang the queue. The engine is created for the first
frame's render size and recreated when the size changes (the layers are built for one token grid).

The NGX shim does not import `amdhip64`: it hung Ghost of Tsushima's device before the first frame next to OptiScaler's
driver hooks. The network is in `d4r_hip_net.dll` (`d3d12/hip_net_bridge.cpp`, `d4rHipNet*` exports) beside `nvngx.dll`,
loaded on the first preset M frame that uses it; `D4R_DX_NET=dxil` forces the DXIL network. Ghost of Tsushima's LDR
flag was verified in game on RX 9070 XT at 1920x1080 -> 3840x2160 preset M (see `docs/experimental-vkhip.md`); the HDR
package is built but unverified in game.

### Ray Reconstruction on the same split (2026-10-30)

`d3d12/hip_net_rr.h` (`d4r::dx::HipNetworkRr`) does the same for the RR engines (`runtime_rr_d3d12.h`'s
`ExternalNetworkRr`). Presets D and E run the model's ten `hip/rr_<stage>.hsaco` stages (enc1 - enc5, dec4 - dec0, the
code objects `kernels/rr/build_vkhip.sh` builds for the Vulkan vkhip route) on a private HIP stream: the engine records
only its front (parameters, reset, exposure, encoder 0) and its back (the output stage), and the network's tensors live
in HIP memory. The plane arena is one D3D12 buffer imported as an opaque NT handle (zero-copy); the three skip images
encoder 0 leaves for the network (surfaces 128/112/120 at level 0) and the three decoder 0 hands back (surface 112 and
surface 144 for each frame parity) travel through D3D12 buffers shared the same way, staged into and out of HIP images
(`hipMemcpy2D`). Those images are row-major; the stages load, sample and store them through hardware image descriptors
that HIP builds for a pitched 2D resource (its texture object's 8 descriptor dwords), and the bilinear sampler is a
constant descriptor (CLAMP on all axes, `FILTER_MODE` in word 2 on gfx12). Preset F runs the fifteen
`hip/rr2_<stage>.hsaco` objects instead, which work directly in the engine's arena through their own parameter blocks
(the engine hands each stage's block and grid over between the front and the back, and the arena reserves the phases'
records for every phased stage, as the Vulkan route does when it gives the recorder device addresses). Encoder 0 is one
of them when the package has `hip/rr2_enc0.hsaco` (the vkhip route's hybrid stage, which runs after the gather pass):
the front then records only the parameters, the exposure and the gather, and hands encoder 0's block over first.
`D4R_HIP_RR_ENC0=0` (or a package without that file) keeps encoder 0 on the engine's DXIL front.

```
d4r-rr-d3d12 MODEL INPUT_DIR OUT [repeats] hip
```

A model's `hip/rr_*.hsaco` (D/E) or `hip/rr2_*.hsaco` (F) take the route by default; `D4R_RR_DX_NET=dxil` keeps the
DXIL network byte for byte, and a missing HIP network, incomplete stage set or failed import logs
`network: hip unavailable (<reason>); using the DXIL network`.

Measured 2026-10-30, RX 9070 XT (gfx1201, driver 32.0.32015.7029), `d4r-rr-d3d12` on the 1280x720 -> 2560x1440 RR
capture (`build/local-preset-compare-20261029/inputs/rr-1440p`): the network takes 1.8-1.9 ms of GPU time per frame
(HIP events, the three staged images included) and a CPU-serialized frame 3.7-4.7 ms from the front's submission to
the back's. The output is finite: on the first frame 46.5 dB (rr-d) and 43.4 dB (rr-e) RGB against the DXIL network at
peak 1, 54.8 and 55.3 dB with the frame's own range as the peak, mean absolute error 0.0023 / 0.0033. Over the full
140-frame sequence rr-d stays at 35.2 dB (peak 1, 58.6 dB at the output range): the temporal history amplifies the
kernels' f16 rounding. The Vulkan SPIR-V route on the same frames is 43.2 dB against the D3D12 DXIL network, so the
HIP stages are closer to this engine than the Vulkan shaders are.

Preset F runs its `hip/rr2_*.hsaco` (built by `kernels/rr/build_vkhip_f.sh`, the same objects as vkhip's F route,
encoder 0 included when the package has it) on the HIP stream too. Each kernel takes one 24-byte by-value argument,
`{param, weights, arena}` as vkhip lays out its kernarg block, so the launch passes that struct, not three pointers: HIP
copies the kernel's declared 24 bytes from the first pointer, and the garbage weight and arena addresses hung the
driver. On four 1440p frames the output is 42.2 dB RGB against the DXIL network at peak 1 (54.0 dB at the output range;
vkhip against the same DXIL run is 42.2 / 54.0 dB), at 4k 43.2 / 51.9 dB, no nonfinite samples. Encoder 0 on HIP is
byte-identical to the DXIL front's (8 frames at 1280x720 -> 2560x1440, `inputs/rr-syn`, 15 repeats) and takes 0.47 ms
instead of 0.55: front 0.85 -> 0.30 ms, network 1.81 -> 2.28 ms, back 1.15 -> 1.12 ms, the frame 3.81 -> 3.69 ms
(3 rounds interleaved, spread 0.2%).

## Linux execution and remaining gates

The runner uses normal D3D12; WMMA additionally uses the AMD extension protocol
already implemented by upstream vkd3d-proton. Build
the Windows executable using an appropriate cross toolchain, or copy a Windows
build and its required C++ runtime. Then run it under Wine/Proton with your
vkd3d-proton DLLs and DXGI configuration. For example, after configuring that
environment, `--launch-prefix wine` lets the Python wrapper launch the executable.
The wrapper translates Unix absolute data paths through the prefix's default
Z: mapping, which must be enabled. Cross compilation and baseline Linux execution
were subsequently verified in the [Linux/RADV report](linux-radv-benchmark.md):
42.57 us native Vulkan versus 91.52 us WMMA through vkd3d, with matching fixtures
and correct outputs. That result predates the tuning variants. The Windows
development session cannot measure their RADV performance: its WSL environment
has no Wine/vkd3d or DRI device. Actual captured K data remains untested.

Before making this the shared production engine:

1. Run the same replay cases on Linux/vkd3d, with captured K weights and inputs.
2. Recheck the tuning variants on Linux/vkd3d and compare GPU time with the
   current Linux Vulkan/HIP paths. Preserve native Vulkan as the Linux baseline;
   the existing WMMA port was 2.15x slower there before these ablations.
3. Add embedding, patch merge/expand and output head, then K's input,
   reconstruction, exposure and temporal stages.
4. Introduce a reusable D3D12 engine that records into caller-owned command
   lists and retains resources until GPU completion; this runner owns its queue
   and waits only for standalone testing.
5. Integrate NGX/OptiScaler, then validate game resource states, resets,
   subrectangles, dynamic resolution and multiple frames in flight.

Keep model packing, frame mathematics and shader algorithms shared as the
backend grows. Preserve the existing engine as a correctness/performance
reference until the new path meets those gates. M/L and Ray Reconstruction are
separate migrations.

## Experimental M/L RDNA 3 backend

Two native D3D12 M/L backends share one model package:

| | Hardware | Network | Host | Shaders | Replay target |
|---|---|---|---|---|---|
| Default | RDNA 4 (FP8 WMMA) | `layer_m.hlsl` | `runtime_m_d3d12.cpp`, `d4r::dx::MModel` | package root | `d4r-m-d3d12` |
| RDNA 3 | no FP8 WMMA | FP16 WMMA, FP8 storage in software (`layer_m_wmma.hlsl`) | `runtime_m_rdna3.inc`, `d4r::dx::rdna3::MModel` | `<package>/rdna3/` | `d4r-m-rdna3` |

The shim tries the default backend first and falls back to the package's `rdna3/`
shaders when those do not load (a driver without FP8 WMMA fails at pipeline
creation). `D4R_ENGINE_M_BACKEND=rdna3` forces the RDNA 3 shaders.
`compile_m.py` builds both by default (`--dxil` and `--rdna3`, each
`auto|yes|no`; `--rdna3-math fast|fp8`); `m_d3d12_build.py` adds only the RDNA 3
shaders to an existing package.

`d4r-m-layer`, `test_m_layer.py`, `test_l_wmma.py` and `bench_m_network.py`
validate native FP16 WMMA network blocks and software FP8 storage. The
`d4r-m-rdna3` target in the parent engine build replays complete M/L frames.
The native NGX shim supports model selection with separate HDR/LDR packages and
runs on the game's command list.

Split-expansion builds now use `D4RML006`, with uint4 input/output views for the
shared network blocks. The shader keeps scalar I/O by default; the native builder
enables the vector schedule. `m_d3d12_build.py --scalar-io` retains earlier package
markers for controlled comparisons. The native host accepts both schedules;
older hosts reject `D4RML006`. L's C32 pixel blocks keep their existing I/O path.
Layer accuracy checks can use `test_m_layer.py --backend wmma --vector-io
--cbv --split-expand` with a rebuilt layer runner.

Fast native builds consume one MLP hidden tile at a time in registers for the
five shared network shapes. This preserves the matrix rounding before GELU,
the scalar packing conversion after GELU, and
each residual's product order. `m_d3d12_build.py --lds-mlp` retains the previous
LDS schedule for comparison. L's pre-existing reconstruction nondeterminism remains
unresolved; the network change is not a fix for it. FP8 math and L's C32 pixel blocks
retain their existing MLP schedules. Layer checks select this schedule with
`--fast --define D4R_M_STREAM_MLP=1`.

Both models have launched in Ghost. Performance targets remain unmet. Isolated network
timings must not be presented as complete-frame or gameplay results.

## Native Windows OptiScaler model selection

Native D3D12 supports K (11), L (12), and M (13). The same preset-following policy
used on Linux also runs on Windows. Set `[OptiScaler] FollowPresets=true` in
`d4r/d4r.ini`. In OptiScaler's menu (Insert), open DLSS Settings, enable Render
Presets Override, choose K/L/M, and Apply Changes. Other model presets require
backends that are not available in this native Windows build.

For an override enabled at launch, OptiScaler.ini uses `[DLSS]` keys
`RenderPresetOverride=true` and `RenderPresetForAll=13` (M). `RenderPreset` is not
the supported setting in the installed OptiScaler version. Default/Latest slots
fall back to `[DLSS] Model` in d4r.ini. The harness accepts
`D4R_HARNESS_DLSS_PRESET` to test the first creation's hints and
`D4R_HARNESS_RECREATE_PRESETS` to test live menu changes.

The 2026-10-23 Ghost install enables both preset following and OptiScaler's
override, initially selecting M. Native integration tests verify M -> K -> L -> M
-> Default, an override already enabled at launch, and FollowPresets=false.
Logs: `build/ml-rdna3/windows-preset-override`. Tests call the native Windows NGX
API with actual packaged models; they do not automate the OptiScaler overlay.
