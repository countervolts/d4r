# Native Windows preset K backend

2026-10-22, RX 9070 XT. D3D12 is a native alternative to K's Vulkan kernels.
There is one packaged D3D12 implementation; no numerical profile selection is
required. Its kernels use:

- Native half conversion and f16 WMMA accumulators.
- Root constant buffers for layer parameters, immutable offset uploads, and
  prefetched column-major 16x16 B-operand weight tiles.
- Wave32 network dispatches, one token tile per wave on Windows, and wave32 input.
- Wave64 reconstruction that filters every subpixel, as Vulkan does. The
  "bright" flag of each 2x2 pixel quad goes through shared memory: AMD's D3D12
  compiler does not give the dialect's subgroupShuffleXor (WaveReadLaneAt with
  a per-lane index) the quad's lanes, which left HDR highlights on the
  non-bright reconstruction table. The twelve-tap centre Gaussian estimate the
  Windows target used before (`D4R_GAUSSIAN_CENTER`) is no longer built: with
  both fixed, an HDR-highlight harness capture went from 46-47 dB to 51-52 dB
  against the CUDA reference, as Vulkan (RX 9070 XT, 2026-10-23).

The arithmetic differs from Vulkan. Vulkan's
SPIR-V, canonical weights and numerical behavior are unchanged. The isolated
matrix/layer test utilities retain their diagnostic implementations; they are
not selectable packaged D3D12 profiles.

## Measurements

Current measurements use fixed-peak-1, unclipped linear RGB PSNR. The
historical measurements are labeled separately and use reference-peak PSNR.

### Cyberpunk application-profile correction (2026-10-23)

On RX 9070 XT, the identical standalone replay executable renamed
`Cyberpunk2077.exe` reproduces the live game's corruption byte for byte.
The model, captured parameters, reconstruction LUT, full network head,
sampled luma history and blend controls agree; the Gaussian accumulation
differs. A captured 4K scene had 454,956 all-zero RGB pixels in that replay
versus none under the normal executable name. This is executable-specific
driver behavior, not a difference in the network weights or shim bindings.

The first correction defined `D4R_PROFILE_SAFE_HALF`: Gaussian sums remained
in float32 registers and each tap was explicitly rounded to binary16 with
integer mantissa rounding and a subnormal-preserving bias. Leaving the
accumulation fully float32, or relying on native conversion rounding, changes
the filter more. The correction is 85.7 dB from the preceding normal-name
output on this frame (fixed peak 1, unclipped linear RGB), with zero black
pixels. The normal-name and Cyberpunk-name corrected outputs are byte-identical.
All 36 saved buffers of a separate six-frame Cyberpunk sequence also match
across those names and are finite. This verifies profile independence on
these captures; it does not establish byte parity with CUDA or Vulkan.

Vulkan SPIR-V is byte-identical before/after this change. The vkd3d target
keeps its original Gaussian accumulation; the workaround belongs to the
Windows package. No new model weights or runtime ABI are required.

Run the application-profile regression with real captured scenes, including
dark detail; flat synthetic fixtures can miss this:

```powershell
python engine/d3d12/test_profile_k.py path/to/model path/to/case.txt build/profile-check --runner build/native-engine-win/Release/d4r-k-d3d12.exe
```

`compare_k.py` and `../test_frame.py` now report fixed-peak-1, unclipped linear
HDR **RGB** PSNR for final output, matching the Linux handoff. Auxiliary
buffers keep their reference-peak metric. Previous Windows final-output
numbers used the frame's brightest value and included alpha: one captured
frame was 79.8 dB that way versus 39.6 dB with the shared metric. Those values
must not be compared across definitions. Nonfinite auxiliary samples now fail
the D3D12 comparison even when both results have NaNs at the same locations.

Three interleaved trials on the captured 2560x1440 → 3840x2160 case,
16-repeat GPU batches under the Cyberpunk executable name, measured median
3505.5 µs before and 3829.1 µs after (about 0.32 ms / 9.2% extra). Clocks were
not pinned; this measures the fix's cost on one replay, not game FPS.
Raw local results: `build/d3d12/profile-fix-timing.json`.

### Speed recovery with a small numerical change (2026-10-23)

The Windows package now defines `D4R_FILTER_ACCUM_F32`: Gaussian sums stay
in float32 without software half rounding after every tap. The final filtered
luma is rounded to half; geometry, exponents, weights and confidence retain
their half precision. Windows enc0 and dec0 also use one token tile per wave.
Vulkan and vkd3d scheduling and accumulation remain unchanged. The manifest
records `d3d12_gaussian_accumulation: f32-sums-f16-result`.

Five rotated trials, 25-repeat GPU batches, profiling/readback disabled,
under the Cyberpunk executable name on the Linux handoff fixtures:

| Output | Corrected per-tap RNE | Current Windows | Saved Linux Vulkan |
|---|---:|---:|---:|
| 3840x2160 | 3188.2 us | 2875.8 us | 2845.4 us |
| 2560x1440 | 1694.2 us | 1580.1 us | 1513.6 us |

This recovers 9.8% at 4K and 6.7% at 1440p. A three-trial verification after
rebuilding the final source measured 2906.6 and 1594.1 us. Clocks were not
pinned; Linux's saved results used a 250 W cap and Windows power was not
matched. These are replay timings, not measured live-game upscaler times.

Against the saved Linux outputs, current fixed-peak-1 RGB PSNR is
56.31–57.01 dB at 4K and 49.39–49.62 dB at 1440p. Compared with the per-tap
RNE correction this loses at most 0.67 dB and 0.09 dB respectively. The final
network scheduling produces the same output as the screened float32-sum
variant on all eight fixture frames. On six real captured game frames,
PSNR against the CUDA references improves by up to 1.94 dB; the largest loss
is 0.10 dB. All six frames have zero all-black RGB pixels. The dark scene
also has zero black pixels, and all 42 saved buffers across these seven
frames are finite and byte-identical between normal and Cyberpunk names.

Raw local results: `build/d3d12/speed-recovery/final-trials.json`,
`final-summary.json`, `final-source-timing.json`, `final-quality.json`, and
the `final-profile-dark` / `final-profile-six` regression directories.
The package uses the same weights and runtime ABI.

### Historical synthetic measurements

Five interleaved trials per package, profiling disabled, on the same synthetic
four-frame fixtures used by the Linux handoff:

| Output resolution | D3D12, us | Linux Vulkan, us | Output PSNR vs Linux |
|---|---:|---:|---:|
| 3840x2160 | 2623.4 | 2845.4 | 62.59–63.29 dB |
| 2560x1440 | 1394.9 | 1513.6 | 58.60–59.02 dB |

D3D12 trial ranges were 2620.8–2624.6 us and 1389.2–1396.5 us. Linux used a
250 W cap; Windows power/clocks were not matched. These synthetic measurements
do not establish game quality or power efficiency. All 48 compared recurrent
buffers were finite and correctly sized. PSNR uses the maximum absolute value
of each Linux reference buffer as its peak; the reference is not NVIDIA.

The accepted implementation was previously measured under the experimental
name `fast`. Raw historical measurements and comparisons remain in local
`build/d3d12/tiles-report.json` and the historical Linux follow-up report.
The normal package is now `build/d3d12/k-d3d12`.

## Build and replay

Use the existing extracted model prefix, reconstruction tables and frame case:

```powershell
cmake --build build/native-engine-win --config Release --target d4r-k-d3d12
python engine/compile_k.py MODEL_PREFIX LUT.bin MODEL_DIR --dxil yes
build/native-engine-win/Release/d4r-k-d3d12.exe MODEL_DIR CASE.txt OUTPUT_DIR 1
build/native-engine-win/Release/d4r-k-d3d12.exe MODEL_DIR CASE.txt - 25
```

On Windows, `--dxil auto` (the default) adds D3D12 kernels when DXC is found.
`--dxil no` packages Vulkan without compiling D3D12. `--ldr` selects the
existing display-referred reconstruction variant for both backends.

## Package contract

`weights.bin` remains canonical for Vulkan/dot2. `weights_d3d12.bin` contains
eight bytes `D4RWTB01`, followed by the D3D12 weight payload. Each 16x16 tile of
qkv, attention-output and MLP matrices is transposed and loaded column-major,
preserving its logical matrix. Biases, gains, attention tables, embedding,
patch merge/expand and head matrices retain their layout. Tensor offsets and
payload size are unchanged.

`d3d12_layer_cbv.bin` contains `D4RCBV02`, selecting root CBVs and the tiled
weight file. The runtime validates the weight header and exact size. Old
runtimes reject this version rather than combining the new shaders with
canonical weights. The current runtime still reads legacy `D4RCBV01` packages
(CBVs/canonical weights) and packages without a marker (structured parameters).
Those compatibility contracts do not expose compiler profiles.

The manifest records `d3d12_reconstruction` and `d3d12_weight_layout`, with
hashes for every packaged shader and weight file.

## Diagnostics

For a live snapshot, set `D4R_D3D12_CAPTURE_DIR` to a writable directory before
starting the game. Create an empty `capture-now` file there while the failing
scene is visible. The runtime records input planes, the actual GPU parameters
after exposure, window geometry, previous/next history, LUT, head and output.
It retains readbacks until the submission fence completes, then saves a
`frame-N` subdirectory and a `complete` marker. Normal recording does no
readback. Discarded command lists discard their pending snapshots.

`python engine/d3d12/prepare_live_k.py path/to/frame-N` prepares `case.txt` for
both replay backends. The helper currently accepts float input formats; it
rejects UNORM depth rather than treating its integer encoding as half floats.

Use `D4R_ENGINE_PROFILE=1` for a separate stage breakdown; omit it from final
timing runs. `D4R_D3D12_DUMP_DIR` saves the exact pipeline `.dxil`, serialized
`.rootsig`, and an opaque `.cached-pso` when available. Capture happens during
model loading; unset the variable afterward. Files are overwritten by later
loads. ISA/resource captures are under local `build/d3d12/windows-isa`.
