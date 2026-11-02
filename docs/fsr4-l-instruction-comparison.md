# FSR4 versus preset L: RGP instruction mix, 2026-10-29

FSR4 is a useful AMD design reference. The measured advantage includes both
less vector work in its matrix core and a substantially smaller total matrix
budget. It does not demonstrate a kernel-only route to FSR4's time for L.
L's inner tube already has nearly the same vector-to-matrix instruction ratio
as FSR4's core; L's outer blocks are the larger vector-work target.

## Actual path and comparison boundaries

Both captures use the same RX 9070 XT (gfx1201), RADV Mesa 26.2.5, 330 W cap,
and 1920x1080 input / 3840x2160 output. RGP 2.7.1.13 ran under Xvfb.

The locally available signed FSR runtime reports and pins **Upscaling 4.1.0**.
Shaders identify `fsr4_model_v07_fp8_no_scale_*`, and the actual matrix opcode
is `v_wmma_f32_16x16x16_fp8_fp8` in all sampled matrix passes, including the
postpass. No INT8 WMMA is observed. Provider names alone were not used to infer
precision. Runtime/provider/D3D12 binary hashes are recorded in the data file.

FSR4 runs through D3D12 → vkd3d-proton → RADV, using GE-Proton11-3's AMD
driver-extension bridge and an app-local `amdxcffx64.dll`. L uses Vulkan outer
stages and native HIP kernels through vkhip. This controls the physical GPU and
Vulkan driver, but not the language/compiler frontend. It is not a native
Windows AMD-driver comparison. The measured runtime is the local 4.1.0 provider,
not a claim about every release in the [current AMD FSR SDK](https://gpuopen.com/amd-fsr-sdk/).

The inputs are patterned RGBA16F color, zero RG16F motion, depth 0.5, four
temporal jitter offsets and reset on the first frame only. FSR4 enables HDR and
automatic exposure. Its final 4K output has no nonfinite RGB components or
all-zero pixels. This confirms a working dispatch, not comparable image quality.

## Non-matrix instructions per WMMA

The numerator counts executed **vector ALU issue instructions excluding WMMA**.
It includes address arithmetic, conversions and moves, but excludes scalar
instructions, vector memory, LDS and synchronization. A VOPD pair counts as
one issue instruction; a separate arithmetic-component metric expands both
operations. None of these ratios measures cycles or matrix utilization directly.

| Region | Non-matrix VALU / WMMA | Estimated WMMA wave instructions |
|---|---:|---:|
| FSR4 matrix core, passes 1–12 | **4.426** | **11,333,640** |
| L inner network, enc1 through dec1 | **6.555** | **31,513,376** |
| L tube dispatches only | 4.432–4.447 | 9,227,520 |
| L outer encoder, including input work | 14.789 | 10,565,440 |
| L outer decoder, including reconstruction head | 13.819 | 9,331,200 |
| FSR4 postpass, including reconstruction work | **32.100** | 1,944,000 |

L inner stages perform about **48% more non-matrix VALU per WMMA** than the
FSR4 core. Comparing only L's outer blocks with FSR4's core would exaggerate
the network-wide difference. Conversely, FSR4's 32.1-ratio postpass shows that
its entire upscaler cannot be characterized by the 4.43 core ratio.

L's twelve matrix stages execute about **51.41 million WMMA instructions**,
versus **13.28 million** in FSR4's core plus postpass, approximately **3.87×**
as many. All are 16×16×16 tiles; L includes 1.55 million FP16 WMMA and 49.86
million FP8 WMMA, while sampled FSR4 matrix work is FP8. L's ten inner stages
alone execute 2.78× the WMMA count of FSR4's twelve core passes. Different
networks and fusion boundaries prevent interpreting these as equivalent work
or equal quality, but total matrix count is strong evidence of a different
compute budget, beyond instruction scheduling.

| Issue category per WMMA | FSR4 core | L inner network |
|---|---:|---:|
| Standalone floating arithmetic, including mixed precision | 0.053 | 3.309 |
| Packed FP16 arithmetic | **0.000** | **1.105** |
| VOPD issue instructions, all component types | 0.940 | 0.359 |
| FP8 pack instructions | 1.093 | 0.575 |
| Other vector conversions | 0.332 | 0.396 |
| Integer / bit / move / compare | 2.009 | 0.810 |
| Vector memory, separately counted | 1.076 | 0.211 |
| LDS, separately counted | 0.035 | 0.041 |

Much of FSR4's FP32 arithmetic is in VOPD pairs, so the 0.053 standalone
number is **not** its total FP32 work. Counting FP32/mixed arithmetic components
separately, including both VOPD components but excluding conversions and packed
FP16, gives 1.716 per WMMA for FSR4's core and 3.224 for L's inner network.
FP8 packing itself is not universally lower in FSR4: it performs more packing
and vector-memory instructions per WMMA in this comparison.

## Ownership, resources and execution

Core shader dataflow repeatedly keeps WMMA accumulators in registers through
bias/add, max and FP8 quantization before storing intermediate buffers. Most
passes use zero LDS. This supports retaining pointwise work in the accumulator
layout and avoiding a mandatory FP16 intermediate. ISA alone does not establish
the complete network topology or every tensor's logical layout.

| Representative shader | VGPRs | LDS bytes shown by RGP | Theoretical waves / SIMD |
|---|---:|---:|---:|
| FSR4 pass 1 / 2 / 12 | 120 | 0 | 12 / 16 |
| FSR4 pass 4 / 5 / 10 | 192 | 0 | 8 / 16 |
| FSR4 pass 7 / 8 | 256 | 6144 | 5 / 16 |
| FSR4 pass 11 | 96 | 0 | 13 / 16 |
| FSR4 postpass | 96 | 1024 | 13 / 16 |
| L outer encoder | 192 | 8192 | 8 / 16 |
| L outer decoder | 216 | 9216 | 7 / 16 |

These are resource limits, not measured achieved occupancy. FSR4 does not
uniformly win through higher occupancy: passes 7/8 have lower limits than L's
outer stages. Pass 11's sampled issue-latency cost is dominated by load waits
(about 45%), with RGP showing roughly 15% VALU and 22% VMEM busy. FSR4 is not
free of memory stalls. These percentages describe that shader and capture;
they are not a whole-frame utilization diagnosis.

The capture contains exposure, prepass, twelve core passes with auxiliary
`*_post` dispatches, and the final postpass. Intermediates cross global-memory
dispatch boundaries. Its advantage is not one giant fused dispatch. Exact
intermediate byte traffic has not been established here; memory-instruction
count is not a substitute for cache or bandwidth counters. The semantics of
the auxiliary passes are not inferred solely from their names.

The useful transfer is explicit register ownership and inexpensive pointwise
operations around matrix instructions. L's normalization, attention and GELU
cannot simply be removed while preserving the network. Matching FSR4's smaller
matrix budget and activation mix would require network changes with training
and quality validation. The [decoder redesign](l-amd-redesign.md) demonstrates
one exact layout change: 9.57% decoder / 1.57% whole-pass improvement.
The subsequent [AMD quality-budget implementation](l-amd-quality.md) also
uses hardware-saturating FP8 conversions and approximate outer-block arithmetic,
with a per-frame reference PSNR gate instead of equality.

## Method, missing samples and reproduction

Instruction timing was copied from RGP's Events view. Counts are instruction
hit counts divided by the sampled wave count, then weighted by each dispatch's
full wave count. The aggregate is a ratio of summed counts, not an unweighted
average of stage ratios. Exported opcode counts and resource readings are in
[fsr4-l-instruction-comparison.json](benchmarks/gpu-limits-20261029/fsr4-l-instruction-comparison.json).

AMD documents that instruction timing covers selected compute units on a
single shader engine, and that its latency is the interval until the next
instruction issues. Wait costs include outstanding operations; they are not
standalone instruction execution times or directly removable frame time.
See [RGP Events documentation](https://gpuopen.com/manuals/rgp_manual/events_windows/).

FSR4's prepass has no usable instruction-timing waves in this capture. Its
static ISA has no WMMA, and RGP reports 96 VGPRs, 2048 LDS bytes and 13/16
theoretical waves. It is excluded from dynamic aggregates. Consequently there
is **no complete-FSR4 non-matrix-per-WMMA figure** here. Sampled L branches
likewise make estimates input- and scheduling-dependent.

The public probe's separate 40-warmup/100-measured timestamp run is **1.404 ms**.
The earlier capture prototype measured approximately 1.398 ms in another run.
Uploads, initialization and readback are outside its timestamp interval.
Neither was interleaved with L; neither is a time L should be expected to
reach. Trace durations are not used for benchmark timings.

Build the probe with `bash scripts/build_fsr_upscale_probe.sh`, or the MinGW
`fsr_upscale_probe` CMake target. Supply the signed upscaler DLL, compatible AMD
provider DLL and working D3D12 runtime locally. In an isolated Proton prefix,
with a working display or Xvfb and an existing output directory:

```sh
# Copy executable and local DLLs to a Linux filesystem if Wine cannot map PE
# files from the repository's mounted filesystem. Keep them app-local.
WINEDLLOVERRIDES='d3d12,d3d12core=n,b' PROTON_FSR4_UPGRADE=0 \
  "$PROTON/proton" run "$APP/fsr_upscale_probe.exe" \
  'Z:\path\to\app\amd_fidelityfx_upscaler_dx12.dll' \
  'Z:\path\to\existing-output-directory' 1920 1080 3840 2160 140
```

Set `STEAM_COMPAT_CLIENT_INSTALL_PATH`, `STEAM_COMPAT_DATA_PATH` and `DISPLAY`
for the local installation. The probe writes `probe.log` and `output.rgba16f`,
logs providers and loaded module paths, pins an FSR4 provider and refuses an
analytical FSR2/3 fallback. Inspect ISA to distinguish FP8 from INT8.

For a warmed RGP trace, add `MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1`
and run 42 evaluations; retain the last dispatch submit. This RADV configuration
captures **every submit**, even with a trigger file, so allow disk space and
remove only this run's unneeded traces. Do not benchmark with tracing enabled.
Open the retained trace in RGP, select each event's Instruction timing table,
copy it and read the X clipboard with `tools/rgp_clip.c`; analyze with
`tools/rgp_instruction_mix.py`. Count VOPD pairs consistently on both models.

Local traces, screenshots and table exports remain under
`build/fsr4-reference-20261029/`. Runtime DLLs, models and shader ISA are not
included in the published summary data.
