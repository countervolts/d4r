# Preset L: architectural workaround experiments

Measured on 2026-10-29, RX 9070 XT (`gfx1201`), RADV, default 330 W cap. These
experiments follow [the vector-issue analysis](amd-vector-issue-limit.md). They
did **not** find a large improvement or establish NVIDIA parity. All new kernel
options remain experimental; none changes the default build.
Rebuilding with every new option disabled preserves the full assembly of all
four outer-stage shaders and the GPU `.text` sections of all five native layers.

## Explicit dual issue

[bench_l_gelu.hip](../tools/bench_l_gelu.hip) compares the native Swin GELU's
compiler-generated code with scalar assembly and explicitly paired assembly.
[gelu_dual.h](../kernels/m/gelu_dual.h) fixes operand registers so the VOPD
register-bank constraints are satisfied. The constants are the native kernel's
half-rounded coefficients. This is an isolated activation benchmark, not an L
evaluation or a port of L's Vulkan outer stages.

| Implementation | Median kernel time |
|---|---:|
| Compiler | 1.418 ms |
| Scalar assembly | 2.184 ms |
| Explicit dual assembly | 1.582 ms |

Seven rotated/reversed trials, ten launches per timed batch, after 100 warmup
rounds of all three kernels. Each launch evaluates eight independent channels
per thread for 4096 dependent iterations. The timing loop adds 0.25 between
iterations to avoid decaying into subnormals. The single-iteration correctness
check omits that offset and compares 1,048,576 values, including every binary16
encoding and pseudorandom binary32 patterns. Both assembly versions match all
non-NaN result bits; NaN payloads are not compared.

The paired instructions are real and beat the scalar assembly by 28%, but lose
to optimized compiler output by 12%. The compiler uses one median clamp and an
absolute-value source modifier; the explicit paired sequence needs separate
minimum/maximum and absolute-value instructions. Its register count also rises
from 19 to 26. More dual-issued instructions alone is therefore not evidence of
a faster activation. This result does not rule out a better full-kernel schedule.

## Full L evaluations

Geometry is 1920x1080 to 3840x2160, `vkhip`, with the existing five outer-stage
vector reductions enabled in both members of every pair. Each pair has three
interleaved trials, 48 warmup evaluations and 100 timed evaluations per run.
Numbers are medians of trial medians. Compare within a row: the control was
rerun for each experiment. These are standalone GPU evaluation times, using the
official-probe patterned input and four jitters, not game FPS or image-quality
acceptance tests.

| Experiment | Control / candidate, ms | Result |
|---|---:|---|
| Inner network residual and norm 2 in f32 (`SWIN_RES32`) | 4.024 / 4.033 | No gain; changes arithmetic |
| Inner network raw-buffer weight loads (`SWIN_TOK_BUFFER_WEIGHTS`) | 3.991 / 3.968 | 0.56% lower whole time; 1.0% lower network time |
| Same loads without hiding the base from LLVM (`SWIN_TOK_BUFFER_DIRECT`, with the previous flag) | 3.990 / 3.977 | 0.33% lower whole time |
| One window per wave in the optimized outer stages (`WPW=1`) | 4.031 / 4.159 | 3.16% slower |
| Compact, XOR-swizzled decoder head buffer (`M_L_HEAD_COMPACT`) | 4.043 / 4.058 | 0.37% slower |
| Explicitly uniform outer-stage wave index (`M_SCALAR_WAVE`) | 3.996 / 3.988 | 0.19% lower time; too small to promote |

The weight-load option reads exactly the same prepared bytes with a scalar
buffer resource and separate lane/tile offsets. Its register pressure increases
in several layers, despite the small measured network improvement. It remains
opt-in rather than being selected for all gfx12 GPUs from one short benchmark.

The f32 residual experiment removes conversions and reduces static VALU counts
by about 3-6%, but increases f32 norm work. Whole-network time is slightly worse.
Its final RGB frames are 53.3-58.4 dB against the relaxed control (unit peak),
with no non-finite RGB values on these four frames. This is not a comparison
against a reference render and does not establish quality acceptance.

Every other candidate above preserves all four final-frame SHA-256 hashes in
all three trials. That coverage establishes equality on this input sequence,
not on arbitrary games, motion, exposure changes or lifecycle events.

## RGP follow-up

RGP 2.7.1 was run under Xvfb `:78` on a fresh capture of the optimized control.
The trace includes instrumentation overhead; its durations are not used in the
benchmark table.

| Stage | VGPRs | Allocated LDS | Waves/SIMD | VALU busy | VALU / WMMA per sampled wave |
|---|---:|---:|---:|---:|---:|
| `enc0_l` | 192 | 8192 B | 8/16 | 53.7% | 2526 / 160 |
| `dec0_l` | 216 | 9216 B | 7/16 | 54.6% | 2134 / 144 |

RGP identified a decoder occupancy threshold requiring 24 fewer registers and
1024 fewer LDS bytes. That motivated the compact-head experiment: its head
buffer drops from 64x40 to 64x32 binary16 values, with matching swizzled writes
and reads. The compiled decoder does drop from 216 to 192 registers, but its
plane-layout VALU count rises from 2137 to 2239. The layout remains correct,
but its measured decoder time regresses. The uniform-wave-index variant cuts
the two outer stages to 2497 and 2073 VALU instructions, without a convincing
whole-evaluation gain in this run.
Occupancy and static instruction counts must be judged against actual timings.

The remaining work is still largely vector work. These tests provide no basis
for promising that a plain HIP port, more VOPD pairs, or more occupancy will
close the approximately 4.0 versus 3.16 ms gap. They also do not prove a hard
lower bound for a different implementation.

## Reproduction and evidence

The larger [AMD decoder redesign](l-amd-redesign.md) subsequently achieved a
9.57% decoder / 1.57% whole-pass improvement with identical tested outputs.
The [FSR4 instruction comparison](fsr4-l-instruction-comparison.md) adds an
actual FP8 reference and distinguishes L's inner tube from its outer blocks.

Machine-readable trial results, hashes and instruction counts are in
[sr-l-architecture-workarounds.json](benchmarks/gpu-limits-20261029/sr-l-architecture-workarounds.json).
Full logs, model variants, assembly, RGP table exports, screenshots and the paired
runner script are retained locally under `build/l-architecture-20261029/`.
Weights, models and trace binaries are not published.

```sh
hipcc --offload-arch=gfx1201 -O3 tools/bench_l_gelu.hip -o /tmp/bench_l_gelu
/tmp/bench_l_gelu 32

# Build each native-network experiment into its own directory, then compile
# a preset-L model with compile_m.py CAPTURE MODEL --hip LAYERS.
D4R_VKHIP_LAYERS=1 D4R_GPU_ARCH=gfx1201 D4R_NATIVE_FP8=1 \
  D4R_SWIN_CFLAGS=-DSWIN_TOK_BUFFER_WEIGHTS \
  bash kernels/build.sh m /tmp/layers-buffer
# Substitute -DSWIN_RES32 for the residual experiment, or add
# -DSWIN_TOK_BUFFER_DIRECT to the buffer flag for its second form.

# Rebuild only the outer stages in a current model (including its f32 biases).
# Substitute -DM_SCALAR_WAVE for the uniform-index experiment.
python3 tools/build_l_stage_variant.py CURRENT_MODEL COMPACT_MODEL \
  -DM_F32ACT -DM_RES32 -DM_PK16 -DM_BIAS32 -DM_LDAU -DM_L_HEAD_COMPACT
# For WPW=1, set LVAR_BASE='-DM_RELAXED -DWPW=1 -DM_RESIDUAL_F32'
# and omit the compact-head define.
```

Use `D4R_M_WARMUP_PASSES=12 D4R_M_TIME_EACH=1`, 37 passes and four input
frames for 100 timed evaluations, as in the existing local comparison. Dump
outer-stage ISA with `RADV_DEBUG=shaders` on the Vulkan route: debug dumping
disables the cache objects needed by the vkhip native donor loader. Model
initialization compiles both outer-stage plane layouts on that route too.
