# RDNA3 M/L tuning and experimental kernels

## Validated gfx1101 fast-set weight loads

The fast builds now use uniform-offset raw-buffer weight loads for enc1, enc2,
enc3 tube and dec1 on RX 7700 XT. Dec2 retains its previous loads. The weight
addresses, bytes, matrix operands, accumulation and quantization are unchanged;
this selects the existing `SWIN_BUFFER_BLOAD` implementation. Other targets and
accuracy sets retain their previous selection. `D4R_SWIN_CFLAGS=-USWIN_BUFFER_BLOAD_FAST`
builds the previous fast-set control for comparisons.

At 1280×720 → 3840×2160, M/HDR/quality value 2, three alternating 150-frame harness
pairs (first 50 discarded) measured median DLSS GPU time **6.187 → 6.107 ms**,
about **1.3%**. All three pairs improved (1.07–1.36%) and their final frames match.
Selected trained-layer replays improved about 2–7%; these are not whole-frame gains.
For this weight-load change alone, L/HDR/quality value 3 had mixed results: a
0.17% regression and a 1.13% improvement. The separate L endpoint rewrite below
provides the substantial L gain.
These are standalone harness results, not game FPS measurements.

Byte comparisons pass 128 temporal frames spanning M/L, moving and jittered inputs,
M HDR/LDR, quality changes, exposure changes and resets. Four feature recreations
per model at alternating render sizes also match. The existing accuracy kernels'
machine-code sections remain identical. Evidence, sources and reproduction scripts
are under `build/ml-bitexact-20261022/`. No game installation was changed.

The shared token-major, content-cache and native M texture candidates below
remain **opt-in experiments, not faster defaults**. The separate L endpoint
rewrite uses a validated portion of that work. No 1.7–2.0 ms M result has been achieved.

## gfx11 token-major Swin

```
D4R_ROCM_DIR=/path/to/rocm D4R_GPU_ARCH=gfx1101 \
D4R_PREFER_ACCURACY=1 D4R_SWIN_TOKEN_GFX11=1 \
kernels/build.sh m /new/output/directory
```

This builds the five shared M/L Swin layers from `swin_tok.h` using gfx11 FP16
WMMA. It does not need or enable `D4R_NATIVE_FP8`. Other architectures retain
their existing paths. The build rejects mixing this option with a fast set.

Weights are expanded into FP16 operands. Token-major activations stay in VGPRs;
Q, V, attention, projection and MLP intermediates use exact E4M3 rounding in
FP16 registers instead of packing and immediately unpacking bytes. The native
gfx11 accumulator's even/odd rows survive successive k32 steps. Half-wave
exchanges reconstruct operands and logical row pairs; prepared weights load
complete gfx11 operands without repeating their broadcast. Patch merge streams
its operands rather than retaining the full input array. Patch expand/merge
still use LDS across windows. The tube's producer/consumer flags are preserved.

`q8x2_exact` includes nearest-even rounding, subnormal handling, finite saturation,
and signed zero. Simple mantissa masking would fail these conditions. NaNs
retain the existing half-quantizer payload policy; they are not covered by the
finite equivalence assertion. Quantization tests cover all 63,490 finite/infinity
half encodings in both packed positions using an independent E4M3 nearest-value
oracle. Nonuniform tests verify physical accumulator ownership separately from
the logical row round trip. Tensor tests cover shifted/unshifted windows, borders,
all five layer shapes, and unchanged input allocations.

The default exchange is `ds_bpermute`. To compare `permlanex16` instead:
`D4R_SWIN_CFLAGS='-UWM_HALF_BPERMUTE -USWIN_HALF_BPERMUTE'`.
`D4R_SWIN_CFLAGS` also permits isolated register/scheduling experiments. Increasing
VGPR caps removes some spills but can lose occupancy; it is not inherently faster.

## Content-checked enc3 preparation

`D4R_ENC3_CONTENT_CACHE=1 kernels/build.sh m OUTPUT` retains the prep launch and
compares each weight tile's **complete 16 source bytes** with a GPU snapshot.
Only unchanged tiles skip expansion and prepared-image stores. A recycled pointer
or changed byte cannot authorize a stale tile. This is an exact comparison, not
a hash. Per-channel constants in the token implementation are refreshed normally.
Zero-initialized source snapshots and prepared images both represent zero weights.
The existing stream-ordered preparation/consumption contract still applies.

ZLUDA's address-only preparation skip remains disabled for enc3. MicroCUDA already
supports allocation generation plus write epoch reuse; this change additionally
ensures failed prep submissions never publish a new valid identity. The content
option has not demonstrated a performance benefit and remains disabled.

`swin_prep_test.cpp` compares every prepared image, constant and slot key against
an uncached module after mutation at the same address, repeated content, zero and
NaN codes, more keys than available slots, and allocation recycling. The host-only
`tests/microcuda_prep_identity_test.cpp` exercises failed initial and updated prep
submissions and subsequent successful reuse.

## Native M/L texture experiments

```
D4R_DLSS_DLL=/path/to/nvngx_dlss.dll D4R_ROCM_DIR=/path/to/rocm \
scripts/build_m_native_textures.sh /new/output/directory [KERNEL]
```

The isolated builder translates 25 candidates from the user's DLL: four folded M
enc0 variants, folded M dec0, sixteen folded/unfolded post variants, and four
downsample variants. Generated NVIDIA sources remain temporary. A manifest gates
the candidates by module identity. Nonidentical duplicate definitions are refused.
The builder rejects accuracy-set use pending broader temporal validation.

Floating-coordinate `tex.base` uses the sampler at level zero. Integer-coordinate
fetches retain their existing path. Optional operands not implemented by this
translation are rejected. `--max-vgpr=N` exposes a checked opt-in compiler budget;
`D4R_M_TEXTURE_MAX_VGPR` passes it through the builder. Post defaults to wave64,
downsample and matrix tails to wave32. The folded M encoder splice validates the
neural tail against the reference up to a consistent register renaming. Each
variant exports its own correctly named prep kernel. L's unfolded enc0/dec0 are
not substituted with M's folded tails.

The four downsample names are static/dynamic × HDR/LDR **alternatives**, not four
successive encoder stages. Runtime inspection shows downsampling after output
reconstruction; its result participates in temporal history. Fusing it into the
earlier current-frame encoder would violate that dependency. Combining post and
downsample would require a halo/recomputation design; removing the dispatch without
that design is incorrect. No unsafe fusion was enabled.

## Evidence and limits (2026-10-22)

Hardware: RX 7700 XT/gfx1101. Compiler/runtime: cached ROCm 7.2 toolchain.
Detailed private results: `build/ml-rdna3-20261022/RESULTS.md` and adjacent logs.
All 25 texture candidates compile. The token path compiles for gfx1100/gfx1101;
the existing gfx1201 native-FP8 set also compiles. Other GPUs have no runtime test.

Ten shifted/unshifted synthetic cases and ten freshly captured trained-M launches
(enc1, enc2, six tube launches, dec2, dec1) match all allocation bytes against the
native **accuracy** control. Captured inputs came from the fast M pipeline; this
is an isolated layer comparison, not an end-to-end exact-M claim. Small and larger
synthetic shapes were tested separately. The quantization/layout GPU test passes,
as do 40 content-preparation cases and the host cache-failure regression.

The trained capture used 640×360 → 1920×1080, preset M, HDR, render-resolution motion.
Representative isolated medians (40 launches, prep excluded), ms:

| Layer | Existing accuracy kernel | Token candidate |
|---|---:|---:|
| enc1 | 0.274 | 0.322 |
| enc2 | 0.178 | 0.290 |
| tube, first launch | 0.079 | 0.122 |
| dec2 | 0.154 | 0.211 |
| dec1 | 0.269 | 0.296 |

These are sequential diagnostic samples, not a paired end-to-end benchmark.
The 96-channel token kernels still spill. Register exchanges, duplicated operands,
and occupancy outweigh the removed LDS traffic in the tested implementation.

Native M enc0, post and static-HDR downsample each preserve all eight saved harness
frames at 640×360 → 1920×1080. Their separate 80-frame, 1280×720 → 3840×2160 runs
also match the final raw frame. Sequential profile samples regress for each
candidate, so none is promoted. Other native M texture flags remain unvalidated. This experimental M texture
work establishes no image-quality acceptance, game performance or speedup.
L endpoint validation is documented separately below.

The older 3.99 → 1.37 ms Swin and 2.77/2.38 ms whole-evaluation figures in
`native-kernels.md` belong to **RX 9070 XT/gfx1201**, not RDNA3.

## gfx1101 four-wave L encoder and decoder

Fast gfx1101 builds select `D4R_L_GFX11=auto`: native unfolded L enc0/dec0
with one wave per 4×4 window. `D4R_L_GFX11=0` restores the original texture
neural bodies. `D4R_L_GFX11=1` explicitly selects this path on other gfx11 fast
targets; only gfx1101 has runtime validation. Accuracy sets keep their existing
path. Build a complete L set with `kernels/build.sh l OUTPUT` and the usual DLL,
ROCm and offline-emitter variables. No native FP8 setting is required.

Prepared weights contain full scaled FP16 operands. Permuting A rows to match
native gfx11 accumulator ownership removes per-MMA accumulator exchanges;
intermediates stay in registers with the original FP8 quantization and k32
rounding boundaries. Half-wave exchange uses `ds_bpermute`. The original texture
feature head runs on wave zero; a workgroup barrier precedes the four native
windows. Decoder windows publish results before wave zero enters the original
surface epilogue. Post and downsample arithmetic are unchanged.

Two details are necessary for exactness. Texture QK operands retain the original
interleaved low/high-16 pair order, including the split into two k16 operations;
a contiguous split changes rare FP16 ties and temporal history. L's specialized
`q8x2_exact<0x7f80>` canonicalizes NaNs inside the quantizer before operand scaling.
The default quantizer preserves its previous payload behavior and compiled shared
M machine code. Both L prep identity/key exports are zero: weights are refreshed
on both runtimes, including recycled NGX addresses. Address-only caching is unsafe.

Version boundaries matter: `l-safe` preceded complete NaN handling; `final-accepted`
was v2 with a more expensive explicit operand canonicalization; v3 moves that
operation into quantization. Their timings must not be interchanged. Private v3
scripts and logs are in `build/ml-more-20261022/continuation-v3/`.

The selected v3 set passes 128 byte-identical temporal frames across eight cases
and eight recreation/preset-switch cycles (all 34 saved outputs). Fresh exhaustive
GPU boundary tests and 29 focused unit tests pass. All five shared M GPU code
sections remain identical. gfx12 native-FP8 and gfx11 accuracy/token builds pass
compile checks with unchanged GPU code; gfx12 has no new runtime validation.

Three alternating full-set 150-frame pairs at 1280×720 → 3840×2160,
L preset 12 / quality value 3, discard first 50, measured:

| Pair | Control DLSS GPU ms | v3 DLSS GPU ms | Reduction |
|---|---:|---:|---:|
| 1 | 13.7865 | 9.7875 | 29.01% |
| 2 | 13.9420 | 9.7875 | 29.80% |
| 3 | 13.7540 | 9.7685 | 28.98% |

All six final raw-frame hashes match. This is about 29% less whole-evaluation GPU
time in the standalone harness. It does not meet the ≤4 ms target or establish
game performance or RTX parity. M's previous approximately 6.1 ms result uses
preset 13 / quality value 2 and is not an equal-quality comparison with L.
See `build/ml-more-20261022/RESULTS.md`, `l-bench-v3.json`, and adjacent v3
validation files for detailed private evidence and reproduction.

## Further M/L implementation cycles after v3

Sixteen new variants tested compact prepared weights, streamed/in-place window
attention, two- and three-launch phase splits, a separate fallback dispatch,
MLP ping-pong buffers, dense bank-swizzled LDS rows, L two-wave execution,
a split texture head, unscaled arithmetic, buffer operand loads, and two native
reconstruction scheduling/register-budget variants. Most were exact but slower;
the unscaled L path and compact L encoder failed frame equality. Detailed new
sources, correctness hashes and paired diagnostics are preserved in
`build/ml-next-20261022/RESULTS.md` and adjacent experiment directories.

Two changes remain: gfx1101 fast tube MLP scratch uses dense XOR-swizzled 32-half
rows with the original barriers, and the native gfx11 L endpoints load unchanged
prepared operands through buffer instructions with uniform tile offsets. Tube
trained replay improved roughly 11%; that is a single-layer result, not a
whole-evaluation claim. L decoder register use dropped from 148 to 126.
Preparation/invalidation, FP8 boundaries, QK grouping and arithmetic order remain.
Accuracy/token/native-FP8 and other default architecture paths retain their GPU
code. `D4R_L_GFX11=0` retains the original L texture-body fallback.

Three fresh alternating 150-frame pairs per model, discard first 50, at
1280×720 → 3840×2160, measured these medians across runs:

| Model / setting | Starting-tree control ms | New candidate ms | Per-pair reduction |
|---|---:|---:|---:|
| M / preset 13, quality 2 | 6.0735 | 6.0270 | 0.63–1.09% |
| L / preset 12, quality 3 | 9.7315 | 9.6100 | 1.18–1.32% |

L's starting source is the validated v3 path previously measured around 9.78 ms;
the table uses fresh controls, not the old 13.8 ms L implementation. The models
use different quality settings. All paired final hashes match. There are 256
byte-identical temporal frames across sixteen cases and sixteen lifecycle cycles
with all 68 saved outputs exact. Ten shifted/unshifted synthetic cases and ten
trained launches match all allocations. Twenty-two GPU code sections for gfx1100
fast, gfx1201 native-FP8 shared/L, and gfx1101 accuracy/token paths match controls;
only gfx1101 has runtime hardware tests. Twenty-nine focused CPU tests pass.

These gains are modest; ≤4 ms remains unmet. New phase splits amplified global
intermediate traffic, extra LDS/double buffering reduced occupancy, fewer L waves
introduced spills, and reconstruction scheduling traded registers against scratch.
Removing L scaling changed output. No rejected architecture is enabled, and no
RTX parity or game-performance result is established.
