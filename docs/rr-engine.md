# The swin-layer engine: design

Why this exists: a per-phase port of NVIDIA's `cuda_dldn_engine_swin_*` keeps the cost that
makes those kernels slow. Disassembled, the translated `swin_enc0` kernel is 83,373
instructions for 536 FP8 WMMAs. The top of the mix is operand assembly, not arithmetic:

| instruction | count |
| --- | --- |
| `v_cndmask_b32_e64` | 9,124 |
| `ds_bpermute_b32` | 7,976 |
| `v_lshrrev_b32_e32` / `v_pk_lshrrev_b16` | 9,695 |
| `v_and_b32` / `v_perm_b32` / `v_pk_add_u16` / `v_pk_min_u16` | 13,750 |
| `v_cvt_f16_f32` / `v_cvt_f32_f16` (all forms) | 3,568 |
| `v_wmma_f32_16x16x16_fp8_fp8` | 536 |

Those bit operations and cross-lane permutes exist only to reassemble NVIDIA's register
fragments into WMMA operands. A kernel that keeps its activations in a staged e4m3 image and
reads its operands directly from it does not pay them.

## The primitives, already in the tree and already bit-identical to the translated path

`kernels/tex/tex_common.h` (with `kernels/common/wmma_layout.h`), used today by the working
natives `kernels/tex/enc0_tail.hip` and `kernels/tex/dec0_head.hip`:

- `kop_from16(u4v d)` — this half's WMMA operand (8 e4m3 bytes) from 16 e4m3 bytes, in
  NVIDIA's k order. It is a half selection, not a shuffle.
- `kop_image(const kslot* img, int idx)` — the same from a prepared weight image, where a
  `kslot` is the 16 e4m3 bytes of one k16 operand row.
- `kslot_from16(u4v d)` — the weight image's own conversion, done once in a `_prep` kernel,
  so the main kernel never touches the weight layout.
- `k32_e4m3(f8v c, kop a0, kop b0, kop a1, kop b1)` — one m16n8k32 e4m3 step as two gfx12
  FP8 WMMAs. With `D4R_TEX_FP8` it is the native FP8 instruction with no scaling; the
  comment records that its result is bit-identical to ZLUDA's gfx12 lowering of NVIDIA's
  mma, which is what our reference runs through.

The one structural fact that makes this cheap: on gfx12 the accumulator layout *is* the
next GEMM's operand layout (`wmma_layout.h`: "D VGPR i is row i + 8h, which is the gfx12
B-operand layout of the next GEMM (no exchange)"). So an epilogue requantising its
accumulator to e4m3 and staging it as a row-major image in shared memory gives the next
phase its operands with no cross-lane movement at all.

## Shape of a layer

1. `_prep` kernel (once per launch, like `enc0_tail`'s): read the layer's weight image at
   `arg040` and write each phase's B operands as `kslot`s into a global buffer.
2. Main kernel, per token patch:
   - load the layer's input (texture/surface args) and compute the first operand image;
   - for each phase in the descriptor: build A (`kop_from16` from a staged image, or from
     the previous phase's requantised accumulator), run `k32_e4m3` per (m, n) tile with the
     prepared B, apply the epilogue (e4m3 requantise into the next image, clamped cubic
     activation, f16 column-table add, RMS/LayerNorm, softmax, patch expand, plane scatter,
     lane merge, surface scatter);
   - write the layer's output.

## Descriptor

Each layer is a `kernels/rr/descriptor_<layer>.py` with `phases()` (a list of dicts, one per
phase: shapes, A source, B offset, C source, K tiling, epilogue kind and its operands) and
`layer()` (launch geometry, argument offsets, token/tile geometry). Every offset must be a
number with evidence behind it; anything unverified must say so.

## Validation

By output surface against the translated reference: run the layer's native kernel alone
(`D4R_RRSWIN_ONLY=<layer>` in its own native dir) and compare the layer's own output
surfaces with `kernels/tools/compare_launch_dump.py`. Per-phase oracle comparison is a
debugging aid for a phase that is already failing, not the acceptance gate. Note two traps
that cost the agents hours: the debug payload is written into a *window* of a larger block
(the containing argument and the window offset differ per layer, so read the argument the
arena actually lives in, at `arena_offset + base`), and the harness reads its dump after a
context synchronise, so a payload can be masked by later kernels in the frame.

### Current gfx1201 evidence

The checked shared-memory primitive probe (`rrswin_engine_k16_test.cpp`) passes its
GEMM, f16 accumulation rounding, requantization, weight preparation and phase-chain
comparisons. These checks do not establish a complete RR implementation.

The phase-8 pilot now compares all 196,608 f16 results in 64 captured blocks
without a mismatch. Its previous comparison was invalid: a six-token-tile phase
overflowed a four-tile accumulator array, device output had no block stride, and
the host block stride omitted the 32 lanes. Weight preparation now uses per-block
shared memory rather than concurrent writes into one global buffer.

The host phase-8 model also needed all 32 lanes to assemble the activation image,
an 80-register stride for the combined operand/result payload, and f16 output
rounding. With those corrections, the documented fragment and weight maps match
the captured oracle; the previous apparent operand-layout contradiction was not
evidence of a wrong map.

An isolated native phase-8 splice preserves the other NVIDIA phases and produces
a byte-identical final 3840×2160 RGBA16F frame with matching denormal settings.
The initial splice is slower: enc0 measures 14.90 ms versus a 6.59 ms translated
baseline. It is a correctness experiment, not a shipping performance improvement.
The first implementation of patch `0009`'s native FP8 requantization measured
26.65 ms of GPU kernel time at 720p→4K after six warm-up frames. Its same-runtime
software-codec control measured 33.90 ms. Compact integer saturation plus
`0010`'s packed FP8 operand gathers measure 25.33 ms in that 20-frame run;
the complete RGBA16F output remains byte-identical. The matching 80-frame
control measures 24.764 ms after excluding the first 20 frames. Patch `0011`'s
complete-pair shadow-store coalescing and eight-wave register budgeting together
measure 19.146 ms, with all 80 saved frames byte-identical to that control.
Patch `0012`'s signed native conversion reduces the same configuration to
16.234 ms with a byte-identical final frame; its plain and ReLU conversions
match software for every half encoding in both packed positions. Removing
the harness's 200 ms inter-frame idle gap measures 15.144 ms in steady state,
also with the identical final image.
The older installed-runtime FTZ output differs and is not a valid arithmetic
control. Cyberpunk with `0009` + `0010` rendered a loaded 4K save with
720p RR input, coherent moving views and approximately 26.0 ms overlay upscaler
time. With `0011` + `0012` the harness measures 16.234 ms (200 ms idle gap
between frames) and 15.144 ms continuous, with byte-identical final frames.

## Next direct-converter win: B operands from transposed accumulators

Validated mapping (read-only IR investigation, not yet implemented) for gfx12
native-FP8 MMAs whose A operand is a requantization of an earlier accumulator
pair — the encoder's relu and plain chains in
`/tmp/d4r-packed-enc0-analysis/linked.ll`:

* Producer: an MMA pair produces four `<2 x i32>` transposed accumulators
  `(q=0/1, p=1/2)`; today each is converted with `d4r_wmma_t_to_nv` (4 bpermutes),
  stored to registers, requantized by `d4r_cvt_rn_satfinite[_relu]_e4m3x2_f16x2`,
  repacked into four words (`%11374..%11377` for the relu chain), loaded again by
  every consumer and converted by `d4r_wmma12f8_b_from_a` (8 bpermutes per
  consumer). ≥60 consumer sites in enc0, ≥3 fragment groups.
* The composition `t_to_nv → cvt → pack → b_from_a` collapses algebraically: with
  `h = lane/16`, `r = lane&7`, `t = 2h`, the WMMA B word `j` of a lane is
  `pack(cvt(f16x2{lo16(gather(A0, T_j)), lo16(gather(A1, T_j))}), ...)`, where
  `A0/A1 = (r, r+8)` and register `.x` for lanes < 16, `(r+8, l)` and `.y`
  otherwise. One helper per producer fragment —
  `d4r_wmma12f8_b_from_t(uint2 d00, uint2 d01, uint2 d10, uint2 d11)` — replaces
  32 producer bpermutes + the alloca round trip + 8 bpermutes per consumer with
  8 bpermutes total, all consumers CSE to one call.
* Slot order is chain-specific (out0/out1 ← q=0; out2/out3 ← q=1; low/high 16-bit
  half ← p=1/p=2); a matcher must read it off the actual insertelement/bitcast
  structure, and reject fragments with mixed cvt variants or partial updates
  (F32-shadow path needs its own derivation and is not covered).
* The same investigation found no sites where packed-f16 arithmetic could
  commute across `toNv` (the inserted values come from global loads, not shadow
  conversions), so that rewrite is not worth building.

Full derivation with anchors is in the session transcript
(`agent://Fp8ShadowOperandFusion`); the probe extension to gate it (producer →
requantize → consumer chain in `rrswin_fp8_unpaired.ptx`) is described there.


## How a layer feeds the engine (design finding, not yet implemented)

The engine's phases read their activations from a *staged image* — rows of consecutive e4m3 k —
not from NVIDIA's mma fragments. Two ways to get that image exist, and the cheap one is not the
obvious one:

1. Rewrite the layer's prologue natively (load the input planes, the norm, the patch geometry)
   and have it stage the image directly. Correct, but it means re-deriving the layer's whole
   input path, which is where most of the per-layer reverse engineering sits.
2. Cut NVIDIA's PTX *after* its own loading and norming, exactly as the working bodies do today,
   and convert the fragments it hands over into the image **once per phase boundary** instead of
   per mma. Each lane's fragment bytes are documented (row and k per byte, in the body's header);
   writing them to shared memory at their (row, k) positions is a shared-memory scatter, which is
   cheap, whereas ZLUDA's per-mma reassembly is what the 7,976 `ds_bpermute_b32` and 9,124
   `v_cndmask_b32_e64` in the translated kernel are paying for.

Route 2 reuses the cut machinery and the ABI the existing bodies already have, and it is
the reason the engine does not need a prologue rewrite to be useful. What it does need, per layer,
is: the fragment-to-image staging at the cut boundary, the descriptor (offset/layout per phase),
the epilogue vocabulary (the engine has the clamped cubic activation; norm, softmax, column-table
add, patch expand, plane scatter, lane merge and surface scatter are still to come), and the
image-to-fragment conversion back at the exit, where the surviving PTX expects specific registers.

## Rejected direct-converter route

The removed `swin_gemm_zluda` experiment passed widened NVIDIA A/B fragments
straight into ZLUDA's native MMA helpers. Those types matched the helper ABI,
but the data did not: ZLUDA first forms the WMMA A operand from NVIDIA B through
`t_a_from_b`, and the WMMA B operand from NVIDIA A through `t_b_from_a`.
Skipping those cross-lane gathers is not bit-exact by construction.

The staged-image route uses independently checked fragment maps and the existing
`kop_image` / `swin_operand` / `k32_e4m3` primitives instead. The isolated phase-8
recipe in `swin_enc0_recipe.py` preserves the surrounding NVIDIA code. It stages
one token tile and the already-prefetched B fragments in shared memory, computes
the two weight tiles, and hands the rounded result back through lane-major shared
slots. Enc0's 3200-byte `input_tensor` is dead after prologue line 3938 and can be
reused; the final surface epilogue still needs the separate `input_noise` array.

The generic helper model and phase-8 oracle comparison establish these mappings,
not the unvalidated full-layer descriptors or the scalar full-body experiments.
