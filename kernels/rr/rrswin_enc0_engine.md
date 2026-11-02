# `cuda_dldn_engine_swin_enc0_kernel` on the staged-image engine

This file is the implementation record for enc0's port from the mirror-cut scalar body to the
staged-image engine (`kernels/rr/rrswin_engine.h`).  §1–§5 are the design contract the code
implements; §6 is the evidence table that accrues as each offset and layout claim is
validated (per-phase oracle captures, engine unit tests, output-surface compares).

Status: **design fixed; primitives and body port dispatched.**  Gates and their results are
recorded at the bottom as they are measured.

## 1. What the port changes

The mirror-cut body `d4r_swin_enc0` (`kernels/rr/rrswin_enc0.hip`) keeps

* its cut ABI (called from NVIDIA's PTX between the prologue and E22; arguments
  `a0..a23` = the staged block's 24 A-fragment words, `b0..b3` = phase 1's C seeds,
  `w` = arg040 (the 54560-byte prepared weight image), `prm` = the 1032-byte parameter
  block, `scratch` = the cut's `input_tensor`, 3200 bytes of shared memory),
* every epilogue (E5/E10 softmax, E6/E12 norms and table adds, E13/15/17/19 activations,
  E20 arena scatter + lane merge, E21 surface scatter, E22 handover),
* the score/P·V phases 5, 6, 10, 11 (`score_gemm`, `pv_gemm`), phase 21 (`band21`) and
  phase 22's GEMM as they are (scalar f32 dots over exact e4m3 decodes, 136 of 536 mma),

and replaces the sixteen plain GEMM phases **1, 2, 3, 4, 7, 8, 9, 12, 13, 14, 15, 16, 17,
18, 19, 20** (400 of 536 mma) with engine GEMMs: operands read from staged e4m3 images and
prepared slots, one `k32_e4m3` per (m, n) tile, accumulators rounded to f16 after each k32
step — the conservative chain that is bit-identical to the translated kernel's
`mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` (see §2 of docs/rr-engine.md and
`rrswin_engine_k16_test.cpp` step 2).

Replaced with this port: the scalar `gemm()` calls for those sixteen phases, whose per-mma
cost is 64 byte loads from the weight image, 64 e4m3 decodes, 128 f32 FMAs and twelve DPP
permutes per lane.

## 2. Orientation and the two identities that make the port mechanical

Measured facts (all previously validated, not re-derived here):

* **D layout** (`kernels/tools/rrswin_operand_fit.cpp`): engine tile D — lane `l`, VGPR `i`
  holds row `i + 8h`, column `l & 15`, `h = l >> 4`.  With the engine's operand roles
  (`swin_gemm_tile`: weights = A via `kop_image`, activations = B via `swin_operand`) the
  tile's D is `[weight column wr][token tk]` — NVIDIA's D transposed:
  `D[wr = i + 8h + 16*mt][tk = (l&15) + 16*nt]` for tile `(mt, nt)`.
* **Staging D into the next phase's A image** is exactly the existing
  `swin_stage_operand(img, stride, 16*nt, mt, acc)`: it writes
  `img[(16*nt + l&15)*stride + 16*mt + 8h + i] = e4m3(acc[i])`, i.e. image row = token, image
  k = weight column — the A-operand form the next phase's `swin_operand` reads.
* **The residual identity**: the next phase's C operand is a cell-for-cell copy of the
  producer's D.  NVIDIA's mma computes `D = A·B + C` where `C[m][n]` sits at `(token, weight
  column)`; the engine tile's D cell `(wr, tk)` is `C[tk][wr]`.  So phase P+1's engine tile
  seed is `c[i] = f16(P's own tile (mt, nt) at VGPR i)` — the producer's accumulator compacted
  to f16 pairs, no exchange, no shuffle.  This is why phases 12, 16, 18, 20 (residual =
  previous phase's D) and 7 and 14 (residual + per-column table) need no C plumbing.
* **Per-column biases and tables are column-major f16**: every read of the form
  `*(u32*)(w + base + 16*nt + ((laneid<<2)&12))` (E9's seeds, `wbias_regs`, E6/E13's tables)
  places column `c` at byte `2*c` of the blob.  Evidence: the word for `(nt, lane)` covers
  columns `8nt+2t, 8nt+2t+1` at bytes `16nt+4t .. +3 = 2*(8nt+2t) .. +3`.  The engine tile
  seed for a per-column C is therefore `c[i] = f32(f16 at w + 2*(i + 8h + 16*mt))`.

## 3. The dcell exchange (the one cross-lane primitive the port adds)

Epilogues that consume the NVIDIA `d[mi][rh][nt]` view (`norm_as` for phases 2 and 12; the
scalar `score_gemm`/`pv_gemm` for phases 3, 4, 8, 9; E20's `r20` build for phase 20) get it
from the engine tiles through one shared-memory exchange.

* Writer (per tile `(mt, nt)`, lane `l`, its own four f16 pairs
  `pairs[p] = f16(D[2p]) | f16(D[2p+1]) << 16`, `p = 0..3`):
  `lds[(mt*n_nt + nt)*128 + lane*4 + p] = pairs[(mt*n_nt + nt)*4 + p]` — all of it in
  **u32 units: 128 words (= 512 bytes) per tile**, 32 lanes × 4 words.
* Reader: `d[mi][rh][nt] = lds[((nt>>1)*n_nt + mi)*128 + (8rh + g + 16*(nt&1))*4 + t]`
  with `g = laneid>>2`, `t = laneid&3` — same units.

Derivation (checkable against the two measured facts above): the wanted cell is
`D[wr = 8nt + 2t + cb][tk = 16mi + 8rh + g]`; its engine location is tile
`(mt_e = nt>>1, nt_e = mi)`, lane `(tk%16) + 16*((wr%16)>>3) = 8rh + g + 16*(nt&1)`,
VGPR `wr%8 = 2t + cb` — i.e. the reader's own pair index `t`, both halves.  A full
twelve-tile exchange is 12 × 512 B = 6144 B and an eight-tile one 4096 B — both exceed
the scratch at exchange time, so **every exchange runs in 2048-byte chunks** (four tiles
at a time) overlaying the just-dead image region, with the chunk loop's barriers owned by
the phase code (§5).

## 4. New primitives (kernels/rr/rrswin_engine.h)

All are thin, checked by `kernels/tools/rrswin_enc0_engine_test.cpp` against independent
host models before the body relies on them.

| primitive | signature | semantics |
|---|---|---|
| `swin_slots_from_wimg` | `(uint8_t* slots, const uint8_t* w, int base, int n_rows)` | One phase's B fragments (8 u32 per lane at `w + base + 512*t + 16*lane + 4*word`, `t = 0..n_rows/16-1`, `word = 0..3`) scattered into slot layout: slot `(n_rows*(k>>4) + n)*16 + (k&15)`, with `(n, k)` from the verified map `n = 16t + (lane>>2) + 8*(word>>1)`, `k = 2t' + (j&1) + 16*(j>>1) + 8*(word&1)`.  Matches `swin_gemm_tile`'s read with `slot_base = 16*mt`, `slot_step = n_rows`. |
| `swin_dcell_exchange` | writer `(lds_u32*, const uint32_t* pairs, int n_mt, int n_nt)` / reader `(const lds_u32*, uint32_t* d, int n_mi, int n_nt4, ...)` | §3.  Writer and reader are separate calls; the wave barrier between them belongs to the caller. |
| `swin_stage_operand_p` | `(uint8_t* img, int stride, int row_base, int chunk, const uint32_t pairs[4], int mode)` | `swin_stage_operand` from a tile's four f16 pairs with a per-pair transform before `e4m3x2`: mode 0 plain, 1 `relu_pair` (ZLUDA's `relu_keep_nan_halves`), 2 `act_pair` (the clamped cubic).  Byte `i` of the image row holds `e4m3(transform(pair[i/2]) half i%2)`. |

`swin_stage_fragment` needs no row-offset variant: the second half of a split 96-row GEMM
restages its fragment tile group at *local* rows 0–47 (`swin_stage_fragment(img, 32,
frag + 12, 3)`) and runs its GEMM tiles against local `row_base = 0`, emitting results to
the tiles' *global* indices (nt 3–5).  The D output of every split phase is registers
(pairs), so no image write needs the global row position.

The relu and act transforms are per-element, so `e4m3x2(transform(pair))` equals the
translated kernel's `transform`-then-`cvt.rn.satfinite.e4m3x2.f16x2` element for element.

## 5. Shared-memory map (3200 bytes, the cut's `input_tensor`)

`input_noise` (800 B) is a separate `.shared` variable whose offset relative to
`input_tensor` the linked body cannot know, so the budget is the 3200-byte scratch only.

| region | size | live during |
|---|---|---|
| image buffer | 1536 (48 rows × 32 B) or 2048 (64 rows × 32 B) | one phase's A image, split into halves for the 96-row phases |
| slots | 1024 (64 slots × 16 B; 512 for phase 22, unused — phase 22 stays scalar) | one phase's GEMM |
| dcell exchange | 128 × (tile count) | between a phase's GEMM and its dcell-consuming epilogue |

The 96-row phases (1, 2, 3, 4, 8, 9) cannot hold a 3072-byte image and 1024 bytes of slots
at once, so their GEMM runs in two token-tile halves (nt 0–2, nt 3–5): stage rows 0–47,
run, restage rows 48–95, run.  The halves' staging sources are registers (the cut's `as1`,
the normed tile `as3`), so restaging is cheap.  Phases 7 and 12–20 have 64-row images
(2048 B) and fit without splitting.  Every exchange overlays the just-dead image or slot
region of the same phase.

## 6. Per-phase implementation table

| phase | mma | A (engine B operand) | B (engine A operand) | C seed | D consumed as |
|---|---|---|---|---|---|
| 1 | 24 | `as1` fragments → image halves | slots from `w+4096` | per-column, `w+5120` | relu-staged into phase 2's image (mode 1), no dcell |
| 2 | 24 | relu-staged image of D1 | slots from `w+5184` | per-column, `w+6208` | dcell exchange → `norm_as` (gain `w+6272`, eps 2⁻¹³·2⁻⁴? see body) → `as3`; pairs kept for phase 7's C |
| 3 | 24 | `as3` halves | slots from `w+6336` | — | dcell exchange → `d3` |
| 4 | 24 | `as3` halves | slots from `w+7360` | — | dcell exchange → `d4` |
| 5–6 | 48+48 | scalar `score_gemm`/`pv_gemm` unchanged (`d3`, `d4`, bias `w+8384`) | | | |
| 7 | 16 | `as7` (from `d6`) → image | slots from `w+20672` | `d2` pairs + table `w+37056` | pairs kept (phase 12's C) |
| 8 | 24 | `as3` halves | slots from `w+21696` | — | dcell exchange → `d8` |
| 9 | 24 | `as3` halves | slots from `w+22720` | — | dcell exchange → `d9` |
| 10–11 | 48+48 | scalar `score_gemm`/`pv_gemm` unchanged (`d8`, `d9`, bias `w+23744`) | | | |
| 12 | 16 | `as12` (from `d11`) → image | slots from `w+36032` | `d7` pairs | pairs kept (phase 14's C); dcell exchange → `norm_as` (gain `w+37120`, eps 2⁻¹³) → `asN` |
| 13 | 16 | `asN` → image | slots from `w+37184` | per-column, `w+38208` | act-staged into phase 14's image (mode 2) |
| 14 | 16 | act-staged image of D13 | slots from `w+38272` | `d12` pairs + table `w+39296` | pairs kept (phase 16's C) |
| 15 | 16 | `asN` → image | slots from `w+39360` | per-column, `w+40384` | act-staged into phase 16's image |
| 16 | 16 | act-staged image of D15 | slots from `w+40448` | `d14` pairs | pairs kept (phase 18's C) |
| 17 | 16 | `asN` → image | slots from `w+41472` | per-column, `w+42496` | act-staged into phase 18's image |
| 18 | 16 | act-staged image of D17 | slots from `w+42560` | `d16` pairs | pairs kept (phase 20's C) |
| 19 | 16 | `asN` → image | slots from `w+43584` | per-column, `w+44608` | act-staged into phase 20's image |
| 20 | 16 | act-staged image of D19 | slots from `w+44672` | `d18` pairs | dcell exchange → E20's `r20` build (unchanged) |
| 21 | 32 | scalar `band21` unchanged (merged `m21`, weights `w+45696`, bias `w+53888`) | | | |
| 22 | 8 | scalar `gemm(..., nnt=2)` unchanged (`r20`, `w+54016`, bias `w+54528`) | | | |

Weight-image byte ranges are the body's own verified table (`rrswin_enc0.hip`'s
"Weight image layout of the 22 phases" comment); `waddr`'s formula is verified on all 1024
`(n, k)` pairs of a tile.

## 7. Validation gates and results

*(filled as measured)*

| gate | requirement | result |
|---|---|---|
| translated reference set | 22 oracle stages `p1..p22` captured (`rr/gate-enc0/oracle-p<stage>/dumps`, relative to the capture working directory), 6 launches each | **captured** |
| harness determinism | `compare_launch_dump.py` oracle-p1 vs oracle-p3: 108 args, 6 differ — only arg048, first differing byte exactly at 46,000,000 (the debug payload); all surfaces/textures/weights byte-identical | **proven** |
| performance anchor (translated) | `D4R_HARNESS_FRAME_WAIT_MS=0` (continuous, matching the 16.2 ms config), 30 frames, frames 13–30: **enc0 = 3.879 ms/frame**, total 16.79 — the documented 3.79 ms baseline reproduced within ~2 % | **measured** |
| full-frame reference | `control-80`: 80-frame run of the installed runtime with `D4R_HARNESS_SAVE_FRAMES`, 80 saved frames (`rr/gate-enc0/control-80/out.raw.frameN`) | **captured** |
| primitives | `rrswin_enc0_engine_test.cpp` steps vs independent host models | pending |
| per-phase oracle | engine `D4R_RRSWIN_DEBUG_STAGE=<n>` vs the translated stage, payload-window compare | pending |
| layer surfaces | `compare_launch_dump.py` translated vs engine enc0 output surfaces | pending |
| full frame | 80-frame engine run, frames 21–80 byte-identical to `control-80` | pending |
| performance | enc0 < 3.879 ms/frame in the anchored continuous configuration | pending |
