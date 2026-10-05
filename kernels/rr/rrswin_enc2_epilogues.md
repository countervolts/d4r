# `cuda_dldn_engine_swin_enc2_kernel` — the epilogues of phases 1…38

Companion to `kernels/rr/rr_layer_spec.py enc2`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0015-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_enc2_kernel` (14199 statements, file lines 1018–34562).

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at their first line).  **`LNNNNN`
  is the physical line** of the same instruction in the corpus file, so every
  claim can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and the
  next phase's first `mma`.  The phase table (`rr_layer_spec.py enc2`) gives the
  ranges; this file names what each one *computes*.
* `W` denotes the prepared weight image: `ld.param.u64 %rd2,
  [cuda_dldn_engine_swin_enc2_kernel_param_0+40]` (s8), `cvta.to.global.u64 %rd3,
  %rd2` (s9).  Every `ld.weak.global.ca.v4.u32` in an epilogue reads
  `W + 16*laneid + imm`; every `ld.global.v2.u16` / `ld.global.u32` table load
  reads `W + ((laneid<<2)&12) + imm`.
* `Wz` denotes the **per-`%tid.z` weight slab**: `%rd269 = %rd2 + 2*(%tid.z *
  10752) + 16*laneid` (s4589–s4590, s4613–s4615) = `W + 21504*%tid.z +
  16*laneid`.  Phases 1…7 add offsets to *this* pointer; phases 9…36 add them to
  `W + 16*laneid` directly (s8976–s8978); phase 37 to `W + 12288*%tid.z +
  16*laneid` (s13634–s13639).  Every byte offset in the phase table is relative
  to the pointer named here, so the phase-table numbers do **not** include the
  `%tid.z` term.  See §6 U5.
* `S` denotes the plane arena: `ld.param.u64 %rd383, [%rd1+-32]` (s12865) with
  `%rd1 = param_0 + 80` (s4) → **`param_0+48`**, then `cvta.to.global.u64 %rd5,
  %rd383` (s12866).  `I` denotes the input image: `ld.param.u64 %rd10,
  [param_0+8]` (s17) → `%rd4` (s18).  `O` denotes the epilogue-37 destination:
  `ld.param.u64 %rd6, [%rd492+56]` (s13632) → **`param_0+56`**.
* The mma fragment layouts and the weight-addressing formula are the settled ones
  (see `rr_layer_spec.py --help` and the enc0 companion): C/D `reg0 = (row g,
  cols 2t,2t+1)`, `reg1 = (row g+8, same cols)`, `g = laneid>>2`, `t = laneid&3`.
* **Geometry.**  Writing `m = M/16`, `n = N/8`, `k = K/32` for a phase, the mma
  chain fixes the three exactly: `|A fragments| = m*k`, `|B fragments| = n*k`,
  accumulator chains (maximal C←D dataflow runs) `= m*n`, each of length `k`, and
  `mma = m*n*k`.  `rr_layer_spec.py` prints `M = 16*|A fragments|` and
  `N = 8*|B fragments|`, i.e. it omits the division by `k`, so its M and N
  columns are too large by a factor `K/32` whenever a phase makes more than one
  k32 step.  Every row is re-derived from the chain in §2.
* Anything I could not pin down from the PTX is in §6 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `W` (weight image) | `ld.param.u64 %rd2, [param_0+40]` / `cvta.to.global.u64 %rd3, %rd2` | s8 / s9 |
| `Wz = W + 21504*%tid.z + 16*laneid` | `mul.lo.s32 %r16116, %r10, 10752` (`%r10 = %tid.z`, s22), `mul.wide.u32 %rd266, %r16116, 2`, `add.s64 %rd268, %rd2, %rd266`, `add.s64 %rd269, %rd268, %rd267` (`%rd267 = 16*laneid`) | s4589–s4590, s4613–s4615 |
| `I` = input image | `ld.param.u64 %rd10, [param_0+8]` / `cvta.to.global.u64 %rd4, %rd10` | s17–s18 |
| `S` = plane arena (`param_0+48`) | `add.s64 %rd1, %rd8, 80` (`%rd8 = param_0`, s1), `ld.param.u64 %rd383, [%rd1+-32]`, `cvta.to.global.u64 %rd5, %rd383` | s4, s12865–s12866 |
| `O` = `param_0+56` | `ld.param.u64 %rd6, [%rd492+56]` (`%rd492 = param_0`, s12803) | s13632 |
| `T` = texture (`param_0+88`) | `ld.param.u64 %rd480, [%rd491+88]` (`%rd491 = param_0`, s13941) | s13941, s14056 |
| surfaces | `ld.param.u64 [%rd491+112]` (s13953), `[+120]` (s14178), `[+128]` (s14179) | s13953, s14178–s14179 |
| extents `(ex, ey)` | `ld.param.v2.u32 {%r322, %r323}, [param_0+0]` | s14 |
| block origins | `sub.s32 %r1, %r317, %r318` (`%r317 = 8*ctaid.x`) and `sub.s32 %r2, %r321, %r319` (`%r321 = 8*ctaid.y`) | s3–s7, s11–s13 |
| patch origin offsets | `ld.param.v2.u16 {%rs126, %rs127}, [param_0+80]` | s5 |
| `%r5 = 8*ex`, `%r6 = ey*(8*ex)` | `shl.b32 %r5, %r322, 3`, `mul.lo.s32 %r6, %r323, %r5` | s15–s16 |
| `g = laneid>>2`, `t = laneid&3` | `shr.u32 %r8, %r315, 2`, `and.b32 %r9, %r315, 3` | s20–s21 |
| `%r182 = %r2 + 2*%tid.z` | `shl.b32 %r181, %r10, 1`, `add.s32 %r182, %r2, %r181` | s9184–s9185 |
| `%r17795 = %tid.z<<5` | `mov.u32 %r17745, %tid.z`, `shl.b32 %r17795, %r17745, 5` | s13135, s13139 |

The three scalar-table forms matter and are used throughout:

* `ld.weak.global.ca.v4.u32` at `ptr + 512*i` reads one **N-tile of a K=32 slice**
  = 16 bytes/lane = 2 B fragments or 2 C pairs (the tool's own comment,
  `rr_layer_spec.py:189–192`).
* `ld.global.v2.u16` at `W + ((laneid<<2)&12) + off` reads 4 bytes/lane, the four
  lanes with equal `laneid&3` reading the same word, so a run of such loads at
  16-byte stride is one **per-column** vector (a 96-element f16 table is 12 such
  loads: 12 × 4 bytes × 4 distinct lane offsets = 192 bytes).
* `ld.global.u32` at the same addressing reads one word = 2 f16 = one per-column
  bias pair, used **in both halves** of an mma C operand, i.e. replicated across
  the 16 rows of the tile (s9820 `{%r9477, %r9477}`, s13871 `{%r17465, %r17465}`).

### 1.2 Constants that survive between epilogues

| register | first definition | f32/f64 literal | value used (after the `cvt`) | read by |
|---|---|---|---|---|
| `%f476` | s5020 `mov.f32 %f476, 0f3C8CCB50` | 0.017186790704727173 | f16 `0x2466` = 0.017181396484375 | E7 |
| `%fd383` | s5024 `mov.f64 %fd383, 0dBFE1CC0000000000` | −0.55615234375 | f16 `0xB873` = −0.55615234375 | E7 |
| `%fd385` | s5029 `mov.f64 %fd385, 0d3FE1CC0000000000` | +0.55615234375 | f16 `0x3873` = +0.55615234375 | E7 |
| `%f478` | s5035 `mov.f32 %f478, 0f3F6D6000` | 0.92724609375 | f16 `0x3B6B` = 0.92724609375 | E7 |
| `%f480` | s5040 `mov.f32 %f480, 0f3FB00000` | 1.375 | f16 `0x3D80` = 1.375 | E7 |
| `%f861` | s9833 `mov.f32 %f861, 0f3ED306EB` | 0.4121621549129486 | f16 `0x3698` = 0.412109375 | E13…E35 |
| `%f862` | s9836 `mov.f32 %f862, 0f3DA60DD6` | 0.0810810774564743 | f16 `0x2D30` = 0.0810546875 | E13…E35 |
| `%f863` | s9839 `mov.f32 %f863, 0f3F000000` | 0.5 | f16 `0x3800` = 0.5 | E13…E35 |
| `%f864` | s9842 `mov.f32 %f864, 0f40000000` | 2.0 | f16 `0x4000` = 2.0 | E13…E35 |
| `%fd1` | s3662 `mov.f64 %fd1, 0d3F20000000000000` | 2⁻¹³ | f16 `0x0800` = 0.0001220703125 | prologue, E12 |

`%f861…%f864` are defined **inside E13** (s9833–s9842) and re-read by the `cvt`
of every later 164-statement epilogue (E15 s10174–s10181, E17 s10422–…, and so
on to E35), so those epilogues contain `cvt.rn.f16.f32` but only 2 `mov.u32`.

Also live across the region: `%r3037 = 1` (s3318), `%r3046 = 2` (s3611),
`%r3048 = −1` (s3604) — the delta / segmask / membermask operands of every
`shfl.sync.bfly.b32`; and `%r8606 = 0` (s4616), the zero C operand of phases 1,
2 and 8.

### 1.3 Two quantisation helpers

* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).
* `cvt.rn.f16x2.e4m3x2` — the *other* direction (f16x2 → e4m3x2), used only in
  the prologue to quantise the input patch (s3320 ff.).
* `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `div.approx.ftz.f32` — always wrapped as `cvt.f32.f16 → op → cvt.rn.f16.f32`
  (round each step through f16).

## 2. Summary

### 2.1 The 38 epilogues

| epilogue | range | stmts | named operation |
|---|---|---|---|
| E1 | s4645–s4651 | 7 | phase 2 B weight loads |
| E2 | s4676–s4682 | 7 | phase 3 B weight loads |
| E3 | s4707–s4713 | 7 | phase 4 B weight loads |
| E4 | s4738–s4744 | 7 | phase 5 B weight loads |
| E5 | s4769–s4775 | 7 | phase 6 B weight loads |
| E6 | s4800–s4970 | 171 | score-bias C-seed loads + packs of phase-5/6 D → phase 7 A and B |
| E7 | s5019–s8861 | 3843 | softmax of the 64×96 score, V transpose, packs → phase 8 A and B |
| E8 | s8910–s9078 | 169 | phase-8 D → shared, phase 9 A read, phase 9 B loads, per-column table add → phase 9 C |
| E9 | s9091–s9109 | 19 | phase 10 A from shared + phase 10 B loads |
| E10 | s9122–s9140 | 19 | phase 11 A from shared + phase 11 B loads |
| E11 | s9153–s9171 | 19 | phase 12 A from shared + phase 12 B loads |
| E12 | s9184–s9819 | 636 | RMS norm of phase-12 D + gain, packs; phase 13 A/C and phase 15 B |
| E13 | s9832–s10088 | 257 | clamped cubic on phase-13 D + packs; residual+table → phase 14 C; phase 14 B |
| E14 | s10101–s10160 | 60 | phase 15 B + C loads; 24 packs of the normed tile → phase 15 A |
| E15 | s10173–s10336 | 164 | clamped cubic on phase-15 D + packs; phase 16 B |
| E16 | s10349–s10408 | 60 | phase 17 B + C loads; packs of the normed tile → phase 17 A |
| E17 | s10421–s10584 | 164 | clamped cubic on phase-17 D + packs; phase 18 B |
| E18 | s10597–s10656 | 60 | phase 19 B + C loads; packs of the normed tile → phase 19 A |
| E19 | s10669–s10832 | 164 | clamped cubic on phase-19 D + packs; phase 20 B |
| E20 | s10845–s10904 | 60 | phase 21 B + C loads; packs of the normed tile → phase 21 A |
| E21 | s10917–s11080 | 164 | clamped cubic on phase-21 D + packs; phase 22 B |
| E22 | s11093–s11152 | 60 | phase 23 B + C loads; packs of the normed tile → phase 23 A |
| E23 | s11165–s11328 | 164 | clamped cubic on phase-23 D + packs; phase 24 B |
| E24 | s11341–s11400 | 60 | phase 25 B + C loads; packs of the normed tile → phase 25 A |
| E25 | s11413–s11576 | 164 | clamped cubic on phase-25 D + packs; phase 26 B |
| E26 | s11589–s11648 | 60 | phase 27 B + C loads; packs of the normed tile → phase 27 A |
| E27 | s11661–s11824 | 164 | clamped cubic on phase-27 D + packs; phase 28 B |
| E28 | s11837–s11896 | 60 | phase 29 B + C loads; packs of the normed tile → phase 29 A |
| E29 | s11909–s12072 | 164 | clamped cubic on phase-29 D + packs; phase 30 B |
| E30 | s12085–s12144 | 60 | phase 31 B + C loads; packs of the normed tile → phase 31 A |
| E31 | s12157–s12320 | 164 | clamped cubic on phase-31 D + packs; phase 32 B |
| E32 | s12333–s12392 | 60 | phase 33 B + C loads; packs of the normed tile → phase 33 A |
| E33 | s12405–s12568 | 164 | clamped cubic on phase-33 D + packs; phase 34 B |
| E34 | s12581–s12640 | 60 | phase 35 B + C loads; packs of the normed tile → phase 35 A |
| E35 | s12653–s12816 | 164 | clamped cubic on phase-35 D + packs; phase 36 B |
| E36 | s12829–s13698 | 870 | pack phase-36 D → arena scatter + shared staging; phase 37 A/B/C |
| E37 | s13747–s13870 | 124 | pack phase-37 D → 4 stores to `param+56`; phase 38 B/C |
| E38 | s13877–s14199 | 323 | shared staging, real row max, softmax, 9 texture reads, 3×3 gather, sigmoid/0.25/lane sum, 3 surface writes |

### 2.2 Phase geometry re-derived from the mma chain

`|A|` and `|B|` are the distinct A-quad and B-pair counts;
`chains`/`len` come from the C←D dataflow runs; `mma` is the emitted count.
`spec` is the phase table's (`M`, `N`, `K`, `tiles`, `mma`).

| # | A-quads | B-pairs | chains×len | k | re-derived (M,N,K,tiles) | spec (M,N,K,tiles) | M/16·N/8·K/32 vs mma |
|---|---|---|---|---|---|---|---|
| 1–6 | 6 | 4 | 24×1 | 1 | 96,32,32,24 | 96,32,32,24 | 6·4·1 = 24 = mma ✔ |
| 7 | 4 | 12 | 48×1 | 1 | 64,96,32,48 | 64,96,32,48 | 4·12·1 = 48 = mma ✔ |
| 8 | 12 | 12 | 16×3 | 3 | **64,32,96,16** | 192,96,96,16 | 4·4·3 = 48 = mma ✔ (spec M,N too big) |
| 9–12 | 1 | 12 | 12×1 | 1 | 16,96,32,12 | 16,96,32,12 | 1·12·1 = 12 = mma ✔ |
| 13,15,…,35 | 3 | 12 | 4×3 | 3 | **16,32,96,4** | 48,96,96,4 | 1·4·3 = 12 = mma ✔ (spec M,N too big) |
| 14,16,…,36 | 1 | 12 | 12×1 | 1 | 16,96,32,12 | 16,96,32,12 | 1·12·1 = 12 = mma ✔ |
| 37 | 12 | 48 | 4×12 | 12 | **16,32,384,4** | 192,384,384,4 | 1·4·12 = 48 = mma ✔ (spec M,N too big) |
| 38 | 3 | 6 | 2×3 | 3 | **16,16,96,2** | 48,48,96,2 | 1·2·3 = 6 = mma ✔ (spec M,N too big) |

Explicit answer to the flagged rows: **in every one of the 38 rows the identity
`mma = (M/16)·(N/8)·(K/32)` holds, and none of the phase-table rows is
arithmetically inconsistent.**  What is wrong in the table is `M` and `N` (and
hence the `full`/`band` label the tool derives from them): the tool sets
`M = 16·|A|`, `N = 8·|B|` (`rr_layer_spec.py:481`, `:590–592`), which counts every
k32 step as a separate m- and n-tile, so whenever `K > 32` its M and N are too
large by exactly the factor `k = K/32` and its `dense` test
(`nmma == nA*nB*lens[0]`, `:593`) is too strict.  Dividing out `k` — which the
chain's own `m·k = |A|`, `n·k = |B|`, `m·n = chains` fixes uniquely — gives the
row marked `re-derived`, and every one then satisfies the identity:

* rows 1–7, 9–12, 14, 16, …, 36 are correct as printed (`k = 1`);
* row 8: `|A|=12, |B|=12, k=3` → **M=64, N=32, K=96** (tool: 192, 96, 96) —
  dense, 4·4·3 = 48;
* rows 13, 15, …, 35: `|A|=3, |B|=12, k=3` → **M=16, N=32, K=96** (tool: 48, 96,
  96) — dense, 1·4·3 = 12; the tool's `K = 96` is right, only M and N are 3× too
  large;
* row 37: `|A|=12, |B|=48, k=12` → **M=16, N=32, K=384** (tool: 192, 384, 384) —
  dense, 1·4·12 = 48; again only M and N are wrong (12×);
* row 38: `|A|=3, |B|=6, k=3` → **M=16, N=16, K=96** (tool: 48, 48, 96) — dense,
  1·2·3 = 6.

The *operand provenance* of the corrected rows is checked separately from the
counts: for row 8 the 12 A quads are E7's probability packs of a 64×96 matrix
(§3/E7(g)) and the 12 B pairs its `movmatrix` transposes of a 96×32 matrix, giving
4 m-tiles × 3 k and 4 n-tiles × 3 k directly; for rows 13…35 the 3 A quads are
E12's three 16×32 e4m3 slices of the 16×96 normed tile (§3/E12) → 1 m-tile × 3 k;
for row 37 the 12 A quads are read as 48 `ld.shared.u32` from the staged phase-36
output and the 48 B pairs come from 24 512-byte tiles → 1 m-tile × 12 k and 4
n-tiles × 12 k; for row 38 the 3 A quads are staged from phase 37's chain-final D
and the 6 B pairs from 3 tile loads → 1 × 3 and 2 × 3.  The only thing the operand
provenance does *not* settle is which (n, k) each 512-byte B tile holds for rows
13…35 and 37 — see §6 U1.

## 3. The epilogues

### E1 — s4645–s4651: phase 2 B weight loads

```
s4645 L7837: mov.u32 %r3499, %laneid
s4646 L7839: mul.wide.u32 %rd270, %r3499, 16
s4647 L7840: add.s64 %rd271, %rd268, %rd270
s4648 L7841: add.s64 %rd64, %rd271, 3264
s4649 L7843: ld.weak.global.ca.v4.u32 { %r3500,%r3501,%r3502,%r3503},[%rd64]
s4650 L7845: add.s64 %rd65, %rd271, 3776
s4651 L7847: ld.weak.global.ca.v4.u32 { %r3504,%r3505,%r3506,%r3507},[%rd65]
```

Two 16-byte/lane fragments of `Wz + {3264, 3776}` = the weight block
`[3264, 4296)` — phase 2's B operand (`%r3500,%r3501` … `%r3506,%r3507`), which
phase 2 consumes at s4652–s4675.  `%rd268` is the `Wz` pointer of §1.1, so this
is `W + 21504*%tid.z + 16*laneid + off`.  Nothing else.

### E2 — s4676–s4682: phase 3 B weight loads

Same shape, base register `%rd268`, offsets `+1216` (s4679) and `+1728` (s4681)
→ `%r3749…%r3756`.  Weight block `[1216, 2248)`.

### E3 — s4707–s4713: phase 4 B weight loads

Offsets `+4288` (s4710), `+4800` (s4712) → `%r3998…%r4005`.  Block
`[4288, 5320)`.

### E4 — s4738–s4744: phase 5 B weight loads

Offsets `+2240` (s4741), `+2752` (s4743) → `%r4247…%r4254`.  Block
`[2240, 3272)`.

### E5 — s4769–s4775: phase 6 B weight loads

Offsets `+5312` (s4772), `+5824` (s4774) → `%r4496…%r4503`.  Block
`[5312, 6344)`.

---

### E6 — s4800–s4970: score-bias seeds + packs of phases 5 and 6

171 statements: 24 weight loads, 80 `cvt.rn.satfinite.e4m3x2.f16x2`, 40
`mov.b32`, and the address setup.  Two independent things happen.

**(a) The phase-7 score bias, s4800–s4850.** 24 `ld.weak.global.ca.v4.u32` at
`Wz + 16*laneid + 6336 + 512*i`, `i = 0…23`:

```
s4803 L8746: add.s64 %rd74, %rd281, 6336
s4804 L8748: ld.weak.global.ca.v4.u32 { %r4745,%r4746,%r4747,%r4748},[%rd74]
   …  +6848, +7360, +7888, … +18112 …
s4850 L8840: ld.weak.global.ca.v4.u32 { %r4837,%r4838,%r4839,%r4840},[%rd97]
```

24 × 512 bytes = **12288 bytes** = 6144 f16, the byte block `[6336, 18624)` of
the slab.  The 96 registers `%r4745…%r4840` are phase 7's **C operand**: each
mma uses one distinct C pair and the chain analysis of §2.2 shows all 48 pairs
are distinct (grid dump: `C frags: %r4745,%r4746 … %r4839,%r4840`).  So the
score GEMM adds a **full 64×96 f16 additive bias, one value per score element**,
laid out tile-major: with `m` = m-tile 0…3 and `n` = n-tile 0…11,

```
C(m,n) = the register pair at Wz + 6336 + 512*(4*(n>>1) + m) + 16*laneid + 4*((n&1)*2 + j),  j = 0,1
byte base of tile (m,n) = 6336 + 512*(4*(n>>1) + m),  48 tiles × 512 B = 12288 B.
```

Verified directly on the emitted chain: `s4971` uses `C={%r4745,%r4746}` with
`A=%r5033…`, `B={%r5287,%r5288}`; `s4983` (the same m-tile 1 step later) uses
`C={%r4749,%r4750}`; `s4995` `C={%r4753,%r4754}`; `s5007` `C={%r4757,%r4758}` —
stride 4 registers per m-tile, 2 registers per n-tile.  This is the same layout
formula enc0's E9 derived for its 12288-byte score bias
(`C(m,n) = load_tile(4*(n>>1)+m) + 4*(n&1)`); only the base differs (6336 here,
8384 there).  What the table *is* cannot be decided from the PTX — see §6 U2.

**(b) Phase 7's operands are register-resident, s4851–s4970.** 80 `cvt` +
40 `mov.b32` produce **both** A and B of phase 7 from phase 5's and phase 6's D:

```
s4851 L8843: cvt.rn.satfinite.e4m3x2.f16x2 %rs1462, %r4275
s4852 L8846: cvt.rn.satfinite.e4m3x2.f16x2 %rs1461, %r4255
s4853 L8848: mov.b32 %r5287, {%rs1461, %rs1462}          // phase 7 B frag 0
   …
s4925 L9016: mov.b32 %r5033, {%rs1509, %rs1510}          // phase 7 A frag 0
```

`%r5287` and `%r5033` are **bit-identical**: both are
`{pack(%r4255), pack(%r4275)}` (s4852–s4853 vs s4924–s4925, with `%r4255` ← the
phase-5 D of s4745 and `%r4275` ← that of s4747).  Every further pair follows
the same pattern with the two halves swapped: s4865 `%r5307 =
{pack(%r4295), pack(%r4315)}` against s4937 `%r5153 = {pack(%r4315),
pack(%r4295)}`.  So phase 7's B operand carries the *same data* as its A operand
(re-ordered for the row/col fragment layouts), i.e. the score GEMM multiplies one
projection by its own transpose; and 4 A quads + 12 B pairs = 64×32 · 32×96.

**Inputs:** phase 5's D `%r4255…%r4486`, phase 6's D `%r4504…%r4735`, `Wz`.
**Outputs:** the 48 C seeds `%r4745…%r4840`, the 4 A quads
(`%r5033…%r5036`, `%r5153…%r5156`, `%r5273…%r5276`, `%r5393…%r5396`) and the 12
B pairs (`%r5287…%r5398`).  **Memory:** 24 weight loads.

---

### E7 — s5019–s8861: softmax of the 64×96 score, then the V transpose

3843 statements, **no memory access at all** (0 `ld.*`, 0 `st.*`).  Sub-sections,
verified by counting each op's first and last statement:

| range | stmts | what |
|---|---|---|
| s5019–s7136 | 2118 | 96 per-element groups: scale, clamp, cubic, exponent extraction |
| s7137–s7222 | 86 | in-lane partial sum tree |
| s7223–s7344 | 122 | 16 `shfl.sync.bfly` → 8 row sums |
| s7345–s8117 | 773 | 192 `rcp.approx.ftz.f32` → 96 reciprocals |
| s8118–s8597 | 480 | 192/288 `mul.f16` → 96 probabilities |
| s8598–s8645 | 48 | `movmatrix` transpose of phase 6's D |
| s8646–s8861 | 216 | 144 `cvt.rn.satfinite.e4m3x2.f16x2` packs |

**(a) per-element exponent, s5019–s7136.** 96 groups of 22 statements.  A group:

```
s5020 L9461: mov.f32 %f476, 0f3C8CCB50
s5021 L9464: cvt.rn.f16.f32 low, %f476
s5022 L9465: mov.b32 %r5402, {low,low}
s5023 L9469: mul.f16x2 %r5403,%r4921,%r5402
s5025 L9474: cvt.rn.f16.f64 %rs1542, %fd383          // -0.55615234375
s5028 L9483: max.f16x2 %r5406,%r5403,%r5408
s5033 L9497: min.f16x2 %r5409,%r5406,%r5411          // +0.55615234375
s5034 L9501: neg.f16x2 %r5412,%r5409
s5039 L9515: fma.rn.f16x2 %r5414,%r5409,%r5412,%r5417 // 0.92724609375
s5044 L9529: fma.rn.f16x2 %r5418,%r5409,%r5414,%r5421 // 1.375
s5045 L9532: shl.b32 %r16192, %r5418, 5
s5046 L9533: and.b32 %r7323, %r16192, 2145419232      // 0x7FE07FE0
```

For each of phase 7's 96 f16x2 D registers, per half (an f16 value `x`):

```
m  = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)
t  = f16(m*(-m) + 0.92724609375)
u  = f16(m*t    + 1.375)  =  f16(1.375 + m*(0.92724609375 - m^2))
E  = bits(u)[9:5]     // floor(32*frac(u))    -- top five mantissa bits
F  = bits(u)[4:0]     // low five mantissa bits
expval = f16 from bits (E << 10) | (F << 5)   // = 2^(E-15)*(1 + F/32)
```

Exactly the enc0 E5 scheme (`((u_bits<<5) & 0x7FE07FE0)` drops `u`'s sign and
exponent so the mantissa lands in the exponent field): `expval ≈
2^(32·frac(u) − 15) = 2^(32u − 47)` with `u ∈ [1.03125, 1.71875]` because the
clamp `±0.55615… = √(0.92724609375/3)` is the turning point of
`0.92724609375·m − m³`.  **This is not `exp`**: it is the cubic
`32·(0.375 + 0.92724609375·m − m³)` truncated to a 6-bit significand.

**(b),(c) row sums, s7137–s7344.** 80 `add.f16x2` reduce the 96 `expval`
registers in-lane; then eight times (one per row `g`, `g+8` of the four m-tiles):

```
s7223 L16329: shfl.sync.bfly.b32 %r7567,%r7563,%r3037,%r7570,%r3048   // delta 1
s7224 L16333: add.f16x2 %r7572,%r7563,%r7567
s7229 L16344: shfl.sync.bfly.b32 %r7576,%r7572,%r3046,%r7579,%r3048   // delta 2
s7230 L16348: add.f16x2 %r7581,%r7572,%r7576
s7232 L16353: add.f16 %rs2309,%rs2310,%rs2311
s7233 L16356: mov.b32 %r7778, {%rs2309, %rs2309}
```

Two butterfly steps over lane bits 0 (delta 1, `%r3037`) and 1 (delta 2,
`%r3046`), then `add.f16` of the two halves and a broadcast into both halves.
Result: **8 row sums** `%r7778 %r7782 %r7826 %r7830 %r7874 %r7878 %r7922
%r7926`, one per row of the 64-row window.

**There is no row maximum anywhere in this epilogue.**  Enumerated: every one of
the 96 `max.f16x2` (s5028…s7121) is immediately followed by a `min.f16x2`
(s5033…s7125) and preceded by the `mul.f16x2` with `%r5402` (the 0.017181 scale)
— one clamp triple per element group.  The 96 `neg.f16x2` (s5034…s7126) feed the
first `fma`, and the 16 `shfl.sync.bfly` are the row sums above.  Nothing else
touches a D register.  Contrast E38 (§3/E38), which *does* compute a real row
max.

**(d) reciprocals, s7345–s8117.** 192 `rcp.approx.ftz.f32` = 96 f16x2
reciprocals, each `cvt.f32.f16 → rcp.approx.ftz.f32 → cvt.rn.f16.f32` on a row
sum (`s7350 L16652: mov.b32 {hl, hu}, %r7778` / `s7353 L16655:
rcp.approx.ftz.f32 fl, fl`), duplicated four times per row sum (once per n-tile).

**(e) probabilities, s8118–s8597.**

```
s8118 L17897: mov.b32 {%rs2334, %rs2337}, %r7323     // expval
s8119 L17898: mov.b32 {%rs2335, %rs2338}, %r7735     // 1/rowsum
s8120 L17900: mul.f16 %rs2336,%rs2337,%rs2338
s8121 L17904: mul.f16 %rs2333,%rs2334,%rs2335
s8122 L17907: mov.b32 %r8071, {%rs2333, %rs2336}
```

`p = expval · (1/rowsum)`; the row sums cancel the constant `2⁻⁴⁷` of (a), so
`p_ij ∝ 2^(32·u_ij)`.  96 f16x2 results.

**(f) V transpose, s8598–s8645.** 48 `movmatrix.sync.trans.aligned.m8n8.b16`, one
per register of **phase 6's D** (`%r4504, %r4505, %r4514, %r4515, %r4544,
%r4545, … %r4734, %r4735` — exactly the 24 C/D pairs listed at s4776–s4799, each
once):

```
s8598 L18954: movmatrix.sync.trans.aligned.m8n8.b16 %r7927, %r4504
s8645 L19095: movmatrix.sync.trans.aligned.m8n8.b16 %r8021, %r4735
```

48 × 64 = 3072 values = the whole 96×32 tile of phase 6's D.

**(g) packs, s8646–s8861.** 144 `cvt` / 72 `mov.b32` in two groups:

* s8646–s8717 — from the **movmatrix** results → phase 8's 12 B pairs
  (`%r8533,%r8534`, `%r8543,%r8544`, … `%r8633,%r8644`), i.e. the transposed V:
  `s8646 L19098: cvt.rn.satfinite.e4m3x2.f16x2 %rs2910, %r7935` /
  `s8648 L19103: mov.b32 %r8533, {%rs2909, %rs2910}`.
* s8718–s8861 — from the **probabilities** → phase 8's 12 A quads
  (`%r8239…%r8242`, `%r8259…`, `%r8279…`, `%r8359…`, `%r8379…`, `%r8399…`,
  `%r8479…`, `%r8499…`, `%r8519…`, `%r8599…`, `%r8619…`, `%r8639…%r8642`):
  `s8720 L19271: mov.b32 %r8239, {%rs2957, %rs2958}`.

**Inputs:** phase 7's 96 D f16x2 (`%r4921…%r5392`), phase 6's 48 D
(`%r4504…%r4735`), `%r3037/%r3046/%r3048`, the five constants of §1.2.
**Outputs:** the 5 constants + phase 8's 12 A quads and 12 B pairs.
**Memory:** none.

---

### E8 — s8910–s9078: phase-8 D → shared, phase 9 A, table add → phase 9 C

169 statements, four moves.

**(a) stage phase 8's D into shared, s8910–s8968.**

```
s8910 L19937: bar.sync 0
s8911 L19938: shl.b32 %r16320, %r10, 11            // %tid.z * 2048
s8912 L19939: add.s32 %r16321, %r3216, %r16320     // %r3216 = smem symbol
s8915 L19944: add.s32 %r16323, %r16321, %r16322    // + 16*laneid
s8928 L19973: st.shared.v4.u32 [%r16323],        {%r16327, %r16326, %r16325, %r16324}
s8941 L20002: st.shared.v4.u32 [%r16323+512],    {…}
s8954 L20031: st.shared.v4.u32 [%r16323+1024],   {…}
s8967 L20060: st.shared.v4.u32 [%r16323+1536],   {…}
s8968 L20061: bar.sync 0
```

Four v4 stores (16 bytes/lane) at `smem + 2048*%tid.z + 16*laneid + {0,512,1024,
1536}`.  Each v4 word is an e4m3 pack of **two** distinct phase-8 D registers
(e.g. s8916 `cvt %rs3054, %r8267`, s8917 `cvt %rs3053, %r8207`, s8924
`mov.b32 %r16324, {%rs3059, %rs3060}`); all 32 `cvt` in the block name distinct
registers, so **32 of phase 8's 48 D registers are staged** here, 16 words per
lane.

**(b) phase 9's A, s8969–s8974.**

```
s8969 L20062: shl.b32 %r16340, %r10, 9             // %tid.z * 512
s8970 L20063: add.s32 %r16341, %r3216, %r16340
s8973 L20068: add.s32 %r16343, %r16341, %r16342    // + 16*laneid
s8974 L20069: ld.shared.v4.u32 {%r16344, %r16345, %r16346, %r16347}, [%r16343]
```

One A quad (`%r16344…%r16347`), i.e. the e4m3-packed phase-8 D, re-read through
shared.  Phase 9's mma (s9079) uses exactly this quad.

**(c) phase 9's B, s8975–s8989.** 6 `ld.weak.global.ca.v4.u32` at
`W + 16*laneid + 18624 + 512*i`, `i = 0…5` (s8978 `add.s64 %rd98, %rd283,
18624`) → `%r8682…%r8705`, the 12 B pairs of phase 9.  Note the base is `%rd283 =
%rd2 + 16*laneid` (s8977) — **no `%tid.z` term**, unlike E1–E6.

**(d) per-column residual table → phase 9's C, s8990–s9078.**

```
s8995 L20106: ld.global.v2.u16 {%rs4364, %rs4365}, [%rd286+86224]
   …  86240, 86256, 86272, 86288, 86304, 86320, 86336, 86352, 86368, 86384, 86208
s9007 L20119: add.f16 %rs3088,%rs4387,%rs1103
s9008 L20123: add.f16 %rs3085,%rs4386,%rs1100
s9009 L20126: mov.b32 %r8715, {%rs3085, %rs3088}
```

12 `ld.global.v2.u16` at `W + ((laneid<<2)&12) + {86208, 86224, …, 86384}` = 192
bytes = 96 f16, a per-column vector over the 96-wide N axis.  48 `add.f16` add
them (pair-swapped) to the 48 halves of 24 **prologue** registers and 24
`mov.b32` build the 24 registers `%r8715…%r8826` = phase 9's **C operand**.
The 48 added operands are `%rs1100 … %rs1361`, and every one of them is defined
by a prologue `mov.b32` from the *quantised input patch* — `%rs1100/%rs1103` ←
`%r2798` (s3366, source line 4684), `%rs1106/%rs1109` ← `%r2802`, … — where
`%r2798` is itself `cvt.rn.f16x2.e4m3x2 %r2798, %rs128` (s3320), i.e. the first
element of the input image read at s176.  So

```
C9 = column_table[86208] + (the block's quantised input patch, re-read as f16)
```

— an *input* skip connection, **not** phase 8's D.  (This is a surprising but
direct reading: the 48 `add.f16` sources are the halves of `%r2798…%r2845`, the
prologue's `cvt.rn.f16x2.e4m3x2` results, and no statement anywhere redefines
them; see §6 U9.)

**Inputs:** phase 8's 48 D (staged in (a)), the prologue's 24 quantised input
registers `%r2798…`, table `86208`, `%tid.z`, `%r3216`, `W`.
**Outputs (live 49):** `%r16344…%r16347` (via shared), the 24 C seeds, the 12 B
pairs.  **Memory:** 6 weight loads + 12 table reads + 2 barriers + 4 shared
stores + 1 shared load.

---

### E9 — s9091–s9109: phase 10 A from shared + phase 10 B

19 statements.

```
s9091 L20419: mov.u32 %r8827, %laneid
s9092 L20421: shl.b32 %r16349, %r8827, 4
s9094 L20423: ld.shared.v4.u32 {%r16351, %r16352, %r16353, %r16354}, [%r16350+2048]
s9099 L20431: ld.weak.global.ca.v4.u32 { %r8829,%r8830,%r8831,%r8832},[%rd104]
   … 5 more at 512 stride …
s9109 L20451: ld.weak.global.ca.v4.u32 { %r8849,%r8850,%r8851,%r8852},[%rd109]
```

One A quad read from `smem + 16*laneid + 2048` (i.e. the *next* staged slot of
(a) in E8), and 6 weight fragments at `W + 16*laneid + 40128 + 512*i` = phase
10's B (`%r8829…%r8852`).  The shared slot is 2048 bytes above the slot phase 9
read, so the four staged v4 words serve phases 9, 10, 11, 12 at
`base + {0, 2048, 4096, 6144}`.

### E10 — s9122–s9140: phase 11 A from shared + phase 11 B

Same shape; `s9125 L20542: ld.shared.v4.u32 {%r16357…%r16360}, [%r16356+4096]` and
6 weight fragments at `W + 16*laneid + 61632 + 512*i` (s9130 …) → `%r8975…%r8998`.

### E11 — s9153–s9171: phase 12 A from shared + phase 12 B

`s9156 L20661: ld.shared.v4.u32 {%r16363…%r16366}, [%r16362+6144]` and 6 weight
fragments at `W + 16*laneid + 83136 + 512*i` (s9161 …) → `%r9121…%r9144`.
Note phases 9–12 also chain their C: phase 12's C is
`{%r8999,%r9000}` = phase 11's D (s9141 / s9172), and likewise 10←9, 11←10.

---

### E12 — s9184–s9819: the RMS norm before the MLP band

636 statements.  Structure (counts from the op histogram of s9184–s9819: 182
`mov.b32`, 144 `mul.f16`, 50 `add.f16`, 48 each `cvt.f32.f16` /
`rsqrt.approx.ftz.f32` / `cvt.rn.f16.f32`, 26 `add.f16x2`, 24
`cvt.rn.satfinite.e4m3x2.f16x2`, 12 `ld.global.v2.u16`, 6 weight loads, 4
`shfl.sync.bfly`):

| range | stmts | op |
|---|---|---|
| s9184–s9281 | 98 | 48 `mul.f16` — the squares of phase 12's D |
| s9282–s9301 | 20 | 21 `add.f16x2` — in-lane reduction tree |
| s9302–s9344 | 43 | 4 `shfl.sync.bfly` + `add.f16x2` + `add.f16` + `+ eps`, broadcast into 24 b32 |
| s9345–s9405 | 61 | 48 `add.f16` more (24 duplicated eps-bearing sums) |
| s9406–s9598 | 193 | 48 `rsqrt.approx.ftz.f32` (24 inverses) |
| s9599–s9615 | 17 | 12 gain loads `+86400 … +86576` |
| s9616–s9687 | 72 | 48 `mul.f16` — `inv · gain` |
| s9688–s9759 | 72 | 48 `mul.f16` — `x · (inv·gain)` |
| s9760–s9774 | 15 | 6 weight loads `+86632 … +89192` (phase 13's B) |
| s9775–s9783 | 9 | 4 `ld.global.u32` `+89664 … +89712` (phase 13's C seeds) |
| s9784–s9819 | 36 | 24 packs → phase 13's A |

Evidence:

```
s9185 L20776: add.s32 %r182, %r2, %r181            // %r2 + 2*%tid.z
s9186 L20777: mov.b32 {%rs3838, %rs3841}, %r9145   // unpack a D register
s9187 L20779: mul.f16 %rs3232,%rs3841,%rs3841      // square (high half)
s9188 L20783: mul.f16 %rs3229,%rs3838,%rs3838      // square (low half)
s9282 L21018: add.f16x2 %r9265,%r9266,%r9267
s9303 L21101: add.f16x2 %r9326,%r9313,%r9319
s9308 L21112: shfl.sync.bfly.b32 %r9330,%r9326,%r3037,%r9333,%r3048
s9317 L21136: add.f16 %rs3373,%rs3374,%rs3375
s9333 L21179: cvt.rn.f16.f64 %rs3379, %fd1         // eps = 2^-13
s9334 L21183: add.f16 %rs3383,%rs3373,%rs3379
s9410 L21407: rsqrt.approx.ftz.f32 fl, fl
s9604 L21720: ld.global.v2.u16 {%rs4388, %rs4389}, [%rd295+86416]
s9616 L21732: mov.b32 {%rs3525, %rs3528}, %r9369
s9688 L21949: mul.f16 %rs3671,%rs3841,%rs3527      // x * (inv*gain)
s9690 L21956: mov.b32 %r15561, {%rs3668, %rs3671}  // the normed register
s9784 L22205: cvt.rn.satfinite.e4m3x2.f16x2 %rs3813, %r15562
s9786 L22210: mov.b32 %r9540, {%rs3812, %rs3813}
s9780 L22200: ld.global.u32 %r9477, [%rd300+89664]
```

**What it computes.**  Phase 12's D is 24 registers = 48 f16/lane; each register
pairs `(row g, cols 2t,2t+1)` with `(row g+8, cols 2t,2t+1)`, and two registers
(`%r9145`,`%r9146`) are one n-tile.  For each of the two rows a lane holds:

```
sumsq[row] = Σ_{c=0..95} x[row,c]^2                     (f16, one shfl.bfly pair)
inv        = rsqrt.approx.ftz.f32(f32(f16(sumsq + 2^-13)))    // rounded to f16
out[row,c] = x[row,c] * inv * gain[c]
```

* **There is no mean subtraction and no `1/N`**: `%rs3232/%rs3229` are raw
  squares (`mul.f16 %rs3232,%rs3841,%rs3841`), the tree only adds, and the only
  constant added is `%fd1 = 2⁻¹³` (s9333/s9334).  The quantity is the raw sum of
  96 squares plus `2⁻¹³`; the mean is never formed.  This is an **RMS norm**, not
  a mean-subtracting LayerNorm.
* `gain[c]` is read from `W + ((laneid<<2)&12) + {86400, 86416, …, 86576}` (12
  `v2.u16` at 16-byte stride = 192 bytes = 96 f16), i.e. a per-column vector over
  the 96 columns, constant across rows.
* the two butterfly steps (`%r3037 = 1`, `%r3046 = 2`, segmask 31, membermask
  −1 at s9307/s9313) reduce over the four lanes with equal `laneid>>2`, which
  between them hold all 96 columns of the row; `add.f16 %rs3373,%rs3374,%rs3375`
  folds the two column halves.  The inverse is broadcast into both halves
  (`mov.b32 %r9370, {%rs3380, %rs3383}`) so one row value serves all n-tiles.
* the norm's *bias* is **not** added here.  It is the 4 `ld.global.u32` at the end
  (s9780–s9783), which become phase 13's C seeds as `{%r9477, %r9477}` etc. —
  a per-column bias replicated over the 16 rows.

**The prologue's `normalise:96@s3812` is the same function.**  Its op block
(s3250–s4497) has the same shape and the same missing pieces: 48
`cvt.rn.f16x2.e4m3x2` quantise the 24 `ld.global.v2.u16` input loads (s176,
s328, … s3316, all through `%rd12 = %rd4 + 4*index`, `%rd4 = param_0+8`, i.e.
the input image); 96 `mul.f16` square them in place
(`s3431 L4846: mul.f16 %rs321,%rs1199,%rs1199`); 52 `add.f16x2` + 8
`shfl.sync.bfly` + `add.f16` reduce; `s3662 L5488: mov.f64 %fd1,
0d3F20000000000000` / `s3664 L5494: add.f16 %rs526,%rs510,%rs522` add the
*same* `2⁻¹³`; 96 `rsqrt.approx.ftz.f32` (48 pairs, `s3812 L5934:
rsqrt.approx.ftz.f32 fl, fl`) invert; the 96 products are multiplied by a
per-column gain read as 12 `ld.global.v2.u16` at `W + ((laneid<<2)&12) +
{16, 32, …, 176, 0}` (s4198–s4209, 192 bytes = 96 f16); 96 more `mul.f16`
(`s4211 L6573: mul.f16 %rs811,%rs812,%rs1459`) apply the gain; 48
`cvt.rn.satfinite.e4m3x2.f16x2` (s4498–s4521) requantise and six
`st.shared.v4.u32` (s4540, s4553, s4566, s4576, s4581, s4586) stage the 18 A
quads.  No mean, no `1/N`, no bias add — **also an RMS norm**, with its bias
deferred to a later GEMM's C exactly as in E12.

**The normed tile is the A operand of every odd phase 13…35.**  E12's output
registers `%r15561…%r15584` (24 b32) are re-packed not only here (s9784–s9819,
→ `%r9540,%r9541,%r9542,%r9543` … `%r9580…%r9583`, i.e. three 16×32 A slices)
but again in **every even-numbered epilogue E14, E16, …, E34**:
`%r15561` is read by exactly the s9785 (E12), s10126 (E14), s10374 (E16),
s10622 (E18), s10870 (E20), s11118 (E22), s11366 (E24), s11614 (E26), s11862
(E28), s12110 (E30), s12358 (E32), s12606 (E34) — twelve repacks, one per band
phase.  So all twelve odd phases 13, 15, …, 35 share **one** input tile.

**Inputs:** phase 12's 24 D, `%fd1`, the gain table, `W`.  **Outputs (54 live):**
phase 13's 3 A quads (`%r9540…%r9543`, `%r9560…%r9563`, `%r9580…%r9583`), phase
13's 4 C seeds (`%r9477, %r9487, %r9537, %r9547`) and 12 B pairs (`%r9419…%r9442`,
6 loads at `+86592…+89152`), the 24 normed registers `%r15561…%r15584` (re-read by
E14, E16, …, E34).  **Memory:** 12 table + 6 weight loads.

---

### E13 — s9832–s10088: clamped cubic + residual/table, packs

257 statements.  This is the epilogue that materialises `%f861…%f864`.

**(a) the activation, s9832–s10076.** 8 groups (one per chain-final D register of
phase 13: `%r9508 %r9518 %r9509 %r9519 %r9568 %r9578 %r9569 %r9579`), 17
statements each:

```
s9833 L22375: mov.f32 %f861, 0f3ED306EB
s9836 L22382: mov.f32 %f862, 0f3DA60DD6
s9839 L22389: mov.f32 %f863, 0f3F000000
s9842 L22396: mov.f32 %f864, 0f40000000
s9845 L22404: neg.f16x2 %r9593,%r9592              // -2.0
s9846 L22408: max.f16x2 %r9595,%r9508,%r9593       // max(x, -2)
s9847 L22412: min.f16x2 %r9598,%r9595,%r9592       // min(·, +2)
s9848 L22416: abs.f16x2 %r9601,%r9598              // |y|
s9849 L22420: mul.f16x2 %r9603,%r9590,%r9601       // 0.0810546875*|y|
s9850 L22424: sub.f16x2 %r9606,%r9589,%r9603       // 0.412109375 - that
s9851 L22428: mul.f16x2 %r9609,%r9598,%r9606       // y * that
s9852 L22432: add.f16x2 %r9612,%r9591,%r9609       // 0.5 + that
s9853 L22436: mul.f16x2 %r9615,%r9508,%r9612       // x * g
```

Per element, `y = clamp(x, −2, +2)` and

```
g   = 0.5 + y * (0.412109375 − 0.0810546875 * |y|)        // f16 throughout
out = x * g
```

`g` is a cubic ramp: `g(+2) = 1` so `out = x` for `x ≥ 2`; `g(−2) = 0` so
`out = 0` for `x ≤ −2`.  The constants and the op order are **byte-identical to
enc0's E13** (`0f3ED306EB`, `0f3DA60DD6`, `0.5`, `2.0`, `neg/max/min/abs/mul/
sub/mul/add/mul`): this is the same clamped cubic, not a GELU, not a sigmoid and
not a polynomial in `x`.  Reproduce the formula, not `x·sigmoid(x)`.

**(b) the residual + per-column table → phase 14's C, s9990–s10076.**

```
s9993 L22895: ld.global.v2.u16 {%rs4412, %rs4413}, [%rd305+92816]
   …  92832, 92848, 92864, 92880, 92896, 92912, 92928, 92944, 92960, 92976, 92800
s10005 L22908: add.f16 %rs3839,%rs4435,%rs3841
s10006 L22912: add.f16 %rs3836,%rs4434,%rs3838
s10007 L22915: mov.b32 %r9863, {%rs3836, %rs3839}
```

12 `ld.global.v2.u16` at `W + ((laneid<<2)&12) + {92800…92976}` (192 bytes = 96
f16, one per column), then 48 `add.f16` and 24 `mov.b32` producing
`%r9863…%r9974`, the 24 registers phase 14 uses as its C operand.  `%rs3838/
%rs3841` are the two halves of `%r9145` — **phase 12's D**, not phase 13's, so

```
C14 = D12 + column_table[92800]        (a residual, added to all 16 rows)
```

**(c) phase 14's B, s9977–s9989.** 6 `ld.weak.global.ca.v4.u32` at
`W + 16*laneid + 89728 + 512*i` → `%r9822…%r9845`.

**(d) packs, s10077–s10088.** 8 `cvt` + 4 `mov.b32` → `%r9967, %r9969, %r9968,
%r9970` = phase 14's A quad, from the activation outputs (`%r9615`, `%r9644`,
`%r9673`, `%r9731`, `%r9760`, `%r9789`, `%r9818`, `%r9702`):

```
s10077 L23124: cvt.rn.satfinite.e4m3x2.f16x2 %rs3981, %r9731
s10078 L23127: cvt.rn.satfinite.e4m3x2.f16x2 %rs3980, %r9615
s10079 L23129: mov.b32 %r9967, {%rs3980, %rs3981}
```

**Inputs:** phase 13's 8 chain-final D, phase 12's 24 D, table `92800`, `W`.
**Outputs (24 live):** `%f861…%f864`, `%r9967…%r9970`, `%r9863…%r9974`,
`%r9822…%r9845`.  **Memory:** 6 weight + 12 table loads.

### E14 — s10101–s10160: phase 15 B/C + packs of the normed tile

60 statements: 6 weight loads at `+92992…+95552` → `%r9976…%r9999` (phase 15's
B); 4 `ld.global.u32` at `+96064, +96080, +96096, +96112` → `%r10034, %r10044,
%r10094, %r10104` (phase 15's C seeds, each used as `{%rX, %rX}`); and 24 packs
of **E12's normed registers** into phase 15's 3 A quads:

```
s10126 L23279: cvt.rn.satfinite.e4m3x2.f16x2 %rs3988, %r15561
s10127 L23281: mov.b32 %r10097, {%rs3988, %rs3989}
   …  %r10098, %r10099, %r10100 (quad 1) ; %r10117…%r10120; %r10137…%r10140
```

Note the pack order is `%r10097←(%r15561,%r15562)`, `%r10099←(%r15563,%r15564)`,
`%r10098←(%r15565,%r15566)`, `%r10100←(%r15567,%r15568)` (s10125–s10136), i.e.
the whole 16×96 normed tile is re-quantised into 12 b32 = three 16×32 A slices.

### E15 — s10173–s10336: clamped cubic on phase 15's D

164 statements = 8 activation groups + 6 weight loads + 4 packed b32.  Identical
to E13(a) but on phase 15's chain-final D (`%r10065 %r10075 %r10066 %r10076
%r10125 %r10135 %r10126 %r10136`) and with **no constants of its own** and **no
`ld.global`** — the four `cvt.rn.f16.f32` read `%f861…%f864` defined in E13:

```
s10174 L23448: cvt.rn.f16.f32 low, %f861
s10183 L23475: max.f16x2 %r10152,%r10065,%r10150
s10190 L23503: mul.f16x2 %r10172,%r10065,%r10169
s10314 L23933: ld.weak.global.ca.v4.u32 { %r10379,…},[%rd140]     // +96128
s10325 L23956: cvt.rn.satfinite.e4m3x2.f16x2 %rs4013, %r10288
s10327 L23961: mov.b32 %r10523, {%rs4012, %rs4013}
```

Outputs phase 16's A quad (`%r10523…%r10526`) and phase 16's B (6 loads at
`+96128…+98688`).  **No column-bias add.**

### E16 — s10349–s10408: phase 17 B/C + packs of the normed tile

Identical in shape to E14: B at `+99200 … +101760`, C words at
`+102272, +102288, +102304, +102320` (`s10369 L24103: ld.global.u32 %r10590,
[%rd317+102272]`), and 24 packs of `%r15561…%r15584` (s10373 ff.) → phase 17's
three A quads `%r10653…%r10656`, `%r10673…%r10676`, `%r10693…%r10696`.

### E17 — s10421–s10584: clamped cubic on phase 17's D

164 statements.  8 activation groups (17 statements each, the E13(a) shape) on
phase 17's chain-final D registers `%r10621 %r10631 %r10622 %r10632 %r10681
%r10691 %r10682 %r10692`; no constants of its own (`cvt.rn.f16.f32 low, %f861`
opens the first group at s10422); 6 weight loads at `W + 16*laneid +
102336 + 512*i` = phase 18's B; 8 `cvt` + 4 `mov.b32` → phase 18's A quad
`%r11079…%r11082`.

### E18 — s10597–s10656: phase 19 B/C loads + packs of the normed tile

60 statements: 6 weight loads at `+105408 … +107968` (phase 19's B), 4
`ld.global.u32` at `+108480, +108496, +108512, +108528` (phase 19's C seeds,
each used as `{%rX, %rX}`), and 24 `cvt` / 12 `mov.b32` of `%r15561…%r15584` →
phase 19's three A quads `%r11209…%r11212`, `%r11229…%r11232`,
`%r11249…%r11252`.  Evidence:
`s10601 L24906: ld.weak.global.ca.v4.u32 { %r11088,…},[%rd158]`;
`s10617 L24935: ld.global.u32 %r11146, [%rd324+108480]`.

### E19 — s10669–s10832: clamped cubic on phase 19's D

164 statements, identical to E17.  Chain-final D `%r11177 %r11187 %r11178
%r11188 %r11237 %r11247 %r11238 %r11248`; phase 20's B at `+108544 … +111104`;
phase 20's A quad `%r11635…%r11638`.

### E20 — s10845–s10904: phase 21 B/C loads + packs of the normed tile

60 statements.  B `+111616 … +114176`; C `+114688, +114704, +114720, +114736`;
A quads `%r11765…%r11768`, `%r11785…%r11788`, `%r11805…%r11808`.  Evidence:
`s10849 L25738: ld.weak.global.ca.v4.u32 { %r11644,…},[%rd170]`;
`s10865 L25767: ld.global.u32 %r11702, [%rd331+114688]`.

### E21 — s10917–s11080: clamped cubic on phase 21's D

164 statements.  Chain-final D `%r11733 %r11743 %r11734 %r11744 %r11793
%r11803 %r11794 %r11804`; phase 22's B at `+114752 … +117312`; phase 22's A
quad `%r12191…%r12194`.

### E22 — s11093–s11152: phase 23 B/C loads + packs of the normed tile

60 statements.  B `+117824 … +120384`; C `+120896, +120912, +120928, +120944`;
A quads `%r12321…%r12324`, `%r12341…%r12344`, `%r12361…%r12364`.  Evidence:
`s11097 L26570: ld.weak.global.ca.v4.u32 { %r12200,…},[%rd182]`;
`s11113 L26599: ld.global.u32 %r12258, [%rd338+120896]`.

### E23 — s11165–s11328: clamped cubic on phase 23's D

164 statements.  Chain-final D `%r12289 %r12299 %r12290 %r12300 %r12349
%r12359 %r12350 %r12360`; phase 24's B at `+120960 … +123520`; phase 24's A
quad `%r12747…%r12750`.

### E24 — s11341–s11400: phase 25 B/C loads + packs of the normed tile

60 statements.  B `+124032 … +126592`; C `+127104, +127120, +127136, +127152`;
A quads `%r12877…%r12880`, `%r12897…%r12900`, `%r12917…%r12920`.  Evidence:
`s11345 L27402: ld.weak.global.ca.v4.u32 { %r12756,…},[%rd194]`;
`s11361 L27431: ld.global.u32 %r12814, [%rd345+127104]`.

### E25 — s11413–s11576: clamped cubic on phase 25's D

164 statements.  Chain-final D `%r12845 %r12855 %r12846 %r12856 %r12905
%r12915 %r12906 %r12916`; phase 26's B at `+127168 … +129728`; phase 26's A
quad `%r13303…%r13306`.

### E26 — s11589–s11648: phase 27 B/C loads + packs of the normed tile

60 statements.  B `+130240 … +132800`; C `+133312, +133328, +133344, +133360`;
A quads `%r13433…%r13436`, `%r13453…%r13456`, `%r13473…%r13476`.  Evidence:
`s11593 L28234: ld.weak.global.ca.v4.u32 { %r13312,…},[%rd206]`;
`s11609 L28263: ld.global.u32 %r13370, [%rd352+133312]`.

### E27 — s11661–s11824: clamped cubic on phase 27's D

164 statements.  Chain-final D `%r13401 %r13411 %r13402 %r13412 %r13461
%r13471 %r13462 %r13472`; phase 28's B at `+133376 … +135936`; phase 28's A
quad `%r13859…%r13862`.

### E28 — s11837–s11896: phase 29 B/C loads + packs of the normed tile

60 statements.  B `+136448 … +139008`; C `+139520, +139536, +139552, +139568`;
A quads `%r13989…%r13992`, `%r14009…%r14012`, `%r14029…%r14032`.  Evidence:
`s11841 L29066: ld.weak.global.ca.v4.u32 { %r13868,…},[%rd218]`;
`s11857 L29095: ld.global.u32 %r13926, [%rd359+139520]`.

### E29 — s11909–s12072: clamped cubic on phase 29's D

164 statements.  Chain-final D `%r13957 %r13967 %r13958 %r13968 %r14017
%r14027 %r14018 %r14028`; phase 30's B at `+139584 … +142144`; phase 30's A
quad `%r14415…%r14418`.

### E30 — s12085–s12144: phase 31 B/C loads + packs of the normed tile

60 statements.  B `+142656 … +145216`; C `+145728, +145744, +145760, +145776`;
A quads `%r14545…%r14548`, `%r14565…%r14568`, `%r14585…%r14588`.  Evidence:
`s12089 L29898: ld.weak.global.ca.v4.u32 { %r14424,…},[%rd230]`;
`s12105 L29927: ld.global.u32 %r14482, [%rd366+145728]`.

### E31 — s12157–s12320: clamped cubic on phase 31's D

164 statements.  Chain-final D `%r14513 %r14523 %r14514 %r14524 %r14573
%r14583 %r14574 %r14584`; phase 32's B at `+145792 … +148352`; phase 32's A
quad `%r14971…%r14974`.

### E32 — s12333–s12392: phase 33 B/C loads + packs of the normed tile

60 statements.  B `+148864 … +151424`; C `+151936, +151952, +151968, +151984`;
A quads `%r15101…%r15104`, `%r15121…%r15124`, `%r15141…%r15144`.  Evidence:
`s12337 L30730: ld.weak.global.ca.v4.u32 { %r14980,…},[%rd242]`;
`s12353 L30759: ld.global.u32 %r15038, [%rd373+151936]`.

### E33 — s12405–s12568: clamped cubic on phase 33's D

164 statements.  Chain-final D `%r15069 %r15079 %r15070 %r15080 %r15129
%r15139 %r15130 %r15140`; phase 34's B at `+152000 … +154560`; phase 34's A
quad `%r15527…%r15530`.

### E34 — s12581–s12640: phase 35 B/C loads + packs of the normed tile

60 statements.  B `+155072 … +157632`; C `+158144, +158160, +158176, +158192`;
A quads `%r15657…%r15660`, `%r15677…%r15680`, `%r15697…%r15700`.  Evidence:
`s12585 L31562: ld.weak.global.ca.v4.u32 { %r15536,…},[%rd254]`;
`s12601 L31591: ld.global.u32 %r15594, [%rd380+158144]`.

### E35 — s12653–s12816: clamped cubic on phase 35's D

164 statements.  Chain-final D `%r15625 %r15635 %r15626 %r15636 %r15685
%r15695 %r15686 %r15696`; phase 36's B at `+158208 … +160768`
(`s12794 L32253: ld.weak.global.ca.v4.u32 { %r15939,…},[%rd260]`); phase 36's A
quad `%r16083…%r16086`.  This is the last of the twelve band activations; its
output feeds phase 36, whose output E36 packs (and this is where the A-quad
chain from the normed tile ends).

#### Shape of the whole MLP band (E12–E36)

```
E12  normed tile N (24 b32, %r15561…%r15584) — RMS norm of the attention output
  ├─ pack → P13 A (s9784)     P13 : A = N,        B = W+86592,  C = bias@89664   → D13
  │      E13: clamped cubic(D13) pack → P14 A;  D12 + table@92800 → P14 C;  P14 B = W+89728
  ├─ pack → P15 A (s10126)    P15 : A = N,        B = W+92992,  C = bias@96064   → D15
  │      E15: clamped cubic(D15) pack → P16 A;  P16 B = W+96128, C = D14
  ├─ pack → P17 A (s10373)    P17 : A = N,        B = W+99240,  C = bias@102272  → D17
  │      E17: clamped cubic(D17) pack → P18 A;  P18 B = W+102336, C = D16
  ⋮  (twenty-four phases, twelve repetitions of the pair)
  ├─ pack → P35 A (s12606)    P35 : A = N,        B = W+155072, C = bias@158144  → D35
  │      E35: clamped cubic(D35) pack → P36 A
  └─────────────────────────  P36 : A = act(D35), B = W+158208, C = D34           → D36
```

So the band is **twelve branches off one shared normed tile**, each branch being
the pair (odd phase: 48×96×32 banded GEMM with clamped cubic and a per-column
bias; even phase: 16×96×32 GEMM whose C is the running residual `D12 + table`
and whose output is fed to the next odd phase as `B`), and the residual chain
runs `D12 → D14 → D16 → … → D36`.  `D36` (12 registers, 16×96) is the band's
output and is the input to E36.

---

### E36 — s12829–s13698: arena scatter, shared staging, phase 37 A/B/C

870 statements.  Four sequential moves.

**(a) requantise, s12829–s12864.** 24 `cvt.rn.satfinite.e4m3x2.f16x2` + 12
`mov.b32` → `%r183…%r194` (12 b32), the requantised 16×96 output of phase 36.
Each pair of e4m3 registers is built from `(reg0, reg2)` / `(reg1, reg3)` of a
phase-36 D brace group:

```
s12829 L32388: cvt.rn.satfinite.e4m3x2.f16x2 %rs4341, %r15991
s12830 L32391: cvt.rn.satfinite.e4m3x2.f16x2 %rs4340, %r15971
s12831 L32393: mov.b32 %r183, {%rs4340, %rs4341}
```

**(b) plane-arena scatter to `param+48`, s12865–s13009.**

```
s12865 L32471: ld.param.u64 %rd383, [%rd1+-32]          // %rd1 = param_0+80 → param_0+48
s12866 L32472: cvta.to.global.u64 %rd5, %rd383
s12868 L32476: shr.u32 %r195, %r16115, 2                // g = laneid>>2
s12869 L32477: and.b32 %r196, %r16115, 3                // t = laneid&3
s12870 L32478: shr.u32 %r16389, %r16115, 5
s12872 L32480: add.s32 %r16391, %r182, %r16389          // y = %r182 + (laneid>>5)
s12878 L32486: add.s32 %r16392, %r1, %r16390            // x = %r1 + (g&7)
s12884 L32492: mad.lo.s32 %r197, %r5, %r16391, %r16393  // %r5 = 8*ex
s12885 L32493: add.s32 %r16394, %r196, %r197            // + t (+ guard bit)
s12886 L32494: selp.b32 %r198, -1, %r16394, %p1
s12887 L32495: setp.lt.s32 %p197, %r198, 0
s12888 L32496: @%p197 bra $L__BB1_196
s12889 L32498: mul.wide.s32 %rd384, %r198, 4
s12890 L32499: add.s64 %rd385, %rd5, %rd384
s12891 L32500: st.global.u32 [%rd385], %r183
```

Twelve `st.global.u32`, all with `idx = 8·ex·y + 8·x + d` in units of 4 bytes:

| # | stmt | value | index base | guard | d |
|---|---|---|---|---|---|
| 1 | s12891 | `%r183` | `%r197 = 8*ex*(%r182 + (laneid>>5)) + 8*(%r1 + (g&7))` | p1 | `t` |
| 2 | s12900 | `%r184` | `%r197` | p1 | `t\|4` |
| 3 | s12933 | `%r185` | `%r201 = 8*ex*(%r182 + ((g+8)>>3)) + 8*(%r1 + ((g+8)&7))` | p2 | `t` |
| 4 | s12941 | `%r186` | `%r201` | p2 | `t\|4` |
| 5 | s12950 | `%r187` | `%r204 = %r197 + %r6` | p1 | `t` |
| 6 | s12958 | `%r188` | `%r204` | p1 | `t\|4` |
| 7 | s12967 | `%r189` | `%r207 = %r201 + %r6` | p2 | `t` |
| 8 | s12975 | `%r190` | `%r207` | p2 | `t\|4` |
| 9 | s12984 | `%r191` | `%r210 = %r204 + %r6` | p1 | `t` |
| 10 | s12992 | `%r192` | `%r210` | p1 | `t\|4` |
| 11 | s13001 | `%r193` | `%r213 = %r207 + %r6` | p2 | `t` |
| 12 | s13009 | `%r194` | `%r213` | p2 | `t\|4` |

with `%r5 = 8*ex` (s15), `%r6 = ey·8·ex` (s16, one image plane), `%r182 =
%r2 + 2·%tid.z` (s9185), `g = laneid>>2`, `t = laneid&3`.  Each store block is
guarded by `p1`/`p2` = `(y<0)|(y≥ey)|(laneid>63)|(x<0)|(x≥ex)` (s12874–s12882,
s12916–s12924) and a guard failure stores `-1` as the index, which the following
`setp.lt.s32 %p, %idx, 0` + `bra` turns into "skip the store" (the pattern of
s12886–s12888).  Note the index also has the guard *bit* added
(`add.s32 %r16394, %r196, %r197` at s12885 uses `%r196` — a predicate register
read as 0/1), which the `selp`/`bra` pair makes harmless.

**(c) shared staging, s13013–s13131.**

```
s13013 L32645: bar.sync 0
s13014 L32647: mov.u32 %r16412, %laneid
s13018 L32652: add.s32 %r16414, %r17741, %r16413   // %r17741 = 2*%tid.z, %r16413 = laneid>>5
s13022 L32656: shl.b32 %r16415, %r16414, 5
s13023 L32657: and.b32 %r16416, %r16412, 28
s13025 L32659: or.b32 %r218, %r217, %r16417
s13031 L32666: st.shared.u32 [%r16420], %r183
s13039 L32676: st.shared.u32 [%r16424], %r184       // index + 256
s13059 L32698: st.shared.u32 [%r16434], %r185       // group 2, base from (laneid>>2)+8
s13067 L32708: st.shared.u32 [%r16438], %r186
   …  +512, +768, +1024, +1280 variants for %r187…%r194 …
s13131 L32788: st.shared.u32 [%r16470], %r194
```

Twelve `st.shared.u32`, word index for the first six
`(2·%tid.z)·32 + (laneid & 28)` and `+256`, and for the last six
`((2·%tid.z)+((laneid>>2)+8)>>3)·32 + (((laneid>>2)+8)<<2 & 28)` and `+256`
(s13043–s13058, s13061–s13074).  See §6 U3 for the address collision this
produces.  A second `bar.sync 0` follows at s13136.

**(d) phase 37's A operand from shared, s13138–s13625.**

```
s13138 L32798: ld.param.v2.u32 {%r16472, %r16473}, [%rd489+-80]     // param_0+0 = (ex, ey)
s13143 L32804: shr.u32 %r16475, %r16471, 2
s13146 L32807: and.b32 %r16478, %r16477, -64
s13148 L32809: and.b32 %r16480, %r16479, 24
s13150 L32811: or.b32 %r16482, %r16476, %r16481
s13151 L32812: shl.b32 %r16483, %r16482, 2
s13152 L32813: add.s32 %r16485, %r3216, %r16483
s13153 L32814: ld.shared.u32 %r235, [%r16485]
s13165 L32826: ld.shared.u32 %r236, [%r16490]
   …  48 loads total, %r235…%r282 …
s13625 L33286: ld.shared.u32 %r282, [%r16812+144]
```

48 `ld.shared.u32` = 12 A quads (one `{a0,a2,a1,a3}`-ordered quad each).  The
first word's index is

```
word = ((laneid<<2) & ~63) | ((laneid<<1) & 24) | ((laneid>>2)&3)
     = 64·(laneid>>4) + 8·((laneid>>2)&1) + ((laneid>>2)&3)
byte = smem + 4·word
```

and the remaining 47 use the same base plus `+4, +8, +12, +16, +20` and a
divisor-by-24 decomposition of `(laneid&3)|4` / `|8` / `|12` … (`s13156
mul.wide.u16 %r16486, %rs4437, -21845` / `s13157 shr.u32 %r16487, %r16486, 20`
computes `⌊x/24⌋`, `s13159–s13160` the remainder, `s13161–s13163` select the
byte plane), with the whole block guarded by `setp.gt.u32 %p233, %r17795, 127`
(s13140) and by bounds checks against `(ex, ey)` at s13287–s13295.

**(e) phase 37's B, s13634–s13687.**

```
s13634 L33296: mul.lo.s32 %r17412, %r17795, 384        // %r17795 = %tid.z<<5
s13635 L33297: cvt.u64.u32 %rd433, %r17412
s13636 L33298: add.s64 %rd434, %rd2, %rd433
s13638 L33302: mul.wide.u32 %rd435, %r16825, 16
s13639 L33303: add.s64 %rd436, %rd434, %rd435
s13640 L33304: add.s64 %rd409, %rd436, 161280
s13641 L33306: ld.weak.global.ca.v4.u32 { %r16826,%r16827,%r16828,%r16829},[%rd409]
   …  24 loads at 512 stride, last at +173056 …
s13687 L33398: ld.weak.global.ca.v4.u32 { %r16918,%r16919,%r16920,%r16921},[%rd432]
```

24 weight fragments at `W + 12288·%tid.z + 16·laneid + 161280 + 512·i`, `i =
0…23` = 12288 bytes = 48 B pairs = phase 37's B.  The whole block is inside the
loop `$L__BB1_244` (s13633) which advances `%r17795` by 128 and branches back at
s13847–s13850 while `%r17795 < 128`; with `%r17795 = %tid.z<<5` this is a single
iteration for `%tid.z ≤ 3`.

**(f) phase 37's C, s13688–s13698.** 4 `ld.global.u32` at
`W + ((laneid*4)&12) + {210432, 210448, 210464, 210480}` → `%r16932, %r16942,
%r17172, %r17182`, each used as `{%rX, %rX}` — a per-column bias replicated over
the 16 rows, seeding the four chains of phase 37.

**Inputs:** phase 36's 24 D, `%r3216`, `W`, `(ex,ey)`.  **Outputs:** the 12 arena
stores (side effect), `%r183…%r194`, phase 37's 12 A quads, 48 B pairs, 4 C
seeds.  **Memory:** 12 global stores, 12 shared stores, 48 shared loads, 24
weight loads, 4 table loads, 2 barriers.

---

### E37 — s13747–s13870: packs + 4 stores to `param+56`

124 statements.

**(a) packs, s13747–s13756.** 8 `cvt` + 3 `mov.b32` from phase 37's chain-final
D registers → `%r287` (2 b32), `%rs4534/%rs4535`, `%rs4540/%rs4541`:

```
s13747 L33750: cvt.rn.satfinite.e4m3x2.f16x2 %rs4534, %r17143
s13748 L33753: cvt.rn.satfinite.e4m3x2.f16x2 %rs4535, %r17383
s13749 L33756: cvt.rn.satfinite.e4m3x2.f16x2 %rs4537, %r17393
s13750 L33759: cvt.rn.satfinite.e4m3x2.f16x2 %rs4536, %r17153
s13751 L33761: mov.b32 %r287, {%rs4536, %rs4537}
```

**(b) the four stores to `param+56` (`O`), s13757–s13845.**

```
s13763 L33783: shr.u32 %r17418, %r16473, 1         // ey>>1
s13764 L33784: shr.u32 %r17419, %r16472, 1         // ex>>1
s13765 L33785: shl.b32 %r17420, %r17419, 3         // 8*(ex>>1)
s13766 L33786: mul.lo.s32 %r17421, %r17418, %r17420
s13767 L33787: shr.u32 %r17422, %r17795, 5         // %tid.z (loop-carried)
s13768 L33788: mul.lo.s32 %r290, %r17421, %r17422
s13769 L33789: add.s32 %r17423, %r283, %r17416     // y = (%r2>>1) + (laneid>>4)
s13776 L33796: mad.lo.s32 %r17424, %r17420, %r17423, %r290
s13777 L33797: add.s32 %r17425, %r284, %r17417     // x = (%r1>>1) + ((laneid>>2)&3)
s13783 L33803: add.s32 %r291, %r17426, %r17424
s13784 L33804: or.b32 %r17427, %r17415, %r291      // + t
s13791 L33812: st.global.v2.u16 [%rd445], {%rs4534, %rs4535}
s13802 L33825: st.global.u32 [%rd448], %r287
s13836 L33861: st.global.u32 [%rd451], %r288
s13845 L33872: st.global.v2.u16 [%rd454], {%rs4540, %rs4541}
```

`%r283 = %r2>>1` (s13628), `%r284 = %r1>>1` (s13631), `%r17416 = laneid>>4`,
`%r17417 = (laneid>>2)&3`, `t = laneid&3`; the second pair recomputes
`y' = (%r2>>1) + (((laneid>>2)+8)>>2)`, `x' = (%r1>>1) + (((laneid>>2)+8)&3)`
(s13811–s13822) at the same base `%r290`.  Extents are **halved** (`ex>>1`,
`ey>>1`) and the destination is `param+56`, not the plane arena.  Guards are
`p6 = (y<0)|(y≥(ey>>1))|(laneid>63)|(y-constrained tid.z>127)|(x<0)|(x≥(ex>>1))`
(s13762–s13781), with the same `selp −1` + `bra` skip.

**(c) loop close and phase 38's operands, s13846–s13870.**

```
s13847 L33875: add.s32 %r299, %r17795, 128
s13848 L33876: setp.gt.u32 %p256, %r17795, -129
s13849 L33877: mov.u32 %r17795, %r299
s13850 L33878: @%p256 bra $L__BB1_244
s13852 L33881: mov.u32 %r17758, _ZZ…E4smem
s13858 L33889: add.s64 %rd455, %rd459, 210688
s13859 L33891: ld.weak.global.ca.v4.u32 { %r17443,%r17444,%r17445,%r17446},[%rd455]
s13861 L33895: ld.weak.global.ca.v4.u32 { %r17447,…},[%rd456]     // +211200
s13863 L33899: ld.weak.global.ca.v4.u32 { %r17451,…},[%rd457]     // +211712
s13869 L33908: ld.global.u32 %r17465, [%rd463+212224]
s13870 L33909: ld.global.u32 %r17475, [%rd463+212240]
```

Three weight fragments at `W + 16*laneid + {210688, 211200, 211712}` = phase
38's 6 B pairs, and 2 `ld.global.u32` at `W + ((laneid*4)&12) + {212224, 212240}`
= phase 38's C seeds (each used as `{%rX, %rX}`).

---

### E38 — s13877–s14199: the final epilogue

323 statements: 4 `st.shared.u32` (staging phase 38's D), 5 `ld.shared.u32` +
1 `ld.shared.v2.u16` (the softmax window), 1 real row max over 5 values, 5
`ex2.approx.ftz.f32` + 5 `div.approx.ftz.f32`… no — 9 `tex.base.2d.v4.f16.s32`,
3 `sust.b.2d.v4.b16.zero`, 4 `shfl.sync.down.b32`, 3 `ld.param.u64`, 1 `ret`.

**(a) stage phase 38's D for the softmax, s13877–s13907.**

```
s13877 L33952: bar.sync 0
s13878 L33954: mov.u32 %r17516, %laneid
s13882 L33959: add.s32 %r17521, %r304, %r17746            // g + 16*tid.z
s13883 L33960: and.b32 %r17522, %r17521, 1073741816       // & 0x40000008
s13884 L33961: or.b32 %r17523, %r17522, %r17518           // | (laneid>>2)&7
s13885 L33962: mad.lo.s32 %r17524, %r17523, 6, %r305      // *6 + t
s13887 L33964: add.s32 %r306, %r17758, %r17525            // *4 → byte address
s13888 L33965: st.shared.u32 [%r306], %r17496
s13892 L33970: st.shared.u32 [%r306+16], %r17506
s13905 L33984: st.shared.u32 [%r308], %r17497
s13907 L33987: st.shared.u32 [%r308+16], %r17507
```

Four words staged: the phase-38 chain-final pairs `{%r17496,%r17497}` and
`{%r17506,%r17507}`, at a 6-word row pitch (`*6`) plus `t`, in two banks 16 bytes
apart for the two rows `g` and `g+8`.

**(b) bounds gate, s13909–s13939.** `%r311 = %r17766 + (tid.x&7)` with
`%r17766 = 8*ctaid.x - rs126` (the x origin) and `%r312 = %r17762 +
(tid.x>>3)`; `p265 = (tid.z>1)|((%r312|%r311)<0)|(%r311≥ex)|(%r312≥ey)` (s13931–s13939)
skips the whole softmax/gather.  Note `(tid.x + 32*tid.z)` is first reduced mod 8
into `%r309 = ... & -8` and `%r310 = ... >>3` (s13922–s13928).

**(c) the softmax window, s13940–s14038.**

```
s13946 L34028: ld.shared.u32 %r17554, [%r17703]
s13948 L34030: ld.shared.u32 %r17555, [%r17703+4]
s13950 L34032: ld.shared.u32 %r17558, [%r17703+8]
s13951 L34033: ld.shared.u32 %r17561, [%r17703+12]
s13952 L34034: ld.shared.v2.u16 {%rs4674, %rs4675}, [%r17703+20]
s13954 L34036: ld.shared.u32 %r17575, [%r17703+16]
s13957 L34040: cvt.f32.f16 %f865, %rs4542
s13958 L34043: mul.ftz.f32 %f893, %f865, 0fBFB8AA3B     // -log2 e
s13959 L34044: ex2.approx.ftz.f32 %f894, %f893
s13960 L34045: add.ftz.f32 %f895, %f894, 0f3F800000
s13962 L34047: div.approx.ftz.f32 %f866, %f896, %f895   // %f896 = 1.0
s13964 L34053: max.f16x2 %r17553,%r17554,%r17555
s13965 L34057: max.f16x2 %r17556,%r17553,%r17558
s13966 L34061: max.f16x2 %r17559,%r17556,%r17561
s13968 L34066: max.f16 %rs4544,%rs4545,%rs4546
s13969 L34070: max.f16 %rs4547,%rs4544,%rs4549
s13970 L34073: mov.b32 %r17576, {%rs4547, %rs4547}
s13971 L34075: sub.f16x2 %r17562,%r17554,%r17576
s13974 L34083: mul.ftz.f32 %f897, %f867, 0f3FB8AA3B     // +log2 e
s13975 L34084: ex2.approx.ftz.f32 %f869, %f897
```

**This epilogue does have a real row maximum.**  `max.f16x2` twice and `max.f16`
twice (s13964–s13969) reduce the *low* halves of the five words
`%r17554, %r17555, %r17558, %r17561, %r17575` (the last through `%rs4549`) into
`%r17576`, which is broadcast to both halves and subtracted
(`s13971: sub.f16x2 %r17562,%r17554,%r17576`).  Then a textbook f16 softmax:
`2^((x−max)·log2e)` per element through `ex2.approx.ftz.f32` (s13974–s14029),
the row sum accumulated in f32 (s13979, s13983, s13996, s14009, s14022, s14030),
inverted with `div.approx.ftz.f32 1.0/…` (s14031
`div.approx.ftz.f32 %f885, %f896, %f914`), and the five f16x2 registers scaled
(s14034–s14038 `mul.f16x2 %r17577…%r17589,%…,%r17591`).

**(d) the 3×3 texture gather, s14039–s14121.**  The nine
`tex.base.2d.v4.f16.s32` (s14058, s14060, s14077, s14079, s14087, s14089, s14102,
s14104, s14117) sample the surface `param+88` (`%rd480`, s14056) at the nine
combinations of `(x−1, x, x+1)` × `(y−1, y, y+1)`:

```
s14044 L34234: setp.lt.s32 %p267, %r311, 1
s14045 L34235: selp.b32 %r17640, 1, %r17706, %p267     // xm
s14051 L34241: selp.b32 %r17609, 1, %r17709, %p269     // ym
s14055 L34245: selp.b32 %r17642, 1, %r17710, %p271     // xp
s14058 L34249: tex.base.2d.v4.f16.s32 {%rs4569,%rs4570,%rs4571,%rs4572}, [%rd464, {%r17640,%r17609}]
```

`%rd464…%rd480` are all `mov.u64` copies of `%rd480` (s14057, s14059, …).  The
sampled values are combined with the five softmax weights by `fma.rn.f16x2`
(s14062, s14064, s14066, s14081, s14083, s14085, s14091, s14093, s14095, s14106,
s14109, s14112) and the 2×2 accumulator halves folded with `add.f16` (s14114,
s14115, s14116), then one more shared value `%rs4610` folded with `fma.rn.f16`
(s14119–s14121).

**(e) sigmoid, difference, scale, cross-lane sum, s14122–s14179.**

```
s14122 L34385: cvt.f32.f16 %f886, %rs4674
s14123 L34388: mul.ftz.f32 %f915, %f886, 0fBFB8AA3B
s14125 L34390: add.ftz.f32 %f917, %f916, 0f3F800000
s14126 L34391: div.approx.ftz.f32 %f887, %f896, %f917   // sigmoid
s14134 L34409: mul.f16 %rs4630,%rs4543,%rs4614
s14137 L34421: sub.f16 %rs4645,%rs4622,%rs4636          // acc - sigmoid*acc
s14138 L34424: mov.f32 %f892, 0f3E800000                // 0.25
s14145 L34444: mul.f16x2 %r17658,%r17659,%r17660        // * 0.25
s14152 L34463: shfl.sync.down.b32 %r17662,%r17658,%r17683,%r17665,%r17693   // delta 1
s14153 L34467: add.f16x2 %r17667,%r17658,%r17662
s14158 L34478: shfl.sync.down.b32 %r17671,%r17667,%r17691,%r17674,%r17693   // delta 8
s14159 L34482: add.f16x2 %r17676,%r17667,%r17671
s14168 L34506: add.f16 %rs4657,%rs4651,%rs4656
s14177 L34530: add.f16 %rs4663,%rs4657,%rs4662
```

`sigmoid(v) = 1/(1 + 2^(−log2e·v))` on two more shared values
(`%rs4674/%rs4675`, the v2.u16 at `+20`), then three `acc − sigmoid·acc`
differences scaled by `0.25` (f16), then a two-step partial warp reduction:
`shfl.sync.down` delta 1 (`%r17683`, s13956) then delta 8 (`%r17691`, s13949),
each followed by an `add.f16x2`; the scalar path repeats the same two deltas with
`add.f16` (s14168, s14177).  Segmask `%r17665`/`%r17674` is `(WARP_SZ<<8)−8192|31`
= 31 in both, membermask `%r17693 = −1`: a partial (not five-step) reduction.

**(f) the three surface writes, s14178–s14199.**

```
s14178 L34533: ld.param.u64 %rd484, [%rd491+120]
s14179 L34534: ld.param.u64 %rd487, [%rd491+128]
s14180 L34535: shl.b32 %r17697, %r311, 3
s14181 L34537: sust.b.2d.v4.b16.zero [%rd482, {%r17697,%r312}], {%rs4639,%rs4642,%rs4645,%rs4676}
s14182 L34540: sust.b.2d.v4.b16.zero [%rd484, {%r17697,%r312}], {%rs4630,%rs4633,%rs4636,%rs4677}
s14183 L34542: and.b32 %r17724, %r313, 1
s14185 L34544: mov.pred %p279, 0
s14186 L34545: xor.pred %p280, %p278, %p279
s14187 L34546: @%p280 bra $L__BB1_260
s14191 L34551: shl.b32 %r17729, %r17728, 2
s14192 L34552: and.b32 %r17725, %r17729, -8
s14197 L34558: sust.b.2d.v4.b16.zero [%rd487, {%r17725,%r17726}], {%rs4678,%rs4679,%rs4663,%rs4681}
s14199 L34562: ret
```

Three four-channel b16 surface writes to `param+112`, `param+120` and
`param+128`; the third is taken only when `(x|y)` is odd
(`and.b32 %r17724, %r313, 1` at s14183, `@%p280 bra`) and stores
`mov.u16 %rs4681, 0` (s14196) in its fourth channel.  `%r17725 = (x<<2) & −8`,
`%r17726 = y>>1` for that one, versus the plain `(x<<3, y)` of the first two.

**Inputs:** the four staged phase-38 D words, `%rd480` (texture), `(ex,ey)`,
`%r1/%r2/%tid.z/%tid.x`, constants `0fBFB8AA3B`, `0f3FB8AA3B`, `0f3F800000`,
`0f3E800000`.  **Outputs:** none (the kernel returns).

---

## 4. The plane arena (`param_0+48`) and the other `ld.param` buffers

The four `ld.param` that matter, quoted verbatim:

```
s4    L1032: add.s64 %rd1, %rd8, 80
s5    L1033: ld.param.v2.u16 {%rs126, %rs127}, [cuda_dldn_engine_swin_enc2_kernel_param_0+80]
s8    L1036: ld.param.u64 %rd2,  [cuda_dldn_engine_swin_enc2_kernel_param_0+40]
s14   L1042: ld.param.v2.u32 {%r322, %r323}, [cuda_dldn_engine_swin_enc2_kernel_param_0]
s17   L1045: ld.param.u64 %rd10, [cuda_dldn_engine_swin_enc2_kernel_param_0+8]
s12865 L32471: ld.param.u64 %rd383, [%rd1+-32]        // = param_0 + 48
s13632 L33293: ld.param.u64 %rd6,   [%rd492+56]       // = param_0 + 56
s13941 L34023: mov.b64 %rd491, cuda_dldn_engine_swin_enc2_kernel_param_0
s13953 L34035: ld.param.u64 %rd482, [%rd491+112]
s14056 L34246: ld.param.u64 %rd480, [%rd491+88]
s14178 L34533: ld.param.u64 %rd484, [%rd491+120]
s14179 L34534: ld.param.u64 %rd487, [%rd491+128]
```

| offset | role | read by | written by |
|---|---|---|---|
| 0 | `(ex, ey)` v2.u32 | s14 (`%r322/%r323`), s13138 (`%r16472/%r16473`) | — |
| 8 | **input image** `I` | prologue: 24 × `ld.global.v2.u16 [%rd12]` (s176, s328, … s3316) via `%rd4` | — |
| 40 | **weight image** `W` | every weight load, §1.1 | — |
| 48 | **plane arena** `S` | **never** | 12 × `st.global.u32` (s12891…s13009) |
| 56 | phase-37 output buffer `O` | **never** | 4 stores (s13791, s13802, s13836, s13845) |
| 80 | patch origins v2.u16 | s5 (`%rs126/%rs127`), reused at s13910–s13915 | — |
| 88 | input texture | 9 × `tex.base.2d.v4.f16.s32` (s14058…s14117) | — |
| 112, 120, 128 | output surfaces | — | 3 × `sust.b.2d.v4.b16.zero` (s14181, s14182, s14197) |

**Is anything generated into the plane arena?**  Yes, exactly one thing:
E36(b) writes phase 36's requantised 16×96 output — 12 `st.global.u32`, 4 bytes ×
32 lanes × 12 = **1536 bytes per block** — into `S` at
`4·(8·ex·y + 8·x + d)`.  Nothing in the entry ever *reads* `param_0+48`:

```
$ grep -n '%rd5\b\|%rd383' 0015-PREPASS_ENTRYPOINT_NAME.ptx
32471:ld.param.u64 %rd383, [%rd1+-32];
32472:cvta.to.global.u64 %rd5, %rd383;
32499:add.s64 %rd385, %rd5, %rd384;
   … 11 more add.s64 %rd5, all feeding st.global.u32 …
```

`%rd5` appears only in the twelve `st.global.u32` (there is **no** `ld.*` with
`%rd5` as base) and `%rd383` only at s12865.  So the arena write is a pure
side effect for the next launch; the same 12 registers are *also* consumed
inside this kernel via the shared staging of E36(c).

The one `surface-write`-style marker the brief attributes to a "phase 30
epilogue" (`surface-write:32@s25247`) has no counterpart here: s25247 does not
exist (the entry ends at s14199) and **E30 is a 60-statement load/pack epilogue
with no store of any kind** (§3/E30; verified by enumerating every `st.*` in the
entry — the only `st.global` are s12891–s13009 and s13791–s13845, and the only
`st.shared` are the prologue's six, E8's four, E36's twelve and E38's four).
That marker belongs to a different linearisation — see §6 U6.

## 5. Cross-lane reductions worth calling out

Introduction: the entry contains exactly **32** `shfl.sync.bfly.b32` and **4**
`shfl.sync.down.b32`; 8 of the former are in the prologue, 16 in E7, 4 in E12 and
4 in E38.  Each butterfly uses the shared delta operands `%r3037 = 1` (s3318),
`%r3046 = 2` (s3611) and membermask `%r3048 = −1` (s3604), with a segmask built as
`((WARP_SZ<<8) − 8192) | 31` = 31.

1. **The prologue's RMS reduction** (s3605, s3612, s3621, s3627, s3637, s3643,
   s3652, s3658): four butterfly pairs over lane bits 0 (`%r3037`) and 1
   (`%r3046`), each followed by `add.f16x2` and a final `add.f16`
   (`s3646 L5446: add.f16 %rs516,%rs517,%rs518`) — four row sums from the
   prologue's 96 squared input values.
2. **E7's row sums** (s7223–s7344): 16 `shfl.sync.bfly.b32` = eight such pairs,
   each followed by `add.f16x2` and a final `add.f16` of the two halves — 8 row
   sums over 96 columns of the score window.
3. **E12's RMS reduction** (s9308–s9330): the same two butterfly steps
   (`delta 1` at s9308, `delta 2` at s9314) over the four lanes with equal
   `laneid>>2`, folding the 96 squares of one row; then `add.f16` of the two
   column halves (s9317, s9332).  This is the whole cross-lane traffic of the
   norm.
4. **E36's and E38's `bar.sync` pairs** (s8910/s8968, s13013/s13136,
   s13877/s13920): block-wide, guarding the shared tiles.  The seventh
   `bar.sync` is the prologue's, s4588.
5. **E38's partial lane sums** (s14152–s14177): four `shfl.sync.down.b32`, two
   on the f16x2 path (`delta 1` = `%r17683`, `delta 8` = `%r17691`) and two on the
   scalar path, segmask 31, membermask −1.

There is no `red.*` or `atom.*` anywhere in the entry, and `movmatrix` occurs only
in E7 (the 48 of §3/E7(f)).

## 6. Uncharacterised

**U1. The (n,k) indexing of the 512-byte B tiles of phases 13…35 and 37.**
The *geometry* of these phases is not in doubt: the chain fixes `m·k = |A|`,
`n·k = |B|`, `m·n = chains`, `k =` chain length, so phase 13 (and 15…35) is
`m=1, n=4, k=3` → M=16, N=32, K=96 and phase 37 is `m=1, n=4, k=12` → M=16,
N=32, K=384 (§2.2); every row then satisfies `mma = (M/16)(N/8)(K/32)`.  What I
cannot derive is *which* (n-tile, k-step) each 512-byte tile holds.  For phase 13
the 12 B pairs sit in six tiles at `W + 86592 + 512*i` (E12's six loads), and the
emitted chains pair them as chain 0 = tiles {0,2,4} at byte `+0`, chain 1 = tiles
{0,2,4} at byte `+8`, chain 2 = tiles {3,4,5} at `+0`, chain 3 = tiles {3,4,5} at
`+8` (mma s9820/s9822/s9824, s9821/s9823/s9825, s9826/s9828/s9830,
s9827/s9829/s9831).  If a chain is one n-tile's three k-slices, its three tiles
would have to be the same tile — they are not; so either the weight image's tile
stride is not the `(n,k)`-major layout `rr_layer_spec.py` assumes
(`:189–192`, `:491–498`), or `m·n` counts something other than the chains.  The
register/byte evidence above is exact and sufficient to reproduce the *operands*;
only their (n,k) naming is unresolved.  *Missing:* the `_NAME_prep` permutation
that built the weight image, or a run of the kernel with a weight-image dump.

**U2. The identity of the 12288-byte score bias.**  Phase 7 adds the f16 table
at `W + 6336 … W + 18624` as its mma C operand, one value per element of the
64×96 score, tile-major at 512 bytes per output tile (§3/E6(a)).  Whether it is a
relative-position bias, a learned score bias, or something else cannot be decided
from the PTX: it is an opaque blob in a permuted weight image.  *Missing:* the
weight image's layout description (the `_prep` permutation) or the parameter
names from the model file.

**U3. E36(c)'s shared staging produces 4-way write collisions.**  The word index
is `(2·%tid.z)·32 + (laneid & 28)` for the first six stores and
`((2·%tid.z) + ((laneid>>2)+8)>>3)·32 + (((laneid>>2)+8)<<2 & 28)` for the last
six, plus `+256` for the alternating pair.  `laneid & 28` takes only the eight
values 0,4,…,28, so lanes 0…3, 4…7, … all write the *same* word, with *different*
data (`%r183` vs `%r184` are the two halves of the same D pair, and `%r185` is
the other half of `%r183`'s source).  Either the four colliding lanes are
guaranteed to hold equal data in this kernel's inputs (unlikely — they are
distinct e4m3 packs of distinct D registers) or the staging is racy and the
issued values come from whichever lane wins.  I could not decide which from the
PTX.  *Missing:* a run with a shared-memory dump, or the high-level source of the
phase-36→37 hand-off.  (The corresponding *read* side, E36(d), is 48
`ld.shared.u32` with the swizzle of §3/E36(d) and is complete.)

**U4. E8's shared slot layout.**  E8 stages four words at
`smem + 2048*%tid.z + 16*laneid + {0,512,1024,1536}` (s8911–s8967), but phases 9,
10, 11 read single quads at `smem + 16*laneid + {2048, 4096, 6144}` (s8974,
s9094, s9125) and phase 12 at `smem + %tid.z*512 + 16*laneid` (s8969–s8974 uses
`%tid.z*512` for phase 9's *other* read).  For `%tid.z = 0` the written slots are
{0,512,1024,1536} while phase 9 reads 2048, phase 10 4096, phase 11 6144 —
disjoint.  The union over `%tid.z ∈ {0,1,2,3}` of the writes is the 16 slots
{0…15}, which *does* contain 4, 8 and 12 (i.e. the reads 2048/4096/6144 are
slots 4/8/12).  This is consistent only if `blockDim.z = 4` and the four z-planes
write the block's shared tile cooperatively; the module has **no `.maxntid`
declaration** so I cannot confirm the launch shape.  *Missing:* the harness's
launch configuration (gridDim/blockDim), or whatever `.maxntid`/`.reqntid` the
unstripped cubin carries.

**U5. The `%tid.z` weight-slab strides are mutually inconsistent.**
Phases 1–7 index the weight image through `W + 21504*%tid.z + 16*laneid`
(s4589–s4590, s4613–s4615); phases 9–36 index it through `W + 16*laneid` with no
`%tid.z` term at all (s8976–s8978); phase 37 through `W + 12288*%tid.z +
16*laneid` (s13634–s13639); phase 38's bias again without a `%tid.z` term
(s13688–s13694).  Since phase 9's very first weight offset (18624) is smaller
than the 21504-byte stride of phases 1–7, the two regions overlap for `%tid.z ≥ 1`
whatever the value of `%tid.z`.  The addresses are as quoted and are sufficient
to reproduce each load; the *reason* they differ is not derivable from the PTX.
*Missing:* the launcher's `gridDim.z` and the weight-image builder.  For
`%tid.z = 0` the touched byte range of the image is `[192, 212244)` (see below).

**Weight-image extent.**  Relative to the pointer named in U5 (i.e. excluding any
`%tid.z` term and the `16*laneid` lane stride):

* the highest weight-fragment offset is `173056 + 16` (phase 37's last
  `ld.weak.global.ca.v4.u32`, s13687, `add.s64 %rd432, %rd436, 173056`);
* the highest scalar-table offset is `212240 + 4` (phase 38's second C seed,
  `ld.global.u32 [%rd463+212240]`, s13870), and `%rd463 = %rd3 +
  ((laneid*4)&12)` so the true last byte is `212240 + 12 + 4 = 212256`.

**The weight image therefore spans at least 212256 bytes**, and per block the
touched range is `[16*laneid + 192, 16*laneid + 212256)`.  Per-`%tid.z` slabs add
`21504*%tid.z` (phases 1–7) and `12288*%tid.z` (phase 37) on top (U5).

**U6. The brief's stale landmarks.**  Three statement indices given with this
ticket do not exist in this linearisation: `s25247` (the entry has 14199
statements), and the range markers "phase 30's epilogue has
`surface-write:32@s25247`".  `s12891` (`surface-write:12`) and `s13791`
(`surface-write:4`) do match, and `s13959` / `s14058` match the reported
`softmax:12` and `texture-read:9`.  So only the `s25247` landmark is foreign; it
is most likely from an enc0-era numbering.  *Missing:* the numbering that
produced `s25247`, if it is meant to be in this layer.

**U7. The semantic names of the per-column tables.**  `W + 0 … 192` (the
prologue's norm gain, 12 loads at `+0,16,…,176`: s4198–s4209), `W + 86208 … 86400`
(E8's residual add), `W + 86400 … 86592` (E12's norm gain), `W + 92800 … 92992`
(E13's residual add), and the four-word biases at `89664`, `96064`, `102272`, …,
`158144`, `210432`, `212224` are all read with the same
`W + ((laneid<<2)&12) + off` pattern and are 96 f16 or 32 f16 wide.  Their *roles*
are computable from the arithmetic (gain in the two RMS norms, additive per-column
bias elsewhere) but their *names* are not.  *Missing:* the weight-image layout or
the model's parameter names.

**U8. Phase 37's K = 384 against a 96-wide A.**  The chain gives phase 37
`m=1, n=4, k=12` → M=16, N=32, K=384, and 48 mma (verified: 4 chains of 12, each
step a distinct A quad and a distinct B pair).  But the A operand is 12 quads =
48 registers read from a shared tile whose source (E36(c)) stages only 12 words
per lane, and E36(a) packs only 12 of phase 36's 24 D registers — i.e. one of the
twelve 16×8 n-tiles of phase 36's output is staged per staged word.  The
mechanical relation between the 12 staged words, the 48 A registers, and a
384-deep k axis is not determined without the swizzle's intent.  *Missing:* the
high-level structure of the phase-36→37 merge, or a shared-memory dump.

**U9. The residual operand added into phase 9's C.**  E8(d) adds a per-column
table (at `W+86208`) to 48 f16 values that are the halves of the prologue's
quantised-input registers `%r2798…%r2845`.  The mechanics are unambiguous (the
defining `mov.b32 {%rs1100, %rs1103}, %r2798` at s3366 has no successor
anywhere in the entry, and `%r2798` is `cvt.rn.f16x2.e4m3x2 %r2798, %rs128` of
the input element loaded at s176).  What is *not* derivable from the PTX is
whether this really is a semantic skip connection from the block input (which
would require the input patch's per-lane layout to coincide with an mma C
operand), or whether the input patch registers happen to alias a value the
compiler could have recomputed from phase 8's D.  *Missing:* the high-level
source, or a numeric run showing whether phase 9's output equals
`pack(D8)·W9 + table + input`.

## 7. Quick acceptance index

| requirement | where |
|---|---|
| phase table for all 38 phases, M/16·N/8·K/32 == mma checked, mismatches named | §2.2: the identity holds on **all 38** once the tool's `M = 16·\|A\|`, `N = 8·\|B\|` are divided by `k = K/32`; the flagged rows 8, 13…35, 37, 38 are re-derived there (the tool's `K` column is right, its `M`,`N` and `full`/`band` label wrong) |
| weight image extent | §6 U5 (212256 bytes, per-block) |
| phases that read no weight operand | §2.2 and §3/E6(b), E7(f): phases 7 and 8 — phase 7's A and B operands both come from E6's packs of phase 5's D, with A frag 0 **bit-identical** to B frag 0 (s4853 vs s4925) and the later ones the same data with the two halves swapped (s4865 vs s4937); phase 8's A is E7's probability packs (s8720 ff.) and its B is E7's `movmatrix` transposes of phase 6's D (s8598 ff.) |
| every epilogue named | §2.1 (38 rows), §3 (38 `###` sections) |
| the norms: mean-subtracting LN or RMS? | §3/E12: raw sum of squares, only `2⁻¹³` added, no mean and no `1/N`; prologue (s3812 ff.) identical structure → **both are RMS norms**, with a per-column gain and the bias deferred to the next GEMM's C |
| attention: does the score have a row max? | §3/E7(b),(c): **no** — all 96 `max.f16x2` are the clamp lower bound `−0.55615234375`, all 96 `min.f16x2` the upper; the only warp shuffles are the 16 `shfl.sync.bfly` row sums |
| attention: the score's bias operand | §3/E6(a): the mma C operand, 48 distinct pairs from the 12288-byte f16 table at `W+6336`, tile-major `512*(4*(n>>1)+m)`, one value per score element; identity → §6 U2 |
| activations | §3/E13(a), E15, E17…E33: `out = x·(0.5 + y·(0.412109375 − 0.0810546875·|y|))`, `y = clamp(x,±2)` — the **same clamped cubic as enc0's E13**, byte-identical constants |
| patch merge / surface writes | §3/E36(b) (12 stores to `param+48`), E37(b) (4 stores to `param+56`), E38(f) (3 `sust` to `param+112/+120/+128`) |
| final texture-reading epilogue | §3/E38: 4 staged words, real 5-value row max, softmax, 9 `tex.2d`, 3×3 `fma` gather, sigmoid, 0.25 scale, two `shfl.sync.down` steps |
| per-launch table into the plane arena | §4: exactly one — E36's 1536 bytes per block; `param+48` is written and never read |
| cross-lane reductions | §5 |
| uncharacterised | §6 U1–U9 |
