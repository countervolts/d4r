# `cuda_dldn_engine_swin_enc5_kernel` — the epilogues of phases 1…61

Companion to `kernels/rr/rr_layer_spec.py enc5`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0018-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_enc5_kernel` (file lines 1018–51175).

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at their first line).  **`LNNNNN`
  is the physical line** of the same instruction in the corpus file, so every
  claim can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and
  the next phase's first `mma`; `Exx` below means *the epilogue of phase xx*.
  The phase table (`rr_layer_spec.py enc5`) gives the ranges.
* `W` denotes the prepared weight image: `ld.param.u64 %rd2,
  [cuda_dldn_engine_swin_enc5_kernel_param_0+40]` (s9, L1037) and
  `cvta.to.global.u64 %rd3, %rd2` (s10, L1038).  Every `ld.weak.global.ca.v4.u32`
  reads `W + 16*laneid + imm`; the table loads (see §1.1) read
  `W + ((laneid<<2)&12) + imm`.  Exception: the B loads of phases 2–10, the
  phase-1 B load and the phase-11 C seeds carry an extra `%tid.z*27648` term
  (see §1.1's last row and §2.3).
* `%rd1` = the parameter-block base at `+80`: `mov.b64 %rd6, param_0` (s1, L1018)
  then `add.s64 %rd1, %rd6, 80` (s4, L1032).  So `[%rd1+n]` is `param_0+80+n`
  and `[%rd1-72]` is `param_0+8`.
* `smem` denotes the module's shared array
  `_ZZ33cuda_dldn_engine_swin_enc5_kernel33DldnEngineSwinEncParamsStructBaseE4smem`,
  declared 16384 bytes by the cut-ABI dump.
* The C/D register layout and the weight-addressing formula are fixed (see the
  settled notes): `C/D reg0 = (row g, cols 2t,2t+1)`, `reg1 = (row g+8, same
  cols)`, `g = laneid>>2`, `t = laneid&3`; the A fragment's four registers are
  `reg0 = row g, k = 2t + (j&1) + 16*((j>>1)&1)`, `reg1 = row g+8, same k`,
  `reg2 = row g, k+8`, `reg3 = row g+8, k+8`.
* Anything I could not pin down from the PTX is in §5 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `W` (weight image, 431124 bytes) | `ld.param.u64 %rd2, [param_0+40]` / `cvta.to.global.u64 %rd3, %rd2` | s9 / s10 |
| `smem` | `mov.u32 %r11623, _ZZ…BaseE4smem` | s7511 (L11833), s7405 (L11639), s20122 (L50821) |
| `%rd1` = `param_0+80` | `mov.b64 %rd6, param_0` + `add.s64 %rd1, %rd6, 80` | s1 / s4 |
| extents `(ex, ey)` | `ld.param.v2.u32 {%r446, %r447}, [param_0]` | s15 (L1043) |
| block offsets `(%rs119, %rs120)` | `ld.param.v2.u16 {%rs119, %rs120}, [param_0+80]` | s5 (L1033) |
| input buffer | `ld.param.u64 %rd8, [%rd1+-72]` = `param_0+8`, then `cvta %rd4` | s22 / s23 |
| plane arena | `ld.param.u64 %rd805, [%rd1+-32]` = `param_0+48`, then `cvta %rd5` | s19817 / s19818 |
| output surface (`sust`) | `ld.param.u64 %rd874, [%rd881+32]` = `param_0+112` | s20134 (L50833) |
| texture handle | `ld.param.u64 %rd872, [%rd881+8]` = `param_0+88` | s20229 (L51030) |
| per-lane weight address | `mul.wide.u32 %rdX, %laneid, 16` + `add.s64 %rdX, %rd3, %rdX` | every load block |
| per-lane weight address, phases 1–11 | `add.s64 %rd138, %rd2, %rd136` with `%rd136 = mul.wide(%tid.z*13824, 2)` then `+ 16*laneid` | s7507–s7508, s7545–s7546 |
| per-lane *table* address | `shl.b32 %rX, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rdX, %rX, 12` → `add.s64 %rdX, %rd3, %rdX` = `W + ((laneid*4) & 12)` | s15140–s15142 and 60 copies |

The last line matters: every `ld.global.v2.u16` / `ld.global.u32` table load in
an epilogue reads `W + ((laneid<<2)&12) + off`, i.e. **the four lanes with equal
`laneid&3` read the same address, so one load instruction covers one 16-byte
word of the table and 32 lanes cover a 512-byte span**.  A 20-load group with
`step 16` therefore covers 320 bytes = 160 f16 = the 160 columns of an
`N=160` GEMM's C operand (one f16 per column, replicated over the 16 rows of a
tile); a 4-load group covers 64 bytes = 32 f16 = the 32 columns of an `N=32`
GEMM's C operand.

### 1.2 Constants and helpers

| register | first definition | f32/f64 literal | f16 value used |
|---|---|---|---|
| `%f476` | s8077 L14399 `mov.f32 %f476, 0f3C8CCB50` | 0.017186790704727173 | `0x2466` = 0.017181396484375 |
| `%fd383` | s8081 L14410 `mov.f64 %fd383, 0dBFE1CC0000000000` | −0.55615234375 | `0xB873` |
| `%fd385` | s8086 L14424 `mov.f64 %fd385, 0d3FE1CC0000000000` | +0.55615234375 | `0x3873` |
| `%f478` | s8092 L14442 `mov.f32 %f478, 0f3F6D6000` | 0.92724609375 | `0x3B6B` |
| `%f480` | s8097 L14456 `mov.f32 %f480, 0f3FB00000` | 1.375 | `0x3D80` |
| `%f1117` | s13641 L29561 `mov.f32 %f1117, 0f3ED306EB` | 0.4121621549129486 | `0x3698` |
| `%f1118` | s13644 L29568 `mov.f32 %f1118, 0f3DA60DD6` | 0.0810810774564743 | `0x2D30` |
| `%f1119` | s13647 L29575 `mov.f32 %f1119, 0f3F000000` | 0.5 | `0x3800` |
| `%f1120` | s13650 L29582 `mov.f32 %f1120, 0f40000000` | 2.0 | `0x4000` |
| `%fd770` | L8337 (prologue) `mov.f64 %fd770, 0d3F20000000000000` | 2⁻¹³ | `0x0800` = 0.0001220703125 |

`%f1117…%f1120` are materialised **inside E21** (s13641–s13652) and re-used by
E23 … E59, which contain no `mov.f32`/`mov.f64` of their own.

* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).
* `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `div.approx.ftz.f32` — always wrapped as
  `cvt.f32.f16 → op → cvt.rn.f16.f32` (each step rounded through f16).
* `mov.u32 %r11547, 0` (s7547, L11871) is the constant-zero C operand used by
  every mma whose `C` field is `{%r11547, %r11547}`.

## 2. Phase table, geometry and weight image

### 2.1 The phase table

`rr_layer_spec.py enc5` prints 61 phases.  The columns are reproduced here with
the M/N/K arithmetic of §2.2:

| # | stmt range | tool M | tool N | tool K | mma | grid | weight bytes | bias bytes | epi |
|---|---|---|---|---|---|---|---|---|---|
| 1 | s7554–s7577 | 96 | 32 | 32 | 24 | full | 320..844 | – | 7 |
| 2 | s7585–s7608 | 96 | 32 | 32 | 24 | full | 5440..5964 | – | 7 |
| 3 | s7616–s7639 | 96 | 32 | 32 | 24 | full | 1344..1868 | – | 7 |
| 4 | s7647–s7670 | 96 | 32 | 32 | 24 | full | 6464..6988 | – | 7 |
| 5 | s7678–s7701 | 96 | 32 | 32 | 24 | full | 2368..2892 | – | 7 |
| 6 | s7709–s7732 | 96 | 32 | 32 | 24 | full | 7488..8012 | – | 7 |
| 7 | s7740–s7763 | 96 | 32 | 32 | 24 | full | 3392..3916 | – | 7 |
| 8 | s7771–s7794 | 96 | 32 | 32 | 24 | full | 8512..9036 | – | 7 |
| 9 | s7802–s7825 | 96 | 32 | 32 | 24 | full | 4416..4940 | – | 7 |
| 10 | s7833–s7856 | 96 | 32 | 32 | 24 | full | 9536..10060 | – | 171 |
| 11 | s8028–s8075 | 64 | 96 | 32 | 48 | full | *(none)* | 10560..22348 | 3844 |
| 12 | s11920–s11967 | 192 | 96 | 96 | 48 | band | *(none)* | – | 274 |
| 13 | s12242–s12261 | 16 | 160 | 32 | 20 | full | 22848..27468 | – | 27 |
| 14 | s12289–s12308 | 16 | 160 | 32 | 20 | full | 50496..55116 | – | 27 |
| 15 | s12336–s12355 | 16 | 160 | 32 | 20 | full | 78144..82764 | – | 27 |
| 16 | s12383–s12402 | 16 | 160 | 32 | 20 | full | 105792..110412 | – | 27 |
| 17 | s12430–s12449 | 16 | 160 | 32 | 20 | full | 133440..138060 | – | 27 |
| 18 | s12477–s12496 | 16 | 160 | 32 | 20 | full | 161088..165708 | – | 27 |
| 19 | s12524–s12543 | 16 | 160 | 32 | 20 | full | 188736..193356 | – | 27 |
| 20 | s12571–s12590 | 16 | 160 | 32 | 20 | full | 216384..221004 | – | 1029 |
| 21 | s13620–s13639 | 80 | 160 | 160 | 20 | band | 222144..226764 | 227264..227312 | 321 |
| 22 | s13961–s13980 | 16 | 160 | 32 | 20 | full | 227328..231948 | – | 92 |
| 23 | s14073–s14092 | 80 | 160 | 160 | 20 | band | 232768..237388 | 237888..237936 | 172 |
| 24 | s14265–s14284 | 16 | 160 | 32 | 20 | full | 237952..242572 | – | 92 |
| 25 | s14377–s14396 | 80 | 160 | 160 | 20 | band | 243072..247692 | 248192..248240 | 172 |
| … | *(the 23/24 pair repeats with a fixed stride)* | | | | | | | | |
| 59 | s19545–s19564 | 80 | 160 | 160 | 20 | band | 418240..422860 | 423360..423408 | 172 |
| 60 | s19737–s19756 | 16 | 160 | 32 | 20 | full | 423424..428044 | – | 299 |
| 61 | s20056–s20065 | 80 | 80 | 160 | 10 | band | 428544..430604 | 431104..431120 | 234 |

The odd phases 25…59 follow the exact pattern of 23 (weight
`= 243072 + 10240*((n-23)/2)`, bias `= 248192 + 10240*((n-23)/2)`, 172
statements), and the even phases 26…60 follow 24 (weight
`= 237952 + 10240*((n-24)/2)`, 92 statements).

### 2.2 The M/N/K arithmetic (`M/16 · N/8 · K/32 == mma`)

Ranks hold for **53 of the 61 phases**.  The eight failures are the banded
phases; in every one of them the tool's `M` (and, for P12, its `K`) is an
overcount because *18 of the 32 A/B register sets are k-slices, not distinct
row/column tiles*.  The chain analysis that fixes them:

| # | mma | tool M×N×K | product | fails? | correct reading | evidence |
|---|---|---|---|---|---|---|
| 1–10 | 24 | 96×32×32 | 24 | – | 96×32×32, dense | s7554: 6 A quads × 4 B pairs × 1 k-step |
| 11 | 48 | 64×96×32 | 48 | – | 64×96×32, dense | s8028: 4 A × 12 B × 1 |
| **12** | 48 | 192×96×96 | 432 | **no** | **64×32×96** = 4 m-tiles × 4 n-tiles × 3 k-steps | 16 chains of length 3, `chains_of` at s11920–s11967 |
| 13–20 | 20 | 16×160×32 | 20 | – | 16×160×32, dense | s12242: 1 A quad × 20 B pairs |
| **21,23,…,59** | 20 | 80×160×160 | 500 | **no** | **16×32×160** = 1 m-tile × 4 n-tiles × 5 k-steps | 4 chains of length 5 (s13620–s13639): A advances one k-slice per step; B advances 512 B per step, 2 n-tiles inside each 512-byte tile |
| 22,24,…,60 | 20 | 16×160×32 | 20 | – | 16×160×32, dense | s13961: 1 A quad × 20 B pairs |
| **61** | 10 | 80×80×160 | 250 | **no** | **16×16×160** = 1 × 2 × 5 | 2 chains of length 5, s20056–s20065 |

For P12 the mma chain is
`s11920 A=%r11180 B=%r11474 → s11922 A=%r11200 B=%r11494 → s11924 A=%r11220
B=%r11514`: three *different* A quads and three *different* B pairs accumulate
into one D, so the three are k-slices (k = 0/32/64) of one 16-row tile.  Chains
0…3 share the A triple and take four different B triples; chains 4…7 use a
second A triple, and so on.  The grid is therefore 4 A-tiles × 4 B-tiles × 3
k-steps = 48, i.e. `(64×96)·(96×32)`, with `M=64, N=32, K=96`.

For phase 21 the chain is
`s13620 A=%r14257 B=%r14064@222144 → s13622 A=%r14277 B=%r14068@222656 →
s13624 A=%r14297 B=%r14072@223168 → s13626 A=%r14317 B=%r14076@223680 →
s13628 A=%r14337 B=%r14080@224192`.  The A quads are the five k-slices of one
16-row tile (K = 5·32 = 160) and the B pairs step 512 bytes, i.e. one k-slice
each; chain 1 uses the same five A quads with the B pairs at `+8` (the second
n-tile of the same 512-byte tile), which is why a 512-byte weight tile is
shared by two chains.  Four chains → N = 32.

The tool's own note on phase 12/21/61 (`grid: band`) is thus correct as a
*selection* remark but its `M`, `N`, `K` columns overcount by the k-slice
factor; the corrected shapes above are what a native replacement has to
produce.

### 2.3 Weight image extent

The prepared weight image is **431124 bytes**.  The largest byte actually
loaded is `W+431120` (`ld.global.u32` at s20055, L50687), and 431120+4 = 431124;
no load reads past it.  The corpus' largest `ld.weak.global.ca.v4.u32` offset is
`W+430592` (s20064, L50745) plus 12 bytes of the `v4` = 430604, inside the same
extent.

One caveat, established in §3/E1–E9: the B loads of phase 1, phases 2–10 and
the C seeds of phase 11 are issued from `W + %tid.z*27648 + 16*laneid + off`
(`%rd138 = %rd2 + mul.wide(%tid.z*13824, 2)`, s7507–s7580), so those
phase-table offsets name four interleaved `%tid.z` blocks rather than a single
contiguous run.  The image is laid out as 27648-byte slots: slot `k` holds the
`%tid.z = k` group's phases-1…11 weights in its first 22848 bytes and one later
phase's weights in its tail (`%tid.z = 0` slot tail = phase 13 at 22848, `k = 1`
tail = phase 14 at 50496, `k = 2` tail = phase 15 at 78144, …).  Phases 12 and
later read from the plain `W + 16*laneid + off` base (e.g. s12036 L25015
`add.s64 %rd641, %rd2, %rd640`).

### 2.4 Phases that read no weight operand

Every `B` operand of every phase resolves to the weight image **except two**:

* **phase 11** — its B registers `%r11474…%r11584` are written by `mov.b32` at
  s11705… (inside E11) and carry the transposed V (see §3/E11); the phase-table
  B column is `?`.
* **phase 12** — its B registers are the same `movmatrix`/pack output written in
  E11; the phase-table B column is `?`.

Phase 11's *C* operand is a weight read (the 12288-byte score bias at
`W+10560…W+22348`, loaded in E10), and phase 12's C operand is the constant
`%r11547 = 0` (s7547), so neither phase reads a `B` from `W`.

## 3. The epilogues

| epilogue | range | stmts | named operation |
|---|---|---|---|
| E1–E9 | s7578–s7832 (nine 7-statement runs) | 7 each | next-B weight loads |
| E10 | s7857–s8027 | 171 | 24 B-fragment loads (phase 11's C bias) + pack phase 10's D |
| E11 | s8076–s11919 | 3844 | softmax of the 64×96 score block, V-transpose, pack |
| E12 | s11968–s12241 | 274 | pack phase 12's D → shared, 160-column table add, next B |
| E13–E19 | s12262–s12570 (seven 27-statement runs) | 27 each | shared A fetch + next-B loads |
| E20 | s12591–s13619 | 1029 | RMS norm (², sum, rsqrt, gain) + packs + next B/C |
| E21 | s13640–s13960 | 321 | clamped cubic activation + 160-column C build + pack |
| E23, E25, …, E59 | 172 statements each | 172 | clamped cubic activation + pack + next B |
| E22, E24, …, E58 | 92 statements each | 92 | pack the phase-20 output × gain → next odd phase's A + next B/C |
| E60 | s19757–s20055 | 299 | pack → plane-arena scatter → next B/C |
| E61 | s20066–s20299 | 234 | shared fade-in, softmax, texture gather, surface write |

---

### E1–E9 — s7578–s7832: next-B weight loads

Nine byte-identical 7-statement runs, each two `ld.weak.global.ca.v4.u32` at
`W + 16*laneid + {off, off+512}` = the *next* phase's B operand.  The first
run:

```
s7578 L12051: mov.u32 %r5444, %laneid
s7579 L12053: mul.wide.u32 %rd140, %r5444, 16
s7580 L12054: add.s64 %rd141, %rd138, %rd140
s7581 L12055: add.s64 %rd94, %rd141, 5440
s7582 L12057: ld.weak.global.ca.v4.u32 { %r5445,%r5446,%r5447,%r5448},[%rd94]
s7583 L12059: add.s64 %rd95, %rd141, 5952
s7584 L12061: ld.weak.global.ca.v4.u32 { %r5449,%r5450,%r5451,%r5452},[%rd95]
```

The offsets, and the phase whose B they are, are:

| run | statement range | offsets | next phase (weight bytes) |
|---|---|---|---|
| E1 | s7578–s7584 | +5440, +5952 | P2 (5440..5964) |
| E2 | s7609–s7615 | +1344, +1856 | P3 (1344..1868) |
| E3 | s7640–s7646 | +6464, +6976 | P4 (6464..6988) |
| E4 | s7671–s7677 | +2368, +2880 | P5 (2368..2892) |
| E5 | s7702–s7708 | +7488, +8000 | P6 (7488..8012) |
| E6 | s7733–s7739 | +3392, +3904 | P7 (3392..3916) |
| E7 | s7764–s7770 | +8512, +9024 | P8 (8512..9036) |
| E8 | s7795–s7801 | +4416, +4928 | P9 (4416..4940) |
| E9 | s7826–s7832 | +9536, +10048 | P10 (9536..10060) |

The run base is `%rd138 = %rd2 + %rd136` (s7580), where
`%rd136 = mul.wide.u32(%r11621, 2)` and `%r11621 = %r9*13824` with
`%r9 = %tid.z` (s19 L1047, s7507 L11827, s7508 L11828).  So the address these
nine runs and the prologue's phase-1 B load use is

```
W + %tid.z*27648 + 16*laneid + <offset>
```

**not** `W + 16*laneid + <offset>`: `%rd138` already carries a `%tid.z*27648`
term.  (`rr_layer_spec.py`'s `track_weights` drops that term because it is the
result of a `mul.wide`, so the phase table's byte offsets for phase 1's B,
phases 2–10's B and phase 11's C bias are relative to
`W + %tid.z*27648`, whereas every other offset in the table is relative to
`W`.  See §5 U7.)

---

### E10 — s7857–s8027: phase 11's C-seed loads + pack of phase 10's D

171 statements, 24 weight loads + 80 packs (histogram: `80 cvt.rn.satfinite`,
`40 mov.b32`, `25 add.s64`, `24 ld.weak.global.ca.v4.u32`).

```
s7861 L13686: ld.weak.global.ca.v4.u32 { %r7686,%r7687,%r7688,%r7689},[%rd112]  // +10560
   …
s7907 L13778: ld.weak.global.ca.v4.u32 { %r7778,%r7779,%r7780,%r7781},[%rd135]  // +22336
s7908 L13781: cvt.rn.satfinite.e4m3x2.f16x2 %rs, %r…                             // first pack
```

* **24 loads** at `W + 16*laneid + 10560 + 512*i`, `i = 0…23` — the phase-11
  score-bias C seeds, byte block `[10560, 22348)` = 12288 bytes = 6144 f16.
  Same layout as enc0's E9: the four C registers of one mma are the four u32 of
  one 16-byte lane slot, tile index `4*(n>>1) + m`, stride 512 bytes.
* **80 packs / 40 b32** at s7908–s8027, whose sources are registers written by
  **phase 9 and phase 10's mma** (traced through the `mov.b32` packs: 38 packs
  come from a register whose newest definition is an `mma` of P9, 2 from P10,
  40 from sources the walk cannot resolve in one hop).  These 40 b32 are
  phase 11's A and B operands.

**Outputs:** phase 11's 4+24 operand registers plus the 24 C-seed registers
(`%r7686…%r7781`).  **Memory:** 24 weight loads, nothing else.

---

### E11 — s8076–s11919: softmax of the 64×96 score block

**Phase 11** = 48 `mma` (s8028–s8075), D registers 96 f16x2 (64 rows × 96
columns, 8 rows × 8 cols per lane).  The epilogue is byte-for-byte the same
algorithm as enc0's E5, with its own constants (they are *not* shared with any
other enc5 epilogue).  Sub-sections, by first occurrence of each op class:

| range | stmts | what |
|---|---|---|
| s8076–s10193 | 2118 | 96 per-element groups: scale, clamp, cubic, exponent extract |
| s10194–s10277 | 84 | in-lane partial row-sum trees (`add.f16x2` ×104 back to s10194) |
| s10278–s10311 | 34 | 16 `shfl.sync.bfly` → 8 row sums (`s10281 L21268` … `s10402 L21572`) |
| s10409–s11174 | 766 | 192 `rcp.approx.ftz.f32` → 96 reciprocals (`s10411 L21594`) |
| s11178–s11654 | 477 | 192 `mul.f16` → 96 probabilities (`s11178 L22839`) |
| s11656–s11703 | 48 | `movmatrix.sync.trans.aligned.m8n8.b16` (V-transpose) `s11656 L23893` |
| s11704–s11918 | 215 | 144 packs (`s11704 L24037`) |

#### (a) per-element exponent, s8076–s10193

```
s8077 L14399: mov.f32 %f476, 0f3C8CCB50
s8080 L14407: mul.f16x2 %r8344,%r7862,%r8343
s8085 L14421: max.f16x2 %r8347,%r8344,%r8349
s8090 L14435: min.f16x2 %r8350,%r8347,%r8352
s8096 L14453: fma.rn.f16x2 %r8355,%r8350,%r8353,%r8358
s8101 L14467: fma.rn.f16x2 %r8359,%r8350,%r8355,%r8362
s8102 L14470: shl.b32 %r11745, %r8359, 5
s8103 L14471: and.b32 %r10264, %r11745, 2145419232   // 0x7FE07FE0
```

Per f16 half `x` of each of the 96 D registers:

```
m  = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)
t  = f16(m*(-m) + 0.92724609375)
u  = f16(m*t    + 1.375)
E  = bits(u)[9:5]      // floor(32*frac(u)) — the top five mantissa bits
F  = bits(u)[4:0]      // the low five mantissa bits
expval = f16 from bits  (E << 10) | (F << 5)     // = 2^(E-15) * (1 + F/32)
```

i.e. `((u_bits << 5) & 0x7FE07FE0)` keeps `u`'s mantissa and drops its sign and
exponent, so the mantissa lands in the exponent field.  This is not `exp` and
not `exp2(x*log2e)`: the exponent is the *cubic*
`32*(0.375 + 0.92724609375*m − m³)` quantised to a 6-bit significand.  The
clamp bounds are the guard: `d/dm [0.92724609375m − m³]` vanishes at
`m = √(0.92724609375/3) = 0.55615…`.

#### (b) in-lane partial sums, s10194–s10277

96 `expval` registers summed in groups with `add.f16x2` (104), giving 4 f16x2
(= 8 f16) per group, i.e. **each lane reduces its 96 values to 24 partial sums
in 12 f16x2 registers**.

#### (c) cross-lane row sums, s10278–s10311

```
s10281 L21268: shfl.sync.bfly.b32 %r…, %r…, 1, %r…, -1
s10290 L21292: add.f16 %rs2318,%rs2319,%rs2320
```

Eight times (one per row `g, g+8` of the four m-tiles): add the two f16x2
partials, butterfly over lane bits 0 and 1 (operands 1 and 2), add the two
halves with `add.f16`, then broadcast into both halves.  Result: **8 row sums**,
covering all 96 columns because the four lanes with equal `laneid&7` hold
different column pairs.

**There is no row maximum anywhere in this epilogue.**  `max.f16x2` (96×) and
`min.f16x2` (96×) are the clamp of (a); the only warp shuffles are the 16
`shfl.sync.bfly` above.

#### (d) reciprocal, s10409–s11174

192 `rcp.approx.ftz.f32` = 96 f16x2 reciprocals, each
`cvt.f32.f16 → rcp.approx.ftz.f32 → cvt.rn.f16.f32` on a row-sum register
(duplicated, so each row sum is inverted once per n-tile of that row):

```
s10408 L21591: mov.b32 {hl, hu}, %r10719
s10411 L21594: rcp.approx.ftz.f32 fl, fl
s10413 L21596: cvt.rn.f16.f32 hl, fl
```

#### (e) probabilities, s11178–s11654

192 `mul.f16` → 96 f16x2 results `p = expval * (1/rowsum)`; the column sums
computed in (b)/(c) cancel the constant `2⁻⁴⁷` of (a), so `p_ij ∝ 2^(32·u_ij)`
and `Σ_j p_ij = 1` up to the `rcp.approx` error.

#### (f) V-transpose, s11656–s11703

**48** `movmatrix.sync.trans.aligned.m8n8.b16` on 48 distinct phase-10 D
registers (`%r7445`, `%r7446`, `%r7455`, … each exactly once).  Each
transposes one 8×8 b16 tile built from the 32 threads' `%r`, i.e. 48 × 64 =
3072 values = the whole 96×32 D tile.

#### (g) packs, s11704–s11918

144 packs in three groups:

1. s11704–s11775 — from the **movmatrix** results → **phase 12's B**
   fragments (`%r11474 … %r11584`, traced: `%r11474` is written by
   `mov.b32 @s11705`, `%r11584` by `@s11768`).
2. s11776–s11893 — from the **probabilities** → phase 12's A fragments.
3. s11894–s11918 — more probability packs whose destinations are not read
   after s11919 (compiler spill).

**Inputs:** phase 11's 96 D f16x2, phase 10's 48 D f16x2, the three `shfl`
operands.  **Outputs:** phase 12's 12 A quads / 12 B pairs and the five
constants of §1.2 (read again by E21–E59).  **Memory:** none.

---

### E12 — s11968–s12241: pack phase 12's D → shared, 160-column table, next B

274 statements.  Sub-sections:

```
s11968 L24876: bar.sync 0
s11974 L24885: cvt.rn.satfinite.e4m3x2.f16x2 %rs3929, %r11208
s11986 L24912: st.shared.v4.u32 [%r11876], {%r11880, %r11879, %r11878, %r11877}
s11999 L24941: st.shared.v4.u32 [%r11876+512], {…}
s12012: st.shared.v4.u32 [%r11876+1024], {…}
s12025: st.shared.v4.u32 [%r11876+1536], {…}
s12026 L25000: bar.sync 0
s12033 L25010: ld.shared.v4.u32 {%r29390, %r29391, %r29392, %r29393}, [%r29389]
s12038 L25018: ld.weak.global.ca.v4.u32 { %r11895,…},[%rd160]   // +22848
   … ten loads at +22848 + 512*i (i = 0…9), last s12056 L25054 (+27456)
s12062 L25063: ld.global.v2.u16 {%rs6407, %rs6408}, [%rd644+221520]
   … twenty loads at +221520 + 16*i, last s12081 L25082 (+221808)
s12083 L25085: add.f16 %rs…, %rs6445, %rs…      // first of 80
```

1. **32 packs** of phase 12's D (s11974–s12025) → 16 b32, stored by **4
   `st.shared.v4.u32`** at `smem + %r9*512 + laneid*16` offsets
   `0, 512, 1024, 1536` (`%r11876 = %r11623 + %r9<<11 + laneid*16`, s11969–
   s11973).  This is the A operand that E13–E19 fetch back.
2. **`bar.sync 0`**, then **one `ld.shared.v4.u32`** at `[%r29389]`
   (`%r29387 = %r11623 + %r9<<9`, s12029) — a 512-byte reload of the same
   staging, used as the source of the table add below.
3. **Ten weight loads** at `W + 16*laneid + 22848 + 512*i` → phase 13's 20 B
   fragments.
4. **Twenty table loads** at `W + ((laneid<<2)&12) + 221520 + 16*i` (320 bytes,
   160 f16 = one f16 per column) → the 160-column vector.
5. **80 `add.f16`** (s12083–s12240) `= table[column] + shared_operand` →
   40 b32 = **phase 13's C seeds** (the epilogue's 96 `mov.b32` are the
   pack/repack plumbing).

**Outputs:** phase 13's 20 B fragments, 40 C seeds, and the shared staging.
**Memory:** 4 shared stores, 1 shared load, 10 + 20 weight loads.

---

### E13–E19 — s12262–s12570: shared A fetch + next-B loads

Seven identical 27-statement runs.  Each begins with one `ld.shared.v4.u32`
that fetches the *next* phase's single A quad from the E12 staging, at a
2048-byte stride:

| run | range | shared load | next-phase A base | B loads (phase, bytes) |
|---|---|---|---|---|
| E13 | s12262–s12288 | s12265 L25628 `[%r29396+2048]` | +2048 | P14, +50496 … |
| E14 | s12309–s12335 | s12312 `[%r29402+4096]` | +4096 | P15, +78144 … |
| E15 | s12356–s12382 | s12359 `[%r29408+6144]` | +6144 | P16, +105792 … |
| E16 | s12403–s12429 | s12406 `[%r29414+8192]` | +8192 | P17, +133440 … |
| E17 | s12450–s12476 | s12453 `[%r29420+10240]` | +10240 | P18, +161088 … |
| E18 | s12497–s12523 | s12500 `[%r29426+12288]` | +12288 | P19, +188736 … |
| E19 | s12544–s12570 | s12547 `[%r29432+14336]` | +14336 | P20, +216384 … |

Each run also contains **ten `ld.weak.global.ca.v4.u32`** at
`W + 16*laneid + base + 512*i`, `i = 0…9` — the next phase's 20 B fragments
(`s12270 L25636` is the first of E13's).  The A quad itself is loaded by the
preceding phase's epilogue at `+0, +2048, …`, so the shared array holds eight
2048-byte A tiles.  Nothing else happens.

---

### E20 — s12591–s13619: RMS norm before the MLP, and the odd phases' operand

1029 statements, no stores, 20 table loads, 10 weight loads, 4 bias loads,
40 packs.  Op boundaries:

| range | op |
|---|---|
| s12591–s12596 | first of 240 `mul.f16` — 96 f16x2 squares of phase 20's D |
| s12753–s12817 | 42 `add.f16x2` + 4 `shfl.sync.bfly.b32` (`s12795 L27521`) → 8 row sums |
| s12804–s12940 | 82 `add.f16` |
| s12820–s12821 | `mov.f64 %fd386, 0d3F20000000000000` (L27587) + `cvt.rn.f16.f64 %rs4446, %fd386` — the `2⁻¹³` epsilon |
| s12944–s13261 | 80 `cvt.f32.f16` + 80 `rsqrt.approx.ftz.f32` (`s12946 L27961`) + 80 `cvt.rn.f16.f32` |
| s13268–s13287 | 20 `ld.global.v2.u16` at `W + ((laneid<<2)&12) + 221840 + 16*i` — the norm gain |
| s13300–s13528 | `mul.f16` ×(gain) |
| s13532–s13550 | 10 `ld.weak.global.ca.v4.u32` at `W + 16*laneid + 222144 + 512*i` → phase 21's B |
| s13556–s13559 | 4 `ld.global.u32` at `W + ((laneid<<2)&12) + 227264 + 16*i` → phase 21's C seed |
| s13560–s13618 | 40 packs |

**What it computes.**  For each of the 8 rows a lane holds:

```
sumsq[row] = Σ_c x[row,c]²                        (f16, 4-lane butterfly)
inv        = rsqrt.approx.ftz.f32(f32(f16(sumsq + 2⁻¹³)))     // rounded to f16
out[row,c] = x[row,c] * inv * gain[c]
```

with **no mean subtraction and no `1/N`** (the quantity is the raw sum of
squares plus `2⁻¹³`), and `gain[c]` read per column from the 320-byte table at
`W+221840…W+222128`.  The norm's bias is not added here; it arrives as a C
operand later.

The 40 packs are the crucial part: tracing each pack's source shows that
**19 packs read a register whose newest definition is a phase-20 `mma`, 1 reads
phase 20's last `mma`, and 20 read one of the 20 gain loads at
`W+221840…W+222128`**.  So E20 produces **the odd phases' A operand**: the
normalised tile scaled by the gain table, packed to e4m3.  This same pack
content is reproduced in E22, E24, …, E58 (§3).

**Outputs:** phase 21's B (`%r14064…`) and C (`%r14154, %r14164, %r14254,
%r14264`), phase 21's A (in E21), and the shared staging.  **Memory:** 20 table
loads, 1 shared load, 10 + 4 weight loads.

---

### E21 — s13640–s13960: clamped cubic activation + 160-column C build

321 statements.  This is the only epilogue that materialises the four activation
constants (s13641–s13652).

```
s13641 L29561: mov.f32 %f1117, 0f3ED306EB
s13648 L29578: cvt.rn.f16.f32 low, %f1119            // 0.5
s13650 L29582: mov.f32 %f1120, 0f40000000            // 2.0
s13653 L29590: neg.f16x2 %r14350,%r14349
s13654 L29594: max.f16x2 %r14352,%r14225,%r14350
s13655 L29598: min.f16x2 %r14355,%r14352,%r14349
s13656 L29602: abs.f16x2 %r14358,%r14355
s13657 L29606: mul.f16x2 %r14360,%r14347,%r14358
s13658 L29610: sub.f16x2 %r14363,%r14346,%r14360
s13659 L29614: mul.f16x2 %r14366,%r14355,%r14363
s13660 L29618: add.f16x2 %r14369,%r14348,%r14366
s13661 L29622: mul.f16x2 %r14372,%r14225,%r14369
```

Per element, over each of the 8 f16x2 D registers of phase 21 (its four chains,
two registers each):

```
y   = clamp(x, -2, +2)
g   = 0.5 + y * (0.412109375 - 0.0810546875 * |y|)      // f16 throughout
out = x * g
```

`g` is a *cubic ramp*, not a sigmoid: `g(+2) = 1` ⇒ `out = x` for `x ≥ 2`;
`g(-2) = 0` ⇒ `out = 0` for `x ≤ -2`.  **Reproduce the formula, not SiLU/GELU.**

Then, in this order:

| range | op |
|---|---|
| s13640–s13781 | 8 activation groups (each 4 `cvt` + 4 `mov.f32` + 10 f16x2 ops) |
| s13785–s13803 | 10 `ld.weak.global.ca.v4.u32` at `W + 16*laneid + 227328 + 512*i` → phase 22's B |
| s13809–s13828 | 20 `ld.global.v2.u16` at `W + ((laneid<<2)&12) + 232448 + 16*i` (320 bytes) |
| s13829–s13947 | 80 `add.f16` `= table[column] + residual` → 40 b32 |
| s13949–s13960 | 8 packs **of the activation outputs** → phase 22's A |

Evidence for the split:

```
s13829 L30118: add.f16 %rs5210,%rs6526,%rs5212       // table + residual
s13831 L30125: mov.b32 %r14636, {%rs5207, %rs5210}   // → phase 22's C
s13949 L30478: cvt.rn.satfinite.e4m3x2.f16x2 %rs5448, %r14488   // activation output
s13960 L30504: mov.b32 %r14823, {%rs5453, %rs5454}   // → phase 22's A
s13961 L30506: mma … {%r14820, %r14821, %r14822, %r14823}, {%r14579, %r14580}, {%r14636, %r14637}
```

**Outputs:** phase 22's A (8 b32), phase 22's C (40 b32), phase 22's B
(`%r14579…`), and the four constants (consumed again by E23 … E59).

---

### E23, E25, …, E59 — 172 statements each: activation + pack + next B

Twenty runs, one after each odd banded phase, byte-identical in shape
(histogram: 32 `cvt.rn.f16.f32`, 36 `mov.b32`, 8 each `neg/max/min/abs.f16x2`,
24 `mul.f16x2`, 8 `sub.f16x2`, 8 `add.f16x2`, 10 `ld.weak.global.ca.v4.u32`,
8 packs).  Same clamped cubic as E21 but **no table add and no `ld.global`**:

```
s14094 L30986: cvt.rn.f16.f32 low, %f1117            // constant from E21
s14102 L31009: neg.f16x2 %r…,%r…
s14103 L31013: max.f16x2 %r…,%r…,%r…                 // phase 23's D
s14234 L31471: ld.weak.global.ca.v4.u32 {…},[%rd270] // +232768 … (phase 24's B)
s14253 L31510: cvt.rn.satfinite.e4m3x2.f16x2 %rs…, %r…
```

The 10 weight loads are at `W + 16*laneid + base + 512*i` for the next even
phase (232768 for E23, 243072 for E25, …); the 8 packs are the activation
outputs and become the next even phase's A.  No `mov.f32`, no `ld.global`,
no shared access.

---

### E22, E24, …, E58 — 92 statements each: the shared odd-phase operand

Twenty runs (histogram: 40 `cvt.rn.satfinite`, 20 `mov.b32`, 12 `add.s64`,
10 `ld.weak.global.ca.v4.u32`, 4 `ld.global.u32`).  Example:

```
s13985 L30652: ld.weak.global.ca.v4.u32 { %r14579,…},[%rd260]   // +227328 … (phase 23's B)
s14009 L30697: ld.global.u32 %r14919, [%rd676+237888]           // phase 23's C, 4 words
   …
s14013 L30702: cvt.rn.satfinite.e4m3x2.f16x2 %rs…, %r…
```

Tracing the pack sources of every one of these runs gives the *same* answer as
E20's pack group: **19 packs read a phase-20 `mma` register, 1 reads phase 20's
last `mma`, and 20 read one of the gain loads at `W+221840…W+222128`**.  So
E22 … E58 do **not** consume their own phase's D: they re-emit the shared
odd-phase A operand (phase 20's normalised-and-gained output) for the next odd
phase, plus that odd phase's B and C.  This is why every odd phase 21…59 uses
the same A operand (cf. the `a_source` column of `rr_layer_spec.py`).

The four `ld.global.u32` are at `W + ((laneid<<2)&12) + {227264, 237888,
248192, …} + {0,16,32,48}` — a 64-byte, 32-column C seed for the next odd phase.

---

### E60 — s19757–s20055: pack → plane-arena scatter → next B/C

299 statements.  Three sub-sections.

#### (a) requantise, s19757–s19816

40 `cvt.rn.satfinite.e4m3x2.f16x2` + 20 `mov.b32` → `%r370 … %r389` (phase 60's
64×32 output as 20 b32 = 40 e4m3 values per lane).

#### (b) plane-arena scatter, s19817–s20031

```
s19817 L50393: ld.param.u64 %rd805, [%rd1+-32]          // param_0+48
s19818 L50394: cvta.to.global.u64 %rd5, %rd805
s19820 L50398: shr.u32 %r390, %r29384, 2               // g = laneid>>2
s19821 L50399: and.b32 %r391, %r29384, 3               // t = laneid&3
s19822 L50400: shr.u32 %r29468, %r29384, 5             // laneid>>5
s19823 L50401: and.b32 %r29469, %r390, 7
s19824 L50402: add.s32 %r29470, %r369, %r29468         // y = %r369 + (laneid>>5)
s19830 L50408: add.s32 %r29471, %r1, %r29469           // x = %r1 + (g&7)
s19836 L50414: mad.lo.s32 %r392, %r7, %r29470, r29472  // %r7 = 8*ex
s19841 L50420: mul.wide.s32 %rd806, %r393, 4
s19843 L50422: st.global.u32 [%rd807], %r370
```

**Twenty guarded stores** `st.global.u32 [%rd5 + 4*idx]`, in statement order
`idx = <base> + d` with `d = %r391 = t = laneid&3` for the first of a pair and
`d = %r394 = t|4` for the second:

| # | statement | value | base |
|---|---|---|---|
| 1 | s19843 L50422 | `%r370` | `%r392` |
| 2 | s19852 L50433 | `%r371` | `%r392` |
| 3 | s19885 L50468 | `%r372` | `%r396` |
| 4 | s19893 L50478 | `%r373` | `%r396` |
| 5 | s19902 L50489 | `%r374` | `%r399 = %r392 + %r8` |
| 6 | s19910 L50499 | `%r375` | `%r399` |
| 7 | s19919 L50510 | `%r376` | `%r402 = %r396 + %r8` |
| 8 | s19927 L50520 | `%r377` | `%r402` |
| 9 | s19936 L50531 | `%r378` | `%r405 = %r399 + %r8` |
| 10 | s19944 L50541 | `%r379` | `%r405` |
| 11 | s19953 L50552 | `%r380` | `%r408 = %r402 + %r8` |
| 12 | s19961 L50562 | `%r381` | `%r408` |
| 13 | s19970 L50573 | `%r382` | `%r411 = %r405 + %r8` |
| 14 | s19978 L50583 | `%r383` | `%r411` |
| 15 | s19987 L50594 | `%r384` | `%r414 = %r408 + %r8` |
| 16 | s19995 L50604 | `%r385` | `%r414` |
| 17 | s20004 L50615 | `%r386` | `%r417 = %r411 + %r8` |
| 18 | s20012 L50625 | `%r387` | `%r417` |
| 19 | s20021 L50636 | `%r388` | `%r420 = %r414 + %r8` |
| 20 | s20029 L50646 | `%r389` | `%r420` |

The two principal bases are built once:

```
s19823 L50401: and.b32 %r29469, %r390, 7               // g&7
s19824 L50402: add.s32 %r29470, %r369, %r29468         // y = %r369 + (laneid>>5)
s19830 L50408: add.s32 %r29471, %r1, %r29469           // x = %r1 + (g&7)
s19835 L50413: shl.b32 %r29472, %r29471, 3             // 8*x
s19836 L50414: mad.lo.s32 %r392, %r7, %r29470, %r29472 // = 8*ex*y + 8*x
s19862 L50444: shl.b32 %r29772, %r446, 3               // 8*ex
s19864 L50446: shr.u32 %r29476, %r29475, 3             // (g+8)>>3
s19865 L50447: and.b32 %r29477, %r29475, 7             // (g+8)&7
s19866 L50448: add.s32 %r29478, %r29774, %r29476
s19878 L50460: mad.lo.s32 %r396, %r29772, %r29478, %r29480
```

with `g = laneid>>2`, `t = laneid&3`, `%r7 = %r446<<3 = 8*ex` (s17 L1045),
`%r8 = %r447 * %r7 = ey*8*ex` (s18 L1046, the **plane stride**),
`%r369 = %r3 + (tid.z<<1)` (s12592 L26961), `%r1`/`%r3` the block origins
(s7 L1035, s14 L1042), and every store guarded by
`p1/p2 = (y<0) | (y≥ey) | (laneid>15) | (x<0) | (x≥ex)` (s19825–s19834,
s19867–s19877): a guard failure stores `-1` as the index and the following
`setp.lt.s32 …, -1` + `bra` skips the store.  Consecutive pairs therefore walk
**one plane per two stores**, i.e. the arena is a stack of `8*ex*ey`-u32
planes.

#### (c) next B/C, s20032–s20055

```
s20032 L50650: ld.param.u64 %rd878, [param_0+40]
s20033 L50651: ld.param.u64 %rd877, [param_0+40]
s20040 L50661: ld.weak.global.ca.v4.u32 { %r29500,…},[%rd846]   // +428544, phase 61's B
   … five loads at +428544 + 512*i (last s20048 L50677, +430592)
s20054 L50686: ld.global.u32 %r29530, [%rd855+431104]           // phase 61's C seed
s20055 L50687: ld.global.u32 %r29540, [%rd855+431120]
```

**Outputs:** phase 61's B (10 fragments) and C (`%r29530, %r29540`).
**Memory:** 20 plane-arena stores, 5 + 2 weight loads.

---

### E61 — s20066–s20299: the final epilogue

234 statements.  It consumes the cut's four returned accumulators
(`%r29601, %r29611, %r29781, %r29782`).

#### (a) shared fade-in, s20066–s20135

```
s20066 L50758: bar.sync 0
s20078 L50772: st.shared.u32 [%r430], %r29601
s20082 L50777: st.shared.u32 [%r430+16], %r29611
s20093 L50789: st.shared.u32 [%r432], %r29602
s20095 L50792: st.shared.u32 [%r432+16], %r29612
s20100 L50798: bar.sync 0
s20129 L50828: ld.shared.u32 %r29649, [%r29759]
s20131 L50830: ld.shared.u32 %r29650, [%r29759+4]
s20132 L50831: ld.shared.u32 %r29653, [%r29759+8]
s20133 L50832: ld.shared.u32 %r29656, [%r29759+12]
s20135 L50834: ld.shared.u32 %r29670, [%r29759+16]
```

Four `st.shared.u32` write the returned accumulators (16 bytes apart) and five
`ld.shared.u32` read back a 20-byte window at
`%r29759 = smem + ((x*6 + y*48) << 2)` (s20125–s20128) — the two row groups'
values side by side.

#### (b) bounds check, s20101–s20121

`s20111–s20112` form `x = %r1 + %r433`, `y = %r3 + %r434`; `s20116` reads
`param_0+0` (`%r29646, %r29647` = `(ex, ey)`); `s20113–s20120` build
`p350 = (laneid>>5 > 1) | (x|y < 0) | (x ≥ ex) | (y ≥ ey)` and `s20121
bra $L__BB1_370` skips the rest.

#### (c) softmax over 5 shared values, s20136–s20212

```
s20138 L50838: max.f16x2 %r29648,%r29649,%r29650
s20140 L50846: max.f16x2 %r29654,%r29651,%r29656
s20142 L50851: max.f16 %rs…,%rs…,%rs…
s20145 L50860: sub.f16x2 %r29669,%r29670,%r29671
s20148 L50868: mul.ftz.f32 %f…, %f…, 0f3FB8AA3B      // *log2(e)
s20149 L50869: ex2.approx.ftz.f32 %f…, %f…
s20153 L50876: add.ftz.f32 %f…, %f…, %f…             // row sum
s20205 L50988: div.approx.ftz.f32 %f1141, %f1142, %f1160    // 1/rowsum
s20208 L50995: mul.f16x2 %r29672,%r29673,%r29686
   … five f16x2 multiplies (s20208–s20212)
```

A real row max (`max.f16x2` ×3, `max.f16` ×2), `x − max`, `2^((x−max)·log2e)`
through `cvt.f32.f16 → mul → ex2.approx`, an f32 row sum, `div.approx.ftz.f32`
with `1.0`, and five `mul.f16x2` — a textbook f32 softmax, unlike E11's.

#### (d) texture gather, s20213–s20296

`ld.param.u64 %rd872, [%rd881+8]` = `param_0+88` (s20229 L51030) is the texture
handle; nine `tex.base.2d.v4.f16.s32` (s20231 L51033 … s20290 L51153) sample it
at integer coordinates clamped into `[1, ex−2] × [1, ey−2]` (s20213–s20228,
s20240–s20248), and each result is folded in with `fma.rn.f16x2` against the
softmax weights (s20235 L51041, …, 12 total f16x2 fma) starting from
`%r29702 = 0` (s20130 L50829).

#### (e) surface write, s20297

```
s20297 L51171: sust.b.2d.v4.b16.zero [%rd874, {%r29753,%r436}], {%rs6599,%rs6603,%rs6607,%rs6614}
```

`%rd874` = `param_0+112` (s20134 L50833).  One 4-channel b16 surface write per
lane; the fourth channel is the constant `0` (`s20296 L51169: mov.u16 %rs6614,
0`).  `ret` at s20299 L51175.

**Inputs:** the four returned accumulators, `param+0/+48/+88/+112`, the smem
window, `0f3FB8AA3B`, `0f3F800000`.
**Outputs:** none (the kernel returns).

## 4. Memory traffic of the whole entry

### 4.1 Global

* **Weight-image B fragments** — `ld.weak.global.ca.v4.u32` in the E1–E9 runs,
  E12–E19 runs, E20–E60 runs, always `W + 16*laneid + imm`, `imm` a phase-table
  byte offset.  The largest is `W+430592` (E60, s20048).
* **Per-column tables** — `ld.global.v2.u16` / `ld.global.u32` at
  `W + ((laneid<<2)&12) + imm`.  The complete list:

| statements | offsets | loads | block | consumer |
|---|---|---|---|---|
| s6864–s6883 (prologue) | `W+0 … W+304`, step 16 | 20 × v2.u16 | 320 B | prologue normalise |
| s7861–s7907 (E10) | `W+10560 … W+22336`, step 512 | 24 × v4.u32 | 12288 B | phase 11's C (score bias) |
| s12062–s12081 (E12) | `W+221520 … W+221808`, step 16 | 20 × v2.u16 | 320 B | phase 13's C |
| s13268–s13287 (E20) | `W+221840 … W+222128`, step 16 | 20 × v2.u16 | 320 B | norm gain **and** odd-phase A |
| s13556–s13559 (E20) | `W+227264 … W+227312` | 4 × u32 | 64 B | phase 21's C |
| s13809–s13828 (E21) | `W+232448 … W+232752`, step 16 | 20 × v2.u16 | 320 B | phase 22's C |
| s14009–s14012 (E22) | `W+237888 … W+237936` | 4 × u32 | 64 B | phase 23's C |
| *(E24, E26, …, E58)* | `W + 237888 + 10240*k` | 4 × u32 | 64 B each | odd phases 25…61 at `W+431104` |
| s20054–s20055 (E60) | `W+431104, W+431120` | 2 × u32 | 8 B | phase 61's C |

* **Plane arena (param_0+48)** — read once, at s19817 L50393 (E60), and written
  by E60's twenty `st.global.u32` (§3/E60).  Nothing reads it back inside this
  launch.
* **Input buffer (param_0+8)** — read once, at s22 L1051 (`%rd8`), converted to
  `%rd4` (s23 L1052); the prologue's `ld.global.v2.u16` reads (s178 L1218 and 39
  siblings, all `[%rd4 + mul.wide(%r29786, 4)]`) sample it at e4m3 pairs.
* **Texture handle (param_0+88)** — read once, at s20229 L51030 (E61).
* **Output surface (param_0+112)** — read once, at s20134 L50833 (E61), used as
  the `sust.b.2d.v4.b16.zero` descriptor at s20297 L51171.
* **Extents (param_0+0)** — s15 L1043 (prologue) and s20116 L50814 (E61).
* **E61 does not read `param_0+24`, `+32`, `+48`, `+96` or `+104`.**

### 4.2 Shared

* **Prologue staging** — ten `st.shared.v4.u32` per `%tid.z` group, fed by 80
  e4m3 packs of the input buffer (s7364–s7403):
  * five at `%r5148 + {0,512,1024,1536,2048}` where
    `%r5148 = smem + %tid.z*2560 + laneid*16` (s7404…s7422, L11638…L11674,
    last s7474 L11790);
  * five at `%r5173 + {10240,10752,11264,11776,12288}` where
    `%r5173 = %r5146 + laneid*16 = smem + %tid.z*2560 + laneid*16`
    (s7499 region, s7484 L11803 … s7504 L11823).

  followed by a `bar.sync 0` (s7506 L11826).  The staged A operands are read
  back at s7513–s7542: thirty `ld.shared.v4.u32 [%r11624 + 512*i]`,
  `i = 0…29`, with `%r11624 = smem + laneid*16` (s7510–s7512) — the read base
  carries **no** `%tid.z` term, while the store base does; the correspondence
  between the two is in §5 U8.
* **E12 staging** — 4 `st.shared.v4.u32` at
  `smem + %tid.z*512 + laneid*16` offsets `0,512,1024,1536` (s11986 L24912 …
  s12025), read back by the `ld.shared.v4.u32` of E12 (s12033) and E13–E19
  (offsets 2048 … 14336).
* **E61 staging** — 4 `st.shared.u32` + 5 `ld.shared.u32` (§3/E61(a)).

Nothing else in the entry touches shared memory.

## 5. Uncharacterised

**U1. `param_0+24`, `+32`, `+96`, `+104`, `+120`, `+128` are never read.**  A
grep of every `ld.param` in the entry (50 instructions, listed in §1.1) shows
exactly `param_0+0`, `+8`, `+40`, `+48`, `+80`, `+88`, `+112`; there is no read
at `+24`, `+32`, `+96`, `+104`, `+120` or `+128`.  *Missing:* the parameter
struct's field names.

**U2. The identity of the tables.**  The 320-byte blocks at
`W+221504`/`W+221840`/`W+232448` (and the prologue's `W+0…W+304`) are per-column
f16 vectors read with the `W + ((laneid<<2)&12)` pattern.  Their role is
*computable* (§3/E12, E20, E21) but their names are not.  *Missing:* the weight
image's `_prep` permutation or the model's parameter names.

**U3. Phase 11's and 12's B operands.**  Both are register-resident (E11's
`movmatrix`/pack output, §3/E11), so the phase table's `?` weight/`band` columns
are correct; what the two operands *are* in model terms (V vs Vᵀ vs a
concatenation) is not determinable from the PTX alone.  *Missing:* the
high-level attention formulation.

**U4. The exact residual subtracted in E21's C build.**  The 80 `add.f16` at
s13829–s13947 add a per-column table to a register pair whose provenance the
one-hop walk does not resolve (`%rs5212` etc.).  The *arithmetic* is
`C = table[column] + residual` (s13829 L30118, s13831 L30125), but which
register holds the residual is not established.  *Missing:* a full def-use chain
through the phase-20 norm.

**U5. E22…E58's pack source.**  The walk shows 20 of each run's 40 packs read
the gain loads at `W+221840…W+222128` and 20 read phase-20 `mma` registers, i.e.
the same content as E20's pack group.  Whether that is a deliberate re-emission
of the shared A operand or an artefact of how the compiler scheduled one
operand across twenty uses cannot be decided from the PTX.  *Missing:* the
source-level loop structure.

**U6. The plane-arena geometry in E60.**  The index is
`idx = (8*ex)*y + 8*x + d` with `d ∈ [0,8)`, so one row-stride unit is
`8*ex` u32 = `32*ex` bytes, but whether the arena is the same object as
`param_0+8` (the input read in the prologue) is not established — the two
pointers are distinct fields of the parameter block.  *Missing:* the harness.

**U7. The `%tid.z` weight-base term.**  The B loads of phase 1, phases 2–10 and
the C seeds of phase 11 are issued from `W + %tid.z*27648 + 16*laneid + off`
(§2.3, §3/E1–E9), while every later phase uses `W + 16*laneid + off`.  The
phase-table byte offsets for those eleven phases are therefore *not* absolute
image offsets.  Whether the four `%tid.z` groups compute four different output
tiles of a 4×-wide GEMM, or four replicas of the same tile with different
weights, cannot be decided from the PTX: the A operands they read are
`%tid.z`-independent (shared reads at `smem + laneid*16`, §4.2), and their D
registers are physically the same registers.  *Missing:* the block's launch
geometry (`%ntid.z`) and the high-level loop structure.

**U8. The prologue's staging regions.**  The ten staging stores per `%tid.z`
group write `[%tid.z*2560, %tid.z*2560+2560)` and
`[%tid.z*2560+10240, %tid.z*2560+12800)`; for `%tid.z = 3` the second block
ends at 20480, which exceeds the 16384-byte declaration of the shared array
(`.shared .align 4 .b8 _ZZ…smem[16384]`, L1027).  The read side uses a
`%tid.z`-independent base (`smem + laneid*16`) and spans `[0, 15360)`, so the
store and load regions do not obviously line up for every `%tid.z`.  *Missing:*
the launch's `%ntid.z` and the source-level staging loop; I could not decide
whether some `%tid.z` values never reach the stores (`setp.gt.u32 %p3, %r9, 3`
at s20 L1048 only excludes `%tid.z > 3`) or whether the array is over-allocated
by the ABI.

## 6. Quick acceptance index

| requirement | where |
|---|---|
| phase table with M/N/K checked | §2.1, §2.2 |
| weight image extent | §2.3 (431124 bytes) |
| phases that read no weight operand | §2.4 (11 and 12) |
| norm (RMS, no mean, `2⁻¹³`, per-column gain) | §3/E20, s12591–s13619 |
| attention (score, softmax, V-transpose, P·V) | §3/E10, E11, E12 |
| activations (clamped cubic, not GELU) | §3/E21, E23…E59 |
| merge / patch expand | no 8-lane merge (that device appears only in enc0's E20(c)); this layer's per-pixel gather is E61's 5 shared loads + 9 texture reads (§3/E61) |
| per-launch table in the plane arena | §4.1 and §3/E60(b) |
| plane-arena writes, exact value and index | §3/E60(b) |
| final epilogue | §3/E61 |
