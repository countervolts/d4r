# `cuda_dldn_engine_swin_dec4_kernel` — the epilogues of phases 1…62

Companion to `kernels/rr/rr_layer_spec.py dec4`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0019-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_dec4_kernel` (the entry opens at file line 1016, its body
holds statements s1…s24447, and s24447 is the `ret`).

dec4 is the widest of the eleven Swin layers: its attention/MLP width is 160 where
dec3's is 128, so every count below is *not* interchangeable with dec3's.  Where a
region is the same code as dec3's, that is stated and the shared evidence is cited
on dec4's own statements.

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents; **`LNNNNN` is the physical line** of the same instruction, so every
  claim can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.  (Example anchors:
  phase 1's 300 `mma` occupy only L1707–L3799, and s871 = L3807.)
* An **epilogue** is the run of statements between one phase's last `mma` and the
  next phase's first `mma`; a phase's own weight loads live in the previous
  epilogue, and are named in place.
* `W` denotes the prepared weight image: `ld.param.u64 %rd2,
  [cuda_dldn_engine_swin_dec4_kernel_param_0+40]` (s9) then
  `cvta.to.global.u64 %rd3, %rd2` (s10).  Every `ld.weak.global.ca.v4.u32` reads
  `W + 16*laneid + imm`; tables and biases use
  `W + ((laneid<<2)&12) + imm` (canonical construction s16157–s16161).
* `S` denotes the plane arena: `ld.param.u64 %rd1050, [%rd1+-32]` (s23917) then
  `cvta.to.global.u64 %rd5, %rd1050` (s23918), with `%rd1 = param_0+80` (s4), so
  `%rd1-32` is **`param_0+48`**.  This is the entry's only read of `param_0+48`.
* `SM` denotes the 16384-byte shared array
  `_ZZ33cuda_dldn_engine_swin_dec4_kernel33DldnEngineSwinEncParamsStructBaseE4smem`
  (materialised by `mov.u32` at s11611 and others).
* `%r9 = %tid.z` (s19) is the **slab index**, and dec4 **guards it to 0…3**:
  `s20 setp.gt.u32 %p3, %r9, 3` / `s21 @%p3 bra $L__BB1_121`.  It scales the input
  buffer ×25600 (s438), a second input stride ×160 (s543), ×40 (s1052), the shared
  tile ×16 (s2301), ×2560 (s11503), ×2048 (`<<11`, s16069), ×512 (`<<9`, s16128)
  and the phase-2…11 weight group ×27648 (`%r16781 = %r9*13824` s11607,
  `%rd381 = %r16781*2` s11608, consumed at s11645 `add.s64 %rd383, %rd2, %rd381`).
* Register conventions are the settled family ones: `C/D reg0` = (row g, cols
  2t,2t+1), `reg1` = (row g+8, same cols), `g = laneid>>2` (s37/s38), `t = laneid&3`.
* Anything I could not pin down from the PTX is in §6.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `param+0`, extents `(ex, ey)` | `ld.param.v2.u32 {%r1013, %r1014}, [param_0]` | s15 |
| `param+8`, the e4m3 feature buffer | `ld.param.u64 %rd60, [%rd1+-72]` → `cvta.to.global.u64 %rd61, %rd60` | s34–s35 |
| `param+24`, the f16 image buffer (**the skip input**) | `ld.param.u64 %rd253, [%rd1+-56]` → `cvta.to.global.u64 %rd4, %rd253` | s3640–s3641 |
| `param+40`, `W` (weight image) | `ld.param.u64 %rd2, [param_0+40]` → `cvta.to.global.u64 %rd3, %rd2` | s9–s10 |
| `param+48`, `S` (plane arena) | `ld.param.u64 %rd1050, [%rd1+-32]` → `cvta.to.global.u64 %rd5, %rd1050` | s23917–s23918 |
| `param+80`, block origin `(ox, oy)` | `ld.param.v2.u16 {%rs222, %rs223}, [param_0+80]` | s5 |
| `param+88`, low-res feature **texture** | `ld.param.u64 %rd1117, [%rd1129+8]` with `%rd1129 = param_0+80` (s24223) | s24334 |
| `param+96`, **texture** added unconditionally | `ld.param.u64 %rd1119, [%rd1129+16]` | s24410 |
| `param+104`, **texture** added under a NaN guard | `ld.param.u64 %rd1121, [%rd1129+24]` | s24412 |
| `param+112`, output **surface** | `ld.param.u64 %rd1123, [%rd1129+32]` | s24233 |
| `param+136`, `(sx, sy)` pixel→uv scale | `ld.param.v2.f32 {%f1180, %f1181}, [%rd1129+56]` | s24328 |
| `SM` (16384 B) | `mov.u32 %r16783, _ZZ33…E4smem` | s11611 |
| per-lane weight address | `mul.wide.u32 %rd382, %laneid, 16` + `add.s64 %rd384, %rd383, %rd382` | s11644–s11646 |
| per-lane *table* address | `shl.b32 %r, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rd, %r, 12` → `add.s64 %rd, W, %rd` | s16157–s16161 |
| block origins | `%r1 = ctaid.x<<3 − ox` (s7), `%r2 = %r1 − 1` (s8), `%r3 = ctaid.y<<3 − oy` (s14), `%r6 = %r3 − 1` (s16) | s7–s16 |
| `%r7 = ex<<3`, `%r8 = ey*%r7` | s17–s18 | s17–s18 |

The `param+80`-relative form is worth stressing: dec4 keeps `%rd1127` and `%rd1129`
(both `= %rd1128/%rd1130 + 80`, s24198 / s24223) and then reads the decoder
parameters as `[%rd1127-80]` (= `param+0`), `[%rd1129+8]`, `[%rd1129+16]`,
`[%rd1129+24]`, `[%rd1129+32]`, `[%rd1129+56]`.  So **`+96` and `+104` are exactly
`%rd1129+16` and `%rd1129+24`** (s24410 / s24412).

### 1.2 Constants

| register | first definition | literal | identity |
|---|---|---|---|
| `%f476` | s12177 `mov.f32 %f476, 0f3C8CCB50` | 0.017186790704727173 | score scale |
| `%fd383` | s12181 `mov.f64 %fd383, 0dBFE1CC0000000000` | −0.55615234375 | clamp low |
| `%fd385` | s12186 `mov.f64 %fd385, 0d3FE1CC0000000000` | +0.55615234375 | clamp high |
| `%f478` | s12192 `mov.f32 %f478, 0f3F6D6000` | 0.92724609375 | cubic coefficient |
| `%f480` | s12197 `mov.f32 %f480, 0f3FB00000` | 1.375 | cubic offset |
| `%fd1` | s10075 `mov.f64 %fd1, 0d3F20000000000000` | 2⁻¹³ | norm epsilon in E1 |
| `%fd386` | s16920 `mov.f64 %fd386, 0d3F20000000000000` | 2⁻¹³ | norm epsilon in E21 |
| `%f1117` | s17741 `mov.f32 %f1117, 0f3ED306EB` | 0.4121621549129486 | activation ramp |
| `%f1118` | s17744 `mov.f32 %f1118, 0f3DA60DD6` | 0.0810810774564743 | activation width |
| `%f1119` | s17747 `mov.f32 %f1119, 0f3F000000` | 0.5 | activation offset |
| `%f1120` | s17750 `mov.f32 %f1120, 0f40000000` | 2.0 | activation clamp |

`0f3FB8AA3B` (log2e) occurs 9 times in the final epilogue, first at s24246; the
negated form `0fBFB8AA3B` once at s24415; `0f3F800000` (1.0) at s24251's
`add.ftz.f32`.  Neither layer ever uses the `.relu` requantiser: the string
`satfinite.relu` does not occur.

### 1.3 Two quantisation helpers

* `cvt.rn.satfinite.e4m3x2.f16x2` — requantisation.
* `cvt.rn.f16x2.e4m3x2` — the inverse unpack (one e4m3 byte pair → f16x2).

## 2. Phase table summary

From `rr_layer_spec.py dec4`: **62 phases, 1606 `mma`, 24447 statements**.

The tool prints M and N from the number of *distinct* A and B fragments, so the
identity `M/16 * N/8 * K/32 == mma` is the dense-GEMM check:

* holds for phases 2–12, 14–21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47,
  49, 51, 53, 55, 57, 59, 61;
* **fails for 23 phases — 1, 13, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
  46, 48, 50, 52, 54, 56, 58, 60, 62.**  For the even phases 22…60 the tool prints
  M=80 N=160 K=160 (`5*20*5 = 500`) against 20 `mma`: those phases are 5-step
  cascades, not dense GEMMs (see E22).  For **P62** it prints M=80 N=80 K=160
  against 10 `mma`; P62 is a 5-fragment cascade (A = `%r938…%r957`) as §4/E62
  describes.

Weight-image extent: the largest byte read from `W` is phase 62's second C seed at
`W + ((laneid<<2)&12) + 534800`, i.e. the image is **at least 534804 bytes**.
Phase 1's B block is `[0, 25600)`, phase 1's C seeds are `[102400, 102704]`, the
score bias (phase 12's C) is `[114240, 126028)`, the per-slab copy of the phase-2…11
weight group starts at `104000 + z*27648`, and phase 62's last weight tile is
`[532224, 534796)`.

**Phases that read no weight operand at all: 12 and 13.**  Phase 12's A and B both
come from e4m3 packs of phase 10's D (E11), and phase 13's B is the `movmatrix`
transpose of registers (E12); see §4/E11, E12.

## 3. Summary of epilogues

| epilogue | range | stmts | named operation |
|---|---|---|---|
| E1 | s871–s11653 | 10783 | requantise P1's D, scatter to `SM`, read back, blend `param+24`, inverse-std, requantise, stage for P2 |
| E2–E10 | 7 stmts each | 7 | next-B weight loads only (s11678–s11900) |
| E11 | s11957–s12127 | 171 | score-bias C-seed loads + 80 packs of P10's D → P12 A and B |
| E12 | s12176–s16019 | 3844 | softmax of the 64×96 score block, V-transpose, pack |
| E13 | s16068–s16341 | 274 | stage P·V, P14 A from `SM`, next-B loads, column table + skip → P14 C |
| E14–E20 | 27 stmts each | 27 | P15…P21 A from `SM` + next-B loads (s16362–s16670) |
| E21 | s16691–s17719 | 1029 | RMS norm (², sum, rsqrt, gain) + packs + next B/C |
| E22 | s17740–s18060 | 321 | clamped cubic activation + column-bias add + packs + next B |
| E23 | s18081–s18172 | 92 | 40 packs → P24 A, next-B, next-C bias |
| E24…E60 | 172/92 alternating | | E22/E23 repeated for P24…P60 |
| E61 | s23857–s24154 | 298 | pack → plane-arena scatter (`param+48`) → next B/C |
| E62 | s24165–s24446 | 282 | `SM` stage, softmax over 5 taps, 3×3 texture gather, aux textures, surface write |

---

## 4. The epilogues

### E1 — s871–s11653: patch expand, image blend, inverse standard deviation

10783 statements (44 % of the entry), the same pipeline as dec3's E1 with every
count scaled by 160/128 = 1.25.

Evidence:

```
s871   L3807  cvt.rn.satfinite.e4m3x2.f16x2 %rs226, %r1298
s875   L3818  mov.b32 %r10, {%rs228, %rs229}
s1090  L4268  st.shared.v2.u16 [%r4580], {%rs226, %rs227}
s1117  L4297  st.shared.u32 [%r4593], %r10
s2295  L5592  bar.sync 0
s2340  L5647  ld.shared.v2.u16 {%rs346, %rs347}, [%r5246]
s3640  L7252  ld.param.u64 %rd253, [%rd1+-56]
s3794  L7417  ld.global.v2.u16 {%rs426, %rs427}, [%rd255]
s9055  L12990 cvt.rn.f16x2.e4m3x2 %r9582, %rs426
s9217  L13472 add.f16 …                        // first of 324
s10018 L15459 shfl.sync.bfly.b32 %r9963,%r9959,%r10029,%r9966,%r10040
s10075 L15607 mov.f64 %fd1, 0d3F20000000000000
s10321 L16341 rsqrt.approx.ftz.f32 fl, fl
s10963 L17382 ld.global.v2.u16 {%rs3159, %rs3160}, [%rd336+103696]
s11503 L18962 mul.lo.s32 %r936, %r9, 2560
s11521 L18998 st.shared.v4.u32 [%r10307], {%r10311, %r10310, %r10309, %r10308}
s11606 L19151 bar.sync 0
s11613 L19160 ld.shared.v4.u32 {%r16785, %r16786, %r16787, %r16788}, [%r16784]
s11651 L19201 ld.weak.global.ca.v4.u32 { %r10356,%r10357,%r10358,%r10359},[%rd337]
```

| range | stmts | what | operations (count) |
|---|---|---|---|
| s871–s1048 | 178 | requantise **phase 1's D** and pack | 120 `cvt.rn.satfinite.e4m3x2.f16x2`, 58 `mov.b32` → `%r10…%r67` |
| s1049–s1089 | 41 | lane / `%r9` → shared-index setup for the first store | `mov.u32 %r4338, %laneid` (s1049), `mul.wide.u32 %rd204, %r4338, -1431655765` (s1051) |
| s1090–s2295 | 1206 | guarded scatter into `SM`, one basic block per store | **58 `st.shared.u32`** + 2 `st.shared.v2.u16` |
| s2295 | 1 | `bar.sync 0` (L5592) | — |
| s2296–s3639 | 1344 | read back a window from `SM` (5-way switch per load) | **40 `ld.shared.v2.u16`** (s2340…s3639) |
| s3640–s3641 | 2 | **the `param+24` buffer pointer** | `ld.param.u64 %rd253, [%rd1+-56]`, `cvta` |
| s3642–s9054 | 5413 | clamped-coordinate gather from the `param+24` buffer | **40 `ld.global.v2.u16`** (s3794…s9054) |
| s9055–s9214 | 160 | dequantise both sources | 160 `cvt.rn.f16x2.e4m3x2` → `%r9582…%r9741` |
| s9215–s10017 | 803 | blend + reduction tree | 480 `mul.f16`, 324 `add.f16`, 84 `add.f16x2` |
| s10018–s10074 | 57 | cross-lane sums | **8 `shfl.sync.bfly.b32`** (s10018…s10071) |
| s10075–s10076 | 2 | epsilon `2⁻¹³` | `mov.f64 %fd1, 0d3F20000000000000` (s10075) |
| s10077–s10320 | 244 | epsilon add + packing of the sums | 160 `add.f16`, 81 `mov.b32` |
| s10321–s10956 | 636 | inverse standard deviation | **160 `rsqrt.approx.ftz.f32`** (s10321…s10954) |
| s10957–s10982 | 26 | per-column gain read | **20 `ld.global.v2.u16`** at `W + ((laneid<<2)&12) + {103680…103984}` (s10963…s10982) |
| s10983–s11502 | 520 | normalise and combine | 320 `mul.f16`, 160 `mov.b32`, 40 `cvt.rn.satfinite.e4m3x2.f16x2` |
| s11503–s11605 | 103 | requantise and stage for phase 2 | **10 `st.shared.v4.u32`** |
| s11606 | 1 | `bar.sync 0` (L19151) | — |
| s11607–s11642 | 36 | phase 2's A fragments | **30 `ld.shared.v4.u32`** → `%r16785…%r16904` |
| s11643–s11653 | 11 | phase 2's B | 2 `ld.weak.global.ca.v4.u32` at `W + z*27648 + 16*laneid + {104000, 104512}` |

**The scatter (s1090–s2295).**  The 60 stores are guarded (`setp.gt.u32 %p, …, 9`
at s1052 and its siblings) and the index is
`byte = 4 * ( (col & 3) | (40*row + 400*(col>>2)) )` into `SM`
(`shr.u32 %r4584, %r79, 2` / `mul.lo.s32 %r4585, %r4584, 400` /
`mad.lo.s32 %r4586, %r80, 40, %r4585` / `shl.b32 %r4587, %r81, 2` /
`add.s32 %r4588, %r4587, %r4586` / `and.b32 %r4589, %r79, 3` /
`or.b32 %r4590, %r4589, %r4588` — s1107–s1113).  So dec4 writes **10-by-10
sub-blocks of 400 bytes** and 58 distinct `(row, col)` triples, versus dec3's
literal `+{0,1600,…,11200}` plane offsets.  As in dec3 the `(row, col)` order is a
permutation of `laneid` derived with the golden-ratio multiplier
(`mul.wide.u32 %rd204, %r4338, -1431655765`, s1051).  The store count differs from
dec3's 48 because dec4's phase 1 produces 800 columns instead of 640.

**The `param+24` gather (s3642–s9054).**  `%rd1-56` is `param_0+24` (s4, s3640);
`%rd4 = cvta.to.global.u64(%rd253)` (s3641).  Each of the 40 loads computes a
clamped halo index and issues
`mul.wide.u32 %rd254, %r34933, 4` / `add.s64 %rd255, %rd4, %rd254` /
`ld.global.v2.u16 {%rs426, %rs427}, [%rd255]` (s3792–s3794).  The gather's
interior index is `clamped_row * %r7 + 8*clamped_col + min(laneid&3, 7)`, i.e. the
same **8-u32-per-image-row f16 buffer** as dec3's (§4/E1 of `rrswin_dec3_epilogues.md`).

**The inverse-std block.**  The residual `add.f16` fold (first at s9217) yields a
value that is squared (`mul.f16`, s9620ff), reduced in-lane, summed with the
two-step butterfly `shfl.sync.bfly` (s10018, s10071), offset by `2⁻¹³` (s10075) and
inverted by 160 `rsqrt.approx.ftz.f32` (s10321…s10954).  The gain table is 20
`ld.global.v2.u16` at `W + ((laneid<<2)&12) + {103680…103984}` (s10963…s10982) —
40 f16 per lane = 160 columns.

**Staging for phase 2.**  `%r936 = %r9*2560` (s11503); the 10 `st.shared.v4.u32`
write at `SM + z*2048 + 16*laneid + {0,512,1024,1536}` (s11521, s11533, s11545,
s11557) and at `+{10240,10752,11264,11776,12288}` in a second group guarded by
`setp.gt.u32 %p626, %r10328, 31` with `%r10328 = %r9<<4` (s11574–s11575), i.e. only
`z ∈ {0,1}`.  After `bar.sync` (s11606) the **30 `ld.shared.v4.u32`** at
`SM + 16*laneid + 512*i` (s11613…s11642) become **phase 2's A**
`%r16785…%r16904` (30 A fragments = 6 m-tiles × 5 k-steps), and the 2
`ld.weak.global.ca.v4.u32` at s11651/s11653 are phase 2's B.

### E2–E10 — s11678…s11900: next-B weight loads only

Seven statements each, exactly dec3's E2–E8 shape: 8 B fragments (16 registers) per
phase, read as `W + z*27648 + 16*laneid + base + 512*i`.  The bases are
`104000` (P2), `109120` (P3), `105024` (P4), `110144` (P5), `106048` (P6),
`111168` (P7), `107072` (P8), `112192` (P9), `108096` (P10), `113216` (P11) — the
even phases walk `104000, 105024, 106048, 107072, 108096` and the odd phases
`109120, 110144, 111168, 112192, 113216`, i.e. **five k-group pairs** (dec3 had
four), so phases 2–11 together form one **96×64×160** GEMM split into ten
`96×32×32` phases chained through C (verified from the operand sets: P2/P3 use A
fragment set 0, P4/P5 set 1, P6/P7 set 2, P8/P9 set 3, P10/P11 set 4; P2/P3 have
C = the single zero register `%r16707` while P4…P11 take the previous phase's D).

### E11 — s11957–s12127: score-bias seeds and 80 packs of phase 10's D

171 statements, 24 weight loads + 80 packs — the dec3 E9 design with the wider N.

```
s11961 L21128 ld.weak.global.ca.v4.u32 { … }, [%rd…]   // first of 24
s12008 L21313 cvt.rn.satfinite.e4m3x2.f16x2 …          // first of 80 packs
```

* **24 loads** at `W + z*27648 + 16*laneid + 114240 + 512*i`, `i = 0…23`
  (s11960 `add.s64 %rd357, %rd404, 114240` with `%rd404 = %rd383 + 16*laneid`,
  `%rd383 = W + z*27648`; s11961 the first load, s11962 the next at `+114752`) →
  **phase 12's C seeds** (96 registers = 24 C pairs).  The block is
  `[114240, 126028)`.  Note this is a different *kind* of C operand from phase 1's:
  phase 1 seeds each mma with a single u32 duplicated into both halves
  (`{%r…, %r…}`), loaded as **20** `ld.global.u32` at
  `[%rd203 + {102400, 102416, …, 102704}]` (s551–s570), whereas phase 12 takes 24
  full 512-byte tiles.
* **80 packs** whose sources are **phase 10's D**, not phase 11's; they split into
  phase 12's A fragments (`%r13134…%r13497`, 16 registers) and B fragments
  (`%r13388…%r13499`, 24 registers = 12 fragments).  Phase 11's own D reaches
  E12's `movmatrix` block instead.
* The C-seed layout is the same as dec3's §4/E9:
  `byte = base + 512*(4*(n>>1) + m) + 16*laneid + 8*(n&1)`, so the tile index runs
  over 4 m-tiles and 6 `(n>>1)` groups = 24 tiles = 12288 bytes.

### E12 — s12176–s16019: softmax of the 64×96 score block, V-transpose, pack

**Phase 12** = 48 `mma` of 64×96 with C = the 24 bias pairs (E11).  The epilogue has
**no memory access at all** (0 `ld.*`, 0 `st.*`) and no `bar.sync`.  Its op
histogram is byte-for-byte dec3's E10: 1048 `mov.b32`, 672 `cvt.rn.f16.f32`, 384
`cvt.rn.f16.f64`, 192 each of `fma.rn.f16x2` / `cvt.f32.f16` /
`rcp.approx.ftz.f32` / `mul.f16`, 144 packs, 112 `shl.b32`, 104 `add.f16x2`,
96 each of `mul.f16x2` / `max.f16x2` / `min.f16x2` / `neg.f16x2` / `and.b32`,
48 `movmatrix`, 16 `shfl.sync.bfly.b32`, 8 `add.f16`.

```
s12177 L21724 mov.f32 %f476, 0f3C8CCB50
s12178 L21727 cvt.rn.f16.f32 low, %f476
s12182 L21737 cvt.rn.f16.f64 %rs3280, %fd383
s14381 L28593 shfl.sync.bfly.b32 %r15668,%r15664,%r15820,%r15671,%r15831   // first of 16
s14511 L28919 rcp.approx.ftz.f32 fl, fl                                    // first of 192
s15278 L30164 mul.f16 %rs4074,%rs4075,%rs4076                              // first of 192
s15756 L31218 movmatrix.sync.trans.aligned.m8n8.b16 %r16028, %r12605       // first of 48
s15804 L31362 cvt.rn.satfinite.e4m3x2.f16x2 %rs4648, %r16036               // first of 144 packs
```

Sub-sections (the same partitioning dec3's §4/E10 documents):

| range | what |
|---|---|
| s12176–s14293 | 96 per-element groups: scale, clamp, cubic, exponent extract (all 96 `max.f16x2` ≤ s14278, `min.f16x2` ≤ s14282, `neg.f16x2` ≤ s14283, `and.b32` ≤ s14293) |
| s14294–s14508 | in-lane partial sums and the butterfly: 104 `add.f16x2` (s14294–s14503), 16 `add.s32` (s14378–s14500), 16 `or.b32` (s14379–s14501), **16 `shfl.sync.bfly.b32`** (s14381–s14502) → 8 row sums, 8 `add.f16` (s14390–s14505) |
| s14509–s15277 | 192 `cvt.f32.f16` (s14509–s15270) → **192 `rcp.approx.ftz.f32`** (s14511–s15272) → 192 `cvt.rn.f16.f32` (…s15274) = 96 reciprocals |
| s15278–s15755 | **192 `mul.f16`** (s15278–s15754) → 96 probabilities |
| s15756–s15803 | 48 `movmatrix` transposing **phase 11's D** |
| s15804–s16019 | 144 `cvt.rn.satfinite.e4m3x2.f16x2` packs (s15804–s16018) |

(The boundaries above are the runs between the first/last occurrence of each op,
verified with the lineariser; the exponent region's end is fixed by the last
`and.b32` at s14293 and the sum region's by the first `add.f16x2` at s14294.)

Per element the exponent is the clamped cubic of dec3's §4/E10:

```
m      = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)
t      = f16(m*(-m) + 0.92724609375)
u      = f16(m*t + 1.375)
expval = f16 from bits ((bits(u)<<5) & 0x7FE07FE0)
```

and **there is no row maximum**: `max.f16x2` (96×) and `min.f16x2` (96×) are the
clamp, and the only warp shuffles are the 16 `shfl.sync.bfly` above.  Contrast E62,
which does compute a real row max over 5 taps.

### E13 — s16068–s16341: stage P·V, seed phase 14

274 statements.

```
s16068 L32201 bar.sync 0
s16069 L32202 shl.b32 %r17033, %r9, 11
s16070 L32203 add.s32 %r17034, %r16783, %r17033
s16074 L32210 cvt.rn.satfinite.e4m3x2.f16x2 %rs4792, %r16368   // first of 32 packs
s16086 L32237 st.shared.v4.u32 [%r17036], {%r17040, %r17039, %r17038, %r17037}
s16128 L32328 shl.b32 %r34545, %r9, 9
s16133 L32335 ld.shared.v4.u32 {%r34550, %r34551, %r34552, %r34553}, [%r34549]
s16138 L32343 ld.weak.global.ca.v4.u32 { %r17055,%r17056,%r17057,%r17058},[%rd405]
s16162 L32388 ld.global.v2.u16 {%rs7270, %rs7271}, [%rd889+325200]   // first of 20
s16183 L32410 add.f16 %rs4826,%rs7309,%rs4828                        // first of 80
```

* s16068 `bar.sync`; s16069–s16073 build `%r17036 = SM + z*2048 + 16*laneid`;
* 32 `cvt.rn.satfinite.e4m3x2.f16x2` packing phase 13's D and 4
  `st.shared.v4.u32` at `SM + z*2048 + 16*laneid + {0,512,1024,1536}`
  (s16086, s16100, s16114, s16127);
* s16128–s16132 `%r34549 = SM + z*512 + 16*laneid`; s16133
  `ld.shared.v4.u32` → **phase 14's A fragment #0** (`%r34550…%r34553`);
* s16137–s16156 **10 weight loads** at `W + 16*laneid + {126528, 127040, 128064,
  128576, 129088, 129600, 130112, 130624, 131136}` → phase 14's B (20 fragments =
  40 registers);
* s16161–s16181 **20 table loads** at `W + ((laneid<<2)&12) + {325184, 325200, …
  325488}` (16-byte stride) → 40 f16 per lane = a per-column table;
* s16182–s16341 **80 `add.f16`** → 40 b32 `%r17104, %r17105, …` = **phase 14's C
  seeds**, exactly `table[col] + residual` (s16183 `add.f16 %rs4826,%rs7309,%rs4828`
  where the second operand is the E1 residual register).

So `phase 14: D = A·B + (skip_residual + column_table)`.

### E14–E20 — s16362–s16670: A from `SM` + next-B, seven times

27 statements each: one `ld.shared.v4.u32` (one A fragment, `%r34557…%r34560` in
E14, then `%r34563…`, `%r34569…`, `%r34575…`, `%r34581…`, `%r34587…`, `%r34593…`),
10 `ld.weak.global.ca.v4.u32` (the next phase's B, 20 fragments = 40 registers) and
the address arithmetic.  Example, E14 (s16362–s16388): `s16365 ld.shared.v4.u32`,
`s16370 ld.weak.global.ca.v4.u32` (first of 10).

Together with E13 this supplies the A k-fragments of phases 14–21, which are one
GEMM **16×160×256** split across eight `mma` phases chained through C (P15 C = P14's
D, …, P21 C = P20's D).  The B blocks are `126528` (P14), `154176` (P15), `181824`
(P16), `209472` (P17), `237120` (P18), `264768` (P19), `292416` (P20), `320064`
(P21).

### E21 — s16691–s17719: RMS norm before the MLP

1029 statements, no stores.  The dec3 E15 design scaled by 160/128.

```
s16691 L34285 shl.b32 %r34597, %r9, 1
s16693 L34287 mov.b32 {%rs6072, %rs6075}, %r18790   // first of 302
s16694 L34289 mul.f16 %rs5066,%rs6075,%rs6075        // first of 240 squares
s16895 L34846 shfl.sync.bfly.b32 %r19103,%r19099,%r15820,%r19106,%r15831   // first of 4
s16920 L34912 mov.f64 %fd386, 0d3F20000000000000     // eps = 2^-13
s16921 L34914 cvt.rn.f16.f64 %rs5309, %fd386
s17046 L35286 rsqrt.approx.ftz.f32 fl, fl            // first of 80
s17368 L35807 ld.global.v2.u16 {%rs7310, %rs7311}, [%rd906+325520]   // first of 20 gain loads
s17632 L36554 ld.weak.global.ca.v4.u32 { %r19224,%r19225,%r19226,%r19227},[%rd485]  // first of 10 next-B
s17656 L36599 ld.global.u32 %r19314, [%rd911+330944] // first of 4 next-C bias words
s17660 L36604 cvt.rn.satfinite.e4m3x2.f16x2 %rs6031, %r33783   // first of 40 packs
```

| range | stmts | op |
|---|---|---|
| s16691–s16903 | 213 | lane/origin setup and 160 `mul.f16` squares (the remaining 80 of the 240 are in the normalise) |
| s16853–s17040 | 188 | in-lane/butterfly sums: 42 `add.f16x2` (s16853–s16917), 4 `or.b32` (s16894–s16915), **4 `shfl.sync.bfly.b32`** (s16895–s16916), 82 `add.f16` (s16904–s17040) |
| s16920–s16921 | 2 | eps `2⁻¹³` in `%fd386` (`cvt.rn.f16.f64 %rs5309, %fd386` at s16921) |
| s17044–s17361 | 318 | 80 `cvt.f32.f16` (s17044–s17357) → **80 `rsqrt.approx.ftz.f32`** (s17046–s17359) → 80 `cvt.rn.f16.f32` (s17048–s17361) |
| s17368–s17387 | 20 | gain loads `W + ((laneid<<2)&12) + {325504…325808}` (`+325504` appears last, `+325520` first) |
| s17632–s17659 | 28 | 10 `ld.weak.global.ca.v4.u32` (phase 22's B, `320064 + 512*i`, s17632–s17650) then 4 `ld.global.u32` at `330944/330960/330976/330992` (phase 22's C bias word, s17656–s17659) |
| s17660–s17718 | 59 | 40 packs → phase 22's A |

**What it computes** — the same expression as dec3's §4/E15:
```
sumsq[row] = Σ_c x[row,c]²                             (f16, in-lane tree + 4-lane butterfly)
inv        = rsqrt.approx.ftz.f32(f32(f16(sumsq + 2⁻¹³)))   // rounded to f16
out[row,c] = x[row,c] * inv * gain[c]
```
with **no mean subtraction and no `1/N`**, and the gain is a per-column table read
with the `W + ((laneid<<2)&12) + off` pattern.  The norm's bias is not added here:
it is phase 22's C seed, loaded at s17656.

### E22 — s17740–s18060: clamped cubic activation + column bias

321 statements.

```
s17740 L36884 mov.u32 %r19505, %laneid
s17741 L36886 mov.f32 %f1117, 0f3ED306EB
s17744 L36893 mov.f32 %f1118, 0f3DA60DD6
s17747 L36900 mov.f32 %f1119, 0f3F000000
s17750 L36907 mov.f32 %f1120, 0f40000000
s17753 L36915 neg.f16x2 %r19510,%r19509      // first of 8
s17754 max.f16x2 …                           // first of 8
s17755 min.f16x2 …                           // first of 8
s17756 abs.f16x2 …                           // first of 8
s17757 L36931 mul.f16x2 %r19520,%r19507,%r19518   // first of 24
s17758 sub.f16x2 …                           // first of 8
s17760 add.f16x2 …                           // first of 8
s17883 add.s64 …                             // column-table addressing
s17885 L37377 ld.weak.global.ca.v4.u32 …     // first of 10 next-B loads
s17909 L37422 ld.global.v2.u16 {%rs7350, %rs7351}, [%rd916+336144]   // first of 20 table loads
s17929 L37469 add.f16 …                      // first of 80 bias adds
s18049 L37803 cvt.rn.satfinite.e4m3x2.f16x2 %rs6311, %r19648   // first of 8 packs
```

Per element, over each of the 8 f16x2 groups of phase 22's D:

```
y   = clamp(x, -2, +2)                              // neg/max/min, s17753–s17755
g   = 0.5 + y * (0.412109375 - 0.0810546875 * |y|)  // abs/mul/sub/mul/add, s17756–s17760
out = x * g
```

`g` is a **cubic ramp, not a sigmoid**: `g(+2) = 1` ⇒ `out = x` for `x ≥ 2`;
`g(-2) = 0` ⇒ `out = 0` for `x ≤ -2`.  **Reproduce the formula, not SiLU/GELU.**
The constants `%f1117…%f1120` are defined here and are re-read as
`cvt.rn.f16.f32 low, %f1117…%f1120` by every later activation epilogue.

Then 20 table loads (s17909 = `+336144`) and 80 `add.f16` (s17929–s18047) write the
table over the activation **inputs**, producing **phase 23's C seeds**; the final 8
packs (s18049…) build **phase 23's A** (one 4-register fragment); and the 10
`ld.weak.global.ca.v4.u32` from s17885 are phase 23's B block `[331008, 336128)`.

### E23 — s18081–s18172: packs, then next B/C

92 statements:

```
s18081 L37971 mov.u32 %r19988, %laneid
s18085 L37979 ld.weak.global.ca.v4.u32 …      // first of 10 next-B loads
s18109 L38022 ld.global.u32 %r20079, [%rd921+341568]   // first of 4 next-C bias words
s18113 L38027 cvt.rn.satfinite.e4m3x2.f16x2 %rs6319, %r33783   // first of 40 packs
```

40 packs (20 b32) → **phase 24's A**; 10 `ld.weak.global.ca.v4.u32` → phase 24's B
`[336448, 341568)`; 4 `ld.global.u32` at `341568/341584/341600/341616` → phase 24's
C bias.

### E24 … E60 — the repeating MLP stack

The pattern `(activation, first one with a table add) → (40 packs + next B/C)`
repeats with a period of two phases.  Numbering the phases: the **even** phases 22,
24, …, 60 each have a 321/172-statement activation epilogue; the **odd** phases 23,
25, …, 59 each have a 92-statement pack epilogue.  Verified op histograms:

* P22 epi (s17740–s18060, 321): 80 `add.f16`, 76 `mov.b32`, 32 `cvt.rn.f16.f32`,
  24 `mul.f16x2`, 20 `ld.global.v2.u16`, 10 `ld.weak`, 8 each of
  `neg/max/min/abs/sub/add.f16x2`, 8 packs.
* P23 epi (s18081–s18172, 92): 40 packs, 20 `mov.b32`, 10 `ld.weak`, 4 `ld.global.u32`.
* P24 epi (s18193–s18364, 172): 172 statements with 8 packs and 10 `ld.weak` and
  **no** `ld.global.v2.u16`, i.e. no column table add — the residual is carried by
  the C operand that E23 loaded.

The B blocks advance 4128 bytes per phase: `336448` (P24), `341632` (P25),
`346752` (P26), `351936` (P27), `357056` (P28), … `521920` (P60), `527104` (P61),
`532224` (P62), and each phase's C bias words sit in the 48-byte window 48 bytes
below the next phase's B base (e.g. P24's C at `341568..341616`, loaded at s18109).

### E61 — s23857–s24154: pack → plane-arena scatter → next B/C

298 statements.

```
s23857 L57579 cvt.rn.satfinite.e4m3x2.f16x2 %rs7231, %r34324   // first of 40 packs
s23917 L57718 ld.param.u64 %rd1050, [%rd1+-32]                  // S = plane arena (param+48)
s23918 L57719 cvta.to.global.u64 %rd5, %rd1050
s23920 L57723 shr.u32 %r958, %r34544, 2                         // g = laneid>>2
s23921 L57724 and.b32 %r959, %r34544, 3                         // t = laneid&3
s23924 L57727 add.s32 %r34630, %r937, %r34628                   // y = %r937 + (laneid>>5)
s23930 L57733 add.s32 %r34631, %r1, %r34629                     // x = %r1 + (g&7)
s23943 L57747 st.global.u32 [%rd1052], %r938                    // first of 20 stores
s24139 L57579-… ld.weak.global.ca.v4.u32                        // first of 5 next-B loads
s24153 ld.global.u32 …                                          // first of 2 next-C seeds
```

* 40 packs → 20 b32 `%r938…%r957`, which are simultaneously **phase 62's A
  fragments** — the same 20 registers are scattered to memory and fed to the last
  GEMM.
* `S = cvta.to.global.u64(param_0+48)` via `%rd1-32` (s23917–s23918) — the entry's
  only read of `param_0+48`.
* **20 `st.global.u32`** (first at s23943 with value `%r938`), each guarded by
  `setp.lt.s32 %p, %r34630, 0` / `setp.ge.u32 %p, %r34630, %r1014` /
  `setp.gt.u32 %p, %r34544, 63` combined with `or.pred` (s23925–s23929) and the
  matching x guards, then a `selp.b32 …, -1, …, %p1` + `setp.lt.s32` + `bra` that
  turns a guard failure into "skip the store".
* The index is `idx = (8*ex)*y + 8*x + d` with `%r7 = ex<<3` (s17),
  `y = %r937 + (laneid>>5)` where **`%r937 = %r3 + 2*%r9`** (s16691–s16692), and
  `x = %r1 + ((laneid>>2)&7)`, `d = t` (or `d = t|4`), exactly dec3's §5 formula.
* after the stores: 5 `ld.weak.global.ca.v4.u32` at
  `W + 16*laneid + {532224, 532736, 533248, 533760, 534272}` (s24138–s24147,
  `%r34660…%r34679`) = phase 62's B (10 fragments = 20 registers), then 2
  `ld.global.u32` at `W + ((laneid<<2)&12) + {534784, 534800}` (s24153–s24154,
  `%r34690`/`%r34700`) = phase 62's C seeds.

### E62 — s24165–s24446: softmax over five taps, 3×3 texture gather, surface write

282 statements; the entry's last statement is the `ret`.

```
s24165 L58082 bar.sync 0
s24170 L58089 shl.b32 %r997, %r34888, 4
s24177 L58096 st.shared.u32 [%r998], %r34761
s24181 L58101 st.shared.u32 [%r998+16], %r34771
s24198 L58121 add.s64 %rd1127, %rd1128, 80
s24209 L58132 add.s32 %r1003, %r1, %r1001                       // x
s24210 L58133 add.s32 %r1004, %r3, %r1002                       // y
s24215 L58138 ld.param.v2.u32 {%r34806, %r34807}, [%rd1127+-80]  // ex, ey
s24223 L58147 add.s64 %rd1129, %rd1130, 80
s24233 L58157 ld.param.u64 %rd1123, [%rd1129+32]                 // output surface (param+112)
s24246 L58191 (first of 12 mul.ftz.f32 by 0f3FB8AA3B)
s24247 L58192 ex2.approx.ftz.f32
s24328 L58360 ld.param.v2.f32 {%f1180, %f1181}, [%rd1129+56]     // param+136 scale pair
s24334 L58366 ld.param.u64 %rd1117, [%rd1129+8]                  // param+88 low-res texture
s24336 L58369 tex.base.2d.v4.f16.f32 {%rs7418,…,}, [%rd1101, {%f1140,%f1141}]
s24410 L58541 ld.param.u64 %rd1119, [%rd1129+16]                 // param+96
s24411 L58543 tex.base.2d.v4.f16.s32 {%rs7562,…,}, [%rd1119, {%r1003,%r1004}]
s24412 L58545 ld.param.u64 %rd1121, [%rd1129+24]                 // param+104
s24413 L58547 tex.base.2d.v4.f16.s32 {%rs7566,…,}, [%rd1121, {%r1003,%r1004}]
s24445 L58635 sust.b.2d.v4.b16.zero [%rd1123, {%r34877,%r1004}], {%rs7619,%rs7620,%rs7621,%rs7618}
```

Op histogram (s24165–s24446): 30 `fma.rn.f16`, 17 `mov.b32`, 16 `add.s32`,
13 `setp.lt.s32`, 12 each of `mul.ftz.f32` / `cvt.rn.f16.f32` / `selp.b32`,
10 each of `cvt.f32.f16` / `ex2.approx.ftz.f32` / `add.ftz.f32`,
9 `tex.base.2d.v4.f16.f32`, 8 `mov.u64`, 6 each of `mov.u32` / `and.b32` /
`shl.b32` / `bra` / `cvt.rn.f32.s32` / `fma.rn.ftz.f32`, 5 `ld.shared.u32`,
5 `sub.f16x2`, 5 `mul.f16x2`, and 1 `sust`.

| range | stmts | what |
|---|---|---|
| s24165–s24197 | 33 | `bar.sync` (s24165) + stage phase 62's D into `SM` at `(48*y + 6*x)` slots (4 `st.shared.u32`, s24177–s24194) |
| s24198–s24227 | 30 | pointer setup: `%rd1127 = %rd1128 + 80` (s24198), second `bar.sync` (s24199), the integer coordinates `%r1003 = %r1+%r1001` / `%r1004 = %r3+%r1002` (s24209–s24210), `(ex, ey)` read from `%rd1127-80` (s24215), bounds guard (s24216–s24222), `%rd1129 = %rd1130 + 80` (s24223) |
| s24228–s24241 | 14 | surface pointer `param+112` = `[%rd1129+32]` (s24233), **5 `ld.shared.u32`** reading 5 consecutive taps (s24228–s24234), and the row max: 3 `max.f16x2` (s24236–s24238) + 2 `max.f16` (s24240–s24241) |
| s24233 / s24243–s24326 | — | surface pointer `param+112` (s24233); softmax: 5 `sub.f16x2` (s24243–s24295), 12 `mul.ftz.f32` (s24246–), 10 `ex2.approx.ftz.f32` (s24247–), `div.approx.ftz.f32` (s24304, s24418), 5 `mul.f16x2` (s24307–s24311), 3 `cvt.rn.f16.s32` (s24312–s24314), 12 `selp.b32` (s24319–) |
| s24327–s24333 | 7 | clamped coordinates `clamp(p-1, 1, ex-2)` / `clamp(p-1, 1, ey-2)` (6 `cvt.rn.f32.s32` s24327–, 6 `fma.rn.ftz.f32` s24330–, `ld.param.v2.f32` s24328) |
| s24334–s24410 | 77 | 9 `tex.base.2d.v4.f16.f32` (s24336–s24406) at the 3×3 stencil of `param+88`, accumulated with 30 `fma.rn.f16` (s24338–s24427) |
| s24411–s24413 | 3 | 2 `tex.base.2d.v4.f16.s32` at `param+96` (s24410–s24411) and `param+104` (s24412–s24413), both at `{%r1003, %r1004}` |
| s24414–s24444 | 31 | `sigmoid`-like blend (s24418 `div.approx`, s24421 `sub.f16`, 3 `mul.f16` s24422–s24426, 3 `fma.rn.f16`) then the per-channel NaN guard on `param+104`: 3 `set.nan.f16.f16` (s24428–s24438) → `setp.ne.s16` → `@p bra` → 3 `add.f16` (s24431–s24441) |
| s24445 | 1 | `sust.b.2d.v4.b16.zero` surface write |

**Where the +96/+104 reads go.**  `%rd1129+16` = `param+96` (s24410) and
`%rd1129+24` = `param+104` (s24412).  Both are `tex.base.2d.v4.f16.s32` reads at the
same integer coordinate `{%r1003, %r1004}` = `(x, y)` (s24209–s24210);
`param+96`'s channels are added unconditionally, while `param+104`'s are added
**per channel only when the channel is not NaN** (`set.nan.f16.f16`).  The surface
written is `%rd1123 = param+112` (s24233), and the write lands at
`(%r34877, %r1004)` with `0` in the fourth channel (s24445).

**What the +24 skip input supplies.**  `param_0+24` (reached as `%rd1-56`, s3640)
supplies the f16 image values that E1 adds to the shared-staged features (s9217
onward) to form the residual.  That residual is used in E1's own normalisation
(`mul.f16`) and is added to the per-column table at `W+325184` in E13
(s16182–s16341) to seed phase 14's C operand.

## 5. The plane-arena writes (`param_0+48`)

`S = cvta.to.global.u64(%rd1050)` with `%rd1050 = ld.param.u64 [%rd1+-32]`
(s23917–s23918) is the **only** occurrence of `param_0+48` in the entry.  There are
exactly two store groups in the whole entry:

**(i) E61, 20 × `st.global.u32 [%rd1052 + …], v` — s23943 … s24129.**  The first
is `s23943 L57747 st.global.u32 [%rd1052], %r938` and the last is
`s24129 L57971 st.global.u32 [%rd1090], %r957`; the values are `%r938…%r957` in
file order (the same 20 registers that become phase 62's A).  Each is guarded as
described in §4/E61:

```
idx = (8*ex)*y + 8*x + d,  guard  y >= 0, y < ey, laneid <= 63, x >= 0, x < ex
y = %r937 + (laneid>>5),  %r937 = %r3 + 2*%r9,  x = %r1 + ((laneid>>2)&7),
d = t (first of each pair) | d = t|4, with t = laneid&3
where ex = %r1013, ey = %r1014 (s15), %r1 = block origin x (s7), %r3 = block origin y (s14)
```

**(ii) E62, the shared staging** at s24177/s24181/s24190/s24194: `st.shared.u32`
into `SM`, not the arena.

**What is written:** phase 61's requantised 16×160 output (20 u32 per lane).
**What reads it back:** nothing inside this kernel — `param_0+48` is never the base
of an `ld.*`, and phase 62's A comes from registers (`%r938…%r957`, E61's own
packs).  See §6.

## 6. Uncharacterised

**U1. The shared-memory layout of E1's first scatter.**  dec4 writes with
`byte = 4 * ((col&3) | (40*row + 400*(col>>2)))` (s1107–s1113) into a 16 KB array,
rather than dec3's literal `+{0,1600,…,11200}` plane offsets.  I did not reduce the
`(row, col)` permutation of `laneid` (built with `mul.wide.u32 %rd204, %r4338,
-1431655765`, s1051) to a closed form, nor map each plane to a source element of the
800-column phase-1 output.  *Missing:* a shared-memory dump or the producer's D
layout.

**U2. `%r16781 = %r9*13824` (s11607) feeding `%rd381` (s11608).**  `%rd381` is used
at s11645 as the slab term `W + z*27648` for phase 2's B.  The *derivation* `13824*2`
looks like a per-slab stride expressed for a different tile width, so the constant
13824 is presumably the per-slab output-tile size divided by 2, but nothing in the
PTX states it.  *Missing:* the weight-image layout description.

**U3. The identity of the per-column tables.**  `W+325184` (E13), `W+336144` (E22),
`W+103680…103984` (E1's gain), `W+325504…325808` (E21's gain) are all read with the
same `W + ((laneid<<2)&12) + off` pattern.  Their *roles* are computable (E13: added
to the skip before phase 14; E22: added after the activation; E1/E21: multiply the
inverse std) but their *names* are not.  The same is true of the GEMM biases at
`102400…102704` (P12), `126528`-adjacent words and `534784/534800` (P62).
*Missing:* the weight image's `_prep` permutation or the model's parameter names.

**U4. The identity of the 12288-byte score bias.**  Phase 12 adds a 12288-byte f16
table (`W + z*27648 + 114240 … 126028`) as its GEMM C operand, laid out as §4/E11
describes.  Whether it is a relative position bias, a learned score bias, or a
per-lane constant cannot be decided from the PTX.

**U5. The geometry of the banded phases.**  Twenty-three phases print an M/N/K that
fails `M/16·N/8·K/32 == mma` (P1, 13, 22–60 even, 62).  For the even phases
22…60 I verified the 5-step cascade shape (A = 5 fragments, B = 20 fragments,
`mma = 20 = 5*4` with the C-chain running over the A fragments), and for P62
likewise (A = 5 fragments, B = 10 fragments, `mma = 10 = 5*2`).  For **P1**
(300 mma, 100 B fragments, 250 distinct C/D pairs) and **P13** (48 mma, C set of 65
registers) I did not reduce the full (M, N, K) reading.  The epilogue contracts
above do not depend on resolving this.

**U6. The logical meaning of E62's blend.**  E62 computes a blend of a 3×3
`tex.base.2d.v4.f16.f32` gather of `param+88` (weighted by a 5-tap softmax) with the
`param+96` texture and, per channel under a NaN guard, the `param+104` texture, then
writes `param+112` at `(8·x, y)`.  The PTX gives the arithmetic exactly but not
which model term this is.  *Missing:* the high-level source or a reference.

**U7. The count of distinct input-buffer strides.**  dec4's prologue/`E1` use
×25600 (s438), ×160 (s543) and ×40 (s1052) for `%r9` in addition to the shared and
weight strides.  I did not determine which of the three addresses the prologue's
gather (`param+8`) versus E1's gather (`param+24`) versus some other buffer.
*Missing:* a mapping from `%r9` to buffer identity.

## 7. Quick acceptance index

| requirement | where |
|---|---|
| patch expand | §4/E1 (s871–s11653) and phase 1 (§2) |
| the norms | E1's rsqrt block (s9215–s10956, eps s10075) and E21 (s16691–s17719, eps s16920) |
| the attention | E11 (score seeds + packs), E12 (softmax + `movmatrix` s15756–s15803 + packs), E13 (P·V staging) |
| the activations | E22, E24, …, E60 — the clamped cubic `x·(0.5 + y·(0.412109375 − 0.0810546875·|y|))`, `y = clamp(x,±2)`; constants s17741–s17750 |
| the per-launch table in the plane arena | §5, s23917/s23943ff (`param_0+48`) |
| every epilogue of every phase | §3 summary table + §4 (E1–E62) |
| texture reads at +96/+104 and where the surface write goes | §4/E62, s24410–s24413 and s24233/s24445 |
| what the skip input at +24 supplies | §1.1 (s3640) and §4/E1, E13 |
| phase table M/N/K arithmetic checked | §2 (23 phases fail; listed) |
| weight-image extent | §2 (≥ 534804 bytes; last read s24153 + 2 words) |
| phases with no weight operand | §2: 12 and 13 |
| uncharacterised | §6 |
