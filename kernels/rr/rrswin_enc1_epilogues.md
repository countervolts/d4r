# `cuda_dldn_engine_swin_enc1_kernel` — the epilogues of phases 1…32

Companion to `kernels/rr/rr_layer_spec.py enc1`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0014-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_enc1_kernel`.

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at their first line).  **`LNNNNN`
  is the physical line** of the same instruction in the corpus file, so every
  claim can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and the
  next phase's first `mma`.  The phase table (`rr_layer_spec.py enc1`) gives the
  ranges; this file names what each one *computes*.
* `W` denotes the prepared weight image: `ld.param.u64 %rd2,
  [cuda_dldn_engine_swin_enc1_kernel_param_0+40]` (s11) and
  `cvta.to.global.u64 %rd3, %rd2` (s12).  Every `ld.weak.global.ca.v4.u32` in an
  epilogue reads `W + 16*laneid + imm`.
* `S` denotes the destination buffer at `param_0+48`: `ld.param.u64 %rd337,
  [%rd1+-32]` (s25221) with `%rd1 = param_0+80` (s4).  The buffer at
  `param_0+56` is `ld.param.u64 [%rd535+-24]` (s25963), same base.
* The C/D register layout and the weight-addressing formula are fixed (see the
  settled notes): `C/D reg0 = (row g, cols 2t,2t+1)`, `reg1 = (row g+8, same cols)`,
  `g = laneid>>2`, `t = laneid&3`.  A reg = `reg0 = (row g, k=2t+(j&1)+16*((j>>1)&1))`,
  `reg1 = (row g+8, same k)`, `reg2 = (row g, k+8)`, `reg3 = (row g+8, k+8)`.
* Block origins, defined once in the prologue and read by every epilogue:
  `%r1 = (ctaid.x<<3) - ext.x` (s3, s7), `%r7 = ((ctaid.y+tid.y)<<3) - ext.y`
  (s10, s16, s19), `%r5 = ext.x<<3` (s17), `%r6 = ext.y * %r5` (s18),
  `%r465 = ext.x`, `%r466 = ext.y` (s13).  Every `add.s32 %rX, %r7, …` is a y
  origin and every `add.s32 %rX, %r1, …` an x origin.
* Anything I could not pin down from the PTX is in §6 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `W` (weight image, ≥96480 bytes) | `ld.param.u64 %rd2, [param_0+40]` / `cvta.to.global.u64 %rd3, %rd2` | s11 / s12 |
| `S` (destination buffer) | `ld.param.u64 %rd337, [%rd1+-32]` with `%rd1 = param_0+80` | s4, s25221–s25222 |
| `D` (2nd destination buffer) | `ld.param.u64 %rd455, [%rd535+-24]`, `%rd535 = param_0+80` | s25635, s25963–s25964 |
| input tile buffer | `ld.param.u64 %rd12, [param_0+8]` / `cvta.to.global.u64 %rd4, %rd12` | s21 / s22 |
| extents `(ex, ey)` | `ld.param.v2.u32 {%r465, %r466}, [param_0+0]` | s13 |
| `%r1`, `%r5`, `%r6`, `%r7` | `sub.s32 %r1, %r460, %r461`; `shl %r5, %r465, 3`; `mul %r6, %r466, %r5`; `sub.s32 %r7, %r468, %r467` | s7, s17, s18, s19 |
| per-lane weight address | `mul.wide.u32 %rd, %laneid, 16` + `add.s64 %rd, W, %rd` | every load block |
| per-lane *table* address | `shl.b32 %r, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rd, %r, 12` → `add.s64 %rd, %rd3, %rd` = `W + ((laneid*4) & 12)` | s12509–s12512 and copies |

The last line matters: the eight `ld.global.v2.u16` / four `ld.global.u32` loads
in an epilogue read `W + ((laneid<<2)&12) + off`, i.e. **the same 64-byte record
is read by the four lanes with equal `laneid&3`, and the eight loads pick eight
16-byte words inside it.**  So those tables are indexed by column, not by row.

`param_0+80` is used *as a base as well as* a value: `ld.param.v2.u16 {%rs103,
%rs104}, [param_0+80]` (s5) reads two u16 at the offset that `%rd1 = param_0+80`
(s4) points at, and every later param read is relative to `param_0+80`:
`[%rd1+-32]` = +48, `[%rd535+-24]` = +56, `[%rd535+-80]` = +0,
`[%rd539+8]` = +88, `[%rd539+32]` = +112, `[%rd539+40]` = +120,
`[%rd539+48]` = +128.

### 1.2 Constants that survive between epilogues

| register | first definition | f32/f64 literal | value used (after the `cvt`) |
|---|---|---|---|
| `%f956` | s8607 `mov.f32 %f956, 0f3C8CCB50` | 0.017186790704727173 | f16 `0x2466` = 0.017181396484375 |
| `%fd767` | s8611 `mov.f64 %fd767, 0dBFE1CC0000000000` | −0.55615234375 | f16 `0xB873` = −0.55615234375 |
| `%fd769` | s8616 `mov.f64 %fd769, 0d3FE1CC0000000000` | +0.55615234375 | f16 `0x3873` = +0.55615234375 |
| `%f958` | s8622 `mov.f32 %f958, 0f3F6D6000` | 0.92724609375 | f16 `0x3B6B` = 0.92724609375 |
| `%f960` | s8627 `mov.f32 %f960, 0f3FB00000` | 1.375 | f16 `0x3D80` = 1.375 |
| `%f1981` | s18793 `mov.f32 %f1981, 0f3ED306EB` | 0.4121621549129486 | f16 `0x3698` = 0.412109375 |
| `%f1982` | s18796 `mov.f32 %f1982, 0f3DA60DD6` | 0.0810810774564743 | f16 `0x2D30` = 0.0810546875 |
| `%f1983` | s18799 `mov.f32 %f1983, 0f3F000000` | 0.5 | f16 `0x3800` = 0.5 |
| `%f1984` | s18802 `mov.f32 %f1984, 0f40000000` | 2.0 | f16 `0x4000` = 2.0 |
| `%fd770` | s6471 `mov.f64 %fd770, 0d3F20000000000000` | 2⁻¹³ | f16 `0x0800` = 0.0001220703125 |

Also live across the region (defined in the prologue, read by every warp
reduction): `%r16376` = `mov.u32 %r16376, 1` (s5731), `%r16385` =
`mov.u32 %r16385, 2` (s6296), `%r16387` = `mov.u32 %r16387, -1` (s6289) —
the three operands of every `shfl.sync.bfly.b32`.

`%f1981…%f1984` are defined **inside E15** (phase 15's epilogue) and reused by
the epilogues of phases 17, 19, 21, 23, 25, 27 and 29 — those epilogues contain
no `mov.f32` at all (E17 at s19785 has none; E19's histogram at s20569 likewise).
`%f956/%fd767/%fd769/%f958/%f960` are defined inside E5 and reused by E10/E12.

### 1.3 Two quantisation helpers

* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).
* `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `div.approx.ftz.f32` — always wrapped as
  `cvt.f32.f16 → op → cvt.rn.f16.f32` (round each step through f16), except in
  E32 where the f32 softmax is kept in f32 until its final `cvt.rn.f16.f32`.

## 2. Summary

### 2.1 Phase table with the M/N/K arithmetic checked

The `rr_layer_spec.py` phase table's `M`,`N`,`K` columns come from counting
distinct A and B fragments, which is unambiguous only for a one-k32-step phase.
For every phase I followed the `D → C` chains of the phase's own `mma` run and
counted `k_steps`, `m_tiles = distinctA/k_steps`, `n_tiles = distinctB/k_steps`:

| # | stmts | mma | k_steps | M | N | K | `M/16·N/8·K/32 == mma` | spec row |
|---|---|---|---|---|---|---|---|---|
| 1 | s8270–s8293 | 24 | 1 | 96 | 32 | 32 | 24 == 24 OK | 96/32/32 OK |
| 2 | s8301–s8324 | 24 | 1 | 96 | 32 | 32 | 24 OK | OK |
| 3 | s8332–s8355 | 24 | 1 | 96 | 32 | 32 | 24 OK | OK |
| 4 | s8363–s8386 | 24 | 1 | 96 | 32 | 32 | 24 OK | OK |
| 5 | s8558–s8605 | 48 | 1 | 64 | 96 | 32 | 48 OK | OK |
| 6 | s12449–s12496 | 48 | **3** | **64** | **32** | **96** | 4·4·3 = 48 OK | spec says 192/96/96 → **wrong** |
| 7 | s12761–s12792 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 8 | s12800–s12823 | 24 | 1 | 96 | 32 | 32 | 24 OK | OK |
| 9 | s12831–s12854 | 24 | 1 | 96 | 32 | 32 | 24 OK | OK |
| 10 | s12862–s12885 | 24 | 1 | 96 | 32 | 32 | 24 OK | OK |
| 11 | s12893–s12916 | 24 | 1 | 96 | 32 | 32 | 24 OK | OK |
| 12 | s13088–s13135 | 48 | 1 | 64 | 96 | 32 | 48 OK | OK |
| 13 | s16974–s17021 | 48 | **3** | **64** | **32** | **96** | 48 OK | spec says 192/96/96 → **wrong** |
| 14 | s17081–s17112 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 15 | s18760–s18791 | 32 | **2** | **64** | **32** | **64** | 4·4·2 = 32 OK | spec says 128/64/64 → **wrong** |
| 16 | s19605–s19636 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 17 | s19753–s19784 | 32 | **2** | **64** | **32** | **64** | 32 OK | spec says 128/64/64 → **wrong** |
| 18 | s20389–s20420 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 19 | s20537–s20568 | 32 | **2** | **64** | **32** | **64** | 32 OK | spec says 128/64/64 → **wrong** |
| 20 | s21173–s21204 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 21 | s21321–s21352 | 32 | **2** | **64** | **32** | **64** | 32 OK | spec says 128/64/64 → **wrong** |
| 22 | s21957–s21988 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 23 | s22105–s22136 | 32 | **2** | **64** | **32** | **64** | 32 OK | spec says 128/64/64 → **wrong** |
| 24 | s22741–s22772 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 25 | s22889–s22920 | 32 | **2** | **64** | **32** | **64** | 32 OK | spec says 128/64/64 → **wrong** |
| 26 | s23525–s23556 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 27 | s23673–s23704 | 32 | **2** | **64** | **32** | **64** | 32 OK | spec says 128/64/64 → **wrong** |
| 28 | s24309–s24340 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 29 | s24457–s24488 | 32 | **2** | **64** | **32** | **64** | 32 OK | spec says 128/64/64 → **wrong** |
| 30 | s25093–s25124 | 32 | 1 | 64 | 64 | 32 | 32 OK | OK |
| 31 | s25824–s25919 | 96 | **8** | **16** | **96** | **256** | 1·12·8 = 96 OK | spec says 128/768/256 → **wrong** |
| 32 | s26124–s26139 | 16 | **2** | **64** | **16** | **64** | 4·2·2 = 16 OK | spec says 128/32/64 → **wrong** |

Method, on phase 6 as the example: the 48 `mma` (s12449–s12496) split into 16
chains of 3, e.g. `s12449 → s12451 → s12453` (C of each is the D of the
previous), each chain consuming three A fragments and three B fragments.  So
`k_steps = 3`, `distinctA = 12 = 4 m_tiles × 3`, `distinctB = 12 = 4 n_tiles × 3`.
The spec's `M=192 N=96` double-counts the k-steps on both sides; the true shape
is `64×32 = (64×96)·(96×32)`, i.e. the same P·V GEMM enc0's phase 6 is.

Every row above satisfies `M/16 · N/8 · K/32 == mma` once `k_steps` is taken out
of the fragment counts.  **No row is left unexplained.**

### 2.2 Weight image extent

`rr_layer_spec.py --geometry` reports `enc1 … image=96468`, which is
`(highest immediate) + (width of that access)`.  That number is exact for the
`W + 16*laneid + imm` B-fragment loads but **undercounts the per-lane *table*
loads**, whose base is `W + ((laneid*4)&12)`: the highest such immediate is
`+96464` (`ld.global.u32 %r32027, [%rd485+96464]` at s26123, the second of phase
32's C seeds), and `%rd485 = %rd3 + ((laneid<<2)&12)` (s26118–s26121), so for
lanes with `laneid&3 == 3` the address reaches `96464 + 12 + 4`.

```
s26117 L66827: mov.u32 %r31887, %laneid
s26118 L66829: shl.b32 %r32050, %r31887, 2
s26120 L66831: and.b64 %rd484, %rd483, 12
s26121 L66832: add.s64 %rd485, %rd543, %rd484
s26123 L66834: ld.global.u32 %r32027, [%rd485+96464]
```

Taking the per-lane bases into account over every load in the entry, the highest
byte touched is **96480**; the largest `W + 16*laneid + imm` load ends at 96448
(phase 32's B, `+95936` tile).  So the image spans at least **96480 bytes**.

### 2.3 Phases that read no weight operand

* **Phase 6** (`s12449–s12496`) and **phase 13** (`s16974–s17021`) read **no
  weight bytes at all**: both A and B come from registers.  Phase 6's A is the
  packed softmax of phase 5's D and its B is the `movmatrix` transpose of phase
  4's D, both packed in E5 (s12233–s12448, see §3/E5(g)); phase 13's A and B are
  packed in E12 (s16758–s16973) from phase 12's softmax and phase 11's D.  Their
  C operands are the zero register `%r15665` (§3/E5).
* **Phase 5** (`s8558–s8605`) and **phase 12** (`s13088–s13135`) read only a
  **C bias operand**, never a B operand: their B comes from registers packed by
  E4 (s8438–s8521) resp. E11 (s12968–s12990), while C is the 12288-byte score
  table at `W+4224…W+16512` (24 tiles, E4 s8390–s8437) resp. `W+22656…W+34848`
  (24 tiles, E11 s12920–s12967).

So phases 1–4, 7–11, 14–32 read weight operands; 5 and 12 read only a bias
table; 6 and 13 read neither.

### 2.4 Summary of the epilogues

| epilogue | range | stmts | named operation |
|---|---|---|---|
| E1 | s8294–s8300 | 7 | next-B weight loads |
| E2 | s8325–s8331 | 7 | next-B weight loads |
| E3 | s8356–s8362 | 7 | next-B weight loads |
| E4 | s8387–s8557 | 171 | score-bias C-seed loads (24 tiles) + pack of D3 and D4 |
| E5 | s8606–s12448 | 3843 | softmax of the 64×96 score block, V-transpose, pack |
| E6 | s12497–s12760 | 264 | input-residual + per-column table add, pack, next B |
| E7 | s12793–s12799 | 7 | next-B weight loads |
| E8 | s12824–s12830 | 7 | next-B weight loads |
| E9 | s12855–s12861 | 7 | next-B weight loads |
| E10 | s12886–s12892 | 7 | next-B weight loads |
| E11 | s12917–s13087 | 171 | score-bias C-seed loads (24 tiles) + pack of D10 |
| E12 | s13136–s16973 | 3838 | softmax (2nd window, same code as E5) |
| E13 | s17022–s17080 | 59 | pack of D13 → next A; next B loads |
| E14 | s17113–s18759 | 1647 | RMS norm + gain, 8 next-B / next-C loads, pack of the normed tile |
| E15 | s18792–s19604 | 813 | clamped cubic activation + column table add + pack |
| E16 | s19637–s19752 | 116 | next B/C loads + pack of the normed tile |
| E17 | s19785–s20388 | 604 | clamped cubic activation + pack |
| E18 | s20421–s20536 | 116 | next B/C loads + pack of the normed tile |
| E19 | s20569–s21172 | 604 | clamped cubic activation + pack |
| E20 | s21205–s21320 | 116 | next B/C loads + pack of the normed tile |
| E21 | s21353–s21956 | 604 | clamped cubic activation + pack |
| E22 | s21989–s22104 | 116 | next B/C loads + pack of the normed tile |
| E23 | s22137–s22740 | 604 | clamped cubic activation + pack |
| E24 | s22773–s22888 | 116 | next B/C loads + pack of the normed tile |
| E25 | s22921–s23524 | 604 | clamped cubic activation + pack |
| E26 | s23557–s23672 | 116 | next B/C loads + pack of the normed tile |
| E27 | s23705–s24308 | 604 | clamped cubic activation + pack |
| E28 | s24341–s24456 | 116 | next B/C loads + pack of the normed tile |
| E29 | s24489–s25092 | 604 | clamped cubic activation + pack |
| E30 | s25125–s25823 | 699 | pack → destination-buffer scatter → 8-lane merge → next B/C |
| E31 | s25920–s26123 | 204 | pack → 2nd-buffer scatter → `%tid.x` → next B/C |
| E32 | s26140–s26819 | 680 | cross-lane row max, softmax, texture gather, surface stores |

---

## 3. The epilogues

### E1–E3, E7–E10 — s8294–s8300, s8325–s8331, s8356–s8362, s12793–s12799, s12824–s12830, s12855–s12861, s12886–s12892: next-B loads

Seven statements each, all the same shape:

```
s8294 L13511: mov.u32 %r5787, %laneid
s8298 L13517: ld.weak.global.ca.v4.u32 { %r5788,%r5789,%r5790,%r5791},[%rd105]
s8300 L13521: ld.weak.global.ca.v4.u32 { %r5792,%r5793,%r5794,%r5795},[%rd106]
```

`mul.wide.u32` + two `add.s64` + two `ld.weak.global.ca.v4.u32` =
`W + 16*laneid + {P, P+512}` for the next phase's 4 B fragments.  Offsets:

| epilogue | phase it feeds | offsets |
|---|---|---|
| E1 | 2 | +2176, +2688 (s8296, s8299) |
| E2 | 3 | +1152, +1664 |
| E3 | 4 | +3200, +3712 |
| E7 | 8 | +18560, +19072 |
| E8 | 9 | +20608, +21120 |
| E9 | 10 | +19584, +20096 |
| E10 | 11 | +21632, +22144 |

### E4 — s8387–s8557: score-bias C-seed loads + pack of D3

171 statements: **24 `ld.weak.global.ca.v4.u32`** (s8391–s8437, one per `add.s64`
at s8390–s8436) and **80 packs / 40 b32** (s8438–s8557).

1. **Score bias** (s8388–s8437): `ld.weak.global.ca.v4.u32 …,
   [W + 16*laneid + 4224 + 512*i]`, `i = 0…23` — the phase-5 score-bias C seeds,
   byte block `[4224, 16512)` = 12288 bytes = 6144 f16 = 24 tiles of 512 bytes.
   The last offset is `+16000` (s8436, `%rd134`), whose 16 bytes end at 16512.
2. **Packs** (s8438–s8557): 80 `cvt.rn.satfinite.e4m3x2.f16x2` + 40 `mov.b32`,
   whose sources are **all 48 registers of phase 3's D** (`%r6045…%r6276`,
   verified: the set of cvt source registers is exactly that range).  They form
   *both* operands of phase 5:
   * `%r7077 %r7078`, `%r7087 %r7088`, … `%r7187 %r7188` (24 b32) = phase 5's
     **B** fragments, built by shuffling phase 3's D (`s8438: cvt …%r6065`,
     `s8439: cvt …%r6045`, `s8440: mov.b32 %r7077, {%rs2730,%rs2731}`); and
   * `%r6823 %r6824 %r6825 %r6826`, `%r6943…`, `%r7063…`, `%r7183…` (16 b32) =
     phase 5's **A** fragments, also from phase 3's D (`s8512: mov.b32 %r6823,
     {%rs2778, %rs2779}` from `%r6045`/`%r6065`).

   So phase 5 computes `scores = pack(D3)·pack(D3)` — **Q and K are the same
   projection**, and the V operand is phase 4's D, which E5's `movmatrix` block
   transposes into phase 6's B (§3/E5(f)).

**The C-seed layout** (identical for phase 5 and phase 12): the four C registers
of one mma are the four `u32` of one 16-byte lane slot, and

```
C(m, n) = tile(4*(n>>1) + m) + 4*(n&1)      [tile index within the block]
byte    = base + 512*(4*(n>>1) + m) + 16*laneid + 8*(n&1)
```

so the tile index runs `4*(n>>1) + m` (stride 512 bytes) over 4 m-tiles and
24 `(n>>1)` groups = 24 tiles = 12288 bytes.  Verified from phase 5's own
operands: `s8558 C={%r6535,%r6536}` … `s8570 C={%r6539,%r6540}` (the second
m-tile row), i.e. consecutive `v4` registers from one 16-byte slot.

### E5 — s8606–s12448: softmax of the 64×96 score block

**Phase 5** = 48 `mma` (s8558–s8605), D registers `%r6711…%r7182` (96 f16x2 =
64 rows × 96 columns held 8 rows × 24 cols per lane).  The epilogue has **no memory access at all** (0 `ld.*`,
0 `st.*`); it reads **phase 5's D** (the 96 score registers, for the softmax) and
**phase 4's D** (through the `movmatrix` block), plus the constants, and writes
phase 6's operands.

Sub-sections (boundaries found from the op sequence; the per-element group is
22 statements and there are exactly 96 `and.b32 …, 2145419232`):

| range | stmts | what |
|---|---|---|
| s8606–s10723 | 2118 | 96 per-element groups: scale, clamp, cubic, exponent extract |
| s10724–s10803 | 80 | partial row-sum trees (in-lane `add.f16x2`) |
| s10804–s10939 | 136 | 16 `shfl.sync.bfly` → 8 row sums |
| s10940–s11701 | 762 | 192 `rcp.approx.ftz.f32` → 96 reciprocals |
| s11702–s12184 | 483 | 192 `mul.f16` → 96 probabilities |
| s12185–s12232 | 48 | `movmatrix` transpose of phase 4's D |
| s12233–s12448 | 216 | 72 packs (144 `cvt.rn.satfinite.e4m3x2.f16x2`) |

#### (a) per-element exponent, s8606–s10723

```
s8607 L14773: mov.f32 %f956, 0f3C8CCB50
s8610 L14781: mul.f16x2 %r7193,%r6711,%r7192
s8615 L14795: max.f16x2 %r7196,%r7193,%r7198
s8620 L14809: min.f16x2 %r7199,%r7196,%r7201
s8626 L14827: fma.rn.f16x2 %r7204,%r7199,%r7202,%r7207
s8631 L14841: fma.rn.f16x2 %r7208,%r7199,%r7204,%r7211
s8632 L14844: shl.b32 %r30241, %r7208, 5
s8633 L14845: and.b32 %r9113, %r30241, 2145419232      // 0x7FE07FE0
```

For each of the 96 f16x2 D registers, per half (an f16 value `x`):

```
m  = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)   // s8610,s8615,s8620
t  = f16(m*(-m) + 0.92724609375)                                         // s8626 fma
u  = f16(m*t    + 1.375)                                                 // s8631 fma
    = f16(1.375 + m*(0.92724609375 - m^2))
E  = bits(u)[9:5]      // floor(32*frac(u))
F  = bits(u)[4:0]      // the low five mantissa bits
expval = f16 from bits  (E << 10) | (F << 5)      // = 2^(E-15) * (1 + F/32)
```

`((u_bits << 5) & 0x7FE07FE0)` keeps `u`'s mantissa and drops sign and exponent,
so the mantissa lands in the exponent field.  Numerically
`expval ≈ 2^(32*frac(u) - 15) = 2^(32u - 47)` and, because the clamp keeps
`u ∈ [1.03125, 1.71875]`, `expval ∈ [2⁻¹⁴, 2⁸]`.  **This is not `exp` and not
`exp2(x*log2e)`**: the exponent is the *cubic* `32*(0.375 + 0.92724609375*m − m³)`
quantised to a 6-bit significand (1.5 % relative granularity).  The two `f64`
literals exist only so the clamp is the *exact* f16 value.

#### (b) in-lane partial sums, s10724–s10803

80 `add.f16x2` reduce the 96 `expval` registers in tree fashion:

```
s10724 L21307: add.f16x2 %r9112,%r9113,%r9114
s10750 L21411: add.f16x2 %r9190,%r9191,%r9192
s10782 L21539: add.f16x2 %r9286,%r9274,%r9262
```

Each lane reduces its 96 values (8 rows × 12 registers) to 16 f16x2 partials.

#### (c) cross-lane row sums, s10804–s10939

```
s10810 L21641: shfl.sync.bfly.b32 %r9357,%r9353,%r16376,%r9360,%r16387
s10811 L21645: add.f16x2 %r9362,%r9353,%r9357
s10816 L21656: shfl.sync.bfly.b32 %r9366,%r9362,%r16385,%r9369,%r16387
s10817 L21660: add.f16x2 %r9371,%r9362,%r9366
s10818 L21663: mov.b32 {%rs3579, %rs3580}, %r9371
s10819 L21665: add.f16 %rs3578,%rs3579,%rs3580
s10820 L21668: mov.b32 %r9568, {%rs3578, %rs3578}
```

Eight times (one per row `g, g+8` of the four m-tiles): add the two f16x2
partials, butterfly over lane bits 0 and 1 (`%r16376 = 1`, `%r16385 = 2`), add
the two halves with `add.f16`, then broadcast into both halves.  Result:
**8 row sums** `%r9568`, `%r9572`, … (rows `g`, `g+8` of m-tiles 0…3), covering
all 96 columns because the four lanes with equal `laneid&7` hold different
column pairs.

The shuffle's `c` operand is built as `(WARP_SZ<<8) - 8192 | 31` (s10807–s10809)
rather than a plain constant; since `WARP_SZ = 32`, that is `8192 - 8192 | 31 = 31`.

**There is no row maximum anywhere in this epilogue**: the only `max.f16x2`
(96×) and `min.f16x2` (96×) are the clamp of (a); the only warp shuffles are the
16 `shfl.sync.bfly` above.  See §6.

#### (d) reciprocal, s10940–s11701

192 `rcp.approx.ftz.f32` = 96 f16x2 reciprocals, each computed as
`cvt.f32.f16 → rcp.approx.ftz.f32 → cvt.rn.f16.f32` on a row-sum register
(duplicated, so each row sum is inverted once per n-tile of that row):

```
s11025 L22107: mov.b32 {hl, hu}, %r9572
s11028 L22110: rcp.approx.ftz.f32 fl, fl
s11029 L22111: rcp.approx.ftz.f32 fu, fu
s11032 L22114: mov.b32 %r9547, {hl, hu}
```

#### (e) probabilities, s11702–s12184

```
s11705 L23209: mov.b32 {%rs3603, %rs3606}, %r9113      // expval (row g)
s11706 L23210: mov.b32 {%rs3604, %rs3607}, %r9525      // 1/rowsum
s11707 L23212: mul.f16 %rs3605,%rs3606,%rs3607
s11708 L23216: mul.f16 %rs3602,%rs3603,%rs3604
s11709 L23219: mov.b32 %r9861, {%rs3602, %rs3605}
```

192 `mul.f16` → 96 f16x2 results `p = expval * (1/rowsum)` — the row sums
cancel the constant `2⁻⁴⁷` of (a), so `p_ij ∝ 2^(32·u_ij)` and
`Σ_j p_ij = 1` up to the `rcp.approx` error.

#### (f) V-transpose, s12185–s12232

```
s12185 L24266: movmatrix.sync.trans.aligned.m8n8.b16 %r9717, %r6294
s12186 L24269: movmatrix.sync.trans.aligned.m8n8.b16 %r9719, %r6295
s12189 L24278: movmatrix.sync.trans.aligned.m8n8.b16 %r9725, %r6334
```

**48** `movmatrix` on 48 distinct **phase 4 D** registers (`%r6294…%r6525`, each
exactly once — the source list runs `%r6294 %r6295 %r6304 %r6305 %r6334 …
%r6524 %r6525`).  Each instruction transposes one 8×8 b16 tile built from the
32 threads, i.e. 48 × 64 = 3072 values = the whole 96×32 D tile.

#### (g) packs, s12233–s12448

72 packs, in two groups (classified by which phase-6 operand register they
write):

1. **s12235–s12304** — 24 packs from the `movmatrix` results → phase 6's
   **12 B fragments** (`%r10323 %r10333 %r10343 … %r10423 %r10433`, 24 b32):
   `s12235: mov.b32 %r10323, {%rs4178, %rs4179}` where `%rs4178/%rs4179` come
   from `%r9717`/`%r9725` (s12233–s12234).
2. **s12307–s12448** — 48 packs from the **probabilities** → phase 6's
   **12 A fragments** (`%r10029 %r10031 %r10049 … %r10427 %r10428`, 48 b32):
   `s12307: mov.b32 %r10029, {%rs4226, %rs4227}` where `%rs4226/%rs4227` come
   from `%r9861`/`%r9862` — probability products (s11709 s11789).

**Inputs:** phase 5's 96 D f16x2, phase 4's 48 D f16x2, `%r16376/%r16385/%r16387`.
**Outputs:** the 5 constants + phase 6's 48 A-register / 24 B-register operands.
**Memory read/written:** none.

### E6 — s12497–s12760: input residual + per-column table, then pack

264 statements; op mix: `add.f16:128, mov.b32:80, cvt…e4m3:32, ld.global.v2.u16:8,
ld.weak.global.ca.v4.u32:4`, 6 × `add.s64`.

```
s12501 L25256: ld.weak.global.ca.v4.u32 { %r10438,%r10439,%r10440,%r10441},[%rd135]
s12509 L25273: shl.b32 %r30369, %r10454, 2
s12512 L25276: add.s64 %rd262, %rd3, %rd261          // W + ((laneid<<2)&12)
s12513 L25277: ld.global.v2.u16 {%rs9161, %rs9162}, [%rd262+37008]
s12520 L25284: ld.global.v2.u16 {%rs9175, %rs9176}, [%rd262+36992]
s12521 L25286: add.f16 %rs4325,%rs9176,%rs4327
s12522 L25290: add.f16 %rs4322,%rs9175,%rs4324
s12523 L25293: mov.b32 %r10495, {%rs4322, %rs4325}
s12713 L25862: cvt.rn.satfinite.e4m3x2.f16x2 %rs4707, %r10057
s12715 L25867: mov.b32 %r10559, {%rs4706, %rs4707}
```

Four things happen, in this order:

1. **Next-B loads** (s12497–s12507): four `v4.u32` fragments from
   `W + 16*laneid + {16512, 17024, 17536, 18048}` → `%r10438…%r10453`, phase 7's
   B operand (weight block `[16512, 18060)`).
2. **Table loads** (s12508–s12520): **eight** `ld.global.v2.u16` from
   `W + ((laneid<<2)&12) + {36992, 37008, 37024, 37040, 37056, 37072, 37088,
   37104}` → 16 f16 (`%rs9161…%rs9176`).  16 bytes/lane × 32 lanes × 8 words =
   4096 bytes = **a per-column vector covering 32 columns** (8 words × 4 lanes
   with distinct `laneid&3`).
3. **Residual + column table add** (s12521–s12712): 128 `add.f16`, i.e. 64 f16x2,
   computed on the halves of the **dequantised input tile** — each
   `%rs4324/%rs4327` is unpacked at s5827 `mov.b32 {%rs4324, %rs4327}, %r4678`
   from `cvt.rn.f16x2.e4m3x2 %r4678, %rs107` (s5730), and `%rs107` is
   `ld.global.v2.u16 …` at s173 from the `param_0+8` buffer.  So:

   ```
   new_low  = table[2t]   + input_low      s12522
   new_high = table[2t+1] + input_high     s12521
   → packed into %r10495, %r10496, %r10505, %r10506, … (64 registers)
   ```

   These 64 registers are **phase 7's C operand** (`s12761` first mma
   `C={%r10495,%r10496}`), so phase 7 computes
   `D7 = A7·B7 + (input_tile + column_table)` — the attention out-projection
   plus the *un-normalised input* residual, exactly enc0's E6 shape but with the
   input patch instead of a second GEMM output.
4. **Pack** (s12713–s12760): 32 packs of **phase 6's D** (`%r9957…%r10428`,
   16 b32 = phase 7's A operand `%r10559` … `%r10802`) — the P·V output
   requantised for the out-projection.

**Outputs (80 live):** `%r10438…%r10453` (B), the 64 C seeds `%r10495…%r10806`,
the 16 packed A registers.  **Memory:** 4 weight loads + 8 table reads.

### E11 — s12917–s13087: score-bias C-seed loads + pack of D10

171 statements, identical in shape to E4 with different offsets:

* **24 loads** at `W + 16*laneid + 22656 + 512*i`, `i = 0…23` (s12920–s12967) —
  the phase-12 score-bias seeds, byte block `[22656, 34848)` = 12288 bytes.
* **80 packs / 40 b32** (s12968–s13087), whose sources are all registers written
  by **phase 10's mma** (`%r11314…%r11465`, the D pairs of s12862–s12885).
  Phase 11's D (`%r11555…`) is **not** read here; it is consumed by E12's
  `movmatrix` block instead (s16710: `movmatrix… %r14986, %r11563`).
  The 40 packed b32 split into phase 12's A fragments
  (`s12968: cvt …%r11334`, `s12969: cvt …%r11314`, `s12970: mov.b32 %r12346,
  {%rs4738, %rs4739}`) and phase 12's B fragments.

### E12 — s13136–s16973: softmax, second window

**Identical code to E5** with shifted registers; the op histogram is
byte-for-byte the same (1048 `mov.b32`, 672 `cvt.rn.f16.f32`, 384
`cvt.rn.f16.f64`, 192 each `fma.rn.f16x2` / `cvt.f32.f16` /
`rcp.approx.ftz.f32` / `mul.f16`, 144 packs, 48 `movmatrix`).  It re-uses E5's
constants (`%f956 %fd767 %fd769 %f958 %f960`, defined at s8607–s8627) and E5's
shfl operands `%r16376 %r16385 %r16387`, so it contains **no `mov.f32`/`mov.f64`
of its own**.

Sub-sections (by the same markers):

| range | what |
|---|---|
| s13136–s15248 | 96 exponent groups (`and.b32 …, 2145419232`, s13158…s15248) |
| s15249–s15457 | 80 in-lane partial-sum `add.f16x2` |
| s15335–s15456 | 16 `shfl.sync.bfly` → 8 row sums |
| s15465–s16226 | 192 `rcp.approx.ftz.f32` |
| s16232–s16708 | 192 `mul.f16` → 96 probabilities |
| s16710–s16757 | 48 `movmatrix` on phase 11's D (`s16710 L37129`) |
| s16758–s16973 | 144 packs (72 `mov.b32`) |

(The `movmatrix` block is *before* the reciprocal block in program order here,
unlike enc0 where it also sits late; the statement index is what identifies it.)

**Inputs:** phase 12's 96 D (`%r11980…`), phase 11's 48 D.  **Outputs:** phase
13's 12 A fragments + 12 B fragments + the 5 constants.  **Memory:** none.

### E13 — s17022–s17080: pack of D13, next B

59 statements: **4 `ld.weak.global.ca.v4.u32`** at
`W + 16*laneid + {34944, 35456, 35968, 36480}` (s17026–s17032) and **16 packs**
(s17035–s17080, 32 `cvt…e4m3` + 16 `mov.b32`) from **phase 13's D**
(`%r15226…%r15697`) → phase 14's A fragments
(`s17035: mov.b32 %r15827, {%rs6330, %rs6331}`).

Phase 14's C is phase 7's D (see §3/E14 for the trace), so phase 14 is the
second window's out-projection whose residual is the *first window's whole
attention output*.

### E14 — s17113–s18759: RMS norm before the MLP

1647 statements, no stores.  Structure (by op sequence):

| range | stmts | op |
|---|---|---|
| s17113–s17368 | 256 | 32 × unpack of phase 14's D + 128 × `x·x` squares (last at s17367) |
| s17369–s17416 | 48 | in-lane `add.f16x2` |
| s17417–s17539 | 123 | 8 rows × (`add.f16x2` + 2 `shfl.sync.bfly` + `add.f16x2` + `add.f16` + broadcast) |
| s17540–s17543 | 4 | `+ eps` (2⁻¹³) |
| s17544–s18246 | 703 | 128 × `rsqrt.approx.ftz.f32` |
| s18247–s18260 | 14 | gain loads `+37120/37136/…/37232` |
| s18261–s18642 | 382 | 256 `mul.f16`: 128 × `inv · gain`, then 128 × `x · (inv·gain)` |
| s18643–s18654 | 12 | next-B loads `+37248/+37760/+38272/+38784` |
| s18655–s18663 | 9 | next-C loads `+39296/39312/39328/39344` |
| s18664–s18759 | 96 | 32 packs → phase 15's A |

Evidence:

```
s17114 L38471: mul.f16 %rs6365,%rs7992,%rs7992          // square of a D half
s17423 L39316: shfl.sync.bfly.b32 %r16224,%r16220,%r16376,%r16227,%r16387
s17541 L39626: cvt.rn.f16.f64 %rs6770, %fd770           // eps = 2^-13
s17738 L40214: rsqrt.approx.ftz.f32 fl, fl
s18252 L41047: ld.global.v2.u16 {%rs9177, %rs9178}, [%rd277+37136]
s18261 L41057: mul.f16 %rs7155,%rs7156,%rs9191          // inv * gain
s18641 L42199: mul.f16 %rs7920,%rs8370,%rs7536          // x * (inv*gain)
s18664 L42240: cvt.rn.satfinite.e4m3x2.f16x2 %rs7924, %r28441
```

**What it computes.**  For each of the 8 rows a lane holds (8 columns each,
covering the 64×64 phase-14 D):

```
sumsq[row] = Σ_{c=0..31} x[row,c]^2                      (f16, 4-lane butterfly)
inv        = rsqrt.approx.ftz.f32(f32(f16(sumsq + 2^-13)))    // rounded to f16
out[row,c] = x[row,c] * inv * gain[c]
```

* **no mean subtraction and no `1/N`**: the quantity is the *raw* sum of the 32
  squares in a row plus the constant `2^-13`; the mean is not formed.  This is an
  RMS norm, not a LayerNorm.
* `gain[c]` is read from `W + ((laneid<<2)&12) + {37120, 37136, 37152, 37168,
  37184, 37200, 37216, 37232}` (16 f16 per lane = 32 columns), so the gain is
  per column, constant across rows.
* the `inv` register is duplicated into both halves of a b32 before use
  (`mov.b32 %r16392, {hl, hu}`), so one row-sum value serves all n-tiles.
* the norm's *bias* is not added here — it is folded into phase 15's C operand,
  loaded at the end of this epilogue (`+39296…+39344`, four `ld.global.u32`).

**The normed tile** (32 f16x2 = `%r28440…%r28503`) is packed **eight times**,
once in E14 (s18664–s18759 → phase 15's A) and again in E16, E18, E20, E22, E24,
E26, E28 (each s19…/s20… pack of `%r28441`/`%r28440`, e.g. `s19657 L45595:
cvt.rn.satfinite.e4m3x2.f16x2 %rs8404, %r28441`).  So all eight odd phases
15, 17, 19, 21, 23, 25, 27, 29 share one input tile.

### E15 — s18792–s19604: clamped cubic activation, column table, pack

813 statements.  This is the only epilogue that materialises `%f1981…%f1984`
(the four activation constants of §1.2):

```
s18793 L42690: mov.f32 %f1981, 0f3ED306EB
s18796 L42697: mov.f32 %f1982, 0f3DA60DD6
s18799 L42704: mov.f32 %f1983, 0f3F000000          // 0.5
s18802 L42711: mov.f32 %f1984, 0f40000000          // 2.0
s18805 L42719: neg.f16x2 %r16928,%r16927
s18806 L42723: max.f16x2 %r16930,%r16623,%r16928
s18807 L42727: min.f16x2 %r16933,%r16930,%r16927
s18808 L42731: abs.f16x2 %r16936,%r16933
s18809 L42735: mul.f16x2 %r16938,%r16925,%r16936
s18810 L42739: sub.f16x2 %r16941,%r16924,%r16938
s18811 L42743: mul.f16x2 %r16944,%r16933,%r16941
s18812 L42747: add.f16x2 %r16947,%r16926,%r16944
s18813 L42751: mul.f16x2 %r16950,%r16623,%r16947
```

Per element, over each of the 32 f16x2 D registers of phase 15
(`%r16603 %r16613 %r16623 …`, 64 f16 per lane):

```
y   = clamp(x, -2, +2)
g   = 0.5 + y * (0.412109375 - 0.0810546875 * |y|)      // f16 throughout
out = x * g
```

`g` is a *cubic ramp*, not a sigmoid: `g(+2) = 1` ⇒ `out = x` for `x ≥ 2`;
`g(-2) = 0` ⇒ `out = 0` for `x ≤ -2`; `g(1) = 0.831055` where `x·sigmoid(x) =
0.730957`.  **Reproduce the formula, not SiLU/GELU.**

Then:

| range | op |
|---|---|
| s18803–s19340 | 32 activation groups (each 4 constants + 11 f16x2 ops) |
| s19341–s19351 | next-B loads: `+40448 +512*i`, `i=0..3` → phase 16's B |
| s19352–s19364 | lane-shift setup + 8 `ld.global.v2.u16` at `W + ((laneid<<2)&12) + {41408,41424,41440,41456,41472,41488,41504,41520}` |
| s19365–s19555 | 128 `add.f16` (64 f16x2) — table add, written over **phase 14's D** |
| s19559–s19604 | 16 packs of the activation outputs → phase 16's A |

Evidence for the table add:

```
s19365 L44651: add.f16 %rs7990,%rs9208,%rs7992
s19366 L44655: add.f16 %rs7987,%rs9207,%rs7989
s19367 L44658: mov.b32 %r17910, {%rs7987, %rs7990}
s19559 L45232: mov.b32 %r17974, {%rs8371, %rs8372}
```

`%rs7989/%rs7992` are the halves of `%r15755` (phase 14's D, unpacked at
`s17113: mov.b32 {%rs7989, %rs7992}, %r15755`), so
`%r17910 = phase14_D + table[column]`.  The 64 such registers
(`%r17910 %r17920 %r17930 …`) are **phase 16's C operand**
(`s19605 C={%r17910,%r17911}`); `%r17853 %r17857 %r17861 %r17865` are phase 16's
B fragments (weight image, `+40448…`).

**Outputs:** `%f1981…%f1984` (consumed again by E17, E19, E21, E23, E25, E27,
E29), the 64 C seeds (`%r17910…%r18221`), the 16 packed A registers
(`%r17974…%r18217`), and the 4 next-B fragments (`%r17853 %r17857 %r17861
%r17865`, from the s19345–s19351 loads).

### E16, E18, E20, E22, E24, E26, E28 — s19637–s19752 etc.: next B/C loads + pack of the normed tile

116 statements each, identical shape.  Taking E16 as the example:

```
s19638 L45565: mul.wide.u32 %rd288, %r18222, 16
s19641 L45569: ld.weak.global.ca.v4.u32 { %r18223,…, %r18226},[%rd183]   // +41536
s19643 L45573: ld.weak.global.ca.v4.u32 { %r18227,…, %r18230},[%rd184]   // +42048
s19645 L45577: ld.weak.global.ca.v4.u32 { %r18231,…, %r18234},[%rd185]   // +42560
s19647 L45581: ld.weak.global.ca.v4.u32 { %r18235,…, %r18238},[%rd186]   // +43072
s19653 L45590: ld.global.u32 %r18553, [%rd292+43584]
s19654 L45591: ld.global.u32 %r18563, [%rd292+43600]
s19655 L45592: ld.global.u32 %r18593, [%rd292+43616]
s19656 L45593: ld.global.u32 %r18603, [%rd292+43632]
s19657 L45595: cvt.rn.satfinite.e4m3x2.f16x2 %rs8404, %r28441
```

* phase 17's B at `W + 16*laneid + {41536, 42048, 42560, 43072}`;
* **phase 17's C/bias**: 4 `ld.global.u32` at `W + ((laneid<<2)&12) +
  {43584, 43600, 43616, 43632}` → `%r18553 %r18563 %r18593 %r18603`.  Each of
  those four u32 is used in **both halves** of an mma C operand
  (`{%r18553, %r18553}` at s19753), i.e. a per-column bias replicated over the
  16 rows of a tile and reused by all four m-tiles;
* **32 packs** of `%r28440…%r28441` — **the normed tile E14 produced** — giving
  phase 17's A (`%r18356…%r18619`).

The offsets of the other six epilogues (each 4 × B + 4 × C + 32 packs):

| epilogue | phase it feeds | B offsets | C/bias offsets |
|---|---|---|---|
| E16 | 17 | 41536, 42048, 42560, 43072 | 43584, 43600, 43616, 43632 |
| E18 | 19 | 45696, 46208, 46720, 47232 | 47744, 47760, 47776, 47792 |
| E20 | 21 | 49856, 50368, 50880, 51392 | 51904, 51920, 51936, 51952 |
| E22 | 23 | 54016, 54528, 55040, 55552 | 56064, 56080, 56096, 56112 |
| E24 | 25 | 58176, 58688, 59200, 59712 | 60224, 60240, 60256, 60272 |
| E26 | 27 | 62336, 62848, 63360, 63872 | 64384, 64400, 64416, 64432 |
| E28 | 29 | 66496, 67008, 67520, 68032 | 68544, 68560, 68576, 68592 |

(from the phase table's `bias bytes` column; the B offsets are the four `add.s64`
immediates preceding the four `ld.weak.global.ca.v4.u32`.)

### E17, E19, E21, E23, E25, E27, E29 — s19785–s20388 etc.: clamped cubic activation, pack

604 statements each; identical histograms (`mov.b32:144, cvt.rn.f16.f32:128,
mul.f16x2:96, neg/max/min/abs/sub/add f16x2: 32 each, cvt…e4m3:32, 4 weight
loads`, **no `ld.global`, no constants of its own**).

```
s19795 L46074: max.f16x2 %r18631,%r18324,%r18629       // phase 17's D
s19796 L46078: min.f16x2 %r18634,%r18631,%r18628
s19812 L46134: max.f16x2 %r18660,%r18334,%r18658
s20334 L47972: ld.weak.global.ca.v4.u32 { %r19554,…, %r19557},[%rd187]   // +43648
s20343 L47992: mov.b32 %r19674, {%rs8467, %rs8468}
```

Same function as E15: `out = x·(0.5 + y·(0.412109375 − 0.0810546875|y|))`,
`y = clamp(x,±2)`, over the 32 D registers of the phase, then 32 packs **of the
activation outputs** (`%r19674…%r19917`) → the next phase's A, plus 4 weight
loads → the next phase's B.  **No column-bias add and no column table.**

The four activation constants are re-read by `cvt.rn.f16.f32` from
`%f1981…%f1984` (e.g. E17 has `128 cvt.rn.f16.f32` and zero `mov.f32`).

#### Shape of the whole MLP region (E14–E30)

Putting the pack destinations and the C operands together (all verified by
tracing each `mov.b32 %rX, {%rsA, %rsB}` back through its two `cvt` statements,
and each C register back to its `mma` in the *previous even* phase):

```
E14  normed tile N (32 f16x2, %r28440…%r28503)
 ├─ pack → P15 A (s18664)  P15: A=N, B=W15, C=bias@39296   → D15 → E15 activation
 │                              E15 packs act(D15) → P16 A (s19559)
 │                              P16: C = D14 + table@41408 → D16        (s19605)
 ├─ pack → P17 A (s19657)  P17: A=N, B=W17, C=bias@43584   → D17 → E17 activation
 │                              E17 packs act(D17) → P18 A (s20343)
 │                              P18: C = D16               → D18        (s20389)
 ├─ pack → P19 A (s20443)  P19: A=N, B=W19, C=bias@47744   → D19 → E19 activation
 │                              E19 packs act(D19) → P20 A (s21127)
 │                              P20: C = D18               → D20        (s21173)
 ├─ pack → P21 A (s21227)  P21: A=N, B=W21, C=bias@51904   → D21 → E21 activation
 │                              E21 packs act(D21) → P22 A (s21911)
 │                              P22: C = D20               → D22        (s21957)
 ├─ pack → P23 A (s22011)  P23: A=N, B=W23, C=bias@56064   → D23 → E23 activation
 │                              E23 packs act(D23) → P24 A (s22695)
 │                              P24: C = D22               → D24        (s22741)
 ├─ pack → P25 A (s22795)  P25: A=N, B=W25, C=bias@60224   → D25 → E25 activation
 │                              E25 packs act(D25) → P26 A (s23479)
 │                              P26: C = D24               → D26        (s23525)
 ├─ pack → P27 A (s23579)  P27: A=N, B=W27, C=bias@64384   → D27 → E27 activation
 │                              E27 packs act(D27) → P28 A (s24263)
 │                              P28: C = D26               → D28        (s24309)
 └─ pack → P29 A (s24363)  P29: A=N, B=W29, C=bias@68544   → D29 → E29 activation
                                E29 packs act(D29) → P30 A (s25045)
                                P30: C = D28               → D30        (s25093)
```

So the region is **eight bottleneck residual blocks** `64 → 32 → 64`: every odd
phase expands the shared normed tile into 32 channels with its own weight and
per-column bias, its epilogue applies the clamped cubic activation, and the
following even phase projects back to 64 channels and adds the running residual
(`D16 = D14 + table + act(D15)·W16`, then `D18 = D16 + act(D17)·W18`, …,
`D30 = D28 + act(D29)·W30`).  The only difference from enc0 is the count
(eight blocks instead of four) and the first residual's `+ table` term.

### E30 — s25125–s25823: pack → destination-buffer scatter → 8-lane merge → next B/C

699 statements.  Four sub-sections:

#### (a) requantise, s25125–s25220

64 `cvt.rn.satfinite.e4m3x2.f16x2` + 32 `mov.b32` → `%r317 … %r348`
(phase 30's 64×64 output as 32 b32 = 64 e4m3 per lane).

#### (b) destination-buffer scatter, s25221–s25630

```
s25221 L65106: ld.param.u64 %rd337, [%rd1+-32]        // %rd1 = param_0+80 → param+48
s25222 L65107: cvta.to.global.u64 %rd5, %rd337
s25223 L65109: mov.u32 %r30186, %laneid
s25224 L65111: shr.u32 %r349, %r30186, 2              // g = laneid>>2
s25225 L65112: and.b32 %r350, %r30186, 3              // t = laneid&3
s25226 L65113: shr.u32 %r30540, %r30186, 5            // laneid>>5
s25227 L65114: and.b32 %r30541, %r349, 7              // g&7
s25228 L65115: add.s32 %r30542, %r7, %r30540          // y = %r7 + (laneid>>5)
s25234 L65121: add.s32 %r30543, %r1, %r30541          // x = %r1 + (g&7)
s25239 L65126: shl.b32 %r30544, %r30543, 3
s25240 L65127: mad.lo.s32 %r351, %r5, %r30542, %r30544
s25241 L65128: add.s32 %r30545, %r350, %r351
s25247 L65135: st.global.u32 [%rd339], %r317
```

Thirty-two stores `st.global.u32 [%rd5 + 4*idx], v` (s25247…s25630, one per
`selp.b32`+`bra` guard), writing `%r317…%r348` **in that order**.  The index is

```
idx = %r5 * y + 8*x + d,     %r5 = ex<<3
d   = t (even-numbered store) or t|4 (odd-numbered store)
```

and the y/x bases are the eight values `(g + 8j)`, `j = 0…7`:

```
y = %r7 + ((g + 8j) >> 3)          // j=0 uses the raw laneid>>5 (s25226)
x = %r1 + ((g + 8j) & 7)           // j=0 uses the raw g&7 (s25227)
```

`g = laneid>>2`, `t = laneid&3`.  The bases are recomputed at s25260 (`+8`),
s25328 (`+16`), s25362 (`+24`), s25430 (`+32`), s25464 (`+40`), s25532 (`+48`),
s25566 (`+56`); group 0's base is the prologue's raw `%r30540`/`%r30541`.  Each
store is guarded by the compiler's standard triplet,
`selp.b32 %r, -1, idx, %p` (s25242 and 31 siblings) + `setp.lt.s32 %p, %r, 0` +
`@%p bra`, where `%p` is the disjunction
`(y<0) | (y ≥ ey) | (laneid > 255 or (g+8j) > 63) | (x<0) | (x ≥ ex)`
(`s25229–s25238` for group 0), so a guard failure becomes "skip the store".

The stride is `%r5 = ex<<3`, so the buffer is `8*ex` 4-byte words = 32*ex bytes
wide per row.  Which of the eight bases each individual store uses is not
determined by the PTX alone — see §6 U6.

#### (c) 8-lane merge, s25632–s25707

```
s25638 L65591: and.b32 %r31799, %r30619, 4
s25640 L65593: setp.eq.s32 %p422, %r31799, 0
s25644 L65597: shfl.sync.down.b32 %r31804|%p423, %r317, %r31800, %r31802, %r31803
s25645 L65598: shfl.sync.up.b32 %r31805|%p424, %r325, %r31800, %r31801, %r31803
s25646 L65599: selp.b32 %r31626, %r317, %r31805, %p422
s25647 L65600: selp.b32 %r31666, %r31804, %r325, %p422
```

**Sixteen** `shfl.sync.down/up` pairs, all with **delta = 4** (`%r31800 =
mov.u32 4`, s25639), width operand `31` for the `.down` (`%r31802`, s25642) and
`0` for the `.up` (`%r31801`, s25641), membermask `-1` (`%r31803`, s25643), and
the selection predicate `%p422 = (laneid & 4) == 0`:

```
down = a[lane+4]  (shfl.sync.down, delta 4)
up   = a[lane-4]  (shfl.sync.up,   delta 4)
dst1 = p422 ? own_low  : up
dst2 = p422 ? down     : own_high
```

so each pair re-pairs the two members of `{lane, lane±4}`.  The sixteen
destinations are **phase 31's A operand** — `s25824` first mma uses
`{%r31626, %r31627, %r31628, %r31629}`.

#### (d) next B/C loads, s25708–s25823

```
s25712 L65668: ld.weak.global.ca.v4.u32 { %r30621,…, %r30624},[%rd402]   // +70656
s25714 L65672: ld.weak.global.ca.v4.u32 { %r30625,…, %r30628},[%rd403]   // +71168
   … one every 512 bytes, 48 of them, up to +94720 (s25805)
s25812 L65865: ld.global.u32 %r30823, [%rd454+95232]
   … twelve in all, +95232 … +95408 (s25823)
```

* **48 weight fragments** at `W + 16*laneid + 70656 + 512*i` → phase 31's B;
* **12 bias words** at `W + ((laneid<<2)&12) + {95232…95408}` → phase 31's C
  seeds, each used in both halves (`s25824 C={%r30823,%r30823}`).

**Inputs:** phase 30's D, `%r1/%r5/%r7/%r465/%r466`.  **Outputs (live):**
`%r317…%r348` (read again by the merge *and* by phase 32), the 16 merged A
registers, phase 31's B and C.

### E31 — s25920–s26123: pack → 2nd-buffer scatter → `%tid.x` → next B/C

204 statements.

#### (a) requantise, s25921–s25954

24 `cvt.rn.satfinite.e4m3x2.f16x2` (each on a phase-31 D register
`%r30954…%r31765`) grouped 2-by-2 into **12 e4m3 pairs**: 10 of them are
immediately combined with `mov.b32` into `%r400 … %r409`, and the first two pairs
are kept as the raw `{%rs9209,%rs9210}` and `{%rs9231,%rs9232}` for the
`v2.u16` stores.

```
s25921 L66551: cvt.rn.satfinite.e4m3x2.f16x2 %rs9209, %r30954
s25922 L66554: cvt.rn.satfinite.e4m3x2.f16x2 %rs9210, %r31114
s25925 L66562: mov.b32 %r400, {%rs9211, %rs9212}
s25953 L66627: cvt.rn.satfinite.e4m3x2.f16x2 %rs9231, %r31605
s25954 L66630: cvt.rn.satfinite.e4m3x2.f16x2 %rs9232, %r31765
```

#### (b) coordinate / 2nd-buffer scatter, s25955–s26107

```
s25955 L66632: ld.param.v2.u32 {%r31837, %r31838}, [%rd535+-80]   // param+0, extents
s25956 L66633: shr.u32 %r412, %r31838, 1                          // ey>>1
s25957 L66634: shr.u32 %r413, %r31837, 1                          // ex>>1
s25958 L66635: shl.b32 %r414, %r413, 3                            // 8*(ex>>1)
s25959 L66636: mul.lo.s32 %r415, %r412, %r414                     // row stride
s25963 L66642: ld.param.u64 %rd455, [%rd535+-24]                  // param+56
s25964 L66643: cvta.to.global.u64 %rd7, %rd455
s25965 L66644: and.b32 %r31839, %r416, 1                          // g&1
s25969 L66648: bfi.b32 %r31843, %r31842, %r31839, 1, 31           // ((laneid>>5)&~2)|((g&1)<<1)
s25973 L66652: add.s32 %r31846, %r418, %r31843                    // y
s25982 L66661: add.s32 %r31849, %r419, %r31841                    // x
s25988 L66667: mad.lo.s32 %r420, %r414, %r31846, %r31850
s25995 L66675: st.global.v2.u16 [%rd457], {%rs9209, %rs9210}
```

The destination is **`param_0+56`**, not the plane arena; extents come from
`param+0` and are *halved*.  Two index bases are formed, `%r421` (s25990, the
`(g&1)`-bfi'd `y` with `x = (%r1>>1) + ((laneid>>3)&3)`) and `%r424` (s26026,
the same shape recomputed from a shifted index), each guarded by a
`(y<0)|(y≥(ey>>1))|(tile bits>3)|(x<0)|(x≥(ex>>1))` predicate (`%p9`, `%p10`).

**Twelve guarded stores**, all at `base + (t or t|4)` in 4-byte units:

| # | statement | value | base | d |
|---|---|---|---|---|
| 1 | s25995 `st.global.v2.u16` | `{%rs9209, %rs9210}` | `%r421` | `t` |
| 2 | s26004 `u32` | `%r400` | `%r421` | `t\|4` |
| 3 | s26031 `u32` | `%r401` | `%r424` | `t` |
| 4 | s26039 `u32` | `%r402` | `%r424` | `t\|4` |
| 5 | s26048 `u32` | `%r403` | `%r421` | `t` |
| 6 | s26056 `u32` | `%r404` | `%r421` | `t\|4` |
| 7 | s26065 `u32` | `%r405` | `%r424` | `t` |
| 8 | s26073 `u32` | `%r406` | `%r424` | `t\|4` |
| 9 | s26082 `u32` | `%r407` | `%r421` | `t` |
| 10 | s26090 `u32` | `%r408` | `%r421` | `t\|4` |
| 11 | s26099 `u32` | `%r409` | `%r424` | `t` |
| 12 | s26107 `st.global.v2.u16` | `{%rs9231, %rs9232}` | `%r424` | `t\|4` |

(the guard predicates alternate `%p9`/`%p10` exactly as listed by the `selp.b32`
at s25999, s26034, s26043, s26051, s26060, s26068, s26077, s26085, s26094,
s26102.)

#### (c) next B/C and `%tid.x`, s26109–s26123

```
s26114 L66820: ld.weak.global.ca.v4.u32 { %r31879,…, %r31882},[%rd480]   // +95424
s26116 L66824: ld.weak.global.ca.v4.u32 { %r31883,…, %r31886},[%rd481]   // +95936
s26122 L66833: ld.global.u32 %r32017, [%rd485+96448]
s26123 L66834: ld.global.u32 %r32027, [%rd485+96464]
```

phase 32's B fragments (`W + 16*laneid + 95424 + 512*i`) and two C seeds
(`W + ((laneid<<2)&12) + {96448, 96464}`).

### E32 — s26140–s26819: the final epilogue

679 statements in **two identical passes** (the second at s26554 onward).  Each
pass selects the four f16 values of a row from phase 32's D registers, does a
cross-lane row maximum, an f32 softmax, nine `tex.base.2d.v4.f16.s32` gathers,
an f16 fma accumulation, a sigmoid difference, a two-step cross-lane sum, and
three `sust.b.2d.v4.b16.zero` surface writes.

#### (a) cross-lane row maximum / row selection, s26141–s26290

```
s26141 L66950: and.b32 %r32051, %r32048, 7                // laneid&7
s26142 L66951: shr.u32 %r32052, %r32048, 3                // laneid>>3
s26143 L66952: and.b32 %r32053, %r32048, 3
s26145 L66954: selp.b32 %r32054, %r31909, %r31908, %p483  // pick by (laneid&3)
s26161 L66970: bfi.b32 %r32067, %r32051, %r32066, 2, 3
s26164 L66973: shfl.sync.idx.b32 %r32070|%p487, %r32056, %r32067, %r32068, %r32069
s26182 L66991: selp.b32 %r438, %r32082, %r32084, %p493
```

Two blocks (s26141–s26216 for the first two 4-value rows, s26217–s26290 for the
other two), each doing:

1. a `selp` chain over `(laneid&3)` selecting one of four **phase-32 D**
   registers (`%r31908/%r31909`, `%r31948/%r31949`, `%r32028/%r32029`,
   `%r32038/%r32039`) into four registers;
2. four `shfl.sync.idx.b32` with `srcLane = (laneid&7) | ((((laneid>>3)+d)&3)<<2)`
   for `d = 0, 1, 2, -1` (`%r32068 = mov.u32 31`, `%r32069 = mov.u32 -1`);
3. a `selp` chain over `laneid>>3` picking between the four shuffled values.

Result: `%r438 %r439 %r440 %r441` (and `%r443…%r446`) hold four values per lane;
`%r442`, `%r447`, `%r448` are the corresponding single values of the other two
register sets.

#### (b) bounds check, s26291–s26306

```
s26291 L67102: mov.u32 %r449, %tid.x
s26296 L67107: sub.s32 %r32174, %r449, %r32173      // tid.x - ((tid.x+round)>>3<<3)
s26298 L67109: add.s32 %r450, %r32174, %r1          // x
s26299 L67110: add.s32 %r451, %r32175, %r7          // y
s26300 L67111: setp.ge.u32 %p515, %r450, %r31837    // x >= ex
s26301 L67112: setp.ge.u32 %p516, %r451, %r31838    // y >= ey
s26306 L67117: @%p519 bra $L__BB1_438
```

Out-of-range lanes jump past (b)…(f).  The second pass recomputes with
`tid.x + 32` (s26554: `add.s32 %r32350, %r449, 32`).

#### (c) softmax, s26307–s26394

```
s26311 L67124: cvt.f32.f16 %f1985, %rs96
s26312 L67127: mul.ftz.f32 %f2013, %f1985, 0fBFB8AA3B   // -log2(e)
s26313 L67128: ex2.approx.ftz.f32 %f2014, %f2013
s26314 L67129: add.ftz.f32 %f2015, %f2014, 0f3F800000
s26316 L67131: div.approx.ftz.f32 %f1986, 0f3F800000, %f2015   // sigmoid(%rs96)
s26318 L67137: max.f16x2 %r32176,%r438,%r439
s26320 L67145: max.f16x2 %r32182,%r32179,%r441
s26323 L67151: max.f16 %rs9235,%rs9236,%rs9237
s26324 L67155: max.f16 %rs9238,%rs9235,%rs95            // the real row max
s26326 L67160: sub.f16x2 %r32185,%r438,%r32199          // x - rowmax
s26329 L67168: mul.ftz.f32 %f2017, %f1987, 0f3FB8AA3B   // *log2(e)
s26330 L67169: ex2.approx.ftz.f32 %f1989, %f2017
s26386 L67288: div.approx.ftz.f32 %f2005, 0f3F800000, %f2034   // 1/rowsum
s26389 L67295: mul.f16x2 %r32200,%r32201,%r32214
```

A textbook f32 softmax over the row's four values plus the fifth: `max` via
`max.f16x2`/`max.f16`, `y = 2^((x-max)*log2e)` in f32
(`cvt.f32.f16 → mul by 0f3FB8AA3B → ex2.approx`), accumulate the row sum in f32,
invert with `div.approx.ftz.f32 1.0/…`, round to f16, multiply the five f16x2
registers `%r32200 %r32203 %r32206 %r32209 %r32212`.  Separately,
`1/(1+2^(-log2e·c))` is built at s26311–s26317 from `%r442` — a sigmoid of the
fifth value, kept in `%rs9234` for the second pass.  **This epilogue does have a
real row maximum** (unlike E5), done with four `shfl.sync.idx` per row group.

#### (d) texture gather and accumulate, s26395–s26471

```
s26397 L67317: selp.b32 %r32324, %r32322, %r32323, %p520   // clamp x
s26412 L67333: tex.base.2d.v4.f16.s32 {%rs9260,…, %rs9263}, [%rd486, {%r32263,%r32232}]
s26414 L67337: tex.base.2d.v4.f16.s32 {%rs9264,…, %rs9267}, [%rd488, {%r32265,%r32232}]
s26444 L67394: mov.b32 %r32252, {%rs9276, %rs9280}
s26445 L67396: fma.rn.f16x2 %r32251,%r32252,%r32206,%r32235
s26468 L67441: add.f16 %rs9292,%rs9293,%rs9294
s26473 L67457: fma.rn.f16 %rs9305,%rs9301,%rs9307,%rs9292
```

**Nine** `tex.base.2d.v4.f16.s32` (s26412, s26414, s26431, s26433, s26441,
s26443, s26456, s26458, s26471) at a 2×2-plus-one stencil of the texture at
`param_0+88` (`%rd486 %rd488 %rd490 %rd492 %rd494 %rd496 %rd498 %rd500 %rd502`,
handles built from `[%rd539+8]`), indexed by clamped `(x,y)` neighbourhoods.
The 2×2 blocks are multiplied by the five softmax registers with `fma.rn.f16x2`
and accumulated; the accumulator halves are summed with `add.f16`, then one more
texture value is folded in with `fma.rn.f16` (`* %rs9307`, the 5th weight).

#### (e) sigmoid, difference, scale, cross-lane sum, s26472–s26552

```
s26476 L67469: cvt.f32.f16 %f2006, %rs97
s26477 L67472: mul.ftz.f32 %f2035, %f2006, 0fBFB8AA3B
s26478 L67473: ex2.approx.ftz.f32 %f2036, %f2035
s26479 L67474: add.ftz.f32 %f2037, %f2036, 0f3F800000
s26480 L67475: div.approx.ftz.f32 %f2007, 0f3F800000, %f2037   // sigmoid
s26488 L67493: mul.f16 %rs9321,%rs9234,%rs9305
s26492 L67508: mov.f32 %f2012, 0f3E800000                        // 0.25
s26493 L67510: sub.f16 %rs9333,%rs9309,%rs9324
s26507 L67548: shfl.sync.down.b32 %r32285,%r32281,%r32306,%r32288,%r32069
s26508 L67552: add.f16x2 %r32290,%r32281,%r32285
s26513 L67563: shfl.sync.down.b32 %r32294,%r32290,%r32314,%r32297,%r32069
s26514 L67567: add.f16x2 %r32299,%r32290,%r32294
s26529 L67606: shfl.sync.down.b32 %r32312,%r32310,%r32314,%r32315,%r32069
```

`sigmoid(v) = 1/(1 + 2^(-log2e·v))`, then three `acc − sigmoid(acc)` differences
scaled by `0.25` (`%f2012`), then a two-step cross-lane sum: `shfl.sync.down`
with the delta operand `%r32306 = 1` (s26504) then `%r32314 = 8` (s26504), each
followed by an `add.f16x2`; the same two steps repeat on the scalar path
(`add.f16` at s26523 and s26532).  The width operand is 31, so this is a partial
(not five-step) warp reduction.

#### (f) surface stores, s26533–s26552

```
s26533 L67618: ld.param.u64 %rd506, [%rd539+40]      // param_0+120
s26534 L67619: ld.param.u64 %rd508, [%rd539+48]      // param_0+128
s26536 L67622: sust.b.2d.v4.b16.zero [%rd504, {%r32320,%r451}], {%rs9330,%rs9333,%rs9336,%rs9365}
s26537 L67625: sust.b.2d.v4.b16.zero [%rd506, {%r32320,%r451}], {%rs9321,%rs9324,%rs9327,%rs9366}
s26538 L67627: and.b32 %r32342, %r452, 1
s26552 L67643: sust.b.2d.v4.b16.zero [%rd508, {%r32343,%r32344}], {%rs9367,%rs9368,%rs9354,%rs9370}
```

Three 4-channel b16 surface writes (`param_0+112`, `+120`, `+128`; `%rd504 =
[%rd539+32]` at s26310); the third is taken only when `(%r452 & 1) != 1` (s26538–
s26542) and stores `0` in its fourth channel (`%rs9370 = mov.u16 0`).

**Inputs:** phase 32's D, the surface handles and the texture handle from
`param_0+88/+112/+120/+128`, `param_0+0` extents, constants `0fBFB8AA3B`,
`0f3FB8AA3B`, `0f3F800000`, `0f3E800000`.
**Outputs:** none (the kernel returns at s26819).

---

## 4. The destination-buffer writes (`param_0+48` and `param_0+56`)

There are exactly **three** store groups in the whole entry:

**(i) E30, 32 × `st.global.u32 [%rd5 + 4*idx], v` — s25247…s25630**, into
`%rd5 = cvta(param_0+48)`.  In one line per store, with
`%r5 = ex<<3`, `g = laneid>>2`, `t = laneid&3`:

```
idx = %r5*y + 8*x + d,   d = t (even store) or t|4 (odd store)
guard: 0 <= y < ey  and  0 <= x < ex  and  (laneid <= 255 or (g+8j) <= 63)
bases: y = %r7 + ((g+8j)>>3),  x = %r1 + ((g+8j)&7),  j = 0…7
       recomputed at s25260 (+8), s25328 (+16), s25362 (+24), s25430 (+32),
       s25464 (+40), s25532 (+48), s25566 (+56); j=0 uses the raw laneid>>5 / g&7
```

| # | statement | value | # | statement | value |
|---|---|---|---|---|---|
| 1 | s25247 | `%r317` | 17 | s25452 | `%r333` |
| 2 | s25256 | `%r318` | 18 | s25460 | `%r334` |
| 3 | s25282 | `%r319` | 19 | s25486 | `%r335` |
| 4 | s25290 | `%r320` | 20 | s25494 | `%r336` |
| 5 | s25299 | `%r321` | 21 | s25503 | `%r337` |
| 6 | s25307 | `%r322` | 22 | s25511 | `%r338` |
| 7 | s25316 | `%r323` | 23 | s25520 | `%r339` |
| 8 | s25324 | `%r324` | 24 | s25528 | `%r340` |
| 9 | s25350 | `%r325` | 25 | s25554 | `%r341` |
| 10 | s25358 | `%r326` | 26 | s25562 | `%r342` |
| 11 | s25384 | `%r327` | 27 | s25588 | `%r343` |
| 12 | s25392 | `%r328` | 28 | s25596 | `%r344` |
| 13 | s25401 | `%r329` | 29 | s25605 | `%r345` |
| 14 | s25409 | `%r330` | 30 | s25613 | `%r346` |
| 15 | s25418 | `%r331` | 31 | s25622 | `%r347` |
| 16 | s25426 | `%r332` | 32 | s25630 | `%r348` |

Which of the eight bases each individual store uses is not determined by the PTX
alone — see §6 U6.

**(ii) E31, 12 stores to `param_0+56`** — s25995 (`v2.u16` of `%rs9209/%rs9210`),
s26004, s26031, s26039, s26048, s26056, s26065, s26073, s26082, s26090, s26099
(`u32` of `%r400…%r409`), s26107 (`v2.u16` of `%rs9231/%rs9232`).  See §3/E31 for
the indices.

**(iii) E32, 6 × `sust.b.2d.v4.b16.zero`** to `param_0+112/+120/+128`
(s26536, s26537, s26552 and their second-pass copies s26801, s26802, s26817).

**What is written to `param_0+48`:** phase 30's requantised 64×64 output — 32 b32
per lane, one per store — i.e. 4 bytes × 32 lanes × 32 stores = 4096 bytes per
block.  **Nothing inside this kernel reads `param_0+48` back**: the only read of
that offset are the `ld.param.u64 [%rd1+-32]` at s25221 that *creates* the base
pointer.

## 5. Cross-lane reductions of the merge

The merge's "shfl reductions" are the **sixteen `shfl.sync.down/up` pairs of
E30(c), s25644–s25705**.  Each pair uses delta operand 4 (PTX `b`), group/width
operand 31 (`.down`) or 0 (`.up`), and `laneid & 4` as the selection predicate,
so each of the 32 payloads is re-paired between lanes 0-3 and 4-7 of every
`laneid&7` group before it becomes phase 31's A operand.

There is a *second*, unrelated cross-lane reduction family:
* E5/E12's 16 `shfl.sync.bfly` (lane bits 0 and 1) for the softmax row sums;
* E14's 16 `shfl.sync.bfly` for the RMS sums;
* E32's four `shfl.sync.idx` per row group (row maximum) and its
  `shfl.sync.down` pairs at delta 1 and 8 after the sigmoid.

## 6. Uncharacterised

**U1. The identity of the 12288-byte score bias.**  Both score phases add a
12288-byte f16 table as their GEMM C operand (P5: `W+4224…W+16512`, P12:
`W+22656…W+34848`), indexed as §3/E4 describes.  Whether it is a relative
position bias, a learned score bias, or a per-lane constant cannot be decided
from the PTX: it is an opaque blob in a permuted weight image.  *Missing:* the
weight image's layout description (the `_prep` permutation) or the parameter
names from the model file.

**U2. Why phase 5's two operands are the same projection.**  E4 packs phase 3's
D into *both* phase 5's A and its B (fragment-permuted), and phase 4's D becomes
the transposed V of phase 6.  So `scores = pack(D3)·pack(D3)ᵀ`.  The PTX proves
the arithmetic; it does not say which of D3/D4 is "Q", "K" or "V" in the model.
*Missing:* the high-level source or the model's projection names.

**U3. The identity of the per-column tables and biases.**  `W+36992…37104` (E6),
`W+37120…37232` (E14, used as the norm gain), `W+41408…41520` (E15), and the
per-column biases at `+39296/+43584/+47744/+51904/+56064/+60224/+64384/+68544`
are all read with the same `W + ((laneid<<2)&12) + off` pattern and are 16 f16
per lane (32 columns).  Their role is *computable* (E6: added to the input
residual before phase 7; E14: multiplies the inverse std; E15: added after the
input tile) but their *names* are not.  The biases at `+95232…95408` and
`+96448/+96464` are the same case.

**U4. The exact purpose of phase 32's texture reads.**  Nine
`tex.base.2d.v4.f16.s32` per pass fetch a 2×2-plus-one stencil through handles
built from `param_0+88`; the arithmetic is §3/E32(d)–(e), but which input surface
that handle is and what model term `(acc − sigmoid(acc))·0.25` corresponds to is
not decidable from the PTX.  Note that enc0's equivalent epilogue reads a
*shared* array (`input_noise`) instead, so enc1 and enc0 differ here.  *Missing:*
the high-level source (a `shrink`/`gate`/`residual` term) or a reference
implementation.

**U5. The geometry the spec reports for the band phases.**  `rr_layer_spec.py`
prints `M=192 N=96 K=96` for phases 6 and 13 and `M=128 N=64 K=64` for phases
15…29; §2.1 shows the true shapes are `64×32×96` and `64×32×64`.  The tool's
`band` grid marker comes from the address pattern, not from the chain; I could
not find a rule that recovers `k_steps` from the tool's own data, so the table in
§2.1 was derived by walking each phase's `D → C` chain by hand.  The epilogue
contracts above do not depend on the tool's numbers.

**U6. E30's base-to-store assignment.**  The 32 stores write `%r317…%r348` in
order and there are eight `(g+8j)` bases (recomputed at s25260, s25328, s25362,
s25430, s25464, s25532, s25566 plus the prologue's raw `g`), but the PTX does
not pair them one-to-one: between two consecutive recomputations there are
sometimes two and sometimes six stores, and the guard predicates used inside one
such run alternate between the run's own predicate and an earlier one (e.g.
`s25282/s25290` use `%p2` while `s25299/s25307` use `%p1`).  I could not
determine whether the six-store runs cover a second base or merely re-store the
same addresses.  *Missing:* a run of the kernel with a `param_0+48` buffer dump,
or the corresponding high-level code.

**U7. Why `%r442`/`%r447` differ from `%r438…%r441`.**  In E32 the first
`selp`+`shfl` block produces `%r438…%r441` from `%r31908/%r31909/%r31948/%r31949`
and `%r442…` from `%r31918/%r31919/%r31958/%r31959`, i.e. two disjoint sets of
phase-32 D registers.  The softmax uses the first set and a sigmoid a member of
the second; the PTX does not say why.  *Missing:* the high-level source.

## 7. Quick acceptance index

| requirement | where |
|---|---|
| phase table with M/N/K checked | §2.1 (twelve rows differ from the tool's numbers, all explained) |
| weight image extent | §2.2 — ≥ 96480 bytes (tool says 96468; the per-lane table base adds 12) |
| phases with no weight operand | §2.3 — 6 and 13 read none; 5 and 12 read only a C bias |
| E5 row max / exp / row sum / reciprocal | §3/E5(a)–(e); **no row max**; the exponent is a cubic-with-6-bit-mantissa, not `exp2` |
| P5 score bias | §3/E4 — 24 × `ld.weak.global.ca.v4.u32` at `W+4224+512*i`, used as the GEMM C operand, *not* in the epilogue |
| P6 `P·V`, the transposes | §3/E5(f) (48 `movmatrix`, s12185–s12232) + E6 |
| second window P8–P14 | §3/E8–E13 |
| the RMS norm | §3/E14 — sum of squares + 2⁻¹³, `rsqrt`, ·gain, **no mean, no 1/N, no bias** |
| the activation | §3/E15/E17/… — clamped cubic `x·(0.5 + y·(0.412109375 − 0.0810546875|y|))`, `y = clamp(x,±2)` |
| the eight bottleneck residual blocks | §3/E15…E30 ("Shape of the whole MLP region") |
| patch merge / destination-buffer writes | §4(i) — 32 stores at `param_0+48`; §4(ii) — 12 at `param_0+56` |
| final epilogue | §3/E32 — row max, f32 softmax, 9 texture reads, surface stores |
| plane-arena table generated per launch | §4 — the buffer at `param_0+48` is write-only inside this kernel (see §3/E30(b)) |
