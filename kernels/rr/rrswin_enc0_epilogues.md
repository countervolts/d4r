# `cuda_dldn_engine_swin_enc0_kernel` — the epilogues of phases 5…22

Companion to `kernels/rr/rr_layer_spec.py enc0`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0013-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_enc0_kernel`.

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at their first line).  **`LNNNNN`
  is the physical line** of the same instruction in the corpus file, so every
  claim can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and the
  next phase's first `mma`.  The phase table (`rr_layer_spec.py enc0`) gives the
  ranges; this file names what each one *computes*.
* `W` denotes the prepared weight image: `ld.param.u64 %rd367,
  [cuda_dldn_engine_swin_enc0_kernel_param_0+40]` (s2221) and
  `cvta.to.global.u64 %rd2, %rd1` (s12, `%rd1` = the same `+40` load at s11).
  Every `ld.weak.global.ca.v4.u32` in an epilogue reads `W + 16*laneid + imm`.
  `S` denotes the plane arena: `ld.param.u64 %rd276, [%rd368+48]` (s16052).
* The C/D register layout and the weight-addressing formula are fixed (see the
  settled notes): `C/D reg0 = (row g, cols 2t,2t+1)`, `reg1 = (row g+8, same cols)`,
  `g = laneid>>2`, `t = laneid&3`.
* Anything I could not pin down from the PTX is in §6 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `W` (weight image, 54560 bytes) | `ld.param.u64 %rd367, [param_0+40]` / `cvta.to.global.u64 %rd2, %rd1` | s2221 / s11 |
| `S` (plane arena) | `ld.param.u64 %rd276, [%rd368+48]` then `cvta.to.global.u64 %rd37, %rd276` | s16052–s16053 |
| `%rd368` | `mov.b64 %rd368, cuda_dldn_engine_swin_enc0_kernel_param_0` | s2220 |
| extents `(ex, ey)` | `ld.param.v2.u32 {%r364, %r365}, [param_0+0]` | s13 |
| `%r1`, `%r6` (block origins) | `sub.s32 %r1, %r359, %r360` and `sub.s32 %r6, %r367, %r366` | s6, s16 |
| per-lane weight address | `mul.wide.u32 %rd, %laneid, 16` + `add.s64 %rd, W, %rd` | every load block |
| per-lane *table* address | `shl.b32 %r, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rd, %r, 12` → `add.s64 %rd, %rd2, %rd` = `W + ((laneid*4) & 12)` | s7881–s7885 and copies |

The last line matters: the four `ld.global.v2.u16` / `ld.global.u32` loads in an
epilogue read `W + ((laneid<<2)&12) + off`, i.e. **the same 64-byte record is
read by the four lanes with equal `laneid&3`, and the four loads pick four
16-byte words inside it.**  So those tables are indexed by column, not by row.

### 1.2 Constants that survive between epilogues

| register | first definition | f32/f64 literal | value used (after the `cvt`) |
|---|---|---|---|
| `%f2085` | s3984 `mov.f32 %f2085, 0f3C8CCB50` | 0.017186790704727173 | f16 `0x2466` = 0.017181396484375 |
| `%fd767` | s3988 `mov.f64 %fd767, 0dBFE1CC0000000000` | −0.55615234375 | f16 `0xB873` = −0.55615234375 |
| `%fd769` | s3993 `mov.f64 %fd769, 0d3FE1CC0000000000` | +0.55615234375 | f16 `0x3873` = +0.55615234375 |
| `%f2087` | s3999 `mov.f32 %f2087, 0f3F6D6000` | 0.92724609375 | f16 `0x3B6B` = 0.92724609375 |
| `%f2089` | s4004 `mov.f32 %f2089, 0f3FB00000` | 1.375 | f16 `0x3D80` = 1.375 |
| `%fd770` | s2802 `mov.f64 %fd770, 0d3F20000000000000` | 2⁻¹³ | f16 `0x0800` = 0.0001220703125 |
| `%f2598` | s13192 `mov.f32 %f2598, 0f3ED306EB` | 0.4121621549129486 | f16 `0x3698` = 0.412109375 |
| `%f2599` | s13195 `mov.f32 %f2599, 0f3DA60DD6` | 0.0810810774564743 | f16 `0x2D30` = 0.0810546875 |
| `%f2600` | s13198 `mov.f32 %f2600, 0f3F000000` | 0.5 | f16 `0x3800` = 0.5 |
| `%f2601` | s13201 `mov.f32 %f2601, 0f40000000` | 2.0 | f16 `0x4000` = 2.0 |

Also live across the region (defined in the prologue, read by every warp
reduction): `%r11301` = `mov.u32 %r11301, 1` (s2286), `%r11310` =
`mov.u32 %r11310, 2` (s2225), `%r11312` = `mov.u32 %r11312, -1` (s2621) —
the three operands of every `shfl.sync.bfly.b32`.

`%f2598…%f2601` are defined **inside phase 13's epilogue** and reused by the
epilogues of phases 15, 17 and 19 (e.g. s14001 and s14693 read them).

### 1.3 Two quantisation helpers

* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).
* `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `div.approx.ftz.f32` — always wrapped as
  `cvt.f32.f16 → op → cvt.rn.f16.f32` (round each step through f16).

## 2. Summary

| epilogue | range | stmts | named operation |
|---|---|---|---|
| E5 | s3983–s7825 | 3843 | softmax of the 64×96 score block, V-transpose, pack |
| E6 | s7874–s8033 | 160 | residual + per-column table add, pack, next B |
| E7 | s8050–s8056 | 7 | next-B weight loads |
| E8 | s8081–s8087 | 7 | next-B weight loads |
| E9 | s8112–s8282 | 171 | score-bias C-seed loads, pack |
| E10 | s8331–s12168 | 3838 | softmax (2nd window, same code as E5) |
| E11 | s12217–s12271 | 55 | next-B weight loads, pack |
| E12 | s12288–s13174 | 887 | RMS norm (², sum, rsqrt, gain) + packs + next C |
| E13 | s13191–s13899 | 709 | clamped cubic activation + column bias + pack |
| E14 | s13916–s13979 | 64 | next B/C loads, pack |
| E15 | s13996–s14595 | 600 | clamped cubic activation + pack |
| E16 | s14612–s14675 | 64 | next B/C loads, pack |
| E17 | s14692–s15291 | 600 | clamped cubic activation + pack |
| E18 | s15308–s15371 | 64 | next B/C loads, pack |
| E19 | s15388–s15987 | 600 | clamped cubic activation + pack |
| E20 | s16004–s16417 | 414 | pack → plane-arena scatter → 8-lane merge → next B/C |
| E21 | s16450–s16606 | 157 | pack → surface scatter → `%tid.x` → next B/C |
| E22 | s16615–s17222 | 608 | cross-lane row max, softmax, shared-patch gather, store |

---

## 3. The epilogues

### E5 — s3983–s7825: softmax of the 64×96 score block

**Phase 5** = 48 `mma` (s3935–s3982), D registers `%r2566…%r3037` (96
f16x2 = 64 rows × 96 columns held 8 rows × 8 cols per lane).  The epilogue has
**no memory access at all** (0 `ld.*`, 0 `st.*`); it also reads phase 4's D
(`%r2149…%r2380`, 48 registers) and writes 77 registers that survive:
the 5 constants of §1.2, phase 6's 12 A fragments (`%r5884 %r5885 %r5886 %r5887`,
`%r5904…`, `… %r6284…%r6287`) and its 12 B fragments
(`%r6178 %r6179 %r6188 %r6189 … %r6288 %r6289`).

Sub-sections:

| range | stmts | what |
|---|---|---|
| s3983–s6100 | 2118 | 96 per-element groups: scale, clamp, cubic, exponent extract |
| s6101–s6180 | 80 | partial row-sum trees (in-lane) |
| s6181–s6312 | 132 | 16 `shfl.sync.bfly` → 8 row sums |
| s6313–s7081 | 769 | 192 `rcp.approx.ftz.f32` → 96 reciprocals |
| s7082–s7561 | 480 | 192 `mul.f16` → 96 probabilities |
| s7562–s7609 | 48 | `movmatrix` transpose of phase 4's D |
| s7610–s7825 | 216 | 144 `cvt.rn.satfinite.e4m3x2.f16x2` packs |

#### (a) per-element exponent, s3983–s6100 (96 groups of 22 statements)

Evidence:

```
s3984  L8744: mov.f32 %f2085, 0f3C8CCB50
s3987  L8752: mul.f16x2 %r3048,%r2566,%r3047
s3992  L8766: max.f16x2 %r3051,%r3048,%r3053
s3997  L8780: min.f16x2 %r3054,%r3051,%r3056
s4003  L8798: fma.rn.f16x2 %r3059,%r3054,%r3057,%r3062
s4008  L8812: fma.rn.f16x2 %r3063,%r3054,%r3059,%r3066
s4009  L8815: shl.b32 %r16813, %r3063, 5
s4010  L8816: and.b32 %r4968, %r16813, 2145419232      // 0x7FE07FE0
```

For each of the 96 f16x2 D registers, per half (an f16 value `x`):

```
m  = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)   // s3987,s3992,s3997
t  = f16(m*(-m) + 0.92724609375)                                         // s4003 fma
u  = f16(m*t    + 1.375)                                                 // s4008 fma
    = f16(1.375 + m*(0.92724609375 - m^2))
E  = bits(u)[9:5]      // floor(32*frac(u))       -- the top five mantissa bits
F  = bits(u)[4:0]      // the low five mantissa bits
expval = f16 from bits  (E << 10) | (F << 5)      // = 2^(E-15) * (1 + F/32)
```

i.e. `((u_bits << 5) & 0x7FE07FE0)` keeps `u`'s mantissa and drops its sign and
exponent, so the mantissa lands in the exponent field.  Numerically
`expval ≈ 2^(32*frac(u) - 15) = 2^(32u - 47)` and, because the clamp keeps
`u ∈ [1.03125, 1.71875]`, `expval ∈ [2⁻¹⁴, 2⁸]`.
**This is not `exp` and not `exp2(x*log2e)`**: the exponent is the *cubic*
`32*(0.375 + 0.92724609375*m − m³)`, quantised to a 6-bit significand (1.5 %
relative granularity).  The 6-bit truncation is why the constant 1.375 is there:
`frac(u)` for `u∈[1,2)` *is* the mantissa.

The clamp bounds are the guard: `d/dm [0.92724609375m − m³] = 0.92724609375 − 3m²`
vanishes at `m = √(0.92724609375/3) = 0.55615…`, exactly the clamp, so the
exponent saturates at `32·(0.92724609375·0.55615 − 0.55615³) + 12 = 23`
(`expval = 2⁸`) at the top and at `E = 1` (`2⁻¹⁴`) at the bottom.  The two
`f64` literals are used only so the clamp is the *exact* f16 value.

#### (b) in-lane partial sums, s6101–s6180

96 `expval` registers are summed in three groups of twelve: each group does
12 `add.f16x2` then `(a+b)+c` four times, giving 4 f16x2 (= 8 f16) per group,
so **each lane reduces its 96 values (8 rows × 12 registers) to 24 partial sums
in 12 f16x2 registers**.

#### (c) cross-lane row sums, s6181–s6312

```
s6182 L15601: add.f16x2 %r5208,%r5015,%r5021
s6187 L15612: shfl.sync.bfly.b32 %r5212,%r5208,%r11301,%r5215,%r11312
s6188 L15616: add.f16x2 %r5217,%r5208,%r5212
s6193 L15627: shfl.sync.bfly.b32 %r5221,%r5217,%r11310,%r5224,%r11312
s6194 L15631: add.f16x2 %r5226,%r5217,%r5221
s6196 L15636: add.f16 %rs2318,%rs2319,%rs2320
s6197 L15639: mov.b32 %r5423, {%rs2318, %rs2318}
```

Eight times (one per row `g, g+8` of the four m-tiles): add the two f16x2
partials, butterfly over lane bits 0 and 1 (`%r11301 = 1`, `%r11310 = 2`),
add the two halves of the result with `add.f16`, then broadcast into both
halves.  Result: **8 row sums** `%r5423 %r5427 %r5471 %r5475 %r5519 %r5523
%r5567 %r5571` (rows `g`, `g+8` of m-tiles 0…3), covering all 96 columns of each
row because the four lanes with equal `laneid&7` hold different column pairs.

**There is no row maximum anywhere in this epilogue**: `max.f16x2` (96×) and
`min.f16x2` (96×) are the clamp of (a); the only warp shuffles are the 16
`shfl.sync.bfly` above.  See §6.

#### (d) reciprocal, s6313–s7081

192 `rcp.approx.ftz.f32` = 96 f16x2 reciprocals, each computed as
`cvt.f32.f16 → rcp.approx.ftz.f32 → cvt.rn.f16.f32` on a row-sum register
(duplicated, so each row sum is inverted four times — once per n-tile of that
row):

```
s6314 L15935: mov.b32 {hl, hu}, %r5423
s6317 L15938: rcp.approx.ftz.f32 fl, fl
s6319 L15940: cvt.rn.f16.f32 hl, fl
s6321 L15942: mov.b32 %r5380, {hl, hu}
```

#### (e) probabilities, s7082–s7561

```
s7082 L17180: mov.b32 {%rs2343, %rs2346}, %r4968     // expval (row g)
s7083 L17181: mov.b32 {%rs2344, %rs2347}, %r5380     // 1/rowsum (row g)
s7084 L17183: mul.f16 %rs2345,%rs2346,%rs2347
s7085 L17187: mul.f16 %rs2342,%rs2343,%rs2344
s7086 L17190: mov.b32 %r5716, {%rs2342, %rs2345}
```

192 `mul.f16` → 96 f16x2 results `p = expval * (1/rowsum)` — the row sums
computed in (b)/(c) cancel the constant `2⁻⁴⁷` of (a), so `p_ij ∝
2^(32·u_ij)`, and `Σ_j p_ij = 1` up to the `rcp.approx` error.

#### (f) V-transpose, s7562–s7609

```
s7562 L18237: movmatrix.sync.trans.aligned.m8n8.b16 %r5572, %r2149
s7563 L18240: movmatrix.sync.trans.aligned.m8n8.b16 %r5574, %r2150
```

**48** `movmatrix` on 48 distinct phase-4 D registers (`%r2149…%r2380`, each
exactly once).  Each instruction transposes one 8×8 b16 tile built from the 32
threads' `%r`, i.e. 48 × 64 = 3072 values = the whole 96×32 D tile.  (The
epilogue therefore contains 48, not 96, `movmatrix`.)

#### (g) packs, s7610–s7825

```
s7611 L18384: cvt.rn.satfinite.e4m3x2.f16x2 %rs2918, %r5572
s7612 L18386: mov.b32 %r6178, {%rs2918, %rs2919}
s7683 L18552: cvt.rn.satfinite.e4m3x2.f16x2 %rs2966, %r5716
s7684 L18554: mov.b32 %r5884, {%rs2966, %rs2967}
```

144 packs in three groups:

1. s7610–s7681 — from the **movmatrix** results → phase 6's 12 B fragments
   (`%r6178…%r6289`, 24 b32), i.e. the transposed V.
2. s7682–s7799 — from the **probabilities** → phase 6's 12 A fragments
   (`%r5884…%r6287`, 48 b32).  Each fragment consumes 8 consecutive products;
   the four halves are re-ordered (`%r5884 ← (p0,p1)`, `%r5886 ← (p2,p3)`,
   `%r5885 ← (p4,p5)`, `%r5887 ← (p6,p7)`) to match the e4m3 fragment byte order.
3. s7800–s7825 — more probability packs whose destinations are *not* read after
   s7825 (compiler spill; harmless).

**Inputs:** phase 5's 96 D f16x2, phase 4's 48 D f16x2, `%r11301/%r11310/%r11312`.
**Outputs:** the 5 constants + phase 6's 48 A-register / 24 B-register operands.
**Memory read/written:** none.

---

### E6 — s7874–s8033: residual + per-column table, then pack

160 statements.  Evidence:

```
s7877 L19225: add.s64 %rd161, %rd229, 20672
s7878 L19227: ld.weak.global.ca.v4.u32 { %r6293,%r6294,%r6295,%r6296},[%rd161]
s7880 L19231: ld.weak.global.ca.v4.u32 { %r6297,%r6298,%r6299,%r6300},[%rd162]   // +21184
s7886 L19240: ld.global.v2.u16 {%rs6191, %rs6192}, [%rd232+37072]
s7891 L19252: {add.f16 %rs3062,%rs6197,%rs3064;}
s7892 L19254: mov.b32 %r6342, {%rs3062, %rs3065}
s7987 L19536: cvt.rn.satfinite.e4m3x2.f16x2 %rs3254, %r5852
```

Four things happen, in this order:

1. **Next-B loads** (s7874–s7880): two `v4.u32` fragments from
   `W + 16*laneid + {20672, 21184}` → `%r6293…%r6300`, phase 7's B operand.
2. **Table loads** (s7881–s7889): four `ld.global.v2.u16` from
   `W + ((laneid<<2)&12) + {37056, 37072, 37088, 37104}` → 8 f16
   (`%rs6191…%rs6198`).  16 bytes/lane × 32 lanes × 4 words = 512 bytes: a
   **per-column** vector covering 32 columns (4 words × 8 columns, each lane
   holding the two columns `2t,2t+1` of two of the four 8-column groups).
3. **Residual + column table add** (s7890–s7985): 64 `add.f16`, i.e. 32 f16x2,
   computed *in place* on the f16 halves of **phase 2's D**
   (`%r1174…%r1325`, the halves unpacked at s2399–s2555 by
   `mov.b32 {%rs3064, %rs3067}, %r1175` and its 38 siblings):

   ```
   new_low  = table[2t]   + D2_half_low        s7891
   new_high = table[2t+1] + D2_half_high       s7890
   → packed into %r6342, %r6343, %r6352, … %r6493   (32 registers)
   ```

   These 32 registers are **phase 7's C operand**, so phase 7 computes
   `D7 = A7·B7 + (D2 + column_table)` — an out-projection plus the residual.
4. **Pack** (s7986–s8033): 32 packs of **phase 6's D** (`%r5852`, `%r5862`,
   `%r5912`, `%r5922`, `%r5972`, … — 32 of the 48 D registers, i.e. the 64×32
   output) → 16 b32 = phase 7's A operand (`%r6366 %r6367 %r6368 %r6369`,
   `%r6406…`, `%r6446…`, `%r6486…`).

**Outputs (50 live):** `%r6293 %r6297` (B), the 32 C seeds `%r6342…%r6493`, the
16 packed A registers.  **Memory:** 2 weight loads + 4 table reads.

---

### E7 — s8050–s8056: next-B loads

```
s8053 L19761: add.s64 %rd163, %rd234, 21696
s8054 L19763: ld.weak.global.ca.v4.u32 { %r6495,%r6496,%r6497,%r6498},[%rd163]
s8055 L19765: add.s64 %rd164, %rd234, 22208
s8056 L19767: ld.weak.global.ca.v4.u32 { %r6499,%r6500,%r6501,%r6502},[%rd164]
```

Two 16-byte/lane fragments of `W + 16*laneid + {21696, 22208}` = phase 8's B
operand (weight block `[21696, 22720)` per the phase table).  Nothing else.

### E8 — s8081–s8087: next-B loads

Same shape, offsets `+22720` and `+23232`:

```
s8084 L19942: add.s64 %rd165, %rd236, 22720
s8085 L19944: ld.weak.global.ca.v4.u32 { %r6744,%r6745,%r6746,%r6747},[%rd165]
s8086 L19946: add.s64 %rd166, %rd236, 23232
s8087 L19948: ld.weak.global.ca.v4.u32 { %r6748,%r6749,%r6750,%r6751},[%rd166]
```

→ phase 9's B operand, weight block `[22720, 23744)`.

### E9 — s8112–s8282: score-bias C-seed loads + pack of phase 8's D

171 statements, 24 loads + 80 packs:

```
s8116 L20125: ld.weak.global.ca.v4.u32 { %r6993,…%r6996},[%rd167]   // +23744
   …
s8162 L20217: ld.weak.global.ca.v4.u32 { %r7085,…%r7088},[%rd190]   // +23744+23*512 = +35520
```

* **24 loads** at `W + 16*laneid + 23744 + 512*i`, `i = 0…23` — the phase-10
  score-bias seeds, byte block `[23744, 36032)` = 12288 bytes = 6144 f16.  This
  is the twin of phase 5's C-seed block (`[8384, 20672)`), loaded the same way
  in phase 4's epilogue (s3764–s3814).
* **80 packs / 40 b32**, whose sources are *all* registers written by **phase 8's
  mma** (`%r6503…%r6744`, the D pairs of s8057–s8080) — phase 9's D
  (`%r6752…%r6983`) is **not** read here; it is consumed by E10's `movmatrix`
  block instead (s11905: `movmatrix.sync.trans.aligned.m8n8.b16 %r10175,
  %r6752`).  The 40 packed b32 split into
  * phase 10's four A-fragment sets: `%r7281 %r7282 %r7283 %r7284`,
    `%r7401…`, `%r7521…`, `%r7641…` (16 b32, e.g. `s8237: mov.b32 %r7281,
    {%rs3334, %rs3335}` from `cvt` of `%r6503`/`%r6523` at s8235–s8236), and
  * phase 10's twelve B-fragment pairs `%r7535 %r7536 … %r7645 %r7646`
    (24 b32, e.g. `s8163–s8165: cvt …%r6523; cvt …%r6503; mov.b32 %r7535,
    {%rs3286, %rs3287}`).

  The first A set and the first B set are **bit-identical**: both are
  `{pack(%r6503), pack(%r6523)}` (`%r7281` from s8235–s8237, `%r7535` from
  s8163–s8165).  Whatever the score GEMM's two operands are, phase 10 gets one
  of them twice at fragment 0.

**The C-seed layout** (identical for phase 5 and phase 10): the four C registers
of one mma are the four u32 of one 16-byte lane slot, and

```
C(m, n) = load_tile(4*(n>>1) + m) + 4*(n&1)      [register index within the block]
byte    = base + 512*(4*(n>>1) + m) + 16*laneid + 8*(n&1)
```

so the bias tile index runs `4*(n>>1) + m` (stride 512 bytes) over 4 m-tiles and
6 `(n>>1)` groups = 24 tiles = 12288 bytes.  Verified from phase 5's own mma
operands: `s3935 C=%r2390, s3936 C=%r2392, s3937 C=%r2406, s3947 C=%r2394,
s3959 C=%r2398, s3971 C=%r2402, s3982 C=%r2484`.

---

### E10 — s8331–s12168: softmax, second window

**Identical code to E5** with shifted registers; the op histogram is byte-for-byte
the same (1048 `mov.b32`, 672 `cvt.rn.f16.f32`, 384 `cvt.rn.f16.f64`, 192 each
`fma.rn.f16x2` / `cvt.f32.f16` / `rcp.approx.ftz.f32` / `mul.f16`, 144 packs,
48 `movmatrix`).  It re-uses E5's constants (`%f2085 %fd767 %fd769 %f2087
%f2089`, defined at s3984–s4004) and E5's shfl operands `%r11301 %r11310
%r11312`, so it contains no `mov.f32`/`mov.f64` of its own.

Sub-sections:

| range | what |
|---|---|
| s8331–s10443 | 96 exponent groups (scale/clamp/cubic/`and.b32 0x7FE07FE0`) |
| s10444–s10523 | 80 in-lane partial-sum `add.f16x2` |
| s10524–s10655 | 16 `shfl.sync.bfly` → 8 row sums (`s10530 L27701`, `s10651 L27990`) |
| s10656–s11424 | 192 `rcp.approx.ftz.f32` (`s10660 L28027`) |
| s11425–s11904 | 192 `mul.f16` → 96 probabilities |
| s11905–s11952 | 48 `movmatrix` (`s11905 L30326: movmatrix… %r10175, %r6752`) |
| s11953–s12168 | 144 packs (`s11953 L30470`) |

**Inputs:** phase 10's 96 D (`%r7169…`), phase 9's 48 D.  **Outputs (72 live):**
phase 11's 12 A fragments (`%r10487…%r10892`) + 12 B fragments (`%r10781…`)
+ the 4 constants.  **Memory:** none.

---

### E11 — s12217–s12271: next-B loads + pack

55 statements:

```
s12220 L31314: add.s64 %rd191, %rd240, 36032
s12221 L31316: ld.weak.global.ca.v4.u32 { %r10896,…%r10899},[%rd191]
s12222 L31318: add.s64 %rd192, %rd240, 36544
s12223 L31320: ld.weak.global.ca.v4.u32 { %r10900,…%r10903},[%rd192]
s12224 L31323: cvt.rn.satfinite.e4m3x2.f16x2 %rs4879, %r10515
s12226 L31328: mov.b32 %r10968, {%rs4878, %rs4879}
```

* two weight fragments at `W + 16*laneid + {36032, 36544}` → phase 12's B;
* **32 packs** of phase 11's D (`%r10455…%r10645`) → 16 b32 = phase 12's A
  (`%r10968 %r10969 %r10970 %r10971`, `%r11008…`, `%r11048…`, `%r11088…`).

**Unlike E6, no table**: phase 11's output is requantised only, because
the residual for the second attention branch is folded into phase 12's C operand
(`%r6334 %r6335` = phase 7's D, see the phase table).

#### Shape of the attention region (E5–E12)

```
P3  A = prologue tile, B = W+6336  C = 0        → D_K   (96×32)
P4  A = prologue tile, B = W+7360  C = 0        → D_QV  (96×32)   epilogue E4 packs both
     E4 packs D_K  → P5 B  (s3817: %r2932),     E4 loads P5 C = bias@8384 (s3764–s3814)
P5  A = pack(D_QV)   B = pack(D_K)  C = bias@8384    → scores 64×96
     E5: softmax over the 96 columns (no row max), 48 movmatrix(D_QV),
         pack → P6 A (48 b32) + P6 B (24 b32)
P6  A = pack(P)      B = pack(transpose(D_QV))  C = 0 → D_PV (64×32)
     E6: pack(PV) → P7 A (16 b32);  P7 C = D_QV2 + column table@37056;  loads P7 B
P7  C = residual       → D7                      epilogue E7 loads P8 B
P8  A = prologue tile, B = W+21696, C = 0  → D8    epilogue E9 packs D8 → P10 A and B
P9  A = prologue tile, B = W+22720, C = 0  → D9    (D9 only reaches E10's movmatrix)
P10 A = pack(D8)   B = pack(D8)   C = bias@23744 → scores 64×96
      E10: same softmax + transpose(D9) + packs → P11 A/B
P11 … → E11 packs → P12 A;  P12 C = D7 (residual)
```

Window 2 therefore mirrors window 1 except that phase 10's *two* operands both
come out of phase 8 (see E9), while the transpose of phase 9's D supplies
phase 11.

---

### E12 — s12288–s13174: RMS norm before the MLP

887 statements, no stores, one `ld.global.v2.u16` group, two weight groups.
Structure:

| range | stmts | op |
|---|---|---|
| s12288–s12415 | 128 | 32 × `x·x` squares of phase 12's D |
| s12416–s12431 | 16 | in-lane pairwise `add.f16x2` |
| s12432–s12555 | 124 | 8 rows × (`add` + 2 `shfl.sync.bfly` + `add` + `add.f16`) |
| s12556–s12652 | 97 | `+ eps`, broadcast into 16 b32 |
| s12653–s12909 | 257 | 32 × `rsqrt.approx.ftz.f32` |
| s12910–s12918 | 9 | gain loads `+37120/37136/37152/37168` |
| s12919–s13014 | 96 | `inv · gain` |
| s13015–s13110 | 96 | `x · (inv·gain)` |
| s13111–s13117 | 7 | next-B loads `+37184/+37696` |
| s13118–s13126 | 9 | next-C loads `+38208/38224/38240/38256` |
| s13127–s13174 | 48 | 32 packs → phase 13's A |

Evidence:

```
s12289 L31548: mul.f16 %rs4913,%rs5740,%rs5740          // square of a D half
s12438 L31945: shfl.sync.bfly.b32 %r11149,%r11145,%r11301,%r11152,%r11312
s12556 L32255: cvt.rn.f16.f64 %rs5126, %fd770           // eps = 2^-13
s12657 L32555: rsqrt.approx.ftz.f32 fl, fl
s12915 L32972: ld.global.v2.u16 {%rs6199, %rs6200}, [%rd243+37136]
s13015 L33265: mul.f16 %rs5514,%rs5740,%rs5322          // D_half * (inv*gain)
s13114 L33557: add.s64 %rd193, %rd245, 37184
s13123 L33572: ld.global.u32 %r11553, [%rd248+38208]
s13128 L33580: cvt.rn.satfinite.e4m3x2.f16x2 %rs5703, %r15389
```

**What it computes.** For each of the 8 rows a lane holds (8 columns each):

```
sumsq[row] = Σ_{c=0..31} x[row,c]^2                      (f16, 4-lane butterfly)
inv        = rsqrt.approx.ftz.f32(f32(f16(sumsq + 2^-13)))    // rounded to f16
out[row,c] = x[row,c] * inv * gain[c]
```

* **no mean subtraction and no `1/N`**: the quantity is the *raw* sum of the 32
  squares in a row plus the constant `2^-13`; the mean is not formed.
* `gain[c]` is read from `W + ((laneid<<2)&12) + {37120, 37136, 37152, 37168}`
  (8 f16 per lane = 32 columns), so the gain is per column, constant across rows.
* the `inv` register is duplicated into both halves of a b32 before use
  (`mov.b32 %r11318, {%rs5127, %rs5130}` at s12559), so one row-sum value serves
  all four n-tiles of that row (32 `rsqrt` for 8 row sums).
* the norm's *bias* is not added here — it is folded into phase 13's C operand,
  loaded at the end of this epilogue (`+38208…+38256`).

**Outputs (22):** `%r11383 %r11387` (phase 13 B, from `+37184/+37696`),
`%r11553 %r11563 %r11573 %r11583` (phase 13 C), 16 packed A registers
(`%r11456 %r11457 %r11458 %r11459`, `%r11496…`, `%r11536…%r11539`), the 32 f16x2
normalised values `%r15389…%r15420` (re-packed in E14), and the 32
half-registers `%rs5737 %rs5743 … %rs5923` still read by E13's bias add.

---

### E13 — s13191–s13899: clamped cubic activation, column bias, pack

709 statements.  This is the only epilogue that materialises `%f2598…%f2601`
(the four activation constants of §1.2).

```
s13192 L33803: mov.f32 %f2598, 0f3ED306EB
s13198 L33817: mov.f32 %f2600, 0f3F000000          // 0.5
s13201 L33824: mov.f32 %f2601, 0f40000000          // 2.0
s13204 L33832: neg.f16x2 %r11589,%r11588
s13205 L33836: max.f16x2 %r11591,%r11424,%r11589
s13206 L33840: min.f16x2 %r11594,%r11591,%r11588
s13207 L33844: abs.f16x2 %r11597,%r11594
s13208 L33848: mul.f16x2 %r11599,%r11586,%r11597
s13209 L33852: sub.f16x2 %r11602,%r11585,%r11599
s13210 L33856: mul.f16x2 %r11605,%r11594,%r11602
s13211 L33860: add.f16x2 %r11608,%r11587,%r11605
s13212 L33864: mul.f16x2 %r11611,%r11424,%r11608
```

Per element, over each of the 32 f16x2 D registers of phase 13
(`%r11424 %r11425 %r11504…`, 64 f16 per lane):

```
y = clamp(x, -2, +2)
g = 0.5 + y * (0.412109375 - 0.0810546875 * |y|)      // f16 throughout
out = x * g
```

`g` is a *cubic ramp*, not a sigmoid: `g(+2) = 1` ⇒ `out = x` for `x ≥ 2`;
`g(-2) = 0` ⇒ `out = 0` for `x ≤ -2`; `g(1) = 0.831055` where `x·sigmoid(x) =
0.730957`.  **Reproduce the formula, not SiLU/GELU.**

Then:

| range | op |
|---|---|
| s13191–s13739 | 32 activation groups (each group 4 constants + 9 f16x2 ops) |
| s13740–s13746 | next-B loads: `+38272` → `%r12514…%r12517`, `+38784` → `%r12518…%r12521` |
| s13747–s13755 | lane-shift setup + `ld.global.v2.u16` at `W + ((laneid<<2)&12) + {39296, 39312, 39328, 39344}` |
| s13756–s13851 | 64 `add.f16` — table add, written over the **activation inputs** |
| s13852–s13899 | 32 packs of the activation outputs → phase 14's A |

Evidence for the table add:

```
s13752 L35747: ld.global.v2.u16 {%rs6207, %rs6208}, [%rd253+39312]
s13756 L35752: {add.f16 %rs5738,%rs6214,%rs5740;}
s13757 L35756: {add.f16 %rs5735,%rs6213,%rs5737;}
s13758 L35759: mov.b32 %r12563, {%rs5735, %rs5738}
s13853 L36043: cvt.rn.satfinite.e4m3x2.f16x2 %rs5927, %r11611
```

`%rs5737/%rs5740` are the halves of `%r11424` (phase 13's D, split at s13191),
so `%r12563 = phase13_D + table[column]`.  The 32 such registers
(`%r12563 %r12564 %r12573 %r12574 … %r12713 %r12714`) are **phase 14's C
operand**; `%r12514 %r12518` are phase 14's B fragments (weight image).

**Outputs (54 live):** `%f2598…%f2601` (the constants — consumed again by E15,
E17, E19), the 32 C seeds, the 16 packed A registers, `%r12514 %r12518`.

---

### E14 — s13916–s13979: next B/C loads + pack

64 statements:

```
s13919 L36268: add.s64 %rd197, %rd255, 39360
s13920 L36270: ld.weak.global.ca.v4.u32 { %r12716,…%r12719},[%rd197]
s13921 L36272: add.s64 %rd198, %rd255, 39872
s13922 L36274: ld.weak.global.ca.v4.u32 { %r12720,…%r12723},[%rd198]
s13928 L36283: ld.global.u32 %r12886, [%rd258+40384]
s13929 L36284: ld.global.u32 %r12896, [%rd258+40400]
s13930 L36285: ld.global.u32 %r12906, [%rd258+40416]
s13931 L36286: ld.global.u32 %r12916, [%rd258+40432]
s13933 L36293: cvt.rn.satfinite.e4m3x2.f16x2 %rs5959, %r15389
s13934 L36295: mov.b32 %r12789, {%rs5959, %rs5960}
```

* phase 15's B at `W + 16*laneid + {39360, 39872}`;
* **phase 15's C/bias**: 4 `ld.global.u32` at `W + ((laneid<<2)&12) +
  {40384, 40400, 40416, 40432}` → `%r12886 %r12896 %r12906 %r12916`.  Each of
  those four u32 is used **in both halves** of an mma C operand
  (`{%r12886, %r12886}` at s13980), i.e. a per-column bias replicated over the
  16 rows of a tile and reused by all four m-tiles;
* **32 packs of `%r15389…%r15420`** — the *normalised* values E12 produced, so
  phase 15's A operand is bit-identical to phase 13's A operand (phases 13 and
  15 are two parallel GEMMs off the same normed tile).

---

### E15 — s13996–s14595: clamped cubic activation, pack

600 statements, histogram: 96 `mul.f16x2`, 32 each `neg/max/min/abs/sub/add`
f16x2, 128 `cvt.rn.f16.f32`, 144 `mov.b32`, 32 packs, 2 weight loads — and
**no constants of its own** and **no `ld.global`**.

```
s13997 L36516: cvt.rn.f16.f32 low, %f2598            // constants from E13
s14005 L36539: neg.f16x2 %r12922,%r12921
s14006 L36543: max.f16x2 %r12924,%r12757,%r12922
s14009 L36555: mul.f16x2 %r12932,%r12919,%r12930
s14013 L36571: mul.f16x2 %r12944,%r12757,%r12941
s14545 L38441: ld.weak.global.ca.v4.u32 { %r13847,…%r13850},[%rd199]   // +40448
s14547 L38445: ld.weak.global.ca.v4.u32 { %r13851,…%r13854},[%rd200]   // +40960
```

Same function as E13: `out = x·(0.5 + y·(0.412109375 − 0.0810546875|y|))`,
`y = clamp(x,±2)`, over phase 15's 32 D registers (`%r12757 %r12758 …`), then
32 packs **of the activation outputs** (`%r12944` at s14549, `%r13060` at s14548,
…) → phase 16's A (`%r13919…%r14042`), plus phase 17's B fragments from
`W + 16*laneid + {40448, 40960}`.  **No column-bias add.**

### E16 — s14612–s14675: next B/C loads + pack

Identical shape to E14 with offsets `+41472/+41984` (B) and
`+42496/42512/42528/42544` (C, 4 `ld.global.u32` → `%r14218 %r14228 %r14238
%r14248`), followed by **32 packs of `%r15389…%r15420` — again the normed tile
E12 produced** — giving phase 17's A (`%r14121 %r14122 %r14123 %r14124`,
`%r14161…`, `%r14201…`).

Evidence: `s14616 L38678: ld.weak.global…[%rd201]`, `s14624 L38691: ld.global.u32
%r14218, [%rd265+42496]`, `s14629 L38699: cvt.rn.satfinite.e4m3x2.f16x2 %rs6023,
%r15389` (the pack source is E12's `%r15389`, defined at s13017).

### E17 — s14692–s15291: clamped cubic activation, pack

600 statements; identical histogram to E15.

```
s14693 L38924: cvt.rn.f16.f32 low, %f2598
s14701 L38947: neg.f16x2 %r14254,%r14253
s14702 L38951: max.f16x2 %r14256,%r14089,%r14254      // phase 17's D
s14709 L38979: mul.f16x2 %r14276,%r14089,%r14273
s15241 L40849: ld.weak.global.ca.v4.u32 { %r15179,…},[%rd203]      // +42560
s15244: cvt.rn.satfinite.e4m3x2.f16x2 …
```

Activation on phase 17's D, 32 packs **of those activation outputs** (`%r14276`
at s15245, `%r14392` at s15244, …) → phase 18's A, 2 weight loads at
`W + 16*laneid + {42560, 43072}` → phase 18's B.  No constants, no `ld.global`.

### E18 — s15308–s15371: next B/C loads + pack

Same shape as E14/E16: B at `+43584/+44096`, C at
`+44608/44624/44640/44656` → `%r15550 %r15560 %r15570 %r15580`, then **32 packs
of `%r15389…%r15420` — the normed tile for the fourth time** → phase 19's A
(`%r15453…%r15576`).  Evidence: `s15312 L41086: ld.weak.global…`,
`s15320 L41100: ld.global.u32 %r15550, [%rd272+44608]`, and
`s15325 L41107: cvt.rn.satfinite.e4m3x2.f16x2 %rs6087, %r15389` (source defined
at s13017).

### E19 — s15388–s15987: clamped cubic activation, pack

600 statements; identical histogram to E15/E17.

```
s15389 L41332: cvt.rn.f16.f32 low, %f2598
s15397 L41355: neg.f16x2 %r15586,%r15585
s15398 L41359: max.f16x2 %r15588,%r15421,%r15586      // phase 19's D
s15399 L41363: min.f16x2 %r15591,%r15588,%r15585
s15937 L43257: ld.weak.global.ca.v4.u32 { %r16511,…},[%rd207]      // +44672
s15940 L43270: cvt.rn.satfinite.e4m3x2.f16x2 …
```

Activation on phase 19's D, 32 packs **of those activation outputs** (`%r15608`
at s15941, `%r15724` at s15940, …) → phase 20's A (`%r16583…%r16706`), 2
weight loads at `W + 16*laneid + {44672, 45184}` → phase 20's B.

#### Shape of the whole MLP region (E12–E20)

Putting the pack sources and the C operands together (all verified by tracing each
`mov.b32 %rX, {%rsA, %rsB}` back through its two `cvt` statements):

```
E12  normed tile N (32 f16x2, %r15389…%r15420)
 ├─ pack → phase 13 A   (s13127)   phase 13: A=N, B=W13, C=bias@38208  → SiLU (E13)
 │                                              pack of SiLU → phase 14 A (s13852)
 │                                              phase 14: C = 0          → D14
 ├─ pack → phase 15 A   (s13932)   phase 15: A=N, B=W15, C=bias@40384  → SiLU (E15)
 │                                              pack of SiLU → phase 16 A (s14548)
 │                                              phase 16: C = D14        → D16
 ├─ pack → phase 17 A   (s14628)   phase 17: A=N, B=W17, C=bias@42496  → SiLU (E17)
 │                                              pack of SiLU → phase 18 A (s15244)
 │                                              phase 18: C = D16        → D18
 └─ pack → phase 19 A   (s15324)   phase 19: A=N, B=W19, C=bias@44608  → SiLU (E19)
                                                pack of SiLU → phase 20 A (s15940)
                                                phase 20: C = D18        → D20 (E20)
```

so the four branches share one input tile, each has its own weight+bias+SiLU, and
their outputs are folded into a running residual (`D16 = … + D14`,
`D18 = … + D16`, `D20 = … + D18`).  The four `SiLU` sites are byte-identical
blocks of the same f16 expression with the same four constants.

---

### E20 — s16004–s16417: pack → plane-arena scatter → 8-lane merge → next B/C

414 statements.  Four sub-sections:

#### (a) requantise, s16004–s16051

32 `cvt.rn.satfinite.e4m3x2.f16x2` + 16 `mov.b32` → `%r262 … %r277`
(phase 20's 64×32 output as 16 b32 = 32 e4m3 values per lane).

#### (b) plane-arena scatter, s16052–s16326

```
s16052 L43599: ld.param.u64 %rd276, [%rd368+48]
s16053 L43600: cvta.to.global.u64 %rd37, %rd276
s16055 L43604: shr.u32 %r278, %r16743, 2                 // g = laneid>>2
s16056 L43605: and.b32 %r279, %r16743, 3                 // t = laneid&3
s16059 L43608: add.s32 %r17110, %r6, %r17108             // y = %r6 + (laneid>>5)
s16065 L43614: add.s32 %r17111, %r1, %r17109             // x = %r1 + (g&7)
s16070 L43619: shl.b32 %r17112, %r17111, 3
s16071 L43620: mad.lo.s32 %r280, %r18179, %r17110, %r17112
s16072 L43621: add.s32 %r17113, %r279, %r280
s16073 L43622: selp.b32 %r281, -1, %r17113, %p35
s16074 L43623: setp.lt.s32 %p317, %r281, 0
s16075 L43624: @%p317 bra $L__BB1_258
s16078 L43628: st.global.u32 [%rd278], %r262
s16080 L43631: or.b32 %r282, %r279, 4
s16087 L43639: st.global.u32 [%rd280], %r263
```

The stride register is `%r18179 = shl.b32 %r18179, %r364, 3` (s2219): **8·ex**,
and `%r364/%r365 = ld.param.v2.u32 [param_0+0]`.

**All sixteen stores** (`s` = `st.global.u32 [%rd37 + 4*index], value`), where
`idx = stride*y + (x<<3) + d`:

| # | statement | value | stride | y | x | d |
|---|---|---|---|---|---|---|
| 1 | s16078 | `%r262` | `%r18179` | `%r6 + (laneid>>5)` | `%r1 + ((laneid>>2)&7)` | `t` |
| 2 | s16087 | `%r263` | `%r18179` | as above | as above | `t\|4` |
| 3 | s16113 | `%r264` | `%r18180 = %r364<<3` | `%r6 + ((g+8)>>3)` | `%r1 + ((g+8)&7)` | `t` |
| 4 | s16121 | `%r265` | `%r18180` | as above | as above | `t\|4` |
| 5 | s16147 | `%r266` | `%r18181` | `%r6 + ((g+16)>>3)` | `%r1 + ((g+16)&7)` | `t` |
| 6 | s16155 | `%r267` | `%r18181` | as above | as above | `t\|4` |
| 7 | s16181 | `%r268` | `%r18182` | `%r6 + ((g+24)>>3)` | `%r1 + ((g+24)&7)` | `t` |
| 8 | s16189 | `%r269` | `%r18182` | as above | as above | `t\|4` |
| 9 | s16215 | `%r270` | `%r18183` | `%r6 + ((g+32)>>3)` | `%r1 + ((g+32)&7)` | `t` |
| 10 | s16223 | `%r271` | `%r18183` | as above | as above | `t\|4` |
| 11 | s16249 | `%r272` | `%r18184` | `%r6 + ((g+40)>>3)` | `%r1 + ((g+40)&7)` | `t` |
| 12 | s16257 | `%r273` | `%r18184` | as above | as above | `t\|4` |
| 13 | s16283 | `%r274` | `%r18185` | `%r6 + ((g+48)>>3)` | `%r1 + ((g+48)&7)` | `t` |
| 14 | s16291 | `%r275` | `%r18185` | as above | as above | `t\|4` |
| 15 | s16317 | `%r276` | `%r18186` | `%r6 + ((g+56)>>3)` | `%r1 + ((g+56)&7)` | `t` |
| 16 | s16325 | `%r277` | `%r18186` | as above | as above | `t\|4` |

with `g = laneid>>2`, `t = laneid&3`, every `%r1818N` equal to `%r364<<3`
(s16090, s16124, s16158, s16192, s16226, s16260, s16294), and each block guarded
by `p35…p43 = (y<0) | (y ≥ %r365) | (laneid>255) | (x<0) | (x ≥ %r364)`; a
guard failure stores `-1` as the index, which the following `setp.lt.s32
%p, %idx, 0` + `bra` turns into "skip the store".  Note `(g+8j)&7 = g&7` and
`(g+8j)>>3 = (g>>3)+j`, so blocks 1…7 only differ from block 0 in the row they
land on (`%r6 + g>>3 + j`) and in the value written.

The 16 stored b32 are the requantised 64×32 output of phase 20, so this writes
`8*32 = 256` bytes per row-stride unit of a `%r364`-wide arena.

#### (c) 8-lane merge, s16327–s16369

```
s16332 L43915: and.b32 %r17575, %r17171, 4              // laneid & 4
s16334 L43917: setp.eq.s32 %p389, %r17575, 0
s16338 L43921: shfl.sync.down.b32 %r17580|%p390, %r262, %r17576, %r17578, %r17579
s16339 L43922: shfl.sync.up.b32 %r17581|%p391, %r266, %r17577, %r17578, %r17579
s16340 L43923: selp.b32 %r17490, %r262, %r17581, %p389
s16341 L43924: selp.b32 %r17510, %r17580, %r266, %p389
```

**Eight** `shfl.sync.down/up` pairs, all with **delta = 4** — the *third*
operand (`%r17576 = mov.u32 4`, s16333) is the PTX `b` (delta/mask) operand and
the fourth is the group/width operand `c`: `31` for the `.down` (`%r17578`,
s16336) and `0` for the `.up` (`%r17577`, s16335) — with `c = 0` the "same
group" test `~(c-1) = 0` always succeeds, so the up-shuffle is unrestricted.
Membermask is `%r17579 = mov.u32 -1` (s16337).  The selection predicate is
`%p389 = (laneid & 4) == 0`:

```
down = a[lane+4]  (shfl.sync.down, delta 4)
up   = a[lane-4]  (shfl.sync.up,   delta 4)
dst1 = p389 ? own_low  : up        // lanes 0-3: own low ; lanes 4-7: lane-4's high
dst2 = p389 ? down     : own_high  // lanes 0-3: lane+4's low ; lanes 4-7: own high
```

so each of the eight destinations ends up holding, in its two halves, the values
of both members of the pair `{lane, lane±4}`.

The eight pairs take `(src_low, src_high)` =
`(%r262,%r266) (%r264,%r268) (%r263,%r267) (%r265,%r269) (%r270,%r274)
(%r272,%r276) (%r271,%r275) (%r273,%r277)` and write, in order,
`%r17490 %r17510 %r17530 %r17550 %r17492 %r17512 %r17532 %r17552 %r17491 %r17511
%r17531 %r17551 %r17493 %r17513 %r17533 %r17553` (s16340–s16369).  Those sixteen
are **phase 21's A operand**: `{%r17490 %r17491 %r17492 %r17493}`, `{%r17510 …}`,
`{%r17530 …}`, `{%r17550 …}` (four fragments of four, confirmed from the 32 mma
at s16418–s16449); the same sixteen are re-read as
`%r17490 %r17491 %r17492 %r17493 %r17497 %r17510 %r17511 %r17512 %r17513 %r17530
%r17531 %r17532 %r17533 %r17550 %r17551 %r17552 %r17553` by the loads that
follow.  Net effect: the sixteen 32-bit payloads that eight lanes scattered are
re-paired into four A fragments of four before the next GEMM.

#### (d) next B/C loads, s16370–s16417

```
s16373 L43958: add.s64 %rd309, %rd326, 45696
s16374 L43960: ld.weak.global.ca.v4.u32 { %r17173,…},[%rd309]   // … +45696+512*i, i=0..15
s16410 L44029: ld.global.u32 %r17247, [%rd329+53888]
s16411 L44030: ld.global.u32 %r17257, [%rd329+53904]
   … +53920, +53936, +53952, +53968, +53984, +54000
```

* **16 weight fragments** at `W + 16*laneid + 45696 + 512*i` → phase 21's B
  (`%r17173 … %r17236`);
* **8 bias words** at `W + ((laneid<<2)&12) + {53888…54000}` → `%r17247 %r17257
  %r17327 %r17337 %r17407 %r17417 %r17487 %r17497` = **phase 21's C seed**
  (128 bytes per lane = 64 f16 per lane).

**Inputs:** phase 20's D, `%r1/%r6/%r364/%r365/%r18179`, `W`, `S`.
**Outputs (58 live):** `%r262…%r277` (they are *read again* inside this same
epilogue by the merge), the 16 merged A registers, phase 21's B and C,
`%rd369 %rd370` (cached weight pointer).

---

### E21 — s16450–s16606: pack → surface scatter → `%tid.x` → next B/C

157 statements.

#### (a) requantise, s16450–s16472

```
s16450 L44261: add.s64 %rd38, %rd369, 54016
s16451 L44263: cvt.rn.satfinite.e4m3x2.f16x2 %rs6215, %r17298
s16453 L44269: cvt.rn.satfinite.e4m3x2.f16x2 %rs6218, %r17388
s16455 L44274: mov.b32 %r305, {%rs6217, %rs6218}
```

16 packs of phase 21's D → six b32 `%r305…%r310` plus the two `v2.u16` payloads
`{%rs6215,%rs6216}` and `{%rs6229,%rs6230}`.

#### (b) coordinate / surface scatter, s16473–s16592

```
s16473 L44316: ld.param.v2.u32 {%r17597, %r17598}, [%rd370]      // param block +0
s16474 L44317: shr.u32 %r313, %r17598, 1                         // ey>>1
s16475 L44318: shr.u32 %r314, %r17597, 1                         // ex>>1
s16476 L44319: shl.b32 %r315, %r314, 3                           // 8*(ex>>1)
s16477 L44320: mul.lo.s32 %r316, %r313, %r315                    // row stride
s16481 L44326: ld.param.u64 %rd331, [%rd370+56]
s16482 L44327: cvta.to.global.u64 %rd39, %rd331                  // destination buffer
s16487 L44332: bfi.b32 %r17603, %r17602, %r17599, 1, 31
s16490 L44335: shr.s32 %r319, %r17605, 1                         // y0 = %r6 >> 1 (arith)
s16506 L44351: mad.lo.s32 %r321, %r315, %r17606, %r17610
s16507 L44352: or.b32 %r17611, %r318, %r321
s16513 L44359: st.global.v2.u16 [%rd333], {%rs6215, %rs6216}
```

The destination pointer is **param block +56**, not the plane arena; extents come
from `param+0` and are *halved*.  Four index bases are formed, each guarded by
`p43 = (y<0)|(y≥(ey>>1))|(tile bits>3)|(x<0)|(x≥(ex>>1))`:

| base | statement | y | x | stores |
|---|---|---|---|---|
| `%r321` | s16506 | `(%r6>>1) + (((laneid>>5)&~2)\|((g&1)<<1))` | `(%r1>>1) + ((laneid>>3)&3)` | s16513 `v2.u16 {%rs6215,%rs6216}` at `+t`; s16522 `u32 %r305` at `+(t\|4)` |
| `%r325` | s16542 | recomputed from `g+8` | recomputed | s16549 `u32 %r306` at `+t`; s16557 `u32 %r307` at `+(t\|4)` |
| `%r328 = %r321 + %r316` | s16559 | (as `%r321`) | (as `%r321`) | s16566 `u32 %r308` at `+t`; s16574 `u32 %r309` at `+(t\|4)` |
| `%r331 = %r325 + %r316` | s16576 | (as `%r325`) | (as `%r325`) | s16583 `u32 %r310` at `+t`; s16591 `v2.u16 {%rs6229,%rs6230}` at `+(t\|4)` |

So: **8 guarded stores**, 6 × `st.global.u32` and 2 × `st.global.v2.u16`, index
`base + (t or t|4)` in units of 4 bytes.  (`%r325` and `%r328` are equal — see §6.)

#### (c) next B/C and `%tid.x`, s16593–s16606

```
s16595 L44456: mov.u32 %r18203, %tid.x
s16599 L44463: ld.weak.global.ca.v4.u32 { %r17629,…%r17632},[%rd348]   // +54016
s16605 L44472: ld.global.u32 %r17703, [%rd352+54528]
s16606 L44473: ld.global.u32 %r17713, [%rd352+54544]
```

phase 22's B fragment (`W + 16*laneid + 54016`) and two C seeds
(`W + ((laneid<<2)&12) + {54528, 54544}`).

**Outputs (5 live):** `%r17597` (the `param+0` v2.u32), `%r17629`,
`%r17703 %r17713`, `%r18203` (`%tid.x`) — two of the eighteen values the cut
returns.

---

### E22 — s16615–s17222: the final epilogue

608 statements; it consumes the eighteen values returned by the cut
(`%r17597`, `%r18203` and the sixteen D registers `%r17634 %r17635 %r17644
%r17645 %r17654 %r17655 %r17664 %r17665 %r17674 %r17675 %r17684 %r17685
%r17694 %r17695 %r17704 %r17705`) and writes the surfaces.

#### (a) cross-lane row max, s16615–s16765

```
s16616 L44534: and.b32 %r17717, %r17714, 7                 // row key = laneid & 7
s16617 L44535: shr.u32 %r17718, %r17714, 3                 // column group = laneid >> 3
s16618 L44536: and.b32 %r17719, %r17714, 3
s16620 L44537: selp.b32 %r17720, %r17635, %r17634, %p430   // pick by (laneid&3)
s16636 L44554: bfi.b32 %r17733, %r17717, %r17732, 2, 3
s16639 L44556: shfl.sync.idx.b32 %r17736|%p434, %r17722, %r17733, %r17734, %r17735
s16657 L44574: selp.b32 %r334, %r17748, %r17750, %p440
```

Two identical blocks (rows `g` from the D pairs `{r17634,r17635}`, `{r17654,r17655}`
and rows `g+8` from `{r17674,r17675}`, `{r17694,r17695}`), each doing:

1. a `selp` chain over `(laneid&3)` selects one of the four D pairs into
   `%r17722/%r17725/%r17728/%r17731` (the four n-tiles of the row segment);
2. four `shfl.sync.idx.b32` with `srcLane = (laneid&7) | ((((laneid>>3)+d)&3)<<2)`
   for `d = 0, 1, 2, -1` (`%r17734 = mov.u32 31`, `%r17735 = mov.u32 -1`);
3. a `selp` chain over `laneid>>3` picks between the four shuffled values.

Result: `%r334…%r337` (row `g`) and `%r339…%r342` (row `g+8`) hold the
**per-lane row maximum over the 8 columns**; the following `max.f16` / `max.f16x2`
chain (`s16792–s16799`, `s16800…`) reduces those four registers, then `sub.f16x2
%r17849, %r334, %r17863` (`s16800`) subtracts the row max: **this is a real row
max**, done with `shfl.sync.idx`, unlike E5's.

#### (b) bounds check, s16766–s16784

```
s16774 L44693: setp.ge.u32 %p462, %r346, %r17597           // x >= ex  (%r17597 = param+0 low)
s16775 L44694: setp.ge.u32 %p463, %r348, %r17598           // y >= ey
s16780 L44699: @%p466 bra $L__BB1_307
s16784 L44704: ld.param.u64 %rd353, [%rd372+112]            // surface A (param+112)
s16766 L44685: shr.s32 %r17836, %r18203, 31                // %r18203 = %tid.x (E21)
```

`%r18203` is used to derive `%r345 = tid.x - ((tid.x+round)>>3<<3)` (row/tile
offset), `%r346 = %r345 + %r1` (x), `%r348 = (…>>3) + %r6` (y); out-of-range
lanes jump to `$L__BB1_307` and skip (b)…(e).

#### (c) softmax, s16785–s16867

```
s16786 L44710: mul.ftz.f32 %f2630, %f2602, 0fBFB8AA3B      // -log2(e)
s16787 L44711: ex2.approx.ftz.f32 %f2631, %f2630
s16788 L44712: add.ftz.f32 %f2632, %f2631, 0f3F800000
s16790 L44713: div.approx.ftz.f32 %f2603, %f2633, %f2632   // 1/(1+2^(-max*log2e))
s16792 L44719: max.f16x2 %r17840,%r334,%r335
s16800 L44742: sub.f16x2 %r17849,%r334,%r17863             // x - rowmax
s16803 L44750: mul.ftz.f32 %f2634, %f2604, 0f3FB8AA3B      // *log2(e)
s16804 L44751: ex2.approx.ftz.f32 %f2606, %f2634           // 2^(…)
s16859 L44869: add.ftz.f32 %f2651, %f2649, %f2621          // row sum
s16860 L44870: div.approx.ftz.f32 %f2622, %f2633, %f2651   // 1/rowsum
s16863 L44875: mul.f16x2 %r17864,%r17865,%r17878
```

A textbook f32 softmax over each row's 8 values: `max` over the four lane
registers, `y = 2^((x-max)*log2e)` in f32 (`cvt.f32.f16 → mul by 0f3FB8AA3B →
ex2.approx`), accumulate the row sum in f32, invert with `div.approx.ftz.f32 1.0/…`,
round to f16, and multiply the five f16x2 registers
(`%r17864 %r17867 %r17870 %r17873 %r17876`).  The `%f2633` dividend is the
`mov.f32 %f2633, 0f3F800000` at s16789 (1.0).  Note `s16786` uses `-log2(e)` on
the *max* to build `1/(1+2^(-max·log2e))` (a per-row scale kept in `%rs6232`),
which is applied later to the second row group.

#### (d) shared-patch gather, s16868–s17143

```
s16871 L44899: mov.u32 %r17960, 8
s16868 L44896: mad.lo.s32 %r17969, %r347, 80, %r18206       // input_noise + 80*y
s16869 L44897: shl.b32 %r17970, %r345, 3
s16870 L44898: add.s32 %r17971, %r17969, %r17970
s16872 L44900: ld.shared.v4.u16 {%rs6327,…%rs6330}, [%r17971+8]
s16873 L44901: ld.shared.v4.u16 {%rs6331,…%rs6334}, [%r17971]
s16874 L44902: mov.b32 %r17880, {%rs6331, %rs6327}           // 2x2 transpose pair
s16875 L44904: fma.rn.f16x2 %r17879,%r17880,%r17864,%r17890  // acc += w * p
s16880 L44917: ld.shared.v4.u16 {%rs6341,…}, [%r17971+80]
s16881 L44918: ld.shared.v4.u16 {%rs6348,…}, [%r17971+16]
s16888 L44934: ld.shared.v4.u16 {%rs6355,…}, [%r17971+96]
s16889 L44935: ld.shared.v4.u16 {%rs6362,…}, [%r17971+88]
s16896 L44951: ld.shared.v4.u16 {…}, [%r17971+168]
s16897 L44952: ld.shared.v4.u16 {…}, [%r17971+160]
s16910 L44983: ld.shared.v4.u16 {…}, [%r17971+176]
```

18 `ld.shared.v4.u16` at `+0 +8 +16 | +80 +88 +96 | +160 +168 | +176`, i.e. a
2×2 (plus one extra column) stencil of the shared array `input_noise`
(`_ZZ33cuda_dldn_engine_swin_enc0_kernel35DldnEngineFusedSwinEnc0ParamsStructE11input_noise`),
which is filled in the prologue (s1457–s1547) as
`st.shared.v4.u16 [input_noise + y*80 + x*8], {f16c0, f16c1, f16c2, 0}` — a
10×10 grid of `f16x2` triples (three channels + a zero pad), 800 bytes.

The 2×2 blocks are multiplied by the five softmax registers with
`fma.rn.f16x2`, accumulated into `%r17879 %r17883 %r17887 %r17891…` (a 2×2
accumulator per row group), then the accumulator halves are summed with
`add.f16` (`s17135–s17137`) and one more shared value is folded in with
`fma.rn.f16` (`s17140–s17142`).

#### (e) sigmoid, difference, scale, cross-lane sum, s17143–s17200

```
s17145 L45494: mul.ftz.f32 %f2708, %f2679, 0fBFB8AA3B
s17146 L45495: ex2.approx.ftz.f32 %f2709, %f2708
s17148 L45497: div.approx.ftz.f32 %f2680, %f2689, %f2710    // sigmoid(acc)
s17156 L45515: mul.f16 %rs6445,%rs6394,%rs6429
s17160 L45530: mov.f32 %f2685, 0f3E800000                   // 0.25
s17162 L45536: sub.f16 %rs6454,%rs6429,%rs6445              // acc - sigmoid term
s17167 L45550: mul.f16x2 %r18080,%r18081,%r18082            // * 0.25
s17175 L45568: shfl.sync.down.b32 %r18084,%r18080,%r18105,%r18087,%r18115
s17176 L45574: add.f16x2 %r18089,%r18080,%r18084
s17191 L45613: add.f16 %rs6472,%rs6466,%rs6471
s17200 L45637: add.f16 %rs6478,%rs6472,%rs6477
```

`sigmoid(v) = 1/(1 + 2^(-log2e·v))`, then three `acc − sigmoid(acc)` differences
scaled by `0.25`, then a two-step cross-lane sum: `shfl.sync.down` with the
delta operand `%r18105 = 1` (s17175) then `%r18113 = 8` (s17181), each followed
by an `add.f16x2`; the same two steps repeat on the scalar path (s17188 delta 1,
s17197 delta 8, `add.f16` at s17191 and s17200).  The group/width operand is 31
in all four, so this is a partial (not five-step) warp reduction.

#### (f) surface stores, s17201–s17222

```
s17201 L45640: ld.param.u64 %rd362, [%rd371+120]            // surface B
s17202 L45641: ld.param.u64 %rd365, [%rd371+128]            // surface half
s17204 L45644: sust.b.2d.v4.b16.zero [%rd360, {%r18119,%r354}], {%rs6454,%rs6457,%rs6460,%rs6549}
s17205 L45647: sust.b.2d.v4.b16.zero [%rd362, {%r18119,%r354}], {%rs6445,%rs6448,%rs6451,%rs6550}
s17210 L45653: @%p477 bra $L__BB1_310                       // gated on (%r355&1)==0
s17220 L45665: sust.b.2d.v4.b16.zero [%rd365, {%r18134,%r18135}], {%rs6551,%rs6552,%rs6478,%rs6554}
s17222 L45669: ret
```

Three 4-channel b16 surface writes (params +112, +120, +128); the third is taken
only by even `%r355 = %r315|%r354` lanes and stores `0` in its fourth channel.

**Inputs:** the eighteen returned values, `W` (+54016/54528), `param+0/+112/+120/+128`,
the shared `input_noise` array, constants `0fBFB8AA3B`, `0f3FB8AA3B`,
`0f3F800000`, `0f3E800000`.
**Outputs:** none (the kernel returns).

---

## 4. The plane-arena writes (param block +48)

`S = cvta.to.global.u64(%rd276)` with `%rd276 = ld.param.u64 [%rd368+48]`
(s16052–s16053) is the **only** occurrence of `param_0+48` in the entry: I
grep'ed every `ld.param` and this offset is read once.

There are exactly **two** store groups in the whole entry (24 `st.global.*` in
the file, all after line 43628):

**(i) E20, 16 × `st.global.u32 [%rd37 + 4*idx], v` — s16078…s16325** (see §3/E20
for the full table).  In one line per store:

```
idx = (8*ex)*y + 8*x + d,   guard  0 <= y < ey  and  0 <= x < ex  and  laneid <= 255
store1  (s16078): v = %r262,  y = %r6 + (laneid>>5),                 x = %r1 + ((laneid>>2)&7),    d = t
store2  (s16087): v = %r263,  same                                    d = t|4
store3  (s16113): v = %r264,  y = %r6 + ((g+8)>>3),                   x = %r1 + ((g+8)&7),         d = t
store4  (s16121): v = %r265,  same                                    d = t|4
store5  (s16147): v = %r266,  y = %r6 + ((g+16)>>3),                  x = %r1 + ((g+16)&7),        d = t
store6  (s16155): v = %r267,  same                                    d = t|4
store7  (s16181): v = %r268,  y = %r6 + ((g+24)>>3),                  x = %r1 + ((g+24)&7),        d = t
store8  (s16189): v = %r269,  same                                    d = t|4
store9  (s16215): v = %r270,  y = %r6 + ((g+32)>>3),                  x = %r1 + ((g+32)&7),        d = t
store10 (s16223): v = %r271,  same                                    d = t|4
store11 (s16249): v = %r272,  y = %r6 + ((g+40)>>3),                  x = %r1 + ((g+40)&7),        d = t
store12 (s16257): v = %r273,  same                                    d = t|4
store13 (s16283): v = %r274,  y = %r6 + ((g+48)>>3),                  x = %r1 + ((g+48)&7),        d = t
store14 (s16291): v = %r275,  same                                    d = t|4
store15 (s16317): v = %r276,  y = %r6 + ((g+56)>>3),                  x = %r1 + ((g+56)&7),        d = t
store16 (s16325): v = %r277,  same                                    d = t|4
where g = laneid>>2, t = laneid&3, ex = %r364, ey = %r365,
      %r1 = x origin (s6), %r6 = y origin (s16)
```

**(ii) E21, 8 stores to `param+56`, not to the arena** — s16513 (`v2.u16` of
`%rs6215/%rs6216`), s16522, s16549, s16557, s16566, s16574, s16583 (`u32` of
`%r305…%r310`), s16591 (`v2.u16` of `%rs6229/%rs6230`).  See §3/E21 for the
indices.

**What is written:** phase 20's requantised 64×32 output — 16 b32 per lane, one
per store — i.e. 4 bytes × 32 lanes × 16 stores = 2048 bytes per block (4096
e4m3 elements' worth is not what lands in memory; only 16 of phase 20's 32 D
registers per lane are scattered).

**What reads it back:** nothing inside this kernel.  `param_0+48` is never the
base of an `ld.*`; P5 and P10 take their score bias from the *weight image*
(E9's 24 loads at `W+23744+512*i`, and phase 4's epilogue's 24 loads at
`W+8384+512*i`), and P5/P10 take their `B` operand from registers (phases 3/9
D → e4m3 packs).  See §6.

## 5. The eight shfl reductions of the merge

The brief's "eight shfl reductions" are the **eight `shfl.sync.down/up`
pairs of E20(c), s16338–s16368** (statement range s16327–s16369 including the
setup).  Each pair uses the same delta operand `%r17576 = 4` (the PTX `b`
operand), with the group/width operand `c` = 31 (`.down`) or 0 (`.up`), and
`laneid & 4` as the selection predicate, so each of the eight 32-bit payloads
(`%r262…%r277`) is re-paired between lanes 0-3 and 4-7 of every `laneid&7`
group before it becomes phase 21's A operand.

(There is a *second*, unrelated cross-lane reduction in E22: the four
`shfl.sync.idx` per row group at s16639–s16651 / s16680–s16683 / s16714–s16726 /
s16755–s16758 that implement the row maximum, plus the `shfl.sync.down` pair
after the sigmoid at s17175/s17181 and its scalar copy at s17188/s17197.)

## 6. Uncharacterised

**U1. "The plane-arena writes at stmts 15053–15198".**  In this file, with the
`rr_layer_spec.py` numbering:

* `s15053`–`s15198` lie **inside phase 17's epilogue** (s14692–s15291).  They
  are the per-output-channel activation coefficient materialisation
  (`cvt.rn.f16.f32 %rs, %f2601` / `neg` / `max` / `min` / `abs` / `mul` / `sub`
  / `mul` / `add` / `mul`) and contain **no store of any kind** — the entry has
  zero `st.*` between line 1450 and line 43628.
* The only writes to the plane arena are (i) and (ii) of §4, at s16078–s16325
  and s16513–s16591.
* No statement reads `param_0+48`, so nothing written there can be read back by
  P5 or P10 *within this launch*.

I could not reconcile 15053–15198 with this corpus file.  The two other
landmarks supplied with that range also disagree with it: `%r17597` (the
`ld.param.v2.u32` of `param+0`) is **s16473 / L44316**, and `%r18203` (`%tid.x`)
is **s16595 / L44456**, not 15448 and 15570.  *Missing:* the statement numbering
that produced 15053/15198/15448/15570 — most likely a different linearisation
(every `;`-terminated physical line of the whole file gives 14946 for
L44316 and 15375 for L43628; entry-only gives 14567 and 19913 for the
non-`;` variant, neither matches).  Give me that numbering (or the tool) and the
range will map directly.

**U2. The identity of the 12288-byte score bias.**  Both score phases add a
12288-byte f16 table as their GEMM C operand (P5: `W+8384…W+20672`, P10:
`W+23744…W+36032`), indexed as §3/E9 describes.  Whether it is a relative
position bias, a learned score bias, or a per-lane constant cannot be decided
from the PTX: it is an opaque blob in a permuted weight image.  *Missing:* the
weight image's layout description (the `_prep` permutation) or the parameter
names from the model file.

**U3. The identity of the per-column tables.**  `W+37056/37072/37088/37104`
(E6), `W+37120/37136/37152/37168` (E12, used as the norm gain), and
`W+39296/39312/39328/39344` (E13) are all read with the same
`W + ((laneid<<2)&12) + off` pattern and are 8 f16 per lane (32 columns).  Their
role is *computable* (E6: added to the residual before phase 7; E12: multiplies
the inverse std; E13: added after the activation) but their *names* are not.
The GEMM biases at `+38208/40384/42496/44608/53888/54528` are the same case.

**U4. E21's index bases.**  `%r325` (s16542) and `%r328` (s16559) are both
`%r321 + %r316`, so the stores at s16549/s16557 and s16566/s16574 target the
same two indices in program order — the later store wins.  I could not tell from
the PTX whether `%r316`'s value or the `g+8` recomputation in the second block
is meant to separate them (it looks like the source of a dropped output, but it
may equally be a degenerate case that the guard prevents).  *Missing:* either a
run of the kernel with a plane-arena/surface dump, or the corresponding
high-level code.

**U5. The geometry of P6 (M/N/K).**  E5 produces 12 A fragments and 12 B
fragments for phase 6; the A side is the 96 probabilities and the B side the 48
transposed-V registers, so phase 6 is `64×32 = (64×96)·(96×32)` with three
k-steps of 32 (verified from the mma chain at s7826–s7873: `D ← A·B + D_prev`
for exactly three steps).  `rr_layer_spec.py`'s phase table still prints
M=192 N=96 K=96 for that phase; the *epilogue* contract above does not depend on
resolving it.

**U6. The logical meaning of E22's final expression.**  The tail computes
`out = (acc − sigmoid(acc)) * 0.25` on three channels gathered from the shared
input patch.  The PTX gives the arithmetic exactly (§3/E22(e)) but not what
model term it corresponds to.  *Missing:* the high-level source (a `shrink` /
`gate` / `residual` term) or a reference implementation to compare against.

## 7. Quick acceptance index

| requirement | where |
|---|---|
| P5 epilogue: row max / exp / row sum / reciprocal / position bias | §3-E5(a)–(e); **no row max and no bias add are present — see §6 U1/U2** |
| P5: the position bias | added as the GEMM **C operand** (s3764–s3814, `W+8384+512*i`), *not* in the epilogue |
| P6: `P*V`, the transposes | §3-E5(f) (48 `movmatrix`, s7562–s7609) + E6 |
| second window P8–P11 | §3-E8, E9, E10, E11 |
| MLP P12–P20, the norm | §3-E12 |
| fc1/gelu/fc2 triples | §3-E13…E19 (activation is the clamped cubic of E13, not GELU) |
| gains/biases at 38208/40384/42496/44608/53888 | §1.1, §3-E12/E13/E14/E16/E18/E20 |
| P21/P22 merge + eight shfl | §5, §3-E20(c), E21 |
| plane-arena writes, exact value and index | §4 |
| final epilogue after L44528 and the 18 values | §3-E22 |