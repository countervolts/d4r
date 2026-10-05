# `cuda_dldn_engine_swin_dec2_kernel` — the epilogues of phases 1…38

Companion to `kernels/rr/rr_layer_spec.py dec2`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0021-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_dec2_kernel` (15796 statements).

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at their first line).  **`LNNNNN`
  is the physical line** of the same instruction in the corpus file, so every
  claim can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and
  the next phase's first `mma`.  Because phase 1 is the first GEMM, the prologue
  (s1–s451) doubles as phase 1's prelude and is described here too.
* `W` denotes the prepared weight image: `ld.param.u64 %rd1,
  [cuda_dldn_engine_swin_dec2_kernel_param_0+40]` (s7, `L1035`) and
  `cvta.to.global.u64 %rd32, %rd1` (s8, `L1036`).  Every
  `ld.weak.global.ca.v4.u32` reads `W + 16*laneid + imm`.
* The C/D register layout and the weight-addressing formula are fixed (see the
  settled notes): `C/D reg0 = (row g, cols 2t,2t+1)`, `reg1 = (row g+8, same
  cols)`, `g = laneid>>2`, `t = laneid&3`.
* Anything I could not pin down from the PTX is in §7 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `param_0` | `mov.b64 %rd7, …_param_0` | s1, L1018 |
| `W` (weight image) | `ld.param.u64 %rd1, [param_0+40]` → `cvta… %rd32, %rd1` | s7/s8, L1035–L1036 |
| extents `(ex, ey)` | `ld.param.v2.u32 {%r2147, %r2148}, [param_0+0]` | s13, L1041 |
| `%r7 = 8·ex`, `%r8 = ey·8·ex` | `shl.b32 %r7, %r2147, 3` / `mul.lo.s32 %r8, %r2148, %r7` | s14/s15, L1042–L1043 |
| block origins `%r1`, `%r2` | `sub.s32 %r1, %r2142, %r2143` / `sub.s32 %r2, %r2146, %r2144` | s6/s12, L1034/L1040 |
| `%rd33` = `param_0+8` | `ld.param.u64 %rd33, [param_0+8]` | s28, L1056 |
| **texture / skip input** | `ld.param.u64 %rd2, [%rd7+24]` = `param_0+24` | s2288, L4638 |
| **output plane arena** | `ld.param.u64 %rd571, [%rd7+48]` = `param_0+48` → `cvta… %rd4, %rd571` | s15333/s15334, L36645–L36646 |
| block-origin table | `ld.param.v2.u16 {%rs264, %rs265}, [param_0+80]` | s4, L1032 |
| window index `tid.z` | `mov.u32 %r2350, %tid.z` | s378, L1408 |
| shared arena | `.shared .align 4 .b8 _ZZ33…dec2_kernel…E4smem[9600]` | L1027 (declaration; not a statement) |
| per-lane weight address | `mul.wide.u32 %rd, %laneid, 16` + `add.s64 %rd, W, %rd` | every load block, e.g. s11444–s11445, L24247–L24248 |
| per-lane *table* address | `shl.b32 %r, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rd, %r, 12` → `add.s64 %rd, %rd32, %rd` = `W + ((laneid·4)&12)` | e.g. s12456–s12460, L27063–L27068 |

The last row matters: the `ld.global.v2.u16` / `ld.global.u32` table loads in an
epilogue read `W + ((laneid<<2)&12) + off`, i.e. **the same 16-byte slot is read
by the four lanes with equal `laneid&3`, and consecutive loads pick consecutive
4-byte words inside one 64-byte record** — so those tables are indexed by
column, not by row.

Two per-`tid.z` strides exist, both in phase 1's prelude and nowhere else:

| group | base | statements |
|---|---|---|
| phase 1's B | `W + 12288·tid.z + 16·laneid` | s379–s384, L1409–L1416 |
| phase 1's C bias | `W + 192·tid.z + ((laneid·4)&12) + 49152…49328` | s432–s439, L1512–L1521 |

`%r2350·12288` (s379) is the only `×12288` in the file and `%r2350·96 → ×2`
(s432/s433) the only `×192`, so no other phase's loads carry a window term.

### 1.2 Constants that survive between epilogues

| register | first definition | f32/f64 literal | value used (after the `cvt` to f16) |
|---|---|---|---|
| `%f476` | s7488 `mov.f32 %f476, 0f3C8CCB50` | 0.017186790704727173 | f16 `0x2466` = 0.017181396484375 |
| `%fd383` | s7492 `mov.f64 %fd383, 0dBFE1CC0000000000` | −0.55615234375 | f16 `0xB873` |
| `%fd385` | s7497 `mov.f64 %fd385, 0d3FE1CC0000000000` | +0.55615234375 | f16 `0x3873` |
| `%f478` | s7503 `mov.f32 %f478, 0f3F6D6000` | 0.92724609375 | f16 `0x3B6B` |
| `%f480` | s7508 `mov.f32 %f480, 0f3FB00000` | 1.375 | f16 `0x3D80` |
| `%fd1` | s6132 `mov.f64 %fd1, 0d3F20000000000000` | 2⁻¹³ | f16 `0x0800` = 0.0001220703125 |
| `%f861` | s12301 `mov.f32 %f861, 0f3ED306EB` | 0.4121621549129486 | f16 `0x3698` = 0.412109375 |
| `%f862` | s12304 `mov.f32 %f862, 0f3DA60DD6` | 0.0810810774564743 | f16 `0x2D30` = 0.0810546875 |
| `%f863` | s12307 `mov.f32 %f863, 0f3F000000` | 0.5 | f16 `0x3800` |
| `%f864` | s12310 `mov.f32 %f864, 0f40000000` | 2.0 | f16 `0x4000` |

`%f861…%f864` are materialised **inside E14** (s12301–s12312) and only re-`cvt`
by E16, E18, E20, …, E36 (`s12642 cvt.rn.f16.f32 low, %f861`, …).  `%fd1` is
defined once, in E1 (s6132), and only re-`cvt` by E13.

Also live across the region, as in the sibling kernel: `%r5908 = 1`
(s5595, `L8140`) and `%r5919 = -1` (s6074, `L9586`) supply the butterfly delta
and the membermask of every `shfl.sync.bfly.b32` in the epilogues
(`s6075 shfl.sync.bfly.b32 %r5842,%r5838,%r5908,%r5845,%r5919`); the
group/width operand is per-instruction.

### 1.3 Two quantisation helpers

* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).
* `cvt.rn.f16x2.e4m3x2` — its inverse, the *unpack*, used only when a shared
  tile that was stored e4m3-packed is read back (E1: 96 of them, s5594–s5690,
  L8139–L8437).
* `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `div.approx.ftz.f32` — always wrapped as
  `cvt.f32.f16 → op → cvt.rn.f16.f32` (round each step through f16).

## 2. Summary

| epilogue | range | stmts | named operation |
|---|---|---|---|
| E1 | s596–s7088 | 6493 | patch expand into shared + per-column RMS norm + gain + restage |
| E2 | s7113–s7119 | 7 | next-B weight loads (P3) |
| E3 | s7144–s7150 | 7 | next-B weight loads (P4) |
| E4 | s7175–s7181 | 7 | next-B weight loads (P5) |
| E5 | s7206–s7212 | 7 | next-B weight loads (P6) |
| E6 | s7237–s7243 | 7 | next-B weight loads (P7) |
| E7 | s7268–s7438 | 171 | score-bias C-seed loads (24×512 B) + 80 packs → P8's A and B |
| E8 | s7487–s11329 | 3843 | cubic-exponent softmax, V transpose, pack |
| E9 | s11378–s11546 | 169 | pack P9's D → shared, per-column table → P10's C, next B |
| E10 | s11559–s11577 | 19 | 1 shared A fragment + 6 next-B loads (P11) |
| E11 | s11590–s11608 | 19 | 1 shared A fragment + 6 next-B loads (P12) |
| E12 | s11621–s11639 | 19 | 1 shared A fragment + 6 next-B loads (P13) |
| E13 | s11652–s12287 | 636 | RMS norm of P13's D + gain, packs → P14's A and P16's A, next B/C |
| E14 | s12300–s12556 | 257 | clamped-cubic activation of P14's D, per-column table + residual → P15's C |
| E15 | s12569–s12628 | 60 | 24 packs → P16's A, next B, P16's C bias |
| E16 | s12641–s12804 | 164 | clamped-cubic activation of P16's D → P17's A, next B |
| E17 | s12817–s12876 | 60 | 24 packs → P18's A, next B, P18's C bias |
| E18 | s12889–s13052 | 164 | activation of P18's D → P19's A, next B |
| E19 | s13065–s13124 | 60 | 24 packs → P20's A, next B, P20's C bias |
| E20 | s13137–s13300 | 164 | activation of P20's D → P21's A, next B |
| E21 | s13313–s13372 | 60 | 24 packs → P22's A, next B, P22's C bias |
| E22 | s13385–s13548 | 164 | activation of P22's D → P23's A, next B |
| E23 | s13561–s13620 | 60 | 24 packs → P24's A, next B, P24's C bias |
| E24 | s13633–s13796 | 164 | activation of P24's D → P25's A, next B |
| E25 | s13809–s13868 | 60 | 24 packs → P26's A, next B, P26's C bias |
| E26 | s13881–s14044 | 164 | activation of P26's D → P27's A, next B |
| E27 | s14057–s14116 | 60 | 24 packs → P28's A, next B, P28's C bias |
| E28 | s14129–s14292 | 164 | activation of P28's D → P29's A, next B |
| E29 | s14305–s14364 | 60 | 24 packs → P30's A, next B, P30's C bias |
| E30 | s14377–s14540 | 164 | activation of P30's D → P31's A, next B |
| E31 | s14553–s14612 | 60 | 24 packs → P32's A, next B, P32's C bias |
| E32 | s14625–s14788 | 164 | activation of P32's D → P33's A, next B |
| E33 | s14801–s14860 | 60 | 24 packs → P34's A, next B, P34's C bias |
| E34 | s14873–s15036 | 164 | activation of P34's D → P35's A, next B |
| E35 | s15049–s15108 | 60 | 24 packs → P36's A, next B, P36's C bias |
| E36 | s15121–s15284 | 164 | activation of P36's D → P37's A, next B |
| E37 | s15297–s15504 | 208 | pack → 12 guarded plane-arena scatter → B/C for P38 |
| E38 | s15511–s15796 | 286 | 4 D stores → shared, row max, softmax, 9+2 texture reads, surface store |

---

## 3. The epilogues

### E1 — s596–s7088: patch expand, RMS norm, gain, restage

6493 statements; op histogram (top): 1103 `add.s32`, 477 `shl.b32`, 446
`mov.b32`, 413 `$label`, 409 `sub.s32`, 355 `min.u32`, 288 `mul.f16`, 120
`cvt.rn.satfinite.e4m3x2.f16x2`, 96 `cvt.rn.f16x2.e4m3x2`, 96
`rsqrt.approx.ftz.f32`, 42 `st.shared.*` (34 `u32` + 2 `v2.u16` + 6 `v4.u32`),
42 `ld.shared.*` (24 `v2.u16` + 18 `v4.u32`), 36 `ld.global.v2.u16`, 8
`shfl.sync.bfly.b32`, 1 `ld.param`, 2 `ld.weak…`.

Sub-sections (bounded by the first/last statement of each op):

| range | what |
|---|---|
| s596–s702 | the head packs of phase 1's D (72 `cvt.rn.satfinite…`), → `%r9…%r41` |
| s716–s1478 | per-lane index arithmetic + the 36 guarded patch stores (34 `st.shared.u32` s770–s1460 + 2 `st.shared.v2.u16` s743/s1478); 1st `bar.sync` at s1480 |
| s1524–s2287 | 24 `ld.shared.v2.u16` read back of the packed patch |
| s2288 | `ld.param.u64 %rd2, [%rd7+24]`: **the patch-gather buffer** |
| s2444–s5593 | 24 clamped `ld.global.v2.u16` gathered from that buffer |
| s5594–s5690 | 96 `cvt.rn.f16x2.e4m3x2` unpacks of the read-back patch |
| s5693–s6276 | the sum trees (196 `add.f16` exact + 52 `add.f16x2`) |
| s6132 | the 2⁻¹³ constant `%fd1` (s6132 `mov.f64 %fd1, 0d3F20000000000000`) |
| s6075–s6128 | the 8 `shfl.sync.bfly.b32` (first at s6075) |
| s6282–s6659 | 96 `rsqrt.approx.ftz.f32`, each `cvt.f32.f16 → rsqrt → cvt.rn.f16.f32` |
| s6668–s6679 | **gain loads** `W + ((laneid·4)&12) + {49920…50096}` (12 `ld.global.v2.u16`) |
| s5693–s6966 | the 288 `mul.f16` (two groups: `inv · gain` then `x · (inv·gain)`) |
| s7008–s7054 | 24 packs → the 6 `st.shared.v4.u32` at s7008/7021/7034/7044/7049/7054 |
| s7059–s7082 | `bar.sync` (s7059) + 18 `ld.shared.v4.u32` (s7064–s7082) → phases 2…7's A |
| s7083–s7088 | 2 `ld.weak` (s7086/s7088) → **phase 2's B** |

Evidence for the patch scatter (36 stores):

```
s743  L2836: st.shared.v2.u16 [%r2373], {%rs192, %rs193}
s770  L2865: st.shared.u32 [%r2386], %r9
s800  L2897: st.shared.u32 [%r2403], %r10
s1478 L3641: st.shared.v2.u16 [%r2787], {%rs262, %rs263}
s7008 L11717: st.shared.v4.u32 [%r6091], {%r6095, %r6094, %r6093, %r6092}
s7044 L11788: st.shared.v4.u32 [%r6110+6144], {%r6114, %r6113, %r6112, %r6111}
s7054 L11798: st.shared.v4.u32 [%r6110+7168], {%r6122, %r6121, %r6120, %r6119}
```

Each patch store is guarded exactly as in the sibling kernel
(`setp.gt.u32 %p…, %r…, 9` / `or.pred` / `@%p bra …`; the same
`(row·40 + col·8 + t)·4` index form — cf. `rrswin_dec1_epilogues.md` §3/E1,
which carries the same 400-block emitter).  The second staging area is at
`[%r6091]`, `[%r6091+512]`, `[%r6091+1024]`, `[%r6110+6144]`,
`[%r6110+6656]`, `[%r6110+7168]`, where `%r6091 = smem + tid.z·2048`
(s6994 `shl.b32 … %r2350, 11`) and `%r6110 = smem + tid.z·3072` (s7040).

The norm:

```
s6132 L9664:  mov.f64 %fd1, 0d3F20000000000000        // 2^-13
s5693 L8430:  add.f16 %rs456,%rs457,%rs458
s6282 L10110: rsqrt.approx.ftz.f32 fl, fl
s6668 L10735: ld.global.v2.u16 {%rs1959, %rs1960}, [%rd247+49936]
s6681 L10749: mul.f16 %rs…, %rs…, %rs…                // inv * gain
```

**There is no mean subtraction and no `1/N`.**  The accumulated quantity is the
raw sum of squares of the 3 patch channels plus the constant 2⁻¹³, and the norm
is applied per *column* (the gain is read at `W + ((laneid<<2)&12) + off`, so it
is constant across rows).

**Inputs:** phase 1's D (`%r593…%r826`, 12 fragments — phase 1 has `nA = 12`),
`W`, `param_0+8`, `param_0+24`, `%r1/%r2/%r7/%r8/%r2147/%r2148`, `%r2350`
(`tid.z`).
**Outputs (live past s7088):** the patch-expand `st.shared` regions, the 18
`ld.shared.v4.u32` A fragments `%r18995…%r19066`, phase 2's B
`%r6125…%r6132`, the 4 constants of §1.2 (`%fd1`, `%f861…%f864` are *not* yet
defined here).

### E2…E6 — s7113–s7243: next-B loads

Five 7-statement blocks of identical shape:

```
s7113 L12005: mov.u32 %r6373, %laneid
s7114 L12007: mul.wide.u32 %rd…, %r6373, 16
s7115 L12008: add.s64 %rd…, %rd2?, %rd…
s7117 L12017: ld.weak.global.ca.v4.u32 { %r6374,…%r6377},[%rd250]
s7119 L12021: ld.weak.global.ca.v4.u32 { %r6378,…%r6381},[%rd251]
```

| epilogue | weight block | destination phase |
|---|---|---|
| E2 | 50112…50636 | P3's B |
| E3 | 53184…53708 | P4's B |
| E4 | 51136…51660 | P5's B |
| E5 | 54208…54732 | P6's B |
| E6 | 52160…52684 | P7's B |

The weight sub-image is laid out **P2, P4, P6, P3, P5, P7** (blocks of 1024
bytes at `W + 50112 + 1024·k`), not in phase order.

### E7 — s7268–s7438: score-bias C-seed loads + packs → P8's A and B

171 statements, no `ld.global` outside the two groups below.

* **24 loads** `W + 16·laneid + 56256 + 512·i`, `i = 0…23`, at s7272…s7318
  (L12922…L13014) into `%r7619…%r7714`; the last block's `add.s64` is
  `s7317 add.s64 %rd283, %rd467, 17920`.  Byte block
  `[56256, 68544)` = 24 × 512 = 12288 bytes — **phase 8's C score-bias seed**
  (the phase table prints the last *slot* offset, 68044).
* **80 packs / 40 b32**, s7319–s7438, whose sources are registers written by
  **phases 3…7's mma** (`%r7129/%r7149/…`, `%r7229/…`, `%r7360…`).  The two
  destinations are `%r8161…%r8272` (**phase 8's B**) and `%r7907…%r8030`
  (**phase 8's A**), e.g.

```
s7320 L13020: cvt.rn.satfinite.e4m3x2.f16x2 %rs1984, %r7149
s7321 L13022: mov.b32 %r8161, {%rs1983, %rs1984}
s7391 L13185: cvt.rn.satfinite.e4m3x2.f16x2 %rs2032, %r7149
s7393 L13190: mov.b32 %r7907, {%rs2031, %rs2032}
```

`s7439` then consumes them: `mma … {A=%r7907,%r7908,%r7909,%r7910},
{B=%r8161,%r8162}, {C=%r7619,%r7620}`.  `%r7149` is phase 3's first D
(s7089–s7112) and `%r7129`/`%r7360` likewise belong to phases 3/4/6/7, so both
of phase 8's operands are packs of the six preceding projections.

### E8 — s7487–s11329: cubic-exponent softmax, V transpose, pack

**Byte-identical in kind to `cuda_dldn_engine_swin_enc0_kernel`'s E5** (see
`rrswin_enc0_epilogues.md` §3/E5): 1048 `mov.b32`, 672 `cvt.rn.f16.f32`, 384
`cvt.rn.f16.f64`, 192 each of `fma.rn.f16x2` / `cvt.f32.f16` /
`rcp.approx.ftz.f32` / `mul.f16`, 144 packs, 48 `movmatrix`, 16
`shfl.sync.bfly.b32`, 96 each of `max/min/neg/abs.f16x2`.

Sub-sections:

| range | what |
|---|---|
| s7488–s7513 | the 5 constants (§1.2 of the sibling file): s7488 `0f3C8CCB50`, s7492/s7497 the f64 clamps, s7503/s7508 the cubic pair |
| s7514–s9604 | 96 per-element groups (scale, clamp, cubic, `and.b32 …,2145419232`) |
| s9605–s9690 | in-lane partial sums (`add.f16x2`) |
| s9691–s9820 | 16 `shfl.sync.bfly` → 8 row sums |
| s9821–s10582 | 192 `rcp.approx.ftz.f32` → 96 reciprocals |
| s10583–s11065 | 192 `mul.f16` → 96 probabilities |
| s11066–s11113 | 48 `movmatrix.sync.trans.aligned.m8n8.b16` of **phase 7's** D |
| s11114–s11328 | 144 packs |

Evidence:

```
s7514 L13706: and.b32 %r…, %r…, 2145419232               // 0x7FE07FE0
s9691 L20503: shfl.sync.bfly.b32 %r10441,%r10437,%r5908,%r10444,%r5919
s9821 L20829: rcp.approx.ftz.f32 fl, fl
s11066 L23128: movmatrix.sync.trans.aligned.m8n8.b16 %r10801, %r7378
s11114 L23272: cvt.rn.satfinite.e4m3x2.f16x2 %rs3432, %r10809
```

`%r10801` is the first movmatrix result and its source `%r7378` is phase 7's
first D (s7244–s7267); the movmatrix sources run over phase 7's 48 D registers.
The transposed results become **phase 9's B** (`%r11407 ← mov.b32
{%rs3431,%rs3432}` at s11116 after the pack at s11114), and the probability packs
(`%r11113 ← mov.b32 {%rs3479,%rs3480}` at s11188) become **phase 9's A**.

The per-element function is the same cubic-exponent approximation as enc0's E5
(no row maximum, no bias add):

```
m  = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)
u  = f16(1.375 + m*(0.92724609375 - m*m))
expval = f16 from bits  ((u_bits << 5) & 0x7FE07FE0)
```

**Memory:** none (0 `ld.*`, 0 `st.*` in 3843 statements).

### E9 — s11378–s11546: pack P9's D → shared, per-column table → P10's C

169 statements: 32 packs, 8 `add.f16`, 6 `ld.weak`, 12 `ld.global.v2.u16`,
4 `st.shared.v4.u32`, 1 `ld.shared.v4.u32`.

```
s11378 L24111: bar.sync 0
s11384 L24120: cvt.rn.satfinite.e4m3x2.f16x2 %rs3576, %r11141
s11396 L24147: st.shared.v4.u32 [%r19198], {%r19202, %r19201, %r19200, %r19199}
s11435 L24234: st.shared.v4.u32 [%r19198+1536], {%r19214, %r19213, %r19212, %r19211}
s11436 L24235: bar.sync 0
s11442 L24243: ld.shared.v4.u32 {%r19219,…%r19222}, [%r19218]
s11446 L24249: add.s64 %rd284, %rd469, 68544
s11451 L24259: add.s64 %rd286, %rd469, 69568
s11463 L24280: ld.global.v2.u16 {%rs4886, %rs4887}, [%rd473+136144]
s11475 L24293: add.f16 %rs3610,%rs4909,%rs459
s11477 L24300: mov.b32 %r11589, {%rs3607, %rs3610}
```

* **32 packs of phase 9's D** → 16 b32 → four `st.shared.v4.u32` at
  `[%r19198 + {0,512,1024,1536}]`; the `bar.sync` pair makes them visible, then
  **one `ld.shared.v4.u32` gives the first A fragment of phase 10**.
* **6 `ld.weak` at `W + 16·laneid + {68544, 69056, 69568, 70080, 70592, 71104}`**
  → phase 10's B.
* **12 `ld.global.v2.u16` at `W + ((laneid·4)&12) + {136128…136304}`** — a
  per-column vector (12 words per lane = 96 columns).
* **48 `add.f16`** adding that vector to register pairs, producing the 24
  registers `%r11589 %r11590 %r11599 %r11600 …` = **phase 10's C**
  (confirmed at s11547: `C = {%r11589, %r11590}`).

**Which registers the table is added to is ambiguous in this file.**  The
operands are `%rs456/%rs459` and their 22 siblings, and the *only* textual
definition of `%rs456/%rs459` is at `s5693/s5694` (L8430/L8434) inside **E1**
(the E1 sum-of-squares tree); nothing between s5694 and s11475 writes them.  So
either E9's residual is genuinely carried from E1, or the printed PTX reuses a
register name across two disjoint live ranges without a rename.  See §7/U1.

### E10/E11/E12 — s11559–s11639: one shared A fragment + next-B loads

Three 19-statement blocks:

```
s11562 L24597: ld.shared.v4.u32 {%r19226,…%r19229}, [%r19225+2048]
s11566 L24603: add.s64 %rd290, %rd475, 90048
s11571 L24613: add.s64 %rd292, %rd475, 91072
```

| epilogue | shared A offset | weight block | destination phase |
|---|---|---|---|
| E10 | `+2048` | 90048…92620 | P11's A, B |
| E11 | `+4096` | 111552…114124 | P12's A, B |
| E12 | `+6144` | 133056…135628 | P13's A, B |

Each takes its single A fragment from the same shared staging region E9 wrote
(offsets 2048/4096/6144 relative to `%r19216`) and reads 6 consecutive 512-byte
weight fragments.  All three phases take `C = 0`.

### E13 — s11652–s12287: RMS norm of P13's D + gain

636 statements; 182 `mov.b32`, 144 `mul.f16`, 76 `add.f16`, 26 `add.f16x2`,
48 `rsqrt.approx.ftz.f32`, 4 `shfl.sync.bfly`, 24 packs, 12 `ld.global.v2.u16`.

| range | op |
|---|---|
| s11654–s11749 | unpack P13's D halves (`%r12019…`) and square them (48 `mul.f16`, s11655–s11749) |
| s11750–s11804 | in-lane `add.f16x2` partial sums |
| s11776–s11794 | 4 `shfl.sync.bfly.b32` |
| s11875–s11878 | `add.f16` of the halves, then `+ 2⁻¹³` (`%fd1`, defined at s6132) |
| s11883–s12068 | 48 `rsqrt.approx.ftz.f32` |
| s12072–s12083 | **gain loads** `W + ((laneid·4)&12) + {136336, 136352, …, 136496, 136320}` (12 `ld.global.v2.u16`) |
| s12085–s12226 | 96 `mul.f16` — `inv · gain` then `x · (inv·gain)` |
| s12231–s12242 | next-B loads `W + {136512, 137024, 137536, 138048, 138560, 139072}` → phase 14's B |
| s12248–s12251 | `ld.global.u32` at `W + ((laneid·4)&12) + {139584, 139600, 139616, 139632}` → **phase 14's C bias** |
| s12252–s12287 | 24 packs → `%r12414…%r12457`, **phase 14's A** |

At s12157/s12158 the same epilogue also emits a *second* scaled copy of P13's D:

```
s12157 L26127: mul.f16 %rs4190,%rs4360,%rs4046
s12158 L26130: mov.b32 %r18435, {%rs4190, %rs4193}
```

`%rs4360` is the unpack of P13's first D (s11654).  The 24 registers
`%r18435…%r18458` are re-used by **every** even phase's A pack (E15 s12594,
E17 s12842, E19 …, and E37 s15297's neighbours), i.e. **phases 14, 16, 18, …,
P36 all share one A operand** derived from phase 13.

### E14 — s12300–s12556: activation of P14's D, table + residual → P15's C

257 statements.  Three things happen:

1. **s12301–s12440: the clamped-cubic activation** on the 8 f16x2 registers
   `%r12382 %r12392 %r12383 %r12393 %r12442 %r12452 %r12443 %r12453` (phase 14's
   D).  This is the epilogue that materialises `%f861…%f864` (s12301–s12312):

```
s12313 L26578: neg.f16x2 %r12467,%r12466
s12314 L26582: max.f16x2 %r12469,%r12382,%r12467
s12315 L26586: min.f16x2 %r12472,%r12469,%r12466
s12316 L26590: abs.f16x2 %r12475,%r12472
s12317 L26594: mul.f16x2 %r12477,%r12464,%r12475
s12318 L26598: sub.f16x2 %r12480,%r12463,%r12477
s12319 L26602: mul.f16x2 %r12483,%r12472,%r12480
s12320 L26606: add.f16x2 %r12486,%r12465,%r12483
s12321 L26610: mul.f16x2 %r12489,%r12382,%r12486
```

   Per element: `y = clamp(x,−2,+2)`, `g = 0.5 + y·(0.412109375 − 0.0810546875·|y|)`,
   `out = x·g` — the same cubic ramp as the sibling kernel's E10.  **Reproduce
   the formula, not SiLU/GELU.**
2. **s12441–s12472: next-B and table loads.**  Six `ld.weak` at
   `W + {139648, 140160, 140672, 141184, 141696, 142208}` → phase 15's B; and
   12 `ld.global.v2.u16` at `W + ((laneid·4)&12) + {142720…142896}`.
3. **s12473–s12556: the residual/table add and the packs.**  48 `add.f16`
   (`add.f16 %rs4361,%rs4957,%rs4363` at s12473) combine the table with the 24
   registers whose halves come from `s11654 L24951: mov.b32 {%rs4360, %rs4363},
   %r12019` — **phase 13's D**.  The 24 results `%r12737…%r12848` are
   **phase 15's C**; the 8 packs at s12545–s12556 give **phase 15's A**.

So phase 15 computes `D15 = act(D14)·B15 + (D13 + table@142720)`.

### E15 — s12569–s12628: packs of the shared A, next B, P16's C bias

60 statements:

```
s12572 L27414: add.s64 %rd320, %rd494, 142912
s12589 L27445: ld.global.u32 %r12908, [%rd497+145984]
s12593 L27450: cvt.rn.satfinite.e4m3x2.f16x2 %rs4511, %r18436
s12594 L27453: cvt.rn.satfinite.e4m3x2.f16x2 %rs4510, %r18435
```

* six `ld.weak` at `W + {142912, 143424, 143936, 144448, 144960, 145472}`
  → phase 16's B;
* four `ld.global.u32` at `W + ((laneid·4)&12) + {145984, 146000, 146016,
  146032}` → **phase 16's C bias**;
* **24 packs of `%r18435…%r18458`** — the *shared* A that E13 produced — so
  phase 16's A is bit-identical to phase 14's… (checked: `%r12971 ← mov.b32
  {%rs4510, %rs4511}` at s12595 and `%r12971` is phase 16's first A register;
  contrast phase 14's first A register `%r12414 ← mov.b32 {%rs4334, %rs4335}` at
  s12254, sourced from `%r18435` at s12253 — the same register, so yes,
  **identical**).

### E16/E18/…/E36 — the 164-statement activation epilogues

Twelve shape-identical 164-statement epilogues (shape hash `706fcc703f68`):
36 `mov.b32`, 32 `cvt.rn.f16.f32`, 24 `mul.f16x2`, 8 each
`neg/max/min/abs/sub/add.f16x2`, 6 `ld.weak`, no `ld.global`, no constants of
their own (they re-`cvt` `%f861…%f864`).

Each applies the E14 activation to its own phase's D and packs 8 b32 → the next
odd phase's A, and reads 6 weight fragments → the next odd phase's B.
Evidence (E16):

```
s12642 L27622: cvt.rn.f16.f32 low, %f861
s12651 L27649: max.f16x2 %r13026,%r12939,%r13024      // phase 16's D
s12658 L27677: mul.f16x2 %r13046,%r12939,%r13043
s12782 L28107: ld.weak.global.ca.v4.u32 { %r13253,…},[%rd326]   // W+146048
s12795 L28135: mov.b32 %r13397, {%rs4534, %rs4535}    // phase 17's A
```

**No table add and no residual**: these epilogues produce only A and B.

### E17/E19/…/E37-pre — the 60-statement pack epilogues

Eleven shape-identical 60-statement epilogues (shape hash `a7b578456a5a`):
24 packs, 12 `mov.b32`, 6 `ld.weak`, 4 `ld.global.u32`.

```
s12594 L27453: cvt.rn.satfinite.e4m3x2.f16x2 %rs4510, %r18435
s12842 L28285: cvt.rn.satfinite.e4m3x2.f16x2 %rs4542, %r18435
```

Each packs the same 24 registers `%r18435…%r18458` (so every even phase
P16, P18, …, P36 gets the identical A operand), reads 6 weight fragments → the
next even phase's B, and 4 `ld.global.u32` → the next even phase's C bias
(E15→P16@145984, E17→P18@152192, E19→P20@158400, E21→P22@164608,
E23→P24@170816, E25→P26@177024, E27→P28@183232, E29→P30@189440,
E31→P32@195648, E33→P34@201856, E35→P36@208064).

#### Shape of the whole MLP region (E13–E37)

```
E13  normed tile  → P14's A (s12252)      ; shared tile T = %r18435… (s12158)
P14: A = f(D13),  B = W14, C = bias@139584  → D14   (48×96×96)
E14: act(D14) → P15's A ; P15's C = D13 + table@142720
P15: A = act(D14), B = W15, C = D13+table   → D15   (16×96×32)
E15: pack T → P16's A ; B = W16 ; P16's C = bias@145984
P16: A = pack(T),  B = W16, C = bias@145984 → D16   (48×96×96)
E16: act(D16) → P17's A ; B = W17
P17: A = act(D16), B = W17, C = D15         → D17   (16×96×32)
E17: pack T → P18's A ; B = W18 ; P18's C = bias@152192
P18: A = pack(T),  B = W18, C = bias        → D18
E18: act(D18) → P19's A ; P19's C = D17
 …   (the same two-phase motif repeats with the bias offsets of §2 until)
P37: A = act(D36), B = W37, C = D35         → D37
E37: pack D37 → 12 plane-arena stores ; P38: B = W38, C = bias@212736
P38: A = those packs, B = W38, C = bias     → D38   (48×48×96)
```

The claim "odd phase's C = previous odd phase's D" is machine-checked for every
odd phase from P17 on (P15 is the exception, whose C is the register pair E14
builds).  The claim "even phase's C comes from a `ld.global.u32`" holds for
P14…P38.

### E37 — s15297–s15504: pack → 12 guarded plane-arena stores → B/C for P38

208 statements; the only `st.global` in the entry (12 of them, all here).

**(a) requantise, s15297–s15332.**  12 groups of `2 cvt + 1 mov.b32` → the 12
b32 `%r481…%r492` (phase 37's D).

**(b) plane-arena scatter, s15333–s15482.**

```
s15333 L36645: ld.param.u64 %rd571, [%rd7+48]        // = param_0+48
s15336 L36650: shr.u32 %r493, %r18989, 2             // g = laneid>>2
s15337 L36651: and.b32 %r494, %r18989, 3             // t = laneid&3
s15340 L36654: add.s32 %r19271, %r480, %r19269       // y = %r480 + (laneid>>5)
s15346 L36660: add.s32 %r19277, %r1, %r19270         // x = %r1 + (g&7)
s15351 L36665: shl.b32 %r19278, %r19277, 3           // x<<3
s15352 L36666: mad.lo.s32 %r495, %r7, %r19271, %r19278
s15354 L36668: selp.b32 %r496, -1, %r19279, %p1
s15359 L36674: st.global.u32 [%rd573], %r481
```

Stride register `%r7 = 8·ex` (s14).  Each store is `idx = 8·ex·y + 8·x + d`,
`d = t` or `t|4`, guarded by
`(y<0)|(y≥ey)|(laneid>63)|(x<0)|(x≥ex)` (s15341–s15350, L36655–L36664); a guard
failure stores `-1` as the index, which the following `setp.lt.s32` + `bra`
turns into "skip the store".

Store values, bases and column offsets (`g = laneid>>2`, `t = laneid&3`,
`%r480 = %r2 + 2·tid.z` at s11653/L24950, `%r8 = ey·8·ex`):

| store | stmt | value | index base | d |
|---|---|---|---|---|
| 1 | s15359 | `%r481` | `idx1` | `t` |
| 2 | s15368 | `%r482` | `idx1` | `t\|4` |
| 3 | s15401 | `%r483` | `idx2` | `t` |
| 4 | s15409 | `%r484` | `idx2` | `t\|4` |
| 5 | s15419 | `%r485` | `idx1 + %r8` | `t` |
| 6 | s15427 | `%r486` | `idx1 + %r8` | `t\|4` |
| 7 | s15437 | `%r487` | `idx2 + %r8` | `t` |
| 8 | s15445 | `%r488` | `idx2 + %r8` | `t\|4` |
| 9 | s15455 | `%r489` | `idx1 + 2·%r8` | `t` |
| 10 | s15463 | `%r490` | `idx1 + 2·%r8` | `t\|4` |
| 11 | s15473 | `%r491` | `idx2 + 2·%r8` | `t` |
| 12 | s15481 | `%r492` | `idx2 + 2·%r8` | `t\|4` |

with the two bases built as `mad.lo.s32 %rBASE, 8·ex, y, 8·x`:

| base | stmt | y | x |
|---|---|---|---|
| `idx1 = %r495` | s15352, L36666 | `%r480 + (laneid>>5)` | `%r1 + (g&7)` |
| `idx2 = %r499` | s15394, L36712 | `%r19505 + ((g+8)>>3)` | `%r1 + ((g+8)&7)` |

`%r19505` is a *second* row origin:
`s15370 mov.u32 %r19511, %ctaid.y` / `s15372 shl.b32 %r19509, %r19511, 3` /
`s15375 sub.s32 %r19506, %r19509, %r19510` / `s15376 add.s32 %r19505, %r19506,
%r19507` with `%r19507 = 2·tid.z` (s15373/s15374), i.e.
`ctaid.y·8 − origin_y + 2·tid.z` — so the two halves of this scatter start at
different rows.  The `+%r8` (s15412 `mad.lo.s32 %r502, %r2148, %r19493, %r495`)
and `+2·%r8` (s15466 `mad.lo.s32 %r511, %r2148, %r19496, %r505`) forms then step
one and two rows further.

**(c) next B/C, s15483–s15504.**

```
s15492 L36832: add.s64 %rd596, %rd600, 211200
s15493 L36834: ld.weak.global.ca.v4.u32 { %r19307,…%r19310},[%rd596]
s15497 L36842: ld.weak.global.ca.v4.u32 { %r19315,…%r19318},[%rd598]
s15503 L36851: ld.global.u32 %r19329, [%rd604+212736]
s15504 L36852: ld.global.u32 %r19339, [%rd604+212752]
```

Three weight fragments at `W + {211200, 211712, 212224}` → phase 38's B, two
`u32` at `W + ((laneid&3)·4 &12) + {212736, 212752}` → phase 38's C bias.

### E38 — s15511–s15796: the final epilogue

286 statements; it consumes phase 38's four D accumulators (`%r19360 %r19370
%r19497 %r19498`) and writes the surface.

**(a) staging, s15511–s15542.**

```
s15511 L36895: bar.sync 0
s15522 L36908: st.shared.u32 [%r520], %r19360
s15526 L36913: st.shared.u32 [%r520+16], %r19370
s15539 L36927: st.shared.u32 [%r522], %r19361
s15541 L36930: st.shared.u32 [%r522+16], %r19371
```

Four guarded stores of the phase-38 D pairs into shared, indexed from
`%r518 = laneid>>2`, `%r519 = laneid&3` and `%tid.z`.

**(b) coordinates and bounds check, s15543–s15570.**

```
s15551 L36941: mov.u32 %r19403, %tid.x
s15559 L36949: add.s32 %r525, %r1, %r523             // x
s15560 L36950: add.s32 %r526, %r19512, %r524         // y
s15565 L36955: ld.param.v2.u32 {%r19418, %r19419}, [%rd634]
s15570 L36960: @%p412 bra $L__BB1_486
```

**(c) row max over 5 shared values, s15571–s15591.**

```
s15577 L36968: ld.shared.u32 %r19421, [%r19470]
s15583 L36974: ld.shared.u32 %r19442, [%r19470+16]
s15585 L36977: max.f16x2 %r19420,%r19421,%r19422
s15590 L36994: max.f16 %rs4961,%rs4958,%rs4963
s15592 L36999: sub.f16x2 %r19429,%r19421,%r19443
```

Five `u32` (= 10 f16) per row, reduced in registers — **no warp shuffle**.

**(d) softmax, s15592–s15676.**

```
s15595 L37007: mul.ftz.f32 %f905, %f865, 0f3FB8AA3B   // * log2e
s15596 L37008: ex2.approx.ftz.f32 %f867, %f905
s15651 L37126: add.ftz.f32 %f922, %f920, %f882        // running row sum
s15653 L37128: div.approx.ftz.f32 %f883, %f904, %f922 // 1/rowsum, %f904 = 1.0
```

`y = 2^((x−max)·log2e)`, `p = y·(1/Σy)` in f16, five f16x2 registers.

**(e) 3×3 colour texture, s15677–s15758.**

```
s15677 L37176: ld.param.v2.f32 {%f924, %f925}, [%rd635+136]
s15683 L37182: ld.param.u64 %rd622, [%rd635+88]
s15685 L37185: tex.base.2d.v4.f16.f32 {%rs4986,%rs4987,%rs4988,%rs4989}, [%rd606, {%f884,%f885}]
s15687 L37189: fma.rn.f16 %rs4990,%rs4986,%rs5000,%rs4983
```

`%rd635 = param_0` (s15571), so the colour texture handle is **`param_0+88`**
and the coordinate scales are **`param_0+136`**.  Nine
`tex.base.2d.v4.f16.f32` (s15685, 15697, 15709, 15721, 15726, 15732, 15744,
15750, 15755) cover the 3×3 neighbourhood of clamped `(x−1, x, x+1) ×
(y−1, y, y+1)`, accumulated into 3 channels with `fma.rn.f16`.

**(f) the two integer-coordinate textures, s15759–s15762.**

```
s15759 L37357: ld.param.u64 %rd624, [%rd635+96]
s15760 L37359: tex.base.2d.v4.f16.s32 {%rs5130,%rs5131,%rs5132,%rs5133}, [%rd624, {%r525,%r526}]
s15761 L37361: ld.param.u64 %rd626, [%rd635+104]
s15762 L37363: tex.base.2d.v4.f16.s32 {%rs5134,%rs5135,%rs5136,%rs5137}, [%rd626, {%r525,%r526}]
```

**These are the `+96` / `+104` texture reads the brief asks about** — handles at
`param_0+96` and `param_0+104`, sampled at the *integer* pixel coordinate
`(%r525, %r526)`.

**(g) combine, sigmoid gate, surface store, s15763–s15796.**

```
s15763 L37366: cvt.f32.f16 %f902, %rs5138
s15764 L37369: mul.ftz.f32 %f935, %f902, 0fBFB8AA3B
s15765 L37370: ex2.approx.ftz.f32 %f936, %f935
s15766 L37371: add.ftz.f32 %f937, %f936, 0f3F800000
s15767 L37372: div.approx.ftz.f32 %f903, %f904, %f937    // 1/(1+2^(-log2e·v))
s15770 L37382: sub.f16 %rs5141,%rs5140,%rs5139           // 1 - sigmoid
s15771 L37386: mul.f16 %rs5144,%rs5118,%rs5139           // acc * sigmoid
s15772 L37390: fma.rn.f16 %rs5187,%rs5130,%rs5141,%rs5144// + tex96 * (1-sig)
s15777 L37410: set.nan.f16.f16 %rs5165,%rs5134,%rs5134   // + tex104 unless NaN
s15793 L37449: mov.u16 %rs5186, 0
s15794 L37451: sust.b.2d.v4.b16.zero [%rd629, {%r19489,%r526}], {%rs5187,%rs5188,%rs5189,%rs5186}
```

`%rd629 = param_0+112` (s15582).  The store column is `%r525·8`
(s15792 `shl.b32 %r19489, %r525, 3`), the row `%r526`; the fourth channel is
the zero literal, so this is a **3-channel write into a 4-channel surface**.
The `set.nan`/`bra` triad (s15777–s15790) adds the `+104` read only when it is
not NaN.

**Inputs:** the 4 returned D registers, `param_0+{48,88,96,104,112,136}`, the
shared scratch, constants `0f3FB8AA3B`, `0fBFB8AA3B`, `0f3F800000`.
**Outputs:** none (the kernel returns, s15796).

---

## 4. Texture reads, surface writes, and the `+24` input

| what | where |
|---|---|
| `param_0+8` — input feature plane arena, read by phase 1's A loads | prologue s66…s451 (60 `ld.global`) |
| `param_0+24` — the patch-gather source E1 reads | s2288 (`ld.param.u64 %rd2, [%rd7+24]`) and the 24 `ld.global.v2.u16` at s2444…s5593 |
| `param_0+48` — the plane arena E37 scatters phase 37's output to | s15333, 12 `st.global.u32` at s15359…s15481 |
| `param_0+88` — the colour texture handle | s15683, 9 `tex.base.2d.v4.f16.f32` at s15685…s15755 |
| `param_0+96` — texture handle #2 (integer coords) | s15759/s15760 |
| `param_0+104` — texture handle #3 (integer coords) | s15761/s15762 |
| `param_0+112` — the output surface | s15582, `sust.b.2d.v4.b16.zero` s15794 |
| `param_0+136` — two f32 texture coordinate scales | s15677 |

**The skip input at `+24` supplies the patch-gather source.**  E1 reads it once
per lane per stencil position and never writes it; the buffer is addressed in
4-byte elements (one f16x2 = 2 channels) with index
`clamp(row)·8·ex + clamp(col)·8 + clamp(t,0,7)`:

```
s2318 L4671: mad.lo.s32 %r3078, %r3068, %r7, %r3077   // clamp_row*(8*ex) + clamp_col*8
s2440 L4800: add.s32 %r19540, %r2996, %r2992          // + clamp(t, - , 7)
s2442 L4803: mul.wide.u32 %rd191, %r19540, 4
s2444 L4805: ld.global.v2.u16 {%rs314, %rs315}, [%rd192]
```

The same index form and the same 24-gather shape as the sibling kernel's §3/E1.

**The surface writes** are `param_0+112` (E38, s15794), a 4-channel b16 surface
written with 3 live channels.  The **plane-arena writes** are E37's 12
`st.global.u32` to `param_0+48` (s15359…s15481).

**Nothing reads `param_0+48` back** in this entry: `param_0+48` is the source of
exactly one `ld.param` (s15333) and never the base of a load.  Likewise the
texture handles at `+88/+96/+104` are only ever `tex.*` operands.

## 5. The phase table, M/N/K checked

`rr_layer_spec.py dec2`'s phase table with `M/16 · N/8 · K/32 == mma` evaluated
on every row:

| # | stmts | M | N | K | mma | M/16·N/8·K/32 | holds? | weight bytes (rel. `W`) | bias bytes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | s452–s595 | 192 | 384 | 128 | 144 | 12·48·4 = 2304 | **no** | 0…11788 (+12288·`tid.z`) | 49152…49328 (+192·`tid.z`) |
| 2 | s7089–s7112 | 96 | 32 | 32 | 24 | 6·4·1 = 24 | yes | 50112…50636 | – |
| 3 | s7120–s7143 | 96 | 32 | 32 | 24 | 24 | yes | 53184…53708 | – |
| 4 | s7151–s7174 | 96 | 32 | 32 | 24 | 24 | yes | 51136…51660 | – |
| 5 | s7182–s7205 | 96 | 32 | 32 | 24 | 24 | yes | 54208…54732 | – |
| 6 | s7213–s7236 | 96 | 32 | 32 | 24 | 24 | yes | 52160…52684 | – |
| 7 | s7244–s7267 | 96 | 32 | 32 | 24 | 24 | yes | 55232…55756 | – |
| 8 | s7439–s7486 | 64 | 96 | 32 | 48 | 4·12·1 = 48 | yes | **none** | 56256…68044 |
| 9 | s11330–s11377 | 192 | 96 | 96 | 48 | 12·12·3 = 432 | **no** | **none** | – |
| 10 | s11547–s11558 | 16 | 96 | 32 | 12 | 1·12·1 = 12 | yes | 68544…71116 | – |
| 11 | s11578–s11589 | 16 | 96 | 32 | 12 | 12 | yes | 90048…92620 | – |
| 12 | s11609–s11620 | 16 | 96 | 32 | 12 | 12 | yes | 111552…114124 | – |
| 13 | s11640–s11651 | 16 | 96 | 32 | 12 | 12 | yes | 133056…135628 | – |
| 14 | s12288–s12299 | 48 | 96 | 96 | 12 | 3·12·3 = 108 | **no** | 136512…139084 | 139584…139632 |
| 15 | s12557–s12568 | 16 | 96 | 32 | 12 | 12 | yes | 139648…142220 | – |
| 16 | s12629–s12640 | 48 | 96 | 96 | 12 | 108 | **no** | 142912…145484 | 145984…146032 |
| 17 | s12805–s12816 | 16 | 96 | 32 | 12 | 12 | yes | 146048…148620 | – |
| 18 | s12877–s12888 | 48 | 96 | 96 | 12 | 108 | **no** | 149120…151692 | 152192…152240 |
| 19 | s13053–s13064 | 16 | 96 | 32 | 12 | 12 | yes | 152256…154828 | – |
| 20 | s13125–s13136 | 48 | 96 | 96 | 12 | 108 | **no** | 155328…157900 | 158400…158448 |
| 21 | s13301–s13312 | 16 | 96 | 32 | 12 | 12 | yes | 158464…161036 | – |
| 22 | s13373–s13384 | 48 | 96 | 96 | 12 | 108 | **no** | 161536…164108 | 164608…164656 |
| 23 | s13549–s13560 | 16 | 96 | 32 | 12 | 12 | yes | 164672…167244 | – |
| 24 | s13621–s13632 | 48 | 96 | 96 | 12 | 108 | **no** | 167744…170316 | 170816…170864 |
| 25 | s13797–s13808 | 16 | 96 | 32 | 12 | 12 | yes | 170880…173452 | – |
| 26 | s13869–s13880 | 48 | 96 | 96 | 12 | 108 | **no** | 173952…176524 | 177024…177072 |
| 27 | s14045–s14056 | 16 | 96 | 32 | 12 | 12 | yes | 177088…179660 | – |
| 28 | s14117–s14128 | 48 | 96 | 96 | 12 | 108 | **no** | 180160…182732 | 183232…183280 |
| 29 | s14293–s14304 | 16 | 96 | 32 | 12 | 12 | yes | 183296…185868 | – |
| 30 | s14365–s14376 | 48 | 96 | 96 | 12 | 108 | **no** | 186368…188940 | 189440…189488 |
| 31 | s14541–s14552 | 16 | 96 | 32 | 12 | 12 | yes | 189504…192076 | – |
| 32 | s14613–s14624 | 48 | 96 | 96 | 12 | 108 | **no** | 192576…195148 | 195648…195696 |
| 33 | s14789–s14800 | 16 | 96 | 32 | 12 | 12 | yes | 195712…198284 | – |
| 34 | s14861–s14872 | 48 | 96 | 96 | 12 | 108 | **no** | 198784…201356 | 201856…201904 |
| 35 | s15037–s15048 | 16 | 96 | 32 | 12 | 12 | yes | 201920…204492 | – |
| 36 | s15109–s15120 | 48 | 96 | 96 | 12 | 108 | **no** | 204992…207564 | 208064…208112 |
| 37 | s15285–s15296 | 16 | 96 | 32 | 12 | 12 | yes | 208128…210700 | – |
| 38 | s15505–s15510 | 48 | 48 | 96 | 6 | 3·6·3 = 54 | **no** | 211200…212236 | 212736…212752 |

**Every phase whose `M/16·N/8·K/32` does not equal `mma` is one the tool also
marks `grid = band`**, and every phase that holds is marked `full`.  The
mismatch factors are not constant (P1 ×16, P9 ×9, P14/…/P36 ×9, P38 ×9), so `M`,
`N` and `K` cannot all be taken at face value for those rows.  For P14 the reason
is checkable and structural:

```
s12288 D=%r12342 A=%r12414 B=%r12293 C=bias@139584      (bo 136512)
s12290 D=%r12362 A=%r12434 B=%r12297 C=%r12342           (bo 137024)
```

— the same D accumulates products with B fragments from *different* 512-byte
blocks and two *different* A quads.  Under the settled weight formula
(`n = 16·tt + 8·(b>>3) + (L>>2)`, verified on enc0) P14's six blocks span
`n = 0…95`, so `N = 96` is supported; then the 3-mma chains must be summing over
`M` or `K`, and neither is separately readable from the fragment structure.
See §7/U2.

**Which phases read no weight operand:** **P8 and P9**.  Both take `B` from
registers the preceding epilogue packed (E7 packs phases 3…7's D for P8's A and
B; E8 packs the probabilities and the transposed P7 for P9's A and B).  Their
`bo` is `None` for every mma.

**Weight image extent as the tool measures it:** the largest resolved offset is
**212752** (s15504, 4 bytes wide) ⇒ **212756 bytes** (1082 distinct slots).
Phase 1's B and C bias carry the only `tid.z` terms (`W + 12288·tid.z`,
`W + 192·tid.z`); with `tid.z ∈ {0,1}` (E38 rejects `tid.z > 1` at s15561
`setp.gt.u32 %p406, %r19503, 1`) the extra 12288 bytes stay inside 212756.
**(INFERENCE** — the tool drops the `tid.z` term, so the per-window base cannot
be read off it directly.)

## 6. The plane arena (param_0 + 48) and the per-launch tables

* `param_0+8` is read 60 times in the prologue (s66…s451) as the **input feature
  plane** that builds phase 1's 12 A fragments.
* `param_0+24` is the **patch-gather source** (see §4); it is read, never written.
* `param_0+48` is the **output plane arena**: written by E37's 12
  `st.global.u32` (s15359…s15481), never read.  The entry has exactly 12
  `st.global` and no other global store.
* The only other `ld.param` sites are the 4 in the prologue (§1.1) and the 5 in
  E38 (`[%rd635+88/96/104/112/136]`, s15683/15759/15761/15582/15677) plus
  `[%rd7+48]` (s15333), `[%rd7+24]` (s2288) and the two `param_0+40` re-reads
  (s15483/s15484).

There is **no per-launch table generated into the plane arena by this kernel**:
the arena is only written (E37), and every table that looks like one
(`W + ((laneid<<2)&12) + {49920, 136128, 136320, 142720, …}`) lives in the
*weight image*, not the arena.

## 7. Uncharacterised

**U1. E9's residual registers.**  The 24 registers that E9's per-column table
adds into (`%rs456/%rs459` and their siblings, s11473–s11544) have exactly one
textual definition each, at s5693/s5694 (L8430/L8434) inside **E1** — nothing
between s5694 and s11473 writes them (`grep -n '%rs456\b'` returns L8430, L8913,
L11184, L24297 only).  So either the value is genuinely carried from E1's
sum-of-squares tree, or the printed PTX reuses a register name across two
disjoint live ranges without a rename.  The two readings differ in what phase
10's C *is* (a table plus E1's partial sums, versus a table plus something
unnameable), so I will not guess.  *Missing:* a shared-memory/register dump at
s11473, or the pre-rename SSA.

**U2. The true `M/N/K` of the thirteen banded phases (P1, P9, P14, P16, …,
P36, P38).**  For those rows `M/16·N/8·K/32 ≠ mma` (§5).  `nA` counts distinct A
*register quads*, which can be either distinct `M`-tiles or distinct `K`-slices
of the same tile, and the accumulator chains mix distinct B operands, so the
two cannot be separated from the fragment structure.  *Missing:* a run with a
weight-image dump, the `_prep` permutation, or the high-level layer definition.

**U3. The identity of the 12288-byte score bias (P8's C).**  E7 loads it as 24
tiles × 512 bytes at `W + 56256`.  Whether it is a relative position bias, a
learned score bias, or a per-lane constant cannot be decided from the PTX: it is
an opaque blob in a permuted weight image.  *Missing:* the image's layout
description or the model's parameter names.

**U4. The identity of the per-column tables.**  `W+49920…50096` (E1 gain),
`W+136128…136304` (E9), `W+136320…136496` (E13 gain), `W+142720…142896` (E14)
are all read with the same `W + ((laneid<<2)&12) + off` pattern (12 words per
lane).  Their *role* is computable (E1/E13: multiply the inverse RMS; E9/E14:
added to a residual before the next GEMM) but their *names* are not.  The GEMM
biases at `+49152`, `+139584`, `+145984`, `+152192`, `+158400`, `+164608`,
`+170816`, `+177024`, `+183232`, `+189440`, `+195648`, `+201856`, `+208064`,
`+212736` are the same case.

**U5. E1's patch-expand geometry.**  As in the sibling kernel, the 36 guarded
`st.shared` are emitted as ~400 basic blocks with 413 labels and I could not
reduce the lane→(row,col) map to a closed form.  Established: 34 `u32` + 2
`v2.u16` patch stores, a second staging area of 6 `v4.u32` stores at
`smem + tid.z·{2048,3072} + {0,512,1024,6144,6656,7168}`, and the whole 9600-byte
shared allocation.  *Missing:* the source-level patchify expression or a
shared-memory dump.

**U6. Whether `param_0+8` is an input plane or a scratch table.**  The prologue
reads it 60 times with 4-byte elements and never writes it, and E1 does not read
it (E1 reads `param_0+24`).  So `+8` and `+24` are two different input buffers;
which one holds the noisy colour plane and which the feature planes cannot be
decided from the PTX.  *Missing:* the host-side launch arguments.

**U7. The `tid.z` strides.**  Phase 1's B and bias carry `×12288` and `×192`
per-window strides and no later phase carries one.  Whether the image is a
concatenation of per-window sub-images or `tid.z` selects a batch element whose
weights are interleaved cannot be decided here; consequently the "212756 bytes"
of §5 is a lower bound.

**U8. The logical meaning of E38's final expression.**  The tail computes
`out = acc·σ(v) + tex96·(1−σ(v)) + tex104` on the three channels gathered from
the 3×3 colour stencil and the two integer-coordinate textures.  The PTX gives
the arithmetic exactly (§3/E38(g)) but not what model term it corresponds to.
*Missing:* the high-level source or a reference implementation.

**U9. Why every even phase shares one A operand.**  E13 emits two scaled copies
of phase 13's D (the normed packs `%r12414…` at s12252 and `%r18435…` at
s12158) and phases 14, 16, …, 36 all pack the second one.  That makes the A
operands of those twelve GEMMs bit-identical; whether that is the intended
computation (one shared projection) or a consequence of a missed optimisation
cannot be decided from the PTX.  *Missing:* the high-level layer definition.

## 8. Quick acceptance index

| requirement | where |
|---|---|
| every epilogue named and ranged | §2, §3 (E1…E38) |
| phase-1 epilogue's patch expand + norm | §3/E1 |
| the norm (`E1`, `E13`) | §3/E1, §3/E13 — raw sum of squares + 2⁻¹³, no mean, no 1/N |
| the attention (score, softmax, P·V) | §3/E7, E8, E9; §3/E9's table add |
| the activations | §3/E14 (materialises `%f861…%f864`), E16, E18, …, E36 |
| the patch expand | §3/E1 (36 `st.shared`), §7/U5 |
| the final epilogue, textures at +96/+104, surface | §3/E38, §4 |
| the texture reads at +96/+104 | §3/E38(f), s15760 / s15762 |
| the skip input at +24 | §3/E1 (s2288), §4 |
| the plane-arena writes | §3/E37 (s15359…s15481), §4, §6 |
| the phase table with M/N/K checked | §5 |
| weight image extent | §5 (212756 bytes, 1082 slots) |
| which phases read no weight | §5 (P8, P9) |
| uncharacterised | §7 |
