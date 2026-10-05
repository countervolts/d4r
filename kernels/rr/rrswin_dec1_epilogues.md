# `cuda_dldn_engine_swin_dec1_kernel` — the epilogues of phases 1…26

Companion to `kernels/rr/rr_layer_spec.py dec1`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0022-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_dec1_kernel` (16304 statements, 44528 physical lines).

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at their first line).  **`LNNNNN`
  is the physical line** of the same instruction in the corpus file, so every
  claim can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and
  the next phase's first `mma`.  `rr_layer_spec.py dec1` prints the ranges; this
  file names what each one *computes*.  Because phase 1 is the first GEMM, the
  prologue (s1–s402) doubles as phase 1's prelude and is described here too.
* `W` denotes the prepared weight image: `ld.param.u64 %rd2,
  [cuda_dldn_engine_swin_dec1_kernel_param_0+40]` (s8, `L1036`) and
  `cvta.to.global.u64 %rd3, %rd2` (s9, `L1037`).  Every
  `ld.weak.global.ca.v4.u32` reads `W + 16*laneid + imm`.
* **Byte offsets are relative to `W` and (where noted) carry a `tid.z` term.**
  Two epilogues' load chains add a per-`tid.z` stride that the rest of the entry
  does not, so the offsets in the phase table are *not* all on one base — §1.1
  and §7/U6.
* The C/D register layout and the weight-addressing formula are fixed (see the
  settled notes): `C/D reg0 = (row g, cols 2t,2t+1)`, `reg1 = (row g+8, same
  cols)`, `g = laneid>>2`, `t = laneid&3`.
* Anything I could not pin down from the PTX is in §7 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `param_0` | `mov.b64 %rd8, …_param_0` | s1, L1018 |
| `W` (weight image) | `ld.param.u64 %rd2, [param_0+40]` → `cvta… %rd3, %rd2` | s8/s9, L1036–L1037 |
| extents `(ex, ey)` | `ld.param.v2.u32 {%r2195, %r2196}, [param_0+0]` | s14, L1042 |
| `%r7 = 8·ex`, `%r8 = ey·8·ex` | `shl.b32 %r7, %r2195, 3` / `mul.lo.s32 %r8, %r2196, %r7` | s15/s16, L1043–L1044 |
| block origins `%r1`, `%r2` | `sub.s32 %r1, %r2190, %r2191` / `sub.s32 %r2, %r2194, %r2192` | s7/s13, L1035/L1041 |
| `%rd1` = `param_0+80` | `add.s64 %rd1, %rd8, 80` | s4, L1032 |
| **input plane arena** | `ld.param.u64 %rd34, [param_0+8]` → `cvta… %rd35, %rd34` | s29/s30, L1057–L1058 |
| **texture / skip input** | `ld.param.u64 %rd149, [%rd1+-56]` = `param_0+24` → `cvta… %rd4, %rd149` | s2215/s2216, L4641–L4642 |
| **output plane arena** | `ld.param.u64 %rd388, [%rd8+48]` → `cvta… %rd5, %rd388` | s15754/s15755, L38150/L38151 |
| block-origin table | `ld.param.v2.u16 {%rs296, %rs297}, [param_0+80]` | s5, L1033 |
| window index `tid.z` | `mov.u32 %r9, %tid.z` | s325, L1355 |
| shared arena | `.shared .align 4 .b8 _ZZ33…dec1_kernel…E4smem[6400]` | L1027 (declaration; not a statement) |
| per-lane weight address | `mul.wide.u32 %rd, %laneid, 16` + `add.s64 %rd, W, %rd` | every load block, e.g. s7214–s7215, L12685–L12686 |
| per-lane *table* address | `shl.b32 %r, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rd, %r, 12` → `add.s64 %rd, %rd3, %rd` = `W + ((laneid·4)&12)` | e.g. s6684–s6686, L10864–L10866 |

The last row matters: the `ld.global.v2.u16` / `ld.global.u32` table loads in an
epilogue read `W + ((laneid<<2)&12) + off`, i.e. **the same 16-byte slot is read
by the four lanes with equal `laneid&3`, and consecutive loads pick consecutive
4-byte words inside one 64-byte record** — so those tables are indexed by
column, not by row.

Two per-`tid.z` strides exist and are used by *different* groups:

| group | base | statements |
|---|---|---|
| phase 1's B | `W + 12288·tid.z + 16·laneid` | s326–s331, L1356–L1363 |
| phase 1's C bias | `W + 256·tid.z + ((laneid·4)&12) + 24576…24816` | s379–s402, L1459–L1484 |
| phases 2…5 B + phase 6's bias | `W + 18432·tid.z + 25216` (`%rd307`, s7073) | s7068–s7073, L11935–L11940 |

`%rd307` is written exactly once (s7073) and consumed only by s7092–s7095
(phase 2's B) and by the four load blocks of E2…E5 (s7122, s7153, s7184, s7215).
Every other phase's loads go straight off `W` with absolute offsets.

### 1.2 Constants that survive between epilogues

| register | first definition | f32/f64 literal | value used (after the `cvt` to f16) |
|---|---|---|---|
| `%f476` | s7433 `mov.f32 %f476, 0f3C8CCB50` | 0.017186790704727173 | f16 `0x2466` = 0.017181396484375 |
| `%fd383` | s7437 `mov.f64 %fd383, 0dBFE1CC0000000000` | −0.55615234375 | f16 `0xB873` |
| `%fd385` | s7442 `mov.f64 %fd385, 0d3FE1CC0000000000` | +0.55615234375 | f16 `0x3873` |
| `%f478` | s7448 `mov.f32 %f478, 0f3F6D6000` | 0.92724609375 | f16 `0x3B6B` |
| `%f480` | s7453 `mov.f32 %f480, 0f3FB00000` | 1.375 | f16 `0x3D80` |
| `%fd1` | s6143 `mov.f64 %fd1, 0d3F20000000000000` | 2⁻¹³ | f16 `0x0800` = 0.0001220703125 |
| `%f989` | s12414 `mov.f32 %f989, 0f3ED306EB` | 0.4121621549129486 | f16 `0x3698` = 0.412109375 |
| `%f990` | s12417 `mov.f32 %f990, 0f3DA60DD6` | 0.0810810774564743 | f16 `0x2D30` = 0.0810546875 |
| `%f991` | s12420 `mov.f32 %f991, 0f3F000000` | 0.5 | f16 `0x3800` |
| `%f992` | s12423 `mov.f32 %f992, 0f40000000` | 2.0 | f16 `0x4000` |

`%f476/%fd383/%fd385/%f478/%f480` are defined **inside E6** and reused by the
second softmax — there is no second softmax here, so they are consumed by E6
alone.  `%f989…%f992` are defined **inside E10** (s12414–s12425) and re-read by
E12, E14, E16, E18, E20, E22, E24 (e.g. s12642 `cvt.rn.f16.f32 low, %f989`);
E12 is the only one that materialises them; the others only `cvt` them.

Also live across the region (defined in the prologue, read by every warp
reduction): `%r5735 = 1` (s5579, L8209), `%r5744 = 2` (s6061, L9582),
`%r5746 = -1` (s6054, L9566) — the three operands of every
`shfl.sync.bfly.b32` in the epilogues.  (`%r5746` is also reused as the
membermask of the E6 shuffles.)

### 1.3 Two quantisation helpers

* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).
* `cvt.rn.f16x2.e4m3x2` — its inverse, the *unpack*, used only when a shared or
  register tile that was stored e4m3-packed is read back (E1: 144 of them,
  s5590–s5676, L8241–L9556).
* `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `div.approx.ftz.f32` — always wrapped as
  `cvt.f32.f16 → op → cvt.rn.f16.f32` (round each step through f16).

## 2. Summary

| epilogue | range | stmts | named operation |
|---|---|---|---|
| E1 | s547–s7095 | 6549 | patch expand into shared + per-column RMS norm + gain + restage |
| E2 | s7120–s7126 | 7 | next-B weight loads (P3) |
| E3 | s7151–s7157 | 7 | next-B weight loads (P4) |
| E4 | s7182–s7188 | 7 | next-B weight loads (P5) |
| E5 | s7213–s7383 | 171 | score-bias C-seed loads (24×512 B) + 80 packs of P2/P3's D |
| E6 | s7432–s11274 | 3843 | cubic-exponent softmax, V transpose, pack |
| E7 | s11323–s11505 | 183 | pack P7's D → shared, restage A, next-B, per-column table + residual → P8's C |
| E8 | s11522–s11537 | 16 | 2 shared A fragments + 4 next-B loads (P9) |
| E9 | s11554–s12396 | 843 | RMS norm (squares, butterfly, +2⁻¹³, rsqrt, gain), packs, next B/C |
| E10 | s12413–s12833 | 421 | clamped-cubic activation + per-column table + residual, packs, next B |
| E11 | s12850–s12917 | 68 | 32 packs → P12's A, next B/C |
| E12 | s12934–s13241 | 308 | clamped-cubic activation, packs, next B |
| E13 | s13258–s13325 | 68 | 32 packs → P14's A, next B/C |
| E14 | s13342–s13649 | 308 | clamped-cubic activation, packs, next B |
| E15 | s13666–s13733 | 68 | 32 packs → P16's A, next B/C |
| E16 | s13750–s14057 | 308 | clamped-cubic activation, packs, next B |
| E17 | s14074–s14141 | 68 | 32 packs → P18's A, next B/C |
| E18 | s14158–s14465 | 308 | clamped-cubic activation, packs, next B |
| E19 | s14482–s14549 | 68 | 32 packs → P20's A, next B/C |
| E20 | s14566–s14873 | 308 | clamped-cubic activation, packs, next B |
| E21 | s14890–s14957 | 68 | 32 packs → P22's A, next B/C |
| E22 | s14974–s15281 | 308 | clamped-cubic activation, packs, next B |
| E23 | s15298–s15365 | 68 | 32 packs → P24's A, next B/C |
| E24 | s15382–s15689 | 308 | clamped-cubic activation, packs, next B |
| E25 | s15706–s15980 | 275 | pack → 16 guarded plane-arena scatter → B/C for P26 |
| E26 | s15989–s16304 | 316 | 6 D stores → shared, row max, softmax, 9+2 texture reads, surface store |

---

## 3. The epilogues

### E1 — s547–s7095: patch expand, RMS norm, gain, restage

6549 statements; op histogram (top): 1102 `add.s32`, 523 `shl.b32`, 460
`mov.b32`, 438 `$label`, 414 `sub.s32`, 368 `min.u32`, 341 `bra.uni`, 288
`mul.f16`, 198 `add.f16`, 144 `cvt.rn.satfinite.e4m3x2.f16x2`, 96
`cvt.rn.f16x2.e4m3x2`, 96 `rsqrt.approx.ftz.f32`, 96 `cvt.f32.f16`, 96
`cvt.rn.f16.f32`, 54 `add.f16x2`, 54 `st.shared.*` (46 `u32` + 2 `v2.u16` + 6
`v4.u32`), 36 `ld.shared.*` (24 `v2.u16` + 12 `v4.u32`), 32
`ld.global.v2.u16`, 12 `shfl.sync.bfly.b32`, 1 `ld.param`, 2 `ld.weak…`.

Sub-sections (bounded by the first/last statement of each op):

| range | what |
|---|---|
| s547–s688 | 144 packs of phase 1's D → the 48 b32 `%r10…%r55` |
| s689–s1367 | per-lane index arithmetic + the 48 guarded patch stores (46 `st.shared.u32` s732–s1355 + 2 `st.shared.v2.u16` s718/s1367); 1st `bar.sync` at s1369 |
| s1413–s2214 | 24 `ld.shared.v2.u16` read back of the packed patch |
| s2215–s2216 | `param_0+24` → `%rd4`: **the patch-gather buffer** |
| s2371–s5577 | 24 clamped `ld.global.v2.u16` gathered from that buffer |
| s5578–s5674 | 96 `cvt.rn.f16x2.e4m3x2` unpacks of the read-back patch |
| s5677–s6287 | the sum-of-squares trees (198 `add.f16`, 54 `add.f16x2`) |
| s6143–s6144 | the 2⁻¹³ round (s6143 `mov.f64 %fd1`, s6144 `cvt.rn.f16.f64 %rs1082, %fd1`) |
| s6293–s6670 | 96 `rsqrt.approx.ftz.f32`, each `cvt.f32.f16 → rsqrt → cvt.rn.f16.f32` |
| s6668–s6686 | 12 statements: **gain loads** `W + ((laneid·4)&12) + {25088…25200}` (8 `ld.global.v2.u16`) |
| s5868–s6973 | the 288 `mul.f16` (two groups: `inv · gain` then `x · (inv·gain)`) |
| s6990–s7047 | 24 packs → the 6 `st.shared.v4.u32` at s7008/7021/7034/7047/7061/7066 |
| s7058–s7089 | `bar.sync` (s7072) + 12 `ld.shared.v4.u32` (s7077–s7089) → **phase 2's A** |
| s7068–s7095 | `%rd307 = W + 18432·tid.z + 25216` (s7073) and 2 `ld.weak` → **phase 2's B** |

Evidence for the patch scatter (48 stores):

```
s718  L2859: st.shared.v2.u16 [%r2393], {%rs200, %rs201}
s732  L2875: st.shared.u32 [%r2400+1600], %r10
s771  L2916: st.shared.u32 [%r2418+1600], %r12
s783  L2932: st.shared.u32 [%r2425+3200], %r13
s797  L2948: st.shared.u32 [%r2432+4800], %r14
s7066 L11932: st.shared.v4.u32 [%r5949+4608], {%r5957, %r5956, %r5955, %r5954}
```

The 48 patch stores are 12 blocks of 4: base offsets `{0, 1600, 3200, 4800}`
(12 stores each) plus two `v2.u16` stores.  Each block is guarded by

```
s708  L2848: setp.gt.u32 %p5, %r64, 9
s709  L2849: setp.gt.u32 %p6, %r62, 9
s711  L2851: @%p7 bra $L__BB1_2
```

and the index is `(row_clamped·40 + col_clamped·8 + t)·4` bytes
(s714 `mad.lo.s32 %r2389, %r64, 40, %r2388`, s715 `or.b32 %r2390, %r57, %r2389`,
s716 `shl.b32 %r2391, %r2390, 2`), i.e. a **10×10 grid of 4-word cells × 4
sub-arrays = 6400 bytes = the whole shared allocation**.  (§7/U4: the guards
bound row/col at 9; I did not reduce the 438-label duplication to a closed form.)

The gather from `param_0+24` (s2215–s2371) uses

```
s2245 L4674: mad.lo.s32 %r3019, %r3009, %r7, %r3018     // clamp_row·(8·ex) + clamp_col·8
s2250 L4679: add.s32 %r19034, %r3023, %r3019            // + clamp(t, - , 7)
s2369 L4806: mul.wide.u32 %rd150, %r19034, 4
s2371 L4808: ld.global.v2.u16 {%rs346, %rs347}, [%rd151]
```

so the element stride of the `+24` buffer is 4 bytes (one f16x2 = two channels),
indexed `row·(8·ex) + col·8 + clamp(t,0,7)`; the same buffer is read 24 times
per lane, once per `(row,col)` of the stencil.

The norm itself:

```
s6143 L9797:  mov.f64 %fd1, 0d3F20000000000000        // 2^-13
s6144 L9799:  cvt.rn.f16.f64 %rs1082, %fd1
s6186 L9927:  add.f16 %rs1161,%rs1067,%rs1082         // ...+ eps
s6293 L10243: rsqrt.approx.ftz.f32 fl, fl
s6688 L10878: mul.f16 %rs1371,%rs1372,%rs2011         // inv * gain
s6832 L11313: mul.f16 %rs1659,%rs488,%rs1371          // value * (inv*gain)
```

**There is no mean subtraction and no `1/N`.**  The accumulated quantity is the
raw sum of squares of the patch's 3 channels plus the constant 2⁻¹³, and the
norm is applied per *column* (the gain is read at `W + ((laneid<<2)&12) + off`
so it is constant across rows).  The 12 `shfl.sync.bfly.b32` sum across the four
lanes of each `laneid&3` group.

**Inputs:** phase 1's D (`%r692…%r2083`), `W`, `param_0+8`, `param_0+24`,
`%r1/%r2/%r7/%r8/%r2195/%r2196`, `%r9` (`tid.z`).
**Outputs (live past s7095):** the 6 patch-expand `st.shared` regions, the 12
`ld.shared.v4.u32` A fragments `%r18458…%r18505`, phase 2's B
`%r5960…%r5967`, `%rd307`, the 5 constants of §1.2.

### E2/E3/E4 — s7120–s7126, s7151–s7157, s7182–s7188: next-B loads

Three 7-statement blocks, each identical in shape:

```
s7120 L12140: mov.u32 %r6208, %laneid
s7121 L12142: mul.wide.u32 %rd309, %r6208, 16
s7122 L12143: add.s64 %rd310, %rd307, %rd309
s7123 L12144: add.s64 %rd203, %rd310, 2048
s7124 L12146: ld.weak.global.ca.v4.u32 { %r6209,…%r6212},[%rd203]     // +2560 as well
```

| epilogue | offsets added to `%rd307` (`W + 18432·tid.z + 25216`) | destination phase |
|---|---|---|
| E2 | `+2048`, `+2560` | P3's B |
| E3 | `+1024`, `+1536` | P4's B |
| E4 | `+3072`, `+3584` | P5's B |

Nothing else.  The offsets are deliberately out of phase order: the weight
sub-image is laid out P2, P4, P3, P5 (offset blocks 0, 1, 2, 3 of 1024 bytes).

### E5 — s7213–s7383: score-bias C-seed loads + pack of P2/P3's D

171 statements; the only epilogue whose weight loads carry the `tid.z` stride.

* **24 loads** `W + 18432·tid.z + 16·laneid + 25216 + 4096 + 512·i`,
  `i = 0…23`, at s7217…s7263 (L12689…L12781), into `%r6956…%r7051`.  Byte block
  `[18432·tid.z + 29312, +41100)` = 12288 bytes = 24 tiles × 512 — **phase 6's C
  score-bias seed**.
* **80 packs / 40 b32**, s7264–s7383, whose sources are registers written by
  **phase 2's and phase 3's mma** (`%r6466/6467/6486/6487/…` from s7096–s7126
  and s7127–s7136).  Two interleaved destinations — `%r7498 %r7508 …` and
  `%r7244 %r7246 …` — which become phase 6's B and A respectively (verified:
  `%r7244 ← mov.b32 {%rs2061,%rs2062}` at s7338 whose sources are
  `cvt` of `%r6466/%r6486`; `%r7498 ← mov.b32 {%rs2013,%rs2014}` at s7266 whose
  source is `cvt` of `%r6466` at s7265).

**The C-seed layout.**  The four C registers of one mma are the four u32 of one
16-byte lane slot, and the tile index runs `4·(n>>1) + m` — the same scheme
`rrswin_enc0_epilogues.md` §3/E9 established.  Checked against phase 6's own
operands: `s7384 C=%r6956,%r6957`, `s7385 C=%r6958,%r6959`,
`s7390 C=%r7004,%r7005`, `s7391 C=%r7006,%r7007` — consecutive pairs from the
load block, i.e. mma `(m, n)` takes the slot `(n>>1)` of the load block.

### E6 — s7432–s11274: cubic-exponent softmax, V transpose, pack

**Identical in kind to `cuda_dldn_engine_swin_enc0_kernel`'s E5** (see
`rrswin_enc0_epilogues.md` §3/E5) and byte-for-byte the same op histogram:
1048 `mov.b32`, 672 `cvt.rn.f16.f32`, 384 `cvt.rn.f16.f64`, 192 each of
`fma.rn.f16x2` / `cvt.f32.f16` / `rcp.approx.ftz.f32` / `mul.f16`, 144 packs,
48 `movmatrix`, 16 `shfl.sync.bfly.b32`, 96 each of `max/min/neg/abs.f16x2`.

Sub-sections:

| range | stmts | what |
|---|---|---|
| s7433–s7458 | 26 | the 5 constants (§1.2): s7433 `mov.f32 %f476, 0f3C8CCB50`, s7437/s7442 the two f64 clamps, s7448/s7453 the cubic pair |
| s7459–s9549 | 2091 | 96 per-element groups (scale, clamp, cubic, `and.b32 …,2145419232`) |
| s9550–s9635 | 86 | in-lane partial sums (`add.f16x2`) |
| s9636–s9765 | 130 | 16 `shfl.sync.bfly` → 8 row sums |
| s9766–s10587 | 822 | 192 `rcp.approx.ftz.f32` → 96 reciprocals |
| s10588–s11010 | 423 | 192 `mul.f16` → 96 probabilities |
| s11011–s11058 | 48 | `movmatrix.sync.trans.aligned.m8n8.b16` of **phase 5's** D |
| s11059–s11273 | 215 | 144 `cvt.rn.satfinite.e4m3x2.f16x2` packs |

Evidence:

```
s7459 L13474: and.b32 %r9534, %r18506, 2145419232          // 0x7FE07FE0
s9636 L20270: shfl.sync.bfly.b32 %r9778,%r9774,%r5735,%r9781,%r5746
s9766 L20596: rcp.approx.ftz.f32 fl, fl
s11011 L22895: movmatrix.sync.trans.aligned.m8n8.b16 %r10138, %r6715
s11059 L23039: cvt.rn.satfinite.e4m3x2.f16x2 %rs3462, %r10146
```

The transposed registers `%r6715…%r6946` are exactly phase 5's D (s7189–s7212).
`%r10138/%r10146/…` are the movmatrix results; the pack at s11059 takes
`%r10146`, i.e. the transposed V is what becomes **phase 7's B** (checked:
`%r10744 ← mov.b32 {%rs3461,%rs3462}` at s11061, and `%r10744` is phase 7's
first B pair; the probability packs, e.g. `%r10450 ← mov.b32
{%rs3509,%rs3510}` at s11133, feed phase 7's A).  `%r6715` is phase 5's first D
and `%r6946` its last, so all 48 movmatrix inputs are phase 5's D.

For the record, the per-element function (identical to enc0's E5(a), reproduced
here because it is the only arithmetic in this file that is *not* obviously a
named primitive):

```
m  = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)
u  = f16(1.375 + m*(0.92724609375 - m*m))
expval = f16 from bits  ((u_bits << 5) & 0x7FE07FE0)
```

**There is no row maximum in E6** (the only `max.f16x2`/`min.f16x2` are the 96
clamp pairs of (a)), and no bias add — the bias is phase 6's GEMM C operand.

**Inputs:** phase 6's 96 D f16x2, phase 5's 48 D f16x2, `%r5735/%r5744/%r5746`.
**Outputs:** the 5 constants, phase 7's 12 A fragments + 12 B fragments.
**Memory:** none (0 `ld.*`, 0 `st.*` in 3843 statements).

### E7 — s11323–s11505: pack P7's D → shared, restage, next-B, table → P8's C

183 statements: 64 `add.f16`, 48 `mov.b32`, 32 packs, 8 `ld.global.v2.u16`,
4 `st.shared.v4.u32`, 4 `ld.shared.v4.u32`, 4 `ld.weak`, 6 `add.s64`.

```
s11323 L23878: bar.sync 0
s11339 L23912: st.shared.v4.u32 [%r18635], {%r18639, %r18638, %r18637, %r18636}
s11352 L23941: st.shared.v4.u32 [%r18635+512], {…}
s11365 L23970: st.shared.v4.u32 [%r18635+1024], {…}
s11378 L23999: st.shared.v4.u32 [%r18635+1536], {…}
s11379 L24000: bar.sync 0
s11384 L24007: ld.shared.v4.u32 {%r18655,…%r18658}, [%r18654]
s11385 L24008: ld.shared.v4.u32 {%r18659,…%r18662}, [%r18654+512]
s11390 L24016: ld.weak.global.ca.v4.u32 { %r10893,…%r10896},[%rd233]   // W+41600
s11402 L24037: ld.global.v2.u16 {%rs5218, %rs5219}, [%rd322+62096]
s11410 L24046: add.f16 %rs3640,%rs5233,%rs491
```

* **32 packs of phase 7's D** (`%r10378…%r10428`, the 192×96 banded P·V output)
  → 16 b32 → four `st.shared.v4.u32` at `[%r18635 + {0,512,1024,1536}]`; the
  `bar.sync` pair makes them visible, then **two `ld.shared.v4.u32` give phase
  8's A** (`%r18655…%r18662`).
* **4 `ld.weak` at `W + 16·laneid + {41600, 42112, 42624, 43136}`** → phase 8's B.
* **8 `ld.global.v2.u16` at `W + ((laneid·4)&12) + {62080…62192}`** — a
  per-column vector, 8 words per lane = 32 columns.
* **64 `add.f16`** adding that vector to the *f16 halves of phase 7's D* which
  were unpacked at s6186–s6288 (`%rs488/%rs491/%rs494…`), producing
  `%r10918 %r10919 %r10928 %r10929 …` — **phase 8's C operand** (confirmed:
  `s11506 C=%r10918,%r10919`, `s11507 C=%r10928,%r10929`).  So phase 8 computes
  `D8 = A8·B8 + (D7 + column_table)`.

### E8 — s11522–s11537: next-A and next-B for P9

```
s11525 L24450: ld.shared.v4.u32 {%r18666,…%r18669}, [%r18665+2048]
s11526 L24451: ld.shared.v4.u32 {%r18670,…%r18673}, [%r18665+2560]
s11531 L24459: ld.weak.global.ca.v4.u32 { %r11072,…%r11075},[%rd237]   // W+60032
s11537 L24471: ld.weak.global.ca.v4.u32 { %r11084,…%r11087},[%rd240]   // W+61568
```

Two shared A fragments (the upper half of the same staging region, offsets 2048
and 2560 relative to `%r18652`) and four weight fragments at
`W + {60032, 60544, 61056, 61568}` → phase 9's A and B.

### E9 — s11554–s12396: RMS norm before the second attention window

843 statements, no stores, 16 memory ops.  Structure:

| range | stmts | op |
|---|---|---|
| s11554–s11699 | 146 | 32 × `x·x` squares of phase 9's D |
| s11700–s11757 | 58 | in-lane pairwise `add.f16x2` |
| s11714–s11757 | 44 | 8 `shfl.sync.bfly` (2 per row) → row sums |
| s11764–s11779 | 16 | `add.f16` of the two halves, then `+ 2⁻¹³` (§1.2 `%fd1`) |
| s11779–s11880 | 102 | broadcast each row sum into 4 b32 |
| s11883–s12128 | 246 | 64 `rsqrt.approx.ftz.f32` |
| s12129–s12136 | 8 | **gain loads** `W + ((laneid·4)&12) + {62208…62320}` |
| s12137–s12232 | 96 | `mul.f16` — `inv · gain` |
| s12233–s12328 | 96 | `mul.f16` — `x · (inv·gain)` → 32 b32 `%r17567…%r17598` |
| s12332–s12339 | 8 | next-B loads `W + {62336, 62848, 63360, 63872}` → phase 10's B |
| s12341–s12348 | 8 | next-C loads `W + ((laneid·4)&12) + {64384,64400,64416,64432}` → phase 10's C |
| s12349–s12396 | 48 | 32 packs → phase 10's A (`%r11574…%r11577`, `%r11594…`, …) |

Evidence:

```
s11557 L24589: mul.f16 %rs3832,%rs4647,%rs4647          // square of a D half
s11714 L25018: shfl.sync.bfly.b32 %r11325,%r11321,%r5735,%r11328,%r5746
s11776 L25176: cvt.rn.f16.f64 %rs4033, %fd1             // eps = 2^-13
s11883 L25588: rsqrt.approx.ftz.f32 fl, fl
s12129 L25883: ld.global.v2.u16 {%rs5234, %rs5235}, [%rd327+62224]
s12233 L26180: mul.f16 %rs4421,%rs4647,%rs4229          // D_half * (inv*gain)
s12332 L26472: add.s64 %rd241, %rd329, 62336
s12349 L26500: cvt.rn.satfinite.e4m3x2.f16x2 %rs4611, %r17568
```

**What it computes.**  For each of the 8 rows a lane holds:

```
sumsq[row] = Σ_c x[row,c]²                      (f16, 4-lane butterfly)
inv        = rsqrt.approx.ftz.f32(f32(f16(sumsq + 2⁻¹³)))   (rounded to f16)
out[row,c] = x[row,c] · inv · gain[c]
```

* **no mean subtraction and no `1/N`** — the quantity is the raw sum of squares
  plus the constant 2⁻¹³.
* `gain[c]` is per *column* (the `W + ((laneid<<2)&12)` pattern) and constant
  across rows.
* the `inv` register is duplicated into both halves of a b32 before use, so one
  row-sum serves every n-tile of that row (64 `rsqrt` for 16 row sums).
* the norm's *bias* is not added here; it is phase 10's C, loaded at the end of
  this epilogue (s12345–s12348).

**Outputs:** `%r11473 %r11477` (P10 B), `%r11611 %r11621 %r11651 %r11661`
(P10 C), 16 packed A registers, the 32 f16x2 normed values `%r17567…%r17598`
(re-packed by E11).

#### Shape of the attention region (E5–E9)

```
P2  A = patch(shared), B = W+25216+2048,  C = 0        → D2  (96×32)
P3  A = patch(shared), B = W+25216+   0,  C = 0        → D3  (96×32)
P4  A = patch(shared), B = W+25216+1024,  C = D2       → D4
P5  A = patch(shared), B = W+25216+3072,  C = D3       → D5
     E5 packs D2 ∪ D3 → P6 A and B;  E5 loads P6 C = bias@29312
P6  A = pack(D2∪D3), B = pack(D2∪D3), C = bias → scores 64×96
     E6: softmax over the 96 columns (no row max), 48 movmatrix(D5), pack
P7  A = pack(prob), B = pack(transpose(D5)), C = 0 → D7 (192×96)
     E7: pack D7 → shared;  P8 C = D7 + column table@62080;  P8 B = W+41600
P8  A = shared, B = W+41600, C = D7+table → D8
P9  A = shared+2048, B = W+60032, C = 0 → D9
     E9: RMS norm of D9 + gain, packs → P10 A
```

### E10 — s12413–s12833: clamped-cubic activation, per-column table, residual

421 statements; this is the only epilogue that materialises `%f989…%f992`.

```
s12414 L26726: mov.f32 %f989, 0f3ED306EB
s12420 L26740: mov.f32 %f991, 0f3F000000          // 0.5
s12426 L26755: neg.f16x2 %r11687,%r11686
s12427 L26759: max.f16x2 %r11689,%r11542,%r11687
s12428 L26763: min.f16x2 %r11692,%r11689,%r11686
s12429 L26767: abs.f16x2 %r11695,%r11692
s12432 L26779: mul.f16x2 %r11703,%r11692,%r11700
s12433 L26783: add.f16x2 %r11706,%r11685,%r11703
s12434 L26787: mul.f16x2 %r11709,%r11542,%r11706
```

Per element, over each of the 16 f16x2 D registers of phase 10:

```
y   = clamp(x, -2, +2)
g   = 0.5 + y * (0.412109375 - 0.0810546875 * |y|)      // f16 throughout
out = x * g
```

`g` is a *cubic ramp*, not a sigmoid: `g(+2)=1 ⇒ out=x` for `x ≥ 2`;
`g(-2)=0 ⇒ out=0` for `x ≤ -2`.  **Reproduce the formula, not SiLU/GELU.**

Then:

| range | op |
|---|---|
| s12413–s12689 | 16 activation groups |
| s12693–s12700 | next-B loads `W + {64448,64960,65472,65984}` → phase 11's B |
| s12701–s12713 | `ld.global.v2.u16` at `W + ((laneid·4)&12) + {66496…66608}` |
| s12714–s12809 | 64 `add.f16` — residual add, written over the **activation inputs** (`%rs4644/%rs4647`, unpacked at s11556/… from phase 9's D) |
| s12810–s12833 | 16 packs of the activation outputs → phase 12's A |

Evidence for the residual add:

```
s12706 L27718: ld.global.v2.u16 {%rs5250, %rs5251}, [%rd337+66512]
s12714 L27727: add.f16 %rs4645,%rs5265,%rs4647
s12716 L27734: mov.b32 %r12189, {%rs4642, %rs4645}
s12810 L28015: cvt.rn.satfinite.e4m3x2.f16x2 %rs4835, %r11825
```

The residual is **phase 9's D**, not phase 10's: `%rs4644/%rs4647` are the halves
of `%r11088`, unpacked at s11556 (`mov.b32 {%rs4644, %rs4647}, %r11088`) at the
very start of E9, and `%r11088` is written by phase 9's mma (s11538–s11553).
The 32 such registers `%r12189 %r12190 … %r12329 %r12330` are **phase 11's C**;
`%r12148 %r12152 …` are phase 11's B.

Note the two pack streams: the *activation* outputs (`%r11709`, `%r11825`, …) go
to phase 12's A, while the *residual+table* registers go to phase 11's C.

### E11 — s12850–s12917: packs of the normed tile, next B/C

68 statements: 32 packs, 4 `ld.weak`, 4 `ld.global.u32`.

```
s12854 L28189: ld.weak.global.ca.v4.u32 { %r12342,…%r12345},[%rd249]
s12866 L28210: ld.global.u32 %r12480, [%rd342+68672]
s12870 L28216: cvt.rn.satfinite.e4m3x2.f16x2 %rs…, %r17567
```

* four `v4.u32` fragments at `W + {66624, 67136, 67648, 68160}` → phase 12's B;
* four `ld.global.u32` at `W + ((laneid·4)&12) + {68672, 68688, 68704, 68720}`
  → **phase 12's C seed**, each used in both halves of the C operand;
* **32 packs of `%r17567…%r17598`** — the *normalised* tile E9 produced — so
  phase 12's A operand is bit-identical to phase 10's A.  Phases 10 and 12 are
  two parallel GEMMs off the same normed tile.

### E12 — s12934–s13241: clamped-cubic activation, no residual

308 statements; op mix 72 `mov.b32`, 64 `cvt.rn.f16.f32`, 48 `mul.f16x2`, 16
each `neg/max/min/abs/sub/add.f16x2`, 4 `ld.weak`, 16 packs.  **No constants of
its own** (it re-`cvt`s `%f989` etc. at s12642-equivalent sites) and **no
`ld.global`**.

Same function as E10; 16 packs of the activation outputs
(`%r13046` at s13191, …) → phase 13's A, plus 4 weight loads at
`W + {68736, 69248, 69760, 70272}` → phase 13's B.  **No table add and no
residual**, so phase 13's C is zero.

### E13 — s13258–s13325: packs of the normed tile again, next B/C

68 statements, shape-identical to E11:

```
s13262 L29597: ld.weak.global.ca.v4.u32 { %r13210,…},[%rd257]
s13274 L29618: ld.global.u32 %r13348, [%rd349+72832]
```

B at `W + {70784, 71296, 71808, 72320}` → phase 14's B; **32 packs of
`%r17567…%r17598` — the normed tile for the second time**; C at
`W + ((laneid·4)&12) + {72832, 72848, 72864, 72880}` → phase 14's C.

### E14/E16/E18/E20/E22/E24 — s13342–s13649, s13750–s14057, s14158–s14465, s14566–s14873, s14974–s15281, s15382–s15689: activation + packs + next B

Six shape-identical 308-statement epilogues (shape hash `4234ec4de215`).  Each
applies the E10 activation to the D of its own phase, then 16 packs of the
activation outputs → the next odd phase's A, and 4 weight loads → the next odd
phase's B.  Evidence (E14):

```
s13376 L30936: cvt.rn.f16.f32 low, %f989
s13393 L30985: mul.f16x2 %r13727,%r13634,%r13724
s13619 L30816: ld.weak.global.ca.v4.u32 { %r13885,…},[%rd261]   // W+70784
s13633 L30986: cvt.rn.satfinite.e4m3x2.f16x2 %rs…, %r13727
```

Offsets: E12→{68736,69248,69760,70272}, E14→{70784,71296,71808,72320},
E16→{72896,73408,73920,74432}, E18→{74944,75456,75968,76480},
E20→{77056,77568,78080,78592}, E21's targets … (from the phase table's weight
column: P13=68736, P15=72896, P17=77056, P19=81216, P21=85376, P23=89536,
P25=93696).

No `ld.global`, no table add, no residual: these six produce only A and B.

### E15/E17/E19/E21/E23 — s13666–s13733, s14074–s14141, s14482–s14549, s14890–s14957, s15298–s15365: packs + next B/C

Five shape-identical 68-statement epilogues (shape hash `819f37d58cd2`), the
twin of E11/E13.  Example (E15):

```
s13670 L31005: ld.weak.global.ca.v4.u32 { %r14078,…},[%rd265]
s13682 L31026: ld.global.u32 %r14216, [%rd356+76992]
s13686 L31031: cvt.rn.satfinite.e4m3x2.f16x2 %rs…, %r17567
```

B at `W + {72896+1024…}` etc. per the phase table; C at
`W + ((laneid·4)&12) + {76992, 77008, 77024, 77040}` (E15 → phase 16's C);
**32 packs of `%r17567…%r17598`, the normed tile — third, fourth, fifth, sixth
and seventh time** (E13, E15, E17, E19, E21, E23 all pack the same 32 f16x2).

#### Shape of the whole MLP region (E9–E25)

```
E9  normed tile N (32 f16x2, %r17567…%r17598, s12349 packs)
 ├─ pack → P10 A  (s12349)   P10: A=N, B=W10,  C=bias@64384   → act (E10)
 │                            pack of act → P12 A (s12870)
 │                            E10: P11 C = D9 + table@66496
 ├─ pack → P12 A  (s12870)   P12: A=N, B=W12,  C=bias@68672   → act (E12)
 │                            pack of act → P14 A (s13394)
 ├─ pack → P14 A  (s13394)   P14: A=N, B=W14,  C=bias@72832   → act (E14)
 ├─ pack → P16 A  (s13806)   P16: A=N, B=W16,  C=bias@76992   → act (E16)
 ├─ pack → P18 A  (s14218)   P18: A=N, B=W18,  C=bias@81152   → act (E18)
 ├─ pack → P20 A  (s14630)   P20: A=N, B=W20,  C=bias@85312   → act (E20)
 ├─ pack → P22 A  (s15042)   P22: A=N, B=W22,  C=bias@89472   → act (E22)
 └─ pack → P24 A  (s15454)   P24: A=N, B=W24,  C=bias@93632   → act (E24)
                              pack of act → P25 A (s15666)
```

The odd phases P11, P13, …, P25 take `C = 0` and produce the residual pair that
the following even phase's table adds.

### E25 — s15706–s15980: pack → 16 guarded plane-arena stores → B/C for P26

275 statements; the only `st.global` in the entry (16 of them, all here).

**(a) requantise, s15706–s15753.**  24 groups of `2 cvt + 1 mov.b32` → 24 b32;
the 16 consumed are `%r489…%r504`.

**(b) plane-arena scatter, s15754–s15960.**

```
s15754 L38150: ld.param.u64 %rd388, [%rd8+48]        // = param_0+48
s15755 L38151: cvta.to.global.u64 %rd5, %rd388
s15757 L38155: shr.u32 %r505, %r18449, 2             // g = laneid>>2
s15758 L38156: and.b32 %r506, %r18449, 3             // t = laneid&3
s15761 L38159: add.s32 %r18707, %r488, %r18705       // y = %r488 + (laneid>>5)
s15767 L38165: add.s32 %r18713, %r1, %r18706         // x = %r1 + (g&7)
s15772 L38170: shl.b32 %r18714, %r18713, 3           // x<<3
s15773 L38171: mad.lo.s32 %r507, %r7, %r18707, %r18714
s15775 L38173: selp.b32 %r508, -1, %r18715, %p1
s15780 L38179: st.global.u32 [%rd390], %r489
```

Stride register `%r7 = 8·ex` (s15).  Every store is
`idx = 8·ex·y + 8·x + d`, with `d = t` or `t|4`; each of the four base slices is
guarded by its own predicate of the form
`(y<0)|(y≥ey)|(laneid>127)|(x<0)|(x≥ex)` (e.g. s15762–s15771,
L38160–L38169); a guard failure stores `-1` as the index, which the following
`setp.lt.s32 %p, %idx, 0` + `bra` turns into "skip the store"
(s15774–s15777).

Store values, bases and column offsets (`g = laneid>>2`, `t = laneid&3`,
`%r488 = %r2 + 4·tid.z` at s11555/L24586, `%r8 = ey·8·ex`):

| store | stmt | value | index base | d |
|---|---|---|---|---|
| 1 | s15780 | `%r489` | `idx1` | `t` |
| 2 | s15789 | `%r490` | `idx1` | `t\|4` |
| 3 | s15815 | `%r491` | `idx2` | `t` |
| 4 | s15823 | `%r492` | `idx2` | `t\|4` |
| 5 | s15832 | `%r493` | `idx1 + %r8` | `t` |
| 6 | s15840 | `%r494` | `idx1 + %r8` | `t\|4` |
| 7 | s15849 | `%r495` | `idx2 + %r8` | `t` |
| 8 | s15857 | `%r496` | `idx2 + %r8` | `t\|4` |
| 9 | s15883 | `%r497` | `idx3` | `t` |
| 10 | s15891 | `%r498` | `idx3` | `t\|4` |
| 11 | s15917 | `%r499` | `idx4` | `t` |
| 12 | s15925 | `%r500` | `idx4` | `t\|4` |
| 13 | s15934 | `%r501` | `idx3 + %r8` | `t` |
| 14 | s15942 | `%r502` | `idx3 + %r8` | `t\|4` |
| 15 | s15951 | `%r503` | `idx4 + %r8` | `t` |
| 16 | s15959 | `%r504` | `idx4 + %r8` | `t\|4` |

with the four bases built as `mad.lo.s32 %rBASE, 8·ex, y, 8·x`:

| base | stmt | y | x |
|---|---|---|---|
| `idx1 = %r507` | s15773, L38171 | `%r488 + (laneid>>5)` | `%r1 + (g&7)` |
| `idx2 = %r511` | s15808, L38210 | `%r488 + ((g+8)>>3)` | `%r1 + ((g+8)&7)` |
| `idx3 = %r520` | s15876, L38290 | `%r488 + ((g+16)>>3)` | `%r1 + ((g+16)&7)` |
| `idx4 = %r523` | s15910, L38328 | `%r488 + ((g+24)>>3)` | `%r1 + ((g+24)&7)` |

The `g+8j` slices come from `add.s32 %r18717, %r18986, 8` (s15793, L38195),
`add.s32 %r18734, %r18988, 16` (s15861, L38275) and
`add.s32 %r18747, %r18990, 24` (s15895, L38313); each base carries its own
guard predicate (`p1…p4`) of the form
`(y<0)|(y≥ey)|(slice>31)|(x<0)|(x≥ex)`.

**(c) next B/C, s15961–s15980.**

```
s15962 L38392: ld.param.u64 %rd456, [param_0+40]
s15969 L38401: add.s64 %rd424, %rd456, %rd423        // + 16*laneid
s15971 L38404: ld.weak.global.ca.v4.u32 { %r18765,…%r18768},[%rd421]   // +95744
s15973 L38408: ld.weak.global.ca.v4.u32 { %r18769,…%r18772},[%rd422]   // +96256
s15979 L38417: ld.global.u32 %r18823, [%rd428+96768]
s15980 L38418: ld.global.u32 %r18833, [%rd428+96784]
```

Note `%rd424` is built from `%rd456` (the *un*cvtad pointer) while `%rd455` is
cvtad; on this corpus both are the same address, and the offsets still read
`W + ….`

**Outputs:** `%r489…%r504` (re-read as P26's A, see s15981), `%r18765 %r18769`
(P26 B), `%r18823 %r18833` (P26 C), `%rd388` (`param_0+48`).

### E26 — s15989–s16304: the final epilogue

316 statements; it consumes the six D accumulators of phase 26
(`%r18794 %r18795 %r18804 %r18805 %r18834 %r18835 %r18844 %r18845` — the
`s15996`…`s15988` mma results) and writes the surface.

**(a) staging, s15989–s16050.**  12 guarded `st.shared.u32` of the phase-26 D
pairs into a shared scratch, indexed from `%r540 = laneid>>2`, `%r541 = laneid&3`
and `%tid.z`:

```
s16000 L38488: st.shared.u32 [%r542], %r18794
s16004 L38493: st.shared.u32 [%r542+16], %r18804
s16017 L38507: st.shared.u32 [%r544], %r18795
s16019 L38510: st.shared.u32 [%r544+16], %r18805
```

**(b) coordinates and bounds check, s16051–s16084.**

```
s16059 L38555: mov.u32 %r18897, %tid.x
s16060 L38556: add.s32 %r18898, %r19003, %r18897     // %r19003 = tid.z*32
s16065 L38561: sub.s32 %r547, %r18898, %r18902
s16067 L38563: add.s32 %r549, %r1, %r547             // x
s16068 L38564: add.s32 %r550, %r19004, %r548         // y, %r19004 = ctaid.y*8 - origin
s16073 L38569: ld.param.v2.u32 {%r18912, %r18913}, [%rd458]
s16078 L38574: @%p480 bra $L__BB1_534
```

Out-of-range lanes jump straight to `$L__BB1_534` (s16303) and skip (c)…(g).

**(c) row max over 5 shared values, s16085–s16100.**

```
s16085 L38582: ld.shared.u32 %r18915, [%r18964]
s16087 L38584: ld.shared.u32 %r18916, [%r18964+4]
s16091 L38588: ld.shared.u32 %r18936, [%r18964+16]
s16093 L38591: max.f16x2 %r18914,%r18915,%r18916
s16097 L38604: max.f16 %rs5266,%rs5267,%rs5268
s16099 L38611: mov.b32 %r18937, {%rs5269, %rs5269}
s16100 L38613: sub.f16x2 %r18923,%r18915,%r18937
```

Five `u32` (= 10 f16) are read per row and reduced in registers — **no warp
shuffle** — giving the row max, which is subtracted.

**(d) softmax, s16101–s16168.**  Textbook f32 softmax over the row's 5 values:

```
s16103 L38621: mul.ftz.f32 %f1033, %f993, 0f3FB8AA3B    // * log2(e)
s16104 L38622: ex2.approx.ftz.f32 %f995, %f1033
s16160 L38741: mov.f32 %f1032, 0f3F800000
s16161 L38742: div.approx.ftz.f32 %f1011, %f1032, %f1050
s16164 L38749: mul.f16x2 %r18938,%r18939,%r18952
```

`y = 2^((x−max)·log2e)`, `p = y · (1/Σy)` in f16, five f16x2 registers
(`%r18938 %r18941 %r18944 %r18947 %r18950`).

**(e) texture gather, s16169–s16266.**  A 3×3 stencil of a colour texture:

```
s16184 L38789: cvt.rn.f32.s32 %f1051, %r18968
s16185 L38790: ld.param.v2.f32 {%f1052, %f1053}, [%rd459+136]
s16187 L38792: fma.rn.ftz.f32 %f1012, %f1052, %f1051, %f1056
s16191 L38796: ld.param.u64 %rd446, [%rd459+88]
s16193 L38799: tex.base.2d.v4.f16.f32 {%rs5294,%rs5295,%rs5296,%rs5297}, [%rd430, {%f1012,%f1013}]
s16195 L38803: fma.rn.f16 %rs5298,%rs5294,%rs5308,%rs5291
```

`%rd459 = param_0` (s16079), so the texture handle is **`param_0+88`**, the
scale/offset are **`param_0+136`** (two f32: `f1052·(k−0.5·?)`) and the
accumulator weights come from the softmax registers.  Nine
`tex.base.2d.v4.f16.f32` are issued (s16193, 16205, 16217, 16229, 16234, 16240,
16252, 16258, 16263) for the 3×3 neighbourhood (clamped coordinates at
s16172–s16183 compute `(x−1, x, x+1)` and `(y−1, y, y+1)`), accumulated with
`fma.rn.f16` into 3 channels.

**(f) two integer-coordinate textures, s16267–s16270.**

```
s16267 L38971: ld.param.u64 %rd448, [%rd459+96]
s16268 L38973: tex.base.2d.v4.f16.s32 {%rs5438,%rs5439,%rs5440,%rs5441}, [%rd448, {%r549,%r550}]
s16269 L38975: ld.param.u64 %rd450, [%rd459+104]
s16270 L38977: tex.base.2d.v4.f16.s32 {%rs5442,%rs5443,%rs5444,%rs5445}, [%rd450, {%r549,%r550}]
```

**These are the `+96` / `+104` texture reads the brief asks about** — handles at
`param_0+96` and `param_0+104`, sampled at the *integer* pixel coordinate
`(%r549, %r550)`, returning two 4-channel f16 vectors.

**(g) combine, sigmoid gate, surface store, s16271–s16304.**

```
s16271 L38980: cvt.f32.f16 %f1030, %rs5446
s16272 L38983: mul.ftz.f32 %f1063, %f1030, 0fBFB8AA3B
s16273 L38984: ex2.approx.ftz.f32 %f1064, %f1063
s16274 L38985: add.ftz.f32 %f1065, %f1064, 0f3F800000
s16275 L38986: div.approx.ftz.f32 %f1031, %f1032, %f1065     // 1/(1+2^(-log2e·v))
s16278 L38996: sub.f16 %rs5449,%rs5448,%rs5447               // 1 - sigmoid
s16279 L39000: mul.f16 %rs5452,%rs5426,%rs5447               // acc * sigmoid
s16280 L39004: fma.rn.f16 %rs5495,%rs5438,%rs5449,%rs5452    // + tex96 * (1-sig)
s16285 L39024: set.nan.f16.f16 %rs5473,%rs5442,%rs5442       // + tex104 unless NaN
s16302 L39065: sust.b.2d.v4.b16.zero [%rd453, {%r18983,%r550}], {%rs5495,%rs5496,%rs5497,%rs5494}
```

`%rd453 = param_0+112` (s16090).  The store address column is `%r549·8` (s16300
`shl.b32 %r18983, %r549, 3`), the row `%r550`; the fourth channel is the zero
literal `0x0000` (s16301 `mov.u16 %rs5494, 0`), so this is a **3-channel write
into a 4-channel surface**.  The `set.nan`/`bra` triad (s16285–s16298) adds the
`+104` read only when it is not NaN.

**Inputs:** the 6 returned D registers, `param_0+{48,88,96,104,112,136}`, the
shared scratch, constants `0f3FB8AA3B`, `0fBFB8AA3B`, `0f3F800000`.
**Outputs:** none (the kernel returns, s16304).

---

## 4. Texture reads, surface writes, and the `+24` input

| what | where |
|---|---|
| `param_0+8` — input feature plane arena, read by phase 1's A loads | prologue s67…s324 (52 `ld.global.u32`) |
| `param_0+24` — the patch-gather source E1 reads | s2215 (`ld.param.u64 %rd149, [%rd1+-56]`, `%rd1 = param_0+80`) and the 24 `ld.global.v2.u16` at s2371…s5577 |
| `param_0+48` — the plane arena E25 scatters phase 25's output to | s15754 / s15755, 16 `st.global.u32` at s15780…s15959 |
| `param_0+88` — the colour texture handle | s16191, 9 `tex.base.2d.v4.f16.f32` at s16193…s16263 |
| `param_0+96` — texture handle #2 (integer coords) | s16267/s16268 |
| `param_0+104` — texture handle #3 (integer coords) | s16269/s16270 |
| `param_0+112` — the output surface | s16090, `sust.b.2d.v4.b16.zero` s16302 |
| `param_0+136` — two f32 texture coordinate scales | s16185 |

**The skip input at `+24` supplies the patch-gather source.**  E1 reads it once
per lane per stencil position and never writes it; the buffer is addressed in
elements of 4 bytes (one f16x2 = 2 channels) with index
`clamp(row)·8·ex + clamp(col)·8 + clamp(t,0,7)` (s2245/s2250, L4674/L4679).  It
is the *input* the patch expand tile is built from — the same operand the
window-1 patch is expanded out of.

**The surface writes** are `param_0+112` (E26, s16302), a 4-channel b16 surface
written with 3 live channels.  The **plane-arena writes** are E25's 16
`st.global.u32` to `param_0+48` (s15780…s15959).

**Nothing reads `param_0+48` back** in this entry: `param_0+48` is the source of
exactly one `ld.param` (s15754) and never the base of a load.  Likewise the
texture handles at `+88/+96/+104` are only ever `tex.*` operands.

## 5. The phase table, M/N/K checked

`rr_layer_spec.py dec1`'s phase table with the arithmetic `M/16 · N/8 · K/32 ==
mma` evaluated on every row (the brief's check):

| # | stmts | M | N | K | mma | M/16·N/8·K/32 | holds? | weight bytes (rel. `W`) | bias bytes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | s403–s546 | 144 | 384 | 96 | 144 | 9·48·3 = 1296 | **no** | 0…11788 (+12288·`tid.z`) | 24576…24816 (+256·`tid.z`) |
| 2 | s7096–s7119 | 96 | 32 | 32 | 24 | 6·4·1 = 24 | yes | 25216…25740 (+18432·`tid.z`) | – |
| 3 | s7127–s7150 | 96 | 32 | 32 | 24 | 24 | yes | 27264…27788 (+18432·`tid.z`) | – |
| 4 | s7158–s7181 | 96 | 32 | 32 | 24 | 24 | yes | 26240…26764 (+18432·`tid.z`) | – |
| 5 | s7189–s7212 | 96 | 32 | 32 | 24 | 24 | yes | 28288…28812 (+18432·`tid.z`) | – |
| 6 | s7384–s7431 | 64 | 96 | 32 | 48 | 4·12·1 = 48 | yes | **none** | 29312…41100 (+18432·`tid.z`) |
| 7 | s11275–s11322 | 192 | 96 | 96 | 48 | 12·12·3 = 432 | **no** | **none** | – |
| 8 | s11506–s11521 | 32 | 64 | 32 | 16 | 2·8·1 = 16 | yes | 41600…43148 | – |
| 9 | s11538–s11553 | 32 | 64 | 32 | 16 | 16 | yes | 60032…61580 | – |
| 10 | s12397–s12412 | 64 | 64 | 64 | 16 | 4·8·2 = 64 | **no** | 62336…63884 | 64384…64432 |
| 11 | s12834–s12849 | 32 | 64 | 32 | 16 | 16 | yes | 64448…65996 | – |
| 12 | s12918–s12933 | 64 | 64 | 64 | 16 | 64 | **no** | 66624…68172 | 68672…68720 |
| 13 | s13242–s13257 | 32 | 64 | 32 | 16 | 16 | yes | 68736…70284 | – |
| 14 | s13326–s13341 | 64 | 64 | 64 | 16 | 64 | **no** | 70784…72332 | 72832…72880 |
| 15 | s13650–s13665 | 32 | 64 | 32 | 16 | 16 | yes | 72896…74444 | – |
| 16 | s13734–s13749 | 64 | 64 | 64 | 16 | 64 | **no** | 74944…76492 | 76992…77040 |
| 17 | s14058–s14073 | 32 | 64 | 32 | 16 | 16 | yes | 77056…78604 | – |
| 18 | s14142–s14157 | 64 | 64 | 64 | 16 | 64 | **no** | 79104…80652 | 81152…81200 |
| 19 | s14466–s14481 | 32 | 64 | 32 | 16 | 16 | yes | 81216…82764 | – |
| 20 | s14550–s14565 | 64 | 64 | 64 | 16 | 64 | **no** | 83264…84812 | 85312…85360 |
| 21 | s14874–s14889 | 32 | 64 | 32 | 16 | 16 | yes | 85376…86924 | – |
| 22 | s14958–s14973 | 64 | 64 | 64 | 16 | 64 | **no** | 87424…88972 | 89472…89520 |
| 23 | s15282–s15297 | 32 | 64 | 32 | 16 | 16 | yes | 89536…91084 | – |
| 24 | s15366–s15381 | 64 | 64 | 64 | 16 | 64 | **no** | 91584…93132 | 93632…93680 |
| 25 | s15690–s15705 | 32 | 64 | 32 | 16 | 16 | yes | 93696…95244 | – |
| 26 | s15981–s15988 | 64 | 32 | 64 | 8 | 4·4·2 = 32 | **no** | 95744…96268 | 96768…96784 |

**Every phase whose `M/16·N/8·K/32` does not equal `mma` is one the tool also
marks `grid = band`**, and every phase that holds is marked `full`.  The
mismatching rows are *not* off by a constant factor (P1 is ×9, P7 ×9, P10/12/…
×4, P26 ×4), so `M`, `N` and `K` cannot all be taken at face value for those
rows: `nA` counts distinct A *register quads*, which can be either distinct
`M`-tiles or distinct `K`-slices of the same tile, and the tool cannot tell them
apart.  See §7/U1.

**Which phases read no weight operand:** **P6 and P7**.  Both take `B` from
registers that the preceding epilogue packed (E5 packs P2/P3's D for P6's A and
B; E6 packs the probabilities and the transposed P5 for P7's A and B).  Their
`bo` is `None` for every mma.

**Weight image extent as the tool measures it:** the largest resolved offset is
**96784** (s15980, 4 bytes wide) ⇒ **96788 bytes** (634 distinct slots).  This
excludes the two `tid.z`-strided groups in §1.1: phase 1's B and bias are read
at `W + 12288·tid.z` / `W + 256·tid.z`, and phases 2…5's B and phase 6's bias at
`W + 18432·tid.z + 25216`.  With `tid.z ∈ {0,1}` (E26 rejects `tid.z > 1` at
s16069 `setp.gt.u32 %p474, %r19008, 1`) the true image must reach at least
`18432 + 41100 = 59532`, which is inside 96788, so the two `tid.z` slices do not
extend the image beyond what the tool already sees.  **(INFERENCE** — the tool
drops the `tid.z` term, so the per-window base cannot be read off it directly.)

## 6. The plane arena (param_0 + 48) and the per-launch tables

* `param_0+8` is read 52 times in the prologue (s67…s324) as the **input feature
  plane** that builds phase 1's 9 A fragments.  The index arithmetic is
  `(clamp_r·ey + clamp_c)·4` style (`shl.b32 … 3`, `mad.lo.s32`, `mul.wide.u32 …,
  4`, `add.s64 …, %rd35`): the arena is a `{ex, ey}`-wide f16/byte plane read
  with 4-byte elements.
* `param_0+24` is the **patch-gather source** (see §4); it is read, never written.
* `param_0+48` is the **output plane arena**: written by E25's 16
  `st.global.u32` (s15780…s15959), never read.  There is no second `st.*` group
  in the entry (the entry has exactly 16 `st.global` and 0 other global stores
  before L38179).
* The only other `ld.param` sites are the 4 in the prologue (§1.1) and the 5 in
  E26 (`[%rd459+88/96/104/112/136]`, s16191/16267/16269/16090/16185) plus
  `[%rd8+48]` (s15754) and the two `param_0+40` re-reads (s15961/s15962).

There is **no per-launch table generated into the plane arena by this kernel**:
the plane arena is only written (E25) and the tables that look like per-launch
tables (the `W + ((laneid<<2)&12) + off` blocks at 25088, 62080, 66496) all live
in the *weight image*, not the arena.  See §7/U5.

## 7. Uncharacterised

**U1. The true `M/N/K` of the nine banded phases (P1, P7, P10, P12, …, P24,
P26).**  For those rows `M/16·N/8·K/32 ≠ mma` (§5).  The reason is structural
and checkable: the accumulator chains of a banded phase mix *distinct* B
operands.  For P10 the four chains are

```
s12397 D=%r11522 A=%r11574 B=%r11473 C=bias      (bo 62336)
s12399 D=%r11542 A=%r11594 B=%r11477 C=%r11522   (bo 62848)
s12401 D=%r11562 A=%r11574 B=%r11481 C=bias      (bo 63360)
s12403 D=%r11582 A=%r11594 B=%r11485 C=%r11562   (bo 63872)
```

— the same D accumulates products with B fragments from four *different*
512-byte blocks and two *different* A quads.  Under the settled weight formula
(`n = 16·tt + 8·(b>>3) + (L>>2)`, verified on enc0) those four blocks span
`n = 0…63`, so `N = 64` is supported; but then the two-mma chains must be
summing over `M` or `K`, and neither M nor K is separately readable from the
fragment structure.  For P1 the same argument gives `N = 384` (48 B fragments =
24 blocks × 16 columns) while the arithmetic needs `M/16 · K/32 = 3`.
*Missing:* either a run with a weight-image dump, or the `_prep` permutation
that produces the image, or the high-level layer definition.  Until then I will
not assign M/K to these rows.

**U2. The identity of the 12288-byte score bias (P6's C).**  E5 loads it as 24
tiles × 512 bytes at `W + 18432·tid.z + 29312`.  Whether it is a relative
position bias, a learned score bias, or a per-lane constant cannot be decided
from the PTX: it is an opaque blob in a permuted weight image.  *Missing:* the
image's layout description (the `_prep` permutation) or the model's parameter
names.

**U3. The identity of the per-column tables.**  `W+25088…25200` (E1 gain),
`W+62080…62192` (E7), `W+62208…62320` (E9 gain), `W+66496…66608` (E10) are all
read with the same `W + ((laneid<<2)&12) + off` pattern and are 8 words per lane.
Their *role* is computable (E1/E9: multiply the inverse RMS; E7/E10: added to a
residual before the next GEMM) but their *names* are not.  The GEMM biases at
`+24576`, `+64384`, `+68672`, `+72832`, `+76992`, `+81152`, `+85312`, `+89472`,
`+93632`, `+96768` are the same case.

**U4. E1's patch-expand geometry.**  The 48 `st.shared` of §3/E1 are guarded by
row/col limits of 9 and index `(row·40 + col·8 + t)·4` into four 1600-byte
sub-arrays, but they are emitted as ~400 basic blocks with 438 labels and I could
not reduce the lane→(row,col) map of the 10×10 patch to a closed form.  What is
established: the four sub-arrays, the 40-word row pitch, the `t` and `t|4`
column split, the two `v2.u16` cells, and the total 6400 bytes.
*Missing:* the source-level im2col/patchify expression, or a shared-memory dump.

**U5. Whether `param_0+8` is an input plane or a scratch table.**  The prologue
reads it 52 times with 4-byte elements and never writes it, and E1 does not read
it at all (E1 reads `param_0+24`).  So `+8` and `+24` are two different input
buffers; which one holds the noisy colour plane and which holds the feature
planes cannot be decided from the PTX.  *Missing:* the host-side launch
arguments.

**U6. The `tid.z` strides.**  Two different per-`tid.z` strides (12288 for phase
1, 18432 for phases 2…5 + phase 6's bias, 256 for phase 1's bias, 192 for
phase 1's bias in the companion layout) appear, and no later phase carries one.
Whether this means the weight image is a concatenation of per-window sub-images
or that `tid.z` selects a batch element whose weights are interleaved cannot be
decided from the PTX alone.  Consequence: the "96788 bytes" of §5 is a lower
bound for the image, not necessarily its size.

**U7. The logical meaning of E26's final expression.**  The tail computes
`out = acc·σ(v) + tex96·(1−σ(v)) + tex104` on the three channels gathered from
the 3×3 colour stencil and the two integer-coordinate textures.  The PTX gives
the arithmetic exactly (§3/E26(g)) but not what model term it corresponds to.
*Missing:* the high-level source or a reference implementation.

**U8. E12/E14/…/E24's `%f989…%f992`.**  These six epilogues re-`cvt` the four
constants but never re-`mov.f32` them, so they read the registers E10 wrote
(s12414–s12425).  That is only valid if the register allocator kept them live
across phases 11…23; the PTX shows no redefinition, so it is consistent, but
nothing in the file asserts the interference-freedom.  *Missing:* an SSA check.

## 8. Quick acceptance index

| requirement | where |
|---|---|
| every epilogue named and ranged | §2, §3 (E1…E26) |
| phase-1 epilogue's patch expand + norm | §3/E1 |
| the norm (`E1`, `E9`) | §3/E1, §3/E9 — raw sum of squares + 2⁻¹³, no mean, no 1/N |
| the attention (score, softmax, P·V) | §3/E5, E6, E7; §3/E9 (region diagram) |
| the activations | §3/E10 (materialises `%f989…%f992`), E12, E14, E16, E18, E20, E22, E24 |
| the patch expand | §3/E1 (48 `st.shared`), §7/U4 |
| the final epilogue, textures at +96/+104, surface | §3/E26, §4 |
| the texture reads at +96/+104 | §3/E26(f), s16268 / s16270 |
| the skip input at +24 | §3/E1 (s2215), §4 |
| the plane-arena writes | §3/E25 (s15780…s15959), §4, §6 |
| the phase table with M/N/K checked | §5 |
| weight image extent | §5 (96788 bytes, 634 slots) |
| which phases read no weight | §5 (P6, P7) |
| uncharacterised | §7 |
