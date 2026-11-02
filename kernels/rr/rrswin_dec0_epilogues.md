# `cuda_dldn_engine_swin_dec0_kernel` — the epilogues of phases 1…19

Companion to `kernels/rr/rr_layer_spec.py dec0`.  Everything below was read out
of `~/.cache/d4r-rr-corpus/0023-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_dec0_kernel` (file lines 1018–47791).

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at their first line).  **`LNNNNN`
  is the physical line** of the same instruction in the corpus file, so every
  claim can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and
  the next phase's first `mma`; `Exx` below means *the epilogue of phase xx*.
* `W` denotes the prepared weight image: `ld.param.u64 %rd2,
  [cuda_dldn_engine_swin_dec0_kernel_param_0+40]` (s11, L1039) and
  `cvta.to.global.u64 %rd3, %rd2` (s12, L1040).  Every
  `ld.weak.global.ca.v4.u32` reads `W + 16*laneid + imm`; the table loads read
  `W + ((laneid<<2)&12) + imm`.
* `%rd1` = the parameter-block base at `+80`: `mov.b64 %rd25,
  cuda_dldn_engine_swin_dec0_kernel_param_0` (s1, L1018) then `add.s64 %rd1,
  %rd25, 80` (s4, L1032).  So `[%rd1+n]` is `param_0+80+n`, `[%rd1-56]` is
  `param_0+24` and `[%rd1-80]` is `param_0+0`.
* `smem` denotes `_ZZ33cuda_dldn_engine_swin_dec0_kernel30DldnEngineSwinDec0ParamsStructE4smem`,
  declared 3200 bytes by the cut-ABI dump.
* The C/D register layout and the weight-addressing formula are fixed (see the
  settled notes): `C/D reg0 = (row g, cols 2t,2t+1)`, `reg1 = (row g+8, same
  cols)`, `g = laneid>>2`, `t = laneid&3`; the A fragment's four registers are
  `reg0 = row g, k = 2t + (j&1) + 16*((j>>1)&1)`, `reg1 = row g+8, same k`,
  `reg2 = row g, k+8`, `reg3 = row g+8, k+8`.
* Anything I could not pin down from the PTX is in §6 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `W` (weight image, 47376 bytes) | `ld.param.u64 %rd2, [param_0+40]` / `cvta.to.global.u64 %rd3, %rd2` | s11 / s12 |
| `smem` | `mov.u32 %r1706, _ZZ…SwinDec0ParamsStructE4smem` | s560 (L2364) |
| `%rd1` = `param_0+80` | `mov.b64 %rd25, param_0` + `add.s64 %rd1, %rd25, 80` | s1 / s4 |
| extents `(ex, ey)` | `ld.param.v2.u32 {%r1527, %r1528}, [param_0]` | s13 (L1041) |
| block offsets `(%rs235, %rs236)` | `ld.param.v2.u16 {%rs235, %rs236}, [param_0+80]` | s5 (L1033) |
| **input buffer** | `ld.param.u64 %rd27, [param_0+8]`, then `cvta %rd28` | s29 / s30 |
| **skip input** | `ld.param.u64 %rd114, [%rd1+-56]` = `param_0+24`, then `cvta %rd4` | s1459 / s1460 |
| **plane arena** | `ld.param.u64 %rd319, [%rd357+-32]` = `param_0+48`, then `cvta %rd320` | s18493 / s18494 |
| **texture A** (18 f32-coord reads) | `ld.param.u64 %rd313, [%rd357+8]` = `param_0+88` | s18702 (L46878) |
| **texture B** (s32-coord reads) | `ld.param.u64 %rd315, [%rd357+16]` = `param_0+96` | s18778 (L47053) |
| **texture C** (s32-coord reads) | `ld.param.u64 %rd317, [%rd357+24]` = `param_0+104` | s18780 (L47057) |
| **surface A** (`sust.b`) | `ld.param.u64 %rd323, [%rd357+32]` = `param_0+112` | s18604 (L46672) |
| **surface B** (`sust.p`) | `ld.param.u64 %rd295, [%rd357+64]` = `param_0+144` | s18602 (L46668) |
| texture coord scale/origin | `ld.param.v2.f32 {%f1602, %f1603}, [%rd357+56]` = `param_0+136` | s18696 (L46872) |
| per-lane weight address | `mul.wide.u32 %rdX, %laneid, 16` + `add.s64 %rdX, %rd3, %rdX` | every load block |
| per-lane *table* address | `shl.b32 %rX, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rdX, %rX, 12` → `add.s64 %rdX, %rd3, %rdX` = `W + ((laneid*4) & 12)` | s306–s309 and 60 copies |

The last line matters: every `ld.global.v2.u16` / `ld.global.u32` table load in
an epilogue reads `W + ((laneid<<2)&12) + off`, i.e. **the four lanes with equal
`laneid&3` read the same address, so one instruction covers one 16-byte word and
32 lanes cover a 512-byte span**.  A 4-load group with `step 16` therefore
covers 64 bytes = 32 f16 = one f16 per column of an `N=32` GEMM's C operand,
replicated over the 16 rows of a tile.

### 1.2 Constants and helpers

| register | first definition | f32/f64 literal | f16 value used |
|---|---|---|---|
| `%fd770` | s5030 L8337 `mov.f64 %fd770, 0d3F20000000000000` | 2⁻¹³ | `0x0800` = 0.0001220703125 |
| `%f1469` | s15420 L36585 `mov.f32 %f1469, 0f3ED306EB` | 0.4121621549129486 | `0x3698` = 0.412109375 |
| `%f1470` | s15423 L36592 `mov.f32 %f1470, 0f3DA60DD6` | 0.0810810774564743 | `0x2D30` = 0.0810546875 |
| `%f1471` | s15426 L36599 `mov.f32 %f1471, 0f3F000000` | 0.5 | `0x3800` |
| `%f1472` | s15429 L36606 `mov.f32 %f1472, 0f40000000` | 2.0 | `0x4000` |
| `%f1520` | s18515 L46581 `mov.f32 %f1520, 0f3F800000` | 1.0 | `0x3C00` (E19) |
| `%f1526` | s18517 L46583 `mov.f32 %f1526, 0fC0000000` | −2.0 | `0xC000` (E19) |

`%f1469…%f1472` are materialised **inside E12** (s15420–s15431) and re-used by
E14, E16 and E18, which contain no `mov.f32` of their own.

* `mov.u32 %r13609, 0` (L6552) is the constant-zero C operand used by every mma
  whose `C` field is `{%r13609, %r13609}` (phases 2, 3, 7, 8, 11 …); the same
  register is later re-read as a zero f16 by E19 (§3/E19).
* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).
* `cvt.rn.f16x2.e4m3x2` — the *inverse* direction, used only by E1 to
  dequantise the skip input (96 occurrences, s4386–s4481).
* `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `div.approx.ftz.f32`, `abs.ftz.f32` — always wrapped as
  `cvt.f32.f16 → op → cvt.rn.f16.f32` (each step rounded through f16).

## 2. Phase table, geometry and weight image

### 2.1 The phase table

`rr_layer_spec.py dec0` prints 19 phases:

| # | stmt range | tool M | tool N | tool K | mma | grid | weight bytes | bias bytes | epi |
|---|---|---|---|---|---|---|---|---|---|
| 1 | s326–s421 | 96 | 256 | 64 | 96 | band | 0..7692 | 8192..8432 | 5515 |
| 2 | s5937–s5960 | 96 | 32 | 32 | 24 | full | 8512..9036 | – | 7 |
| 3 | s5968–s5991 | 96 | 32 | 32 | 24 | full | 9536..10060 | – | 171 |
| 4 | s6163–s6210 | 64 | 96 | 32 | 48 | full | *(none)* | 10560..22348 | 3843 |
| 5 | s10054–s10101 | 192 | 96 | 96 | 48 | band | *(none)* | – | 160 |
| 6 | s10262–s10277 | 64 | 32 | 32 | 16 | full | 22848..23372 | – | 7 |
| 7 | s10285–s10308 | 96 | 32 | 32 | 24 | full | 23872..24396 | – | 7 |
| 8 | s10316–s10339 | 96 | 32 | 32 | 24 | full | 24896..25420 | – | 171 |
| 9 | s10511–s10558 | 64 | 96 | 32 | 48 | full | *(none)* | 25920..37708 | 3838 |
| 10 | s14397–s14444 | 192 | 96 | 96 | 48 | band | *(none)* | – | 55 |
| 11 | s14500–s14515 | 64 | 32 | 32 | 16 | full | 38208..38732 | – | 887 |
| 12 | s15403–s15418 | 64 | 32 | 32 | 16 | full | 39360..39884 | 40384..40432 | 709 |
| 13 | s16128–s16143 | 64 | 32 | 32 | 16 | full | 40448..40972 | – | 64 |
| 14 | s16208–s16223 | 64 | 32 | 32 | 16 | full | 41536..42060 | 42560..42608 | 600 |
| 15 | s16824–s16839 | 64 | 32 | 32 | 16 | full | 42624..43148 | – | 64 |
| 16 | s16904–s16919 | 64 | 32 | 32 | 16 | full | 43648..44172 | 44672..44720 | 600 |
| 17 | s17520–s17535 | 64 | 32 | 32 | 16 | full | 44736..45260 | – | 64 |
| 18 | s17600–s17615 | 64 | 32 | 32 | 16 | full | 45760..46284 | 46784..46832 | 600 |
| 19 | s18216–s18231 | 64 | 32 | 32 | 16 | full | 46848..47372 | – | 929 |

### 2.2 The M/N/K arithmetic (`M/16 · N/8 · K/32 == mma`)

Ranks hold for **15 of the 19 phases**.  The four failures are the banded
phases; in each the tool's `M` overcounts because A and B fragments are
reused across k-steps, and in two of them `K` is also wrong.

* **Phase 1 — 48×128 with K=64, not 96×256.**
  The phase has 96 mma, 6 distinct A quads and 32 distinct B pairs, but the
  chains (`chains_of`, s326–s421) are **48 chains of length 2**.  Each chain
  uses one A quad and one B pair per k-step:
  `s326 A=%r756 B=%r399@0 → s328 A=%r776 B=%r403@512`,
  `s330 A=%r756 B=%r407@1024 → s332 A=%r776 B=%r411@1536`, …  So the 6 A quads
  are 3 m-tiles × 2 k-slices and the 32 B pairs are 16 n-tiles × 2 k-slices:
  `M = 48`, `N = 128`, `K = 64`, and `3·16·2 = 96`.  mma count over 6·32 = 192
  A×B pairs would be 384 if the tool's M/N were taken literally; only the
  48-chain reading reproduces 96.
* **Phase 5 and phase 10 — 64×32 with K=96, not 192×96.**
  Each has 48 mma in 16 chains of length 3.  Chains 0…3 share the A triple
  `{%r8639, %r8659, %r8679}` (phase 5) and take four different B triples;
  chains 4…7 use a second A triple, etc.  So the 12 A quads are 4 m-tiles × 3
  k-slices and the 12 B pairs are 4 n-tiles × 3 k-slices:
  `M = 64`, `N = 32`, `K = 96`, and `4·4·3 = 48`.  (This is the
  `(64×96)·(96×32)` banded attention product.)
* **Phase 1's tool `N=256`** comes from counting the 32 B pairs as 8 columns
  each; the 16 real n-tiles are the 16 × 1024-byte-strided groups seen at
  `s274…s304` (`W+0, +512, …, +7680`, i.e. 16 loads × 2 fragments).

So the phase-table `M`/`N`/`K` columns overcount the four banded phases by the
k-slice factor; the corrected shapes are the ones a native replacement must
produce.

### 2.3 Weight image extent

The prepared weight image is **47376 bytes**.  The largest byte actually loaded
is `W+47372` (`ld.weak.global.ca.v4.u32` at s18167, L46043) and 47372+4 = 47376;
no load reads past it.  The B-operand offsets resolved from the mma alone span
`W+0 … W+47372` (176 distinct B word offsets, 224 distinct C word offsets).

Unlike enc5, **every** weight load in this entry is issued from the plain
`W + 16*laneid + off` base: each load block builds `add.s64 %rdX, %rd2,
%rdY` with `%rdY = mul.wide.u32(%laneid, 16)` and then adds the offset
(s5931…s5934, L10447…L10451, for phase 2's B; s5993…s5996, L10809…L10813, for
phase 4's C; s10340…s10344, L22901…L22907, for phase 9's C; s12036…s12038 for
E12's B).  There is no `%tid.z` term and therefore no per-group interleaving of
the byte offsets.

### 2.4 Phases that read no weight operand

Every `B` operand of every phase resolves to the weight image **except four**:

* **phase 4** — its B registers `%r8933…%r9043` are written by the
  `cvt.rn.satfinite`/`mov.b32` packs of E3 (s6043 L10908 onwards); the
  phase-table B column is `?`.
* **phase 5** — its B registers are E4's `movmatrix`/pack output (`?` in the
  table), i.e. the transposed V.
* **phase 9** — like phase 4: its B comes from E8's packs (s10391 L23002).
* **phase 10** — like phase 5: its B comes from E9's `movmatrix`/pack output.

Phases 4 and 9 read their *C* operand from the weight image (the 12288-byte
score biases at `W+10560…W+22348` and `W+25920…W+37708`, loaded in E3 and E8);
phases 5 and 10 use the constant-zero C `%r13609` (L6552).

## 3. The epilogues

| epilogue | range | stmts | named operation |
|---|---|---|---|
| E1 | s422–s5936 | 5515 | pack phase 1's D, stage a 10×10 window in shared, read the skip input, RMS norm, pack |
| E2 | s5961–s5967 | 7 | next-B weight loads |
| E3 | s5992–s6162 | 171 | 24 B-fragment loads (phase 4's C bias) + pack phase 3's D |
| E4 | s6211–s10053 | 3843 | softmax of the 64×96 score block, V-transpose, pack |
| E5 | s10102–s10261 | 160 | residual + per-column table, pack, next B |
| E6 | s10278–s10284 | 7 | next-B weight loads |
| E7 | s10309–s10315 | 7 | next-B weight loads |
| E8 | s10340–s10510 | 171 | 24 B-fragment loads (phase 9's C bias) + pack phase 8's D |
| E9 | s10559–s14396 | 3838 | softmax (second window, same code as E4) |
| E10 | s14445–s14499 | 55 | next-B loads + pack phase 10's D |
| E11 | s14516–s15402 | 887 | RMS norm (², sum, rsqrt, gain) + packs + next B/C |
| E12 | s15419–s16127 | 709 | clamped cubic activation + column table + pack |
| E13 | s16144–s16207 | 64 | next B/C loads + pack |
| E14 | s16224–s16823 | 600 | clamped cubic activation + pack |
| E15 | s16840–s16903 | 64 | next B/C loads + pack |
| E16 | s16920–s17519 | 600 | clamped cubic activation + pack |
| E17 | s17536–s17599 | 64 | next B/C loads + pack |
| E18 | s17616–s18215 | 600 | clamped cubic activation + pack |
| E19 | s18232–s19160 | 929 | final epilogue: row max, softmax, texture gather, softmax, surface writes |

---

### E1 — s422–s5936: pack, shared window staging, skip input, RMS norm

5515 statements — the largest epilogue, and the only one with control flow per
element group (339 `$label`s, 278 `bra.uni`, 159 `bra`).  Op order:

| range | op |
|---|---|
| s422–s600 | 80 `cvt.rn.satfinite.e4m3x2.f16x2` + 37 `mov.b32` → 40 b32 of phase 1's D |
| s540–s765 | address arithmetic; 37 `st.shared.u32` + 3 `st.shared.v2.u16` into `smem` |
| s771–s800 | guard predicates for the staging |
| s805–s1457 | 24 `ld.shared.v2.u16` read the staged patch back |
| s1459 | `ld.param.u64 %rd114, [%rd1+-56]` = `param_0+24` (the skip input) |
| s1613–s4308 | 21 `ld.global.v2.u16` from `%rd4` (= `param_0+24`) |
| s4357–s4384 | 3 more `ld.global.v2.u16` from the same pointer |
| s4386–s4481 | 96 `cvt.rn.f16x2.e4m3x2` (dequantise the skip values) |
| s4484–s5174 | 204 `add.f16` (pairwise residual adds) |
| s4674–s5856 | 288 `mul.f16` (squares and gain multiplies) |
| s4818–s5027 | 60 `add.f16x2` (in-lane sums) |
| s4849–s5026 | 24 `shfl.sync.bfly.b32` |
| s5030–s5031 | `mov.f64 %fd770, 0d3F20000000000000` + `cvt.rn.f16.f64` — `2⁻¹³` |
| s5178–s5559 | 96 `cvt.f32.f16` + 96 `rsqrt.approx.ftz.f32` + 96 `cvt.rn.f16.f32` |
| s5563–s5569 | `W + ((laneid<<2)&12)` + 4 `ld.global.v2.u16` at `W+8448…8496` |
| s5860–s5928 | 48 `cvt.rn.satfinite` packs (+ 44 `mov.b32`) of the normalised values |
| s5934–s5936 | 2 `ld.weak.global.ca.v4.u32` at `W+8512`, `W+9024` → phase 2's B |

Evidence:

```
s564 L2369: st.shared.v2.u16 [%r48+-176], {%rs139, %rs140}
s588 L2395: st.shared.u32 [%r53], %r5
s1459 L3450: ld.param.u64 %rd114, [%rd1+-56]
s1613 L3615: ld.global.v2.u16 {%rs287, %rs288}, [%rd116]
s4386 L6554: cvt.rn.f16x2.e4m3x2 %r4074, %rs287
s4484 L6844: {add.f16 %rs425,%rs426,%rs427;}
s4674 L7322: {mul.f16 %rs716,%rs428,%rs428;}        // square
s4849 L7865: shfl.sync.bfly.b32 %r4247,%r4243,%r14056,%r4250,%r14067
s5030 L8337: mov.f64 %fd770, 0d3F20000000000000
s5180 L8783: rsqrt.approx.ftz.f32 fl, fl
s5566 L9408: ld.global.v2.u16 {%rs6669, %rs6670}, [%rd241+8464]
s5860 L10282: mov.b32 %r9539, {%rs1902, %rs1903}
s5934 L10451: ld.weak.global.ca.v4.u32 { %r4647,%r4648,%r4649,%r4650},[%rd157]
```

**What it computes.**

1. **Requantise phase 1's D** and stage it into `smem` as a 10×10 grid of
   `f16x2` pairs (the stores step by ±16, ±160, ±1600) — a padded window patch.
2. **Read it back** as a shifted 24-load window (s805–s1457) — the receptive
   field of the next phase.
3. **Read the skip input** at `param_0+24`, `ld.global.v2.u16` pairs, and
   dequantise it with `cvt.rn.f16x2.e4m3x2` (E1 is the only epilogue that uses
   that op).
4. **Add** the two (204 `add.f16`) — the values live on: the sums `%rs425` /
   `%rs428` defined at s4484/s4485 are still read in **E5** (s10118, L22027) and in
   E19.
5. **RMS norm**, exactly as enc0's E12: sum of squares (`mul.f16`), in-lane
   `add.f16x2` trees, `shfl.sync.bfly` across lanes, `+ 2⁻¹³`, `rsqrt`, multiply
   by the per-column gain read from `W+8448…8496` (4 × `v2.u16` = 8 f16 per
   lane = 32 columns).  There is **no mean subtraction and no `1/N`**.
6. **Repack** the normalised values (48 packs, s5860–s5928) as phases 2 and 3's
   A operands, and load phase 2's B (`W+8512`, `W+9024`).

**Outputs:** phases 2/3's A and phase 2's B.  **Memory:** 24 global skip loads,
24 shared loads, 4 gain loads, 40 shared stores, 2 weight loads.

---

### E2 — s5961–s5967: next-B weight loads

```
s5961 L10626: mov.u32 %r4895, %laneid
s5963 L10629: add.s64 %rd245, %rd2, %rd244
s5964 L10630: add.s64 %rd159, %rd245, 9536
s5965 L10632: ld.weak.global.ca.v4.u32 { %r4896,…},[%rd159]   // +9536
s5966 L10634: add.s64 %rd160, %rd245, 10048
s5967 L10636: ld.weak.global.ca.v4.u32 { %r4900,…},[%rd160]   // +10048
```

Two 16-byte/lane fragments of `W + 16*laneid + {9536, 10048}` = phase 3's B
(weight block `[9536, 10060)` per the phase table).  Nothing else.

---

### E3 — s5992–s6162: phase 4's C-seed loads + pack of phase 3's D

171 statements, 24 weight loads + 80 packs (identical shape to enc5's E10 and
enc0's E9):

```
s5996 L10813: ld.weak.global.ca.v4.u32 { %r5145,…},[%rd161]   // +10560
   …
s6042 L10905: ld.weak.global.ca.v4.u32 { %r5237,…},[%rd184]   // +22336
s6043 L10908: cvt.rn.satfinite.e4m3x2.f16x2 %rs1951, %r4675
```

* **24 loads** at `W + 16*laneid + 10560 + 512*i`, `i = 0…23` — phase 4's
  score-bias C seeds, byte block `[10560, 22348)` = 12288 bytes = 6144 f16,
  tile index `4*(n>>1) + m`, stride 512.
* **80 packs / 40 b32** (s6043–s6162) whose sources are phase 3's D registers
  (`%r4655`, `%r4675`, …) → phase 4's A and B operands.

**Memory:** 24 weight loads, nothing else.

---

### E4 — s6211–s10053: softmax of the 64×96 score block

**Phase 4** = 48 `mma` (s6163–s6210), D registers 96 f16x2 (64 rows × 96
columns).  The epilogue is byte-for-byte the same code as enc0's E5 and
enc5's E11.  Sub-sections:

| range | stmts | what |
|---|---|---|
| s6211–s8328 | 2118 | 96 per-element groups: scale, clamp, cubic, exponent extract |
| s8329–s8412 | 84 | in-lane partial row-sum trees |
| s8413–s8539 | 127 | 16 `shfl.sync.bfly` → 8 row sums (`s8415 L18394` … `s8536 L18698`) |
| s8543–s9308 | 766 | 192 `rcp.approx.ftz.f32` → 96 reciprocals (`s8545 L18720`) |
| s9312–s9788 | 477 | 192 `mul.f16` → 96 probabilities (`s9312 L19965`) |
| s9790–s9837 | 48 | `movmatrix.sync.trans.aligned.m8n8.b16` (`s9790 L21019`) |
| s9838–s10052 | 215 | 144 packs (`s9838 L21163`) |

#### (a) per-element exponent, s6211–s8328

```
s6212 L11526: mov.f32 %f?, 0f3C8CCB50
s6215 L11534: mul.f16x2 %r?, %r?, %r?
s6220 L11548: max.f16x2 %r?,%r?,%r?
s6225 L11562: min.f16x2 %r?,%r?,%r?
s6231 L11580: fma.rn.f16x2 %r?,%r?,%r?,%r?
s6237 L11597: shl.b32 %r?, %r?, 5
s6238 L11598: and.b32 %r?, %r?, 2145419232      // 0x7FE07FE0
```

Per f16 half `x`:

```
m  = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)
u  = f16(1.375 + m*(0.92724609375 - m*m))
expval = f16 from bits ((u_bits << 5) & 0x7FE07FE0)   // 2^(32·frac(u) − 47)
```

The same cubic-exponent trick as enc5's E11 and enc0's E5; **not `exp`**.  The
clamp bounds are the guard (`d/dm[0.92724609375m − m³]` vanishes at
`m = 0.55615…`).

#### (b) in-lane partial sums, s8329–s8412

`add.f16x2` trees reduce each lane's 96 `expval` values to 24 partial sums in
12 f16x2 registers.

#### (c) cross-lane row sums, s8413–s8539

```
s8415 L18394: shfl.sync.bfly.b32 %r?,%r?,1,%r?,-1
s8424 L18418: add.f16 %rs?,%rs?,%rs?
```

Eight times: add the two f16x2 partials, butterfly over lane bits 0 and 1, add
the halves, broadcast.  **8 row sums**, one per row `g, g+8` of the four
m-tiles.  **There is no row maximum in this epilogue**; `max.f16x2`/`min.f16x2`
(96 each) are the clamp of (a).

#### (d)–(g)

192 `rcp.approx.ftz.f32` → 96 reciprocals; 192 `mul.f16` → 96 probabilities
`p = expval/rowsum`; 48 `movmatrix` transpose phase 3's D (the V matrix); 144
packs → phase 5's A and B.  **Inputs:** phase 4's 96 D, phase 3's 48 D.
**Outputs:** phase 5's operands.  **Memory:** none.

---

### E5 — s10102–s10261: residual + per-column table, then pack

160 statements.  This is the twin of enc0's E6.

```
s10106 L22009: ld.weak.global.ca.v4.u32 { %r9048,…},[%rd185]   // +22848
s10108 L22013: ld.weak.global.ca.v4.u32 { %r9052,…},[%rd186]   // +23360
s10114 L22022: ld.global.v2.u16 {%rs6677, %rs6678}, [%rd252+39248]
s10115 L22023: ld.global.v2.u16 {%rs6679, %rs6680}, [%rd252+39264]
s10116 L22024: ld.global.v2.u16 {%rs6681, %rs6682}, [%rd252+39280]
s10117 L22025: ld.global.v2.u16 {%rs6683, %rs6684}, [%rd252+39232]
s10118 L22027: add.f16 %rs3545,%rs6684,%rs428
s10214 L22315: cvt.rn.satfinite.e4m3x2.f16x2 %rs?, %r?
```

1. **Next-B loads** (s10106–s10108): two `v4.u32` fragments from
   `W + 16*laneid + {22848, 23360}` → phase 6's B.
2. **Table loads** (s10110–s10117): four `ld.global.v2.u16` from
   `W + ((laneid<<2)&12) + {39232, 39248, 39264, 39280}` → 8 f16 = 32 columns.
3. **Residual + table add** (s10118–s10212): 64 `add.f16`, i.e. 32 f16x2, each
   `table[2t{,+1}] + %rs425/%rs428`.  `%rs425`/`%rs428` are the skip-input sums
   **defined in E1** (s4484/s4485, L6844/L6848); they have no other
   definition in the entry (`grep -n "%rs428\b"` returns lines 6848, 7322,
   9845, 22027).  The 32 resulting registers are **phase 6's C operand**, so
   phase 6 computes `D6 = A6·B6 + (skip + column_table)`.
4. **Pack** (s10214–s10261): 32 packs of phase 5's D (the banded P·V output)
   → 16 b32 = phase 6's A.

**Memory:** 2 weight loads + 4 table reads.

---

### E6 / E7 — s10278–s10284 and s10309–s10315: next-B weight loads

Two 7-statement runs, each two `ld.weak.global.ca.v4.u32`:

* E6: `W + 16*laneid + {23872, 24384}` (s10282 L22545, s10284 L22549) →
  phase 7's B (weight block `[23872, 24396)`).
* E7: `W + 16*laneid + {24896, 25408}` (s10313 L22726, s10315 L22730) →
  phase 8's B (`[24896, 25420)`).

---

### E8 — s10340–s10510: phase 9's C-seed loads + pack of phase 8's D

Identical to E3 with the second window's offsets: 24 loads at
`W + 16*laneid + 25920 + 512*i` (s10344 L22907 … s10390 L22999) = the byte
block `[25920, 37708)`, and 80 packs of phase 8's D (s10391 L23002 onward) →
phase 9's operands.

---

### E9 — s10559–s14396: softmax, second window

**Identical code to E4** with shifted registers; the op histogram is the same
(1048 `mov.b32`, 672 `cvt.rn.f16.f32`, 384 `cvt.rn.f16.f64`, 192 each
`fma.rn.f16x2` / `cvt.f32.f16` / `rcp.approx.ftz.f32` / `mul.f16`, 144 packs,
48 `movmatrix`).  Sub-section boundaries: exponent groups s10559–s12671, in-lane
sums s12672–s12755, 16 `shfl.sync.bfly` s12756–s12885, 192 `rcp.approx`
s12886–s13653, 192 `mul.f16` s13655–s14131, 48 `movmatrix` s14133–s14180, 144
packs s14181–s14395.

**Inputs:** phase 9's 96 D, phase 8's 48 D.  **Outputs:** phase 10's operands.
**Memory:** none.

#### Shape of the attention region (E1–E11)

```
P1  A = input buffer (param+8)   B = W+0     C = bias@8192      → D1   (48×128, K=64)
     E1 packs D1 into smem, dequantises the skip (param+24), RMS-norms, packs
P2  A = E1's normed tile  B = W+8512   C = 0        → D2  (96×32)
P3  A = E1's normed tile  B = W+9536   C = 0        → D3
     E3 packs D3 → P4 A, loads P4 C = bias@10560 (12288 B)
P4  A = pack(D3)  B = pack(D2)  C = bias@10560   → scores 64×96
     E4: softmax over the 96 columns (no row max), 48 movmatrix(D2), packs → P5
P5  A = pack(P)   B = pack(transpose(D2))  C = 0  → D5 (64×32, K=96)
     E5: pack → P6 A; P6 C = skip(E1) + column table@39232; loads P6 B
P6  A = pack(D5)  B = W+22848  C = skip+table  → D6 (64×32)
P7  A = E1's normed tile  B = W+23872  C = 0  → D7  (96×32)
P8  A = E1's normed tile  B = W+24896  C = 0  → D8
     E8 packs D8 → P9 A, loads P9 C = bias@25920 (12288 B)
P9  A = pack(D8)  B = pack(D7)  C = bias@25920  → scores 64×96
     E9: same softmax + transpose(D7) + packs → P10
P10 A = pack(P')  B = pack(transpose(D7))  C = 0  → D10 (64×32, K=96)
     E10 packs D10 → P11 A
P11 A = pack(D10)  B = W+38208  C = D6 (residual)  → D11 (64×32)
```

Window 2 therefore mirrors window 1 except that P9's two operands both come out
of P8 (E8), while the transpose of P7 supplies P10 (E9).

---

### E10 — s14445–s14499: next-B loads + pack

```
s14449 L34098: ld.weak.global.ca.v4.u32 { %r13651,…},[%rd215]   // +38208
s14451 L34102: ld.weak.global.ca.v4.u32 { %r13655,…},[%rd216]   // +38720
s14452 L34105: cvt.rn.satfinite.e4m3x2.f16x2 %rs5359, %r13270
```

* two weight fragments at `W + 16*laneid + {38208, 38720}` → phase 11's B;
* **32 packs** of phase 10's D (s14452–s14498) → 16 b32 = phase 11's A.

**Unlike E5, no table**: phase 11's C is phase 6's D (`%r9097,%r9098` etc. —
`rr_layer_spec.py`'s C-in list for the cut contains `%r13609`-style pairs and
the residual D6), so phase 11 computes `D11 = A11·B11 + D6`.

---

### E11 — s14516–s15402: RMS norm before the MLP

887 statements, no stores.  Structure:

| range | op |
|---|---|
| s14516–s14643 | 192 `mul.f16` (squares of phase 11's D; first `s14517 L34330`) |
| s14644–s14659 | 16 `add.f16x2` (in-lane pairs) |
| s14660–s14783 | 16 `shfl.sync.bfly.b32` (`s14666 L34727` … `s14780`) + `add.f16` |
| s14784 | `cvt.rn.f16.f64` of `%fd770` = `2⁻¹³` |
| s14881–s15136 | 64 `cvt.f32.f16` + 64 `rsqrt.approx.ftz.f32` (`s14885 L35337`) + 64 `cvt.rn.f16.f32` |
| s15140–s15146 | gain loads `W + ((laneid<<2)&12) + {39296, 39312, 39328, 39344}` |
| s15147–s15337 | 192 `mul.f16` (`inv · gain` then `x · (inv·gain)`) |
| s15340–s15345 | next-B loads `W + 16*laneid + {39360, 39872}` |
| s15346–s15354 | next-C loads `W + ((laneid<<2)&12) + {40384, 40400, 40416, 40432}` |
| s15355–s15401 | 32 packs → phase 12's A |

Evidence:

```
s14517 L34330: mul.f16 %rs5393,%rs6220,%rs6220
s14666 L34727: shfl.sync.bfly.b32 %r13904,%r13900,%r14056,%r13907,%r14067
s14885 L35337: rsqrt.approx.ftz.f32 fl, fl
s15143 L35754: ld.global.v2.u16 {%rs6685, %rs6686}, [%rd263+39312]
s15355 L36359: cvt.rn.satfinite.e4m3x2.f16x2 %rs6184, %r18145
```

**What it computes.**  For each row a lane holds:

```
sumsq[row] = Σ_c x[row,c]²                        (f16, butterfly over 4 lanes)
inv        = rsqrt.approx.ftz.f32(f32(f16(sumsq + 2⁻¹³)))   // rounded to f16
out[row,c] = x[row,c] * inv * gain[c]
```

with **no mean subtraction and no `1/N`**; `gain[c]` is read per column from
the 64-byte table at `W+39296…39344`.  The norm's bias is not added here — it is
phase 12's C operand, loaded at the end of this epilogue (`+40384…+40432`).

**Outputs:** phase 12's B (`%r14138…`) and C (`%r14308 %r14318 %r14328
%r14338`), 16 packed A registers, and the 32 f16x2 normalised values that E13,
E15 and E17 repack.

---

### E12 — s15419–s16127: clamped cubic activation + column table + pack

709 statements.  Histogram: 128 `cvt.rn.f16.f32`, 176 `mov.b32`, 32 each
`neg/max/min/abs.f16x2`, 96 `mul.f16x2`, 32 `sub.f16x2`, 32 `add.f16x2`,
2 `ld.weak`, 4 `ld.global.v2.u16`, 64 `add.f16`, 32 packs.

```
s15420 L36585: mov.f32 %f?, 0f3ED306EB
s15432 L36614: neg.f16x2 %r?,%r?
s15433 L36618: max.f16x2 %r?,%r?,%r?
s15434 L36622: min.f16x2 %r?,%r?,%r?
s15435 L36626: abs.f16x2 %r?,%r?
s15436 L36630: mul.f16x2 %r?,%r?,%r?
s15437 L36634: sub.f16x2 %r?,%r?,%r?
s15439 L36642: add.f16x2 %r?,%r?,%r?
s15972 L38516: ld.weak.global.ca.v4.u32 { %r15269,…},[%rd219]   // +40448
s15974 L38520: ld.weak.global.ca.v4.u32 { %r15273,…},[%rd220]   // +40960
s15980 L38529: ld.global.v2.u16 {%rs6693, %rs6694}, [%rd273+41488]
   … +41504, +41520, +41472
s15984 L38534: add.f16 %rs?,%rs?,%rs?
s16080 L38822: cvt.rn.satfinite.e4m3x2.f16x2 %rs?, %r?
```

Per element, over each of the 32 f16x2 D registers of phase 12:

```
y   = clamp(x, -2, +2)
g   = 0.5 + y * (0.412109375 - 0.0810546875 * |y|)      // f16 throughout
out = x * g
```

`g` is a *cubic ramp*, not a sigmoid: `g(+2) = 1` ⇒ `out = x` for `x ≥ 2`;
`g(-2) = 0` ⇒ `out = 0` for `x ≤ -2`; `g(1) = 0.831055` where
`x·sigmoid(x) = 0.730957`.  **Reproduce the formula, not SiLU/GELU.**

Then: 4 `ld.global.v2.u16` at `W + ((laneid<<2)&12) + {41472, 41488, 41504,
41520}` (64 bytes, 32 columns), 64 `add.f16` = `table[column] + D12` (written
over the activation inputs, s15984 L38534), 32 packs of the **activation
outputs** → phase 13's A, and 2 weight loads at `W+40448/+40960` → phase 13's B.

**Outputs:** phase 13's A, B and C seeds.

---

### E13 / E15 / E17 — 64 statements each: next B/C loads + pack

Three identical runs:

| run | range | B loads | C loads (4 × `ld.global.u32`) |
|---|---|---|---|
| E13 | s16144–s16207 | `+40448`, `+40960` (s16148 L39052, s16150 L39056) | `+42560/42576/42592/42608` (s16156 L39065 …) |
| E15 | s16840–s16903 | `+42624`, `+43136` (s16844 L41460, s16846 L41464) | `+44672/44688/44704/44720` (s16852 L41473 …) |
| E17 | s17536–s17599 | `+44736`, `+45248` (s17540 L43868, s17542 L43872) | `+46784/46800/46816/46832` (s17548 L43881 …) |

Each loads the next phase's B and C and then **32 packs of the normalised tile
that E11 produced** (the source register has its newest definition at
s15355–s15401), so phases 13, 15 and 17 all take the same A operand — three
parallel GEMMs off one normed tile.

---

### E14 / E16 / E18 — 600 statements each: clamped cubic activation + pack

Three identical runs (E14 s16224–s16823, E16 s16920–s17519, E18
s17616–s18215), histogram in each: 128 `cvt.rn.f16.f32`, 144 `mov.b32`, 32 each
`neg/max/min/abs.f16x2`, 96 `mul.f16x2`, 32 `sub.f16x2`, 32 `add.f16x2`,
2 `ld.weak`, 32 packs — and **no `ld.global` and no table add**.

```
s16225 L39298: cvt.rn.f16.f32 low, %f…
s16233 L39321: neg.f16x2 %r?,%r?
s16234 L39325: max.f16x2 %r?,%r?,%r?          // phase 14's D
s16770 L41219: mul.wide.u32 …
s16773 L41223: ld.weak.global.ca.v4.u32 {…},[%rd223]    // +42624 (phase 15's B)
s16776 L41230: cvt.rn.satfinite.e4m3x2.f16x2 %rs?, %r?
```

Same clamped cubic as E12, applied to the phase's own D; the 32 packs are the
activation outputs → the next phase's A; the 2 weight loads are the next
phase's B.

#### Shape of the whole MLP region (E11–E18)

```
E11  normed tile N (32 f16x2)
 ├─ pack → phase 13 A   (s15355)   phase 13: A=N, B=W13, C=bias@40384  → activation (E12: +table@41472)
 │                                              pack of activation → phase 14 A (s16080)
 │                                              phase 14: C = D13        → D14
 ├─ pack → phase 15 A   (s16160)   phase 15: A=N, B=W15, C=bias@42560  → activation (E14)
 │                                              pack of activation → phase 16 A (s16776)
 │                                              phase 16: C = D15        → D16
 └─ pack → phase 17 A   (s17552)   phase 17: A=N, B=W17, C=bias@44672  → activation (E16)
                                                pack of activation → phase 18 A (s18168)
                                                phase 18: C = D17        → D18 (E18)
```

Only the *first* of the three branches (phase 13's) has a per-column table add
(E12, `W+41472…41520`); the other two activations are pure cubic + pack.  The
three branches share one input tile and fold into a running residual
(`D14 = … + D13`, `D16 = … + D15`, `D18 = … + D17`).

---

### E19 — s18232–s19160: the final epilogue

929 statements; it consumes the cut's returned accumulators (`%r13609`,
`%r14067`, `%r19306`, `%r19316`, …, listed by `rr_layer_spec.py`).

#### (a) cross-lane row max, s18232–s18474

```
s18232 L46270: mov.u32 %r19498, %laneid
s18233 L46272: and.b32 %r19867, %r19498, 7              // row key = laneid & 7
s18234 L46273: shr.u32 %r19868, %r19498, 3              // column group
s18236 L46275: setp.eq.s32 %p294, %r19869, 1
s18237 L46276: selp.b32 %r19870, %r19307, %r19306, %p294
s18253 L46292: bfi.b32 %r19883, %r19867, %r19882, 2, 3
s18254 L46293: shfl.sync.idx.b32 %r19884|%p298, %r19872, %r19883, %r19524, %r14067
s18268 L46307: selp.b32 %r19897, %r19888, %r19884, %p302
```

Two identical blocks (rows `g` from the D pairs `{r19306,r19307}`, … and rows
`g+8` from `{r19346,r19347}`, …, s18353 L46393 onwards), each doing:

1. a `selp` chain over `laneid&3` selects one of the four D pairs into
   `%r19872/%r19875/%r19878/%r19881` (the four n-tiles of the row segment);
2. four `shfl.sync.idx.b32` with
   `srcLane = (laneid&7) | ((((laneid>>3)+d)&3)<<2)` for `d = 0, 1, 2, -1`;
3. a `selp` chain over `laneid>>3` picks between the four shuffled values.

Result: `%r368…%r371` (row `g`) and `%r372…%r375` (row `g+8`) hold the per-lane
row maximum over the 8 columns.  The `max.f16x2` / `max.f16` chain at
s18605–s18611 then reduces them and `s18612 L46696 sub.f16x2 %r20095,%r368,
%r20109` subtracts the row max: **this is a real row max**, done with
`shfl.sync.idx`, unlike E4's.

#### (b) bounds check, s18475–s18490

```
s18483 L46524: ld.param.v2.u32 {%r20073, %r20074}, [%rd1+-80]     // param_0+0
s18484 L46525: setp.ge.u32 %p342, %r391, %r20073                  // x >= ex
s18486 L46527: or.pred %p344, %p342, %p343
s18490 L46531: @%p346 bra $L__BB1_367
```

Out-of-range lanes jump to `$L__BB1_367` and skip (c)…(e).

#### (c) surface 1 write (`param_0+48`), s18491–s18502

```
s18493 L46535: ld.param.u64 %rd319, [%rd357+-32]
s18494 L46536: cvta.to.global.u64 %rd320, %rd319
s18495 L46537: mad.lo.s32 %r20132, %r20073, %r392, %r391
s18497 L46539: add.s64 %rd322, %rd320, %rd321
s18498 L46541: cvt.rn.satfinite.e4m3x2.f16x2 %rs6704, %r378
s18502 L46552: st.global.v4.u16 [%rd322], {%rs6701, %rs6702, %rs6703, %rs6704}
```

Four packs of `%r373/%r376/%r377/%r378` and one `st.global.v4.u16` at
`4*(ex*y + x)` bytes — the e4m3 surface.  The same block repeats for the second
row group at s18835–s18847 (line 47192).

#### (d) per-element `abs`/`ex2` evaluation, s18503–s18601

```
s18504 L46556: cvt.f32.f16 %f1473, low
s18511 L46577: abs.ftz.f32 %f1521, %f1473
s18512 L46578: mul.ftz.f32 %f1522, %f1521, 0f4038AA3B
s18513 L46579: ex2.approx.ftz.f32 %f1523, %f1522
s18514 L46580: add.ftz.f32 %f1524, %f1523, 0f3F800000
s18516 L46582: rcp.approx.ftz.f32 %f1525, %f1524
s18518 L46584: fma.rn.ftz.f32 %f1527, %f1525, %f1526, %f1520   // f1526 = -2.0
s18519 L46585: setp.ge.ftz.f32 %p347, %f1521, 0f41102CB4
s18520 L46586: selp.f32 %f1528, 0f3F800000, %f1527, %p347
s18526 L46592: mul.ftz.f32 %f1530, %f1473, %f1473
s18529 L46595: fma.rn.ftz.f32 %f1533, %f1532, %f1530, %f1531
   …
s18599 L46665: fma.rn.ftz.f32 %f1582, %f1581, %f1476, %f1476
s18600 L46666: setp.ge.ftz.f32 %p354, %f1569, 0f3F19999A
s18601 L46667: selp.f32 %f1480, %f1576, %f1582, %p354
```

Per element `v`: a `1 - 2/(1+2^(|v|·log2e))` branch (with a hard `1.0` above
`0f41102CB4` = 9.0109…) and a polynomial branch (`fma` chain over `v²`) selected
by `|v| ≥ 0f3F19999A` = 0.6.

#### (e) surface 2 write (`param_0+144`, `sust.p`), s18602–s18603

```
s18602 L46668: ld.param.u64 %rd295, [%rd357+64]
s18603 L46670: sust.p.2d.v4.b32.zero [%rd295, {%r391,%r392}], {%f1477,%f1478,%f1479,%f1480}
```

Four f32 channels `%f1477…%f1480` written at integer coordinates `(x, y)`.
Repeated at s18947–s18948 (line 47310) for the second row group.

#### (f) row softmax on the f16x2 values, s18605–s18679

```
s18605 L46674: max.f16x2 %r20086,%r368,%r369
s18609 L46687: max.f16 %rs6705,%rs6706,%rs6707
s18612 L46696: sub.f16x2 %r20095,%r368,%r20109
s18615 L46704: mul.ftz.f32 %f1583, %f1481, 0f3FB8AA3B
s18616 L46705: ex2.approx.ftz.f32 %f1483, %f1583
s18672 L46824: div.approx.ftz.f32 %f1499, %f1520, %f1600   // 1/rowsum
s18675 L46831: mul.f16x2 %r20110,%r20111,%r20124
   … five f16x2 multiplies (s18675–s18679)
```

A real row max, `x − max`, `2^((x−max)·log2e)` in f32, an f32 row sum,
`div.approx.ftz.f32` with 1.0, then five `mul.f16x2` — the same softmax as
enc5's E61(c).

#### (g) texture reads and surface 3 write, s18680–s19159

```
s18696 L46872: ld.param.v2.f32 {%f1602, %f1603}, [%rd357+56]      // param_0+136
s18697 L46873: mul.ftz.f32 %f1606, %f1602, 0f3F000000
s18698 L46874: fma.rn.ftz.f32 %f1500, %f1602, %f1601, %f1606      // x coord
s18702 L46878: ld.param.u64 %rd313, [%rd357+8]                     // param_0+88
s18704 L46881: tex.base.2d.v4.f16.f32 {%rs6733,…}, [%rd297, {%f1500,%f1501}]
s18706 L46885: fma.rn.f16 %rs6737,%rs6733,%rs6747,%rs6730
   …
s18778 L47053: ld.param.u64 %rd315, [%rd357+16]                    // param_0+96
s18779 L47055: tex.base.2d.v4.f16.s32 {%rs6877,%rs6878,%rs6879,%rs6880}, [%rd315, {%r391,%r392}]
s18780 L47057: ld.param.u64 %rd317, [%rd357+24]                    // param_0+104
s18781 L47059: tex.base.2d.v4.f16.s32 {%rs6881,%rs6882,%rs6883,%rs6884}, [%rd317, {%r391,%r392}]
s18782 L47062: cvt.f32.f16 %f1518, %rs98
s18783 L47065: mul.ftz.f32 %f1613, %f1518, 0fBFB8AA3B         // -log2e
s18784 L47066: ex2.approx.ftz.f32 %f1614, %f1613
s18785 L47067: add.ftz.f32 %f1615, %f1614, 0f3F800000
s18786 L47068: div.approx.ftz.f32 %f1519, %f1520, %f1615      // 1/(1+2^(-v·log2e))
s18789 L47078: sub.f16 %rs6888,%rs6887,%rs6886
s18790 L47082: mul.f16 %rs6891,%rs6865,%rs6886
s18791 L47086: fma.rn.f16 %rs7167,%rs6877,%rs6888,%rs6891
s18796 L47106: set.nan.f16.f16 %rs6912,%rs6881,%rs6881
s18797 L47109: setp.ne.s16 %p367, %rs6912, 0
s18798 L47110: @%p367 bra $L__BB1_362
s18799 L47113: add.f16 %rs7167,%rs7167,%rs6881
s18811 L47144: shl.b32 %r20167, %r391, 3
s18812 L47145: mov.u16 %rs6933, 0
s18813 L47147: sust.b.2d.v4.b16.zero [%rd323, {%r20167,%r392}], {%rs7167,%rs7168,%rs7169,%rs6933}
```

**Eighteen** `tex.base.2d.v4.f16.f32` sample `param_0+88` at *floating-point*
coordinates derived from `param_0+136` (a `v2.f32` scale/origin, `x = 2*…·(n−1)
+ w/2` style, s18697–s18701), and two `tex.base.2d.v4.f16.s32` sample
`param_0+96` and `param_0+104` at the *integer* coordinates `(x, y)`.  The
sampled values are folded into `%rs7167…%rs7169` with `fma.rn.f16`, a `NaN`
guard is applied (`set.nan.f16.f16` / `setp.ne.s16` — if the sampled value is
NaN the add is skipped, s18796–s18810), and the result is written by
**`sust.b.2d.v4.b16.zero [%rd323, {%r20167,%r392}], {%rs7167,%rs7168,
%rs7169, 0}`** where `%rd323` = `param_0+112` (s18604 L46672) — the fourth
channel is the constant `0`.  The whole tail is repeated for the second row
group at s18838–s19158 (lines 47175–47787).

**Inputs:** the returned accumulators, `param+0/+8/+24` (E1), `W`, `smem`,
`param+0/+48/+88/+96/+104/+112/+136/+144`.
**Outputs:** none (the kernel returns at s19160 L47791).

## 4. The parameter block, and the surfaces

`rr_layer_spec.py dec0` reports a **152-byte parameter block with 3 `ld.param`
over 3 distinct offsets `[8, 40, 80]`**; that count only covers the literal
`[param_0+N]` forms.  The complete set of `ld.param` in the entry (22
instructions) resolves, via `%rd1 = param_0+80` (s4) and `%rd357 = param_0+80`
(s18492), to:

| offset | type | statements | what it supplies |
|---|---|---|---|
| +0 | `v2.u32` | s13 L1041, s18483 L46524 | `(ex, ey)`, the extents |
| +8 | `u64` | s29 L1057 | the **input buffer**, read by the 24 prologue `ld.global.u32` (s67 L1097 … s270 L1300) that form phase 1's A (24 loads = its 6 A quads × 4 registers); never read again |
| +24 | `u64` | s1459 L3450 | the **skip input** |
| +40 | `u64` | s11 L1039 | `W`, the weight image |
| +48 | `u64` | s18493 L46535, s18838 L47175 | **output e4m3 surface** (`st.global.v4.u16` at s18502 L46552 and s18847 L47192) |
| +80 | `v2.u16` | s5 L1033 | block origin `(%rs235, %rs236)` |
| +88 | `u64` | s18702 L46878, s19047 L47518 | **texture A**, the handle of the 18 `tex.base.2d.v4.f16.f32` (s18704 L46881 … s18774 L47039) |
| +96 | `u64` | s18778 L47053, s19123 L47693 | **texture B**, the handle of the first `tex.base.2d.v4.f16.s32` (s18779 L47055) |
| +104 | `u64` | s18780 L47057, s19125 L47697 | **texture C**, the handle of the second `tex.base.2d.v4.f16.s32` (s18781 L47059) |
| +112 | `u64` | s18604 L46672, s18949 L47312 | **surface A**, the `sust.b.2d.v4.b16.zero` descriptor (s18813 L47147, s19158) |
| +136 | `v2.f32` | s18696 L46872, s19041 L47512 | the texture coordinate scale/origin used to build the f32 coords (s18697–s18701) |
| +144 | `u64` | s18602 L46668, s18947 L47308 | **surface B**, the `sust.p.2d.v4.b32.zero` descriptor (s18603 L46670, s18948 L47310) |

### 4.1 The two texture reads at `+96`/`+104`

Both are `u64` texture handles, loaded once per row group, and both feed
`tex.base.2d.v4.f16.s32` **with integer coordinates** (`{%r391, %r392}`, the
tile's `x`/`y`):

* `+96` → `%rd315` at s18778 L47053 → `tex.base.2d.v4.f16.s32 {%rs6877, %rs6878,
  %rs6879, %rs6880}, [%rd315, {%r391,%r392}]` at **s18779 L47055** (and the
  second row group at s19124).
* `+104` → `%rd317` at s18780 L47057 → `tex.base.2d.v4.f16.s32 {%rs6881,
  %rs6882, %rs6883, %rs6884}, [%rd317, {%r391,%r392}]` at **s18781 L47059**
  (second row group at s19126).

Their results are consumed immediately:
`%rs6881…%rs6884` (from `+104`) are the second term of the `sub.f16 %rs6888,
%rs6887,%rs6886` / `add.f16 %rs7167,%rs7167,%rs6881` path (s18789–s18799), and
`%rs6883`/`%rs6884` are the third channel folded into `%rs7169`;  `%rs6877…`
(from `+96`) are the weights of the `fma.rn.f16` at s18791–s18795.  Both are
"sample the same pixel as the surfaceless input" reads, which is why they use
integer coordinates.

### 4.2 The surface writes: `+112` and `+144` (there is no `+120` or `+128`)

A grep of every `st.global` / `sust` in the entry returns exactly **six**
instructions:

| statement | instruction | descriptor | offset |
|---|---|---|---|
| s18502 L46552 | `st.global.v4.u16 [%rd322], {%rs6701,%rs6702,%rs6703,%rs6704}` | `%rd322 = cvta(param_0+48) + 8*(ex*y+x)` | +48 |
| s18847 L47192 | `st.global.v4.u16 [%rd352], {%rs6934,%rs6935,%rs6936,%rs6937}` | same, second row group | +48 |
| s18603 L46670 | `sust.p.2d.v4.b32.zero [%rd295, {%r391,%r392}], {%f1477,%f1478,%f1479,%f1480}` | `%rd295 = param_0+144` | +144 |
| s18948 L47310 | `sust.p.2d.v4.b32.zero [%rd325, {%r395,%r396}], {%f1620,…}` | same, second row group | +144 |
| s18813 L47147 | `sust.b.2d.v4.b16.zero [%rd323, {%r20167,%r392}], {%rs7167,%rs7168,%rs7169,%rs6933}` | `%rd323 = param_0+112` | +112 |
| s19158 L47787 | `sust.b.2d.v4.b16.zero [%rd353, {%r20268,%r396}], {%rs7170,%rs7171,%rs7172,%rs7166}` | same, second row group | +112 |

So there are **three surface writes per row group** — `st.global.v4.u16` to
`param_0+48`, `sust.p.2d.v4.b32.zero` to `param_0+144`, and
`sust.b.2d.v4.b16.zero` to `param_0+112` — consuming respectively the
requantised accumulators (s18498–s18501), the four `%f1477…%f1480` of §3/E19(d),
and the texture-fused `%rs7167…%rs7169`.  **`param_0+120` and `param_0+128` are
not read anywhere in the entry** (`grep -n 'ld\.param'` returns offsets 0, 8,
24, 40, 48, 80, 88, 96, 104, 112, 136, 144 and nothing else), so the write set
is `{+48, +112, +144}`; see §6 U1.

### 4.3 What the skip input at `+24` supplies

`param_0+24` is loaded **once**, at s1459 L3450:
`ld.param.u64 %rd114, [%rd1+-56]` (`%rd1 = param_0+80` ⇒ `param_0+24`), then
`cvta.to.global.u64 %rd4, %rd114` (s1460 L3451).  `%rd4` is the base of **24
`ld.global.v2.u16`** loads, all at `[%rd4 + mul.wide(%rX, 4)]`:

* s1613 L3615, s1761 L3772, s1911 L3931, s2058 L4087, s2208 L4246, s2355 L4402,
  s2505 L4561, s2652 L4717, s2802 L4876, s2949 L5032, s3099 L5191, s3246 L5347,
  s3396 L5506, s3543 L5662, s3693 L5821, s3840 L5977, s3960 L6104, s4078 L6229,
  s4166 L6322, s4253 L6414, s4308 L6472 (21 loads), plus
  s4357 L6524, s4380 L6547, s4384 L6551 (3 more).

Each pair is dequantised by `cvt.rn.f16x2.e4m3x2` (96 of them, s4386 L6554 …
s4481) and added to the value read back from the staged shared window by the
204 `add.f16` at s4484 L6844 … s5174.  So **the skip input at `+24` supplies a
per-position e4m3 feature vector that is added to the staged patch before the
RMS norm**; the sum `%rs425/%rs428` computed at s4484/s4485 (L6844/L6848) is then used

* as the square summand in E1 (`mul.f16` at s4674, L7322),
* as the norm's multiplicand in E1 (`mul.f16` at s9845),
* and as **the residual seed of phase 6's C operand in E5**
  (`add.f16 %rs3545,%rs6684,%rs428` at s10118 L22027; `grep -n "%rs428\b"`
  finds only lines 6848, 7322, 9845, 22027, so there is no second definition).

It is **not** read anywhere else — in particular the final epilogue (E19) never
mentions `param_0+24`.

## 5. The plane arena

There is no plane-arena *write* in this entry: `param_0+48` is read once
(s18493 L46535, s18838 L47175) and written by the two `st.global.v4.u16` of
§4.2.  The only other global stores are the two `sust` pairs.  So dec0 has no
per-launch table generated into an arena of the kind enc5 builds in its E60;
the six store instructions above are the entry's entire global-write traffic.

## 6. Uncharacterised

**U1. The parameter offsets `+120` and `+128`.**  The brief's "three surface
writes at +112/+120/+128" does not match this corpus: the actual write
descriptors are `param_0+48`, `param_0+112` and `param_0+144` (§4.2), and no
instruction reads `param_0+120` or `param_0+128`.  Either the brief's numbering
uses a different base (e.g. relative to `param_0+80 - 48`, under which `+48 →
96`, `+112 → 160`, `+144 → 192`, still not matching) or it describes a
different revision of the harness.  *Missing:* the harness' parameter-struct
declaration, which would name the fields and settle the numbering.

**U2. `param_0+32`, `+56`, `+96`, `+104`, `+120`, `+128` as *fields*.**  No
`ld.param` in the entry reads offsets `+32`, `+56`, `+120` or `+128`; if the
struct declares them they are unused by this kernel.  *Missing:* the struct
declaration.

**U3. The identity of the two 12288-byte score biases** (`W+10560…22348` for
phase 4, `W+25920…37708` for phase 9) and of the per-column tables
(`W+8448…8496`, `W+39232…39280`, `W+39296…39344`, `W+41472…41520`,
`W+42560…42608`, `W+44672…44720`, `W+46784…46832`).  Their arithmetic role is
established (§3), their names are not.  *Missing:* the weight image's `_prep`
permutation or the model's parameter names.

**U4. The long liveness of `%rs425`/`%rs428`.**  These two registers are
defined once each, inside E1 (s4484 L6844 / s4485 L6848), and read in E1
(s4674 L7322, s5714 L9845) **and in E5** (s10118 L22027) — i.e. the
values cross phases 2, 3 and 4 (and E4's 3843-statement softmax) live.  The PTX
is unambiguous about this, but whether it is a deliberate "add the skip at the
end of the first residual" or a register-allocation accident cannot be decided
from the PTX.  *Missing:* the high-level source.

**U5. E19's two evaluation branches.**  §3/E19(d) computes both a
`1 − 2/(1+2^(|v|·log2e))` form and a `fma` polynomial over `v²`, selected by
`|v| ≥ 0.6` (and, inside the first form, by `|v| ≥ 9.0109…`).  The arithmetic is
exact (s18511–s18601); which model function it is (erf/exp-like tone mapper?) is
not determinable.  *Missing:* the reference implementation.

**U6. Phase 1's true output width.**  Phase 1 is 48×128×64 (§2.2) but its D
registers are 96 (48 chains × 2), and E1 packs them 128 times.  Why 128 packs
for 96 registers — the extra 32 are presumably the second half of a 64-wide
value built from the same D — could not be tied down within this analysis.
*Missing:* a full def-use map of E1's pack destinations.

## 7. Quick acceptance index

| requirement | where |
|---|---|
| phase table with M/N/K checked; phase 1 is 48×128 with K=64 | §2.1, §2.2 |
| weight image extent | §2.3 (47376 bytes) |
| phases that read no weight operand | §2.4 (4, 5, 9, 10) |
| norm (RMS, no mean, `2⁻¹³`, per-column gain) | §3/E11, §3/E1 |
| attention (score, softmax, V-transpose, P·V) | §3/E3, E4, E5, E8, E9, E10 |
| activations (clamped cubic, not GELU) | §3/E12, E14, E16, E18 |
| merge / patch expand | no 8-lane merge (that device appears only in enc0's E20(c)); this layer's per-pixel gather is E1's shared window staging/readback plus E19's texture gather (§3/E1, §3/E19) |
| per-launch table in the plane arena | none in this entry (§5) |
| where the `+96`/`+104` texture reads are consumed | §4.1 |
| the surface writes | §4.2 (`+48`, `+112`, `+144`; no `+120`/`+128`) |
| what the `+24` skip input supplies | §4.3 |
| final epilogue | §3/E19 |
