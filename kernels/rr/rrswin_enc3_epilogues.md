# `cuda_dldn_engine_swin_enc3_kernel` — the epilogues of phases 1…48

Companion to `kernels/rr/rr_layer_spec.py enc3`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0016-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_enc3_kernel`.

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at its first line).  **`LNNNNN` is
  the physical line** of the same instruction in the corpus file, so every claim
  can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and the
  next phase's first `mma`.  The phase table gives the ranges; this file names
  what each one *computes*.
* `W` denotes the prepared weight image: `ld.param.u64 %rd2,
  [cuda_dldn_engine_swin_enc3_kernel_param_0+40]` (s8).  Every
  `ld.weak.global.ca.v4.u32` in an epilogue reads `W + 16*laneid + imm`, and the
  global alias `%rd74 = cvta.to.global.u64 %rd2` (s5564) is what the table loads
  in the epilogues use.
* `S` denotes the plane arena: `param_0+48`, reached as
  `ld.param.u64 %rd556, [%rd1+-32]` with `%rd1 = param_0+80` (`add.s64 %rd1,
  %rd7, 80`, s4) then `cvta.to.global.u64 %rd4, %rd556` (s16148–s16149).
* The C/D register layout and the weight-addressing formula are fixed (see the
  settled notes): `C/D reg0 = (row g, cols 2t,2t+1)`, `reg1 = (row g+8, same cols)`,
  `g = laneid>>2`, `t = laneid&3`.
* `rr_layer_spec.py`'s printed `M`/`N`/`K` come from `geom()`, i.e. from the
  distinct A-fragment count (`M = 16·nA`), the distinct B-fragment count
  (`N = 8·nB`) and the accumulator-chain length (`K = 32·len`).  The phase
  object's own `w_min..w_max`/`n_tiles`/`k_slots` fields use a *different*
  (512-byte-block) convention and disagree with `geom()` on the stem phases; §2
  uses `geom()` and checks the arithmetic explicitly.
* Anything I could not pin down from the PTX is in §6 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `W` (weight image, 315732 bytes) | `ld.param.u64 %rd2, [param_0+40]` | s8 |
| `W`(global alias) | `cvta.to.global.u64 %rd74, %rd2` | s5564 |
| `S` (plane arena, `param+48`) | `ld.param.u64 %rd556, [%rd1+-32]` → `cvta.to.global.u64 %rd4, %rd556` | s16148–s16149 |
| output buffer (`param+56`) | `ld.param.u64 %rd5, [%rd682+56]` | s16581 |
| input plane (`param+8`) | `ld.param.u64 %rd9, [param_0+8]` → `cvta.to.global.u64 %rd3, %rd9` | s16–s17 |
| extents `(ex, ey)` | `ld.param.v2.u32 {%r407, %r408}, [param_0+0]` | s13 |
| block origin `(x0, y0)` | `%r402 = ctaid.x<<3`, `%r1 = %r402 - %rs166`; `%r406 = ctaid.y<<3`, `%r2 = %r406 - %rs167`; `%rs166/%rs167` from `ld.param.v2.u16 [param_0+80]` | s2, s3, s7, s10, s11, s12, s5 |
| `%rd7` = `param_0` | `mov.b64 %rd7, cuda_dldn_engine_swin_enc3_kernel_param_0` | s1 |
| shared arena base `%r4244` | `mov.u32 %r4244, _ZZ33cuda_dldn_engine_swin_enc3_kernel33DldnEngineSwinEncParamsStructBaseE4smem` | s6002 |
| `%r235` = `%r4244 + (tid.z<<11)` | `shl.b32 %r4243, %r12, 11` then `add.s32 %r235, %r4244, %r4243` | s6001, s6003 |
| per-lane weight address | `mul.wide.u32 %rd, %laneid, 16` + `add.s64 %rd, %rd2, %rd` | e.g. s6115–s6117 |
| per-lane *table* address | `shl.b32 %r, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rd, %r, 12` → `add.s64 %rd, %rd74, %rd` = `W + ((laneid<<2)&12)` | e.g. s10558–s10561 |

The last line matters exactly as in enc0: a `ld.global.v2.u16` / `ld.global.u32`
with that addressing reads **the same 16-byte record from the four lanes with
equal `laneid&3`**, so those tables are indexed by column, not by row.

The shadow copy `ld.param.u64 %rd947, [param_0+40]` appears again at s20229 in the
final epilogue — the weight image pointer is re-read, not cached.

### 1.2 Constants that survive between epilogues

| register | first definition | literal | meaning |
|---|---|---|---|
| `%f476` | s6585 `mov.f32 %f476, 0f3C8CCB50` | 0.017186790704727173 | f16 `0x2466` = 0.017181396484375 (softmax input scale) |
| `%fd383` | s6589 `mov.f64 %fd383, 0dBFE1CC0000000000` | −0.55615234375 | f16 clamp lower bound |
| `%fd385` | s6594 `mov.f64 %fd385, 0d3FE1CC0000000000` | +0.55615234375 | f16 clamp upper bound |
| `%f478` | s6600 `mov.f32 %f478, 0f3F6D6000` | 0.92724609375 | cubic coefficient |
| `%f480` | s6605 `mov.f32 %f480, 0f3FB00000` | 1.375 | cubic constant term |
| `%fd1` | s4856 `mov.f64 %fd1, 0d3F20000000000000` | 2⁻¹³ | norm epsilon |
| `%f989` | s11656 `mov.f32 %f989, 0f3ED306EB` | 0.4121621549129486 | activation coefficient |
| `%f990` | s11659 `mov.f32 %f990, 0f3DA60DD6` | 0.0810810774564743 | activation coefficient |
| `%f991` | s11662 `mov.f32 %f991, 0f3F000000` | 0.5 | activation offset |
| `%f992` | s11665 `mov.f32 %f992, 0f40000000` | 2.0 | activation clamp |

`%f989…%f992` are defined inside phase 15's epilogue and re-materialised (as
`mov.f32` + `cvt.rn.f16.f32` pairs) inside the epilogue of every odd phase
17…45, e.g. s12054–s12064 (E17) — unlike enc0 the *f32* constants are not reused,
only their values.

Live across the whole mma region (defined in the prologue, read by every warp
reduction):

| register | statement | value | used as |
|---|---|---|---|
| `%r4016` | s4416 | 1 | `shfl.sync.bfly` `b` operand, bit 0 |
| `%r4025` | s4805 | 2 | `shfl.sync.bfly` `b` operand, bit 1 |
| `%r4027` | s4798 | −1 | `shfl.sync.bfly` membermask |

The `c` (clamp/segment) operand of every butterfly is built as
`%r = (WARP_SZ << 8) - 8192` then `or.b32 %r, %r, 31` (e.g. s8785–s8787), i.e.
31 for a 32-lane warp.

The long-lived **residual vector**: 48 `mov.b32 {%rsA, %rsB}, %r3713+4k`
statements at s4480 ff. split `%r3713, %r3717, %r3721 …` into f16 halves
(`%rs1460/%rs1463`, `%rs1466/%rs1469`, `%rs1472/%rs1475`, …).  Each `%r3713+4k` is
`cvt.rn.f16x2.e4m3x2 %r3713, %rs168` (s4415), i.e. the dequantisation of two
packed e4m3 bytes loaded from the input plane by `ld.global.v2.u16 {%rs168,
%rs169}, [%rd11]` with `%rd11 = %rd3 + …` and `%rd3 = cvta.to.global.param_0+8`
(s173–s175).  These halves are added to the column tables in E10 (§3/E10) and
read nowhere else.

### 1.3 Quantisation helpers

* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).
* `cvt.rn.f16x2.e4m3x2` — its inverse, used only in the prologue (s4415 ff.).
* `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `div.approx.ftz.f32` — always wrapped as
  `cvt.f32.f16 → op → cvt.rn.f16.f32` (each step rounded through f16).

### 1.4 The shared arena

`_ZZ…E4smem` is declared 12800 bytes.  The mma A operands of phases 1…8 come from
24 `ld.shared.v4.u32` at `%r4244 + 16*laneid + 512*k`, `k = 0…23`
(s6090–s6113, lines 9771–9794); phases 11…14 come from 4 such loads at
`%r4244 + (tid.z<<9) + 16*laneid + {0, 2048, 4096, 6144}` (s10537, s10693, s10732,
s10770); phase 47 comes from 64 `ld.shared.u32` (s16502–s16574).  The arena is
written by the prologue (s6054–s6083), by the P*V epilogue E10 (s10491–s10530)
and by E46 (s16348–s16480).  See §6/U3.

## 2. Phase table summary

Reproduced from `rr_layer_spec.py enc3` with `M`, `N`, `K` from `geom()` (fragment
counts and chain lengths), and `check = (M/16)·(N/8)·(K/32)`:

| # | stmt range | lines | M | N | K | nA | nB | chains | klens | mma | check | =? | weight bytes | C bias bytes | epi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | s6124–s6147 | L9812–9973 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 256..780 | – | 7 |
| 2 | s6155–s6178 | L9993–10154 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 4352..4876 | – | 7 |
| 3 | s6186–s6209 | L10174–10335 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 1280..1804 | – | 7 |
| 4 | s6217–s6240 | L10355–10516 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 5376..5900 | – | 7 |
| 5 | s6248–s6271 | L10536–10697 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 2304..2828 | – | 7 |
| 6 | s6279–s6302 | L10717–10878 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 6400..6924 | – | 7 |
| 7 | s6310–s6333 | L10898–11059 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 3328..3852 | – | 7 |
| 8 | s6341–s6364 | L11079–11240 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 7424..7948 | – | 171 |
| 9 | s6536–s6583 | L11628–11957 | 64 | 96 | 32 | 4 | 12 | 48 | 1 | 48 | 48 | Y | – | 8448..20236 | 3843 |
| 10 | s10427–s10474 | L22107–22436 | 192 | 96 | 96 | 12 | 12 | 16 | 3 | 48 | 432 | **N** | – | – | 199 |
| 11 | s10674–s10689 | L22922–23027 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 20736..24332 | – | 23 |
| 12 | s10713–s10728 | L23077–23182 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 45312..48908 | – | 23 |
| 13 | s10752–s10767 | L23232–23337 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 69888..73484 | – | 23 |
| 14 | s10791–s10806 | L23387–23492 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 94464..98060 | – | 832 |
| 15 | s11639–s11654 | L25484–25589 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 99072..102668 | 103168..103216 | 289 |
| 16 | s11944–s11959 | L26459–26564 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 103232..106828 | – | 76 |
| 17 | s12036–s12051 | L26731–26836 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 107584..111180 | 111680..111728 | 168 |
| 18 | s12220–s12235 | L27391–27496 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 111744..115340 | – | 76 |
| 19 | s12312–s12327 | L27663–27768 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 115840..119436 | 119936..119984 | 168 |
| 20 | s12496–s12511 | L28323–28428 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 120000..123596 | – | 76 |
| 21 | s12588–s12603 | L28595–28700 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 124096..127692 | 128192..128240 | 168 |
| 22 | s12772–s12787 | L29255–29360 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 128256..131852 | – | 76 |
| 23 | s12864–s12879 | L29527–29632 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 132352..135948 | 136448..136496 | 168 |
| 24 | s13048–s13063 | L30187–30292 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 136512..140108 | – | 76 |
| 25 | s13140–s13155 | L30459–30564 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 140608..144204 | 144704..144752 | 168 |
| 26 | s13324–s13339 | L31119–31224 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 144768..148364 | – | 76 |
| 27 | s13416–s13431 | L31391–31496 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 148864..152460 | 152960..153008 | 168 |
| 28 | s13600–s13615 | L32051–32156 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 153024..156620 | – | 76 |
| 29 | s13692–s13707 | L32323–32428 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 157120..160716 | 161216..161264 | 168 |
| 30 | s13876–s13891 | L32983–33088 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 161280..164876 | – | 76 |
| 31 | s13968–s13983 | L33255–33360 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 165376..168972 | 169472..169520 | 168 |
| 32 | s14152–s14167 | L33915–34020 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 169536..173132 | – | 76 |
| 33 | s14244–s14259 | L34187–34292 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 173632..177228 | 177728..177776 | 168 |
| 34 | s14428–s14443 | L34847–34952 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 177792..181388 | – | 76 |
| 35 | s14520–s14535 | L35119–35224 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 181888..185484 | 185984..186032 | 168 |
| 36 | s14704–s14719 | L35779–35884 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 186048..189644 | – | 76 |
| 37 | s14796–s14811 | L36051–36156 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 190144..193740 | 194240..194288 | 168 |
| 38 | s14980–s14995 | L36711–36816 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 194304..197900 | – | 76 |
| 39 | s15072–s15087 | L36983–37088 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 198400..201996 | 202496..202544 | 168 |
| 40 | s15256–s15271 | L37643–37748 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 202560..206156 | – | 76 |
| 41 | s15348–s15363 | L37915–38020 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 206656..210252 | 210752..210800 | 168 |
| 42 | s15532–s15547 | L38575–38680 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 210816..214412 | – | 76 |
| 43 | s15624–s15639 | L38847–38952 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 214912..218508 | 219008..219056 | 168 |
| 44 | s15808–s15823 | L39507–39612 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 219072..222668 | – | 76 |
| 45 | s15900–s15915 | L39779–39884 | 64 | 128 | 128 | 4 | 16 | 4 | 4 | 16 | 256 | **N** | 223168..226764 | 227264..227312 | 168 |
| 46 | s16084–s16099 | L40439–40544 | 16 | 128 | 32 | 1 | 16 | 16 | 1 | 16 | 16 | Y | 227328..230924 | – | 564 |
| 47 | s16664–s16727 | L41319–41760 | 256 | 512 | 512 | 16 | 64 | 4 | 16 | 64 | 16384 | **N** | 231424..247308 | 313344..313392 | 125 |
| 48 | s16853–s16860 | L41931–41980 | 64 | 64 | 128 | 4 | 8 | 2 | 4 | 8 | 128 | **N** | 313664..315212 | 315712..315728 | 323 |

**Arithmetic check.**  Rows 1–9, 11–16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
38, 40, 42, 44 and 46 satisfy
`M/16 · N/8 · K/32 == mma`, i.e. every A×B tile of the M×N tile grid is
evaluated exactly once, so `M`/`N`/`K` are the true GEMM shape.

Rows 10, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47 and 48
do **not**: the phase is *banded* (a diagonal block of the M×N grid) and the
printed `M`, `N` are the *extents spanned by the fragment operands*, not the shape
of the result.  The chain structure makes the real shape explicit:

* **P10** (`48 mma`, 16 chains of 3): chain `c` is
  `(A_{3·(c>>2)+k}, B_{2·(c&3)+k})` for `k = 0,1,2` — 4 row tiles × 4 column-pair
  groups, each accumulating 3 k-steps.  Real shape **64×32×96**, all
  4×4 = 16 output tiles evaluated.  (Same shape as enc0's P6.)
* **P15/17/…/45** (`16 mma`, 4 chains of 4): chain 0 is
  `(A0,B0) (A1,B2) (A2,B4) (A3,B6)`, chain 1 is `(A0,B1) (A1,B3) (A2,B5) (A3,B7)`,
  chain 2 repeats with `B8…B15`, chain 3 with `B9…B15` — 4 row tiles × 2 column
  tiles per 8-B group, 4 k-steps.  Real shape **64×64×128** with only the
  4×2 = 8 diagonal output tiles evaluated (of a 4×8 grid).
* **P47** (`64 mma`, 4 chains of 16): chain 0 is
  `(A_j, B_{2j})`, chain 1 `(A_j, B_{2j+1})`, chains 2/3 `B_{32+2j}`, `B_{32+2j+1}`,
  `j = 0…15`.  16 k-steps.  Real shape **256×256×512** over 4 output tiles.
* **P48** (`8 mma`, 2 chains of 4): `(A_j, B_{2j})` / `(A_j, B_{2j+1})`,
  `j = 0…3`.  Real shape **64×32×128** over 2 tiles.
* **P15…P45 odd** likewise.

**Phases that read no weight-image B operand:** **P9** (`Bo = [None, None]` on all
48 mma — its `B` is a register operand packed from P7's D, §3/E8) and **P10**
(its `B` is packed from the `movmatrix` of P8's D, §3/E9(f)).  Every other phase's
`B` is loaded from `W`.

**Weight-image extent.**  The largest byte offset used anywhere is 315728 (P48's
second C seed, `ld.global.u32 [%rd653+315728]`, s16852; the first is `+315712`,
s16851).  With the 4-byte read that makes the image **315732 bytes**; 1654
distinct weight-image byte offsets are touched (`weight_image_size`).  The single largest interior gap is
`247308…313344` — 66036 bytes between the end of P47's B block and P47's bias that
no phase in this entry reads (the analogous hole in enc4 is 82420 bytes, §5/U9 of
`rrswin_enc4_epilogues.md`).

## 3. The epilogues

Phases 1…8 (the stem), 11…45 (the norm + MLP block) and 46…48 (the merge) are
highly repetitive; each distinct body is written out once and the repeats are
tabulated.

### E1…E7 — s6148–s6154, s6179–s6185, s6210–s6216, s6241–s6247, s6272–s6278, s6303–s6309, s6334–s6340: next-B loads

Seven identical 7-statement epilogues (`add.s64`×3, 2 `ld.weak.global.ca.v4.u32`,
`mov.u32`, `mul.wide.u32`).  Each issues the two `v4.u32` fragments of the *next*
phase's B operand at `W + 16*laneid + {base, base+512}`:

| epilogue | range | loads (statement / line) | offsets | next phase's W block |
|---|---|---|---|---|
| E1 | s6148–s6154 | s6152 / L9986, s6154 / L9990 | 4352, 4864 | P2 `4352..4876` |
| E2 | s6179–s6185 | s6183 / L10167, s6185 / L10171 | 1280, 1792 | P3 `1280..1804` |
| E3 | s6210–s6216 | s6214 / L10348, s6216 / L10352 | 5376, 5888 | P4 `5376..5900` |
| E4 | s6241–s6247 | s6245 / L10529, s6247 / L10533 | 2304, 2816 | P5 `2304..2828` |
| E5 | s6272–s6278 | s6276 / L10710, s6278 / L10714 | 6400, 6912 | P6 `6400..6924` |
| E6 | s6303–s6309 | s6307 / L10891, s6309 / L10895 | 3328, 3840 | P7 `3328..3852` |
| E7 | s6334–s6340 | s6338 / L11072, s6340 / L11076 | 7424, 7936 | P8 `7424..7948` |

Nothing else.  **Outputs:** the 8 registers of the next phase's B.

### E8 — s6365–s6535: the score-phase bias tiles and the phase-9 operands

171 statements: 24 `ld.weak.global.ca.v4.u32`, 80
`cvt.rn.satfinite.e4m3x2.f16x2`, 40 `mov.b32`, 25 `add.s64`, plus one
`mov.u32`/`mul.wide.u32` per-lane address pair.

```
s6368  L11249: add.s64 %rd94, %rd425, 8448
s6369  L11253: ld.weak.global.ca.v4.u32 { %r6280,%r6281,%r6282,%r6283},[%rd94]
s6415  L11345: ld.weak.global.ca.v4.u32 { %r6372,…%r6375},[%rd117]   // 8448+23*512 = +20224
s6416  L11348: cvt.rn.satfinite.e4m3x2.f16x2 %rs1942, %r5810
s6489  L11522: cvt.rn.satfinite.e4m3x2.f16x2 %rs1989, %r5790
s6535  L11626: mov.b32 %r6933, {%rs2034, %rs2035}
```

1. **Score-bias C seeds** (s6368–s6415): 24 `v4.u32` loads at
   `W + 16*laneid + 8448 + 512*i`, `i = 0…23` → `%r6280 … %r6375`
   (96 registers = the 96 `C` operands of phase 9).  The index is the same
   `4*(n>>1) + m` scheme enc0 documents: `4*(n>>1)+m` runs 0…23 for the 4 m-tiles
   and 6 `(n>>1)` groups, and phase 9's own mma confirm the assignment
   (`s6536 C=%r6280,%r6281` first, `s6538 C=%r6296,%r6297` = the `+2048` tile).
   Byte block `[8448, 20236]` = 11792 bytes of f16, structurally identical to
   enc0's `[8384, 20672)`.
2. **B packs** (s6418–s6487, sources from s6416): the first 24 `mov.b32`
   destinations
   (`%r6822 %r6823 %r6832 %r6833 … %r6932 %r6933`) = phase 9's twelve B fragments
   (24 b32).  Every source pair is
   `cvt.rn.satfinite.e4m3x2.f16x2` of two registers written by **phase 7's mma**
   (`%r5790 … %r6021`), verified by walking each `mov.b32`'s operands through its
   two `cvt` statements (e.g. `s6417: cvt %rs1941, %r5790`, `s6416: cvt %rs1942,
   %r5810`, `s6418: mov.b32 %r6822, {%rs1941, %rs1942}`).
3. **A packs** (s6490–s6535): the remaining 16 `mov.b32` destinations
   (`%r6568 %r6569 %r6570 %r6571`, `%r6688…%r6691`, `%r6808…%r6811`,
   `%r6928…%r6931`) = phase 9's four A fragments (16 b32), also from phase 7's D
   (`s6489: cvt %rs1989, %r5790`).

**Phase 9's A fragment 0 and B fragment 0 are bit-identical**: both are
`{pack(%r5790,%r5810), …}` — `%r6568` (s6489–s6491) and `%r6822` (s6416–s6418)
read the same two source registers.  Whatever the score GEMM's two operands mean,
fragment 0 of each is the same value, exactly as in enc0 (E9).

**Inputs:** phase 7's 48 D registers, phase 8's `%r10141`-adjacent state.
**Outputs:** phase 9's 16 A + 24 B registers and its 96 C seeds.

### E9 — s6584–s10426: the 64×96 score softmax

3843 statements, **zero memory access**.  This is the enc0 E5 block verbatim with
shifted registers; the op histogram is the same (1048 `mov.b32`, 672
`cvt.rn.f16.f32`, 384 `cvt.rn.f16.f64`, 192 each `fma.rn.f16x2` /
`cvt.f32.f16` / `rcp.approx.ftz.f32` / `mul.f16`, 96 each `max.f16x2`/`min.f16x2`/
`neg.f16x2`/`mul.f16x2`/`and.b32`, 48 `movmatrix`, 16 `shfl.sync.bfly`, 144 packs).

Sub-ranges (first/last statement with the physical line):

| range | statements | what |
|---|---|---|
| s6584–s8701 | 2118 | 96 per-element exponent groups (scale, clamp, cubic, mantissa shift/and) |
| s8702–s8781 | 80 | in-lane partial sum trees (`add.f16x2`) |
| s8782–s8916 | 135 | 16 `shfl.sync.bfly.b32` (s8788/L18834 … s8909/L19138) → 8 row sums |
| s8918–s9679 | 769 | 192 `rcp.approx.ftz.f32` → 96 reciprocals |
| s9685–s10161 | 477 | 192 `mul.f16` → 96 probabilities |
| s10163–s10210 | 48 | `movmatrix.sync.trans.aligned.m8n8.b16` of **phase 8's** 48 D registers |
| s10211–s10425 | 216 | 144 `cvt.rn.satfinite.e4m3x2.f16x2` packs |

**(a) per-element exponent, s6584–s8701.**  Per f16 half `x` of each of phase 9's
96 D f16x2 registers:

```
m  = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)
   s6588 mul.f16x2, s6593 max.f16x2, s6598 min.f16x2
u  = f16(1.375 + m*(0.92724609375 - m^2))
   s6604 fma.rn.f16x2 (m*(-m) + 0.92724609375), s6609 fma.rn.f16x2 (m*u + 1.375)
E  = (u_bits << 5) & 0x7FE07FE0            // s6610 shl.b32, s6611 and.b32
```

i.e. the same "cubic in the exponent field" approximation enc0's E5(a) derives and
which the acceptance text asks about.  The constants are re-materialised per group
(`s6585 mov.f32 %f476 …`, `s6589/s6594 mov.f64 %fd383/%fd385 …`) — unlike enc0,
this layer's E9 does not hoist them.

**(b) row sums, s8702–s8781 + s8782–s8916.**  96 `expval` registers are added in
eight groups of six (`add.f16x2`), then across lane bits 0 and 1 with
`shfl.sync.bfly` using `%r4016 = 1` and `%r4025 = 2`, then the f16x2 halves are
folded with `add.f16` and broadcast (`s8796 mov.b32 {%rs2790,%rs2791}, %r9116` /
`s8797 add.f16` / `s8798 mov.b32 %r9313, {%rs2789,%rs2789}`).

**There is no row maximum anywhere in this epilogue.**  `max.f16x2`/`min.f16x2`
(96 each, s6593…s8691) are the clamp of (a); the only shuffles are the 16
`shfl.sync.bfly` above.  (Contrast E48(b), which *does* take a row max.)

**(c) reciprocal / probability, s8918–s10161.**  192 `rcp.approx.ftz.f32` wrapped
as `cvt.f32.f16 → rcp → cvt.rn.f16.f32` on the 8 row sums (each inverted once per
row-tile), then 192 `mul.f16` giving `p = expval * (1/rowsum)`.

**(d) V transpose, s10163–s10210.**  Exactly 48
`movmatrix.sync.trans.aligned.m8n8.b16` on phase **8**'s 48 D registers
(`%r6039 %r6040 %r6049 %r6050 … %r6270`; `s10163 L21459: movmatrix… %r9462,
%r6039`, `s10210 L21600: … %r9556, %r6270`) → 48 b32 of transposed V.

**(e) packs, s10211–s10425.**  144 packs: s10211–s10314 from the movmatrix
results → phase 10's 12 B fragments (`%r10068 …%r10179`, e.g. `s10213: mov.b32
%r10068, {%rs3389, %rs3390}`); the rest from the probabilities → phase 10's 12 A
fragments (`%r9774 …%r10177`, e.g. `s10427 A={%r9774,%r9775,%r9776,%r9777}`).

**Inputs:** phase 9's 96 D, phase 8's 48 D, `%r4016/%r4025/%r4027`.
**Outputs:** phase 10's 48 A / 24 B registers (plus the `%f476`, `%fd383`,
`%fd385`, `%f478`, `%f480` constants).  **Memory:** none.

### E10 — s10475–s10673: P·V epilogue — pack, shared staging, residual table, next B/C

199 statements.  Histogram: `add.f16`×64, `mov.b32`×48,
`cvt.rn.satfinite.e4m3x2.f16x2`×32, `ld.global.v2.u16`×16, `add.s64`×10,
`ld.weak.global.ca.v4.u32`×8, `mov.u32`×4, `shl.b32`×4, `st.shared.v4.u32`×4,
`add.s32`×3, `bar.sync`×2, `ld.shared.v4.u32`×1, `mul.wide.u32`×1,
`cvt.u64.u32`×1, `and.b64`×1.

**(a) require, s10479–s10530.**  32 packs of phase 10's D → 16 b32
(`%r22010 … %r22025`, `s10487 L22472: mov.b32 %r22010, {%rs3539, %rs3540}`),
stored one `v4.u32` per 512 bytes at
`%r22009 = %r235 + (laneid<<4)` (`s10478 add.s32 %r22009, %r235, %r22008`,
`%r22008 = laneid<<4` s10477):
`s10491 st.shared.v4.u32 [%r22009], …`, `s10504 [+512]`, `s10517 [+1024]`,
`s10530 [+1536]`.

**(b) stage read-back, s10531–s10541.**  `s10531 bar.sync 0`, then one
`ld.shared.v4.u32 [%r22029]` at
`%r22029 = %r4244 + (tid.z<<9) + (laneid<<4)` (`s10532–s10536`) → `%r22030 …
%r22033` (one 16-byte A fragment).  These are **phase 11's A operand** (P11
`A = %r22030,%r22031,%r22032,%r22033`).

**(c) next B, s10541–s10556.**  8 `v4.u32` at `W + 16*laneid + 20736 + 512*j`,
`j = 0..7` → phase 11's 16 B fragments (block `[20736, 24332)`).

**(d) residual column table, s10557–s10577.**  16 `ld.global.v2.u16` at
`W + ((laneid<<2)&12) + {98560 + 16*i}`, `i = 0…15` → 32 f16
(`%rs5396 … %rs5427`) covering the 128 columns of phase 11's output.

**(e) residual add, s10578–s10673.**  64 `add.f16` — two per f16x2 — each
`table[column] + <half of the long-lived prologue residual>`, e.g.

```
s10579 L22638: add.f16 %rs3565,%rs5426,%rs1460      // %rs5426 = W+98560, %rs1460 = residual
s10578 L22634: add.f16 %rs3568,%rs5427,%rs1463
s10580 L22641: mov.b32 %r10258, {%rs3565, %rs3568}
```

The 32 results `%r10258 %r10259 %r10268 … %r10409` are **phase 11's C operands**
(`s10674 C={%r10258,%r10259}`).  So P11 computes
`D11 = A11·B11 + (table + residual)`.

**Inputs:** phase 10's D, phase 8's D staging, `%r4244/%r235`, `W`.
**Outputs:** phase 11's A (4 regs), B (32 regs) and C (32 regs).

### E11…E13 — s10690–s10712, s10729–s10751, s10768–s10790: next-A from shared, next-B from W

Three identical 23-statement epilogues (`add.s64`×9, 8 `v4.u32`, `mov.u32`×2,
`shl.b32`, `add.s32`, 1 `ld.shared.v4.u32`, `mul.wide.u32`):

```
s10692 L23037: add.s32 %r22036, %r22027, %r22035      // %r22027 = %r4244 + (tid.z<<9)
s10693 L23038: ld.shared.v4.u32 {…}, [%r22036+2048]
s10697 L23044: add.s64 %rd126, %rd433, 45312
s10698 L23046: ld.weak.global.ca.v4.u32 { %r10412,…},[%rd126]
```

| epilogue | range | shared read (offset) | becomes | B offsets | next phase's W block |
|---|---|---|---|---|---|
| E11 | s10690–s10712 | s10693, `+2048` | P12 A | 45312 … 48896 step 512 (8 loads) | P12 `45312..48908` |
| E12 | s10729–s10751 | s10732, `+4096` | P13 A | 69888 … 73472 | P13 `69888..73484` |
| E13 | s10768–s10790 | s10771, `+6144` | P14 A | 94464 … 98048 | P14 `94464..98060` |

So phases 11…14 (which all have `nA = 1`) each take their single 16×32 A fragment
from one 512-byte slot of the shared arena and their B from `W`.

### E14 — s10807–s11638: the RMS norm before the MLP

832 statements: 242 `mov.b32`, 192 `mul.f16`, 66 `add.f16`, 64 `cvt.f32.f16`,
64 `rsqrt.approx.ftz.f32`, 64 `cvt.rn.f16.f32`, 34 `add.f16x2`, 4
`shfl.sync.bfly`, 32 packs, 16 `ld.global.v2.u16`, 8 weight `v4.u32`, 4
`ld.global.u32`.

| range | what |
|---|---|
| s10810–s10935 | 64 `mul.f16` — squares of phase 14's 32 D f16x2 (`s10809 L23500: mov.b32 {%rs4566,%rs4569}, %r10832`, `s10810 mul.f16 %rs3760,%rs4569,%rs4569`) |
| s10937–s10999 | 34 `add.f16x2` — in-lane reduction, with the 4 `shfl.sync.bfly` over lane bits 0,1 at s10971/L23947, s10977, s10983, s10989 |
| s10996–s11001 | `add.f16x2` + `add.f16` — the two row sums per lane |
| s11002–s11097 | `+ 2⁻¹³` (`s11003 add.f16 %rs3959,%rs3949,%rs3955`, `%rs3955 = cvt.rn.f16.f64 %fd1` s11002), broadcast into 32 f16x2 |
| s11102–s11351 | 64 `rsqrt.approx.ftz.f32` (2 halves each) → 32 reciprocals |
| s11355–s11370 | 16 `ld.global.v2.u16` gain at `W + ((laneid<<2)&12) + {98816 + 16*i}` |
| s11372–s11466 | 64 `mul.f16` — `inv · gain` (e.g. `s11372 L24749: mul.f16 %rs4148,%rs4149,%rs5458` with `%rs5458 = W+98816`) |
| s11467–s11561 | 64 `mul.f16` — `x · (inv·gain)` (`s11468 mul.f16 %rs4340,%rs4566,%rs4148`) |
| s11567–s11581 | 8 weight `v4.u32` = phase 15's B (`99072 + 512*j`) |
| s11587–s11590 | 4 `ld.global.u32` at `W + ((laneid<<2)&12) + {103168,103184,103200,103216}` = phase 15's C bias |
| s11591–s11637 | 32 packs of the normalised values → phase 15's A |

**What it computes.**  One RMS reduction per **row** over all 128 columns, with
no mean subtraction and no `1/N`: the lane's 16 one-per-n-tile partials for that
row are tree-summed in-lane (s10937–s10965), the 4 lanes holding the four
`t = laneid&3` column pairs are folded with two butterfly steps, and the f16x2
halves are added (`s10998 add.f16 %rs3949,%rs3950,%rs3951`).  Then

```
inv        = f16(rsqrt.approx(f32(f16(sumsq + 2^-13))))
out[row,c] = x[row,c] · inv · gain[c]
```

32 `rsqrt` results are produced because the compiler re-emits the reciprocal once
per (row, n-tile) pair, not because there are 32 distinct sums: the 32 `mov.b32`
that feed them copy only **two** distinct halves (`%rs3949` row `g`, `%rs3952` row
`g+8`, e.g. `s11005 mov.b32 %r11121, {%rs3956, %rs3959}` where `%rs3956 = %rs3959
= %rs3949 + 2^-13`).

`gain[c]` is per column and constant across rows (the four lanes with equal
`laneid&3` read the same 16-byte record).  **The norm's bias is not applied
here** — it is phase 15's C operand, loaded at the end of this epilogue
(`+103168 … +103216`).

**Outputs:** phase 15's B (8 regs) and C (4 regs), 16 packed A registers, and the
32 normalised f16x2 registers that E15/E17/E19 re-read as `%rs4566 …`.

### E15 — s11655–s11943: clamped cubic activation + column table + pack

289 statements: `mov.b32`×68, `add.f16`×64, `cvt.rn.f16.f32`×32, `mul.f16x2`×24,
16 `ld.global.v2.u16`, 8 each `neg/max/min/abs/sub/add.f16x2`, 8 weight `v4.u32`,
8 packs, `mov.f32`×4.

**(a) the constants and the activation, s11656–s11795.**  Eight groups, each
materialising `%f989…%f992` (`s11656 L25598: mov.f32 %f989, 0f3ED306EB`, `s11659
mov.f32 %f990, 0f3DA60DD6`, `s11662 mov.f32 %f991, 0f3F000000`, `s11665 mov.f32
%f992, 0f40000000`) and then doing, on **phase 15's D**
(`%r11311 %r11321 …`):

```
y   = clamp(x, -2, +2)                              // s11668 neg, s11669 max, s11670 min
g   = 0.5 + y*(0.412109375 - 0.0810546875*|y|)      // s11671 abs, s11672 mul, s11673 sub,
                                                    // s11674 mul, s11675 add
out = x * g                                         // s11676 mul.f16x2
```

`g` is the same cubic ramp as enc0's E13 (`g(+2) = 1`, `g(-2) = 0`), **not** a
sigmoid/SiLU.  8 groups cover the 8 register-pairs of the 4 chains of P15.

**(b) column bias, s11815–s11931.**  16 `ld.global.v2.u16` at
`W + ((laneid<<2)&12) + {107328 + 16*i}` then 64 `add.f16`, e.g.
`s11837 L26147: add.f16 %rs4564,%rs5490,%rs4566` — the second addend is a half of
phase **14**'s D (`%rs4566/%rs4569` ← `%r10832`, s10809), i.e. a residual.  The
48 results `%r11694 … %r11845` are **phase 16's C operand**.

**(c) pack, s11932–s11943.**  8 packs of the **activation outputs**
(`s11932 cvt %rs4757, %r11554`, `s11933 cvt %rs4756, %r11438`, `s11934 mov.b32
%r11838, {%rs4756,%rs4757}`, …) → phase 16's A fragment (`s11944
A={%r11838,%r11839,%r11840,%r11841}`).

**(d) next B, s11800–s11814.**  8 `v4.u32` at `103232 + 512*j` → phase 16's B.

**No stores, no `ld.global` other than the two tables.**  Inputs: phase 15's D,
phase 14's D, `W`.  Outputs: phase 16's A (4 regs), B (32), C (32).

### E16 / E18 / … / E44 — s11960–s12035 and its 14 repeats: next B/C loads + pack

76 statements, identical shape, for every even phase 16…44:

```
s11964 L26577: ld.weak.global.ca.v4.u32 { %r11847,…},[%rd166]      // B 103232 + 512*j, 8 loads
s11984 L26614: ld.global.u32 %r11921, [%rd455+111680]              // C bias, 4 words at +111680+16*i
s11988 L26619: cvt.rn.satfinite.e4m3x2.f16x2 %rs…, %r…             // 32 packs
s11990 L26624: mov.b32 %r…, {%rs…, %rs…}                           // 16 b32 = 4 A fragments
```

The 32 pack sources (`%r21120 %r21121 %r21122 …`, `s11988 L26619: cvt
%rs4765, %r21121`) are the **normalised values** E14 produced with the second of
its multi-sets (`s11469 mov.b32 %r21120, {%rs4340, %rs4343}`), i.e. phase 17's A
operand is the *same* normed tile that phase 15's A was built from — phases 15
and 17 are two parallel GEMMs off one normed tile, each with its own weight block
and bias.  The 4 C words are `W + ((laneid<<2)&12) + {bias_base + 16*i}`, and
each is used in **both halves** of the *odd* phase's mma C operand:
`s12036 C={%r11921,%r11921}` with `[%r11921=W+111680]`, `s12037
C={%r11931,%r11931}` with `W+111696`.

| epilogue | range | next-B block | next-C bytes |
|---|---|---|---|
| E16 | s11960–s12035 | 103232..106828 | 111680..111728 |
| E18 | s12236–s12311 | 111744..115340 | 119936..119984 |
| E20 | s12512–s12587 | 120000..123596 | 128192..128240 |
| E22 | s12788–s12863 | 128256..131852 | 136448..136496 |
| E24 | s13064–s13139 | 136512..140108 | 144704..144752 |
| E26 | s13340–s13415 | 144768..148364 | 152960..153008 |
| E28 | s13616–s13691 | 153024..156620 | 161216..161264 |
| E30 | s13892–s13967 | 161280..164876 | 169472..169520 |
| E32 | s14168–s14243 | 169536..173132 | 177728..177776 |
| E34 | s14444–s14519 | 177792..181388 | 185984..186032 |
| E36 | s14720–s14795 | 186048..189644 | 194240..194288 |
| E38 | s14996–s15071 | 194304..197900 | 202496..202544 |
| E40 | s15272–s15347 | 202560..206156 | 210752..210800 |
| E42 | s15548–s15623 | 210816..214412 | 219008..219056 |
| E44 | s15824–s15899 | 219072..222668 | 227264..227312 |

### E17 / E19 / … / E45 — s12052–s12219 and its 15 repeats: clamped cubic activation + pack

168 statements; identical op mix to E15 minus the table loads and the `mov.f32`
block (36 `mov.b32`, 32 `cvt.rn.f16.f32`, 24 `mul.f16x2`, 8 each
`neg/max/min/abs/sub/add.f16x2`, 8 weight `v4.u32`, 8 packs):

```
s12061 L26870: neg.f16x2 %r…,%r…
s12062 L26874: max.f16x2 %r…,%r11912,%r…
s12065 L26886: mul.f16x2 %r…,%r…,%r…
s12193 L27332: ld.weak.global.ca.v4.u32 { …},[%rd174]      // next B, 8 loads
s12208 L27363: cvt.rn.satfinite.e4m3x2.f16x2 …             // 8 packs of activation outputs
```

Same function as E15(a) on the odd phase's D (`%r11912 %r11922 …`), 8 packs of the
activation *outputs* → the next even phase's A, and 2 weight groups → the next even
phase's B.  **No column bias, no `ld.global`.**  The 15 repeats are at
s12052, s12328, s12604, s12880, s13156, s13432, s13708, s13984, s14260, s14536,
s14812, s15088, s15364, s15640, s15916 (one per odd phase 17…45).

#### Shape of the attention + MLP region, E10–E46

```
P9   A = pack(P7.D)   B = pack(P7.D)  C = bias@8448+512i  → scores 64×96
     E9: cubic-exponent softmax (no row max), 48 movmatrix(P8.D) + packs
P10  A = pack(scores)  B = pack(transpose(P8.D))  C = 0  → 64×32
     E10: pack → shared slot, read back as P11 A; table@98560 + residual → P11 C
P11  A = shared, B = W@20736,  C = table+residual   → D11
     E11: read shared slot +2048 → P12 A; load P12 B
P12  A = shared, B = W@45312,  C = D11             → D12   E12: same for P13
P13  A = shared, B = W@69888,  C = D12             → D13   E13: same for P14
P14  A = shared, B = W@94464,  C = D13             → D14 (16×128)
     E14: row-RMS norm over 128 columns (+2^-13, rsqrt, gain@98816), pack → P15 A
P15  A = normed tile, B = W@99072, C = bias@103168  → D15  (banded 64×64×128)
     E15: cubic activation(D15); table@107328 + D14 → P16 C; pack → P16 A
P16  A = pack(activation), B = W@103232, C = D14+table → D16
     E16: pack of the *normed* tile → P17 A; load P17 B and C
P17  A = normed tile, B = W@107584, C = bias@111680  → D17
     …the P15/P16 pair then repeats 15 more times (P17/P18 … P45/P46),
     each odd phase with its own weight block + bias and each even phase
     re-using the same normed tile as A and the previous even phase's D as C.
P46  A = pack(activation P45), B = W@227328, C = D44 → D46 (16×128)
     E46: arena scatter, shared staging, next B/C
P47  A = 16 shared slots, B = W@231424, C = bias@313344 → 4×16 k-steps
     E47: pack + surface scatter to param+56, next B/C
P48  A = pack(E47), B = W@313664, C = bias@315712 → D48
     E48: the final epilogue
```

### E46 — s16100–s16663: pack → plane-arena scatter → shared staging → next B/C

564 statements.  Histogram: `add.s32`×66, `ld.shared.u32`×64, `add.s64`×53,
`setp.lt.s32`×36, `shl.b32`×35, `bra`×33, `cvt.rn.satfinite.e4m3x2.f16x2`×32,
`selp.b32`×32, `ld.weak.global.ca.v4.u32`×32, `mov.b32`×16, `mul.wide.s32`×16,
`st.global.u32`×16, `st.shared.u32`×16, `and.b32`×11, `mov.u32`×10, `shr.u32`×10,
`or.b32`×9, `setp.ge.u32`×4, `ld.global.u32`×4, `ld.param.u64`×2, `bar.sync`×2.

**(a) packs, s16100–s16147.**  32 `cvt.rn.satfinite.e4m3x2.f16x2` + 16 `mov.b32`
→ `%r238 … %r253` = phase 46's 16×128 output as 16 b32 = 32 e4m3 per lane.

**(b) plane-arena scatter, s16148–s16326.**  `%rd4 = cvta(ld.param.u64 [%rd1+-32])`
(s16148–s16149, `%rd1 = param_0+80` ⇒ **`param_0+48`**), lane decomposition
`s16151 shr %r254, laneid, 2` (`g`), `s16152 and %r255, laneid, 3` (`t`), and 8
index bases formed from `(%r1 + ((g+8j)&7))`, `(%r6' + ((g+8j)>>3))` and
`(%r255 | 4)`, each guarded by `setp.lt.s32 %p, idx, 0` + `bra`:

| store | value | index |
|---|---|---|
| s16174 / L40691 | `%r238` | `t + 4*(g&7) + 32*(y)` |
| s16183 / L40702 | `%r239` | same, `t` or 4 |
| s16216 / L40737 | `%r240` | second base (`y2 = tid.z*2 + ((g+8)>>3)`) |
| s16224 | `%r241` | `t` or 4 |
| s16233 / s16241 | `%r242`, `%r243` | `+512` of the two bases |
| s16250 / s16258 | `%r244`, `%r245` | |
| s16267 / s16275 | `%r246`, `%r247` | `+1024` |
| s16284 / s16292 | `%r248`, `%r249` | |
| s16301 / s16309 | `%r250`, `%r251` | `+1536` |
| s16318 / s16326 | `%r252`, `%r253` | |

The exact index algebra is in §6/U4 — the *base* splits as
`t + 4*(g&7) + 32*y` with `y = (tid.z<<1) + (laneid>>5)` for the first group and
`y = (tid.z<<1) + ((g+8)>>3)` for the second, and the second group's `+512`
strides.  What is certain: 16 guarded 4-byte stores, i.e. `8*32 = 256` bytes per
row-stride unit of the arena.

**(c) shared staging, s16330–s16480.**  16 `st.shared.u32` at
`%r22114 = %r4244 + (idx<<2)` with `idx = t | ((laneid&28) | (y<<5))` and
`y = (tid.z<<1) + (laneid>>5)`, then `+256` and `+512*j`, `j = 0…7`
(`s16348 L40899: st.shared.u32 [%r22114], %r238`, `s16356 [+256] …`).  A
`bar.sync 0` at s16485 separates the two halves.

**(d) staging read-back, s16486–s16574.**  64 `ld.shared.u32` at
`%r4244 + (idx2<<2)` with
`idx2 = ((laneid&12)<<1) | ((laneid&~15)<<2) | (laneid&3)  (= 16*g + t)` for the
first 32 and `idx2 + 128` for the second 32, at byte offsets `{0, 1024, 2048,
3072, 4096, 5120, 6144, 7168}` (`s16502 L41087: ld.shared.u32 %r304, [%r22196]`,
`s16503 [+1024]`, `s16503–s16522`, then `s16527 [+16] … s16542`).  These 64 values
`%r304 … %r367` are **phase 47's 16 A fragments**.

**(e) next B/C, s16581–s16663.**  Inside a loop `$L__BB1_324` (`s16582`),
`s16583 shl.b32 %r22998, %r23413, 9` then 32 `v4.u32` at
`W + 16*laneid + 231424 + 512*(%r23413 + j)`, `j = 0…31` (`s16590 – s16652`) =
phase 47's 64 B fragments, and 4 `ld.global.u32` at
`W + ((laneid<<2)&12) + {313344, 313360, 313376, 313392}` (s16660–s16663) =
phase 47's C bias.

**Outputs:** `%r238…%r253` (read again by (c)), phase 47's A/B/C.

### E47 — s16728–s16852: pack → surface scatter → next B/C

125 statements: 8 packs + 2 `mov.b32` (s16728–s16737), 4 guarded stores to
**`param_0+56`** (s16772 `st.global.v2.u16`, s16783 `st.global.u32`,
s16817 `st.global.u32`, s16826 `st.global.v2.u16`), then 4 weight `v4.u32`
(s16839–s16845) and 2 `ld.global.u32` (s16851–s16852).

```
s16738 L41793: mov.u32 %r22997, %laneid
s16744 L41800: shr.u32 %r23004, %r22183, 1          // ey>>1
s16745 L41801: shr.u32 %r23005, %r22182, 1          // ex>>1
s16746 L41802: shl.b32 %r23006, %r23005, 3          // 8*(ex>>1)
s16747 L41803: mul.lo.s32 %r23007, %r23004, %r23006 // row stride
s16769 L41826: cvta.to.global.u64 %rd632, %rd5      // %rd5 = param_0+56 (s16581)
s16772 L41829: st.global.v2.u16 [%rd634], {%rs5492, %rs5493}
```

The destination is `param+56`, **not** the plane arena; extents come from
`ld.param.v2.u32 [param_0+0]` re-read as `%r22182/%r22183` in E46 (s16487) and are
**halved**.  Two index bases are formed, each guarded by
`(y<0)|(y≥(ey>>1))|(laneid>63)|(x<0)|(x≥(ex>>1))|(%r23413>159)`: base 1
`y = (%r2>>1) + (laneid>>4)`, `x = (%r1>>1) + ((laneid>>2)&3)`; base 2 uses
`(g+8)`.  The loop at s16828–s16830 steps `%r23413` by 128 while `< 160`, so the
8-pack/4-store body runs once or twice depending on `tid.z` (§6/U4).

Then phase 48's B (4 `v4.u32` at `313664 + 512*j`) and C (2 `ld.global.u32` at
`315712`, `315728`).

### E48 — s16861–s17183: the final epilogue

323 statements: 4 `st.shared.u32` (s16872–s16891), 2 `bar.sync`, 5 `ld.shared.u32`
(s16930–s16938), 12 `mul.ftz.f32`/`ex2.approx`/`add.ftz.f32`/`div.approx`,
4 `shfl.sync.down.b32`, 9 `tex.base.2d.v4.f16.s32` (s17042–s17101), 3
`sust.b.2d.v4.b16.zero` (s17165, s17166, s17181).

**(a) shared staging, s16861–s16891.**  Phase 48's D (`%r23106/%r23107` and
`%r23116/%r23117`) is written to a shared array at
`%r391 = %r23368 + 4*(6*((laneid>>2) + (tid.z<<4) ... ) + t)` and `%r393` (the
`(g+8)` variant), two `u32` per lane with a `%r392 > 5` guard on the second.

**(b) row max, then softmax, s16904–s17022.**  After `bar.sync 0` (s16904)
`%tid.x` picks a 6-element row inside a shared array
(`s16930–s16936: ld.shared.u32 [%r23313] … [+20]`, with `%r23313 = %r23380 +
4*(6*x + 48*y)`), and:

```
s16948 L42087: max.f16x2 %r23163,%r23164,%r23165      // 5-element row max
s16952 L42100: max.f16 %rs5502,%rs5503,%rs5504
s16954 L42107: mov.b32 %r23186, {%rs5505, %rs5505}
s16955 L42109: sub.f16x2 %r23172,%r23164,%r23186      // x - rowmax
s16959 L42118: ex2.approx.ftz.f32 %f997, %f1025       // 2^((x-max)*log2e)
s17015 L42237: div.approx.ftz.f32 %f1013, %f1024, %f1042   // 1/rowsum
s17018 L42244: mul.f16x2 %r23187,%r23188,%r23201
```

**This one has a real row maximum** (unlike E9).  The `ex2` is a *true* `2^u` via
`mul.ftz.f32` by `0f3FB8AA3B` (= log2 e) plus `ex2.approx.ftz.f32`, not the cubic
mantissa trick.  Note the extra `add.ftz.f32 %f1028, %f1027, 0f00000000` at
s16967 (a zero add the compiler left in).

**(c) texture gather + weighted sum, s17023–s17105.**  A 3×3 spatial stencil of
`tex.base.2d.v4.f16.s32` reads from the texture at `param_0+88`
(`s16937 ld.param.u64 %rd672, [%rd681+112]` is a surface; the texture is
`s17040 ld.param.u64 %rd670, [%rd681+88]`), with the 8 neighbour coordinates
clamped by `selp` chains (`s17024–s17035`: `x-1`, `x+1`, `y-1`, `y+1`, both with
`1` as the out-of-range sentinel).  Each 3×3 texel is multiplied by one of the
five softmax values with `fma.rn.f16x2` (`s17046 fma.rn.f16x2 %r23206,%r23207,
%r23187,%r23217` with `%r23217 = 0`), the halves are summed
(`s17098 add.f16 %rs5559,%rs5560,%rs5561`) and one more texel is folded with
`fma.rn.f16` (s17103–s17105).

**(d) second sigmoid and difference, s17106–s17161.**  Two more
`1/(1+2^(-log2e·v))` reciprocals (`s17107–s17111`, `s17113–s17117`), three
`mul.f16` products (`s17118–s17120`) and three `sub.f16` differences
(`s17121–s17124`), the `0.25` (`s17122 mov.f32 %f1020, 0f3E800000`) scale
(s17129 `mul.f16x2 %r23268, %r23269, %r23270`), then a **two-step cross-lane
sum**: `shfl.sync.down.b32` with `delta = %r23293 = 1` (s16940) and
`delta = %r23301 = 8` (s16933), each followed by `add.f16x2`, and the same two
steps again on the scalar path (s17149 with delta 1, s17158 with delta 8, then
`add.f16` at s17152 and s17161).  The clamp/segment operand is again
`(WARP_SZ<<8)-8192 | 31` and the membermask `%r23303 = -1` (s17026).

**(e) surface stores, s17162–s17183.**

```
s17162 L42567: ld.param.u64 %rd674, [%rd681+120]
s17163 L42568: ld.param.u64 %rd677, [%rd681+128]
s17165 L42571: sust.b.2d.v4.b16.zero [%rd672, {%r23307,%r397}], {%rs5597,%rs5600,%rs5603,%rs5634}
s17166 L42574: sust.b.2d.v4.b16.zero [%rd674, {%r23307,%r397}], {%rs5588,%rs5591,%rs5594,%rs5635}
s17181 L42592: sust.b.2d.v4.b16.zero [%rd677, {%r23335,%r23336}], {%rs5636,%rs5637,%rs5621,%rs5639}
```

Three 4-channel b16 surface writes (`param_0+112`, `+120`, `+128`); the third is
taken only for odd `%r398 & 1` (`s17167 and.b32 %r23334, %r398, 1`, `s17169
mov.pred %p347, 0`, `s17170 xor.pred %p348, %p346, %p347`, s17171 `@%p348 bra`) and
stores `0` in its fourth channel (`s17180 mov.u16 %rs5639, 0`).

**Inputs:** phase 48's D, the 18 cut values, `param+0/+88/+112/+120/+128`, the
shared staging of (a) and the normed tile's leftovers.
**Outputs:** none (the kernel `ret`s at s17183 / L42596).

## 4. The plane-arena and surface writes

`param_0+48` is reached once (`s16148 ld.param.u64 %rd556, [%rd1+-32]` with
`%rd1 = param_0+80` at s4) and is never the base of an `ld.*` in this entry: the
arena is write-only within a launch.  The complete set of `st.global.*` /
`sust.b.*` in the entry is:

* **E46, 16 × `st.global.u32 [%rd4 + 4*idx], v`** — s16174, s16183, s16216,
  s16224, s16233, s16241, s16250, s16258, s16267, s16275, s16284, s16292, s16301,
  s16309, s16318, s16326 (§3/E46(b)).  Value: phase 46's requantised 16×128
  output, 16 b32 per lane.
* **E47, 4 stores to `param_0+56`** — s16772 (`v2.u16`), s16783 (`u32`),
  s16817 (`u32`), s16826 (`v2.u16`).
* **E48, 3 `sust.b.2d.v4.b16.zero`** — s17165, s17166, s17181 to surfaces at
  `param_0+112/+120/+128`.

Nothing else in the 17183-statement entry writes to global memory.

## 5. Uncharacterised

**U1. The identity of the 11792-byte score bias (P9's C operand).**  Phase 9 adds
the f16 table `[8448, 20236)` indexed as
`byte = 8448 + 512*(4*(n>>1) + m) + 16*laneid + 8*(n&1)` (verified against phase
9's own C operands: `s6536 C=%r6280,%r6281` ↔ `i=0`, `s6538 C=%r6296,%r6297` ↔
`i=4`, both loaded by `s6369`/`s6377`).  Whether it is a relative position bias, a
learned score bias or a per-lane constant cannot be decided from the PTX.
*Missing:* the weight image's layout description (the `_prep` permutation).

**U2. The identity of the per-column tables.**  `W+98560` (E10's C seed add),
`W+98816` (E14's norm gain), `W+107328` (E15's post-activation add) and
`W+103168`/`111680`/… (the odd phases' GEMM bias) are all read through the same
`W + ((laneid<<2)&12) + off` pattern.  Their *role* is computable (see §3) but
their model names are not.

**U3. The shared-arena layout.**  The arena is written by the prologue
(s6054–s6083), by E10 (s10491–s10530) and by E46 (s16348–s16480), and read by
P1–P8 (s6090–s6113), P11–P14 (s10537, s10693, s10732, s10770) and P47
(s16502–s16574).  The prologue writes at `%r235 + 16*laneid + {0,512,1024,1536}`
and `%r235 + 16*laneid + {8192…9728}` (s6058, s6068–s6083) while the P1–P8 reads
use `%r4244 + 16*laneid + 512*k` with **no** `tid.z` term (s6090 ff.) — so at
least one of the two index derivations is not reconstructible without the
`_prep`/launch geometry.  *Missing:* the launch's `%tid.z`/`ctaid` ranges, or a
memory dump.

**U4. E46's/E47's index algebra.**  E46(b)'s arena stores and E47's
`param+56` stores each form two index bases from `(laneid&28) | (y<<5)` and
`((g+8)<<2)&28 | (y2<<5)` plus `+512*j`, with `y`/`y2` differing by one in the
row (they are derived from `%r23351` and `%r23353`, both `tid.z<<1`); the shared
staging read in E46(d) uses a *different* index
(`16*g + t` and `16*g + t + 128`).  I could not prove the store↔load permutation
from the PTX alone, nor whether `%r23413`'s loop in E47 (s16828–s16830) executes
once or twice per `tid.z` (the guard `%r23413 > 159` at s16489/s16743 is checked
before the first body).  *Missing:* a run of the kernel with a plane-arena dump.

**U5. The MLP block's shape.**  §3 establishes, from the pack sources and each
phase's C operand, that the MLP region is: one normed tile feeding every *odd*
phase (P15, P17, …, P45) as its `A`, each odd phase's activation feeding its
*paired* even phase as `A`, and the even phases' `C` chaining (`D16 = … + D14 +
table@107328`, `D18 = … + D16`, …, verified from `P18 C` tracing to P16's mma).
Only the **first** even phase adds the column table `@107328` to the residual
(s11837 versus the table-free `E18`/`E20`/…); every later even phase's `C` is the
previous even phase's `D` alone (E18's `C` operands have no `ld.` in their def-use
chain).  Whether the 16 pairs are one wide MLP, a stack of residual blocks, or
something else is a model-level question the PTX cannot answer.
*Missing:* the model source.

**U6. The stem's shift structure.**  Phases 1…8 read the same 24 shared A
fragments in four shift groups (P1/P2 at `+0/1536/…`, P3/P4 at `+512/2048/…`,
P5/P6 at `+1024/…`, P7/P8 at `+1536/…`, s6090–s6113) and accumulate in two
chains (`P1→P3→P5→P7` and `P2→P4→P6→P8`, verified from each phase's C operands:
`s6186 C={%r4296,%r4297}` = P1's D).  Whether that is a 2×2 stencil, a 4-tap
1-D convolution, or something else is a property of the high-level model, not of
the PTX.  (enc4 has five shift groups and ten phases, so the tap count is a
per-stage constant.)  *Missing:* the model source.

**U7. The meaning of the prologue residual vector.**  §1.2 tracks the f16 halves
`%rs1460 …` back to `cvt.rn.f16x2.e4m3x2` of e4m3 values loaded from the plane at
`param_0+8` (`s175`, `s480`, …).  What that plane holds (an input colour buffer,
a noise texture, a previous frame) is not determinable from the PTX.

**U8. The logical meaning of E48's final expression.**  The tail computes
`out = (acc − sigmoid(acc)) * 0.25` on three channels gathered by texture from a
3×3 stencil (same shape as enc0's E22(e), whose §6/U6 flags the same gap).

## 6. Quick acceptance index

| requirement | where |
|---|---|
| every phase's M/N/K with the arithmetic checked | §2 (rows flagged **N** are banded; the real shapes are derived there) |
| weight image extent | §2 (315732 bytes, max offset 315728 at s16852) |
| phases that read no weight operand | §2 (P9, P10) |
| score epilogue: row max? | **none** — §3/E9, 16 `shfl.sync.bfly` only, no `max` used for a maximum |
| score bias operand | §3/E8 (24 `v4.u32` at `W+8448+512*i`, indexed `4*(n>>1)+m`) |
| attention Q/K/V | §3/E8 (P9 A and B both from P7's D), §3/E9(d) (P8's D transposed) |
| the norm | §3/E14 (row RMS over 128 columns, `+2^-13`, `rsqrt.approx`, gain) |
| the activations | §3/E15, §3/E17 (clamped cubic ramp, 8 groups per phase) |
| patch merge | §3/E46 (arena + shared staging), §3/E47 (`param+56`), §3/E48 (textures + surfaces) |
| per-launch plane-arena writes | §4 |
