# `cuda_dldn_engine_swin_enc4_kernel` — the epilogues of phases 1…62

Companion to `kernels/rr/rr_layer_spec.py enc4`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0017-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_enc4_kernel`.

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at its first line).  **`LNNNNN` is
  the physical line** of the same instruction in the corpus file, so every claim
  can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and the
  next phase's first `mma`.
* `W` denotes the prepared weight image: `ld.param.u64 %rd2,
  [cuda_dldn_engine_swin_enc4_kernel_param_0+40]` (s9), globalised as
  `cvta.to.global.u64 %rd3, %rd2` (s10).  Every `ld.weak.global.ca.v4.u32` in an
  epilogue reads `W + 16*laneid + imm`.  (s20229 re-reads the same parameter; the
  pointer is not cached.)
* `S` denotes the plane arena: `param_0+48`, reached as
  `ld.param.u64 %rd808, [%rd1+-32]` with `%rd1 = param_0+80` (`add.s64 %rd1,
  %rd9, 80`, s4) then `cvta.to.global.u64 %rd5, %rd808` (s19817–s19818).
* The C/D register layout and the weight-addressing formula are fixed (see the
  settled notes): `C/D reg0 = (row g, cols 2t,2t+1)`, `reg1 = (row g+8, same cols)`,
  `g = laneid>>2`, `t = laneid&3`.
* `rr_layer_spec.py`'s printed `M`/`N`/`K` come from `geom()` (distinct A/B
  fragment counts and accumulator-chain length).  The `Phase` object's own
  `w_min..w_max`/`n_tiles`/`k_slots` fields use a different (512-byte-block)
  convention and disagree with `geom()` on the stem phases; §2 uses `geom()` and
  checks the arithmetic explicitly.
* Anything I could not pin down from the PTX is in §6 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `W` (weight image, 533844 bytes) | `ld.param.u64 %rd2, [param_0+40]` | s9 (re-read s20229, s21313) |
| `S` (plane arena, `param+48`) | `ld.param.u64 %rd808, [%rd1+-32]` → `cvta.to.global.u64 %rd5, %rd808` | s19817–s19818 |
| output buffer (`param+56`) | `ld.param.u64 %rd849, [%rd945+-24]` with `%rd945 = param_0+80` | s21037–s21038 |
| input plane (`param+8`) | `ld.param.u64 %rd11, [%rd1+-72]` → `cvta.to.global.u64 %rd4, %rd11` | s22–s23 |
| extents `(ex, ey)` | `ld.param.v2.u32 {%r575, %r576}, [param_0+0]`; re-read as `ld.param.v2.u32 {%r29719, %r29720}, [%rd940+-80]` | s15; s20222 |
| block origin `(x0, y0)` | `%r570 = ctaid.x<<3`, `%r1 = %r570 - %rs124`; `%r574 = ctaid.y<<3`, `%r3 = %r574 - %rs125`; `%r2 = %r1 - 1`, `%r6 = %r3 - 1`; `%rs124/%rs125` from `ld.param.v2.u16 [param_0+80]` | s2, s3, s7, s12, s13, s14, s16, s5 |
| `%rd9` = `param_0` | `mov.b64 %rd9, cuda_dldn_engine_swin_enc4_kernel_param_0` | s1 |
| shared base `%r11752` | `mov.u32 %r11752, _ZZ33cuda_dldn_engine_swin_enc4_kernel33DldnEngineSwinEncParamsStructBaseE4smem` | s7511 |
| `%r31615` (second copy) | same `mov.u32` in E62 | s21312 |
| per-lane weight address | `mul.wide.u32 %rd, %laneid, 16` + `add.s64 %rd, %rd2, %rd` | e.g. s11983–s11984, s12267–s12268 |
| per-lane *table* address | `shl.b32 %r, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rd, %r, 12` → `add.s64 %rd, %rd74/…, %rd` = `W + ((laneid<<2)&12)` | e.g. s13552–s13555 |

As in enc0/enc3, a `ld.global.v2.u16` or `ld.global.u32` with that addressing reads
the **same 16-byte record from the four lanes with equal `laneid&3`**, so those
tables are indexed by column, not by row.

The `%tid.z` guard `setp.gt.u32 %p8, %r9, 3; @%p8 bra` at s20–s21 bounds `%tid.z`
to 0…3 for the whole mma region; the eight phases that stage through shared memory
(E12→P13…P20) add `(tid.z << 11)` (s11969) resp. `(tid.z << 9)` (s12028) to the
arena base.

### 1.2 Constants that survive between epilogues

| register | first definition | literal | meaning |
|---|---|---|---|
| `%f476` | s8077 `mov.f32 %f476, 0f3C8CCB50` | 0.017186790704727173 | f16 `0x2466` = 0.017181396484375 (softmax input scale) |
| `%fd383` | s8081 `mov.f64 %fd383, 0dBFE1CC0000000000` | −0.55615234375 | f16 clamp lower bound |
| `%fd385` | s8086 `mov.f64 %fd385, 0d3FE1CC0000000000` | +0.55615234375 | f16 clamp upper bound |
| `%f478` | s8092 `mov.f32 %f478, 0f3F6D6000` | 0.92724609375 | cubic coefficient |
| `%f480` | s8097 `mov.f32 %f480, 0f3FB00000` | 1.375 | cubic constant term |
| `%fd1` | s5976 `mov.f64 %fd1, 0d3F20000000000000` | 2⁻¹³ | norm epsilon |
| `%f1117` | s13641 `mov.f32 %f1117, 0f3ED306EB` | 0.4121621549129486 | activation coefficient |
| `%f1118` | s13644 `mov.f32 %f1118, 0f3DA60DD6` | 0.0810810774564743 | activation coefficient |
| `%f1119` | s13647 `mov.f32 %f1119, 0f3F000000` | 0.5 | activation offset |
| `%f1120` | s13650 `mov.f32 %f1120, 0f40000000` | 2.0 | activation clamp |
| `%f1152` | s21429 `mov.f32 %f1152, 0f3F800000` | 1.0 | softmax dividend |
| `%f1148` | s21606 `mov.f32 %f1148, 0f3E800000` | 0.25 | final output scale |

`%f1117…%f1120` are defined inside E21 (phase 21's epilogue) and re-materialised
inside every odd phase's epilogue (E23, E25, … E59, e.g. `s14094–s14100
cvt.rn.f16.f32 low, %f1117 … %f1120`), exactly as in enc3.

Live across the whole mma region (defined in the prologue, read by every warp
reduction):

| register | statement | value | used as |
|---|---|---|---|
| `%r10789` | s7548 | 1 | `shfl.sync.bfly` `b` operand, bit 0 |
| `%r10798` | s7549 | 2 | `shfl.sync.bfly` `b` operand, bit 1 |
| `%r10800` | s10280 | −1 | `shfl.sync.bfly` membermask |

The `c` (clamp/segment) operand of every butterfly is built as
`%r = (WARP_SZ << 8) - 8192` then `or.b32 %r, %r, 31` (e.g. s10279), i.e. 31 for a
32-lane warp.  The `shfl.sync.down` cascade in E62 uses `%r31545 = 1` (s21424),
`%r31553 = 8` (s21417) and membermask `%r31555 = -1` (s21510).

The long-lived **residual vector**: 48 `mov.b32 {%rsA, %rsB}, %r317xx`
statements (e.g. s5443, s5446 …) split registers produced by
`cvt.rn.f16x2.e4m3x2 %r31707, %rs128` (s5439) into f16 halves.  `%rs128` comes from
a prologue `ld.global.v2.u16` off the input plane (`param_0+8`, s22–s23).  These
halves are the second addend of the E12 column-table add (§3/E12).

### 1.3 Quantisation helpers

* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).
* `cvt.rn.f16x2.e4m3x2` — its inverse, used only in the prologue (s5439 ff.).
* `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `div.approx.ftz.f32` — always wrapped as
  `cvt.f32.f16 → op → cvt.rn.f16.f32` (each step rounded through f16).

### 1.4 The shared arena

`_ZZ…E4smem` is declared **16384 bytes**.  The mma A operands of phases 1…10 come
from 30 `ld.shared.v4.u32` at `%r11753 = %r11752 + 16*laneid` plus
`512*k`, `k = 0…29` (s7513–s7542, lines 11835–11864); phases 13…20 come from 8
such loads at `%r11752 + (tid.z<<9) + 16*laneid + {0,2048,…,14336}` (s12033,
s12265, s12312, …, s12547); phase 61 comes from 80 `ld.shared.u32`
(s20242–s21030).  The arena is written by the prologue (s7494–s7504), by E12
(s11986–s12025) and by E60 (s20051–s20215).  See §6/U3.

## 2. Phase table summary

Reproduced from `rr_layer_spec.py enc4` with `M`, `N`, `K` from `geom()` and
`check = (M/16)·(N/8)·(K/32)`:

| # | stmt range | lines | M | N | K | nA | nB | chains | klens | mma | check | =? | weight bytes | C bias bytes | epi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | s7554–s7577 | L11883–12044 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 320..844 | – | 7 |
| 2 | s7585–s7608 | L12064–12225 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 5440..5964 | – | 7 |
| 3 | s7616–s7639 | L12245–12406 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 1344..1868 | – | 7 |
| 4 | s7647–s7670 | L12426–12587 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 6464..6988 | – | 7 |
| 5 | s7678–s7701 | L12607–12768 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 2368..2892 | – | 7 |
| 6 | s7709–s7732 | L12788–12949 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 7488..8012 | – | 7 |
| 7 | s7740–s7763 | L12969–13130 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 3392..3916 | – | 7 |
| 8 | s7771–s7794 | L13150–13311 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 8512..9036 | – | 7 |
| 9 | s7802–s7825 | L13331–13492 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 4416..4940 | – | 7 |
| 10 | s7833–s7856 | L13512–13673 | 96 | 32 | 32 | 6 | 4 | 24 | 1 | 24 | 24 | Y | 9536..10060 | – | 171 |
| 11 | s8028–s8075 | L14061–14390 | 64 | 96 | 32 | 4 | 12 | 48 | 1 | 48 | 48 | Y | – | 10560..22348 | 3844 |
| 12 | s11920–s11967 | L24541–24870 | 192 | 96 | 96 | 12 | 12 | 16 | 3 | 48 | 432 | **N** | – | – | 274 |
| 13 | s12242–s12261 | L25484–25617 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 22848..27468 | – | 27 |
| 14 | s12289–s12308 | L25675–25808 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 50496..55116 | – | 27 |
| 15 | s12336–s12355 | L25866–25999 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 78144..82764 | – | 27 |
| 16 | s12383–s12402 | L26057–26190 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 105792..110412 | – | 27 |
| 17 | s12430–s12449 | L26248–26381 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 133440..138060 | – | 27 |
| 18 | s12477–s12496 | L26439–26572 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 161088..165708 | – | 27 |
| 19 | s12524–s12543 | L26630–26763 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 188736..193356 | – | 27 |
| 20 | s12571–s12590 | L26821–26954 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 216384..221004 | – | 1029 |
| 21 | s13620–s13639 | L29419–29552 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 222144..226764 | 227264..227312 | 321 |
| 22 | s13961–s13980 | L30506–30639 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 227328..231948 | – | 92 |
| 23 | s14073–s14092 | L30842–30975 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 232768..237388 | 237888..237936 | 172 |
| 24 | s14265–s14284 | L31538–31671 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 237952..242572 | – | 92 |
| 25 | s14377–s14396 | L31874–32007 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 243072..247692 | 248192..248240 | 172 |
| 26 | s14569–s14588 | L32570–32703 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 248256..252876 | – | 92 |
| 27 | s14681–s14700 | L32906–33039 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 253376..257996 | 258496..258544 | 172 |
| 28 | s14873–s14892 | L33602–33735 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 258560..263180 | – | 92 |
| 29 | s14985–s15004 | L33938–34071 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 263680..268300 | 268800..268848 | 172 |
| 30 | s15177–s15196 | L34634–34767 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 268864..273484 | – | 92 |
| 31 | s15289–s15308 | L34970–35103 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 273984..278604 | 279104..279152 | 172 |
| 32 | s15481–s15500 | L35666–35799 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 279168..283788 | – | 92 |
| 33 | s15593–s15612 | L36002–36135 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 284288..288908 | 289408..289456 | 172 |
| 34 | s15785–s15804 | L36698–36831 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 289472..294092 | – | 92 |
| 35 | s15897–s15916 | L37034–37167 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 294592..299212 | 299712..299760 | 172 |
| 36 | s16089–s16108 | L37730–37863 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 299776..304396 | – | 92 |
| 37 | s16201–s16220 | L38066–38199 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 304896..309516 | 310016..310064 | 172 |
| 38 | s16393–s16412 | L38762–38895 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 310080..314700 | – | 92 |
| 39 | s16505–s16524 | L39098–39231 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 315200..319820 | 320320..320368 | 172 |
| 40 | s16697–s16716 | L39794–39927 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 320384..325004 | – | 92 |
| 41 | s16809–s16828 | L40130–40263 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 325504..330124 | 330624..330672 | 172 |
| 42 | s17001–s17020 | L40826–40959 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 330688..335308 | – | 92 |
| 43 | s17113–s17132 | L41162–41295 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 335808..340428 | 340928..340976 | 172 |
| 44 | s17305–s17324 | L41858–41991 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 340992..345612 | – | 92 |
| 45 | s17417–s17436 | L42194–42327 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 346112..350732 | 351232..351280 | 172 |
| 46 | s17609–s17628 | L42890–43023 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 351296..355916 | – | 92 |
| 47 | s17721–s17740 | L43226–43359 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 356416..361036 | 361536..361584 | 172 |
| 48 | s17913–s17932 | L43922–44055 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 361600..366220 | – | 92 |
| 49 | s18025–s18044 | L44258–44391 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 366720..371340 | 371840..371888 | 172 |
| 50 | s18217–s18236 | L44954–45087 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 371904..376524 | – | 92 |
| 51 | s18329–s18348 | L45290–45423 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 377024..381644 | 382144..382192 | 172 |
| 52 | s18521–s18540 | L45986–46119 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 382208..386828 | – | 92 |
| 53 | s18633–s18652 | L46322–46455 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 387328..391948 | 392448..392496 | 172 |
| 54 | s18825–s18844 | L47018–47151 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 392512..397132 | – | 92 |
| 55 | s18937–s18956 | L47354–47487 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 397632..402252 | 402752..402800 | 172 |
| 56 | s19129–s19148 | L48050–48183 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 402816..407436 | – | 92 |
| 57 | s19241–s19260 | L48386–48519 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 407936..412556 | 413056..413104 | 172 |
| 58 | s19433–s19452 | L49082–49215 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 413120..417740 | – | 92 |
| 59 | s19545–s19564 | L49418–49551 | 80 | 160 | 160 | 5 | 20 | 4 | 5 | 20 | 500 | **N** | 418240..422860 | 423360..423408 | 172 |
| 60 | s19737–s19756 | L50114–50247 | 16 | 160 | 32 | 1 | 20 | 20 | 1 | 20 | 20 | Y | 423424..428044 | – | 1381 |
| 61 | s21138–s21217 | L51887–52440 | 320 | 640 | 640 | 20 | 80 | 4 | 20 | 80 | 32000 | **N** | 428544..448524 | 530944..530992 | 117 |
| 62 | s21335–s21344 | L52605–52668 | 80 | 80 | 160 | 5 | 10 | 2 | 5 | 10 | 250 | **N** | 531264..533324 | 533824..533840 | 323 |

**Arithmetic check.**  Rows 1–11, 13–16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58 and 60 satisfy
`M/16 · N/8 · K/32 == mma`: every A×B tile of the grid is evaluated once and the
printed `M`/`N`/`K` is the true GEMM shape.

Rows 12, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
57, 59, 61 and 62 do **not** and are *banded*: only a diagonal band of the M×N
grid is evaluated, so the printed `M`/`N` are the extents spanned by the fragment
operands.  The chain structure pins the real shape down:

* **P12** (`48 mma`, 16 chains of 3): chain `c = 4·(c>>2) + (c&3)` is
  `(A_{3·(c>>2)+k}, B_{2·(c&3)+k})`, `k = 0,1,2` — 4 row tiles × 4 column-pair
  groups, 3 k-steps.  Real shape **64×32×96**, all 16 output tiles evaluated.
  (Same as enc0's P6 and enc3's P10.)
* **P21/23/…/59** (`20 mma`, 4 chains of 5): chain 0 is
  `(A0,B0) (A1,B2) (A2,B4) (A3,B6) (A4,B8)`, chain 1 `(A_j, B_{2j+1})`, chains 2/3
  the same with `B10…B19`.  Real shape **80×80×160** with only the 5×2 = 10
  diagonal output tiles of a 5×10 grid evaluated.
* **P61** (`80 mma`, 4 chains of 20): chain 0 is `(A_j, B_{2j})`, chain 1
  `(A_j, B_{2j+1})`, chains 2/3 with `B_{40+2j}`, `B_{40+2j+1}`, `j = 0…19`.
  Real shape **320×320×640** over 4 output tiles.
* **P62** (`10 mma`, 2 chains of 5): `(A_j, B_{2j})` / `(A_j, B_{2j+1})`,
  `j = 0…4`.  Real shape **80×80×160** over 2 tiles.

**Phases that read no weight-image B operand:** **P11** (`Bo = [None, None]` on all
48 mma; its `B` is a register pack of P9's D, §3/E10) and **P12** (its `B` is
packed from the `movmatrix` of P10's D, §3/E11).  Every other phase loads its `B`
from `W`.

**Weight-image extent.**  The largest byte offset used anywhere is 533840 (P62's
second C seed, `ld.global.u32 [%rd900+533840]`); with the 4-byte read the image is
**533844 bytes**, 2522 distinct weight-image byte offsets (`weight_image_size`).  The single largest
interior gap is `448524…530944` (the 82 KiB between P61's B block and P61's bias).

## 3. The epilogues

Phases 1…10 (the stem), 13…59 (the norm + 20 MLP branch pairs) and 60…62 (the
merge) are highly repetitive; each distinct body is written out once and the
repeats are tabulated.

### E1…E9 — s7578–s7584, s7609–s7615, s7640–s7646, s7671–s7677, s7702–s7708, s7733–s7739, s7764–s7770, s7795–s7801, s7826–s7832: next-B loads

Nine identical 7-statement epilogues.  Each issues the two `v4.u32` fragments of
the *next* phase's B operand (`W + 16*laneid + {base, base+512}`):

| epilogue | range | loads (statement / line) | offsets | next phase's W block |
|---|---|---|---|---|
| E1 | s7578–s7584 | s7582 / L12057, s7584 / L12061 | 5440, 5952 | P2 `5440..5964` |
| E2 | s7609–s7615 | s7613 / L12238, s7615 / L12242 | 1344, 1856 | P3 `1344..1868` |
| E3 | s7640–s7646 | s7644 / L12419, s7646 / L12423 | 6464, 6976 | P4 `6464..6988` |
| E4 | s7671–s7677 | s7675 / L12600, s7677 / L12604 | 2368, 2880 | P5 `2368..2892` |
| E5 | s7702–s7708 | s7706 / L12781, s7708 / L12785 | 7488, 8000 | P6 `7488..8012` |
| E6 | s7733–s7739 | s7737 / L12962, s7739 / L12966 | 3392, 3904 | P7 `3392..3916` |
| E7 | s7764–s7770 | s7768 / L13143, s7770 / L13147 | 8512, 9024 | P8 `8512..9036` |
| E8 | s7795–s7801 | s7799 / L13324, s7801 / L13328 | 4416, 4928 | P9 `4416..4940` |
| E9 | s7826–s7832 | s7830 / L13505, s7832 / L13509 | 9536, 10048 | P10 `9536..10060` |

**Outputs:** the 8 registers of the next phase's B.  Nothing else.

### E10 — s7857–s8027: the score-phase bias tiles and the phase-11 operands

171 statements: 24 `ld.weak.global.ca.v4.u32` (s7861/L13686 … s7907/L13778), 80
`cvt.rn.satfinite.e4m3x2.f16x2` (s7908/L13781 … s8026/L14057), 40 `mov.b32`
(s7910/L13786 … s8027/L14059), 25 `add.s64`, plus the per-lane address pair.

1. **Score-bias C seeds** (s7860–s7907): 24 `v4.u32` at
   `W + 16*laneid + 10560 + 512*i`, `i = 0…23` → the 96 `C` operands of phase 11
   (byte block `[10560, 22348]`, 11792 bytes of f16).  Phase 11's own mma confirm
   the `4*(n>>1)+m` tile index: for the first A-tile the C offsets run
   `10560, 10568` (s8028, s8029), `12608, 12616` (s8030, s8031), `14656, 14664`,
   `16704, 16712`, `18752, 18760`, `20800, 20808` (s8038, s8039) — i.e.
   `10560 + 2048*(n>>1) + 8*(n&1)` — and the second A-tile starts at `11072`
   (`s8040`) = `10560 + 512`.  Structurally identical to enc0's `[8384, 20672)`
   and enc3's `[8448, 20236)`.
2. **B packs** (s7910–s7979, sources from s7908): the first 24 `mov.b32`
   destinations = phase 11's twelve B fragments (24 b32), each a
   `cvt.rn.satfinite.e4m3x2.f16x2` pair of two registers written by **phase 9's
   mma** (e.g. `s7909: cvt %rs2341, %r7325`, `s7915: cvt %rs2345, %r7335`,
   `s7910: mov.b32 %r8357, {%rs2341, %rs2345}` = B fragment 0).
3. **A packs** (s7982–s8027): the remaining 16 `mov.b32` destinations
   (`%r8103…%r8106`, `%r8223…%r8226`, `%r8343…%r8346`, `%r8463…%r8466`) = phase
   11's four A fragments (16 b32), also from phase 9's D
   (`s8017: cvt %rs2413, %r7445`, `s8021: mov.b32 %r8463, {%rs2413, %rs2414}`).

As in enc3, **fragment 0 of phase 11's A and of its B are bit-identical** (both
packs read the same two source registers).

**Inputs:** phase 9's 48 D registers.  **Outputs:** phase 11's 16 A + 24 B
registers and its 96 C seeds.  **Memory:** 24 weight loads only.

### E11 — s8076–s11919: the 64×96 score softmax

3844 statements, **zero memory access**; byte-for-byte the same code as enc3's E9
(and enc0's E5) with shifted registers.  Histogram: 1048 `mov.b32`, 672
`cvt.rn.f16.f32`, 384 `cvt.rn.f16.f64`, 192 each `fma.rn.f16x2` /
`cvt.f32.f16` / `rcp.approx.ftz.f32` / `mul.f16`, 96 each `max.f16x2`/`min.f16x2`/
`neg.f16x2`/`mul.f16x2`/`and.b32`, 48 `movmatrix`, 16 `shfl.sync.bfly`, 144 packs.

| range | statements | what |
|---|---|---|
| s8076–s10193 | 2118 | 96 per-element exponent groups |
| s10194–s10273 | 80 | in-lane partial-sum trees (`add.f16x2`) |
| s10274–s10408 | 135 | 16 `shfl.sync.bfly.b32` (s10281/L21268 … s10402/L21572) → 8 row sums |
| s10411–s11172 | 769 | 192 `rcp.approx.ftz.f32` → 96 reciprocals |
| s11178–s11654 | 477 | 192 `mul.f16` → 96 probabilities |
| s11656–s11703 | 48 | `movmatrix.sync.trans.aligned.m8n8.b16` of **phase 10's** 48 D registers |
| s11704–s11918 | 216 | 144 packs |

**(a) per-element exponent, s8076–s10193.**  Per f16 half `x` of each of phase
11's 96 D f16x2 registers:

```
m = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)
u = f16(1.375 + m*(0.92724609375 - m^2))
E = (u_bits << 5) & 0x7FE07FE0        // first: s8102 shl.b32, s8103 and.b32
```

identical to enc0's E5(a): **not** `exp`, and the two `fma.rn.f16x2` at
s8096/L14453 and s8097 are the cubic, `neg.f16x2` at s8091/L14439 the `−m`.

**(b) row sums, s10194–s10408.**  In-lane `add.f16x2` tree, then `shfl.sync.bfly`
over lane bits 0 and 1 (`%r10789 = 1`, `%r10798 = 2`, membermask `%r10800 = −1`),
then `add.f16x` of the halves and broadcast.

**There is no row maximum in this epilogue.**  Every `max.f16x2` (96, s8091 …
s10183) and `min.f16x2` belongs to the clamp of (a); the only shuffles are the 16
butterflies above.

**(c) reciprocity and probability, s10411–s11654.**  192 `rcp.approx.ftz.f32`
wrapped as `cvt.f32.f16 → rcp → cvt.rn.f16.f32`, then 192 `mul.f16`.

**(d) V transpose, s11656–s11703.**  Exactly 48 `movmatrix…m8n8.b16` on phase
**10**'s 48 D registers (`s11656 L23893` … `s11703 L24034`).

**(e) packs, s11704–s11918.**  144 packs: the first 24 `mov.b32` → phase 12's 12
B fragments (the transposed V), the rest → phase 12's 12 A fragments.

**Inputs:** phase 11's 96 D, phase 10's 48 D, `%r10789/%r10798/%r10800`.
**Outputs:** phase 12's 48 A / 24 B registers plus the `%f476`, `%fd383`,
`%fd385`, `%f478`, `%f480` constants.  **Memory:** none.

### E12 — s11968–s12241: P·V epilogue — pack, shared staging, residual table, next B/C

274 statements.  Histogram: `mov.b32`×96, `add.f16`×80,
`cvt.rn.satfinite.e4m3x2.f16x2`×32, `ld.global.v2.u16`×20, `add.s64`×12,
`ld.weak.global.ca.v4.u32`×10, `shl.b32`×5, `add.s32`×4, `mov.u32`×4,
`st.shared.v4.u32`×4, `bar.sync`×2, `bra`×1, `ld.shared.v4.u32`×1,
`mul.wide.u32`×1, `cvt.u64.u32`×1, `and.b64`×1.

**(a) require and shared staging, s11969–s12025.**

```
s11969 L24877: shl.b32 %r12002, %r9, 11             // tid.z << 11
s11970 L24878: add.s32 %r12003, %r11752, %r12002
s11973 L24883: add.s32 %r12005, %r12003, %r12004    // + 16*laneid
s11986 L24912: st.shared.v4.u32 [%r12005], {%r12009, %r12008, %r12007, %r12006}
```

32 packs of phase 12's D → 16 b32 (96 `mov.b32` in the whole epilogue; the other
80 pair the table adds) stored one `v4.u32` each at `+0`, `+512`, `+1024`, `+1536`
(s11986, s11999, s12012, s12025).

**(b) `bar.sync 0` (s12026) and skip guard.**  `s12027 L25001: @%p8 bra
$L__BB1_422` with `%p8 = (tid.z > 3)` (s20–s21) — phases 13…20 are only run for
`tid.z ≤ 3`.

**(c) staging read-back, s12028–s12033.**  `%r29518 = %r11752 + (tid.z<<9) +
(laneid<<4)` (s12028–s12032), `ld.shared.v4.u32 [%r29518]` → `%r29519 … %r29522` =
**phase 13's A operand** (one 16-byte fragment, `nA = 1`).

**(d) next B, s12034–s12056.**  10 `v4.u32` at `W + 16*laneid + 22848 + 512*j`,
`j = 0..9` → phase 13's 20 B fragments (block `[22848, 27468)`).

**(e) residual column table, s12062–s12081.**  20 `ld.global.v2.u16` at
`W + ((laneid<<2)&12) + {221504 + 16*i}`, `i = 0…19` → 40 f16 covering the 160
columns of phase 13's output.

**(f) residual add, s12082–s12241.**  80 `add.f16` — two per f16x2 — each
`table[column] + <half of the long-lived prologue residual>`:

```
s12084 L25089: add.f16 %rs3965,%rs6450,%rs3967    // %rs6450 = W+221504, %rs3967 = residual
s12083 L25085: add.f16 %rs3968,%rs6451,%rs3970
s12085 L25092: mov.b32 %r12073, {%rs3965, %rs3968}
```

The 40 results `%r12073 …` are **phase 13's C operands**.  The residual halves are
`%r31707`, `%r31706`, … split at s5443 ff. (§1.2).

**Inputs:** phase 12's D, the shared staging, `W`.  **Outputs:** phase 13's A
(4 regs), B (40 regs) and C (40 regs).

### E13…E19 — s12262–s12288, s12309–s12335, s12356–s12382, s12403–s12429, s12450–s12476, s12497–s12523, s12544–s12570: next-A from shared, next-B from W

Seven identical 27-statement epilogues (`add.s64`×11, 10 `v4.u32`, `mov.u32`×2,
`shl.b32`, `add.s32`, 1 `ld.shared.v4.u32`, `mul.wide.u32`):

```
s12264 L25627: add.s32 %r29525, %r29516, %r29524   // %r29516 = %r11752 + (tid.z<<9)
s12265 L25628: ld.shared.v4.u32 {…}, [%r29525+2048]
s12267 L25632: mul.wide.u32 %rd648, %r12266, 16
s12268 L25633: add.s64 %rd649, %rd2, %rd648
```

| epilogue | range | shared read offset | becomes | B offsets (10 loads) | next phase's W block |
|---|---|---|---|---|---|
| E13 | s12262–s12288 | s12265, `+2048` | P14 A | 50496 … 55104 | P14 `50496..55116` |
| E14 | s12309–s12335 | s12312, `+4096` | P15 A | 78144 … 82752 | P15 `78144..82764` |
| E15 | s12356–s12382 | s12359, `+6144` | P16 A | 105792 … 110400 | P16 `105792..110412` |
| E16 | s12403–s12429 | s12406, `+8192` | P17 A | 133440 … 138048 | P17 `133440..138060` |
| E17 | s12450–s12476 | s12453, `+10240` | P18 A | 161088 … 165696 | P18 `161088..165708` |
| E18 | s12497–s12523 | s12500, `+12288` | P19 A | 188736 … 193344 | P19 `188736..193356` |
| E19 | s12544–s12570 | s12547, `+14336` | P20 A | 216384 … 220992 | P20 `216384..221004` |

### E20 — s12591–s13619: the RMS norm before the MLP

1029 statements: 302 `mov.b32`, 240 `mul.f16`, 82 `add.f16`, 80 `cvt.f32.f16`,
80 `rsqrt.approx.ftz.f32`, 80 `cvt.rn.f16.f32`, 42 `add.f16x2`, 4
`shfl.sync.bfly`, 40 packs, 20 `ld.global.v2.u16`, 10 weight `v4.u32`, 4
`ld.global.u32`.

| range | what |
|---|---|
| s12594–s12751 | 80 `mul.f16` — squares of phase 20's 40 D f16x2 |
| s12753–s12817 | 42 `add.f16x2` — in-lane reduction, with the 4 `shfl.sync.bfly` over lane bits 0,1 at s12795/L27521, s12801, s12810, s12816/L27575 |
| s12819–s12940 | 82 `add.f16` — `+ 2⁻¹³` (`%fd1`, s5976) broadcast into the f16x2 pairs |
| s12946–s13259 | 80 `rsqrt.approx.ftz.f32` (s12946/L27961 … s13259/L28469) |
| s13268–s13287 | 20 `ld.global.v2.u16` gain at `W + ((laneid<<2)&12) + {221824 + 16*i}` (`s13289 mul.f16 %rs4692,%rs4693,%rs6490` with `%rs6490 = W+221824`) |
| s13289–s13407 | 80 `mul.f16` — `inv · gain` |
| s13408–s13526 | 80 `mul.f16` — `x · (inv·gain)` (`s13408 mul.f16 %rs4935,%rs5217,%rs4695`) |
| s13532–s13550 | 10 weight `v4.u32` = phase 21's B (`222144 + 512*j`) |
| s13556–s13559 | 4 `ld.global.u32` at `W + ((laneid<<2)&12) + {227264,227280,227296,227312}` = phase 21's C bias |
| s13560–s13618 | 40 packs of the normalised values → phase 21's A |

**What it computes.**  The same per-**row** RMS reduction as enc3's E14, over the
**160** columns of a M=16/N=160 tile: the lane's 20 one-per-n-tile partials for
each of its two rows are tree-summed in-lane, the four `t = laneid&3` lanes are
folded with two butterfly steps, the f16x2 halves are added, then

```
inv        = f16(rsqrt.approx(f32(f16(sumsq + 2^-13))))
out[row,c] = x[row,c] · inv · gain[c]
```

There is no mean subtraction and no `1/N`.  The 80 `rsqrt` results are the two row
sums re-emitted once per (row, n-tile) pair, not 80 distinct sums (the half-copies
that feed them are duplicates, e.g. `s12818`-range `add.f16 … , %rs…, %rs…` with
identical addends).  `gain[c]` is per column.

**Outputs:** phase 21's B (40 regs) and C (4 regs), 20 packed A registers, and the
normalised f16x2 registers that E21/E22 re-read.

### E21 — s13640–s13960: clamped cubic activation + column bias + pack

321 statements: `add.f16`×80, `mov.b32`×76, `cvt.rn.f16.f32`×32, `mul.f16x2`×24,
20 `ld.global.v2.u16`, 10 weight `v4.u32`, 8 each `neg/max/min/abs/sub/add.f16x2`,
8 packs, `mov.f32`×4.

**(a) constants and activation, s13641–s13780.**  Eight groups; each materialises
the activation constants (`s13641 L29561: mov.f32 %f1117, 0f3ED306EB`, `s13644
%f1118, 0f3DA60DD6`, `s13647 %f1119, 0f3F000000`, `s13650 %f1120, 0f40000000`)
and then, on **phase 21's D**,

```
y   = clamp(x, -2, +2)                              // s13653 neg, s13654 max, s13655 min
g   = 0.5 + y*(0.412109375 - 0.0810546875*|y|)      // abs, mul, sub, mul, add
out = x * g                                         // s13661 mul.f16x2
```

The same cubic ramp as enc0's E13 / enc3's E15 — **not** a sigmoid, GELU or SiLU.

**(b) column bias, s13809–s13948.**  20 `ld.global.v2.u16` at
`W + ((laneid<<2)&12) + {232464 + 16*i}` then 80 `add.f16`, e.g.
`s13932 L30428: add.f16 %rs5416,%rs6524,%rs5418` — the second addend is a half of
phase **20**'s D (a residual).  The 40 results `%r14926 … %r14956` are **phase
22's C operand**.

**(c) pack, s13949–s13959.**  8 packs of the **activation outputs**
(`s13949 L30478: cvt.rn.satfinite.e4m3x2.f16x2 %rs5453, %r14617`) → phase 22's A
fragment.

**(d) next B, s13785–s13803.**  10 `v4.u32` at `227328 + 512*j` → phase 22's B.

**No stores and no `ld.global` beyond the two tables.**

### E22 / E24 / … / E58 — s13981–s14072 and its 19 repeats: next B/C loads + pack

92 statements, identical shape, for every even phase 22…58:

```
s13984 L30650: add.s64 %rd263, %rd676, 232768
s13985 L30652: ld.weak.global.ca.v4.u32 { %r14958,…},[%rd263]    // 232768 + 512*j, 10 loads
s14009 L30697: ld.global.u32 %r…, [%rd…+237888]                  // C bias, 4 words at +16*i
s14013 L30702: cvt.rn.satfinite.e4m3x2.f16x2 %rs…, %r…           // 40 packs
s14015 L30707: mov.b32 %r…, {%rs…, %rs…}                         // 20 b32 = 5 A fragments
```

The 40 pack sources (`%r28751 %r28752 %r28753 …`, `s14013 L30702: cvt %rs5461,
%r28752`) are the **normalised values** E20 produced with the second of its
multi-sets, i.e. phase 23's A operand is the *same* normed tile phase 21's A was
built from — phases 21 and 23 are two parallel GEMMs off one normed tile, each
with its own weight block and bias.  The 4 C words are
`W + ((laneid<<2)&12) + {bias_base + 16*i}` (`s13556 L29274:
ld.global.u32 %r14283, [%rd669+227264]`, `s13557`/`s13558`/`s13559`) and each is
used in **both halves** of the *next odd* phase's mma C operand: for the P23 bias
loaded by E22, `s14073 C={%r15048,%r15048}` with `[%r15048=W+237888]`,
`s14074 C={%r15058,%r15058}` with `W+237904`.

| epilogue | range | next-B block | next-C bytes |
|---|---|---|---|
| E22 | s13981–s14072 | 232768..237388 | 237888..237936 |
| E24 | s14285–s14376 | 243072..247692 | 248192..248240 |
| E26 | s14589–s14680 | 253376..257996 | 258496..258544 |
| E28 | s14893–s14984 | 263680..268300 | 268800..268848 |
| E30 | s15197–s15288 | 273984..278604 | 279104..279152 |
| E32 | s15501–s15592 | 284288..288908 | 289408..289456 |
| E34 | s15805–s15896 | 294592..299212 | 299712..299760 |
| E36 | s16109–s16200 | 304896..309516 | 310016..310064 |
| E38 | s16413–s16504 | 315200..319820 | 320320..320368 |
| E40 | s16717–s16808 | 325504..330124 | 330624..330672 |
| E42 | s17021–s17112 | 335808..340428 | 340928..340976 |
| E44 | s17325–s17416 | 346112..350732 | 351232..351280 |
| E46 | s17629–s17720 | 356416..361036 | 361536..361584 |
| E48 | s17933–s18024 | 366720..371340 | 371840..371888 |
| E50 | s18237–s18328 | 377024..381644 | 382144..382192 |
| E52 | s18541–s18632 | 387328..391948 | 392448..392496 |
| E54 | s18845–s18936 | 397632..402252 | 402752..402800 |
| E56 | s19149–s19240 | 407936..412556 | 413056..413104 |
| E58 | s19453–s19544 | 418240..422860 | 423360..423408 |

### E23 / E25 / … / E59 — s14093–s14264 and its 18 repeats: clamped cubic activation + pack

172 statements; identical op mix to E21 minus the table loads and the `mov.f32`
block (36 `mov.b32`, 32 `cvt.rn.f16.f32`, 24 `mul.f16x2`, 8 each
`neg/max/min/abs/sub/add.f16x2`, 10 weight `v4.u32`, 8 packs):

```
s14094 L30986: cvt.rn.f16.f32 low, %f1117
s14102 L31009: neg.f16x2 %r15244,%r15243
s14109 L31037: add.f16x2 %r15250,%r15243,%r15249
s14234 L31471: ld.weak.global.ca.v4.u32 { …},[%rd…]      // next B, 10 loads
s14253 L31510: cvt.rn.satfinite.e4m3x2.f16x2 …            // 8 packs of activation outputs
```

Same function as E21(a) on the odd phase's D, 8 packs of the activation *outputs*
→ the next even phase's A, and 10 weight loads → its B.  **No column bias, no
`ld.global`.**  The 18 repeats are at s14093, s14377, s14681, s14985, s15289,
s15593, s15897, s16201, s16505, s16809, s17113, s17417, s17721, s18025, s18329,
s18637, s18937, s19241, s19545 (phases 23…59 odd).

#### Shape of the attention + MLP region, E12–E60

```
P11  A = pack(P9.D)   B = pack(P9.D)  C = bias@10560+512i  → scores 64×96
     E11: cubic-exponent softmax (no row max), 48 movmatrix(P10.D) + packs
P12  A = pack(scores)  B = pack(transpose(P10.D))  C = 0  → 64×32
     E12: pack → shared slot, read back as P13 A; table@221504 + residual → P13 C
P13  A = shared, B = W@22848,  C = table+residual  → D13
     E13: shared slot +2048 → P14 A; load P14 B
P14…P19  the same for the shared slots +4096 …, +14336
P20  A = shared, B = W@216384, C = D19  → D20 (16×160)
     E20: row-RMS norm over 160 columns (+2^-13, rsqrt, gain@221824), pack → P21 A
P21  A = normed tile, B = W@222144, C = bias@227264  → D21 (banded 80×80×160)
     E21: cubic activation(D21); table@232464 + D20 → P22 C; pack → P22 A
P22  A = pack(activation), B = W@232768, C = D20+table  → D22
     E22: pack of the *normed* tile → P23 A; load P23 B and C
P23  A = normed tile, B = W@237952, C = bias@237888 → D23
     … the P21/P22 pair then repeats 19 more times (P23/P24 … P59/P60).
P60  A = pack(activation P59), B = W@423424, C = D58 → D60 (16×160)
     E60: arena scatter, shared staging, next B/C
P61  A = 20 shared slots, B = W@428544, C = bias@530944 → 4×20 k-steps
     E61: pack + scatter to param+56, next B/C
P62  A = pack(E61), B = W@531264, C = bias@533824 → D62
     E62: the final epilogue
```

### E60 — s19757–s21137: pack → plane-arena scatter → shared staging → next B/C

1381 statements.  Histogram: `add.s32`×218, `or.b32`×129, `shl.b32`×99,
`and.b32`×80, `ld.shared.u32`×80, `mul.wide.u16`×69, `add.s64`×66, `shr.u32`×51,
`setp.lt.s32`×44, `bra`×41, `$label`×41, `cvt.rn.satfinite.e4m3x2.f16x2`×40,
`selp.b32`×40, `ld.weak.global.ca.v4.u32`×40, `mov.b32`×20, `st.shared.u32`×20,
`st.global.u32`×20, `bar.sync`×2, `ld.global.u32`×4.

**(a) packs, s19757–s19816.**  40 `cvt.rn.satfinite.e4m3x2.f16x2` + 20 `mov.b32`
→ `%r371 … %r390` = phase 60's 16×160 output as 20 b32 = 40 e4m3 per lane.

**(b) plane-arena scatter, s19817–s20029.**  `%rd5 = cvta(ld.param.u64 [%rd1+-32])`
(s19817–s19818; `%rd1 = param_0+80` ⇒ **`param_0+48`**), lane decomposition
`s20035 shr %r424, laneid, 2` (`g`), `s20036 and %r425, laneid, 3` (`t`), and 20
guarded `st.global.u32` (`s19843 L50422: st.global.u32 [%rd810], %r371` …
`s20029 L50646: … %r390`), each preceded by a
`setp.lt.s32 %p, idx, 0` + `bra` guard.  The index is built from
`(%r1 + ((g+8j)&7))`, `((tid.z<<1) + ((g+8j)>>3))` and `(%r425 | 4)` in the same
way as enc3's E46(b) (§6/U4).

**(c) shared staging, s20033–s20215.**  `bar.sync 0`, then 20 `st.shared.u32` at
`%r29635 = %r11752 + (idx<<2)` with
`idx = t | ((laneid&28) | (y<<5))`, `y = (tid.z<<1) + (laneid>>5)`
(`s20051 L50672: st.shared.u32 [%r29635], %r371`), then the `+512*j` variants.

**(d) staging read-back, s20222–s21030.**  `bar.sync 0` (s20220), then 80
`ld.shared.u32` at `%r11752 + (idx2<<2)` with
`idx2 = ((laneid<<1)&24) | ((laneid<<2)&0xFFFFFFC0) | (laneid&3)`
(`s20234–s20241`) plus byte offsets `{0, 1024, …, 7168}` and the `+16` variants
(`s20242 L50905: ld.shared.u32 …`).  These 80 values are **phase 61's 20 A
fragments**.  `ld.param.v2.u32 [%rd940+-80]` at s20222 re-reads `param_0+0`.

**(e) next B/C, s21039–s21137.**  Inside the loop `$L__BB1_406` (s21041),
`s21039 add.s64 %rd7, %rd947, 428544`, `s21042: mul.lo.s32 %r31250, %r31708, 640`
then 40 `v4.u32` at `W + 16*laneid + 428544 + 640*%r31708 + 512*j`, `j = 0…39`
(`s21048–s21126`) = phase 61's 80 B fragments, and 4 `ld.global.u32` at
`W + ((laneid<<2)&12) + {530944, 530960, 530976, 530992}` (s21134–s21137) =
phase 61's C bias.

**Outputs:** `%r371…%r390` (read again by (c)), phase 61's A/B/C.
The shared arena is re-used four times in this epilogue — `%r11752` appears as the
base of the E60 stores and of the P61 read-back.

### E61 — s21218–s21334: pack → surface scatter → next B/C

117 statements: 8 packs + 2 `mov.b32` (s21218–s21227), 4 guarded stores to
**`param_0+56`** (s21257 `st.global.v2.u16`, s21267 `st.global.u32`,
s21298 `st.global.u32`, s21306 `st.global.v2.u16`), 5 `bra`, then 5 weight
`v4.u32` (s21319–s21327) and 2 `ld.global.u32` (s21333–s21334).

```
s21253 L52499: setp.lt.s32 %p381, %r545, 0
s21254 L52500: @%p381 bra $L__BB1_408
s21255 L52502: mul.wide.s32 %rd898, %r545, 4
s21257 L52504: st.global.v2.u16 [%rd899], {%rs6698, %rs6699}
```

The destination is `param+56` (`%rd6 = cvta(%rd849)`, `%rd849 = ld.param.u64
[%rd945+-24]`, s21037–s21038), **not** the plane arena; the extents are
`param_0+0`, halved (`s20223 shr %r451, %r29720, 1` = `ey>>1`, `s20224 shr %r452,
%r29719, 1` = `ex>>1`, `s20225 shl %r453, %r452, 3` = the row stride).  The index
bases use `%r536 = %r1>>1` (s21036) and `%r535 = %r3>>1` (s21033).

Then phase 62's B (5 `v4.u32` at `531264 + 512*j`) and C (2 `ld.global.u32` at
`533824`, `533840`).

### E62 — s21345–s21667: the final epilogue

323 statements: 4 `st.shared.u32` (s21357–s21374), 2 `bar.sync`, 5 `ld.shared.u32`
(s21414–s21422), `mul.ftz.f32`/`ex2.approx`/`add.ftz.f32`/`div.approx`×12,
12 `fma.rn.f16x2`, 4 `shfl.sync.down.b32`, 9 `tex.base.2d.v4.f16.s32`
(s21526–s21585), 3 `sust.b.2d.v4.b16.zero` (s21649, s21650, s21665).

**(a) shared staging, s21345–s21374.**

```
s21348 L52679: and.b32 %r557, %r31392, 3                    // t
s21350 L52681: shl.b32 %r558, %r31608, 4                    // %r31608 = tid.z
s21351 L52682: add.s32 %r31395, %r556, %r558                // g + (tid.z<<4)
s21354 L52685: mad.lo.s32 %r31398, %r31397, 6, %r557
s21357 L52688: st.shared.u32 [%r559], %r31372
```

Phase 62's D (`%r31372/%r31382` and `%r31392`-group counterparts) is written to
`%r31615 + 4*(6*index + t)` with a `%r560 > 5` guard on the second word.

**(b) row max and softmax, s21387–s21613.**  After `bar.sync 0` (s21387),
`%r31565 = smem + 4*(6*%r562 + 48*%r563)` (s21410–s21413) and
`s21414–s21422` read a **6-element row** (`ld.shared.u32 [%r31565] … [+20]`); then

```
max.f16x2 / max.f16 over the row                                   // row maximum
sub.f16x2  x - rowmax                                              // e.g. s21502-region
mul.ftz.f32 by log2(e), ex2.approx.ftz.f32                          // 2^((x-max)*log2e)
div.approx.ftz.f32 1.0(rowsum)   with %f1152 = 0f3F800000 (s21429)
mul.f16x2  p = e / rowsum
```

**This epilogue has a real row maximum** (unlike E11), and a true `2^u` `ex2`
rather than the cubic mantissa trick.

**(c) texture gather, s21421–s21590.**  One texture (`load.param.u64 [%rd943+8]` at
s21524, copied into `%rd916 … %rd932`, s21525–s21572) with 9
`tex.base.2d.v4.f16.s32` reads forming a 3×3 stencil; the 8 neighbour coordinates
are clamped by `selp` chains with `1` as the out-of-range sentinel (the same
pattern as enc3's E48(c)).  Each texel is weighted by one of the five softmax
values with `fma.rn.f16x2` (`s21530 L52979`, `s21547 L53007`, …,
`s21580 L53074`) and one extra texel is folded in with `fma.rn.f16`.

**(d) second sigmoid, difference, cross-lane sum, s21591–s21648.**  Three
`mul.f16` + `sub.f16` products and differences, `%f1148 = 0.25` (s21606), then the
`shfl.sync.down` cascade: delta `1` (`%r31545`, s21424) and `8` (`%r31553`,
s21417), membermask `%r31555 = -1` (s21510), clamp operand `(WARP_SZ<<8)-8192 | 31`
(s21619, s21625, s21632, s21641), each followed by `add.f16x2` / `add.f16`.

**(e) surface stores, s21646–s21667.**

```
s21646 L53255: ld.param.u64 %rd936, [%rd943+40]
s21647 L53256: ld.param.u64 %rd938, [%rd943+48]
s21649 L53259: sust.b.2d.v4.b16.zero [%rd934, {%r31559,%r565}], {%rs6803,%rs6806,%rs6809,%rs6840}
s21650 L53262: sust.b.2d.v4.b16.zero [%rd936, {%r31559,%r565}], {%rs6794,%rs6797,%rs6800,%rs6841}
s21665 L53280: sust.b.2d.v4.b16.zero [%rd938, {%r31587,%r31588}], {%rs6842,%rs6843,%rs6827,%rs6845}
```

Three 4-channel b16 surface writes.  The handles are `param_0+32` (`%rd934`, read
at s21421), `+40` and `+48`; the texel source is the texture at `param_0+8`
(s21524).  The third store is guarded (`s21655 … bra`) and stores `0` in its
fourth channel as in enc3.

**Inputs:** phase 62's D, the shared staging of (a), `param+0/+8/+32/+40/+48`.
**Outputs:** none (the kernel `ret`s at s21667 / L53284).

## 4. The plane-arena and surface writes

`param_0+48` is reached once (`s19817 ld.param.u64 %rd808, [%rd1+-32]` with
`%rd1 = param_0+80` at s4) and is never the base of an `ld.*` in this entry.  The
complete set of global writes in the 21667-statement entry:

* **E60, 20 × `st.global.u32 [%rd5 + 4*idx], v`** — s19843, s19852, s19885, s19893,
  s19902, s19910, s19919, s19927, s19936, s19944, s19953, s19961, s19970, s19978,
  s19987, s19995, s20004, s20012, s20021, s20029 (§3/E60(b)).  Value: phase 60's
  requantised 16×160 output, 20 b32 per lane.
* **E61, 4 stores to `param_0+56`** — s21257 (`v2.u16`), s21267 (`u32`),
  s21298 (`u32`), s21306 (`v2.u16`).
* **E62, 3 `sust.b.2d.v4.b16.zero`** — s21649, s21650, s21665.

Nothing else writes to global memory.

## 5. Uncharacterised

**U1. The identity of the 11792-byte score bias (P11's C operand).**  Phase 11
adds the f16 table `[10560, 22348]` indexed as
`byte = 10560 + 512*(4*(n>>1) + m) + 16*laneid + 8*(n&1)`.  Whether it is a
relative position bias, a learned score bias or a per-lane constant cannot be
decided from the PTX.  *Missing:* the weight image's layout description (the
`_prep` permutation).

**U2. The identity of the per-column tables.**  `W+221504` (E12's C seed add),
`W+221824` (E20's norm gain), `W+232464` (E21's post-activation add) and
`W+227264`/`227264+65536`… (the odd phases' GEMM bias) all use the
`W + ((laneid<<2)&12) + off` pattern.  Their role is computable (§3) but their
model names are not.

**U3. The shared-arena layout.**  The arena (16384 bytes) is written by the
prologue (s7494–s7504), E12 (s11986–s12025) and E60 (s20051–s20215), and read by
P1–P10 (s7513–s7542, at `%r11752 + 16*laneid + 512*k` with **no** `tid.z` term),
P13–P20 (s12033, s12265, …, s12547, at `%r11752 + (tid.z<<9) + 16*laneid + 2048*j`)
and P61 (s20242–s21030).  The prologue's store side and the E60 store side add
`tid.z` terms that the P1–P10 read side does not, so the mapping cannot be
reconstructed from the PTX alone.  *Missing:* the launch's `%tid.z`/`ctaid`
ranges, or a memory dump.

**U4. E60's / E61's index algebra.**  E60(b)'s arena stores, E60(c)'s shared
stores and E61's `param+56` stores each form index bases from
`(laneid&28) | (y<<5)` and `((g+8)<<2)&28 | (y2<<5)` plus `+512*j`, with
`y = (tid.z<<1) + (laneid>>5)` and `y2 = (tid.z<<1) + ((g+8)>>3)`, while E60(d)'s
read side uses `16*g + t` and `16*g + t + 128`.  I could not prove the
store↔load permutation, nor whether the `%r31708` loop (s21041, `+128` at s21308)
runs once or twice per `tid.z` (the guard `%r31708 > 159` at s20227/s21046 is
checked before the first body, exactly as in enc3).  *Missing:* a plane-arena dump.

**U5. The MLP block's shape.**  §3 establishes, from the pack sources and each
phase's C operand, that the MLP region is: one normed tile feeding every *odd*
phase (P21, P23, …, P59) as its `A`, each odd phase's activation feeding its
*paired* even phase as `A`, and the even phases' `C` chaining
(`D22 = … + D20 + table@232464`, `D24 = … + D22`, …, verified from each even
phase's C operands tracing to the previous even phase's mma).  Only the **first**
even phase adds the column table `@232464` to the residual
(`s13932 add.f16 %rs5416,%rs6524,%rs5418`); every later even phase's `C` is the
previous even phase's `D` alone.  Whether the 20 pairs are one wide MLP, a stack
of residual blocks, or something else is a model-level question the PTX cannot
answer.  *Missing:* the model source.

**U6. The stem's shift structure.**  Phases 1…10 read the same 30 shared A
fragments in five shift groups (P1/P2 at `+0/512/…`, P3/P4 at `+512/…`, P5/P6 at
`+1024/…`, P7/P8 at `+1536/…`, P9/P10 at `+2048/…`, s7513–s7542) and accumulate
in two chains (`P1→P3→P5→P7→P9` and `P2→P4→P6→P8→P10`; each phase's C operands name
the previous one's D, e.g. `s7616 C={%r5574,%r5575}` = P1's D).  Whether that is a
2×2 / 5-tap stencil is a property of the high-level model.  *Missing:* the model
source.  (enc3 has four shift groups and eight phases, so the tap count is a
per-stage constant.)

**U7. The meaning of the prologue residual vector.**  §1.2 tracks the f16 halves
`%rs…` back through `cvt.rn.f16x2.e4m3x2 %r31707, %rs128` (s5439) to the input
plane at `param_0+8` (s22–s23).  What that plane holds is not determinable from
the PTX.

**U8. The logical meaning of E62's final expression.**  The tail computes
`out = (acc − sigmoid(acc)) * 0.25` on three channels gathered by texture from a
3×3 stencil.  The arithmetic is exact in §3/E62(d); what model term it corresponds
to is not derivable.  (Same gap as enc0 §6/U6.)

**U9. The 82 KiB hole in the weight image.**  Between P61's B block
(`428544..448524`) and P61's bias (`530944`) there are 82420 bytes that no phase
in this entry reads.  Whether that region is padding, is unused by this stage, or
belongs to another kernel's parameters cannot be told from this entry.
*Missing:* the `_prep` code that builds the image.

## 6. Quick acceptance index

| requirement | where |
|---|---|
| every phase's M/N/K with the arithmetic checked | §2 (rows flagged **N** are banded; the real shapes are derived there) |
| weight image extent | §2 (533844 bytes, max offset 533840) |
| phases that read no weight operand | §2 (P11, P12) |
| score epilogue: row max? | **none** — §3/E11 (16 `shfl.sync.bfly`, no maximum) |
| score bias operand | §3/E10 (24 `v4.u32` at `W+10560+512*i`, indexed `4*(n>>1)+m`) |
| attention Q/K/V | §3/E10 (P11 A and B both from P9's D), §3/E11(d) (P10's D transposed) |
| the norm | §3/E20 (row RMS over 160 columns, `+2^-13`, `rsqrt.approx`, gain) |
| the activations | §3/E21, §3/E23 (clamped cubic ramp, 8 groups per phase) |
| patch merge | §3/E60 (arena + shared staging), §3/E61 (`param+56`), §3/E62 (textures + surfaces) |
| per-launch plane-arena writes | §4 |
