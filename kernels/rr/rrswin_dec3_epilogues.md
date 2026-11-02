# `cuda_dldn_engine_swin_dec3_kernel` — the epilogues of phases 1…48

Companion to `kernels/rr/rr_layer_spec.py dec3`.  Everything below was read out of
`~/.cache/d4r-rr-corpus/0020-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_dec3_kernel` (the entry opens at file line 1016, its body
holds statements s1…s19663, and s19663 / L47124 is the `ret`).

## 0. Conventions

* **`sNNNN` is a statement index** in the linearisation `rr_layer_spec.py`
  documents (statements of the entry body; a wrapped `{ … }` block and a 4-line
  `mma.sync` count as one statement, numbered at their first line).  **`LNNNNN` is
  the physical line** of the same instruction in the corpus file, so every claim
  can be checked with `sed -n 'LNNNNN,LNNNNN p' …ptx`.
* An **epilogue** is the run of statements between one phase's last `mma` and the
  next phase's first `mma`.  A phase's *prologue* (its weight loads and, for the
  register-fed phases, its operand construction) is written at the tail of the
  previous epilogue, so a few loads attributed here actually serve the next phase;
  each is named in place.
* `W` denotes the prepared weight image: `ld.param.u64 %rd2,
  [cuda_dldn_engine_swin_dec3_kernel_param_0+40]` (s8) then
  `cvta.to.global.u64 %rd50, %rd2` (s9).  Every `ld.weak.global.ca.v4.u32` in an
  epilogue reads `W + 16*laneid + imm`, and every bias/table row is read with the
  pattern `W + ((laneid<<2)&12) + imm` (s523–s526 is the canonical construction).
* `S` denotes the plane arena: `ld.param.u64 %rd763, [%rd8+48]` (s19164) then
  `cvta.to.global.u64 %rd5, %rd763` (s19165).  This is the entry's **only** read
  of `param_0+48`.
* `SM` denotes the 12800-byte shared array
  `_ZZ33cuda_dldn_engine_swin_dec3_kernel33DldnEngineSwinEncParamsStructBaseE4smem`
  (declared at file line 1027); it is materialised by `mov.u32` into a register at
  s954, s972, s1700, s1746, s19355 and others.
* `%r9 = %tid.z` (s433) is the **slab index**; it scales the input-buffer stride
  (×20480, s434), the shared-tile stride (×2048 via `<<11`, s9018; ×16 at s1709),
  the weight-image stride (×24576: `%r25647 = %r9*12288` s9101 then `*2` s9102),
  the bias stride (×256: `%r3569 = %r9<<7` s519) and ×512 (s13548).  dec3 has **no
  guard** on `%r9`; only a derived test `setp.gt.u32 %p499, %r186, 31` with
  `%r186 = %r9<<4` (s9075) restricts one store group.
* The C/D register layout and the weight-addressing formula are the settled family
  ones: `C/D reg0 = (row g, cols 2t,2t+1)`, `reg1 = (row g+8, same cols)`,
  `g = laneid>>2`, `t = laneid&3`; A `reg0 = row g, k = 2t+(j&1)+16*((j>>1)&1)`,
  `reg1 = row g+8 same k`, `reg2 = row g, k+8`, `reg3 = row g+8, k+8`.
* Anything I could not pin down from the PTX is in §6 under **uncharacterised**.

## 1. Shared vocabulary

### 1.1 Addresses and parameters an epilogue touches

| symbol | how it is obtained | statements |
|---|---|---|
| `param+0`, extents `(ex, ey)` | `ld.param.v2.u32 {%r3347, %r3348}, [param_0]` | s14 |
| `param+8`, the e4m3 feature buffer | `ld.param.u64 %rd51, [param_0+8]` → `cvta.to.global.u64 %rd52, %rd51` | s29–s30 |
| `param+24`, the f16 image buffer (**the skip input**) | `ld.param.u64 %rd3, [%rd1+-56]` → `cvta.to.global.u64 %rd4, %rd3`, where `%rd1 = param_0+80` | s4, s2780–s2781 |
| `param+40`, `W` (weight image) | `ld.param.u64 %rd2, [param_0+40]` → `cvta.to.global.u64 %rd50, %rd2` | s8–s9 |
| `param+48`, `S` (plane arena) | `ld.param.u64 %rd763, [%rd8+48]` → `cvta.to.global.u64 %rd5, %rd763` | s19164–s19165 |
| `param+80`, block origin `(ox, oy)` | `ld.param.v2.u16 {%rs344, %rs345}, [param_0+80]` | s5 |
| `param+88`, low-res feature **texture** | `ld.param.u64 %rd823, [%rd836+88]` with `%rd836 = param_0` | s19438, s19550 |
| `param+96`, **texture** added unconditionally | `ld.param.u64 %rd825, [%rd836+96]` | s19626 |
| `param+104`, **texture** added per-channel under a NaN guard | `ld.param.u64 %rd827, [%rd836+104]` | s19628 |
| `param+112`, output **surface** | `ld.param.u64 %rd830, [%rd836+112]` | s19449 |
| `param+136`, `(sx, sy)` pixel→uv scale | `ld.param.v2.f32 {%f1052, %f1053}, [%rd836+136]` | s19544 |
| `SM` (12800 B) | `mov.u32 %r3926, _ZZ33…E4smem` | s1746, s9554 etc. |
| per-lane weight address | `mul.wide.u32 %rd, %laneid, 16` + `add.s64 %rd, W, %rd` | s438–s439 and every load block |
| per-lane *table* address | `shl.b32 %r, %laneid, 2` → `cvt.u64.u32` → `and.b64 %rd, %r, 12` → `add.s64 %rd, W, %rd` | s523–s526 and copies |
| block origins `%r1`, `%r2` | `sub.s32 %r1, %r3342, %r3343` (s7) and `sub.s32 %r2, %r3346, %r3344` (s13); `%r3342 = ctaid.x<<3` (s3), `%r3346 = ctaid.y<<3` (s12) | s3, s7, s12–s13 |
| `%r7 = ex<<3`, `%r8 = ey*%r7` | s15, s16 | s15–s16 |

The per-lane table pattern matters: those loads read
`W + ((laneid<<2) & 12) + off`, so **the four lanes with equal `laneid&3` read the
same 16-byte record and the four words of a group are picked by the `off` stride**.
Those tables are indexed by **column**, not by row (same conclusion as enc0 §1.1).

### 1.2 Constants that survive between epilogues

| register | first definition | literal | identity |
|---|---|---|---|
| `%f476` | s9601 `mov.f32 %f476, 0f3C8CCB50` | 0.017186790704727173 | score scale (f16 `0x2466`) |
| `%fd383` | s9605 `mov.f64 %fd383, 0dBFE1CC0000000000` | −0.55615234375 | clamp low |
| `%fd385` | s9610 `mov.f64 %fd385, 0d3FE1CC0000000000` | +0.55615234375 | clamp high |
| `%f478` | s9616 `mov.f32 %f478, 0f3F6D6000` | 0.92724609375 | cubic coefficient |
| `%f480` | s9621 `mov.f32 %f480, 0f3FB00000` | 1.375 | cubic offset |
| `%fd1` | s7874 `mov.f64 %fd1, 0d3F20000000000000` | 2⁻¹³ | RMS-norm epsilon |
| `%f989` | s14672 `mov.f32 %f989, 0f3ED306EB` | 0.4121621549129486 | activation ramp |
| `%f990` | s14675 `mov.f32 %f990, 0f3DA60DD6` | 0.0810810774564743 | activation width |
| `%f991` | s14678 `mov.f32 %f991, 0f3F000000` | 0.5 | activation offset |
| `%f992` | s14681 `mov.f32 %f992, 0f40000000` | 2.0 | activation clamp |

The softmax constants `%f476/%fd383/%fd385/%f478/%f480` are defined **once**, in
phase 10's epilogue (s9601–s9621), and re-read by `cvt.rn.f16.f32`/`cvt.rn.f16.f64`
in the same block.  The activation constants `%f989…%f992` are defined **once** in
phase 16's epilogue (s14672–s14681, paired with
`s14673 cvt.rn.f16.f32 low, %f989`) and are then read back by the same
`cvt.rn.f16.f32 low, %f989…%f992` sequence in every later activation epilogue —
first occurrence of `0f3ED306EB` after s14672 is none; e.g. E18 does
`s15069 cvt.rn.f16.f32 low, %f989`, `s15071 …, %f990`, `s15073 …, %f991`,
`s15075 …, %f992`.

`rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
`div.approx.ftz.f32` are always wrapped as `cvt.f32.f16 → op → cvt.rn.f16.f32`, so
each step is rounded through f16.

### 1.3 Two quantisation helpers

* `cvt.rn.satfinite.e4m3x2.f16x2` — the requantisation used everywhere
  (`f16x2_to_e4m3x2_satfinite_bits` in `ptx/lib/zluda_ptx_impl.cpp`).  Neither dec3
  nor its siblings ever use the `.relu` form: the string `satfinite.relu` does not
  occur in the entry.
* `cvt.rn.f16x2.e4m3x2` — the inverse (unpack one e4m3 byte pair to an f16x2).

## 2. Phase table summary

From `rr_layer_spec.py dec3`: 48 phases, 1112 `mma`.  The tool prints M and N from
the count of *distinct* A and B fragments, so the identity
`M/16 * N/8 * K/32 == mma` is the check that the phase really is a dense GEMM:

* holds for phases 2–9, 12–15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
  43, 45, 47 (all `dense=True` in the tool);
* **fails for 19 phases — 1, 11, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
  40, 42, 44, 46, 48** (tool `dense=False`).  Example: P16 prints M=64 N=128 K=128
  (`4*16*4 = 256`) against 16 `mma`; the B operand of P16 has 16 fragments but only
  4 of them are used per A fragment (the C-chain is a 4-step cascade, §3/E16).

Weight-image extent: the largest byte the entry reads from `W` is the second of
phase 48's two C seeds, `W + ((laneid<<2)&12) + 316432` (s19373), i.e. the image is
**at least 316436 bytes**.  Phase 1's B block occupies `[0, 20480)` (40
`ld.weak.global.ca.v4.u32` at `+0…+19968`, s440–s518), the score bias
`[81920, 82160]` (s527–s542), and the last weight tile `[314368, 315904]`
(s19359–s19366).

**Phases that read no weight operand at all: 10 and 11.**  Phase 10's A and B both
come from e4m3 packs of phase 8's D (E9), and phase 11's B is the `movmatrix`
transpose of registers (E10); see §3/E9, E10.

## 3. Summary of epilogues

| epilogue | range | stmts | named operation |
|---|---|---|---|
| E1 | s783–s9138 | 8357 | requantise P1's D, scatter to `SM`, read back, blend the `param+24` image, inverse-std, requantise, stage for P2 |
| E2–E8 | 7 stmts each | 7 | next-B weight loads only (s9164–s9356) |
| E9 | s9381–s9551 | 171 | score-bias C-seed loads + 80 packs of P8's D → P10 A and B |
| E10 | s9600–s13442 | 3843 | softmax of the 64×96 score block, V-transpose, pack |
| E11 | s13491–s13689 | 199 | stage P·V, P12 A from `SM`, next-B loads, column table + skip → P12 C |
| E12–E14 | 23 stmts each | 23 | P13/P14/P15 A from `SM` + next-B loads (s13706–s13806) |
| E15 | s13823–s14654 | 832 | RMS norm (², sum, rsqrt, gain) + packs + next B/C |
| E16 | s14671–s14959 | 289 | clamped cubic activation + column-bias add + packs + next B |
| E17 | s14976–s15051 | 76 | 32 packs → P18 A, next-B, next-C bias |
| E18 | s15068–s15235 | 168 | activation + packs + next B (no table add) |
| E19 | s15252–s15327 | 76 | 32 packs → P20 A, next-B, next-C bias |
| E20–E46 | 168/76 alternating | | the E18/E19 pair repeated for P20…P46 |
| E47 | s19116–s19373 | 258 | pack → plane-arena scatter (`param+48`) → next B/C |
| E48 | s19382–s19662 | 281 | `SM` stage, softmax over 5 taps, 3×3 texture gather, aux textures, surface write |

---

## 4. The epilogues

### E1 — s783–s9138: patch expand, image blend, inverse standard deviation

This is 42.5 % of the entry.  It is one long pipeline; the sub-ranges below are
maximal runs of one kind of work, all verified with the lineariser.

Evidence:

```
s783   L3338  cvt.rn.satfinite.e4m3x2.f16x2 %rs248, %r924
s787   L3349  mov.b32 %r10, {%rs250, %rs251}
s956   L3705  st.shared.v2.u16 [%r3583], {%rs248, %rs249}
s974   L3725  st.shared.u32 [%r3591+1600], %r10
s1704  L4548  bar.sync 0
s1748  L4601  ld.shared.v2.u16 {%rs346, %rs347}, [%r3927]
s2780  L5874  ld.param.u64 %rd3, [%rd1+-56]
s2936  L6041  ld.global.v2.u16 {%rs410, %rs411}, [%rd215]
s7176  L10529 cvt.rn.f16x2.e4m3x2 %r7510, %rs410
s7307  L10916 add.f16 %rs600,%rs601,%rs602
s7562  L11555 mul.f16 %rs984,%rs600,%rs600
s7817  L12370 shfl.sync.bfly.b32 (first of 8)
s7874  L12518 mov.f64 %fd1, 0d3F20000000000000
s8072  L13108 rsqrt.approx.ftz.f32 fl, fl
s8586  L13941 ld.global.v2.u16 {%rs2599, %rs2600}, [%rd283+82960]
s9018  L15205 shl.b32 %r8105, %r9, 11
s9035  L15240 st.shared.v4.u32 [%r8109], {%r8113, %r8112, %r8111, %r8110}
s9099  L15355 st.shared.v4.u32 [%r8133+9728], {%r8149, %r8148, %r8147, %r8146}
s9104  L15361 bar.sync 0
s9109  L15368 ld.shared.v4.u32 {%r25650, %r25651, %r25652, %r25653}, [%r25649]
s9137  L15399 ld.weak.global.ca.v4.u32 { %r8152,%r8153,%r8154,%r8155},[%rd284]
```

| range | stmts | what | operations (count) |
|---|---|---|---|
| s783–s924 | 142 | requantise **phase 1's D** pairs to e4m3 and pack them | 96 `cvt.rn.satfinite.e4m3x2.f16x2`, 46 `mov.b32` → `%r10…%r55`, plus the loose halves `%rs248/%rs249` (s783/s784) and `%rs342/%rs343` (s923/s924) |
| s925–s1703 | 779 | guarded scatter of the packed words into `SM` | 46 `st.shared.u32`, 2 `st.shared.v2.u16` |
| s1704 | 1 | `bar.sync 0` (L4548) | — |
| s1705–s2779 | 1075 | read back a window from `SM` | 32 `ld.shared.v2.u16` (s1748…s2779) |
| s2780–s2781 | 2 | **the `param+24` buffer pointer** | `ld.param.u64 %rd3, [%rd1+-56]`, `cvta` |
| s2782–s7175 | 4394 | clamped-coordinate gather from the `param+24` buffer | 32 `ld.global.v2.u16` (s2936…s7175) |
| s7176–s7304 | 129 | dequantise both sources | 128 `cvt.rn.f16x2.e4m3x2` → `%r7510…%r7637` |
| s7305–s7560 | 256 | blend staged value + image value | 260 `add.f16` → `%rs600, %rs603, %rs606 …` |
| s7561 | 1 | `bar.sync 0` (L11553) | — |
| s7562–s8067 | 506 | squares + in-lane sums | 384 `mul.f16`, 68 `add.f16x2`, 8 `shfl.sync.bfly.b32` (s7817…s7870) |
| s8068–s8580 | 513 | inverse standard deviation | 128 `rsqrt.approx.ftz.f32` (s8072…s8577) with 128 `cvt.f32.f16` / 128 `cvt.rn.f16.f32` |
| s8581–s8601 | 21 | per-column gain read | 16 `ld.global.v2.u16` at `W + ((laneid<<2)&12) + {82944…83184}` |
| s8602–s9034 | 433 | normalise and combine | `mul.f16` (e.g. s8795 `mul.f16 %rs2149,%rs600,%rs1765`) |
| s9035–s9099 | 65 | requantise and stage for phase 2 | 8 `st.shared.v4.u32` |
| s9104 | 1 | `bar.sync 0` (L15361) | — |
| s9106–s9133 | 28 | phase 2's A fragments | 24 `ld.shared.v4.u32` → `%r25650…%r25745` |
| s9134–s9138 | 5 | phase 2's B | 1 `ld.weak.global.ca.v4.u32` at `W + z*24576 + 83200 + 16*laneid` |

**The scatter (s925–s1703).**  The 48 stores are guarded by
`setp.gt.u32 %p, row, 9` / `…, col, 9` (s945, s947, s963, s965 …) and the store
index is `40*row + 4*col + t` bytes into a plane whose base is `{0, 1600, 3200,
4800, 6400, 8000, 9600, 11200}` — **8 planes of 1600 bytes = 12800 bytes, exactly
the whole shared array**.  The plane offsets appear literally at s974 (`+1600`),
s1032 (`+3200`), s1050 (`+4800`), s1096 (`+6400`), s1114 (`+8000`), s1160
(`+9600`), s1178 (`+11200`).  `row` and `col` are not the natural coordinates of
`laneid`: they come out of a golden-ratio multiply, e.g. s929–s936
`mul.wide.u32 %rd195, %r3340, -1431655765` / `shr.u64 %rd196, %rd195, 35` /
`mul.wide.u32 %rd197, %r56, -1431655765` / `shr.u64 %rd198, %rd197, 34`, i.e. the
emitted PTX walks the 10×10 grid in a permuted lane order.

**The `param+24` gather (s2782–s7175).**  `%rd1` is `param_0+80` (s4), so `%rd1-56`
is `param_0+24`; `%rd4 = cvta.to.global.u64(%rd3)` (s2781).  Each of the 32 loads
computes a **clamped halo coordinate** from the block origin and `ex`/`ey`: the
pattern `add.s32 %r4243, %r2, %r403` / `abs.s32 %r4244, %r4243` /
`shl.b32 %r4245, %r3348, 1` / `sub.s32 %r4246, %r4245, %r4244` /
`add.s32 %r4247, %r4246, -2` / `min.s32 %r4248, %r4244, %r4247` /
`min.u32 %r4250, %r4249, %r4248` (s2793–s2800) mirrors the prologue's `A`-gather
(s42–s57) with a different row stride.  The interior branch builds the index with
`shl.b32 %r4373, %r4372, 3` (s2956) then
`mad.lo.s32 %r4374, %r4364, %r7, %r4373` (s2957), where `%r7 = ex<<3` (s15), so the
buffer row stride is **8 u32 = 8 e4m3-words per image row**; one of the boundary
branches instead uses `shl.b32 %r4189, %r4188, 3` (s2906) with a literal 8.
The load itself is `mul.wide.u32 %rd214, %r26257, 4` /
`add.s64 %rd215, %rd4, %rd214` / `ld.global.v2.u16 {%rs410, %rs411}, [%rd215]`
(s2934–s2936), i.e. **one u32 (two f16) per image column position**, four columns
per lane group selected by `min.u32 %r4194, %r4193, 7` (s2911).

**The blend (s7305–s7560).**  The staged values coming out of `SM` are combined
with the buffer values: `mov.b32 {%rs601, %rs604}, %r7510` /
`mov.b32 {%rs602, %rs605}, %r7574` / `add.f16 %rs600,%rs601,%rs602` /
`add.f16 %rs603,%rs604,%rs605` (s7305–s7308).  The results `%rs600…%rs957`
(**32 f16 pairs per lane**) are the layer's **skip/residual term**; they are read
much later, at s13595 (`add.f16 %rs4255,%rs6116,%rs600`) inside E11 and at s8795
(`mul.f16 %rs2149,%rs600,%rs1765`).

**The inverse-std block (s7561–s8580).**  384 `mul.f16` square the residual
(`mul.f16 %rs984,%rs600,%rs600`, s7563), the `add.f16x2`/`add.f16` tree folds
them, 8 `shfl.sync.bfly.b32` (s7817…s7870) sum across the four lanes of a row
segment, the constant `2⁻¹³` is added (s7874–s7875) and 128
`rsqrt.approx.ftz.f32` (s8072…s8577) invert.  The gain table is read at
`W + ((laneid<<2)&12) + {82944, 82960, … 83184}` (s8586–s8601, 16 loads, 16-byte
stride, last at 82944).  So E1 computes, per output element,

```
acc  = staged_SM_value + param24_buffer_value
inv  = rsqrt.approx(f16(sum_lane(acc^2) + 2^-13))
out  = (acc * gain[col]) * inv        // the mul.f16 at s8795
```

**What is staged for phase 2 (s9035–s9133).**  `%r8105 = %r9<<11` (s9018) and
`%r8107 = SM + %r8105` (s9019); the 8 `st.shared.v4.u32` write at
`SM + z*2048 + 16*laneid + {0,512,1024,1536}` for every `z` and at
`+{8192,8704,9216,9728}` only when `%r186 = z<<4` is `<= 31` (s9075–s9076), i.e.
only `z ∈ {0,1}`.  After `bar.sync` (s9104) the 24 `ld.shared.v4.u32` at
`SM + 16*laneid + 512*i` (s9109–s9133) become **phase 2's A** `%r25650…%r25745`,
and the final `ld.weak.global.ca.v4.u32` (s9137) is phase 2's B.

### E2–E8 — s9164…s9356: next-B weight loads only

Each of these is exactly 7 statements: one `mov.u32 %laneid`-style setup, two
`add.s64` and two `ld.weak.global.ca.v4.u32` (4 fragments = 16 B registers), plus
the label.  Example, E7 (s9318–s9324):

```
s9320  ld.weak.global.ca.v4.u32 { … }, [%rd, 177408+…]×  // two 512-byte-strided loads
```

The B blocks, read verbatim off the phase table (the tool's `weight bytes` column
minus its usual +12 on the upper end), are `83200/83712` (P2), `87296/87808` (P3),
`84224/84736` (P4), `88320/88832` (P5), `85248/85760` (P6), `89344/89856` (P7),
`86272/86784` (P8), `90368/90880` (P9) — each read as `W + z*24576 + 16*laneid` plus
`{0, 512}`.  Note the interleave: the even phases walk `83200, 84224, 85248, 86272`
and the odd phases `87296, 88320, 89344, 90368`, so the file's phase order is
(k0n0, k0n1, k1n0, k1n1, …) with the two n-groups stored contiguously.

### E9 — s9381–s9551: score-bias seeds and 80 packs of phase 8's D

171 statements, 24 weight loads + 80 packs.

```
s9381  L16841 mov.u32 %r10143, %laneid
s9384  L16845 add.s64 %rd300, %rd631, 8192
s9385  L16847 ld.weak.global.ca.v4.u32 { %r10144,%r10145,%r10146,%r10147},[%rd300]
s9432  L16943 cvt.rn.satfinite.e4m3x2.f16x2 …          // first of 80 packs
s9502  L17126 …                                          // last pack group
```

* **24 loads** at `W + z*24576 + 16*laneid + 8192 + 512*i`, `i = 0…23`
  (s9385…s9500) → **phase 10's C seeds** `%r10144…` (96 registers = 24 C pairs).
  The byte block is `[8192, 20480)` inside the phase-2…9 weight group.  Note this
  is a different *kind* of C operand from phase 1's: phase 1 seeds each mma with a
  single u32 duplicated into both halves (`{%r2453, %r2453}`, s543), loaded as 16
  `ld.global.u32` at `W + z*256 + ((laneid<<2)&12) + {81920…82160}` (s527–s542),
  whereas phase 10 takes 24 full 512-byte tiles.
* **80 packs** whose sources are **phase 8's D** (`%r9654…%r9885`), not phase 9's:
  `%r10686 = (pack(r9654), pack(r9674))` (s9434), `%r10696 = (pack(r9655),
  pack(r9675))` (s9437), …  They split into phase 10's A fragments
  (`%r10432…%r10795`, 16 registers) and B fragments (`%r10686…%r10797`, 24
  registers = 12 fragments).  Phase 9's own D (`%r9903…%r10134`) is not read here;
  it reaches E10's `movmatrix` block instead (see E10).

**The C-seed layout.**  The four C registers of one mma are the four u32 of one
16-byte lane slot, and

```
C(m, n) = load_tile(4*(n>>1) + m) + 4*(n&1)      [register index within the block]
byte    = base + 512*(4*(n>>1) + m) + 16*laneid + 8*(n&1)
```

so the bias tile index runs `4*(n>>1) + m` (512-byte stride) over 4 m-tiles and 6
`(n>>1)` groups = 24 tiles = 12288 bytes.  Verified against phase 10's operands:
s9552 `C={%r10144,%r10145}`, s9553 `C={%r10152,%r10153}`, …, and the first block's
`%r10144` is the load at `+8192` (s9385) while `%r10152` is the load at `+8704`
(s9387).

### E10 — s9600–s13442: softmax of the 64×96 score block, V-transpose, pack

**Phase 10** = 48 `mma` of 64×96 with C = the 24 bias pairs (E9).  The epilogue has
**no memory access at all** (0 `ld.*`, 0 `st.*`) and no `bar.sync`.

```
s9601  L17560 mov.f32 %f476, 0f3C8CCB50
s9604  L17568 mul.f16x2 %r10802,%r10320,%r10801
s9609  L17582 max.f16x2 %r10805,%r10802,%r10807
s9614  L17596 min.f16x2 %r10808,%r10805,%r10810
s9620  L17614 fma.rn.f16x2 %r10813,%r10808,%r10811,%r10816
s9625  L17628 fma.rn.f16x2 %r10817,%r10808,%r10813,%r10820
s9626  L17631 shl.b32 %r25746, %r10817, 5
s9627  L17632 and.b32 %r12722, %r25746, 2145419232      // 0x7FE07FE0
s11804 L24428 shfl.sync.bfly.b32 (first of 16)
s11813 L24452 add.f16 (first of 8)
s11934 L24754 rcp.approx.ftz.f32 (first of 192)
s12701 L25999 mul.f16 (first of 192)
s13179 L27053 movmatrix.sync.trans.aligned.m8n8.b16 %r13326, %r9903 (first of 48)
s13227 L27197 cvt.rn.satfinite.e4m3x2.f16x2 (first of 144 packs)
```

Sub-sections:

| range | what |
|---|---|
| s9600–s11717 | 96 per-element groups: scale, clamp, cubic, exponent extract |
| s11718–s11801 | in-lane partial-sum trees (104 `add.f16x2`) |
| s11802–s11821 | 16 `shfl.sync.bfly.b32` → 8 row sums |
| s11822–s12700 | 192 `rcp.approx.ftz.f32` (s11934…) → 96 reciprocals |
| s12701–s13178 | 192 `mul.f16` → 96 probabilities |
| s13179–s13226 | 48 `movmatrix` transposing **phase 9's D** |
| s13227–s13442 | 144 `cvt.rn.satfinite.e4m3x2.f16x2` packs |

**(a) per-element exponent.**  For each of the 96 f16x2 D registers, per half (an
f16 `x`):

```
m      = clamp(f16(x * 0.017181396484375), -0.55615234375, +0.55615234375)  // s9604,s9609,s9614
t      = f16(m*(-m) + 0.92724609375)                                        // s9620
u      = f16(m*t + 1.375) = f16(1.375 + m*(0.92724609375 - m²))             // s9625
expval = f16 from bits ((bits(u)<<5) & 0x7FE07FE0)   // 2^(32·frac(u) − 15)
```

This is **not `exp` and not `exp2(x·log2e)`**: the exponent is the cubic
`32*(0.375 + 0.92724609375·m − m³)` quantised to a 6-bit significand.  The clamp
bounds are the guard: `d/dm[0.92724609375·m − m³]` vanishes at
`m = √(0.92724609375/3) = 0.55615…`, exactly the clamp, so the exponent saturates
at the top.

**(b)/(c) row sums.**  In-lane `add.f16x2` trees, then eight times: add the two
partials, butterfly over lane bits 0 and 1, `add.f16` the halves, broadcast.  The
shuffles use the operands `%r11301 = 1` and `%r11310 = 2`.

**(d)/(e) probabilities.**  `rcp.approx.ftz.f32` (each as
`cvt.f32.f16 → rcp → cvt.rn.f16.f32`) on the 8 row sums, then
`p = expval * (1/rowsum)` with 192 `mul.f16`, so `Σ_j p_ij = 1` up to the
`rcp.approx` error.

**(f) V-transpose.**  48 `movmatrix.sync.trans.aligned.m8n8.b16` on **phase 9's**
48 D registers.  (The first is `s13179 L27053`.)

**(g) packs.**  144 packs → phase 11's 12 A fragments (the probabilities) and 12 B
fragments (the transposed V).  E11's A operand list (`%r13638…%r14041`, 48
registers) and B operand list (`%r13932…%r14043`, 24 registers) confirm both.

**There is no row maximum anywhere in this epilogue** — `max.f16x2` (96×) and
`min.f16x2` (96×) are the clamp of (a), and the only shuffles are the 16
`shfl.sync.bfly` above.  Contrast E48, which does have a real row max.

### E11 — s13491–s13689: stage P·V, seed phase 12

199 statements.

```
s13491 L28036 bar.sync 0
s13495 L28043 cvt.rn.satfinite.e4m3x2.f16x2 …            // first of 32 packs
s13507 L28070 st.shared.v4.u32 [%r25875], {%r25879, %r25878, %r25877, %r25876}
s13547 L28158 bar.sync 0
s13548 L28159 shl.b32 %r25892, %r9, 9                    // z*512
s13553 L28166 ld.shared.v4.u32 {%r25896, %r25897, %r25898, %r25899}, [%r25895]
s13558 L28174 ld.weak.global.ca.v4.u32 { %r14081,%r14082,%r14083,%r14084},[%rd324]
s13578 L28211 ld.global.v2.u16 {%rs6086, %rs6087}, [%rd637+181520]
s13594 L28228 add.f16 %rs4258,%rs6117,%rs603
s13595 L28232 add.f16 %rs4255,%rs6116,%rs600
```

* s13491 `bar.sync`; then 32 `cvt.rn.satfinite.e4m3x2.f16x2` packing phase 11's D
  (`%r13606, %r13666, %r13646, %r13846, …`) and 4 (or 5, when the extra store group
  runs) `st.shared.v4.u32` into `SM + z*2048 + 16*laneid` at `+{0,512,1024,1536}`
  (s13507, s13520, s13533, s13546);
* s13547 `bar.sync`; s13548–s13549 `%r25893 = SM + z*512`;
* s13553 `ld.shared.v4.u32` at `SM + z*512 + 16*laneid` → **phase 12's A fragment
  #0** (`%r25896…%r25899`);
* s13558–s13572, **8 weight loads** at `W + 16*laneid + {103680, 104192, 105216,
  105728, 106240, 106752, 107264}`… → phase 12's B (16 fragments = 32 registers);
* s13578–s13593, **16 table loads** at `W + ((laneid<<2)&12) + {181504, 181520, …
  181744}` (16-byte stride) → 32 f16 per lane = a per-column table;
* s13594–s13689, **64 `add.f16`** → 32 b32 `%r14122…%r14273` = **phase 12's C
  seeds**, exactly `table[col] + residual`, e.g. s13595
  `add.f16 %rs4255,%rs6116,%rs600` where `%rs600` is the **E1 residual**.

So `phase 12: D = A·B + (skip_residual + column_table)`.

### E12–E14 — s13706–s13806: A from `SM` + next-B, three times

23 statements each, one `ld.shared.v4.u32` (one A fragment: `%r25903…%r25906` in
E12, `%r25909…%r25912` in E13, `%r25915…%r25918` in E14) plus 8
`ld.weak.global.ca.v4.u32` for the next phase's B (16 fragments = 32 registers,
e.g. `%r14276…%r14307`).  The shared base is the same `SM + z*512 + 16*laneid`.

Together with E11, this supplies the four A k-fragments of phases 12–15, which are
one GEMM 16×128×128 split across four `mma` phases chained through C
(P12 C = the E11 seeds; P13 C = P12's D; P14 C = P13's D; P15 C = P14's D — the
operand sets in §2's summary table).

### E15 — s13823–s14654: RMS norm before the MLP

832 statements, no stores.

```
s13825 L29094 mov.b32 {%rs5256, %rs5259}, %r14696   // first of 302
s13826 L29096 mul.f16 %rs4450,%rs5259,%rs5259        // first of 192 squares
s13987 L29541 shfl.sync.bfly.b32 (first of 4)
s14018 L29618 cvt.rn.f16.f64 %rs4645, %fd1           // eps = 2^-13
s14118 L29915 rsqrt.approx.ftz.f32 fl, fl            // first of 64
s14371 L30325 ld.global.v2.u16 {%rs6118, %rs6119}, [%rd646+181776]   // first of 16 gain loads
s14583 L30924 ld.weak.global.ca.v4.u32 …             // first of 8 next-B loads (182016+512*i)
s14603 L30961 ld.global.u32 %r15124, [%rd651+186112] // phase 16's C bias words
s14607 L31227 cvt.rn.satfinite.e4m3x2.f16x2 …        // first of 32 packs
```

| range | stmts | op |
|---|---|---|
| s13823–s14010 | 188 | 192 `mul.f16` — squares of phase 12–15's D halves |
| s14011–s14017 | 7 | `add.f16` / `add.f16x2` in-lane pairwise sums |
| s13981–s14009 | — | 9 `mov.u32` lane-shift setup, 4 `or.b32`, 4 `shfl.sync.bfly.b32` (s13987…) |
| s14116–s14135 | — | 64 `cvt.f32.f16` → 64 `rsqrt.approx.ftz.f32` (s14118) → 64 `cvt.rn.f16.f32` |
| s14371–s14386 | 16 | gain loads `W + ((laneid<<2)&12) + {…}` (16 loads, 16-byte stride) |
| s14583–s14606 | 24 | 8 `ld.weak.global.ca.v4.u32` (phase 16's B, `182016 + 512*i`, s14583–s14597) then 4 `ld.global.u32` at `186112/186128/186144/186160` (phase 16's C bias, s14603–s14606) |
| s14607–s14653 | 47 | 32 packs → phase 16's A |

**What it computes.**  For each row a lane holds:
```
sumsq[row] = Σ_c x[row,c]²                        (f16, in-lane tree + 4-lane butterfly)
inv        = rsqrt.approx.ftz.f32(f32(f16(sumsq + 2^-13)))   // rounded to f16
out[row,c] = x[row,c] * inv * gain[c]
```
with **no mean subtraction and no `1/N`** — the quantity is the raw sum of squares
plus `2^-13`.  The eps is materialised at s14018 (`cvt.rn.f16.f64 %rs…, %fd1`, and
`%fd1 = 0d3F20000000000000` at s7874).  The gain is the **same shape of table as
E1's** (`W + ((laneid<<2)&12) + off`) and the norm's bias is not added here: it is
folded into phase 16's C operand, loaded by the 4 `ld.global.u32` at s14603.

### E16 — s14671–s14959: clamped cubic activation + column bias

289 statements.

```
s14671 L31190 mov.u32 %r15275, %laneid
s14672 L31192 mov.f32 %f989, 0f3ED306EB
s14673 L31195 cvt.rn.f16.f32 low, %f989       // first of 32 constant→f16 converts
s14684 L31221 neg.f16x2 %r15280,%r15279       // first of 8
s14685 L31225 max.f16x2 …                     // first of 8
s14686 L31229 min.f16x2 …                     // first of 8
s14687 L31233 abs.f16x2 …                     // first of 8
s14688 L31237 mul.f16x2 %r15290,%r15277,%r15288   // first of 24
s14689 L31241 sub.f16x2 …                     // first of 8
s14691 L31249 add.f16x2 …                     // first of 8
s14814 L31680 add.s64 %rd653, %rd2, %rd652    // column-table addressing
s14816 L31683 ld.weak.global.ca.v4.u32 { %r15509,%r15510,%r15511,%r15512},[%rd364]  // first of 8, 186176+512*i
s14836 L31720 ld.global.v2.u16 {%rs6150, %rs6151}, [%rd656+190288]   // first of 16 table loads
s14852 L31737 add.f16 %rs5257,%rs6181,%rs5259 // first of 64 bias adds
s14948 L32025 cvt.rn.satfinite.e4m3x2.f16x2 %rs5447, %r15418
s14950 L32030 mov.b32 %r15702, {%rs5446, %rs5447}
```

Per element, over each of the 8 f16x2 groups of phase 16's D:

```
y   = clamp(x, -2, +2)                              // neg/max/min, s14684–s14686
g   = 0.5 + y * (0.412109375 - 0.0810546875 * |y|)  // abs/mul/sub/mul/add, s14687–s14691
out = x * g                                         // mul.f16x2, s14688/s14691 chain
```

`g` is a **cubic ramp, not a sigmoid**: `g(+2) = 1` ⇒ `out = x` for `x ≥ 2`;
`g(-2) = 0` ⇒ `out = 0` for `x ≤ -2`; `g(1) = 0.831055` where `x·sigmoid(x) =
0.730957`.  **Reproduce the formula, not SiLU/GELU.**

Then (s14836–s14851) the 16 table loads
`ld.global.v2.u16 … [%rd656+190288]` (s14836, the first, at a 16-byte stride, so 32
f16 per lane = a per-column table, using the same
`W + ((laneid<<2)&12) + off` pattern as E1/E11/E15) and 64 `add.f16`
(s14852–s14939) write the table over the activation **inputs**:
`s14940 add.f16 %rs5428,%rs6176,%rs5430` / `s14941 mov.b32 %r15699, {%rs5428,
%rs5431}` → `%r15699`, `%r15708`, `%r15709`, … are **phase 17's C seeds**.  The
final 8 packs (s14948–s14959) build **phase 17's A**: `%r15702 = (pack(r15302),
pack(r15418))` (s14950), `%r15704 = (pack(r15331), pack(r15447))` (s14953),
`%r15703 = (pack(r15360), pack(r15476))` (s14956), `%r15705 = (pack(r15389),
pack(r15505))` (s14959).  The 8 `ld.weak.global.ca.v4.u32` from s14816 are phase
17's B block `[186176, 190284)`: s14815 `add.s64 %rd364, %rd653, 186176`, then
`+512` per load up to `+189760` at s14829.

### E17 — s14976–s15051: packs, then next B/C

76 statements, identical form to E12–E14 plus packs:

```
s14980 L32171 ld.weak.global.ca.v4.u32 { %r15711,%r15712,%r15713,%r15714},[%rd37…]  // first of 8
s15000 L32208 ld.global.u32 %r15785, [%rd661+194624]   // first of 4 next-C bias words
s15004 L32213 cvt.rn.satfinite.e4m3x2.f16x2 %rs5455, %r24985   // first of 32 packs
```

32 packs (16 b32) → **phase 18's A**; 8 `ld.weak.global.ca.v4.u32` → phase 18's B
block `[190528, 194636)`; 4 `ld.global.u32` at `W + ((laneid<<2)&12) +
{194624,194640,194656,194672}` → phase 18's C bias (s15000 is the first, at
`+194624`).

### E18 — s15068–s15235: activation, no column bias

168 statements, the E16 formula without the table add:

```
s15068 L32437 mov.u32 %r15936, %laneid
s15069 L32441 cvt.rn.f16.f32 low, %f989      // 32 constant→f16, reusing %f989…%f992
s15077 L32464 neg.f16x2 %r15941,%r15940
s15078 L32468 max.f16x2 %r15943,%r15836,%r15941
s15079 L32472 min.f16x2 %r15946,%r15943,%r15940
s15080 L32476 abs.f16x2 %r15949,%r15946
s15081 L32480 mul.f16x2 %r15951,%r15938,%r15949   // first of 24
s15209 L32926 ld.weak.global.ca.v4.u32 { %r16170,%r16171,%r16172,%r16173},[%rd38…]  // 8 next-B loads
s15224 L32957 cvt.rn.satfinite.e4m3x2.f16x2 %rs5487, %r16079   // 8 packs → 1 A fragment
```

There is **no `ld.global.v2.u16`** in this epilogue, i.e. no per-column table add:
the residual for this branch is carried entirely by the C operand that E17 loaded.

### E19 — s15252–s15327: packs, then next B/C

76 statements, byte-identical in shape to E17:

```
s15256 L33103 ld.weak.global.ca.v4.u32 { %r16371,%r16372,%r16373,%r16374},[%rd388]  // first of 8
s15276 L33140 ld.global.u32 %r16445, [%rd668+202880]   // first of 4 next-C bias words
s15280 L33145 cvt.rn.satfinite.e4m3x2.f16x2 %rs5495, %r24985   // first of 32 packs
```

The 8 weight loads are `W + z*24576 + 16*laneid + 198784 + 512*i`, i.e. **phase
20's B block** `[198784, 202892)`; the 4 `ld.global.u32` at
`202880/202896/202912/202928` are phase 20's C bias (`202880..202928`).

**E20 … E46.**  The pattern `(activation, no table) → (32 packs + next B/C)`
repeats with a period of two phases.  Numbering the phases, not the epilogues: the
**even** phases 16, 18, …, 46 each have a 289/168/…-statement **activation**
epilogue; the **odd** phases 17, 19, …, 45 each have a 76-statement **pack**
epilogue.  The occurrence of each is fixed by the classification the tool prints
(`… / after phase N … e4m3-pack:32@sNNNN` for odd N, `… normalise/pack` for even N).

The per-phase B blocks advance by 4096/4160 alternately, i.e. **8256 bytes per
two-phase pair**:

```
P16 [182016,186124)   P17 [186176,190284)   P18 [190528,194636)   P19 [194688,198796)
P20 [198784,202892)   P21 [202944,207052)   P22 [207040,211148)   P23 [211200,215308)
P24 [215296,219404)   P25 [219456,223564)   P26 [223552,227660)   P27 [227712,231820)
…                                              P47 [310272,314380)   P48 [314368,316428)
```

Each epilogue loads the block of the phase that *follows* it, and that phase's C
bias words live in the 64 bytes immediately above its B block (e.g. phase 18's C is
at `194624..194672`, loaded at s15000).  Only **E16** has the column-table add
(`ld.global.v2.u16`); every other activation epilogue (E18, E20, …, E46) has none.

#### Shape of the whole tail (E11–E48)

```
E11  stage P·V → SM;  A12 = ld.shared(SM);  C12 = E1_skip + table@181504
P12..P15  one GEMM 16×128×128 in four k-chained phases  → D15
E15  RMS norm on D15 (eps 2^-13, gain@…), no mean
     ├─ pack → P16 A (ld.shared)       P16: C = norm-bias@(4 ld.global.u32)  → activation (E16) + table → P17 C
     │                                                                        pack → P17 A
     ├─ P17: 16×128, C = E16 seeds, packs → P18 A (E17)
     ├─ P18: activation on its own D (E18), packs → P19 A
     └─ … 14 more (activation, pack) pairs, each with its own weight block
E47  pack → plane arena (`param+48`)
E48  SM stage, softmax over 5 taps, 3×3 texture gather, aux textures, surface write
```

### E47 — s19116–s19373: pack → plane-arena scatter → next B/C

258 statements.

```
s19116 L46145 cvt.rn.satfinite.e4m3x2.f16x2 %rs6055, %r25470    // first of 32 packs
s19164 L46256 ld.param.u64 %rd763, [%rd8+48]                    // S = plane arena
s19165 L46257 cvta.to.global.u64 %rd5, %rd763
s19167 L46261 shr.u32 %r639, %r25642, 2                         // g = laneid>>2
s19168 L46262 and.b32 %r640, %r25642, 3                         // t = laneid&3
s19170 L46264 and.b32 %r25951, %r639, 7
s19171 L46265 add.s32 %r25952, %r622, %r25950                   // y
s19183 L46277 mad.lo.s32 %r641, %r7, %r25952, %r25959           // idx
s19190 L46285 st.global.u32 [%rd765], %r623                     // first of 16 stores
s19359 L46487 add.s64 %rd796, %rd801, 314368
s19360 L46489 ld.weak.global.ca.v4.u32 { %r25994,%r25995,%r25996,%r25997},[%rd796]   // next B
s19372 L46510 ld.global.u32 %r26020, [%rd805+316416]            // next C seed
```

* 32 packs → 16 b32 `%r623…%r638`, which are simultaneously **phase 48's A
  fragments** (see E48) — the same 16 registers are scattered to memory and fed to
  the last GEMM.
* `S = cvta.to.global.u64(param_0+48)` (s19164–s19165) — the entry's only read of
  `param_0+48`.
* **16 `st.global.u32`** (s19190…s19348), each guarded by
  `setp.lt.s32 %p, y, 0` / `setp.ge.u32 %p, y, ey` / `setp.gt.u32 %p, laneid, 63` /
  `setp.lt.s32 %p, x, 0` / `setp.ge.u32 %p, x, ex` combined with `or.pred`
  (s19172–s19181) and a `selp.b32 %r642, -1, %r25960, %p1` + `setp.lt.s32 … -1` +
  `bra` (s19185–s19187) that turns a guard failure into "skip the store".

  The store index is `idx = (8*ex)*y + 8*x + d` with `%r7 = ex<<3` (s15),
  `y = %r622 + (laneid>>5)`, `x = %r1 + ((laneid>>2)&7)`, `d = t` (s19183–s19184)
  — the same formula as enc0's E20 (§4 of the enc0 file).
* after the stores: 4 `ld.weak.global.ca.v4.u32` at
  `W + 16*laneid + {314368, 314880, 315392, 315904}` (s19360–s19366) = phase 48's B
  (8 fragments = 16 registers), and 2 `ld.global.u32` at
  `W + ((laneid<<2)&12) + {316416, 316432}` (s19372–s19373) = phase 48's C seeds.

### E48 — s19382–s19662: softmax over five taps, 3×3 texture gather, surface write

281 statements; the entry's last statement is `s19663 L47124 ret`.

```
s19382 L46568 bar.sync 0
s19387 L46575 add.s32 %r26096, %r670, %r26211
s19393 L46581 st.shared.u32 [%r672], %r26071
s19397 L46586 st.shared.u32 [%r672+16], %r26081
s19414 L46606 mov.b64 %rd835, param_0
s19432 L46624 ld.param.v2.u32 {%r26129, %r26130}, [%rd835]      // ex, ey
s19438 L46631 mov.b64 %rd836, param_0
s19449 L46642 ld.param.u64 %rd830, [%rd836+112]                 // output surface
s19452 L46646 max.f16x2 %r26131,%r26132,%r26133                 // first of 4
s19456 L46659 max.f16 %rs6182,%rs6183,%rs6184
s19458 L46666 mov.b32 %r26154, {%rs6185, %rs6185}               // broadcast row max
s19462 L46676 mul.ftz.f32 %f1033, %f993, 0f3FB8AA3B             // ×log2e
s19463 L46677 ex2.approx.ftz.f32 %f995, %f1033
s19519 L46796 mov.f32 %f1032, 0f3F800000
s19520 L46797 div.approx.ftz.f32 %f1011, %f1032, %f1050         // 1/rowsum
s19544 L46845 ld.param.v2.f32 {%f1052, %f1053}, [%rd836+136]
s19546 L46847 fma.rn.ftz.f32 %f1012, %f1052, %f1051, %f1056     // sx*(i+0.5)
s19550 L46851 ld.param.u64 %rd823, [%rd836+88]                  // low-res texture
s19552 L46854 tex.base.2d.v4.f16.f32 {%rs6210,%rs6211,%rs6212,%rs6213}, [%rd807, {%f1012,%f1013}]
s19626 L47026 ld.param.u64 %rd825, [%rd836+96]
s19627 L47028 tex.base.2d.v4.f16.s32 {%rs6354,%rs6355,%rs6356,%rs6357}, [%rd825, {%r677,%r678}]
s19628 L47030 ld.param.u64 %rd827, [%rd836+104]
s19629 L47032 tex.base.2d.v4.f16.s32 {%rs6358,%rs6359,%rs6360,%rs6361}, [%rd827, {%r677,%r678}]
s19644 L47079 set.nan.f16.f16 %rs6389,%rs6358,%rs6358
s19647 L47086 add.f16 %rs6411,%rs6411,%rs6358                   // only if not NaN
s19659 L47117 shl.b32 %r26200, %r677, 3                         // 8*x
s19661 L47120 sust.b.2d.v4.b16.zero [%rd830, {%r26200,%r678}], {%rs6411,%rs6412,%rs6413,%rs6410}
```

Sub-sections:

| range | stmts | what |
|---|---|---|
| s19382–s19413 | 32 | `bar.sync` + stage phase 48's D into `SM` at `(48*y + 6*x)` slots |
| s19414–s19437 | 24 | `param_0` reload, `(ex, ey)` read, bounds guard (`tid.z > 1` or x/y out of range → skip to `$L__BB1_646`) |
| s19438–s19448 | 11 | surface pointer `param+112`, 5 `ld.shared.u32` reading **5 consecutive taps** |
| s19449–s19530 | 82 | row max over the 5 taps, `2^((v-max)·log2e)` softmax, `div.approx` reciprocal |
| s19531–s19543 | 13 | clamped coordinates `clamp(p-1, 1, ex-2)` / `clamp(p-1, 1, ey-2)` |
| s19544–s19549 | 6 | `param+136` scale pair → normalized `u = sx·(i+0.5)`, `v = sy·(j+0.5)` |
| s19550–s19625 | 76 | 9 `tex.base.2d.v4.f16.f32` at the 3×3 stencil of `param+88`, accumulated with `fma.rn.f16` (30 in total) |
| s19626–s19629 | 4 | `param+96` and `param+104` textures at the integer coordinate `(x, y)` |
| s19630–s19643 | 14 | `sigmoid`-like `div.approx` blend: `out = acc·σ + tex·(1-σ)` (e.g. s19638 `mul.f16 %rs6368,%rs6342,%rs6363`, s19639 `fma.rn.f16 %rs6411,%rs6354,%rs6365,%rs6368`) |
| s19644–s19658 | 15 | per-channel NaN guard on the `param+104` texture: `set.nan.f16.f16` → `setp.ne.s16` → `@p bra` → `add.f16` |
| s19659–s19661 | 3 | `8*x` and the `sust.b.2d.v4.b16.zero` surface write |

**Where the +96/+104 reads go.**  Both are `tex.base.2d.v4.f16.s32` reads at the
same integer coordinate `{%r677, %r678}` = `(x, y)` of the block:
`param+96`'s three channels `%rs6354/%rs6355/%rs6356` are added unconditionally
(s19639–s19643), while `param+104`'s three `%rs6358/%rs6359/%rs6360` are added
**per channel only when the channel is not NaN** (s19644–s19658).  The surface
written is `param+112` (s19449), and the write lands at `(8·x, y)` (s19659–s19661),
tap `%rs6410 = 0` in the fourth channel.

**What the +24 skip input supplies.**  `param_0+24` (reached as `%rd1-56`, s2780)
supplies the f16 image values that E1 adds to the shared-staged features
(s7307 onward) to form the residual `%rs600…%rs957`.  That residual is (i) the
`mul.f16` input at s8795 in E1's normalisation, and (ii) added to the per-column
table at `W+181504` in E11 (s13594–s13689) to seed phase 12's C operand.

## 5. The plane-arena writes (`param_0+48`)

`S = cvta.to.global.u64(%rd763)` with `%rd763 = ld.param.u64 [%rd8+48]`
(s19164–s19165) is the **only** occurrence of `param_0+48` in the entry.  There are
exactly two store groups in the whole entry:

**(i) E47, 16 × `st.global.u32 [%rd5 + 4*idx], v`.**  Statement and value of each
store, in file order:

| # | stmt | line | value | # | stmt | line | value |
|---|---|---|---|---|---|---|---|
| 1 | s19190 | L46285 | `%r623` | 9 | s19286 | L46397 | `%r631` |
| 2 | s19199 | L46296 | `%r624` | 10 | s19294 | L46407 | `%r632` |
| 3 | s19232 | L46331 | `%r625` | 11 | s19304 | L46419 | `%r633` |
| 4 | s19240 | L46341 | `%r626` | 12 | s19312 | L46429 | `%r634` |
| 5 | s19250 | L46353 | `%r627` | 13 | s19322 | L46441 | `%r635` |
| 6 | s19258 | L46363 | `%r628` | 14 | s19330 | L46451 | `%r636` |
| 7 | s19268 | L46375 | `%r629` | 15 | s19340 | L46463 | `%r637` |
| 8 | s19276 | L46385 | `%r630` | 16 | s19348 | L46473 | `%r638` |

```
idx = (8*ex)*y + 8*x + d,  guard  y >= 0, y < ey, laneid <= 63, x >= 0, x < ex
y = %r622 + (laneid>>5),  x = %r1 + ((laneid>>2)&7),  d = t  (odd stores) | d = t|4 (even stores)
where t = laneid&3, ex = %r3347, ey = %r3348, %r1 = block origin x (s7),
      %r622 = %r2 + 2*%r9  (s13823–s13824: `shl.b32 %r25919, %r9, 1` then
      `add.s32 %r622, %r2, %r25919`), i.e. the y origin also carries the slab index.
```

(The 16 indices are all built from the single `%r641 = %r7*%r25952 + %r25959`
computed at s19183, with `%r25959 = x<<3` and the second of each pair differing by
`or.b32 %r643, %r640, 4`, s19192.)  The stores write 4 bytes × 32 lanes × 16 =
2048 bytes per block of phase 47's requantised output.

**(ii) E48, the shared staging** at s19393/s19397/s19410/s19412: `st.shared.u32`
into `SM`, not the arena.

**What is written:** phase 47's requantised 16×128 output.  **What reads it back:**
nothing inside this kernel — `param_0+48` is never the base of an `ld.*` in the
entry, and phase 48's A comes from registers (`%r623…%r638`, E47's own packs).
See §6.

## 6. Uncharacterised

**U1. The shared-memory layout of E1's first scatter.**  The 48 stores write
`40*row + 4*col + t` bytes into one of 8 planes of 1600 bytes (offsets literal at
s974, s1032, s1050, s1096, s1114, s1160, s1178), with `row`/`col` produced by a
golden-ratio permutation of `laneid` and `%r9` (s929–s936).  I did not reduce that
permutation to a closed form in `(laneid, %r9)`, so I cannot say which element of
the 96×128 tile each plane holds.  *Missing:* either the layout of the producer's
`mma` D fragments in the destination tile, or a run with a shared-memory dump.

**U2. The 12 `%r8xxx` gain/normalisation wiring inside E1.**  E1 has 384 `mul.f16`
and 260 `add.f16` spread over s7305–s9034; I traced the first of each chain
(s7307, s7563, s8795) but not all 644.  *Missing:* a dataflow dump of the f16
halves.

**U3. The identity of the per-column tables.**  `W+181504` (E11),
`W+39296/39312/39328/39344` (E16), `W+82944…83184` (E1's gain),
`W+…` (E15's gain) are all read with the same `W + ((laneid<<2)&12) + off` pattern.
Their *role* is computable (E11: added to the skip before phase 12; E16: added
after the activation; E1/E15: multiply the inverse std) but their *names* are not.
The same is true of the GEMM biases at `81920…82160` (P1), `8192…20480` (P10),
`103680…107264`-adjacent words (P12), `316416/316432` (P48).  *Missing:* the weight
image's layout description (the `_prep` permutation) or the model's parameter names.

**U4. The identity of the 12288-byte score bias.**  Phase 10 adds a 12288-byte f16
table (`W + z*24576 + 8192 … +20480`) as its GEMM C operand, laid out as §4/E9
describes.  Whether it is a relative position bias, a learned score bias, or a
per-lane constant cannot be decided from the PTX.

**U5. The geometry of the banded phases.**  Nineteen phases print an M/N/K that
fails `M/16·N/8·K/32 == mma` (P1, 11, 16–46 even, 48).  For P16/P18/…/P46 I traced
the cascade explicitly (D_m = A_m·B_m + D_{m-1} over the 4 A fragments, so the
*effective* per-phase shape is a 4-step prefix sum, not 64×128×128), and for P48
likewise (4 A fragments, 8 mma).  For **P1** (5-step cascades, 24 groups of 10) and
**P11** (48 mma, C set of 65 registers) I did not reduce the full (M,N,K) reading.
The epilogue contracts above do not depend on resolving this.

**U6. The logical meaning of E48's blend.**  E48 computes
`out = acc·σ(t) + tex96·(1-σ(t))`, plus `tex104` per channel under a NaN guard, and
writes the low-res feature `acc` obtained by a 3×3 `tex.base.2d.v4.f16.f32` gather
weighted by a 5-tap softmax.  The PTX gives the arithmetic exactly but not which
model term this is.  *Missing:* the high-level source or a reference implementation.

**U7. `%r9`'s range.**  dec3 has no guard on `%r9 = %tid.z` (s433), unlike dec4
(`setp.gt.u32 %p3, %r9, 3`).  Since `SM` is 12800 bytes and the E1 store bases step
by 2048, the usable `%r9` range is bounded by `z*2048 + 2032 < 12800`, i.e.
`z <= 5`, but nothing in the PTX states it.  *Missing:* the launch configuration.

## 7. Quick acceptance index

| requirement | where |
|---|---|
| patch expand | §4/E1 (s783–s9138), and phase 1 itself (§2) |
| the norms | E1's rsqrt block (s7562–s8580, eps s7874) and E15 (s13823–s14654, eps s14018) |
| the attention | E9 (score seeds + packs), E10 (softmax + movmatrix + packs), E11 (P·V staging) |
| the activations | E16, E18, …, E46 — the clamped cubic `x·(0.5 + y·(0.412109375 − 0.0810546875·|y|))`, `y = clamp(x,±2)`; constants s14672–s14681 |
| the per-launch table in the plane arena | §5, s19164/s19190–s19348 (`param_0+48`) |
| every epilogue of every phase | §3 summary table + §4 (E1–E48) |
| texture reads at +96/+104 and where the surface write goes | §4/E48, s19626–s19629 and s19449/s19661 |
| what the skip input at +24 supplies | §1.1 (s4, s2780) and §4/E1, E11 |
| phase table M/N/K arithmetic checked | §2 (19 phases fail; listed) |
| weight-image extent | §2 (≥ 316436 bytes; last read s19373) |
| phases with no weight operand | §2: 10 and 11 |
| uncharacterised | §6 |
