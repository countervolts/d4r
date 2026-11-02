"""Machine-readable descriptor for `cuda_dldn_engine_swin_dec0_kernel`.

The WMMA engine's per-layer input.  Every number here is read out of
`~/.cache/d4r-rr-corpus/0023-PREPASS_ENTRYPOINT_NAME.ptx` (entry lines 1018-47791)
with `kernels/rr/rr_layer_spec.py`'s parser, or measured on the GPU; the source of
each family of numbers is named in the comments and anything not established that
way carries an `"unverified"` key.

Conventions used throughout:

* `m_tiles` counts 16-row mma tiles, `n_tiles` counts 8-column n tiles, `k` is the
  contraction depth of the phase.  `mma` is the shape NVIDIA emits,
  `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`.
* The engine's unit is `k32_e4m3` (one `wmma_f32_16x16x16_fp8_fp8` pair), so a
  phase of depth K is `k // 16` WMMA steps per output tile, two per k32 slice.
* The weight image is arg040, 47376 bytes.  A *tile* is 512 bytes: 32 lanes x 16
  bytes, holding two n-tiles' B fragments.  For a phase with `k_slices` k32 steps
  the tile index of n-tile `nt`, step `ks` is `k_slices * (nt >> 1) + ks`, and
  inside a tile the fragment for n-tile `nt` sits at bytes `0..7` when `nt` is
  even and `8..15` when it is odd (both read by the same `ld.weak.global.ca.v4`
  at `W + 16*laneid + base`).  The column a lane owns inside a tile is
  `8*(nt & 1) + 2*t + r` for register `r`, and the address bit assignment inside
  the tile is
  `bit0 = k&1, bit1 = (k>>4)&1, bit2 = column&1, bit3 = column>>3,
   bits4-5 = (column>>1)&3, bits6-8 = (k>>1)&7`
  for the interleaved byte->k map, or
  `bit0-1 = k&3, bit2 = column&1, bit3 = column>>3, bits4-5 = (column>>1)&3,
   bits6-8 = k>>2`
  for the consecutive one; see `rrswin_dec0.hip` for both and for the A side.
  `"unverified"`: which of the two byte maps the mma uses was not settled --
  the first attempt to prove it was invalidated by an ambient-memory comparison
  (see the "Validation" section of rrswin_dec0.hip), so both are carried.
* A *column table* (E1's gain, the per-column addends) is read with the table
  address `W + ((laneid << 2) & 12) + off`: the four lanes with equal `laneid & 3`
  share a 16-byte word, so one load covers 32 columns.
* Register names (`%rNNNN`) are cited so each field can be checked against the
  PTX with `sed -n`.
"""


def layer():
    """Launch geometry, argument map and token geometry."""
    return dict(
        entry="cuda_dldn_engine_swin_dec0_kernel",
        corpus="0023-PREPASS_ENTRYPOINT_NAME.ptx",
        entry_lines=(1018, 47791),
        role="first Swin decoder (encoders' outputs -> feature planes)",
        # one warp per 8x8 token patch; blockDim 32x1x1 is a single wave, so no
        # %tid.z term appears in any address
        grid=(161, 93, 1),
        block=(32, 1, 1),
        wave=32,
        blocks=14973,
        token_patch=(8, 8),
        param_bytes=152,
        # param_0+80 is the block origin as a v2.u16 (s5, L1033); the token
        # geometry is 1280x720 -> 161x93 patches of 8x8
        image=(1280, 736),
        args=dict(
            input=8,               # u64, e4m3 input plane; read once, 24 ld.global.u32
            skip=24,               # u64, e4m3 skip input; 24 ld.global.v2.u16
            weights=40,            # u64, prepared weight image, 47376 bytes
            plane=48,              # u64, output e4m3 surface (see writes below)
            origin=80,             # v2.u16 block origin
            texture_a=88,          # u64, 18 tex.base.2d.v4.f16.f32 samples
            texture_b=96,          # u64, tex.base.2d.v4.f16.s32 at (x, y)
            texture_c=104,         # u64, tex.base.2d.v4.f16.s32 at (x, y)
            surface_a=112,         # u64, sust.b.2d.v4.b16.zero destination
            coord_scale=136,       # v2.f32, texture coordinate scale/origin
            surface_b=144,         # u64, sust.p.2d.v4.b32.zero destination
        ),
        # measured on the GPU: the launch dump files the 65,945,600-byte block
        # that holds these pointers as arg008 at base+0x1cc0000, so a pointer at
        # `block + N` is dumped at arg008[N + 0x1cc0000]
        arena=dict(
            dump_arg=8,
            dump_offset=0x1CC0000,
            dump_bytes=65945600,
            input=30146560,        # param+8   (Main's measurement, shared)
            plane=45219840,        # param+48  (measured here: the p1 oracle's
                                  # payload at D4R_RRSWIN_DEBUG_BASE lands at
                                  # arg008 offset 61,997,056, i.e. 45,219,840 +
                                  # 16,777,216, and shifting the base by 8192
                                  # shifts the first differing byte by 8192)
            param56=60293120,      # param+56  (Main's measurement, shared)
        ),
        writes=(
            # (statement, instruction, argument, address)
            (18502, "st.global.v4.u16", 48, "4*(ex*y + x)"),          # e4m3 planes
            (18603, "sust.p.2d.v4.b32.zero", 144, "(x, y)"),         # four f32
            (18813, "sust.b.2d.v4.b16.zero", 112, "(8*x, y)"),        # texture fuse
        ),
        unverified="param_0+32, +56, +72, +120 and +128 are never read by the "
                   "entry; the 4-byte value of param+0 is the (ex, ey) extents "
                   "pair (s13), and the offset of param+48 inside its block is "
                   "the only arena figure measured for this layer.",
    )


def phases():
    """The 19 GEMM phases in program order, each with its epilogue."""
    return [p1(), p2(), p3(), p4(), p5(), p6(), p7(), p8(), p9(), p10(),
            p11(), p12(), p13(), p14(), p15(), p16(), p17(), p18(), p19()]


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------

def _e1():
    """E1: patch expand, skip residual, RMS norm.  Returns the normalised tile."""
    return dict(
        name="E1",
        stmts=(422, 5936),
        steps=[
            dict(op="requantise", detail="D1 (96 f16x2 registers, %r464..%r775 "
                 "written by the 96 mma) -> 37 `mov.b32` of two "
                 "cvt.rn.satfinite.e4m3x2.f16x2 each plus 3 v2 pairs; 4 e4m3 "
                 "codes per b32, the two codes of two adjacent n-tiles for one "
                 "row"),
            dict(op="stage", detail="st.shared into the entry's 3200-byte smem as "
                 "two 400-cell planes of a 10x10 grid of 4-byte cells, cell = "
                 "row*40 + col*4 + t, plane stride 1600 bytes; 25 cells per lane "
                 "are written and all 800 cells are covered exactly once"),
            dict(op="gather", detail="24 ld.shared.v2.u16 per lane at "
                 "cell = plane*400 + row*40 + col*4 + t with "
                 "col = (laneid>>2) + 1, for row = 1..8, 0, 9 in both planes, "
                 "then (row = col, col = 0) and (row = col, col = 9) in both "
                 "planes -- 24 v2 loads = 48 f16x2, one b16 per e4m3x2"),
            dict(op="dequantise_skip", detail="24 ld.global.v2.u16 from param+24 "
                 "(s1613-s4384) then 96 cvt.rn.f16x2.e4m3x2"),
            dict(op="add", detail="204 add.f16: staged patch + dequantised skip"),
            dict(op="rms_norm", detail="mul.f16 squares, 60 add.f16x2 in-lane "
                 "sums, 24 shfl.sync.bfly.b32, + 2^-13 (mov.f64 0x3F20000000000000, "
                 "s5030), rsqrt.approx.ftz.f32, times the column gain read at "
                 "W + ((laneid<<2)&12) + {8448, 8464, 8480, 8496} (4 x "
                 "ld.global.v2.u16 = 8 f16 per lane = 32 columns).  No mean "
                 "subtraction and no 1/N"),
            dict(op="requantise", detail="48 cvt.rn.satfinite.e4m3x2.f16x2 "
                 "(s5860-s5928) -> 24 b32 = phases 2 and 3's A operand"),
        ],
        # the two registers E5 still reads after the softmax: the skip residual
        # sums %rs425 / %rs428 (s4484/s4485), defined once here and read again at
        # s10118 as phase 6's C seed
        residual_registers=("%rs425", "%rs428"),
        unverified="the lane -> (row, col) map of the staging and of the "
                   "gather was derived by executing the region s539-s1457 as "
                   "integer PTX for all 32 lanes; the resulting map is "
                   "self-consistent (exactly 800 distinct cells, one writer "
                   "each, every read cell written) but has not been checked "
                   "against NVIDIA's own values.",
    )


def _softmax(scale=0.017181396484375):
    """The cubic-exponent softmax E4 and E9 use (identical code, shifted regs)."""
    return dict(
        clamp=(-0.55615234375, 0.55615234375),
        scale=scale,
        cube=dict(c0=0.92724609375, note="u = f16(1.375 + m*(c0 - m*m))"),
        trick="expval = f16 from bits ((u_bits << 5) & 0x7FE07FE0), i.e. "
              "2^(32*frac(u) - 47)",
        row_sum="8 row sums (one per row g, g+8 of the four m-tiles) from 12 "
                "add.f16x2 partials and 16 shfl.sync.bfly.b32; there is NO row "
                "maximum in this epilogue, the max/min f16x2 are the clamp",
        normalise="192 rcp.approx.ftz.f32 -> 96 reciprocals, then 192 mul.f16",
    )


def _activation():
    """The clamped cubic activation of E12, E14, E16 and E18."""
    return dict(
        clamp=(-2.0, 2.0),
        gain="g = f16(0.5 + y*(0.412109375 - 0.0810546875*|y|)) with "
             "y = clamp(x, -2, 2), all in f16",
        out="x * g, computed in f32 and rounded to f16",
        note="a cubic ramp, not sigmoid/GELU: g(+2) = 1 so out = x above 2, "
             "g(-2) = 0 so out = 0 below -2",
        requantise="32 cvt.rn.satfinite.e4m3x2.f16x2 -> 16 b32 = the next "
                   "phase's A",
    )


# ---------------------------------------------------------------------------
# the phases
# ---------------------------------------------------------------------------

def p1():
    return dict(
        index=1,
        stmts=(326, 421), lines=(1393, 2061),
        gemm=dict(m_tiles=3, n_tiles=16, k=64, k_slices=2, mma=96,
                  mma_shape="m16n8k32", wmma_steps=4,
                  note="the tool prints 96x256 K=64; the 48 chains of two steps "
                       "reproduce 3*16*2 = 96, so N is 128 and the 32 B "
                       "fragments are 16 n-tiles x 2 k slices"),
        mma_order="for mi in 0..2, for p in 0..7, for ks in 0..1, for j in 0..1: "
                  "a_fragment = 2*mi + ks, n_tile = 2*p + j, b_fragment = "
                  "4*p + 2*ks + j",
        a=dict(source="arg+8 input plane, staged by NVIDIA's prologue into 24 "
            "registers (%r756..%r759, %r776..%r779, %r1076..%r1079, "
            "%r1096..%r1099, %r1396..%r1399, %r1416..%r1419) in mma first-use "
            "order", fragments=6, registers=24, requantise="none: already e4m3"),
        b=dict(source="weight image", base=0,
               tile="k_slices*(nt>>1) + ks = 2*(nt>>1) + ks",
               fragments=32, note="16 n-tiles x 2 k slices; the largest word "
               "offset is 7688, so the phase reads [0, 7692)"),
        c=dict(kind="per-n-tile f16x2 bias word", count=16, base=8192, stride=16,
               addr="W + ((laneid<<2)&12) + 8192 + 16*nt",
               regs=["%r1113", "%r1123", "%r1153", "%r1163", "%r1193", "%r1203",
                     "%r1233", "%r1243", "%r1273", "%r1283", "%r1313", "%r1323",
                     "%r1353", "%r1363", "%r1393", "%r1403"],
               note="the first k step of each chain; the second reads the first "
                    "step's f16 result"),
        d_registers=96,
        epilogue=_e1(),
    )


def _proj(index, stmts, lines, a_source, a_registers, b_base, c, m_tiles,
          n_tiles, epilogue, note=""):
    """One K=32 projection: the shape of phases 2, 3, 6, 7, 8 and 11..19."""
    return dict(
        index=index, stmts=stmts, lines=lines,
        gemm=dict(m_tiles=m_tiles, n_tiles=n_tiles, k=32, k_slices=1,
                  mma=m_tiles * n_tiles, mma_shape="m16n8k32", wmma_steps=2,
                  note=note),
        mma_order="for mi in 0..%d, for nt in 0..%d: a_fragment = mi, "
                  "n_tile = nt" % (m_tiles - 1, n_tiles - 1),
        a=a_source,
        b=dict(source="weight image", base=b_base, tile="(nt>>1)",
               fragments=n_tiles,
               span="[%d, %d)" % (b_base, b_base + 512 * ((n_tiles + 1) // 2))),
        c=c,
        d_registers=2 * m_tiles * n_tiles,
        epilogue=epilogue,
    )


def _zero_c():
    return dict(kind="constant zero", regs=["%r13609"],
                note="mov.u32 %r13609, 0 at L6552; the mma's C field is "
                     "{%r13609, %r13609}")


def _norm_tile_a(regs):
    return dict(source="E1's normalised tile (the 24 packed b32 E1's last "
                "requantise writes)", fragments=6, registers=24, regs=regs,
                requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E1")


def _bias_c(base, count, stride=16):
    return dict(kind="per-n-tile f16x2 bias word", count=count, base=base,
                stride=stride,
                addr="W + ((laneid<<2)&12) + %d + %d*nt" % (base, stride))


def _acc_c(src, regs, n_tiles, m_tiles):
    return dict(kind="the previous phase's f16x2 accumulators",
                count=2 * m_tiles * n_tiles, regs=regs, note=src)


def p2():
    return _proj(2, (5937, 5960), (10458, 10622),
                 _norm_tile_a(["%r9539", "%r9579", "%r9619", "%r9659", "%r9699",
                               "%r9739"]),
                 None, 8512, _zero_c(), 6, 4,
                 dict(name="E2", stmts=(5961, 5967),
                      steps=[dict(op="load", detail="phase 3's B, two "
                                  "ld.weak.global.ca.v4 at W + 16*laneid + "
                                  "{9536, 10048}")]))


def p3():
    return _proj(3, (5968, 5991), (10639, 10803),
                 _norm_tile_a(["%r9539", "%r9579", "%r9619", "%r9659", "%r9699",
                               "%r9739"]),
                 None, 9536, _zero_c(), 6, 4,
                 dict(name="E3", stmts=(5992, 6162),
                      steps=[
                          dict(op="load", detail="phase 4's C bias: 24 "
                                  "ld.weak.global.ca.v4 at "
                                  "W + 16*laneid + 10560 + 512*i, i = 0..23 "
                                  "= [10560, 22348)"),
                          dict(op="requantise", detail="phase 3's D (%r4904.."
                                  "%r5135) -> 80 cvt.rn.satfinite.e4m3x2 in 40 "
                                  "b32 = phase 4's A (16 b32) and B (24 b32)"),
                      ]))


def p4():
    return dict(
        index=4, stmts=(6163, 6210), lines=(11188, 11520),
        gemm=dict(m_tiles=4, n_tiles=12, k=32, k_slices=1, mma=48,
                  mma_shape="m16n8k32", wmma_steps=2,
                  note="the 64x96 score block of the first attention window"),
        mma_order="for mi in 0..3, for nt in 0..11: a_fragment = mi, "
                  "n_tile = nt",
        a=dict(source="E3's pack of phase 3's D", fragments=4, registers=16,
               regs=["%r5433", "%r5553", "%r5673", "%r5793"],
               requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E3"),
        b=dict(source="E3's pack of phase 2's D (the V of the first window)",
               fragments=12, registers=24,
               regs=["%r5687", "%r5697", "%r5707", "%r5717", "%r5727", "%r5737",
                     "%r5747", "%r5757", "%r5767", "%r5777", "%r5787", "%r5797"],
               requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E3"),
        c=dict(kind="per-(m-tile, n-tile) f16x2 seed pair", count=48,
               base=10560, span=(10560, 22348),
               word_offset="base + 2048*(i >> 1) + 8*(i & 1) for mma index i, "
                           "the pair at +0 and +4",
               addr="W + ((laneid<<2)&12) + word_offset",
               note="the 12288-byte position bias of the first window; four "
                    "consecutive mma share a 512-byte stride"),
        d_registers=96,
        epilogue=dict(
            name="E4", stmts=(6211, 10053), softmax=_softmax(),
            steps=[
                dict(op="exponent", detail="96 element groups: "
                     "m = clamp(f16(x * 0.017181396484375), -0.55615234375, "
                     "+0.55615234375); u = f16(1.375 + m*(0.92724609375 - m*m)); "
                     "expval = f16 from bits ((u_bits << 5) & 0x7FE07FE0)"),
                dict(op="row_sum", detail="12 add.f16x2 partials per lane, 16 "
                     "shfl.sync.bfly.b32, 8 row sums; no row maximum"),
                dict(op="normalise", detail="192 rcp.approx.ftz.f32 -> 96 "
                     "reciprocals; 192 mul.f16 -> the 96 probabilities"),
                dict(op="transpose", detail="48 movmatrix.sync.trans.aligned."
                     "m8n8.b16 over phase 2's D (the V matrix)"),
                dict(op="requantise", detail="144 cvt.rn.satfinite.e4m3x2 in 48 "
                     "b32 = phase 5's A (48 b32, 4 m-tiles x 3 k slices) and B "
                     "(24 b32, 4 n-tiles x 3 k slices)"),
            ]),
    )


def p5():
    return dict(
        index=5, stmts=(10054, 10101), lines=(21667, 21999),
        gemm=dict(m_tiles=4, n_tiles=4, k=96, k_slices=3, mma=48,
                  mma_shape="m16n8k32", wmma_steps=6,
                  note="16 chains of three steps; the tool prints 192x96 "
                       "because it attributes the k slices to M and N"),
        mma_order="for mi in 0..3, for nt in 0..3, for ks in 0..2: "
                  "a_fragment = 3*mi + ks, b_fragment = 3*nt + ks",
        a=dict(source="E4's pack of the softmax probabilities (P of the first "
            "window)", fragments=12, registers=48,
            regs=["%r8639", "%r8659", "%r8679", "%r8759", "%r8779", "%r8799",
                  "%r8879", "%r8899", "%r8919", "%r8999", "%r9019", "%r9039"],
            requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E4"),
        b=dict(source="E4's pack of the transposed phase 2's D (the V matrix)",
               fragments=12, registers=24,
               regs=["%r8933", "%r8943", "%r8953", "%r8963", "%r8973", "%r8983",
                     "%r8993", "%r9003", "%r9013", "%r9023", "%r9033", "%r9043"],
               requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E4"),
        c=dict(kind="zero then the chain's own accumulator", count=1,
               regs=["%r13609"],
               note="ks = 0 reads {%r13609, %r13609}; ks > 0 reads the previous "
                    "step's D pair"),
        d_registers=96,
        epilogue=dict(
            name="E5", stmts=(10102, 10261),
            steps=[
                dict(op="load", detail="phase 6's B, two v4 at "
                     "W + 16*laneid + {22848, 23360}"),
                dict(op="table", detail="phase 6's column table: four "
                     "ld.global.v2.u16 at W + ((laneid<<2)&12) + "
                     "{39232, 39248, 39264, 39280} = 8 f16 = 32 columns"),
                dict(op="add", detail="64 add.f16: table[column] + the E1 "
                     "residual %rs425 / %rs428 (defined at s4484/s4485, the only "
                     "definition in the entry), giving phase 6's C operand"),
                dict(op="requantise", detail="32 cvt.rn.satfinite.e4m3x2 of "
                     "phase 5's D -> 16 b32 = phase 6's A"),
            ]),
    )


def p6():
    return _proj(
        6, (10262, 10277), (22427, 22535),
        dict(source="E5's pack of phase 5's D (P*V of the first window)",
             fragments=4, registers=16, regs=["%r9121", "%r9161", "%r9201",
                                              "%r9241"],
             requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E5"),
        None, 22848,
        _acc_c("E1's skip residual plus the per-column table at W + 39232",
               ["%r9097", "%r9107", "%r9117", "%r9127", "%r9137", "%r9147",
                "%r9157", "%r9167", "%r9177", "%r9187", "%r9197", "%r9207",
                "%r9217", "%r9227", "%r9237", "%r9247"], 4, 4),
        4, 4,
        dict(name="E6", stmts=(10278, 10284),
             steps=[dict(op="load", detail="phase 7's B at "
                         "W + 16*laneid + {23872, 24384}")]))


def p7():
    return _proj(
        7, (10285, 10308), (22552, 22716),
        _norm_tile_a(["%r9539", "%r9579", "%r9619", "%r9659", "%r9699", "%r9739"]),
        None, 23872, _zero_c(), 6, 4,
        dict(name="E7", stmts=(10309, 10315),
             steps=[dict(op="load", detail="phase 8's B at "
                         "W + 16*laneid + {24896, 25408}")]))


def p8():
    return _proj(
        8, (10316, 10339), (22733, 22897),
        _norm_tile_a(["%r9539", "%r9579", "%r9619", "%r9659", "%r9699", "%r9739"]),
        None, 24896, _zero_c(), 6, 4,
        dict(name="E8", stmts=(10340, 10510),
             steps=[
                 dict(op="load", detail="phase 9's C bias: 24 v4 at "
                     "W + 16*laneid + 25920 + 512*i, i = 0..23 = [25920, 37708)"),
                 dict(op="requantise", detail="phase 8's D (%r9507..%r9737) -> "
                     "80 cvt.rn.satfinite.e4m3x2 in 40 b32 = phase 9's A (16) "
                     "and B (24)"),
             ]))


def p9():
    out = dict(p4())
    out.update(index=9, stmts=(10511, 10558), lines=(23282, 23614))
    out["gemm"] = dict(p4()["gemm"], note="the 64x96 score block of the second "
                         "attention window")
    out["a"] = dict(p4()["a"], source="E8's pack of phase 8's D",
                    regs=["%r10036", "%r10156", "%r11036", "%r11156"],
                    requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E8")
    out["b"] = dict(p4()["b"], source="E8's pack of phase 7's D (the V of the "
                    "second window)",
                    regs=["%r11287", "%r11397", "%r11507", "%r11617", "%r11727",
                          "%r11837", "%r11947", "%r12057", "%r12167", "%r12277",
                          "%r12387", "%r12497"],
                    requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E8")
    out["c"] = dict(p4()["c"], base=25920, span=(25920, 37708),
                    note="the second window's 12288-byte position bias")
    out["epilogue"] = dict(name="E9", stmts=(10559, 14396), softmax=_softmax(),
                           steps=[
                               dict(op="exponent", detail="identical code to "
                                    "E4's, registers shifted; 96 element groups"),
                               dict(op="row_sum", detail="12 add.f16x2 partials, "
                                    "16 shfl.sync.bfly.b32, 8 row sums"),
                               dict(op="normalise", detail="192 rcp.approx."
                                    "ftz.f32 then 192 mul.f16"),
                               dict(op="transpose", detail="48 movmatrix.sync."
                                    "trans.aligned.m8n8.b16 over phase 7's D"),
                               dict(op="requantise", detail="144 "
                                    "cvt.rn.satfinite.e4m3x2 in 48 b32 = "
                                    "phase 10's A (48) and B (24)"),
                           ])
    return out


def p10():
    out = dict(p5())
    out.update(index=10, stmts=(14397, 14444), lines=(33756, 34088))
    out["gemm"] = dict(p5()["gemm"], note="P*V of the second attention window")
    out["a"] = dict(p5()["a"], source="E9's pack of the second window's "
                    "probabilities",
                    regs=["%r13242", "%r13262", "%r13282", "%r13302", "%r13322",
                          "%r13342", "%r13362", "%r13382", "%r13402", "%r13422",
                          "%r13442", "%r13462"],
                    requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E9")
    out["b"] = dict(p5()["b"], source="E9's pack of the transposed phase 7's D",
                    regs=["%r13933", "%r13943", "%r13953", "%r13963", "%r13973",
                          "%r13983", "%r13993", "%r14003", "%r14013", "%r14023",
                          "%r14033", "%r14043"],
                    requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E9")
    out["epilogue"] = dict(name="E10", stmts=(14445, 14499), steps=[
        dict(op="load", detail="phase 11's B at W + 16*laneid + {38208, 38720}"),
        dict(op="requantise", detail="32 cvt.rn.satfinite.e4m3x2 of phase 10's "
             "D -> 16 b32 = phase 11's A; no column table"),
    ], note="phase 11's C is phase 6's accumulator (the residual), so phase 11 "
           "computes D11 = A11*B11 + D6")
    return out


def p11():
    return _proj(
        11, (14500, 14515), (34217, 34325),
        dict(source="E10's pack of phase 10's D", fragments=4, registers=16,
             regs=["%r13723", "%r13763", "%r13803", "%r13843"],
             requantise="cvt.rn.satfinite.e4m3x2.f16x2 in E10"),
        None, 38208,
        _acc_c("phase 6's accumulators (the running residual)",
               ["%r9089", "%r9099", "%r9109", "%r9119", "%r9129", "%r9139",
                "%r9149", "%r9159", "%r9169", "%r9179", "%r9189", "%r9199",
                "%r9209", "%r9219", "%r9229", "%r9239"], 4, 4),
        4, 4,
        dict(name="E11", stmts=(14516, 15402), op="rms_norm",
             steps=[
                 dict(op="squares", detail="192 mul.f16 of phase 11's D"),
                 dict(op="sum", detail="16 add.f16x2 in-lane, 16 "
                      "shfl.sync.bfly.b32 + add.f16 across the four lanes"),
                 dict(op="scale", detail="+ 2^-13 then rsqrt.approx.ftz.f32, 64 "
                      "cvt.f32.f16 / rsqrt / cvt.rn.f16.f32, rounded to f16"),
                 dict(op="gain", detail="times the column gain at "
                      "W + ((laneid<<2)&12) + {39296, 39312, 39328, 39344}; "
                      "192 mul.f16 (inv*gain, then x*(inv*gain))"),
                 dict(op="load", detail="phase 12's B at W + 16*laneid + "
                      "{39360, 39872} and its C bias at "
                      "W + ((laneid<<2)&12) + {40384, 40400, 40416, 40432}"),
                 dict(op="requantise", detail="32 cvt.rn.satfinite.e4m3x2 -> "
                      "16 b32 = phase 12's A"),
             ],
             note="no mean subtraction and no 1/N; the normalised tile N (32 "
                  "f16x2) is also what E13, E15 and E17 repack, so phases 12, "
                  "14, 16 and 18 all take the same A operand"))


def _mlp_a(source, regs, note=""):
    return dict(source=source, fragments=4, registers=16, regs=regs,
                requantise="cvt.rn.satfinite.e4m3x2.f16x2 in the preceding "
                           "epilogue", note=note)


N_REGS = ["%r15544", "%r15584", "%r15624", "%r15664"]        # E13's pack of N
ACT14 = ["%r16674", "%r16714", "%r16754", "%r16794"]         # E14's pack
ACT16 = ["%r16876", "%r16916", "%r16956", "%r16996"]         # E15's pack of N
ACT17 = ["%r18006", "%r18046", "%r18086", "%r18126"]         # E16's pack
ACT18 = ["%r18208", "%r18248", "%r18288", "%r18328"]         # E17's pack of N
ACT19 = ["%r19338", "%r19378", "%r19418", "%r19458"]         # E18's pack


def p12():
    return _proj(
        12, (15403, 15418), (36471, 36579),
        _mlp_a("E11's pack of the RMS-normed tile N",
               ["%r14211", "%r14251", "%r14291", "%r14331"]),
        None, 39360, _bias_c(40384, 4), 4, 4,
        dict(name="E12", stmts=(15419, 16127), activation=_activation(),
             steps=[
                 dict(op="activate", detail="over each of phase 12's 32 f16x2 D "
                      "registers: y = clamp(x, -2, 2), "
                      "g = 0.5 + y*(0.412109375 - 0.0810546875*|y|), out = x*g, "
                      "with 128 cvt.rn.f16.f32, 32 each neg/max/min/abs.f16x2, "
                      "96 mul.f16x2, 32 sub.f16x2, 32 add.f16x2"),
                 dict(op="table", detail="64 add.f16 of the column table read at "
                      "W + ((laneid<<2)&12) + {41472, 41488, 41504, 41520}, "
                      "written over the *activation inputs*, so phase 13's "
                      "activation input is D12 + table[column]"),
                 dict(op="load", detail="phase 13's B at W + 16*laneid + "
                      "{40448, 40960}"),
                 dict(op="requantise", detail="32 packs of the activation "
                      "outputs -> 16 b32 = phase 13's A"),
             ],
             note="the only one of the four branches with a per-column table"))


def p13():
    return _proj(
        13, (16128, 16143), (38934, 39042),
        _mlp_a("E12's pack of the activated (and table-added) tile",
               ["%r15342", "%r15382", "%r15422", "%r15462"]),
        None, 40448,
        _acc_c("phase 12's accumulators plus the per-column table at W + 41472, "
               "i.e. the values E12 wrote over the activation inputs",
               ["%r15269", "%r15329", "%r15389", "%r15449", "%r15509", "%r15569",
                "%r15629", "%r15689", "%r15749", "%r15809", "%r15869", "%r15929",
                "%r15989", "%r16049", "%r16109", "%r16169"], 4, 4),
        4, 4,
        dict(name="E13", stmts=(16144, 16207), steps=[
            dict(op="load", detail="phase 14's B at W + 16*laneid + {41536, "
                 "42048} and its C bias at W + ((laneid<<2)&12) + "
                 "{42560, 42576, 42592, 42608}"),
            dict(op="requantise", detail="32 packs of the *normed tile* N (not "
                 "of phase 13's D) -> 16 b32 = phase 14's A"),
        ]))


def p14():
    return _proj(
        14, (16208, 16223), (39182, 39290),
        _mlp_a("E13's pack of the RMS-normed tile N", N_REGS,
               note="the same 16 registers as phase 16's A"),
        None, 41536, _bias_c(42560, 4), 4, 4,
        dict(name="E14", stmts=(16224, 16823), activation=_activation(),
             steps=[
                 dict(op="activate", detail="the same clamped cubic as E12, over "
                      "phase 14's own D; no ld.global and no table add"),
                 dict(op="load", detail="phase 15's B at W + 16*laneid + "
                      "{42624, 43136}"),
                 dict(op="requantise", detail="32 packs of the activation "
                      "outputs -> 16 b32 = phase 15's A"),
             ]))


def p15():
    return _proj(
        15, (16824, 16839), (41342, 41450),
        _mlp_a("E14's pack of the activated tile", ACT14), None, 42624,
        _bias_c(44672, 4), 4, 4,
        dict(name="E15", stmts=(16840, 16903), steps=[
            dict(op="load", detail="phase 16's B at W + 16*laneid + {43648, "
                 "44160} and its C bias at W + ((laneid<<2)&12) + "
                 "{44672, 44688, 44704, 44720}"),
            dict(op="requantise", detail="32 packs of the normed tile N -> "
                 "16 b32 = phase 16's A"),
        ]))


def p16():
    return _proj(
        16, (16904, 16919), (41590, 41698),
        _mlp_a("E15's pack of the RMS-normed tile N", ACT16), None, 43648,
        _bias_c(46784, 4), 4, 4,
        dict(name="E16", stmts=(16920, 17519), activation=_activation(),
             steps=[
                 dict(op="activate", detail="the same clamped cubic as E12, over "
                      "phase 16's own D"),
                 dict(op="load", detail="phase 17's B at W + 16*laneid + "
                      "{44736, 45248}"),
                 dict(op="requantise", detail="32 packs of the activation "
                      "outputs -> 16 b32 = phase 17's A"),
             ]))


def p17():
    return _proj(
        17, (17520, 17535), (43750, 43858),
        _mlp_a("E16's pack of the activated tile", ACT17), None, 44736,
        _bias_c(42560, 4) if False else dict(
            kind="per-n-tile f16x2 bias word", count=4, base=46784, stride=16,
            addr="W + ((laneid<<2)&12) + 46784 + 16*nt",
            note="phase 17's C is read in E17 as "
                 "W + ((laneid<<2)&12) + {46784, 46800, 46816, 46832}"),
        4, 4,
        dict(name="E17", stmts=(17536, 17599), steps=[
            dict(op="load", detail="phase 18's B at W + 16*laneid + {45760, "
                 "46272} and its C bias at W + ((laneid<<2)&12) + "
                 "{46784, 46800, 46816, 46832}"),
            dict(op="requantise", detail="32 packs of the normed tile N -> "
                 "16 b32 = phase 18's A"),
        ]))


def p18():
    return _proj(
        18, (17600, 17615), (43998, 44106),
        _mlp_a("E17's pack of the RMS-normed tile N", ACT18), None, 45760,
        _bias_c(46784, 4), 4, 4,
        dict(name="E18", stmts=(17616, 18215), activation=_activation(),
             steps=[
                 dict(op="activate", detail="the same clamped cubic as E12, over "
                      "phase 18's own D; this is the last requantisation in "
                      "the entry"),
                 dict(op="load", detail="phase 19's B at W + 16*laneid + "
                      "{46848, 46960}"),
                 dict(op="requantise", detail="32 packs of the activation "
                      "outputs -> 16 b32 = phase 19's A"),
             ]))


def p19():
    return _proj(
        19, (18216, 18231), (46158, 46266),
        _mlp_a("E18's pack of the activated tile", ACT19), None, 46848,
        _acc_c("phase 18's accumulators",
               ["%r19089", "%r19149", "%r19209", "%r19269", "%r19329", "%r19389",
                "%r19449", "%r19509"], 4, 2),
        4, 4,
        dict(name="E19", stmts=(18232, 19160),
             note="the final epilogue stays NVIDIA's in the mirror cut; it is "
                  "described here because the engine's output feeds it",
             steps=[
                 dict(op="row_max", detail="a real row maximum (unlike E4's): a "
                      "selp chain over laneid&3 picks one of the four D pairs "
                      "per lane, then four shfl.sync.idx.b32 with "
                      "srcLane = (laneid&7) | ((((laneid>>3)+d)&3)<<2) for "
                      "d = 0,1,2,-1, then a selp chain over laneid>>3; max.f16x2 "
                      "/ max.f16 and sub.f16x2 subtract it"),
                 dict(op="bounds", detail="ld.param.v2.u32 at param+0 gives "
                      "(ex, ey); out-of-range lanes branch to $L__BB1_367 and "
                      "skip the rest"),
                 dict(op="surface1", detail="4 cvt.rn.satfinite.e4m3x2 and one "
                      "st.global.v4.u16 at 4*(ex*y + x) into param+48"),
                 dict(op="surface2", detail="per element v: a "
                      "1 - 2/(1 + 2^(|v|*log2e)) branch (with a hard 1.0 above "
                      "9.0109...) and an fma polynomial over v^2 selected by "
                      "|v| >= 0.6, then sust.p.2d.v4.b32.zero at (x, y) into "
                      "param+144"),
                 dict(op="softmax", detail="x - max in f16, 2^((x-max)*log2e) "
                      "in f32, an f32 row sum, div.approx.ftz.f32 by 1.0, five "
                      "mul.f16x2"),
                 dict(op="texture_fuse", detail="18 tex.base.2d.v4.f16.f32 of "
                      "param+88 at f32 coordinates built from param+136, plus "
                      "tex.base.2d.v4.f16.s32 of param+96 and param+104 at "
                      "integer (x, y); folded with fma.rn.f16, guarded by "
                      "set.nan.f16.f16 / setp.ne.s16 so a NaN sample is "
                      "skipped, written by sust.b.2d.v4.b16.zero at (8*x, y) "
                      "into param+112 with a constant 0 fourth channel"),
             ],
             unverified="E19's two evaluation branches are exact in the PTX "
                        "(every instruction is cited) but which model function "
                        "they implement is not determinable from it."),
        )






def _final():
    return None
