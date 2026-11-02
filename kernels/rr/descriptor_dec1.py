#!/usr/bin/env python3
"""Per-layer descriptor for cuda_dldn_engine_swin_dec1_kernel (nvngx_dlssd 310.7).

The parameterised-engine deliverable: every number here is either (a) read out of
~/.cache/d4r-rr-corpus/0022-PREPASS_ENTRYPOINT_NAME.ptx, entry
`cuda_dldn_engine_swin_dec1_kernel`, and reproducible with

    python3 kernels/rr/rr_layer_spec.py dec1

or (b) marked `"unverified"` with the reason.  Phase boundaries, mma operand
register lists and the weight-image offsets were extracted mechanically from the
linearised statement list; the epilogue operations are named from
kernels/rr/rrswin_dec1_epilogues.md, which cites the statement ranges.

Geometry recovered by interpreting NVIDIA's own index arithmetic (not guessed)
is recorded in `e1_patch_grid`.

    python3 -c "import descriptor_dec1 as d; print(d.layer())"
"""

# ---------------------------------------------------------------------------
# launch geometry, from the entry's prologue (s1-s325) and rr_layer_spec.py
# ---------------------------------------------------------------------------

def layer():
    """Launch geometry, parameter-block layout and tile shape.

    `param_offsets` are byte offsets into the 144-byte parameter block; the entry
    declares it as one `.param .align 8 .b8 ..._param_0[144]`, so every offset below
    is a plain byte read (`rr_layer_spec.py dec1`: "param block 144 bytes (5
    ld.param over 3 distinct offsets)", "param offsets [8, 40, 80]").
    """
    return {
        "kernel": "cuda_dldn_engine_swin_dec1_kernel",
        "corpus": "0022-PREPASS_ENTRYPOINT_NAME.ptx",
        "grid": (81, 47, 1),
        "block": (32, 1, 2),          # two waves per block; %tid.z is the window index
        "waves": 2,
        "laneid": 32,
        "param_block_bytes": 144,
        "param_offsets": {
            0: {"kind": "u32x2", "name": "extents", "meaning": "ex, ey"},
            8: {"kind": "ptr", "name": "in_plane",
                "meaning": "input feature plane arena, 52 ld.global.u32 in the prologue"},
            24: {"kind": "ptr", "name": "patch_gather",
                 "meaning": "the +24 skip input E1 gathers the patch from; reached as "
                            "[param_0+80 - 56], not as its own ld.param"},
            40: {"kind": "ptr", "name": "weights",
                 "meaning": "NVIDIA's prepared weight image (arg040, 96,800 bytes)"},
            48: {"kind": "ptr", "name": "out_plane",
                 "meaning": "output plane arena; E25's 16 guarded st.global.u32 write here"},
            80: {"kind": "v2u16", "name": "block_origin",
                 "meaning": "the per-block origin table; %rd1 = param_0+80 and "
                            "param_0+24 is [rd1-56]"},
            88: {"kind": "tex", "name": "colour_tex"},
            96: {"kind": "tex", "name": "tex_int_0"},
            104: {"kind": "tex", "name": "tex_int_1"},
            112: {"kind": "surface", "name": "surface_0"},
            120: {"kind": "surface", "name": "surface_1"},
            136: {"kind": "f32x2", "name": "tex_coord_scale"},
        },
        "tid_z_strides": {
            # two per-tid.z strides exist and are used by different groups
            # (rrswin_dec1_epilogues.md §1.1, read off the load chains).
            "phase1_B": {"bytes": 12288, "phase": 1},
            "phase1_bias": {"bytes": 256, "phase": 1},
            "phase2_5_B_and_phase6_bias": {"bytes": 18432, "phases": [2, 3, 4, 5, 6]},
        },
        "tid_z_range": [0, 1],
        "shared_bytes": 6400,
        "token_geometry": {
            "patch": "8x8 tokens per warp; a 10x10 halo grid of 4-word cells",
            "unverified": "the 10x10 grid's mapping to image coordinates was not "
                          "reduced to a closed form; see e1_patch_grid for what the "
                          "interpreter measured and rrswin_dec1_epilogues.md U4 for "
                          "what remains open.",
        },
        "constants_f16": {
            # §1.2: value after the cvt to f16, i.e. what the arithmetic actually uses
            "softsign_scale": 0x2466,        # 0.017181396484375
            "clamp_lo": 0xB873,               # -0.55615234375
            "clamp_hi": 0x3873,               # +0.55615234375
            "cubic_a": 0x3B6B,                # 0.92724609375
            "cubic_b": 0x3D80,                # 1.375
            "rms_eps": 0x0800,                # 2^-13
            "act_a": 0x3698,                  # 0.412109375
            "act_b": 0x2D30,                  # 0.0810546875
            "act_half": 0x3800,               # 0.5
            "act_two": 0x4000,                # 2.0
        },
    }


# ---------------------------------------------------------------------------
# E1's patch grid, recovered by interpreting the PTX
# ---------------------------------------------------------------------------

def e1_patch_grid():
    """The 6400-byte shared tile E1 builds and reads, as measured.

    `/tmp/d4r-dec1/interp.py` evaluates NVIDIA's own index arithmetic for all 32
    lanes x tid.z in {0,1} over s689-s1367 (the 48 guarded `st.shared`) and
    s1370-s2214 (the 24 `ld.shared.v2.u16`).  The layout below is that output,
    not an inference.
    """
    return {
        "sub_arrays": 4,
        "sub_array_bytes": 1600,
        "word_index": "sub*400 + row*40 + col*4 + t   (row 0..9, col 0..9, t = laneid&3)",
        "grid": "10 rows x 10 cols x 4 words",
        "pack_index": "p = 16*mgrp + 4*(gp>>1) + 2*h + row; "
                      "pk[p] = e4m3x2(D[mgrp][2q][j=2][h][row]) "
                      "| e4m3x2(D[mgrp][2q+1][j=2][h][row])<<16, q = gp>>1",
        "live_packs": [p for p in range(48) if p not in
                       (33, 35, 37, 39, 41, 43, 45, 47)],
        "row_parity": {"tid_z_0": [1, 3, 5, 7, 9], "tid_z_1": [0, 2, 4, 6, 8]},
        "coverage": "the 48 stores tile all 400 cells exactly once, zero duplicates",
        "cells_per_g_tid_z_0": [28, 28, 28, 24, 20, 20, 24, 28],
        "cells_per_g_tid_z_1": [24, 28, 28, 28, 28, 24, 20, 20],
        "readback_positions": {
            "count": 24,
            "note": "6 grid positions x 4 sub-arrays",
            "tid_z_0": "rows {1,2,3,4,0,9} at column g+1",
            "tid_z_1": "rows {5,6,7,8} at column g+1, plus column halos "
                       "(g+1,0) and (g+1,9)",
        },
        "unverified": "the gather source arithmetic (param_0+24, index "
                      "clamp_row*(8*ex) + clamp_col*8 + clamp(t,0,7)), the "
                      "sum-of-squares trees and the restage were not decoded; the "
                      "epilogue cites them at s2215-s7095.",
    }


# ---------------------------------------------------------------------------
# the 26 phases
# ---------------------------------------------------------------------------

def _gemm(mt, nt, k, mma, a, b, c, ksplit=None):
    d = {
        "m_tiles": mt, "n_tiles": nt, "k": k, "mma": mma,
        "a": a, "b": b, "c": c,
        "k_tiling": ksplit or "one 32-wide k step; the engine emits two 16-wide "
                             "gfx12 WMMA steps in NVIDIA's slot order",
    }
    return d


# Weight-image byte offsets are relative to W (param_0+40).  `tid_z` says whether
# the offset carries a per-tid.z stride and which one.
_PATCH = "E1's restaged patch grid (shared), m-tile index selects the A fragment"
_NORMT = "the RMS-normalised tile N of E9 (32 f16x2), requantised to e4m3"


def phases():
    """One dict per phase, in execution order.

    `m_tiles`, `n_tiles`, `k` and `mma` are exact: `mma` is the statement count and
    the tiles come from counting the distinct A quads and B pairs the phase's mma
    read (rr_layer_spec.py's phase table).  `a`, `b`, `c` name the operand's source;
    where the source is a weight-image read the byte offset and the per-tid.z
    stride are given.
    """
    out = []
    P = out.append

    P(dict(
        index=1, stmts=(403, 546), epilogue="E1", epilogue_stmts=(547, 7095),
        gemm=_gemm(9, 48, 96, 144,
                   {"source": "the staged 8x8 block patch (param_0+8)",
                    "operands": "9 .f16.e4m3 fragments of four b32, 36 registers, "
                                "passed across the cut",
                    "requantise": "already e4m3"},
                   {"source": "weight image",
                    "byte_offset": 0, "byte_span": 11788, "stride": "12288 * tid.z",
                    "note": "48 B fragments = 24 blocks x 16 columns; banded, the "
                            "three m-groups are a running sum along j, not a "
                            "k-accumulation"},
                   {"source": "per-chain bias seed",
                    "byte_offset": 24576, "byte_span": 240, "stride": "256 * tid.z",
                    "kind": "per-(chain, n-tile) seed pair broadcast into both C registers"},
    )))

    # ---- the four 96x32 projections -------------------------------------------------
    for idx, a_mt, b_off, c_src in (
            (2, [0, 2, 4, 6, 8, 10], 0, "zero"),
            (3, [0, 2, 4, 6, 8, 10], 2048, "zero"),
            (4, [1, 3, 5, 7, 9, 11], 1024, "phase 2"),
            (5, [1, 3, 5, 7, 9, 11], 3072, "phase 3")):
        st = {2: (7096, 7119), 3: (7127, 7150), 4: (7158, 7181), 5: (7189, 7212)}[idx]
        P(dict(
            index=idx, stmts=st, epilogue="E%d" % (idx + 1 if idx in (2, 3) else idx),
            gemm=_gemm(6, 4, 32, 24,
                       {"source": _PATCH, "m_tiles_used": a_mt,
                        "requantise": "already e4m3 (read back from shared by E1)"},
                       {"source": "weight image",
                        "byte_offset": 25216 + b_off, "byte_span": 1024,
                        "stride": "18432 * tid.z"},
                       {"source": c_src,
                        "kind": "per-(m,n) seed pair" if c_src != "zero" else "zero register"},
        )))

    P(dict(
        index=6, stmts=(7384, 7431), epilogue="E6", epilogue_stmts=(7432, 11274),
        gemm=_gemm(4, 12, 32, 48,
                   {"source": "E5's pack of phase 2's and phase 3's accumulators",
                    "requantise": "cvt.rn.satfinite.e4m3x2.f16x2 then mov.b32"},
                   {"source": "E5's second pack stream of the same two accumulators",
                    "weight_bytes": None,
                    "note": "phase 6 reads no weight image at all (phase table: "
                            "weight column `-` for every mma)"},
                   {"source": "the score bias",
                    "byte_offset": 29312, "byte_span": 12288, "stride": "18432 * tid.z",
                    "kind": "per-(m,n) seed pair, 24 tiles x 512 bytes"},
    )))

    P(dict(
        index=7, stmts=(11275, 11322), epilogue="E7", epilogue_stmts=(11323, 11505),
        gemm=_gemm(12, 12, 96, 48,
                   {"source": "E6's pack of the softmax probabilities",
                    "requantise": "cvt.rn.satfinite.e4m3x2.f16x2"},
                   {"source": "E6's movmatrix.trans of phase 5's D",
                    "weight_bytes": None,
                    "note": "48 movmatrix.sync.trans.aligned.m8n8.b16 at s11011-s11058, "
                            "inputs are exactly phase 5's 48 D registers (%r6715..%r6946)"},
                   {"source": "zero"},
        ),
    ))

    P(dict(
        index=8, stmts=(11506, 11521), epilogue="E8", epilogue_stmts=(11522, 11537),
        gemm=_gemm(2, 8, 32, 16,
                   {"source": "shared, E7's pack of phase 7's D at offsets 0/512/1024/1536"},
                   {"source": "weight image", "byte_offset": 41600, "byte_span": 1548},
                   {"source": "phase 7's D plus the per-column table",
                    "byte_offset": 62080, "byte_span": 112,
                    "kind": "per-column f16 pair, 8 words per lane"},
        )))

    P(dict(
        index=9, stmts=(11538, 11553), epilogue="E9", epilogue_stmts=(11554, 12396),
        gemm=_gemm(2, 8, 32, 16,
                   {"source": "shared, offsets +2048 and +2560"},
                   {"source": "weight image", "byte_offset": 60032, "byte_span": 1548},
                   {"source": "zero"},
    )))

    # ---- the MLP band, phases 10..25 ------------------------------------------------
    # Odd phases are the 32x64 projections whose C is zero; even phases are the
    # 64x64 ones whose C is a bias.  Every odd phase's A is the same normalised tile N.
    band = [
        # (index, m_tiles, n_tiles, k, mma, weight_off, c_off, c_kind)
        (10, 4, 8, 64, 16, 62336, 64384, "per-(m,n) seed pair"),
        (11, 2, 8, 32, 16, 64448, None, "zero"),
        (12, 4, 8, 64, 16, 66624, 68672, "per-(m,n) seed pair"),
        (13, 2, 8, 32, 16, 68736, None, "zero"),
        (14, 4, 8, 64, 16, 70784, 72832, "per-(m,n) seed pair"),
        (15, 2, 8, 32, 16, 72896, None, "zero"),
        (16, 4, 8, 64, 16, 74944, 76992, "per-(m,n) seed pair"),
        (17, 2, 8, 32, 16, 77056, None, "zero"),
        (18, 4, 8, 64, 16, 79104, 81152, "per-(m,n) seed pair"),
        (19, 2, 8, 32, 16, 81216, None, "zero"),
        (20, 4, 8, 64, 16, 83264, 85312, "per-(m,n) seed pair"),
        (21, 2, 8, 32, 16, 85376, None, "zero"),
        (22, 4, 8, 64, 16, 87424, 89472, "per-(m,n) seed pair"),
        (23, 2, 8, 32, 16, 89536, None, "zero"),
        (24, 4, 8, 64, 16, 91584, 93632, "per-(m,n) seed pair"),
        (25, 2, 8, 32, 16, 93696, None, "zero"),
    ]
    stmts = {10: (12397, 12412), 11: (12834, 12849), 12: (12918, 12933),
             13: (13242, 13257), 14: (13326, 13341), 15: (13650, 13665),
             16: (13734, 13749), 17: (14058, 14073), 18: (14142, 14157),
             19: (14466, 14481), 20: (14550, 14565), 21: (14874, 14889),
             22: (14958, 14973), 23: (15282, 15297), 24: (15366, 15381),
             25: (15690, 15705)}
    epi = {10: "E10", 11: "E11", 12: "E12", 13: "E13", 14: "E14", 15: "E15",
           16: "E16", 17: "E17", 18: "E18", 19: "E19", 20: "E20", 21: "E21",
           22: "E22", 23: "E23", 24: "E24", 25: "E25"}
    for idx, mt, nt, k, mma, w, c, ckind in band:
        even = idx % 2 == 0
        P(dict(
            index=idx, stmts=stmts[idx], epilogue=epi[idx],
            gemm=_gemm(mt, nt, k, mma,
                       {"source": _NORMT,
                        "note": "every odd phase packs the same 32 f16x2 tile N; "
                                "E11/E13/E15/E17/E19/E21/E23 re-pack the identical "
                                "registers %r17567..%r17598",
                        "requantise": "cvt.rn.satfinite.e4m3x2.f16x2"},
                       {"source": "weight image", "byte_offset": w, "byte_span": 1548,
                        "stride": None},
                       {"source": "bias" if even else "zero",
                        "byte_offset": c, "kind": ckind} if even else
                       {"source": "zero"},
            ),
        ))

    P(dict(
        index=26, stmts=(15981, 15988), epilogue="E26", epilogue_stmts=(15989, 16304),
        gemm=_gemm(4, 4, 64, 8,
                   {"source": "E25's requantised words %r489..%r504, scattered to "
                              "param_0+48 and read back as P26's A",
                    "requantise": "24 groups of 2 cvt + 1 mov.b32 at s15706-s15753"},
                   {"source": "weight image", "byte_offset": 95744, "byte_span": 524},
                   {"source": "bias", "byte_offset": 96768, "byte_span": 16,
                    "kind": "per-(m,n) seed pair"},
        ),
    ))

    return out


# ---------------------------------------------------------------------------
# the epilogues, as an operation list the engine can dispatch on
# ---------------------------------------------------------------------------

def epilogues():
    """name -> the operations it performs, with the offsets that carry them.

    Operation names are the engine's vocabulary: e4m3_requantise, pack_next_A,
    clamped_cubic_activation, add_column_table, rms_norm, softmax, patch_expand,
    plane_scatter, lane_merge, surface_scatter.
    """
    return {
        "E1": {"after": 1, "stmts": (547, 7095), "ops": [
            {"op": "e4m3_requantise", "src": "phase 1's D", "count": 144,
             "dst": "shared, 10x10x4 grid (see e1_patch_grid)"},
            {"op": "patch_expand", "grid": [10, 10], "words_per_cell": 4},
            {"op": "gather", "arg": 24, "index": "clamp_row*(8*ex) + clamp_col*8 + clamp(t,0,7)",
             "stride_bytes": 4, "count": 24},
            {"op": "e4m3_unpack", "count": 96},
            {"op": "rms_norm", "eps": "2^-13", "reduction": "4-lane butterfly over laneid&3",
             "gain_byte_offset": 25088, "gain_span": 112,
             "note": "raw sum of squares + eps, no mean subtraction and no 1/N; "
                     "applied per column"},
            {"op": "pack_next_A", "dst": "shared", "tiles": [0, 512, 1024, 1536, 4096, 4608],
             "base": "tid.z*2048 + laneid*16", "guarded_by": "tid.z*16 < 32"},
            {"op": "lane_merge", "note": "tid.z*2048 splits the two waves' A tiles; "
                                          "each wave reads the other's"},
            {"op": "load_B", "byte_offset": 25216, "stride": "18432 * tid.z"},
        ]},
        "E2": {"after": 2, "ops": [{"op": "load_B", "byte_offset": 25216 + 2048,
                                     "span": 1024, "stride": "18432 * tid.z"}]},
        "E3": {"after": 3, "ops": [{"op": "load_B", "byte_offset": 25216 + 1024,
                                     "span": 1024, "stride": "18432 * tid.z"}]},
        "E4": {"after": 4, "ops": [{"op": "load_B", "byte_offset": 25216 + 3072,
                                     "span": 1024, "stride": "18432 * tid.z"}]},
        "E5": {"after": 5, "stmts": (7213, 7383), "ops": [
            {"op": "load_bias", "byte_offset": 29312, "span": 12288,
             "stride": "18432 * tid.z", "dst": "phase 6's C"},
            {"op": "e4m3_requantise", "src": "phase 2's and phase 3's D", "count": 80,
             "dst": "phase 6's A (80 regs) and B (80 regs)"},
        ]},
        "E6": {"after": 6, "stmts": (7432, 11274), "ops": [
            {"op": "softmax", "form": "cubic exponent, no row maximum, no bias add",
             "scale": 0.017181396484375, "clamp": [-0.55615234375, 0.55615234375],
             "cubic": [0.92724609375, 1.375],
             "note": "expval = f16 from bits ((u<<5) & 0x7FE07FE0); the 8 row sums come "
                     "from two shfl.sync.bfly steps"},
            {"op": "lane_merge", "note": "48 movmatrix.sync.trans.aligned.m8n8.b16 of "
                                          "phase 5's D"},
            {"op": "e4m3_requantise", "src": "the probabilities and the transposed V",
             "count": 144, "dst": "phase 7's A and B"},
        ]},
        "E7": {"after": 7, "ops": [
            {"op": "e4m3_requantise", "count": 32, "dst": "shared"},
            {"op": "lane_merge", "note": "pack D7 to shared, bar.sync, read back as phase 8's A"},
            {"op": "add_column_table", "byte_offset": 62080, "span": 112,
             "dst": "phase 8's C = phase 7's D + the table"},
            {"op": "load_B", "byte_offset": 41600, "span": 1548},
        ]},
        "E8": {"after": 8, "ops": [
            {"op": "load_A", "from": "shared", "offsets": [2048, 2560]},
            {"op": "load_B", "byte_offset": 60032, "span": 1548},
        ]},
        "E9": {"after": 9, "stmts": (11554, 12396), "ops": [
            {"op": "rms_norm", "eps": "2^-13", "reduction": "4-lane butterfly",
             "gain_byte_offset": 62208, "gain_span": 112,
             "note": "produces the normalised tile N, 32 f16x2, re-packed by E11, "
                     "E13, E15, E17, E19, E21 and E23"},
            {"op": "e4m3_requantise", "count": 32, "dst": "phase 10's A"},
            {"op": "load_B", "byte_offset": 62336, "span": 1548},
            {"op": "load_bias", "byte_offset": 64384, "span": 48, "dst": "phase 10's C"},
        ]},
        "E10": {"after": 10, "stmts": (12413, 12833), "ops": [
            {"op": "clamped_cubic_activation",
             "clamp": [-2.0, 2.0],
             "form": "g = 0.5 + y*(0.412109375 - 0.0810546875*|y|); out = x*g",
             "note": "a cubic ramp, not SiLU or GELU; g(+2)=1 and g(-2)=0"},
            {"op": "add_column_table", "byte_offset": 66496, "span": 112,
             "dst": "phase 11's C = phase 9's D + the table"},
            {"op": "e4m3_requantise", "count": 16, "dst": "phase 12's A"},
            {"op": "load_B", "byte_offset": 64448, "span": 1548},
        ]},
        "E25": {"after": 25, "stmts": (15706, 15980), "ops": [
            {"op": "e4m3_requantise", "count": 24, "kept": 16,
             "dst": "phase 26's A"},
            {"op": "plane_scatter", "arg": 48,
             "index": "8*ex*y + 8*x + d, d = t or t|4",
             "bases": ["%r488 + (laneid>>5)", "%r488 + ((g+8)>>3)",
                       "%r488 + ((g+16)>>3)", "%r488 + ((g+24)>>3)"],
             "stores": 16, "guarded": "(y<0)|(y>=ey)|(slice>31)|(x<0)|(x>=ex)"},
            {"op": "load_B", "byte_offset": 95744, "span": 524},
            {"op": "load_bias", "byte_offset": 96768, "span": 16, "dst": "phase 26's C"},
        ]},
        "E26": {"after": 26, "stmts": (15989, 16304), "ops": [
            {"op": "lane_merge", "note": "12 guarded st.shared of the eight phase-26 D "
                                          "registers, then a register-only row max over "
                                          "5 u32 (10 f16) with no warp shuffle"},
            {"op": "softmax", "form": "f32 exp2 with the row maximum subtracted",
             "log2e": 0x3FB8AA3B},
            {"op": "surface_scatter", "arg": 112,
             "stencil": "3x3 colour tex at param_0+88 with the param_0+136 scales, "
                        "plus param_0+96 and param_0+104 at integer (x, y)",
             "combine": "acc*sigmoid(v) + tex96*(1-sigmoid(v)) + tex104 (unless NaN)",
             "channels": 3, "channel_pitch_bytes": 8, "fourth_channel": 0x0000},
        ]},
        "_shared_band": {
            "note": "E11/E13/E15/E17/E19/E21/E23 are 68-statement twins: 32 packs of "
                    "the same normalised tile, four v4 B loads and four per-(m,n) C "
                    "seed words.  E12/E14/E16/E18/E20/E22/E24 are 308-statement twins "
                    "of E10's activation with no table add and no residual.",
            "unverified": "the per-phase B and C byte offsets of phases 11-24 are "
                          "taken from rr_layer_spec.py's phase table (weight and bias "
                          "columns), which reads them off the load chains; they were "
                          "not re-derived from the ld.weak immediates here.",
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"layer": layer(), "patch_grid": e1_patch_grid(),
                      "phases": phases(), "epilogues": epilogues()}, indent=2))