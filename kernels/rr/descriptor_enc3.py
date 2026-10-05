"""Descriptor for `cuda_dldn_engine_swin_enc3_kernel` (DLSS 310.7).

Machine-readable geometry for the parameterised WMMA engine: what every one of the
entry's 48 GEMMs consumes and produces, and what each epilogue does in between.
Every number here is either

  * measured at run time (the launch geometry and the parameter-block layout,
    from a replay capture -- see `layer()`), or
  * derived mechanically from the corpus PTX
    (`python3 kernels/rr/rr_layer_spec.py enc3`, which prints the same numbers,
    and `kernels/rr/rrswin_enc3_epilogues.md`, which decodes the epilogues), or
  * explicitly flagged `"unverified"` with the reason.

Nothing here is transcribed by hand from a disassembly: the per-phase GEMM table
in `phases()` is checked field by field against rr_layer_spec's `geom()`,
`chains_of()` and per-mma weight/C byte offsets over the `mma.sync` runs.  Where
the two disagree this file is wrong and the fix belongs here.

Vocabulary used in the phase dicts
----------------------------------
m_tiles   16-row mma tiles the fragment operands span (M = 16 * m_tiles)
n_tiles   8-column nma tiles the fragment operands span (N = 8 * n_tiles)
k32       number of m16n8k32 accumulation steps chained into one output tile;
          the engine splits each into `k32 * 2` gfx12 WMMA 16-deep steps
dense     every (m_tile, n_tile) of the M x N grid is evaluated.  When false the
          phase is banded: `band` says which tiles are, and `gemm_real` gives the
          true result shape (M/N above are then only the extent spanned)
a         where the activation operand comes from and how it reaches e4m3
b         where the weight operand comes from
c         what the accumulator is seeded with before the chain runs

Only the first ten phases have a bit-exact native body today
(kernels/rr/rrswin_enc3.hip); phases 11..48 are specified here but not yet
implemented, which is why that body is a reference for this descriptor and not
the deliverable.
"""

# ---------------------------------------------------------------------------
# The stem's eight phases are a 2 x 4 shift stencil over one 96-row tile; see
# U6 in rrswin_enc3_epilogues.md for what the shift means at model level.
_STEM_W = {1: 256, 2: 4352, 3: 1280, 4: 5376, 5: 2304, 6: 6400, 7: 3328, 8: 7424}

# The MLP block is 16 odd/even pairs.  Each odd phase is a banded 64x64x128 GEMM
# off the *same* normed tile (E14) with its own weights and per-column bias; each
# even phase is a dense 16x128x32 GEMM whose A is the previous odd phase's
# activation and whose C is the previous even phase's accumulator (plus, for the
# first pair only, a column table -- U5).
_ODD_W = {15: 99072, 17: 107584, 19: 115840, 21: 124096, 23: 132352, 25: 140608,
          27: 148864, 29: 157120, 31: 165376, 33: 173632, 35: 181888, 37: 190144,
          39: 198400, 41: 206656, 43: 214912, 45: 223168}
_ODD_BIAS = {15: 103168, 17: 111680, 19: 119936, 21: 128192, 23: 136448,
             25: 144704, 27: 152960, 29: 161216, 31: 169472, 33: 177728,
             35: 185984, 37: 194240, 39: 202496, 41: 210752, 43: 219008,
             45: 227264}
_EVEN_W = {16: 103232, 18: 111744, 20: 120000, 22: 128256, 24: 136512,
           26: 144768, 28: 153024, 30: 161280, 32: 169536, 34: 177792,
           36: 186048, 38: 194304, 40: 202560, 42: 210816, 44: 219072,
           46: 227328}
_PROJ_W = {11: 20736, 12: 45312, 13: 69888, 14: 94464}

# Column tables, all read through the `W + ((laneid<<2)&12) + off` pattern, i.e.
# indexed by column (the four lanes with equal laneid&3 read the same record).
_TBL_RESIDUAL = 98560      # E10, feeds phase 11's C
_TBL_GAIN = 98816          # E14, the RMS norm's per-column gain
_TBL_P16 = 107328          # E15, added to phase 14's D into phase 16's C


def layer():
    """Launch geometry, parameter block and token geometry."""
    return {
        "entry": "cuda_dldn_engine_swin_enc3_kernel",
        "corpus": "0016-PREPASS_ENTRYPOINT_NAME.ptx",
        "grid": (21, 13, 1),
        "block": (32, 1, 4),
        "wave_size": 32,
        "waves_per_block": 4,
        "waves_are_distinct": True,
        "waves_note": (
            "wave z reads its own 24576-byte slice of the weight image "
            "(line 9765 `mul.lo.s32 %r21780, %r12, 12288`), so the wave index "
            "is part of every per-wave offset, not a redundant copy"),
        "shared_arena": {
            "symbol": "_ZZ33cuda_dldn_engine_swin_enc3_kernel"
                      "33DldnEngineSwinEncParamsStructBaseE4smem",
            "bytes": 12800,
            "window_per_wave": 3200,
        },
        "param_block_bytes": 144,
        "params": {
            0: "u32[2] (ex, ey) output extents",
            8: "input plane pointer",
            40: "prepared weight image pointer",
            48: "plane arena pointer (write-only inside this entry)",
            56: "output buffer pointer",
            80: "u16[2] block-origin offsets",
            88: "texture object",
            112: "surface object",
            120: "surface object",
            128: "surface object",
        },
        # Measured from a replay capture of a real launch
        # (D4R_CUDA_REPLAY_DUMP_DIR): the three pointers above are windows into
        # one 65,945,600-byte allocation, which is what the launch dumper writes
        # out under the name `arg008`.
        "measured_allocation": {
            "bytes": 65945600,
            "dumped_as": "arg008",
            "param_8": 35799040,
            "param_48": 37683200,
            "param_56": 39567360,
            "arena_bytes": 28262400,
            "arena_to_output": 1884160,
        },
        "weight_image": {
            "bytes": 315732,
            "distinct_offsets": 1654,
            "prep": (
                "a NAME_prep companion may permute the image once, so the engine "
                "is free to consume the operands in its own order"),
        },
        "tokens": {
            "grid_tokens": (168, 104),
            "grid_note": "ctaid.x<<3 / ctaid.y<<3 minus the u16 pair at param+80",
            "input_texture": "160x92x4 f16 (param+88)",
            "output_surfaces": [
                "160x92x4 f16 (param+112)",
                "160x92x4 f16 (param+120)",
                "80x46x4 f16 (param+128)",
            ],
            "patch": "one wave per 8x8 token patch",
        },
    }


def _gemm(m_tiles, n_tiles, k32, chains, dense, mma, band=None):
    """One GEMM.  m_tiles/n_tiles are the tiles the fragment operands *span*,
    which for a banded phase is more than the result needs -- gemm_real carries
    the true shape in that case, and `mma` is the statement count either way."""
    g = {"m_tiles": m_tiles, "n_tiles": n_tiles, "k32": k32,
         "M": 16 * m_tiles, "N": 8 * n_tiles, "K": 32 * k32,
         "chains": chains, "mma": mma, "dense": dense,
         "wmma_k": 2 * k32}
    if band is not None:
        g["band"] = band
    return g


def _b(base, n_offsets):
    return {"source": "weight_image", "base": base, "fragments": n_offsets,
            "tile_stride": 512, "addr": "W + 16*laneid + base + 512*j"}


def phases():
    """One dict per GEMM, in the entry's own order."""
    out = []

    # ---- phases 1..8: the stem ------------------------------------------
    # Two accumulator chains (P1->P3->P5->P7 and P2->P4->P6->P8) over four
    # shift groups; phase p uses group (p-1)>>2 and chain (p-1)&1, and its m-tile
    # mi reads A fragment 4*mi + group out of the shared arena.  E1..E7 are
    # identical seven-statement epilogues that only prefetch the next B.
    for p in range(1, 9):
        group = (p - 1) >> 2
        chain = (p - 1) & 1
        out.append({
            "n": p,
            "name": f"stem_g{group}_c{chain}",
            "gemm": _gemm(6, 4, 1, 24, True, 24),
            "a": {"source": "shared_arena", "fragments": 6,
                  "addr": f"smem + 16*laneid + 512*(4*mi + {group}), mi = 0..5",
                  "already_e4m3": True},
            "b": _b(_STEM_W[p], 8),
            "b_wave_slice": "w + tid.z*24576",
            "c": ({"kind": "zero"} if p in (1, 2)
                  else {"kind": "accumulator", "from_phase": p - 2}),
            "epilogue": {
                "name": f"E{p}",
                "range": [6148 + 31 * (p - 1), 6154 + 31 * (p - 1)],
                "ops": [f"load phase {p + 1}'s B from W + {(_STEM_W.get(p + 1, 0))}"],
            },
        })

    # ---- phase 9: the score GEMM Q.K^T ----------------------------------
    out.append({
        "n": 9,
        "name": "score",
        "gemm": _gemm(4, 12, 1, 48, True, 48),
        "a": {"source": "accumulator", "from_phase": 7, "roles": "query rows",
              "rows": "16*mi + g + 8*row, mi = 0..3",
              "requant": "e4m3; word 0 packs n-tiles 0 and 2, word 1 packs 1 and 3",
              "note": "P7's accumulators serve as both the queries and the keys"},
        "b": {"source": "accumulator", "from_phase": 7, "roles": "key rows",
              "rows": "16*(nb>>1) + 8*(nb&1) + n, nb = 0..11",
              "requant": "e4m3, same packing as A"},
        "c": {"kind": "bias_table", "base": 8448, "words": 96,
              "addr": "W + 8448 + 512*(4*(n>>1) + m) + 16*laneid + 8*(n&1) + 4*row",
              "note": "24 tiles of 512 bytes; role computable, model name unknown (U1)"},
        "epilogue": {
            "name": "E9",
            "range": [6584, 10426],
            "ops": ["expval cubic-exponent softmax", "no row maximum",
                    "two butterfly steps over lane bits 0 and 1",
                    "rcp.approx.ftz.f32", "movmatrix transpose of phase 8's D"],
            "scale": "0x2466", "clamp": ["0xb873", "0x3873"],
            "exponent": "((u << 5) & 0x7FE07FE0) over the packed pair",
        },
    })

    # ---- phase 10: P.V ---------------------------------------------------
    out.append({
        "n": 10,
        "name": "pv",
        "gemm": _gemm(12, 12, 3, 16, False, 48,
                      band="4 row tiles x 4 column-pair groups; chain c is "
                           "(A_{3*(c>>2)+k}, B_{2*(c&3)+k}) for k = 0,1,2"),
        "gemm_real": {"M": 64, "N": 32, "K": 96},
        "a": {"source": "softmax", "from_phase": 9,
              "requant": "e4m3; k-step ks packs key columns {4ks, 4ks+2} into "
                         "the fragment's first word and {4ks+1, 4ks+3} into its second"},
        "b": {"source": "accumulator", "from_phase": 8, "roles": "transposed V",
              "requant": "e4m3"},
        "c": {"kind": "zero"},
        "epilogue": {
            "name": "E10",
            "range": [10475, 10673],
            "ops": ["e4m3 requantise and pack",
                    "shared staging at +0/+512/+1024/+1536",
                    "read back as phase 11's A",
                    "column table + prologue residual -> phase 11's C"],
            "table": _TBL_RESIDUAL,
            "table_addr": "W + ((laneid<<2)&12) + 98560 + 16*i",
        },
    })

    # ---- phases 11..14: the dense projection, C chained -------------------
    for p in range(11, 15):
        out.append({
            "n": p,
            "name": f"proj{p - 10}",
            "gemm": _gemm(1, 16, 1, 16, True, 16),
            "a": {"source": "shared_arena", "fragments": 1,
                  "addr": f"smem + (tid.z<<9) + 16*laneid + {2048 * (p - 11)}",
                  "already_e4m3": True},
            "b": _b(_PROJ_W[p], 32),
            "c": ({"kind": "table_plus_residual", "table": _TBL_RESIDUAL,
                   "residual": "long-lived prologue vector"} if p == 11
                  else {"kind": "accumulator", "from_phase": p - 1}),
            "epilogue": {
                "name": f"E{9 + (p - 10)}" if p < 14 else "E14",
                "ops": (["read the next shared slot as the next phase's A",
                         "load the next phase's B"] if p < 14
                        else ["row-RMS norm over all 128 columns",
                              "tree-sum in lane", "two butterfly steps",
                              "+ 2^-13", "rsqrt.approx.ftz.f32",
                              "x * inv * gain", "pack -> the odd phases' A"]),
                "gain_table": _TBL_GAIN if p == 14 else None,
                "gain_addr": "W + ((laneid<<2)&12) + 98816 + 16*i" if p == 14 else None,
                "next_bias": 103168 if p == 14 else None,
                "next_w": 99072 if p == 14 else None,
            },
        })

    # ---- phases 15..46: the MLP block, 16 odd/even pairs ------------------
    for i in range(16):
        odd = 15 + 2 * i
        even = 16 + 2 * i
        out.append({
            "n": odd,
            "name": f"mlp_up{odd - 14}",
            "gemm": _gemm(4, 16, 4, 4, False, 16,
                          band="4 row tiles x 2 of the 8 column tiles per 8-B "
                               "group; chain 0 is (A0,B0)(A1,B2)(A2,B4)(A3,B6), "
                               "chain 1 the B1,B3,B5,B7 partner, chains 2/3 the "
                               "same over B8..B15"),
            "gemm_real": {"M": 64, "N": 64, "K": 128},
            "a": {"source": "normed_tile", "from_phase": 14,
                  "note": "all sixteen odd phases read the SAME normed tile E14 "
                          "produced from phase 14's D"},
            "b": _b(_ODD_W[odd], 32),
            "c": {"kind": "column_bias", "base": _ODD_BIAS[odd], "words": 4,
                  "addr": "W + ((laneid<<2)&12) + bias + 16*i",
                  "used": "both halves of each mma's C operand"},
            "epilogue": {
                "name": "E15" if i == 0 else f"E{2 * odd + 1}",
                "ops": ["clamped cubic activation", "pack -> the even phase's A",
                        "load the even phase's B"],
                "activation": {
                    "y": "clamp(x, -2, +2)",
                    "g": "0.5 + y*(0.412109375 - 0.0810546875*|y|)",
                    "out": "x * g",
                },
                "next_table": _TBL_P16 if i == 0 else None,
                "next_table_adds_phase": 14 if i == 0 else None,
                "next_w": _EVEN_W[even],
            },
        })
        out.append({
            "n": even,
            "name": f"mlp_down{even - 15}",
            "gemm": _gemm(1, 16, 1, 16, True, 16),
            "a": {"source": "activation", "from_phase": odd},
            "b": _b(_EVEN_W[even], 32),
            "c": ({"kind": "accumulator_plus_table", "from_phase": 14,
                   "table": _TBL_P16} if even == 16
                  else {"kind": "accumulator", "from_phase": even - 2}),
            "epilogue": {
                "name": "E16" if even == 16 else f"E{even + 2}",
                "ops": ["e4m3 requantise and pack", "load the next odd phase's B",
                        "load the next odd phase's C bias"],
                "next_w": _ODD_W[odd + 2] if even < 46 else None,
                "next_bias": _ODD_BIAS[odd + 2] if even < 46 else None,
            },
        })

    # ---- phase 47: the merge ---------------------------------------------
    out.append({
        "n": 47,
        "name": "merge",
        "gemm": _gemm(16, 64, 16, 4, False, 64,
                      band="chain 0 is (A_j, B_{2j}), chain 1 (A_j, B_{2j+1}), "
                           "chains 2/3 the same over B32..B63, j = 0..15"),
        "gemm_real": {"M": 256, "N": 256, "K": 512},
        "a": {"source": "shared_arena", "fragments": 16,
              "addr": "smem + 16*g + t and +128, at byte offsets {0,1024,...,7168}",
              "already_e4m3": True},
        "b": _b(231424, 128),
        "c": {"kind": "column_bias", "base": 313344, "words": 4,
              "addr": "W + ((laneid<<2)&12) + 313344 + 16*i"},
        "epilogue": {
            "name": "E47",
            "range": [16728, 16852],
            "ops": ["e4m3 requantise and pack", "surface scatter to param+56",
                    "load phase 48's B and C"],
            "surface_stores": 4,
            "surface_index": "(y<0)|(y>=(ey>>1))|(laneid>63)|(x<0)|(x>=(ex>>1))|"
                             "(loop counter > 159); y = (origin_y>>1) + (laneid>>4), "
                             "x = (origin_x>>1) + ((laneid>>2)&3), second base uses g+8",
            "loop": {"steps_by": 128, "while": "< 160",
                     "unverified": "rrswin_enc3_epilogues.md U4: whether the body "
                                   "runs once or twice per tid.z could not be "
                                   "decided from the PTX alone"},
        },
    })

    # ---- phase 48: the tail ----------------------------------------------
    out.append({
        "n": 48,
        "name": "tail",
        "gemm": _gemm(4, 8, 4, 2, False, 8,
                      band="(A_j, B_{2j}) and (A_j, B_{2j+1}) for j = 0..3"),
        "gemm_real": {"M": 64, "N": 32, "K": 128},
        "a": {"source": "phase47_surface_pack", "from_phase": 47},
        "b": _b(313664, 16),
        "c": {"kind": "column_bias", "base": 315712, "words": 2,
              "addr": "W + ((laneid<<2)&12) + 315712 / +315728"},
        "returns": ["%r23106", "%r23107", "%r23116", "%r23117"],
        "returns_note": "the four accumulators the surviving epilogue reads",
        "epilogue": {
            "name": "E48",
            "range": [16861, 17183],
            "ops": ["shared staging of phase 48's D", "bar.sync",
                    "row maximum (unlike E9)",
                    "exp2 via mul.ftz by 0f3FB8AA3B",
                    "1/rowsum via div.approx.ftz.f32",
                    "3x3 texture gather from param+88 with clamped coordinates",
                    "fma.rn.f16x2 weighted sum",
                    "second sigmoid", "(acc - sigmoid(acc)) * 0.25",
                    "two-step cross-lane sum (shfl.down delta 1 then 8, twice)",
                    "three sust.b.2d.v4.b16.zero"],
            "surfaces": [112, 120, 128],
            "third_surface_note": "taken only when %r398 & 1; fourth channel 0",
            "meaning": {"unverified": "rrswin_enc3_epilogues.md U8: the logical "
                                      "meaning of (acc - sigmoid(acc)) * 0.25 "
                                      "over the 3x3 stencil is not derivable "
                                      "from the PTX"},
        },
    })

    return out


def epilogue_index():
    """Statement ranges of every distinct epilogue, for the engine's scheduler.

    The odd/even MLP epilogues repeat, so only the first of each family is listed
    with its own range; `repeats` says how many times the body runs unchanged."""
    return {
        "E1..E7": {"ranges": [[6148, 6154], [6179, 6185], [6210, 6216],
                              [6241, 6247], [6272, 6278], [6303, 6309],
                              [6334, 6340]],
                   "ops": ["load the next phase's B"]},
        "E9": {"range": [6584, 10426], "ops": ["score softmax", "V transpose"]},
        "E10": {"range": [10475, 10673],
                "ops": ["pack", "shared staging", "table + residual"]},
        "E11..E13": {"ranges": [[10690, 10712], [10729, 10751], [10768, 10790]],
                     "ops": ["shared re-read as the next A", "load the next B"]},
        "E14": {"range": [10807, 11638], "ops": ["row-RMS norm", "pack"]},
        "E15": {"range": [11655, 11943],
                "ops": ["clamped cubic", "column table + residual", "pack"]},
        "E16": {"range": [11960, 12035], "repeats": 15,
                "ops": ["pack the normed tile", "load the next B and C"]},
        "E17": {"range": [12052, 12219], "repeats": 16,
                "ops": ["clamped cubic", "pack", "load the next B"]},
        "E46": {"range": [16100, 16663],
                "ops": ["pack", "plane-arena scatter", "shared staging",
                        "load the next B and C"],
                "scatter_unverified": "rrswin_enc3_epilogues.md U4: the store "
                                      "index splits as t + 4*(g&7) + 32*y with "
                                      "y = (tid.z<<1) + (laneid>>5) for the first "
                                      "group and (tid.z<<1) + ((g+8)>>3) for the "
                                      "second, but the store-to-load permutation "
                                      "was not provable from the PTX alone"},
        "E47": {"range": [16728, 16852], "ops": ["pack", "surface scatter"]},
        "E48": {"range": [16861, 17183],
                "ops": ["softmax", "texture gather", "surface stores"]},
    }


def open_questions():
    """The descriptor items that are NOT settled, with the reason."""
    return {
        "shared_arena_tid_z": "rrswin_enc3_epilogues.md U3: the prologue writes at "
                              "smem + (tid.z<<11) + 16*laneid + {0,512,1024,1536} "
                              "and {8192..9728}, while phases 1..8 read at "
                              "smem + 16*laneid + 512*k with NO tid.z term, so one "
                              "of the two index derivations is not reconstructible "
                              "from the PTX alone",
        "e46_scatter": "rrswin_enc3_epilogues.md U4",
        "e47_loop_count": "rrswin_enc3_epilogues.md U4",
        "score_bias_identity": "rrswin_enc3_epilogues.md U1: the 11792-byte table "
                               "at W+8448 is indexed and its role is computable, "
                               "but whether it is a relative position bias, a "
                               "learned score bias or a per-lane constant is not "
                               "decidable from the PTX",
        "column_tables": "rrswin_enc3_epilogues.md U2: W+98560, W+98816 and "
                         "W+107328 are all read through the same column-indexed "
                         "pattern; their roles are computed here, their model "
                         "names are not",
        "prologue_residual": "rrswin_enc3_epilogues.md U7: the long-lived f16 "
                             "vector phase 11's C adds traces back to e4m3 values "
                             "loaded from the plane at param+8; what that plane "
                             "holds is not determinable",
        "mlp_shape": "rrswin_enc3_epilogues.md U5: whether the 16 pairs are one "
                     "wide MLP or a stack of residual blocks is a model-level "
                     "question the PTX cannot answer",
        "stem_shape": "rrswin_enc3_epilogues.md U6: the four shift groups read as "
                      "a 2x2 stencil or a 4-tap 1-D convolution; enc4 has five "
                      "groups and ten phases, so the tap count is per-stage",
        "tail_meaning": "rrswin_enc3_epilogues.md U8",
    }