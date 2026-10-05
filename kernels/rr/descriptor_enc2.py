"""Descriptor for `cuda_dldn_engine_swin_enc2_kernel` (nvngx_dlssd.dll 310.7).

The parameterised WMMA engine's description of enc2: launch geometry, per-phase
GEMM shape, where each operand comes from, and what each epilogue computes.

Provenance of every field
-------------------------
* Geometry (m/n/k tiles, mma count, B-operand register sets, C-operand register
  sets and their producers) is read out of the corpus module
  `~/.cache/d4r-rr-corpus/0015-PREPASS_ENTRYPOINT_NAME.ptx` by
  `kernels/rr/rr_layer_spec.py enc2`, whose linearisation is reproducible with
  `python3 kernels/rr/rr_layer_spec.py enc2`.  The M/N/K columns of that tool's
  phase table are 3x/12x too large for every phase with k > 1 (it sets
  `M = 16*|A|` and `N = 8*|B|`, counting each k32 step as another m/n-tile); the
  values here divide those out by k, which is fixed uniquely by the chain's own
  `m*k = |A|`, `n*k = |B|`, `m*n = chains`, and every row then satisfies
  `mma = m*n*k`.
* Byte offsets are the `ld.weak.global.ca.v4.u32` / `ld.global.v2.u16` /
  `ld.global.u32` immediates read directly from the PTX, and are relative to the
  pointer the epilogue builds (see each phase's `weight_pointer`).
* Epilogue arithmetic is read from the op sequences at the statement ranges
  `kernels/rr/rrswin_enc2_epilogues.md` names (E1-E38).
* Launch geometry and the arena/parameter-block layout are from a measured
  replay manifest, captured with
  `D4R_CUDA_REPLAY_DUMP_DIR=... D4R_CUDA_REPLAY_DUMP_FILTER=swin_enc2`.

Anything not established that way carries `"unverified"` with the reason.
"""

# ---------------------------------------------------------------------------
# Launch geometry, measured
# ---------------------------------------------------------------------------
_LAUNCH = {
    "entry": "cuda_dldn_engine_swin_enc2_kernel",
    "corpus_module": "0015-PREPASS_ENTRYPOINT_NAME.ptx",
    "grid": [41, 24, 1],
    "block": [32, 1, 4],
    "wavefront": 32,
    "warps_per_block": 4,          # one warp per work-item in z
    "lanes_per_warp": 32,
    "shared_bytes": 9600,
    "param_bytes": 144,
    "param_ld_offsets": [8, 40, 80],
    "work_per_warp": "one 8x8 token patch per warp; four waves deep in z",
    # one measurement, tools/wine_nvcuda_bridge.c replay manifest
    "measured_launch_line": "launch 41 24 1 32 1 4 0",
    # token geometry: ex x ey is read from param+0 at run time; at the harness
    # resolution it is 320x184, and the grid is (ex/2 + 1) x (ey/2 + 1)
    "extents_arg": 0,
    "extents_layout": "v2.u32 (ex, ey)",
    "extents_measured": [320, 184],
}

# ---------------------------------------------------------------------------
# Argument windows.  Measured: a single 65,945,600-byte allocation (alloc 0)
# holds three of the buffers, so each parameter is an offset into it; the weight
# image is its own allocation.  `offset` is the byte offset of the window inside
# its allocation.
# ---------------------------------------------------------------------------
_ARGS = {
    "extents":      {"arg": 0,   "kind": "u32x2", "window_bytes": 8},
    "input_image":  {"arg": 8,   "kind": "buffer", "alloc": 0, "offset": 60293120,
                     "window_bytes": 5652480, "note": "320x184x4 f16, the block input patch source"},
    "weights":      {"arg": 40,  "kind": "buffer", "alloc": 1, "offset": 0,
                     "window_bytes": 212256,
                     "note": "prepared e4m3 weight image, read-only; 212256 bytes is the whole image"},
    "plane_arena":  {"arg": 48,  "kind": "buffer", "alloc": 0, "offset": 30146560,
                     "window_bytes": 5652480,
                     "note": "written by E36's scatter, never read back"},
    "output_O":     {"arg": 56,  "kind": "buffer", "alloc": 0, "offset": 35799040,
                     "window_bytes": 2450560,
                     "note": "phase 37's four guarded stores land here"},
    "origins":      {"arg": 80,  "kind": "u16x2", "window_bytes": 4,
                     "note": "(rs126, rs127), the block's x/y patch origin offsets"},
    "texture_in":   {"arg": 88,  "kind": "texture", "shape": "320x184x4", "fmt": 16},
    "surface_0":    {"arg": 112, "kind": "surface", "shape": "320x184x4", "fmt": 16},
    "surface_1":    {"arg": 120, "kind": "surface", "shape": "320x184x4", "fmt": 16},
    "surface_2":    {"arg": 128, "kind": "surface", "shape": "160x92x4", "fmt": 16},
}

# ---------------------------------------------------------------------------
# Operand vocabulary shared by the phases
# ---------------------------------------------------------------------------
# The pointer each phase adds its byte offsets to.  Phases 1-7 index a per-plane
# weight slab; phases 9-36 carry no %tid.z term; phase 37 uses a different
# stride.  The inconsistency is real and is reproduced verbatim (epilogues.md U5).
_WZ = {"expr": "W + 21504*tid.z + 16*laneid", "phases": "1-7"}
_W = {"expr": "W + 16*laneid", "phases": "9-36, 38"}
_W37 = {"expr": "W + 12288*tid.z + 16*laneid", "phases": "37"}

# a phase's B operand comes from consecutive 512-byte tiles at the given base
_WTILE = 512


def _band(w_base, bias_base, a_from, act_from):
    """One (odd, even) branch of the MLP band, phases 13/14 .. 35/36."""
    return {"odd": w_base, "bias": bias_base, "a": a_from, "act": act_from}


# The band is twelve repetitions of the same pair, hanging off one shared
# normalised tile.  `odd` is the 48x96x32 banded GEMM, `even` the 16x96x32 one
# whose C is the running residual.
_BAND_BASES = [(86592, 89664), (92992, 96064), (99200, 102272),
               (105408, 108480), (111616, 114688), (117824, 120896),
               (124032, 127104), (130240, 133312), (136448, 139520),
               (142656, 145728), (148864, 151936), (155072, 158144)]
_BAND_EVEN_B = [89728, 96128, 102336, 108544, 114752, 120960,
                127168, 133376, 139584, 145792, 152000, 158208]
_RESIDUAL_TABLE = 92800        # E13(b): added to phase 12's D to seed phase 14's C


def phases():
    """The 38 phases in order.  Each dict is one mma chain plus its epilogue."""
    out = []
    add = out.append

    # ---- phases 1-6: the 96x96 -> 96x32 residual projection ----------------
    # A is the entry's staged input tile: three K=32 slices of the 96x96 patch,
    # RMS-normalised with a per-column gain and requantised to e4m3 by the
    # prologue, staged at smem + 512*(3*slice + m-tile) + 16*laneid.
    slices = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    w_base = {1: 192, 2: 3264, 3: 1216, 4: 4288, 5: 2240, 6: 5312}
    c_from = {1: None, 2: None, 3: 1, 4: 2, 5: 3, 6: 4}
    for p in range(1, 7):
        add({
            "id": p,
            "gemm": {
                "m_tiles": 6, "n_tiles": 4, "k": 1,
                "M": 96, "N": 32, "K": 32, "mma": 24,
                "A": {
                    "source": "staged_input_tile",
                    "slice": slices[p],
                    "shared_addr": "smem + 512*(3*slice + m_tile) + 16*laneid",
                    "dtype": "e4m3",
                    "note": "18 A quads total, six m-tiles x three K=32 slices; "
                            "the prologue writes them, requantised from the "
                            "RMS-normalised f16 patch",
                },
                "B": {
                    "source": "weights",
                    "weight_pointer": _WZ,
                    "base": w_base[p],
                    "stride": _WTILE,
                    "tiles": 4,
                    "dtype": "e4m3",
                },
                "C": ({"source": "zero", "value": 0} if c_from[p] is None else
                      {"source": "phase_D", "phase": c_from[p],
                       "note": "the phase two before that shares this A fragment"}),
            },
            "k_tiling": {"k": 32, "wmma_steps": 2, "step_k": 16,
                         "f16_round_per_k32": True},
            "epilogue": {
                "ops": ["load_B_tiles"],
                "for_phase": p + 1 if p < 6 else 7,
                "offsets": {"next_B_base": {1: 3264, 2: 1216, 3: 4288,
                                            4: 2240, 5: 5312}.get(p)},
                "note": "E%d: six weight loads for the next phase's B, nothing else"
                        % p,
            },
        })

    # ---- phase 7: the 64x96 attention score ------------------------------
    add({
        "id": 7,
        "gemm": {
            "m_tiles": 4, "n_tiles": 12, "k": 1,
            "M": 64, "N": 96, "K": 32, "mma": 48,
            "A": {
                "source": "phase_D_requantised",
                "phase": 5,
                "m_tiles_used": [0, 1, 2, 3],
                "role": "queries: phase 5's D read as a 64-row tile",
                "dtype": "e4m3",
                "note": "packed by E6(b) from phase 5's D; A frag 0 is bit-identical "
                        "to B frag 0 (s4853 vs s4925) because both are phase 5's D",
            },
            "B": {
                "source": "phase_D_requantised",
                "phase": 5,
                "n_tiles_used": list(range(12)),
                "role": "keys: the same 96x32 tile read as rows",
                "dtype": "e4m3",
                "note": "the score multiplies one projection by its own transpose",
            },
            "C": {
                "source": "weight_bias_table",
                "weight_pointer": _WZ,
                "base": 6336,
                "bytes": 12288,
                "layout": "512*(4*(n>>1) + m) + 16*laneid + 8*(n&1), "
                          "one f16x2 per score element, 48 distinct pairs",
                "dtype": "f16",
                "note": "96 C words per lane; an opaque relative-position or "
                        "learned score bias (epilogues.md U2)",
            },
        },
        "k_tiling": {"k": 32, "wmma_steps": 2, "step_k": 16,
                     "f16_round_per_k32": True},
        "epilogue": {
            "ops": ["softmax_no_max", "transpose_movmatrix", "requantise_pack"],
            "softmax": {
                "kind": "cubic_exp",
                "elements": 96,
                "scale": 0.017181396484375,      # f16 0x2466, from 0f3C8CCB50
                "clamp": [-0.55615234375, 0.55615234375],   # 0dBFE1CC0 / 0d3FE1CC0
                "cubic": {"a": 0.92724609375, "b": 1.375},  # 0f3F6D6000, 0f3FB00000
                "extraction": "E = bits(u)[10:5], F = bits(u)[4:0]; "
                              "expval = f16 from bits (E<<10) | (F<<5)",
                "row_sum": {
                    "in_lane_tree": "A = ((e0+e2)+(e4+e6)) + (e8+e10); "
                                    "B = ((e1+e3)+(e5+e7)) + (e9+e11); sum = A + B",
                    "butterfly": [1, 2],
                    "fold": "add.f16 of the two halves, then broadcast to both",
                    "note": "f16 addition is not associative, so the tree order is "
                            "part of the result; there is no row maximum anywhere",
                },
                "reciprocal": "rcp.approx.ftz.f32 on f32(f16 sum), rounded to f16",
            "transpose": {"op": "movmatrix.sync.trans.aligned.m8n8.b16",
                          "count": 48, "source_phase": 6,
                          "note": "one per phase 6 D register; 48 x 64 = the whole "
                                  "96x32 tile"},
            "requantise_pack": {"ops": ["e4m3_requantise", "pack"],
                                "target_phase": 8},
            "for_phase": 8,
        },
    })

    # ---- phase 8: the attention output projection -------------------------
    add({
        "id": 8,
        "gemm": {
            "m_tiles": 4, "n_tiles": 4, "k": 3,
            "M": 64, "N": 32, "K": 96, "mma": 48,
            "A": {"source": "phase_D_requantised", "phase": 7,
                  "role": "the 64x96 softmax probabilities",
                  "quads": 12, "dtype": "e4m3"},
            "B": {"source": "phase_D_transposed_requantised", "phase": 6,
                  "role": "movmatrix transposes of phase 6's 96x32 D, i.e. 32x96",
                  "pairs": 12, "dtype": "e4m3"},
            "C": {"source": "zero", "value": 0, "note": "%r8606, the region's "
                  "zero accumulator start; four chains of three k-steps"},
        },
        "k_tiling": {"k": 96, "wmma_steps": 6, "step_k": 16,
                     "k32_chain_length": 3, "f16_round_per_k32": True},
        "epilogue": {
            "ops": ["requantise", "shared_stage", "load_B_tiles",
                    "column_table_add", "residual_from_input"],
            "shared_stage": {
                "addr": "smem + 2048*tid.z + 16*laneid + {0, 512, 1024, 1536}",
                "stores": 4, "bytes_each": 16, "per_lane": 64,
                "content": "32 of phase 8's 48 D registers, packed two per word",
                "unverified": "E8's slot layout is self-inconsistent: the writes "
                              "cover slots {0..15} over the four z-planes while the "
                              "reads are at 512*tid.z + {0,2048,4096,6144}. It is "
                              "only consistent if blockDim.z = 4 and the four "
                              "z-planes write the tile cooperatively "
                              "(epilogues.md U4)",
            },
            "column_table_add": {
                "weight_pointer": _W,
                "base": 86208, "bytes": 192,
                "layout": "W + ((laneid<<2)&12) + {86208, 86224, ..., 86384}",
                "dtype": "f16", "count": 96,
                "note": "one value per column of phase 9's N axis",
            },
            "residual_from_input": {
                "regs": ["%r2798", "%r2800", "%r2799", "%r2801", "%r2806",
                         "%r2808", "%r2807", "%r2809", "%r2814", "%r2816",
                         "%r2815", "%r2817"],
                "note": "the residual added into phase 9's C is the prologue's "
                        "cvt.rn.f16x2.e4m3x2 of the quantised input patch, not "
                        "phase 8's D (epilogues.md U9). The prologue's addressing "
                        "for these is not reproduced here.",
                "unverified": "the prologue computes %r17771/%r17794 through a "
                              "five-way select on laneid, the z-plane bias and "
                              "clamped (x, y); re-deriving it needs the E-prologue "
                              "index arithmetic, which is outside the mma region",
            },
            "for_phase": 9,
        },
    })

    # ---- phases 9-12: four 16x96 projections on phase 8's D ---------------
    w9 = {9: 18624, 10: 40128, 11: 61632, 12: 83136}
    for p in range(9, 13):
        add({
            "id": p,
            "gemm": {
                "m_tiles": 1, "n_tiles": 12, "k": 1,
                "M": 16, "N": 96, "K": 32, "mma": 12,
                "A": {"source": "shared", "from_phase": 8,
                      "addr": "smem + 512*tid.z + 16*laneid + "
                              + str(2048 * (p - 9)),
                      "quads": 1, "dtype": "e4m3",
                      "note": "the e4m3 pack of one m-tile of phase 8's D"},
                "B": {"source": "weights", "weight_pointer": _W,
                      "base": w9[p], "stride": _WTILE, "tiles": 6,
                      "dtype": "e4m3"},
                "C": ({"source": "column_table_plus_input_patch",
                       "table_base": 86208, "input_regs": "%r2798..%r2821",
                       "pairs": 24, "note": "E8(d): a per-column f16 table added "
                       "to the prologue's quantised input patch"}
                      if p == 9 else
                      {"source": "phase_D", "phase": p - 1,
                       "note": "the C chain runs 9 -> 10 -> 11 -> 12"}),
            },
            "k_tiling": {"k": 32, "wmma_steps": 2, "step_k": 16,
                         "f16_round_per_k32": True},
            "epilogue": {
                "ops": (["shared_stage", "load_B_tiles", "column_table_add",
                         "residual_from_input"] if p == 9 else
                        ["shared_read", "load_B_tiles"]),
                "load_B_tiles": {"base": w9[p + 1] if p < 12 else None,
                                 "stride": _WTILE, "tiles": 6},
                "for_phase": p + 1 if p < 12 else 13,
            },
        })

    # ---- the MLP band: twelve branches off one normalised tile -----------
    # phase 12's D is RMS-normalised once (E12) into a 16x96 f16 tile; all
    # twelve odd phases read that same tile, and the even phases carry the
    # residual D12 -> D14 -> ... -> D36.
    add({
        "id": "norm12",
        "gemm": None,
        "k_tiling": None,
        "epilogue": {
            "ops": ["rms_norm", "gain_multiply", "requantise_pack"],
            "after_phase": 12,
            "rms_norm": {
                "kind": "rms",
                "extent": 96,
                "eps": 0.0001220703125,       # 2^-13, %fd1 = 0d3F20000000000000
                "accumulate": "raw sum of squares, f16 tree then two butterfly "
                              "steps (delta 1, delta 2) over the four lanes with "
                              "equal laneid>>2",
                "inverse": "rsqrt.approx.ftz.f32(f32(f16(sum + eps))), rounded to f16",
                "no_mean": True,
                "note": "no mean subtraction and no 1/N; this is an RMS norm, "
                        "not a mean-subtracting LayerNorm. The prologue's norm "
                        "(s3812 ff.) has the identical shape",
            },
            "gain": {"weight_pointer": _W, "base": 86400, "bytes": 192,
                     "count": 96, "dtype": "f16",
                     "layout": "W + ((laneid<<2)&12) + {86400, 86416, ..., 86576}",
                     "note": "per-column gain, constant across rows"},
            "bias_deferred": "the norm's bias is not added here; it becomes the "
                             "odd phase's C seeds",
            "output": {"shape": [16, 96], "dtype": "f16", "regs": 24,
                       "register_range": "%r15561..%r15584"},
        },
    })

    for i, ((odd_w, bias), even_w) in enumerate(zip(_BAND_BASES, _BAND_EVEN_B)):
        odd, even = 13 + 2 * i, 14 + 2 * i
        add({
            "id": odd,
            "gemm": {
                "m_tiles": 1, "n_tiles": 4, "k": 3,
                "M": 16, "N": 32, "K": 96, "mma": 12,
                "A": {"source": "normalised_tile", "from": "norm12",
                      "quads": 3, "dtype": "e4m3",
                      "note": "the 16x96 normed tile requantised into three "
                              "16x32 k-slices; E12 packs it at s9784, and every "
                              "even epilogue E14, E16, ... repacks the same tile"},
                "B": {"source": "weights", "weight_pointer": _W,
                      "base": odd_w, "stride": _WTILE, "tiles": 6,
                      "dtype": "e4m3",
                      "unverified": "which (n-tile, k-step) each 512-byte tile "
                                    "holds is not derivable from the PTX: the four "
                                    "chains pair the six tiles as {0,2,4} and "
                                    "{3,4,5}, so the image's tile stride is not "
                                    "(n,k)-major (epilogues.md U1). The byte "
                                    "offsets and the six loads are exact."},
                "C": {"source": "weight_bias_table", "weight_pointer": _W,
                      "base": bias, "stride": 16, "words": 4, "dtype": "f16",
                      "layout": "W + ((laneid*4)&12) + {%d, +16, +32, +48}, each "
                                "used as {rX, rX}" % bias,
                      "note": "per-column bias replicated over the tile's 16 rows; "
                              "four chains of three k-steps"},
            },
            "k_tiling": {"k": 96, "wmma_steps": 6, "step_k": 16,
                         "k32_chain_length": 3, "f16_round_per_k32": True},
            "epilogue": {
                "ops": ["clamped_cubic", "requantise_pack"],
                "activation": {
                    "kind": "clamped_cubic",
                    "clamp": [-2.0, 2.0],
                    "formula": "y = clamp(x, -2, 2); g = 0.5 + y*(0.412109375 - "
                               "0.0810546875*|y|); out = x*g",
                    "constants_f32": [0.4121621549129486, 0.0810810774564743,
                                      0.5, 2.0],
                    "constants_f16": [0.412109375, 0.0810546875, 0.5, 2.0],
                    "op_order": "neg, max, min, abs, mul, sub, mul, add, mul",
                    "dtype": "f16x2 throughout",
                    "note": "byte-identical to enc0's E13; g(+2) = 1 so out = x "
                            "for x >= 2, g(-2) = 0 so out = 0 for x <= -2",
                },
                "requantise_pack": {"target_phase": even,
                                    "quads": 1,
                                    "note": "8 cvt + 4 mov.b32 -> one A quad"},
                "for_phase": even,
            },
        })
        add({
            "id": even,
            "gemm": {
                "m_tiles": 1, "n_tiles": 12, "k": 1,
                "M": 16, "N": 96, "K": 32, "mma": 12,
                "A": {"source": "activation_pack", "from_phase": odd,
                      "quads": 1, "dtype": "e4m3"},
                "B": {"source": "weights", "weight_pointer": _W,
                      "base": even_w, "stride": _WTILE, "tiles": 6,
                      "dtype": "e4m3"},
                "C": ({"source": "residual_plus_column_table",
                       "table_weight_pointer": _W,
                       "table_base": _RESIDUAL_TABLE, "table_bytes": 192,
                       "residual_phase": 12, "pairs": 24,
                       "note": "E13(b): phase 12's D plus a per-column f16 table, "
                               "added to all 16 rows"}
                      if even == 14 else
                      {"source": "phase_D", "phase": even - 2,
                       "note": "the residual chain D12 -> D14 -> ... -> D36"}),
            },
            "k_tiling": {"k": 32, "wmma_steps": 2, "step_k": 16,
                         "f16_round_per_k32": True},
            "epilogue": {
                "ops": (["load_B_tiles", "load_C_bias", "requantise_pack_norm"]
                        if even < 36 else
                        ["load_B_tiles", "requantise"]),
                "note": "E%d: phase %d's B loads and C seeds, then 24 packs of the "
                        "shared normed tile into phase %d's three A quads"
                        % (even, even + 1, even + 1),
                "for_phase": even + 1 if even < 36 else 37,
            },
        })

    # ---- the patch merge: phase 37 ---------------------------------------
    add({
        "id": 37,
        "gemm": {
            "m_tiles": 1, "n_tiles": 4, "k": 12,
            "M": 16, "N": 32, "K": 384, "mma": 48,
            "A": {"source": "shared", "from_phase": 36,
                  "quads": 12, "dtype": "e4m3", "regs": 48,
                  "addr": "smem, swizzled",
                  "loads": "48 ld.shared.u32, the first at word "
                           "((laneid<<2) & ~63) | ((laneid<<1) & 24) | "
                           "((laneid>>2)&3), the rest at +4,+8,+12,+16,+20 and a "
                           "floor-divide-by-24 decomposition for the byte plane",
                  "note": "the 12 staged words per lane are phase 36's requantised "
                          "output, one 16x8 n-tile each",
                  "unverified": "how 12 staged words per lane become a 384-deep k "
                                "axis is not determined without the swizzle's "
                                "intent (epilogues.md U8)"},
            "B": {"source": "weights", "weight_pointer": _W37,
                  "base": 161280, "stride": _WTILE, "tiles": 24,
                  "pairs": 48, "dtype": "e4m3",
                  "loop": "inside $L__BB1_244, whose trip count is 1 for "
                          "tid.z <= 3"},
            "C": {"source": "weight_bias_table", "weight_pointer": _W,
                  "base": 210432, "stride": 16, "words": 4, "dtype": "f16",
                  "layout": "W + ((laneid*4)&12) + {%d, +16, +32, +48}" % 210432,
                  "note": "per-column bias seeding four chains of twelve k-steps"},
        },
        "k_tiling": {"k": 384, "wmma_steps": 24, "step_k": 16,
                     "k32_chain_length": 12, "f16_round_per_k32": True},
        "epilogue": {
            "ops": ["requantise", "surface_scatter", "load_B_tiles", "load_C_bias"],
            "requantise": {"source_phase": 37, "words": 12, "target": "stores",
                           "note": "8 cvt + 3 mov.b32 -> %r287 (2 b32) and "
                                   "%rs4534/%rs4535, %rs4540/%rs4541"},
            "surface_scatter": {
                "target": "output_O", "arg": 56,
                "stores": 4,
                "extents": "halved: ex>>1, ey>>1",
                "base": "y = (%r2>>1) + (laneid>>4); x = (%r1>>1) + ((laneid>>2)&3)",
                "plane": "plus tid.z * (ey>>1) * 8*(ex>>1)",
                "guard": "p6 = (y<0)|(y>=(ey>>1))|(laneid>63)|(x<0)|(x>=(ex>>1))",
                "index": "8*(ex>>1)*y + 8*x + (t | 4)",
                "note": "the second pair recomputes y', x' from (laneid>>2)+8 at "
                        "the same base",
            },
            "load_B_tiles": {"base": 210688, "stride": _WTILE, "tiles": 3,
                             "weight_pointer": _W},
            "load_C_bias": {"base": 212224, "stride": 16, "words": 2,
                            "weight_pointer": _W},
            "for_phase": 38,
        },
    })

    # ---- the attention window: phase 38 ----------------------------------
    add({
        "id": 38,
        "gemm": {
            "m_tiles": 1, "n_tiles": 2, "k": 3,
            "M": 16, "N": 16, "K": 96, "mma": 6,
            "A": {"source": "phase_D_requantised", "phase": 36,
                  "quads": 3, "pairs": 6, "dtype": "e4m3",
                  "note": "the A quads are %r183,%r185,%r184,%r186 and the two "
                          "further pairs, i.e. E36's requantised registers read "
                          "directly, not phase 37's output"},
            "B": {"source": "weights", "weight_pointer": _W,
                  "base": 210688, "stride": _WTILE, "tiles": 3,
                  "pairs": 6, "dtype": "e4m3"},
            "C": {"source": "weight_bias_table", "weight_pointer": _W,
                  "base": 212224, "stride": 16, "words": 2, "dtype": "f16",
                  "layout": "W + ((laneid*4)&12) + {212224, 212240}",
                  "note": "two chains of three k-steps"},
        },
        "k_tiling": {"k": 96, "wmma_steps": 6, "step_k": 16,
                     "k32_chain_length": 3, "f16_round_per_k32": True},
        "epilogue": {
            "ops": ["shared_stage", "bounds_gate", "softmax_with_max",
                    "texture_gather", "sigmoid", "scale", "lane_reduce",
                    "surface_scatter"],
            "shared_stage": {
                "addr": "&smem + ((g + 16*tid.z) & 0x40000008 | (laneid>>2)&7)"
                        " * 6 * 4 + t * 4, and +16 for the second row bank",
                "stores": 4, "row_pitch_words": 6,
                "content": "phase 38's four chain-final D pairs",
            },
            "bounds_gate": "p265 = (tid.z>1) | ((x|y) < 0) | (x >= ex) | (y >= ey)",
            "softmax": {
                "kind": "softmax_with_max",
                "window": 5, "dtype": "f16",
                "max": "max.f16x2 twice then max.f16 twice over the low halves of "
                       "five words, broadcast to both halves",
                "exp": "2^((x - max) * log2e) through ex2.approx.ftz.f32",
                "sum": "f32 row sum over the five weights",
                "invert": "div.approx.ftz.f32",
                "note": "a real row maximum, unlike E7",
            },
            "texture_gather": {
                "surface": "texture_in", "arg": 88,
                "reads": 9, "offsets": "(x-1, x, x+1) x (y-1, y, y+1)",
                "combine": "fma.rn.f16x2 with the five softmax weights, 2x2 "
                           "accumulator halves folded with add.f16, then one more "
                           "shared value with fma.rn.f16",
                "clamp": "coordinates selp'd to 1 when out of range",
            },
            "post": {
                "sigmoid": "1/(1 + 2^(-log2e*v)) on two more shared values",
                "difference": "acc - sigmoid*acc",
                "scale": 0.25,
                "lane_reduce": "shfl.sync.down delta 1 then delta 8, f16x2 path "
                               "and scalar path, segmask 31, membermask -1",
            },
            "surface_scatter": {
                "targets": [{"arg": 112, "shape": "320x184x4"},
                            {"arg": 120, "shape": "320x184x4"},
                            {"arg": 128, "shape": "160x92x4"}],
                "op": "sust.b.2d.v4.b16.zero",
                "index": "(x<<3, y) for the first two, ((x<<2) & -8, y>>1) for "
                         "the third",
                "guard": "the third is taken only when (x|y) is odd",
                "note": "the third store's fourth channel is a literal zero",
            },
            "for_phase": None,
        },
    })

    return out


def layer():
    """The launch geometry and argument windows, as one record."""
    return {
        "name": "enc2",
        "entry": _LAUNCH["entry"],
        "corpus_module": _LAUNCH["corpus_module"],
        "grid": _LAUNCH["grid"],
        "block": _LAUNCH["block"],
        "wavefront": _LAUNCH["wavefront"],
        "warps_per_block": _LAUNCH["warps_per_block"],
        "shared_bytes": _LAUNCH["shared_bytes"],
        "param_bytes": _LAUNCH["param_bytes"],
        "args": _ARGS,
        "weight_pointers": {"Wz": _WZ, "W": _W, "W37": _W37},
        "weight_image_bytes": 212256,
        "weight_slab_strides": [21504, 12288],
        "token_geometry": {
            "patch": [8, 8],
            "grid_from_extents": "((ex >> 1) + 1, (ey >> 1) + 1, 1)",
            "origin_arg": 80,
            "note": "the block's x/y patch origin offsets are param+80's u16 pair; "
                    "the patch origin is x0 = 8*ctaid.x - rs126, "
                    "y0 = 8*(ctaid.y + tid.y) - rs127",
        },
        "measured_launch_line": _LAUNCH["measured_launch_line"],
        "phases": phases(),
    }
