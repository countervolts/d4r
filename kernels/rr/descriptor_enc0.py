#!/usr/bin/env python3
"""Descriptor of cuda_dldn_engine_swin_enc0_kernel for the WMMA engine.

Source of every number below: kernels/rr/rrswin_enc0.hip's own comments and call sites (the
weight-image ranges are listed there, and each phase's `gemm(...)` call gives its operand
sources and its tile counts), plus kernels/rr/rrswin_enc0_epilogues.md for the epilogue kinds.
The card's M/N/K columns in that document are known to be wrong wherever K > 32 (its own note),
so the shapes here come from the body's call sites instead.

Two orientations are recorded per phase.  NVIDIA's mma has the *activation* as its A operand
(16-row tiles of the token patch) and the weights as B; the engine uses the weights as its A
operand and the activations as its B (kernels/tex/enc0_tail.hip's convention), so the engine's
m_tiles is NVIDIA's N/16 and its n_tiles is NVIDIA's M/16.

Nothing here is verified against the oracle yet: this is a transcription of the body, and the
engine pilot is what will test it.
"""

# Weight-image byte ranges, from the body's "Weight image layout of the 22 phases" comment.
WEIGHTS = {
    1: (4096, 5120), 2: (5184, 6208), 3: (6336, 7360), 4: (7360, 8384),
    5: (8384, 20672), 6: (None, None), 7: (20672, 21696), 8: (21696, 22720), 9: (22720, 23744),
    10: (23744, 36032), 11: (None, None), 12: (36032, 37056), 13: (37184, 38208),
    14: (38272, 39296), 15: (39360, 40384), 16: (40448, 41472), 17: (41472, 42496),
    18: (42560, 43584), 19: (43584, 44608), 20: (44672, 45696), 21: (45696, 53888),
    22: (54016, 54544),
}
BIAS = {
    1: (5120, 5184), 2: (6208, 6272), 5: (8384, 20672), 10: (23744, 36032),
    13: (38208, 38272), 15: (40384, 40448), 17: (42496, 42560), 19: (44608, 44672),
    21: (53888, 54016), 22: (54528, 54560),
}

# Per phase: (NVIDIA M, NVIDIA N, K, A source, B source, C source).  M and N are the mma's
# extents; the mma count is (M/16)*(N/8)*(K/32), which the self-check verifies per phase.
LAYER_PARAMS = 1032          # enc0's kernel argument block, from rr_layer_spec.py
GRID = (161, 93, 1)          # ctaid.x, ctaid.y, ctaid.z; block 32x1x1
BLOCK = (32, 1, 1)
PLANE_ARG = 48               # arg048, the plane arena the layer writes
WEIGHT_ARG = 40              # arg040, the weight image


def phases():
    """One dict per phase, in execution order."""
    # mma counts are the corpus's per-phase runs (22 runs, 536 mma, measured with
    # kernels/tools/swin_mma.py-style linearisation).  M/N/K are the body's own shapes; where
    # the count alone is ambiguous the body's comment decides.
    return [
        _ph(1, 96, 32, 32, "normed tile as1", "P1", "seeds"),
        _ph(2, 96, 32, 32, "normed tile as2", "P2", "bias"),
        _ph(3, 96, 32, 32, "normed tile as3", "P3", "zero"),
        _ph(4, 96, 32, 32, "normed tile as3", "P4", "zero"),
        _ph(5, 64, 96, 32, "phases 3/4 accumulators", "P5", "the 64x96 score bias table"),
        _ph(6, 96, 64, 32, "softmax result", "PV of the first window", "zero"),
        _ph(7, 64, 32, 32, "as7", "P7", "c7"),
        _ph(8, 96, 32, 32, "normed tile as3", "P8", "zero"),
        _ph(9, 96, 32, 32, "normed tile as3", "P9", "zero"),
        _ph(10, 64, 96, 32, "phases 8/9 accumulators", "P10", "the second 64x96 score bias table"),
        _ph(11, 96, 64, 32, "softmax result", "PV of the second window", "zero"),
        _ph(12, 64, 32, 32, "as12", "P12", "c12"),
        _ph(13, 64, 32, 32, "normed MLP tile asN", "P13", "bias"),
        _ph(14, 64, 32, 32, "E13's packs as14", "P14", "E13's table add c14"),
        _ph(15, 64, 32, 32, "asN", "P15", "bias"),
        _ph(16, 64, 32, 32, "E15's packs as16", "P16", "phase 14's accumulators"),
        _ph(17, 64, 32, 32, "asN", "P17", "bias"),
        _ph(18, 64, 32, 32, "E17's packs as18", "P18", "phase 16's accumulators"),
        _ph(19, 64, 32, 32, "asN", "P19", "bias"),
        _ph(20, 64, 32, 32, "E19's packs as20", "P20", "phase 18's accumulators"),
        _ph(21, 16, 64, 128, "E20's merge m21", "P21", "bias"),
        _ph(22, 64, 16, 32, "E20's packs r20", "P22", "bias"),
    ]


def _ph(index, m, n, k, a_src, b_src, c_src, epilogue=None):
    wlo, whi = WEIGHTS[index]
    blo, bhi = BIAS.get(index, (None, None))
    return dict(
        index=index,
        gemm=dict(
            m=m, n=n, k=k, mma=(m // 16) * (n // 8) * (k // 32),
            nvidia_m_tiles=m // 16, nvidia_n_tiles=n // 16,
            # the engine's convention: weights are A, activations are B
            engine_m_tiles=n // 16 if wlo is not None else None,
            engine_n_tiles=m // 16 if wlo is not None else None,
            a_source=a_src,
            b_source=dict(kind="weight image", byte_offset=wlo, byte_span=None if wlo is None else whi - wlo),
            c_source=dict(kind=c_src, byte_offset=blo,
                          byte_span=None if blo is None else bhi - blo),
            k_tiling="NVIDIA's k32 step = two gfx12 k16 WMMA steps (swin_gemm_tile)",
        ),
        epilogue=epilogue,
    )


def layer():
    return dict(
        kernel="cuda_dldn_engine_swin_enc0_kernel",
        corpus="0013-PREPASS_ENTRYPOINT_NAME.ptx",
        grid=GRID, block=BLOCK, kernarg_bytes=LAYER_PARAMS,
        arg_plane=PLANE_ARG, arg_weights=WEIGHT_ARG,
        weight_layout=("addr(n,k) = base + 512*(n>>4) + 64*(n&7) + 16*((k>>1)&3) "
                       "+ 8*((n>>3)&1) + 4*((k>>3)&1) + 2*((k>>4)&1) + (k&1) "
                       "-- kernels/rr/rrswin_enc0.hip, swin_waddr_enc0 in the engine"),
        notes=[
            "The engine's m_tiles/n_tiles are swapped relative to NVIDIA's, because the engine "
            "uses the prepared weights as its A operand (enc0_tail's convention).",
            "Phases 5/10's score and 6/11's P*V take both operands from accumulators; the "
            "engine still consumes them through the weight-slot path, so their 'B' is a staged "
            "image, not the weight image.",
            "Phases 21 and 22 read E20's merge output (an arena scatter) and phase 21 writes a "
            "surface; those are the epilogues the engine does not have yet.",
        ],
    )


def self_check():
    """Compare this descriptor's mma total with the corpus.

    The per-phase mma counts came from the corpus's contiguous runs; this checks that the
    M/N/K above still reproduce them, so a shape edit that breaks the arithmetic is caught.
    """
    import importlib.util as _i
    import os
    import sys as _s
    _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import rr_layer_spec as R
    path, entry = R.resolve("enc0", None, None)
    lines = open(path).read().split("\n")
    first, last = next((a, b) for (n, a, b) in R.find_entries(lines) if n == entry)
    stmts, _ = R.linearise(lines, first, last)
    runs, cur = [], 0
    for s in stmts:
        if s.is_mma:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    mine = [p["gemm"]["mma"] for p in phases()]
    return dict(corpus_total=sum(runs), descriptor_total=sum(mine),
                corpus_runs=runs, descriptor_runs=mine, agrees=runs == mine)


if __name__ == "__main__":
    import json
    print(json.dumps(dict(layer=layer(), phases=phases(), self_check=self_check()), indent=1))
