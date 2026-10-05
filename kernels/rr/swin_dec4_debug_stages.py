#!/usr/bin/env python3
"""Oracle debug stages for cuda_dldn_engine_swin_dec4_kernel's phase 1.

Each stage is a (file line after which the store block goes, registers in order) tuple for
kernels/tex/make_ptx.py's SWIN_DEBUG_STAGES, so D4R_RRSWIN_DEBUG=p1 builds NVIDIA's own
body with that stage's registers stored into the plane arena, which
kernels/rr/rrswin_dec4.hip's D4R_RRSWIN_DEBUG_STAGE reproduces from its own registers.

Phase 1 is 300 mma in sixty five-step chains: the C operand of each mma is the D pair of
the mma two statements earlier, so a chain accumulates over five k-steps and is seeded
once. A stage therefore keeps the mma whose D pair no other mma of the phase reads as its
C -- the last k-step of each chain -- and lists the sixty pairs in mma statement order, each
as the (row g, row g+8) pair its mma wrote.

The register numbers are read out of the corpus PTX
(~/.cache/d4r-rr-corpus/0019-PREPASS_ENTRYPOINT_NAME.ptx, entry
cuda_dldn_engine_swin_dec4_kernel): statement i of the phase writes {%r(1218+10*i),
%r(1219+10*i)}, verified over all 300 mma of file lines 1707-3803.
"""


def stages():
    """The stage table: name -> (splice line, registers)."""
    # phase 1: 300 mma, 60 output tiles, splice after file line 3803 (the last mma's `;`)
    last = [(1218 + 10 * i, 1219 + 10 * i) for i in range(300) if i % 10 in (8, 9)]
    return {
        "p1": (3803, [f"%r{a}" for pair in last for a in pair]),
        # p1a: phase 1's fifteen A fragments as the mma reads them (four b32 each), the
        # sixty A registers the splice hands the callee
        "p1a": (3803, [f"%r{b + o}" for b in (2130, 2150, 2170, 2190, 2210,
                                             3130, 3150, 3170, 3190, 3210,
                                             4130, 4150, 4170, 4190, 4210)
                       for o in range(4)]),
        # p1s: phase 1's twenty bias seeds, the C operands the splice hands the callee
        "p1s": (3803, [f"%r{3227 + 100 * (j // 2) + 10 * (j % 2)}" for j in range(20)]),
    }
