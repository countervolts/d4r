#!/usr/bin/env python3
"""Splice recipe for cuda_dldn_engine_swin_dec2_kernel (nvngx_dlssd.dll 310.7).

kernels/tex/make_ptx.py's per-layer hook imports this module by path and returns
`recipe()`.  Everything here is self-contained: nothing is imported from
make_ptx.py or rr_layer_spec.py.

The cut replaces the whole 38-phase mma region of the entry -- physical lines
1535-36889 of ~/.cache/d4r-rr-corpus/0021-PREPASS_ENTRYPOINT_NAME.ptx, statement
range s452-s15510, 726 mma -- with one `call.uni d4r_swin_dec2`.  NVIDIA's
prologue (lines 1-1534: the 60 plane-arena loads that stage phase 1's twelve A
fragments, its 24 weight-fragment loads and its twelve bias seeds, all with the
per-tid.z strides 12288 and 192) and its final epilogue (line 36893 to the end:
the plane-arena guards, the shared staging of phase 38's four accumulators, the
row max, the softmax, the 3x3 colour gather, the two integer-coordinate texture
reads at param_0+96/+104 and the surface store to param_0+112) stay in PTX.

Everything between -- all 38 phases and all 37 epilogues -- is the native body's
job.  Only four registers cross back into PTX (rr_layer_spec.py's live_across on
s452-s15510): phase 38's four D halves %r19360 %r19361 %r19370 %r19371.

The callee is handed, in this order:

  * the 48 A registers of phase 1 (twelve .f16.e4m3 fragments of four b32, in
    the order the 144 mma first use them: m-tile 0's four k-slices, then m-tile
    1's, then m-tile 2's),
  * the twelve C bias seeds phase 1 reads from the weight image at
    +49152..+49328 (every other mma's C operand is another mma's D register,
    which the body computes itself, so no other C register is passed),
  * the weight image pointer (%rd32, the `cvta.to.global` of param_0+40),
  * the parameter block pointer (%rd7, param_0 itself),
  * the address of the entry's 9600-byte .shared array.

`call.uni` return values cannot come back to PTX (ZLUDA lowers them to an LLVM
return type with no .param-space object), so the four accumulators cross the cut
through the shared scratch, 16 bytes per lane, as dec1's do.

Phase 1's geometry, read off the mma list and NOT off rr_layer_spec.py's M/N
columns (those count the four k-slices of each fragment as four m-tiles and four
n-tiles, so they print 192x384 for what is a 48x96x128 GEMM):

  * twelve A fragments = 3 m-tiles of 16 rows x 4 k-slices of 32 k,
  * forty-eight B fragments = 12 n-tiles of 8 columns x 4 k-slices,
  * 144 mma = 3 m-tiles x 12 n-tiles x 4 k-steps, in 36 chains of four, so
    M = 48, N = 96, K = 128 and 3*12*4 = 144.

D4R_RRSWIN_DEBUG=<stage> is handled before this module is consulted: make_ptx.py's
swin_debug() finds the stage in kernels/rr/swin_dec2_debug_stages.py (which this
module's A_REGS/C_REGS and phase 1's mma order generate) and keeps NVIDIA's whole
body with that stage's registers stored into the plane arena, which
kernels/rr/rrswin_dec2.hip reproduces with -DD4R_RRSWIN_DEBUG_STAGE=<n>.
"""

MODULE = "0021-PREPASS_ENTRYPOINT_NAME.ptx"
CALLEE = "d4r_swin_dec2"

# the mma region: the first mma's first physical line, and the line after the last
# mma's last physical line
CUT_CALL = (1535, 36893)

# phase 1's A fragments, in the order the 144 mma first use them (the first mma
# reads %r1040..%r1043; s595 reads %r2060..%r2063)
A_REGS = [
    "%r1040", "%r1041", "%r1042", "%r1043",
    "%r1060", "%r1061", "%r1062", "%r1063",
    "%r1080", "%r1081", "%r1082", "%r1083",
    "%r1100", "%r1101", "%r1102", "%r1103",
    "%r1520", "%r1521", "%r1522", "%r1523",
    "%r1540", "%r1541", "%r1542", "%r1543",
    "%r1560", "%r1561", "%r1562", "%r1563",
    "%r1580", "%r1581", "%r1582", "%r1583",
    "%r2000", "%r2001", "%r2002", "%r2003",
    "%r2020", "%r2021", "%r2022", "%r2023",
    "%r2040", "%r2041", "%r2042", "%r2043",
    "%r2060", "%r2061", "%r2062", "%r2063",]

# phase 1's twelve bias seeds, in weight-image offset order (+49152, +49168, ...)
C_REGS = [
    "%r1597", "%r1607", "%r1677", "%r1687",
    "%r1757", "%r1767", "%r1837", "%r1847",
    "%r1917", "%r1927", "%r1997", "%r2007",
]

# what the epilogue reads back out of the deleted region: phase 38's four D halves
RET_REGS = ["%r19360", "%r19361", "%r19370", "%r19371"]
RET_STRIDE = 4 * len(RET_REGS)

SCRATCH = "_ZZ33cuda_dldn_engine_swin_dec2_kernel33DldnEngineSwinEncParamsStructBaseE4smem"

# registers the splice reads: %rd32 is the globalised weight image (s8), %rd7 is
# param_0 itself (s1)
W_REG = "%rd32"
V_REG = "%rd7"


def cut_call():
    """The PTX that replaces the mma region."""
    args = A_REGS + C_REGS
    decl = "\n".join("\t.param .b32 p%d;" % i for i in range(len(args)))
    store = "\n".join("\tst.param.b32 [p%d+0], %s;" % (i, r)
                      for i, r in enumerate(args))
    decl += "\n\t.param .b64 pw;\n\t.param .b64 pv;\n\t.param .b32 ps;"
    store += (f"\n\tst.param.b64 [pw+0], {W_REG};"
              f"\n\tst.param.b64 [pv+0], {V_REG};"
              f"\n\tmov.u32 %rswin0, {SCRATCH};"
              "\n\tst.param.b32 [ps+0], %rswin0;")
    hel = ", ".join(["p%d" % i for i in range(len(args))] + ["pw", "pv", "ps"])
    gets = "\n".join("\tld.shared.b32 %s, [%%rswin1+%d];" % (r, 4 * i)
                     for i, r in enumerate(RET_REGS))
    return f"""{{
.reg .b32 %rswin<2>;
{decl}
{store}
\tcall.uni {CALLEE}, ({hel});
\tmad.lo.s32 %rswin1, %laneid, {RET_STRIDE}, %rswin0;
{gets}
}}"""


def extern():
    """The .extern .func declaration of the callee, matching cut_call()'s arguments."""
    args = ", ".join([".param .b32 d4r_a%d" % i
                      for i in range(len(A_REGS) + len(C_REGS))]
                     + [".param .b64 d4r_w", ".param .b64 d4r_v", ".param .b32 d4r_s"])
    return f".extern .func {CALLEE}\n(\n\t{args}\n)\n;\n"


def recipe():
    """make_ptx.py's recipe for this kernel."""
    return dict(file=MODULE, cut_call=CUT_CALL, call=cut_call(), extern=extern())
