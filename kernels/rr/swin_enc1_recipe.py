#!/usr/bin/env python3
"""Splice recipe for `cuda_dldn_engine_swin_enc1_kernel` (corpus module 0014).

Self-contained: `kernels/tex/make_ptx.py` imports this module by path when the kernel
name matches and uses what `recipe()` returns.  Everything below was derived from the
corpus PTX with `kernels/rr/rr_layer_spec.py enc1` and the register traces recorded in
`kernels/rr/rrswin_enc1_epilogues.md`:

  * the cut is the mma region, statements s8270-s26139, i.e. file lines 13343-66944
    (1072 `mma.sync` in 32 phases).  `make_ptx.py` requires line 13343 to start an mma
    and line 66944 to end one, so CUT's second entry is 66945.
  * the live-in set is the two staged 96x32 e4m3 A operand sets the prologue built
    (%r11097.. for phases 1/2/8/9, %r11595.. for phases 3/4/10/11), phase 1's C seed
    register (%r15665, the constant zero of `mov.u32 %r15665, 0`), and the 64 packed
    f16x2 registers of the dequantised input tile (%r4678..%r4741) that E6 adds to
    phase 7's per-column table.  The tile is passed in the order E6 consumes it:
    (m-tile 0..3, n-tile 0..7, row half 0..1).
  * the live-out set is phase 32's eight mma D pairs (16 registers).  The two extents
    %r31837/%r31838 come from `ld.param.v2.u32 [param_0+0]` inside the cut, so the
    splice re-reads them instead of carrying them across.

ZLUDA lowers `call.uni`'s return parameters to the callee's LLVM return type and forms
no parameter-space object for them (ptx/src/pass/llvm/emit.rs, `emit_call`), so the
accumulators cross the cut through LDS instead.  This module's entry declares no
`.shared` array at all -- 0014 has none -- so the splice declares `d4r_scratch` itself:
ZLUDA parses a `.shared` declaration in the middle of a body and sizes the workgroup's
LDS from it (verified with d4r_emit).  `kernels/rr/rrswin_enc1.hip` writes the sixteen
accumulators at `base + 64*laneid + 4*i`; the splice reads them back with ld.shared.
"""

ENTRY = "cuda_dldn_engine_swin_enc1_kernel"
FILE = "0014-PREPASS_ENTRYPOINT_NAME.ptx"

# cut: the mma region, 1-based inclusive file lines
CUT = (13343, 66945)

# A operand set 1: phases 1, 2, 8 and 9 (six 16-row fragments x four b32)
AS1 = [
    "%r11097", "%r11098", "%r11099", "%r11100", "%r11137", "%r11138", "%r11139", "%r11140",
    "%r11177", "%r11178", "%r11179", "%r11180", "%r11217", "%r11218", "%r11219", "%r11220",
    "%r11257", "%r11258", "%r11259", "%r11260", "%r11297", "%r11298", "%r11299", "%r11300"
]

# A operand set 2: phases 3, 4, 10 and 11 (the prologue's second, normalised packing)
AS2 = [
    "%r11595", "%r11596", "%r11597", "%r11598", "%r11635", "%r11636", "%r11637", "%r11638",
    "%r11675", "%r11676", "%r11677", "%r11678", "%r11715", "%r11716", "%r11717", "%r11718",
    "%r11755", "%r11756", "%r11757", "%r11758", "%r11795", "%r11796", "%r11797", "%r11798"
]

# phase 1's C seed: the constant-zero register %r15665
CSEED = ["%r15665"]

# the dequantised input tile E6 adds to phase 7's column table, indexed
# (m-tile 0..3, n-tile 0..7, row half 0..1)
RES = [
    "%r4678", "%r4682", "%r4680", "%r4684", "%r4679", "%r4683", "%r4681", "%r4685",
    "%r4686", "%r4690", "%r4688", "%r4692", "%r4687", "%r4691", "%r4689", "%r4693",
    "%r4694", "%r4698", "%r4696", "%r4700", "%r4695", "%r4699", "%r4697", "%r4701",
    "%r4702", "%r4706", "%r4704", "%r4708", "%r4703", "%r4707", "%r4705", "%r4709",
    "%r4710", "%r4714", "%r4712", "%r4716", "%r4711", "%r4715", "%r4713", "%r4717",
    "%r4718", "%r4722", "%r4720", "%r4724", "%r4719", "%r4723", "%r4721", "%r4725",
    "%r4726", "%r4730", "%r4728", "%r4732", "%r4727", "%r4731", "%r4729", "%r4733",
    "%r4734", "%r4738", "%r4736", "%r4740", "%r4735", "%r4739", "%r4737", "%r4741"
]

# live-out: phase 32's eight mma D pairs.  %r31837/%r31838 (the extents) are re-read
# from param_0 by the splice, so only the sixteen accumulators cross the cut.
RET_ACCS = [
    "%r31908", "%r31909", "%r31918", "%r31919", "%r31948", "%r31949", "%r31958", "%r31959",
    "%r31988", "%r31989", "%r31998", "%r31999", "%r32028", "%r32029", "%r32038", "%r32039"
]

NP = len(AS1) + len(AS2) + len(CSEED) + len(RES)
B32 = AS1 + AS2 + CSEED + RES
SCRATCH_WORDS = 2048          # 8 KiB of LDS: the phase 5/6 exchange, then the accumulators
RET_STRIDE = 4 * len(RET_ACCS)


def extern():
    """`.extern .func` for the callee: the b32 live-ins, then the weight image, the
    parameter block and the LDS scratch address."""
    names = [".param .b32 d4r_a{}".format(i) for i in range(NP)]
    names += [".param .b64 d4r_w", ".param .b64 d4r_v", ".param .b32 d4r_s"]
    return ".extern .func d4r_swin_enc1\n(\n\t" + ", ".join(names) + "\n)\n;\n"


def call():
    """The splice that replaces the mma region: hand the live-ins to the native body,
    then read the accumulators back out of LDS."""
    stores = "\n".join("\tst.param.b32 [q%d+0], %s;" % (i, r) for i, r in enumerate(B32))
    hel = ", ".join(["q%d" % i for i in range(NP)] + ["qw", "qv", "qs"])
    gets = "\n".join("\tld.shared.b32 %s, [%%rsw1+%d];" % (RET_ACCS[i], 4 * i)
                     for i in range(len(RET_ACCS)))
    return "\n".join([
        "{",
        ".reg .b32 %rsw<3>;",
        ".reg .b64 %rdw<2>;",
        ".shared .b32 d4r_scratch[%d];" % SCRATCH_WORDS,
        "mov.u32 %rsw0, d4r_scratch;",
        ".param .b32 q<%d>;" % NP,
        stores,
        ".param .b64 qw;",
        "ld.param.u64 %%rdw0, [%s_param_0+40];" % ENTRY,
        "cvta.to.global.u64 %rdw0, %rdw0;",
        "st.param.b64 [qw+0], %rdw0;",
        ".param .b64 qv;",
        "mov.b64 %%rdw1, %s_param_0;" % ENTRY,
        "st.param.b64 [qv+0], %rdw1;",
        ".param .b32 qs;",
        "st.param.b32 [qs+0], %rsw0;",
        "call.uni d4r_swin_enc1, (%s);" % hel,
        "ld.param.v2.u32 {%%r31837, %%r31838}, [%s_param_0+0];" % ENTRY,
        "mad.lo.s32 %%rsw1, %%laneid, %d, %%rsw0;" % RET_STRIDE,
        gets,
        "}",
    ])


def recipe():
    return dict(file=FILE, cut_call=CUT, extern=extern(), call=call())
