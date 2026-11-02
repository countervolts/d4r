#!/usr/bin/env python3
"""Mirror-cut splice for cuda_dldn_engine_swin_dec4_kernel (nvngx_dlssd.dll 310.7).

dec4 is the widest of the eleven Swin layers: its attention/MLP width is 160, so
every register count below is *not* interchangeable with dec3's.  Everything here is
read out of `kernels/rr/rr_layer_spec.py dec4`'s "cut ABI" section and out of the
corpus PTX itself (`~/.cache/d4r-rr-corpus/0019-PREPASS_ENTRYPOINT_NAME.ptx`, entry
`cuda_dldn_engine_swin_dec4_kernel`, which opens at file line 1016).

The cut keeps NVIDIA's prologue (the texture/plane gathers of lines 1-1706) and its
surface epilogue (from line 58080) and replaces the 1606 `mma` of the 62 phases
(file lines 1707-58079) with one `call.uni` into `kernels/rr/rrswin_dec4.hip`.

    phase table      62 phases, 1606 mma, 24447 statements
    cut              s571-s24164, file lines 1707-58076 (`entry` is line 1707)
    parameter block  144 bytes (arg040 = the prepared weight image, arg048 = S)
    launch           no .maxntid in this module; reads %tid.x, %tid.z, %ctaid.x,
                     %ctaid.y, %laneid; block z is eight waves
    A in             60 regs = phase 1's fifteen 16-row A fragments, in the first
                     mma's operand order (four b32 each)
    C in             phase 1's twenty bias seeds, each read as a single u32 and
                     duplicated into both C halves of its mma
    returns          %r34761, %r34762, %r34771, %r34772 -- phase 62's four
                     accumulators, stored to `SM` past the cut (lines 58096, 58101,
                     58113, 58116) and read back there for E62's softmax
    weight image     534804 bytes (phase 62's second C seed at +534800)

The accumulators cross the cut through shared memory rather than `call.uni`'s return
parameters: ZLUDA lowers a return value to the *function's* LLVM return type
(llvm_zluda/ptx/src/pass/llvm/emit.rs, emit_call), so there is no .param-space object
behind it and the callee would read 0x0 in every slot.  `SM`
(`...33DldnEngineSwinEncParamsStructBaseE4smem`, 16384 bytes) is *not* dead after the
cut -- E62 reads it back at line 58152 -- so the handoff uses the top of the array,
past every offset the epilogue touches.  See the offset arithmetic in `swin_cut_call`.

`recipe()` is self-contained: it imports nothing from make_ptx.py and re-derives the
small amount of splice text it needs, so kernels/tex/make_ptx.py stays untouched by
the ten layers being written in parallel.
"""


def _a_regs():
    """Phase 1's fifteen A fragments, in the order the first mma lists them.

    VERIFIED against the corpus: 300 mma, 15 distinct four-b32 A fragments, and every
    one of the 60 registers is defined in the prologue before file line 1707 (the cut's
    first line), so the splice can read all of them.  The register numbers come from
    the mma statements themselves, s571-s870 / file lines 1707-3803.
    """
    bases = [2130, 2150, 2170, 2190, 2210,           # %r2130.., %r2150.., ... %r2210..
             3130, 3150, 3170, 3190, 3210,           # %r3130.., ... %r3210..
             4130, 4150, 4170, 4190, 4210]           # %r4130.., ... %r4210..
    return [f"%r{b + o}" for b in bases for o in range(4)]


def _c_regs():
    """Phase 1's twenty bias seeds, in chain-root order.

    Phase 1 is 300 mma in sixty five-step chains (the C operand of each mma is the D
    pair of the mma two statements earlier, so each chain accumulates over the five
    k-steps of one tile and is seeded once).  The sixty roots use twenty distinct seed
    registers, three chains each: `ld.global.u32 %r3227, [%rd203+102400]` and its
    nineteen successors at +16 bytes (file lines 1686-1705), read at
    `W + z*320 + ((laneid<<2)&12) + 102400 + 16*j`.

    Order matters: it is the order the roots appear in the mma stream, which is
    %r3227, %r3237, %r3327, %r3337, ... (each pair 10 apart, each decade 100 apart).
    VERIFIED by walking the chain structure of all 300 mma.
    """
    return [f"%r{3227 + 100 * (j // 2) + 10 * (j % 2)}" for j in range(20)]


A_REGS = _a_regs()
C_REGS = _c_regs()
# The registers the surviving epilogue (file line 58080 on) reads and that the cut
# deletes the definition of.  VERIFIED by walking every instruction from line 58080 to
# the `ret` and listing the registers used but never defined there: six of them,
# three mma D pairs --
#   s24177 `st.shared.u32 [%r998], %r34761` / s24178 with %r34762,
#   s24181 `st.shared.u32 [%r998+16], %r34771` / s24182 with %r34772,
#   s24170 `shl.b32 %r997, %r34888, 4` / s24173 `add.s32 %r998, %r34889, %r34788`.
# Leaving any of them undefined makes ZLUDA's optimizer fold the whole entry to
# `unreachable` (VERIFIED: the run then dies with HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION),
# because PTX registers are `undef` on first use, which is poison once it reaches a
# branch or a memory address.  `rr_layer_spec.py dec4` names the first member of each
# pair (%r34761, %r34771, %r34888, %r34889); the partners come with them.
RET_REGS = ["%r34761", "%r34762", "%r34771", "%r34772", "%r34888", "%r34889"]

# The shared scratch the accumulators cross the cut through.  `SM` is 16384 bytes.
#
# Only two pieces of NVIDIA's code survive the cut, and both are shallow in `SM`:
#
#   * the prologue (file lines 1-1706) touches shared memory exactly zero times -- its
#     only mention of the array is the `.shared` declaration itself, line 1027;
#   * the epilogue (from line 58080) has eleven shared accesses, all of them E62's
#     five-tap staging.  Its deepest is `%r34858 = SM + (%r1001*6 + %r1002*48)*4` at
#     line 58151, where `%r1001 = tid.x & 7` <= 7 and `%r1002 = (z*32 + tid.x) >> 3`
#     <= 32, so the last byte it can reach is (7*6 + 32*48)*4 + 16 = 6328.
#
# Everything between the two -- including the `+14848` immediates of phase 2's A-fragment
# read at line 19189 -- is deleted by the cut and replaced by this call, so those do not
# constrain the choice.  The handoff therefore takes the top 32 lanes * 16 bytes = 512
# bytes of the array, at 16384-512 = 15872, which leaves a 9544-byte margin over the
# epilogue's 6328.
RET_SCRATCH_SYMBOL = "_ZZ33cuda_dldn_engine_swin_dec4_kernel33DldnEngineSwinEncParamsStructBaseE4smem"
# The handoff therefore takes the top 32 lanes * 24 bytes = 768 bytes of the array, at
# 16384-768 = 15616, which leaves a 9288-byte margin over the epilogue's 6328.
RET_STRIDE = 4 * len(RET_REGS)          # 24 bytes per lane
RET_BASE = 16384 - 32 * RET_STRIDE      # 15616: the top of the array

# The two addresses the callee needs besides its registers.  enc0 hands over the weight
# image and the parameter block as .b64; dec4's equivalents are %rd2 (`ld.param.u64 %rd2,
# [param_0+40]`, file line 1037, still live at the cut -- its last read before line 1707
# is line 1471) and %rd8 (`mov.b64 %rd8, cuda_dldn_engine_swin_dec4_kernel_param_0`,
# line 1029, whose only other use is line 1032).
WEIGHT_REG = "%rd2"
PARAM_REG = "%rd8"

# File lines: the cut replaces 1707-58079 inclusive.  `before` is the first mma's opcode
# line (make_ptx.py checks lines[before-1] starts with "mma.sync.aligned.m16n8k32") and
# `after` is one past the last mma's closing line (lines 58076-58079 are the final mma,
# its operands, and its `;`).
CUT_BEFORE = 1707
CUT_AFTER = 58080

CALLEE = "d4r_swin_dec4"

# The label the prologue's slab guard branches to (file line 1049), re-emitted at the end
# of the splice because its definition at line 5591 falls inside the replaced region.
SKIP_LABEL = "$L__BB1_121"

ENTRY = "cuda_dldn_engine_swin_dec4_kernel"
MODULE = "0019-PREPASS_ENTRYPOINT_NAME.ptx"


def cut_call():
    """The PTX that replaces the mma region.

    The callee is handed phase 1's sixty A registers and twenty C seeds, the weight
    image, the parameter block and the address of the shared scratch; it writes the four
    accumulators there and this splice reads them back into the epilogue's registers.
    Mirrors make_ptx.py's swin_cut_call() for enc0, with dec4's register lists.

    The trailing `$L__BB1_121:` is load-bearing.  The prologue guards the slab index at
    line 1049 -- `setp.gt.u32 %p3, %r9, 3` / `@%p3 bra $L__BB1_121` -- to send the four
    idle waves of the eight-wave block past all the work.  Its target was defined at
    file line 5591, *inside* the region this call replaces, so without a replacement
    label the branch dangles and the PTX translator fails with
    `UnknownSymbol("$L__BB1_121")`.  Re-emitting it here lands those waves just before
    the epilogue's opening `bar.sync 0` (line 58082), which is where they must arrive:
    they take no part in the call, and the only barriers the working waves execute from
    here on are that one and the one at line 58122, both of which they reach.
    """
    args = A_REGS + C_REGS
    decl = "\n".join(f"\t.param .b32 p{i};" for i in range(len(args)))
    store_args = "\n".join(f"\tst.param.b32 [p{i}+0], {r};" for i, r in enumerate(args))
    decl += "\n\t.param .b64 pw;\n\t.param .b64 pv;\n\t.param .b32 ps;"
    store_args += (f"\n\tst.param.b64 [pw+0], {WEIGHT_REG};"
                   f"\n\tst.param.b64 [pv+0], {PARAM_REG};"
                   f"\n\tmov.u32 %rswin0, {RET_SCRATCH_SYMBOL};"
                   "\n\tst.param.b32 [ps+0], %rswin0;")
    hel = ", ".join([f"p{i}" for i in range(len(args))] + ["pw", "pv", "ps"])
    gets = "\n".join(f"\tld.shared.b32 {r}, [%rswin1+{RET_BASE + 4 * i}];"
                     for i, r in enumerate(RET_REGS))
    return f"""{{
.reg .b32 %rswin<2>;
{decl}
{store_args}
\tcall.uni {CALLEE}, ({hel});
\tmad.lo.s32 %rswin1, %laneid, {RET_STRIDE}, %rswin0;
{gets}
}}
{SKIP_LABEL}:"""


def extern():
    """The `.extern .func` declaration, in the same argument order as cut_call()."""
    args = ", ".join([f".param .b32 d4r_a{i}" for i in range(len(A_REGS) + len(C_REGS))]
                     + [".param .b64 d4r_w", ".param .b64 d4r_v", ".param .b32 d4r_s"])
    return f".extern .func {CALLEE}\n(\n\t{args}\n)\n;\n"


def recipe():
    """make_ptx.py's per-layer hook: the kernel spec for cuda_dldn_engine_swin_dec4_kernel."""
    return dict(file=MODULE,
                cut_call=(CUT_BEFORE, CUT_AFTER),
                call=cut_call(),
                extern=extern())
