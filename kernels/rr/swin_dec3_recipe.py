#!/usr/bin/env python3
"""Splice recipe for cuda_dldn_engine_swin_dec3_kernel (Ray Reconstruction denoiser, decoder 3).

`kernels/tex/make_ptx.py` imports this module by path and calls `recipe()` whenever the
kernel name matches, so the layer owns its own splice and make_ptx.py stays untouched.

What the recipe produces
  * `cut_call` = (1658, 46566): the file lines of the mma region.  Line 1658 is the
    first `mma.sync` (statement s543) and line 46565 is the `;` of the last one
    (s19381), so everything outside is NVIDIA's texture prologue and the surface
    epilogue of E48.
  * `extern`: the `.extern .func` declaration of the callee `d4r_swin_dec3`.
  * `call`: the block spliced in place of the cut.  It hands the callee the staged
    A fragments, phase 1's sixteen bias seeds, the weight image, the parameter block
    and the base of the shared scratch, then reads the accumulators back.

Everything below was read out of `~/.cache/d4r-rr-corpus/0020-PREPASS_ENTRYPOINT_NAME.ptx`
by `kernels/rr/rr_layer_spec.py dec3` and by hand; each block says where its numbers
come from.

Set D4R_RRSWIN_DEBUG_DEC3=<stage> to build the *oracle* instead of the cut: NVIDIA's
whole body with a store of that stage's registers spliced into the plane arena at
+46,000,000 (the region the kernel never writes), which `rrswin_dec3.hip`'s
-D4R_RRSWIN_DEBUG_STAGE=<n> reproduces from its own registers.
"""

import os

# ---------------------------------------------------------------------------
# The cut.  File lines, not statement indices: make_ptx.py slices `lines` with these.
# rr_layer_spec.py dec3 prints "cut s543-s19381 (file lines 1658-46562)"; the last mma
# is a four-line instruction, so its `;` -- the line the region ends on -- is 46565.
# ---------------------------------------------------------------------------
CUT_FIRST = 1658
CUT_LAST = 46566

KERNEL = "cuda_dldn_engine_swin_dec3_kernel"
PARAM = KERNEL + "_param_0"
# the entry's only shared array (declared at file line 1027, 12800 bytes)
SMEM = "_ZZ33cuda_dldn_engine_swin_dec3_kernel33DldnEngineSwinEncParamsStructBaseE4smem"

# ---------------------------------------------------------------------------
# Live-in registers.
#
# A in (60 b32 = 15 fragments of four): the staged block input the prologue gathered
# with `ld.global.u32` at s67, s109, s127, s145, s163, s200, s237, s255, s273, s291,
# s328, s365, s383, s401, s419 -- fifteen 16-row x 32-k e4m3 fragments.  Listed in the
# order the mma read them (fragment f is %r<base>+0..3, ascending, which for dec3 is
# also the numerical order the tool prints).
# ---------------------------------------------------------------------------
A_BASES = [1556, 1576, 1596, 1616, 1636,      # m-block 0, k-slices 0..4
           2356, 2376, 2396, 2416, 2436,      # m-block 1
           3156, 3176, 3196, 3216, 3236]      # m-block 2
A_REGS = ["%%r%d" % (b + o) for b in A_BASES for o in range(4)]

# C in: the sixteen bias seeds, the only C operands of the region that come from
# outside it (the other 384 are D registers of an earlier mma of phase 1 itself).
# They are the sixteen `ld.global.u32` at file lines 1633-1648, reading
# W + z*256 + ((laneid<<2)&12) + 81920 + 16*i; the callee reads the same words
# itself from the weight image, so they are passed for the record only.
C_REGS = ["%%r%d" % r for r in (2453, 2463, 2553, 2563, 2653, 2663, 2753, 2763,
                                2853, 2863, 2953, 2963, 3053, 3063, 3153, 3163)]

# ---------------------------------------------------------------------------
# Live-out registers: written inside the cut, read by E48 after it.  rr_layer_spec.py
# prints four of them; a corrected destination scan finds six, because an mma writes a
# brace group and `dest_regs` truncates at the first comma inside it:
#   %r26071/%r26072  phase 48's mma #7 D  (row g, row g+8), read at 46581/46600
#   %r26081/%r26082  phase 48's mma #8 D,                      read at 46586/46603
#   %r26210         the shared array's base,                    read at 46580
#   %r26211         %tid.z << 4,                                read at 46575
# %r26212 (the %tid.z they come from) and %r26073/%r26083 are dead after the cut, so
# the splice re-materialises the first two instead of paying for four parameters.
# ---------------------------------------------------------------------------
RET_REGS = ["%r26071", "%r26072", "%r26081", "%r26082"]
RET_STRIDE = 4 * len(RET_REGS)

# ---------------------------------------------------------------------------
# Oracle debug stages: name -> (file line the store block goes after, registers).
#
# A stage is reproducible from its tuple alone: make_ptx.py's swin_debug_block writes
# the registers to arena + 46,000,000 + block*slot + lane*len*4, and rrswin_dec3.hip's
# debug_store writes the same registers of its own phase in the same order.  Only the
# last k-step of each phase's chains carries a value into the epilogue, so those are the
# stages; operand stages (the A fragments and C seeds as the mma reads them) fix the
# fragment order independently of the arithmetic.
# ---------------------------------------------------------------------------
# phase 1: the 48 chain finals (n-tile pair per group, m-tile major, group minor),
# each the (row g, row g+8) pair the mma writes.
PHASE1_FINALS = sum((["%%r%d" % (844 + 100 * g + 80 + 10 * c), "%%r%d" % (845 + 100 * g + 80 + 10 * c)]
                     for g in range(24) for c in range(2)), [])

DEBUG_STAGES = {
    # phase 1's 48 accumulators (the last k-step of every chain)
    "p1": (3334, PHASE1_FINALS),
    # phase 1's A fragments and C seeds as the mma reads them: 60 + 16 registers
    "a": (1657, A_REGS + C_REGS),
}

# 32 lanes x 8 bytes per wave-pair; blockDim is (32, 1, 4), so the debug slot is keyed
# by (ctaid.y, ctaid.x, tid.z) without needing the grid's extent.
DEBUG_WAVES = 4
DEBUG_BLOCKS_X = 8      # ctaid.x < 8
DEBUG_BLOCKS_Y = 2      # ctaid.y < 2  ->  8*2*4 = 64 slots, one per block/wave pair
# dec3's plane arena is the tail of a 65,945,600-byte scratch block that all eleven
# denoiser layers share: dec3's arg048 sits at block offset 35,799,040, so the arena
# is [35,799,040, 65,945,600) and the family default of 46,000,000 lands outside the
# block entirely -- the dump of arg008 would never see the payload.  A payload must
# also clear every other layer's sub-buffer inside the arena, because a stage store
# that overlaps one corrupts the frame.  D4R_CUDA_REPLAY_DUMP_FILTER=swin_ manifests
# for all eleven launches record the block offsets in use: 0, 30146560, 35799040,
# 36387840, 36535040, 37683200, 39567360, 45219840, 60293120.  dec3's own inputs are
# 37683200 (param+24, the skip image) and 39567360 (param+8, the e4m3 features), and
# dec3 itself writes only the first 8*ex*ey = 8*160*92 words = 471,040 bytes of its
# arena (E47's scatter).  Nothing is placed above 60293120, so the tail is the only
# region large enough for 64 slots x 32 KiB.
# D4R_RRSWIN_DEBUG_BASE steers it, exactly as it steers make_ptx.py's
# swin_debug_block; the default is the tail described above.  A native build must be
# given the same value (-DD4R_RRSWIN_DEBUG_BASE=<n> in D4R_TEX_CFLAGS), because that
# side bakes the constant into debug_store().
DEBUG_BASE = int(os.environ.get("D4R_RRSWIN_DEBUG_BASE", 27000000))
DEBUG_ARENA_BYTES = 65945600 - 35799040
DEBUG_MIN_SLOT = 8192
if DEBUG_BASE + DEBUG_BLOCKS_X * DEBUG_BLOCKS_Y * DEBUG_WAVES * 32768 >= DEBUG_ARENA_BYTES:
    raise SystemExit("swin_dec3_recipe: the debug payload does not fit dec3's plane arena")


def debug_slot(nregs):
    """Per-block-slot bytes: 32 lanes x NREGS x 4, at least DEBUG_MIN_SLOT, a power of two."""
    want = max(DEBUG_MIN_SLOT, 32 * nregs * 4)
    return 1 << (want - 1).bit_length()


def debug_block(regs):
    """PTX storing REGS (u32 each) into the plane arena, one slot per (block, wave).

    Same shape as make_ptx.py's swin_debug_block, but keyed by (ctaid.y, ctaid.x, tid.z)
    instead of enc0's flat block index: dec3's block is four waves deep (blockDim.z = 4,
    %tid.z is the slab index the prologue scales its loads by), so a flat index would have
    the four waves of one block overwrite each other.
    """
    n = len(regs)
    slot = debug_slot(n)
    stores = "\n".join(
        f"@%pdbg1 st.global.v4.u32 [%rddbg3+{i * 16}], {{{', '.join(regs[i * 4:i * 4 + 4])}}};"
        for i in range((n + 3) // 4))
    return f"""// d4r debug stage (swin_dec3_recipe.py)
{{
.reg .pred %pdbg<4>;
.reg .b32 %rdbg<10>;
.reg .b64 %rddbg<4>;
ld.param.u64 %rddbg1, [{PARAM}+48];
mov.u32 %rdbg1, %ctaid.x;
mov.u32 %rdbg2, %ctaid.y;
mov.u32 %rdbg3, %tid.z;
setp.lt.u32 %pdbg1, %rdbg1, {DEBUG_BLOCKS_X};
setp.lt.u32 %pdbg2, %rdbg2, {DEBUG_BLOCKS_Y};
setp.lt.u32 %pdbg3, %rdbg3, {DEBUG_WAVES};
and.pred %pdbg1, %pdbg1, %pdbg2;
and.pred %pdbg1, %pdbg1, %pdbg3;
mad.lo.s32 %rdbg4, %rdbg2, {DEBUG_BLOCKS_X}, %rdbg1;
mad.lo.s32 %rdbg4, %rdbg4, {DEBUG_WAVES}, %rdbg3;
shl.b32 %rdbg5, %rdbg4, {slot.bit_length() - 1};
mov.u32 %rdbg6, %laneid;
mad.lo.s32 %rdbg7, %rdbg6, {n * 4}, %rdbg5;
add.s32 %rdbg8, %rdbg7, {DEBUG_BASE};
cvt.u64.u32 %rddbg2, %rdbg8;
add.s64 %rddbg3, %rddbg1, %rddbg2;
{stores}
}}
"""


def extern():
    """The `.extern .func` declaration of the callee: A registers, C registers, the weight
    image, the parameter block and the shared scratch, in that order."""
    args = ", ".join([f".param .b32 d4r_p{i}" for i in range(len(A_REGS) + len(C_REGS))]
                     + [".param .b64 d4r_w", ".param .b64 d4r_v", ".param .b32 d4r_s"])
    return f".extern .func d4r_swin_dec3\n(\n\t{args}\n)\n;\n"


def call():
    """The block spliced in place of the cut.

    The parameter space of `call.uni` is built explicitly (ZLUDA lowers the arguments as
    PTX parameter-space objects, not as an LLVM argument list), the weight image is
    materialised here from the parameter block so the callee gets a global pointer, and
    the four live-out accumulators come back through the shared scratch: ZLUDA turns a
    multi-value C++ return into LLVM's function return type, so there is no callee-side
    .param-space object to read them from.

    %r26210 (the shared base) and %r26211 (%tid.z << 4) are re-materialised rather than
    handed back -- the epilogue's own code at 46481/46480 shows exactly what they are.
    """
    args = A_REGS + C_REGS
    decl = "\n".join(f"\t.param .b32 p{i};" for i in range(len(args)))
    decl += "\n\t.param .b64 pw;\n\t.param .b64 pv;\n\t.param .b32 ps;"
    store = "\n".join(f"\tst.param.b32 [p{i}+0], {r};" for i, r in enumerate(args))
    store += ("\n\tst.param.b64 [pw+0], %rd4;\n\tst.param.b64 [pv+0], %rd2;"
              "\n\tst.param.b32 [ps+0], %r26210;")
    hel = ", ".join([f"p{i}" for i in range(len(args))] + ["pw", "pv", "ps"])
    gets = "\n".join(f"\tld.shared.b32 {r}, [%r681+{4 * i}];"
                     for i, r in enumerate(RET_REGS))
    return f"""{{
{decl}
	mov.b64 %rd2, {PARAM};
	ld.param.u64 %rd3, [%rd2+40];
	cvta.to.global.u64 %rd4, %rd3;
	mov.u32 %r26210, {SMEM};
{store}
	call.uni d4r_swin_dec3, ({hel});
	mov.u32 %r26212, %tid.z;
	shl.b32 %r26211, %r26212, 4;
	mov.u32 %r681, %r26210;
	mad.lo.s32 %r681, %laneid, {RET_STRIDE}, %r681;
{gets}
}}"""


def recipe():
    """The dict make_ptx.py consumes.  D4R_RRSWIN_DEBUG_DEC3 builds the oracle instead."""
    stage = os.environ.get("D4R_RRSWIN_DEBUG_DEC3", "")
    if stage:
        if stage not in DEBUG_STAGES:
            raise SystemExit(f"swin_dec3_recipe: unknown debug stage {stage!r}; "
                             f"have {', '.join(sorted(DEBUG_STAGES))}")
        after, regs = DEBUG_STAGES[stage]
        return dict(file="0020-PREPASS_ENTRYPOINT_NAME.ptx", debug=(after, regs),
                    debug_block=debug_block)
    return dict(file="0020-PREPASS_ENTRYPOINT_NAME.ptx", cut_call=(CUT_FIRST, CUT_LAST),
                extern=extern(), call=call())


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "recipe"
    value = recipe()
    if what == "recipe":
        for k, v in sorted(value.items()):
            print(f"== {k} ==")
            print(v)
    else:
        print(value[what])