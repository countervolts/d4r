#!/usr/bin/env python3
"""Mirror-cut splice for cuda_dldn_engine_swin_enc5_kernel.

`recipe()` is what kernels/tex/make_ptx.py calls for this layer (see its
`kernel_spec`): it returns the recipe dict that replaces NVIDIA's mma region
with one `call.uni` into kernels/rr/rrswin_enc5.hip.

Cut ABI (from `python3 kernels/rr/rr_layer_spec.py enc5`, cut section):

  * the mma region is statements s7554-s20065, 1306 mma in 61 phases, at file
    lines 11883-50752 of 0018-PREPASS_ENTRYPOINT_NAME.ptx.  An mma occupies
    four physical lines, so `cut_call` ends one line *past* the last mma's
    first line -- that is what make_ptx.py's `lines[after - 1:]` splice wants,
    and it is the same convention the built-in enc0 entry uses (44524 + 4).
  * the callee takes no operand registers.  Unlike enc0 the thirty staged A
    fragments are still in shared memory when the cut is taken (the PTX reads
    them back into registers at lines 11835-11864, immediately before the
    first mma, and never writes shared memory again until E61), so the body
    re-reads them from `smem + laneid*16 + 512*i` itself.  That keeps the
    parameter list at the three pointers every layer shares.
  * six registers cross the cut back: phase 61's four accumulators
    (%r29601, %r29602, %r29611, %r29612) and the two values E61 reads that are
    *computed inside* the region -- %r29781 (`%tid.z`) and %r29782 (the shared
    base, line 50649).  rr_layer_spec's "returns" line misses the latter two
    because it only looks for mma destinations; E61 stores %r29601/%r29611 and
    loads %r29782 with them, so all six have to come back.
  * `call.uni` return values cannot carry them (ZLUDA lowers them to the
    function's LLVM return type), so they travel through shared memory, which
    E61's own shared window (at most 2832 bytes in) never touches.  The return
    window is the last 768 bytes of the 16384-byte array.

Debug stages: with D4R_RRSWIN_DEBUG=w<n> (n = 1..12) this module returns
make_ptx.py's `debug` recipe instead, keeping NVIDIA's whole body and storing
phase n's accumulator registers; kernels/rr/rrswin_enc5.hip stores the same
registers under -DD4R_RRSWIN_DEBUG_STAGE=<n>, so the two payload regions at
plane-arena + 46,000,000 compare byte for byte.
"""

import os

FILE = "0018-PREPASS_ENTRYPOINT_NAME.ptx"
ENTRY = "cuda_dldn_engine_swin_enc5_kernel"
SMEM = "_ZZ33cuda_dldn_engine_swin_enc5_kernel33DldnEngineSwinEncParamsStructBaseE4smem"

CUT_FIRST = 11883            # the first mma's line
CUT_AFTER = 50756            # the last mma's line (50752) plus its three continuation lines

CALLEE = "d4r_swin_enc5"
EXTERN = f""".extern .func {CALLEE}
(
\t.param .b64 d4r_w, .param .b64 d4r_v, .param .b32 d4r_s
)
;
"""

# %r29601, %r29602, %r29611, %r29612, %r29781, %r29782 -- in the order the splice reads them
RET_REGS = ["%r29601", "%r29602", "%r29611", "%r29612", "%r29781", "%r29782"]
RET_SLOT = 16384 - 4 * len(RET_REGS) * 32   # the last 768 bytes of the shared array


def call():
    gets = "\n".join(f"\tld.shared.b32 {r}, [%rswin2+{4 * i}];" for i, r in enumerate(RET_REGS))
    return f"""{{
.reg .b32 %rswin<3>;
.reg .b64 %rswinp<2>;
.param .b64 pw;
.param .b64 pv;
.param .b32 ps;
ld.param.u64 %rswinp0, [{ENTRY}_param_0+40];
mov.b64 %rswinp1, {ENTRY}_param_0;
mov.u32 %rswin0, {SMEM};
add.s32 %rswin1, %rswin0, {RET_SLOT};
st.param.b64 [pw+0], %rswinp0;
st.param.b64 [pv+0], %rswinp1;
st.param.b32 [ps+0], %rswin0;
\tcall.uni {CALLEE}, (pw, pv, ps);
mad.lo.s32 %rswin2, %laneid, {4 * len(RET_REGS)}, %rswin1;
{gets}
}}"""


# ---------------------------------------------------------------------------
# debug oracle: keep NVIDIA's body, store one stage's registers in the plane arena

DEBUG_BASE = int(os.environ.get("D4R_RRSWIN_DEBUG_BASE", "4000000"))
                             # bytes into arg048 (this layer's plane arena) the kernel
                             # never writes: past E60's 147,200-byte scatter and inside
                             # the allocation the harness dumps.  Both sides take it from
                             # the environment, so the two payloads land at the same place.
DEBUG_SLOT = 8192            # the per-block slot floor (32 lanes x 64 registers x 4 bytes)


def debug_slot(nregs):
    """Per-block slot for a stage of NREGS registers: 32 lanes x NREGS x 4 bytes, at least
    DEBUG_SLOT, rounded up to a power of two so the splice's `shl` can form block * slot."""
    want = max(DEBUG_SLOT, 32 * nregs * 4)
    return 1 << (want - 1).bit_length()


def debug_block(regs):
    """PTX that stores REGS (u32 each) to plane arena + DEBUG_BASE + block*slot + lane*len*4.

    This layer launches grid (5,3,1) x block (32,1,8), so `block = ctaid.x + 5*ctaid.y` and only
    one %tid.z group fits in a slot: the store is predicated on %tid.z == 0, which is what
    kernels/rr/rrswin_enc5.hip's debug_store does too.
    """
    slot = debug_slot(len(regs))
    stores = "\n".join(
        f"@%pdbg1 st.global.v4.u32 [%rddbg3+{i * 16}], {{{', '.join(regs[i * 4:i * 4 + 4])}}};"
        for i in range((len(regs) + 3) // 4))
    return f"""// d4r debug stage
{{
.reg .pred %pdbg<3>;
.reg .b32 %rdbg<8>;
.reg .b64 %rddbg<4>;
ld.param.u64 %rddbg1, [{ENTRY}_param_0+48];
mov.u32 %rdbg1, %ctaid.x;
mov.u32 %rdbg2, %ctaid.y;
mad.lo.s32 %rdbg3, %rdbg2, 5, %rdbg1;
mov.u32 %rdbg7, %tid.z;
setp.eq.u32 %pdbg2, %rdbg7, 0;
setp.lt.u32 %pdbg1, %rdbg3, 64;
and.pred %pdbg1, %pdbg1, %pdbg2;
shl.b32 %rdbg4, %rdbg3, {slot.bit_length() - 1};
mov.u32 %rdbg5, %laneid;
mad.lo.s32 %rdbg6, %rdbg5, {len(regs) * 4}, %rdbg4;
add.s32 %rdbg7, %rdbg6, {DEBUG_BASE};
cvt.u64.u32 %rddbg2, %rdbg7;
add.s64 %rddbg3, %rddbg1, %rddbg2;
{stores}
}}
"""


def _win_dregs(base):
    """Phase 1-10's forty-eight accumulator registers: m-tile major, n-tile minor, the row pair
    (row g, row g+8) each of the phase's twenty-four mma writes.  The mma of m-tile mi and n-tile
    nt writes base + 40*mi + 10*nt and its second register *one* further on -- the five-apart
    registers the pair does not touch are dead virtuals, and a store of one of those reads back
    whatever physical register the translator had left it aliased to."""
    return [f"%r{base + 40 * mi + 10 * nt + rh}"
            for mi in range(6) for nt in range(4) for rh in (0, 1)]


def _score_dregs():
    """Phase 11's ninety-six accumulator registers: m-tile major, n-tile minor, the row pair.
    Its mma j = 12*mi + nt writes %r7862 + 120*mi + 10*nt and its second register one on."""
    return [f"%r{7862 + 120 * mi + 10 * nt + rh}"
            for mi in range(4) for nt in range(12) for rh in (0, 1)]


def _pv_dregs():
    """Phase 12's thirty-two accumulator registers: the last k-step of each of its sixteen
    chains, m-tile major and n-tile minor.  The chain of (mi, nt) ends at mma
    j = 12*mi + 6*(nt>>1) + 4 + (nt&1), whose D pair is %r11108 + 10*j."""
    return [f"%r{11108 + 10 * (12 * mi + 6 * (nt >> 1) + 4 + (nt & 1)) + rh}"
            for mi in range(4) for nt in range(4) for rh in (0, 1)]


# (line after which to insert, registers to store).  The line is the last mma's line plus four.
STAGES = {
    "w1": (12048, _win_dregs(5204)),
    "w2": (12229, _win_dregs(5453)),
    "w3": (12410, _win_dregs(5702)),
    "w4": (12591, _win_dregs(5951)),
    "w5": (12772, _win_dregs(6200)),
    "w6": (12953, _win_dregs(6449)),
    "w7": (13134, _win_dregs(6698)),
    "w8": (13315, _win_dregs(6947)),
    "w9": (13496, _win_dregs(7196)),
    "w10": (13677, _win_dregs(7445)),
    "w11": (14394, _score_dregs()),
    "w12": (24874, _pv_dregs()),
    # phase 13: its forty C seeds (%r11944 + 10*nt is row g's column pair, the
    # register one on row g+8's) and its forty accumulators (%r11936 + 10*nt)
    "p13c": (25621, [f"%r{11944 + 10 * nt + rh}" for nt in range(20) for rh in (0, 1)]),
    "p13d": (25621, [f"%r{11936 + 10 * nt + rh}" for nt in range(20) for rh in (0, 1)]),
}


def recipe():
    stage = os.environ.get("D4R_RRSWIN_DEBUG", "")
    if stage:
        if stage not in STAGES:
            raise SystemExit(f"swin_enc5_recipe: no debug stage {stage!r} (have {sorted(STAGES)})")
        return dict(file=FILE, debug=STAGES[stage], debug_block=debug_block)
    return dict(file=FILE, cut_call=(CUT_FIRST, CUT_AFTER), extern=EXTERN, call=call())
