#!/usr/bin/env python3
"""Mirror-cut splice for cuda_dldn_engine_swin_enc3_kernel.

`recipe()` is what kernels/tex/make_ptx.py calls for this layer (see its
`kernel_spec`): it returns the recipe dict that replaces NVIDIA's 936 mma of
48 phases with one `call.uni` into kernels/rr/rrswin_enc3.hip.  Without
D4R_RRSWIN_DEBUG in the environment it returns the cut; with
D4R_RRSWIN_DEBUG=w<n> it returns make_ptx.py's `debug` recipe instead, which
keeps NVIDIA's whole body and stores phase n's accumulator registers -- that is
the oracle kernels/rr/rrswin_enc3.hip is checked against.

Layer facts (kernels/rr/rrswin_enc3_epilogues.md, `python3 kernels/rr/rr_layer_spec.py enc3`):

  * grid (21, 13, 1) x block (32, 1, 4): four waves per block, `%tid.z` in 0..3.
    The waves are not redundant -- each reads its own 24576-byte slice of the
    weight image (s9765 `mul.lo.s32 %r21780, %r12, 12288` -> `mul.wide.u32
    %rd406, %r21780, 2`), so the oracle's per-block debug slots must be
    wave-distinct (see debug_block).
  * parameter block 144 bytes; the weight image (arg040) is 315732 bytes.
  * the shared arena `_ZZ...smem` is 12800 bytes.  The mma region rewrites it
    (E10 stages P11..P14's A operand, E46 stages P47's), and E48 -- which stays
    in PTX -- reads *other waves'* E48(a) rows out of it, so the body may use
    the whole array as scratch: E48 writes every word it later reads.

Cut ABI.  The mma region is statements s6124-s16860, at file lines 9812-41980
of 0016-PREPASS_ENTRYPOINT_NAME.ptx; an mma occupies four physical lines, so
`cut_call` spans 9812..(41980+4).  Both numbers are re-derived from the module
below rather than transcribed, so they cannot drift.

Registers crossing the cut *in*: nothing.  The 24 staged A fragments are read
into registers at lines 9771-9794, immediately before the first mma, and the
body is free to re-read them out of the arena at `smem + 16*laneid + 512*i`
(the arena still holds them: its next writer is E10, s10491).  The B operands
are loaded from the weight image inside the region and the C seeds are either
zero or a previous phase's accumulator, so both are recomputable.  That keeps
the parameter list at the three pointers every layer shares.

Registers crossing the cut *out*: eight, four of them phase 48's last two
accumulator pairs (%r23106/%r23107 and %r23116/%r23117, the D registers of P48's
mma 6 and 7, stored by E48 at s41999/42004/42018/42021).  The other four are
not accumulators and rr_layer_spec.py's "returns" line misses them because it
only looks at mma destinations:

  * %r22182, %r22183 -- the (ex, ey) extents, re-loaded from `param_0+0` inside
    E46 (s16487) and read by E48's bounds tests (s42050, s42052, s42263...).  The
    splice re-loads them instead of the body returning them.
  * %r23356 -- `%tid.z << 4`, built inside E47 (s41894) and read by E48(a)
    (s41993).  The splice re-derives it from `%tid.z`.
  * %r23368 -- the shared arena's base address (s41891), read by E48(a) at
    s41998 and s42017.  The splice re-materialises the symbol.

The four accumulators cannot travel as `call.uni` return values (ZLUDA lowers
those to the function's LLVM return type -- llvm_zluda/ptx/src/pass/llvm/
emit.rs, emit_call: one value, or an get_or_create_struct_type struct for
several, distributed with LLVMBuildExtractValue; there is no .param-space object
behind them, so an N x i32 C++ struct return lowers to an sret pointer, which
is a different LLVM signature and will not link).  They go through shared
memory instead, in the last 512 bytes of the arena, which E48 never touches
(its own window ends at word 397, byte 1588).
"""

import os
import re
from pathlib import Path

FILE = "0016-PREPASS_ENTRYPOINT_NAME.ptx"
ENTRY = "cuda_dldn_engine_swin_enc3_kernel"
SMEM = "_ZZ33cuda_dldn_engine_swin_enc3_kernel33DldnEngineSwinEncParamsStructBaseE4smem"
SMEM_BYTES = 12800

CALLEE = "d4r_swin_enc3"

# ---------------------------------------------------------------------------
# The four accumulators that cross the cut back, and the three shared pointers.
#
# An mma statement is four physical lines (`mma.sync... {d},`, the A, the B and
# the C operand lines, the last ending in `;`).  D2R has REG_EXT_REGS mapping
# register name -> (phase, chain index, half) for every mma destination; the
# entries below are P48's chains 2 and 3, i.e. the last two of its eight mma.
RET_REGS = ["%r23106", "%r23107", "%r23116", "%r23117"]
RET_SLOT = SMEM_BYTES - 4 * len(RET_REGS) * 32   # the last 512 bytes: 4 b32 per lane

EXTERN = f""".extern .func {CALLEE}
(
\t.param .b64 d4r_w, .param .b64 d4r_v, .param .b32 d4r_s
)
;
"""


def module_text():
    """The corpus module make_ptx.py will splice.  D4R_DLSS_PTX_DIR is what
    kernels/build.sh hands both of them (RR_PTX_DIR)."""
    d = Path(os.environ.get("D4R_DLSS_PTX_DIR", Path.home() / ".cache/d4r-rr-corpus"))
    return (d / FILE).read_text()


def statements(lines):
    """(line number, text) of every statement of the entry body, in the same
    linearisation rr_layer_spec.py documents: a line that is neither blank nor
    a bare `//` comment starts a statement, and an `mma.sync` -- this entry's
    only multi-line statement, four physical lines -- is joined into one
    statement numbered at its first line."""
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip() or ln.strip() == "//":
            i += 1
            continue
        if ln.startswith("mma.sync"):
            start = i
            while not lines[i].rstrip().endswith(";"):
                i += 1
            out.append((start + 1, " ".join(p.strip() for p in lines[start:i + 1])))
            i += 1
            continue
        out.append((i + 1, ln))
        i += 1
    return out


RE_MMA = re.compile(r"^mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16 "
                    r"\{([^}]*)\}, \{([^}]*)\}, \{([^}]*)\}, \{([^}]*)\};?$")


def phases(lines):
    """[(first line, last mma line, [(D regs, C regs)])] over the entry, as
    physical lines of the module file.

    Two mma belong to the same phase iff no other statement separates them --
    NVIDIA emits one contiguous mma block per GEMM, so the gaps between those
    blocks are exactly the epilogues."""
    stmts = statements(lines)
    idx = [i for i, (_, t) in enumerate(stmts) if RE_MMA.match(t)]
    out = []
    for j, i in enumerate(idx):
        if j == 0 or i != idx[j - 1] + 1:
            out.append([stmts[i][0], stmts[i][0], []])
        m = RE_MMA.match(stmts[i][1])
        out[-1][1] = stmts[i][0]
        out[-1][2].append((tuple(r.strip() for r in m.group(1).split(",")),
                           tuple(r.strip() for r in m.group(4).split(","))))
    return out


def accumulators(mmas):
    """The D pairs a phase's accumulators end up in: the last mma of every
    accumulator chain, in chain order.

    A banded phase (P10, P15..P45, P47, P48) chains several mma into one output
    tile, so its intermediate D registers are overwritten by the next k-step and
    no native body reproduces them -- only the surviving accumulators are
    comparable.  The chain partition is rr_layer_spec.chains_of's: an mma joins
    the chain that produced one of its C registers, otherwise it starts one."""
    owner = {}
    chains = []
    for dregs, cregs in mmas:
        hit = {owner[r] for r in cregs if r in owner}
        ci = hit.pop() if len(hit) == 1 else (chains.append([]) or len(chains) - 1)
        chains[ci].append(dregs)
        for r in dregs:
            owner[r] = ci
    return [c[-1] for c in chains]


def cut_range(lines):
    """(first mma's line, last mma's line + 4): make_ptx.py splices
    `lines[:before-1] + call + lines[after-1:]`, so the region removed is the
    first mma's line through the last mma's fourth (statement-terminating)
    line, and `lines[after-2]` must end in `;`."""
    ph = phases(lines)
    assert len(ph) == 48, f"enc3: {len(ph)} phases, expected 48"
    return ph[0][0], ph[-1][1] + 4


def entry_lines(lines):
    """Every line of the module up to the entry's closing brace.  The slice
    starts at line 0 so the line numbers phases() derives are the module's own
    physical lines, which is what make_ptx.py's cut_call and debug recipes
    index.  A `}` alone on a line also closes the epilogues' `{ cvt ... ;`
    blocks, so the extent is found by brace depth from the entry's opening
    brace, exactly as make_ptx.py does."""
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f".visible .entry {ENTRY}("))
    b0 = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "{")
    depth = 0
    for i in range(b0, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if i > b0 and depth == 0:
            return lines[:i]
    raise SystemExit("enc3: entry closing brace not found")


def call():
    """The splice: hand the callee the weight image, the parameter block and the
    arena's base address; read the four accumulators back out of the arena and
    re-materialise the four values E48 reads that were computed inside the
    region."""
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
// E48 reads these four inside the cut and cannot recompute them.
mov.b64 %rswinp0, {ENTRY}_param_0;
ld.param.v2.u32 {{%r22182, %r22183}}, [%rswinp0+0];
mov.u32 %rswin2, %tid.z;
shl.b32 %r23356, %rswin2, 4;
mov.u32 %r23368, {SMEM};
}}"""


# ---------------------------------------------------------------------------
# debug oracle: keep NVIDIA's body, store one phase's accumulators in the plane
# arena past everything the kernel writes.

DEBUG_BASE = 4194304         # bytes into arg048 past everything the kernel writes


def debug_base():
    """Byte offset into the plane arena (arg048) where a stage's payload lands.

    NOT the shared 46,000,000: enc3's arena starts at byte 37,683,200 of a
    65,945,600-byte allocation, so only 28,262,400 bytes exist and 46,000,000 is
    past the end of it.  Measured: with the base at 524,288 the payload window
    comes back byte-identical to the plain translated run -- something later in
    the frame rewrites arena bytes 524,288..2,473,984, so a stage store there is
    destroyed before the dump is taken.  4,194,304 is above that window, and
    4 MiB + 64 * 16384 = 5,242,880 stays inside the arena.  The output buffer
    (param+56, arena+1,884,160) is below the base, so the surface comparison is
    unaffected.  kernels/rr/rrswin_enc3.hip's debug_store uses the same default
    and honours D4R_RRSWIN_DEBUG_BASE."""
    return int(os.environ.get("D4R_RRSWIN_DEBUG_BASE", DEBUG_BASE))


DEBUG_SLOT = 8192            # the per-block slot floor (32 lanes x 64 registers x 4 bytes)
GRID_X = 21                  # grid (21, 13, 1)
WAVES = 4                    # block (32, 1, 4)


def debug_slot(nregs):
    """Per-block slot for a stage of NREGS registers: 32 lanes x NREGS x 4 bytes, at least
    DEBUG_SLOT, rounded up to a power of two so the splice's `shl` can form block * slot."""
    want = max(DEBUG_SLOT, 32 * nregs * 4)
    return 1 << (want - 1).bit_length()


def debug_block(regs):
    """PTX that stores REGS (u32 each) to plane arena + DEBUG_BASE + block*slot + lane*len*4.

    This layer runs four waves per block and each reads a different slice of the
    weight image, so the slot index carries `%tid.z` too: block =
    (ctaid.x + 21*ctaid.y)*4 + tid.z, and only the first 64 of those are stored.
    kernels/rr/rrswin_enc3.hip's debug_store computes the same index.
    """
    slot = debug_slot(len(regs))
    base = debug_base()
    stores = "\n".join(
        f"        @%pdbg1 st.global.v4.u32 [%rddbg3+{i * 16}], {{{', '.join(regs[i * 4:i * 4 + 4])}}};"
        for i in range((len(regs) + 3) // 4))
    return f"""// d4r debug stage
{{
.reg .pred %pdbg<2>;
.reg .b32 %rdbg<8>;
.reg .b64 %rddbg<4>;
ld.param.u64 %rddbg1, [{ENTRY}_param_0+48];
mov.u32 %rdbg1, %ctaid.x;
mov.u32 %rdbg2, %ctaid.y;
mad.lo.s32 %rdbg3, %rdbg2, {GRID_X}, %rdbg1;
mov.u32 %rdbg7, %tid.z;
mad.lo.s32 %rdbg3, %rdbg3, {WAVES}, %rdbg7;
setp.lt.u32 %pdbg1, %rdbg3, 64;
shl.b32 %rdbg4, %rdbg3, {slot.bit_length() - 1};
mov.u32 %rdbg5, %laneid;
mad.lo.s32 %rdbg6, %rdbg5, {len(regs) * 4}, %rdbg4;
add.s32 %rdbg7, %rdbg6, {base};
cvt.u64.u32 %rddbg2, %rdbg7;
add.s64 %rddbg3, %rddbg1, %rddbg2;
{stores}
}}
"""


def debug_stages(lines):
    """w<n> -> (line to insert the store after, registers to store), for n = 1..48.

    The insert point is the fourth line of the phase's last mma -- the line the
    `;` is on.  The registers are the phase's surviving accumulators (see
    accumulators()), which for a single-k-step phase is every mma's D pair in
    mma order: m-tile major, n-tile minor, the (row g, row g+8) pair each mma
    writes.  kernels/rr/rrswin_enc3.hip emits the same order."""
    out = {}
    for n, (first, last, mmas) in enumerate(phases(lines), 1):
        out[f"w{n}"] = (last + 4, [r for pair in accumulators(mmas) for r in pair])
    return out


def recipe():
    lines = entry_lines(module_text().split("\n"))
    stage = os.environ.get("D4R_RRSWIN_DEBUG", "")
    if stage:
        stages = debug_stages(lines)
        if stage not in stages:
            raise SystemExit(f"swin_enc3_recipe: no debug stage {stage!r} "
                             f"(have w1..w{max(int(k[1:]) for k in stages)})")
        return dict(file=FILE, debug=stages[stage], debug_block=debug_block)
    first, after = cut_range(lines)
    return dict(file=FILE, cut_call=(first, after), extern=EXTERN, call=call())