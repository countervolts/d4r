#!/usr/bin/env python3
"""The mirror cut for cuda_dldn_engine_swin_enc4_kernel (Ray Reconstruction denoiser, encoder 4).

make_ptx.py's per-layer hook imports this module by path and returns whatever `recipe()` gives,
so the module is self-contained: it duplicates the small generator logic it needs instead of
importing make_ptx.py, and it derives every register list from the corpus PTX rather than trusting
a hand-copied number.  Re-derive with

    python3 kernels/rr/rr_layer_spec.py enc4

and cross-check the printed `cut ABI` block against the constants below.

What the cut is
---------------
The layer is one 62-phase mma region: the ten-phase stem (phases 1-10, two chains of five 96x32x32
GEMMs over the shared-staged block input), the 64x96 score GEMM and its cubic-exponent softmax
(11-12), the sixteen 16x160 attention projections with their row-RMS norm (13-20), forty MLP
phases in twenty banded pairs (21-60) and the patch merge (61-62).  `rr_layer_spec.py enc4` puts the
region at statements s7554-s21344, file lines 11883-52671: 1386 mma in 62 phases.

NVIDIA's texture prologue (the input plane read, the e4m3 requantisation, the 30-fragment
shared-memory staging and the first weight loads) and the surface epilogue (E62: the 3x3 texture
gather and the three sust.b writes) stay in PTX; only the mma region becomes

    call.uni d4r_swin_enc4, (...)

`kernels/rr/rrswin_enc4.hip` is the callee.

The callee's arguments
----------------------
The prologue leaves the cut with

  * 24 A registers -- the six staged A fragments of phase 1, in the first mma's operand order
    (%r11754..%r11757, %r11774..%r11777, %r11794..%r11797, %r11814..%r11817, %r11834..%r11837,
    %r11854..%r11857).  These are shift group 0's fragments.  Groups 1..4 (what phases 3/4, 5/6,
    7/8 and 9/10 read) are *not* among the live-out registers: the callee reads them from the
    shared arena itself, exactly as the spliced PTX did, which is why the arena's address is
    handed over as well.
  * 1 C register -- %r11676, which the prologue sets to 0 at line 11871 (s7547) and which is every
    phase 1/2 mma's C operand.  The later phases' C operands are the previous phase's D registers,
    all inside the cut.
  * arg040, the prepared weight image (%rd2), and the parameter block (%rd9).
  * the address of the shared arena, so the callee can both read the staged A fragments and hand
    its accumulators back.

Why the accumulators come back through shared memory
---------------------------------------------------
Exactly as in enc0 (see the comment block above SWIN_A_REGS in kernels/tex/make_ptx.py): ZLUDA
lowers `call.uni`'s return values to the callee's *LLVM* return type, so there is no .param-space
object behind them and the callee would read 0x0 in every return slot.  enc4 has five live-out
registers

    %r29719  the x extent, ld.param.v2.u32 at line 50892 (param_0+0)
    %r31372  phase 62's D pair
    %r31382  phase 62's D pair
    %r31608  %tid.z
    %r31615  mov.u32 of the shared arena symbol, line 52567

so the callee writes five words per lane at `scratch + 20*laneid + 4*i` and the splice reads them
back with five ld.shared.b32.  5 * 32 lanes * 4 bytes = 640 bytes.  enc4 has exactly one shared
array (16384 bytes, `_ZZ...smem`), and unlike enc0's `input_tensor` it is *not* dead: phases 13-20
and 61 still use it.  The callee therefore writes the five words as its very last act, at arena
offset 0 (shift group 0's staging slots, which nothing reads after phase 10), and the splice reads
them back before E62 issues a single shared store.  The read-back is therefore ordered against
every other user of the arena by the cut itself.
"""

import os
import re

# ---------------------------------------------------------------------------
# The corpus module.  make_ptx.py resolves it by entry name over the whole directory, so `file` is
# only a hint for the human reader; the path is used here to derive the debug stages.
FILE = "0017-PREPASS_ENTRYPOINT_NAME.ptx"
ENTRY = "cuda_dldn_engine_swin_enc4_kernel"

# `cut s7554-s21344 (file lines 11883-52671)`: 11883 is the first mma's first line, 52671 the last
# mma's last line, and make_ptx's `after` is the first line it keeps.
CUT_FIRST = 11883
CUT_AFTER = 52672

# The 24 A registers the prologue leaves live, in the first mma's operand order: six fragments of
# four b32, m-tile major.
A_REGS = [
    "%r11754", "%r11755", "%r11756", "%r11757",
    "%r11774", "%r11775", "%r11776", "%r11777",
    "%r11794", "%r11795", "%r11796", "%r11797",
    "%r11814", "%r11815", "%r11816", "%r11817",
    "%r11834", "%r11835", "%r11836", "%r11837",
    "%r11854", "%r11855", "%r11856", "%r11857",
]

# The one C seed register (%r11676 = 0).
C_REGS = ["%r11676"]

# The five live-out registers, in the order the callee writes them.
RET_REGS = ["%r29719", "%r31372", "%r31382", "%r31608", "%r31615"]
RET_STRIDE = 4 * len(RET_REGS)

# The only shared array in the entry: 16384 bytes, live across the cut (see the module docstring).
SCRATCH = "_ZZ33cuda_dldn_engine_swin_enc4_kernel33DldnEngineSwinEncParamsStructBaseE4smem"

CALLEE = "d4r_swin_enc4"

# ---------------------------------------------------------------------------
# Debug-store geometry, identical to the enc0 splice's (SWIN_DEBUG_BASE / SWIN_DEBUG_SLOT in
# make_ptx.py) so a payload region can be compared the same way: bytes into the plane arena the
# kernel never writes, plus block*slot plus lane*len(regs)*4.
#
# enc4's plane arena is arg008, not enc0's arg048: the weight image is arg040 and there is no
# arg048 among this kernel's resources.  The launch dump writes arg008 as a 65,945,600-byte file
# whose index 0 is `arena - 0x25bc000`, so an arena offset A appears at dump index A + 39,567,360
# and only arena offsets below 26,378,240 are observable at all.
#
# The payload sits well inside the arena and 64 slots deep (BLOCK_LIMIT slots would not fit for the
# widest stage: 160 registers at a 32,768-byte slot).  D4R_RRSWIN_DEBUG_BASE moves it; both sides
# read the same variable, so a sweep is automatically consistent.
PLANE_ARG = 8
DEBUG_BASE = int(os.environ.get("D4R_RRSWIN_DEBUG_BASE", "20000000"))
DEBUG_BLOCKS = 64
DEBUG_SLOT_FLOOR = 8192

# enc4 is one 8x8 token patch per block over an 80x46 token grid, four waves per block (tid.z bounds
# to 0..3), so grid (11, 7, 1) x block (32, 1, 4).  The four waves of a block compute *different*
# results (each reads its own 27648-byte weight slice and its own staging slots), so the wave has to
# be part of the slot index.  The slot is `ctaid.x + 32*ctaid.y + 256*tid.z`: injective over the
# launch's 11x7x4 = 308 waves and, unlike enc0's 64-block scheme, it keeps every wave's registers
# addressable.  Only the first DEBUG_BLOCKS slots are compared, because the arena cannot hold all
# 1024 of them; blocks past the limit are not stored by either side.
SLOT_X = 32
SLOT_Y = 8
SLOT_Z = 4
BLOCK_LIMIT = SLOT_X * SLOT_Y * SLOT_Z


def debug_slot(nregs):
    """Per-block slot for NREGS registers: 32 lanes x NREGS x 4 bytes, at least DEBUG_SLOT_FLOOR,
    rounded up to a power of two (the splice's `shl` forms block*slot with a shift)."""
    want = max(DEBUG_SLOT_FLOOR, 32 * nregs * 4)
    return 1 << (want - 1).bit_length()


# ---------------------------------------------------------------------------
# A small PTX lineariser, duplicating rr_layer_spec.py's convention (a statement ends at its `;`,
# a wrapped `{ ... }` scope is a delimiter but an operand register list is not, blank/`//`/`.reg`
# lines are not statements).  Only what the debug stages need is parsed: the runs of adjacent mma.
RE_ENTRY = re.compile(r"\.entry\s+([\w$]+)\s*\(")
RE_MMA = re.compile(r"^mma\.sync")


def _entry_lines():
    """The entry's body lines of the corpus module, or None when the corpus is not reachable."""
    roots = []
    if os.environ.get("D4R_DLSSD_PTX_DIR"):
        roots.append(os.environ["D4R_DLSSD_PTX_DIR"])
    roots.append(os.path.expanduser("~/.cache/d4r-rr-corpus"))
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extracted", "rr"))
    for root in roots:
        path = os.path.join(root, FILE)
        if os.path.isfile(path):
            lines = open(path).read().split("\n")
            for index, line in enumerate(lines):
                if not RE_ENTRY.search(line) or ENTRY not in line:
                    continue
                depth, started = 0, False
                for j in range(index, len(lines)):
                    depth += lines[j].count("{") - lines[j].count("}")
                    if "{" in lines[j]:
                        started = True
                    if started and depth == 0:
                        return lines[index:j + 1], index + 1
            return None, None
    return None, None


def _mma_runs(body, first_line):
    """(last_line, [D regs]) for every maximal run of adjacent mma statements of the entry whose
    body starts at file line `first_line`.

    `last_line` is the file line the run's final mma ends on -- the line the debug splice is
    inserted after.  D regs are the two b32 each mma writes, in statement order; that is the
    register list the oracle stores for a phase and the list the native body's own debug_store
    reproduces in the same order."""
    runs = []
    cur = None
    buf = None
    for index, raw in enumerate(body):
        line = first_line + index
        s = raw.strip()
        if not s or s.startswith("//") or s.startswith("."):
            continue
        while s and s[0] in "{}" and s[1:2] != "%":
            s = s[1:].lstrip()
        while s.endswith("}") and s[:-1].rstrip().endswith(";"):
            s = s[:-1].rstrip()
        if not s:
            continue
        if buf is None:
            buf = s
        else:
            buf += " " + s
        while buf is not None and buf.endswith(";"):
            text, buf, last = buf[:-1].strip(), None, line
            if RE_MMA.match(text):
                groups = re.findall(r"\{([^}]*)\}", text)
                dest = re.findall(r"%r\d+", groups[0]) if groups else []
                if cur is None:
                    cur = [last, []]
                cur[0] = last
                cur[1].extend(dest)
            elif cur is not None:
                runs.append(tuple(cur))
                cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def _stage_lines():
    """[(line, D registers)] per phase: the line the oracle's debug store goes after, and the
    phase's D registers in mma statement order."""
    body, first = _entry_lines()
    if body is None:
        return []
    entry = next(i for i, line in enumerate(body) if line.startswith(".visible .entry"))
    return _mma_runs(body[entry + 1:], first + entry + 1)


# ---------------------------------------------------------------------------
# The splice and its declaration
def cut_call():
    """The PTX that replaces the mma region.

    The callee is handed the staged A fragments, the C seed, the weight image, the parameter block
    and the address of the shared arena; it writes the five live-out registers into the arena and
    this splice reads them back into the epilogue's registers."""
    args = A_REGS + C_REGS
    decl = "\n".join(f"\t.param .b32 p{i};" for i in range(len(args)))
    store = "\n".join(f"\tst.param.b32 [p{i}+0], {r};" for i, r in enumerate(args))
    decl += "\n\t.param .b64 pw;\n\t.param .b64 pv;\n\t.param .b32 ps;"
    store += ("\n\tst.param.b64 [pw+0], %rd2;\n\tst.param.b64 [pv+0], %rd9;"
              f"\n\tmov.u32 %rswin0, {SCRATCH};"
              "\n\tst.param.b32 [ps+0], %rswin0;")
    hel = ", ".join([f"p{i}" for i in range(len(args))] + ["pw", "pv", "ps"])
    gets = "\n".join(f"\tld.shared.b32 {r}, [%rswin1+{4 * i}];" for i, r in enumerate(RET_REGS))
    return f"""{{
.reg .b32 %rswin<2>;
{decl}
{store}
\tcall.uni {CALLEE}, ({hel});
\tmad.lo.s32 %rswin1, %laneid, {RET_STRIDE}, %rswin0;
{gets}
}}"""


def cut_extern():
    args = ", ".join([f".param .b32 d4r_a{i}" for i in range(len(A_REGS) + len(C_REGS))]
                     + [".param .b64 d4r_w", ".param .b64 d4r_v", ".param .b32 d4r_s"])
    return f".extern .func {CALLEE}\n(\n\t{args}\n)\n;\n"


# ---------------------------------------------------------------------------
# The oracle's debug stage: keep NVIDIA's whole body and store one phase's D registers past the
# end of the plane arena the kernel writes, so the launch dump holds the translated values for a
# block/lane.  Built with -DD4R_RRSWIN_DEBUG_STAGE=<n> (kernels/build.sh rrswin, D4R_TEX_CFLAGS),
# kernels/rr/rrswin_enc4.hip stores the same registers of its own phase n in the same order, so the
# two dumps compare byte for byte.
def debug_block(regs):
    """PTX that stores REGS (u32 each) to plane arena + DEBUG_BASE + block*slot + lane*len(regs)*4.

    Works on either side of the splice: it is inserted into NVIDIA's own PTX for the reference and
    reproduced by the native body's debug flag, and both sides compute the same slot index.  The
    predicate is `ctaid.x < SLOT_X && ctaid.y < SLOT_Y && tid.z < SLOT_Z && slot < DEBUG_BLOCKS`,
    so the mapping stays injective however the kernel is launched.  The DEBUG_BLOCKS term is what
    keeps the region inside arg008: the arena cannot hold all BLOCK_LIMIT slots of the widest
    stage, and both sides drop the rest rather than write past the end."""
    slot = debug_slot(len(regs))
    # `st.global.v4.u32` needs a multiple of four operands, and stage e4a stores 25 registers
    # (24 A + the C seed); pad the tail by repeating the last register.  The comparison reads
    # only the first len(regs) words, so the padding is invisible on both sides.
    padded = list(regs) + [regs[-1]] * (-len(regs) % 4)
    stores = "\n".join(
        f"@%pdbg5 st.global.v4.u32 [%rddbg3+{i * 16}], {{{', '.join(padded[i * 4:i * 4 + 4])}}};"
        for i in range(len(padded) // 4))
    return f"""// d4r debug stage
{{
.reg .pred %pdbg<7>;
.reg .b32 %rdbg<8>;
.reg .b64 %rddbg<4>;
ld.param.u64 %rddbg1, [{ENTRY}_param_0+{PLANE_ARG}];
mov.u32 %rdbg1, %ctaid.x;
mov.u32 %rdbg2, %ctaid.y;
mov.u32 %rdbg3, %tid.z;
setp.lt.u32 %pdbg1, %rdbg1, {SLOT_X};
setp.lt.u32 %pdbg2, %rdbg2, {SLOT_Y};
setp.lt.u32 %pdbg3, %rdbg3, {SLOT_Z};
and.pred %pdbg4, %pdbg1, %pdbg2;
and.pred %pdbg5, %pdbg4, %pdbg3;
shl.b32 %rdbg4, %rdbg2, 5;
add.s32 %rdbg4, %rdbg4, %rdbg1;
shl.b32 %rdbg5, %rdbg3, 8;
add.s32 %rdbg4, %rdbg4, %rdbg5;
setp.lt.u32 %pdbg6, %rdbg4, {DEBUG_BLOCKS};
and.pred %pdbg5, %pdbg5, %pdbg6;
shl.b32 %rdbg4, %rdbg4, {slot.bit_length() - 1};
mov.u32 %rdbg5, %laneid;
mad.lo.s32 %rdbg6, %rdbg5, {len(regs) * 4}, %rdbg4;
add.s32 %rdbg6, %rdbg6, {DEBUG_BASE};
cvt.u64.u32 %rddbg2, %rdbg6;
add.s64 %rddbg3, %rddbg1, %rddbg2;
{stores}
}}
"""


def _stage_table():
    """{stage name: (line, D registers)} for every phase of the entry.

    Every name carries the `e4` prefix.  make_ptx.py's swin_debug() is consulted *before* this
    recipe, and its unkeyed SWIN_DEBUG_STAGES table holds enc0's `a`, `p1`..`p7` with enc0's line
    numbers and enc0's param_0+48 plane address; an unprefixed name would silently splice enc0's
    registers into enc4's module and store them to a parameter enc4 never had.  The keyed
    `("enc4", name)` hook (swin_enc4_debug_stages.py) is checked first, and an unrecognised name
    falls through here -- so `e4p*` is what actually reaches this table.

    `p<n>` is phase n's D registers (however many the phase has) in mma statement order; `e4a` is
    the staged A operand the callee is handed, which pins the prologue independently of the body's
    arithmetic.

    Every stage's register count is padded to a multiple of four.  The lane stride is
    `len(regs)*4`, and the store is a `st.global.v4.u32`, which needs 16-byte alignment: with 25
    registers the stride is 100 bytes, so every odd lane's store is misaligned and the hardware
    drops it (measured: the oracle stored zeros for odd lanes while the native side's scalar
    stores wrote real values).  Phase stages are already 40, 48, 96 or 160 registers wide and so
    need no padding; only the 25-register `e4a` does.
    """
    table = {}
    stages = _stage_lines()
    a_regs = A_REGS + C_REGS
    table["e4a"] = (CUT_FIRST - 1, a_regs + [a_regs[-1]] * (-len(a_regs) % 4))
    for n, (line, regs) in enumerate(stages, start=1):
        table[f"e4p{n}"] = (line, regs)
    return table


def recipe():
    """The make_ptx.py spec for this layer.

    D4R_RRSWIN_DEBUG=<stage> selects the oracle build: NVIDIA's whole body with that stage's
    registers stored past the plane arena.  Otherwise the mirror cut, built and served."""
    stage = os.environ.get("D4R_RRSWIN_DEBUG", "")
    if stage:
        table = _stage_table()
        if stage not in table:
            raise SystemExit(
                f"swin_enc4_recipe: no stage {stage!r}; known: " + ", ".join(sorted(table)))
        line, regs = table[stage]
        return dict(file=FILE, debug=(line, regs), debug_block=debug_block)
    return dict(file=FILE, cut_call=(CUT_FIRST, CUT_AFTER), extern=cut_extern(), call=cut_call())


if __name__ == "__main__":
    # self-check: print the recipe and the stage table
    import pprint
    pprint.pprint({k: (v if k != "call" else "<generated>") for k, v in recipe().items()})
    for name, (line, regs) in sorted(_stage_table().items(),
                                     key=lambda kv: (len(kv[1][1]), kv[0])):
        print(f"{name:>4s}  line {line:6d}  {len(regs):3d} registers")