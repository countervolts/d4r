#!/usr/bin/env python3
"""Oracle debug stages for cuda_dldn_engine_swin_dec2_kernel.

make_ptx.py loads this module and keys its stages() by (layer, stage), so these
names do not collide with enc0's unkeyed table.  Each entry is
(line after which the store block goes, the registers to store); the store block
itself is make_ptx.py's shared swin_debug_block(), which writes to param_0+48 +
D4R_RRSWIN_DEBUG_BASE + block*slot + lane*n*4 with block = ctaid.y*161 + ctaid.x
below 64.  rrswin_dec2.hip's debug_store() writes the same bytes from its own
values, so the two launch dumps compare byte for byte.

**Why every stage splices after E37 (line 36892).**  dec2's plane arena is a
5,652,480-byte window at byte 60,293,120 of the 65,945,600-byte block the bridge
dumps as arg048's allocation (measured from a D4R_CUDA_REPLAY_DUMP capture:
`pointer 48 0 60293120`).  E37 -- the 12 `st.global.u32` plane-arena scatters --
rewrites essentially all of it: diffing a pre-launch replay capture against the
post-launch reference dump shows 5,564,929 of 5,652,480 arena bytes changed.  A
splice placed next to a phase's mma is therefore erased before the dump.  Line
36892 is the `;` of phase 38's last mma, i.e. after E37 and before E38 (which
never touches param_0+48), so the payload survives.

The accumulators are intact at line 36892: a last-definition scan over the whole
entry shows every phase-n D register is written for the last time before line
33,227, and never rewritten after.

make_ptx.py additionally requires the first stored register to appear in the 4000
lines before the splice point.  Phases 2..28 name registers the late window never
mentions, so those lists start with %r19329 -- phase 38's C bias seed, a
weight-image load the native body reproduces trivially -- and the phase's own
accumulators follow.

The lists are generated from the corpus PTX rather than written out, so they
follow the corpus: each phase's mma run in statement order, contributing its D
pair.
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORPUS = os.environ.get("D4R_DLSSD_PTX_DIR", os.path.expanduser("~/.cache/d4r-rr-corpus"))
_PTX = os.path.join(_CORPUS, "0021-PREPASS_ENTRYPOINT_NAME.ptx")

# the line the store block follows: phase 38's last mma
SPLICE_LINE = 36892
# phase 38's first C bias seed: live at the splice point and inside make_ptx's window
SENTINEL = "%r19329"

RE_ENTRY = re.compile(r"\.entry\s+cuda_dldn_engine_swin_dec2_kernel\s*\(")
RE_MMA = re.compile(r"mma\.sync[^(]*\{([^}]*)\}")


def _entry_lines():
    lines = open(_PTX).read().split("\n")
    next(i for i, l in enumerate(lines) if RE_ENTRY.search(l))
    return lines


def _mma_runs():
    """Per phase (1-based), that phase's D registers in mma statement order."""
    sys.path.insert(0, _HERE)
    import rr_layer_spec as R
    lines = open(_PTX).read().split("\n")
    body = next((a, b) for n, a, b in R.find_entries(lines)
                if n == "cuda_dldn_engine_swin_dec2_kernel")
    stmts, _ = R.linearise(lines, *body)
    runs, prev = [], False
    for s in stmts:
        if not s.is_mma:
            prev = False
            continue
        if not prev:
            runs.append([])
        runs[-1].extend(x.strip() for x in RE_MMA.match(s.text).group(1).split(","))
        prev = True
    return runs


def stages():
    """name -> (line after which the store block goes, registers)."""
    window = "\n".join(_entry_lines()[SPLICE_LINE - 4000:SPLICE_LINE])
    # swin_debug_block() emits v4 stores, so a list whose length is not a multiple
    # of four would produce a short vector and fail PTX translation; pad with the
    # sentinel, which both sides store identically.
    out = {}
    for phase, regs in enumerate(_mma_runs(), start=1):
        # make_ptx.py's own check: the first stored register must appear in the window
        lead = [] if regs[0] in window else [SENTINEL]
        full = lead + regs
        full += [SENTINEL] * (-len(full) % 4)
        out["p%d" % phase] = (SPLICE_LINE, full)
    return out