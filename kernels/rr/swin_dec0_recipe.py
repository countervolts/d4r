"""Splice recipe for `cuda_dldn_engine_swin_dec0_kernel`.

Self-contained: `kernels/tex/make_ptx.py` imports this module by path and returns
`recipe()` for the kernel name, so nothing here may import make_ptx.  Everything
below is derived from the corpus PTX with `kernels/rr/rr_layer_spec.py`'s own
parser (the sibling module, imported for its lineariser and mma collector): the
mma region is `s326-s18231` (file lines 1393-46263), 544 mma in 19 phases, the
weight image is 47376 bytes at arg040, the parameter block is 152 bytes, and the
launch is grid 161x93 with block 32x1x1 (one wave).

**Shape of the splice.** NVIDIA's plane/texture prologue (lines 1-1392) and its
surface epilogue (46264-end) are kept verbatim, and so is every epilogue between
the phases: E1's shared-window staging, the RMS norms, both attention softmaxes
and the clamped-cubic activations all stay NVIDIA's PTX, so the body only has to
reproduce the arithmetic the mma statements perform.  Each phase's contiguous mma
run is replaced by one `call.uni` into `d4r_swin_dec0` with that phase's operands.

**Why the calls need spill slots.** `call.uni` has no ABI: every register the
surrounding PTX still needs across the call is destroyed.  So each block

  1. stores the registers that are live across this call (defined before the
     phase's first mma, read after its last) into `d4r_spill`,
  2. copies the phase's A/B/C operand registers into the parameter bank,
  3. calls,
  4. reads the phase's D registers back out of `d4r_spill` into the exact
     register names the mma would have written, and
  5. reloads the spilled registers.

The largest live-across set is 508 bytes per lane (phase 3), so the spill area
is `SPILL_SLOT` bytes per lane, and the largest D return is phase 1's 192 b32,
so the return area is `RET_SLOT` bytes per lane.  Both are indexed off
`laneid * LANE_STRIDE` inside a single module-scope `.shared` array; the array is
declared in the `extern` string, which `make_ptx.py` emits at module scope (a
`.shared` declaration is not legal inside a block).

**Operand encoding.** The callee receives, in this order, the phase number, the
A fragments (four b32 each, in first-use order), the B fragments (two b32 each,
in first-use order), the C words (two b32 per distinct C operand, in first-use
order), the weight image, the parameter block and its shared row base.  B
fragments and C operands that `rr_layer_spec` resolves into the weight image are
*not* passed: the callee reads them from arg040 at the offsets the mma resolved.
Chain C operands (a previous k step's D) and the constant `%r13609` are not
passed either: the callee computes them.
"""

import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import rr_layer_spec as SPEC   # noqa: E402  (path is fixed up above)

MODULE = "0023-PREPASS_ENTRYPOINT_NAME.ptx"
ENTRY = "cuda_dldn_engine_swin_dec0_kernel"
PARAM = ENTRY + "_param_0"

# the cut: phase 1's first mma (line 1393) through phase 19's last (line 46266);
# make_ptx.py replaces lines[before-1:after-1]
CUT = (1393, 46267)

# shared-memory geometry, bytes per lane
LANE_STRIDE = 1280
RET_SLOT = 768        # 192 b32, phase 1's D
SPILL_SLOT = 512      # the widest live-across set is 508
SPILL = RET_SLOT      # spill area starts after the return area
SPILL_BYTES = LANE_STRIDE * 32

# the callee's parameter bank: MAXP b32 then b64, b64, b32.  Phase 4 passes the
# most (137); rrswin_dec0.hip's d4r_swin_dec0 must declare the same count.
MAXP = 144

SPILL_SYMBOL = "d4r_spill"


def _ptx_dir():
    return Path(os.environ.get("D4R_DLSS_PTX_DIR",
                               _HERE.parent / "extracted" / "ptx"))


def _key(reg):
    return (reg[:3], int(reg[3:]) if reg[3:] else 0)


def _analyse():
    """Every number the splice needs, read out of the corpus PTX."""
    lines = (_ptx_dir() / MODULE).read_text().split("\n")
    body = next(((a, b) for n, a, b in SPEC.find_entries(lines) if n == ENTRY))
    if body is None:
        raise SystemExit(f"{MODULE}: no entry {ENTRY}")
    stmts, _ = SPEC.linearise(lines, *body)
    defs = SPEC.track_weights(stmts, ENTRY)
    mmas = SPEC.collect_mma(stmts, defs)
    phases = SPEC.build_phases(stmts, mmas)

    last_w, last_r = {}, {}
    for s in stmts:
        for r in SPEC.dest_regs(s.text):
            last_w[r] = s.idx
        for r in SPEC.read_regs(s.text):
            last_r[r] = max(last_r.get(r, 0), s.idx)
        for group in SPEC.RE_OPERAND.findall(s.text):
            for r in SPEC.REG_SRC.findall(group):
                last_r[r] = max(last_r.get(r, 0), s.idx)

    out = []
    for p in phases:
        lo, hi = p.mma[0].idx, p.mma[-1].idx
        a_frag, b_frag, c_frag = [], [], []
        c_index, d_regs = [], []
        for m in p.mma:
            a = tuple(m.regs("A"))
            b = tuple(m.regs("B"))
            if a not in a_frag:
                a_frag.append(a)
            if b not in b_frag:
                b_frag.append(b)
            d_regs.extend(m.regs("D"))
            cregs = tuple(m.regs("C"))
            if all(r in d_regs or r == "%r13609" for r in cregs):
                c_index.append(-1)                      # chain or the zero constant
            elif all(o is None for o in m.co):
                c_index.append(-1)                      # resolved into the weight image
            else:
                if cregs not in c_frag:
                    c_frag.append(cregs)
                c_index.append(c_frag.index(cregs))
        out.append(dict(
            idx=p.idx, lo=lo, hi=hi,
            line0=p.mma[0].line, line1=p.mma[-1].line + 3,
            a=[r for f in a_frag for r in f],
            b=[r for f in b_frag for r in f if defs.from_weight(r, lo) is None],
            c=[r for f in c_frag for r in f],
            c_index=c_index, d=d_regs,
            spill=sorted((r for r in last_w if last_w[r] < lo and last_r.get(r, 0) > hi),
                         key=_key),
        ))
    return lines, out


def _extern():
    """The callee's declaration and the shared spill array, at module scope."""
    params = ", ".join(f".param .b32 d4r_p{i}" for i in range(MAXP))
    return (f".shared .align 4 .b8 {SPILL_SYMBOL}[{SPILL_BYTES}];\n"
            f".extern .func d4r_swin_dec0\n(\n\t{params},\n"
            "\t.param .b64 d4r_w, .param .b64 d4r_v, .param .b32 d4r_s\n)\n;\n")


def _spill_ops(ph):
    """The st.shared / ld.shared pair that carries one live-across register."""
    lines = []
    for j, reg in enumerate(ph["spill"]):
        off = SPILL + 4 * j
        if reg.startswith("%rd"):
            lines.append(f"\tmov.b64 {{%rtlo,%rthi}}, {reg};")
            lines.append(f"\tst.shared.v2.b32 [%rk+{off}], {{%rtlo,%rthi}};")
        elif reg.startswith("%p"):
            lines.append(f"\tselp.b32 %rtlo, 1, 0, {reg};")
            lines.append(f"\tst.shared.u32 [%rk+{off}], %rtlo;")
        else:
            lines.append(f"\tst.shared.u32 [%rk+{off}], {reg};")
    return lines


def _reload_ops(ph):
    lines = []
    for j, reg in enumerate(ph["spill"]):
        off = SPILL + 4 * j
        if reg.startswith("%rd"):
            lines.append(f"\tld.shared.v2.b32 {{%rtlo,%rthi}}, [%rk+{off}];")
            lines.append(f"\tmov.b64 {reg}, {{%rtlo,%rthi}};")
        elif reg.startswith("%p"):
            lines.append(f"\tld.shared.u32 %rtlo, [%rk+{off}];")
            lines.append(f"\tsetp.ne.u32 {reg}, %rtlo, 0;")
        else:
            lines.append(f"\tld.shared.u32 {reg}, [%rk+{off}];")
    return lines


def _block(ph):
    """The `{ ... }` that replaces one phase's mma run."""
    k = ph["idx"]
    n = 1 + len(ph["a"]) + len(ph["b"]) + len(ph["c"])
    if n > MAXP:
        raise SystemExit(f"phase {k} needs {n} parameters, MAXP is {MAXP}")
    decl = "\n".join(f"\t.param .b32 q{k}_{i};" for i in range(n))
    decl += f"\n\t.param .b64 qw{k};\n\t.param .b64 qv{k};\n\t.param .b32 qs{k};"
    zero = "\n".join(f"\tst.param.b32 [q{k}_{i}+0], 0;" for i in range(n))
    operands = [f"\tst.param.b32 [q{k}_{i + 1}+0], {r};"
                for i, r in enumerate(ph["a"] + ph["b"] + ph["c"])]
    args = ", ".join([f"q{k}_{i}" for i in range(n)] + [f"qw{k}", f"qv{k}", f"qs{k}"])
    body = [
        f"mov.u32 %rk, {SPILL_SYMBOL};",
        f"mad.lo.s32 %rk, %laneid, {LANE_STRIDE}, %rk;",
    ]
    body += _spill_ops(ph)
    body += [decl, zero, "\n".join(operands)]
    body += [
        f"ld.param.u64 %rv, [{PARAM}+40];",
        f"st.param.b64 [qw{k}+0], %rv;",
        f"mov.b64 %rv, {PARAM};",
        f"st.param.b64 [qv{k}+0], %rv;",
        f"st.param.b32 [qs{k}+0], %rk;",
        f"call.uni d4r_swin_dec0, ({args});",
    ]
    body += [f"\tld.shared.u32 {ph['d'][i]}, [%rk+{4 * i}];" for i in range(len(ph["d"]))]
    body += _reload_ops(ph)
    return "\n".join(["{",
                      ".reg .b32 %rk, %rv, %rtlo, %rthi;",
                      "\n".join(body),
                      "}"])


def _call():
    lines, phases = _analyse()
    out = []
    cursor = CUT[0]
    for ph in phases:
        if not ph["spill"] and not ph["c"] and len(ph["a"]) + len(ph["b"]) > 0:
            pass
        out += lines[cursor - 1:ph["line0"] - 1]
        out.append(_block(ph))
        cursor = ph["line1"] + 1
    out += lines[cursor - 1:CUT[1] - 1]
    return "\n".join(out)


def recipe():
    return dict(file=MODULE, cut_call=CUT, extern=_extern(), call=_call())