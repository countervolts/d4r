#!/usr/bin/env python3
"""Structural spec for one Ray Reconstruction (DLSS-FG) denoiser layer.

Reads NVIDIA's extracted PTX for a single `cuda_dldn_engine_swin_*` entry out of
`~/.cache/d4r-rr-corpus` and reports, with no GPU involved:

  * the linearised statement list for the entry,
  * a phase table (one row per maximal run of adjacent `mma.sync`),
  * a classification of the code between the phases,
  * the cut ABI: what a native device function would receive and return.

Usage:
    rr_layer_spec.py enc0                 # canonical corpus file for enc0
    rr_layer_spec.py --file F.ptx --entry NAME
    rr_layer_spec.py --list                # corpus alias table

Statement indexing convention (identical for every entry, so reports compare):
  * scope is the `.visible .entry NAME(...) { ... }` body;
  * a statement is a source construct terminated by `;`;
  * blank lines, `//` comment lines and `.directive` lines are not statements;
  * `}` and `{` that stand alone on a line are not statements;
  * a label (`$L__BB0_12:`) is its own statement;
  * multi-line instructions (the 4-line `mma.sync`, wrapped `tex.`) are joined
    into one statement, numbered at the line its first line is on.

Every statement therefore also carries its 1-based physical line number in the
PTX, and every range in the output below is printed as `s<first>-<last>`, so a
reader can check any claim against `sed -n` on the corpus file.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from dataclasses import dataclass, field

CORPUS = os.path.expanduser("~/.cache/d4r-rr-corpus")

# alias -> (corpus file, entry name)
ALIASES = {
    "enc0": ("0013-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_enc0_kernel"),
    "enc1": ("0014-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_enc1_kernel"),
    "enc2": ("0015-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_enc2_kernel"),
    "enc3": ("0016-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_enc3_kernel"),
    "enc4": ("0017-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_enc4_kernel"),
    "enc5": ("0018-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_enc5_kernel"),
    "dec0": ("0023-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_dec0_kernel"),
    "dec1": ("0022-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_dec1_kernel"),
    "dec2": ("0021-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_dec2_kernel"),
    "dec3": ("0020-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_dec3_kernel"),
    "dec4": ("0019-PREPASS_ENTRYPOINT_NAME.ptx", "cuda_dldn_engine_swin_dec4_kernel"),
    "prepass": ("0013-PREPASS_ENTRYPOINT_NAME.ptx", "PREPASS_ENTRYPOINT_NAME"),
}

# ---------------------------------------------------------------------------
# linearisation
# ---------------------------------------------------------------------------

RE_ENTRY = re.compile(r"\.entry\s+([\w$]+)\s*\(")
RE_LABEL = re.compile(r"^\$[\w$]+:$")


@dataclass
class Stmt:
    idx: int  # 1-based statement index inside the entry
    line: int  # 1-based physical line in the PTX file
    text: str
    op: str

    @property
    def is_mma(self) -> bool:
        return self.op.startswith("mma.sync")

    @property
    def is_label(self) -> bool:
        return bool(RE_LABEL.match(self.text))


def find_entries(lines):
    """Yield (name, first_line_index, last_line_index) for every .entry in the file."""
    for idx, line in enumerate(lines):
        m = RE_ENTRY.search(line)
        if not m:
            continue
        depth = 0
        started = False
        for j in range(idx, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if "{" in lines[j]:
                started = True
            if started and depth == 0:
                yield m.group(1), idx, j
                break


def op_of(text: str) -> str:
    m = re.match(r"(@!?%p\d+\s+)?([a-z][a-z0-9_.:]*)", text)
    return m.group(2) if m else text


def linearise(lines, first, last):
    """Linearise one entry body.  Returns (statements, leftover_text).

    NVIDIA writes each instruction either bare (`mma f;`) or wrapped in a
    scoped block (`{ cvt f;}`), and the 4-line `mma.sync` continuation lines
    start with `{` because their operands are register lists.  So braces are
    stripped as block delimiters and a statement ends at its `;`.
    """
    stmts: list[Stmt] = []
    buf = None
    start = None
    for i in range(first, last + 1):
        s = lines[i].strip()
        if not s or s.startswith("//"):
            continue
        if s.startswith("."):  # .reg / .shared / .pragma / .maxntid / ...
            continue
        if s.startswith("{."):  # scoped declaration: `{.reg .f16 low, high;}`
            continue
        # A brace is a block delimiter unless it introduces an operand register
        # list: `{ cvt...` and `{mul.f16 ...` open a scope, `{%r0, %r1},` is an
        # operand list and must survive so the mma parses back into fragments.
        # A trailing `}` closing a `;` closes a scope.
        while s and s[0] in "{}" and s[1:2] != "%":
            s = s[1:].lstrip()
        while s.endswith("}") and s[:-1].rstrip().endswith(";"):
            s = s[:-1].rstrip()
        if not s:
            continue
        if s.startswith("."):  # scoped `{.reg .f16 low, high;}` declaration
            continue
        if RE_LABEL.match(s):
            stmts.append(Stmt(len(stmts) + 1, i + 1, s, "$label"))
            continue
        if buf is None:
            start = i + 1
            buf = s
        else:
            buf += " " + s
        while buf is not None and buf.endswith(";"):
            body = buf[:-1].strip()
            stmts.append(Stmt(len(stmts) + 1, start, body, op_of(body)))
            buf = None
    return stmts, buf


# ---------------------------------------------------------------------------
# every PTX virtual register mentioned anywhere in a statement
REGS = re.compile(r"%[a-z]+\d+")


def resolve(layer, path, entry):
    """(ptx path, entry name) for a layer alias or an explicit --file/--entry."""
    if path is None:
        fn = ALIASES.get(layer, (None, None))[0]
        if fn is None:
            stem = layer.replace("cuda_dldn_engine_swin_", "")
            fn = ALIASES.get(stem, (None, None))[0]
        if fn is None:
            hits = [f for f in sorted(os.listdir(CORPUS)) if stem in f]
            if not hits:
                print(f"no corpus file for {layer}", file=sys.stderr)
                return None, None
            fn = hits[0]
        path = os.path.join(CORPUS, fn)
    if entry is None:
        entry = ALIASES.get(layer, (None, None))[1]
    if entry is None:
        names = [n for n, _, _ in find_entries(open(path).read().split("\n"))]
        stem = layer.replace("cuda_dldn_engine_swin_", "")
        entry = next((n for n in names if stem in n), names[0] if names else None)
    return path, entry

# ---------------------------------------------------------------------------
# weight-pointer tracking
# ---------------------------------------------------------------------------

# ld.{weak.}global[.ca].<ty> {a,b}, [ptr]  /  ld.global.u32 %r, [ptr+imm]
RE_LD_VEC = re.compile(
    r"^ld\.(?:weak\.)?global(?:\.ca)?\.[a-z0-9.]*\s*\{([^}]*)\},\s*\[(%\w+)(?:\+(\d+))?\]$")
RE_LD_SCAL = re.compile(
    r"^ld\.(?:weak\.)?global(?:\.ca)?\.[a-z0-9.]*\s*(%\w+),\s*\[(%\w+)(?:\+(\d+))?\]$")

# `param` offset (bytes) of the kernel's 64-bit weight pointer
WEIGHT_PARAM_OFFSET = 40
# bytes between consecutive N-tiles of one K=32 slice of a packed weight matrix
NTILE_STRIDE = 512
# bytes per m16n8k32 mma step along K
KSTEP = 8


class Defs:
    """Versioned register definitions.

    NVIDIA reuses physical registers across phases, so a `reg -> offset` map
    built with last-write-wins is wrong.  Every definition is kept as
    `(statement_index, info)` in a per-register list, and lookups are resolved
    *at a program point* by taking the newest definition at or before it.

    `info` is:
      ("W", offset)  register holds a value read from the weight image at
                     `offset` bytes from the image base (the param+40 pointer),
      ("P", param)   register holds a kernel parameter pointer at byte `param`,
      ("G",)         register holds anything else loaded from global memory.
    """

    def __init__(self):
        self.defs = {}

    def put(self, reg, idx, info):
        self.defs.setdefault(reg, []).append((idx, info))

    def at(self, reg, idx):
        """Newest definition of `reg` at or before statement `idx`."""
        best = None
        for i, info in self.defs.get(reg, ()):
            if i <= idx:
                best = info
            else:
                break
        return best

    def from_weight(self, reg, idx):
        info = self.at(reg, idx)
        return info[1] if info and info[0] == "W" else None

    def walk_origin(self, reg, idx, stmts, max_steps=200):
        """Follow a register's def-use chain back to the memory or mma operation
        that produced it.  Returns (op, statement_index, packed) or None.

        An A fragment reaches the mma through `mov.b32 %rX, {%rsA, %rsB}`
        packing two b16 registers that `cvt.rn.satfinite.e4m3x2.f16x2` wrote
        from f32 registers, so a one-hop lookup on the mma operand is not
        enough - the walk has to pass *through* that quantisation to reach the
        real producer.  `packed` is True when an e4m3 quantisation was crossed.
        """
        seen = set()
        frontier = [(reg, idx)]
        packed = False
        for _ in range(max_steps):
            nxt = []
            for r, at in frontier:
                if r in seen:
                    continue
                seen.add(r)
                hits = [i for i, _ in self.defs.get(r, ()) if i <= at]
                if not hits:
                    continue
                i = hits[-1]
                st = stmts[i]
                if st.op.startswith(("ld.", "tex.", "mma.sync")):
                    return st.op, i, packed
                if st.op.startswith("cvt.rn.satfinite.e4m3") or \
                        st.op.startswith("cvt.rn.satfinite.relu.e4m3"):
                    packed = True
                nxt.extend((src, i) for src in read_regs(st.text))
            frontier = nxt
            if not frontier:
                return None
        return None


REGS = re.compile(r"%[a-z]+\d+")
REG_SRC = REGS


def strip_block_braces(text):
    """Drop leading block braces.  `{ cvt...` and `{mul.f16 ...` open a scope;
    `{%r0, %r1},` is an operand list and must survive."""
    while text and text[0] in "{}" and not text[1:2] == "%":
        text = text[1:].lstrip()
    return text


def read_regs(text):
    """Source registers of a statement: every register after its destination."""
    text = strip_block_braces(text)
    depth = 0
    for i, ch in enumerate(text):
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "," and depth == 0:
            return REG_SRC.findall(text[i + 1:])
    return []


def track_weights(stmts, entry):
    """Resolve every register to the global-memory load that defines it.

    A pointer register is tracked symbolically.  `ld.param.u64` at +40 is the
    weight pointer (arg040 in the harness dump: the 54560-byte prepared weight
    image).  `add.s64 rd, rW, x` keeps the weight lineage and accumulates `x`
    whether `x` is a literal or a lane-strided addend, so a weight load's byte
    offset is exact.
    """
    d = Defs()
    val = {}
    rparam = re.compile(r"^ld\.param\.u64 (%\w+), \[" + entry + r"_param_0\+(\d+)\]$")
    rcvta = re.compile(r"^cvta\.to\.global\.u64 (%\w+), (%\w+)$")
    radd = re.compile(r"^add\.s64 (%\w+), (%\w+), (\S+)$")

    for k, st in enumerate(stmts):
        t = st.text
        m = rparam.match(t)
        if m:
            off = int(m.group(2))
            val[m.group(1)] = ("W", 0) if off == WEIGHT_PARAM_OFFSET else ("P", off)
            continue
        m = rcvta.match(t)
        if m and m.group(2) in val:
            val[m.group(1)] = val[m.group(2)]
            continue
        m = radd.match(t)
        if m:
            dst, a, b = m.group(1), m.group(2), m.group(3)
            va = val.get(a)
            vb = val.get(b) if b.startswith("%") else ("K", int(b))
            if va and va[0] == "W" and vb is None:
                val[dst] = ("W", va[1])
            elif va and vb and va[0] == "W" and vb[0] in ("W", "K"):
                val[dst] = ("W", va[1] + vb[1])
            continue
        m = RE_LD_SHARED.match(t)
        if m:
            g = m.group(1) if m.group(1) is not None else m.group(2)
            imm = int(m.group(4) or 0)
            regs = ([x.strip() for x in g.split(",")] if "," in g
                    else [g.strip()])
            for i, r in enumerate(regs):
                d.put(r, k, ("S", imm + i * 4))
            continue
        for pat in (RE_LD_VEC, RE_LD_SCAL):
            m = pat.match(t)
            if not m:
                continue
            g, ptr, imm = m.group(1), m.group(2), int(m.group(3) or 0)
            regs = ([x.strip() for x in g.split(",")] if "," in g
                    else [g.strip()])
            vp = val.get(ptr)
            base = vp[1] if vp and vp[0] == "W" else None
            for i, r in enumerate(regs):
                d.put(r, k, ("W", base + imm + i * 4) if base is not None else ("G",))
            break
        else:
            # any other write: record it so a def-use walk can follow the
            # chain back to a memory operation
            for r in dest_regs(t):
                d.put(r, k, ("V", st.op))
    return d


def dest_regs(text):
    """Destination registers of a statement: the operand(s) before the first
    top-level comma, minus any predicate."""
    depth = 0
    body = text.split(None, 1)
    if body[0].startswith("@"):
        text = body[1] if len(body) > 1 else ""
    for ch in text:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "," and depth == 0:
            return REG_SRC.findall(text[:text.index(",")])
    return REG_SRC.findall(text)


RE_LD_SHARED = re.compile(
    r"^ld\.shared\.[a-z0-9.]*\s*(?:\{([^}]*)\}|([^,\[]+?))\s*,\s*\[(%\w+)(?:\+(\d+))?\]$")


# ---------------------------------------------------------------------------
# mma extraction and phases
# ---------------------------------------------------------------------------

RE_MMA = re.compile(
    r"^mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16 "
    r"\{([^}]*)\}, \{([^}]*)\}, \{([^}]*)\}, \{([^}]*)\}$")


@dataclass
class Mma:
    idx: int          # statement index
    line: int         # physical line
    D: str            # output fragment registers   {d}
    A: str            # activation fragment registers {a}
    B: str            # weight fragment registers    {b}
    C: str            # accumulator input registers  {c}
    bo: tuple = ()    # byte offsets of the B regs inside the weight image
    co: tuple = ()    # byte offsets of the C regs (bias) inside the weight image

    def regs(self, which):
        return tuple(r.strip() for r in getattr(self, which).split(","))


@dataclass
class Phase:
    """One GEMM: a maximal run of *adjacent* mma statements.  NVIDIA emits a
    prologue, one contiguous mma block and an epilogue per GEMM, so the gaps
    between these runs are exactly what has to be classified."""

    idx: int
    first: int          # statement index of the first mma
    last: int           # statement index of the last mma
    mma: list = field(default_factory=list)
    M: int = 0
    N: int = 0
    K: int = 0
    n_tiles: int = 0    # 8-wide N tiles
    k_slots: int = 0    # 32-deep K steps per N tile
    w_min: int = 0
    w_max: int = 0
    w_known: bool = False
    bias: tuple = ()
    epilogue: int = 0
    epilogue_range: tuple = (0, 0)
    a_tiles: int = 0
    b_tiles: int = 0
    a_regs: tuple = ()
    b_regs: tuple = ()
    c_regs: tuple = ()
    d_regs: tuple = ()

    @property
    def nmma(self) -> int:
        return len(self.mma)

    @property
    def rank(self) -> bool:
        """mma count equals (M/16)*(N/8)*(K/32)?  i.e. one accumulation pass."""
        return self.nmma == (self.M // 16) * (self.N // 8) * (self.K // 32)

    @property
    def weight_bytes(self) -> int:
        return self.w_max - self.w_min + 4 if self.w_known else 0


def collect_mma(stmts, defs):
    """Every mma, with its B and C operands resolved against the weight image at
    *that* mma's program point."""
    out = []
    for st in stmts:
        m = RE_MMA.match(st.text)
        if not m:
            continue
        d = Mma(idx=st.idx, line=st.line, D=m.group(1), A=m.group(2),
                B=m.group(3), C=m.group(4))
        d.bo = tuple(defs.from_weight(r, d.idx) for r in d.regs("B"))
        d.co = tuple(defs.from_weight(r, d.idx) for r in d.regs("C"))
        out.append(d)
    return out


def build_phases(stmts, mmas):
    phases = []
    cur = None
    for m in mmas:
        if cur is not None and m.idx == cur.mma[-1].idx + 1:
            cur.mma.append(m)
        else:
            cur = Phase(idx=len(phases) + 1, first=m.idx, last=m.idx, mma=[m])
            phases.append(cur)

    for p in phases:
        p.last = p.mma[-1].idx
        a_quads, b_pairs, c_pairs = [], [], []
        for m in p.mma:
            if m.A not in a_quads:
                a_quads.append(m.A)
            if m.B not in b_pairs:
                b_pairs.append(m.B)
            if m.C not in c_pairs:
                c_pairs.append(m.C)
        p.a_tiles, p.b_tiles = len(a_quads), len(b_pairs)
        p.M = 16 * p.a_tiles
        p.a_regs = p.mma[0].regs("A")
        p.b_regs = p.mma[0].regs("B")
        p.c_regs = p.mma[0].regs("C")
        p.d_regs = p.mma[0].regs("D")

        offs = sorted({o for m in p.mma for o in m.bo if o is not None})
        if offs:
            base = offs[0]
            p.w_min, p.w_max, p.w_known = base, offs[-1], True
            # NVIDIA packs one K=32 slice of an N=8 column tile in 512 bytes and
            # steps 8 bytes per K=32; the residue mod 512 identifies the K step
            # and the quotient the N tile.
            p.k_slots = len({(o - base) % NTILE_STRIDE for o in offs
                             if (o - base) % NTILE_STRIDE < KSTEP * 8})
            p.n_tiles = 1 + max((o - base) // NTILE_STRIDE for o in offs)
            p.N = 8 * p.n_tiles
            p.K = 32 * max(1, p.k_slots)
        p.bias = tuple(sorted({o for m in p.mma for o in m.co if o is not None}))

    for i, p in enumerate(phases):
        if i + 1 < len(phases):
            nxt = phases[i + 1].first
            p.epilogue = nxt - p.last - 1
            p.epilogue_range = (p.last + 1, nxt - 1)
        else:
            p.epilogue = len(stmts) - p.last
            p.epilogue_range = (p.last + 1, len(stmts))
    return phases


# ---------------------------------------------------------------------------
# stage classification evidence
# ---------------------------------------------------------------------------

EVIDENCE_OPS = {
    "normalise": ("rsqrt.approx", "rcp.approx", "ex2.approx", "shfl.sync", "bar.warp"),
    "softmax": ("ex2.approx", "lg2.approx", "max.f32", "add.f32", "rcp.approx"),
    "tex": ("tex.",),
    "surface": ("st.global", "st.shared", "red.global", "atom.global"),
    "shared": ("ld.shared", "st.shared", "bar.sync"),
    "mma": ("mma.sync",),
    "e4m3": ("cvt.rn.satfinite.e4m3x2.f16x2", "cvt.rn.satfinite.relu.e4m3x2.f16x2"),
    "movmatrix": ("movmatrix.sync",),
}


def op_histogram(stmts):
    h = {}
    for s in stmts:
        h[s.op] = h.get(s.op, 0) + 1
    return h


def evidence(stmts, lo, hi, keys, limit=6):
    """Top ops mentioning any of `keys` in the statement range [lo, hi] (1-based,
    inclusive), as evidence strings a reader can check."""
    out = []
    for s in stmts[lo - 1:hi]:
        for k in keys:
            if k in s.op:
                out.append(f"s{s.idx}(l{s.line}) {s.op}")
                break
    seen = {}
    for e in out:
        seen[e.split()[1]] = seen.get(e.split()[1], 0) + 1
    head = sorted(seen.items(), key=lambda kv: -kv[1])[:limit]
    return ", ".join(f"{op}x{n}" for op, n in head)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def chains_of(phase):
    """Partition a phase's mma into independent accumulator chains.

    An mma joins the chain that produced one of its C registers (D -> C
    dataflow).  Each chain is one output 16x8 tile of the m16n8k32 tile grid;
    a chain of length L accumulates K = 32*L.
    """
    owner = {}
    out = []
    for m in phase.mma:
        hit = {owner[r] for r in m.regs("C") if r in owner}
        if len(hit) == 1:
            ci = hit.pop()
        else:
            out.append([])
            ci = len(out) - 1
        out[ci].append(m)
        for r in m.regs("D"):
            owner[r] = ci
    return out


def geom(phase):
    """(M, N, K, n_out_tiles) for a phase, from the fragment structure alone.

    A m16n8k32 fragment is 16 rows (A, 4 regs), 8 columns (B, 2 regs) and
    32 deep (K, one mma).  So M = 16 * distinct A fragments, N = 8 * distinct
    B fragments, and K = 32 * chain length.  `dense` is False when the phase
    does not evaluate the full M x N grid of A x B fragments - that is the
    signature of a banded/diagonal attention block.
    """
    chains = chains_of(phase)
    nA = len({m.A for m in phase.mma})
    nB = len({m.B for m in phase.mma})
    lens = sorted({len(c) for c in chains})
    return dict(M=16 * nA, N=8 * nB, nA=nA, nB=nB, nchains=len(chains),
                klens=lens, K=(32 * lens[0] if len(lens) == 1 else None),
                tiles=len(chains), dense=len(phase.mma) == nA * nB * lens[0])


def a_source(phase, defs, stmts):
    """Where each of the phase's A fragments came from, following the def-use
    chain back to the memory operation that produced it.  A fragments reach the
    mma through `mov.b32 %rX, {%rsA, %rsB}` packs of two b16 registers written
    by `cvt.rn.satfinite.e4m3x2.f16x2`, so a one-hop lookup is not enough.

    Returns {kind: count}, where kind is the memory op's mnemonic family:
    `ld.shared` (an earlier mma result staged through shared memory),
    `ld.global`/`ld.weak.global` (an input activation plane), `tex` (a texture
    fetch), or `mma.sync` (a value produced inside this layer's mma region).
    """
    kinds = {}
    for a in phase.a_regs:
        got = defs.walk_origin(a, phase.first, stmts)
        if got is None:
            key = "unresolved"
        else:
            op, _, packed = got
            if op.startswith("mma.sync"):
                key = "mma"
            elif op.startswith("ld.shared"):
                key = "ld.shared"
            elif op.startswith("ld."):
                key = "ld.global"
            elif op.startswith("tex."):
                key = "tex"
            else:
                key = op.split(".")[0]
            if packed and key != "unresolved":
                key += "+e4m3"
        kinds[key] = kinds.get(key, 0) + 1
    return kinds


OP_EVIDENCE = {
    "e4m3-pack": ("cvt.rn.satfinite.e4m3x2.f16x2", "cvt.rn.satfinite.relu.e4m3x2.f16x2"),
    "transpose": ("movmatrix.sync.trans.aligned.m8n8.b16",),
    "softmax": ("ex2.approx.ftz.f32", "lg2.approx.ftz.f32"),
    "reciprocal": ("rcp.approx.ftz.f32",),
    "normalise": ("rsqrt.approx.ftz.f32",),
    "shared-staging": ("ld.shared", "st.shared", "bar.sync"),
    "surface-write": ("st.global", "st.shared", "red.global"),
    "texture-read": ("tex.",),
}


def op_evidence(stmts, lo, hi):
    """{stage: (count, first statement, first line)} for a statement range."""
    out = {}
    for s in stmts[lo - 1:hi]:
        for stage, keys in OP_EVIDENCE.items():
            if any(k in s.op for k in keys):
                c, first, line = out.get(stage, (0, 0, 0))
                out[stage] = (c + 1, first or s.idx, line or s.line)
                break
    return out


def fmt_ev(ev):
    if not ev:
        return "-"
    return " ".join(f"{k}:{v[0]}@s{v[1]}" for k, v in sorted(ev.items()))


def chain_summary(phase, limit=2):
    """The A x B fragment grid as a checkable picture."""
    ao, bo = {}, {}
    for m in phase.mma:
        ao.setdefault(m.A, len(ao))
        bo.setdefault(m.B, len(bo))
    used = {(ao[m.A], bo[m.B]) for m in phase.mma}
    rows = []
    for r in range(len(ao)):
        rows.append("".join("X" if (r, c) in used else "." for c in range(len(bo))))
    return ao, bo, rows[:limit]


def print_phase_table(phases):
    print()
    print("phase table  (a phase is one maximal run of adjacent mma statements;")
    print(" 'epi' = statements after its last mma and before the next phase's first)")
    print(f"{'#':>3} {'stmt range':>13} {'M':>4} {'N':>4} {'K':>5} {'tiles':>5} "
          f"{'mma':>4} {'grid':>5} {'weight bytes':>22} {'bias bytes':>26} {'epi':>6}")
    for p in phases:
        g = p.geo
        wb = f"{p.w_min}..{p.w_max}" if p.w_known else "?"
        bb = (f"{p.bias[0]}..{p.bias[-1]}" if p.bias else "-")
        K = str(g["K"]) if g["K"] else str(g["klens"])
        print(f"{p.idx:>3} s{p.first:>5}-s{p.last:<5} {g['M']:>4} {g['N']:>4} {K:>5} "
              f"{g['tiles']:>5} {p.nmma:>4} {'full' if g['dense'] else 'band':>5} "
              f"{wb:>22} {bb:>26} {p.epilogue:>6}")


def print_classification(stmts, phases, defs):
    print()
    print("stage classification")
    print("  Each entry is a statement range plus the operations in it.  Counts are")
    print("  exact; `sNNN` is the first statement index in the range.")

    def show(name, lo, hi, note=""):
        ev = op_evidence(stmts, lo, hi)
        print(f"  {name:22s} s{lo}-s{hi} ({hi - lo + 1:5d} stmts)  {fmt_ev(ev)}")
        if note:
            print(f"  {'':22s} {note}")

    show("prologue", 1, phases[0].first - 1)
    for p in phases:
        lo, hi = p.epilogue_range
        g = p.geo
        src = a_source(p, defs, stmts)
        note = (f"M={g['M']} N={g['N']} {'K=' + str(g['K']) if g['K'] else 'K mixed'} "
                f"tiles={g['tiles']} A-src={src}")
        show(f"after phase {p.idx}", lo, hi, note)


SPECIAL = ("%tid.x", "%tid.y", "%tid.z", "%ctaid.x", "%ctaid.y", "%ctaid.z",
           "%laneid", "%warpid", "%nwarpid")


def kernel_entry_facts(stmts, lines, body):
    """Launch shape and parameter-block facts, read straight off the PTX."""
    first = lines[body[0]]
    psz = re.search(r"param_0\[(\d+)\]", lines[body[0] + 1])
    maxntid = next((l.strip() for l in lines[body[0]:body[0] + 40]
                    if ".maxntid" in l), None)
    offs = collections.Counter()
    for s in stmts:
        for m in re.finditer(r"_param_0\+(\d+)\]", s.text):
            offs[int(m.group(1))] += 1
    used = [r for r in SPECIAL if any(r in s.text for s in stmts)]
    shared = re.findall(r"\.shared[^\n;]*?(\w+)\[(\d+)\]",
                        "\n".join(lines[body[0]:body[0] + 12]))
    return dict(entry=first.strip(), param_bytes=int(psz.group(1)) if psz else None,
                maxntid=maxntid, param_offsets=sorted(offs), specials=used,
                shared=shared, reads=sum(offs.values()))


def live_across(stmts, lo, hi):
    """Registers written inside [lo, hi] and read after it.  Those are the
    values a native replacement must hand back to the surrounding PTX."""
    written = set()
    for s in stmts[lo - 1:hi]:
        written.update(dest_regs(s.text))
    after = set()
    for s in stmts[hi:]:
        after.update(read_regs(s.text))
        after.update(r for g in RE_OPERAND.findall(s.text) for r in REG_SRC.findall(g))
    return sorted(written & after)


RE_OPERAND = re.compile(r"\{([^}]*)\}")


def print_cut_abi(stmts, phases, defs, entry, path, lines, body):
    mmas = [m for p in phases for m in p.mma]
    first, last = mmas[0], mmas[-1]
    lo, hi = first.idx, last.idx
    k = kernel_entry_facts(stmts, lines, body)

    print()
    print("cut ABI")
    print()
    print("  (a) kernel entry - what a NAME.hsaco built from this module receives")
    print(f"      launch         {k['maxntid'] or 'no .maxntid in this module'}; "
          f"reads {k['specials']}")
    print(f"      param block    {k['param_bytes']} bytes "
          f"({k['reads']} ld.param over {len(k['param_offsets'])} distinct offsets)")
    print(f"      param offsets  {k['param_offsets']}")
    print(f"      shared         {k['shared'] or 'none'}")
    print("      weights        arg040, the prepared weight image; every byte offset")
    print("                      in the phase table is relative to its base")

    print()
    print("  (b) mma region - the cut that keeps texture reads and the surface")
    print("      epilogue in PTX and sends the GEMMs to a native device function")
    print(f"      cut            s{lo}-s{hi}  (file lines {first.line}-{last.line}), "
          f"{len(mmas)} mma in {len(phases)} phases")
    print(f"      entry          s{lo}: {stmts[lo - 1].text}")
    a_in, b_in, c_in = set(), set(), set()
    for m in phases[0].mma:
        a_in.update(m.regs("A"))
        b_in.update(m.regs("B"))
        c_in.update(m.regs("C"))
    print(f"      A in           {len(a_in)} regs (6 fragments x 4): "
          f"{sorted(a_in, key=lambda r: int(r[2:]))}")
    print(f"      B in           {len(b_in)} regs (4 fragments x 2): {sorted(b_in, key=lambda r: int(r[2:]))}")
    print(f"      C in           {sorted(c_in, key=lambda r: int(r[2:]))} "
          f"(bias seeds, read from the weight image at +{phases[0].bias})")
    live = live_across(stmts, lo, hi)
    print(f"      returns        {len(live)} accumulators written in the region and "
          f"read after s{hi}")
    print(f"                    {live[:12]}{' ...' if len(live) > 12 else ''}")
    weight_b = sorted({m.B for p in phases for m in p.mma})
    print(f"      weight image   {len(weight_b)} distinct B fragments; byte ranges "
          f"per phase are in the table above")
    print("      note           a NAME_prep companion may permute the weight image")
    print("                      once, so the native side is free to consume it in")
    print("                      its own fragment order rather than NVIDIA's")

def weight_image_size(stmts, defs):
    """Size of the prepared weight image: the largest byte offset actually
    touched, plus the width of the widest access at that offset."""
    widest = {}
    for v in defs.defs.values():
        for i, info in v:
            if info[0] != "W":
                continue
            g = stmts[i].op.split(".")[-1]
            w = {"u8": 1, "u16": 2, "v2.u16": 4, "u32": 4, "v2.u32": 8,
                 "v4.u32": 16, "v4": 16}.get(g, 4)
            widest[info[1]] = max(widest.get(info[1], 0), w)
    if not widest:
        return None, 0
    top = max(widest)
    return top + widest[top], len(widest)


def print_geometry(name, stmts, defs, phases):
    size, nslots = weight_image_size(stmts, defs)
    geos = [p.geo for p in phases]
    print(f"{name:6s} mma={sum(p.nmma for p in phases):5d} "
          f"phases={len(phases):3d} image={str(size):>7} slots={nslots:4d} "
          f"M={sorted({g['M'] for g in geos})} N={sorted({g['N'] for g in geos})}")


def geometry_all():
    """Per-layer geometry for every interior layer in the corpus."""
    order = ["enc0", "enc1", "enc2", "enc3", "enc4", "enc5",
             "dec0", "dec1", "dec2", "dec3", "dec4"]
    rows = []
    for alias in order:
        fn, ent = ALIASES.get(alias, (None, None))
        if ent is None:
            continue
        lines = open(os.path.join(CORPUS, fn)).read().split("\n")
        body = next(((a, b) for n, a, b in find_entries(lines) if n == ent), None)
        if body is None:
            continue
        stmts, _ = linearise(lines, *body)
        defs = track_weights(stmts, ent)
        phases = build_phases(stmts, collect_mma(stmts, defs))
        for p in phases:
            p.geo = geom(p)
        rows.append((alias, fn, stmts, defs, phases))
    print(f"{'layer':6s} {'file':32s} {'mma':>5} {'ph':>3} {'image':>8} {'slots':>6}")
    for alias, fn, stmts, defs, phases in rows:
        size, nslots = weight_image_size(stmts, defs)
        print(f"{alias:6s} {fn:32s} {sum(p.nmma for p in phases):>5} "
              f"{len(phases):>3} {size:>8} {nslots:>6}")
    print()
    print("mma tile shapes (M values and N values each layer produces)")
    for alias, fn, stmts, defs, phases in rows:
        geos = [p.geo for p in phases]
        print(f"{alias:6s} M={sorted({g['M'] for g in geos})} "
              f"N={sorted({g['N'] for g in geos})} "
              f"K={sorted({g['K'] for g in geos if g['K']})} "
              f"banded={sum(1 for g in geos if not g['dense'])}")

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer", nargs="?", help="alias (enc0..enc5, dec0) or layer name")
    ap.add_argument("--file", help="PTX file (overrides the corpus alias)")
    ap.add_argument("--entry", help="entry name inside --file")
    ap.add_argument("--list", action="store_true", help="print the corpus alias table")
    ap.add_argument("--geometry", action="store_true",
                    help="per-layer geometry table for every interior layer")
    args = ap.parse_args(argv)

    if args.geometry:
        return geometry_all()


    if args.list or not args.layer:
        for k, (f, e) in ALIASES.items():
            print(f"{k:8s} {f}  {e}")
        return 0

    path, entry = resolve(args.layer, args.file, args.entry)
    if path is None:
        return 2
    lines = open(path).read().split("\n")
    body = next(((a, b) for n, a, b in find_entries(lines) if n == entry), None)
    if body is None:
        print(f"entry {entry} not in {path}", file=sys.stderr)
        return 2

    stmts, leftover = linearise(lines, *body)
    defs = track_weights(stmts, entry)
    mmas = collect_mma(stmts, defs)
    phases = build_phases(stmts, mmas)
    for p in phases:
        p.geo = geom(p)

    print(f"layer {args.layer}: {entry}")
    print(f"file  {path}")
    print(f"statements {len(stmts)} (unterminated tail: {leftover!r})")
    h = op_histogram(stmts)
    for op in sorted(h, key=lambda o: -h[o])[:12]:
        print(f"    {op:62s} {h[op]}")
    print_phase_table(phases)
    print_classification(stmts, phases, defs)
    print_cut_abi(stmts, phases, defs, entry, path, lines, body)
    return 0



if __name__ == "__main__":
    sys.exit(main())