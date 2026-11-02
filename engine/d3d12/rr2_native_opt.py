"""HLSL rewrites for the native RR network stages (preset F).

The builder calls :func:`optimize` on the HLSL that ``rr_hlsl.translate``
produced, before DXC sees it.  Every pass is a local textual rewrite of the
generated stage; resource bindings, barriers, the shared ABI, the workgroup
shape are unchanged. Rounding follows gfx12; the FP8 NaN behavior change is
documented below. A pass whose predicate does not match returns its input
unchanged, and source without a generated entry point is left untouched.

``quant_fusion`` (exact)
    The translator leaves ``fe4m3vec2 t_; saturatedConvertEXT(t_, S);`` in
    front of a quantization that runs again on the same values
    (``rrQuantSet`` or ``pack4``).  ``saturatedConvertEXT`` is ``rrQuant(s)``:
    an integer round-to-nearest-even codec (about a dozen VALU per element),
    while ``rrQuantSet`` is one AGS matrix conversion and ``pack4`` is
    ``rrCode`` per element.  ``code(rrQuant(x)) == code(x)`` for every f32 --
    the same e4m3 grid, tie-break and 448 saturation, verified over 77M inputs
    including all subnormals, every grid point +-1 ulp and NaN/inf -- and the
    ``isinf`` -> NaN select is kept, so a NaN reaches the consumer's clamp in
    both forms.  The first conversion is dead work and is dropped.

``half_bits_roundtrip`` (exact)
    ``float16_t(f16tof32(w))`` widens the binary16 bits ``w`` to f32 and
    rounds them back through ``d4rRne16`` (an AGS conversion).  The widening
    is exact and the round trip returns ``w``, so the expression is
    ``asfloat16(uint16_t(w))``: the f16 bits without the conversion pair.

``e4pair_native_half`` (exact)
    ``e4pair`` decodes two FP8 codes through ``rrDecode2`` (AGS B8 -> Acc16
    plus two ``f16tof32``) and then builds an ``f16vec2`` from the widened
    values, which rounds each one through ``d4rRne16`` again.  The helper is
    replaced by one that takes the two binary16 values straight out of the
    conversion's register, dropping the f32 round trip.

``pack4_quantset`` (quality sensitive)
    A local ``floate4m3_t X[8]`` filled by a pair loop and stored through two
    ``pack4`` calls (``dec0_0``'s output word, ``dec*_0``'s XT skip) encodes
    every element with ``rrCode``, eight software codecs per word.  One
    ``rrQuantSet`` conversion produces the same two words: both paths round to
    nearest even on the same e4m3 grid and saturate at 448.  Difference kept
    in mind: ``rrQuantSet``'s clamp maps a NaN residual to the bound instead of
    the NaN code. The decoder's ``Nb`` tensor already uses that conversion
    for the same residual class. End-to-end replay and CUDA-comparison
    measurements, including the output-word path, are in windows-rr-tuning.md.

``inf_select`` (relaxed, infinity only)
    The ``q8s`` templates guard each saturating FP8 conversion with
    ``isinf(x) ? NaN : x``. On gfx12 the D3D12 compiler lowers the select to a
    class compare, an SGPR mask inversion, an SALU->VALU hazard wait and a
    VALU select per element: about a tenth of a network stage's dynamic
    instructions. RADV's saturating conversion needs no such guard. The
    consumer is the clamp-only quantization (``RR_QUANT_CLAMPONLY``), which
    already bounds every finite value at +-448 and turns NaN into a bound;
    without the select an infinity saturates to its own sign's bound instead
    of becoming that NaN bound (-448). Finite values and NaN are unchanged.

``residual_pack`` (exact)
    ``dec0_0`` holds its f16 residual ``xr[4][2][8]`` across the head loop.
    The D3D12 compiler sinks the residual sum past the loop and keeps both
    operands live, one half per VGPR: 255 VGPRs requested, 264 allocated
    (granule 24), five waves per SIMD against RADV's seven.  The array is
    stored two halves per ``uint`` and pinned before the loop by a no-op
    loop-carried XOR; 216 VGPRs, seven waves, about 14% off the stage.

``branch_pin`` (exact)
    The same sink in ``dec0_1``: the packed f16 residual ``rr2_r16`` is
    unpacked to the f32 ``rv`` across the branch loop.  The no-op XOR pins the
    packed words; 192 VGPRs, eight waves, about 3.5% off the stage.

``half_rne8`` (exact)
    The branch bias of ``enc0``/``dec0`` and the beta path of every
    ``enc*``/``dec*`` stage converts each of eight MMA accumulator elements
    with ``float16_t(x)`` (``d4rRne16``: a pair AGS conversion plus two
    widenings per element) only to use the bits as a table index.
    ``half_rne8`` emits one ``rrHalfRne8`` helper (a single AGS Acc32 -> Acc16
    conversion; element 2i/2i+1 in ``reg[i]``, the order ``unpackHalf2x16``
    gives) and rewrites both spellings (a separate ``c[i] = float16_t(p_[i])``
    loop and the inline ``bh(float16_t(p_[i]))``).  Per-element rounding is
    unchanged.
"""
from __future__ import annotations

import re

__all__ = ['optimize', 'optimize_with_report', 'PASSES', 'ORDER']


def _close(s: str, start: int) -> int:
    """Index of the delimiter closing the one at ``start`` (()[], {})."""
    pairs = {'(': ')', '[': ']', '{': '}'}
    opens = set(pairs)
    stack = [pairs[s[start]]]
    for i in range(start + 1, len(s)):
        c = s[i]
        if c in opens:
            stack.append(pairs[c])
        elif stack and c == stack[-1]:
            stack.pop()
            if not stack:
                return i
    raise ValueError('unbalanced expression')

def _split_args(s: str) -> list[str]:
    """Split a call's argument text at top-level commas."""
    out, depth, at = [], 0, 0
    for i, c in enumerate(s):
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == ',' and depth == 0:
            out.append(s[at:i].strip())
            at = i + 1
    return out + [s[at:].strip()]

def _body_start(s: str) -> int | None:
    """First index of a generated entry point, so helpers stay untouched."""
    m = re.search(r'\nvoid\s+(?:main|rr2_block)\s*\(\s*\)', s)
    return m.start() if m else None

# --------------------------------------------------------------------------- quant_fusion

_LOOP = re.compile(r'\[unroll\]\s*for\s*\(\s*int\s+(i_?)\s*=\s*0\s*;\s*\1\s*<\s*8\s*;\s*\1\s*\+=\s*2\s*\)\s*\{')
_CONV = re.compile(r'fe4m3vec2\s+t_\s*;\s*saturatedConvertEXT')
# NAME[idx] = t_.x; NAME[idx + 1] = t_.y; }   (parentheses around the name or
# the index are optional in the emitted text)
_ASSIGN = re.compile(
    r'\s*;?\s*\(?\s*([A-Za-z_]\w*)\s*\)?\s*\[\s*\(?\s*(i_?)\s*\)?\s*\]\s*=\s*t_\.x\s*;\s*'
    r'\(?\s*([A-Za-z_]\w*)\s*\)?\s*\[\s*\(?\s*(i_?)\s*\)?\s*\+\s*1\s*\]\s*=\s*t_\.y\s*;\s*\}')
_SETTER = r'(?:\(?[A-Za-z_]\w*(?:\s*\[[^\]]*\])*\)?\s*=\s*)?'
_CONSUMER_SET = re.compile(r'\s*\}?\s*' + _SETTER + r'rrQuantSet\s*\(\s*%s\s*\)')
_DECL = re.compile(r'(?:float|RRFloat|floate4m3_t|float16_t)\s+$')


def _quant_uses(body: str, name: str, allowed: list[tuple[int, int]],
                lo: int = 0, hi: int | None = None) -> bool:
    """Every mention of ``name`` in [lo, hi) is a declaration or allowed."""
    hi = len(body) if hi is None else hi
    for m in re.finditer(r'\b' + re.escape(name) + r'\b', body):
        if not lo <= m.start() < hi:
            continue
        if _DECL.search(body[max(0, m.start() - 40):m.start()]):
            continue
        if any(a <= m.start() < b for a, b in allowed):
            continue
        return False
    return True


def _block_span(s: str, pos: int) -> tuple[int, int]:
    """Span of the innermost brace block that contains ``pos``."""
    depth = 0
    for i in range(pos - 1, -1, -1):
        if s[i] == '}':
            depth += 1
        elif s[i] == '{':
            if depth == 0:
                return i, _close(s, i)
            depth -= 1
    return 0, len(s)


def quant_fusion(source: str) -> tuple[str, int]:
    # The conversion is only dead when the same float array is quantized again
    # right after the loop.  Restrict the rewrite to the emitted templates.
    out, pos, sites = [], 0, 0
    while True:
        m = _CONV.search(source, pos)
        if not m:
            out.append(source[pos:])
            break
        loop = None
        for cand in _LOOP.finditer(source, max(0, m.start() - 4000), m.start()):
            try:
                if _close(source, cand.end() - 1) > m.start():
                    loop = cand
            except ValueError:
                continue
        if loop is None:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        arg = m.end()
        while arg < len(source) and source[arg].isspace():
            arg += 1
        if arg >= len(source) or source[arg] != '(':
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        try:
            s_end = _close(source, arg)
        except ValueError:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        conv = _split_args(source[arg + 1:s_end])
        if len(conv) != 2 or conv[0] != 't_':
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        expr = conv[1]
        a = _ASSIGN.match(source, s_end + 1)
        if not a or a.group(1) != a.group(3) or a.group(2) != a.group(4):
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        name = a.group(1)
        # The consumer: rrQuantSet over the array, or the two pack4 words.
        tail = source[a.end():a.end() + 800]
        consumer = _CONSUMER_SET.pattern % re.escape(name)
        cm = re.match(consumer, tail)
        span_end = a.end() + (cm.end() if cm else 0)
        if not cm:
            p0 = tail.find('pack4(' + name + '[0]')
            p1 = tail.find('pack4(' + name + '[4]')
            if p0 < 0 or p1 < 0 or p1 < p0:
                out.append(source[pos:m.end()])
                pos = m.end()
                continue
            semi = tail.find(';', p1)
            span_end = a.end() + (semi + 1 if semi >= 0 else p1)
        allowed = [(loop.start(), a.end()), (a.end(), span_end)]
        lo, hi = _block_span(source, loop.start())
        if not _quant_uses(source, name, allowed, lo, hi + 1):
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        out.append(source[pos:m.start()])
        out.append('const float2 rrCv%d_ = %s; %s[%s] = rrCv%d_.x; %s[(%s) + 1] = rrCv%d_.y; }'
                   % (sites, expr, name, a.group(2), sites, name, a.group(2), sites))
        pos = a.end()
        sites += 1
    return ''.join(out), sites


# --------------------------------------------------------------------------- half_bits_roundtrip

_WIDEN = re.compile(r'float16_t\s*\(\s*f16tof32\s*\(')


def half_bits_roundtrip(source: str) -> tuple[str, int]:
    out, pos, sites = [], 0, 0
    while True:
        m = _WIDEN.search(source, pos)
        if not m:
            out.append(source[pos:])
            break
        try:
            end = _close(source, m.end() - 1)
        except ValueError:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        close = re.match(r'\s*\)', source[end + 1:])
        if not close:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        out.append(source[pos:m.start()])
        out.append('asfloat16(uint16_t(%s))' % source[m.end():end])
        pos = end + 1 + close.end()
        sites += 1
    return ''.join(out), sites


# --------------------------------------------------------------------------- e4pair_native_half

_E4PAIR = re.compile(
    r'f16vec2\s+e4pair\s*\(\s*uint\s+w\s*,\s*uint\s+h\s*\)\s*\{\s*'
    r'return\s+f16vec2\s*\(\s*uintBitsToFloate4m3EXT\s*\(\s*h\s*==\s*0u\s*\?\s*'
    r'unpack8\s*\(\s*w\s*\)\.xy\s*:\s*unpack8\s*\(\s*w\s*\)\.zw\s*\)\s*\)\s*;\s*\}')
_E4PAIR_NEW = (
    'f16vec2 e4pair(uint w, uint h)\n'
    '{\n'
    '#if D4R_VKD3D\n'
    '    // vkd3d-proton: B8 -> Acc32 of the whole word (byte i in register i,\n'
    '    // one conversion for both halves after CSE) and a plain cast, which\n'
    '    // RADV drops into an f32 consumer as the Vulkan build\'s f16vec2 decode;\n'
    '    // through Acc16 the binary16 conversion and a mode switch per pair stay.\n'
    '    const Matrix16 r = convert(MatrixB8, MatrixAcc32, 0, m16(uint2(w, 0u)));\n'
    '    return h == 0u ? f16vec2(float16_t(asfloat(r.reg[0])), float16_t(asfloat(r.reg[1])))\n'
    '                   : f16vec2(float16_t(asfloat(r.reg[2])), float16_t(asfloat(r.reg[3])));\n'
    '#else\n'
    '    // The codes are already the low/high half of w; one B8 -> Acc16\n'
    '    // conversion yields both binary16 values, with no f32 round trip.\n'
    '    const uint word = h == 0u ? (w & 0xffffu) : (w >> 16u);\n'
    '    const Matrix16 r = convert(MatrixB8, MatrixAcc16, 0, m16(uint2(word, 0u)));\n'
    '    return f16vec2(asfloat16(uint16_t(r.reg[0] & 0xffffu)), asfloat16(uint16_t(r.reg[0] >> 16u)));\n'
    '#endif\n'
    '}')


def e4pair_native_half(source: str) -> tuple[str, int]:
    return _E4PAIR.subn(_E4PAIR_NEW, source)


# --------------------------------------------------------------------------- pack4_quantset

_PACK4_DECL = re.compile(r'floate4m3_t\s+(\w+)\s*\[\s*8\s*\]\s*;')


def _stmt_start(s: str, pos: int, lo: int) -> int:
    i = pos
    while i > lo:
        if s[i - 1] in ';{}':
            return i
        i -= 1
    return lo


def pack4_quantset(source: str) -> tuple[str, int]:
    """One AGS conversion instead of eight ``rrCode`` codecs (opt-in).

    A local ``floate4m3_t NAME[8]`` filled by a pair loop and stored through
    two ``pack4`` calls is quantized again by ``pack4`` (``rrCode`` per
    element).  ``rrQuantSet(NAME)`` produces the same two code words: both
    paths are nearest-even with saturation at 448, and the residual values
    that reach here are finite.  The pass replaces the declaration, inserts
    one ``rrQuantSet`` before the first ``pack4`` statement and reads the two
    words from it.  Quality delta: a NaN residual becomes the clamp bound
    instead of the NaN code, and fp8 subnormals rely on the AGS conversion
    (the same one the decoder's ``Nb`` tensor already uses).
    """
    out, pos, sites = [], 0, 0
    while True:
        m = _PACK4_DECL.search(source, pos)
        if not m:
            out.append(source[pos:])
            break
        name = m.group(1)
        lo, hi = _block_span(source, m.start())
        loop = _LOOP.search(source, m.end(), hi)
        body_end = -1
        if loop is not None:
            try:
                body_end = _close(source, loop.end() - 1)
            except ValueError:
                body_end = -1
        calls = []
        if body_end > 0:
            call = re.compile(r'pack4\s*\(\s*' + re.escape(name) + r'\s*\[')
            for cm in call.finditer(source, body_end, hi):
                open_ = source.index('(', cm.start())
                try:
                    calls.append((cm.start(), open_, _close(source, open_)))
                except ValueError:
                    calls = []
                    break
        if len(calls) != 2 or body_end < 0:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        args = [_split_args(source[o + 1:e]) for _, o, e in calls]
        if args != [['%s[0]' % name, '%s[1]' % name, '%s[2]' % name, '%s[3]' % name],
                    ['%s[4]' % name, '%s[5]' % name, '%s[6]' % name, '%s[7]' % name]]:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        # every other mention of the array is an assignment inside the loop
        ok = True
        for u in re.finditer(r'\b' + re.escape(name) + r'\b', source[lo:hi + 1]):
            at = lo + u.start()
            if at <= m.end() or body_end > at >= loop.start():
                continue
            if any(c[0] <= at <= c[2] for c in calls):
                continue
            ok = False
            break
        if not ok:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        ins = _stmt_start(source, calls[0][0], lo)
        edit = []
        edit.append((m.start(), m.end(), 'RRFloat %s;' % name))
        edit.append((ins, ins, ' const RRByte %s_q = rrQuantSet(%s); ' % (name, name)))
        edit.append((calls[0][0], calls[0][2] + 1, '%s_q.x' % name))
        edit.append((calls[1][0], calls[1][2] + 1, '%s_q.y' % name))
        edit.sort()
        out.append(source[pos:m.start()])
        cur = m.start()
        for a, b, text in edit:
            out.append(source[cur:a])
            out.append(text)
            cur = b
        out.append(source[cur:max(b for _, b, _ in edit)])
        pos = max(b for _, b, _ in edit)
        sites += 1
    return out and ''.join(out) or source, sites


# --------------------------------------------------------------------------- half_rne8

_HALF8 = re.compile(
    r'\[unroll\]\s*for\s*\(\s*int\s+(i_?)\s*=\s*0\s*;\s*\1\s*<\s*8\s*;\s*\+\+\s*\1\s*\)\s*'
    r'([A-Za-z_]\w*)\s*\[\s*\1\s*\]\s*=\s*float16_t\s*\(\s*([A-Za-z_]\w*)\s*\[\s*\1\s*\]\s*\)\s*;')
_HALF8_INLINE = re.compile(
    r'\[unroll\]\s*for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*8\s*;\s*\+\+\s*i\s*\)\s*'
    r'[A-Za-z_]\w*(?:\s*\[[^\]]*\])+\s*=\s*hb\s*\(\s*uint\s*\(\s*rrB1\.Load<uint16_t>\s*\(')
_HALF8_HELPER = (
    '// One AGS Acc32 -> Acc16 conversion for eight values: the scalar\n'
    '// float16_t(x) spelling widens through d4rRne16 once per element.\n'
    'RRHalf rrHalfRne8(RRFloat v)\n'
    '{\n'
    '    const Matrix16 h = convert(MatrixAcc32, MatrixAcc16, 0, f32m(v));\n'
    '    RRHalf r;\n'
    '    [unroll] for (uint i = 0; i < 4; ++ i) {\n'
    '        r[2u * i] = asfloat16(uint16_t(h.reg[i] & 0xffffu));\n'
    '        r[2u * i + 1u] = asfloat16(uint16_t(h.reg[i] >> 16u));\n'
    '    }\n'
    '    return r;\n'
    '}\n')


def _declared(s: str, name: str, kind: str, pos: int, window: int = 6000) -> bool:
    """True when the nearest declaration of ``name`` before ``pos`` is ``kind``."""
    pat = re.compile(r'\b' + kind + r'\b[^;{}]*?\b' + re.escape(name) + r'\s*[\[=;,)]')
    return pat.search(s[max(0, pos - window):pos]) is not None


def _inject_half8(source: str) -> str:
    if 'RRHalf rrHalfRne8' in source:
        return source
    m = re.search(r'#include\s+"rr_wmma\.hlsl[^"]*"\n', source)
    at = m.end() if m else None
    if at is None:
        b = _body_start(source)
        at = b if b is not None else 0
    return source[:at] + '\n' + _HALF8_HELPER + source[at:]


def half_rne8(source: str) -> tuple[str, int]:
    """Batch the scalar f32 -> binary16 RNE conversions of the beta path.

    The model converts each of the eight MMA accumulator elements with
    ``float16_t(x)`` (``d4rRne16``: a pair conversion plus two widenings per
    element) only to use the bits as a table index.  ``rrHalfRne8`` performs
    the same AGS Acc32 -> Acc16 conversion for all eight at once, so the
    rounding of every element is unchanged.
    """
    sites = 0
    # form 1: a separate loop filling an RRHalf from an RRFloat
    out, pos = [], 0
    while True:
        m = _HALF8.search(source, pos)
        if not m:
            out.append(source[pos:])
            break
        name, src = m.group(2), m.group(3)
        if not (_declared(source, name, 'RRHalf', m.start())
                and _declared(source, src, 'RRFloat', m.start())):
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        out.append(source[pos:m.start()])
        out.append('%s = rrHalfRne8(%s);' % (name, src))
        pos = m.end()
        sites += 1
    source = ''.join(out)
    # form 2: the conversion inline in the table-index loop
    out, pos = [], 0
    while True:
        m = _HALF8_INLINE.search(source, pos)
        if not m:
            out.append(source[pos:])
            break
        # the single statement after the loop header, then the site inside it
        try:
            hdr = _close(source, source.index('(', m.start())) + 1
        except ValueError:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        depth, end = 0, -1
        for i in range(hdr, len(source)):
            c = source[i]
            if c in '([{':
                depth += 1
            elif c in ')]}':
                depth -= 1
            elif c == ';' and depth == 0:
                end = i
                break
        if end < 0:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        body = source[m.start():end + 1]
        conv = re.search(r'float16_t\s*\(\s*([A-Za-z_]\w*)\s*\[\s*i\s*\]\s*\)', body)
        if not conv:
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        name = conv.group(1)
        lo, hi = _block_span(source, m.start())
        if not _declared(source, name, 'RRFloat', m.start()):
            out.append(source[pos:m.end()])
            pos = m.end()
            continue
        out.append(source[pos:m.start()])
        out.append('{ const RRHalf rrHb_ = rrHalfRne8(%s); ' % name)
        out.append(body[:conv.start()] + 'rrHb_[i]' + body[conv.end():])
        out.append(' }')
        pos = end + 1
        sites += 1
    source = ''.join(out)
    if not sites:
        return source, 0
    return _inject_half8(source), sites


# --------------------------------------------------------------------------- inf_select

_INF = re.compile(r'\(\s*isinf\s*\(')
_NAN_ARM = re.compile(r'\s*\?\s*uintBitsToFloat\s*\(\s*0x7fc00000u\s*\)\s*:\s*\(')


def inf_select(source: str) -> tuple[str, int]:
    body = _body_start(source)
    if body is None or 'RR_QUANT_CLAMPONLY 1' not in source[:body]:
        return source, 0
    out, pos, sites = [], body, 0
    out.append(source[:body])
    for m in _INF.finditer(source, body):
        if m.start() < pos:
            continue
        try:
            a_end = _close(source, m.end() - 1)
            arm = _NAN_ARM.match(source, a_end + 1)
            if not arm:
                continue
            b_end = _close(source, arm.end() - 1)
            outer = _close(source, m.start())
        except ValueError:
            continue
        a = source[m.end():a_end].strip()
        b = source[arm.end():b_end].strip()
        # the parenthesised select is exactly `(isinf(A) ? NaN : (A))`
        if a != b or source[b_end + 1:outer].strip():
            continue
        out.append(source[pos:m.start()])
        out.append('((' + b + '))')
        pos = outer + 1
        sites += 1
    out.append(source[pos:])
    return ''.join(out), sites


# --------------------------------------------------------------------------- residual_pack

_XR_DECL = re.compile(r'float16_t\s+xr((?:\[[^\]\[;]+\])+)\[\s*8\s*\]\s*;')
_XR_IDX = r'((?:\[\s*\w+\s*\])+)'
# xr[...][A] = x.x;  xr[...][A + 1u] = x.y;   A is 4u * h2 + 2u * pair or 4u * h2 + b
_XR_PAIR = re.compile(r'xr' + _XR_IDX + r'\[\s*(4u \* \w+ \+ (?:2u \* \w+|(\w+)))\s*\]\s*=\s*x\.x\s*;\s*'
                      r'xr\1\[\s*\2 \+ 1u\s*\]\s*=\s*x\.y\s*;')
_XR_HELPER = '''RRHalf rrUnpackHalf8(uint p[4]) {
  RRHalf r;
  [unroll] for (int i = 0; i < 8; ++ i) r[i] = asfloat16(uint16_t(p[i >> 1] >> (16u * uint(i & 1))));
  return r;
}
'''


def residual_pack(source: str) -> tuple[str, int]:
    """Keep the decoder's f16 residual ``xr`` packed across the attention loop.

    ``dec*_0`` computes ``xr = e + sk * s3`` before the head loop and reads it
    after.  The D3D12 compiler sinks that sum past the loop and holds both
    operands, one half per VGPR, through it.  On ``dec0_0`` that is about 75
    registers: 255 requested, 264 allocated, five waves per SIMD where RADV
    runs seven.  Two halves per ``uint`` plus a loop-carried ``^= hd >> 31u``
    (zero for any reachable trip count) pin the packed sum before the loop.
    The bits read back are the bits written: exact.  Only the four-tile
    ``xr[wv][m][8]`` of ``dec0_0`` is rewritten (216 VGPRs, seven waves).
    """
    body = _body_start(source)
    if body is None:
        return source, 0
    decl = _XR_DECL.search(source, body)
    if not decl:
        return source, 0
    dims = re.findall(r'\[([^\]]+)\]', decl.group(1))
    if len(dims) < 2:
        # dec1..4_0 (xr[m][8]) measured neutral to 16 us slower: their
        # allocation does not cross an occupancy step.
        return source, 0
    pairs = list(_XR_PAIR.finditer(source, decl.end()))
    if not pairs:
        return source, 0
    for p in pairs:
        # A bare step variable must come from `for (uint b = 0u; b < 4u; b += 2u)`
        if p.group(3) and not re.search(r'for \(uint %s = 0u; %s < 4u; %s \+= 2u\)'
                                        % ((re.escape(p.group(3)),) * 3),
                                        source[max(0, p.start() - 600):p.start()]):
            return source, 0
    loop = re.compile(r'\[loop\] for \(uint (\w+) = 0u; [^{]*\{\n').search(source, pairs[-1].end())
    if not loop:
        return source, 0
    out, pos, reads = [source[:decl.start()]], decl.start(), 0
    out.append('uint xr%s[4];' % decl.group(1))
    pos = decl.end()
    edits = [(p.start(), p.end(), 'xr%s[(%s) >> 1u] = uint(asuint16(x.x)) | (uint(asuint16(x.y)) << 16u);'
              % (p.group(1), p.group(2))) for p in pairs]
    elem = re.compile(r'(?:\(\s*xr' + _XR_IDX + r'\s*\)|xr' + _XR_IDX + r')\s*\[\s*([^\]]+?)\s*\]')
    row = re.compile(r'rrQuantMatrix\(\s*xr' + _XR_IDX + r'\s*,')
    for m in re.finditer(r'\bxr\b', source[decl.end():]):
        at = decl.end() + m.start()
        if any(a <= at < b for a, b, _ in edits):
            continue
        r = row.match(source, at - len('rrQuantMatrix(')) if source[:at].endswith('rrQuantMatrix(') else None
        if r:
            edits.append((r.start(), r.end(), 'rrQuantMatrix(rrUnpackHalf8(xr%s),' % r.group(1)))
            continue
        paren = re.search(r'\(\s*$', source[:at])
        e = (elem.match(source, paren.start()) if paren else None) or elem.match(source, at)
        idx = e and (e.group(1) or e.group(2))
        if not e or idx.count('[') != len(dims):
            return source, 0
        j = e.group(3)
        edits.append((e.start(), e.end(), 'asfloat16(uint16_t(xr%s[uint(%s) >> 1u] >> (16u * (uint(%s) & 1u))))'
                      % (idx, j, j)))
        reads += 1
    if not reads:
        return source, 0
    idx = ''.join('[j%d_]' % k for k in range(len(dims) + 1))
    pin = '        ' + ' '.join('[unroll] for (uint j%d_ = 0u; j%d_ < %s; ++ j%d_)' % (k, k, d.strip(), k)
                                for k, d in enumerate(dims + ['4u'])) + ' xr%s ^= %s >> 31u;\n' % (idx, loop.group(1))
    edits.append((loop.end(), loop.end(), pin))
    edits.sort()
    for a, b, r in edits:
        out.append(source[pos:a])
        out.append(r)
        pos = b
    out.append(source[pos:])
    text = ''.join(out)
    if 'RRHalf rrUnpackHalf8' not in text and 'rrUnpackHalf8(' in text:
        m = re.search(r'#include\s+"rr_wmma\.hlsl[^"]*"\n', text)
        at = m.end() if m else _body_start(text)
        text = text[:at] + '\n' + _XR_HELPER + text[at:]
    return text, len(pairs)


# --------------------------------------------------------------------------- branch_pin

_R16_DECL = re.compile(r'static\s+uint\s+rr2_r16((?:\[[^\]\[;]+\])+)\s*[,;]')


def branch_pin(source: str) -> tuple[str, int]:
    """Pin ``dec0_1``'s packed residual ``rr2_r16`` before the branch loop.

    The residual is packed to f16 pairs before the ``br`` loop and stored after
    it.  The D3D12 compiler sinks the packing past the loop and holds the f32
    ``rv`` instead, 64 VGPRs where the packed form needs 32 (216 allocated,
    seven waves).  A loop-carried ``^= br >> 31u`` (zero for any reachable trip
    count) keeps the packed words: 192 VGPRs, eight waves.  Exact.
    """
    body = _body_start(source)
    if body is None:
        return source, 0
    decl = _R16_DECL.search(source)
    if not decl:
        return source, 0
    writes = [m.end() for m in re.compile(r'\brr2_r16(?:\[[^\]]+\])+\s*=[^=]').finditer(source, body)]
    if not writes:
        return source, 0
    loop = re.compile(r'\[loop\] for \(uint (br) = 0u; [^{]*\{\n').search(source, writes[-1])
    if not loop:
        return source, 0
    end = _close(source, loop.end() - 2)
    if re.search(r'\brr2_r16\b', source[loop.end():end]) or not re.search(r'\brr2_r16\b', source[end:]):
        return source, 0
    dims = re.findall(r'\[([^\]]+)\]', decl.group(1))
    idx = ''.join('[j%d_]' % k for k in range(len(dims)))
    pin = '        ' + ' '.join('[unroll] for (uint j%d_ = 0u; j%d_ < %s; ++ j%d_)' % (k, k, d.strip(), k)
                                for k, d in enumerate(dims)) + ' rr2_r16%s ^= br >> 31u;\n' % idx
    return source[:loop.end()] + pin + source[loop.end():], 1


# --------------------------------------------------------------------------- driver

PASSES = {
    'quant_fusion': quant_fusion,
    'half_bits_roundtrip': half_bits_roundtrip,
    'e4pair_native_half': e4pair_native_half,
    'pack4_quantset': pack4_quantset,
    'half_rne8': half_rne8,
    'inf_select': inf_select,
    'residual_pack': residual_pack,
    'branch_pin': branch_pin,
}
# Only one ordering constraint: quant_fusion exposes the raw values that
# pack4_quantset then encodes once.
ORDER = ('quant_fusion', 'half_bits_roundtrip', 'e4pair_native_half',
         'pack4_quantset', 'half_rne8', 'inf_select', 'residual_pack',
         'branch_pin')


def optimize_with_report(source: str, stage: str = '',
                         passes=None) -> tuple[str, list[tuple[str, int]]]:
    """Apply the enabled passes; return the text and (pass, sites) counts."""
    names = tuple(passes) if passes is not None else ORDER
    report = []
    text = source
    for name in names:
        try:
            fn = PASSES[name]
        except KeyError:
            raise KeyError('unknown pass %r (known: %s)'
                           % (name, ', '.join(PASSES))) from None
        text, n = fn(text)
        report.append((name, n))
    return text, report


def optimize(source: str, stage: str = '', passes=None) -> str:
    """Return ``source`` with the native-RR rewrites applied (idempotent)."""
    if not source or _body_start(source) is None:
        # Not a generated stage (helper or hand-written shader): leave it.
        return source
    return optimize_with_report(source, stage, passes)[0]
