"""Lower translated F pixel work without changing bindings or dispatch geometry.

The split gather, hybrid enc0/dec0_1, and reconstruction passes share generated
bitcast, binary16-packing and FP8-encoding idioms. Proof-gated rewrites remove
round trips, redundant masks and duplicate shared reads, and combine adjacent
halfword buffer accesses at their original two-byte alignment.
Texture.Load supplies the zero border natively, avoiding redundant bounds tests.

Native float tanh replaces the FP64 OCML emulation. This is deliberately
approximate: F prioritizes throughput rather than bit-identical arithmetic.
The codec and memory rewrites preserve values, saturation and NaN handling;
divergent pixel work never uses the driver's unsafe AGS conversion lowering.
"""

import re

__all__ = ['optimize']


# --------------------------------------------------------------------------- scanners

_PAIRS = {'(': ')', '[': ']', '{': '}'}
_CALL = re.compile(r'([A-Za-z_]\w*)\s*\(')


def _skip(source, i):
    """First index at or after i that is not whitespace or a comment."""
    n = len(source)
    while i < n:
        c = source[i]
        if c in ' \t\r\n':
            i += 1
        elif source.startswith('//', i):
            j = source.find('\n', i)
            i = n if j < 0 else j + 1
        elif source.startswith('/*', i):
            j = source.find('*/', i + 2)
            i = n if j < 0 else j + 2
        else:
            break
    return i


def _match(source, i):
    """Index of the bracket closing the one at i (same bracket type)."""
    opening = source[i]
    if opening not in _PAIRS:
        raise ValueError('not an opening bracket: ' + opening)
    closing = _PAIRS[opening]
    depth = 0
    j = i
    n = len(source)
    while j < n:
        if source.startswith('//', j):
            k = source.find('\n', j)
            j = n if k < 0 else k
            continue
        if source.startswith('/*', j):
            k = source.find('*/', j + 2)
            j = n if k < 0 else k + 2
            continue
        c = source[j]
        if c == opening:
            depth += 1
        elif c == closing:
            depth -= 1
            if depth == 0:
                return j
        elif c in '"\'':
            quote = c
            j += 1
            while j < n and source[j] != quote:
                j += 2 if source[j] == '\\' else 1
        j += 1
    raise ValueError('unbalanced ' + opening)


def _calls(source, names):
    """(name, start, open) for every call of one of names, nested ones included."""
    pos = 0
    while True:
        m = _CALL.search(source, pos)
        if not m:
            return
        pos = m.end()
        if m.group(1) not in names:
            continue
        at = m.end() - 1
        if source[at] != '(':
            continue
        yield m.group(1), m.start(), at


def _arguments(text):
    """Split call arguments on top-level commas."""
    result = []
    depth = 0
    start = 0
    for i, c in enumerate(text):
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == ',' and depth == 0:
            result.append(text[start:i])
            start = i + 1
    result.append(text[start:])
    return result


def _sole_call(expr):
    """(name, argument, rest) when expr starts with exactly one call."""
    i = _skip(expr, 0)
    m = _CALL.match(expr, i)
    if not m:
        return None
    op = m.end() - 1
    close = _match(expr, op)
    return m.group(1), expr[op + 1:close], expr[close + 1:]


def _strip_parens(expr):
    e = expr.strip()
    while e.startswith('(') and _match(e, 0) == len(e) - 1:
        e = e[1:-1].strip()
    return e


def _apply(source, edits):
    """Apply (start, end, replacement) edits, right to left.

    Edits nested inside another edit (an inner call of a rewritten outer call)
    are dropped: their span is no longer valid once the outer replacement has
    been inserted, and applying it would delete the text between the two stale
    offsets.  The outer replacement keeps the inner text, so the caller's
    fixpoint loop rewrites it on the next pass.
    """
    kept = []
    for start, end, text in sorted(edits, key=lambda e: (e[0], -(e[1] - e[0]))):
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, text))
    for start, end, text in reversed(kept):
        source = source[:start] + text + source[end:]
    return source


def _replace_calls(source, name, rewrite):
    """Rewrite every call of name to fixpoint; rewrite(name, argument) -> text."""
    for _ in range(8):
        edits = []
        for call, start, op in _calls(source, (name,)):
            close = _match(source, op)
            arg = source[op + 1:close]
            text = rewrite(call, arg)
            if text is not None and text != source[start:close + 1]:
                edits.append((start, close + 1, text))
        if not edits:
            return source
        source = _apply(source, edits)
    return source


# --------------------------------------------------------------------------- bit casts

_BITCAST_INVERSE = {'uintBitsToFloat': 'floatBitsToUint', 'intBitsToFloat': 'floatBitsToInt',
                    'floatBitsToUint': 'uintBitsToFloat', 'floatBitsToInt': 'intBitsToFloat'}


def _bitcast(source):
    """`uintBitsToFloat(floatBitsToUint(x))` (optionally `| 0u`) is x."""
    def rewrite(name, arg):
        call = _sole_call(arg)
        if not call or call[0] != _BITCAST_INVERSE.get(name):
            return None
        if not re.fullmatch(r'(?:\s*\|\s*0u?)*\s*', call[2]):
            return None
        return '(' + call[1] + ')'

    for _ in range(4):
        before = source
        for name in _BITCAST_INVERSE:
            source = _replace_calls(source, name, rewrite)
        if source == before:
            break
    return source


# --------------------------------------------------------------------------- half round trips

def _half_roundtrips(source):
    """`H(HU(x))`, with the identity masks the generated code adds, is x.

    `H(u)` is `asfloat16(uint16_t(u))`, `HU(h)` is `uint(asuint16(h))`: the
    pair reinterprets the binary16 and returns it unchanged.  This keeps NaN
    payloads and subnormals exactly.
    """
    def rewrite(name, arg):
        if name != 'H':
            return None
        call = _sole_call(arg)
        if not call or call[0] != 'HU' or call[2].strip():
            return None
        return '(' + call[1] + ')'

    # `| 0u` and `& 0xffffu` around HU's result are inside H's argument here,
    # so drop them from the argument before matching the sole call.
    def clean(source):
        edits = []
        for _, start, op in _calls(source, ('H',)):
            close = _match(source, op)
            arg = source[op + 1:close].strip()
            m = re.fullmatch(r'HU\(\s*(.*?)\s*\)\s*(?:(?:\|\s*0u|&\s*0x[fF]{4}u)\s*)*', arg, re.S)
            if not m:
                continue
            # balanced check: HU's argument must end where the regex says
            call = _sole_call(arg)
            if not call or call[0] != 'HU' or call[2].strip() and not re.fullmatch(
                    r'(?:\s*(?:\|\s*0u|&\s*0x[fF]{4}u))*\s*', call[2]):
                continue
            edits.append((start, close + 1, '(' + call[1] + ')'))
        return _apply(source, edits)

    for _ in range(4):
        before = source
        source = clean(source)
        source = _replace_calls(source, 'H', rewrite)
        if source == before:
            break
    return source


# --------------------------------------------------------------------------- FP8 encode

_CODEC_BODY = re.compile(
    r'(uint\s+h2e4m3_hw\s*\(\s*float16_t\s+h\s*,\s*bool\s+relu\s*\)\s*\{)'
    r'(?:(?!\n\}).)*?floate4m3BitsToUintEXT(?:(?!\n\}).)*?(\n\})', re.S)

_CODEC_HEAD = '''uint h2e4m3_hw(float16_t h, bool relu)
{
    if (isnan(h)) return 0x7fu;
    if (relu) h = h > float16_t(0.0) ? h : float16_t(0.0);
    return rrCode(float(h));
'''


def _codec(source):
    """Strength-reduce the generated scalar FP8 activation encode.

    `floate4m3BitsToUintEXT(floate4m3_t(x))` is `rrCode(rrQuant(x))`.  rrQuant
    returns the value rrCode(x) decodes to; rrCode of that is the identical
    code, so rrQuant's round/subnormal half is dead.  rrCode saturates at 448
    and maps NaN to 127, so for the non-NaN inputs the function has already
    filtered, the f16 clamp before the encode adds nothing.  The explicit
    `isnan` return keeps the unsigned 0x7f that rrCode's sign copy would turn
    into 0xff for a negative NaN.
    """
    def rewrite(name, arg):
        call = _sole_call(arg)
        if not call or call[0] != 'floate4m3_t' or call[2].strip():
            return None
        return 'rrCode(' + call[1] + ')'

    match = _CODEC_BODY.search(source)
    if match:
        source = (source[:match.start()] + _CODEC_HEAD + match.group(2) +
                  source[match.end():])
    return _replace_calls(source, 'floate4m3BitsToUintEXT', rewrite)


_E4_HELPERS = '''uint rrE4H(uint h)
{
    const uint a = h & 0x7fffu;
    return a > 0x7c00u ? 0x7fu : rrCode(float(asfloat16(uint16_t(h))));
}
uint rrE4x2(uint w) { return rrE4H(w & 0xffffu) | (rrE4H(w >> 16u) << 8u); }
'''

# rrE4x2 where the wave runs uniformly, on AMD's driver: one hardware conversion
# of both halves clamped to +-448 (lane-local, v_cvt_pk_fp8_f32) instead of two
# software codecs. The clamp takes an infinity to its own bound, as rrCode does,
# and a NaN to a bound, which the selects replace by 0x7f: every input keeps its
# code. vkd3d-proton's rrCode is already the hardware conversion; the macro keeps
# its preprocessed source unchanged.
_E4_UNIFORM = '''#if D4R_VKD3D
#define rrE4x2W rrE4x2
#else
uint rrE4x2W(uint w)
{
    const vector<float16_t, 2> v = clamp(vector<float16_t, 2>(asfloat16(uint16_t(w & 0xffffu)),
                                                              asfloat16(uint16_t(w >> 16u))),
                                         float16_t(-448.0), float16_t(448.0));
    const uint4 h = uint4(uint(asuint16(v.x)) | (uint(asuint16(v.y)) << 16), 0u, 0u, 0u);
    uint c = convert(MatrixAcc16, MatrixB8, 0, m16(h)).reg[0] & 0xffffu;
    c = (w & 0x7fffu) > 0x7c00u ? (c & 0xff00u) | 0x7fu : c;
    c = ((w >> 16u) & 0x7fffu) > 0x7c00u ? (c & 0x00ffu) | 0x7f00u : c;
    return c;
}
#endif
'''

_ENTRY = re.compile(r'(?m)^void main\(\)')
# the split stages' window test (rr/gen_rr2_enc0.py): workgroup and subgroup
# indices against a uniform bound, the same for every lane of a wave
_VALID_DEF = re.compile(r'const bool rr2_valid = ((?:gl_SubgroupID < \d+u && )?gl_WorkGroupID\.x < rr2_gx);')
_VALID_IF = re.compile(r'if\s*\(\s*rr2_valid\s*\)$')


def _uniform_flow(source, at):
    """Whether `at`, inside main, runs with every lane of its wave.

    Each block open between main's brace and `at` must be a bare scope or the
    wave-uniform `if (rr2_valid)`; any other if / else / loop body may be
    divergent, where the driver's lowering of an AGS conversion is unsafe.
    """
    entry = _ENTRY.search(source)
    if not entry or at < entry.end():
        return False
    valid = bool(_VALID_DEF.search(source, entry.end(), at))
    i = source.index('{', entry.end())
    stack = []
    j = i + 1
    while True:
        j = _skip(source, j)
        if j >= at:
            break
        c = source[j]
        if c == '{':
            head = source[max(i + 1, j - 200):j].rstrip()
            stack.append(head[-1:] in (';', '{', '}') or (valid and bool(_VALID_IF.search(head))))
        elif c == '}':
            if not stack:
                return False
            stack.pop()
        elif c in '"\'':
            return False
        j += 1
    return all(stack)


def _packed_encode(source):
    """`f16x2_to_e4m3x2(H2(w), false)` encodes w's two halves; rrE4x2(w) does.

    H2 splits w into two binary16 values and the codec raises them to f32; the
    helper takes the stored half bits directly.  Same codes (NaN -> 0x7f,
    saturating round to nearest even), one pack/unpack less per pair. In
    wave-uniform flow it is rrE4x2W, the hardware conversion (_E4_UNIFORM).
    """
    edits = []
    for _, start, op in _calls(source, ('f16x2_to_e4m3x2',)):
        close = _match(source, op)
        args = _arguments(source[op + 1:close])
        if len(args) != 2 or args[1].strip() != 'false':
            continue
        call = _sole_call(args[0])
        if not call or call[0] != 'H2' or call[2].strip():
            continue
        word = _arguments(call[1])
        if len(word) != 1 or not re.fullmatch(r'\s*[A-Za-z_]\w*\s*', word[0]):
            continue
        name = 'rrE4x2W' if _uniform_flow(source, start) else 'rrE4x2'
        edits.append((start, close + 1, name + '(' + word[0].strip() + ')'))
    if not edits:
        return source
    uniform = any(e[2].startswith('rrE4x2W(') for e in edits)
    source = _apply(source, edits)
    if re.search(r'(?m)^void main\(\)', source):
        at = re.search(r'(?m)^void main\(\)', source).start()
    else:
        at = source.find('#include "rr_wmma.hlsli"')
        at = source.find('\n', at) + 1
    return source[:at] + _E4_HELPERS + (_E4_UNIFORM if uniform else '') + source[at:]


# --------------------------------------------------------------------------- pair accessors

_PAIR_LOAD = re.compile(
    r'return\s+uint\(\s*(rrB\d+)\.Load<uint16_t>\(\s*\(\s*a\s*>>\s*1\s*\)\s*\*\s*2u\s*\)\s*\)\s*'
    r'\|\s*\(\s*uint\(\s*\1\.Load<uint16_t>\(\s*\(\s*\(\s*a\s*>>\s*1\s*\)\s*\+\s*1u\s*\)\s*\*\s*2u\s*\)\s*\)\s*<<\s*16\s*\)')
_PAIR_STORE = re.compile(
    r'(rrB\d+)\.Store<uint16_t>\(\s*\(\s*a\s*>>\s*1\s*\)\s*\*\s*2u\s*,\s*uint16_t\(\s*v\s*&\s*0x[fF]{4}u\s*\)\s*\)\s*;\s*'
    r'\1\.Store<uint16_t>\(\s*\(\s*\(\s*a\s*>>\s*1\s*\)\s*\+\s*1u\s*\)\s*\*\s*2u\s*,\s*uint16_t\(\s*v\s*>>\s*16\s*\)\s*\)\s*;')
_ACCESSOR = re.compile(r'\b(uint|void)\s+([A-Za-z_]\w*)\s*\(\s*uint\s+a\s*(,\s*uint\s+v\s*)?\)\s*\{')


def _pair_accessors(source):
    """Two adjacent binary16 words accessed as one 2-wide templated accessor.

    The address is `(a >> 1) * 2`, always even, so the template's element
    alignment (2 bytes) is all the load needs.  Values are identical: the
    second word is at offset +2.
    """
    pos = 0
    while True:
        m = _ACCESSOR.search(source, pos)
        if not m:
            return source
        open_ = m.end() - 1
        close = _match(source, open_)
        body = ' '.join(source[open_ + 1:close].split())
        ret, name, second = m.group(1), m.group(2), m.group(3)
        replacement = None
        if ret == 'uint' and not second:
            lm = _PAIR_LOAD.search(body)
            if lm:
                buf = lm.group(1)
                replacement = ('{ const vector<uint16_t, 2> rrPair_ = '
                               f'{buf}.Load<vector<uint16_t, 2> >((a >> 1) * 2u); '
                               'return uint(rrPair_.x) | (uint(rrPair_.y) << 16); }')
        elif ret == 'void' and second:
            sm = _PAIR_STORE.search(body)
            if sm:
                buf = sm.group(1)
                replacement = (f'{{ {buf}.Store<vector<uint16_t, 2> >((a >> 1) * 2u, '
                               'vector<uint16_t, 2>(uint16_t(v & 0xffffu), uint16_t(v >> 16))); }')
        if replacement is None:
            pos = close + 1
            continue
        head = f'{ret} {name}(uint a{second or ""}) '
        source = source[:m.start()] + head + replacement + source[close + 1:]
        pos = m.start() + len(head) + len(replacement)
    return source


# --------------------------------------------------------------------------- masks and casts

_TYPE = ('bool|int|uint|float|double|half|float16_t|uint16_t|int16_t|uint8_t|int8_t|'
         'uint64_t|int64_t|f16vec2|f16vec3|f16vec4|vec2|vec3|vec4|ivec2|ivec3|ivec4|'
         'uvec2|uvec3|uvec4|uint2|uint3|uint4|u16vec2|u16vec4|int2|int3|int4|'
         'bvec2|bvec3|bvec4|RRByte|RRHalf|RRFloat|RRHalf8|RRBytes8|Matrix16')

_UINT16_CALL = re.compile(
    r'(?:HU|PARAM16|SH16|GA16|GW16|GX\d+_16|rrE4H|rrE4x2W?)$')


def _declarations(source):
    """name -> declared type, for the flat declarations the translator emits."""
    result = {}
    pattern = re.compile(r'(?m)^[ \t]*(?:static\s+|shared\s+|groupshared\s+)?(?:const\s+)?'
                         r'(' + _TYPE + r')\s+([^;{}]+);')
    for m in pattern.finditer(source):
        typ = m.group(1)
        for item in _arguments(m.group(2)):
            v = re.match(r'\s*([A-Za-z_]\w*)', item)
            if v:
                result.setdefault(v.group(1), typ)
    return result


def _is16(expr, state, types):
    """Provably below 2^16, as the generated expressions spell it."""
    e = _strip_parens(expr)
    if re.fullmatch(r'\d+u?', e):
        return int(e.rstrip('u')) <= 0xffff
    if re.fullmatch(r'0[xX][0-9a-fA-F]+u?', e):
        return int(e.rstrip('u'), 16) <= 0xffff
    call = _sole_call(e)
    if call and not call[2].strip() and _UINT16_CALL.match(call[0]):
        return True
    if re.fullmatch(r'[A-Za-z_]\w*', e):
        return state.get(e, False)
    if re.fullmatch(r'.*&\s*0x[fF]{4}u', e, re.S):
        return True
    m = re.fullmatch(r'(.*?)\s*\|\s*0u?', e, re.S)
    if m:
        return _is16(m.group(1), state, types)
    return False


def _split_top(expr, op):
    """Split on a binary operator that is not inside brackets or a longer op."""
    result = []
    depth = 0
    start = 0
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif depth == 0 and expr.startswith(op, i):
            if op in ('|', '&') and (expr.startswith(op * 2, i)):
                i += 2
                continue
            if op in ('<<', '>>') and i and expr[i - 1] == op[0]:
                i += 1
                continue
            result.append(expr[start:i])
            start = i + len(op)
            i = start
            continue
        i += 1
    result.append(expr[start:])
    return result


def _is_uint_literal(text):
    return bool(re.fullmatch(r'\s*(?:\d+u|0[xX][0-9a-fA-F]+u)\s*', text))


def _top_ternary(expr):
    """Split `cond ? a : b` at the top-level ? and matching :."""
    depth = 0
    nested = 0
    question = -1
    for i, c in enumerate(expr):
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif depth:
            continue
        elif c == '?' and not (i + 1 < len(expr) and expr[i + 1] == '?'):
            if question < 0:
                question = i
            else:
                nested += 1
        elif c == ':' and question >= 0:
            if nested:
                nested -= 1
            else:
                return expr[:question], expr[question + 1:i], expr[i + 1:]
    return None


def _uint_typed(expr, types):
    """The expression's type is already uint (so uint(expr) is a no-op)."""
    e = _strip_parens(expr)
    if _is_uint_literal(e):
        return True
    if re.fullmatch(r'[A-Za-z_]\w*', e):
        return types.get(e) == 'uint'
    if re.fullmatch(r'[A-Za-z_]\w*(?:\[[^]]*\])+', e):
        return types.get(re.match(r'[A-Za-z_]\w*', e).group(0)) == 'uint'
    call = _sole_call(e)
    if call and not call[2].strip():
        name = call[0]
        return (name == 'uint' or name in ('HU', 'HU2_', 'PARAM32', 'PARAM16', 'PARAM8',
                                           'SH32', 'SH16', 'bfi32', 'rrCode', 'rrE4H',
                                           'rrE4x2', 'rrE4x2W', 'h2e4m3_hw', 'f16x2_to_e4m3x2')
                or bool(re.fullmatch(r'(?:GA|GW|GX)\d+(?:_\d+)?', name)))
    m = re.fullmatch(r'(.+?)\s*&\s*0x[fF]{4}u', e, re.S)
    if m:
        return True
    ternary = _top_ternary(e)
    if ternary:
        return _uint_typed(ternary[1], types) and _uint_typed(ternary[2], types)
    for op in ('|', '&', '^', '+', '-', '*'):
        parts = _split_top(e, op)
        if len(parts) > 1:
            return all(_uint_typed(p, types) or _is_uint_literal(p) for p in parts)
    for op in ('<<', '>>'):
        parts = _split_top(e, op)
        if len(parts) == 2:
            shift = _strip_parens(parts[1])
            return (_uint_typed(parts[0], types) and
                    (bool(re.fullmatch(r'\d+u?', shift)) or
                     _uint_typed(shift, types)))
    return False


def _back_match(text, j, opening, closing):
    """Index of the opening bracket matching the closing one at j."""
    depth = 0
    k = j
    while k >= 0:
        if text[k] == closing:
            depth += 1
        elif text[k] == opening:
            depth -= 1
            if depth == 0:
                return k
        k -= 1
    return -1


def _drop_masks(text, state, types):
    """Drop `& 0xffffu` on values already truncated to 16 bits.

    Also unwraps `uint(...)` when its argument is a uint expression, and
    `uint16_t(v & 0xffffu)` (the conversion truncates already).
    """
    # uint16_t(x & 0xffffu): the cast truncates, the mask is redundant.
    text = re.sub(r'\buint16_t\(\s*([A-Za-z_]\w*)\s*&\s*0x[fF]{4}u\s*\)',
                  r'uint16_t(\1)', text)
    mask = re.compile(r'&\s*0x[fF]{4}u')
    pos = 0
    while True:
        m = mask.search(text, pos)
        if not m:
            return text
        j = m.start() - 1
        while j >= 0 and text[j] in ' \t\r\n':
            j -= 1
        if j < 0:
            pos = m.end()
            continue
        if text[j] == ')':
            k = _back_match(text, j, '(', ')')
            pre = re.search(r'([A-Za-z_]\w*)\s*$', text[:k]) if k >= 0 else None
            if pre:
                callee, arg = pre.group(1), text[k + 1:j]
                if _UINT16_CALL.match(callee) or callee == 'uint16_t':
                    text = text[:m.start()] + text[m.end():]
                    pos = m.start()
                    continue
                if callee == 'uint' and _uint_typed(arg, types):
                    repl = _strip_parens(arg)
                    text = text[:pre.start()] + repl + text[m.end():]
                    pos = pre.start() + len(repl)
                    continue
            pos = m.end()
            continue
        pre = re.search(r'[A-Za-z_]\w*$', text[:j + 1])
        name = pre.group(0) if pre else None
        if name and (state.get(name, False) or types.get(name) == 'uint16_t'):
            text = text[:pre.start()] + name + text[m.end():]
            pos = pre.start() + len(name)
            continue
        pos = m.end()


def _drop_high_half_masks(source, types):
    """`(uint_value >> 16u) & 0xffffu` is just the high half.

    A uint shifted right by 16 is already below 2^16.  The generated code masks
    it again after reading the upper half of a shared word (and the pair loads
    introduced above make the same shape); the AND is one VALU per read.
    """
    pattern = re.compile(r'(\b(?:[A-Za-z_]\w*)(?:\[[^]]*\])*)\s*>>\s*16u\s*&\s*0x[fF]{4}u')
    def rewrite(m):
        base = re.match(r'[A-Za-z_]\w*', m.group(1)).group(0)
        return m.group(1) + ' >> 16u' if types.get(base) == 'uint' else m.group(0)
    return pattern.sub(rewrite, source)


def _collapse_masks(text):
    """`x & 0xffffu & 0xffffu` is `x & 0xffffu` (the repeated masks are idempotent).

    The generated code masks some shared-memory reads twice (through the
    accessor and again at the use); removing the redundant outer operation can
    leave the inner one repeated, which this collapses.
    """
    return re.sub(r'&\s*0x([fF]{4})u(?:\s*&\s*0x[fF]{4}u)+',
                  lambda m: '& 0x' + m.group(1) + 'u', text)


def _rewrite_assignments(source, types):
    """Walk assignments in order, using the types/values proven above them.

    Only variables with a single assignment carry a proven width: the generated
    SSA temporaries mostly are single-assignment, and a variable written inside
    one branch could otherwise keep an older, wider value on the merge path.
    """
    state = {}
    assignments = {}
    for m in re.finditer(r'(?m)^([ \t]*)([A-Za-z_]\w*)\s*=(?!=)\s*([^;]*);', source):
        assignments[m.group(2)] = assignments.get(m.group(2), 0) + 1
    out = []
    last = 0
    for m in re.finditer(r'(?m)^([ \t]*)([A-Za-z_]\w*)\s*=(?!=)\s*([^;]*);', source):
        lhs = m.group(2)
        rhs = _drop_masks(m.group(3), state, types)
        out.append(source[last:m.start(3)])
        out.append(rhs)
        last = m.end(3)
        state[lhs] = _is16(rhs, state, types) and assignments[lhs] == 1
    out.append(source[last:])
    return ''.join(out)


def _drop_redundant_casts(source, types):
    """`uint(x)` where x is already uint, and `T(T(x))`."""
    def rewrite(name, arg):
        call = _sole_call(arg)
        if call and not call[2].strip() and call[0] == name:
            return name + '(' + call[1] + ')'
        if name == 'uint' and _uint_typed(arg, types):
            return _strip_parens(arg)
        return None

    for _ in range(6):
        before = source
        for name in ('uint', 'int', 'float'):
            source = _replace_calls(source, name, rewrite)
        if source == before:
            break
    return source


# --------------------------------------------------------------------------- shared read pairs

_SH_READ = re.compile(r'sh\[\s*\(\s*([A-Za-z_]\w*)\s*\+\s*(\d+)u\s*\)\s*>>\s*2\s*\]')


def _residue(expr, res):
    """Residue mod 4 of a uint address expression, or None when unknown."""
    e = _strip_parens(expr)
    call = _sole_call(e)
    if call and not call[2].strip() and call[0] in ('uint', 'uint16_t'):
        return _residue(call[1], res)
    if re.fullmatch(r'\d+u?', e):
        return int(e.rstrip('u')) % 4
    if re.fullmatch(r'0[xX][0-9a-fA-F]+u?', e):
        return int(e.rstrip('u'), 16) % 4
    if re.fullmatch(r'[A-Za-z_]\w*', e):
        return res.get(e)
    parts = _split_top(e, '+')
    if len(parts) > 1:
        values = [_residue(p, res) for p in parts]
        return None if any(v is None for v in values) else sum(values) % 4
    parts = _split_top(e, '-')
    if len(parts) == 2:
        a, b = _residue(parts[0], res), _residue(parts[1], res)
        return None if a is None or b is None else (a - b) % 4
    parts = _split_top(e, '*')
    if len(parts) > 1:
        values = [_residue(p, res) for p in parts]
        return 0 if any(v == 0 for v in values) else None
    parts = _split_top(e, '<<')
    if len(parts) == 2:
        a = _residue(parts[0], res)
        shift = _strip_parens(parts[1])
        if re.fullmatch(r'\d+u?', shift):
            k = int(shift.rstrip('u'))
            if k >= 2:
                return 0
            return None if a is None else (a << k) % 4
        return 0 if a == 0 else None
    return None


def _residues(source):
    """Mod 4 of every single-assignment uint variable (None when unknown)."""
    counts = {}
    initialized = {}
    for m in re.finditer(r'(?m)^([ \t]*)([A-Za-z_]\w*)\s*=(?!=)\s*([^;]*);', source):
        counts[m.group(2)] = counts.get(m.group(2), 0) + 1
    for m in re.finditer(r'(?m)^[ \t]*(?:static\s+|shared\s+|groupshared\s+)?(?:const\s+)?uint\s+([^;]+);', source):
        for item in _arguments(m.group(1)):
            v = re.fullmatch(r'\s*([A-Za-z_]\w*)\s*=\s*([^,]+?)\s*', item)
            if v:
                initialized[v.group(1)] = v.group(2)
    res = {}
    for m in re.finditer(r'(?m)^([ \t]*)([A-Za-z_]\w*)\s*=(?!=)\s*([^;]*);', source):
        name = m.group(2)
        if counts[name] != 1:
            res[name] = None
            continue
        res[name] = _residue(m.group(3), res)
    for name, value in initialized.items():
        if name not in res:
            res[name] = _residue(value, res)
    return res


def _shared_pairs(source):
    """Read the two halves of one shared word with one load.

    The generated gather and output passes read a binary16 pair as

        a = (sh[(base + 0u) >> 2] & 0xffffu);
        b = (sh[(base + 2u) >> 2] >> 16u & 0xffffu);

    (offsets 0/2, 4/6, 96/98, ...).  With `base` a multiple of 4 the two
    addresses are the same word; the shader's writers pack exactly those pairs
    into `sh[(. + 0u) >> 2]`, so the layout already requires the alignment.
    Proving it by residues mod 4, both reads become one local.  The two
    statements are adjacent and the first is a plain local assignment, so
    hoisting the load above it moves nothing across a barrier or a shared
    write.
    """
    res = _residues(source)
    pattern = r'(?m)^([ \t]*)([A-Za-z_]\w*)\s*=(?!=)\s*([^;]*);'
    statements = [(m.start(), m.end(), m.group(1), m.group(2), m.group(3))
                  for m in re.finditer(pattern, source)]
    edits = []
    n = 0
    i = 0
    while i < len(statements) - 1:
        first, second = statements[i], statements[i + 1]
        a = _SH_READ.search(first[4])
        b = _SH_READ.search(second[4])
        ok = (a and b and a.group(1) == b.group(1) and
              first[2] == second[2] and
              int(b.group(2)) == int(a.group(2)) + 2 and
              re.search(r'&\s*0x[fF]{4}u', first[4]) and
              re.search(r'>>\s*16u', second[4]))
        if ok:
            offset = int(a.group(2))
            residue = res.get(a.group(1))
            ok = residue is not None and (residue + offset) % 4 == 0
        if not ok:
            i += 1
            continue
        name = f'rrSh{n}_'
        n += 1
        load = a.group(0)
        replacement = (f'{first[2]}const uint {name} = {load};\n'
                       + source[first[0]:first[1]].replace(load, name) + '\n'
                       + source[second[0]:second[1]].replace(b.group(0), name))
        edits.append((first[0], second[1], replacement))
        i += 2
    return _apply(source, edits)


# --------------------------------------------------------------------------- constant guards

_GUARD = re.compile(r'\(\s*\(?\s*(\d+)u?\s*\)?\s*>=\s*32u\s*\?\s*0u\s*:\s*')


def _shift_guards(source):
    """`(k >= 32u ? 0u : x << k)` with literal k < 32 is the shift itself."""
    edits = []
    pos = 0
    while True:
        m = _GUARD.search(source, pos)
        if not m:
            break
        if int(m.group(1)) >= 32:
            pos = m.end()
            continue
        close = _match(source, m.start())
        value = source[m.end():close].strip()
        edits.append((m.start(), close + 1, value))
        pos = close + 1
    source = _apply(source, edits)
    source = re.sub(r'min\(\s*(\d+)u?\s*,\s*31u\s*\)',
                    lambda m: m.group(1) + 'u' if int(m.group(1)) <= 31 else m.group(0), source)
    source = re.sub(r'min\(\s*31u\s*,\s*(\d+)u?\s*\)',
                    lambda m: m.group(1) + 'u' if int(m.group(1)) <= 31 else m.group(0), source)
    return source


def _native_texture_loads(source):
    """D3D Texture.Load returns zero in every component outside the texture."""
    m = re.search(r'\bvec4\s+fetch0\(sampler2D t,\s*ivec2 c\)\s*\{', source)
    if not m:
        return source
    start = m.end() - 1
    end = _match(source, start)
    expected = ('constboolout_=any(greaterThanEqual(uvec2(c),uvec2(textureSize(t,0))));'
                'if(RR_ROBUST_IMAGES){constvec4v=texelFetch(t,c,0);'
                'returnvec4(v.xyz,out_?0.0:v.w);}'
                'returnout_?vec4(0.0):texelFetch(t,c,0);')
    if ''.join(source[start + 1:end].split()) != expected:
        return source
    return source[:start] + '{ return texelFetch(t, c, 0); }' + source[end + 1:]


def _native_tanh(source):
    """Avoid the FP64 polynomial in the hybrid decoder's per-pixel correction."""
    m = re.search(r'\bfloat\s+tanh_ocml\(float x\)\s*\{', source)
    if not m:
        return source
    start = m.end() - 1
    end = _match(source, start)
    if 'double' not in source[start:end]:
        return source
    return source[:start] + '{ return tanh(x); }' + source[end + 1:]


# --------------------------------------------------------------------------- driver

def _truncating_masks(source):
    """`uint16_t(x & 0xffffu)` is `uint16_t(x)`: the conversion truncates."""
    return re.sub(r'\buint16_t\(\s*([A-Za-z_]\w*)\s*&\s*0x[fF]{4}u\s*\)',
                  r'uint16_t(\1)', source)


def optimize(source, stage):
    """Peephole-optimize translated F reconstruction HLSL (bindings unchanged).

    `stage` selects the context, but every rule is safe for any translated F
    stage that includes rr_wmma.hlsli; only the shapes present change.
    """
    if not source or 'rr_wmma.hlsli' not in source:
        return source
    source = _native_texture_loads(source)
    source = _native_tanh(source)
    source = _shift_guards(source)
    source = _bitcast(source)
    source = _half_roundtrips(source)
    source = _codec(source)
    source = _packed_encode(source)
    source = _pair_accessors(source)
    types = _declarations(source)
    source = _drop_redundant_casts(source, types)
    source = _rewrite_assignments(source, types)
    source = _collapse_masks(source)
    source = _shared_pairs(source)
    types = _declarations(source)
    source = _rewrite_assignments(source, types)
    source = _collapse_masks(source)
    source = _drop_high_half_masks(source, types)
    source = _drop_redundant_casts(source, types)
    source = _collapse_masks(source)
    return _truncating_masks(source)
