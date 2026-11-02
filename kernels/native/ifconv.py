"""If-conversion of pure branch diamonds in PTX (used by ptx2hip.py --ifconv).

NVIDIA's DLSS kernels branch around small pure arithmetic alternatives (e.g. the inverse tonemap
`v < 0.998 ? 1 / (1 - v) : (exp2(...) - 1) / (9 v)`), laid out as

    @%p bra $A;          (or @!%p)
    bra.uni $B;
  $A: <pure ops>  bra.uni $M;
  $B: <pure ops>                 (falls through)
  $M:

Branches like these split the code into basic blocks that no compiler schedules memory operations across, so
independent fetch chains on either side are serialized. This pass rewrites such a diamond into straight-line code:
both arms compute into fresh registers and every register an arm assigns is then selected by the predicate. Each lane
keeps exactly the values of the arm it would have taken (GPU arithmetic has no traps), so results are bit-identical.
Only arms made of register arithmetic are converted (no memory, texture, barrier, shuffle, branch or return; no
integer division, which is undefined behaviour in C for some inputs).
"""
import re

REG = re.compile(r'%([a-zA-Z]+)(\d+)\b')
PURE_BASES = {'add', 'sub', 'mul', 'fma', 'mad', 'div', 'rcp', 'sqrt', 'rsqrt', 'ex2', 'lg2', 'min', 'max', 'neg',
              'abs', 'cvt', 'mov', 'selp', 'setp', 'and', 'or', 'xor', 'not', 'shl', 'shr', 'set', 'bfi', 'copysign',
              'sin', 'cos', 'cvta'}
SEL = {'f': 'selp.f32', 'r': 'selp.b32', 'rd': 'selp.b64', 'rs': 'selp.b16'}


def _pure(it):
    if it['kind'] != 'stmt' or it['pred'] is not None:
        return False
    op = it['op']
    base = op.split('.')[0]
    if base not in PURE_BASES:
        return False
    if base == 'div' and not op.endswith('.f32') and 'approx' not in op:
        return False  # integer division (and IEEE f32 division, which the translator rejects anyway)
    if base == 'cvta':
        return False
    return True


def _target(it):
    if it['kind'] == 'stmt' and it['op'].startswith('bra'):
        return it['rest'].strip().rstrip(';').strip()
    return None


def apply(body, tokenize, parse, split_operands):
    items = parse(body, tokenize)
    # label -> number of branches targeting it
    refs = {}
    for it in items:
        t = _target(it)
        if t:
            refs[t] = refs.get(t, 0) + 1
    # fresh register numbers per kind
    top = {}
    for m in REG.finditer(body):
        k, n = m.group(1), int(m.group(2))
        top[k] = max(top.get(k, 0), n)
    fresh = {k: v + 100000 for k, v in top.items()}

    def new_reg(kind):
        fresh[kind] = fresh.get(kind, 100000) + 1
        return f'%{kind}{fresh[kind]}'

    use_sites = {}
    for x in items:
        if x['kind'] == 'stmt':
            for r in x['src']:
                use_sites.setdefault(r, []).append(id(x))

    def uses_outside(r, arm_ids):
        return any(u not in arm_ids for u in use_sites.get(r, []))

    defined_before = set()
    out, i, n_conv = [], 0, 0
    while i < len(items):
        it = items[i]
        conv = None
        if it['kind'] == 'stmt' and it['op'].startswith('bra') and it['pred'] is not None:
            conv = _match(items, i, refs)
        if conv is None:
            if it['kind'] == 'stmt':
                defined_before.update(it['dst'])
            out.append(_text(it))
            i += 1
            continue
        neg, pred, arm_a, arm_b, merge_idx = conv
        lines, last = [], [{}, {}]
        for side, arm in enumerate((arm_a, arm_b)):
            ren = {}
            for st in arm:
                op, rest = st['text'].split(None, 1)
                ops = split_operands(rest)
                if len(st['dst']) != 1 or ops[0] != st['dst'][0]:
                    raise ValueError('unexpected destination form: ' + st['text'])
                srcs = [REG.sub(lambda mm: ren.get(mm.group(0), mm.group(0)), o) for o in ops[1:]]
                d = st['dst'][0]
                kind = REG.match(d).group(1)
                if kind not in SEL:
                    raise ValueError('predicate assigned in a diamond arm: ' + st['text'])
                nr = new_reg(kind)
                ren[d] = nr
                lines.append(f"{op} {', '.join([nr] + srcs)}")
            last[side] = ren
        # only registers read outside the two arms need a select (arm temporaries are dropped)
        arm_ids = {id(x) for x in arm_a} | {id(x) for x in arm_b}
        assigned = sorted(r for r in set(last[0]) | set(last[1]) if uses_outside(r, arm_ids))
        cond = pred
        for r in assigned:
            kind = REG.match(r).group(1)
            a = last[0].get(r, r if r in defined_before else last[1].get(r))
            b = last[1].get(r, r if r in defined_before else last[0].get(r))
            # selp d, a, b, p: d = p ? a : b ; arm A is taken when (p xor neg)
            if neg:
                a, b = b, a
            lines.append(f'{SEL[kind]} {r}, {a}, {b}, {cond}')
        defined_before.update(assigned)
        out.extend(l + ';' for l in lines)
        n_conv += 1
        i = merge_idx  # the merge label itself is emitted by the main loop
    return '\n'.join(out), n_conv


def _text(it):
    if it['kind'] == 'open':
        return '{'
    if it['kind'] == 'close':
        return '}'
    if it['kind'] == 'label':
        return it['text'] + ':'
    return it['text'] + ';'


def _match(items, i, refs):
    """@p bra A; bra.uni B; A: ... bra.uni M; B: ... M:   -> (neg, pred, armA, armB, index of M)"""
    br = items[i]
    m = re.match(r'@(!?)(%p\d+)\s', br['text'])
    if not m:
        return None
    neg, pred = m.group(1) == '!', m.group(2)
    a_lab = _target(br)
    if i + 2 >= len(items):
        return None
    j = items[i + 1]
    if not (j['kind'] == 'stmt' and j['op'].startswith('bra') and j['pred'] is None):
        return None
    b_lab = _target(j)
    if refs.get(a_lab) != 1 or refs.get(b_lab) != 1:
        return None

    def arm(start, label):
        if not (items[start]['kind'] == 'label' and items[start]['text'] == label):
            return None
        body, k = [], start + 1
        while k < len(items):
            x = items[k]
            if x['kind'] == 'label':
                return body, k, None
            if x['kind'] in ('open', 'close'):
                return None
            if x['kind'] == 'stmt' and x['op'].startswith('bra'):
                if x['pred'] is not None:
                    return None
                return body, k + 1, _target(x)
            if not _pure(x):
                return None
            body.append(x)
            k += 1
        return None

    first = arm(i + 2, a_lab)
    order = 'AB'
    if first is None:
        first = arm(i + 2, b_lab)
        order = 'BA'
        if first is None:
            return None
    body1, k, jump1 = first
    if jump1 is None:
        return None
    second = arm(k, b_lab if order == 'AB' else a_lab)
    if second is None:
        return None
    body2, k2, jump2 = second
    if jump2 is not None and jump2 != jump1:
        return None  # the second arm may also jump to the merge label explicitly
    if not (k2 < len(items) and items[k2]['kind'] == 'label' and items[k2]['text'] == jump1):
        return None
    arm_a, arm_b = (body1, body2) if order == 'AB' else (body2, body1)
    return neg, pred, arm_a, arm_b, k2


def sink_stores(body, tokenize, parse):
    """Drop labels nothing branches to, then move each st.shared / st.local down to just before the next statement
    that could observe or reorder it: a shared/local load or any other store, a barrier, shuffle, atomic, label,
    branch or return, or a statement that writes one of the store's registers. Stores keep their relative order, so
    every memory location sees the same sequence of writes; texture loads (read-only data) and arithmetic move ahead
    of them, which lets independent fetch chains issue together. Returns (body, number of stores moved)."""
    items = parse(body, tokenize)
    refs = set()
    for it in items:
        t = _target(it)
        if t:
            refs.add(t)
    items = [it for it in items if not (it['kind'] == 'label' and it['text'] not in refs)]

    def is_fence(it):
        if it['kind'] != 'stmt':
            return True  # labels and scope braces
        op = it['op']
        base = op.split('.')[0]
        if base in ('bar', 'shfl', 'atom', 'red', 'bra', 'ret', 'membar', 'fence', 'st', 'sust', 'd4r'):
            return True
        if base == 'ld' and ('.shared' in op or '.local' in op):
            return True
        return False

    moved = 0
    i = len(items) - 1
    while i >= 0:
        it = items[i]
        if it['kind'] == 'stmt' and it['pred'] is None and it['op'].startswith('st.') and ('.shared' in it['op'] or '.local' in it['op']):
            regs = set(it['src'])
            j = i + 1
            while j < len(items) and not is_fence(items[j]) and not (regs & set(items[j]['dst'])):
                j += 1
            if j > i + 1:
                # after the pop the fence that stopped the scan is at j - 1: the store goes right before it
                items.insert(j - 1, items.pop(i))
                moved += 1
        i -= 1
    return '\n'.join(_text(it) for it in items), moved
