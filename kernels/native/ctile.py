"""Color-tap LDS tile pass for the hiluma output kernels (used by ptx2hip.py --ctile).

Path A of hiluma_engine_output_* (taken when a block's render footprint is wider than path B's 24-texel LDS tile,
e.g. at Quality) fetches 16 color texels per thread and converts each one (YCoCg, exposure, tonemap weight) right
after the fetch. The conversion of a texel depends only on the texel and the exposure %f1, and neighbouring
threads convert the same texels (~9x at ratio 1.5). This pass:

1. finds every tex.base on the color texture inside path A,
2. slices the instructions that compute the tap's packed result (cvt.rn.f16x2 x2 or cvt.rn.f16.f32) from the
   fetched texel (forward taint + backward need, including the tonemap branch diamond), and checks that no other
   value escapes the slice,
3. replaces the slice with `d4r.ctap.G` (tile lookup, falling back to fetch + the same conversion), and
4. emits one conversion function per slice shape (generated from the slice's own instructions, so the tile holds
   bit-identical values) and a cooperative tile fill at the start of path A.

The tile is keyed by the final (clamped) texel coordinates, so any tap outside it just takes the fallback.
"""
import re

REG = re.compile(r'%[A-Za-z_$][\w$]*')
CAP = 1024          # tile texels (TW*TH); a larger footprint falls back to direct fetches


class CtileErr(Exception):
    pass


def parse(body, tokenize):
    """-> list of dicts: kind ('stmt'|'label'|'open'|'close'), text, plus op/dst/src/pred for stmts"""
    items = []
    for kind, text in tokenize(body):
        it = {'kind': kind, 'text': text}
        if kind == 'stmt':
            s = text
            m = re.match(r'@(!?)(%p\d+)\s+(.*)$', s)
            it['pred'] = m.group(2) if m else None
            if m:
                s = m.group(3)
            parts = s.split(None, 1)
            it['op'] = parts[0]
            rest = parts[1] if len(parts) > 1 else ''
            it['rest'] = rest
            dst, src = [], []
            op = it['op']
            if op.startswith('.') or op.startswith('bra') or op.startswith('bar') or op.startswith('ret'):
                src = REG.findall(rest)
            elif op.startswith('st.') or op.startswith('sust'):
                src = REG.findall(rest)
            else:
                # first operand = destination(s)
                depth, i = 0, 0
                while i < len(rest):
                    c = rest[i]
                    if c in '{[':
                        depth += 1
                    elif c in '}]':
                        depth -= 1
                    elif c == ',' and depth == 0:
                        break
                    i += 1
                dst = REG.findall(rest[:i])
                src = REG.findall(rest[i:])
            if it['pred']:
                src = src + [it['pred']]
            it['dst'], it['src'] = dst, src
        items.append(it)
    return items


def branch_target(it):
    return it['rest'].strip().rstrip(';').strip().lstrip('$') if it['kind'] == 'stmt' and it['op'].startswith('bra') else None


def find_taps(items, color_aliases):
    return [i for i, it in enumerate(items) if it['kind'] == 'stmt' and it['op'] == 'tex.base.2d.v4.f32.s32'
            and re.search(r'\[\s*(%rd\d+)', it['rest']).group(1) in color_aliases]


def slice_tap(items, t):
    """-> (removed index set, results list, kind 'in'|'out', externals set)"""
    tex = items[t]
    taint = set(tex['dst'])
    tainted_idx = [t]
    results, kind = [], None
    i = t + 1
    while i < len(items):
        it = items[i]
        if it['kind'] == 'stmt' and any(s in taint for s in it['src']):
            op = it['op']
            if op.startswith('cvt.rn.f16x2.f32') and kind in (None, 'in'):
                kind = 'in'
                results.append((i, it['dst'][0]))
                tainted_idx.append(i)
                if len(results) == 2:
                    break
            elif op.startswith('cvt.rn.f16.f32') and kind is None:
                kind = 'out'
                results.append((i, it['dst'][0]))
                tainted_idx.append(i)
                break
            else:
                taint.update(it['dst'])
                tainted_idx.append(i)
        i += 1
    else:
        raise CtileErr(f'tap at {t}: no packed result found')
    last = results[-1][0]
    # backward need from the results (within [t, last])
    need_regs = set()
    for ri, _ in results:
        need_regs.update(items[ri]['src'])
    needed = {ri for ri, _ in results}
    # conditional branches on tainted predicates (the tonemap branch) are part of the slice
    for j in tainted_idx:
        it = items[j]
        if it['op'].startswith('bra') and it['pred'] is not None:
            needed.add(j)
            need_regs.add(it['pred'])
    for j in range(last, t - 1, -1):
        it = items[j]
        if j in needed:
            continue
        if it['kind'] == 'stmt' and j in tainted_idx and any(d in need_regs for d in it['dst']):
            needed.add(j)
            need_regs.update(it['src'])
    extra = [j for j in tainted_idx if j not in needed]
    if extra:
        raise CtileErr(f'tap at {t}: tainted but not needed: {[items[j]["text"] for j in extra]}')
    removed = set(needed)
    # branch diamonds: region from each conditional branch to its merge label
    for j in sorted(needed):
        it = items[j]
        if not it['op'].startswith('bra'):
            continue
        targets, seen = {branch_target(it)}, set()
        k = j + 1
        while True:
            x = items[k]
            if x['kind'] == 'label':
                lab = x['text'].lstrip('$')
                if lab in targets and targets - seen == {lab} and lab not in {branch_target(it)}:
                    removed.add(k)
                    break
                seen.add(lab)
                removed.add(k)
            elif x['kind'] == 'stmt':
                if x['op'].startswith('bra'):
                    if x['pred'] is not None and k not in needed:
                        raise CtileErr(f'tap at {t}: foreign conditional branch in diamond')
                    targets.add(branch_target(x))
                    removed.add(k)
                elif k not in needed:
                    # constant moves used only inside the slice are part of it
                    if x['op'].startswith('mov') and not any(REG.match(s) for s in x['src']):
                        removed.add(k)
                    else:
                        raise CtileErr(f'tap at {t}: foreign instruction in diamond: {x["text"]}')
            k += 1
            if k > last:
                raise CtileErr(f'tap at {t}: diamond does not close before the result')
    # escape check: registers defined in the slice (except results) used outside it
    defs = set()
    for j in removed:
        if items[j]['kind'] == 'stmt':
            defs.update(items[j]['dst'])
    res_regs = {r for _, r in results}
    inner_defs = defs - res_regs
    for j, it in enumerate(items):
        if j in removed or it['kind'] != 'stmt':
            continue
        bad = inner_defs.intersection(it['src'])
        if bad:
            raise CtileErr(f'tap at {t}: {bad} escapes to {it["text"]}')
        if defs.intersection(it['dst']) - res_regs or res_regs.intersection(it['dst']):
            raise CtileErr(f'tap at {t}: slice register redefined by {it["text"]}')
    # labels removed must not be targeted from outside
    labels = {items[j]['text'].lstrip('$') for j in removed if items[j]['kind'] == 'label'}
    for j, it in enumerate(items):
        if j not in removed and branch_target(it) in labels:
            raise CtileErr(f'tap at {t}: label targeted from outside the slice')
    used = set()
    for j in removed:
        if items[j]['kind'] == 'stmt' and j != t:
            used.update(items[j]['src'])
    externals = used - defs - set(tex['dst'])
    externals.discard(None)
    return removed, results, kind, externals


def canon(items, removed, t, results):
    """canonical text of the slice (registers by first appearance, texel by position)"""
    names, labels = {}, {}
    for k, r in enumerate(items[t]['dst']):
        names[r] = f'T{k}'
    for k, (_, r) in enumerate(results):
        names[r] = f'R{k}'
    out = []
    for j in sorted(removed):
        if j == t:
            continue
        it = items[j]
        if it['kind'] == 'label':
            lab = it['text'].lstrip('$')
            labels.setdefault(lab, f'L{len(labels)}')
            out.append(labels[lab] + ':')
            continue
        if it['kind'] != 'stmt':
            continue
        s = it['text']

        def rn(m):
            r = m.group(0)
            if r not in names:
                names[r] = f'V{len(names)}'
            return names[r]
        s = REG.sub(rn, s)
        s = re.sub(r'\$(L__\w+)', lambda m: labels.setdefault(m.group(1), f'L{len(labels)}'), s)
        out.append(s)
    return '\n'.join(out), names


def apply(tr, tokenize):
    """Rewrite tr.body in place for the color tile. Returns the HIP prelude (conversion functions) and info."""
    body = tr.body
    items = parse(body, tokenize)
    # color texture handle: ld.param.u64 [%rd1+-56] (param+200, %rd10 / %rd69); taps use it or its mov aliases
    # (some variants, e.g. mvlo, re-materialize the parameter base: %rdN = param_0 + 256, the same address as %rd1)
    param0 = {it['dst'][0] for it in items if it['kind'] == 'stmt' and it['op'] == 'mov.b64'
              and re.search(r',\s*\w+_param_0\s*$', it['rest'])}
    bases = {it['dst'][0] for it in items if it['kind'] == 'stmt' and it['op'] == 'add.s64'
             and len(it['src']) == 1 and it['src'][0] in param0 and re.search(r',\s*256\s*$', it['rest'])}
    if '%rd1' not in bases:
        raise CtileErr(f'parameter base %rd1 not found ({sorted(bases)})')
    color = {it['dst'][0] for it in items if it['kind'] == 'stmt' and it['op'] == 'ld.param.u64'
             and any(re.search(r'\[' + re.escape(b) + r'\+-56\]', it['rest']) for b in bases)}
    aliases = set(color)
    for it in items:
        if it['kind'] == 'stmt' and it['op'] == 'mov.u64' and len(it['src']) == 1 and it['src'][0] in color:
            aliases.add(it['dst'][0])
    # path A's color taps lie between $L__BB0_1 and $L__BB0_77 (the per-tap blocks); path B's are elsewhere
    labels = {it['text'].lstrip('$'): i for i, it in enumerate(items) if it['kind'] == 'label'}
    a0, a1 = labels['L__BB0_1'], labels['L__BB0_77']
    taps = [t for t in find_taps(items, aliases) if a0 < t < a1]
    if len(taps) != 16:
        raise CtileErr(f'expected 16 path A color taps, found {len(taps)}')
    groups, per_tap = {}, []
    all_removed = set()
    for t in taps:
        removed, results, kind, ext = slice_tap(items, t)
        if removed & all_removed:
            raise CtileErr('overlapping slices')
        all_removed |= removed
        text, names = canon(items, removed, t, results)
        ext_names = sorted(ext, key=lambda r: names.get(r, r))
        key = (kind, text, tuple(names.get(r) for r in ext_names))
        g = groups.setdefault(key, len(groups))
        per_tap.append((t, removed, results, kind, ext_names, g))
    if len(groups) > 3:
        raise CtileErr(f'{len(groups)} slice shapes')
    # externals: %f1 (exposure) or constant registers (single mov-immediate definition)
    const_def = {}
    for it in items:
        if it['kind'] == 'stmt' and it['op'].startswith('mov') and len(it['dst']) == 1 and not it['src']:
            const_def.setdefault(it['dst'][0], []).append(it)
    group_info = {}
    for (t, removed, results, kind, ext_names, g) in per_tap:
        if g in group_info:
            continue
        params, consts = [], []
        for r in ext_names:
            if r == '%f1':
                params.append(r)
            elif r in const_def and len(const_def[r]) == 1:
                consts.append(const_def[r][0]['text'])
            else:
                raise CtileErr(f'external {r} of tap {t}')
        group_info[g] = dict(t=t, removed=removed, results=results, kind=kind, params=params, consts=consts)
    return items, per_tap, group_info, color


def emit(tr, tokenize, Translator):
    """Apply the pass to translator tr (before translate()). Installs the pseudo-op handler; returns prelude text."""
    items, per_tap, group_info, color = apply(tr, tokenize)
    # ---- conversion functions, one per group, translated from the group's first slice
    prelude = []
    for g, info in sorted(group_info.items()):
        t, removed, results = info['t'], info['removed'], info['results']
        sub = Translator.__new__(Translator)
        sub.kernel = tr.kernel
        sub.param_name = tr.param_name
        sub.regs, sub.shared, sub.locals, sub.lines = {}, [], [], []
        sub.scope_names, sub.scope_id = [{}], 0
        tex = items[t]
        # texel inputs
        for k, r in enumerate(tex['dst']):
            sub.reg(r)
            sub.lines.append(f'    {sub.reg(r)} = t_[{k}];')
        for c in info['consts']:
            sub.stmt(c)
        for j in sorted(removed):
            if j == t:
                continue
            it = items[j]
            if it['kind'] == 'label':
                sub.lines.append(f'{it["text"].replace("$", "")}:;')
            elif it['kind'] == 'stmt':
                sub.stmt(it['text'])
        outs = [sub.reg(r) for _, r in results]
        decl = []
        for name, ty in sorted(sub.regs.items()):
            if name in [sub.reg(p) for p in info['params']]:
                continue
            decl.append(f'    {ty} {name};')
        args = ', '.join(['f4 t_'] + [f'float {sub.reg(p)}' for p in info['params']] + [f'uint32_t* o{k}' for k in range(len(outs))])
        prelude.append(f'RT void d4r_ctap_conv{g}({args})\n{{')
        prelude.extend(decl)
        prelude.extend(sub.lines)
        for k, o in enumerate(outs):
            prelude.append(f'    *o{k} = (uint32_t){o};')
        prelude.append('}')
    # ---- tile storage (aliases the kernel's LDS through d4r_lds)
    offs, off = {}, 0
    for g, info in sorted(group_info.items()):
        for k in range(len(info['results'])):
            offs[(g, k)] = off
            off += CAP * 4
    # ---- rewrite the body: removed statements dropped, pseudo-op at each tap, fill at the start of path A
    tap_at = {t: (results, ext, g) for (t, removed, results, kind, ext, g) in per_tap}
    removed_all = set()
    for (t, removed, *_r) in per_tap:
        removed_all |= removed
    out = []
    for i, it in enumerate(items):
        if it['kind'] == 'open':
            out.append('{'); continue
        if it['kind'] == 'close':
            out.append('}'); continue
        if it['kind'] == 'label':
            if i in removed_all:
                continue
            out.append(it['text'] + ':')
            if it['text'].lstrip('$') == 'L__BB0_1':
                out.append('d4r.ctile_fill;')
            continue
        if i in tap_at:
            results, ext, g = tap_at[i]
            m = re.search(r'\[\s*(%rd\d+)\s*,\s*\{([^}]*)\}\s*\]', it['rest'])
            regs = ', '.join(r for _, r in results)
            out.append(f'd4r.ctap.{g} {regs}, {m.group(1)}, {m.group(2)};')
            continue
        if i in removed_all:
            continue
        out.append(it['text'] + ';')
    tr.body = '\n'.join(out)
    summary = dict(groups=group_info, offs=offs, lds_bytes=off, taps=len(per_tap))

    orig_op = tr.op

    def op(opname, a):
        if opname == 'd4r.ctile_fill':
            tr.emit('d4r_ctile_fill(KARGP, (int32_t)r3, (int32_t)r4, (int32_t)r5, (int32_t)r6, f1, &ct_x0, &ct_y0, &ct_w, &ct_h);')
            return
        if opname.startswith('d4r.ctap.'):
            g = int(opname.split('.')[2])
            nres = len(group_info[g]['results'])
            res = [tr.reg(x) for x in a[:nres]]
            h = tr.val(a[nres], 'u64')
            x, y = tr.val(a[nres + 1], 'u32'), tr.val(a[nres + 2], 'u32')
            prm = ''.join(f', {tr.val(p, "f32")}' for p in group_info[g]['params'])
            tr.emit(f'{{ int32_t dx_ = (int32_t){x} - ct_x0, dy_ = (int32_t){y} - ct_y0;')
            tr.emit(f'  if ((uint32_t)dx_ < (uint32_t)ct_w && (uint32_t)dy_ < (uint32_t)ct_h) {{ uint32_t i_ = (uint32_t)(dy_ * ct_w + dx_);')
            for k, r in enumerate(res):
                tr.emit(f'    {r} = (uint32_t)((const AS3 uint32_t*)(d4r_lds + {offs[(g, k)]}))[i_];')
            outs = ', '.join(f'o_ + {k}' for k in range(nres))
            tr.emit(f'  }} else {{ uint32_t o_[2]; d4r_ctap_conv{g}(tex_fetch({h}, (int32_t){x}, (int32_t){y}){prm}, {outs});')
            for k, r in enumerate(res):
                tr.emit(f'    {r} = o_[{k}];')
            tr.emit('  } }')
            return
        orig_op(opname, a)
    tr.op = op
    # result register types: f16x2 packs are u32 regs, f16 packs are u16 regs: both assigned via (uint32_t) casts
    # fill function (needs the conversion functions and offsets)
    fill = ['RT void d4r_ctile_fill(const AS4 uint8_t* KARGP, int32_t x0l, int32_t y0l, int32_t x1l, int32_t y1l, float f1,',
            '                       int32_t* ox0, int32_t* oy0, int32_t* ow, int32_t* oh)',
            '{',
            '    // final texel coordinate of a tile-relative logical coordinate u: u > 0 ? min(lo + u, hi) : lo',
            '    // (the clamp every color tap applies; params +208/+212 = lo, +216/+220 = hi)',
            '    int32_t lox = *(const AS4 int32_t*)(KARGP + 208), loy = *(const AS4 int32_t*)(KARGP + 212);',
            '    int32_t hix = *(const AS4 int32_t*)(KARGP + 216), hiy = *(const AS4 int32_t*)(KARGP + 220);',
            '    int32_t fx0 = x0l > 0 ? __builtin_elementwise_min(lox + x0l, hix) : lox;',
            '    int32_t fy0 = y0l > 0 ? __builtin_elementwise_min(loy + y0l, hiy) : loy;',
            '    int32_t fx1 = x1l > 0 ? __builtin_elementwise_min(lox + x1l, hix) : lox;',
            '    int32_t fy1 = y1l > 0 ? __builtin_elementwise_min(loy + y1l, hiy) : loy;',
            '    int32_t w = fx1 - fx0 + 1, h = fy1 - fy0 + 1;',
            f'    if (w <= 0 || h <= 0 || w * h > {CAP}) {{ w = 0; h = 0; }}',
            '    uint64_t tex = *(const AS4 uint64_t*)(KARGP + 200);',
            '    int32_t n = w * h;',
            '    for (int32_t i = (int32_t)(__builtin_amdgcn_workitem_id_y() * 16u + __builtin_amdgcn_workitem_id_x()); i < n; i += 256)',
            '    {',
            '        int32_t ty = i / w, tx = i - ty * w;',
            '        f4 t_ = tex_fetch(tex, fx0 + tx, fy0 + ty);']
    for g, info in sorted(group_info.items()):
        nres = len(info['results'])
        prm = ''.join(', f1' for _ in info['params'])
        outs = ', '.join(f'o_ + {k}' for k in range(nres))
        fill.append(f'        {{ uint32_t o_[2]; d4r_ctap_conv{g}(t_{prm}, {outs});')
        for k in range(nres):
            fill.append(f'          ((AS3 uint32_t*)(d4r_lds + {offs[(g, k)]}))[i] = o_[{k}];')
        fill.append('        }')
    fill += ['    }',
             '    bar_sync();',
             '    *ox0 = fx0; *oy0 = fy0; *ow = w; *oh = h;',
             '}']
    for p in sum((info['params'] for info in group_info.values()), []):
        if p != '%f1':
            raise CtileErr('only %f1 is supported as a conversion parameter')
    return prelude, fill, summary
