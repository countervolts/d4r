#!/usr/bin/env python3
"""Translate one NVIDIA PTX texture kernel into a Vulkan GLSL compute shader.

usage: ptx2glsl.py MODULE.ptx KERNEL [--tex OFF:linear|nearest ...] > KERNEL.comp

Every PTX instruction becomes the GLSL expression of the same operation that kernels/native/ptx2hip.py
emits for HIP, so the shader computes what the translated kernel computes. Registers become locals and
basic blocks become guarded sections of a program-counter loop (GLSL has no goto). The 8-byte handles in
the parameter block become bindings: textures by their parameter offset (T<off>), surfaces (S<off>);
pointers stay 64-bit addresses and are read through buffer references.
"""
import re
import sys

REG_RE = re.compile(r'%([a-zA-Z]+)(\d+)$')
GT = {'f': 'float', 'r': 'uint', 'rd': 'uint64_t', 'rs': 'uint', 'p': 'bool'}
SREG = {'%tid.x': 'gl_LocalInvocationID.x', '%tid.y': 'gl_LocalInvocationID.y', '%tid.z': 'gl_LocalInvocationID.z',
        '%ctaid.x': 'gl_WorkGroupID.x', '%ctaid.y': 'gl_WorkGroupID.y', '%ctaid.z': 'gl_WorkGroupID.z',
        '%ntid.x': 'gl_WorkGroupSize.x', '%ntid.y': 'gl_WorkGroupSize.y',
        '%nctaid.x': 'gl_NumWorkGroups.x', '%nctaid.y': 'gl_NumWorkGroups.y', '%laneid': 'gl_SubgroupInvocationID'}


class Err(Exception):
    pass


def tokenize(body):
    out, i, n = [], 0, len(body)
    while i < n:
        c = body[i]
        if c.isspace():
            i += 1; continue
        if c == '{':
            out.append(('open', None)); i += 1; continue
        if c == '}':
            out.append(('close', None)); i += 1; continue
        m = re.match(r'(\$?[A-Za-z_][\w$]*):', body[i:])
        if m:
            out.append(('label', m.group(1))); i += m.end(); continue
        j = body.index(';', i)
        out.append(('stmt', ' '.join(body[i:j].split())))
        i = j + 1
    return out


def split_operands(s):
    ops, depth, cur = [], 0, ''
    for ch in s:
        if ch in '{[':
            depth += 1
        elif ch in '}]':
            depth -= 1
        if ch == ',' and depth == 0:
            ops.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip():
        ops.append(cur.strip())
    return ops


class T:
    def __init__(self, ptx, kernel):
        text = re.sub(r'//[^\n]*', '', ptx)
        m = re.search(r'\.entry\s+' + re.escape(kernel) + r'\s*\((.*?)\)', text, re.S)
        if not m:
            raise Err('kernel not found')
        pm = re.search(r'\.param\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]', m.group(1))
        self.kernel, self.pname, self.psize = kernel, pm.group(2), int(pm.group(3))
        rest = text[m.end():]
        mt = re.search(r'\.maxntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', rest[:rest.index('{')])
        self.maxntid = [int(x) for x in mt.groups()] if mt else [256, 1, 1]
        start = rest.index('{'); depth, k = 0, start
        while True:
            if rest[k] == '{':
                depth += 1
            elif rest[k] == '}':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        self.toks = tokenize(rest[start + 1:k])
        self.regs = {}
        self.shared = {}       # name -> byte base
        self.shared_bytes = 0
        self.scopes = [{}]
        self.scope_id = 0
        self.blocks = [[]]     # lists of lines
        self.labels = {}       # label -> block index
        self.textures, self.surfaces = set(), {}
        self.dyn_tex = False
        self.fast = False      # --fast: no 'precise', the compiler may contract and reassociate
        self.collect = False   # first pass: gather how every 16/32-bit register is used
        self.ctx, self.links, self.width, self.half = {}, [], {}, set()
        self.exp_lut = False
        self.prescan()

    # ---------------------------------------------------------------- static facts about registers
    def prescan(self):
        defs = {}
        for kind, s in self.toks:
            if kind != 'stmt' or s.startswith('.'):
                continue
            s = re.sub(r'^@!?%p\d+\s+', '', s)
            parts = s.split(None, 1)
            if len(parts) < 2:
                continue
            a = split_operands(parts[1])
            for d in re.findall(r'%rd\d+', a[0]) if not a[0].startswith('[') else []:
                defs.setdefault(d, []).append((parts[0], a))
        self.pbase = {}        # reg -> constant offset from the parameter block
        self.handle = {}       # reg -> parameter offset its 8-byte value was loaded from
        changed = True
        while changed:
            changed = False
            for r, ds in defs.items():
                if len(ds) != 1:
                    continue
                op, a = ds[0]
                if r not in self.pbase:
                    v = None
                    if op.startswith('mov') and a[1] == self.pname:
                        v = 0
                    elif op in ('add.s64', 'add.u64') and a[1] in self.pbase and re.match(r'-?\d+$', a[2]):
                        v = self.pbase[a[1]] + int(a[2])
                    if v is not None:
                        self.pbase[r] = v; changed = True
                if r not in self.handle:
                    v = None
                    if op == 'ld.param.u64':
                        v = self.param_off(a[1])
                    elif op.startswith('mov') and a[1] in self.handle:
                        v = self.handle[a[1]]
                    if v is not None:
                        self.handle[r] = v; changed = True

    def param_off(self, tok):
        m = re.match(r'\[\s*([^\]+]+?)\s*(?:\+\s*(-?\d+))?\s*\]$', tok)
        base, off = m.group(1), int(m.group(2) or 0)
        if base == self.pname:
            return off
        if base in self.pbase:
            return self.pbase[base] + off
        return None

    # ---------------------------------------------------------------- operands
    def reg(self, tok):
        tok = tok.strip()
        for sc in reversed(self.scopes):
            if tok in sc:
                return sc[tok]
        m = REG_RE.match(tok)
        if not m or m.group(1) not in GT:
            raise Err('not a register: ' + tok)
        name = m.group(1) + m.group(2)
        if m.group(1) in ('r', 'rs'):
            self.width[name] = 32 if m.group(1) == 'r' else 16
        self.regs[name] = self.htype(name) or GT[m.group(1)]
        return name

    def htype(self, name):
        return ('f16vec2' if self.width[name] == 32 else 'float16_t') if name in self.half else None

    def note(self, name, cls):
        if self.collect:
            self.ctx.setdefault(name, set()).add(cls)

    def hk(self, name):
        return 'f16x2' if self.width.get(name) == 32 else 'f16'

    def is_reg(self, tok):
        return tok.startswith('%') and tok not in SREG or any(tok in sc for sc in self.scopes)

    def val(self, tok, t):
        tok = tok.strip()
        if tok in SREG:
            return f'uint({SREG[tok]})' if t != 's32' else f'int({SREG[tok]})'
        if tok in ('%SP', '%SPL'):
            return 'uint64_t(0ul)'          # the local depot is its own array; addresses in it start at 0
        if self.is_reg(tok):
            n = self.reg(tok); g = self.regs[n]
            self.note(n, 'h' if t in ('f16', 'f16x2') else 'u')
            if g in ('float16_t', 'f16vec2'):
                if t in ('f16', 'f16x2'):
                    return n
                bits = f'HU({n})' if g == 'float16_t' else f'H2U({n})'
                if t == 's16':
                    return f's16({bits})'
                if t == 's32':
                    return f'int({bits})'
                if t in ('u64', 'b64'):
                    return f'uint64_t({bits})'
                return bits
            if t == 'f32':
                return n if g == 'float' else f'uintBitsToFloat({n})'
            if t in ('u32', 'b32'):
                return f'floatBitsToUint({n})' if g == 'float' else (f'uint({n})' if g == 'uint64_t' else n)
            if t == 's32':
                return f'floatBitsToInt({n})' if g == 'float' else f'int({n})'
            if t in ('u64', 'b64'):
                return n if g == 'uint64_t' else f'uint64_t({n})'
            if t == 's64':
                return f'int64_t({n})'
            if t in ('u16', 'b16'):
                return n
            if t == 's16':
                return f's16({n})'
            if t == 'f16':
                return f'H({n})'
            if t == 'f16x2':
                return f'H2({n})'
            if t == 'pred':
                return n
            raise Err('type ' + t)
        if tok in self.shared:
            return f'{self.shared[tok]}u'
        if re.match(r'0[fF][0-9A-Fa-f]{8}$', tok):
            bits = int(tok[2:], 16)
            return f'uintBitsToFloat(0x{bits:08X}u)' if t == 'f32' else f'0x{bits:08X}u'
        if re.match(r'-?\d+$', tok) or re.match(r'0x[0-9a-fA-F]+$', tok):
            v = int(tok, 0)
            if t == 'f32':
                raise Err('int immediate as float')
            if t in ('u64', 'b64'):
                return f'uint64_t({v & 0xffffffffffffffff}ul)'
            if t == 's64':
                return f'int64_t({v}l)'
            if t in ('u16', 'b16'):
                return f'{v & 0xffff}u'
            if t == 's16':
                v &= 0xffff
                return f'({v - 0x10000 if v >= 0x8000 else v})'
            if t == 'f16':
                return f'H({v & 0xffff}u)'
            if t == 'f16x2':
                return f'H2({v & 0xffffffff}u)'
            if t == 's32':
                v &= 0xffffffff
                return f'int(0x{v:08X}u)' if v >= 1 << 31 else f'{v}'
            return f'{v & 0xffffffff}u'
        raise Err('operand ' + tok)

    def A(self, dst, expr, st):
        n = self.reg(dst); g = self.regs[n]
        self.note(n, 'h' if st in ('f16', 'f16x2') else 'u')
        if g == 'bool' and st in ('u32', 'u16', 'u64'):
            self.E(f'{n} = ({expr}) != 0u;'); return          # mov.pred with an immediate
        if g in ('float16_t', 'f16vec2'):
            if st in ('f16', 'f16x2'):
                self.E(f'{n} = {expr};')
            else:
                self.E(f'{n} = {"H" if g == "float16_t" else "H2"}(uint({expr}));')
            return
        if st == 'float':
            e = expr if g == 'float' else f'floatBitsToUint({expr})'
        elif st == 'u32':
            e = f'uintBitsToFloat({expr})' if g == 'float' else (f'uint64_t({expr})' if g == 'uint64_t' else f'uint({expr})')
        elif st == 'u64':
            e = f'uint64_t({expr})'
        elif st == 'u16':
            e = f'(uint({expr}) & 0xffffu)'
        elif st == 'f16':
            e = f'HU({expr})'
        elif st == 'f16x2':
            e = f'H2U({expr})'
        elif st == 'bool':
            e = f'({expr})'
        else:
            raise Err('src type')
        self.E(f'{n} = {e};')

    def E(self, s):
        self.blocks[-1].append(s)

    # ---------------------------------------------------------------- driver
    def translate(self):
        for kind, text in self.toks:
            if kind == 'open':
                self.scopes.append({}); continue
            if kind == 'close':
                self.scopes.pop(); continue
            if kind == 'label':
                self.blocks.append([])
                self.labels[text] = len(self.blocks) - 1
                continue
            self.stmt(text)

    def stmt(self, s):
        if s.startswith('.reg'):
            m = re.match(r'\.reg\s+\.(\w+)\s+(.*)$', s)
            ty, names = m.groups()
            if '<' in names:
                return
            self.scope_id += 1
            for nm in names.split(','):
                nm = nm.strip()
                c = f'l{self.scope_id}_' + re.sub(r'\W', '_', nm)
                self.scopes[-1][nm] = c
                if ty in ('f16', 'b16', 'f16x2', 'b32', 'u32', 's32'):
                    self.width[c] = 16 if ty in ('f16', 'b16') else 32
                self.regs[c] = self.htype(c) if c in self.half else {'f16': 'uint', 'b16': 'uint', 'f32': 'float', 'b32': 'uint', 'pred': 'bool', 'f16x2': 'uint',
                                'u32': 'uint', 's32': 'uint', 'b64': 'uint64_t', 'u64': 'uint64_t'}[ty]
            return
        if s.startswith('.local'):
            m = re.match(r'\.local\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]', s)
            self.local_bytes = max(getattr(self, 'local_bytes', 0), int(m.group(3)))
            return
        if re.match(r'mov\.u64 %SPL?, __local_depot', s):
            return
        if s.startswith('.shared'):
            m = re.match(r'\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]', s)
            self.shared[m.group(2)] = self.shared_bytes
            self.shared_bytes += (int(m.group(3)) + 3) & ~3
            return
        if s.startswith('.'):
            return
        pred = None
        m = re.match(r'@(!?)(%p\d+)\s+(.*)$', s)
        if m:
            pred = ('!' if m.group(1) else '') + self.reg(m.group(2)); s = m.group(3)
        parts = s.split(None, 1)
        op = parts[0]
        a = split_operands(parts[1]) if len(parts) > 1 else []
        if op.startswith('bra'):
            tgt = ('L', a[0])
            if pred is not None:
                self.E(('BR', pred, tgt))          # conditional: falls through into a new block
            else:
                self.E(('BR', None, tgt))
            self.blocks.append([])
            return
        if op == 'ret':
            self.E(('BR', pred, ('END', None)))
            self.blocks.append([])
            return
        if pred is not None:
            self.E(f'if ({pred}) {{')
            self.op(op, a)
            self.E('}')
            return
        self.op(op, a)

    def cmp(self, mods, isf):
        table = {'lt': '{0} < {1}', 'gt': '{0} > {1}', 'le': '{0} <= {1}', 'ge': '{0} >= {1}', 'eq': '{0} == {1}', 'ne': '{0} != {1}',
                 'ltu': '!({0} >= {1})', 'gtu': '!({0} <= {1})', 'leu': '!({0} > {1})', 'geu': '!({0} < {1})',
                 'equ': '!({0} != {1})', 'neu': '!({0} == {1})'}
        for k in table:
            if k in mods:
                if isf and k == 'ne':
                    return '({0} < {1} || {0} > {1})'
                if isf and k == 'equ':
                    return '!({0} < {1} || {0} > {1})'
                return table[k]
        raise Err('compare')

    def texref(self, tok):
        m = re.match(r'\[\s*(%rd\d+)\s*,\s*\{([^}]*)\}\s*\]', tok)
        return m.group(1), [x.strip() for x in m.group(2).split(',')]

    def op(self, op, a):
        o = op.split('.'); base = o[0]; mods = set(o[1:])
        E, A, V = self.E, self.A, self.val
        ft = next((t for t in ('f32', 'f16x2', 'f16') if t in mods), None)
        if base == 'bar':
            E('memoryBarrierShared(); barrier();'); return
        if base == 'mov':
            dst, src = a
            if dst.startswith('{'):
                n = [x.strip() for x in dst[1:-1].split(',')]
                sr = self.reg(src) if self.is_reg(src) else None
                if self.collect and sr:
                    for x in [sr] + [self.reg(y) for y in n]:
                        self.ctx.setdefault(x, set()).add('h')
                    return
                if sr and (sr in self.half or any(self.reg(y) in self.half for y in n)):
                    A(n[0], f'{V(src, "f16x2")}.x', 'f16'); A(n[1], f'{V(src, "f16x2")}.y', 'f16'); return
                A(n[0], f'{V(src, "u32")} & 0xffffu', 'u16'); A(n[1], f'{V(src, "u32")} >> 16', 'u16'); return
            if src.startswith('{'):
                n = [x.strip() for x in src[1:-1].split(',')]
                dr = self.reg(dst)
                if self.collect:
                    for x in [dr] + [self.reg(y) for y in n if self.is_reg(y)]:
                        self.ctx.setdefault(x, set()).add('h')
                    return
                if dr in self.half:
                    A(dst, f'f16vec2({V(n[0], "f16")}, {V(n[1], "f16")})', 'f16x2'); return
                A(dst, f'({V(n[0], "u16")} & 0xffffu) | ({V(n[1], "u16")} << 16)', 'u32'); return
            if self.is_reg(dst) and self.reg(dst) in self.width and not (mods & {'f32', 'u64', 'b64', 's64'}):
                dr = self.reg(dst)
                srcreg = self.reg(src) if self.is_reg(src) and src not in SREG else None
                isimm = bool(re.match(r'-?\d+$', src) or re.match(r'0x[0-9a-fA-F]+$', src))
                if self.collect and srcreg and srcreg in self.width:
                    self.links.append((dr, srcreg)); return
                if self.collect and isimm:
                    return
                if dr in self.half and (srcreg or isimm):
                    A(dst, V(src, self.hk(dr)), self.hk(dr)); return
            if src == self.pname:
                A(dst, '0ul', 'u64'); return
            if 'f32' in mods:
                A(dst, V(src, 'f32'), 'float'); return
            if mods & {'u64', 'b64', 's64'}:
                A(dst, V(src, 'u64'), 'u64'); return
            if mods & {'u16', 'b16', 's16'}:
                A(dst, V(src, 'u16'), 'u16'); return
            A(dst, V(src, 'u32'), 'u32'); return
        # ------------------------------------------------ f32
        if ft == 'f32' and base in ('add', 'sub', 'mul'):
            if 'rz' in mods:
                A(a[0], f'round_half_away({V(a[1], "f32")})', 'float'); return     # the roundf idiom: + copysign(0.5), then trunc
            A(a[0], f'{V(a[1], "f32")} {"+-*"[["add", "sub", "mul"].index(base)]} {V(a[2], "f32")}', 'float'); return
        if ft == 'f32' and base == 'fma':
            A(a[0], f'fma({V(a[1], "f32")}, {V(a[2], "f32")}, {V(a[3], "f32")})', 'float'); return
        if ft == 'f32' and base in ('max', 'min'):
            A(a[0], f'{base}({V(a[1], "f32")}, {V(a[2], "f32")})', 'float'); return
        if ft == 'f32' and base == 'neg':
            A(a[0], f'-{V(a[1], "f32")}', 'float'); return
        if ft == 'f32' and base == 'abs':
            A(a[0], f'abs({V(a[1], "f32")})', 'float'); return
        if ft == 'f32' and base in ('rcp', 'ex2', 'lg2', 'rsqrt', 'sqrt'):
            fn = {'rcp': 'rcp_a', 'ex2': 'exp2', 'lg2': 'log2', 'rsqrt': 'inversesqrt', 'sqrt': 'sqrt'}[base]
            A(a[0], f'{fn}({V(a[1], "f32")})', 'float'); return
        if ft == 'f32' and base == 'div':
            A(a[0], f'{V(a[1], "f32")} * rcp_a({V(a[2], "f32")})', 'float'); return
        # ------------------------------------------------ f16
        if ft == 'f16x2' and base in ('add', 'sub', 'mul', 'fma'):
            x = [V(t, 'f16x2') for t in a[1:]]
            e = f'fma({x[0]}, {x[1]}, {x[2]})' if base == 'fma' else f'({x[0]} {"+-*"[["add", "sub", "mul"].index(base)]} {x[1]})'
            if 'sat' in mods:
                e = f'min(max({e}, f16vec2(0.0hf)), f16vec2(1.0hf))'
            A(a[0], e, 'f16x2'); return
        if ft == 'f16x2' and base == 'abs':
            A(a[0], f'abs({V(a[1], "f16x2")})', 'f16x2'); return
        if ft == 'f16x2' and base == 'ex2':
            if self.exp_lut:
                A(a[0], f'ex2_lut({V(a[1], "u32")})', 'u32'); return
            A(a[0], f'f16vec2(exp2(vec2({V(a[1], "f16x2")})))', 'f16x2'); return
        if ft in ('f16', 'f16x2') and base in ('min', 'max'):
            A(a[0], f'{base}({V(a[1], ft)}, {V(a[2], ft)})', ft); return
        if ft == 'f16' and base in ('add', 'sub', 'mul', 'fma'):
            x = [V(t, 'f16') for t in a[1:]]
            e = f'fma({x[0]}, {x[1]}, {x[2]})' if base == 'fma' else f'({x[0]} {"+-*"[["add", "sub", "mul"].index(base)]} {x[1]})'
            if 'sat' in mods:
                e = f'min(max({e}, 0.0hf), 1.0hf)'
            A(a[0], e, 'f16'); return
        if base == 'set' and 'f16x2' in mods and 'u32' in mods:
            c = self.cmp(mods, True); x, y = V(a[1], 'f16x2'), V(a[2], 'f16x2')
            A(a[0], f'(({c.format(x + ".x", y + ".x")}) ? 0xffffu : 0u) | (({c.format(x + ".y", y + ".y")}) ? 0xffff0000u : 0u)', 'u32'); return
        # ------------------------------------------------ compares / selects
        if base == 'setp':
            isf = bool(mods & {'f32', 'f16'})
            t = next(t for t in ('f32', 'f16', 's32', 'u32', 'b32', 's16', 'u16', 'b16', 's64', 'u64') if t in mods)
            if '|' in a[0]:
                raise Err('setp form')
            A(a[0], self.cmp(mods, isf).format(V(a[1], t), V(a[2], t)), 'bool'); return
        if base == 'selp' and not (mods & {'f32', 'u64', 'b64', 's64'}) and self.reg(a[0]) in self.width:
            dr = self.reg(a[0])
            srcs = [self.reg(x) for x in a[1:3] if self.is_reg(x) and x not in SREG]
            if self.collect:
                self.links += [(dr, x) for x in srcs if x in self.width]
                V(a[3], 'pred')
                if any(x not in self.width for x in srcs):
                    self.note(dr, 'u')
                return
            if dr in self.half:
                k = self.hk(dr)
                A(a[0], f'{V(a[3], "pred")} ? {V(a[1], k)} : {V(a[2], k)}', k); return
        if base == 'selp':
            t = 'f32' if 'f32' in mods else 'u16' if mods & {'u16', 'b16', 's16'} else 'u64' if mods & {'u64', 'b64', 's64'} else 'u32'
            A(a[0], f'{V(a[3], "pred")} ? {V(a[1], t)} : {V(a[2], t)}', {'f32': 'float', 'u16': 'u16', 'u64': 'u64', 'u32': 'u32'}[t]); return
        if 'pred' in mods and base in ('and', 'or', 'xor', 'not'):
            if base == 'not':
                A(a[0], f'!{V(a[1], "pred")}', 'bool'); return
            A(a[0], f'{V(a[1], "pred")} {{"and": "&&", "or": "||", "xor": "!="}}'.replace('{"and": "&&", "or": "||", "xor": "!="}', {'and': '&&', 'or': '||', 'xor': '!='}[base]) + f' {V(a[2], "pred")}', 'bool'); return
        # ------------------------------------------------ integer
        if base in ('and', 'or', 'xor', 'not', 'shl', 'shr', 'add', 'sub', 'mul', 'mad', 'min', 'max', 'div', 'rem', 'neg', 'abs') and ft is None:
            w = 64 if mods & {'u64', 's64', 'b64'} else 16 if mods & {'u16', 's16', 'b16'} else 32
            signed = bool(mods & {'s32', 's64', 's16'})
            ut, st = {64: ('u64', 's64'), 32: ('u32', 's32'), 16: ('u16', 's16')}[w]
            if base == 'mul' and 'wide' in mods:
                if 's32' in mods:
                    A(a[0], f'uint64_t(int64_t({V(a[1], "s32")}) * int64_t({V(a[2], "s32")}))', 'u64'); return
                if 'u32' in mods:
                    A(a[0], f'uint64_t({V(a[1], "u32")}) * uint64_t({V(a[2], "u32")})', 'u64'); return
                if 'u16' in mods:
                    A(a[0], f'{V(a[1], "u16")} * {V(a[2], "u16")}', 'u32'); return
                raise Err(op)
            if base == 'not':
                A(a[0], f'~{V(a[1], ut)}', ut); return
            if base == 'neg':
                A(a[0], f'0u - {V(a[1], ut)}', ut); return
            if base == 'shl':
                A(a[0], f'{V(a[1], ut)} << ({V(a[2], "u32")} & {w - 1}u)', ut); return
            if base == 'shr':
                x = V(a[1], st if signed else ut)
                A(a[0], f'{x} >> ({V(a[2], "u32")} & {w - 1}u)', ut); return
            if base in ('and', 'or', 'xor'):
                A(a[0], f'{V(a[1], ut)} {{"and": "&", "or": "|", "xor": "^"}}'.replace('{"and": "&", "or": "|", "xor": "^"}', {'and': '&', 'or': '|', 'xor': '^'}[base]) + f' {V(a[2], ut)}', ut); return
            if base in ('add', 'sub'):
                A(a[0], f'{V(a[1], ut)} {"+" if base == "add" else "-"} {V(a[2], ut)}', ut); return
            if base == 'mul' and 'hi' in mods:
                if signed:
                    A(a[0], f'uint(int((int64_t({V(a[1], "s32")}) * int64_t({V(a[2], "s32")})) >> 32))', 'u32'); return
                A(a[0], f'uint((uint64_t({V(a[1], "u32")}) * uint64_t({V(a[2], "u32")})) >> 32)', 'u32'); return
            if base == 'mul':
                A(a[0], f'{V(a[1], ut)} * {V(a[2], ut)}', ut); return
            if base == 'mad':
                A(a[0], f'{V(a[1], ut)} * {V(a[2], ut)} + {V(a[3], ut)}', ut); return
            if base == 'abs':
                A(a[0], f'uint(abs({V(a[1], st)}))', ut); return
            if base == 'neg' and len(a) == 2:
                A(a[0], f'uint(-{V(a[1], st)})', ut); return
            x, y = V(a[1], st if signed else ut), V(a[2], st if signed else ut)
            if base in ('min', 'max'):
                A(a[0], f'{base}({x}, {y})', ut); return
            if base == 'div':
                A(a[0], f'{x} / {y}', ut); return
            if base == 'rem':
                A(a[0], f'{x} % {y}', ut); return
        if base == 'bfi':
            A(a[0], f'bfi32({V(a[1], "u32")}, {V(a[2], "u32")}, {V(a[3], "u32")}, {V(a[4], "u32")})', 'u32'); return
        if base == 'cvt':
            return self.cvt(op, o[1:], a)
        if base == 'cvta':
            A(a[0], V(a[1], 'u64'), 'u64'); return
        if base == 'ld':
            return self.ld(o[1:], a)
        if base == 'st':
            return self.st(o[1:], a)
        if base in ('tex', 'tld4'):
            dst = [x.strip() for x in a[0][1:-1].split(',')]
            h, c = self.texref(a[1])
            off = self.handle.get(h)
            if off is None:
                raise Err('texture handle of ' + h + ' is not a single parameter load')
            self.textures.add(off)
            if base == 'tld4':
                E(f'{{ vec4 t_ = textureGather(T{off}, vec2({V(c[0], "f32")}, {V(c[1], "f32")}), 0);')
            elif 'base' in mods:
                E(f'{{ vec4 t_ = texelFetch(T{off}, ivec2({V(c[0], "s32")}, {V(c[1], "s32")}), 0);')
            elif 'level' in mods:
                E(f'{{ vec4 t_ = textureLod(T{off}, vec2({V(c[0], "f32")}, {V(c[1], "f32")}), {V(a[2], "f32")});')
            else:
                raise Err(op)
            for i, d in enumerate(dst):
                if 'f16' in o:
                    A(d, f'float16_t(t_[{i}])', 'f16')       # the texel as a half float
                else:
                    A(d, f't_[{i}]', 'float')
            E('}')
            return
        if base == 'sust':
            h, c = self.texref(a[0])
            off = self.handle.get(h)
            if off is None:
                raise Err('surface handle of ' + h)
            v = [x.strip() for x in a[1][1:-1].split(',')]
            if op == 'sust.p.2d.v4.b32.zero':
                self.surfaces.setdefault(off, 'p'); assert self.surfaces[off] == 'p'
                E(f'sust_p(S{off}, {V(c[0], "s32")}, {V(c[1], "s32")}, vec4({", ".join(V(x, "f32") for x in v)}));'); return
            if op == 'sust.b.2d.b32.zero':
                self.surfaces.setdefault(off, 'b'); assert self.surfaces[off] == 'b'
                E(f'sust_b(S{off}, {V(c[0], "s32")}, {V(c[1], "s32")}, {V(v[0], "u32")});'); return
            if op == 'sust.b.2d.v2.b16.zero':       # reading only: two 16-bit values at a byte-addressed position
                self.surfaces.setdefault(off, 'b')
                E(f'sust_b2x16(S{off}, {V(c[0], "s32")}, {V(c[1], "s32")}, {V(v[0], "u16")}, {V(v[1], "u16")});'); return
            raise Err(op)
        if op == 'shfl.sync.bfly.b32':
            d = a[0].split('|')
            E('{ bool ib_;')
            A(d[0], f'shfl_bfly({V(a[1], "u32")}, {V(a[2], "s32")}, {V(a[3], "u32")}, ib_)', 'u32')
            if len(d) > 1:
                A(d[1], 'ib_', 'bool')
            E('}')
            return
        if op == 'shfl.sync.idx.b32':
            d = a[0].split('|')
            A(d[0], f'subgroupShuffle({V(a[1], "u32")}, {V(a[2], "u32")} & 31u)', 'u32')
            if len(d) > 1:
                A(d[1], 'true', 'bool')
            return
        if op == 'tanh.approx.f16':
            A(a[0], f'tanh({V(a[1], "f16")})', 'f16'); return
        if op == 'prmt.b32':
            A(a[0], f'prmt({V(a[1], "u32")}, {V(a[2], "u32")}, {V(a[3], "u32")})', 'u32'); return
        if op.startswith('mma.sync'):
            # not executable here: the kernel is translated for reading only (tensor-core fragments are not emulated)
            self.E(f'// {op} ' + ' '.join(a))
            for d in a[0][1:-1].split(','):
                A(d.strip(), '0u', 'u32')
            return
        if op == 'vote.sync.ballot.b32':
            # the warp is the subgroup (32 lanes in thread order), as for the shuffles
            A(a[0], f'subgroupBallot({V(a[1], "pred")}).x', 'u32'); return
        raise Err('unhandled ' + op)

    def cvt(self, op, o, a):
        A, V = self.A, self.val
        mods = set(o)
        types = [t for t in o if re.match(r'^(f|s|u|b)\d+(x2)?$', t)]
        types = [t for t in o if re.match(r'^(f|s|u|b)\d+(x2)?$|^e4m3x2$', t)]
        dt, st = types[0], types[1]
        # FP8 pairs (reading only: the helpers are not defined)
        if dt == 'f16x2' and st == 'e4m3x2':
            A(a[0], f'e4m3x2_to_f16x2({V(a[1], "u16")})', 'f16x2'); return
        if dt == 'e4m3x2' and st == 'f16x2':
            A(a[0], f'f16x2_to_e4m3x2({V(a[1], "f16x2")})', 'u16'); return
        if dt == 'f32' and st == 'f32':
            x = V(a[1], 'f32')
            if 'sat' in mods:
                A(a[0], f'min(max({x}, 0.0), 1.0)', 'float'); return
            fn = 'trunc' if 'rzi' in mods else 'floor' if 'rmi' in mods else 'roundEven' if 'rni' in mods else 'ceil' if 'rpi' in mods else ''
            A(a[0], f'{fn}({x})', 'float'); return
        if dt == 's32' and st == 'f32':
            x = V(a[1], 'f32')
            fn = '' if 'rzi' in mods else 'floor' if 'rmi' in mods else 'roundEven'
            A(a[0], f'uint(int({fn}({x})))', 'u32'); return
        if dt == 'u32' and st == 'f32':
            x = V(a[1], 'f32')
            fn = 'trunc' if 'rzi' in mods else 'floor' if 'rmi' in mods else 'roundEven'
            A(a[0], f'uint(clamp({fn}({x}), 0.0, 4294967040.0))', 'u32'); return      # CUDA saturates
        if dt == 'f32' and st == 's32':
            A(a[0], f'float({V(a[1], "s32")})', 'float'); return
        if dt == 'f32' and st == 'u32':
            A(a[0], f'float({V(a[1], "u32")})', 'float'); return
        if dt == 'f32' and st == 'f16':
            A(a[0], f'float({V(a[1], "f16")})', 'float'); return
        if dt == 'f16' and st == 'f32':
            A(a[0], f'float16_t({V(a[1], "f32")})', 'f16'); return
        if dt == 'f16x2' and st == 'f32':
            A(a[0], f'f16vec2(float16_t({V(a[2], "f32")}), float16_t({V(a[1], "f32")}))', 'f16x2'); return
        if dt == 'f16' and st == 'f16':
            fn = 'roundEven' if 'rni' in mods else 'floor'
            A(a[0], f'{fn}({V(a[1], "f16")})', 'f16'); return
        if dt == 's32' and st == 'f16':
            fn = 'roundEven' if 'rni' in mods else 'floor'
            A(a[0], f'uint(int(float({fn}({V(a[1], "f16")}))))', 'u32'); return
        if dt == 'f16' and st == 's32':
            A(a[0], f'float16_t({V(a[1], "s32")})', 'f16'); return
        if dt == 's64' and st == 's32':
            A(a[0], f'uint64_t(int64_t({V(a[1], "s32")}))', 'u64'); return
        if dt == 'u64' and st == 'u32':
            A(a[0], f'uint64_t({V(a[1], "u32")})', 'u64'); return
        if dt in ('u16', 's16') and st in ('u32', 's32'):
            A(a[0], V(a[1], 'u32'), 'u16'); return
        if dt in ('u32', 's32') and st in ('u16',):
            A(a[0], V(a[1], 'u16'), 'u32'); return
        if dt == 's32' and st == 's16':
            A(a[0], f'uint({V(a[1], "s16")})', 'u32'); return
        raise Err('cvt ' + op)

    def addr(self, tok):
        m = re.match(r'\[\s*([^\]+]+?)\s*(?:\+\s*(-?\d+))?\s*\]$', tok)
        return m.group(1), int(m.group(2) or 0)

    def ld(self, o, a):
        A, V, E = self.A, self.val, self.E
        mods = set(o)
        ty = next(t for t in o if re.match(r'^(f|s|u|b)\d+$', t))
        vec = next((int(t[1]) for t in o if re.match(r'^v\d$', t)), 1)
        size = int(ty[1:]) // 8
        dsts = [x.strip() for x in a[0][1:-1].split(',')] if a[0].startswith('{') else [a[0]]
        base, off = self.addr(a[1])
        for i, d in enumerate(dsts):
            if 'param' in mods:
                po = self.param_off(a[1])
                if po is None:
                    raise Err('param address ' + a[1])
                e = f'PARAM{size * 8}({po + i * size}u)'
            elif 'shared' in mods:
                e = f'SH{size * 8}({V(base, "u32")} + {off + i * size}u)'
            elif 'local' in mods:
                e = f'loc[(uint({V(base, "u64")}) + {off + i * size}u) >> 2]'
            else:
                e = f'G{size * 8}({V(base, "u64")} + uint64_t({off + i * size}l))'
            if size == 8:
                A(d, e, 'u64')
            elif size == 2:
                A(d, e, 'u16')
            elif size == 1:
                A(d, e, 'u16')
            else:
                A(d, e, 'u32')

    def st(self, o, a):
        V, E = self.val, self.E
        mods = set(o)
        ty = next(t for t in o if re.match(r'^(f|s|u|b)\d+$', t))
        size = int(ty[1:]) // 8
        base, off = self.addr(a[0])
        if 'shared' in mods:
            # vector stores go word by word; 16-bit pairs share a word (the kernels keep them 4-byte aligned)
            vals = [x.strip() for x in a[1][1:-1].split(',')] if a[1].startswith('{') else [a[1]]
            if size == 4:
                for i, v in enumerate(vals):
                    E(f'sh[({V(base, "u32")} + {off + 4 * i}u) >> 2] = {V(v, "u32")};')
                return
            if size == 2 and len(vals) % 2 == 0:
                for i in range(0, len(vals), 2):
                    E(f'sh[({V(base, "u32")} + {off + 2 * i}u) >> 2] = (uint({V(vals[i], "u16")}) & 0xFFFFu) | (uint({V(vals[i + 1], "u16")}) << 16);')
                return
        if 'local' in mods and size == 4:
            vals = [x.strip() for x in a[1][1:-1].split(',')] if a[1].startswith('{') else [a[1]]
            for i, v in enumerate(vals):
                E(f'loc[(uint({V(base, "u64")}) + {off + 4 * i}u) >> 2] = {V(v, "u32")};')
            return
        if 'global' in mods and size == 2 and not a[1].startswith('{'):
            E(f'W16({V(base, "u64")} + uint64_t({off}l)).v[0] = uint16_t({V(a[1], "u16")});'); return
        if 'global' in mods and size == 2 and a[1].startswith('{'):
            vals = [x.strip() for x in a[1][1:-1].split(',')]
            E(f'{{ W16 w_ = W16({V(base, "u64")} + uint64_t({off}l)); ' + ' '.join(f'w_.v[{i}] = uint16_t({V(v, "u16")});' for i, v in enumerate(vals)) + ' }'); return
        if 'global' in mods and size == 4:
            E(f'{{ W16 w_ = W16({V(base, "u64")} + uint64_t({off}l)); uint v_ = {V(a[1], "u32")}; w_.v[0] = uint16_t(v_ & 0xffffu); w_.v[1] = uint16_t(v_ >> 16); }}'); return
        raise Err('store ' + '.'.join(o))

    # ---------------------------------------------------------------- output
    def emit(self, tex_filter):
        nb = len(self.blocks)
        out = []
        w = out.append
        w('#version 460')
        # scripts/check_redistributable.py refuses files carrying this line: they are NVIDIA's code.
        w(f'// Generated by vulkan/ptx2glsl.py from NVIDIA PTX ({self.kernel}); do not commit or redistribute.')
        for e in ('GL_EXT_shader_explicit_arithmetic_types', 'GL_EXT_buffer_reference', 'GL_EXT_buffer_reference2', 'GL_EXT_scalar_block_layout',
                  'GL_KHR_shader_subgroup_basic', 'GL_KHR_shader_subgroup_shuffle', 'GL_KHR_shader_subgroup_ballot', 'GL_EXT_samplerless_texture_functions'):
            w(f'#extension {e} : require')
        bx = 16 if self.maxntid[0] == 256 else self.maxntid[0]
        by = self.maxntid[0] // bx
        w(f'layout(local_size_x = {bx}, local_size_y = {by}) in;')
        w(f'layout(set = 0, binding = 0, std430) readonly buffer ParamBlock {{ uint w[{(self.psize + 3) // 4}]; }} param;')
        b = 1
        for off in sorted(self.textures):
            w(f'layout(set = 0, binding = {b}) uniform sampler2D T{off};   // {tex_filter.get(off, "?")}')
            b += 1
        for off in sorted(self.surfaces):
            fmt = 'r16ui' if self.surfaces[off] == 'b' else 'rgba16f'
            typ = 'uimage2D' if self.surfaces[off] == 'b' else 'image2D'
            w(f'layout(set = 0, binding = {b}, {fmt}) uniform writeonly {typ} S{off};')
            b += 1
        if self.exp_lut:
            w(f'layout(set = 0, binding = {b}, scalar) readonly buffer ExpLut {{ uint16_t e[65536]; }} explut;')
            w('uint ex2_lut(uint u) { return uint(explut.e[u & 0xffffu]) | (uint(explut.e[u >> 16]) << 16); }')
        w(f'shared uint sh[{max(self.shared_bytes // 4, 1)}];')
        w(RUNTIME)
        w('void main()\n{')
        if getattr(self, 'local_bytes', 0):
            w(f'    uint loc[{(self.local_bytes + 3) // 4}];')
        for g in ('float', 'float16_t', 'f16vec2', 'uint', 'uint64_t', 'bool'):
            names = sorted(n for n, t in self.regs.items() if t == g)
            for i in range(0, len(names), 24):
                init = {'float': '0.0', 'float16_t': '0.0hf', 'f16vec2': 'f16vec2(0.0hf)', 'uint': '0u', 'uint64_t': '0ul', 'bool': 'false'}[g]
                w(f'    {"precise " if g in ("float", "float16_t", "f16vec2") and not self.fast else ""}{g} ' + ', '.join(f'{n} = {init}' for n in names[i:i + 24]) + ';')
        # control flow graph over the blocks; END is block nb
        succ = []
        for i, blk in enumerate(self.blocks):
            br = blk[-1] if blk and isinstance(blk[-1], tuple) else None
            if br is None:
                succ.append([min(i + 1, nb)])
            else:
                tgt = nb if br[2][0] == 'END' else self.labels[br[2][1]]
                succ.append([tgt] if br[1] is None else [tgt, i + 1])
        # ---- structured emission: if/else arms that rejoin at the branch's immediate post-dominator --------------
        END = nb
        # reachable nodes, back edges (to a node on the DFS stack)
        color, back = {}, set()
        st = [(0, iter(succ[0]))]; color[0] = 1
        while st:
            n, it = st[-1]
            for m in it:
                if m == END:
                    continue
                if color.get(m) == 1:
                    back.add((n, m))
                elif m not in color:
                    color[m] = 1; st.append((m, iter(succ[m]))); break
            else:
                color[n] = 2; st.pop()
        nodes = sorted(color)
        # each loop header h gets a synthetic "continue" node: back edges go there, and it flows to the loop's exit
        headers = sorted({h for _, h in back})
        cont = {h: END + 1 + i for i, h in enumerate(headers)}
        g = {n: [(cont[m] if (n, m) in back else m) for m in succ[n]] for n in nodes}
        for h in headers:
            latches = [n for n, m in back if m == h]
            exits = {m for n in latches for m in succ[n] if (n, m) not in back}
            if len(exits) != 1:
                raise Err('loop without a single latch exit')
            g[cont[h]] = [exits.pop()]
        g[END] = []
        allnodes = list(g)
        # immediate post-dominators (iterative, on the acyclic graph g)
        rpo, seen = [], set()
        def dfs(n):
            stack = [(n, iter(g[n]))]; seen.add(n)
            while stack:
                x, it = stack[-1]
                for m in it:
                    if m not in seen:
                        seen.add(m); stack.append((m, iter(g[m]))); break
                else:
                    rpo.append(x); stack.pop()
        dfs(0)
        topo = rpo[::-1]                      # entry first
        index = {n: i for i, n in enumerate(topo)}
        ipdom = {END: END}
        def inter(x, y):
            while x != y:
                while index[x] < index[y]:
                    x = ipdom[x]
                while index[y] < index[x]:
                    y = ipdom[y]
            return x
        for n in reversed(topo):              # END first
            if n == END:
                continue
            d = None
            for m in g[n]:
                d = m if d is None else inter(d, m)
            ipdom[n] = d
        emitted = [0]
        def body(n, ind):
            if n > END:
                w(ind + f'again_{n - END - 1} = true;'); return
            for line in self.blocks[n]:
                if not isinstance(line, tuple):
                    w(ind + line)
            emitted[0] += 1
        def region(n, stop, ind):
            while n != stop:
                if n in cont and (n, 'open') not in region.active:
                    # loop: body once per iteration until no back edge is taken
                    e = g[cont[n]][0]
                    w(ind + f'bool again_{cont[n] - END - 1};'); w(ind + 'do\n' + ind + '{')
                    w(ind + f'    again_{cont[n] - END - 1} = false;')
                    region.active.add((n, 'open'))
                    region(n, e, ind + '    ')
                    region.active.discard((n, 'open'))
                    w(ind + f'}} while (again_{cont[n] - END - 1});')
                    n = e
                    continue
                body(n, ind)
                sc = g[n]
                if len(sc) == 1:
                    n = sc[0]; continue
                br = self.blocks[n][-1]
                m = ipdom[n]
                t_empty, f_empty = sc[0] == m, sc[1] == m
                if t_empty and f_empty:
                    pass
                elif f_empty:
                    w(ind + f'if ({br[1]})\n' + ind + '{'); region(sc[0], m, ind + '    '); w(ind + '}')
                elif t_empty:
                    w(ind + f'if (!({br[1]}))\n' + ind + '{'); region(sc[1], m, ind + '    '); w(ind + '}')
                else:
                    w(ind + f'if ({br[1]})\n' + ind + '{'); region(sc[0], m, ind + '    ')
                    w(ind + '}\n' + ind + 'else\n' + ind + '{'); region(sc[1], m, ind + '    '); w(ind + '}')
                n = m
        region.active = set()
        import sys as _s
        _s.setrecursionlimit(20000)
        region(0, END, '    ')
        w('}')
        sys.stderr.write(f'reachable blocks={len(nodes)} emitted block copies={emitted[0]} loops={len(headers)}\n')
        return '\n'.join(out) + '\n'


RUNTIME = r'''
layout(buffer_reference, scalar, buffer_reference_align = 2) readonly buffer R16 { uint16_t v[4]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) writeonly buffer W16 { uint16_t v[4]; };
float16_t H(uint u) { return uint16BitsToFloat16(uint16_t(u)); }
uint HU(float16_t h) { return uint(float16BitsToUint16(h)); }
f16vec2 H2(uint u) { return uint16BitsToFloat16(u16vec2(uint16_t(u & 0xffffu), uint16_t(u >> 16))); }
uint HU2_(u16vec2 u) { return uint(u.x) | (uint(u.y) << 16); }
uint H2U(f16vec2 h) { return HU2_(float16BitsToUint16(h)); }
int s16(uint u) { return (int(u) << 16) >> 16; }
float rcp_a(float x) { return 1.0 / x; }
// x + copysign(0.5, x) rounded toward zero and then truncated: round half away from zero, exactly
float round_half_away(float x) { float t = trunc(x); return abs(x - t) >= 0.5 ? t + sign(x) : t; }
uint PARAM32(uint o) { return param.w[o >> 2]; }
uint64_t PARAM64(uint o) { return uint64_t(param.w[o >> 2]) | (uint64_t(param.w[(o >> 2) + 1u]) << 32); }
uint PARAM16(uint o) { return (param.w[o >> 2] >> ((o & 2u) * 8u)) & 0xffffu; }
uint PARAM8(uint o) { return (param.w[o >> 2] >> ((o & 3u) * 8u)) & 0xffu; }
uint SH32(uint a) { return sh[a >> 2]; }
uint SH16(uint a) { return (sh[a >> 2] >> ((a & 2u) * 8u)) & 0xffffu; }
uint G16(uint64_t a) { return uint(R16(a).v[0]); }
uint G32(uint64_t a) { R16 r = R16(a); return uint(r.v[0]) | (uint(r.v[1]) << 16); }
uint bfi32(uint ins, uint base, uint pos, uint len)
{
    pos &= 0xffu; len &= 0xffu;
    if (pos >= 32u) return base;
    uint mask = len >= 32u ? (0xffffffffu << pos) : (((1u << len) - 1u) << pos);
    return (~mask & base) | (mask & (ins << pos));
}
// prmt.b32 (default mode): byte i of the result is byte (selector >> 4i) & 7 of the eight bytes b:a
uint prmt(uint a, uint b, uint sel)
{
    uint r = 0u;
    for (uint i = 0u; i < 4u; ++i) { const uint k = (sel >> (4u * i)) & 7u; r |= (((k < 4u ? a : b) >> (8u * (k & 3u))) & 0xffu) << (8u * i); }
    return r;
}
// shfl.sync.bfly within a 32-thread warp (ZLUDA's SHFL_SYNC_IMPL)
uint shfl_bfly(uint v, int delta, uint opts, out bool in_bounds)
{
    int section_mask = int((opts >> 8) & 31u), warp_end = int(opts & 31u);
    int self = int(gl_SubgroupInvocationID & 31u);
    int sub_end = (section_mask & self) | (~section_mask & warp_end);
    int idx = self ^ (delta & 31);
    bool oob = idx > sub_end;
    if (oob) idx = self;
    in_bounds = !oob;
    return subgroupShuffle(v, (gl_SubgroupInvocationID & 32u) | uint(idx));
}
void sust_p(writeonly image2D s, int x, int y, vec4 v)
{
    ivec2 n = imageSize(s);
    if (uint(x) < uint(n.x) && uint(y) < uint(n.y)) imageStore(s, ivec2(x, y), v);
}
// byte-addressed 32-bit store into a 16-bit single-channel surface: two texels
void sust_b(writeonly uimage2D s, int xb, int y, uint v)
{
    ivec2 n = imageSize(s);
    int x = xb >> 1;
    if (uint(y) < uint(n.y))
    {
        if (uint(x) < uint(n.x)) imageStore(s, ivec2(x, y), uvec4(v & 0xffffu));
        if (uint(x + 1) < uint(n.x)) imageStore(s, ivec2(x + 1, y), uvec4(v >> 16));
    }
}
'''

if __name__ == '__main__':
    filt = {}
    args = sys.argv[1:]
    while '--tex' in args:
        i = args.index('--tex'); k, v = args[i + 1].split(':'); filt[int(k)] = v; del args[i:i + 2]
    lut = '--exp-lut' in args
    if lut:
        args.remove('--exp-lut')
    fast = '--fast' in args
    if fast:
        args.remove('--fast')
    src = open(args[0]).read()
    t = T(src, args[1])
    t.exp_lut = lut
    if '--no-half-types' not in args:
        t.collect = True
        t.translate()
        parent = {}
        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for x, y in t.links:
            if t.width.get(x) == t.width.get(y):
                parent[find(x)] = find(y)
            else:
                t.ctx.setdefault(x, set()).add('u'); t.ctx.setdefault(y, set()).add('u')
        groups = {}
        for x in set(t.ctx) | set(parent):
            groups.setdefault(find(x), set()).update(t.ctx.get(x, set()))
        half = {x for x in set(t.ctx) | set(parent) if x in t.width and groups[find(x)] == {'h'}}
        sys.stderr.write(f'half-typed registers: {len(half)} of {len(t.width)}\n')
        lut2 = t.exp_lut
        t = T(src, args[1]); t.exp_lut = lut2; t.half = half
    t.fast = fast
    t.translate()
    sys.stdout.write(t.emit(filt))
    sys.stderr.write(f'blocks={len(t.blocks)} textures={sorted(t.textures)} surfaces={t.surfaces} shared={t.shared_bytes}\n')
