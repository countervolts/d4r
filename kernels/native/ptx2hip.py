#!/usr/bin/env python3
"""Translate one NVIDIA PTX kernel (DLSS texture kernels) into native HIP C++.

usage: ptx2hip.py MODULE.ptx KERNEL > KERNEL.hip

The output is a plain HIP kernel with the same name and parameter block. Every PTX instruction becomes
the C expression of ZLUDA's lowering of it (see ptx_rt.h), so the translation computes what ZLUDA's own
compile of the PTX computes, but it is compiled natively by clang and can then be optimised as C++.
Registers become locals, basic blocks become labels, branches become gotos.
"""
import re
import sys

TYPES = {'f': 'float', 'r': 'uint32_t', 'rd': 'uint64_t', 'rs': 'uint16_t', 'p': 'bool', 'fd': 'double',
         'SP': 'uint64_t', 'SPL': 'uint64_t'}
SREG = {'%tid.x': '__builtin_amdgcn_workitem_id_x()', '%tid.y': '__builtin_amdgcn_workitem_id_y()',
        '%tid.z': '__builtin_amdgcn_workitem_id_z()', '%ctaid.x': '__builtin_amdgcn_workgroup_id_x()',
        '%ctaid.y': '__builtin_amdgcn_workgroup_id_y()', '%ctaid.z': '__builtin_amdgcn_workgroup_id_z()',
        '%laneid': 'laneid()',
        '%ntid.x': '__builtin_amdgcn_workgroup_size_x()', '%ntid.y': '__builtin_amdgcn_workgroup_size_y()',
        '%nctaid.x': '(__builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x())',
        '%nctaid.y': '(__builtin_amdgcn_grid_size_y() / __builtin_amdgcn_workgroup_size_y())'}
REG_RE = re.compile(r'%([a-zA-Z]+)(\d+)$')


class Err(Exception):
    pass


def strip_comments(text):
    return re.sub(r'//[^\n]*', '', text)


def tokenize(body):
    """-> list of ('open'|'close'|'label'|'stmt', text)"""
    out, i, n = [], 0, len(body)
    while i < n:
        c = body[i]
        if c.isspace():
            i += 1
            continue
        if c == '{':
            out.append(('open', None)); i += 1; continue
        if c == '}':
            out.append(('close', None)); i += 1; continue
        m = re.match(r'(\$?[A-Za-z_][\w$]*):', body[i:])
        if m and not body[i:].startswith('%'):
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


class Translator:
    def __init__(self, ptx, kernel):
        self.kernel = kernel
        text = strip_comments(ptx)
        m = re.search(r'\.entry\s+' + re.escape(kernel) + r'\s*\((.*?)\)', text, re.S)
        if not m:
            raise Err('kernel not found')
        params = m.group(1)
        pm = re.search(r'\.param\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]', params)
        self.param_name, self.param_size, self.param_align = pm.group(2), int(pm.group(3)), int(pm.group(1))
        rest = text[m.end():]
        self.attrs = rest[:rest.index('{')]
        start = rest.index('{')
        depth, k = 0, start
        while True:
            if rest[k] == '{':
                depth += 1
            elif rest[k] == '}':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        self.body = rest[start + 1:k]
        self.globals_text = text[:m.start()]
        self.regs = {}        # c name -> type
        self.shared = []      # (name, size, align)
        self.locals = []
        self.scope_names = [{}]
        self.scope_id = 0
        self.lines = []

    # ------------------------------------------------------------ operands
    def reg(self, tok):
        tok = tok.strip()
        for scope in reversed(self.scope_names):
            if tok in scope:
                return scope[tok]
        m = REG_RE.match(tok)
        if not m:
            raise Err('not a register: ' + tok)
        kind, num = m.groups()
        name = f'{kind}{num}'
        if kind not in TYPES:
            raise Err('register kind ' + kind)
        self.regs[name] = TYPES[kind]
        return name

    def rtype(self, name):
        return self.regs[name]

    def val(self, tok, as_type):
        """operand as a C expression of type as_type ('f32','u32','s32','u64','s64','u16','s16','pred','f16x2')"""
        tok = tok.strip()
        if tok in SREG:
            return f'(uint32_t){SREG[tok]}'
        if tok.startswith('%') or any(tok in s for s in self.scope_names):
            name = self.reg(tok)
            t = self.rtype(name)
            if as_type == 'f32':
                return name if t == 'float' else f'asf({name})'
            if as_type in ('u32', 'b32'):
                return f'asu({name})' if t == 'float' else name
            if as_type == 's32':
                return f'(int32_t)asu({name})' if t == 'float' else f'(int32_t){name}'
            if as_type in ('u64', 'b64'):
                return name
            if as_type == 's64':
                return f'(int64_t){name}'
            if as_type in ('u16', 'b16'):
                return name
            if as_type == 's16':
                return f'(int16_t){name}'
            if as_type == 'f16':
                return f'ash({name})'
            if as_type == 'f16x2':
                return f'ash2({name})'
            if as_type == 'pred':
                return name
            raise Err('type ' + as_type)
        if tok in self.shared_names():
            return f'(uint32_t)(uintptr_t)(AS3 uint8_t*){tok}'
        if tok == '__local_depot0':
            # private (address space 5) address of the depot, zero-extended: ld/st.local use it as an AS5 pointer
            return '(uint64_t)(uint32_t)(uintptr_t)(AS5 uint8_t*)__local_depot0'
        # immediates
        if re.match(r'0[fF][0-9A-Fa-f]{8}$', tok):
            bits = int(tok[2:], 16)
            return f'asf(0x{bits:08X}u)' if as_type == 'f32' else f'0x{bits:08X}u'
        if re.match(r'-?\d+$', tok) or re.match(r'0x[0-9a-fA-F]+$', tok):
            v = int(tok, 0)
            if as_type == 'f32':
                raise Err('int immediate as float')
            if as_type in ('u64', 's64', 'b64'):
                return f'(uint64_t){v}ull' if v >= 0 else f'(uint64_t)({v}ll)'
            if as_type in ('u16', 'b16', 's16'):
                return f'(uint16_t){v & 0xffff}'
            if as_type == 's32':
                v = v & 0xffffffff
                v = v - (1 << 32) if v >= (1 << 31) else v
                return f'(int32_t){v}' if v != -(1 << 31) else '(int32_t)0x80000000u'
            return f'{v & 0xffffffff}u'
        raise Err('operand ' + tok)

    def shared_names(self):
        return [s[0] for s in self.shared]

    def assign(self, dst, expr, src_type):
        """dst = expr, expr of C type src_type ('float','u32','u64','u16','bool','f16x2','f16')"""
        name = self.reg(dst)
        t = self.rtype(name)
        if src_type == 'float':
            e = expr if t == 'float' else f'asu({expr})'
        elif src_type == 'u32':
            e = f'asf({expr})' if t == 'float' else f'(uint32_t)({expr})'
        elif src_type == 'u64':
            e = f'(uint64_t)({expr})'
        elif src_type == 'u16':
            e = f'(uint16_t)({expr})'
        elif src_type == 'f16':
            e = f'ashu({expr})'
        elif src_type == 'f16x2':
            e = f'ash2u({expr})'
        elif src_type == 'bool':
            e = f'({expr})'
        else:
            raise Err('src type')
        self.emit(f'{name} = {e};')

    def emit(self, s):
        self.lines.append('    ' + s)

    def addr(self, tok):
        """[reg+off] / [name+off] -> (base_expr, offset_int, space_hint)"""
        m = re.match(r'\[\s*([^\]+]+?)\s*(?:\+\s*(-?\d+))?\s*\]$', tok)
        if not m:
            raise Err('address ' + tok)
        base, off = m.group(1), int(m.group(2) or 0)
        if base == self.param_name:
            return 'KARG', off
        if base in self.shared_names():
            return f'(uint32_t)(uintptr_t)(AS3 uint8_t*){base}', off
        return self.val(base, 'u64' if not base.startswith('%r') or base.startswith('%rd') else 'u32'), off

    # ------------------------------------------------------------ instructions
    def translate(self):
        for kind, text in tokenize(self.body):
            if kind == 'open':
                self.scope_names.append({})
                continue
            if kind == 'close':
                self.scope_names.pop()
                continue
            if kind == 'label':
                self.lines.append(f'{text.replace("$", "")}:;')
                lab = text.replace('$', '')
                if lab in getattr(self, 'stamps', []):
                    self.lines.append(f'    D4R_STAMP({self.stamps.index(lab) + 1});')
                if getattr(self, 'bbcount', False):
                    self.bblabels.append(lab)
                    self.lines.append(f'    D4R_BBC({len(self.bblabels) - 1});')
                continue
            self.stmt(text)

    def stmt(self, s):
        if s.startswith('.reg'):
            m = re.match(r'\.reg\s+\.(\w+)\s+(.*)$', s)
            ty, names = m.groups()
            if '<' in names:
                return  # numbered register banks: declared on use
            self.scope_id += 1
            for nm in names.split(','):
                nm = nm.strip()
                cname = f'l{self.scope_id}_{nm.lstrip("%")}'
                ctype = {'f16': 'uint16_t', 'b16': 'uint16_t', 'f32': 'float', 'b32': 'uint32_t', 'pred': 'bool',
                         'f16x2': 'uint32_t', 'u32': 'uint32_t', 's32': 'uint32_t', 'b64': 'uint64_t', 'u64': 'uint64_t'}[ty]
                self.scope_names[-1][nm] = cname
                self.regs[cname] = ctype
            return
        if s.startswith('.shared'):
            m = re.match(r'\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]', s)
            self.shared.append((m.group(2), int(m.group(3)), int(m.group(1))))
            return
        if s.startswith('.local'):
            m = re.match(r'\.local\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]', s)
            self.locals.append((m.group(2), int(m.group(3)), int(m.group(1))))
            return
        if s.startswith('.pragma'):
            return
        pred = None
        m = re.match(r'@(!?)(%p\d+)\s+(.*)$', s)
        if m:
            pred = ('!' if m.group(1) else '') + self.reg(m.group(2))
            s = m.group(3)
        parts = s.split(None, 1)
        op = parts[0]
        args = split_operands(parts[1]) if len(parts) > 1 else []
        if pred is not None:
            if op.startswith('bra'):
                self.emit(f'if ({pred}) goto {args[0].replace("$", "")};')
                return
            self.emit(f'if ({pred}) {{')
            self.op(op, args)
            self.emit('}')
            return
        self.op(op, args)

    def op(self, op, a):
        o = op.split('.')
        base = o[0]
        mods = set(o[1:])
        E, A = self.emit, self.assign
        V = self.val

        def ftype():
            for t in ('f32', 'f16x2', 'f16', 'f64'):
                if t in mods:
                    return t
            return None

        if base == 'bra':
            E(f'goto {a[0].replace("$", "")};'); return
        if base == 'ret':
            if getattr(self, 'stamps', None):
                E(f'D4R_STAMP({len(self.stamps) + 1});')
            E('return;'); return
        if base == 'bar':
            E('bar_sync();'); return
        if base == 'mov':
            dst, src = a
            if dst.startswith('{'):
                names = [x.strip() for x in dst[1:-1].split(',')]
                srcv = V(src, 'u32')
                A(names[0], f'{srcv} & 0xffffu', 'u16')
                A(names[1], f'{srcv} >> 16', 'u16')
                return
            if src.startswith('{'):
                names = [x.strip() for x in src[1:-1].split(',')]
                if 'b64' in mods:
                    A(dst, f'(uint64_t){V(names[0], "u32")} | ((uint64_t){V(names[1], "u32")} << 32)', 'u64'); return
                A(dst, f'pack16({V(names[0], "u16")}, {V(names[1], "u16")})', 'u32'); return
            if src == self.param_name:
                A(dst, 'KARGU', 'u64'); return
            if 'f32' in mods:
                A(dst, V(src, 'f32'), 'float'); return
            if mods & {'u64', 'b64', 's64'}:
                A(dst, V(src, 'u64'), 'u64'); return
            if mods & {'u16', 'b16', 's16'}:
                A(dst, V(src, 'u16'), 'u16'); return
            A(dst, V(src, 'u32'), 'u32'); return

        ft = ftype()
        # ---------------------------------------------------- float arithmetic
        if ft == 'f32' and base in ('add', 'sub', 'mul'):
            if 'rz' in mods:
                raise Err('add.rz outside the roundf idiom')
            sym = {'add': '+', 'sub': '-', 'mul': '*'}[base]
            A(a[0], f'{V(a[1], "f32")} {sym} {V(a[2], "f32")}', 'float'); return
        if ft == 'f32' and base == 'fma':
            A(a[0], f'__builtin_fmaf({V(a[1], "f32")}, {V(a[2], "f32")}, {V(a[3], "f32")})', 'float'); return
        if ft == 'f32' and base in ('max', 'min'):
            if 'NaN' in mods:
                raise Err('NaN min/max')
            A(a[0], f'{"fmaxn" if base == "max" else "fminn"}({V(a[1], "f32")}, {V(a[2], "f32")})', 'float'); return
        if ft == 'f32' and base == 'neg':
            A(a[0], f'neg_ftz({V(a[1], "f32")})' if 'ftz' in mods else f'-{V(a[1], "f32")}', 'float'); return
        if ft == 'f32' and base == 'abs':
            A(a[0], f'abs_ftz({V(a[1], "f32")})' if 'ftz' in mods else f'__builtin_fabsf({V(a[1], "f32")})', 'float'); return
        if ft == 'f32' and base in ('rcp', 'ex2', 'lg2', 'rsqrt', 'sqrt'):
            if 'approx' not in mods:
                raise Err(op)
            fn = {'rcp': 'rcp_a', 'ex2': 'ex2_a', 'lg2': 'lg2_a', 'rsqrt': 'rsq_a', 'sqrt': '__builtin_amdgcn_sqrtf'}[base]
            A(a[0], f'{fn}({V(a[1], "f32")})', 'float'); return
        if ft == 'f32' and base == 'div':
            if 'approx' not in mods:
                raise Err(op)
            A(a[0], f'div_a({V(a[1], "f32")}, {V(a[2], "f32")})', 'float'); return
        # ---------------------------------------------------- f16 / f16x2
        if ft == 'f16x2' and base in ('add', 'sub', 'mul', 'fma'):
            x = [V(t, 'f16x2') for t in a[1:]]
            e = f'h2fma({x[0]}, {x[1]}, {x[2]})' if base == 'fma' else f'({x[0]} {"+-*"[["add", "sub", "mul"].index(base)]} {x[1]})'
            if 'sat' in mods:
                e = f'h2sat({e})'
            A(a[0], e, 'f16x2'); return
        if ft == 'f16x2' and base == 'abs':
            A(a[0], f'h2abs({V(a[1], "f16x2")})', 'f16x2'); return
        if ft == 'f16x2' and base == 'ex2':
            A(a[0], f'h2ex2({V(a[1], "f16x2")})', 'f16x2'); return
        if ft in ('f16', 'f16x2') and base in ('min', 'max'):
            if 'NaN' in mods:
                raise Err('NaN-propagating half min/max is not supported')
            fn = 'hminn' if base == 'min' else 'hmaxn'
            if ft == 'f16':
                A(a[0], f'{fn}({V(a[1], "f16")}, {V(a[2], "f16")})', 'f16'); return
            x, y = V(a[1], 'f16x2'), V(a[2], 'f16x2')
            A(a[0], f'(h2){{{fn}({x}.x, {y}.x), {fn}({x}.y, {y}.y)}}', 'f16x2'); return
        if ft == 'f16' and base in ('add', 'sub', 'mul', 'fma'):
            x = [V(t, 'f16') for t in a[1:]]
            e = f'hfma({x[0]}, {x[1]}, {x[2]})' if base == 'fma' else f'(h1)({x[0]} {"+-*"[["add", "sub", "mul"].index(base)]} {x[1]})'
            if 'sat' in mods:
                e = f'hsat({e})'
            A(a[0], e, 'f16'); return
        if op == 'tanh.approx.f16':
            A(a[0], f'__ocml_tanh_f16({V(a[1], "f16")})', 'f16'); return
        if base == 'set' and 'f16x2' in mods and 'u32' in mods:
            cmp = self.cmp_op(mods)
            x, y = V(a[1], 'f16x2'), V(a[2], 'f16x2')
            A(a[0], f'pack16(({cmp.format(f"{x}.x", f"{y}.x")}) ? 0xffffu : 0u, ({cmp.format(f"{x}.y", f"{y}.y")}) ? 0xffffu : 0u)', 'u32'); return
        # ---------------------------------------------------- compares / selects
        if base == 'setp':
            cmp = self.cmp_op(mods)
            if mods & {'f32'}:
                x, y = V(a[1], 'f32'), V(a[2], 'f32')
            elif 'f16' in mods:
                x, y = V(a[1], 'f16'), V(a[2], 'f16')
            elif mods & {'s32'}:
                x, y = V(a[1], 's32'), V(a[2], 's32')
            elif mods & {'u32', 'b32'}:
                x, y = V(a[1], 'u32'), V(a[2], 'u32')
            elif mods & {'s16'}:
                x, y = V(a[1], 's16'), V(a[2], 's16')
            elif mods & {'u16', 'b16'}:
                x, y = V(a[1], 'u16'), V(a[2], 'u16')
            elif mods & {'s64'}:
                x, y = V(a[1], 's64'), V(a[2], 's64')
            elif mods & {'u64'}:
                x, y = V(a[1], 'u64'), V(a[2], 'u64')
            else:
                raise Err(op)
            if len(a) != 3 or '|' in a[0]:
                raise Err('setp form ' + op)
            A(a[0], cmp.format(x, y), 'bool'); return
        if base == 'selp':
            t = 'f32' if 'f32' in mods else 'u16' if mods & {'u16', 'b16', 's16'} else 'u64' if mods & {'u64', 'b64', 's64'} else 'u32'
            ct = {'f32': 'float', 'u16': 'u16', 'u64': 'u64', 'u32': 'u32'}[t]
            A(a[0], f'{V(a[3], "pred")} ? {V(a[1], t)} : {V(a[2], t)}', ct); return
        # ---------------------------------------------------- predicates / integer
        if 'pred' in mods and base in ('and', 'or', 'xor', 'not'):
            if base == 'not':
                A(a[0], f'!{V(a[1], "pred")}', 'bool'); return
            sym = {'and': '&&', 'or': '||', 'xor': '!='}[base]
            A(a[0], f'{V(a[1], "pred")} {sym} {V(a[2], "pred")}', 'bool'); return
        if base in ('and', 'or', 'xor', 'not', 'shl', 'shr', 'add', 'sub', 'mul', 'mad', 'min', 'max', 'div', 'rem', 'neg', 'abs'):
            if ft is not None:
                raise Err('unhandled floating operation: ' + op)
            width = 64 if mods & {'u64', 's64', 'b64'} else 16 if mods & {'u16', 's16', 'b16'} else 32
            signed = bool(mods & {'s32', 's64', 's16'})
            ut = {64: 'u64', 32: 'u32', 16: 'u16'}[width]
            st = {64: 's64', 32: 's32', 16: 's16'}[width]
            ct = {64: 'u64', 32: 'u32', 16: 'u16'}[width]
            if base == 'mul' and 'hi' in mods and width == 32:
                wide = 'int64_t' if signed else 'uint64_t'
                typ = 's32' if signed else 'u32'
                A(a[0], f'(({wide}){V(a[1], typ)} * ({wide}){V(a[2], typ)}) >> 32', 'u32'); return
            if base == 'mul' and 'wide' in mods:
                if mods & {'s32'}:
                    A(a[0], f'(int64_t){V(a[1], "s32")} * (int64_t){V(a[2], "s32")}', 'u64'); return
                if mods & {'u32'}:
                    A(a[0], f'(uint64_t){V(a[1], "u32")} * (uint64_t){V(a[2], "u32")}', 'u64'); return
                if mods & {'u16'}:
                    A(a[0], f'(uint32_t){V(a[1], "u16")} * (uint32_t){V(a[2], "u16")}', 'u32'); return
                raise Err(op)
            if base == 'not':
                A(a[0], f'~{V(a[1], ut)}', ct); return
            if base in ('neg', 'abs'):
                if base == 'abs' and signed:
                    A(a[0], f'{V(a[1], st)} < 0 ? (uint{width}_t)(0 - {V(a[1], ut)}) : {V(a[1], ut)}', ct); return
                if base == 'neg':
                    A(a[0], f'(uint{width}_t)(0 - {V(a[1], ut)})', ct); return
                raise Err(op)
            x = V(a[1], st if signed else ut)
            if base == 'shl':
                A(a[0], f'{V(a[1], ut)} << ({V(a[2], "u32")} & {width - 1})', ct); return
            if base == 'shr':
                if signed:
                    A(a[0], f'{V(a[1], st)} >> ({V(a[2], "u32")} & {width - 1})', ct); return
                A(a[0], f'{V(a[1], ut)} >> ({V(a[2], "u32")} & {width - 1})', ct); return
            y = V(a[2], st if signed else ut)
            if base in ('and', 'or', 'xor'):
                sym = {'and': '&', 'or': '|', 'xor': '^'}[base]
                A(a[0], f'{V(a[1], ut)} {sym} {V(a[2], ut)}', ct); return
            if base in ('add', 'sub'):
                A(a[0], f'{V(a[1], ut)} {"+" if base == "add" else "-"} {V(a[2], ut)}', ct); return
            if base == 'mul':
                if 'lo' not in mods:
                    raise Err(op)
                A(a[0], f'{V(a[1], ut)} * {V(a[2], ut)}', ct); return
            if base == 'mad':
                if 'lo' not in mods:
                    raise Err(op)
                A(a[0], f'{V(a[1], ut)} * {V(a[2], ut)} + {V(a[3], ut)}', ct); return
            if base in ('min', 'max'):
                fn = 'min' if base == 'min' else 'max'
                A(a[0], f'__builtin_elementwise_{fn}({x}, {y})', ct); return
            if base == 'div':
                A(a[0], f'{x} / {y}', ct); return
            if base == 'rem':
                A(a[0], f'{x} % {y}', ct); return
        if base == 'bfi':
            A(a[0], f'bfi32({V(a[1], "u32")}, {V(a[2], "u32")}, {V(a[3], "u32")}, {V(a[4], "u32")})', 'u32'); return
        if op == 'prmt.b32':
            A(a[0], f'prmt32({V(a[1], "u32")}, {V(a[2], "u32")}, {V(a[3], "u32")})', 'u32'); return
        # ---------------------------------------------------- conversions
        if base == 'cvt':
            return self.cvt(op, o[1:], a)
        if base == 'cvta':
            A(a[0], V(a[1], 'u64'), 'u64'); return
        # ---------------------------------------------------- memory
        if base == 'ld':
            return self.ld(o[1:], a)
        if base == 'st':
            return self.st(o[1:], a)
        if base == 'tex':
            dst = [x.strip() for x in a[0][1:-1].split(',')]
            m = re.match(r'\[\s*(%rd\d+)\s*,\s*\{([^}]*)\}\s*\]', a[1])
            h = V(m.group(1), 'u64')
            c = [x.strip() for x in m.group(2).split(',')]
            if 'base' in mods and mods >= {'2d', 'v4', 's32'} and mods & {'f32', 'f16'}:
                E(f'{{ ::f4 t_ = tex_fetch({h}, (int32_t){V(c[0], "u32")}, (int32_t){V(c[1], "u32")});')
            elif mods >= {'level', '2d', 'v4', 'f32'} and op.endswith('f32.f32'):
                E(f'{{ ::f4 t_ = tex_level({h}, {V(c[0], "f32")}, {V(c[1], "f32")}, {V(a[2], "f32")});')
            else:
                raise Err(op)
            for i, d in enumerate(dst):
                A(d, f'(h1)t_[{i}]', 'f16') if 'f16' in mods else A(d, f't_[{i}]', 'float')
            E('}')
            return
        if base == 'tld4':
            if not (mods >= {'r', '2d', 'v4', 'f32'}):
                raise Err(op)
            dst = [x.strip() for x in a[0][1:-1].split(',')]
            m = re.match(r'\[\s*(%rd\d+)\s*,\s*\{([^}]*)\}\s*\]', a[1])
            c = [x.strip() for x in m.group(2).split(',')]
            E(f'{{ ::f4 t_ = tex_gather_r({V(m.group(1), "u64")}, {V(c[0], "f32")}, {V(c[1], "f32")});')
            for i, d in enumerate(dst):
                A(d, f't_[{i}]', 'float')
            E('}')
            return
        if base == 'sust':
            m = re.match(r'\[\s*(%rd\d+)\s*,\s*\{([^}]*)\}\s*\]', a[0])
            h = V(m.group(1), 'u64')
            c = [V(x.strip(), 'u32') for x in m.group(2).split(',')]
            v = [V(x.strip(), 'u32') for x in a[1][1:-1].split(',')]
            if op == 'sust.p.2d.v4.b32.zero':
                E(f'd4r_sust_p_v4b32({h}, (int32_t){c[0]}, (int32_t){c[1]}, {v[0]}, {v[1]}, {v[2]}, {v[3]});'); return
            if op == 'sust.b.2d.b32.zero':
                E(f'd4r_sust_b32({h}, (int32_t){c[0]}, (int32_t){c[1]}, {v[0]});'); return
            if op == 'sust.b.2d.v2.b16.zero':
                v = [V(x.strip(), 'u16') for x in a[1][1:-1].split(',')]
                E(f'd4r_sust_v2b16({h}, (int32_t){c[0]}, (int32_t){c[1]}, {v[0]}, {v[1]});'); return
            raise Err(op)
        if base == 'shfl':
            if op == 'shfl.sync.idx.b32':
                d = a[0].split('|')
                pred = f'&{self.reg(d[1])}' if len(d) > 1 else 'nullptr'
                A(d[0], f'shfl_idx({V(a[1], "u32")}, {V(a[2], "u32")}, {V(a[3], "u32")}, {pred})', 'u32'); return
            if op != 'shfl.sync.bfly.b32':
                raise Err(op)
            d = a[0].split('|')
            p = f'&{self.reg(d[1])}' if len(d) > 1 else 'nullptr'
            A(d[0], f'shfl_bfly({V(a[1], "u32")}, (int32_t){V(a[2], "u32")}, {V(a[3], "u32")}, {p})', 'u32'); return
        if op == 'vote.sync.ballot.b32':
            A(a[0], f'(uint32_t)(__builtin_amdgcn_ballot_w64({V(a[1], "pred")}) >> (wave_lane() & 32u))', 'u32'); return
        raise Err('unhandled ' + op)

    def cmp_op(self, mods):
        table = {'lt': '{} < {}', 'gt': '{} > {}', 'le': '{} <= {}', 'ge': '{} >= {}', 'eq': '{} == {}',
                 'ne': '{} != {}',
                 # unordered: true if either is NaN
                 'ltu': '!({} >= {})', 'gtu': '!({} <= {})', 'leu': '!({} > {})', 'geu': '!({} < {})',
                 'equ': '!({} != {})', 'neu': '!({} == {})'}
        for k in table:
            if k in mods:
                if k in ('ne', 'equ') and mods & {'f32', 'f16', 'f16x2', 'f64'}:
                    # PTX setp.ne.f32 is ordered (false on NaN) and equ is unordered (true on NaN), while C's
                    # != is unordered: spell both out. Integer ne is plain.
                    return '({0} < {1} || {0} > {1})' if k == 'ne' else '!({0} < {1} || {0} > {1})'
                return table[k]
        raise Err('compare ' + str(mods))

    def cvt(self, op, o, a):
        A, V = self.assign, self.val
        mods = set(o)
        if op == 'cvt.rn.satfinite.e4m3x2.f16x2':
            A(a[0], f'e4m3x2({V(a[1], "u32")})', 'u16'); return
        if op == 'cvt.rn.f16x2.e4m3x2':
            A(a[0], f'd4r_decode_e4m3x2({V(a[1], "u16")})', 'u32'); return
        types = [t for t in o if re.match(r'^(f|s|u|b)\d+(x2)?$', t)]
        dt, stp = types[0], types[1]
        if dt == 'f32' and stp == 'f32':
            x = V(a[1], 'f32')
            if 'sat' in mods and 'ftz' in mods and not (mods & {'rzi', 'rmi', 'rni', 'rpi'}):
                A(a[0], f'sat_ftz({x})', 'float'); return
            if mods & {'sat'}:
                raise Err(op)
            if 'rzi' in mods:
                A(a[0], f'__builtin_truncf({x})', 'float'); return
            if 'rmi' in mods:
                A(a[0], f'__builtin_floorf({x})', 'float'); return
            if 'rni' in mods:
                A(a[0], f'__builtin_rintf({x})', 'float'); return
            if 'rpi' in mods:
                A(a[0], f'__builtin_ceilf({x})', 'float'); return
            if 'ftz' in mods:
                A(a[0], f'canon({x})', 'float'); return
            A(a[0], x, 'float'); return
        if dt == 's32' and stp == 'f32':
            x = V(a[1], 'f32')
            if 'rzi' in mods:
                A(a[0], f'(uint32_t)f2i_rz({x})', 'u32'); return
            if 'rmi' in mods:
                A(a[0], f'(uint32_t)f2i_rz(__builtin_floorf({x}))', 'u32'); return
            if 'rni' in mods:
                A(a[0], f'(uint32_t)f2i_rz(__builtin_rintf({x}))', 'u32'); return
            raise Err(op)
        if dt == 'u32' and stp == 'f32' and 'rzi' in mods:
            A(a[0], f'f2u_rz({V(a[1], "f32")})', 'u32'); return
        if dt == 'f32' and stp == 's32':
            A(a[0], f'(float)(int32_t){V(a[1], "u32")}', 'float'); return
        if dt == 'f32' and stp == 'u32':
            A(a[0], f'(float){V(a[1], "u32")}', 'float'); return
        if dt == 'f32' and stp == 'f16':
            A(a[0], f'(float){V(a[1], "f16")}', 'float'); return
        if dt == 'f16' and stp == 'f32':
            if 'rn' not in mods:
                raise Err(op)
            A(a[0], f'f2h_rn({V(a[1], "f32")})', 'f16'); return
        if dt == 'f16x2' and stp == 'f32':
            # d = {cvt(a) in the high half, cvt(b) in the low half}
            A(a[0], f'pack16(ashu(f2h_rn({V(a[2], "f32")})), ashu(f2h_rn({V(a[1], "f32")})))', 'u32'); return
        if dt == 'f16' and stp == 'f16':
            x = V(a[1], 'f16')
            if 'rni' in mods:
                A(a[0], f'h_roundeven({x})', 'f16'); return
            if 'rmi' in mods:
                A(a[0], f'h_floor({x})', 'f16'); return
            raise Err(op)
        if dt == 's32' and stp == 'f16':
            x = V(a[1], 'f16')
            if 'rni' in mods:
                A(a[0], f'(uint32_t)h2i_sat(h_roundeven({x}))', 'u32'); return
            if 'rmi' in mods:
                A(a[0], f'(uint32_t)h2i_sat(h_floor({x}))', 'u32'); return
            raise Err(op)
        if dt == 'f16' and stp == 's32':
            A(a[0], f'(h1)(int32_t){V(a[1], "u32")}', 'f16'); return
        if dt == 's64' and stp == 's32':
            A(a[0], f'(uint64_t)(int64_t)(int32_t){V(a[1], "u32")}', 'u64'); return
        if dt == 'u64' and stp == 'u32':
            A(a[0], f'(uint64_t){V(a[1], "u32")}', 'u64'); return
        if dt in ('u16', 's16') and stp in ('u32', 's32'):
            A(a[0], f'(uint16_t){V(a[1], "u32")}', 'u16'); return
        if dt in ('u32', 's32') and stp in ('u16',):
            A(a[0], f'(uint32_t){V(a[1], "u16")}', 'u32'); return
        if dt in ('u32', 's32') and stp in ('s16',):
            A(a[0], f'(uint32_t)(int32_t)(int16_t){V(a[1], "u16")}', 'u32'); return
        if dt == 'u32' and stp == 'u64':
            A(a[0], f'(uint32_t){V(a[1], "u64")}', 'u32'); return
        raise Err('cvt ' + op)

    def mem_type(self, o):
        for t in ('u8', 's8', 'u16', 's16', 'b16', 'u32', 's32', 'b32', 'f32', 'u64', 's64', 'b64', 'f16'):
            if t in o:
                return t
        raise Err('mem type ' + str(o))

    def ctype_of(self, t):
        return {'u8': 'uint8_t', 's8': 'int8_t', 'u16': 'uint16_t', 's16': 'int16_t', 'b16': 'uint16_t',
                'u32': 'uint32_t', 's32': 'int32_t', 'b32': 'uint32_t', 'f32': 'float', 'u64': 'uint64_t',
                's64': 'int64_t', 'b64': 'uint64_t'}[t]

    def ptr(self, space, base, off, ct):
        if base == 'KARG':
            return f'*(const AS4 {ct}*)(KARGP + {off})'
        if space == 'param':
            return f'*(const AS4 {ct}*)(uintptr_t)({base} + {off})'
        if space == 'shared':
            return f'*(AS3 {ct}*)(uintptr_t)({base} + {off}u)'
        if space == 'local':
            return f'*(AS5 {ct}*)(uintptr_t)(uint32_t)({base} + {off})'
        if space == 'global' and getattr(self, 'as1', False):
            # --as1: ld/st.global through address-space-1 pointers (global_load / global_store instead of flat)
            return f'*(AS1 {ct}*)(uintptr_t)({base} + {off})'
        if space in ('global', 'local'):
            return f'*({ct}*)(uintptr_t)({base} + {off})'
        raise Err('space ' + space)

    def ld(self, o, a):
        space = next(s for s in ('param', 'global', 'shared', 'local') if s in o)
        t = self.mem_type(o)
        ct = self.ctype_of(t)
        base, off = self.addr(a[1])
        dsts = [x.strip() for x in a[0][1:-1].split(',')] if a[0].startswith('{') else [a[0]]
        width = {'uint8_t': 1, 'int8_t': 1, 'uint16_t': 2, 'int16_t': 2, 'uint32_t': 4, 'int32_t': 4, 'float': 4,
                 'uint64_t': 8, 'int64_t': 8}[ct]
        if len(dsts) in (2, 4) and width == 4 and space in ('global', 'shared') and base != 'KARG':
            # ld.v2/v4 (naturally aligned in PTX) as one vector load instead of per-element loads
            vt = 'u4' if len(dsts) == 4 else 'u2v'
            asq = 'AS3 ' if space == 'shared' else ('AS1 ' if getattr(self, 'as1', False) else '')
            ptr = f'(uintptr_t)({base} + {off}u)' if space == 'shared' else f'(uintptr_t)({base} + {off})'
            self.emit(f'{{ const {vt} v_ = *({asq}const {vt}*){ptr};')
            for i, d in enumerate(dsts):
                self.assign(d, f'v_[{i}]', 'u32')
            self.emit('}')
            return
        for i, d in enumerate(dsts):
            e = self.ptr(space, base, off + i * width, ct)
            if ct == 'float':
                self.assign(d, e, 'float')
            elif width == 8:
                self.assign(d, e, 'u64')
            else:
                name = self.reg(d)
                if self.rtype(name) == 'uint16_t':
                    self.assign(d, e, 'u16')
                else:
                    self.assign(d, f'(uint32_t){e}' if ct != 'int8_t' and ct != 'int16_t' else f'(uint32_t)(int32_t){e}', 'u32')

    def st(self, o, a):
        space = next(s for s in ('param', 'global', 'shared', 'local') if s in o)
        t = self.mem_type(o)
        ct = self.ctype_of(t)
        base, off = self.addr(a[0])
        srcs = [x.strip() for x in a[1][1:-1].split(',')] if a[1].startswith('{') else [a[1]]
        width = {'uint8_t': 1, 'uint16_t': 2, 'uint32_t': 4, 'int32_t': 4, 'float': 4, 'uint64_t': 8}[ct]
        for i, s in enumerate(srcs):
            v = self.val(s, 'f32' if ct == 'float' else 'u64' if width == 8 else 'u16' if width == 2 else 'u32')
            self.emit(f'{self.ptr(space, base, off + i * width, ct)} = ({ct})({v});')

    # ------------------------------------------------------------ roundf idiom
    def rewrite_rz(self):
        """add.rz.ftz.f32 t, x, h; cvt.rzi.f32.f32 y, t  (h = copysign(0.5, x)) -> y = roundf(x)"""
        text = self.body
        pat = re.compile(r'add\.rz\.ftz\.f32\s+(%f\d+),\s*(%f\d+),\s*(%f\d+);\s*(?://[^\n]*\s*)*cvt\.rzi\.f32\.f32\s+(%f\d+),\s*(%f\d+);')
        count = 0

        def repl(m):
            nonlocal count
            t, x, h, y, t2 = m.groups()
            if t != t2:
                raise Err('rz idiom mismatch')
            uses = len(re.findall(re.escape(t) + r'\b', text))
            if uses != 2:
                raise Err(f'rz temp {t} used {uses} times')
            # check h = copysign(0.5, x): mov.b32 rA, x; and.b32 rB, rA, -2147483648; or.b32 rC, rB, 1056964608; mov.b32 h, rC
            count += 1
            return f'd4r.roundf {y}, {x};'
        self.body = pat.sub(repl, text)
        return count


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    if len(args) != 2:
        sys.exit(__doc__)
    ptx, kernel = open(args[0]).read(), args[1]
    mpart = None
    if '--m-tail' in flags:
        lines = ptx.replace('\r\n', '\n').split('\n')
        if kernel == 'rrlite_enc0_4x4_mvhi_hdr_folded':
            if lines[2547].strip() != 'bar.warp.sync -1;':
                raise Err('enc0 cut changed')
            lines = lines[:2548] + ['d4r.enc0 %r259, %rd106, %rd78, %r1066, %r1067, %r1230, %r1233;', 'ret;', '}']
            mpart = 'enc0_tail'
        elif kernel == 'rrlite_dec0_4x4_folded':
            if lines[138].strip() != 'mov.u32 %r17, %laneid;' or lines[586].strip() != 'st.shared.v2.u16 [%r570+1168], {%rs63, %rs64};':
                raise Err('dec0 cut changed')
            lines[138:587] = ['mov.u32 %r560, _ZZ22rrlite_dec0_4x4_foldedN6RRLite18Decoder0ParametersEE4smem;',
                              'd4r.dec0 %r560, %rd22, %rd24, %r475, %r476, %r3, %r4;']
            mpart = 'dec0_head'
        else:
            raise Err('--m-tail requires folded M enc0 or dec0')
        ptx = '\n'.join(lines)
    tr = Translator(ptx, kernel)
    n_rz = tr.rewrite_rz()
    if '--ifconv' in flags:
        import ifconv, ctile as _ct
        tr.body, n_if = ifconv.apply(tr.body, tokenize, _ct.parse, split_operands)
        sys.stderr.write(f'ifconv: {n_if} diamonds converted\n')
    stamps = [f.split('=', 1)[1].split(',') for f in flags if f.startswith('--stamps=')]
    tr.stamps = stamps[0] if stamps else []
    tr.bbcount = '--bbcount' in flags
    tr.as1 = '--as1' in flags
    tr.bblabels = []
    ct = None
    if '--ctile' in flags:
        import ctile
        ct = ctile.emit(tr, tokenize, Translator)
    if '--sink' in flags:
        import ifconv, ctile as _ct
        tr.body, n_sink = ifconv.sink_stores(tr.body, tokenize, _ct.parse)
        sys.stderr.write(f'sink: {n_sink} stores moved\n')
    orig_op = tr.op

    def op_with_round(op, a):
        if op in ('d4r.hmin2', 'd4r.hmax2'):
            fn = 'm_min2' if op == 'd4r.hmin2' else 'm_max2'
            left = ', '.join(tr.val(v, 'f16') for v in a[2:4])
            right = ', '.join(tr.val(v, 'f16') for v in a[4:6])
            tr.emit(f'{{ uint32_t packed_ = ash2u({fn}((h2){{{left}}}, (h2){{{right}}}));')
            tr.assign(a[0], 'packed_ & 0xffffu', 'u16')
            tr.assign(a[1], 'packed_ >> 16', 'u16')
            tr.emit('}')
            return
        if op in ('d4r.enc0', 'd4r.dec0'):
            vals = [tr.val(v, t) for v, t in zip(a, ('u32', 'u64', 'u64', 'u32', 'u32', 's32', 's32'))]
            tr.emit(f'd4r_{"enc0_tail" if op == "d4r.enc0" else "dec0_head"}({", ".join(vals)});')
            return
        if op == 'd4r.roundf':
            tr.assign(a[0], f'__builtin_roundf({tr.val(a[1], "f32")})', 'float')
            return
        orig_op(op, a)
    tr.op = op_with_round
    tr.translate()
    out = []
    out.append(f'// Generated by kernels/native/ptx2hip.py from NVIDIA PTX ({kernel}); {n_rz} roundf idioms.')
    out.append('#include "native_rt.h"')
    if kernel.startswith('rrlite_'):
        out.append('#include "m_ptx_rt.h"')
    if 'cvt.rn.f16x2.e4m3x2' in ptx:
        out.append('RT uint32_t d4r_decode_e4m3x2(uint16_t v) { h2 h; for (int i = 0; i < 2; ++i) { '
                   'uint32_t b = (v >> (8*i)) & 255u; '
                   'h[i] = (b & 127u) == 127u ? ash((uint16_t)(0x7fffu | ((b & 128u) << 8))) : '
                   '(h1)(ash((uint16_t)(((b & 127u) << 7) | ((b & 128u) << 8))) * (h1)256.0f); } return ash2u(h); }')
    if mpart:
        out.append(f'#include "../tex/{mpart}.hip"')
    if ct is None:
        for name, size, align in tr.shared:
            out.append(f'__attribute__((shared)) __attribute__((aligned({max(align, 16)}))) static uint8_t {name}[{size}];')
    else:
        # one LDS block: the kernel's own arrays (path B) and the color tile (path A) are never live together
        off = 0
        for name, size, align in tr.shared:
            off = (off + 15) & ~15
            out.append(f'#define {name} (d4r_lds + {off})')
            off += size
        prelude, fill, info = ct
        out.insert(2, f'__attribute__((shared)) __attribute__((aligned(16))) static uint8_t d4r_lds[{max(off, info["lds_bytes"])}];')
        out.insert(2, f'// --ctile: {info["taps"]} color taps through an LDS tile, {len(info["groups"])} conversion shapes')
        out.extend(prelude)
        out.extend(fill)
    out.append(f'struct __attribute__((aligned({tr.param_align}))) {kernel}_params {{ uint8_t b[{tr.param_size}]; }};')
    maxntid = re.search(r'\.maxntid\s+(\d+)', tr.attrs)
    lb = f'__attribute__((amdgpu_flat_work_group_size(1, {maxntid.group(1)})))' if maxntid else ''
    bounds = [f.split('=', 1)[1] for f in flags if f.startswith('--max-block=')]
    if bounds:
        n = int(bounds[0])
        if not 1 <= n <= 1024:
            raise Err('invalid --max-block')
        lb = f'__attribute__((amdgpu_flat_work_group_size(1, {n})))'
    out.append(f'extern "C" __attribute__((global)) {lb} void {kernel}({kernel}_params)')
    out.append('{')
    out.append('    const AS4 uint8_t* KARGP = (const AS4 uint8_t*)__builtin_amdgcn_kernarg_segment_ptr();')
    out.append('    const uint64_t KARGU = (uint64_t)(uintptr_t)KARGP;')
    for name, size, align in tr.locals:
        out.append(f'    __attribute__((aligned({align}))) uint8_t {name}[{size}];')
    if ct is not None:
        out.append('    int32_t ct_x0 = 0, ct_y0 = 0, ct_w = 0, ct_h = 0;')
    if tr.bbcount:
        # diagnostic build: d4r_bbc[k] = number of waves that entered basic block k (label list in a comment)
        out.insert(2, '__attribute__((device)) unsigned long long d4r_bbc[512];\n'
                      '#define D4R_BBC(k) do { if (wave_lane() == (uint32_t)__builtin_ctzll(__builtin_amdgcn_read_exec())) '
                      '__hip_atomic_fetch_add(&d4r_bbc[k], 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); } while (0)\n'
                      '// d4r_bbc labels: ' + ' '.join(tr.bblabels))
    if tr.stamps:
        # diagnostic build: per-region wave cycles (20-bit SHADER_CYCLES deltas) summed into d4r_dbg[k],
        # wave counts into d4r_dbg[32 + k]; region k ends at stamp label k (k = len + 1: kernel return)
        out.insert(2, '__attribute__((device)) unsigned long long d4r_dbg[64];\n'
                      '#define D4R_STAMP(k) do { uint32_t n_ = (uint32_t)__builtin_readcyclecounter(); '
                      'if (wave_lane() == (uint32_t)__builtin_ctzll(__builtin_amdgcn_read_exec())) { '
                      '__hip_atomic_fetch_add(&d4r_dbg[k], (unsigned long long)((n_ - st_last) & 0xfffffu), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); '
                      '__hip_atomic_fetch_add(&d4r_dbg[32 + (k)], 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); } '
                      'st_last = n_; } while (0)')
        out.append('    uint32_t st_last = (uint32_t)__builtin_readcyclecounter();')
    by_type = {}
    for name, t in sorted(tr.regs.items()):
        by_type.setdefault(t, []).append(name)
    for t, names in by_type.items():
        for i in range(0, len(names), 16):
            out.append(f'    {t} {", ".join(names[i:i + 16])};')
    out.extend(tr.lines)
    out.append('}')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
