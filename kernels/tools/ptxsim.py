#!/usr/bin/env python3
"""Minimal vectorised PTX interpreter for DLSS rrlite kernels.

Runs one thread block of a fully unrolled kernel on the CPU (all threads of the
block as numpy lanes), against memory restored from a bridge replay dump
(D4R_CUDA_REPLAY_DUMP_DIR). It is a reference and an instrument: every
statement can be observed (hooks), which is how the logical computation
(GEMMs, weight layouts, normalisations) of a kernel is recovered.

Only the instruction forms the rrlite kernels use are implemented; anything
else raises NotImplementedError naming the statement.
"""
import re
import struct
import sys

import numpy as np

F16, F32, F64 = np.float16, np.float32, np.float64
U64 = np.uint64
PARAM_BASE = 1 << 60  # virtual address of the kernel parameter block
MASK = {8: 0xff, 16: 0xffff, 32: 0xffffffff, 64: 0xffffffffffffffff}


def _e4m3_decode_table():
    t = np.zeros(256, F64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xf, c & 7
        if (c & 0x7f) == 0x7f:
            t[c] = np.nan
        elif e == 0:
            t[c] = s * m * 2.0 ** -9
        else:
            t[c] = s * (1 + m / 8) * 2.0 ** (e - 7)
    return t


E4M3 = _e4m3_decode_table()


def _f16_to_e4m3_table():
    """Exact RNE + satfinite f16 -> e4m3 for all 65536 f16 codes (ZLUDA's helper)."""
    h = np.arange(65536, dtype=np.uint32)
    mag = h & 0x7fff
    normal = (mag - 0x2000 + 0x3f + ((mag >> 7) & 1)) >> 7
    exp = mag >> 10
    sig = (mag & 0x3ff) | np.where(exp != 0, 0x400, 0)
    cl = np.clip(exp, 1, 8)
    sh = 16 - cl
    sub = (sig + (np.left_shift(1, sh - 1)) - 1 + ((sig >> sh) & 1)) >> sh
    code = np.where(mag >= 0x2400, normal, sub)
    code = np.where(mag > 0x5f00, 0x7e, code)
    code = np.where(mag > 0x7c00, 0x7f, code)
    return (((h >> 8) & 0x80) | code).astype(np.uint32)


F16_TO_E4M3 = _f16_to_e4m3_table()


def bits_to_f16(x):
    return (x & 0xffff).astype(np.uint16).view(F16)


def f16_to_bits(v):
    return np.asarray(v, F16).view(np.uint16).astype(U64)


def bits_to_f32(x):
    return (x & 0xffffffff).astype(np.uint32).view(F32)


def f32_to_bits(v):
    return np.asarray(v, F32).view(np.uint32).astype(U64)


class Stmt:
    __slots__ = ('labels', 'pred', 'neg', 'op', 'args', 'text', 'index')


def split_args(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '{[':
            depth += 1
        elif ch in '}]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_kernel(path, name):
    with open(path) as file:
        src = file.read()
    i = src.index('.entry ' + name + '(')
    j = src.find('\n.visible', i + 10)
    j = len(src) if j < 0 else j
    body = src[i:j]
    body = body[body.index(')') + 1:]
    # module-scope .shared arrays (declared before the entry) are visible to it too
    module_shared = {m.group(1): int(m.group(2)) for m in re.finditer(r'^\.shared[^\n;]*?(\w+)\[(\d+)\]', src[:i], re.M)}
    body = re.sub(r'//[^\n]*', '', body)
    body = re.sub(r'\.(maxntid|reqntid|minnctapersm|maxnreg)[^\n{]*', '', body)
    stmts, types, shared = [], {}, {}
    pending_labels = []
    for raw in re.sub(r'\s*\n\s*', ' ', body).split(';'):
        s = raw.strip()
        while True:
            s = s.strip()
            if s.startswith('{') and not s.startswith('{%'):
                s = s[1:]
                continue
            if s.startswith('}'):
                s = s[1:]
                continue
            m = re.match(r'(\$[\w]+):\s*', s)
            if m:
                pending_labels.append(m.group(1))
                s = s[m.end():]
                continue
            break
        if not s:
            continue
        if s.startswith('.reg'):
            m = re.match(r'\.reg\s*\.(\w+)\s+(.*)', s)
            ty = m.group(1)
            for decl in m.group(2).split(','):
                decl = decl.strip()
                m2 = re.match(r'(%[\w]+)<(\d+)>', decl)
                if m2:
                    types[('prefix', m2.group(1))] = ty
                else:
                    types[decl] = ty
            continue
        if s.startswith('.shared'):
            m = re.search(r'(\w+)\[(\d+)\]', s)
            shared[m.group(1)] = int(m.group(2))
            continue
        if s.startswith('.'):
            continue
        st = Stmt()
        st.labels, pending_labels = pending_labels, []
        st.pred, st.neg = None, False
        m = re.match(r'@(!?)(%\w+)\s+', s)
        if m:
            st.neg, st.pred = m.group(1) == '!', m.group(2)
            s = s[m.end():]
        parts = s.split(None, 1)
        st.op = parts[0]
        st.args = split_args(parts[1]) if len(parts) > 1 else []
        st.text = s
        st.index = len(stmts)
        stmts.append(st)
    if pending_labels:
        st = Stmt()
        st.labels, st.pred, st.neg, st.op, st.args, st.text, st.index = pending_labels, None, False, 'nop', [], 'nop', len(stmts)
        stmts.append(st)
    for k, v in module_shared.items():
        shared.setdefault(k, v)
    return stmts, types, shared


def reg_width(types, name):
    ty = types.get(name)
    if ty is None:
        m = re.match(r'(%[a-z]+)', name)
        ty = types.get(('prefix', m.group(1)))
    return {'pred': 1, 'b16': 16, 'u16': 16, 's16': 16, 'f16': 16, 'b32': 32, 'u32': 32, 's32': 32, 'f32': 32,
            'b64': 64, 'u64': 64, 's64': 64, 'f64': 64, 'b8': 8}[ty]


class Memory:
    """Global memory: the replay dump's allocations at their captured addresses."""

    def __init__(self, dump_dir):
        manifest = open(dump_dir + '/manifest.txt').read().split('\n')
        self.args = open(dump_dir + '/args.bin', 'rb').read()
        self.bases, self.data, self.ids = [], [], []
        for line in manifest:
            p = line.split()
            if p and p[0] == 'alloc':
                idx, base, size = int(p[1]), int(p[2], 16), int(p[3])
                buf = np.fromfile(f'{dump_dir}/alloc-{idx}.bin', np.uint8)
                assert buf.size == size
                self.bases.append(base)
                self.data.append(buf)
                self.ids.append(idx)
        order = np.argsort(self.bases)
        self.bases = [self.bases[i] for i in order]
        self.data = [self.data[i] for i in order]
        self.ids = [self.ids[i] for i in order]
        self.base_arr = np.array(self.bases, U64)
        self.stores = []  # (addresses, bytes) of every global store

    def locate(self, addr):
        idx = np.searchsorted(self.base_arr, addr, side='right') - 1
        if np.any(idx < 0):
            raise ValueError('address below every allocation')
        return idx

    def load(self, addr, nbytes, mask):
        out = np.zeros((addr.size, nbytes), np.uint8)
        idx = self.locate(np.where(mask, addr, self.base_arr[0]))
        for a in np.unique(idx[mask]):
            sel = mask & (idx == a)
            off = (addr[sel] - self.bases[a]).astype(np.int64)
            if np.any(off + nbytes > self.data[a].size):
                raise ValueError('load past the end of an allocation')
            out[sel] = self.data[a][off[:, None] + np.arange(nbytes)]
        return out

    def store(self, addr, raw, mask):
        idx = self.locate(np.where(mask, addr, self.base_arr[0]))
        n = raw.shape[1]
        for a in np.unique(idx[mask]):
            sel = mask & (idx == a)
            off = (addr[sel] - self.bases[a]).astype(np.int64)
            self.data[a][off[:, None] + np.arange(n)] = raw[sel]
            self.stores.append((self.ids[a], off.copy(), raw[sel].copy()))


class Block:
    def __init__(self, stmts, types, shared, mem, param_name, block_dim, grid_dim, ctaid):
        self.stmts, self.types, self.mem = stmts, types, mem
        self.param_name = param_name
        bx, by, bz = block_dim
        self.T = bx * by * bz
        t = np.arange(self.T)
        self.tid = {'x': t % bx, 'y': (t // bx) % by, 'z': t // (bx * by)}
        self.lane = t % 32
        self.warp = t // 32
        self.ctaid = ctaid
        self.grid = grid_dim
        self.ntid = block_dim
        self.regs = {}
        self.shared = {k: np.zeros(v, np.uint8) for k, v in shared.items()}
        self.shared_base = {}
        base = 0
        for k, v in shared.items():
            self.shared_base[k] = base
            base += (v + 15) & ~15
        self.shared_mem = np.zeros(max(base, 16), np.uint8)
        self.labels = {}
        for st in stmts:
            for l in st.labels:
                self.labels[l] = st.index
        self.hooks = []
        self.retired = np.zeros(self.T, bool)
        self.prov = {}

    # --- operands -----------------------------------------------------------
    def width(self, name):
        return reg_width(self.types, name)

    def special(self, name):
        m = re.match(r'%(tid|ntid|ctaid|nctaid)\.([xyz])', name)
        if m:
            kind, ax = m.groups()
            k = 'xyz'.index(ax)
            if kind == 'tid':
                return self.tid[ax].astype(U64)
            if kind == 'ntid':
                return np.full(self.T, self.ntid[k], U64)
            if kind == 'ctaid':
                return np.full(self.T, self.ctaid[k], U64)
            return np.full(self.T, self.grid[k], U64)
        if name == 'WARP_SZ':
            return np.full(self.T, 32, U64)
        if name == '%laneid':
            return self.lane.astype(U64)
        if name == '%warpid':
            return self.warp.astype(U64)
        return None

    def imm(self, s, width=64):
        if s.startswith('0f'):
            return np.full(self.T, int(s[2:], 16), U64)
        if s.startswith('0d'):
            return np.full(self.T, int(s[2:], 16), U64)
        v = int(s, 0)
        return np.full(self.T, v & MASK[64], U64)

    def get(self, s, width=None):
        s = s.strip()
        if s.startswith('%') or s in self.types or s == 'WARP_SZ':
            sp = self.special(s) if (s.startswith('%') or s == 'WARP_SZ') else None
            if sp is not None:
                return sp
            if s not in self.regs:
                w = self.width(s)
                self.regs[s] = np.zeros(self.T, bool if w == 1 else U64)
            return self.regs[s]
        if s in self.shared_base:
            return np.full(self.T, self.shared_base[s], U64)
        if s == self.param_name:
            return np.full(self.T, PARAM_BASE, U64)
        if s[0].isdigit() or s[0] == '-':
            v = self.imm(s)
            if width:
                v = v & np.uint64(MASK[width])
            return v
        raise NotImplementedError('operand ' + s)

    def put(self, name, value, mask, keep_prov=False):
        name = name.strip()
        if not keep_prov:
            self.prov.pop(name, None)
        w = self.width(name)
        old = self.get(name)
        if w == 1:
            self.regs[name] = np.where(mask, value.astype(bool), old)
        else:
            self.regs[name] = np.where(mask, value.astype(U64) & np.uint64(MASK[w]), old)

    def vec(self, s):
        return [x.strip() for x in s.strip()[1:-1].split(',')]

    def address(self, s):
        """[reg+off] or [symbol+off] -> (space, address array)."""
        inner = s.strip()[1:-1]
        m = re.match(r'([%\w$.]+)\s*(\+\s*-?\d+)?', inner)
        base, off = m.group(1), int(m.group(2).replace('+', '')) if m.group(2) else 0
        if base == self.param_name:
            return 'param', np.full(self.T, off, U64)
        if base in self.shared_base:
            return 'shared', np.full(self.T, self.shared_base[base] + off, U64)
        return 'reg', (self.get(base) + np.uint64(off & MASK[64])) & np.uint64(MASK[64])

    # --- execution ------------------------------------------------------------
    def run(self, limit=None):
        stmts = self.stmts
        active = np.ones(self.T, bool)
        pending = {}
        pc = 0
        steps = 0
        while pc < len(stmts):
            st = stmts[pc]
            if pc in pending:
                active |= pending.pop(pc)
            if not active.any():
                # Reconverge at either a forward target or a loop fallthrough.
                targets = sorted(pending)
                if not targets:
                    break
                pc = targets[0]
                continue
            active &= ~self.retired
            mask = active
            if st.pred is not None:
                p = self.get(st.pred)
                mask = active & (~p if st.neg else p)
            if st.op == 'bra' or st.op.startswith('bra.'):
                target = self.labels[st.args[0]]
                taking = mask
                if target <= pc:
                    if (taking == active).all():
                        pc = target
                        continue
                    if not taking.any():
                        pc += 1
                        continue
                    # Lanes finishing a loop wait at its fallthrough while the
                    # other lanes execute their remaining iterations. Their
                    # registers and memory writes stay masked until reconvergence.
                    pending[pc + 1] = pending.get(pc + 1, np.zeros(self.T, bool)) | (active & ~taking)
                    active = taking.copy()
                    pc = target
                    continue
                active = active & ~taking
                if taking.any():
                    pending[target] = pending.get(target, np.zeros(self.T, bool)) | taking
                pc += 1
                continue
            if mask.any() or st.op.startswith(('mma', 'shfl', 'movmatrix', 'bar')):
                self.execute(st, mask)
                for hook in self.hooks:
                    hook(self, st, mask)
            pc += 1
            steps += 1
            if limit and steps >= limit:
                break
        return steps

    def execute(self, st, mask):
        op, a = st.op, st.args
        parts = op.split('.')
        base = parts[0]
        fn = getattr(self, 'op_' + base, None)
        if fn is None:
            raise NotImplementedError(st.text)
        try:
            fn(st, parts, a, mask)
        except Exception as e:
            raise RuntimeError(f'{type(e).__name__} {e} in: {st.text}') from e

    # --- helpers --------------------------------------------------------------
    @staticmethod
    def sext(v, w):
        v = v.astype(np.int64) & ((1 << w) - 1) if w < 64 else v.view(np.int64)
        if w < 64:
            v = np.where(v >= (1 << (w - 1)), v - (1 << w), v)
        return v

    def typed(self, s, ty):
        w = int(re.sub(r'\D', '', ty) or 32) if ty not in ('pred',) else 1
        v = self.get(s, w)
        if ty.startswith('s'):
            return self.sext(v, w)
        if ty.startswith('f'):
            if ty == 'f16':
                return bits_to_f16(v).astype(F64)
            if ty == 'f32':
                return bits_to_f32(v).astype(F64)
            return v.view(F64)
        return (v & np.uint64(MASK[w])).astype(np.int64) if w < 64 else v

    def result(self, ty, value):
        if ty == 'f16':
            return f16_to_bits(np.asarray(value).astype(F16))
        if ty == 'f32':
            return f32_to_bits(np.asarray(value).astype(F32))
        if ty == 'f64':
            return np.asarray(value, F64).view(U64)
        w = int(re.sub(r'\D', '', ty))
        return (np.asarray(value).astype(np.int64).astype(U64)) & np.uint64(MASK[w])

    # --- instructions -----------------------------------------------------------
    def op_nop(self, st, parts, a, mask):
        pass

    def op_bar(self, st, parts, a, mask):
        pass

    def op_ret(self, st, parts, a, mask):
        self.retired |= mask

    def op_neg(self, st, parts, a, mask):
        ty = parts[-1]
        if ty == 'f16x2':
            self.put(a[0], self.get(a[1]) ^ np.uint64(0x80008000), mask)
        elif ty == 'f16':
            self.put(a[0], self.get(a[1]) ^ np.uint64(0x8000), mask)
        elif ty == 'f32':
            self.put(a[0], self.get(a[1]) ^ np.uint64(0x80000000), mask)
        else:
            self.put(a[0], self.result('u' + ty[1:], -self.typed(a[1], ty)), mask)

    def op_cvta(self, st, parts, a, mask):
        self.put(a[0], self.get(a[1]), mask)

    def op_mov(self, st, parts, a, mask):
        d, s = a
        if d.startswith('{'):
            names = self.vec(d)
            v = self.get(s)
            w = self.width(names[0])
            for i, n in enumerate(names):
                self.put(n, (v >> np.uint64(i * w)) & np.uint64(MASK[w]), mask)
            return
        if s.startswith('{'):
            names = self.vec(s)
            w = self.width(names[0])
            v = np.zeros(self.T, U64)
            for i, n in enumerate(names):
                v |= (self.get(n) & np.uint64(MASK[w])) << np.uint64(i * w)
            self.put(d, v, mask)
            return
        self.put(d, self.get(s), mask)
        if s in self.prov:
            self.prov[d] = self.prov[s]

    def op_ld(self, st, parts, a, mask):
        space = 'global' if 'global' in parts else 'shared' if 'shared' in parts else 'param' if 'param' in parts else None
        ty = parts[-1]
        vec = next((int(p[1:]) for p in parts if re.fullmatch(r'v\d', p)), 1)
        w = int(re.sub(r'\D', '', ty))
        nbytes = w // 8
        kind, addr = self.address(a[1])
        total = nbytes * vec
        if kind == 'param':
            raw = np.frombuffer(self.mem.args, np.uint8)
            off = int(addr[0])
            data = np.tile(raw[off:off + total], (self.T, 1))
        elif space == 'param':
            # ld.param through a computed address (mov.b64 %rd, <param>; add; ld.param [%rd])
            raw = np.frombuffer(self.mem.args, np.uint8)
            off = np.where(mask, addr - np.uint64(PARAM_BASE), 0).astype(np.int64)
            data = raw[off[:, None] + np.arange(total)]
        elif kind == 'shared' or space == 'shared':
            off = addr.astype(np.int64)
            data = self.shared_mem[off[:, None] + np.arange(total)]
        else:
            data = self.mem.load(addr, total, mask)
        names = self.vec(a[0]) if a[0].startswith('{') else [a[0]]
        for i, n in enumerate(names):
            chunk = data[:, i * nbytes:(i + 1) * nbytes]
            v = np.zeros(self.T, U64)
            for b in range(nbytes):
                v |= chunk[:, b].astype(U64) << np.uint64(8 * b)
            if ty.startswith('s') and w < self.width(n):
                v = (self.sext(v, w).astype(np.int64)).astype(U64)
            self.put(n, v, mask)
            self.prov[n] = (st.index, kind if kind != 'reg' else space, addr + np.uint64(i * nbytes), nbytes)

    def op_st(self, st, parts, a, mask):
        ty = parts[-1]
        vec = next((int(p[1:]) for p in parts if re.fullmatch(r'v\d', p)), 1)
        w = int(re.sub(r'\D', '', ty))
        nbytes = w // 8
        kind, addr = self.address(a[0])
        names = self.vec(a[1]) if a[1].startswith('{') else [a[1]]
        raw = np.zeros((self.T, nbytes * len(names)), np.uint8)
        for i, n in enumerate(names):
            v = self.get(n, w)
            for b in range(nbytes):
                raw[:, i * nbytes + b] = ((v >> np.uint64(8 * b)) & np.uint64(0xff)).astype(np.uint8)
        if kind == 'shared' or 'shared' in parts:
            off = addr.astype(np.int64)[mask]
            self.shared_mem[off[:, None] + np.arange(raw.shape[1])] = raw[mask]
        else:
            self.mem.store(addr, raw, mask)

    def int_binop(self, parts, a, mask, fn):
        ty = parts[-1]
        x, y = self.typed(a[1], ty), self.typed(a[2], ty)
        self.put(a[0], self.result(ty if not ty.startswith('b') else 'u' + ty[1:], fn(x, y)), mask)

    def op_add(self, st, parts, a, mask):
        ty = parts[-1]
        if ty in ('f16', 'f32'):
            self.put(a[0], self.result(ty, self.typed(a[1], ty) + self.typed(a[2], ty)), mask)
        elif ty == 'f16x2':
            self.put(a[0], self.h2(a[1], a[2], lambda x, y: x + y), mask)
        else:
            self.int_binop(parts, a, mask, lambda x, y: x + y)

    def op_sub(self, st, parts, a, mask):
        ty = parts[-1]
        if ty in ('f16', 'f32'):
            self.put(a[0], self.result(ty, self.typed(a[1], ty) - self.typed(a[2], ty)), mask)
        elif ty == 'f16x2':
            self.put(a[0], self.h2(a[1], a[2], lambda x, y: x - y), mask)
        else:
            self.int_binop(parts, a, mask, lambda x, y: x - y)

    def op_mul(self, st, parts, a, mask):
        ty = parts[-1]
        if 'wide' in parts:
            w = int(re.sub(r'\D', '', ty))
            x, y = self.typed(a[1], ty), self.typed(a[2], ty)
            self.put(a[0], (x * y).astype(np.int64).astype(U64), mask)
        elif ty in ('f16', 'f32'):
            self.put(a[0], self.result(ty, self.typed(a[1], ty) * self.typed(a[2], ty)), mask)
        elif ty == 'f16x2':
            self.put(a[0], self.h2(a[1], a[2], lambda x, y: x * y), mask)
        else:
            self.int_binop(parts, a, mask, lambda x, y: x * y)

    def op_mad(self, st, parts, a, mask):
        ty = parts[-1]
        x, y, z = self.typed(a[1], ty), self.typed(a[2], ty), self.typed(a[3], ty)
        self.put(a[0], self.result('u' + ty[1:], x * y + z), mask)

    def h2(self, sa, sb, fn, sc=None):
        va, vb = self.get(sa), self.get(sb)
        out = np.zeros(self.T, U64)
        for i in range(2):
            x = bits_to_f16(va >> np.uint64(16 * i)).astype(F64)
            y = bits_to_f16(vb >> np.uint64(16 * i)).astype(F64)
            if sc is not None:
                z = bits_to_f16(self.get(sc) >> np.uint64(16 * i)).astype(F64)
                r = fn(x, y, z)
            else:
                r = fn(x, y)
            out |= f16_to_bits(np.asarray(r).astype(F16)) << np.uint64(16 * i)
        return out

    def op_fma(self, st, parts, a, mask):
        ty = parts[-1]
        if ty == 'f16x2':
            self.put(a[0], self.h2(a[1], a[2], lambda x, y, z: x * y + z, a[3]), mask)
        elif ty in ('f16', 'f32'):
            self.put(a[0], self.result(ty, self.typed(a[1], ty) * self.typed(a[2], ty) + self.typed(a[3], ty)), mask)
        else:
            raise NotImplementedError(st.text)

    def minmax(self, parts, a, mask, is_max):
        ty = parts[-1]
        if ty == 'f16x2':
            f = np.fmax if is_max else np.fmin
            self.put(a[0], self.h2(a[1], a[2], f), mask)
        elif ty in ('f16', 'f32'):
            f = np.fmax if is_max else np.fmin
            self.put(a[0], self.result(ty, f(self.typed(a[1], ty), self.typed(a[2], ty))), mask)
        else:
            f = np.maximum if is_max else np.minimum
            self.int_binop(parts, a, mask, f)

    def op_min(self, st, parts, a, mask):
        self.minmax(parts, a, mask, False)

    def op_max(self, st, parts, a, mask):
        self.minmax(parts, a, mask, True)

    def op_abs(self, st, parts, a, mask):
        ty = parts[-1]
        if ty in ('f16x2', 'f16', 'f32'):
            m = {'f16x2': 0x7fff7fff, 'f16': 0x7fff, 'f32': 0x7fffffff}[ty]
            self.put(a[0], self.get(a[1]) & np.uint64(m), mask)
            return
        self.put(a[0], self.result('u' + ty[1:], np.abs(self.typed(a[1], ty))), mask)

    def logic(self, a, mask, fn):
        if self.width(a[0]) == 1:
            self.put(a[0], fn(self.get(a[1]), self.get(a[2])), mask)
        else:
            self.put(a[0], fn(self.get(a[1]), self.get(a[2])), mask)

    def op_and(self, st, parts, a, mask):
        self.logic(a, mask, lambda x, y: x & y)

    def op_or(self, st, parts, a, mask):
        self.logic(a, mask, lambda x, y: x | y)

    def op_xor(self, st, parts, a, mask):
        self.logic(a, mask, lambda x, y: x ^ y)

    def op_not(self, st, parts, a, mask):
        self.put(a[0], ~self.get(a[1]), mask)

    def op_shl(self, st, parts, a, mask):
        w = int(re.sub(r'\D', '', parts[-1]))
        n = self.get(a[2]) & np.uint64(0xff)
        v = np.where(n >= w, 0, (self.get(a[1]) << np.minimum(n, np.uint64(63))))
        self.put(a[0], v.astype(U64), mask)

    def op_shr(self, st, parts, a, mask):
        ty = parts[-1]
        w = int(re.sub(r'\D', '', ty))
        n = (self.get(a[2]) & np.uint64(0xff)).astype(np.int64)
        if ty.startswith('s'):
            v = self.sext(self.get(a[1]), w)
            r = v >> np.minimum(n, w - 1)
            self.put(a[0], r.astype(np.int64).astype(U64), mask)
        else:
            v = self.get(a[1]) & np.uint64(MASK[w])
            r = np.where(n >= w, 0, v >> np.minimum(n, 63).astype(U64))
            self.put(a[0], r.astype(U64), mask)

    def op_bfi(self, st, parts, a, mask):
        f, b = self.get(a[1]), self.get(a[2])
        pos = self.get(a[3]).astype(np.int64) & 0xff
        ln = self.get(a[4]).astype(np.int64) & 0xff
        w = int(re.sub(r'\D', '', parts[-1]))
        out = np.array(b, U64)
        for i in range(self.T):
            p, l = int(pos[i]), int(ln[i])
            if l == 0 or p >= w:
                continue
            l = min(l, w - p)
            m = ((1 << l) - 1) << p
            out[i] = (int(b[i]) & ~m & MASK[w]) | ((int(f[i]) << p) & m)
        self.put(a[0], out, mask)

    def op_setp(self, st, parts, a, mask):
        cmp, ty = parts[1], parts[-1]
        x, y = self.typed(a[1], ty), self.typed(a[2], ty)
        r = {'lt': x < y, 'le': x <= y, 'gt': x > y, 'ge': x >= y, 'eq': x == y, 'ne': x != y,
             'lo': x < y, 'ls': x <= y, 'hi': x > y, 'hs': x >= y}[cmp]
        dests = a[0].split('|')
        self.put(dests[0], r, mask)

    def op_selp(self, st, parts, a, mask):
        p = self.get(a[3])
        self.put(a[0], np.where(p, self.get(a[1]), self.get(a[2])), mask)

    def op_ex2(self, st, parts, a, mask):
        ty = parts[-1]
        v = self.get(a[1])
        if ty == 'f16x2':
            out = np.zeros(self.T, U64)
            for i in range(2):
                h = bits_to_f16((v >> np.uint64(16 * i)) & np.uint64(0xffff)).astype(F64)
                out |= f16_to_bits(np.exp2(h).astype(F16)) << np.uint64(16 * i)
            self.put(a[0], out, mask)
        elif ty == 'f16':
            self.put(a[0], f16_to_bits(np.exp2(bits_to_f16(v).astype(F64)).astype(F16)), mask)
        elif ty == 'f32':
            self.put(a[0], f32_to_bits(np.exp2(bits_to_f32(v).astype(F64)).astype(F32)), mask)
        else:
            raise NotImplementedError(st.text)

    def op_rsqrt(self, st, parts, a, mask):
        x = self.typed(a[1], 'f32').astype(F32)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.put(a[0], f32_to_bits((1.0 / np.sqrt(x.astype(F64))).astype(F32)), mask)

    def op_rcp(self, st, parts, a, mask):
        x = self.typed(a[1], 'f32')
        with np.errstate(divide='ignore'):
            self.put(a[0], f32_to_bits((1.0 / x).astype(F32)), mask)

    def op_div(self, st, parts, a, mask):
        x, y = self.typed(a[1], 'f32'), self.typed(a[2], 'f32')
        with np.errstate(divide='ignore', invalid='ignore'):
            self.put(a[0], f32_to_bits((x / y).astype(F32)), mask)

    def op_cvt(self, st, parts, a, mask):
        types = [p for p in parts[1:] if p not in ('rn', 'rz', 'rm', 'rp', 'rni', 'rzi', 'sat', 'satfinite', 'relu', 'ftz')]
        dt, stp = types[0], types[1]
        relu = 'relu' in parts
        v = self.get(a[1])
        if dt == 'e4m3x2' and stp == 'f16x2':
            out = np.zeros(self.T, U64)
            for i in range(2):
                h = (v >> np.uint64(16 * i)) & np.uint64(0xffff)
                if relu:
                    f = bits_to_f16(h)
                    h = np.where((f < 0) & ~np.isnan(f.astype(F32)), 0, h).astype(U64)
                out |= F16_TO_E4M3[h.astype(np.int64)].astype(U64) << np.uint64(8 * i)
            self.put(a[0], out, mask)
        elif dt == 'f16x2' and stp == 'e4m3x2':
            out = np.zeros(self.T, U64)
            for i in range(2):
                c = ((v >> np.uint64(8 * i)) & np.uint64(0xff)).astype(np.int64)
                out |= f16_to_bits(E4M3[c].astype(F16)) << np.uint64(16 * i)
            self.put(a[0], out, mask)
        elif dt == 'f16' and stp == 'f32':
            self.put(a[0], f16_to_bits(bits_to_f32(v).astype(F16)), mask)
        elif dt == 'f16' and stp == 'f64':
            self.put(a[0], f16_to_bits(v.view(F64).astype(F16)), mask)
        elif dt == 'f32' and stp == 'f16':
            self.put(a[0], f32_to_bits(bits_to_f16(v).astype(F32)), mask)
        elif dt in ('f16', 'f32') and stp[0] in 'us':
            ws = int(re.sub(r'\D', '', stp))
            iv = self.sext(v, ws) if stp[0] == 's' else (v & np.uint64(MASK[ws])).astype(np.int64)
            if dt == 'f16':
                self.put(a[0], f16_to_bits(iv.astype(F64).astype(F16)), mask)
            else:
                self.put(a[0], f32_to_bits(iv.astype(F64).astype(F32)), mask)
        elif dt[0] in 'us' and stp[0] in 'us':
            ws = int(re.sub(r'\D', '', stp))
            val = self.sext(v, ws) if stp[0] == 's' else (v & np.uint64(MASK[ws])).astype(np.int64)
            self.put(a[0], val.astype(np.int64).astype(U64), mask)
        else:
            raise NotImplementedError(st.text)

    def op_shfl(self, st, parts, a, mask):
        mode = parts[2]
        v = self.get(a[1])
        b = self.get(a[2]).astype(np.int64)
        out = np.array(v)
        for w in range(self.T // 32):
            sl = slice(32 * w, 32 * w + 32)
            lane = np.arange(32)
            if mode == 'bfly':
                src = lane ^ b[sl]
            elif mode == 'down':
                src = np.where(lane + b[sl] < 32, lane + b[sl], lane)
            elif mode == 'up':
                src = np.where(lane - b[sl] >= 0, lane - b[sl], lane)
            elif mode == 'idx':
                src = b[sl] & 31
            out[sl] = v[sl][src]
        self.put(a[0], out, mask)

    def op_movmatrix(self, st, parts, a, mask):
        v = self.get(a[1])
        out = np.zeros(self.T, U64)
        for w in range(self.T // 32):
            m = np.zeros((8, 8), U64)
            for lane in range(32):
                x = int(v[32 * w + lane])
                r, c = lane // 4, (lane % 4) * 2
                m[r, c], m[r, c + 1] = x & 0xffff, (x >> 16) & 0xffff
            mt = m.T
            for lane in range(32):
                r, c = lane // 4, (lane % 4) * 2
                out[32 * w + lane] = int(mt[r, c]) | (int(mt[r, c + 1]) << 16)
        self.put(a[0], out, mask)

    def op_mma(self, st, parts, a, mask):
        shape = parts[3]
        d_names, a_names, b_names, c_names = (self.vec(x) for x in a)
        A = [self.get(n) for n in a_names]
        Bv = [self.get(n) for n in b_names]
        C = [self.get(n) for n in c_names]
        dout = [np.array(self.get(n)) for n in d_names]
        for w in range(self.T // 32):
            base = 32 * w
            if shape == 'm16n8k32':
                Am = np.zeros((16, 32)); Bm = np.zeros((32, 8)); Cm = np.zeros((16, 8))
                for lane in range(32):
                    g, t = lane >> 2, lane & 3
                    for i in range(16):
                        byte = (int(A[i // 4][base + lane]) >> (8 * (i % 4))) & 0xff
                        row = g if (i < 4 or 8 <= i < 12) else g + 8
                        col = t * 4 + (i & 3) + (16 if i >= 8 else 0)
                        Am[row, col] = E4M3[byte]
                    for i in range(8):
                        byte = (int(Bv[i // 4][base + lane]) >> (8 * (i % 4))) & 0xff
                        Bm[t * 4 + (i & 3) + (16 if i >= 4 else 0), g] = E4M3[byte]
            elif shape == 'm16n8k16':
                Am = np.zeros((16, 16)); Bm = np.zeros((16, 8)); Cm = np.zeros((16, 8))
                for lane in range(32):
                    g, t = lane >> 2, lane & 3
                    for i in range(8):
                        h = (int(A[i // 2][base + lane]) >> (16 * (i % 2))) & 0xffff
                        row = g if (i < 2 or 4 <= i < 6) else g + 8
                        col = t * 2 + (i & 1) + (8 if i >= 4 else 0)
                        Am[row, col] = np.uint16(h).view(F16)
                    for i in range(4):
                        h = (int(Bv[i // 2][base + lane]) >> (16 * (i % 2))) & 0xffff
                        Bm[t * 2 + (i & 1) + (8 if i >= 2 else 0), g] = np.uint16(h).view(F16)
            else:
                raise NotImplementedError(st.text)
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for i in range(4):
                    h = (int(C[i // 2][base + lane]) >> (16 * (i % 2))) & 0xffff
                    Cm[g if i < 2 else g + 8, t * 2 + (i & 1)] = np.uint16(h).view(F16)
            Dm = (Am @ Bm + Cm).astype(F16)
            if self.mma_hook:
                self.mma_hook(self, st, w, Am, Bm, Cm, Dm, [self.prov.get(n) for n in a_names], [self.prov.get(n) for n in b_names])
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for r in range(2):
                    lo = int(Dm[g + 8 * r, t * 2].view(np.uint16))
                    hi = int(Dm[g + 8 * r, t * 2 + 1].view(np.uint16))
                    dout[r][base + lane] = lo | (hi << 16)
        for n, v in zip(d_names, dout):
            self.put(n, v, mask)

    mma_hook = None

    # wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 with register fragments: two m16n8k16 MMAs
    # sharing A registers 0..3, B registers 0,1 / 2,3 and C/D registers 0,1 / 2,3 (the upper A and B
    # registers of the fragment ABI are duplicates; ZLUDA's LLVMZludaSplitWmmaM16n16k16 mapping)
    def op_wmma(self, st, parts, a, mask):
        if 'm16n16k16' not in parts or 'mma' not in parts:
            raise NotImplementedError(st.text)
        d_names, a_names, b_names, c_names = (self.vec(x) for x in a)
        for half in range(2):
            sub = Stmt_like(st, 'mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16',
                            ['{%s}' % ', '.join(d_names[2 * half:2 * half + 2]), '{%s}' % ', '.join(a_names[0:4]),
                             '{%s}' % ', '.join(b_names[2 * half:2 * half + 2]), '{%s}' % ', '.join(c_names[2 * half:2 * half + 2])])
            self.op_mma(sub, sub.op.split('.'), sub.args, mask)


class Stmt_like:
    def __init__(self, st, op, args):
        self.op, self.args, self.text = op, args, st.text + f'  [as {op}]'
        for k in Stmt.__slots__:
            if k not in ('op', 'args', 'text'):
                setattr(self, k, getattr(st, k, None))


def run_block(ptx, kernel, dump, ctaid, hooks=(), mma_hook=None, limit=None):
    stmts, types, shared = parse_kernel(ptx, kernel)
    mem = Memory(dump)
    launch = [l for l in open(dump + '/manifest.txt') if l.startswith('launch')][0].split()
    grid = tuple(int(x) for x in launch[1:4])
    block = tuple(int(x) for x in launch[4:7])
    b = Block(stmts, types, shared, mem, kernel + '_param_0', block, grid, ctaid)
    b.hooks = list(hooks)
    b.mma_hook = mma_hook
    steps = b.run(limit)
    return b, mem, steps


if __name__ == '__main__':
    ptx, kernel, dump = sys.argv[1:4]
    cx, cy = (int(v) for v in sys.argv[4].split(',')) if len(sys.argv) > 4 else (0, 0)
    import time
    t0 = time.time()
    b, mem, steps = run_block(ptx, kernel, dump, (cx, cy, 0))
    print(f'block ({cx},{cy}): {steps} statements in {time.time() - t0:.1f}s, {len(mem.stores)} store groups')
