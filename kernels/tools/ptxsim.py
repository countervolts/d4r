#!/usr/bin/env python3
"""Vectorised PTX interpreter for one thread block (all threads as numpy lanes).

Runs one thread block of a kernel on the CPU against memory restored from a
capture (the replay directory: allocations before the launch). Every statement
can be observed (hooks, Stmt.line gives the .ptx line), which is how the
intermediates of a kernel are read. Textures and surfaces are bound per launch
through Block.resources (handle value -> Texture / Surface).

Only the instruction forms the RR and rrlite kernels use are implemented;
anything else raises NotImplementedError naming the statement.

Fidelity against the GPU reference (ZLUDA on gfx1201), where the last bit can differ:
- add/sub/mul/div/sqrt of two f16 or f32 operands: exact (f64 evaluation and one rounding;
  double rounding cannot occur for a single operation of binary16/binary32 operands).
- fma/mad (f16, f32): x*y+z is evaluated in f64 and rounded once; the product is exact but the
  sum is rounded in f64 first, so a rare last-bit difference from a true fused multiply-add is possible.
- .approx transcendentals (ex2, lg2, rcp, rsqrt, sqrt.approx, div.approx, tanh.approx): correctly
  rounded here. The AMD instructions behind the reference (v_exp_f32, v_rcp_f32, v_rsq_f32, v_log_f32)
  are approximate, about 1 ulp f32, so these can differ in the last bit.
- .ftz: denormal inputs and results of f32 operations are flushed to signed zero.
- mma.sync m16n8k32 e4m3 and m16n8k16 f16: products are exact; the k-sum is evaluated in f64 and rounded
  once to the accumulator type (MMA_MODE='exact'). MMA_MODE='split16' models the ZLUDA lowering of a k32
  step as two k16 WMMAs with C added in the first (each rounded to the accumulator type).
- tex (linear, normalised): texel values are exact; the bilinear weights are exact in f64 unless
  TEX_FRAC_BITS quantises them (hardware filters often keep fewer fraction bits); one rounding to the result type.
- shfl, movmatrix, prmt, cvt (rounding modes), integer ops: exact.
"""
import re
import sys

import numpy as np

F16, F32, F64 = np.float16, np.float32, np.float64
U64 = np.uint64
PARAM_BASE = 1 << 60  # virtual address of the kernel parameter block
MASK = {8: 0xff, 16: 0xffff, 32: 0xffffffff, 64: 0xffffffffffffffff}
F32_TINY = 2.0 ** -126
F16_TINY = 2.0 ** -14

MMA_MODE = 'exact'      # 'exact' | 'split16' (see module docstring)
TEX_FRAC_BITS = None    # None: exact bilinear weights; N: weights quantised to 1/2**N


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


def flush(x, tiny):
    """flush denormal values (held in float64) to signed zero"""
    return np.where(np.abs(x) < tiny, np.copysign(0.0, x), x)


class Stmt:
    __slots__ = ('labels', 'pred', 'neg', 'op', 'args', 'text', 'index', 'line')


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
    """(stmts, types, shared) of entry NAME in a .ptx file; Stmt.line is the source line."""
    src = open(path).read()
    i = src.index('.entry ' + name + '(')
    j = src.find('\n.visible', i + 10)
    j = len(src) if j < 0 else j
    k = src.index(')', i) + 1           # end of the parameter list
    base = src.count('\n', 0, k) + 1     # source line of body position 0
    # module-scope .shared arrays (declared before the entry) are visible to it too
    module_shared = {m.group(1): int(m.group(2)) for m in re.finditer(r'^\.shared[^\n;]*?(\w+)\[(\d+)\]', src[:i], re.M)}
    body = src[k:j]

    def _blank(m):                       # blank a match but keep its newlines: line numbers stay exact
        return re.sub(r'[^\n]', ' ', m.group(0))
    body = re.sub(r'//[^\n]*', _blank, body)
    body = re.sub(r'^[ \t]*\.(loc|file|pragma)[^\n]*', _blank, body, flags=re.M)
    body = re.sub(r'\.(maxntid|reqntid|minnctapersm|maxnreg)[^\n{]*', _blank, body)
    nl = [mm.start() for mm in re.finditer(r'\n', body)]
    nli = 0
    stmts, types, shared = [], {}, {}
    pending_labels = []
    pos = 0
    for m in re.finditer(r';|\Z', body):
        raw = body[pos:m.start()]
        p0 = pos + len(raw) - len(raw.lstrip())     # first character of the statement
        while nli < len(nl) and nl[nli] < p0:
            nli += 1
        seg_line = base + nli
        pos = m.end()
        s = re.sub(r'\s*\n\s*', ' ', raw).strip()
        while True:
            s = s.strip()
            if s.startswith('{') and not s.startswith('{%'):
                s = s[1:]
                continue
            if s.startswith('}'):
                s = s[1:]
                continue
            mm = re.match(r'(\$[\w]+):\s*', s)
            if mm:
                pending_labels.append(mm.group(1))
                s = s[mm.end():]
                continue
            break
        if not s:
            continue
        if s.startswith('.reg'):
            mr = re.match(r'\.reg\s*\.(\w+)\s+(.*)', s)
            ty = mr.group(1)
            for decl in mr.group(2).split(','):
                decl = decl.strip()
                m2 = re.match(r'(%[\w]+)<(\d+)>', decl)
                if m2:
                    types[('prefix', m2.group(1))] = ty
                else:
                    types[decl] = ty
            continue
        if s.startswith('.shared'):
            ms = re.search(r'(\w+)\[(\d+)\]', s)
            shared[ms.group(1)] = int(ms.group(2))
            continue
        if s.startswith('.'):
            continue
        st = Stmt()
        st.labels, pending_labels = pending_labels, []
        st.pred, st.neg = None, False
        mp = re.match(r'@(!?)(%\w+)\s+', s)
        if mp:
            st.neg, st.pred = mp.group(1) == '!', mp.group(2)
            s = s[mp.end():]
        parts = s.split(None, 1)
        st.op = parts[0]
        st.args = split_args(parts[1]) if len(parts) > 1 else []
        st.text = s
        st.index = len(stmts)
        st.line = seg_line
        stmts.append(st)
    if pending_labels:
        st = Stmt()
        st.labels, st.pred, st.neg, st.op, st.args, st.text, st.index = pending_labels, None, False, 'nop', [], 'nop', len(stmts)
        st.line = base + body.count('\n')
        stmts.append(st)
    for kk, v in module_shared.items():
        shared.setdefault(kk, v)
    return stmts, types, shared


def reg_width(types, name):
    ty = types.get(name)
    if ty is None:
        m = re.match(r'(%[a-z]+)', name)
        ty = types.get(('prefix', m.group(1)))
    return {'pred': 1, 'b16': 16, 'u16': 16, 's16': 16, 'f16': 16, 'b32': 32, 'u32': 32, 's32': 32, 'f32': 32,
            'b64': 64, 'u64': 64, 's64': 64, 'f64': 64, 'b8': 8}[ty]


class Memory:
    """Global memory: the allocations of a replay directory (manifest.txt, alloc-N.bin, args.bin)."""

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
        # (alloc id, element offsets within the allocation, raw bytes per lane, element type or None)
        self.stores = []

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

    def store(self, addr, raw, mask, ty=None):
        idx = self.locate(np.where(mask, addr, self.base_arr[0]))
        n = raw.shape[1]
        for a in np.unique(idx[mask]):
            sel = mask & (idx == a)
            off = (addr[sel] - self.bases[a]).astype(np.int64)
            self.data[a][off[:, None] + np.arange(n)] = raw[sel]
            self.stores.append((self.ids[a], off.copy(), raw[sel].copy(), ty))


class Texture:
    """A texture object over captured texels. Texels are decoded to float64 RGBA; missing channels
    are filled as CUDA does (0, 0, 0, 1). bits: 16/32 float, 8 UNORM (v/255), 9 SNORM (max(v/127, -1))."""

    def __init__(self, path, width, height, channels, bits, linear):
        raw = np.fromfile(str(path), np.uint8)
        per = bits // 8 if bits >= 16 else 1
        if raw.size != width * height * channels * per:
            raise ValueError(f'texture {path}: {raw.size} bytes, expected {width * height * channels * per}')
        if bits == 16:
            x = raw.view(F16).astype(F64)
        elif bits == 32:
            x = raw.view(F32).astype(F64)
        elif bits == 8:
            x = raw.astype(F64) / 255.0
        elif bits == 9:
            x = np.maximum(raw.view(np.int8).astype(F64) / 127.0, -1.0)
        else:
            raise NotImplementedError(f'texture bits {bits}')
        x = x.reshape(height, width, channels)
        full = np.zeros((height, width, 4), F64)
        full[..., :channels] = x
        for c in range(channels, 4):
            full[..., c] = 1.0 if c == 3 else 0.0
        self.texels = full
        self.H, self.W = height, width
        self.linear = linear

    def fetch(self, x, y):
        """point fetch of texel (x, y) (int64 arrays), clamped to the texture: (N, 4) float64"""
        return self.texels[np.clip(y, 0, self.H - 1), np.clip(x, 0, self.W - 1)]

    def sample(self, u, v):
        """normalised coordinates, LOD 0: bilinear (linear) or nearest. (N, 4) float64"""
        x = u * self.W
        y = v * self.H
        if not self.linear:
            return self.fetch(np.floor(x).astype(np.int64), np.floor(y).astype(np.int64))
        x0, y0 = np.floor(x - 0.5), np.floor(y - 0.5)
        fx, fy = (x - 0.5) - x0, (y - 0.5) - y0
        if TEX_FRAC_BITS is not None:
            q = float(2 ** TEX_FRAC_BITS)
            fx, fy = np.floor(fx * q) / q, np.floor(fy * q) / q
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        fx, fy = fx[:, None], fy[:, None]
        top = (1 - fx) * self.fetch(x0, y0) + fx * self.fetch(x0 + 1, y0)
        bot = (1 - fx) * self.fetch(x0, y0 + 1) + fx * self.fetch(x0 + 1, y0 + 1)
        return (1 - fy) * top + fy * bot


class Surface:
    """A surface: byte-addressed texels (channels x bits each), zero-initialised. `written` marks stored bytes;
    stores outside the surface are dropped (zero boundary mode)."""

    def __init__(self, width, height, channels, bits):
        self.W, self.H, self.C, self.bits = width, height, channels, bits
        self.texel_bytes = channels * bits // 8
        self.data = np.zeros(width * height * self.texel_bytes, np.uint8)
        self.written = np.zeros(self.data.size, bool)

    def write(self, x, y, vals, bits):
        if bits != self.bits:
            raise ValueError(f'surface of {self.bits}-bit channels written as b{bits}')
        ok = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        base = (y[ok] * self.W + x[ok]) * self.texel_bytes
        nb = bits // 8
        for c, v in enumerate(vals):
            if c >= self.C:
                raise ValueError(f'sust writes channel {c} of a {self.C}-channel surface')
            v = v[ok]
            for b in range(nb):
                idx = base + c * nb + b
                self.data[idx] = ((v >> U64(8 * b)) & U64(0xff)).astype(np.uint8)
                self.written[idx] = True


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
            for lab in st.labels:
                self.labels[lab] = st.index
        self.hooks = []
        self.retired = np.zeros(self.T, bool)
        self.prov = {}
        self.resources = {}     # texture/surface handle value -> Texture / Surface
        self.mma_hook = None
        self.steps = 0

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
        v = int(s.rstrip('Uu'), 0)   # inline-asm literals carry a C suffix (0x3fb8aa3bU)
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
            self.regs[name] = np.where(mask, np.asarray(value).astype(bool), old)
        else:
            self.regs[name] = np.where(mask, np.asarray(value).astype(U64) & np.uint64(MASK[w]), old)

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

    def resource(self, h, cls):
        obj = self.resources.get(int(h))
        if not isinstance(obj, cls):
            raise NotImplementedError(f'no {cls.__name__} bound to handle {int(h):#x}')
        return obj

    # --- execution ------------------------------------------------------------
    def run(self, limit=None):
        stmts = self.stmts
        active = np.ones(self.T, bool)
        pending = {}
        pc = 0
        steps = 0
        while pc < len(stmts):
            st = stmts[pc]
            for lab in st.labels:
                if lab in pending:
                    active |= pending.pop(lab)
            if not active.any():
                # jump to the nearest label someone waits at
                targets = sorted(self.labels[lab] for lab in pending)
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
                    raise NotImplementedError(f'divergent backward branch at line {st.line}: {st.text}')
                active = active & ~taking
                if taking.any():
                    pending[st.args[0]] = pending.get(st.args[0], np.zeros(self.T, bool)) | taking
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
        self.steps = steps
        return steps

    def execute(self, st, mask):
        op, a = st.op, st.args
        parts = op.split('.')
        base = parts[0]
        fn = getattr(self, 'op_' + base, None)
        if fn is None:
            raise NotImplementedError(f'line {st.line}: {st.text}')
        try:
            fn(st, parts, a, mask)
        except NotImplementedError as e:
            raise NotImplementedError(f'{e} (line {st.line}: {st.text})') from e
        except Exception as e:
            raise RuntimeError(f'{type(e).__name__} {e} at line {st.line}: {st.text}') from e

    # --- helpers --------------------------------------------------------------
    @staticmethod
    def sext(v, w):
        v = v.astype(np.int64) & ((1 << w) - 1) if w < 64 else v.view(np.int64)
        if w < 64:
            v = np.where(v >= (1 << (w - 1)), v - (1 << w), v)
        return v

    def typed(self, s, ty, ftz=False):
        w = 1 if ty == 'pred' else int(re.sub(r'\D', '', ty) or 32)
        v = self.get(s, w)
        if ty.startswith('s'):
            return self.sext(v, w)
        if ty.startswith('f'):
            if ty == 'f16':
                return bits_to_f16(v).astype(F64)
            if ty == 'f32':
                x = bits_to_f32(v).astype(F64)
                return flush(x, F32_TINY) if ftz else x
            return v.view(F64)
        return (v & np.uint64(MASK[w])).astype(np.int64) if w < 64 else v

    def result(self, ty, value, ftz=False):
        if ty == 'f16':
            return f16_to_bits(np.asarray(value, F64).astype(F16))
        if ty == 'f32':
            v = np.asarray(value, F64).astype(F32)
            if ftz:
                v = flush(v.astype(F64), F32_TINY).astype(F32)
            return f32_to_bits(v)
        if ty == 'f64':
            return np.asarray(value, F64).view(U64)
        w = int(re.sub(r'\D', '', ty))
        return (np.asarray(value).astype(np.int64).astype(U64)) & np.uint64(MASK[w])

    @staticmethod
    def relu(r, parts):
        return np.where(r < 0, 0.0, r) if 'relu' in parts else r

    # --- instructions -----------------------------------------------------------
    def op_nop(self, st, parts, a, mask):
        pass

    op_bar = op_nop
    op_membar = op_nop
    op_fence = op_nop
    op_prefetch = op_nop
    op_ldu = None  # set below

    def op_ret(self, st, parts, a, mask):
        self.retired |= mask

    op_exit = op_ret

    def op_neg(self, st, parts, a, mask):
        ty = parts[-1]
        if ty == 'f16x2':
            self.put(a[0], self.get(a[1]) ^ U64(0x80008000), mask)
        elif ty == 'f16':
            self.put(a[0], self.get(a[1]) ^ U64(0x8000), mask)
        elif ty == 'f32':
            self.put(a[0], self.get(a[1]) ^ U64(0x80000000), mask)
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
                self.put(n, (v >> U64(i * w)) & U64(MASK[w]), mask)
            return
        if s.startswith('{'):
            names = self.vec(s)
            w = self.width(names[0])
            v = np.zeros(self.T, U64)
            for i, n in enumerate(names):
                v |= (self.get(n) & U64(MASK[w])) << U64(i * w)
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
            off = np.where(mask, addr - U64(PARAM_BASE), 0).astype(np.int64)
            data = raw[off[:, None] + np.arange(total)]
        elif kind == 'shared' or space == 'shared':
            off = np.where(mask, addr, 0).astype(np.int64)
            data = self.shared_mem[off[:, None] + np.arange(total)]
        else:
            data = self.mem.load(addr, total, mask)
        names = self.vec(a[0]) if a[0].startswith('{') else [a[0]]
        for i, n in enumerate(names):
            chunk = data[:, i * nbytes:(i + 1) * nbytes]
            v = np.zeros(self.T, U64)
            for b in range(nbytes):
                v |= chunk[:, b].astype(U64) << U64(8 * b)
            if ty.startswith('s') and w < self.width(n):
                v = (self.sext(v, w).astype(np.int64)).astype(U64)
            self.put(n, v, mask)
            self.prov[n] = (st.index, kind if kind != 'reg' else space, addr + U64(i * nbytes), nbytes)

    op_ldu = op_ld

    def op_st(self, st, parts, a, mask):
        ty = parts[-1]
        w = int(re.sub(r'\D', '', ty))
        nbytes = w // 8
        kind, addr = self.address(a[0])
        names = self.vec(a[1]) if a[1].startswith('{') else [a[1]]
        raw = np.zeros((self.T, nbytes * len(names)), np.uint8)
        for i, n in enumerate(names):
            v = self.get(n, w)
            for b in range(nbytes):
                raw[:, i * nbytes + b] = ((v >> U64(8 * b)) & U64(0xff)).astype(np.uint8)
        if kind == 'shared' or 'shared' in parts:
            off = addr.astype(np.int64)[mask]
            self.shared_mem[off[:, None] + np.arange(raw.shape[1])] = raw[mask]
        else:
            self.mem.store(addr, raw, mask, ty)

    def int_binop(self, parts, a, mask, fn):
        ty = parts[-1]
        x, y = self.typed(a[1], ty), self.typed(a[2], ty)
        self.put(a[0], self.result(ty if not ty.startswith('b') else 'u' + ty[1:], fn(x, y)), mask)

    def h2(self, sa, sb, fn, sc=None, relu=False):
        va, vb = self.get(sa), self.get(sb)
        vc = self.get(sc) if sc is not None else None
        out = np.zeros(self.T, U64)
        for i in range(2):
            x = bits_to_f16(va >> U64(16 * i)).astype(F64)
            y = bits_to_f16(vb >> U64(16 * i)).astype(F64)
            if vc is not None:
                z = bits_to_f16(vc >> U64(16 * i)).astype(F64)
                r = fn(x, y, z)
            else:
                r = fn(x, y)
            if relu:
                r = np.where(r < 0, 0.0, r)
            out |= f16_to_bits(np.asarray(r).astype(F16)) << U64(16 * i)
        return out

    def h2u(self, sa, fn):
        va = self.get(sa)
        out = np.zeros(self.T, U64)
        for i in range(2):
            x = bits_to_f16(va >> U64(16 * i)).astype(F64)
            with np.errstate(all='ignore'):
                out |= f16_to_bits(np.asarray(fn(x)).astype(F16)) << U64(16 * i)
        return out

    def op_add(self, st, parts, a, mask):
        ty, ftz = parts[-1], 'ftz' in parts
        if ty in ('f16', 'f32'):
            r = self.typed(a[1], ty, ftz) + self.typed(a[2], ty, ftz)
            self.put(a[0], self.result(ty, self.relu(r, parts), ftz), mask)
        elif ty == 'f16x2':
            self.put(a[0], self.h2(a[1], a[2], lambda x, y: x + y, relu='relu' in parts), mask)
        else:
            self.int_binop(parts, a, mask, lambda x, y: x + y)

    def op_sub(self, st, parts, a, mask):
        ty, ftz = parts[-1], 'ftz' in parts
        if ty in ('f16', 'f32'):
            self.put(a[0], self.result(ty, self.typed(a[1], ty, ftz) - self.typed(a[2], ty, ftz), ftz), mask)
        elif ty == 'f16x2':
            self.put(a[0], self.h2(a[1], a[2], lambda x, y: x - y), mask)
        else:
            self.int_binop(parts, a, mask, lambda x, y: x - y)

    def int_mul(self, parts, a, mask, z=None):
        """mul/mad .wide and .hi (integers): products through 64-bit modular arithmetic (exact)"""
        ty = parts[-1]
        w = int(re.sub(r'\D', '', ty))
        x = self.typed(a[1], ty).astype(U64)
        y = self.typed(a[2], ty).astype(U64)
        p = x * y
        if 'hi' in parts:
            out = p >> U64(w)
            if z is not None:
                out = out + self.get(z)
            self.put(a[0], out & U64(MASK[w]), mask)
        else:  # wide
            out = p + (self.get(z) if z is not None else U64(0))
            self.put(a[0], out, mask)

    def op_mul(self, st, parts, a, mask):
        ty, ftz = parts[-1], 'ftz' in parts
        if 'wide' in parts or 'hi' in parts:
            self.int_mul(parts, a, mask)
        elif ty in ('f16', 'f32'):
            r = self.typed(a[1], ty, ftz) * self.typed(a[2], ty, ftz)
            self.put(a[0], self.result(ty, self.relu(r, parts), ftz), mask)
        elif ty == 'f16x2':
            self.put(a[0], self.h2(a[1], a[2], lambda x, y: x * y, relu='relu' in parts), mask)
        else:
            self.int_binop(parts, a, mask, lambda x, y: x * y)

    def op_mad(self, st, parts, a, mask):
        ty, ftz = parts[-1], 'ftz' in parts
        if ty in ('f16', 'f32'):
            r = self.typed(a[1], ty, ftz) * self.typed(a[2], ty, ftz) + self.typed(a[3], ty, ftz)
            self.put(a[0], self.result(ty, self.relu(r, parts), ftz), mask)
        elif 'wide' in parts or 'hi' in parts:
            self.int_mul(parts, a, mask, z=a[3])
        else:  # mad.lo
            w = int(re.sub(r'\D', '', ty))
            x = self.typed(a[1], ty).astype(U64)
            y = self.typed(a[2], ty).astype(U64)
            self.put(a[0], (x * y + self.get(a[3])) & U64(MASK[w]), mask)

    def op_fma(self, st, parts, a, mask):
        ty, ftz = parts[-1], 'ftz' in parts
        if ty == 'f16x2':
            self.put(a[0], self.h2(a[1], a[2], lambda x, y, z: x * y + z, a[3], relu='relu' in parts), mask)
        elif ty in ('f16', 'f32'):
            r = self.typed(a[1], ty, ftz) * self.typed(a[2], ty, ftz) + self.typed(a[3], ty, ftz)
            self.put(a[0], self.result(ty, self.relu(r, parts), ftz), mask)
        else:
            raise NotImplementedError(st.text)

    def minmax(self, parts, a, mask, is_max):
        ty, ftz = parts[-1], 'ftz' in parts
        f = np.fmax if is_max else np.fmin
        if ty == 'f16x2':
            self.put(a[0], self.h2(a[1], a[2], f), mask)
        elif ty in ('f16', 'f32'):
            self.put(a[0], self.result(ty, f(self.typed(a[1], ty, ftz), self.typed(a[2], ty, ftz)), ftz), mask)
        else:
            self.int_binop(parts, a, mask, np.maximum if is_max else np.minimum)

    def op_min(self, st, parts, a, mask):
        self.minmax(parts, a, mask, False)

    def op_max(self, st, parts, a, mask):
        self.minmax(parts, a, mask, True)

    def op_abs(self, st, parts, a, mask):
        ty = parts[-1]
        if ty in ('f16x2', 'f16', 'f32'):
            m = {'f16x2': 0x7fff7fff, 'f16': 0x7fff, 'f32': 0x7fffffff}[ty]
            self.put(a[0], self.get(a[1]) & U64(m), mask)
            return
        self.put(a[0], self.result('u' + ty[1:], np.abs(self.typed(a[1], ty))), mask)

    def logic(self, a, mask, fn):
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
        n = self.get(a[2]) & U64(0xff)
        v = np.where(n >= w, 0, (self.get(a[1]) << np.minimum(n, U64(63))))
        self.put(a[0], v.astype(U64), mask)

    def op_shr(self, st, parts, a, mask):
        ty = parts[-1]
        w = int(re.sub(r'\D', '', ty))
        n = (self.get(a[2]) & U64(0xff)).astype(np.int64)
        if ty.startswith('s'):
            v = self.sext(self.get(a[1]), w)
            r = v >> np.minimum(n, w - 1)
            self.put(a[0], r.astype(np.int64).astype(U64), mask)
        else:
            v = self.get(a[1]) & U64(MASK[w])
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

    def op_bfe(self, st, parts, a, mask):
        """bfe.{u,s}N d, a, pos, len"""
        ty = parts[-1]
        w = int(re.sub(r'\D', '', ty))
        x = self.get(a[1]) & U64(MASK[w])
        pos = (self.get(a[2]) & U64(0xff)).astype(np.int64)
        ln = (self.get(a[3]) & U64(0xff)).astype(np.int64)
        pos = np.minimum(pos, w)
        ln = np.minimum(ln, w - pos)
        field = (x >> pos.astype(U64)) & ((U64(1) << ln.astype(U64)) - U64(1))
        if ty.startswith('s'):
            sgn_bit = np.where(ln > 0, (field >> (ln - 1).astype(U64)) & U64(1), 0).astype(bool)
            field = np.where(sgn_bit & (ln > 0), field | (U64(MASK[64]) << ln.astype(U64)), field)
        self.put(a[0], field & U64(MASK[w]) if w < 64 else field, mask)

    def op_prmt(self, st, parts, a, mask):
        """prmt.b32 d, a, b, c (default mode): bytes 0-3 are a, 4-7 are b"""
        if len(parts) > 2:
            raise NotImplementedError(st.text)
        comb = self.get(a[1]) & U64(0xffffffff)
        comb = comb | ((self.get(a[2]) & U64(0xffffffff)) << U64(32))
        sel = self.get(a[3])
        out = np.zeros(self.T, U64)
        for i in range(4):
            s = (sel >> U64(4 * i)) & U64(0xf)
            idx = s & U64(7)
            byte = (comb >> (idx * U64(8))) & U64(0xff)
            sign = (byte >> U64(7)) & U64(1)
            byte = np.where((s & U64(8)) != 0, np.where(sign == 1, U64(0xff), U64(0)), byte)
            out |= byte << U64(8 * i)
        self.put(a[0], out, mask)

    def op_setp(self, st, parts, a, mask):
        cmp, ty, ftz = parts[1], parts[-1], 'ftz' in parts
        x, y = self.typed(a[1], ty, ftz), self.typed(a[2], ty, ftz)
        un = np.isnan(x) | np.isnan(y) if ty in ('f16', 'f32', 'f64') else np.zeros(self.T, bool)
        base = {'lt': x < y, 'le': x <= y, 'gt': x > y, 'ge': x >= y, 'eq': x == y, 'ne': x != y,
                'lo': x < y, 'ls': x <= y, 'hi': x > y, 'hs': x >= y,
                'equ': (x == y) | un, 'neu': (x != y) | un, 'ltu': (x < y) | un, 'leu': (x <= y) | un,
                'gtu': (x > y) | un, 'geu': (x >= y) | un, 'num': ~un, 'nan': un}[cmp]
        r = np.asarray(base, bool)
        dests = a[0].split('|')
        if len(a) == 4:  # setp.CMP.BOOLOP p[|q], a, b, c
            q = self.get(a[3].lstrip('!'))
            if a[3].startswith('!'):
                q = ~q
            op = parts[2]
            r = {'and': r & q, 'or': r | q, 'xor': r ^ q}[op]
        self.put(dests[0], r, mask)
        if len(dests) > 1:
            self.put(dests[1], ~r, mask)

    def op_set(self, st, parts, a, mask):
        """set.CMP.DTYPE.STYPE d, a, b: d = 1.0 (a float DTYPE) or all ones (an integer DTYPE) where the comparison
        holds, else 0"""
        cmp, dty, ty = parts[1], parts[-2], parts[-1]
        x, y = self.typed(a[1], ty, 'ftz' in parts), self.typed(a[2], ty, 'ftz' in parts)
        un = np.isnan(x) | np.isnan(y) if ty in ('f16', 'f32', 'f64') else np.zeros(self.T, bool)
        r = np.asarray({'lt': x < y, 'le': x <= y, 'gt': x > y, 'ge': x >= y, 'eq': x == y, 'ne': x != y,
                        'equ': (x == y) | un, 'neu': (x != y) | un, 'ltu': (x < y) | un, 'leu': (x <= y) | un,
                        'gtu': (x > y) | un, 'geu': (x >= y) | un, 'num': ~un, 'nan': un}[cmp], bool)
        if dty.startswith('f'):
            self.put(a[0], self.result(dty, np.where(r, 1.0, 0.0)), mask)
        else:
            w = int(re.sub(r'\D', '', dty))
            self.put(a[0], np.where(r, np.uint64(MASK[w]), np.uint64(0)).astype(U64), mask)

    def op_selp(self, st, parts, a, mask):
        p = self.get(a[3])
        self.put(a[0], np.where(p, self.get(a[1]), self.get(a[2])), mask)

    def unary(self, parts, a, mask, fn):
        """unary float op on f16, f16x2 or f32 (ftz honoured for f32)"""
        ty, ftz = parts[-1], 'ftz' in parts
        with np.errstate(all='ignore'):
            if ty == 'f16x2':
                self.put(a[0], self.h2u(a[1], fn), mask)
            elif ty == 'f16':
                self.put(a[0], f16_to_bits(np.asarray(fn(self.typed(a[1], 'f16'))).astype(F16)), mask)
            elif ty in ('f32', 'f64'):
                self.put(a[0], self.result(ty, fn(self.typed(a[1], ty, ftz)), ftz), mask)
            else:
                raise NotImplementedError(st.text if False else f'unary {ty}')

    def op_ex2(self, st, parts, a, mask):
        self.unary(parts, a, mask, np.exp2)

    def op_lg2(self, st, parts, a, mask):
        self.unary(parts, a, mask, np.log2)

    def op_tanh(self, st, parts, a, mask):
        self.unary(parts, a, mask, np.tanh)

    def op_sqrt(self, st, parts, a, mask):
        self.unary(parts, a, mask, np.sqrt)

    def op_rsqrt(self, st, parts, a, mask):
        self.unary(parts, a, mask, lambda x: 1.0 / np.sqrt(x))

    def op_rcp(self, st, parts, a, mask):
        self.unary(parts, a, mask, lambda x: 1.0 / x)

    def op_div(self, st, parts, a, mask):
        ty, ftz = parts[-1], 'ftz' in parts
        if ty in ('f32', 'f64', 'f16'):
            with np.errstate(all='ignore'):
                x, y = self.typed(a[1], ty, ftz), self.typed(a[2], ty, ftz)
                self.put(a[0], self.result(ty, x / y, ftz), mask)
            return
        w = int(re.sub(r'\D', '', ty))
        x = self.typed(a[1], ty)
        y = self.typed(a[2], ty)
        safe = np.where(y == 0, 1, y)
        q = np.sign(x) * np.sign(safe) * (np.abs(x) // np.abs(safe))
        self.put(a[0], np.where(y == 0, 0, q).astype(np.int64).astype(U64) & U64(MASK[w]), mask)

    CVT_MODS = {'rn', 'rz', 'rm', 'rp', 'rni', 'rzi', 'rmi', 'rpi', 'sat', 'satfinite', 'relu', 'ftz', 'approx'}

    def op_cvt(self, st, parts, a, mask):
        types = [p for p in parts[1:] if p not in self.CVT_MODS]
        dt, stp = types[0], types[1]
        ftz = 'ftz' in parts
        v = self.get(a[1])
        if dt == 'e4m3x2' and stp == 'f16x2':
            out = np.zeros(self.T, U64)
            for i in range(2):
                h = (v >> U64(16 * i)) & U64(0xffff)
                if 'relu' in parts:
                    f = bits_to_f16(h)
                    h = np.where((f < 0) & ~np.isnan(f.astype(F32)), 0, h).astype(U64)
                out |= F16_TO_E4M3[h.astype(np.int64)].astype(U64) << U64(8 * i)
            self.put(a[0], out, mask)
        elif dt == 'f16x2' and stp == 'e4m3x2':
            out = np.zeros(self.T, U64)
            for i in range(2):
                c = ((v >> U64(8 * i)) & U64(0xff)).astype(np.int64)
                out |= f16_to_bits(E4M3[c].astype(F16)) << U64(16 * i)
            self.put(a[0], out, mask)
        elif dt[0] in 'us' and stp.startswith('f'):   # float -> integer, saturating, NaN -> 0
            x = self.typed(a[1], stp, ftz)
            if any(p in parts for p in ('rzi', 'rz')) or (not any(p in parts for p in ('rni', 'rn', 'rmi', 'rm', 'rpi', 'rp'))):
                r = np.trunc(x)
            elif any(p in parts for p in ('rni', 'rn')):
                r = np.rint(x)
            elif any(p in parts for p in ('rmi', 'rm')):
                r = np.floor(x)
            else:
                r = np.ceil(x)
            ws = int(re.sub(r'\D', '', dt))
            lo, hi = (-2.0 ** (ws - 1), 2.0 ** (ws - 1) - 1) if dt[0] == 's' else (0.0, 2.0 ** ws - 1)
            r = np.where(np.isnan(r), 0.0, np.clip(r, lo, hi))
            self.put(a[0], r.astype(np.int64).astype(U64) & U64(MASK[ws]), mask)
        elif dt.startswith('f') and (stp[0] in 'us'):   # integer -> float
            iv = self.typed(a[1], stp)
            self.put(a[0], self.result(dt, np.asarray(iv).astype(F64), ftz), mask)
        elif dt.startswith('f') and stp.startswith('f'):   # float -> float (rzi / rni / rmi / rpi: to an integral value)
            x = self.typed(a[1], stp, ftz)
            for mod, fn in (('rzi', np.trunc), ('rni', np.rint), ('rmi', np.floor), ('rpi', np.ceil)):
                if mod in parts:
                    x = fn(x)
            self.put(a[0], self.result(dt, x, ftz), mask)
        elif dt[0] in 'us' and stp[0] in 'us':
            ws = int(re.sub(r'\D', '', stp))
            val = self.sext(v, ws) if stp[0] == 's' else (v & U64(MASK[ws])).astype(np.int64)
            self.put(a[0], val.astype(np.int64).astype(U64), mask)
        else:
            raise NotImplementedError(st.text)

    def op_shfl(self, st, parts, a, mask):
        """shfl.sync.MODE.b32 d[|p], a, b, c, membermask (warp-collective; membermask is not checked)"""
        mode = parts[2]
        dst, pred = (a[0].split('|') + [None])[:2]
        v = self.get(a[1]) & U64(0xffffffff)
        b = self.get(a[2]).astype(np.int64)
        c = self.get(a[3]).astype(np.int64) if len(a) > 3 else np.full(self.T, 31, np.int64)
        clamp, seg = c & 0x1f, (c >> 8) & 0x1f
        lane = self.lane.astype(np.int64)
        minl = lane & seg
        maxl = minl | (clamp & ~seg & 0x1f)
        if mode == 'up':
            j = lane - b
            ok = j >= minl
        elif mode == 'down':
            j = lane + b
            ok = j <= maxl
        elif mode == 'bfly':
            j = lane ^ (b & 0x1f)
            ok = (j >= minl) & (j <= maxl)
        elif mode == 'idx':
            j = minl | (b & ~seg & 0x1f)
            ok = j <= maxl
        else:
            raise NotImplementedError(st.text)
        src = np.where(ok, j, lane) & 31
        gidx = self.warp.astype(np.int64) * 32 + src
        self.put(dst, v[gidx], mask)
        if pred is not None:
            self.put(pred, ok, mask)

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

    @staticmethod
    def _mma_maps(k):
        """per-lane (row, col) of every fragment element: A (16 x k), B (k x 8), C/D (16 x 8)"""
        lane = np.arange(32)
        g, t = lane >> 2, lane & 3
        if k == 32:
            i = np.arange(16)
            ar = np.where((i < 4) | ((i >= 8) & (i < 12)), g[:, None], g[:, None] + 8)
            ac = t[:, None] * 4 + (i & 3) + np.where(i >= 8, 16, 0)
            j = np.arange(8)
            br = t[:, None] * 4 + (j & 3) + np.where(j >= 4, 16, 0)
        else:
            i = np.arange(8)
            ar = np.where((i < 2) | ((i >= 4) & (i < 6)), g[:, None], g[:, None] + 8)
            ac = t[:, None] * 2 + (i & 1) + np.where(i >= 4, 8, 0)
            j = np.arange(4)
            br = t[:, None] * 2 + (j & 1) + np.where(j >= 2, 8, 0)
        bc = np.broadcast_to(g[:, None], br.shape)
        i = np.arange(4)
        cr = g[:, None] + np.where(i >= 2, 8, 0)[None, :]
        cc = t[:, None] * 2 + (i & 1)[None, :]
        return ar, ac, br, bc, cr, cc

    def _frag(self, names, ty):
        """register fragment -> (W, 32, n) float64 elements in PTX fragment order"""
        W = self.T // 32
        cols = []
        for n in names:
            v = self.get(n).reshape(W, 32)
            if ty == 'e4m3':
                for k in range(4):
                    cols.append(E4M3[((v >> U64(8 * k)) & U64(0xff)).astype(np.int64)])
            elif ty == 'f16':
                for k in range(2):
                    cols.append(bits_to_f16((v >> U64(16 * k)) & U64(0xffff)).astype(F64))
            elif ty == 'f32':
                cols.append(bits_to_f32(v).astype(F64))
            else:
                raise NotImplementedError(f'mma fragment type {ty}')
        return np.stack(cols, axis=-1)

    def op_mma(self, st, parts, a, mask):
        """mma.sync.aligned.SHAPE.row.col.DTYPE.ATYPE.BTYPE.CTYPE (m16n8k32 e4m3; m16n8k16 f16)"""
        shape = parts[3]
        dtype, atype, btype, ctype = parts[6], parts[7], parts[8], parts[9]
        if shape == 'm16n8k32' and atype == 'e4m3' and btype == 'e4m3':
            k = 32
        elif shape == 'm16n8k16' and atype == 'f16' and btype == 'f16':
            k = 16
        else:
            raise NotImplementedError(st.text)
        d_names, a_names, b_names, c_names = (self.vec(x) for x in a)
        W = self.T // 32
        ar, ac, br, bc, cr, cc = self._mma_maps(k)
        Av = self._frag(a_names, atype)
        Bv = self._frag(b_names, btype)
        Cv = self._frag(c_names, ctype)
        Am = np.zeros((W, 16, k))
        Bm = np.zeros((W, k, 8))
        Cm = np.zeros((W, 16, 8))
        Am[:, ar, ac] = Av
        Bm[:, br, bc] = Bv
        Cm[:, cr, cc] = Cv

        def rnd(x):
            return x.astype(F16 if dtype == 'f16' else F32).astype(F64)
        if MMA_MODE == 'split16' and k == 32:
            d1 = rnd(np.matmul(Am[:, :, :16], Bm[:, :16]) + Cm)
            Dm = rnd(np.matmul(Am[:, :, 16:], Bm[:, 16:]) + d1)
        else:
            Dm = rnd(np.matmul(Am, Bm) + Cm)
        if self.mma_hook:
            pa = [self.prov.get(n) for n in a_names]
            pb = [self.prov.get(n) for n in b_names]
            for w in range(W):
                self.mma_hook(self, st, w, Am[w], Bm[w], Cm[w], Dm[w], pa, pb)
        Dv = Dm[:, cr, cc]      # (W, 32, 4)
        if dtype == 'f16':
            bits = Dv.astype(F16).view(np.uint16).astype(U64)
            outs = [bits[..., 2 * r] | (bits[..., 2 * r + 1] << U64(16)) for r in range(2)]
        else:
            bits = Dv.astype(F32).view(np.uint32).astype(U64)
            outs = [bits[..., i] for i in range(4)]
        for n, vals in zip(d_names, outs):
            self.put(n, vals.reshape(self.T), mask)

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

    # --- textures and surfaces ----------------------------------------------------------
    def op_tex(self, st, parts, a, mask):
        """tex.{base,level}.2d.v4.DTYPE.CTYPE {d0..d3}, [h, {x, y}] (, lod)
        CTYPE s32: unnormalised integer texel coordinates, point fetch. CTYPE f32: normalised, bilinear
        when the texture is linear (level 0 only)."""
        dtype, ctype = parts[4], parts[5]
        names = self.vec(a[0])
        hname, coord = a[1].strip()[1:-1].split(',', 1)
        h = self.get(hname.strip())
        xs, ys = self.vec(coord.strip())
        if ctype == 's32':
            x = self.typed(xs, 's32').astype(F64)
            y = self.typed(ys, 's32').astype(F64)
        else:
            x = bits_to_f32(self.get(xs)).astype(F64)
            y = bits_to_f32(self.get(ys)).astype(F64)
        if parts[1] == 'level' and len(a) > 2:
            lod = bits_to_f32(self.get(a[2])).astype(F64)
            if np.any(lod[mask] != 0.0):
                raise NotImplementedError('tex with LOD != 0 (captured textures are single-level)')
        out = np.zeros((self.T, 4), F64)
        for hv in np.unique(h[mask]):
            tex = self.resource(hv, Texture)
            sel = mask & (h == hv)
            if ctype == 's32':
                out[sel] = tex.fetch(x[sel].astype(np.int64), y[sel].astype(np.int64))
            else:
                out[sel] = tex.sample(x[sel], y[sel])
        for i, n in enumerate(names):
            bits = f32_to_bits(out[:, i]) if dtype == 'f32' else f16_to_bits(out[:, i])
            self.put(n, bits, mask)

    def op_sust(self, st, parts, a, mask):
        """sust.{b,p}.2d.vN.bBITS.zero [h, {x, y}], {v0..}"""
        nch = int(next(p[1:] for p in parts if re.fullmatch(r'v\d', p)))
        bits = int(re.sub(r'\D', '', parts[-2]))
        hname, coord = a[0].strip()[1:-1].split(',', 1)
        h = self.get(hname.strip())
        xs, ys = self.vec(coord.strip())
        x = self.typed(xs, 's32')
        y = self.typed(ys, 's32')
        data = self.vec(a[1])
        if len(data) != nch:
            raise ValueError(f'sust v{nch} with {len(data)} values')
        vals = [self.get(n) & U64(MASK[bits]) for n in data]
        for hv in np.unique(h[mask]):
            surf = self.resource(hv, Surface)
            sel = mask & (h == hv)
            if parts[1] == 'p':
                # formatted store: x counts texels and the f32 values are converted to the surface's format
                if bits != 32 or surf.bits not in (9, 16):
                    raise NotImplementedError(f'sust.p of b{bits} values to a {surf.bits}-bit surface')
                f = [bits_to_f32(v[sel]).astype(F64) for v in vals]
                if surf.bits == 16:
                    # ZLUDA's lowering of the formatted store truncates (round toward zero), as the capture shows
                    conv = []
                    for c in f:
                        h = c.astype(F16)
                        over = np.abs(h.astype(F64)) > np.abs(c)
                        hb = f16_to_bits(h)
                        conv.append(np.where(over, hb - U64(1), hb))
                else:   # SNORM8: round(clamp(v, -1, 1) * 127)
                    conv = [np.rint(np.clip(np.nan_to_num(c), -1.0, 1.0) * 127.0).astype(np.int64).astype(U64) & U64(0xff) for c in f]
                surf.write(x[sel], y[sel], conv, surf.bits)
            else:
                # binary store: x counts bytes
                xb = x[sel]
                if np.any(xb % surf.texel_bytes):
                    raise ValueError('sust.b at an x that is not a texel boundary')
                surf.write(xb // surf.texel_bytes, y[sel], [v[sel] for v in vals], bits)


class Stmt_like:
    def __init__(self, st, op, args):
        self.op, self.args, self.text = op, args, st.text + f'  [as {op}]'
        for k in Stmt.__slots__:
            if k not in ('op', 'args', 'text'):
                setattr(self, k, getattr(st, k, None))


def run_block(ptx, kernel, dump, ctaid, hooks=(), mma_hook=None, limit=None, resources=None):
    """Run one block of KERNEL from PTX against a replay directory. Returns (block, memory, steps)."""
    stmts, types, shared = parse_kernel(ptx, kernel)
    mem = Memory(dump)
    launch = [l for l in open(dump + '/manifest.txt') if l.startswith('launch')][0].split()
    grid = tuple(int(x) for x in launch[1:4])
    block = tuple(int(x) for x in launch[4:7])
    b = Block(stmts, types, shared, mem, kernel + '_param_0', block, grid, ctaid)
    b.hooks = list(hooks)
    b.mma_hook = mma_hook
    b.resources = dict(resources or {})
    steps = b.run(limit)
    return b, mem, steps


if __name__ == '__main__':
    import time
    ptx, kernel, dump = sys.argv[1:4]
    cx, cy = (int(v) for v in sys.argv[4].split(',')) if len(sys.argv) > 4 else (0, 0)
    t0 = time.time()
    b, mem, steps = run_block(ptx, kernel, dump, (cx, cy, 0))
    print(f'block ({cx},{cy}): {steps} statements in {time.time() - t0:.1f}s, {len(mem.stores)} store groups')
