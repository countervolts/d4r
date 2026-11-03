#!/usr/bin/env python3
"""nvxrt T0: translate one PTX kernel entry into a Vulkan GLSL compute shader with PTX semantics.

usage: ptxc.py MODULE.ptx ENTRY OUT_PREFIX      -> OUT_PREFIX.comp, OUT_PREFIX.meta
       ptxc.py --hash MODULE.ptx                 -> print the kernel-cache identity of the module

Design (see PROGRESS.md D1-D4):
 * registers become typed GLSL locals; every float operation is `precise`; fp32 denormals are flushed
   (all f32 arithmetic in DLSS-G is .ftz) and fp16 denormals preserved via float-controls execution modes;
 * control flow is structurized from the dominator tree (reducible CFGs, Ramsey 2022) into nested
   do{}while(false) blocks and while(true) loops with a jump-target variable for multi-level exits;
 * shared memory is one uint array (sub-word stores via atomics), local memory a private uint array,
   global memory goes through buffer references (CUDA pointers are buffer device addresses);
 * parameters live in push constants; texture/surface handles index the layer's bindless arrays;
 * mma.sync runs one RDNA WMMA per NVIDIA instruction, fragments staged in per-subgroup LDS scratch.
"""
import hashlib
import os
import re
import sys

class Err(Exception):
    pass

# ------------------------------------------------------------------------------------------- normalisation
def normalize_ptx(text):
    out = []
    i, n = 0, len(text)
    pending = False
    while i < n:
        c = text[i]
        if c == '/' and i + 1 < n and text[i + 1] == '/':
            while i < n and text[i] != '\n':
                i += 1
            pending = True
            continue
        if c == '/' and i + 1 < n and text[i + 1] == '*':
            i += 2
            while i + 1 < n and not (text[i] == '*' and text[i + 1] == '/'):
                i += 1
            i += 2
            pending = True
            continue
        if c in '\r\n \t\0':
            pending = True
            i += 1
            continue
        if pending and out:
            out.append(' ')
        pending = False
        out.append(c)
        i += 1
    return ''.join(out)

def module_hash(text):
    return hashlib.sha256(normalize_ptx(text).encode('latin-1')).hexdigest()

# ------------------------------------------------------------------------------------------- parsing
def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)

def split_operands(s):
    ops, depth, cur = [], 0, ''
    for ch in s:
        if ch in '{[':
            depth += 1
        elif ch in '}]':
            depth -= 1
        if ch == ',' and depth == 0:
            ops.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        ops.append(cur.strip())
    return ops

TYPE_SIZES = {'b8': 1, 'u8': 1, 's8': 1, 'b16': 2, 'u16': 2, 's16': 2, 'f16': 2, 'b32': 4, 'u32': 4, 's32': 4,
              'f32': 4, 'b64': 8, 'u64': 8, 's64': 8, 'f64': 8, 'f16x2': 4, 'pred': 1}

class Instr:
    __slots__ = ('pred', 'neg', 'op', 'mods', 'args', 'text')
    def __init__(self, pred, neg, op, mods, args, text):
        self.pred, self.neg, self.op, self.mods, self.args, self.text = pred, neg, op, mods, args, text

class Entry:
    pass

def find_block(text, start):
    depth = 0
    k = start
    while True:
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return k
        k += 1

def module_shared(text):
    """(name, align, size) of the .shared declarations at module scope (brace depth 0)"""
    out, depth = [], 0
    for line in text.split('\n'):
        st = line.strip()
        if depth == 0 and st.startswith('.shared'):
            sm = re.match(r'\.shared\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+([\w$]+)(?:\[(\d+)\])?', st)
            if sm:
                align = int(sm.group(1)) if sm.group(1) else TYPE_SIZES[sm.group(2)]
                out.append((sm.group(3), align, TYPE_SIZES[sm.group(2)] * (int(sm.group(4)) if sm.group(4) else 1)))
        depth += line.count('{') - line.count('}')
    return out

def parse_entry(ptx, name):
    text = strip_comments(ptx)
    m = re.search(r'\.entry\s+' + re.escape(name) + r'\s*\(', text)
    if not m:
        raise Err('entry %s not found' % name)
    k = m.end()
    depth = 1
    j = k
    while depth:
        if text[j] == '(':
            depth += 1
        elif text[j] == ')':
            depth -= 1
        j += 1
    plist = text[k:j - 1]
    e = Entry()
    e.name = name
    e.params = []   # (name, offset, size)
    off = 0
    for p in [x.strip() for x in plist.split(',') if x.strip()]:
        pm = re.match(r'\.param\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+([\w$]+)(?:\[(\d+)\])?', p)
        if not pm:
            raise Err('param ' + p)
        align = int(pm.group(1)) if pm.group(1) else TYPE_SIZES[pm.group(2)]
        esz = TYPE_SIZES[pm.group(2)]
        cnt = int(pm.group(4)) if pm.group(4) else 1
        size = esz * cnt
        off = (off + align - 1) // align * align
        e.params.append((pm.group(3), off, size))
        off += size
    e.param_size = off
    # blocks above 248 bytes (push constants) are read through a buffer address (NVX_BIG_PARAMS, meta big_params)
    if off > 4096:
        raise Err('parameter block %d bytes > 4096' % off)
    body_start = text.index('{', j)
    head = text[j:body_start]
    mt = re.search(r'\.maxntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', head) or re.search(r'\.reqntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', head)
    e.maxntid = [int(x) for x in mt.groups()] if mt else None
    body_end = find_block(text, body_start)
    body = text[body_start + 1:body_end]
    # module-scope shared declarations are added after the body is parsed (module_shared)
    e.regs = {}      # name -> ptx type
    e.shared = []    # (name, align, size)
    e.local = []
    e.stmts = []     # ('label', name) | ('instr', Instr)
    scope_stack = [{}]
    uid = [0]
    def resolve(tok):
        for sc in reversed(scope_stack):
            if tok in sc:
                return sc[tok]
        return tok
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c.isspace():
            i += 1
            continue
        if c == '{':
            scope_stack.append({})
            i += 1
            continue
        if c == '}':
            scope_stack.pop()
            i += 1
            continue
        lm = re.match(r'(\$?[A-Za-z_][\w$]*):', body[i:])
        if lm:
            e.stmts.append(('label', lm.group(1)))
            i += lm.end()
            continue
        if re.match(r'\.(loc|file)\s', body[i:i + 6]):
            # debug directives end at the line end (no ';')
            nl = body.find('\n', i)
            i = n if nl < 0 else nl + 1
            continue
        jx = body.index(';', i)
        s = ' '.join(body[i:jx].split())
        i = jx + 1
        if s.startswith('.reg'):
            rm = re.match(r'\.reg\s+\.(\w+)\s+(.*)$', s)
            typ = rm.group(1)
            for d in split_operands(rm.group(2)):
                d = d.strip()
                am = re.match(r'(%?[\w$]+)<(\d+)>$', d)
                names = ['%s%d' % (am.group(1), q) for q in range(int(am.group(2)))] if am else [d]
                for nm in names:
                    if len(scope_stack) > 1:
                        uid[0] += 1
                        new = '%s_s%d' % (nm, uid[0])
                        scope_stack[-1][nm] = new
                        e.regs[new] = typ
                    else:
                        e.regs[nm] = typ
            continue
        if s.startswith('.shared') or s.startswith('.local'):
            sm = re.match(r'\.(shared|local)\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+([\w$]+)(?:\[(\d+)\])?', s)
            if not sm:
                raise Err('decl ' + s)
            align = int(sm.group(2)) if sm.group(2) else TYPE_SIZES[sm.group(3)]
            size = TYPE_SIZES[sm.group(3)] * (int(sm.group(5)) if sm.group(5) else 1)
            (e.shared if sm.group(1) == 'shared' else e.local).append((sm.group(4), align, size))
            continue
        if s.startswith('.pragma') or s.startswith('.loc') or s.startswith('.file'):
            continue
        if s.startswith('.'):
            raise Err('directive ' + s)
        pm = re.match(r'@(!?)(%[\w$]+)\s+(.*)$', s)
        pred, neg = None, False
        if pm:
            neg = pm.group(1) == '!'
            pred = resolve(pm.group(2))
            s = pm.group(3)
        parts = s.split(None, 1)
        op = parts[0]
        args = split_operands(parts[1]) if len(parts) > 1 else []
        # rename scoped registers inside operands
        if len(scope_stack) > 1:
            def ren(mo):
                return resolve(mo.group(0))
            args = [re.sub(r'(?<![\w$.])%?[A-Za-z_$][\w$]*', ren, a) for a in args]
        mods = op.split('.')
        e.stmts.append(('instr', Instr(pred, neg, mods[0], mods[1:], args, s)))
    # module-scope .shared arrays (declared outside any entry, e.g. DLSS-G 3.x) belong to every kernel that uses them
    used = set(re.findall(r'[A-Za-z_$][\w$]*', body))
    for nm, al, sz in module_shared(text):
        if nm in used and nm not in {x[0] for x in e.shared}:
            e.shared.append((nm, al, sz))
    # memory layout
    off = 0
    e.shared_off = {}
    for nm, al, sz in e.shared:
        off = (off + al - 1) // al * al
        e.shared_off[nm] = off
        off += sz
    e.shared_bytes = (off + 15) // 16 * 16
    off = 0
    e.local_off = {}
    for nm, al, sz in e.local:
        off = (off + al - 1) // al * al
        e.local_off[nm] = off
        off += sz
    e.local_bytes = (off + 15) // 16 * 16
    return e

# ------------------------------------------------------------------------------------------- CFG
class Block:
    def __init__(self, idx):
        self.idx = idx
        self.label = None
        self.instrs = []
        self.term = None     # ('ret',) | ('br', tgt) | ('cbr', pred, neg, tgt, fall) | ('fall', nxt)
        self.succ = []

def build_cfg(e):
    blocks = []
    cur = Block(0)
    blocks.append(cur)
    labels = {}
    def new_block():
        b = Block(len(blocks))
        blocks.append(b)
        return b
    for kind, x in e.stmts:
        if kind == 'label':
            if cur.instrs or cur.label is not None or cur.term is not None:
                if cur.term is None:
                    cur.term = ('fall', len(blocks))
                cur = new_block()
            cur.label = x
            labels[x] = cur.idx
            continue
        ins = x
        if cur.term is not None:
            cur = new_block()
        if ins.op == 'bra':
            tgt = ins.args[0]
            if ins.pred is None:
                cur.term = ('br', tgt)
            else:
                cur.term = ('cbr', ins.pred, ins.neg, tgt, len(blocks))
            continue
        if ins.op in ('ret', 'exit'):
            if ins.pred is None:
                cur.term = ('ret',)
            else:
                cur.term = ('cbr_ret', ins.pred, ins.neg, len(blocks))
            continue
        cur.instrs.append(ins)
    if cur.term is None:
        cur.term = ('ret',)
    # resolve label targets, fallthroughs past the end
    nb = len(blocks)
    for b in blocks:
        t = b.term
        if t[0] == 'br':
            b.term = ('br', labels[t[1]])
        elif t[0] == 'cbr':
            if t[4] >= nb:
                raise Err('conditional branch falls off the end')
            b.term = ('cbr', t[1], t[2], labels[t[3]], t[4])
        elif t[0] == 'fall' and t[1] >= nb:
            b.term = ('ret',)
        elif t[0] == 'cbr_ret' and t[3] >= nb:
            b.term = ('cbr_retend', t[1], t[2])
    for b in blocks:
        t = b.term
        if t[0] == 'br':
            b.succ = [t[1]]
        elif t[0] == 'cbr':
            b.succ = [t[3], t[4]]
        elif t[0] == 'fall':
            b.succ = [t[1]]
        elif t[0] == 'cbr_ret':
            b.succ = [t[3]]
        else:
            b.succ = []
    return blocks

def dominators(blocks):
    # reverse postorder
    order, seen = [], set()
    def dfs(v):
        stack = [(v, iter(blocks[v].succ))]
        seen.add(v)
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                order.append(node)
                stack.pop()
            elif nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(blocks[nxt].succ)))
    dfs(0)
    rpo = list(reversed(order))
    num = {v: i for i, v in enumerate(rpo)}
    preds = {v: [] for v in rpo}
    for v in rpo:
        for s in blocks[v].succ:
            preds[s].append(v)
    idom = {rpo[0]: rpo[0]}
    def intersect(a, b):
        while a != b:
            while num[a] > num[b]:
                a = idom[a]
            while num[b] > num[a]:
                b = idom[b]
        return a
    changed = True
    while changed:
        changed = False
        for v in rpo[1:]:
            ps = [p for p in preds[v] if p in idom]
            new = ps[0]
            for p in ps[1:]:
                new = intersect(p, new)
            if idom.get(v) != new:
                idom[v] = new
                changed = True
    return rpo, num, preds, idom

def dominates(idom, a, b):
    while True:
        if a == b:
            return True
        if idom[b] == b:
            return False
        b = idom[b]

# ------------------------------------------------------------------------------------------- emitter
GT = {'pred': 'bool', 'b16': 'uint16_t', 'u16': 'uint16_t', 's16': 'uint16_t', 'f16': 'uint16_t',
      'b32': 'uint', 'u32': 'uint', 's32': 'uint', 'f32': 'float', 'f16x2': 'uint',
      'b64': 'uint64_t', 'u64': 'uint64_t', 's64': 'uint64_t', 'f64': 'uint64_t', 'b8': 'uint16_t', 'u8': 'uint16_t', 's8': 'uint16_t'}
# GLSL value type for a PTX operand type
VT = {'u32': 'uint', 'b32': 'uint', 's32': 'int', 'f32': 'float', 'u16': 'uint16_t', 'b16': 'uint16_t',
      's16': 'int16_t', 'f16': 'float16_t', 'u64': 'uint64_t', 'b64': 'uint64_t', 's64': 'int64_t',
      'pred': 'bool', 'f64': 'uint64_t', 'u8': 'uint', 's8': 'int', 'b8': 'uint', 'f16x2': 'uint'}
# f64 registers hold the IEEE double's bits (uint64_t): DLSS-G only loads doubles and converts them to f32
# (nvx_d2f); no Float64 capability is needed. Any f64 arithmetic or comparison is refused (Err).

SREG = {'%tid.x': 'gl_LocalInvocationID.x', '%tid.y': 'gl_LocalInvocationID.y', '%tid.z': 'gl_LocalInvocationID.z',
        '%ntid.x': 'gl_WorkGroupSize.x', '%ntid.y': 'gl_WorkGroupSize.y', '%ntid.z': 'gl_WorkGroupSize.z',
        '%ctaid.x': 'gl_WorkGroupID.x', '%ctaid.y': 'gl_WorkGroupID.y', '%ctaid.z': 'gl_WorkGroupID.z',
        '%nctaid.x': 'gl_NumWorkGroups.x', '%nctaid.y': 'gl_NumWorkGroups.y', '%nctaid.z': 'gl_NumWorkGroups.z',
        '%laneid': 'gl_SubgroupInvocationID', '%warpid': 'gl_SubgroupID', '%nwarpid': 'gl_NumSubgroups'}

def f32_imm(tok):
    if re.match(r'0[fF][0-9a-fA-F]{8}$', tok):
        return 'uintBitsToFloat(0x%sU)' % tok[2:]
    if re.match(r'0[dD][0-9a-fA-F]{16}$', tok):
        import struct
        v = struct.unpack('>d', bytes.fromhex(tok[2:]))[0]
        return repr(float(v))
    return None

def int_imm(tok):
    t = tok.strip()
    neg = t.startswith('-')
    if neg:
        t = t[1:]
    if re.match(r'0[xX][0-9a-fA-F]+[uU]?$', t):
        v = int(t.rstrip('uU'), 16)
    elif re.match(r'0[0-7]+[uU]?$', t) and len(t) > 1:
        v = int(t.rstrip('uU'), 8)
    elif re.match(r'\d+[uU]?$', t):
        v = int(t.rstrip('uU'))
    else:
        return None
    return -v if neg else v

class Emitter:
    def __init__(self, e, opts):
        self.e = e
        self.opts = opts
        self.out = []
        self.ind = 1
        self.uses_mma = False
        self.uses_smv = False
        self.uses_scr_a = False
        self.uses_local = e.local_bytes > 0
        self.param_off = {nm: off for nm, off, sz in e.params}
        self.jt_used = False
        self.tmp = 0
        self.chain_pairs = []     # list of (d0, d1) register pairs kept in WMMA accumulators
        self.pair_m1, self.pair_hi, self.phys, self.n_paired = {}, {}, {}, 0
        self.reg_pair = {}        # register -> chain pair index
        self.last_a = None        # A-operand tuple cached in ma_c (straight-line reuse)
        self.ldm_src = {}         # register -> (address temp, matrix j, block) of the ldmatrix.x4 that last wrote it
        self.cur_block = None
        self.idom = None
        self.fn_decls = []        # declarations hoisted to the top of main (address temps used across blocks)
        self.gld_src = {}         # register -> (64-bit base register, byte offset) of the ld.global.u32 that wrote it
        self.quad_cache = {}      # base register -> (shuffled quad-base address var, quad-contiguity flag var)
        # WMMA register layout of RADV's 16x16 f16 cooperative matrices (wave32):
        #   gfx12 (RDNA4): A/B lane l holds row/column l&15, K = 8*(l>>4)+i for its 8 elements
        #   gfx11 (RDNA3): A/B lane l holds row/column l&15, K = i for its 16 elements (lanes 16-31 repeat 0-15)
        # On gfx11 every operand is built either through the subgroup's LDS tile and coopMatLoad (any layout) or
        # by the shuffles below with gfx11's element map; the fused loads, which read 16-byte K halves in gfx12's
        # layout, are off.
        self.layout = opts.get('layout', 'gfx12')
        if self.layout not in ('gfx12', 'gfx11'):
            raise Err('unknown WMMA layout ' + self.layout)
        gfx11 = self.layout == 'gfx11'
        self.fuse = not opts.get('no_fuse')
        self.fuse_a = self.fuse and not opts.get('no_fuse_a') and not gfx11
        self.lds_xpose = self.fuse and not opts.get('no_xpose')
        self.f16acc = bool(opts.get('f16acc')) and not gfx11
        self.xpose_load = not opts.get('no_xpose_load') or gfx11
        self.need_bzero = False
        # B-operand fusion (weights loaded in WMMA layout) is correct but slower on RDNA4: the fused loads sit behind
        # a uniform run-time check and cannot be hoisted, so their latency lands on every mma. Off unless asked for.
        self.fuse_b = self.fuse and opts.get('fuse_b', False) and not opts.get('no_fuse_b') and not gfx11
        self.analyze_mma_chains()
        self.global_ptrs = self.analyze_global_ptrs()
        self.addr32 = set() if opts.get('no_addr32') else self.analyze_addr32()
        self.read_regs = set()
        for k_, ins_ in self.e.stmts:
            if k_ == 'instr':
                self.read_regs.update(self.reg_rw(ins_)[0])

    def analyze_addr32(self):
        """Registers written by mul.wide.u32 whose value only ever ends up as part of a global memory address
        (through add.s64/add.u64). Their 64-bit product equals the 32-bit product zero-extended for every
        address a valid program can touch (the byte offset from a buffer pointer stays below 4 GiB), and the
        zero-extended 32-bit form lets the backend use an SGPR base + 32-bit VGPR offset for the access."""
        ins_list = [ins for k, ins in self.e.stmts if k == 'instr']
        defs, uses = {}, {}
        for ins in ins_list:
            r, w = self.reg_rw(ins)
            for x in w:
                defs.setdefault(x, []).append(ins)
            for x in r:
                uses.setdefault(x, []).append(ins)
        def mem_addr_only(ins, reg):
            # reg appears only inside the [address] operand of a global memory access
            if ins.op not in ('ld', 'st', 'atom', 'red') or 'global' not in ins.mods:
                return False
            if ins.pred == reg:
                return False
            addr_tok = ins.args[0] if ins.op in ('st', 'red') else ins.args[1]
            if not addr_tok.startswith('['):
                return False
            base, _ = self.addr(addr_tok)
            if base != reg:
                return False
            others = ins.args[1:] if ins.op in ('st', 'red') else [ins.args[0]] + list(ins.args[2:])
            return not any(reg in re.findall(r'%?[A-Za-z_$][\w$]*', t) for t in others)
        memo = {}
        def only_addr(reg, depth=0):
            if reg in memo:
                return memo[reg]
            memo[reg] = False
            ok = len(defs.get(reg, [])) == 1 and depth < 6 and bool(uses.get(reg))
            for u in uses.get(reg, []) if ok else []:
                if mem_addr_only(u, reg):
                    continue
                if u.op == 'add' and u.pred is None and ('s64' in u.mods or 'u64' in u.mods) and u.args[0] != reg \
                        and reg in (u.args[1].strip(), u.args[2].strip()) and only_addr(u.args[0].strip(), depth + 1):
                    continue
                ok = False
                break
            memo[reg] = ok
            return ok
        out = set()
        for ins in ins_list:
            if ins.op == 'mul' and 'wide' in ins.mods and 'u32' in ins.mods and ins.pred is None:
                d = ins.args[0].strip()
                if len(defs.get(d, [])) == 1 and uses.get(d) and all(
                        u.op == 'add' and u.pred is None and ('s64' in u.mods or 'u64' in u.mods) and u.args[0].strip() != d
                        and only_addr(u.args[0].strip(), 1) for u in uses[d]):
                    out.add(d)
        return out

    def line(self, s):
        self.out.append('    ' * self.ind + s)

    def analyze_mma_chains(self):
        if self.opts.get('no_chain'):
            return
        mmas = [ins for k, ins in self.e.stmts if k == 'instr' and ins.op == 'mma']
        if mmas:
            self.uses_mma = True
        pairs = []
        for ins in mmas:
            D = tuple(x.strip() for x in ins.args[0][1:-1].split(','))
            C = tuple(x.strip() for x in ins.args[3][1:-1].split(','))
            for P in (D, C):
                if len(P) == 2 and P not in pairs:
                    pairs.append(P)
        dpairs = set(tuple(x.strip() for x in ins.args[0][1:-1].split(',')) for ins in mmas)
        owner = {}
        bad = set()
        for P in pairs:
            for r in P:
                if r in owner and owner[r] != P:
                    bad.add(P)
                    bad.add(owner[r])
                owner[r] = P
        for P in pairs:
            if P in dpairs and P not in bad and P[0] != P[1]:
                idx = len(self.chain_pairs)
                self.chain_pairs.append(P)
                for r in P:
                    self.reg_pair[r] = idx
        if self.fuse and self.lds_xpose and self.xpose_load and not self.f16acc and not self.opts.get('no_pair'):
            self.analyze_mma_pairs()

    def analyze_global_ptrs(self):
        """64-bit registers that provably hold global-memory addresses: u64 kernel parameters, cvta.to.global results,
        moves and integer offsets of those, and loads from a local array whose 64-bit stores all store such values
        (DLSS-G 310.7 picks one of four buffer parameters through a local array and loads through the generic result).
        Generic ld/st through these registers are global accesses; any other generic access stays untranslated."""
        glob = set()
        local_ld = []          # (dst, ins) of 64-bit local loads
        local_st_srcs = []     # sources of 64-bit local stores
        for _ in range(4):     # fixpoint over straight-line facts (order-independent)
            before = len(glob)
            for kind, ins in self.e.stmts:
                if kind != 'instr' or ins.pred is not None and ins.op not in ('ld',):
                    continue
                op, ms, a = ins.op, set(ins.mods), ins.args
                is64 = bool({'u64', 'b64', 's64'} & ms)
                if op == 'ld' and 'param' in ms and is64 and not a[0].startswith('{'):
                    glob.add(a[0])
                elif op == 'cvta' and 'to' in ms and 'global' in ms:
                    glob.add(a[0])
                elif op == 'mov' and is64 and a[1] in glob:
                    glob.add(a[0])
                elif op == 'add' and is64 and len(a) == 3 and ((a[1] in glob) != (a[2] in glob)):
                    glob.add(a[0])
                elif op == 'st' and 'local' in ms and is64:
                    srcs = [x.strip() for x in a[1][1:-1].split(',')] if a[1].startswith('{') else [a[1]]
                    local_st_srcs.extend(srcs)
                elif op == 'ld' and 'local' in ms and is64 and not a[0].startswith('{'):
                    local_ld.append(a[0])
            if local_st_srcs and all(x in glob for x in local_st_srcs):
                glob.update(local_ld)
            local_ld, local_st_srcs = [], []
            if len(glob) == before:
                break
        return glob

    def generic_is_global(self, addr_tok):
        base, _ = self.addr(addr_tok)
        return base in self.global_ptrs

    def mma_regs(self, ins):
        f = lambda t: tuple(x.strip() for x in t[1:-1].split(','))
        return f(ins.args[0]), f(ins.args[1]), f(ins.args[2]), f(ins.args[3])

    def chain_of(self, regs):
        i = self.reg_pair.get(regs[0])
        return i if i is not None and self.chain_pairs[i] == tuple(regs) else None

    def analyze_mma_pairs(self):
        """Two consecutive mma.m16n8k16 with the same A and accumulator chains always updated together become one
        16x16x16 WMMA: chain lo in columns 0-7, chain hi in columns 8-15 (columns are independent, so the result is
        identical). M1 is evaluated at M2's position; nothing in between may touch M1's operands or result."""
        stmts = self.e.stmts
        cands = []
        i = 0
        n = len(stmts)
        while i < n:
            k, ins = stmts[i]
            if k != 'instr' or ins.op != 'mma' or ins.pred is not None or 'm16n8k16' not in ins.mods:
                i += 1
                continue
            D1, A1, B1, C1 = self.mma_regs(ins)
            j = i + 1
            ok = True
            touched_w, touched_r = set(), set()
            while j < n:
                kj, ij = stmts[j]
                if kj != 'instr':
                    ok = False
                    break
                if ij.op == 'mma':
                    break
                r, w = self.reg_rw(ij)
                touched_w |= set(w)
                touched_r |= set(r)
                if ij.op in ('bar', 'barrier', 'ret', 'exit', 'bra'):
                    ok = False
                    break
                j += 1
            if not ok or j >= n:
                i += 1
                continue
            k2, m2 = stmts[j]
            if m2.pred is not None or 'm16n8k16' not in m2.mods or m2.mods != ins.mods:
                i += 1
                continue
            D2, A2, B2, C2 = self.mma_regs(m2)
            c1, d1, c2, d2 = self.chain_of(C1), self.chain_of(D1), self.chain_of(C2), self.chain_of(D2)
            # C operands: both accumulator chains (paired like D) or both plain registers (staged through LDS)
            cok = (c1 is None and c2 is None) or (c1 is not None and c2 is not None and c1 != c2)
            if A2 != A1 or None in (d1, d2) or not cok or len({d1, d2}) != 2 or \
                    touched_w & (set(A1) | set(B1) | set(C1) | set(D1)) or touched_r & set(D1) or \
                    set(D1) & (set(A2) | set(B2) | set(C2)) or set(D2) & set(D1):
                i += 1
                continue
            cands.append((i, j, c1, d1, c2, d2))
            i = j + 1
        # partner relation, with every mma touching a paired chain required to be a candidate pair
        while True:
            part = {}
            badc = set()
            for (_, _, c1, d1, c2, d2) in cands:
                for lo, hi in ((c1, c2), (d1, d2)):
                    if lo is None:
                        continue
                    for x, y, h in ((lo, hi, 0), (hi, lo, 1)):
                        if x in part and part[x] != (y, h):
                            badc.add(x)
                        part[x] = (y, h)
            paired_idx = set()
            for (i1, i2, *_rest) in cands:
                paired_idx.add(i1)
                paired_idx.add(i2)
            for idx, (k, ins) in enumerate(stmts):
                if k == 'instr' and ins.op == 'mma' and idx not in paired_idx:
                    D, A, B, C = self.mma_regs(ins)
                    for ch in (self.chain_of(C), self.chain_of(D)):
                        if ch is not None and ch in part:
                            badc.add(ch)
            badc |= set(part[x][0] for x in badc if x in part)
            keep = [c for c in cands if not ((set(c[2:]) - {None}) & badc)]
            if len(keep) == len(cands):
                break
            cands = keep
        self.pair_m1 = {}     # id(M1) -> True (deferred), id(M2) -> M1 instr
        self.pair_hi = {}     # lo chain -> hi chain
        self.phys = {}        # chain -> lo chain of its pair
        for (i1, i2, c1, d1, c2, d2) in cands:
            self.pair_m1[id(stmts[i1][1])] = True
            self.pair_m1[id(stmts[i2][1])] = stmts[i1][1]
            for lo, hi in ((c1, c2), (d1, d2)):
                if lo is None:
                    continue
                self.pair_hi[lo] = hi
                self.phys[lo] = lo
                self.phys[hi] = lo
        self.n_paired = len(cands)

    def ph(self, p):
        return self.phys.get(p, p)

    def materialize_code(self, p):
        """GLSL that writes the NVIDIA fragment registers of chain pair p from its accumulator acc_p (through the
        subgroup's fp16 C/D staging tile: the accumulator holds fp16-exact values, so the conversion is exact)"""
        d0, d1 = self.chain_pairs[p]
        acc = 'acc_%d' % p if self.f16acc else \
            'coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc_%d)' % p
        code = ('{ uint g_ = gl_SubgroupInvocationID >> 2, t_ = gl_SubgroupInvocationID & 3u; '
                'uint sb_ = NVX_SCRATCH_BASE + gl_SubgroupID * NVX_SSTRIDE; subgroupBarrier(); '
                'coopMatStore(%s, smh, (sb_ + NVX_SC) * 2u, 16u, gl_CooperativeMatrixLayoutRowMajor); subgroupBarrier(); '
                '%s = smem[sb_ + NVX_SC + g_ * 8u + t_]; %s = smem[sb_ + NVX_SC + (g_ + 8u) * 8u + t_]; '
                % (acc, self.rname(d0), self.rname(d1)))
        if p in self.pair_hi:
            h0, h1 = self.chain_pairs[self.pair_hi[p]]
            code += ('%s = smem[sb_ + NVX_SC + g_ * 8u + 4u + t_]; %s = smem[sb_ + NVX_SC + (g_ + 8u) * 8u + 4u + t_]; '
                     % (self.rname(h0), self.rname(h1)))
        return code + 'subgroupBarrier(); }'

    def newtmp(self):
        self.tmp += 1
        return '_t%d' % self.tmp

    # ---------------------------------------------------------------- operand access
    def regtype(self, r):
        if r not in self.e.regs:
            raise Err('unknown register ' + r)
        return self.e.regs[r]

    def conv(self, expr, frm, to):
        """convert a GLSL value of type frm to GLSL type to (bit-preserving for same width)"""
        if frm == to:
            return expr
        key = (frm, to)
        table = {
            ('uint', 'int'): 'int(%s)', ('int', 'uint'): 'uint(%s)',
            ('uint', 'float'): 'uintBitsToFloat(%s)', ('float', 'uint'): 'floatBitsToUint(%s)',
            ('int', 'float'): 'intBitsToFloat(%s)', ('float', 'int'): 'floatBitsToInt(%s)',
            ('uint16_t', 'int16_t'): 'int16_t(%s)', ('int16_t', 'uint16_t'): 'uint16_t(%s)',
            ('uint16_t', 'float16_t'): 'uint16BitsToFloat16(%s)', ('float16_t', 'uint16_t'): 'float16BitsToUint16(%s)',
            ('int16_t', 'float16_t'): 'int16BitsToFloat16(%s)', ('float16_t', 'int16_t'): 'float16BitsToInt16(%s)',
            ('uint64_t', 'int64_t'): 'int64_t(%s)', ('int64_t', 'uint64_t'): 'uint64_t(%s)',
            ('uint64_t', 'double'): 'uint64BitsToDouble(%s)', ('double', 'uint64_t'): 'doubleBitsToUint64(%s)',
            # width changes (used for register-size mismatches PTX allows: low bits / zero extension)
            ('uint', 'uint16_t'): 'uint16_t(%s)', ('uint16_t', 'uint'): 'uint(%s)',
            ('uint', 'int16_t'): 'int16_t(%s)', ('int16_t', 'uint'): 'uint(uint16_t(%s))',
            ('uint64_t', 'uint'): 'uint(%s)', ('uint', 'uint64_t'): 'uint64_t(%s)',
            ('uint64_t', 'int'): 'int(uint(%s))', ('int', 'uint64_t'): 'uint64_t(uint(%s))',
            ('int64_t', 'uint'): 'uint(%s)', ('uint', 'int64_t'): 'int64_t(%s)',
            ('uint16_t', 'int'): 'int(uint(%s))', ('int', 'uint16_t'): 'uint16_t(uint(%s))',
            ('bool', 'uint'): 'uint(%s)', ('uint', 'bool'): '((%s) != 0u)',
            ('float', 'uint16_t'): 'uint16_t(floatBitsToUint(%s))', ('uint16_t', 'float'): 'uintBitsToFloat(uint(%s))',
            ('float16_t', 'uint'): 'uint(float16BitsToUint16(%s))', ('uint', 'float16_t'): 'uint16BitsToFloat16(uint16_t(%s))',
            ('float', 'int16_t'): 'int16_t(floatBitsToUint(%s))',
            ('uint64_t', 'uint16_t'): 'uint16_t(%s)', ('uint16_t', 'uint64_t'): 'uint64_t(%s)',
            ('float', 'uint64_t'): 'uint64_t(floatBitsToUint(%s))', ('uint64_t', 'float'): 'uintBitsToFloat(uint(%s))',
            ('int64_t', 'int'): 'int(%s)', ('int', 'int64_t'): 'int64_t(%s)',
        }
        if key not in table:
            raise Err('no conversion %s -> %s' % key)
        return table[key] % ('(' + expr + ')' if not re.match(r'^[\w.]+$', expr) else expr)

    def val(self, tok, ptype):
        """GLSL expression of operand tok read as PTX type ptype"""
        tok = tok.strip()
        want = VT[ptype]
        if tok in SREG:
            return self.conv(SREG[tok], 'uint', want)
        if tok in self.e.regs or tok.startswith('%'):
            st = GT[self.regtype(tok)]
            return self.conv(self.rname(tok), st, want)
        if tok in self.param_off:
            return self.conv('%du' % self.param_off[tok], 'uint', want) if want != 'uint64_t' else '%dul' % self.param_off[tok]
        if tok in self.e.shared_off:
            o = self.e.shared_off[tok]
            return '%dul' % o if want in ('uint64_t', 'int64_t') else self.conv('%du' % o, 'uint', want)
        if tok in self.e.local_off:
            o = self.e.local_off[tok]
            return '%dul' % o if want in ('uint64_t', 'int64_t') else self.conv('%du' % o, 'uint', want)
        if want == 'float':
            f = f32_imm(tok)
            if f:
                return f
            try:
                return repr(float(tok)) if re.match(r'^-?[\d.]+(e-?\d+)?$', tok) else None
            except ValueError:
                pass
            v = int_imm(tok)
            if v is not None:
                return 'uintBitsToFloat(%du)' % (v & 0xffffffff)
            raise Err('float immediate ' + tok)
        if want == 'float16_t':
            v = int_imm(tok)
            if v is not None:
                return 'uint16BitsToFloat16(uint16_t(%d))' % (v & 0xffff)
            if re.match(r'0[hH][0-9a-fA-F]{4}$', tok):
                return 'uint16BitsToFloat16(uint16_t(0x%s))' % tok[2:]
            raise Err('f16 immediate ' + tok)
        if want == 'double':
            f = f32_imm(tok)
            if f:
                return f
            raise Err('f64 immediate ' + tok)
        v = int_imm(tok)
        if v is None:
            f = f32_imm(tok)
            if f and want == 'uint':
                return '%sU' % ('0x' + tok[2:])
            raise Err('immediate %s as %s' % (tok, ptype))
        if want == 'uint':
            return '%du' % (v & 0xffffffff)
        if want == 'int':
            v &= 0xffffffff
            if v >= 1 << 31:
                v -= 1 << 32
            return '%d' % v if v != -(1 << 31) else 'int(0x80000000u)'
        if want == 'uint16_t':
            return 'uint16_t(%du)' % (v & 0xffff)
        if want == 'int16_t':
            v &= 0xffff
            if v >= 1 << 15:
                v -= 1 << 16
            return 'int16_t(%d)' % v
        if want == 'uint64_t':
            return '%dul' % (v & 0xffffffffffffffff)
        if want == 'int64_t':
            v &= 0xffffffffffffffff
            if v >= 1 << 63:
                v -= 1 << 64
            return '%dl' % v if v != -(1 << 63) else 'int64_t(0x8000000000000000ul)'
        if want == 'bool':
            return 'true' if v else 'false'
        raise Err('immediate type ' + ptype)

    def rname(self, r):
        return 'r_' + (r[1:] if r.startswith('%') else 'x_' + r).replace('$', '_')

    def assign(self, dst, expr, ptype):
        dst = dst.strip()
        if dst == '_':
            self.line('%s;' % expr)
            return
        st = GT[self.regtype(dst)]
        self.line('%s = %s;' % (self.rname(dst), self.conv(expr, VT[ptype], st)))

    # ---------------------------------------------------------------- address operands
    def addr(self, tok):
        """[base+off] -> (base_token or None, offset int)"""
        m = re.match(r'\[\s*([^\]+\-]+?)\s*(?:([+-])\s*(-?\w+))?\s*\]$', tok)
        if not m:
            raise Err('address ' + tok)
        base = m.group(1).strip()
        off = 0
        if m.group(3):
            off = int_imm(m.group(3))
            if m.group(2) == '-':
                off = -off
        return base, off

    def addr_expr64(self, tok):
        base, off = self.addr(tok)
        b = self.val(base, 'u64')
        if off:
            return '(%s + %dul)' % (b, off & 0xffffffffffffffff)
        return b

    def addr_expr32(self, tok):
        base, off = self.addr(tok)
        if base.startswith('%') and GT[self.regtype(base)] == 'uint64_t':
            b = 'uint(%s)' % self.rname(base)
        else:
            b = self.val(base, 'u32')
        if off:
            return '(%s + %du)' % (b, off & 0xffffffff)
        return b

    # ---------------------------------------------------------------- instructions
    def ftype(self, mods):
        for t in ('f32', 'f16x2', 'f16'):
            if t in mods:
                return t
        return None

    def ityp(self, mods):
        for t in ('s64', 'u64', 'b64', 's32', 'u32', 'b32', 's16', 'u16', 'b16', 'pred'):
            if t in mods:
                return t
        return None

    def reg_rw(self, ins):
        """(reads, writes): register names an instruction reads / writes"""
        def regs_in(t):
            return [r for r in re.findall(r'%?[A-Za-z_$][\w$]*', t) if r in self.e.regs]
        a = ins.args
        reads, writes = [], []
        if ins.op in ('st', 'sust', 'red', 'bar', 'barrier', 'membar', 'fence'):
            for x in a:
                reads += regs_in(x)
        elif a:
            dst = a[0]
            if dst.startswith('['):
                reads += regs_in(dst)
            else:
                writes += regs_in(dst)
            for x in a[1:]:
                reads += regs_in(x)
        if ins.pred is not None:
            reads.append(ins.pred)
        return reads, writes

    def emit_instr(self, ins):
        if ins.op != 'mma' and self.last_a:
            _, w = self.reg_rw(ins)
            if set(w) & set(self.last_a[0]):
                self.last_a = None
        if self.ldm_src and ins.op not in ('mma', 'ldmatrix'):
            _, w = self.reg_rw(ins)
            for r in w:
                self.ldm_src.pop(r, None)
        if (self.gld_src or self.quad_cache) and ins.op != 'mma':
            _, w = self.reg_rw(ins)
            if ins.op in ('st', 'red', 'atom') and ('global' in ins.mods or not ({'shared', 'local'} & set(ins.mods))):
                self.gld_src = {}   # memory may have changed: no re-reading at the mma
            for r in w:
                self.gld_src.pop(r, None)
                self.quad_cache.pop(r, None)
                for k in [k for k, v in self.gld_src.items() if v[0] == r]:
                    del self.gld_src[k]
        if self.chain_pairs and ins.op != 'mma':
            reads, writes = self.reg_rw(ins)
            for p in sorted(set(self.ph(self.reg_pair[r]) for r in reads if r in self.reg_pair)):
                self.line('if (!regv_%d) { %s regv_%d = true; }' % (p, self.materialize_code(p), p))
            for p in sorted(set(self.ph(self.reg_pair[r]) for r in writes if r in self.reg_pair)):
                self.line('if (!regv_%d) { %s } shv_%d = false; regv_%d = true;' % (p, self.materialize_code(p), p, p))
        if ins.pred is not None:
            p = self.rname(ins.pred)
            self.line('if (%s%s) {' % ('!' if ins.neg else '', p))
            self.ind += 1
            self.instr_body(ins)
            self.ind -= 1
            self.line('}')
        else:
            self.instr_body(ins)

    def instr_body(self, ins):
        op, mods, a = ins.op, ins.mods, ins.args
        A, V = self.assign, self.val
        ms = set(mods)
        ft = self.ftype(mods)
        if 'f64' in ms and op not in ('ld', 'st', 'mov', 'cvt'):
            raise Err('f64 arithmetic (doubles are stored as bits; only ld/st/mov/cvt are translated): ' + ins.text)
        if op == 'mov':
            t = mods[-1]
            if a[1].startswith('{'):
                parts = [x.strip() for x in a[1][1:-1].split(',')]
                if t == 'b32' and len(parts) == 2:
                    A(a[0], 'uint(%s) | (uint(%s) << 16)' % (V(parts[0], 'u16'), V(parts[1], 'u16')), 'u32'); return
                if t == 'b64' and len(parts) == 2:
                    A(a[0], 'packUint2x32(uvec2(%s, %s))' % (V(parts[0], 'u32'), V(parts[1], 'u32')), 'u64'); return
                raise Err('mov pack ' + ins.text)
            if a[0].startswith('{'):
                parts = [x.strip() for x in a[0][1:-1].split(',')]
                if t == 'b32' and len(parts) == 2:
                    s = V(a[1], 'u32')
                    if parts[0] != '_': A(parts[0], 'uint16_t(%s & 0xffffu)' % s, 'u16')
                    if parts[1] != '_': A(parts[1], 'uint16_t(%s >> 16)' % s, 'u16')
                    return
                if t == 'b64' and len(parts) == 2:
                    s = V(a[1], 'u64')
                    tmp = self.newtmp()
                    self.line('uvec2 %s = unpackUint2x32(%s);' % (tmp, s))
                    if parts[0] != '_': A(parts[0], tmp + '.x', 'u32')
                    if parts[1] != '_': A(parts[1], tmp + '.y', 'u32')
                    return
                raise Err('mov unpack ' + ins.text)
            if t == 'pred':
                A(a[0], V(a[1], 'pred'), 'pred'); return
            A(a[0], V(a[1], t), t); return
        if op == 'cvta':
            A(a[0], V(a[1], 'u64'), 'u64'); return
        if op == 'ld':
            return self.ld(ins, ms)
        if op == 'st':
            return self.st(ins, ms)
        if op == 'cvt':
            return self.cvt(ins, mods)
        if op == 'setp':
            return self.setp(ins, mods)
        if op == 'set':
            return self.set_(ins, mods)
        if op == 'selp':
            t = mods[-1]
            A(a[0], '%s ? %s : %s' % (V(a[3], 'pred'), V(a[1], t), V(a[2], t)), t); return
        if op in ('bar', 'barrier'):
            # only full barriers: bar.warp.sync with a full mask, bar.sync / barrier.sync on barrier 0 without a
            # thread count. bar.red (no destination written), bar.arrive (would block) and named or counted barriers
            # (would wait for every thread) are not translated
            if 'warp' in ms:
                if set(mods) - {'warp', 'sync'} or [x.strip().lower() for x in a] not in (['-1'], ['0xffffffff']):
                    raise Err('barrier ' + ins.text)
                self.line('subgroupMemoryBarrierShared(); subgroupBarrier();'); return
            if set(mods) - {'sync', 'aligned', 'cta'} or 'sync' not in mods or [x.strip() for x in a] != ['0']:
                raise Err('barrier ' + ins.text)
            self.line('controlBarrier(gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsShared | gl_StorageSemanticsBuffer | gl_StorageSemanticsImage, gl_SemanticsAcquireRelease);')
            return
        if op == 'membar' or op == 'fence':
            self.line('memoryBarrier();'); return
        if op == 'shfl':
            mode = [x for x in mods if x in ('down', 'bfly', 'up', 'idx')][0]
            d = [x.strip() for x in a[0].split('|')]
            p = self.newtmp()
            self.line('bool %s;' % p)
            # relative-shuffle forms: identical results, but measured no faster on RDNA4 (ds_swizzle instead of
            # ds_bpermute), so only with PTXC_SHFL_K=1
            kform = '_k' if mode in ('down', 'up', 'bfly') and self.opts.get('shfl_k') and re.match(r'^-?(0[xX][0-9a-fA-F]+|\d+)$', a[2].strip()) else ''
            self.line('uint %s_v = nvx_shfl_%s%s(%s, %s, %s, %s);' % (p, mode, kform, V(a[1], 'u32'), V(a[2], 'u32'), V(a[3], 'u32'), p))
            A(d[0], p + '_v', 'u32')
            if len(d) > 1:
                A(d[1], p, 'pred')
            return
        if op == 'mma':
            return self.mma(ins, mods)
        if op == 'ldmatrix':
            return self.ldmatrix(ins, mods)
        if op == 'tex':
            return self.tex(ins, mods)
        if op == 'sust':
            return self.sust(ins, mods)
        if op == 'suld':
            return self.suld(ins, mods)
        if op == 'atom' or op == 'red':
            return self.atom(ins, mods)
        # ---------------------------------------------------------------- float arithmetic
        if ft == 'f32':
            x = lambda i: V(a[i], 'f32')
            if op in ('add', 'sub', 'mul'):
                sym = {'add': '+', 'sub': '-', 'mul': '*'}[op]
                rnd = [r for r in ('rz', 'rm', 'rp') if r in ms]
                if rnd:
                    if op == 'sub':
                        A(a[0], 'nvx_add_%s(%s, -(%s))' % (rnd[0], x(1), x(2)), 'f32'); return
                    A(a[0], 'nvx_%s_%s(%s, %s)' % (op, rnd[0], x(1), x(2)), 'f32'); return
                e = '%s %s %s' % (x(1), sym, x(2))
                if 'sat' in ms:
                    e = 'nvx_sat(%s)' % e
                A(a[0], e, 'f32'); return
            if op == 'fma' or op == 'mad':
                if [r for r in ('rz', 'rm', 'rp') if r in ms]:
                    raise Err('directed fma ' + ins.text)
                e = 'fma(%s, %s, %s)' % (x(1), x(2), x(3))
                if 'sat' in ms:
                    e = 'nvx_sat(%s)' % e
                A(a[0], e, 'f32'); return
            if op in ('min', 'max'):
                if 'NaN' in ms:
                    raise Err('NaN min/max')
                A(a[0], 'nvx_f%s(%s, %s)' % (op, x(1), x(2)), 'f32'); return
            if op == 'abs':
                A(a[0], 'abs(%s)' % x(1), 'f32'); return
            if op == 'neg':
                A(a[0], '-(%s)' % x(1), 'f32'); return
            if op == 'div':
                if 'approx' in ms or 'full' in ms:
                    A(a[0], 'nvx_div_approx(%s, %s)' % (x(1), x(2)), 'f32'); return
                A(a[0], '(%s) / (%s)' % (x(1), x(2)), 'f32'); return
            if op == 'rcp':
                A(a[0], 'nvx_rcp(%s)' % x(1), 'f32'); return
            if op == 'ex2':
                A(a[0], 'nvx_ex2(%s)' % x(1), 'f32'); return
            if op == 'lg2':
                A(a[0], 'log2(%s)' % x(1), 'f32'); return
            if op == 'sqrt':
                A(a[0], 'sqrt(%s)' % x(1), 'f32'); return
            if op == 'rsqrt':
                A(a[0], 'inversesqrt(%s)' % x(1), 'f32'); return
            if op == 'sin':
                A(a[0], 'sin(%s)' % x(1), 'f32'); return
            if op == 'cos':
                A(a[0], 'cos(%s)' % x(1), 'f32'); return
            if op == 'copysign':
                A(a[0], 'nvx_copysign(%s, %s)' % (x(1), x(2)), 'f32'); return
            if op == 'testp':
                raise Err('testp')
        if ft == 'f16':
            h = lambda i: V(a[i], 'f16')
            if op in ('add', 'sub', 'mul'):
                sym = {'add': '+', 'sub': '-', 'mul': '*'}[op]
                e = '%s %s %s' % (h(1), sym, h(2))
                if 'sat' in ms:
                    e = 'clamp(%s, float16_t(0.0), float16_t(1.0))' % e
                A(a[0], e, 'f16'); return
            if op == 'fma':
                e = 'fma(%s, %s, %s)' % (h(1), h(2), h(3))
                if 'sat' in ms:
                    e = 'clamp(%s, float16_t(0.0), float16_t(1.0))' % e
                A(a[0], e, 'f16'); return
            if op in ('min', 'max'):
                A(a[0], 'nvx_h%s(%s, %s)' % (op, h(1), h(2)), 'f16'); return
            if op == 'neg':
                A(a[0], '-(%s)' % h(1), 'f16'); return
            if op == 'abs':
                A(a[0], 'abs(%s)' % h(1), 'f16'); return
        if ft == 'f16x2':
            def h2(i):
                return 'unpackFloat2x16(%s)' % V(a[i], 'u32')
            if op == 'neg':
                A(a[0], '%s ^ 0x80008000u' % V(a[1], 'u32'), 'u32'); return
            if op == 'abs':
                A(a[0], '%s & 0x7fff7fffu' % V(a[1], 'u32'), 'u32'); return
            if op in ('add', 'sub', 'mul', 'fma', 'min', 'max'):
                if op in ('add', 'sub', 'mul'):
                    e = '%s %s %s' % (h2(1), {'add': '+', 'sub': '-', 'mul': '*'}[op], h2(2))
                elif op == 'fma':
                    e = 'fma(%s, %s, %s)' % (h2(1), h2(2), h2(3))
                else:
                    t1, t2 = self.newtmp(), self.newtmp()
                    self.line('f16vec2 %s = %s, %s = %s;' % (t1, h2(1), t2, h2(2)))
                    e = 'f16vec2(nvx_h%s(%s.x, %s.x), nvx_h%s(%s.y, %s.y))' % (op, t1, t2, op, t1, t2)
                if 'sat' in ms:
                    e = 'clamp(%s, f16vec2(0.0), f16vec2(1.0))' % e
                A(a[0], 'packFloat2x16(%s)' % e, 'u32'); return
        # ---------------------------------------------------------------- predicates
        if 'pred' in ms and op in ('and', 'or', 'xor', 'not'):
            if op == 'not':
                A(a[0], '!%s' % V(a[1], 'pred'), 'pred'); return
            sym = {'and': '&&', 'or': '||', 'xor': '!='}[op]
            A(a[0], '(%s) %s (%s)' % (V(a[1], 'pred'), sym, V(a[2], 'pred')), 'pred'); return
        # ---------------------------------------------------------------- integer
        it = self.ityp(mods)
        if it is None and op not in ('bfi', 'bfe', 'bfind', 'prmt', 'popc', 'clz', 'brev'):
            raise Err('unhandled ' + ins.text)
        if op in ('add', 'sub', 'and', 'or', 'xor', 'mul', 'mad', 'min', 'max', 'div', 'rem', 'shl', 'shr', 'not', 'neg', 'abs'):
            w = TYPE_SIZES[it] * 8
            ut = {64: 'u64', 32: 'u32', 16: 'u16'}[w]
            st_ = {64: 's64', 32: 's32', 16: 's16'}[w]
            signed = it.startswith('s')
            if op in ('add', 'sub'):
                if 'sat' in ms:
                    raise Err('saturating int add')
                A(a[0], '%s %s %s' % (V(a[1], ut), '+' if op == 'add' else '-', V(a[2], ut)), ut); return
            if op in ('and', 'or', 'xor'):
                A(a[0], '%s %s %s' % (V(a[1], ut), {'and': '&', 'or': '|', 'xor': '^'}[op], V(a[2], ut)), ut); return
            if op == 'not':
                A(a[0], '~%s' % V(a[1], ut), ut); return
            if op == 'neg':
                A(a[0], '-%s' % V(a[1], st_), st_); return
            if op == 'abs':
                A(a[0], 'abs(%s)' % V(a[1], st_), st_); return
            if op == 'mul' or op == 'mad':
                if 'wide' in ms:
                    if w == 32:
                        if signed:
                            e = 'int64_t(%s) * int64_t(%s)' % (V(a[1], 's32'), V(a[2], 's32'))
                            if op == 'mad': e += ' + %s' % V(a[3], 's64')
                            A(a[0], e, 's64'); return
                        if op == 'mul' and a[0].strip() in self.addr32:
                            A(a[0], 'uint64_t(%s * %s)' % (V(a[1], 'u32'), V(a[2], 'u32')), 'u64'); return
                        e = 'uint64_t(%s) * uint64_t(%s)' % (V(a[1], 'u32'), V(a[2], 'u32'))
                        if op == 'mad': e += ' + %s' % V(a[3], 'u64')
                        A(a[0], e, 'u64'); return
                    if w == 16:
                        if signed:
                            e = 'int(%s) * int(%s)' % (V(a[1], 's16'), V(a[2], 's16'))
                            if op == 'mad': e += ' + %s' % V(a[3], 's32')
                            A(a[0], e, 's32'); return
                        e = 'uint(%s) * uint(%s)' % (V(a[1], 'u16'), V(a[2], 'u16'))
                        if op == 'mad': e += ' + %s' % V(a[3], 'u32')
                        A(a[0], e, 'u32'); return
                    raise Err(ins.text)
                if 'hi' in ms:
                    if w == 32:
                        e = ('nvx_mulhi_s32(%s, %s)' if signed else 'nvx_mulhi_u32(%s, %s)') % (V(a[1], st_ if signed else ut), V(a[2], st_ if signed else ut))
                        if op == 'mad': e = '%s + %s' % (e, V(a[3], st_ if signed else ut))
                        A(a[0], e, st_ if signed else ut); return
                    if w == 16:
                        if signed:
                            e = 'int16_t((int(%s) * int(%s)) >> 16)' % (V(a[1], 's16'), V(a[2], 's16'))
                            A(a[0], e, 's16'); return
                        e = 'uint16_t((uint(%s) * uint(%s)) >> 16)' % (V(a[1], 'u16'), V(a[2], 'u16'))
                        A(a[0], e, 'u16'); return
                    if w == 64:
                        raise Err('mul.hi.64')
                e = '%s * %s' % (V(a[1], ut), V(a[2], ut))
                if op == 'mad':
                    e += ' + %s' % V(a[3], ut)
                A(a[0], e, ut); return
            if op in ('min', 'max'):
                t = st_ if signed else ut
                A(a[0], '%s(%s, %s)' % (op, V(a[1], t), V(a[2], t)), t); return
            if op == 'div':
                t = st_ if signed else ut
                A(a[0], '%s / %s' % (V(a[1], t), V(a[2], t)), t); return
            if op == 'rem':
                if signed:
                    if w != 32:
                        raise Err('rem width')
                    A(a[0], 'nvx_rem32(%s, %s)' % (V(a[1], 's32'), V(a[2], 's32')), 's32'); return
                A(a[0], '%s %% %s' % (V(a[1], ut), V(a[2], ut)), ut); return
            if op in ('shl', 'shr'):
                amt = V(a[2], 'u32')
                if op == 'shl':
                    fn = {32: 'nvx_shl32', 64: 'nvx_shl64', 16: 'nvx_shl16'}[w]
                    A(a[0], '%s(%s, %s)' % (fn, V(a[1], ut), amt), ut); return
                if signed:
                    fn = {32: 'nvx_sar32', 64: 'nvx_sar64', 16: 'nvx_sar16'}[w]
                    A(a[0], '%s(%s, %s)' % (fn, V(a[1], st_), amt), st_); return
                fn = {32: 'nvx_shr32', 64: 'nvx_shr64', 16: 'nvx_shr16'}[w]
                A(a[0], '%s(%s, %s)' % (fn, V(a[1], ut), amt), ut); return
        if op == 'bfe':
            if 'u32' in ms:
                A(a[0], 'nvx_bfe_u32(%s, %s, %s)' % (V(a[1], 'u32'), V(a[2], 'u32'), V(a[3], 'u32')), 'u32'); return
            if 's32' in ms:
                A(a[0], 'nvx_bfe_s32(%s, %s, %s)' % (V(a[1], 's32'), V(a[2], 'u32'), V(a[3], 'u32')), 's32'); return
        if op == 'bfi':
            if 'b32' in ms:
                A(a[0], 'nvx_bfi32(%s, %s, %s, %s)' % (V(a[1], 'u32'), V(a[2], 'u32'), V(a[3], 'u32'), V(a[4], 'u32')), 'u32'); return
            if 'b64' in ms:
                A(a[0], 'nvx_bfi64(%s, %s, %s, %s)' % (V(a[1], 'u64'), V(a[2], 'u64'), V(a[3], 'u32'), V(a[4], 'u32')), 'u64'); return
        if op == 'bfind':
            if 'u32' in ms:
                A(a[0], 'nvx_bfind_u32(%s, %s)' % (V(a[1], 'u32'), 'true' if 'shiftamt' in ms else 'false'), 'u32'); return
        if op == 'popc':
            A(a[0], 'uint(bitCount(%s))' % V(a[1], 'u32'), 'u32'); return
        if op == 'brev':
            A(a[0], 'bitfieldReverse(%s)' % V(a[1], 'u32'), 'u32'); return
        if op == 'clz':
            A(a[0], '31u - uint(findMSB(%s))' % V(a[1], 'u32'), 'u32'); return
        raise Err('unhandled ' + ins.text)

    def setp(self, ins, mods):
        a = ins.args
        cmp_ = mods[0]
        t = [m for m in mods if m in TYPE_SIZES][0]
        if t == 'f64':
            raise Err('f64 compare ' + ins.text)
        x, y = self.val(a[1], t), self.val(a[2], t)
        isf = t in ('f32', 'f16', 'f64')
        tbl = {'eq': '{0} == {1}', 'ne': '{0} != {1}', 'lt': '{0} < {1}', 'le': '{0} <= {1}', 'gt': '{0} > {1}', 'ge': '{0} >= {1}',
               'lo': '{0} < {1}', 'ls': '{0} <= {1}', 'hi': '{0} > {1}', 'hs': '{0} >= {1}'}
        if isf:
            ftbl = {'eq': '{0} == {1}', 'ne': '({0} < {1} || {0} > {1})', 'lt': '{0} < {1}', 'le': '{0} <= {1}',
                    'gt': '{0} > {1}', 'ge': '{0} >= {1}',
                    'equ': '!({0} < {1} || {0} > {1})', 'neu': '!({0} == {1})', 'ltu': '!({0} >= {1})', 'leu': '!({0} > {1})',
                    'gtu': '!({0} <= {1})', 'geu': '!({0} < {1})', 'num': '(!isnan({0}) && !isnan({1}))', 'nan': '(isnan({0}) || isnan({1}))'}
            c = ftbl[cmp_].format('(%s)' % x, '(%s)' % y)
        else:
            if cmp_ in ('lo', 'ls', 'hi', 'hs'):
                t = {'s32': 'u32', 's16': 'u16', 's64': 'u64'}.get(t, t)
                x, y = self.val(a[1], t), self.val(a[2], t)
            if t.startswith('b'):
                t = 'u' + t[1:]
                x, y = self.val(a[1], t), self.val(a[2], t)
            c = tbl[cmp_].format('(%s)' % x, '(%s)' % y)
        # combined forms setp.CmpOp.BoolOp.type p|q, a, b, c
        boolop = [m for m in mods if m in ('and', 'or', 'xor')]
        dst = [d.strip() for d in a[0].split('|')]
        if boolop:
            pc = self.val(a[3], 'pred')
            sym = {'and': '&&', 'or': '||', 'xor': '!='}[boolop[0]]
            tmp = self.newtmp()
            self.line('bool %s = %s;' % (tmp, c))
            self.assign(dst[0], '(%s) %s (%s)' % (tmp, sym, pc), 'pred')
            if len(dst) > 1:
                self.assign(dst[1], '(!%s) %s (%s)' % (tmp, sym, pc), 'pred')
            return
        if len(dst) > 1:
            tmp = self.newtmp()
            self.line('bool %s = %s;' % (tmp, c))
            self.assign(dst[0], tmp, 'pred')
            self.assign(dst[1], '!' + tmp, 'pred')
            return
        self.assign(dst[0], c, 'pred')

    def set_(self, ins, mods):
        a = ins.args
        cmp_ = mods[0]
        types = [m for m in mods if m in TYPE_SIZES]
        dt, stp = types[0], types[1]
        ftbl = {'eq': '{0} == {1}', 'ne': '({0} < {1} || {0} > {1})', 'lt': '{0} < {1}', 'le': '{0} <= {1}',
                'gt': '{0} > {1}', 'ge': '{0} >= {1}',
                'equ': '!({0} < {1} || {0} > {1})', 'neu': '!({0} == {1})', 'ltu': '!({0} >= {1})', 'leu': '!({0} > {1})',
                'gtu': '!({0} <= {1})', 'geu': '!({0} < {1})'}
        if stp == 'f16x2':
            t1, t2 = self.newtmp(), self.newtmp()
            self.line('f16vec2 %s = unpackFloat2x16(%s), %s = unpackFloat2x16(%s);' % (t1, self.val(a[1], 'u32'), t2, self.val(a[2], 'u32')))
            lo = ftbl[cmp_].format(t1 + '.x', t2 + '.x')
            hi = ftbl[cmp_].format(t1 + '.y', t2 + '.y')
            if dt == 'f16x2':
                one = '0x3c00u'
            elif dt in ('u32', 'b32', 's32'):
                one = '0xffffu'
            else:
                raise Err('set dtype ' + ins.text)
            self.assign(a[0], '((%s) ? %s : 0u) | (((%s) ? %s : 0u) << 16)' % (lo, one, hi, one), 'u32')
            return
        if stp in ('f32',) and dt in ('u32', 'b32', 's32', 'f32'):
            c = ftbl[cmp_].format('(%s)' % self.val(a[1], 'f32'), '(%s)' % self.val(a[2], 'f32'))
            if dt == 'f32':
                self.assign(a[0], '(%s) ? 1.0 : 0.0' % c, 'f32')
            else:
                self.assign(a[0], '(%s) ? 0xffffffffu : 0u' % c, 'u32')
            return
        raise Err('set form ' + ins.text)

    def cvt(self, ins, mods):
        a = ins.args
        types = [m for m in mods if m in TYPE_SIZES]
        dt, stp = types[0], types[1]
        rnd = [m for m in mods if m in ('rn', 'rz', 'rm', 'rp', 'rni', 'rzi', 'rmi', 'rpi')]
        rnd = rnd[0] if rnd else None
        sat = 'sat' in mods
        A, V = self.assign, self.val
        fd, fs = dt in ('f32', 'f16', 'f64'), stp in ('f32', 'f16', 'f64')
        if dt == 'f16x2' and stp == 'f32':
            if rnd not in ('rn', None):
                raise Err('cvt rounding ' + ins.text)
            # d.hi = cvt(a), d.lo = cvt(b)
            A(a[0], '(uint(nvx_f2h(%s)) << 16) | uint(nvx_f2h(%s))' % (V(a[1], 'f32'), V(a[2], 'f32')), 'u32'); return
        if fd and fs:
            x = V(a[1], stp)
            if dt == 'f32' and stp == 'f32':
                e = {None: '%s', 'rni': 'roundEven(%s)', 'rzi': 'trunc(%s)', 'rmi': 'floor(%s)', 'rpi': 'ceil(%s)'}[rnd] % x
                if sat:
                    e = 'nvx_sat(%s)' % e
                A(a[0], e, 'f32'); return
            if dt == 'f16' and stp == 'f32':
                if rnd not in ('rn', None):
                    raise Err('cvt rounding ' + ins.text)
                e = 'nvx_f2h(%s)' % x
                if sat:
                    e = 'nvx_f2h(nvx_sat(%s))' % x
                A(a[0], e, 'u16'); return
            if dt == 'f32' and stp == 'f16':
                A(a[0], 'float(%s)' % x, 'f32'); return
            if dt == 'f32' and stp == 'f64' and rnd in ('rn', None) and not sat:
                A(a[0], 'nvx_d2f(%s)' % x, 'f32'); return
            if dt == 'f16' and stp == 'f16':
                e = {None: '%s', 'rni': 'roundEven(%s)', 'rzi': 'trunc(%s)', 'rmi': 'floor(%s)', 'rpi': 'ceil(%s)'}[rnd] % x
                A(a[0], e, 'f16'); return
            raise Err('cvt ' + ins.text)
        if fd and not fs:
            # int -> float (rn)
            x = V(a[1], stp)
            if dt == 'f32':
                if rnd not in ('rn', None):
                    raise Err('cvt int->f32 rounding ' + ins.text)
                A(a[0], 'float(%s)' % x, 'f32'); return
            if dt == 'f16':
                if rnd not in ('rn', None):
                    raise Err('cvt int->f16 rounding ' + ins.text)
                # f16 from an int via f32: exact while |x| < 2^24 (DLSS-G converts small loop indices)
                A(a[0], 'nvx_f2h(float(%s))' % x, 'u16'); return
            raise Err('cvt ' + ins.text)
        if fs and not fd:
            x = V(a[1], stp)
            if stp == 'f16':
                x = 'float(%s)' % x
            if rnd not in ('rni', 'rzi', 'rmi', 'rpi', None):
                raise Err('cvt rounding ' + ins.text)
            f = {'rni': 'roundEven(%s)', 'rzi': 'trunc(%s)', 'rmi': 'floor(%s)', 'rpi': 'ceil(%s)', None: 'trunc(%s)'}[rnd] % x
            # RDNA's v_cvt_i32_f32 / v_cvt_u32_f32 saturate (NaN -> 0) exactly like PTX's cvt.*i: use them directly
            # for run-time values (checked on NaN, +-inf, +-0, out-of-range and negative inputs); immediates keep the
            # explicit form so constant folding cannot differ
            hw = not self.opts.get('no_hwcvt') and a[1].strip().startswith('%')
            f2i, f2u = ('int(%s)', 'uint(%s)') if hw else ('nvx_f2i(%s)', 'nvx_f2u(%s)')
            if dt == 's32':
                A(a[0], f2i % f, 's32'); return
            if dt == 'u32':
                A(a[0], f2u % f, 'u32'); return
            if dt == 's16':
                A(a[0], 'int16_t(clamp(%s, -32768, 32767))' % (f2i % f), 's16'); return
            if dt == 'u16':
                A(a[0], 'uint16_t(min(%s, 65535u))' % (f2u % f), 'u16'); return
            if dt == 's64':
                # saturates at the 64-bit limits (2^63 is exact in fp32), NaN -> 0
                t = self.newtmp()
                self.line('float %s = %s;' % (t, f))
                A(a[0], 'isnan({0}) ? 0l : {0} >= 9223372036854775808.0 ? 9223372036854775807l : '
                        '{0} <= -9223372036854775808.0 ? (-9223372036854775807l - 1l) : int64_t({0})'.format(t), 's64')
                return
            raise Err('cvt ' + ins.text)
        # int -> int
        def width(t):
            return TYPE_SIZES[t] * 8
        ws, wd = width(stp), width(dt)
        ssig = stp.startswith('s')
        if sat:
            raise Err('saturating int cvt ' + ins.text)
        # read source at its declared width
        if ws == 8:
            src = V(a[1], 'u32' if not a[1].startswith('%') or GT[self.regtype(a[1])] != 'uint16_t' else 'u16')
            src = 'uint(%s)' % src
            if ssig:
                val = 'int(%s << 24) >> 24' % src
                A(a[0], self.int_resize(val, 'int', dt), dt); return
            A(a[0], self.int_resize('(%s & 0xffu)' % src, 'uint', dt), dt); return
        st_ = {8: 'u8', 16: 's16' if ssig else 'u16', 32: 's32' if ssig else 'u32', 64: 's64' if ssig else 'u64'}[ws]
        src = V(a[1], st_)
        A(a[0], self.int_resize(src, VT[st_], dt), dt)

    def int_resize(self, expr, frm, dt):
        to = VT[dt]
        if frm == to:
            return expr
        return '%s(%s)' % (to, expr)

    # ---------------------------------------------------------------- memory
    def ld(self, ins, ms):
        a = ins.args
        space = 'global' if 'global' in ms else 'shared' if 'shared' in ms else 'local' if 'local' in ms else 'param' if 'param' in ms else 'const' if 'const' in ms else None
        if space is None and self.generic_is_global(a[1]):
            space = 'global'
        if space is None:
            raise Err('generic ld ' + ins.text)
        vec = 4 if 'v4' in ms else 2 if 'v2' in ms else 1
        t = [m for m in ins.mods if m in TYPE_SIZES][-1]
        dsts = [x.strip() for x in a[0][1:-1].split(',')] if a[0].startswith('{') else [a[0]]
        esz = TYPE_SIZES[t]
        if space == 'param':
            base, off = self.addr(a[1])
            bexpr = self.param_base(base)
            for i, d in enumerate(dsts):
                o = '%s + %du' % (bexpr, off + i * esz) if bexpr != '0u' else '%du' % (off + i * esz)
                self.ld_param(d, t, o)
            return
        if space == 'global':
            ad = self.addr_expr64(a[1])
            for d in dsts:
                self.gld_src.pop(d, None)
            self.mem_global_ld(dsts, t, vec, ad)
            if self.fuse_b and vec == 1 and TYPE_SIZES[t] == 4 and t != 'f32' and ins.pred is None:
                base, off = self.addr(a[1])
                if base.startswith('%') and base in self.e.regs and GT[self.regtype(base)] == 'uint64_t' and dsts[0] != base:
                    self.gld_src[dsts[0]] = (base, off)
            return
        if space == 'shared':
            ad = self.addr_expr32(a[1])
            vol = 'volatile' in ms
            if vol:
                self.line('subgroupMemoryBarrierShared();')
            self.mem_shared_ld(dsts, t, vec, ad)
            return
        if space == 'local':
            ad = self.addr_expr32(a[1])
            self.mem_local_ld(dsts, t, vec, ad)
            return
        raise Err('ld ' + ins.text)

    def param_base(self, base):
        if base in self.param_off:
            return '%du' % self.param_off[base]
        if base.startswith('%'):
            return 'uint(%s)' % self.val(base, 'u64')
        raise Err('param base ' + base)

    def ld_param(self, d, t, o):
        if t in ('u64', 'b64', 's64'):
            self.assign(d, 'nvx_p64(%s)' % o, 'u64')
        elif t in ('u32', 'b32', 's32'):
            self.assign(d, 'nvx_p32(%s)' % o, 'u32')
        elif t == 'f32':
            self.assign(d, 'uintBitsToFloat(nvx_p32(%s))' % o, 'f32')
        elif t in ('u16', 'b16', 's16'):
            self.assign_int(d, 'nvx_p16(%s)' % o, t == 's16', 16)
        elif t in ('u8', 'b8', 's8'):
            self.assign_int(d, 'nvx_p8(%s)' % o, t == 's8', 8)
        else:
            raise Err('ld.param type ' + t)

    def assign_int(self, d, uexpr, signed, width):
        """assign a zero-extended uint value of `width` bits to register d (sign-extending if signed)"""
        if signed:
            e = 'uint(int(%s << %du) >> %du)' % (uexpr, 32 - width, 32 - width)
        else:
            e = uexpr
        st = GT[self.regtype(d)]
        if st == 'uint16_t':
            self.line('%s = uint16_t(%s);' % (self.rname(d), e))
        elif st == 'uint':
            self.line('%s = %s;' % (self.rname(d), e))
        elif st == 'uint64_t':
            if signed:
                self.line('%s = uint64_t(int64_t(int(%s)));' % (self.rname(d), e))
            else:
                self.line('%s = uint64_t(%s);' % (self.rname(d), e))
        else:
            raise Err('int load into %s' % st)

    def mem_global_ld(self, dsts, t, vec, ad):
        esz = TYPE_SIZES[t]
        if esz == 4 and vec in (2, 4):
            ty = 'NvxU32x%d' % vec
            tmp = self.newtmp()
            self.line('uvec%d %s = %s(%s).v;' % (vec, tmp, ty, ad))
            for i, d in enumerate(dsts):
                if d != '_':
                    self.assign(d, '%s.%s' % (tmp, 'xyzw'[i]), 'f32' if t == 'f32' else 'u32') if t != 'f32' else self.assign(d, 'uintBitsToFloat(%s.%s)' % (tmp, 'xyzw'[i]), 'f32')
            return
        if esz == 2 and vec == 2:
            tmp = self.newtmp()
            self.line('u16vec2 %s = NvxU16x2(%s).v;' % (tmp, ad))
            for i, d in enumerate(dsts):
                self.assign(d, '%s.%s' % (tmp, 'xy'[i]), 'u16')
            return
        if esz == 8 and vec == 2:
            tmp = self.newtmp()
            self.line('u64vec2 %s = NvxU64x2(%s).v;' % (tmp, ad))
            for i, d in enumerate(dsts):
                self.assign(d, '%s.%s' % (tmp, 'xy'[i]), 'u64')
            return
        if vec != 1:
            raise Err('global vector load %s v%d' % (t, vec))
        d = dsts[0]
        if esz == 4:
            e = 'NvxU32(%s).v' % ad
            if t == 'f32':
                self.assign(d, 'uintBitsToFloat(%s)' % e, 'f32')
            else:
                self.assign(d, e, 'u32')
        elif esz == 8:
            self.assign(d, 'NvxU64(%s).v' % ad, 'u64')
        elif esz == 2:
            self.assign_int(d, 'uint(NvxU16(%s).v)' % ad, t == 's16', 16)
        elif esz == 1:
            if t == 's8':
                self.assign_int(d, 'uint(uint8_t(NvxI8(%s).v))' % ad, True, 8)
            else:
                self.assign_int(d, 'uint(NvxU8(%s).v)' % ad, False, 8)
        else:
            raise Err('global load size')

    def mem_shared_ld(self, dsts, t, vec, ad):
        esz = TYPE_SIZES[t]
        if esz == 4:
            for i, d in enumerate(dsts):
                e = 'smem[(%s >> 2) + %du]' % (ad, i) if i else 'smem[%s >> 2]' % ad
                if d == '_':
                    continue
                if t == 'f32':
                    self.assign(d, 'uintBitsToFloat(%s)' % e, 'f32')
                else:
                    self.assign(d, e, 'u32')
            return
        if esz == 8:
            for i, d in enumerate(dsts):
                w = '(%s >> 2) + %du' % (ad, 2 * i)
                self.assign(d, 'packUint2x32(uvec2(smem[%s], smem[%s + 1u]))' % (w, w), 'u64')
            return
        if esz == 2:
            for i, d in enumerate(dsts):
                bo = '(%s + %du)' % (ad, 2 * i) if i else ad
                self.assign_int(d, '((smem[%s >> 2] >> ((%s & 2u) * 8u)) & 0xffffu)' % (bo, bo), t == 's16', 16)
            return
        if esz == 1:
            for i, d in enumerate(dsts):
                bo = '(%s + %du)' % (ad, i) if i else ad
                self.assign_int(d, '((smem[%s >> 2] >> ((%s & 3u) * 8u)) & 0xffu)' % (bo, bo), t == 's8', 8)
            return
        raise Err('shared load')

    def mem_local_ld(self, dsts, t, vec, ad):
        esz = TYPE_SIZES[t]
        if esz == 8:
            for i, d in enumerate(dsts):
                w = '(%s >> 2) + %du' % (ad, 2 * i)
                self.assign(d, 'packUint2x32(uvec2(lmem[%s], lmem[%s + 1u]))' % (w, w), 'u64')
            return
        if esz != 4:
            raise Err('local load size %d' % esz)
        for i, d in enumerate(dsts):
            e = 'lmem[(%s >> 2) + %du]' % (ad, i) if i else 'lmem[%s >> 2]' % ad
            if t == 'f32':
                self.assign(d, 'uintBitsToFloat(%s)' % e, 'f32')
            else:
                self.assign(d, e, 'u32')

    def st(self, ins, ms):
        a = ins.args
        space = 'global' if 'global' in ms else 'shared' if 'shared' in ms else 'local' if 'local' in ms else None
        if space is None and self.generic_is_global(a[0]):
            space = 'global'
        if space is None:
            raise Err('generic st ' + ins.text)
        vec = 4 if 'v4' in ms else 2 if 'v2' in ms else 1
        t = [m for m in ins.mods if m in TYPE_SIZES][-1]
        srcs = [x.strip() for x in a[1][1:-1].split(',')] if a[1].startswith('{') else [a[1]]
        esz = TYPE_SIZES[t]
        it32 = 'u32'
        if space == 'global':
            ad = self.addr_expr64(a[0])
            if esz == 4 and vec in (2, 4):
                vals = ', '.join(self.val(s, 'u32') if t != 'f32' else 'floatBitsToUint(%s)' % self.val(s, 'f32') for s in srcs)
                self.line('NvxU32x%d(%s).v = uvec%d(%s);' % (vec, ad, vec, vals))
                return
            if esz == 2 and vec == 2:
                self.line('NvxU16x2(%s).v = u16vec2(%s, %s);' % (ad, self.val(srcs[0], 'u16'), self.val(srcs[1], 'u16')))
                return
            if vec != 1:
                raise Err('global vector store')
            s = srcs[0]
            if esz == 4:
                v = 'floatBitsToUint(%s)' % self.val(s, 'f32') if t == 'f32' else self.val(s, 'u32')
                self.line('NvxU32(%s).v = %s;' % (ad, v))
            elif esz == 8:
                self.line('NvxU64(%s).v = %s;' % (ad, self.val(s, 'u64')))
            elif esz == 2:
                self.line('NvxU16(%s).v = %s;' % (ad, self.val(s, 'u16')))
            elif esz == 1:
                self.line('NvxU8(%s).v = uint8_t(%s);' % (ad, self.lowbits(s)))
            return
        if space == 'shared':
            ad = self.addr_expr32(a[0])
            vol = 'volatile' in ms
            if esz == 4:
                for i, s in enumerate(srcs):
                    v = 'floatBitsToUint(%s)' % self.val(s, 'f32') if t == 'f32' else self.val(s, 'u32')
                    idx = '(%s >> 2) + %du' % (ad, i) if i else '%s >> 2' % ad
                    self.line('smem[%s] = %s;' % (idx, v))
            elif esz == 8:
                for i, s in enumerate(srcs):
                    tmp = self.newtmp()
                    self.line('uvec2 %s = unpackUint2x32(%s);' % (tmp, self.val(s, 'u64')))
                    w = '(%s >> 2) + %du' % (ad, 2 * i)
                    self.line('smem[%s] = %s.x; smem[%s + 1u] = %s.y;' % (w, tmp, w, tmp))
            elif esz == 2 and vec == 2:
                self.line('smem[%s >> 2] = uint(%s) | (uint(%s) << 16);' % (ad, self.val(srcs[0], 'u16'), self.val(srcs[1], 'u16')))
            elif esz == 2 and vec == 4:
                self.line('smem[%s >> 2] = uint(%s) | (uint(%s) << 16); smem[(%s >> 2) + 1u] = uint(%s) | (uint(%s) << 16);' % (
                    ad, self.val(srcs[0], 'u16'), self.val(srcs[1], 'u16'), ad, self.val(srcs[2], 'u16'), self.val(srcs[3], 'u16')))
            elif esz == 2:
                tmp = self.newtmp()
                self.line('uint %s = %s;' % (tmp, ad))
                self.line('{ uint sh_ = (%s & 2u) * 8u; atomicAnd(smem[%s >> 2], ~(0xffffu << sh_)); atomicOr(smem[%s >> 2], uint(%s) << sh_); }' % (tmp, tmp, tmp, self.val(srcs[0], 'u16')))
            elif esz == 1:
                if vec != 1:
                    raise Err('shared u8 vector store')
                tmp = self.newtmp()
                self.line('uint %s = %s;' % (tmp, ad))
                self.line('{ uint sh_ = (%s & 3u) * 8u; atomicAnd(smem[%s >> 2], ~(0xffu << sh_)); atomicOr(smem[%s >> 2], (%s & 0xffu) << sh_); }' % (tmp, tmp, tmp, self.lowbits(srcs[0])))
            else:
                raise Err('shared store')
            if vol:
                self.line('subgroupMemoryBarrierShared();')
            return
        if space == 'local':
            ad = self.addr_expr32(a[0])
            if esz == 8:
                for i, s_ in enumerate(srcs):
                    tmp = self.newtmp()
                    w = '(%s >> 2) + %du' % (ad, 2 * i)
                    self.line('uvec2 %s = unpackUint2x32(%s);' % (tmp, self.val(s_, 'u64')))
                    self.line('lmem[%s] = %s.x; lmem[%s + 1u] = %s.y;' % (w, tmp, w, tmp))
                return
            if esz != 4:
                raise Err('local store size')
            for i, s in enumerate(srcs):
                v = 'floatBitsToUint(%s)' % self.val(s, 'f32') if t == 'f32' else self.val(s, 'u32')
                idx = '(%s >> 2) + %du' % (ad, i) if i else '%s >> 2' % ad
                self.line('lmem[%s] = %s;' % (idx, v))
            return

    def lowbits(self, s):
        if s.startswith('%'):
            st = GT[self.regtype(s)]
            if st == 'uint16_t':
                return 'uint(%s)' % self.rname(s)
            if st == 'uint':
                return self.rname(s)
            if st == 'uint64_t':
                return 'uint(%s)' % self.rname(s)
        return self.val(s, 'u32')

    def atom(self, ins, mods):
        a = ins.args
        ms = set(mods)
        if 'global' not in ms:
            raise Err('atom space ' + ins.text)
        opn = [m for m in mods if m in ('min', 'max', 'add', 'exch', 'or', 'and', 'xor', 'cas', 'inc', 'dec')][0]
        t = [m for m in mods if m in TYPE_SIZES][-1]
        red = ins.op == 'red'
        if red:
            ad, v = self.addr_expr64(a[0]), a[1]
        else:
            ad, v = self.addr_expr64(a[1]), a[2]
        fn = {'min': 'atomicMin', 'max': 'atomicMax', 'add': 'atomicAdd', 'exch': 'atomicExchange', 'or': 'atomicOr',
              'and': 'atomicAnd', 'xor': 'atomicXor'}[opn]
        if t == 's64':
            e = '%s(NvxI64A(%s).v, %s)' % (fn, ad, self.val(v, 's64'))
            rt = 's64'
        elif t in ('u64', 'b64'):
            e = '%s(NvxU64A(%s).v, %s)' % (fn, ad, self.val(v, 'u64'))
            rt = 'u64'
        elif t == 's32':
            e = '%s(NvxI32A(%s).v, %s)' % (fn, ad, self.val(v, 's32'))
            rt = 's32'
        elif t in ('u32', 'b32'):
            e = '%s(NvxU32A(%s).v, %s)' % (fn, ad, self.val(v, 'u32'))
            rt = 'u32'
        elif t == 'f32' and opn == 'add':
            raise Err('float atomics')
        else:
            raise Err('atom ' + ins.text)
        if red:
            self.line('%s;' % e)
        else:
            self.assign(a[0], e, rt)

    # ---------------------------------------------------------------- textures / surfaces
    def tex_operands(self, tok):
        m = re.match(r'\[\s*([^,\]]+?)\s*,\s*\{([^}]*)\}\s*\]$', tok)
        if not m:
            raise Err('tex operand ' + tok)
        return m.group(1).strip(), [x.strip() for x in m.group(2).split(',')]

    def handle(self, h):
        return self.val(h, 'u64')

    def tex(self, ins, mods):
        a = ins.args
        ms = set(mods)
        if '2d' not in ms:
            raise Err('tex geometry ' + ins.text)
        dsts = [x.strip() for x in a[0][1:-1].split(',')]
        h, coords = self.tex_operands(a[1])
        ctype = mods[-1]
        rtype = mods[-2]
        hv = self.handle(h)
        tmp = self.newtmp()
        if ctype == 's32':
            if 'level' in ms or 'grad' in ms:
                raise Err('integer-coordinate tex with lod ' + ins.text)
            x, y = self.val(coords[0], 's32'), self.val(coords[1], 's32')
            if rtype == 'f32':
                self.line('vec4 %s = nvx_tex_fetch_i(%s, %s, %s);' % (tmp, hv, x, y))
            elif rtype in ('u32', 's32'):
                self.line('uvec4 %s = nvx_utex_fetch_i(%s, %s, %s);' % (tmp, hv, x, y))
            else:
                raise Err(ins.text)
        else:
            u, v = self.val(coords[0], 'f32'), self.val(coords[1], 'f32')
            if 'level' in ms:
                lod = self.val(a[2], 'f32')
                if rtype != 'f32':
                    raise Err(ins.text)
                self.line('vec4 %s = nvx_tex2d_level(%s, %s, %s, %s);' % (tmp, hv, u, v, lod))
            elif 'grad' in ms:
                raise Err('tex.grad')
            elif rtype == 'f32':
                # channels nobody reads are not fetched: the emulated bilinear gathers each used channel's 2x2
                # footprint with one textureGather (same texels, same arithmetic per channel)
                used = 0
                for i, d in enumerate(dsts):
                    if d != '_' and d in self.read_regs:
                        used |= 1 << i
                if self.opts.get('no_gather') or used == 0:
                    self.line('vec4 %s = nvx_tex2d(%s, %s, %s);' % (tmp, hv, u, v))
                else:
                    self.line('vec4 %s = nvx_tex2d_m(%s, %s, %s, %du);' % (tmp, hv, u, v, used))
            elif rtype in ('u32', 's32'):
                self.line('uvec4 %s = nvx_utex2d(%s, %s, %s);' % (tmp, hv, u, v))
            else:
                raise Err(ins.text)
        for i, d in enumerate(dsts):
            if d == '_':
                continue
            if rtype == 'f32':
                self.assign(d, '%s.%s' % (tmp, 'xyzw'[i]), 'f32')
            else:
                self.assign(d, '%s.%s' % (tmp, 'xyzw'[i]), 'u32')

    def sust(self, ins, mods):
        a = ins.args
        ms = set(mods)
        if '2d' in ms and 'b' in ms and not ms & {'3d', 'a1d', 'a2d'}:
            return self.sust_b(ins, mods)
        if '2d' not in ms or 'p' not in ms:
            raise Err('sust form ' + ins.text)
        h, coords = self.tex_operands(a[0])
        srcs = [x.strip() for x in a[1][1:-1].split(',')] if a[1].startswith('{') else [a[1]]
        def bits(s):
            if s.startswith('%') and GT[self.regtype(s)] == 'float':
                return 'floatBitsToUint(%s)' % self.rname(s)
            return self.val(s, 'u32')
        v = [bits(s) for s in srcs] + ['0u'] * (4 - len(srcs))
        if len(srcs) < 4:
            v[3] = '0x3f800000u' if len(srcs) < 4 else v[3]
        self.line('nvx_sust_p(%s, %s, %s, uvec4(%s));' % (self.handle(h), self.val(coords[0], 's32'), self.val(coords[1], 's32'), ', '.join(v)))

    def sust_b(self, ins, mods):
        """raw store of one texel (8/16/32-bit components), converted to the typed view (nvx_sust_b)"""
        a = ins.args
        ms = set(mods)
        esz = {'b8': 1, 'b16': 2, 'b32': 4}.get(next((m for m in mods if m in ('b8', 'b16', 'b32', 'b64')), ''), 0)
        if not esz:
            raise Err('sust form ' + ins.text)
        h, coords = self.tex_operands(a[0])
        srcs = [x.strip() for x in a[1][1:-1].split(',')] if a[1].startswith('{') else [a[1].strip()]
        mask = (1 << (8 * esz)) - 1
        v = []
        for s in srcs:
            if s.startswith('%') and GT[self.regtype(s)] == 'float':
                v.append('floatBitsToUint(%s)' % self.rname(s))
            else:
                v.append('(%s & 0x%xu)' % (self.val(s, 'u32'), mask))
        v += ['0u'] * (4 - len(v))
        self.line('nvx_sust_b(%s, %s, %s, uvec4(%s), %du, %du, %s);' % (
            self.handle(h), self.val(coords[0], 's32'), self.val(coords[1], 's32'), ', '.join(v), esz, len(srcs),
            'true' if 'clamp' in ms else 'false'))

    def suld(self, ins, mods):
        a = ins.args
        ms = set(mods)
        if not ({'b', '2d'} <= ms):
            raise Err('suld form ' + ins.text)
        dsts = [x.strip() for x in a[0][1:-1].split(',')] if a[0].startswith('{') else [a[0].strip()]
        h, coords = self.tex_operands(a[1])
        tmp = self.newtmp()
        if {'v4', 'b32'} <= ms:
            # 16-byte texels (RGBA32 formats)
            self.line('uvec4 %s = nvx_suld_b_v4(%s, %s, %s);' % (tmp, self.handle(h), self.val(coords[0], 's32'), self.val(coords[1], 's32')))
            for i, d in enumerate(dsts):
                if d != '_':
                    self.assign(d, '%s.%s' % (tmp, 'xyzw'[i]), 'u32')
            return
        esz = {'b8': 1, 'b16': 2, 'b32': 4}.get(next((m for m in mods if m in ('b8', 'b16', 'b32', 'b64')), ''), 0)
        if not esz or ms & {'3d', 'a1d', 'a2d'}:
            raise Err('suld form ' + ins.text)
        # one texel of len(dsts) components, rebuilt bit-exactly from the typed view (nvx_suld_b)
        self.line('uvec4 %s = nvx_suld_b(%s, %s, %s, %du, %s);' % (tmp, self.handle(h), self.val(coords[0], 's32'),
                                                                 self.val(coords[1], 's32'), esz, 'true' if 'clamp' in ms else 'false'))
        for i, d in enumerate(dsts):
            if d != '_':
                self.assign_int(d, '%s.%s' % (tmp, 'xyzw'[i]), False, 8 * esz)

    # ---------------------------------------------------------------- tensor cores
    def mma(self, ins, mods):
        pm = self.pair_m1.get(id(ins))
        if pm is True:
            self.line('// %s  (computed with the next mma as one 16x16 WMMA)' % ins.text)
            self.uses_mma = True
            return
        m1 = pm  # the deferred first mma of a pair, or None
        ms = set(mods)
        a = ins.args
        shape = [m for m in mods if m.startswith('m16n8k')][0]
        if not ({'row', 'col'} <= ms) or mods[-4:] != ['f16', 'f16', 'f16', 'f16']:
            raise Err('mma form ' + ins.text)
        self.uses_mma = True
        D = tuple(x.strip() for x in a[0][1:-1].split(','))
        Aregs = tuple(x.strip() for x in a[1][1:-1].split(','))
        Bregs = [x.strip() for x in a[2][1:-1].split(',')]
        Cregs = tuple(x.strip() for x in a[3][1:-1].split(','))
        u = lambda r: self.val(r, 'u32')
        k16 = shape == 'm16n8k16'
        pD = self.reg_pair.get(D[0]) if self.reg_pair.get(D[0]) is not None and self.chain_pairs[self.reg_pair[D[0]]] == D else None
        pC = self.reg_pair.get(Cregs[0]) if self.reg_pair.get(Cregs[0]) is not None and self.chain_pairs[self.reg_pair[Cregs[0]]] == Cregs else None
        if m1 is not None:
            # paired: M1 (columns 0-7) and this mma (columns 8-15) share A; chains are stored as one 16x16 accumulator
            D1, _A1, B1, C1 = self.mma_regs(m1)
            pD = self.ph(self.chain_of(D1))
            pC = self.ph(self.chain_of(C1)) if self.chain_of(C1) is not None else None
            Cregs_hi, Cregs = Cregs, C1
            Bregs_hi, Bregs = Bregs, list(B1)
            D_hi, D = D, D1
        elif pD is not None or pC is not None:
            pD = self.ph(pD) if pD is not None else None
            pC = self.ph(pC) if pC is not None else None
        fusedB = None
        if k16 and self.fuse and m1 is None:
            sb = [self.gld_src.get(r) for r in Bregs]
            if len(Bregs) == 2 and all(x is not None for x in sb) and sb[0][0] == sb[1][0]:
                base = sb[0][0]
                if base not in self.quad_cache:
                    # NVIDIA's B fragment: lane 4n+t holds B[2t..2t+1][n] (b0) and B[8+2t..][n] (b1). When the four
                    # lanes of every quad load 4 consecutive words, WMMA lane l (column n = l&15, K 8*(l>>4)..+7)
                    # can load those 16 bytes itself from the address lane 4n used. Checked at run time.
                    qa, qk = self.newtmp(), self.newtmp()
                    bv = self.rname(base)
                    self.line('uint64_t %s = packUint2x32(uvec2(subgroupShuffle(unpackUint2x32(%s).x, 4u * (gl_SubgroupInvocationID & 7u)), subgroupShuffle(unpackUint2x32(%s).y, 4u * (gl_SubgroupInvocationID & 7u))));' % (qa, bv, bv))
                    self.line('bool %s = subgroupAll(%s == packUint2x32(uvec2(subgroupShuffle(unpackUint2x32(%s).x, gl_SubgroupInvocationID & ~3u), subgroupShuffle(unpackUint2x32(%s).y, gl_SubgroupInvocationID & ~3u))) + 4ul * uint64_t(gl_SubgroupInvocationID & 3u));' % (qk, bv, bv, bv))
                    self.quad_cache[base] = (qa, qk)
                fusedB = (self.quad_cache[base], sb[0][1], sb[1][1])
        self.line('{ // %s%s' % (ins.text, ('  + paired with: ' + m1.text) if m1 is not None else ''))
        self.ind += 1
        # Operands A and B are built directly in RDNA4's WMMA register layout (RADV, GFX12, wave32: lane l holds
        # row/column l&15 and K = 8*(l>>4)+i for element i) from NVIDIA's fragments with subgroup shuffles.
        # Accumulator chains stay in WMMA registers (rounded to fp16 after every mma, as NVIDIA's fp16 D);
        # other C/D values go through per-subgroup LDS scratch.
        self.line('uint g_ = gl_SubgroupInvocationID >> 2, t_ = gl_SubgroupInvocationID & 3u;')
        self.line('uint l_ = gl_SubgroupInvocationID, m_ = l_ & 15u, hi_ = l_ >> 4;')
        self.line('uint sb_ = NVX_SCRATCH_BASE + gl_SubgroupID * NVX_SSTRIDE;')
        self.line('coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mb_;')
        self.line('coopmat<%s, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mc_;' % ('float16_t' if self.f16acc else 'float'))
        srcs = [self.ldm_src.get(r) for r in Aregs] if k16 else None
        fusedA = None
        if srcs and all(x is not None for x in srcs) and len(set(x[0] for x in srcs)) == 1 \
                and [x[1] for x in srcs] == [0, 1, 2, 3] and self.idom is not None \
                and dominates(self.idom, srcs[0][2], self.cur_block):
            fusedA = srcs[0][0]
        if self.last_a != (Aregs, k16) and fusedA:
            self.uses_smv = True
            self.line('{ uint ra_ = subgroupShuffle(%s, 8u * ((m_ >> 3) + 2u * hi_) + (m_ & 7u)); uvec4 q_ = smv[ra_ >> 4];' % fusedA)
            for t, comp in enumerate('xyzw'):
                self.line('  ma_c[%d] = uint16BitsToFloat16(uint16_t(q_.%s & 0xffffu)); ma_c[%d] = uint16BitsToFloat16(uint16_t(q_.%s >> 16));' % (2 * t, comp, 2 * t + 1, comp))
            self.line('}')
            self.last_a = (Aregs, k16)
        elif self.last_a != (Aregs, k16) and k16 and self.lds_xpose:
            # A fragment through the subgroup's scratch: lane (g,t) writes a0..a3 to the row-major 16x16 tile
            # (8 words per row), WMMA lane l reads its row's 16 bytes for K 8*(l>>4).. in one load
            self.uses_smv = True
            self.uses_scr_a = True
            self.line('subgroupBarrier(); smem[sb_ + NVX_SA + g_ * 8u + t_] = %s; smem[sb_ + NVX_SA + (g_ + 8u) * 8u + t_] = %s; smem[sb_ + NVX_SA + g_ * 8u + t_ + 4u] = %s; smem[sb_ + NVX_SA + (g_ + 8u) * 8u + t_ + 4u] = %s; subgroupBarrier();' % (u(Aregs[0]), u(Aregs[1]), u(Aregs[2]), u(Aregs[3])))
            if self.xpose_load:
                self.line('coopMatLoad(ma_c, smh, (sb_ + NVX_SA) * 2u, 16u, gl_CooperativeMatrixLayoutRowMajor);')
            else:
                self.line('{ uvec4 q_ = smv[(sb_ + NVX_SA + m_ * 8u + hi_ * 4u) >> 2];')
                for t, comp in enumerate('xyzw'):
                    self.line('  ma_c[%d] = uint16BitsToFloat16(uint16_t(q_.%s & 0xffffu)); ma_c[%d] = uint16BitsToFloat16(uint16_t(q_.%s >> 16));' % (2 * t, comp, 2 * t + 1, comp))
                self.line('}')
            self.last_a = (Aregs, k16)
        elif self.last_a != (Aregs, k16) and self.layout == 'gfx11':
            # gfx11: element 2t'/2t'+1 of lane l is A[l&15][2t'..2t'+1]; K 0-7 from a0/a1, K 8-15 from a2/a3
            for t in range(4):
                if k16:
                    self.line('{ uint s_ = 4u * (m_ & 7u) + %du; uint x0_ = subgroupShuffle(%s, s_), x1_ = subgroupShuffle(%s, s_), x2_ = subgroupShuffle(%s, s_), x3_ = subgroupShuffle(%s, s_);' % (t, u(Aregs[0]), u(Aregs[1]), u(Aregs[2]), u(Aregs[3])))
                    self.line('  uint v_ = m_ < 8u ? x0_ : x1_, w_ = m_ < 8u ? x2_ : x3_;')
                else:
                    self.line('{ uint s_ = 4u * (m_ & 7u) + %du; uint x0_ = subgroupShuffle(%s, s_), x1_ = subgroupShuffle(%s, s_);' % (t, u(Aregs[0]), u(Aregs[1])))
                    self.line('  uint v_ = m_ < 8u ? x0_ : x1_, w_ = 0u;')
                self.line('  ma_c[%d] = uint16BitsToFloat16(uint16_t(v_ & 0xffffu)); ma_c[%d] = uint16BitsToFloat16(uint16_t(v_ >> 16));' % (2 * t, 2 * t + 1))
                self.line('  ma_c[%d] = uint16BitsToFloat16(uint16_t(w_ & 0xffffu)); ma_c[%d] = uint16BitsToFloat16(uint16_t(w_ >> 16)); }' % (8 + 2 * t, 9 + 2 * t))
            self.last_a = (Aregs, k16)
        elif self.last_a != (Aregs, k16):
            for t in range(4):
                if k16:
                    self.line('{ uint s_ = 4u * (m_ & 7u) + %du; uint x0_ = subgroupShuffle(%s, s_), x1_ = subgroupShuffle(%s, s_), x2_ = subgroupShuffle(%s, s_), x3_ = subgroupShuffle(%s, s_);' % (t, u(Aregs[0]), u(Aregs[1]), u(Aregs[2]), u(Aregs[3])))
                    self.line('  uint v_ = hi_ == 0u ? (m_ < 8u ? x0_ : x1_) : (m_ < 8u ? x2_ : x3_);')
                else:
                    self.line('{ uint s_ = 4u * (m_ & 7u) + %du; uint x0_ = subgroupShuffle(%s, s_), x1_ = subgroupShuffle(%s, s_);' % (t, u(Aregs[0]), u(Aregs[1])))
                    self.line('  uint v_ = hi_ == 0u ? (m_ < 8u ? x0_ : x1_) : 0u;')
                self.line('  ma_c[%d] = uint16BitsToFloat16(uint16_t(v_ & 0xffffu)); ma_c[%d] = uint16BitsToFloat16(uint16_t(v_ >> 16)); }' % (2 * t, 2 * t + 1))
            self.last_a = (Aregs, k16)
        if fusedB:
            (qa, qk), o0, o1 = fusedB
            self.line('if (%s) {' % qk)
            self.line('  uvec4 w_ = m_ < 8u ? NvxU32x4a4(%s + (hi_ != 0u ? %dul : %dul)).v : uvec4(0u);' % (qa, o1 & 0xffffffffffffffff, o0 & 0xffffffffffffffff))
            for t, comp in enumerate('xyzw'):
                self.line('  mb_[%d] = uint16BitsToFloat16(uint16_t(w_.%s & 0xffffu)); mb_[%d] = uint16BitsToFloat16(uint16_t(w_.%s >> 16));' % (2 * t, comp, 2 * t + 1, comp))
            self.line('} else {')
            self.ind += 1
        bx = k16 and self.lds_xpose and not fusedB
        if bx:
            # B fragment through scratch: lane L writes b0/b1 to words 128 + 32*kh + L; WMMA lane l (column n = l&15,
            # K 8*kh..) reads the four words of quad n, i.e. 16 bytes (columns 8..15 are zero)
            self.uses_smv = True
            if self.xpose_load and m1 is not None:
                # paired: columns 0-7 from the first mma's B, columns 8-15 from this one's
                self.line('subgroupBarrier(); smem[sb_ + NVX_SB + 8u * g_ + t_] = %s; smem[sb_ + NVX_SB + 4u + 8u * g_ + t_] = %s; smem[sb_ + NVX_SB + 64u + 8u * g_ + t_] = %s; smem[sb_ + NVX_SB + 68u + 8u * g_ + t_] = %s; subgroupBarrier();' % (u(Bregs[0]), u(Bregs[1]), u(Bregs_hi[0]), u(Bregs_hi[1])))
                self.line('coopMatLoad(mb_, smh, (sb_ + NVX_SB) * 2u, 16u, gl_CooperativeMatrixLayoutColumnMajor);')
            elif self.xpose_load:
                # column-major 16x16 tile at words 128..255: column n = 8 words (K 0..15); columns 8..15 stay zero
                # (unless a paired mma left B data there: columns are independent, so they never reach columns 0-7)
                self.need_bzero = True
                self.line('subgroupBarrier(); smem[sb_ + NVX_SB + 8u * g_ + t_] = %s; smem[sb_ + NVX_SB + 4u + 8u * g_ + t_] = %s; subgroupBarrier();' % (u(Bregs[0]), u(Bregs[1])))
                self.line('coopMatLoad(mb_, smh, (sb_ + NVX_SB) * 2u, 16u, gl_CooperativeMatrixLayoutColumnMajor);')
            else:
                self.line('subgroupBarrier(); smem[sb_ + NVX_SB + l_] = %s; smem[sb_ + NVX_SB + 32u + l_] = %s; subgroupBarrier();' % (u(Bregs[0]), u(Bregs[1])))
                self.line('{ uvec4 w_ = smv[(sb_ + NVX_SB + hi_ * 32u + 4u * (m_ & 7u)) >> 2]; if (m_ >= 8u) w_ = uvec4(0u);')
                for t, comp in enumerate('xyzw'):
                    self.line('  mb_[%d] = uint16BitsToFloat16(uint16_t(w_.%s & 0xffffu)); mb_[%d] = uint16BitsToFloat16(uint16_t(w_.%s >> 16));' % (2 * t, comp, 2 * t + 1, comp))
                self.line('}')
        for t in range(4 if not bx and self.layout == 'gfx11' else 0):
            # gfx11: element 2t'/2t'+1 of lane l is B[2t'..2t'+1][l&15]; K 0-7 from b0, K 8-15 from b1; columns 8-15 zero
            if k16:
                self.line('{ uint s_ = 4u * (m_ & 7u) + %du; uint y0_ = subgroupShuffle(%s, s_), y1_ = subgroupShuffle(%s, s_);' % (t, u(Bregs[0]), u(Bregs[1])))
                self.line('  uint v_ = m_ >= 8u ? 0u : y0_, w_ = m_ >= 8u ? 0u : y1_;')
            else:
                self.line('{ uint s_ = 4u * (m_ & 7u) + %du; uint y0_ = subgroupShuffle(%s, s_);' % (t, u(Bregs[0])))
                self.line('  uint v_ = m_ >= 8u ? 0u : y0_, w_ = 0u;')
            self.line('  mb_[%d] = uint16BitsToFloat16(uint16_t(v_ & 0xffffu)); mb_[%d] = uint16BitsToFloat16(uint16_t(v_ >> 16));' % (2 * t, 2 * t + 1))
            self.line('  mb_[%d] = uint16BitsToFloat16(uint16_t(w_ & 0xffffu)); mb_[%d] = uint16BitsToFloat16(uint16_t(w_ >> 16)); }' % (8 + 2 * t, 9 + 2 * t))
        for t in range(4 if not bx and self.layout == 'gfx12' else 0):
            if k16:
                self.line('{ uint s_ = 4u * (m_ & 7u) + %du; uint y0_ = subgroupShuffle(%s, s_), y1_ = subgroupShuffle(%s, s_);' % (t, u(Bregs[0]), u(Bregs[1])))
                self.line('  uint v_ = m_ >= 8u ? 0u : (hi_ == 0u ? y0_ : y1_);')
            else:
                self.line('{ uint s_ = 4u * (m_ & 7u) + %du; uint y0_ = subgroupShuffle(%s, s_);' % (t, u(Bregs[0])))
                self.line('  uint v_ = (m_ >= 8u || hi_ != 0u) ? 0u : y0_;')
            self.line('  mb_[%d] = uint16BitsToFloat16(uint16_t(v_ & 0xffffu)); mb_[%d] = uint16BitsToFloat16(uint16_t(v_ >> 16)); }' % (2 * t, 2 * t + 1))
        if fusedB:
            self.ind -= 1
            self.line('}')
        # C enters through the subgroup's fp16 staging tile (row-major 16x16, 8 words per row): fragment c0 is
        # row g, columns 2t..2t+1, c1 row g+8; a paired mma's second C goes to columns 8..15
        cw = 'smem[sb_ + NVX_SC + g_ * 8u + t_] = %s; smem[sb_ + NVX_SC + (g_ + 8u) * 8u + t_] = %s; ' % (u(Cregs[0]), u(Cregs[1]))
        if m1 is not None:
            cw += 'smem[sb_ + NVX_SC + g_ * 8u + 4u + t_] = %s; smem[sb_ + NVX_SC + (g_ + 8u) * 8u + 4u + t_] = %s; ' % (u(Cregs_hi[0]), u(Cregs_hi[1]))
        if self.f16acc:
            lds_c = 'subgroupBarrier(); ' + cw + 'subgroupBarrier(); coopMatLoad(mc_, smh, (sb_ + NVX_SC) * 2u, 16u, gl_CooperativeMatrixLayoutRowMajor);'
        else:
            lds_c = ('subgroupBarrier(); ' + cw + 'subgroupBarrier(); '
                     '{ coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c16_; '
                     'coopMatLoad(c16_, smh, (sb_ + NVX_SC) * 2u, 16u, gl_CooperativeMatrixLayoutRowMajor); '
                     'mc_ = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(c16_); }')
        if pC is not None:
            self.line('if (shv_%d) { mc_ = acc_%d; } else { %s }' % (pC, pC, lds_c))
        else:
            self.line(lds_c)
        self.line('mc_ = coopMatMulAdd(ma_c, mb_, mc_);')
        # NVIDIA's D is fp16: round every element (RNE) in place; the opaque zero keeps the compiler from folding
        # the f32->f16->f32 round trip. (f16acc: the WMMA itself produces fp16 - only valid where proven identical)
        if not self.f16acc:
            self.line('{ uint z_ = nvx_opaque_zero(); for (int i_ = 0; i_ < mc_.length(); i_ += 2) { uint w_ = packFloat2x16(f16vec2(float16_t(mc_[i_]), float16_t(mc_[i_ + 1]))) ^ z_; f16vec2 h_ = unpackFloat2x16(w_); mc_[i_] = float(h_.x); mc_[i_ + 1] = float(h_.y); } }')
        if pD is not None:
            self.line('acc_%d = mc_; shv_%d = true; regv_%d = false;' % (pD, pD, pD))
        else:
            st = 'mc_' if self.f16acc else 'coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(mc_)'
            self.line('subgroupBarrier(); coopMatStore(%s, smh, (sb_ + NVX_SC) * 2u, 16u, gl_CooperativeMatrixLayoutRowMajor); subgroupBarrier();' % st)
            self.line('uint d0_ = smem[sb_ + NVX_SC + g_ * 8u + t_];')
            self.line('uint d1_ = smem[sb_ + NVX_SC + (g_ + 8u) * 8u + t_];')
            self.assign(D[0], 'd0_', 'u32')
            self.assign(D[1], 'd1_', 'u32')
            self.line('subgroupBarrier();')
        # a D register that is also part of the cached A tuple invalidates it
        if self.last_a and (set(D) | (set(D_hi) if m1 is not None else set())) & set(self.last_a[0]):
            self.last_a = None
        self.ind -= 1
        self.line('}')

    def ldmatrix(self, ins, mods):
        ms = set(mods)
        if 'trans' in ms or 'x4' not in ms or 'm8n8' not in ms:
            raise Err('ldmatrix form ' + ins.text)
        a = ins.args
        D = [x.strip() for x in a[0][1:-1].split(',')]
        ad = self.addr_expr32(a[1])
        tmp = self.newtmp()
        if self.fuse_a and ins.pred is None:
            self.fn_decls.append('uint %s = 0u;' % tmp)
            self.line('%s = %s;' % (tmp, ad))
        else:
            self.line('uint %s = %s;' % (tmp, ad))
        for j, d in enumerate(D):
            self.ldm_src.pop(d, None)
        for j, d in enumerate(D):
            self.assign(d, 'smem[(subgroupShuffle(%s, %du + (gl_SubgroupInvocationID >> 2)) >> 2) + (gl_SubgroupInvocationID & 3u)]' % (tmp, 8 * j), 'u32')
        # remember where the four 8x8 matrices came from: an mma that takes them as its A operand can load its WMMA
        # fragment straight from shared memory (lane l: row l&15, K 8*(l>>4)..+7 = one 16-byte row of matrix
        # j = (row >= 8) + 2*(l >> 4), whose address lane 8j + (row & 7) holds). Unpredicated only (scope).
        if self.fuse_a and ins.pred is None and len(set(D)) == 4:
            for j, d in enumerate(D):
                self.ldm_src[d] = (tmp, j, self.cur_block)


# ------------------------------------------------------------------------------------------- structurizer
class Structurizer:
    """Reducible CFG -> nested do{}while(false) blocks and while(true) loops (Ramsey 2022, with loop exits placed
    after their loop so code after a divergent loop is not re-executed per iteration and barriers stay uniform)."""
    def __init__(self, em, blocks):
        self.em = em
        self.blocks = blocks
        self.rpo, self.num, self.preds, self.idom = dominators(blocks)
        em.idom = self.idom
        self.fwd_preds = {}
        self.loop_header = set()
        backsrc = {}
        for v in self.rpo:
            fp = []
            for p in self.preds[v]:
                if self.num[p] >= self.num[v]:
                    if not dominates(self.idom, v, p):
                        raise Err('irreducible control flow')
                    self.loop_header.add(v)
                    backsrc.setdefault(v, []).append(p)
                else:
                    fp.append(p)
            self.fwd_preds[v] = fp
        # natural loops
        self.loop = {}
        for h in self.loop_header:
            body = {h}
            work = [p for p in backsrc[h]]
            while work:
                n = work.pop()
                if n in body:
                    continue
                body.add(n)
                for p in self.preds[n]:
                    if p not in body:
                        work.append(p)
            self.loop[h] = body
        self.merge = {v for v in self.rpo if len(self.fwd_preds[v]) >= 2}
        # placement parent: idom, but loop-exit nodes move up to the header of the outermost loop that contains
        # their idom and not themselves
        self.exit_nodes = set()
        self.place = {v: [] for v in self.rpo}
        for v in self.rpo[1:]:
            par = self.idom[v]
            best = None
            for h in self.loop_header:
                L = self.loop[h]
                if par in L and v not in L:
                    if best is None or h in self.loop and self.loop[h] >= self.loop[best]:
                        if best is None or len(self.loop[h]) > len(self.loop[best]):
                            best = h
            if best is not None:
                par = best
                self.exit_nodes.add(v)
            self.place[par].append(v)
        self.follow = self.merge | self.exit_nodes
        self.cid = 0
        self.need_check = set()

    def run(self):
        self.do_tree(self.rpo[0], [])

    # ctx entries: ('loop', header, id) | ('block', follow, id) | ('if',)
    def do_tree(self, x, ctx):
        kids = self.place[x]
        if x in self.loop_header:
            L = self.loop[x]
            inside = sorted([c for c in kids if c in self.merge and c in L], key=lambda c: -self.num[c])
            exits = sorted([c for c in kids if c not in L], key=lambda c: -self.num[c])
            self.wrap_exits(x, exits, inside, ctx)
        else:
            merges = sorted([c for c in kids if c in self.follow], key=lambda c: -self.num[c])
            self.node_within(x, merges, ctx)

    def wrap_exits(self, x, exits, inside, ctx):
        em = self.em
        if exits:
            y, rest = exits[0], exits[1:]
            self.cid += 1
            cid = self.cid
            em.line('do { // block -> B%d (loop exit)' % y)
            em.ind += 1
            self.wrap_exits(x, rest, inside, [('block', y, cid)] + ctx)
            em.ind -= 1
            em.line('} while (false);')
            self.after_construct(ctx, cid)
            self.do_tree(y, ctx)
            return
        self.cid += 1
        cid = self.cid
        em.line('while (true) { // loop L%d' % x)
        em.ind += 1
        self.node_within(x, inside, [('loop', x, cid)] + ctx)
        em.line('break;')
        em.ind -= 1
        em.line('}')
        self.after_construct(ctx, cid)

    def node_within(self, x, ys, ctx):
        em = self.em
        if ys:
            y, rest = ys[0], ys[1:]
            self.cid += 1
            cid = self.cid
            em.line('do { // block -> B%d' % y)
            em.ind += 1
            self.node_within(x, rest, [('block', y, cid)] + ctx)
            em.ind -= 1
            em.line('} while (false);')
            self.after_construct(ctx, cid)
            self.do_tree(y, ctx)
            return
        b = self.blocks[x]
        em.last_a = None
        em.cur_block = x
        em.gld_src = {}
        em.quad_cache = {}
        for ins in b.instrs:
            em.emit_instr(ins)
        t = b.term
        if t[0] == 'ret':
            em.line('return;')
        elif t[0] in ('br', 'fall'):
            self.do_branch(x, t[1], ctx)
        elif t[0] == 'cbr':
            p = em.rname(t[1])
            em.line('if (%s%s) {' % ('!' if t[2] else '', p))
            em.ind += 1
            self.do_branch(x, t[3], [('if',)] + ctx)
            em.ind -= 1
            em.line('} else {')
            em.ind += 1
            self.do_branch(x, t[4], [('if',)] + ctx)
            em.ind -= 1
            em.line('}')
        elif t[0] == 'cbr_ret':
            p = em.rname(t[1])
            em.line('if (%s%s) { return; }' % ('!' if t[2] else '', p))
            self.do_branch(x, t[3], ctx)
        elif t[0] == 'cbr_retend':
            em.line('return;')
        else:
            raise Err('terminator ' + str(t))

    def do_branch(self, src, tgt, ctx):
        if self.num[tgt] <= self.num[src]:
            self.jump(ctx, 'loop', tgt)
        elif tgt in self.follow:
            self.jump(ctx, 'block', tgt)
        else:
            self.do_tree(tgt, ctx)

    def jump(self, ctx, kind, tgt):
        em = self.em
        inner = None
        for c in ctx:
            if c[0] in ('loop', 'block'):
                inner = c
                break
        target = None
        for c in ctx:
            if c[0] == kind and c[1] == tgt:
                target = c
                break
        if target is None:
            raise Err('branch target %s B%d not in context' % (kind, tgt))
        if target is inner:
            em.line('continue;' if kind == 'loop' else 'break;')
            return
        looplike = [c for c in ctx if c[0] in ('loop', 'block')]
        if kind == 'block' and inner[0] == 'loop' and len(looplike) > 1 and looplike[1] is target:
            # leaving the loop lands at the end of the block that directly encloses it
            em.line('break;')
            return
        code = target[2] * 2 + (1 if kind == 'loop' else 0)
        em.jt_used = True
        em.line('_jt = %d; break;' % code)
        for c in ctx:
            if c is target:
                break
            if c[0] in ('loop', 'block'):
                self.need_check.add(c[2])

    def after_construct(self, ctx, cid):
        """emitted right after construct cid closes, inside ctx"""
        if cid not in self.need_check:
            return
        em = self.em
        inner = None
        for c in ctx:
            if c[0] in ('loop', 'block'):
                inner = c
                break
        em.line('if (_jt != 0) {')
        em.ind += 1
        if inner is not None:
            if inner[0] == 'loop':
                em.line('if (_jt == %d) { _jt = 0; continue; }' % (inner[2] * 2 + 1))
            else:
                em.line('if (_jt == %d) { _jt = 0; break; }' % (inner[2] * 2))
            em.line('break;')
        else:
            em.line('return; // unreachable')
        em.ind -= 1
        em.line('}')

# ------------------------------------------------------------------------------------------- driver
HEADER = '''#version 460
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_spirv_intrinsics : require
#extension GL_EXT_samplerless_texture_functions : require
spirv_execution_mode(4460, 32);
spirv_execution_mode(4459, 16);
spirv_execution_mode(4461, 32);
spirv_execution_mode(4461, 16);
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(constant_id = 3) const uint NVX_DYN_SMEM = 0u;
'''

# UI-copy shadows (NVXRT_SHADOW=c): the kernels that read or write DLSS-G's hudless/UI copies (kept as RGBA8 shadows by the layer) get
# the class-3 read fix-up and the checked store (NVX_SHADOW8; meta 'shadow8 1'). The layer redirects a class-3 read only
# for kernels built with it, so this list only decides where UI-copy shadows can apply, never exactness.
SHADOW8_ENTRIES = ('Kernel_DownsampleRGBAndEstimateUIFused', 'Kernel_BlendCandidatesFused')

def translate(ptx, entry, opts=None):
    opts = dict(opts or {})
    import os as _os
    if _os.environ.get('PTXC_NO_CHAIN'):
        opts['no_chain'] = True
    if _os.environ.get('PTXC_NO_FUSE'):
        opts['no_fuse'] = True
    if _os.environ.get('PTXC_NO_FUSE_A'):
        opts['no_fuse_a'] = True
    if _os.environ.get('PTXC_NO_FUSE_B'):
        opts['no_fuse_b'] = True
    if _os.environ.get('PTXC_FUSE_B'):
        opts['fuse_b'] = True
    if _os.environ.get('PTXC_NO_XPOSE'):
        opts['no_xpose'] = True
    if _os.environ.get('PTXC_F16ACC'):
        opts['f16acc'] = True
    if _os.environ.get('PTXC_NO_XPOSE_LOAD'):
        opts['no_xpose_load'] = True
    if _os.environ.get('PTXC_SHFL_K'):
        opts['shfl_k'] = True
    if _os.environ.get('PTXC_NO_PAIR'):
        opts['no_pair'] = True
    if _os.environ.get('PTXC_NO_HWCVT'):
        opts['no_hwcvt'] = True
    if _os.environ.get('PTXC_NO_ADDR32'):
        opts['no_addr32'] = True
    if _os.environ.get('PTXC_SW_CLAMP'):
        opts['sw_clamp'] = True
    if _os.environ.get('PTXC_LAYOUT'):
        opts['layout'] = _os.environ['PTXC_LAYOUT']
    if _os.environ.get('PTXC_NO_GATHER'):
        opts['no_gather'] = True
    e = parse_entry(ptx, entry)
    # f32 arithmetic must be .ftz everywhere (the shader flushes fp32 denormals)
    for kind, ins in e.stmts:
        if kind != 'instr':
            continue
        if 'f32' in ins.mods and ins.op in ('add', 'sub', 'mul', 'fma', 'mad', 'div', 'rcp', 'sqrt', 'rsqrt', 'ex2', 'lg2', 'min', 'max', 'abs', 'neg', 'setp') and 'ftz' not in ins.mods:
            if not (ins.op == 'div' and 'rn' in ins.mods):
                raise Err('non-ftz f32 arithmetic (would need denormal preservation): ' + ins.text)
    blocks = build_cfg(e)
    em = Emitter(e, opts)
    st = Structurizer(em, blocks)
    st.run()
    body = em.out
    out = [HEADER]
    static_words = max(1, e.shared_bytes // 4)
    if em.uses_mma:
        out.append('layout(constant_id = 4) const uint NVX_NSUB = 12u;  // subgroups per workgroup, set by the layer')
        sbase = (e.shared_bytes + 15) // 16 * 4   # per-subgroup scratch starts 16-byte aligned (uvec4 accesses)
        out.append('#define NVX_SCRATCH_BASE %du' % sbase)
        # per-subgroup scratch: [A transpose 128 words if used][B tile 128][C/D fp16 tile 128]
        sa = 0
        sb = 128 if em.uses_scr_a else 0
        sc = sb + 128
        stride = sc + 128
        out.append('#define NVX_SA %du' % sa)
        out.append('#define NVX_SB %du' % sb)
        out.append('#define NVX_SC %du' % sc)
        out.append('#define NVX_SSTRIDE %du' % stride)
        # typed views of the same workgroup memory (GL_EXT_shared_memory_block aliases all blocks): RADV loads
        # cooperative matrices correctly only from arrays of the matrix component type
        words = '(%d + NVX_NSUB * NVX_SSTRIDE)' % sbase
        out.append('#extension GL_EXT_shared_memory_block : require')
        out.append('shared NvxSMU { uint smem[%s]; };' % words)
        out.append('shared NvxSMH { float16_t smh[%s * 2u]; };' % words)
        out.append('shared NvxSMF { float smf[%s]; };' % words)
        out.append('shared NvxSMV { uvec4 smv[%du + NVX_NSUB * (NVX_SSTRIDE / 4u)]; };' % (sbase // 4))
    else:
        out.append('shared uint smem[%d];' % static_words)
    if opts.get('sw_clamp'):
        out.append('#define NVX_SW_CLAMP')   # texel fetches clamp in the shader instead of the companion samplers
    if e.param_size > 248:
        out.append('#define NVX_BIG_PARAMS')   # the layer copies the block into a buffer; its address is in pc.w[60..61]
    if entry in SHADOW8_ENTRIES:
        out.append('#define NVX_SHADOW8')    # UI-copy shadows: fix-up for RGBA8 shadow reads, checked shadow stores
    out.append(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rt.glsl')).read())
    out.append('void main() {')
    if e.local_bytes:
        out.append('    uint lmem[%d];' % (e.local_bytes // 4))
        out.append('    for (int i_ = 0; i_ < %d; i_++) lmem[i_] = 0u;' % (e.local_bytes // 4))
    if em.jt_used:
        out.append('    int _jt = 0;')
    for d in em.fn_decls:
        out.append('    ' + d)
    if em.need_bzero:
        # columns 8..15 of the B staging tile are never written: zero them once (each lane 2 words of its subgroup)
        out.append('    { uint sbz_ = NVX_SCRATCH_BASE + gl_SubgroupID * NVX_SSTRIDE + NVX_SB + 64u; smem[sbz_ + gl_SubgroupInvocationID] = 0u; smem[sbz_ + 32u + gl_SubgroupInvocationID] = 0u; subgroupBarrier(); }')
    if em.uses_mma:
        out.append('    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ma_c;')
        for i in range(len(em.chain_pairs)):
            out.append('    coopmat<%s, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc_%d; bool shv_%d = false; bool regv_%d = true;' % ('float16_t' if em.f16acc else 'float', i, i, i))
    init = {'bool': 'false', 'uint16_t': 'uint16_t(0)', 'uint': '0u', 'float': '0.0', 'uint64_t': '0ul', 'double': '0.0lf'}
    for r, t in sorted(e.regs.items()):
        gt = GT[t]
        if r in ('%SP', '%SPL'):
            pass
        q = 'precise ' if gt in ('float', 'double') else ''
        out.append('    %s%s %s = %s;' % (q, gt, em.rname(r), init[gt]))
    out.extend(body)
    out.append('}')
    meta = ['entry %s' % entry, 'param_size %d' % e.param_size]
    for nm, off, sz in e.params:
        meta.append('param %d %d' % (off, sz))
    meta.append('shared_bytes %d' % e.shared_bytes)
    meta.append('local_bytes %d' % e.local_bytes)
    meta.append('mma %d' % (1 if em.uses_mma else 0))
    if e.param_size > 248:
        meta.append('big_params 1')
    if e.maxntid:
        meta.append('maxntid %d %d %d' % tuple(e.maxntid))
    return '\n'.join(out) + '\n', '\n'.join(meta) + '\n'

def _rt_text():
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rt.glsl')).read()

def uses_filtered_tex(glsl):
    """True if the kernel calls the filtered tex.2d helpers (call sites beyond rt.glsl's own definitions)"""
    rt = _rt_text()
    n = lambda t: t.count('nvx_tex2d_m(') + t.count('nvx_tex2d(')
    return n(glsl) > n(rt)

def hw_filter_source(glsl):
    """the fast-mode variant of a kernel: rt.glsl's filtered fetches use the texture unit (NVX_HW_FILTER)"""
    first, rest = glsl.split('\n', 1)
    assert first.startswith('#version')
    return first + '\n#define NVX_HW_FILTER\n' + rest

# Kernels whose fast-filtering variant pays (measured 2026-10-22, harness 4x, per launch: BlendCandidatesFused
# -0.043 ms, DL1Net_Input -0.004, WarpBlendingWeights -0.002; the other filtered-tap kernels gain < 0.002 ms, so they
# keep the exact model and add no image difference).
HW_FILTER_ENTRIES = ('Kernel_BlendCandidatesFused', 'Kernel_BlendCandidatesFGLite', 'Kernel_WarpBlendingWeights',
                     'Kernel_DL1Net_Input')

_HU_LOAD = re.compile(r'\b(r_\w+) = nvx_p(64|32|16|8)\((?:uint\((r_\w+)\) \+ )?(\d+)u\)')
_HU_DYN = re.compile(r'nvx_p(?:64|32|16|8)\((?!(?:uint\(r_\w+\) \+ )?\d+u\))')
_HU_COPY = re.compile(r'\b(r_\w+) = (r_\w+);')
_HU_CALL = re.compile(r'\b(nvx_tex\w*|nvx_utex\w*|nvx_sust\w*|nvx_suld\w*)\((r_\w+)\b')


def handle_meta(glsl):
    """meta lines for the layer's handle rewriting (NVXRT_SHADOW b/c): per 8-byte parameter slot the kernel reads, how the
    value is used - 'huse OFF tex' (only as the image of texture reads), 'huse OFF surf' (surface loads/stores, maybe
    also texture reads), 'huse OFF other' (anything else, or partial loads); slots never loaded get no line.
    'huse_dynamic 1' when a parameter is read at a run-time offset (then every slot counts as other)."""
    rt = _rt_text()
    i = glsl.find(rt)
    body = glsl[i + len(rt):] if i >= 0 else glsl[max(0, glsl.find('void main()')):]
    if _HU_DYN.search(body):
        return ['huse_dynamic 1']
    # ld.param through a register (param_base): the register holds a parameter's byte offset, which the slot is
    # relative to. Only a register assigned one constant (directly or through register copies) gives static slots.
    def const_reg(r, depth=0):
        vals = set(re.findall(r'^\s*%s = ([^;]*);' % r, body, re.M))
        if len(vals) != 1 or depth > 16:
            return None
        v = vals.pop()
        m = re.match(r'^(\d+)ul?$', v)
        if m:
            return int(m.group(1))
        return const_reg(v, depth + 1) if re.match(r'^r_\w+$', v) else None
    base_off = {}
    for b in set(b for v, n, b, o in _HU_LOAD.findall(body) if b):
        base_off[b] = const_reg(b)
        if base_off[b] is None:
            return ['huse_dynamic 1']
    var_offs, partial = {}, {}
    for v, n, b, o in _HU_LOAD.findall(body):
        o = int(o) + base_off.get(b, 0)
        if n == '64' and o % 8 == 0:
            var_offs.setdefault(v, set()).add(o)
        else:
            partial[o - o % 8] = True
    copies = _HU_COPY.findall(body)
    for _ in range(6):
        for d, sv in copies:
            if sv in var_offs:
                var_offs.setdefault(d, set()).update(var_offs[sv])
    kinds = {}
    for fn, v in _HU_CALL.findall(body):
        k = 'surf' if ('sust' in fn or 'suld' in fn) else 'tex'
        for o in var_offs.get(v, ()):
            kinds.setdefault(o, set()).add(k)
    # any other appearance of a handle variable (not its load, a copy, a declaration, or a tex/surf call's image)
    for v, offs in var_offs.items():
        n_all = len(re.findall(r'\b%s\b' % v, body))
        n_load = len(re.findall(r'\b%s = nvx_p' % v, body))
        n_copy = sum(1 for d, sv in copies if d == v) + sum(1 for d, sv in copies if sv == v)
        n_decl = len(re.findall(r'\b(?:uint64_t|uint) %s\b' % v, body))
        n_call = sum(1 for fn, u in _HU_CALL.findall(body) if u == v)
        if n_all != n_load + n_copy + n_decl + n_call:
            for o in offs:
                kinds.setdefault(o, set()).add('other')
    for o in partial:
        kinds.setdefault(o, set()).add('other')
    out = []
    for o in sorted(set(kinds) | set(o for offs in var_offs.values() for o in offs)):
        k = kinds.get(o, set())
        out.append('huse %d %s' % (o, 'other' if ('other' in k or not k) else ('surf' if 'surf' in k else 'tex')))
    return out


def with_handle_meta(meta_text, glsl):
    """meta text with its huse lines replaced by the ones for glsl"""
    lines = [l for l in meta_text.splitlines() if l and not l.startswith('huse') and not l.startswith('shadow8')]
    extra = ['shadow8 1'] if '\n#define NVX_SHADOW8\n' in glsl else []
    return '\n'.join(lines + handle_meta(glsl) + extra) + '\n'


def compile_hw_variant(glsl, base):
    """writes base.hw.spv (fast filtering) for the kernels in HW_FILTER_ENTRIES that have filtered fetches; removes a
    stale one otherwise"""
    import os
    hashes = [h for h in os.environ.get('PTXC_HW_HASHES', '').split(',') if h]
    named = os.path.basename(base) in HW_FILTER_ENTRIES
    by_hash = any(os.path.basename(os.path.dirname(base)).startswith(h) for h in hashes) or 'all' in hashes
    if not (named or by_hash) or not uses_filtered_tex(glsl):
        if os.path.exists(base + '.hw.spv'):
            os.remove(base + '.hw.spv')
        return False
    open(base + '.hw.comp', 'w').write(hw_filter_source(glsl))
    compile_glsl(base + '.hw.comp', base + '.hw.spv.tmp')
    os.replace(base + '.hw.spv.tmp', base + '.hw.spv')
    return True

FLOAT_CAPS = (4464, 4465, 4466)  # DenormPreserve, DenormFlushToZero, SignedZeroInfNanPreserve

def compile_glsl(comp_path, spv_path):
    import struct, subprocess
    r = subprocess.run(['glslangValidator', '--target-env', 'vulkan1.3', '-o', spv_path + '.raw', comp_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise Err('glslang: ' + ' | '.join([l for l in r.stdout.splitlines() if 'ERROR' in l][:4]))
    data = open(spv_path + '.raw', 'rb').read()
    words = list(struct.unpack('<%dI' % (len(data) // 4), data))
    caps = []
    for c in FLOAT_CAPS:
        caps += [(2 << 16) | 17, c]
    words = words[:5] + caps + words[5:]
    open(spv_path, 'wb').write(struct.pack('<%dI' % len(words), *words))
    import os
    os.remove(spv_path + '.raw')
    r = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', '--scalar-block-layout', spv_path], capture_output=True, text=True)
    if r.returncode != 0:
        raise Err('spirv-val: ' + (r.stdout + r.stderr).strip().splitlines()[0])

def main():
    if sys.argv[1] == '--hash':
        print(module_hash(open(sys.argv[2], encoding='latin-1').read()))
        return
    ptx = open(sys.argv[1], encoding='latin-1').read()
    comp, meta = translate(ptx, sys.argv[2])
    open(sys.argv[3] + '.comp', 'w').write(comp)
    open(sys.argv[3] + '.meta', 'w').write(meta)

if __name__ == '__main__':
    try:
        main()
    except Err as ex:
        print('ptxc: error: %s' % ex, file=sys.stderr)
        sys.exit(1)
