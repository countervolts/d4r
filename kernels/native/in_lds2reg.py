#!/usr/bin/env python3
"""in_lds2reg.py IN.hip > OUT.hip: a hiluma input kernel (any variant, translated by ptx2hip.py --ifconv --sink)
with its per-thread scratch kept in registers. Generalizes in_fuse.py + in_regs.hip, which matched one variant's
register numbers and its output address formula (mvlo variants store to a different address).

The kernel stays the translated kernel: same thread ids, same arithmetic, same final stores. Only two
per-thread scratch areas change, both value-preserving:
- LDS: 4 planes x 3 floats at S + 3072 p + 4 j, where S = T * 12 + base is the thread's own slot, written in
  straight-line code before any branch, read back at fixed offsets and once as plane k (address
  (k << 8) * 12 + S) -> 12 registers + a 4-way select. The shared array is removed (12 KB LDS per block).
- the 32-byte local depot: 4 float pairs at depot + 0..28 written before any branch, read once as the pair at
  depot + 8 k -> 8 registers + a 4-way select (it compiled to scratch memory).
The plane index k must be the same register for both reads, and is proven to lie in 0..3 by evaluating its
defining expression (which may depend only on workgroup/workitem ids and constants) for every thread of
grids up to 65536 x 16384 threads. Anything else is refused.
"""
import re
import sys

import numpy as np

src = open(sys.argv[1]).read()
lines = src.split('\n')
k0 = next(i for i, l in enumerate(lines) if l.startswith('extern "C" __attribute__((global))'))
assert lines[k0 + 1] == '{'
body_start = k0 + 2
first_label = next(i for i in range(body_start, len(lines)) if re.match(r'^L__BB0_\d+:;$', lines[i]))

defs = {}
for i, l in enumerate(lines):
    m = re.match(r'^    ((?:r|rd)\d+) = (.*);$', l)
    if m:
        defs.setdefault(m.group(1), []).append((i, m.group(2)))


def single(reg):
    d = defs.get(reg, [])
    assert len(d) == 1, f'{reg}: {len(d)} definitions'
    return d[0]


# ---- LDS slot S, base, plane address P = (k << 8) * 12 + S
ST = re.compile(r'^    \*\(AS3 float\*\)\(uintptr_t\)\((r\d+) \+ (\d+)u\) = \(float\)\((f\d+)\);$')
LD = re.compile(r'^    (f\d+) = \*\(AS3 float\*\)\(uintptr_t\)\((r\d+) \+ (\d+)u\);$')
stores = [(i, ST.match(l)) for i, l in enumerate(lines) if ST.match(l)]
slot = {m.group(1) for _, m in stores}
assert len(slot) == 1, slot
S = slot.pop()
_, sdef = single(S)
m = re.fullmatch(r'\(uint32_t\)\((r\d+) \* 12u \+ (r\d+)\)', sdef)
assert m, sdef
T, BASE = m.groups()
_, bdef = single(BASE)
shared = re.fullmatch(r'\(uint32_t\)\(\(uint32_t\)\(uintptr_t\)\(AS3 uint8_t\*\)(\w+)\)', bdef)
assert shared, bdef
shared_name = shared.group(1)
offsets = {}
for i, m in stores:
    assert i < first_label, f'LDS store inside control flow: {lines[i]}'
    off = int(m.group(2))
    assert off % 4 == 0 and off // 3072 < 4 and (off % 3072) // 4 < 3 and off not in offsets, lines[i]
    offsets[off] = m.group(3)
assert len(offsets) == 12, sorted(offsets)
loads = [(i, LD.match(l)) for i, l in enumerate(lines) if LD.match(l)]
plane_regs = {m.group(2) for _, m in loads if m.group(2) != S}
assert len(plane_regs) == 1, plane_regs
P = plane_regs.pop()
_, pdef = single(P)
m = re.fullmatch(r'\(uint32_t\)\((r\d+) \* 12u \+ ' + S + r'\)', pdef)
assert m, pdef
_, shdef = single(m.group(1))
m = re.fullmatch(r'\(uint32_t\)\((r\d+) << \(8u & 31\)\)', shdef)
assert m, shdef
K = m.group(1)
for i, m in loads:
    if m.group(2) == P:
        assert m.group(3) == '0', lines[i]

# ---- local depot
DBASE = {r for r, d in defs.items() if any(e == '(uint64_t)(l2_SPL + (uint64_t)0ull)' for _, e in d)}
DST = re.compile(r'^    \*\(AS5 float\*\)\(uintptr_t\)\(uint32_t\)\((rd\d+) \+ (\d+)\) = \(float\)\((f\d+)\);$')
DLD = re.compile(r'^    (f\d+) = \*\(AS5 float\*\)\(uintptr_t\)\(uint32_t\)\((rd\d+) \+ ([04])\);$')
dstores = {}
for i, l in enumerate(lines):
    m = DST.match(l)
    if m:
        assert m.group(1) in DBASE and i < first_label, l
        off = int(m.group(2))
        assert off % 4 == 0 and off < 32 and off not in dstores, l
        dstores[off] = m.group(3)
assert len(dstores) == 8, sorted(dstores)
dloads = [(i, DLD.match(l)) for i, l in enumerate(lines) if DLD.match(l)]
assert len(dloads) == 2, len(dloads)
dl = {m.group(2) for _, m in dloads}
assert len(dl) == 1, dl
DL = dl.pop()
_, ddef = single(DL)
m = re.fullmatch(r'\(uint64_t\)\((rd\d+) \+ (rd\d+)\)', ddef)
assert m and m.group(1) in DBASE, ddef
_, kdef = single(m.group(2))
assert kdef == f'(uint64_t)((int64_t)(int32_t){K} * (int64_t)(int32_t)8)', kdef
assert all(re.search(r'\bAS5\b', lines[i]) is None or i in [j for j, _ in dloads] or DST.match(lines[i])
           or 'l2_SPL' in lines[i] or '__local_depot0' in lines[i] for i in range(len(lines))), 'other depot use'
assert sum('AS3' in l for l in lines) == len(stores) + len(loads) + 1, 'other LDS use'  # + the base address

# ---- prove k in 0..3 for every thread: evaluate its expression over the thread-id range
IDS = {'workgroup_id_x': 'wgx', 'workitem_id_x': 'wix', 'workgroup_id_y': 'wgy', 'workitem_id_y': 'wiy'}


def to_np(reg, env, seen=()):
    assert reg not in seen
    if reg in env:
        return env[reg]
    _, e = single(reg)
    m = re.fullmatch(r'\(uint32_t\)\(\(uint32_t\)__builtin_amdgcn_(\w+)\(\)\)', e)
    if m:
        env[reg] = env[IDS[m.group(1)]]
        return env[reg]
    expr = e
    for r in sorted(set(re.findall(r'\br\d+\b', e)), key=len, reverse=True):
        to_np(r, env, seen + (reg,))
    expr = re.sub(r'\(uint32_t\)', '', expr)
    expr = re.sub(r'\(int32_t\)(r\d+)', r's32(\1)', expr)
    expr = re.sub(r'(\d+)u\b', r'np.uint32(\1)', expr)
    expr = re.sub(r'\(np\.uint32\((\d+)\) & np\.uint32\(31\)\)', r'np.uint32(\1)', expr)
    assert re.fullmatch(r'[\w\s().,+\-*&|<>]*', expr), e
    env[reg] = u32(eval(expr, {'np': np, 's32': s32, 'bfi32': bfi32}, env))
    return env[reg]


def u32(x):
    return np.asarray(x).astype(np.int64).astype(np.uint32) if np.asarray(x).dtype.kind == 'i' else np.asarray(x, np.uint32)


def s32(x):
    return x.view(np.int32) if isinstance(x, np.ndarray) else np.int32(x)


def bfi32(a, b, offset, width):
    mask = np.uint32(((1 << int(width)) - 1) << int(offset))
    return (u32(a) << np.uint32(offset)) & mask | (u32(b) & ~mask)


gx = np.arange(65536, dtype=np.uint32)
for gy0 in range(0, 16384, 256):
    gy = np.arange(gy0, gy0 + 256, dtype=np.uint32)[:, None]
    env = {'wgx': np.broadcast_to(gx >> 4, (256, 65536)).copy(), 'wix': np.broadcast_to(gx & 15, (256, 65536)).copy(),
           'wgy': np.broadcast_to(gy >> 4, (256, 65536)).copy(), 'wiy': np.broadcast_to(gy & 15, (256, 65536)).copy()}
    k = to_np(K, env)
    assert k.max() <= 3, f'plane index {K} reaches {k.max()} at rows {gy0}..'

# ---- emit
out = []
sel = lambda name: (f'{K} == 0u ? {name(0)} : {K} == 1u ? {name(1)} : {K} == 2u ? {name(2)} : {name(3)}')
for i, l in enumerate(lines):
    if re.match(rf'^__attribute__\(\(shared\)\).*\b{shared_name}\[', l):
        continue  # the LDS scratch array
    if i == k0 + 1:
        out.append(l)
        out.append('    float ' + ', '.join(f'shv{p}_{j}' for p in range(4) for j in range(3)) + ';')
        out.append('    float ' + ', '.join(f'dpv{p}_{j}' for p in range(4) for j in range(2)) + ';')
        continue
    if i == single(BASE)[0]:
        out.append(f'    {BASE} = 0u; // (LDS scratch kept in registers)')
        continue
    m = ST.match(l)
    if m:
        off = int(m.group(2))
        out.append(f'    shv{off // 3072}_{(off % 3072) // 4} = {m.group(3)};')
        continue
    m = LD.match(l)
    if m:
        if m.group(2) == P:
            out.append(f'    {m.group(1)} = {sel(lambda p: f"shv{p}_0")}; // {K} in 0..3 (proven)')
        else:
            off = int(m.group(3))
            assert off in offsets, f'load before store: {l}'
            out.append(f'    {m.group(1)} = shv{off // 3072}_{(off % 3072) // 4};')
        continue
    m = DST.match(l)
    if m:
        off = int(m.group(2))
        out.append(f'    dpv{off // 8}_{(off % 8) // 4} = {m.group(3)};')
        continue
    m = DLD.match(l)
    if m:
        j = int(m.group(3)) // 4
        out.append(f'    {m.group(1)} = {sel(lambda p: f"dpv{p}_{j}")};')
        continue
    out.append(l)
print(f'// Generated by kernels/native/in_lds2reg.py from {sys.argv[1].split("/")[-1]} (do not edit).')
print('\n'.join(out), end='')
