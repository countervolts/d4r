#!/usr/bin/env python3
"""in_fuse.py IN.hip > in_unit.h: the hiluma input kernel (translated by ptx2hip.py) as a device function

    in_unit(KARGP, gx, gy, out[4])

computing what thread (gx, gy) of the original launch (blocks of 16x16) writes: the 4 halves at network pixel
(gx >> 2, gy), channels 4 (gx & 3) .. 4 (gx & 3) + 3 (used by in_regs.hip).

Changes, all value-preserving:
- thread ids: workgroup/workitem ids -> gx >> 4, gx & 15, gy >> 4, gy & 15 (the kernel only uses gx = 16 wg_x + wi_x,
  gy = 16 wg_y + wi_y and gx & 1, gx & 2);
- the per-thread LDS scratch (4 planes of 3 floats at slot 12 * (16 wi_y + wi_x), written in straight-line code
  before any branch, read at fixed offsets and once at plane r14 in 0..3) -> 12 registers + a 4-way select;
- the 32-byte local depot (4 float pairs written in straight-line code, one pair read at plane r14) -> 8 registers
  + a 4-way select;
- the 4 final f16 stores -> out[0..3].
Refuses any input that does not match these patterns exactly.
"""
import re
import sys

src = open(sys.argv[1]).read()
lines = src.split('\n')
k0 = next(i for i, l in enumerate(lines) if l.startswith('extern "C" __attribute__((global))'))
assert lines[k0 + 1] == '{'
end = len(lines) - 1
while lines[end].strip() != '}':
    end -= 1
body = lines[k0 + 2:end]
first_label = next(i for i, l in enumerate(body) if re.match(r'^L__BB0_\d+:;$', l))

out = []
SUBS = {
    'r39 = (uint32_t)((uint32_t)__builtin_amdgcn_workgroup_id_x());': 'r39 = gx >> 4;',
    'r41 = (uint32_t)((uint32_t)__builtin_amdgcn_workitem_id_x());': 'r41 = gx & 15u;',
    'r46 = (uint32_t)((uint32_t)__builtin_amdgcn_workgroup_id_y());': 'r46 = gy >> 4;',
    'r48 = (uint32_t)((uint32_t)__builtin_amdgcn_workitem_id_y());': 'r48 = gy & 15u;',
    'r191 = (uint32_t)((uint32_t)__builtin_amdgcn_workitem_id_y());': 'r191 = 0u;',
    'r190 = (uint32_t)((uint32_t)__builtin_amdgcn_workitem_id_x());': 'r190 = 0u;',
    'r90 = (uint32_t)((uint32_t)(uintptr_t)(AS3 uint8_t*)sharedWarpedSpaceHistoryYcocg);': 'r90 = 0u; // (LDS scratch replaced by registers)',
}
seen = set()
ST = re.compile(r'^    \*\(AS3 float\*\)\(uintptr_t\)\(r20 \+ (\d+)u\) = \(float\)\((f\d+)\);$')
LD = re.compile(r'^    (f\d+) = \*\(AS3 float\*\)\(uintptr_t\)\(r20 \+ (\d+)u\);$')
LD153 = re.compile(r'^    (f\d+) = \*\(AS3 float\*\)\(uintptr_t\)\(r153 \+ 0u\);$')
STORE = re.compile(r'^    \*\(uint16_t\*\)\(uintptr_t\)\(rd48 \+ (\d)\) = \(uint16_t\)\((rs\d+)\);$')
stored = set()
nst = 0
# the 32-byte local depot: 4 float pairs (offsets 0..28) written through rd6 / rd59 / rd61 = depot + 0 in
# straight-line code, read once as the pair at rd61 + 8 r14 (r14 in 0..3) -> 8 registers + a 4-way select
DST = re.compile(r'^    \*\(AS5 float\*\)\(uintptr_t\)\(uint32_t\)\((rd6|rd59|rd61) \+ (\d+)\) = \(float\)\((f\d+)\);$')
DLD = re.compile(r'^    (f\d+) = \*\(AS5 float\*\)\(uintptr_t\)\(uint32_t\)\(rd44 \+ ([04])\);$')
dstored = set()
for i, l in enumerate(body):
    t = l.strip()
    if t.startswith('const AS4 uint8_t* KARGP =') or t.startswith('const uint64_t KARGU ='):
        continue
    if t in SUBS:
        out.append('    ' + SUBS[t])
        seen.add(t)
        continue
    m = ST.match(l)
    if m:
        off = int(m.group(1))
        assert i < first_label and off % 4 == 0 and off // 3072 < 4 and (off % 3072) // 4 < 3, l
        out.append(f'    shv{off // 3072}_{(off % 3072) // 4} = {m.group(2)};')
        stored.add(off)
        continue
    m = LD.match(l)
    if m:
        off = int(m.group(2))
        assert off in stored, f'load before store: {l}'
        out.append(f'    {m.group(1)} = shv{off // 3072}_{(off % 3072) // 4};')
        continue
    m = LD153.match(l)
    if m:
        assert all(o in stored for o in (0, 3072, 6144, 9216)), 'plane loads need all planes stored'
        out.append(f'    {m.group(1)} = r14 == 0u ? shv0_0 : r14 == 1u ? shv1_0 : r14 == 2u ? shv2_0 : shv3_0; // r14 in 0..3')
        continue
    m = DST.match(l)
    if m:
        off = int(m.group(2))
        assert i < first_label and off % 4 == 0 and off < 32, l
        out.append(f'    dpv{off // 8}_{(off % 8) // 4} = {m.group(3)};')
        dstored.add(off)
        continue
    m = DLD.match(l)
    if m:
        assert len(dstored) == 8, 'depot read before all pairs are stored'
        j = int(m.group(2)) // 4
        out.append(f'    {m.group(1)} = r14 == 0u ? dpv0_{j} : r14 == 1u ? dpv1_{j} : r14 == 2u ? dpv2_{j} : dpv3_{j}; // r14 in 0..3')
        continue
    m = STORE.match(l)
    if m:
        out.append(f'    out[{int(m.group(1)) // 2}] = {m.group(2)};')
        nst += 1
        continue
    assert 'AS3' not in l and 'workitem_id' not in l and 'workgroup_id' not in l, f'unhandled: {l}'
    assert not re.search(r'\(rd48 \+', l), f'unhandled store: {l}'
    out.append(l)
assert len(seen) == len(SUBS), set(SUBS) - seen
assert nst == 4, nst
assert len(stored) == 12, sorted(stored)
assert len(dstored) == 8, sorted(dstored)
for k in ('    rd6 = (uint64_t)(l2_SPL + (uint64_t)0ull);', '    rd59 = (uint64_t)(l2_SPL + (uint64_t)0ull);',
          '    rd61 = (uint64_t)(l2_SPL + (uint64_t)0ull);', '    rd43 = (uint64_t)((int64_t)(int32_t)r14 * (int64_t)(int32_t)8);',
          '    rd44 = (uint64_t)(rd61 + rd43);'):
    assert k in body, k
# r14 must be 2 r58 + (1 - r65) with r58, r65 parity bits (checked textually)
assert '    r14 = (uint32_t)(r68 + r67);' in body and '    r68 = (uint32_t)(r58 << (1u & 31));' in body
assert '    r67 = (uint32_t)(r26 - r65);' in body and '    r26 = (uint32_t)(1u);' in body

hdr = ['// Generated by kernels/native/in_fuse.py from ' + sys.argv[1].split('/')[-1] + ' (do not edit).',
       '#pragma once',
       '#include "native_rt.h"',
       '// Thread (gx, gy) of hiluma_engine_input_depthinv_mvhi_hdr_v2_rel (16x16 blocks): the 4 halves it stores at',
       '// network pixel (gx >> 2, gy), channels 4 (gx & 3) ... KARGP: the input kernel\'s 248-byte parameter block.',
       '__attribute__((device)) inline __attribute__((always_inline)) void in_unit(const AS4 uint8_t* KARGP, uint32_t gx, uint32_t gy, uint16_t out[4])',
       '{',
       '    const uint64_t KARGU = (uint64_t)(uintptr_t)KARGP;',
       '    float ' + ', '.join(f'shv{p}_{j}' for p in range(4) for j in range(3)) + ';',
       '    float ' + ', '.join(f'dpv{p}_{j}' for p in range(4) for j in range(2)) + ';']
print('\n'.join(hdr + out + ['}']))
