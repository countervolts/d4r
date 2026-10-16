#!/usr/bin/env python3
"""Generate preset F's encoder 0 shader: ptx2rr.py's translation of NVIDIA's per-pixel feature gather and of its guide
tail, around the native block (engine/rr/rr2_m.comp, KIND 4).

usage: gen_rr2_enc0.py ENC0.ptx ENC0.generated.comp OUT.comp
       gen_rr2_enc0.py DEC0_1.ptx DEC0_1.generated.comp OUT.comp --dec0_1

NVIDIA's kernel runs one 32-lane warp per 8x8-token window and keeps everything in registers. Its matrix products
(the embedding, the attention block and the guide head) are one contiguous run of the translated main, which is cut
out:
  wave w   the translation up to the first product: the gather; the first product's A fragments (the window's 64
           feature slots per token) go to the shared array XF
  4 waves  rr2_block(): embedding, attention, MLP, the tokens to the arena, the guide head to the shared array HD
  wave w   the last product's D fragments are loaded from HD and the translation continues after it: the guide tail
With --group N (2 to 4) a workgroup holds N windows along x: waves 0 ... N - 1 run the translated code of one window
each and the block then runs once per window; the stage's dispatch is 1 / N as wide ("xN" in rr.txt). Measured on
an RX 9070 XT (2026-10-13) that is slower than one window per workgroup (network +0.4 ms at 1280x720 with N = 4).
With --dec0_1 the same for dec0's second kernel (rr2_m.comp KIND 5): nothing before the block (it reads its tokens
from the arena); after it the last product's D fragments (the MLP sum) and the registers holding r are loaded from
MSH / RSH, and the translated tail forms y = r + MLP * s2 and computes the kernel's outputs from it.
The translated parts hold code derived from the user's DLL: the output is model data, do not commit it.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_dump import mma_statements, phases_of  # noqa: E402


def tiles(mm):
    """per mma of a phase: (m-tile, n-tile, k-step), as phase_dump.py orders them"""
    dset = {tuple(m[2]): n for n, m in enumerate(mm)}
    prev = {n: dset.get(tuple(m[5])) for n, m in enumerate(mm)}
    step = {}
    for n in range(len(mm)):
        s, q = 0, n
        while prev[q] is not None:
            q = prev[q]
            s += 1
        step[n] = s
    a_tile, b_tile = {}, {}
    for s in range(max(step.values()) + 1):
        for t, key in enumerate(dict.fromkeys(tuple(m[3]) for n, m in enumerate(mm) if step[n] == s)):
            a_tile[(s, key)] = t
        for t, key in enumerate(dict.fromkeys(tuple(m[4]) for n, m in enumerate(mm) if step[n] == s)):
            b_tile[(s, key)] = t
    return [(a_tile[(step[n], tuple(m[3]))], b_tile[(step[n], tuple(m[4]))], step[n]) for n, m in enumerate(mm)]


def f16vars_of(decls):
    names = set()
    for l in decls:
        if re.match(r'\s*(precise )?float16_t ', l):
            names |= set(re.findall(r'\b(rs\d+)\b', l))
    return names


REG = r'\b(?:rs|rd|r|f|p|l)\d+(?:_low|_high)?\b'
# what makes a statement of the gather more than index arithmetic on the parameter block
DATA = re.compile(r'fetch0|texture|shfl_|GX\d|GA\d|GW\d|SH\d|sust_|movm_|mma_')


def tail_inputs(pre, tail):
    """The gather's values the tail reads: (recomputed, saved). Recomputed are the statements, in order, of values
    that are index arithmetic on the parameter block (assigned once, outside any block); saved are the names of the
    others, which the gather pass writes to the window's record."""
    assigned, depth = {}, 0
    for i, l in enumerate(pre):
        m = re.match(r'\s*(\w+) = (.*)', l)
        if m:
            assigned.setdefault(m.group(1), []).append((i, depth, m.group(2)))
        depth += l.count('{') - l.count('}')
    used, own = [], set()
    for l in tail:
        m = re.match(r'\s*(\w+) = (.*)', l)
        for v in re.findall(REG, m.group(2) if m else l):
            if v not in own and v in assigned and v not in used:
                used.append(v)
        if m:
            own.add(m.group(1))

    def closure(v, seen):
        """statement indices v needs, or None when it is not plain index arithmetic"""
        if v in seen:
            return seen[v]
        seen[v] = None
        defs = assigned.get(v)
        if defs is None:
            seen[v] = set()   # never assigned in the gather: a declaration's zero
            return seen[v]
        if len(defs) != 1 or defs[0][1] != 0 or DATA.search(defs[0][2]):
            return None
        need = {defs[0][0]}
        for u in re.findall(REG, defs[0][2]):
            c = closure(u, seen)
            if c is None:
                return None
            need |= c
        seen[v] = need
        return need

    lines, saved, seen = set(), [], {}
    for v in used:
        c = closure(v, seen)
        if c is None:
            saved.append(v)
        else:
            lines |= c
    return [pre[i] for i in sorted(lines)], saved


def split_layout(xf_word, pre, tail, f16vars):
    """(recomputed statements, saved names, the record's words, the GLSL locating a window's record)"""
    recomputed, saved = tail_inputs(pre, tail)
    for v in saved:
        if not (v in f16vars or re.fullmatch(r'r\d+', v)):
            raise SystemExit(f'the tail reads {v} from the gather: not a 16- or 32-bit register')
    record = 1024 + 32 * len(saved)
    # the records' rows of windows lie in up to three ranges of the arena (the runtime's choice, where the network's
    # regions are dead during encoder 0): words XF_WORD, +2, +4 their byte addresses, +1 and +3 the first two's rows
    where = (f'    const uint rr2_gx = (param.w[0] + (param.w[20] & 0xffffu) + 7u) / 8u;\n'
             f'    uint rr2_y = gl_WorkGroupID.y, rr2_base = param.w[{xf_word}u];\n'
             f'    if (rr2_y >= param.w[{xf_word + 1}u]) {{\n'
             f'        rr2_y -= param.w[{xf_word + 1}u]; rr2_base = param.w[{xf_word + 2}u];\n'
             f'        if (rr2_y >= param.w[{xf_word + 3}u]) {{ rr2_y -= param.w[{xf_word + 3}u]; rr2_base = param.w[{xf_word + 4}u]; }}\n'
             f'    }}\n'
             f'    const uint rr2_rec = rr2_base / 4u + {record}u * (rr2_y * (param.w[0] / 8u + 1u) + gl_WorkGroupID.x);\n')
    return recomputed, saved, record, where


def gather_pass(xf_word, head, decls, pre, tail, first, f16vars):
    """Encoder 0 as two passes (--split gather | block): the translated gather in NVIDIA's own 32-thread groups, which
    writes each window's record to the arena, and the native block with the translated tail in 128-thread groups.
    One shader holding both runs the gather on one of the block's four waves while the other three wait, which
    costs more than the record's trip through memory (RX 9070 XT, 1280x720, 2026-10-13: gather 0.40 -> 0.24 ms).
    The record: 1024 words of A fragments ([token slot][64 FP8 feature slots], as the block's XF), then per lane the
    gather's values the tail reads. Its address: the byte address of row y's range (parameter words XF_WORD to
    XF_WORD + 4) + 4 * RECORD * (y' * (W / 8 + 1) + x), y' the row within its range."""
    recomputed, saved, record, where = split_layout(xf_word, pre, tail, f16vars)
    var = lambda r: r.lstrip('%')
    store = ['    if (rr2_valid) {   // the first product\'s A fragments: lane 4g + t, register q -> row, 4 k',
             '        const uint l_ = gl_SubgroupInvocationID, g_ = l_ >> 2u, t_ = l_ & 3u;']
    seen = set()
    for mm_, (i, j, s) in zip(first, tiles(first)):
        if (i, s) in seen:
            continue
        seen.add((i, s))
        for q, r in enumerate(mm_[3]):
            store.append(f'        rr2_arena.w[rr2_rec + rr2_uninter({16 * i + 8 * (q & 1)}u + g_) * 16u + {(32 * s + 16 * (q >> 1)) // 4}u + t_] = {var(r)};')
    for n, v in enumerate(saved):
        bits = f'uint(float16BitsToUint16({v}))' if v in f16vars else v
        store.append(f'        rr2_arena.w[rr2_rec + 1024u + {len(saved)}u * l_ + {n}u] = {bits};')
    store.append('    }')
    print(f'gather pass: record of {record} words, the tail reads {", ".join(saved) or "nothing"} from it')
    return (head + '\nlayout(set = 0, binding = 2, std430) buffer Rr2Arena { uint w[]; } rr2_arena;\n'
            'uint rr2_uninter(uint c) { return (c & ~31u) | (((c & 15u) >> 1u) << 2u) | (c & 1u) | (((c >> 4u) & 1u) << 1u); }\n\n'
            'void main()\n{\n' + '\n'.join(decls) + '\n' + where +
            '    const bool rr2_valid = gl_WorkGroupID.x < rr2_gx;\n'
            '    if (rr2_valid) {   // NVIDIA\'s warp: the translation before the first product\n' + '\n'.join(pre) + '\n    }\n' +
            '\n'.join(store) + '\n}\n'), record


def main():
    ptx, src, out = sys.argv[1:4]
    dec = '--dec0_1' in sys.argv
    # --group N: windows per workgroup (1 to 4; wave w runs the translated code of window w)
    G = int(sys.argv[sys.argv.index('--group') + 1]) if '--group' in sys.argv else 1
    kernel = 'rr2_dec0_kernel_1' if dec else 'rr2_enc0_kernel'
    lines = Path(ptx).read_text().replace('\r', '').split('\n')
    k = next(i for i, l in enumerate(lines) if f'.entry {kernel}(' in l)
    phases = phases_of(mma_statements(lines, k, len(lines)), lines)
    first, last = phases[0], phases[-1]
    var = lambda r: r.lstrip('%')

    g = Path(src).read_text().split('\n')
    m = next(i for i, l in enumerate(g) if l.startswith('void main()'))
    # --one-wave (D4R_RR_RELAXED=3): the block as rr2_w.comp, in NVIDIA's own 32-thread group, the tail on the same wave
    one_wave = '--one-wave' in sys.argv
    head = '\n'.join(g[:m])
    if not one_wave:
        head = head.replace('layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;',
                            'layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;')
    if dec:   # the tail's tanh from rr2_pack.py's table (ptx2rr.py's RR_TANH_TABLE; its offset from OUT's rr2_stage.h)
        v = head.index('\n') + 1
        table = re.search(r'#define W_TANH (\d+)u', (Path(out).parent / 'rr2_stage.h').read_text()).group(1)
        head = head[:v] + f'#define RR_TANH_TABLE {table}u\n' + head[v:]
    body = g[m + 2:]
    while body and body[-1].strip() in ('', '}'):
        last_line = body.pop()
        if last_line.strip() == '}':
            break
    nd = next(i for i, l in enumerate(body) if not re.match(r'    (precise |bool |uint64_t |float |uint |int |float16_t |f16vec2 )', l))
    calls = [i for i, l in enumerate(body) if 'mma_e4m3_f16' in l]
    a = calls[0] - 1                     # '{ uint md0_, ...;'
    b = next(i for i in range(calls[-1], len(body)) if body[i].strip() == '}')
    if not body[a].strip().startswith('{ uint md0_'):
        raise SystemExit('unexpected statement before the first product: ' + body[a])
    pre, cut, tail = body[nd:a], body[a:b + 1], body[b + 1:]
    if sum(l.count('{') - l.count('}') for l in pre):
        raise SystemExit('the first product is inside a block of the translated main')

    # values the tail takes from the cut region other than the last product's results
    assigned = lambda ls: {mm_.group(1) for l in ls for mm_ in [re.match(r'\s*(\w+) = ', l)] if mm_}
    restored = {var(r) for mm_ in last for r in mm_[2]}
    rmap = {}   # dec0_1: the register holding r for (row offset, column offset): y = r + (D half) * s2 in the tail
    if dec:
        place = {}
        for mm_, (i, j, s_) in zip(last, tiles(last)):
            for q, r in enumerate(mm_[2]):
                place[r] = (16 * i + 8 * q, 8 * j)
        text_tail = '\n'.join(lines[last[-1][1]:])
        half = {}
        for mo in re.finditer(r'mov\.b32\s+\{(%rs\d+),\s*(%rs\d+)\},\s*(%r\d+);', text_tail):
            if mo.group(3) in place:
                half[mo.group(1)] = (*place[mo.group(3)], 0)
                half[mo.group(2)] = (*place[mo.group(3)], 1)
        prod = {}
        for mo in re.finditer(r'mul\.f16\s+(%rs\d+),\s*(%rs\d+),\s*(%rs\d+);', text_tail):
            if mo.group(2) in half:
                prod[mo.group(1)] = half[mo.group(2)]
        for mo in re.finditer(r'add\.f16\s+(%rs\d+),\s*(%rs\d+),\s*(%rs\d+);', text_tail):
            if mo.group(2) in prod:
                rmap[var(mo.group(3))] = prod[mo.group(2)]
        # the same step in a module rewritten by build/rr/f/nv/ptx_contract.py (D4R_RR_FUSE builds): one fused instruction
        for mo in re.finditer(r'fma\.rn\.f16\s+(%rs\d+),\s*(%rs\d+),\s*(%rs\d+),\s*(%rs\d+);', text_tail):
            if mo.group(2) in half:
                rmap[var(mo.group(4))] = half[mo.group(2)]
        if len(rmap) != 2 * 2 * len(last):
            raise SystemExit(f'found {len(rmap)} residual registers, expected {4 * len(last)}')
        restored |= set(rmap)
    used = set(re.findall(r'\b[a-z]+\d+\b', '\n'.join(tail)))
    lost = sorted((assigned(cut) - assigned(pre) - restored) & used)
    # (values assigned again in the tail before their use are fine; the kernel has none on the reference build)
    keep = []
    cut_set = assigned(cut)
    for name in lost:   # thread indices and the like: their assignment is carried over when it depends on nothing cut
        line = next(l for l in reversed(cut) if re.match(rf'\s*{name} = ', l))
        if set(re.findall(r'\b[a-z]+\d+\b', line.split('=', 1)[1])) & cut_set:
            raise SystemExit(f'the tail needs {name}, which the cut region computes: {line.strip()}')
        keep.append('    ' + line)

    split = sys.argv[sys.argv.index('--split') + 1] if '--split' in sys.argv else None
    xf_word = int(sys.argv[sys.argv.index('--xf-word') + 1]) if split else 0
    if split == 'gather':
        text, record = gather_pass(xf_word, '\n'.join(g[:m]), body[:nd], pre, tail, first, f16vars_of(body[:nd]))
        Path(out).write_text(text)
        Path(out).with_suffix('.record').write_text(str(record))
        return
    if dec:
        first = []
    store = ['    if (rr2_valid) {   // the first product\'s A fragments: lane 4g + t, register q -> row, 4 k',
             '        const uint l_ = gl_SubgroupInvocationID, g_ = l_ >> 2u, t_ = l_ & 3u;']
    seen = set()
    for mm_, (i, j, s) in zip(first, tiles(first) if first else []):
        if (i, s) in seen:
            continue
        seen.add((i, s))
        for q, r in enumerate(mm_[3]):
            store.append(f'        XF[gl_SubgroupID * 1024u + uninter({16 * i + 8 * (q & 1)}u + g_) * 16u + {(32 * s + 16 * (q >> 1)) // 4}u + t_] = {var(r)};')
    store.append('    }')
    load = ['        {   // the guide head as the last product\'s D fragments (rows g, g + 8; columns 2t, 2t + 1)',
            '            const uint l_ = gl_SubgroupInvocationID, g_ = l_ >> 2u, t_ = l_ & 3u;']
    f16vars = set()
    for l in body[:nd]:
        if re.match(r'\s*(precise )?float16_t ', l):
            f16vars |= set(re.findall(r'\b(rs\d+)\b', l))
    for mm_, (i, j, s) in zip(last, tiles(last)):
        for q, r in enumerate(mm_[2]):
            row = f'({16 * i + 8 * q}u + g_)'
            load.append(f'            {var(r)} = HDV({row}, {8 * j}u + 2u * t_) | (HDV({row}, {8 * j + 1}u + 2u * t_) << 16u);')
    for name, (ro, co, hf) in rmap.items():
        v = f'RSV({ro}u + g_, {co + hf}u + 2u * t_)'
        load.append(f'            {name} = ' + (f'uint16BitsToFloat16(uint16_t({v}));' if name in f16vars else f'{v};'))
    load.append('        }')
    # the native block's source, without its #version line (this shader has its own)
    block = (Path(__file__).resolve().parent / ('rr2_w.comp' if one_wave else 'rr2_m.comp')).read_text().split('\n', 1)[1]
    if split == 'block':   # the second pass: the block on the gather pass's record, then the tail on wave 0
        recomputed, saved, record, where = split_layout(xf_word, pre, tail, f16vars)
        restore = ['            ' + (f'{v} = uint16BitsToFloat16(uint16_t(arena.w[rr2_rec + 1024u + {len(saved)}u * l_ + {n}u]));' if v in f16vars
                                     else f'{v} = arena.w[rr2_rec + 1024u + {len(saved)}u * l_ + {n}u];') for n, v in enumerate(saved)]
        load[2:2] = restore
        Path(out).write_text(
            head + '\n#extension GL_GOOGLE_include_directive : require\n#define RR2_EMBED 1\n#define RR2_GROUP 1\n#define RR2_XF_ARENA 1\n' +
            block + '\nuint HDV(uint row, uint col) { return col < 4u ? uint(float16BitsToUint16(HD[gl_SubgroupID * 256u + row * 4u + col])) : 0u; }\n\n'
            'void main()\n{\n' + '\n'.join(body[:nd]) + '\n' + where +
            '    const bool rr2_valid = gl_SubgroupID < 1u && gl_WorkGroupID.x < rr2_gx;\n'
            '    rr2_xf = rr2_rec;\n    rr2_block();\n    barrier();\n    if (rr2_valid) {\n' +
            '    // the gather\'s index arithmetic the tail reads\n' + '\n'.join(recomputed) + '\n' +
            '\n'.join(load + keep) + '\n' + '\n'.join(tail) + '\n    }\n}\n')
        return
    wx = lambda t: t.replace('gl_WorkGroupID.x', f'({G}u * gl_WorkGroupID.x + gl_SubgroupID)')
    text = (head + f'\n#extension GL_GOOGLE_include_directive : require\n#define RR2_EMBED 1\n#define RR2_GROUP {G}\n' + block + '\n' +
            ('uint HDV(uint row, uint col) { return uint(float16BitsToUint16(MSH[gl_SubgroupID * 2048u + row * 32u + col])); }\n'
             'uint RSV(uint row, uint col) { return uint(float16BitsToUint16(RSH[gl_SubgroupID * 2048u + row * 32u + col])); }\n\n' if dec else
             'uint HDV(uint row, uint col) { return col < 4u ? uint(float16BitsToUint16(HD[gl_SubgroupID * 256u + row * 4u + col])) : 0u; }\n\n') +
            'void main()\n{\n' + '\n'.join(body[:nd]) +
            '\n    // this wave\'s window, when the level has it\n'
            '    const uint rr2_gx = (param.w[0] + (param.w[20] & 0xffffu) + 7u) / 8u;\n'
            f'    const bool rr2_valid = gl_SubgroupID < {G}u && {G}u * gl_WorkGroupID.x + gl_SubgroupID < rr2_gx;\n'
            '    if (rr2_valid) {   // NVIDIA\'s warp: the translation before the first product\n' + wx('\n'.join(pre)) + '\n    }\n' +
            ('' if dec else '\n'.join(store)) +
            # (the block runs unconditionally, also for a window past the level's last one: it writes nothing there, and
            # inside a branch the driver compiles it much slower)
            ('\n    barrier();\n    rr2_block();\n    barrier();\n' if G == 1 else
             f'\n    barrier();\n    for (uint w_ = 0u; w_ < {G}u; ++w_) {{\n        rr2_win = w_;\n'
             '        rr2_block();\n        barrier();\n    }\n') + '    if (rr2_valid) {\n' +
            '\n'.join(load + keep) + '\n' + wx('\n'.join(tail)) + '\n    }\n}\n')
    Path(out).write_text(text)
    print(f'{out}: {len(pre)} gather lines, {len(cut)} cut, {len(tail)} tail lines, {len(seen)} A fragment sets')


if __name__ == '__main__':
    main()
