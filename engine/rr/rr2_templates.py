#!/usr/bin/env python3
"""What a preset F model takes from a capture, read from nvngx_dlssd.dll 310.9 instead: the argument-block templates
(<stage>.args), encoder 0's second table (enc0-aux.bin, argument 104) and the stages' plane-arena regions.

usage: rr2_templates.py NVNGX_DLSSD_DLL OUT_DIR                  write the templates and enc0-aux.bin
       rr2_templates.py NVNGX_DLSSD_DLL --check MODEL_DIR         compare with a model compile_rr2.py built from a capture

engine/runtime_rr2.inc fills every field of the blocks that depends on the frame, the geometry or the bound inputs.
Across the preset F captures (2026-10-14: capF, capF900, capF1440, capF4K, capFQ and the capFX* input variants) what
is left is per stage:
  network stages  binary16 s0 (byte 92), s1 (94), s2 (96) and, for the decoders, s3 (98) (engine/rr/rr2_ref.py)
  enc0            two float32 (572, 576), two binary16 widened to float32 (580, 584), 1.0 (588, 1072), 1 (1036)
  dec0_1          8.0 (208) and exp(t) (212); post: exp(t') (128); down / downp: 0.25 (204), 1.0 (296, 300)
NGX's model description is built by one routine of plain stores (mov [rdx + d], imm32 / ax / a pointer); describe()
replays it. It holds every tensor's pointer and size, s1 / s2 of each stage at 0x220 + 0x238 * n (a decoder level's
two kernels side by side, its s3 behind them), encoder 0's float32 pair at 0x230 and the binary16 pair at 0x185a, and
three one-value tensors (t at 0x1838, t' at 0x1848). s0 sits in the weights, before each encoder's relative position
tables (encoder 0: 16 bytes before; a decoder level uses its encoder's).

enc0-aux.bin is preset E's noise table (engine/rr/dll_weights.py) and then encoder 0's embedding, both branches: Wd
(32 x 64 FP8 codes, 43 inputs used), bd, Wu (32 x 32 FP8), bu (once, after the first branch), the FP8 matrices in the
kernel's fragment order: a fixed permutation of the index bits of the natural [in][out] order (FRAG_WD, FRAG_WU).
Everything written is derived from the user's DLL and stays on their machine.
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dll_weights import noise_table  # noqa: E402
from enc1_ref import inter  # noqa: E402
from rr2_ref import ORDER, dll_weights, find_tables  # noqa: E402

H = np.float16
NETWORK = ORDER   # enc0 ... enc5, dec4_0, dec4_1, ... dec0_1
ARG_BYTES = dict({s: 104 for s in NETWORK}, enc0=1088, dec0_1=216, post=320, down=416, downp=416, pooled=32)
# the plane-arena regions of the network stages: argument offset -> region start in units of 1/3152 of NGX's arena
# (W0 * H0 / 32 bytes; the same on every capture)
ARENA = {
    'enc0': '32:0,40:1024,64:2560,72:2240', 'enc1': '8:0,32:1536,40:2048,48:2240,64:3008,72:2304',
    'enc2': '8:2048,32:1024,40:1216,48:2304,64:1472,72:1364', 'enc3': '8:1216,32:1280,40:1344,48:1364,64:1368,72:1372',
    'enc4': '8:1344,32:1236,40:1216,48:1372,64:1261,72:1221', 'enc5': '8:1216,32:1256,48:1221,64:1262',
    'dec4_0': '8:1256,16:1236,32:1216', 'dec4_1': '8:1216,32:1344,48:1262,56:1261,64:1364',
    'dec3_0': '8:1344,16:1280,32:1216', 'dec3_1': '8:1216,32:1408,48:1364,56:1368,64:1488',
    'dec2_0': '8:1408,16:1024,32:1216', 'dec2_1': '8:1216,32:2816,48:1488,56:1472,64:3136',
    'dec1_0': '8:2816,16:1536,32:1024', 'dec1_1': '8:1024,32:2048,48:3136,56:3008,64:3072',
    'dec0_0': '8:2048,16:0,32:1024', 'dec0_1': '8:1024,48:3072,56:2560,64:2816',
}
# fragment order of encoder 0's FP8 matrices in enc0-aux.bin: bit j of the table index is bit FRAG[j] of the natural
# index in * 32 + out (found on 310.9.1 and checked on every entry of both branches)
FRAG_WD = [5, 9, 8, 3, 6, 7, 0, 1, 2, 10, 4]
FRAG_WU = [5, 9, 8, 3, 6, 7, 0, 1, 2, 4]


def sections(dll):
    pe = struct.unpack_from('<I', dll, 0x3c)[0]
    count, opt = struct.unpack_from('<HH', dll, pe + 6)[0], struct.unpack_from('<H', dll, pe + 20)[0]
    base = struct.unpack_from('<Q', dll, pe + 24 + 24)[0]
    out = []
    for i in range(count):
        at = pe + 24 + opt + 40 * i
        vsize, va, rsize, raw = struct.unpack_from('<IIII', dll, at + 8)
        out.append((base + va, max(vsize, rsize), raw))
    return out


def file_offset(dll, va):
    for start, size, raw in sections(dll):
        if start <= va < start + size:
            return va - start + raw
    raise ValueError(f'address {va:#x} is in no section')


def modrm(b, i):
    m = b[i]
    mod, reg, rm = m >> 6, (m >> 3) & 7, m & 7
    if rm == 4 or (mod == 0 and rm == 5) or mod == 3:
        raise ValueError
    base = {2: 'rdx', 1: 'rcx'}.get(rm)
    if mod == 0:
        return base, reg, 0, 1
    if mod == 1:
        return base, reg, struct.unpack_from('<b', b, i + 1)[0], 2
    return base, reg, struct.unpack_from('<i', b, i + 1)[0], 5


def replay(dll, i, image_base):
    """the stores of the routine starting at file offset i, up to its ret: ({offset: bytes}, {offset: pointer})"""
    stores, pointers = {}, {}
    eax, r8, rax_ptr = 0, 0, None
    text = next((s, sz, raw) for s, sz, raw in sections(dll) if raw <= i < raw + sz)
    for _ in range(4000):
        b = dll[i:i + 16]
        p66 = b[0] == 0x66
        j = 1 if p66 else 0
        rex = b[j] if 0x40 <= b[j] <= 0x4f else 0
        j += 1 if rex else 0
        op, w, rexr = b[j], bool(rex & 8), bool(rex & 4)
        size = 8 if w else 2 if p66 else 4
        if op == 0xc3:
            return stores, pointers
        if op == 0x33 and rex == 0x45 and b[j + 1] == 0xc0:   # xor r8d, r8d
            r8 = 0; i += j + 2; continue
        if op == 0xb8 and not rex:                            # mov eax, imm32
            eax = struct.unpack_from('<I', b, j + 1)[0]; i += j + 5; continue
        if op == 0x8d and w and b[j + 1] == 0x05:             # lea rax, [rip + d]
            rel = struct.unpack_from('<i', b, j + 2)[0]
            rax_ptr = text[0] + (i + j + 6 - text[2]) + rel; i += j + 6; continue
        if op == 0x8b:                                        # mov rax, [rip + d] / mov rax, rcx
            rax_ptr = None; i += j + (2 if b[j + 1] >> 6 == 3 else 6); continue
        if op == 0xc7:                                        # mov [rdx + d], imm32
            base, _, d, n = modrm(b, j + 1)
            if base == 'rdx':
                stores[d] = struct.pack('<q' if w else '<i', struct.unpack_from('<i', b, j + 1 + n)[0])
            i += j + 1 + n + 4; continue
        if op == 0x89:                                        # mov [rdx + d], eax / ax / rax / r8
            base, reg, d, n = modrm(b, j + 1)
            if base == 'rdx' and reg == 0:
                if w and not rexr:
                    pointers[d] = rax_ptr
                else:
                    v = r8 if rexr else eax
                    stores[d] = (v & ((1 << (8 * size)) - 1)).to_bytes(size, 'little')
            i += j + 1 + n; continue
        raise ValueError(f'instruction at file offset {i:#x} ({b[:8].hex()})')
    raise ValueError('no ret')


def describe(dll):
    """replay the routine that builds the model description: the one that stores encoder 0's tensor sizes 2752
    (Wd, 2 x 32 x 43) at offset 0 and 64 (bd) at 0x10"""
    marker = b'\xc7\x02\xc0\x0a\x00\x00\xc7\x42\x10\x40\x00\x00\x00'
    found = []
    at = dll.find(marker)
    while at >= 0:
        start = dll.rfind(b'\xcc\xcc', 0, at) + 2
        try:
            stores, pointers = replay(dll, start, None)
            if 0x220 in stores and 0x1858 in stores:
                found.append((stores, pointers))
        except ValueError:
            pass
        at = dll.find(marker, at + 1)
    if len(found) != 1:
        raise SystemExit(f'found {len(found)} preset F model descriptions in the DLL (expected one): not a layout this '
                         'builder knows')
    return found[0]


def h16(b):
    return np.frombuffer(b, H)[0]


def values(dll):
    """{stage: {argument offset: bytes}} of every constant the runtime keeps"""
    stores, pointers = describe(dll)
    image = bytearray(max(d + len(v) for d, v in stores.items()))
    for d in sorted(stores):
        image[d:d + len(stores[d])] = stores[d]
    word = lambda d: bytes(image[d:d + 2])
    tables = find_tables(dll)
    s0 = {}
    for n, stage in enumerate(['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5']):
        at = tables[stage] - (16 if stage == 'enc0' else 8)
        s0[stage] = dll[at:at + 2]
    out = {}
    for n, stage in enumerate(NETWORK):
        v = {}
        if stage.startswith('enc'):
            k = int(stage[3])
            v[92] = s0[stage]
            d = 0x220 + 0x238 * k
            v[94] = bytes(image[d:d + 4])
        else:
            level, half = int(stage[3]), int(stage[5])
            v[92] = s0[f'enc{level + 1}']
            d = 0x220 + 0x238 * (10 - level)
            v[94] = bytes(image[d + 4 * half:d + 4 * half + 4])
            v[98] = word(d + 8)
        out[stage] = v
    f16f32 = lambda b: struct.pack('<f', float(h16(b)))
    one = struct.pack('<f', 1.0)
    out['enc0'].update({572: bytes(image[0x230:0x234]), 576: bytes(image[0x234:0x238]), 580: f16f32(word(0x185c)), 584: f16f32(word(0x185a)),
                        588: one, 1036: struct.pack('<I', 1), 1072: one})
    tensor = lambda d: dll[file_offset(dll, pointers[d]):file_offset(dll, pointers[d]) + 2]
    expf = lambda b: struct.pack('<f', np.float32(math.exp(float(h16(b)))))
    out['dec0_1'].update({208: struct.pack('<f', 8.0), 212: expf(tensor(0x1838))})
    out['post'] = {128: expf(tensor(0x1848))}
    for s in ('down', 'downp'):
        out[s] = {204: struct.pack('<f', 0.25), 296: one, 300: one}
    out['pooled'] = {}
    return out


def templates(dll):
    blocks = {}
    for stage, v in values(dll).items():
        b = bytearray(ARG_BYTES[stage])
        for off, x in v.items():
            b[off:off + len(x)] = x
        blocks[stage] = bytes(b)
    return blocks


def frag(natural, bits):
    """natural [in][out] FP8 codes -> the table's order: bit j of the table index is bit bits[j] of the natural index"""
    flat = natural.ravel()
    idx = np.arange(flat.size)
    src = np.zeros_like(idx)
    for j, b in enumerate(bits):
        src |= ((idx >> j) & 1) << b
    return flat[src].tobytes()


def aux_table(dll):
    noise = noise_table(dll)
    if noise is None:
        raise SystemExit('no noise table found in the DLL')
    w = dll_weights(dll, 'enc0')

    def natural(b_operand):   # B = q8(mat).T[inter(k)] (k x n) -> [in][out]
        n = np.empty_like(b_operand)
        n[np.asarray(inter(b_operand.shape[0]))] = b_operand
        return n
    out = [noise]
    for b in range(2):
        out += [frag(natural(np.asarray(w['ewd'][b])), FRAG_WD), np.asarray(w['ebd'][b], H).tobytes(),
                frag(natural(np.asarray(w['ewu'][b])), FRAG_WU)]
        if b == 0:
            out.append(np.asarray(w['ebu'], H).tobytes())
    return b''.join(out)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    dll = Path(sys.argv[1]).read_bytes()
    blocks, aux = templates(dll), aux_table(dll)
    if sys.argv[2] == '--check':
        model, bad = Path(sys.argv[3]), 0
        theirs = (model / 'enc0-aux.bin').read_bytes()
        print('enc0-aux.bin:', 'same' if theirs == aux else 'DIFFERENT'); bad += theirs != aux
        for stage, ours in blocks.items():
            path = model / f'{stage}.args'
            if not path.exists():
                continue
            t = path.read_bytes()[:len(ours)]
            diff = [i for i in range(len(ours)) if ours[i] != t[i]]
            words = sorted({i // 4 * 4 for i in diff})
            print(f'{stage}: ' + ('same' if not diff else f'{len(words)} words differ: {words[:40]}'))
        sys.exit(1 if bad else 0)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for stage, b in blocks.items():
        (out / f'{stage}.args').write_bytes(b)
    (out / 'enc0-aux.bin').write_bytes(aux)


if __name__ == '__main__':
    main()
