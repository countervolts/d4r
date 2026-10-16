#!/usr/bin/env python3
"""The two argument-block templates of a preset D / E model (enc0.args, hkpn.args) without a capture.

usage: rr_templates.py NVNGX_DLSSD_DLL D|E OUT_DIR
       rr_templates.py NVNGX_DLSSD_DLL D|E --check MODEL_DIR...   (compare with templates compile_rr.py took from captures)

engine/runtime_rr.inc fills every field of the two blocks that depends on the frame, the geometry or the bound inputs.
Across captures of different sizes, inputs and frames (2026-10-14: capD, capDT, capD4K, capDX*, capE, capE3) what is
left is the same few constants for both presets, plus two floats of encoder 0 (bytes 540 and 544) that differ between
the presets. NGX sets those two in code: `sub ecx, 4; je D; cmp ecx, 1; jne ...; mov [rbx + X], E0; mov [rbx + X + 4],
E1; jmp ...; D: mov [rbx + X], D0; mov [rbx + X + 4], D1` (ecx = the preset), which find_floats() looks up in the DLL.
The templates are written on the user's machine and are never packaged.
"""
import re
import struct
import sys
from pathlib import Path

ENC0_BYTES, HKPN_BYTES = 1032, 480
# (offset, packed value) of the constants the runtime keeps
ENC0_CONST = [(360, struct.pack('<f', 1.0)), (532, struct.pack('<I', 2)), (988, struct.pack('<I', 1))]
HKPN_CONST = [(148, struct.pack('<f', 1.0)), (436, struct.pack('<f', 0.25)), (456, struct.pack('<I', 1))]
# the byte ranges engine/runtime_rr.inc writes every frame (setFrame), for --check
ENC0_WRITTEN = [(0, 8), (48, 64), (80, 84), (136, 160), (312, 352), (364, 368), (432, 440), (472, 480), (528, 532),
                (536, 537), (539, 540), (596, 980), (980, 981), (984, 988), (1024, 1032)]
HKPN_WRITTEN = [(0, 48), (56, 64), (140, 148), (152, 156), (160, 161), (256, 264), (336, 344), (432, 433), (440, 444)]

# mov [rbx + disp32], imm32 is c7 83 disp32 imm32
PATTERN = re.compile(rb'\x83\xe9\x04\x74.\x83\xf9\x01\x75.'
                     rb'\xc7\x83(....)(....)\xc7\x83(....)(....)\xeb.'
                     rb'\xc7\x83(....)(....)\xc7\x83(....)(....)', re.S)


def find_floats(dll):
    """{'D': (f0, f1), 'E': (f0, f1)}: encoder 0's bytes 540 / 544 as NGX's code sets them"""
    found = set()
    for m in PATTERN.finditer(dll):
        e0, e0v, e1, e1v, d0, d0v, d1, d1v = m.groups()
        x = struct.unpack('<I', e0)[0]
        if struct.unpack('<I', e1)[0] == x + 4 and e0 == d0 and e1 == d1:
            found.add((d0v + d1v, e0v + e1v))
    if len(found) != 1:
        raise SystemExit(f'found {len(found)} candidates for the preset constants of encoder 0 (expected one): '
                         'not a DLL layout this builder knows')
    d, e = found.pop()
    return {'D': d, 'E': e}


def templates(dll, preset):
    enc0, hkpn = bytearray(ENC0_BYTES), bytearray(HKPN_BYTES)
    for off, v in ENC0_CONST:
        enc0[off:off + len(v)] = v
    for off, v in HKPN_CONST:
        hkpn[off:off + len(v)] = v
    enc0[540:548] = find_floats(dll)[preset]
    return bytes(enc0), bytes(hkpn)


def masked(b, written, handles):
    b = bytearray(b)
    for lo, hi in written:
        b[lo:hi] = bytes(hi - lo)
    for off in handles:
        b[off:off + 8] = bytes(8)
    return bytes(b)


def handles(rr_txt, stage):
    for line in Path(rr_txt).read_text().splitlines():
        f = line.split()
        if f[:2] == ['stage', stage]:
            tex = [] if f[5] == '-' else [int(x) for x in f[5].split(',')]
            surf = [] if f[6] == '-' else [int(x.split(':')[0]) for x in f[6].split(',')]
            return tex + surf
    return []


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    dll, preset = Path(sys.argv[1]).read_bytes(), sys.argv[2].upper()
    if preset not in ('D', 'E'):
        raise SystemExit('preset D or E')
    enc0, hkpn = templates(dll, preset)
    if sys.argv[3] == '--check':
        bad = 0
        for model in sys.argv[4:]:
            for name, ours, written in (('enc0', enc0, ENC0_WRITTEN), ('hkpn', hkpn, HKPN_WRITTEN)):
                theirs = (Path(model) / f'{name}.args').read_bytes()
                h = handles(Path(model) / 'rr.txt', name)
                a, b = masked(ours, written, h), masked(theirs, written, h)
                diff = [i for i in range(len(a)) if a[i] != b[i]]
                bad += bool(diff)
                print(f'{model} {name}: ' + ('same' if not diff else f'{len(diff)} bytes differ, first at {diff[:8]}'))
        sys.exit(1 if bad else 0)
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'enc0.args').write_bytes(enc0)
    (out / 'hkpn.args').write_bytes(hkpn)


if __name__ == '__main__':
    main()
