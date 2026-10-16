#!/usr/bin/env python3
"""Give a compute SPIR-V module PTX's floating-point environment: RoundingModeRTE for 16- and 32-bit results and
DenormPreserve for 16-bit values (SPV_KHR_float_controls, core in SPIR-V 1.4).

usage: spv_float_controls.py IN.spv [OUT.spv]

Without these execution modes the rounding of float conversions is the driver's choice, where every cvt.rn of
NVIDIA's kernels rounds to nearest even. glslang keeps only one execution mode per mode id, so the modes are inserted
into the compiled module instead.
"""
import struct
import sys

OP_EXTENSION, OP_CAPABILITY, OP_ENTRY_POINT, OP_EXECUTION_MODE = 10, 17, 15, 16
CAP_DENORM_PRESERVE, CAP_ROUNDING_RTE = 4464, 4467
MODES = ((4462, 16), (4462, 32), (4459, 16))   # RoundingModeRTE 16/32, DenormPreserve 16
EXTENSION = 'SPV_KHR_float_controls'


def instructions(words):
    i = 5
    while i < len(words):
        count = words[i] >> 16
        yield words[i] & 0xffff, words[i:i + count]
        i += count


def string_words(text):
    data = text.encode() + b'\0'
    data += b'\0' * (-len(data) % 4)
    return list(struct.unpack(f'<{len(data) // 4}I', data))


def patch(words):
    insts = list(instructions(words))
    caps = {w[1] for op, w in insts if op == OP_CAPABILITY}
    has_ext = any(op == OP_EXTENSION and EXTENSION.encode() in struct.pack(f'<{len(w) - 1}I', *w[1:]) for op, w in insts)
    entry = next(w[2] for op, w in insts if op == OP_ENTRY_POINT)
    modes = {(w[2], w[3]) for op, w in insts if op == OP_EXECUTION_MODE and w[1] == entry and len(w) > 3}
    last_cap = max(i for i, (op, _) in enumerate(insts) if op == OP_CAPABILITY)
    first_mode = min(i for i, (op, _) in enumerate(insts) if op == OP_EXECUTION_MODE)
    out = list(words[:5])
    for i, (op, w) in enumerate(insts):
        if i == first_mode:
            for mode, width in MODES:
                if (mode, width) not in modes:
                    out += [4 << 16 | OP_EXECUTION_MODE, entry, mode, width]
        out += w
        if i == last_cap:
            for cap in (CAP_DENORM_PRESERVE, CAP_ROUNDING_RTE):
                if cap not in caps:
                    out += [2 << 16 | OP_CAPABILITY, cap]
            if not has_ext:
                name = string_words(EXTENSION)
                out += [(1 + len(name)) << 16 | OP_EXTENSION, *name]
    return out


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    data = open(src, 'rb').read()
    words = list(struct.unpack(f'<{len(data) // 4}I', data))
    if words[0] != 0x07230203:
        sys.exit(f'{src}: not SPIR-V')
    out = patch(words)
    open(dst, 'wb').write(struct.pack(f'<{len(out)}I', *out))


if __name__ == '__main__':
    main()
