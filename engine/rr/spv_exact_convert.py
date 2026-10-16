#!/usr/bin/env python3
"""Decorate every float conversion of a SPIR-V module NoContraction, so the driver keeps it.

usage: spv_exact_convert.py IN.spv [OUT.spv]

RADV folds float(float16_t(x)) back to x unless the conversions are marked exact; glslang's `precise` does not reach
conversions. With the decoration a binary16 rounding is two instructions (engine/rr/rr_exact.glsl, RR_EXACT_CONVERT)
instead of three with an opaque zero OR-ed into the bits.
"""
import struct
import sys

OP_DECORATE, OP_FCONVERT, NO_CONTRACTION = 71, 115, 42


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    data = open(src, 'rb').read()
    words = list(struct.unpack(f'<{len(data) // 4}I', data))
    if words[0] != 0x07230203:
        sys.exit(f'{src}: not SPIR-V')
    insts, i = [], 5
    while i < len(words):
        count = words[i] >> 16
        insts.append((words[i] & 0xffff, words[i:i + count]))
        i += count
    ids = [w[2] for op, w in insts if op == OP_FCONVERT]
    first_type = next(k for k, (op, _) in enumerate(insts) if 19 <= op <= 39)   # annotations end where types begin
    out = list(words[:5])
    for k, (op, w) in enumerate(insts):
        if k == first_type:
            for result in ids:
                out += [3 << 16 | OP_DECORATE, result, NO_CONTRACTION]
        out += w
    open(dst, 'wb').write(struct.pack(f'<{len(out)}I', *out))
    print(f'{dst}: {len(ids)} conversions marked exact')


if __name__ == '__main__':
    main()
