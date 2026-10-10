#!/usr/bin/env python3
"""Generate activation_m.glsl's ACT table: binary16 sigmoid, tanh and gate of every E4M3 code.

usage: gen_activation_m.py            print the ACT declaration
       gen_activation_m.py --check    exit 1 unless engine/activation_m.glsl holds exactly this table

Row c is uvec2(tanh << 16 | sigmoid, gate) for the E4M3 value of code c:
  sigmoid  1 / (1 + exp2(-1.442695 * x)) in f32, rounded to binary16
  tanh     correctly rounded binary16 tanh(x)
  gate     .5 * tanh(.5 * x) + .5 with each operation rounded to binary16
The NaN codes (127, 255) give quiet NaN; the sigmoid keeps the input's sign, as f32 arithmetic propagates it.
"""
import math
from pathlib import Path
import re
import sys

import numpy as np

GLSL = Path(__file__).resolve().parent / 'activation_m.glsl'
NAN_ROWS = {127: (0x7e007e00, 0x7e00), 255: (0x7e00fe00, 0x7e00)}


def e4m3(code):
    sign = -1.0 if code & 128 else 1.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 0:
        return sign * mantissa / 8 * 2.0 ** -6
    return sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)


def bits(value):
    return int(np.array(value, dtype=np.float16).view(np.uint16))


def row(code):
    if code in NAN_ROWS:
        return NAN_ROWS[code]
    x = e4m3(code)
    with np.errstate(over='ignore'):
        sigmoid = np.float32(1) / (np.float32(1) + np.exp2(np.float32(-1.442695) * np.float32(x), dtype=np.float32))
    # float64 tanh rounded once to binary16 is correctly rounded for every E4M3 input (no value lies near a tie).
    tanh = np.float16(math.tanh(x))
    half = np.float16(0.5)
    gate = np.float16(np.float16(half * np.float16(math.tanh(float(half * np.float16(x))))) + half)
    return bits(tanh) << 16 | bits(sigmoid), bits(gate)


def declaration():
    rows = ',\n'.join(f'    uvec2(0x{x:08x}u, 0x{y:08x}u)' for x, y in map(row, range(256)))
    return f'const uvec2 ACT[256] = uvec2[256](\n{rows}\n);'


def main(argv):
    if argv == ['--check']:
        match = re.search(r'const uvec2 ACT\[256\] = uvec2\[256\]\(.*?\n\);', GLSL.read_text(), re.S)
        if match is None or match.group(0) != declaration():
            print(f'{GLSL} does not hold the generated table', file=sys.stderr)
            return 1
        return 0
    if argv:
        sys.exit(__doc__)
    print(declaration())
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
