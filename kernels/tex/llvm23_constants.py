"""stdin -> stdout: LLVM 23 IR floating-point constants in the forms LLVM 22's llvm-as reads (build_tex.sh).

LLVM 23 (TheRock 10.2's clang) prints hex constants as the type's own bits (`fmul float %x, f0x33800000`) and
half/bfloat constants in decimal (`fmul half %x, 1.953130e-03`). LLVM 22 reads float and double hex constants only as
the bits of the double (`0x3E70000000000000`), half as `0xH....` and bfloat as `0xR....`, and no decimal half or
bfloat. An operand's type is often only the instruction's, earlier on the line: an f0x constant's width names it
(4 digits half, or bfloat on a bfloat line; 8 float; 16 double), and an untyped decimal is converted only on a line
whose one floating-point type is half or bfloat.
"""
import re
import struct
import sys

NUMBER = r'[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)'
TYPES = re.compile(r'\b(half|bfloat|float|double)\b')


def half_bits(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def bfloat_bits(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16   # round to nearest even


def short(kind, value):
    return '0x%s%04X' % ('H', half_bits(value)) if kind == 'half' else '0x%s%04X' % ('R', bfloat_bits(value))


def convert(line):
    if 'f0x' not in line and not re.search(r'\b(half|bfloat)\b', line):
        return line
    kinds = set(TYPES.findall(line))

    def typed_hex(match):
        digits = match.group(1)
        bits = int(digits, 16)
        if len(digits) == 8:
            return '0x%016X' % struct.unpack('<Q', struct.pack('<d', struct.unpack('<f', struct.pack('<I', bits))[0]))[0]
        if len(digits) == 16:
            return '0x' + digits.upper()
        if len(digits) == 4:
            return ('0xR%04X' if 'bfloat' in kinds and 'half' not in kinds else '0xH%04X') % bits
        raise SystemExit('llvm23_constants.py: unexpected constant f0x' + digits)

    line = re.sub(r'\bf0x([0-9A-Fa-f]+)\b', typed_hex, line)
    # typed decimals (`half 1.5`, also inside vectors), then untyped operands of a single-type half/bfloat line
    line = re.sub(r'\b(half|bfloat) (' + NUMBER + r')(?![\w.])', lambda m: m.group(1) + ' ' + short(m.group(1), float(m.group(2))), line)
    fp = kinds & {'half', 'bfloat', 'float', 'double'}
    if len(fp) == 1 and fp <= {'half', 'bfloat'}:
        kind = next(iter(fp))
        line = re.sub(r'(?<![\w.%@"])(' + NUMBER + r')(?![\w.])', lambda m: short(kind, float(m.group(1))), line)
    return line


for line in sys.stdin:
    sys.stdout.write(convert(line))
