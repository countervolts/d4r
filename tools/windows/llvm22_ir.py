"""Adapt textual HIP helper IR to the pinned LLVM 22 reader without relaxing FP."""
import argparse
from pathlib import Path
import re
import struct


def float_hex(match):
    bits = int(match[1], 16)
    sign, exponent, fraction = bits >> 31, (bits >> 23) & 255, bits & 0x7fffff
    if exponent:
        wide_exponent = 2047 if exponent == 255 else exponent + 896
        wide_fraction = fraction << 29
    elif fraction:
        leading = fraction.bit_length() - 1
        wide_exponent = leading - 149 + 1023
        wide_fraction = (fraction - (1 << leading)) << (52 - leading)
    else:
        wide_exponent = wide_fraction = 0
    # LLVM's legacy float spelling uses the exact widening's binary64 bits.
    # Integer operations preserve signed zero and signaling/quiet NaN payloads.
    return f'0x{((sign << 63) | (wide_exponent << 52) | wide_fraction):016X}'


def compatible_ir(text):
    text = re.sub(r'\bf0x([0-9A-Fa-f]{16})\b', r'0x\1', text)
    text = re.sub(r'\bf0x([0-9A-Fa-f]{4})\b', r'0xH\1', text)
    text = re.sub(r'\bf0x([0-9A-Fa-f]{8})\b', float_hex, text)
    text = text.replace('denormal_fpenv(dynamic)', '"denormal-fp-math"="dynamic,dynamic"')
    # This newer optimization promise is absent in LLVM 22. Dropping a promise
    # weakens what the optimizer may assume; it does not change instructions.
    text = re.sub(r'\bnocreateundeforpoison\b', '', text)
    text = re.sub(r'\bmemory\([^)]*\btarget_mem\b[^)]*\)', '', text)
    text = re.sub(r'\b(uitofp|sitofp|fptoui|fptosi)\s+(?:(?:nnan|ninf|nsz|arcp|contract|afn|reassoc|fast)\s+)+',
                  r'\1 ', text)
    # LLVM 24 permits short decimal floats that the older reader rejects.
    # Restrict conversion to unquoted literals whose nearest IR type is float/half;
    # double literals and string attributes retain their original spelling.
    result = []
    decimal = re.compile(r'(?<![\w.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?(?![\w.])')
    for line in text.splitlines():
        pieces = re.split(r'("(?:[^"\\]|\\.)*")', line)
        for index in range(0, len(pieces), 2):
            piece = pieces[index]
            def convert(match):
                types = re.findall(r'\b(float|double|half|bfloat|fp128|i\d+)\b', piece[:match.start()])
                if not types or types[-1] not in ('float', 'half'):
                    return match[0]
                if types[-1] == 'half':
                    bits = struct.unpack('>H', struct.pack('>e', float(match[0])))[0]
                    return f'0xH{bits:04X}'
                bits = struct.unpack('>I', struct.pack('>f', float(match[0])))[0]
                return float_hex(re.match(r'(.*)', f'{bits:08X}'))
            pieces[index] = decimal.sub(convert, piece)
        result.append(''.join(pieces))
    return '\n'.join(result) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(compatible_ir(args.input.read_text()))
