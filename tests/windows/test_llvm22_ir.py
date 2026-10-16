import math
from pathlib import Path
import random
import re
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/windows'))
from llvm22_ir import compatible_ir, float_hex


class CompatibilityTests(unittest.TestCase):
    def test_float_widening_preserves_finite_values(self):
        rng = random.Random(22)
        patterns = [rng.getrandbits(32) for _ in range(20000)]
        patterns += [(sign << 31) | (exponent << 23) | mantissa
                     for sign in (0, 1) for exponent in range(255)
                     for mantissa in (0, 1, 0x3fffff, 0x400000, 0x7fffff)]
        for bits in patterns:
            value = struct.unpack('>f', bits.to_bytes(4, 'big'))[0]
            if not math.isfinite(value):
                continue
            rendered = float_hex(re.match(r'(.*)', f'{bits:08X}'))
            self.assertEqual(bytes.fromhex(rendered[2:]), struct.pack('>d', value))

    def test_nan_payload_and_signaling_bit_preserved(self):
        for source, expected in [('7F800001', '0x7FF0000020000000'),
                                 ('FFC00001', '0xFFF8000020000000')]:
            self.assertEqual(float_hex(re.match(r'(.*)', source)), expected)

    def test_decimal_half_and_float(self):
        result = compatible_ir('%a = fcmp ult float %x, 8.507060e+37\n'
                               '%b = fmul half %h, 1.953130e-03\n')
        self.assertIn('0x47D0000020000000', result)
        self.assertIn('0xH1800', result)

    def test_double_and_strings_not_rounded_to_float(self):
        text = '%a = fmul double %x, 1.23456789012345e+00\n!0 = !{!"float 1.234567e+00"}\n'
        self.assertEqual(compatible_ir(text), text)

    def test_denormal_mode_and_conservative_promises(self):
        result = compatible_ir('attributes #0 = { denormal_fpenv(dynamic) '
                               'nocreateundeforpoison memory(read, target_mem: none) }\n'
                               '%x = uitofp contract nneg i32 %v to double\n')
        self.assertIn('"denormal-fp-math"="dynamic,dynamic"', result)
        self.assertNotIn('nocreateundeforpoison', result)
        self.assertNotIn('target_mem', result)
        self.assertIn('uitofp nneg i32', result)


if __name__ == '__main__':
    unittest.main()
