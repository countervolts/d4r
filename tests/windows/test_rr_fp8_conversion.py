import unittest

import numpy as np

from engine.rr.rrmath import q8


class F16ToE4M3Tests(unittest.TestCase):
    def test_special_values_and_signed_nan(self):
        bits = np.array([0x0000, 0x3C00, 0x7C00, 0xFC00, 0x7E00, 0xFE00], dtype=np.uint16)
        values = bits.view(np.float16)
        self.assertEqual(q8(values).tolist(), [0x00, 0x38, 0x7E, 0xFE, 0x7F, 0x7F])

    def test_every_half_pattern_maps_to_byte(self):
        values = np.arange(65536, dtype=np.uint16).view(np.float16)
        codes = q8(values)
        self.assertEqual(codes.shape, (65536,))
        self.assertEqual(codes.dtype, np.uint8)
        self.assertTrue(np.all(codes[0x7C01:0x8000] == 0x7F))


if __name__ == '__main__':
    unittest.main()
