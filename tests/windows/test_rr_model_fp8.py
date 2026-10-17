import struct
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/windows'))
from build_rr_model import require_fp8_matrices


class NativeFp8Tests(unittest.TestCase):
    def check_module(self, capabilities, trailing=()):
        words = [0x07230203, 0x00010600, 0, 1, 0]
        for capability in capabilities:
            words += [(2 << 16) | 17, capability]
        words.extend(trailing)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stage.spv'
            path.write_bytes(struct.pack('<' + 'I' * len(words), *words))
            require_fp8_matrices(path)

    def test_native_fp8(self):
        self.check_module((4212, 4213, 6022))

    def test_fp16_fallback_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Native FP8'):
            self.check_module((9, 6022))

    def test_scalar_fp8_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Native FP8'):
            self.check_module((4212, 6022))

    def test_truncated_instruction_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'instruction'):
            self.check_module((4212, 4213, 6022), ((3 << 16) | 17, 9))

    def test_zero_length_instruction_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'instruction'):
            self.check_module((4212, 4213, 6022), (17,))


if __name__ == '__main__':
    unittest.main()
