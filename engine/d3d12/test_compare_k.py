"""Keep final-RGB metrics compatible with the Linux handoff, including failures."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import importlib.util

import numpy as np
from compare_k import compare, KINDS


class Metrics(unittest.TestCase):
    def test_fixed_peak_hdr(self):
        reference = np.array([100., 2., 4.])
        result = reference + .125
        psnr, worst, _, invalid = compare(reference, result, peak=1.)
        self.assertAlmostEqual(psnr, -20 * np.log10(.125))
        self.assertEqual(worst, .125)
        self.assertEqual(invalid, 0)

    def test_matching_nan_is_invalid(self):
        self.assertEqual(compare(np.array([0., np.nan]), np.array([0., np.nan]))[3], 1)

    def test_shared_frame_checker_uses_same_rgb_metric(self):
        spec = importlib.util.spec_from_file_location('frame_checker', Path(__file__).parents[1] / 'test_frame.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as tmp:
            reference, output = Path(tmp) / 'reference.bin', Path(tmp) / 'output.bin'
            np.array([100, 2, 4, 1], np.float16).tofile(reference)
            np.array([100, 2, 4.125, 0], np.float16).tofile(output)
            expected = -10 * np.log10(.125 ** 2 / 3)
            self.assertAlmostEqual(checker.compare(reference, output, final_rgb=True)[0], expected)

    def test_cli_rgb_only_and_invalid_auxiliary(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, result = Path(tmp) / 'reference', Path(tmp) / 'result'
            reference.mkdir(); result.mkdir()
            for suffix, _ in KINDS:
                np.ones(4, np.float16).tofile(reference / f'frame-0.{suffix}.bin')
                np.ones(4, np.float16).tofile(result / f'frame-0.{suffix}.bin')
            np.array([1, 1, 1, 0], np.float16).tofile(result / 'frame-0.312.bin')
            command = [sys.executable, str(Path(__file__).with_name('compare_k.py')), str(reference), str(result)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            for directory in (reference, result):
                np.full(4, np.nan, np.float16).tofile(directory / 'frame-0.296.bin')
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)


if __name__ == '__main__':
    unittest.main()
