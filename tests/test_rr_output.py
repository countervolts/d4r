"""Rendered RGB, alpha subrect preservation, and per-frame output comparison."""
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "tools" / "compare_rr_output.py"


class RayReconstructionOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.a = Path(self.directory.name) / "host.raw"
        self.b = Path(self.directory.name) / "gpu.raw"

    def compare(self, width=1, height=1, *extra):
        return subprocess.run(
            ["python3", str(COMPARE), "--a", str(self.a), "--b", str(self.b),
             "--width", str(width), "--height", str(height), *extra],
            capture_output=True, text=True)

    def test_channel_swap_with_identical_luma_is_rejected(self):
        self.a.write_bytes(struct.pack("<eeee", 0.75, 0.125, 0.25, 1.0))
        self.b.write_bytes(struct.pack("<eeee", 0.25, 0.125, 0.75, 1.0))
        self.assertNotEqual(self.compare().returncode, 0)

    def test_matching_nonfinite_pixels_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                raw = struct.pack("<eeee", value, 0.25, 0.5, 1.0)
                self.a.write_bytes(raw)
                self.b.write_bytes(raw)
                self.assertNotEqual(self.compare().returncode, 0)

    def test_truncated_image_is_rejected(self):
        self.a.write_bytes(struct.pack("<eeee", 0.25, 0.5, 0.75, 1.0))
        self.b.write_bytes(self.a.read_bytes()[:-1])
        self.assertNotEqual(self.compare().returncode, 0)

    def alpha_fixture(self, frames=0):
        raw = struct.pack("<eeee", 0.25, 0.5, 0.75, 1.0) * 2
        values = [-2.0] * 12
        values[5:7] = [0.25, 0.75]
        alpha = struct.pack("<12f", *values)
        for run in (self.a, self.b):
            run.write_bytes(raw)
            Path(str(run) + ".alpha.raw").write_bytes(alpha)
            for frame in range(1, frames + 1):
                Path(f"{run}.frame{frame}").write_bytes(raw)
                Path(f"{run}.alpha{frame}").write_bytes(alpha)
        return values

    def compare_alpha(self, frames=0):
        return self.compare(2, 1, "--alpha", "--alpha-width", "4", "--alpha-height", "3",
                            "--alpha-base", "1", "1", "--frames", str(frames), "--same-frames")

    def test_alpha_subrect_corruption_is_rejected_even_when_both_runs_agree(self):
        values = self.alpha_fixture()
        self.assertEqual(self.compare_alpha().returncode, 0)
        values[0] = 0.5
        for run in (self.a, self.b):
            Path(str(run) + ".alpha.raw").write_bytes(struct.pack("<12f", *values))
        self.assertNotEqual(self.compare_alpha().returncode, 0)

    def test_flat_unwritten_alpha_is_rejected_even_when_both_runs_agree(self):
        values = self.alpha_fixture()
        values[5:7] = [0.0, 0.0]
        for run in (self.a, self.b):
            Path(str(run) + ".alpha.raw").write_bytes(struct.pack("<12f", *values))
        self.assertNotEqual(self.compare_alpha().returncode, 0)

    def test_alpha_frame_mismatch_is_not_hidden_by_matching_final_output(self):
        values = self.alpha_fixture(frames=2)
        self.assertEqual(self.compare_alpha(frames=2).returncode, 0)
        values[5:7] = [0.5, 0.5]
        Path(f"{self.b}.alpha1").write_bytes(struct.pack("<12f", *values))
        self.assertNotEqual(self.compare_alpha(frames=2).returncode, 0)

    def test_missing_alpha_frame_is_rejected(self):
        self.alpha_fixture(frames=2)
        Path(f"{self.b}.alpha2").unlink()
        self.assertNotEqual(self.compare_alpha(frames=2).returncode, 0)


if __name__ == "__main__":
    unittest.main()
