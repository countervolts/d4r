"""Opt-in rounded Quality reconstruction regression against the generic kernel.

D4R_RUN_GPU_TESTS=1 D4R_SR_E_MODEL=/private/model/e \
    D4R_SR_REPLAY=/path/to/d4r-e python3 -m unittest tests.test_sr_reconstruction_gpu
Optionally set D4R_SR_F_MODEL to exercise preset F too. No proprietary fixtures are needed.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS') == '1', 'opt-in GPU test')
class SrReconstructionGpuTests(unittest.TestCase):
    def test_rounded_quality_e(self):
        self.check_model('D4R_SR_E_MODEL')

    def test_rounded_quality_f(self):
        self.check_model('D4R_SR_F_MODEL')

    def check_model(self, variable):
        import numpy as np
        if not os.environ.get(variable):
            self.skipTest(f'{variable} must name a private compiled model')
        model = Path(os.environ[variable]).resolve()
        replay = Path(os.environ.get('D4R_SR_REPLAY', ROOT / 'build/native-engine/d4r-e')).resolve()
        env = dict(os.environ, D4R_E_TIME_REPEAT='1', ENABLE_VKBASALT='0')
        # Test automatic selection, independent of development overrides in the shell.
        env.pop('D4R_ENGINE_E_FINAL', None)
        shapes = [(1280, 800), (129, 65), (128, 64), (127, 63)]
        jitters = [(-.5, -.5), (.5, .5), (-.499, .499), (.499, -.499), (0, 0), (.25, -.25)]
        for ow, oh in shapes:
            # Independently round each render axis down/up; also exercise mixed rounding.
            for upx, upy in [(0, 0), (1, 1), (0, 1), (1, 0)]:
                rw, rh = (2 * ow + 2 * upx) // 3, (2 * oh + 2 * upy) // 3
                for hdr, low in [(0, 1), (1, 1), (0, 0), (1, 0)]:
                    with self.subTest(size=(rw, rh, ow, oh), hdr=hdr, low_motion=low):
                        with tempfile.TemporaryDirectory() as temporary:
                            case = Path(temporary)
                            mw, mh = (rw, rh) if low else (ow, oh)
                            rng = np.random.default_rng(123)
                            lines = [f'size {rw} {rh} {ow} {oh} {hdr} {low} {mw} {mh}']
                            for frame, (jx, jy) in enumerate(jitters, 1):
                                color = rng.random((rh, rw, 4), dtype=np.float32) * (8 if hdr else 1)
                                color[..., 3] = 1
                                color.astype('<f2').tofile(case / f'{frame}-color.bin')
                                rng.uniform(-12, 12, (mh, mw, 2)).astype('<f2').tofile(case / f'{frame}-motion.bin')
                                rng.uniform(.01, 1, (rh, rw)).astype('<f4').tofile(case / f'{frame}-depth.bin')
                                lines.append(f'frame {frame} {jx} {jy} 1 1 {int(frame in (1, 6))}')
                            (case / 'frames.txt').write_text('\n'.join(lines) + '\n')
                            for name, override in [('reference', {'D4R_ENGINE_E_FINAL': '3x3_256'}), ('auto', {})]:
                                out = case / name
                                out.mkdir()
                                result = subprocess.run([str(replay), str(model), str(case), str(out)],
                                                        env=dict(env, **override), capture_output=True, text=True, timeout=90)
                                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            for frame in range(1, len(jitters) + 1):
                                name = f'{frame}-out.bin'
                                reference = np.fromfile(case / 'reference' / name, '<f2')
                                actual = np.fromfile(case / 'auto' / name, '<f2')
                                self.assertEqual(actual.size, ow * oh * 4)
                                self.assertTrue(np.isfinite(reference).all())
                                self.assertTrue(np.isfinite(actual).all())
                                self.assertTrue(np.any(actual.reshape(-1, 4)[:, :3] != 0))
                                np.testing.assert_array_equal(actual.view("<u2"), reference.view("<u2"), err_msg=name)


if __name__ == '__main__':
    unittest.main()
