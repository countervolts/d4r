"""PTX texture sampling/budget contracts and guarded M-only neural splices."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRANSLATOR = ROOT/'kernels/native/ptx2hip.py'


def source(dtype='f32', suffix=''):
    dst = '{%f0,%f1,%f2,%f3}' if dtype == 'f32' else '{%rs0,%rs1,%rs2,%rs3}'
    return f'''.version 8.0
.target sm_89
.address_size 64
.visible .entry sample(.param .align 8 .b8 sample_args[16])
.maxntid 32, 1, 1
{{
.reg .b64 %rd<2>;
.reg .f32 %f<8>;
.reg .b16 %rs<4>;
ld.param.u64 %rd0, [sample_args];
mov.f32 %f4, 0f3f000000;
mov.f32 %f5, 0f3e800000;
tex.base.2d.v4.{dtype}.f32 {dst}, [%rd0, {{%f4,%f5}}]{suffix};
ret;
}}
'''


class MNativeTranslationTests(unittest.TestCase):
    def translate(self, text, *flags):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'sample.ptx'; p.write_text(text)
            return subprocess.run(['python3', str(TRANSLATOR), *flags, str(p), 'sample'],
                                  text=True, capture_output=True)

    def test_float_base_samples_level_zero_for_both_output_types(self):
        for dtype in ('f32', 'f16'):
            with self.subTest(dtype=dtype):
                p = self.translate(source(dtype))
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn('tex_level(rd0, f4, f5, 0.0f)', p.stdout)
                self.assertNotIn('tex_fetch(rd0', p.stdout)

    def test_optional_texture_operands_are_not_silently_dropped(self):
        self.assertNotEqual(self.translate(source(suffix=', 1')).returncode, 0)

    def test_register_budget_is_opt_in_and_checked(self):
        self.assertNotIn('amdgpu_num_vgpr', self.translate(source()).stdout)
        p = self.translate(source(), '--max-vgpr=128')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('amdgpu_num_vgpr(128)', p.stdout)
        for bad in (0, 3, 97, 260):
            with self.subTest(budget=bad):
                self.assertNotEqual(self.translate(source(), f'--max-vgpr={bad}').returncode, 0)

    def test_unfolded_l_cannot_use_folded_m_tail(self):
        p = self.translate(source(), '--m-tail')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('--m-tail requires folded M', p.stderr)


if __name__ == '__main__':
    unittest.main()
