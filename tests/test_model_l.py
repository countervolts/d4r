"""L must retain its unfolded neural math when native surface stores are substituted."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModelLTests(unittest.TestCase):
    def test_model_l_config_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            ini = Path(directory) / 'd4r.ini'
            ini.write_text('[DLSS]\nModel = L\n')
            env = dict(os.environ)
            env.pop('D4R_DLSS_PRESET', None)
            cmd = ['python3', str(ROOT / 'scripts/d4r_config.py'), '--config', str(ini)]
            self.assertIn('export D4R_DLSS_PRESET=12', subprocess.check_output(cmd, env=env, text=True))
            env['D4R_DLSS_PRESET'] = '11'
            self.assertNotIn('export D4R_DLSS_PRESET=', subprocess.check_output(cmd, env=env, text=True))

    def test_unfolded_variants_preserve_neural_arithmetic(self):
        names = ['rrlite_dec0_4x4']
        for mv in ('mvhi', 'mvlo'):
            for dynamic_range in ('hdr', 'ldr'):
                names.append(f'rrlite_enc0_4x4_{mv}_{dynamic_range}')
                names.extend(f'rrlite_post_{v}_{mv}_{dynamic_range}' for v in ('3_1', '3_2'))
        math = 'mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r1, %r2}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%r9, %r10};'
        store = 'sust.b.2d.v2.b16.zero [%rd1, {%r1,%r2}], {%rs1,%rs2};'
        with tempfile.TemporaryDirectory() as directory:
            ptx = Path(directory)
            env = dict(os.environ, D4R_DLSS_PTX_DIR=directory, D4R_SUST_KINDS='b16,p,b32')
            for name in names:
                with self.subTest(kernel=name):
                    (ptx / 'input.ptx').write_text(f'.visible .entry {name}(\n)\n{{\n{math}\n{store}\nret;\n}}\n')
                    subprocess.run(['python3', str(ROOT / 'kernels/tex/make_ptx.py'), name, str(ptx / 'out.ptx')],
                                   env=env, check=True, capture_output=True)
                    output = (ptx / 'out.ptx').read_text()
                    self.assertIn(math, output)
                    self.assertNotIn(store, output)
                    self.assertIn('call d4r_sust_v2b16', output)
                    self.assertNotIn('d4r_enc0_tail', output)
                    self.assertNotIn('d4r_dec0_head', output)


if __name__ == '__main__':
    unittest.main()
