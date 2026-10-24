"""Opt-in synthetic tensor comparison of every shared M/L Swin layer.

D4R_RUN_GPU_TESTS=1 D4R_DUMP_RUNNER=... D4R_SWIN_CONTROL_DIR=...
D4R_SWIN_CANDIDATE_DIR=... python3 -m unittest discover -s tests -p test_swin_gfx11_gpu.py
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS') == '1', 'opt-in GPU test')
class SwinGfx11GpuTests(unittest.TestCase):
    def test_shared_layers_shifted_unshifted_and_untouched_allocations(self):
        runner = os.environ['D4R_DUMP_RUNNER']
        control = Path(os.environ['D4R_SWIN_CONTROL_DIR'])
        candidate = Path(os.environ['D4R_SWIN_CANDIDATE_DIR'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['python3', str(ROOT/'kernels/tools/swin_synthetic.py'), str(root/'fixtures')], check=True)
            for fixture in sorted((root/'fixtures').iterdir()):
                with self.subTest(fixture=fixture.name):
                    name = 'rrlite_' + fixture.name.split('-s')[0] + '_4x4'
                    outputs = []
                    for label, kernels in [('control', control), ('candidate', candidate)]:
                        out = root/fixture.name/label; out.mkdir(parents=True)
                        subprocess.run([runner, str(kernels/(name+'.hsaco')), name, str(fixture), str(out), '1'],
                                       check=True, capture_output=True, text=True)
                        outputs.append(out)
                    for p in outputs[0].glob('alloc-*.bin'):
                        self.assertEqual(p.read_bytes(), (outputs[1]/p.name).read_bytes(), p.name)
                    for i in range(3):
                        self.assertEqual((fixture/f'alloc-{i}.bin').read_bytes(),
                                         (outputs[1]/f'alloc-{i}.bin').read_bytes())
