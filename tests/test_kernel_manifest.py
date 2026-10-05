"""Native SR code must never be authorized by a colliding denoiser entry."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'kernels/tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('d4r_kernel_manifest', TOOLS / 'kernel_manifest.py')
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


def fatbin(ptx):
    entry = bytearray(64)
    struct.pack_into('<HHIQ', entry, 0, 1, 0, 64, len(ptx))
    payload = entry + ptx
    return struct.pack('<IHHQ', 0xBA55ED50, 1, 16, len(payload)) + payload


class KernelManifestTests(unittest.TestCase):
    def test_denoiser_owned_name_comes_only_from_the_denoiser_dll(self):
        # A cuda_dldn_engine_* kernel may only be authorized by the DLL passed as --rr-dll; the
        # Super Resolution libraries define names of that shape too, and their modules differ.
        sr = b'.visible .entry dl4rt_input_kernel() { ret; }\n'
        rr = (b'.visible .entry dl4rt_input_kernel() { mov.u32 %r1, 1; ret; }\n'
              b'.visible .entry cuda_dldn_engine_swin_enc1_kernel() { ret; }\n')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sr.dll').write_bytes(fatbin(sr))
            (root / 'rr.dll').write_bytes(fatbin(rr))
            for name in ('dl4rt_input_kernel', 'cuda_dldn_engine_swin_enc1_kernel'):
                (root / (name + '.hsaco')).touch()
            subprocess.run([sys.executable, str(TOOLS / 'kernel_manifest.py'), str(root),
                            str(root / 'sr.dll'), '--rr-dll', str(root / 'rr.dll')],
                           check=True, capture_output=True)
            rows = [line.split() for line in (root / 'd4r-kernels.txt').read_text().splitlines()
                    if line and not line.startswith('#')]
            self.assertEqual(rows, [
                ['cuda_dldn_engine_swin_enc1_kernel', f'{manifest.fnv1a64(rr):016x}'],
                ['dl4rt_input_kernel', f'{manifest.fnv1a64(sr):016x}'],
            ])

    def test_rr_only_input_refuses_unproven_sr_binary(self):
        rr = b'.visible .entry dl4rt_input_kernel() { ret; }\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'rr.dll').write_bytes(fatbin(rr))
            (root / 'dl4rt_input_kernel.hsaco').touch()
            result = subprocess.run([sys.executable, str(TOOLS / 'kernel_manifest.py'), str(root),
                                     '--rr-dll', str(root / 'rr.dll')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('no DLL defines dl4rt_input_kernel', result.stderr)
            self.assertFalse((root / 'd4r-kernels.txt').exists())


if __name__ == '__main__':
    unittest.main()
