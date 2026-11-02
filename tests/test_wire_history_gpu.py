"""Opt-in GPU checks for wire color, history eligibility, padding and immutable inputs."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS') == '1', 'opt-in GPU test')
class WireHistoryGpuTests(unittest.TestCase):
    def test_wire_source_history_reset_and_expiry(self):
        runner = os.environ['D4R_DUMP_RUNNER']
        kernel = Path(os.environ.get('D4R_WIRE_TEST_KERNEL', str(ROOT / 'build/wine-nvcuda/generated' /
                       (os.environ.get('D4R_GPU_ARCH', 'gfx1101') + '.hsaco'))))
        w, h = 65, 67  # partial workgroups on both axes
        pitch, dp, mp = w*8+16, w*4+12, w*4+20
        pixels = w*h
        rgba = struct.pack('<4e', .5, .5, .5, .5)
        observed = struct.pack('<4e', .125, .125, .125, .75)
        repaired = struct.pack('<4e', .125, .125, .125, .5)
        current = b''.join(rgba*w + b'\xa5'*16 for _ in range(h))
        color = b''.join(observed*w + b'\xa5'*16 for _ in range(h))
        native = b''.join(struct.pack('<'+'f'*w, *[.5 if x == 32 else 1. for x in range(w)]) +
                          b'\xa5'*12 for _ in range(h))
        motion = b''.join(b'\0'*(w*4) + b'\xa5'*20 for _ in range(h))
        for sky_only in (False, True):
            for conservative in (False, True):
                for sparse in (False, True):
                    for mode in ('source', 'history', 'reset', 'expired', 'disoccluded', 'missing', 'neighbor', 'far', 'backdrop'):
                        for jitter in (0, -1, 1):
                            with self.subTest(mode=mode, jitter=jitter, conservative=conservative, sparse=sparse, sky_only=sky_only), tempfile.TemporaryDirectory() as tmp:
                                directory = Path(tmp); output = directory/'out'; output.mkdir()
                                source = mode in ('source', 'backdrop')
                                history = mode in ('history', 'expired', 'disoccluded', 'neighbor')
                                z = .999988 if mode == 'far' else .5
                                near = b''.join(struct.pack('<'+'f'*w, *[z if x == 32 else .9 if mode == 'backdrop' else 1. for x in range(w)]) + b'\xa5'*12 for _ in range(h))
                                previous = color
                                if mode == 'neighbor':
                                    previous = b''.join(b''.join(observed if x == 32-jitter else struct.pack('<4e', .01, .01, .01, .75) for x in range(w)) + b'\xa5'*16 for _ in range(h))
                                input_depth = b''.join(struct.pack('<f', z+1e-6 if mode == 'far' else .5 if source else 1.)*w + b'\xa5'*12 for _ in range(h))
                                arrays = [current, color, near, motion, previous,
                                          bytes([(4 if conservative else 16) if mode == 'expired' else (3 if conservative else 15)])*pixels,
                                          struct.pack('<f', 0. if mode == 'disoccluded' else .5)*pixels,
                                          b'\xa5'*(pitch*h), b'\xa5'*(pixels+64), b'\xa5'*(pixels*4+64),
                                          input_depth, b'\xa5'*(pitch*h)]
                                if sparse: arrays.append(b'\xa5'*(pixels+64))
                                addresses = [(i+1)*0x100000000 for i in range(len(arrays))]
                                args = struct.pack('<17Q5I2fI', *addresses[:12], pitch, dp, mp, pitch, dp,
                                                   w, h, w, h, int(history), jitter, -jitter, int(conservative) | (2 if sky_only else 0))
                                if sparse: args += struct.pack('<Q', addresses[12])
                                (directory/'args.bin').write_bytes(args)
                                manifest = ['kernel '+('d4r_wire_history_sparse' if sparse else 'd4r_wire_history'), 'launch 5 5 1 256 1 1 0', f'args {len(args)}']
                                for i, data in enumerate(arrays):
                                    (directory/f'alloc-{i}.bin').write_bytes(data)
                                    manifest += [f'alloc {i} {hex(addresses[i])} {len(data)}', f'pointer {168 if i == 12 else i*8} {i} 0']
                                (directory/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                                if sparse:
                                    subprocess.run([runner, str(kernel), 'd4r_wire_history_sparse', str(directory), str(output), '1'], check=True, capture_output=True)
                                    for i in range(len(arrays)):
                                        (directory/f'alloc-{i}.bin').write_bytes((output/f'alloc-{i}.bin').read_bytes())
                                    manifest[0]='kernel d4r_wire_history_apply'
                                    (directory/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                                    subprocess.run([runner, str(kernel), 'd4r_wire_history_apply', str(directory), str(output), '1'], check=True, capture_output=True)
                                else:
                                    subprocess.run([runner, str(kernel), 'd4r_wire_history', str(directory), str(output), '1'], check=True, capture_output=True)
                                for i in ((1, 2, 3, 4, 5, 6, 10) if sparse else (0, 1, 2, 3, 4, 5, 6, 10)):
                                    self.assertEqual((output/f'alloc-{i}.bin').read_bytes(), arrays[i])
                                expected = bytearray(current)
                                cache = bytearray(current)
                                age = bytearray(b'\xff'*pixels + b'\xa5'*64)
                                depth = bytearray(struct.pack('<f', 1.)*pixels + b'\xa5'*64)
                                for y in range(16, h-16):
                                    if mode == 'backdrop' and sky_only: continue
                                    x = 32-jitter
                                    offset, index = y*pitch+x*8, y*w+x
                                    depth[index*4:index*4+4] = struct.pack('<f', z)
                                    if mode in ('source', 'history', 'neighbor', 'backdrop') or (mode == 'far' and not conservative):
                                        value = .01 if mode == 'neighbor' and not conservative else .125
                                        expected[offset:offset+8] = struct.pack('<4e', value, value, value, .5)
                                        cache[offset:offset+8] = struct.pack('<4e', value, value, value, .75)
                                        age[index] = 0 if source or mode == 'far' else 4 if conservative else 16
                                self.assertEqual((output/f'alloc-{0 if sparse else 7}.bin').read_bytes(), expected)
                                self.assertEqual((output/'alloc-8.bin').read_bytes(), age)
                                self.assertEqual((output/'alloc-9.bin').read_bytes(), depth)
                                self.assertEqual((output/'alloc-11.bin').read_bytes(), cache)
