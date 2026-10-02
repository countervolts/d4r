"""Optional GPU regression: independently specified thin-foreground velocity coverage.

D4R_RUN_GPU_TESTS=1 D4R_DUMP_RUNNER=/path/to/dump_runner python3 -m unittest discover -s tests -v
Build the embedded kernels first with scripts/build_d4r_motion_kernels.sh.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS') == '1', 'opt-in GPU test')
class MotionKernelGpuTests(unittest.TestCase):
    def test_low_resolution_opaque_edges_and_padded_rows(self):
        runner = os.environ['D4R_DUMP_RUNNER']
        kernel = ROOT / 'build/wine-nvcuda/generated' / (os.environ.get('D4R_GPU_ARCH', 'gfx1101') + '.hsaco')
        width, height, stride = 5, 3, 8
        sentinel = 0xA5A5A5A5
        for reversed_depth in (False, True):
            for radius in (1, 2):
                with self.subTest(reversed_depth=reversed_depth, radius=radius), tempfile.TemporaryDirectory() as t:
                    directory = Path(t); output = directory / 'out'; output.mkdir()
                    depth = [0.1 if reversed_depth else 0.9] * (stride*height)
                    depth[stride+2] = 0.9 if reversed_depth else 0.1
                    motion = list(range(stride*height))
                    arrays = [struct.pack('<24f', *depth), struct.pack('<24I', *motion), struct.pack('<24I', *([sentinel]*24))]
                    addresses = (0x100000000, 0x200000000, 0x300000000)
                    args = struct.pack('<5Q4I2f2I', *addresses, stride*4, stride*4,
                                       width, height, width, height, 0.0, 0.0, radius, reversed_depth)
                    (directory/'args.bin').write_bytes(args)
                    manifest = ['kernel d4r_motion_dilate', 'launch 1 1 1 128 1 1 0', 'args 72']
                    for i, data in enumerate(arrays):
                        (directory/f'alloc-{i}.bin').write_bytes(data)
                        manifest += [f'alloc {i} {hex(addresses[i])} {len(data)}', f'pointer {i*8} {i} 0']
                    (directory/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                    subprocess.run([runner, str(kernel), 'd4r_motion_dilate', str(directory), str(output), '1'], check=True, capture_output=True)
                    self.assertEqual((output/'alloc-0.bin').read_bytes(), arrays[0])
                    self.assertEqual((output/'alloc-1.bin').read_bytes(), arrays[1])
                    expected = [sentinel] * 24
                    for y in range(height):
                        for x in range(width):
                            expected[y*stride+x] = motion[stride+2] if abs(x-2) <= radius and abs(y-1) <= radius else motion[y*stride+x]
                    self.assertEqual((output/'alloc-2.bin').read_bytes(), struct.pack('<24I', *expected))

    def test_foreground_coverage_padding_and_input_immutability(self):
        runner = os.environ['D4R_DUMP_RUNNER']
        kernel = ROOT / 'build/wine-nvcuda/generated' / (os.environ.get('D4R_GPU_ARCH', 'gfx1101') + '.hsaco')
        sentinel = 0xA5A5A5A5
        for reversed_depth in (False, True):
            for radius in (1, 2):
                for jitter in ((0.0, 0.0), (0.25, -0.25)):
                    with self.subTest(reversed_depth=reversed_depth, radius=radius, jitter=jitter), tempfile.TemporaryDirectory() as t:
                        directory = Path(t); output = directory / 'out'; output.mkdir()
                        depth = [1000.0] * 16
                        for y in range(2):
                            for x in range(3):
                                depth[y*8+x] = 0.1 if reversed_depth else 0.9
                        depth[1] = 0.9 if reversed_depth else 0.1
                        motion = list(range(96))
                        arrays = [struct.pack('<16f', *depth), struct.pack('<96I', *motion), struct.pack('<96I', *([sentinel]*96))]
                        addresses = (0x100000000, 0x200000000, 0x300000000)
                        args = struct.pack('<5Q4I2f2I', *addresses, 32, 64, 3, 2, 9, 6, *jitter, radius, reversed_depth)
                        (directory/'args.bin').write_bytes(args)
                        manifest = ['kernel d4r_motion_dilate', 'launch 1 1 1 128 1 1 0', 'args 72']
                        for i, data in enumerate(arrays):
                            (directory/f'alloc-{i}.bin').write_bytes(data)
                            manifest += [f'alloc {i} {hex(addresses[i])} {len(data)}', f'pointer {i*8} {i} 0']
                        (directory/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                        subprocess.run([runner, str(kernel), 'd4r_motion_dilate', str(directory), str(output), '1'], check=True, capture_output=True)
                        self.assertEqual((output/'alloc-0.bin').read_bytes(), arrays[0])
                        self.assertEqual((output/'alloc-1.bin').read_bytes(), arrays[1])
                        expected = [sentinel] * 96
                        # Known single foreground cell: all neighbouring background follows its velocity.
                        # Foreground itself preserves its original per-pixel velocity; row padding is untouched.
                        for y in range(6):
                            for x in range(9):
                                inside = (3 <= x <= 5 and y <= 2) if jitter == (0.0, 0.0) else (2 <= x <= 4 and y <= 3)
                                expected[y*16+x] = motion[y*16+x] if inside else motion[1*16+4] if jitter == (0.0, 0.0) else motion[2*16+3]
                        self.assertEqual((output/'alloc-2.bin').read_bytes(), struct.pack('<96I', *expected))

    def test_automatic_sky_fill_circle_nearest_ties_and_empty_sky(self):
        runner = os.environ['D4R_DUMP_RUNNER']
        kernel = ROOT / 'build/wine-nvcuda/generated' / (os.environ.get('D4R_GPU_ARCH', 'gfx1101') + '.hsaco')
        width, height, stride = 23, 23, 32
        sentinel = 0xA5A5A5A5
        for reversed_depth in (False, True):
            for radius in (1, 2):
                for empty, near_far in ((False, False), (True, False), (False, True)):
                    with self.subTest(reversed_depth=reversed_depth, radius=radius, empty=empty, near_far=near_far), tempfile.TemporaryDirectory() as t:
                        directory = Path(t); output = directory / 'out'; output.mkdir()
                        depth = [1000.0] * (stride*height)
                        for y in range(height):
                            for x in range(width):
                                depth[y*stride+x] = (1.e-7 if near_far else 0.0) if reversed_depth else 1.0
                        sources = [] if empty else [(8, 11), (14, 11)]
                        for x, y in sources:
                            depth[y*stride+x] = (1.5e-7 if near_far else 0.9) if reversed_depth else (0.99 if near_far else 0.1)
                        motion = list(range(stride*height))
                        arrays = [struct.pack(f'<{len(depth)}f', *depth), struct.pack(f'<{len(motion)}I', *motion), struct.pack(f'<{len(motion)}I', *([sentinel]*len(motion)))]
                        addresses = (0x100000000, 0x200000000, 0x300000000)
                        args = struct.pack('<5Q4I2f2I', *addresses, stride*4, stride*4, width, height, width, height, 0.0, 0.0, radius, reversed_depth)
                        (directory/'args.bin').write_bytes(args)
                        manifest = ['kernel d4r_motion_dilate', 'launch 5 1 1 128 1 1 0', 'args 72']
                        for i, data in enumerate(arrays):
                            (directory/f'alloc-{i}.bin').write_bytes(data)
                            manifest += [f'alloc {i} {hex(addresses[i])} {len(data)}', f'pointer {i*8} {i} 0']
                        (directory/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                        subprocess.run([runner, str(kernel), 'd4r_motion_dilate', str(directory), str(output), '1'], check=True, capture_output=True)
                        self.assertEqual((output/'alloc-0.bin').read_bytes(), arrays[0])
                        self.assertEqual((output/'alloc-1.bin').read_bytes(), arrays[1])
                        expected = [sentinel] * len(motion)
                        for y in range(height):
                            for x in range(width):
                                candidates = sorted(((x-sx)**2+(y-sy)**2, sx, sy) for sx, sy in sources)
                                expected[y*stride+x] = motion[y*stride+x]
                                if candidates and candidates[0][0] <= 64:
                                    _, sx, sy = candidates[0]
                                    expected[y*stride+x] = motion[sy*stride+sx]
                        self.assertEqual((output/'alloc-2.bin').read_bytes(), struct.pack(f'<{len(expected)}I', *expected))
