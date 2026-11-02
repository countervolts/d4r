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


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS') == '1', 'opt-in GPU test')
class MotionLdsKernelGpuTests(unittest.TestCase):
    def run_kernel(self, name, width, height, stride, allocation_height, depth, motion, radius, reverse):
        runner = os.environ['D4R_DUMP_RUNNER']
        kernel = ROOT / 'build/wine-nvcuda/generated' / (os.environ.get('D4R_GPU_ARCH', 'gfx1101') + '.hsaco')
        # Guard the tail beyond the allocation too. Generic output is pre-copied,
        # matching the bridge's original full-plane copy followed by dilation.
        guard = [0xA5A5A5A5] * 32
        original = motion + guard
        arrays = [struct.pack(f'<{len(depth)}f', *depth),
                  struct.pack(f'<{len(original)}I', *original),
                  struct.pack(f'<{len(original)}I', *(original if name == 'd4r_motion_dilate' else [0xA5A5A5A5]*len(original)))]
        addresses = (0x100000000, 0x200000000, 0x300000000)
        args = struct.pack('<5Q4I2f2I', *addresses, stride*4, stride*4,
                           width, height, width, height, 0., 0., radius, reverse)
        if name.startswith('d4r_motion_dilate_lds'):
            args += struct.pack('<I', allocation_height)
            grid = f'{(stride+15)//16} {(allocation_height+15)//16} 1 256 1 1 0'
        else:
            grid = f'{(width*height+127)//128} 1 1 128 1 1 0'
        with tempfile.TemporaryDirectory() as t:
            directory = Path(t); output = directory/'out'; output.mkdir()
            (directory/'args.bin').write_bytes(args)
            manifest = [f'kernel {name}', f'launch {grid}', f'args {len(args)}']
            for i, data in enumerate(arrays):
                (directory/f'alloc-{i}.bin').write_bytes(data)
                manifest += [f'alloc {i} {hex(addresses[i])} {len(data)}', f'pointer {i*8} {i} 0']
            (directory/'manifest.txt').write_text('\n'.join(manifest)+'\n')
            subprocess.run([runner, str(kernel), name, str(directory), str(output), '1'],
                           check=True, capture_output=True, timeout=30)
            for i in (0, 1):
                self.assertEqual((output/f'alloc-{i}.bin').read_bytes(), arrays[i], 'input mutated')
            result = (output/'alloc-2.bin').read_bytes()
            self.assertEqual(result[-128:], struct.pack('<32I', *guard), 'tail guard overwritten')
            values = struct.unpack(f'<{len(original)}I', result)
            for y in range(allocation_height):
                for x in range(stride):
                    if x >= width or y >= height:
                        self.assertEqual(values[y*stride+x], motion[y*stride+x], 'padding changed')
            return result

    def test_image_edges_block_edges_and_lexicographic_sky_ties(self):
        width, height, stride, allocation_height = 35, 33, 48, 36
        # Equal-distance sources straddle x=16/y=16 tile boundaries. Include
        # corners, circle boundary, and same-x/different-y tie candidates.
        source_sets = [
            [(0, 0), (34, 32), (13, 16), (19, 16), (8, 31)],
            [(0, 0), (34, 32), (16, 13), (16, 19), (8, 31)],
        ]
        for sources in source_sets:
            for reverse in (False, True):
                for radius in (1, 2):
                    with self.subTest(reverse=reverse, radius=radius, sources=sources):
                        depth = [float('nan')] * (stride*allocation_height)
                        motion = list(range(stride*allocation_height))
                        for y in range(height):
                            for x in range(width):
                                depth[y*stride+x] = 0. if reverse else 1.
                        for x, y in sources:
                            depth[y*stride+x] = .9 if reverse else .1
                        output = self.run_kernel('d4r_motion_dilate_lds', width, height, stride, allocation_height,
                                                 depth, motion, radius, reverse)
                        values = struct.unpack(f'<{len(output)//4}I', output)
                        for y in range(height):
                            for x in range(width):
                                nearby = sorted(((x-sx)**2+(y-sy)**2, sx, sy) for sx, sy in sources)
                                expected = motion[y*stride+x]
                                if nearby[0][0] <= 64:
                                    _, sx, sy = nearby[0]
                                    expected = motion[sy*stride+sx]
                                self.assertEqual(values[y*stride+x], expected, (x, y))

    def test_randomized_bit_exact_generic_equivalence(self):
        import random
        rng = random.Random(0xD4A)
        for width, height in ((1, 1), (2, 3), (15, 17), (17, 31), (33, 19), (65, 37)):
            stride, allocation_height = ((width+7)//8+1)*8, height+3
            for scene in ('empty', 'sparse', 'dense', 'special'):
                forward = [float('nan')] * (stride*allocation_height)
                for y in range(height):
                    for x in range(width):
                        d = 1.
                        if scene == 'dense' or (scene == 'sparse' and rng.random() < .12):
                            d = rng.choice((.1, .5, .9, .99))
                        elif scene == 'special':
                            d = rng.choice((0., 1., 1.e-7, 1.-1.e-7, float('nan'), float('inf'), -float('inf'), .98, .99999994))
                        forward[y*stride+x] = d
                motion = [rng.getrandbits(32) for _ in forward]
                for reverse in (False, True):
                    depth = [1.-d for d in forward] if reverse else forward
                    for radius in (1, 2):
                        with self.subTest(size=(width, height), scene=scene, reverse=reverse, radius=radius):
                            old = self.run_kernel('d4r_motion_dilate', width, height, stride, allocation_height,
                                                  depth, motion, radius, reverse)
                            new = self.run_kernel('d4r_motion_dilate_lds', width, height, stride, allocation_height,
                                                  depth, motion, radius, reverse)
                            self.assertEqual(old, new, 'LDS must have zero mismatched bytes')

    def test_thin_sky_gate_matches_global_run_reference(self):
        # Wide roof plus a long pole, horizontal wire, diagonal and a short tip.
        # Moving the shapes across x/y=16 exercises the extra morphology halo.
        for shift in (0, 3):
            width, height, stride, ah = 47, 39, 56, 42
            foreground = {(x, y) for y in range(24, 39) for x in range(20+shift, 47)}
            foreground |= {(8+shift, y) for y in range(2, 29)}
            foreground |= {(x, 8+shift) for x in range(2, 38)}
            foreground |= {(x, x//2+12) for x in range(2, 20)}
            def short_run(x, y, dx, dy):
                length = 1
                for sign in (-1, 1):
                    step = 1
                    while (x+sign*step*dx, y+sign*step*dy) in foreground:
                        length += 1
                        step += 1
                return length <= 3
            eligible = [(x, y) for x, y in foreground
                        if short_run(x,y,1,0) or short_run(x,y,0,1)]
            self.assertIn((8+shift, 18), eligible, 'long pole must survive')
            self.assertIn((18, 8+shift), eligible, 'horizontal wire must survive')
            self.assertNotIn((30+shift, 28), eligible, 'broad roof must be rejected')
            for reverse in (False, True):
                for radius in (1, 2):
                    with self.subTest(shift=shift, reverse=reverse, radius=radius):
                        depth = [float('nan')]*(stride*ah)
                        motion = list(range(stride*ah))
                        for y in range(height):
                            for x in range(width):
                                depth[y*stride+x] = (.9 if reverse else .1) if (x,y) in foreground else (0. if reverse else 1.)
                        out = self.run_kernel('d4r_motion_dilate_lds_thin', width, height, stride, ah,
                                              depth, motion, radius, reverse)
                        values = struct.unpack(f'<{len(out)//4}I',out)
                        for y in range(height):
                            for x in range(width):
                                bx, by = x, y
                                # Opaque dilation is deliberately retained even where sky fill is gated.
                                if (x,y) not in foreground:
                                    found = False
                                    for dy in range(-radius,radius+1):
                                        for dx in range(-radius,radius+1):
                                            nx, ny = max(0,min(width-1,x+dx)), max(0,min(height-1,y+dy))
                                            if not found and (nx,ny) in foreground:
                                                bx, by = nx, ny
                                                found = True
                                    near = sorted(((x-sx)**2+(y-sy)**2,sx,sy) for sx,sy in eligible)
                                    if near and near[0][0] <= 64:
                                        _, bx, by = near[0]
                                self.assertEqual(values[y*stride+x], motion[by*stride+bx], (x,y))
