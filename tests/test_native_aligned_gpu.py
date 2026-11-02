"""Jitter-aligned native 3x3 motion without the extended halo."""
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS') == '1', 'opt-in GPU test')
class NativeAlignedGpuTests(unittest.TestCase):
    def test_jitter_nearest_ties_partial_tiles_and_padding(self):
        runner = os.environ['D4R_DUMP_RUNNER']
        kernel = Path(os.environ.get('D4R_NATIVE_TEST_KERNEL', str(ROOT/'build/wine-nvcuda/generated'/
                       (os.environ.get('D4R_GPU_ARCH', 'gfx1101')+'.hsaco'))))
        w, h, rw, rh = 33, 35, 11, 12
        dp, gp, op = w+3, w+5, w+7
        sentinel = 0xa5a5a5a5
        f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
        depth = [f32(.1+((x*3+y*7)%5)*.1) for y in range(h) for x in range(w)]
        geometry = [struct.unpack('<I', struct.pack('<2e', .001*x, .001*y))[0] for y in range(h) for x in range(w)]
        for jx, jy in ((0., 0.), (.25, -.25), (-.5, .5), (1., -1.)):
            with self.subTest(jitter=(jx, jy)), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp); out = d/'out'; out.mkdir()
                padded_d = [float('nan')]*(dp*h)
                padded_g = [sentinel]*(gp*h)
                expected = [sentinel]*(op*h)
                for y in range(h):
                    for x in range(w):
                        padded_d[y*dp+x] = depth[y*w+x]
                        padded_g[y*gp+x] = geometry[y*w+x]
                        cx = min(w-1, max(0, math.floor(f32(f32(x+.5)+f32(f32(jx*w)/rw)))))
                        cy = min(h-1, max(0, math.floor(f32(f32(y+.5)+f32(f32(jy*h)/rh)))))
                        best, bx, by = depth[cy*w+cx], cx, cy
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                nx, ny = cx+dx, cy+dy
                                if 0 <= nx < w and 0 <= ny < h and depth[ny*w+nx] < best:
                                    best, bx, by = depth[ny*w+nx], nx, ny
                        expected[y*op+x] = geometry[by*w+bx]
                arrays = [struct.pack('<'+'f'*len(padded_d), *padded_d),
                          struct.pack('<'+'I'*len(padded_g), *padded_g), bytes(1136), bytes(64),
                          struct.pack('<'+'I'*(op*h), *([sentinel]*(op*h)))]
                addresses = [(i+1)*0x100000000 for i in range(5)]
                args = struct.pack('<8Q4I2f', *addresses, dp*4, gp*4, op*4, w, h, rw, rh, jx, jy)
                (d/'args.bin').write_bytes(args)
                manifest = ['kernel d4r_native_motion_compose_aligned', 'launch 3 3 1 256 1 1 0', 'args 88']
                for i, data in enumerate(arrays):
                    (d/f'alloc-{i}.bin').write_bytes(data)
                    manifest += [f'alloc {i} {hex(addresses[i])} {len(data)}', f'pointer {i*8} {i} 0']
                (d/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                subprocess.run([runner, str(kernel), 'd4r_native_motion_compose_aligned', str(d), str(out), '1'],
                               check=True, capture_output=True)
                for i in range(4):
                    self.assertEqual((out/f'alloc-{i}.bin').read_bytes(), arrays[i])
                self.assertEqual((out/'alloc-4.bin').read_bytes(), struct.pack('<'+'I'*len(expected), *expected))

    def test_hybrid_preserves_game_vectors_away_from_silhouettes(self):
        runner = os.environ['D4R_DUMP_RUNNER']
        kernel = ROOT/'build/wine-nvcuda/generated'/(os.environ.get('D4R_GPU_ARCH', 'gfx1101')+'.hsaco')
        w,h,rw,rh=33,35,11,12
        pitch=(w+3)*4; gp=(rw+3)*4; sentinel=0xa5a5a5a5
        game=[struct.unpack('<I',struct.pack('<2e',.001*x,.001*y))[0] for y in range(rh) for x in range(rw)]
        padded_game=[sentinel]*((rw+3)*rh)
        for y in range(rh): padded_game[y*(rw+3):y*(rw+3)+rw]=game[y*rw:(y+1)*rw]
        for depth in (1.,.5):
            with self.subTest(depth=depth),tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp);out=d/'out';out.mkdir()
                arrays=[struct.pack('<f',depth)*(w+3)*h,struct.pack('<I',0x3e003e00)*(w+3)*h,bytes(1136),bytes(64),
                        struct.pack('<I',sentinel)*(w+3)*h,struct.pack('<'+'I'*len(padded_game),*padded_game)]
                addresses=[(i+1)*0x100000000 for i in range(6)]
                args=struct.pack('<8Q4I2f2Q',*addresses[:5],pitch,pitch,pitch,w,h,rw,rh,.25,-.25,addresses[5],gp)
                (d/'args.bin').write_bytes(args)
                manifest=['kernel d4r_native_motion_compose_hybrid','launch 3 3 1 256 1 1 0','args 104']
                for i,data in enumerate(arrays):
                    (d/f'alloc-{i}.bin').write_bytes(data)
                    manifest += [f'alloc {i} {hex(addresses[i])} {len(data)}',f'pointer {88 if i==5 else i*8} {i} 0']
                (d/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                subprocess.run([runner,str(kernel),'d4r_native_motion_compose_hybrid',str(d),str(out),'1'],check=True,capture_output=True)
                for i in (0,1,2,3,5): self.assertEqual((out/f'alloc-{i}.bin').read_bytes(),arrays[i])
                expected=[sentinel]*((w+3)*h)
                for y in range(h):
                    for x in range(w): expected[y*(w+3)+x]=game[int((y+.5)*rh/h)*rw+int((x+.5)*rw/w)]
                self.assertEqual((out/'alloc-4.bin').read_bytes(),struct.pack('<'+'I'*len(expected),*expected))
