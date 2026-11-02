"""Prepared geometry keeps rendered output and readable temporal state exact."""
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS') == '1', 'opt-in GPU test')
class PreparedWireGpuTests(unittest.TestCase):
    def test_prepared_matches_reference_with_poisoned_invalid_history(self):
        runner = os.environ['D4R_DUMP_RUNNER']
        kernel = ROOT/'build/wine-nvcuda/generated'/(os.environ.get('D4R_GPU_ARCH', 'gfx1101')+'.hsaco')
        w,h,rw,rh=65,67,21,22
        pitch,dp,mp,cp,idp=w*8+16,w*4+12,w*4+20,rw*8+16,rw*4+12
        pixels=w*h
        rng=random.Random(746)
        rgba=lambda: struct.pack('<4e',rng.uniform(.05,1),rng.uniform(.05,1),rng.uniform(.05,1),.5)
        current=b''.join(b''.join(rgba() for _ in range(w))+b'\xa5'*16 for _ in range(h))
        source=b''.join(struct.pack('<4e',.125,.125,.125,.75)*rw+b'\xa5'*16 for _ in range(rh))
        previous=bytearray(b''.join(struct.pack('<4e',.125,.125,.125,.75)*w+b'\xa5'*16 for _ in range(h)))
        ages=bytes(rng.choice((0,3,4,15,16,255)) for _ in range(pixels))
        previous_depth=bytearray(struct.pack('<f',.5)*pixels)
        for i,age in enumerate(ages):
            if age==255:
                y,x=divmod(i,w)
                previous[y*pitch+x*8:y*pitch+x*8+8]=struct.pack('<4H',0x7e00,0x7e00,0x7e00,0x7e00)
                previous_depth[i*4:i*4+4]=struct.pack('<f',float('nan'))
        for flags in (0,1,2,3):
            for jitter in ((0.,0.),(.25,-.25),(-1.,1.)):
                for pattern in ('sky','flat','wire'):
                    with self.subTest(flags=flags,jitter=jitter,pattern=pattern), tempfile.TemporaryDirectory() as tmp:
                        d=Path(tmp);out=d/'out';out.mkdir()
                        z=[1. if pattern=='sky' else .5 if pattern=='flat' or x==32 else 1.
                           for y in range(h) for x in range(w)]
                        if pattern=='wire':z[10*w+31]=float('nan')
                        depth=b''.join(struct.pack('<'+'f'*w,*z[y*w:(y+1)*w])+b'\xa5'*12 for y in range(h))
                        arrays=[current,source,depth,b''.join(bytes(w*4)+b'\xa5'*20 for _ in range(h)),
                                bytes(previous),ages,bytes(previous_depth),b'\xa5'*(pitch*h),
                                b'\xa5'*(pixels+64),b'\xa5'*(pixels*4+64),
                                b''.join(struct.pack('<f',.5)*rw+b'\xa5'*12 for _ in range(rh)),
                                b'\xa5'*(pitch*h),b'\xa5'*(pixels+64),
                                b'\xa5'*(((w+31)//32)*h*4+64),b'\xa5'*(pixels+64),
                                b'\xa5'*(mp*h),bytes(1136),bytes(64)]
                        addresses=[(i+1)*0x100000000 for i in range(len(arrays))]
                        def reset():
                            for i,data in enumerate(arrays):(d/f'alloc-{i}.bin').write_bytes(data)
                        def launch(name,args,pointers,grid):
                            (d/'args.bin').write_bytes(args)
                            manifest=['kernel '+name,'launch '+' '.join(map(str,(*grid,1,256,1,1,0))),f'args {len(args)}']
                            for i,data in enumerate(arrays):manifest.append(f'alloc {i} {hex(addresses[i])} {len(data)}')
                            manifest += [f'pointer {offset} {allocation} 0' for offset,allocation in pointers]
                            (d/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                            subprocess.run([runner,str(kernel),name,str(d),str(out),'1'],check=True,capture_output=True)
                            for i in range(len(arrays)):(d/f'alloc-{i}.bin').write_bytes((out/f'alloc-{i}.bin').read_bytes())
                        args=struct.pack('<17Q5I2fI',*addresses[:12],cp,dp,mp,pitch,idp,w,h,rw,rh,1,*jitter,flags)
                        base_pointers=[(i*8,i) for i in range(12)]
                        reset();launch('d4r_wire_history',args,base_pointers,((w+15)//16,(h+15)//16))
                        expected={i:(out/f'alloc-{i}.bin').read_bytes() for i in (7,8,9,11)}
                        reset()
                        native=struct.pack('<8Q4I2f2Q2I',addresses[2],addresses[3],addresses[16],addresses[17],addresses[15],
                                           dp,mp,mp,w,h,rw,rh,*jitter,addresses[13],addresses[14],flags,0)
                        native_pointers=[(0,2),(8,3),(16,16),(24,17),(32,15),(88,13),(96,14)]
                        launch('d4r_native_motion_prepare',native,native_pointers,((w+31)//32,(h+31)//32))
                        prepared=args+struct.pack('<2Q',addresses[12],addresses[14])
                        launch('d4r_wire_history_preclassified',prepared,base_pointers+[(168,12),(176,14)],((w+15)//16,(h+15)//16))
                        launch('d4r_wire_history_apply',prepared[:176],base_pointers+[(168,12)],((w+15)//16,(h+15)//16))
                        actual={i:(out/f'alloc-{i}.bin').read_bytes() for i in range(len(arrays))}
                        self.assertEqual(actual[0],expected[7])
                        self.assertEqual(actual[8],expected[8])
                        for i in (1,2,3,4,5,6,10,16,17):self.assertEqual(actual[i],arrays[i])
                        for i in range(pixels):
                            if expected[8][i]!=255:
                                y,x=divmod(i,w);offset=y*pitch+x*8
                                self.assertEqual(actual[11][offset:offset+8],expected[11][offset:offset+8])
                                self.assertEqual(actual[9][i*4:i*4+4],expected[9][i*4:i*4+4])
                        for y in range(h):self.assertEqual(actual[11][y*pitch+w*8:(y+1)*pitch],b'\xa5'*16)
                        for i,start in ((8,pixels),(9,pixels*4),(12,pixels),(13,((w+31)//32)*h*4),(14,pixels)):
                            self.assertEqual(actual[i][start:],b'\xa5'*64)
