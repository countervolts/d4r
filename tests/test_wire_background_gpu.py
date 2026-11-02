"""Fractional wire coverage must preserve a different foreground surface."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS')=='1','opt-in GPU test')
class WireBackgroundGpuTests(unittest.TestCase):
    def test_sky_wire_does_not_spill_onto_other_geometry(self):
        runner=os.environ['D4R_DUMP_RUNNER']
        kernel=Path(os.environ.get('D4R_WIRE_TEST_KERNEL',str(ROOT/'build/wine-nvcuda/generated'/'gfx1101.hsaco')))
        w,h=65,67
        pitch,dp,mp=w*8+16,w*4+12,w*4+20
        pixels=w*h
        rgba=struct.pack('<4e',.5,.5,.5,.5)
        observed=struct.pack('<4e',.125,.125,.125,.75)
        current=b''.join(rgba*w+b'\xa5'*16 for _ in range(h))
        color=b''.join(observed*w+b'\xa5'*16 for _ in range(h))
        depth=b''.join(struct.pack('<'+'f'*w,*[
            .5 if y==32 else .9 if x<=31 and y<=31 else 1. for x in range(w)])+b'\xa5'*12 for y in range(h))
        motion=b''.join(bytes(w*4)+b'\xa5'*20 for _ in range(h))
        input_depth=b''.join(struct.pack('<f',.5)*w+b'\xa5'*12 for _ in range(h))
        for sparse in (False,True):
            with self.subTest(sparse=sparse), tempfile.TemporaryDirectory() as temp:
                directory=Path(temp);out=directory/'out';out.mkdir()
                arrays=[current,color,depth,motion,color,b'\xff'*pixels,struct.pack('<f',1.)*pixels,
                        b'\xa5'*(pitch*h),b'\xa5'*(pixels+64),b'\xa5'*(pixels*4+64),
                        input_depth,b'\xa5'*(pitch*h)]
                if sparse:arrays.append(b'\xa5'*(pixels+64))
                addresses=[(i+1)*0x100000000 for i in range(len(arrays))]
                args=struct.pack('<17Q5I2fI',*addresses[:12],pitch,dp,mp,pitch,dp,w,h,w,h,0,.25,.25,3)
                if sparse:args+=struct.pack('<Q',addresses[12])
                (directory/'args.bin').write_bytes(args)
                manifest=['kernel '+('d4r_wire_history_sparse' if sparse else 'd4r_wire_history'),
                          'launch 5 5 1 256 1 1 0',f'args {len(args)}']
                for i,data in enumerate(arrays):
                    (directory/f'alloc-{i}.bin').write_bytes(data)
                    manifest += [f'alloc {i} {hex(addresses[i])} {len(data)}',f'pointer {168 if i==12 else i*8} {i} 0']
                (directory/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                name='d4r_wire_history_sparse' if sparse else 'd4r_wire_history'
                subprocess.run([runner,str(kernel),name,str(directory),str(out),'1'],check=True,capture_output=True)
                if sparse:
                    for i in range(len(arrays)):(directory/f'alloc-{i}.bin').write_bytes((out/f'alloc-{i}.bin').read_bytes())
                    manifest[0]='kernel d4r_wire_history_apply'
                    (directory/'manifest.txt').write_text('\n'.join(manifest)+'\n')
                    subprocess.run([runner,str(kernel),'d4r_wire_history_apply',str(directory),str(out),'1'],check=True,capture_output=True)
                result=(out/f'alloc-{0 if sparse else 7}.bin').read_bytes()
                # The .0625 wire corner overlaps a different, thick foreground object.
                self.assertEqual(result[31*pitch+31*8:31*pitch+32*8],rgba)
                self.assertEqual((out/'alloc-8.bin').read_bytes()[31*w+31],255)
                # The neighboring actual wire remains repairable; alpha stays untouched.
                wire=result[32*pitch+32*8:32*pitch+33*8]
                self.assertNotEqual(wire,rgba)
                self.assertEqual(wire[6:],rgba[6:])
                for i in ((1,2,3,4,5,6,10) if sparse else (0,1,2,3,4,5,6,10)):
                    self.assertEqual((out/f'alloc-{i}.bin').read_bytes(),arrays[i])
                for y in range(h):self.assertEqual(result[y*pitch+w*8:(y+1)*pitch],b'\xa5'*16)
                for i,end in ((8,pixels),(9,pixels*4)):
                    self.assertEqual((out/f'alloc-{i}.bin').read_bytes()[end:],b'\xa5'*64)
