"""Compile translated surface stores with their real Vulkan image types."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('glslangValidator') and shutil.which('spirv-val'), 'SPIR-V tools required')
class CnnShaderTypeTests(unittest.TestCase):
    def test_mixed_formatted_and_binary_surface_stores_validate(self):
        # A function taking an unformatted image2D loses the concrete format type.
        # Exercise both RG16F and RGBA16F, including the binary half store helper.
        ptx = '''.version 8.0
.target sm_80
.address_size 64
.visible .entry surface_test(.param .align 8 .b8 args[32])
.maxntid 32, 1, 1
{
.reg .b64 %rd<4>;
.reg .b32 %r<4>;
.reg .f32 %f<2>;
ld.param.u64 %rd0, [args+0];
ld.param.u64 %rd1, [args+8];
mov.u32 %r0, 0;
mov.u32 %r1, 15360;
mov.f32 %f0, 0f3F800000;
sust.b.2d.v2.b16.zero [%rd0, {%r0, %r0}], {%r1, %r1};
sust.p.2d.v4.b32.zero [%rd1, {%r0, %r0}], {%f0, %f0, %f0, %f0};
ret;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            d = Path(temporary)
            (d/'surface.ptx').write_text(ptx)
            subprocess.run([sys.executable, str(ROOT/'engine/rr/ptx2rr.py'), str(d/'surface.ptx'), 'surface_test',
                            str(d/'surface.comp'), '--block', '32,1,1', '--weights-arg', '999',
                            '--surface=0=rg16f', '--surface=8=rgba16f'], check=True, capture_output=True)
            for packed in (False, True):
                with self.subTest(packed=packed):
                    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3',
                                    *(['-DD4R_WORD_ARENA', '-DD4R_MMA_PACKED'] if packed else []),
                                    str(d/'surface.comp'), '-o', str(d/'surface.spv')], check=True, capture_output=True)
                    subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(d/'surface.spv')],
                                   check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
