"""fg/dlssg translator: PTX constructs newer DLSS-G builds use (hand-written PTX, no NVIDIA code), the gfx11 WMMA
layout option, and the CPU model of nvx_d2f (cvt.rn.ftz.f32.f64 from a double's bits)."""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fg/dlssg/translator'))
import ptxc  # noqa: E402

HEAD = '.version 8.0\n.target sm_89\n.address_size 64\n'
LOCAL_GENERIC_F64 = HEAD + '''
.visible .entry main_kernel(
    .param .u64 main_kernel_param_0,
    .param .u64 main_kernel_param_1,
    .param .u32 main_kernel_param_2
)
.maxntid 32, 1, 1
{
    .local .align 16 .b8 __local_depot0[16];
    .reg .b64 %SPL;
    .reg .b64 %rd<10>;
    .reg .b32 %r<5>;
    .reg .f32 %f<3>;
    .reg .f64 %fd<2>;
    mov.u64 %SPL, __local_depot0;
    ld.param.u64 %rd1, [main_kernel_param_0];
    ld.param.u64 %rd2, [main_kernel_param_1];
    ld.param.u32 %r1, [main_kernel_param_2];
    add.u64 %rd3, %SPL, 0;
    st.local.v2.u64 [%rd3], {%rd1, %rd2};
    mul.wide.u32 %rd4, %r1, 8;
    add.s64 %rd5, %rd3, %rd4;
    ld.local.u64 %rd6, [%rd5];
    ld.u32 %r2, [%rd6];
    cvta.to.global.u64 %rd7, %rd2;
    ld.global.f64 %fd1, [%rd7+8];
    cvt.rn.ftz.f32.f64 %f1, %fd1;
    st.global.f32 [%rd7], %f1;
    st.global.u32 [%rd7+4], %r2;
    ret;
}
'''


def d2f_model(b):
    """the integer algorithm of rt.glsl's nvx_d2f"""
    hi, lo = b >> 32, b & 0xffffffff
    s, mh, e = hi & 0x80000000, hi & 0xfffff, (hi >> 20) & 0x7ff
    if e == 0x7ff:
        return 0x7fffffff if (mh | lo) else (s | 0x7f800000)
    fe = e - 1023 + 127
    if fe >= 255:
        return s | 0x7f800000
    m, rem = ((mh << 3) | (lo >> 29)) & 0xffffffff, lo & 0x1fffffff
    if fe <= 0:
        return (s | 0x00800000) if (fe == 0 and m == 0x7fffff) else s
    r = (fe << 23) | m
    if rem > 0x10000000 or (rem == 0x10000000 and (m & 1)):
        r += 1
    return s | r


class TranslatorTests(unittest.TestCase):
    def test_local_u64_generic_global_and_f64(self):
        glsl, _ = ptxc.translate(LOCAL_GENERIC_F64, 'main_kernel')
        self.assertIn('nvx_d2f(', glsl)
        self.assertIn('packUint2x32(uvec2(lmem[', glsl)
        self.assertNotIn('double', glsl.split('void main')[1])
        if shutil.which('glslangValidator'):
            with tempfile.TemporaryDirectory() as d:
                comp = Path(d) / 'k.comp'
                comp.write_text(glsl)
                ptxc.compile_glsl(str(comp), str(Path(d) / 'k.spv'))

    def test_unprovable_generic_pointer_is_refused(self):
        ptx = LOCAL_GENERIC_F64.replace('ld.u32 %r2, [%rd6];', 'ld.u32 %r2, [%rd5];')   # the local array's address
        with self.assertRaises(ptxc.Err):
            ptxc.translate(ptx, 'main_kernel')

    def test_f64_arithmetic_is_refused(self):
        ptx = LOCAL_GENERIC_F64.replace('cvt.rn.ftz.f32.f64 %f1, %fd1;', 'add.f64 %fd1, %fd1, %fd1;\n    cvt.rn.ftz.f32.f64 %f1, %fd1;')
        with self.assertRaises(ptxc.Err):
            ptxc.translate(ptx, 'main_kernel')

    def test_d2f_model(self):
        import random
        rnd = random.Random(3)
        fmin = 1.1754943508222875e-38
        xs = [rnd.gauss(0, 1) * 10.0 ** rnd.randint(-45, 39) for _ in range(20000)]
        xs += [fmin * (1 + rnd.uniform(-1e-6, 1e-6)) for _ in range(2000)] + [0.0, -0.0, 3.4028236e38, -1e300]
        for x in xs:
            b = struct.unpack('<Q', struct.pack('<d', x))[0]
            try:
                f = struct.unpack('<f', struct.pack('<f', x))[0]
                ref = struct.unpack('<I', struct.pack('<f', x))[0]
            except OverflowError:
                ref = (0x80000000 if x < 0 else 0) | 0x7f800000
                f = None
            if f is not None and f != 0 and abs(f) < fmin:
                ref &= 0x80000000      # ftz
            self.assertEqual(d2f_model(b), ref, x)

    def test_gfx11_layout_builds_sixteen_k_elements(self):
        # the A/B operand shuffles of an mma in gfx11's layout fill elements 8..15 as well
        saved = os.environ.pop('PTXC_LAYOUT', None)   # the environment would override opts
        try:
            opts = {'layout': 'gfx11', 'no_fuse': True}
            src = HEAD + '''
.visible .entry main_kernel(.param .u64 main_kernel_param_0)
.maxntid 32, 1, 1
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [main_kernel_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd2];
    ld.global.v2.u32 {%r5, %r6}, [%rd2+16];
    ld.global.v2.u32 {%r7, %r8}, [%rd2+32];
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r9, %r10}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r7, %r8};
    st.global.v2.u32 [%rd2+48], {%r9, %r10};
    ret;
}
'''
            glsl, _ = ptxc.translate(src, 'main_kernel', opts)
            self.assertIn('ma_c[15]', glsl)
            self.assertIn('mb_[15]', glsl)
            glsl12, _ = ptxc.translate(src, 'main_kernel', {'no_fuse': True})
            self.assertNotIn('ma_c[15]', glsl12)
        finally:
            if saved is not None:
                os.environ['PTXC_LAYOUT'] = saved


if __name__ == '__main__':
    unittest.main()
