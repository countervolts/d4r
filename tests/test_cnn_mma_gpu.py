"""Opt-in Vulkan regression for CNN f16 fragments, including independent concurrent waves.

D4R_RUN_GPU_TESTS=1 D4R_EVK=/path/to/evk python3 -m unittest tests.test_cnn_mma_gpu
Build evk with: g++ -O2 engine/evk.cpp -lvulkan -o build/evk
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS') == '1', 'opt-in GPU test')
class CnnMatrixGpuTests(unittest.TestCase):
    def test_fragment_mapping_accumulation_and_wave_isolation(self):
        self.check_fragments(packed=False)

    def test_packed_fragment_rounding_and_wave_isolation(self):
        self.check_fragments(packed=True)

    def check_fragments(self, packed):
        import numpy as np
        rng = np.random.default_rng(412)
        fragments = np.empty((8, 32, 8), dtype='<u4')
        expected = np.empty((8, 32, 4), dtype='<u4')

        def packed(values):
            return np.ascontiguousarray(values, dtype='<f2').view('<u4')[0]

        for wave in range(8):
            a = rng.normal(0, 0.7, (16, 16)).astype(np.float16)
            b = rng.normal(0, 0.7, (16, 8)).astype(np.float16)
            c = rng.normal(0, 0.7, (16, 8)).astype(np.float16)
            sums = c.astype(np.float32)
            results = []
            for k in range(16):
                if packed:
                    # A half FMA rounds the exact product plus accumulator once to half.
                    sums = (sums.astype(float) + a[:, k:k+1].astype(float) * b[k:k+1, :].astype(float)).astype(np.float16).astype(np.float32)
                else:
                    # A product of two finite binary16 values is exact in binary32.
                    sums = (sums + a[:, k:k+1].astype(np.float32) * b[k:k+1, :].astype(np.float32)).astype(np.float32)
                if k in (7, 15):
                    results.append(sums.astype(np.float16))
            for lane in range(32):
                g, t = lane // 4, lane % 4
                fragments[wave, lane] = [
                    packed(a[g, 2*t:2*t+2]), packed(a[g+8, 2*t:2*t+2]),
                    packed(a[g, 2*t+8:2*t+10]), packed(a[g+8, 2*t+8:2*t+10]),
                    packed(b[2*t:2*t+2, g]), packed(b[2*t+8:2*t+10, g]),
                    packed(c[g, 2*t:2*t+2]), packed(c[g+8, 2*t:2*t+2]),
                ]
                expected[wave, lane, 0:2] = [packed(results[0][g, 2*t:2*t+2]), packed(results[0][g+8, 2*t:2*t+2])]
                expected[wave, lane, 2:4] = [packed(results[1][g, 2*t:2*t+2]), packed(results[1][g+8, 2*t:2*t+2])]

        source = '''#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#include "../round_half.glsl"
#include "mma_f16.glsl"
layout(local_size_x = 128) in;
layout(binding=0,std430) readonly buffer In { uint v[]; } src;
layout(binding=1,std430) writeonly buffer Out { uint v[]; } dst;
void main() {
    uint i=gl_GlobalInvocationID.x, at=i*8u, outAt=i*4u;
    uint d0,d1;
    mma_f16_m16n8k8(src.v[at],src.v[at+1u],src.v[at+4u],src.v[at+6u],src.v[at+7u],d0,d1);
    dst.v[outAt]=d0;dst.v[outAt+1u]=d1;
    mma_f16_m16n8k16(src.v[at],src.v[at+1u],src.v[at+2u],src.v[at+3u],src.v[at+4u],src.v[at+5u],src.v[at+6u],src.v[at+7u],d0,d1);
    dst.v[outAt+2u]=d0;dst.v[outAt+3u]=d1;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            d = Path(temporary)
            shader, spv = d/'matrix.comp', d/'matrix.spv'
            shader.write_text(source)
            fragments.tofile(d/'in.bin')
            subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3',
                            *(['-DD4R_MMA_PACKED'] if packed else []), '-I'+str(ROOT/'engine/cnn_sr'), str(shader), '-o', str(spv)], check=True, capture_output=True)
            subprocess.run([os.environ.get('D4R_EVK', str(ROOT/'build/evk')), str(spv), '2', '1', '1',
                            'in:0='+str(d/'in.bin'), 'out:1='+str(d/'out.bin')+':'+str(expected.nbytes)],
                           check=True, capture_output=True)
            actual = np.fromfile(d/'out.bin', dtype='<u4').reshape(expected.shape)
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
