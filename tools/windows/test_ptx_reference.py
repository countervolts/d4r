"""Regression for divergent finite loops in the public CPU PTX reference."""
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'kernels/tools'))
import ptxsim


class ControlFlow(unittest.TestCase):
    def test_divergent_backward_loop_reconverges(self):
        source = '''.visible .entry pattern(.param .align 8 .b8 pattern_param_0[8])
        .maxntid 32, 1, 1
        {
        .reg .b32 %r<4>;
        .reg .pred %p<1>;
        mov.u32 %r0, %tid.x;
        and.b32 %r0, %r0, 3;
        add.u32 %r0, %r0, 1;
        mov.u32 %r1, 0;
        $loop:
        add.u32 %r1, %r1, %r0;
        sub.u32 %r0, %r0, 1;
        setp.ne.u32 %p0, %r0, 0;
        @%p0 bra $loop;
        add.u32 %r2, %r1, 7;
        ret;
        }
        '''
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'pattern.ptx'
            path.write_text(source)
            statements, types, shared = ptxsim.parse_kernel(str(path), 'pattern')
            block = ptxsim.Block(statements, types, shared, None, 'pattern_param_0', (32, 1, 1), (1, 1, 1), (0, 0, 0))
            block.run(limit=1000)
        n = np.arange(32) % 4 + 1
        np.testing.assert_array_equal(block.regs['%r1'], n * (n + 1) // 2)
        np.testing.assert_array_equal(block.regs['%r2'], n * (n + 1) // 2 + 7)


if __name__ == '__main__':
    unittest.main()
