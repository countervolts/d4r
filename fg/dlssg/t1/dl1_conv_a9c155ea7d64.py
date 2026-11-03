# T1: DL1Net k_conv_fp16_nhwc a9c155ea7d64: 1x1, 128 -> 128 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = 'a9c155ea7d64f36caa78000c7b867d3a1eebcee7e07b1982439b54fcd0cd6e5a'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    # tile chosen by a gfx11 harness sweep (smaller tiles / 32 output channels per workgroup: more workgroups on
    # this small layer); every output element runs the same chain, output bit-identical
    glsl, meta = _dl1.kconv_g(KS=1, NCH=8, NT=2, NTALL=8, BIAS=32768, order='c',
                              TH=8, R=4, warps=4, act='leaky', hooks=hooks, PF=1)
    return glsl, meta + ['t1_grid 1 1 1 8 4 1']
