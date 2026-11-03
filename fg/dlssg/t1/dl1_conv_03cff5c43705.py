# T1: DL1Net k_conv_fp16_nhwc 03cff5c43705: 1x1, 64 -> 32 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '03cff5c4370588b2b93cca3920ac91a8df46969b20faca20a217dd5b53cd5a0a'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    # tile chosen by a gfx11 harness sweep (smaller tiles / 32 output channels per workgroup: more workgroups on
    # this small layer); every output element runs the same chain, output bit-identical
    glsl, meta = _dl1.kconv_g(KS=1, NCH=4, NT=2, NTALL=2, BIAS=4096, order='c',
                              TH=8, R=2, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 2 1 1']
