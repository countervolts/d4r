# T1: DL1Net k_conv_fp16_nhwc a9c155ea7d64: 1x1, 128 -> 128 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = 'a9c155ea7d64f36caa78000c7b867d3a1eebcee7e07b1982439b54fcd0cd6e5a'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=8, NT=2, NTALL=8, BIAS=32768, order='c',
                              TH=16, R=4, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 16 4 1']
