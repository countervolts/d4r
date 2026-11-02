# T1: DL1Net k_conv_fp16_nhwc 23aab3e63f3f: 1x1, 128 -> 128 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '23aab3e63f3fd2610f9c6e0ba51d8a2a4d93b710405243ac84d1e07ebf960a1f'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=8, NT=2, NTALL=8, BIAS=32768, order='c',
                              TH=8, R=2, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 2 1 2']
