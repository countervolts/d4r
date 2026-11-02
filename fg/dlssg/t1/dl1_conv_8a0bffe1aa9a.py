# T1: DL1Net k_conv_fp16_nhwc 8a0bffe1aa9a: 1x1, 64 -> 64 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '8a0bffe1aa9ae62080db8158407bc49d0ab65b6d3185ac0dcba9f596759f3cfb'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=4, NT=2, NTALL=4, BIAS=8192, order='c',
                              TH=16, R=4, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 8 2 1']
