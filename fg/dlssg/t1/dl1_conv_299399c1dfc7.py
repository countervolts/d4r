# T1: DL1Net k_conv_fp16_nhwc 299399c1dfc7: 1x1, 128 -> 64 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '299399c1dfc7fe5141148eae8e3f63cd11a956b0e0a5214fef4205bd16345eaf'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=8, NT=2, NTALL=4, BIAS=16384, order='c',
                              TH=16, R=4, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 16 1 1']
