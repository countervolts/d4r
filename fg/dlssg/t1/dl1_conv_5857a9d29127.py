# T1: DL1Net k_conv_fp16_nhwc 5857a9d29127: 1x1, 512 -> 256 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '5857a9d291272d3a23e942a459f2e8317539434971bae9f45f1f4f5dbdd7001b'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=32, NT=2, NTALL=16, BIAS=262144, order='c',
                              TH=4, R=1, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 4 1 2']
