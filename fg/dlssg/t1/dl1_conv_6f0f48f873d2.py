# T1: DL1Net k_conv_fp16_nhwc 6f0f48f873d2: 3x3, 64 -> 128 channels,
# chain order ckxky, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '6f0f48f873d2c03e8538d4b66806340e2079ee5da7565e7778dbe001562424d8'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=3, NCH=4, NT=2, NTALL=8, BIAS=147456, order='ckxky',
                              TH=16, R=4, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 16 4 1']
