# T1: DL1Net k_conv_fp16_nhwc 43c9ce68eb44: 3x3, 128 -> 128 channels,
# chain order ckxky, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '43c9ce68eb449c21e5111acf425976d6cd6d8d26818b602dd3d8cefa99dcdbf2'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=3, NCH=8, NT=2, NTALL=8, BIAS=294912, order='ckxky',
                              TH=16, R=4, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 8 4 1']
