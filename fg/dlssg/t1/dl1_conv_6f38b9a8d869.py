# T1: DL1Net k_conv_fp16_nhwc 6f38b9a8d869: 1x1, 256 -> 256 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '6f38b9a8d869929879c8e738981eff2e137a86c44ed2bcaada4e31acf5f10871'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=16, NT=2, NTALL=16, BIAS=131072, order='c',
                              TH=4, R=1, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 1 1 2']
