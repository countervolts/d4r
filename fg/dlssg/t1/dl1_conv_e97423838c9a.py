# T1: DL1Net k_conv_fp16_nhwc e97423838c9a: 1x1, 256 -> 128 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = 'e97423838c9a9060411bca04480f0c2cf02d9037be53ee88cc6fd7442d3f1bbb'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=16, NT=2, NTALL=8, BIAS=65536, order='c',
                              TH=4, R=1, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 1 1 2']
