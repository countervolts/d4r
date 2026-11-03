# T1: DL1Net k_conv_fp16_nhwc 2555f8db7ae1: 1x1, 32 -> 64 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '2555f8db7ae19efe0fbd8e03eb619770d14f6c0eec578c62b668cac8082d2004'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=2, NT=2, NTALL=4, BIAS=4096, order='c',
                              TH=16, R=4, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 8 2 1']
