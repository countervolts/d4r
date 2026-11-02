# T1: DL1Net k_conv_fp16_nhwc 10f7d864ee4b: 1x1, 256 -> 512 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '10f7d864ee4b29b05aaefe4d9b83b5be4519c728d0441d2019c6bb429dfc69f2'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=16, NT=2, NTALL=32, BIAS=262144, order='c',
                              TH=4, R=1, warps=8, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 1 1 2']
