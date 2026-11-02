# T1: DL1Net k_conv_fp16_nhwc 07993acc01dd: 1x1, 32 -> 16 channels,
# chain order c, epilogue none (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '07993acc01dd813531ef6976c8ce6b53cf6aa0343cfa59a58efd98626ec6ec92'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=2, NT=1, NTALL=1, BIAS=1024, order='c',
                              TH=16, R=2, warps=8, act='none', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 4 1 1']
