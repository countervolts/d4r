# T1: DL1Net k_conv_fp16_nhwc 97d6598f61b7: 1x1, 32 -> 32 channels,
# chain order c, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '97d6598f61b72bd3f03661460d6d1480e7c607a10d1c52bc2279e76afbb849d4'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    # tile chosen by a gfx11 harness sweep (smaller tiles / 32 output channels per workgroup: more workgroups on
    # this small layer); every output element runs the same chain, output bit-identical
    glsl, meta = _dl1.kconv_g(KS=1, NCH=2, NT=2, NTALL=2, BIAS=2048, order='c',
                              TH=8, R=4, warps=4, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 2 1 1']
