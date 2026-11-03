# T1: DL1Net k_conv_fp16_nhwc 8e067766abaf: 3x3, 128 -> 256 channels,
# chain order kxcky, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '8e067766abaf7e6889cf9c42523c55752cc9bb8d348290537f667e26a6895d15'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    # tile chosen by a gfx11 harness sweep (smaller tiles / 16 output channels per workgroup: more workgroups on
    # this small layer); every output element runs the same chain, output bit-identical
    glsl, meta = _dl1.kconv_g(KS=3, NCH=8, NT=1, NTALL=16, BIAS=589824, order='kxcky',
                              TH=8, R=4, warps=2, act='leaky', hooks=hooks)
    return glsl, meta + ['t1_grid 1 1 1 8 8 1']
