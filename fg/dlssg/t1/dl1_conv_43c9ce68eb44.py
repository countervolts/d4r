# T1: DL1Net k_conv_fp16_nhwc 43c9ce68eb44: 3x3, 128 -> 128 channels,
# chain order ckxky, epilogue leaky (see _dl1.py; parameters read from this instance's PTX).
import _dl1
HASH = '43c9ce68eb449c21e5111acf425976d6cd6d8d26818b602dd3d8cefa99dcdbf2'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    # 16 output channels and 16x8 pixels per workgroup (NT=1, 2 waves): more, smaller workgroups on this small layer
    # (gfx11 harness: 0.092 -> 0.075 ms with 16x16 tiles, 0.067 -> 0.059 with 16x8 after the staging rewrite;
    # bit-identical, every output element runs the same chain)
    glsl, meta = _dl1.kconv_g(KS=3, NCH=8, NT=1, NTALL=8, BIAS=294912, order='ckxky',
                              TH=8, R=4, warps=2, act='leaky', hooks=hooks, PF=1)
    return glsl, meta + ['t1_grid 1 1 1 4 8 1']
