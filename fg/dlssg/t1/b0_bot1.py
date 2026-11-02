# T1: custom_block0_conv_bot1_hf_kernel (3 slices: upsample 160x92x32 -> conv at
# 320x184, 32 -> 8 channels, slice z keeps its 4/1/3), full replacement, see _conv.py.
import _conv
HASH = 'c7a5b3be81767266e5d1c4196908123d578f602d36536e53aa16c7815619d781'
ENTRY = 'custom_block0_conv_bot1_hf_kernel'

def build(src):
    glsl, meta = _conv.bot1(CB=2, TH=8, warps=8, variant='b0')
    return glsl, meta + ['t1_grid 1 2 1 1 1 1']     # NVIDIA 8x8 tiles -> 16x8
