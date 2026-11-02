# T1: custom_block1_conv_bot1_hf_kernel (3 slices: upsample 320x184x16 -> conv at
# 640x368, 16 -> 8 channels + residual), full replacement, see _conv.py.
import _conv
HASH = 'b5286e1335e4a3c4258fdd247d32a9e78e52ee092e26740f75966f4b321ee20b'
ENTRY = 'custom_block1_conv_bot1_hf_kernel'

def build(src):
    glsl, meta = _conv.bot1(CB=1, TH=8, warps=8, variant='b1')
    return glsl, meta + ['t1_grid 1 1 1 1 1 1']     # same 16x8 tiles as NVIDIA
