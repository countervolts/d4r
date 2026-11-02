# T1: custom_block1_conv0_kernel (640x368, 18 -> 16, pool -> 320x184), see _conv.py.
import _conv
HASH = '8562841e88f60f1194ffb20afe6e1117b0e1d0f1ab741416854534fe07a3577c'
ENTRY = 'custom_block1_conv0_kernel'

def build(src):
    glsl, meta = _conv.conv_pool(CB=2, C_out=16, TH=8, NT=1, wl='oihw18', persistent=112)
    return glsl, meta
