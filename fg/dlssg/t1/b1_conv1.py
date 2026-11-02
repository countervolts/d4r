# T1: custom_block1_conv1_kernel (320x184, 16 -> 32, pool -> 160x92), see _conv.py.
import _conv
HASH = 'c39431b3222a8d0864a02615fa625f1be886bb84bffd63012a004f634d1ea6f3'
ENTRY = 'custom_block1_conv1_kernel'

def build(src):
    glsl, meta = _conv.conv_pool(CB=1, C_out=32, TH=4, NT=2, persistent=168)
    return glsl, meta
