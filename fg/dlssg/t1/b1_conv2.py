# T1: custom_block1_conv2_kernel (160x92, 32 -> 32), full replacement, see _conv.py.
import _conv
HASH = '8a9652e848038fe10da9c72de0cc50adef75fa61e57d7339d4718f2d6dcb40eb'
ENTRY = 'custom_block1_conv2_kernel'
# the T0 translation of exactly this PTX (sanity: the module is the one these semantics were read from)

def build(src):
    glsl, meta = _conv.conv2(C=32, TH=4, NT=2)
    return glsl, meta + ['t1_grid 1 2 1 1 1 1']
