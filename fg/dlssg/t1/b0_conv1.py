# T1: custom_block0_conv1_kernel (160x92, 32 -> 64, pool -> 80x46), see _conv.py.
import _conv
HASH = '7040b3275c3ae48c103e880539dffcff16447b18eb2513bb8560bc2e3cf73c26'
ENTRY = 'custom_block0_conv1_kernel'

def build(src):
    glsl, meta = _conv.conv_pool(CB=2, C_out=64, TH=4, NT=2, NTALL=4, persistent=168)
    return glsl, meta
