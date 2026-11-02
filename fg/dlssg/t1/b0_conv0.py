# T1: custom_block0_conv0_c8_kernel (320x184, 32 -> 32, pool -> 160x92), see _conv.py.
import _conv
HASH = 'b79f8d20736b37b5e69ef1530d7660737c36b738f066e7cb7a398984f6e95ebe'
ENTRY = 'custom_block0_conv0_c8_kernel'

def build(src):
    glsl, meta = _conv.conv_pool(CB=2, C_out=32, TH=4, NT=2, persistent=168)
    return glsl, meta
