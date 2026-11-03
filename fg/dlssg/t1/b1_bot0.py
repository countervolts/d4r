# T1: custom_block1_conv_bot0_hf_kernel (upsample 160x92x32 -> conv at 320x184,
# 32 -> 3 tensors x 16), full replacement, see _conv.py.
import _conv
HASH = 'b27e85d62a69dffa28a3d8b6b2a42c0b0fb398918799c06f18552c2cf72ddf50'
ENTRY = 'custom_block1_conv_bot0_hf_kernel'

def build(src):
    # the PTX clamps only when ctaid.z != 0; this layer is launched with grid z = 1: never
    glsl, meta = _conv.bot0(CB=2, CT=16, NTENS=3, TH=4, NT=3, warps=12, clamp_from=99, persistent=56)
    return glsl, meta
