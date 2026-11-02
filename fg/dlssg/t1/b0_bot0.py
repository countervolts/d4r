# T1: custom_block0_conv_bot0_hf_kernel (upsample 80x46x64 -> conv at 160x92,
# 64 -> 3 tensors x 32), full replacement, see _conv.py (replaces the prologue-only patch).
import _conv
HASH = '5f3520a39959a23ca3f3f83387400a24debef6560036a0c772102a79386607d6'
ENTRY = 'custom_block0_conv_bot0_hf_kernel'

def build(src):
    glsl, meta = _conv.bot0(CB=4, CT=32, NTENS=3, TH=4, NT=2, warps=8, clamp_from=1, persistent=54)
    return glsl, meta
