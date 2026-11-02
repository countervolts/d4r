# T1: custom_block0_conv2_kernel (80x46, 64 -> 64), full replacement, see _conv.py.
import _conv
HASH = 'ff1be329acea5132488a6a96b32845b687319415b4aece39945c1c357c03d9a8'
ENTRY = 'custom_block0_conv2_kernel'

def build(src):
    glsl, meta = _conv.conv2(C=64, TH=4, NT=2, NTALL=4)
    # NVIDIA: 8x2-pixel tiles, all 64 channels per block; here 16x4 pixels x 32 channels (grid z = 2 halves)
    return glsl, meta + ['t1_grid 1 2 1 2 2 1']
