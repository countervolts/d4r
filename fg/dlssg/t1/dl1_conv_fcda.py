# T1: DL1Net first conv k_conv_fp16_nhwc (fcda7303ddc8): 640x384, 48 -> 32, leaky.
import _dl1
HASH = 'fcda7303ddc8d6e8a7472e1063167a3b800d737474dd557196d37a8c42efdf01'
ENTRY = 'k_conv_fp16_nhwc'

def build(src, hooks=None):
    glsl, meta = _dl1.kconv(NCH=3, NT=2, BIAS=27648, TH=16, R=4, hooks=hooks)
    # NVIDIA: 16x12-pixel tiles (grid 40x32 for 640x384); here 16x16 (grid y * 3/4)
    return glsl, meta + ['t1_grid 1 1 3 4 1 1']
