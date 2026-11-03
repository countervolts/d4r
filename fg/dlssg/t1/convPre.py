# T1: custom_block0_convPre_kernel (640x368, 10 -> 32, pool -> 320x184), see _conv.py.
import _conv
HASH = '0d411707c1e30062aa4b8f07230541de7e4db85453edc134fba1806d9a620d9d'
ENTRY = 'custom_block0_convPre_kernel'

def build(src):
    # persistent: 168 workgroups (3 per CU) stage the weights once and loop over the 16x4 conv tiles
    return _conv.conv_pool(CB=1, C_out=32, TH=4, NT=2, persistent=168)
