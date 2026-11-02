# T1 group: the 8 chained custom_block0_conv2 launches of DL2Net (80x46, 64 channels)
# as one persistent dispatch (_chain.conv2_chain): 120 workgroups = 16x4-pixel blocks x 2 channel halves, weights
# streamed from the (L0-cached) weight buffer.
import _chain
import b0_conv2 as M
NAME = 'dl2_b0c2'
MEMBERS = [(M.HASH, M.ENTRY)] * 8
GRID = (120, 1, 1)


def build_group(get_t0):
    if M.build(get_t0(M.HASH, M.ENTRY)) is None:
        return None
    return _chain.conv2_chain(C=64, BW=16, BH=4, NTW=2, bmode='glob', n_members=len(MEMBERS)), (32, 8, 1)
