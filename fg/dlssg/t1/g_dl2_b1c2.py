# T1 group: the 8 chained custom_block1_conv2 launches of DL2Net (160x92, 32 channels)
# as one persistent dispatch (_fuse.fuse over the per-launch T1 body): 112 workgroups (two per CU: 32 KB LDS each)
# loop over the 10x23 virtual tiles of each layer, a grid barrier between layers.
import _conv, _fuse
import b1_conv2 as M
NAME = 'dl2_b1c2'
MEMBERS = [(M.HASH, M.ENTRY)] * 8
GRID = (_conv.fit_persistent(112, 1), 1, 1)   # two per CU (scaled to NVXRT_CUS, see _conv)


def build_group(get_t0):
    r = M.build(get_t0(M.HASH, M.ENTRY))
    if r is None:
        return None
    glsl, meta = r
    mem = {'glsl': glsl, 'grid': _fuse.t1_grid(meta)}
    return _fuse.fuse([mem] * len(MEMBERS), _conv.RT), _fuse.t1_block(meta)
