# T1 helper: shfl.down by 1/2/4/8/16 as an xor exchange
# (subgroupShuffleXor: ACO emits a DPP lane swap fused into the consuming VALU op, or v_permlanex16, instead of a
# ds_bpermute through LDS with a wait). For a lane L that is a multiple of 2*d the xor partner L ^ d equals the
# shfl.down partner L + d, so tree reductions give identical results on every lane whose value is consumed at the
# end of the tree (the block leaders). Only valid for kernels where that holds; each use is checked end to end.
import re

HELPER = '''
uint nvx_sdx(uint a, uint b, uint c, out bool p)
{
    uint lane = gl_SubgroupInvocationID;
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    p = lane + b <= maxl;
    return subgroupShuffleXor(a, b);
}
'''

PAT = re.compile(r'nvx_shfl_down\(([^,()]+), (1|2|4|8|16)u, ([0-9]+)u, ([^()]+)\)')


def down_to_xor(src):
    n = len(PAT.findall(src))
    if n == 0:
        return src, 0
    src = PAT.sub(r'nvx_sdx(\1, \2u, \3u, \4)', src)
    anchor = 'uint nvx_shfl_down(uint a, uint b, uint c, out bool p)'
    assert anchor in src
    src = src.replace(anchor, HELPER.strip() + '\n' + anchor, 1)
    return src, n
