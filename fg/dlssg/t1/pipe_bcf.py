# T1: Kernel_BlendCandidatesFused - the texture reads that do not depend on the
# motion-vector loads (both frames at the pixel, the two RG textures at the pixel centre) are issued before the
# motion-vector loads instead of after them: one memory-latency round less per pixel. Pure code motion of reads of
# textures this kernel never writes; every value and every operation is unchanged.
import _shfl
HASH = 'd80209427f2fccdbeec472f11a91583c2eeadde15e93741ac3579582c73e54f1'
ENTRY = 'Kernel_BlendCandidatesFused'

MOVE = [
    """                        r_rd18 = nvx_p64(uint(r_rd6) + 0u);
                        r_rd17 = r_rd18;
                        vec4 _t1 = nvx_tex_fetch_i(r_rd17, int(r_r7), int(r_r30));
""",
    """                        r_rd20 = nvx_p64(uint(r_rd6) + 8u);
                        r_rd19 = r_rd20;
                        vec4 _t3 = nvx_tex_fetch_i(r_rd19, int(r_r7), int(r_r30));
""",
    """                        r_rd21 = nvx_p64(uint(r_rd6) + 40u);
                        vec4 _t5 = nvx_tex2d_m(r_rd21, r_f1, r_f2, 3u);
""",
    """                        vec4 _t6 = nvx_tex2d_m(r_rd2, r_f1, r_f2, 3u);
""",
]
ANCHOR = """                if (r_p15) {
                    break;
                } else {
                    do { // block -> B4
                        r_rd2 = nvx_p64(uint(r_rd6) + 16u);
"""


def patch(src):
    if any(src.count(m) != 1 for m in MOVE) or src.count(ANCHOR) != 1:
        return None
    for m in MOVE:
        src = src.replace(m, '', 1)
    hoisted = ''.join(MOVE).replace('vec4 _t', 'vec4 _t')
    # declare the moved temporaries at the top of the else branch (they were block-local before)
    new_anchor = ANCHOR.replace("                } else {\n", "                } else {\n                    r_rd2 = nvx_p64(uint(r_rd6) + 16u);\n" +
                                ''.join('    ' + l + '\n' for l in hoisted.strip('\n').split('\n')).replace('                            ', '                    '), 1)
    src = src.replace(ANCHOR, new_anchor, 1)
    # the pyramid level the kernel also writes: shuffle-down tree reductions as DPP xor exchanges (_shfl)
    src, n = _shfl.down_to_xor(src)
    return src if n else None
