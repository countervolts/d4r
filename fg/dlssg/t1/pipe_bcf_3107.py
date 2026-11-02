# T1: DLSS-G 310.7.0's BlendCandidatesFused (its PTX names every kernel main_kernel; pipe_bcf.py patches 310.9.1's).
# Only the second half of pipe_bcf: the pyramid level's shuffle-down tree reductions as xor exchanges (_shfl), which
# give the same values on the lanes that consume them. Checked bit-exact against the plain translation in the harness
# (gfx11, all frames of a 4x run).
import _shfl
HASH = '99eae6a28d63a446b12d419efaf00d988d0cbaa40978f3a25cf89fb6e01f0862'
ENTRY = 'main_kernel'


def patch(src):
    src, n = _shfl.down_to_xor(src)
    return src if n else None
