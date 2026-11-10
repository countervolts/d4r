# T1 group: DL1Net decoder step at 320x192 (k_upscale 7f2b + k_element_wise 7e65 + 1x1 conv 8a0b, 64 channels) as one
# dispatch: the conv computes upscale + skip-add in its input staging (see _dec.py).
import _conv, _dec, _dl1
import dl1_conv_8a0bffe1aa9a as C
NAME = 'dl1_dec8'
MEMBERS = [('7f2b2e87b28cbe5e5669c1fba3660dcfc8957693d42a9db8aa4ad54f264cceb5', 'k_upscale'), ('7e65660011e9190a540811a11f1d36522c4fe9570d09e8aa0900afafba08bf56', 'k_element_wise'), (C.HASH, C.ENTRY)]
GRID = (1024, 1, 1)   # about one workgroup per conv tile (960 tiles at 2560x1440); grid-stride beyond


def conv(src, hooks=None):
    # all 64 output channels per workgroup (one z slice): each E value is computed once (gfx11 harness: 0.121 ms with
    # two slices, 0.076 here)
    glsl, meta = _dl1.kconv_g(KS=1, NCH=4, NT=4, NTALL=4, BIAS=8192, order='c',
                              TH=4, R=4, warps=4, act='leaky', hooks=hooks)
    return glsl, meta


def build_group(get_t0):
    return _dec.decoder_group(8, 64, 4, 1, conv)
