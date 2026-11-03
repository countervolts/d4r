# T1 group: the two DL1Net decoder steps that share k_upscale 7916 + k_element_wise 1d6d (128 channels), followed by
# 1x1 conv 23aa (40x24) or a9c1 (80x48): one dispatch each; the conv computes upscale + skip-add in its input staging
# (see _dec.py). 23aa writes over its skip input, so the conv keeps all 128 output channels in one workgroup (one z
# slice, _dec.nvx_inplace_ok) and both instances take the fused path.
import _conv, _dec, _dl1
import dl1_conv_23aab3e63f3f as C0, dl1_conv_a9c155ea7d64 as C1
NAME = 'dl1_dec16'
MEMBERS = [('79161b8d55565b623e070a1fbb34df1215672ef90784220566554a6118dacde1', 'k_upscale'), ('1d6dd912cd631de2f88c678c792eaa49f39de540d1d367d1c1d73c9f502148c2', 'k_element_wise'),
           [(C0.HASH, C0.ENTRY), (C1.HASH, C1.ENTRY)]]
GRID = (256, 1, 1)   # about one workgroup per conv tile (240 tiles for a9c1 at 160x96); grid-stride beyond


def conv(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=8, NT=8, NTALL=8, BIAS=32768, order='c',
                              TH=4, R=4, warps=8, act='leaky', hooks=hooks)
    return glsl, meta


def build_group(get_t0):
    return _dec.decoder_group(16, 128, 4, 1, conv)
