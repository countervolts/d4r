# T1 group: DL1Net's last decoder step at 640x384 (k_upscale 4d58 + k_element_wise 34c0 + 1x1 conv 97d6, 32 channels)
# as one dispatch: the conv computes upscale + skip-add in its input staging (see _dec.py). The conv writes its output
# over the skip buffer, so it keeps all 32 output channels in one workgroup (NT = NTALL, see _dec.nvx_inplace_ok).
import _conv, _dec, _dl1
import dl1_conv_97d6598f61b7 as C
NAME = 'dl1_dec4'
MEMBERS = [('4d5821c974a133b0fa99ad83cc429049d549c0f9ae49fadaf6d897c06bb35414', 'k_upscale'),
           ('34c059d1d1f3e083fb2d45806e8872985c7b9d2d7c35f0c299fc44efbfb17cd8', 'k_element_wise'),
           (C.HASH, C.ENTRY)]
GRID = (2048, 1, 1)   # about one workgroup per conv tile (1920 tiles at 2560x1440); grid-stride beyond


def conv(src, hooks=None):
    glsl, meta = _dl1.kconv_g(KS=1, NCH=2, NT=2, NTALL=2, BIAS=2048, order='c',
                              TH=8, R=2, warps=8, act='leaky', hooks=hooks)
    return glsl, meta


def build_group(get_t0):
    return _dec.decoder_group(4, 32, 8, 1, conv)
