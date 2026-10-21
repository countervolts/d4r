// Exact single-buffer enc1: reload the original input for the residual, preserve WGP mode.
// d4r-build-flags: -DPOS_VSHARE1 -DPWIN_VEC_SCALAR -DPWIN_PK_ELEM
// RDNA3: two waves normalize one token per lane straight from global memory (POS_DIRECT_NORM: no raw-input LDS
// staging; 237 -> 213 VGPRs, 0.175 -> 0.172 ms at 1440p, byte-identical)
// d4r-build-flags-gfx11: -DPOS_VSHARE1 -DPWIN_VEC_SCALAR -DPWIN_PK_ELEM -DPOS_DIRECT_NORM
#define PWIN_F32ACC
#include "pwin_pos_recompute.h"
POS_RECOMPUTE_ENCODER(dltss_pwin_enc1_layer, 2, 64, 64, 0, 1)
