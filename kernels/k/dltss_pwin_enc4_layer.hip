// op_imgt / uniform wave index: 2-10% slower here (paired gfx1200 replay), old address form kept
// d4r-build-flags: -DPWIN_NO_IMGT -DPWIN_PLAIN_WV
// NG = 8 (32 waves per window): faster on gfx1200 in paired replay (local k9060xt-opt v3)
#include "pwin_wide.h"
WIDE_ENCODER(dltss_pwin_enc4_layer, 4, 128, 160, 8)
