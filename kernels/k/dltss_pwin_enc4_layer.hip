// NG = 8 (32 waves per window): with the padded LDS rows faster than NG = 4 here
#include "pwin_wide.h"
WIDE_ENCODER(dltss_pwin_enc4_layer, 4, 128, 160, 8)
