// WGP mode: with the padded LDS rows (PWIN_PAD) this layer is 2-5% faster in WGP mode than in CU mode
// (CU mode was 1-6% faster before the padding)
// weight images through plain global loads here (buffer loads were 1-3% slower in this layer)
// d4r-build-flags: -DPWIN_NO_BUFFER_IMG -DPWIN_VEC_FROM_WEIGHTS
// RDNA3: distinct per-lane norm ownership; unchanged arithmetic, full producer/consumer barriers.
// d4r-build-flags-gfx11: -DPWIN_NO_BUFFER_IMG -DPWIN_VEC_FROM_WEIGHTS -DPWIN_NORM_OWNER
#define PWIN_F32ACC
#include "pwin_layer.h"
PWIN_ENCODER(dltss_pwin_enc2_layer, 2, 64, 96)
