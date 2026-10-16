// CU mode: the waves of a window share one L0 cache (1-6% faster here; slower in the wide layers)
// QKV chains two at a time (K-race port): -3% on top of op_imgt (paired gfx1200 replay, exact)
// d4r-build-flags: -mcumode -DPWIN_QKV_SPLIT
#define PWIN_F32ACC
#include "pwin_layer.h"
PWIN_ENCODER(dltss_pwin_enc2_layer, 2, 64, 96)
