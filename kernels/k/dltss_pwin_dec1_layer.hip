// CU mode: the waves of a window share one L0 cache (1-6% faster here; slower in the wide layers)
// QKV chains two at a time (K-race port): -1.7% on top of op_imgt (paired gfx1200 replay, exact)
// d4r-build-flags: -mcumode -DPWIN_QKV_SPLIT
#define PWIN_F32ACC
#include "pwin_layer.h"
PWIN_DECODER(dltss_pwin_dec1_layer, 2, 64, 64)
