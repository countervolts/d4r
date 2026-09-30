// CU mode: the waves of a window share one L0 cache (1-6% faster here; slower in the wide layers)
// d4r-build-flags: -mcumode
#define PWIN_F32ACC
#include "pwin_layer.h"
PWIN_DECODER(dltss_pwin_dec1_layer, 2, 64, 64)
