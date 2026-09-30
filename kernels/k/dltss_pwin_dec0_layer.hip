// CU mode: the waves of a window share one L0 cache (1-6% faster here; slower in the wide layers)
// d4r-build-flags: -mcumode
#define PWIN_F32ACC
#include "pwin_pos.h"
POS_DECODER(dltss_pwin_dec0_layer, 2, 32, 64, 40, 48, 2)
