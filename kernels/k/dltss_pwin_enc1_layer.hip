// CU mode: the waves of a window share one L0 cache (1-6% faster here; slower in the wide layers)
// d4r-build-flags: -mcumode
#define PWIN_F32ACC
#include "pwin_layer.h"
PWIN_ENCODER_POS(dltss_pwin_enc1_layer, 2, 64, 64, true)
