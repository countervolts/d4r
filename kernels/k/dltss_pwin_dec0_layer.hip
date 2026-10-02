// CU mode: the waves of a window share one L0 cache (1-6% faster here; slower in the wide layers)
// d4r-build-flags: -mcumode
#define PWIN_F32ACC
#ifndef D4R_K_POS_MT
#define D4R_K_POS_MT 2
#elif !defined(__GFX12__) || !defined(D4R_K_FP16_BASELINE)
#error "experimental position token tiling requires gfx12 strict K baseline"
#endif
#include "pwin_pos.h"
POS_DECODER(dltss_pwin_dec0_layer, 2, 32, 64, 40, 48, D4R_K_POS_MT)
