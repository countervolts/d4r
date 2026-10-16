// CU mode: the waves of a window share one L0 cache (1-6% faster here; slower in the wide layers)
// d4r-build-flags: -mcumode
#define PWIN_F32ACC
#ifdef D4R_K_ENC1_POS_MT
#if !defined(__GFX12__) || !defined(D4R_K_FP16_BASELINE)
#error "experimental enc1 token tiling requires gfx12 strict K baseline"
#endif
#if D4R_K_ENC1_POS_MT != 1 && D4R_K_ENC1_POS_MT != 2 && D4R_K_ENC1_POS_MT != 4
#error "enc1 token tiles per wave must be 1, 2 or 4"
#endif
#include "pwin_pos.h"
POS_ENCODER(dltss_pwin_enc1_layer, 2, 64, 64, 0, D4R_K_ENC1_POS_MT)
#else
#include "pwin_layer.h"
PWIN_ENCODER_POS(dltss_pwin_enc1_layer, 2, 64, 64, true)
#endif
