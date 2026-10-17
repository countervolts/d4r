// CU mode: the waves of a window share one L0 cache (1-6% faster here; slower in the wide layers)
// op_imgt / uniform wave index: 0.6-6% slower here (paired gfx1200 replay), old address form kept
// d4r-build-flags: -mcumode -DPWIN_NO_IMGT -DPWIN_PLAIN_WV
#define PWIN_F32ACC
#ifndef D4R_K_POS_MT
#define D4R_K_POS_MT 2
#elif (!defined(__GFX12__) && !defined(__GFX11__)) || !defined(D4R_K_FP16_BASELINE)
#error "experimental position token tiling requires gfx11/gfx12 strict K baseline"
#endif
// Reuse V across the window's waves. Paired gfx1201 replays retain every half
// code and reduce enc0 time by about 5%; gfx11 and arithmetic experiments keep
// their previous path. D4R_K_PRIVATE_V supplies the original A/B control.
#if defined(__GFX12__) && defined(D4R_K_FP16_BASELINE) && !defined(D4R_K_PRIVATE_V) && \
    !defined(D4R_K_PACKED_ACC) && !defined(D4R_K_F16_WMMA)
#ifndef POS_VSHARE
#define POS_VSHARE
#endif
#endif
#include "pwin_pos.h"
POS_ENCODER(dltss_pwin_enc0_layer, 2, 32, 64, 16, D4R_K_POS_MT)
