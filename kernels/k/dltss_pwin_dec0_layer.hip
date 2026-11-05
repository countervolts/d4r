// CU mode: the waves of a window share one L0 cache (1-6% faster here; slower in the wide layers). The WMMA half
// exchange goes through ds_bpermute (VALU-bound layer: -1..-3%)
// f32 -> f16 pairs converted straight into one register (WM_CVT_ASM: no v_pack copies; -1..-3 %)
// norm / residual / embedding ReLU on packed f16 pairs (PWIN_PK_ELEM with the packed g2/b2 table; same ops, -1 %)
// attention -> residual cross-wave barrier (POS_HB_SYNC): same as in enc0, whose comment spells the race out. One
// s_barrier per window; no slowdown measured on gfx1201, and it is what keeps this layer reproducible run to run.
// d4r-build-flags: -mcumode -DWM_HALF_BPERMUTE -DWM_CVT_ASM -DPWIN_PK_ELEM -DPWIN_PACKED_RESIDUAL -DPOS_HB_SYNC
// RDNA3: one token per lane in the norm (POS_NORM_LANE: the same sums, half the norm work; 0.299 -> 0.296 ms at 1440p,
// byte-identical)
// d4r-build-flags-gfx11: -mcumode -DWM_HALF_BPERMUTE -DWM_CVT_ASM -DPWIN_PK_ELEM -DPWIN_PACKED_RESIDUAL -DPOS_NORM_LANE -DPOS_HB_SYNC
// RDNA4: no WM_CVT_ASM (its inline asm needs the compiler's real-true16 mode on gfx12; the plain conversion
// costs one pack per pair and gfx12 operands need no lane exchange anyway)
// d4r-build-flags-gfx12: -mcumode -DWM_HALF_BPERMUTE -DPWIN_PK_ELEM -DPWIN_PACKED_RESIDUAL -DPOS_HB_SYNC
#define PWIN_F32ACC
// the reference folds the residual into the output projection in this layer (accuracy builds follow it)
#ifdef D4R_ACCURACY
#define PWIN_RESIDUAL_IN_WO
#endif
#include "pwin_pos.h"
POS_DECODER(dltss_pwin_dec0_layer, 2, 32, 64, 40, 48, 2)
