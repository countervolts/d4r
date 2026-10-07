// fast numerics: intermediate activations f16 (no FP8 re-quantisation), f32 accumulation through GEMM chains
// RX 7700 XT: VGPR cap 160 removes the scratch use; byte-identical to the default build, gfx1101 only
// d4r-build-flags-gfx1101: -DSWIN_MAX_VGPR=160
#define SWIN_NO_Q8
#define SWIN_F32ACC
#ifndef SWIN_MAX_VGPR
#define SWIN_MAX_VGPR 128
#endif
#ifndef SWIN_MLP_R
#define SWIN_MLP_R(NW) 2
#endif
// RDNA4 native FP8 builds use the token-lane block (swin_tok.h); -DSWIN_NO_TOKEN_LANES keeps swin_block.h
#ifndef SWIN_NO_TOKEN_LANES
#define SWIN_TOKEN_LANES
#endif
#include "swin_block.h"
#ifndef SWIN_NW
#define SWIN_NW 8
#endif
SWIN_MODULE_W(rrlite_dec1_4x4, 64, 2, SWIN_NW, 0, false, CommonParams, 96)
