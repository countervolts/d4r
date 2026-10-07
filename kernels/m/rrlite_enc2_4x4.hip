// fast numerics: intermediate activations f16 (no FP8 re-quantisation), f32 accumulation through GEMM chains
// RX 7700 XT: packed FP8 input decode, packed norm1 sums, half-wave exchange via ds_bpermute, MLP loop unrolled 4x;
// byte-identical to the default build, gfx1101 only (not measured on other targets)
// d4r-build-flags-gfx1101: -DSWIN_PACKED_INPUT -DSWIN_PACKED_NORM1 -DSWIN_HALF_BPERMUTE -DSWIN_UNROLL_MLP=4
#define SWIN_NO_Q8
#define SWIN_F32ACC
#ifndef SWIN_MAX_VGPR
#define SWIN_MAX_VGPR 192
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
SWIN_MODULE_W(rrlite_enc2_4x4, 96, 4, SWIN_NW, 128, false, CommonParams, 0)
