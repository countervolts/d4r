// fast numerics: intermediate activations f16 (no FP8 re-quantisation), f32 accumulation through GEMM chains
#define SWIN_NO_Q8
#define SWIN_F32ACC
#ifndef SWIN_MAX_VGPR
#define SWIN_MAX_VGPR 192
#endif
#ifndef SWIN_MLP_R
#define SWIN_MLP_R(NW) 1
#endif
#define SWIN_SCALAR_BLOAD // scalar-base weight loads: faster here (no spills), slower in the 8-wave layers
// RDNA4 native FP8 builds use the token-lane block (swin_tok.h); -DSWIN_NO_TOKEN_LANES keeps swin_block.h
#ifndef SWIN_NO_TOKEN_LANES
#define SWIN_TOKEN_LANES
#endif
#include "swin_block.h"
#ifndef SWIN_NW
#define SWIN_NW 4
#endif
SWIN_MODULE_W(rrlite_enc1_4x4, 64, 2, SWIN_NW, 96, false, CommonParams, 0)
