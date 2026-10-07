// fast numerics: intermediate activations f16 (no FP8 re-quantisation), f32 accumulation through GEMM chains
#define SWIN_NO_Q8
#define SWIN_F32ACC
#ifndef SWIN_MAX_VGPR
#define SWIN_MAX_VGPR 192
#endif
#ifndef SWIN_MLP_R
#define SWIN_MLP_R(NW) 2
#endif
// NGX can recycle a weights address for another tube block after feature
// recreation. Address-only caching then serves stale packed weights, so
// d4r_prep_key_at stays 0 (ZLUDA prepares every launch). MicroCUDA instead keys
// preparation by allocation identity (d4r_prep_identity_at): one image per tube
// weight set (6 per evaluation), re-prepared on any free/reallocation or write.
#define SWIN_PREP_KEY_AT 0
#define SWIN_PREP_IDENTITY_AT 1
#define SWIN_PREP_SLOTS 8
// RDNA4 native FP8 builds use the token-lane block (swin_tok.h); -DSWIN_NO_TOKEN_LANES keeps swin_block.h
#ifndef SWIN_NO_TOKEN_LANES
#define SWIN_TOKEN_LANES
#endif
#include "swin_block.h"
#ifndef SWIN_NW
#define SWIN_NW 8
#endif
SWIN_MODULE_W(rrlite_enc3_tube_4x4, 128, 4, SWIN_NW, 0, true, TubeParams, 0)
