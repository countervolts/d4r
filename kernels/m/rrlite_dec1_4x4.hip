// fast numerics: intermediate activations f16 (no FP8 re-quantisation), f32 accumulation through GEMM chains
#define SWIN_NO_Q8
#define SWIN_F32ACC
#define SWIN_MAX_VGPR 128
#define SWIN_MLP_R(NW) 2
#include "swin_block.h"
SWIN_MODULE_W(rrlite_dec1_4x4, 64, 2, 8, 0, false, CommonParams, 96)
