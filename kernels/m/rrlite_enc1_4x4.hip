// fast numerics: intermediate activations f16 (no FP8 re-quantisation), f32 accumulation through GEMM chains
#define SWIN_NO_Q8
#define SWIN_F32ACC
#define SWIN_MAX_VGPR 192
#define SWIN_MLP_R(NW) 1
#define SWIN_SCALAR_BLOAD // scalar-base weight loads: faster here (no spills), slower in the 8-wave layers
#include "swin_block.h"
SWIN_MODULE_W(rrlite_enc1_4x4, 64, 2, 4, 96, false, CommonParams, 0)
