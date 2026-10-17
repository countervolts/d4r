// fast numerics: intermediate activations f16 (no FP8 re-quantisation), f32 accumulation through GEMM chains
#define SWIN_NO_Q8
#define SWIN_F32ACC
#define SWIN_MAX_VGPR 192
#define SWIN_MLP_R(NW) 2
#if defined(D4R_WINDOWS_GFX11_LANE_EXCHANGE) && defined(__GFX11__)
#define SWIN_BOUNDED_PV // retain FP64 only when the FP16 rounding interval is ambiguous
#endif
#define SWIN_PREP_SLOTS 8 // one weight image per tube block (six per frame)
#include "swin_block.h"
SWIN_MODULE_W(rrlite_enc3_tube_4x4, 128, 4, 8, 0, true, TubeParams, 0)
