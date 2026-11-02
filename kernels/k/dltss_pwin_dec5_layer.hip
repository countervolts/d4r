// dec5 (bottleneck, H 8, C 160) as four phase kernels (pwin_phase.h). Needs the patched
// ZLUDA hook (post kernels). MAXW 64 windows (1440p: 9).
#include "pwin_phase.h"
PHASE_PLAIN(dltss_pwin_dec5_layer, 8, 160, 64, 128, 96, 384, 96)
