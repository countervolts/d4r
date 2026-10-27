// gfx11: the compiler raises the wave priority while a wave issues its first loads (amdgpu-set-wave-priority;
// otherwise the same code, byte-identical output): -0.9 / -2.4 us
// d4r-build-flags-gfx11: -mllvm -amdgpu-set-wave-priority
#include "pwin_wide.h"
WIDE_ENCODER(dltss_pwin_enc3_layer, 4, 96, 128, 2)
