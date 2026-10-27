// the WMMA half exchange through ds_bpermute instead of v_permlanex16: -1.5% here
// d4r-build-flags: -DWM_HALF_BPERMUTE
// gfx11: the compiler raises the wave priority while a wave issues its first loads (amdgpu-set-wave-priority;
// otherwise the same code, byte-identical output): -0.7 / -2.2 us
// d4r-build-flags-gfx11: -DWM_HALF_BPERMUTE -mllvm -amdgpu-set-wave-priority
#include "pwin_wide.h"
WIDE_DECODER(dltss_pwin_dec3_layer, 4, 96, 128, 2)
