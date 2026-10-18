#pragma once
// Publishes the GPU the game renders on to the rest of the process. The NVAPI
// compatibility DLL, the NGX core's CUDA initialization and any later shim
// start from the environment, so the loader and the shim state the adapter's
// real architecture and LUID before they load them.
#include "gpu_selection.h"
#include <windows.h>
#include <cstdlib>

namespace d4r::diag {
// Updates the OS block (other DLLs, ZLUDA) and the CRT copy that getenv reads.
inline void set_process_variable(const char* name, const std::string& value) {
    SetEnvironmentVariableA(name, value.c_str());
    _putenv_s(name, value.c_str());
}
inline void publish_selected_gpu(const std::string& architecture, const AdapterLuid* luid = nullptr) {
    set_process_variable("D4R_GPU_ARCH", architecture);
    if (luid) set_process_variable("D4R_GPU_LUID", format_luid(*luid));
}
}
