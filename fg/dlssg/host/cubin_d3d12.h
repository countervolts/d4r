// cubin_d3d12: NVIDIA's D3D12 cubin interface (NvAPI_D3D12_*Cubin*, *Cu*, GetCuda*Object) for nvngx_dlssg.dll on a
// native Windows AMD D3D12 driver, where nothing else provides it. See cubin_d3d12.cpp.
#pragma once
#include <string>

namespace cux {
struct Config {
    std::wstring cacheDir;   // the kernel cache (dlssg-setup / d4r-features.py --fg dlssg)
    std::wstring logPath;    // nvxrt.log
    bool texHw = false;      // ENTRY.hw.spv (texture-unit filtering)
    bool accFast = false;    // ENTRY.fa.spv / groups/NAME.fa.spv (fp32 accumulation)
    bool groups = true;      // T1 launch groups (persistent fused kernels)
    bool trace = false;      // every launch in the log
    bool splitBarriers = false; // every ResourceBarrier ends a kernel batch (else only non-shader transitions)
    bool gpuSync = true;     // false: hand-offs between the D3D12 queue and the kernels wait on the CPU
    bool vkClears = true;    // ClearUnorderedAccessView* between kernels run on the kernels' queue (no hand-off)
    bool splitLog = false;   // nvxrt.log: the sequence of kernels and D3D12 calls of a few lists, batch ends marked
    bool splitQueries = false; // timestamps, query resolves to readback memory and PIX markers end a kernel batch
};

// Once, before the first query: reads the config. The executor itself starts on the first cubin call.
void configure(const Config& config);
// The implementation of an NVAPI function DLSS-G asks nvapi_QueryInterface for, or nullptr (not ours).
void* queryInterface(unsigned id);
// The D3D12 device DLSS-G runs on (before its Init): descriptor creation hooks.
void attach(void* device);
// Calls into nvngx_dlssg.dll run inside a scope: descriptor creation in it is recorded (CPU handle -> view), and the
// command list an evaluate records into gets its bound state tracked (see "bound state").
struct FgScope {
    explicit FgScope(void* list = nullptr);
    ~FgScope();
    bool outer;
};
// A one-line state summary for the host log
std::string summary();
} // namespace cux
