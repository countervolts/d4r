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
    bool kernelTiming = false; // nvxrt.log: GPU ms per kernel on the kernels' queue (Vulkan timestamps), every 1000 batches
    bool computeQueue = true;  // the kernels on a compute-only queue family (Windows' compute engine: overlaps the game's
                               // graphics queue as Streamline's async compute queue would; else time-sliced with it)
    bool vkCopies = true;    // CopyResource between kernels runs on the kernels' queue where it is exact (no hand-off)
    unsigned queuePriority = 0; // global priority of the kernels' queue: 0 driver default, 1 low, 2 medium, 3 high, 4 realtime
    bool dxil = false;         // the kernels as D3D12 compute in DLSS-G's own command lists (no Vulkan device)
    std::wstring dxc;          // dxc.exe that compiles them on first use (CACHE/hlsl -> CACHE/dxil)
    std::wstring dxilCompare;  // [Host] DxilCompare: one kernel on Vulkan as usual and on DXIL into a shadow of its output,
                               // the two compared on the GPU and logged (finds where a translation differs in a game)
    unsigned dxilCompareOut = 128; // [Host] DxilCompareOut: byte offset of that kernel's output surface parameter
    std::wstring dxilKernels;  // [Host] DxilKernels: only these kernels (comma-separated entry names) on the DXIL route,
                               // the rest on Vulkan; empty or * = all (for bisecting a kernel that differs in a game)
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
// Before nvngx_dlssg.dll creates a frame generation feature: its buffers start a pointer heap of their own (the DXIL
// route then finds all of one launch's buffers in one heap while the previous feature is still alive)
void newFeature();
// Test captures: keep the state each resource was last transitioned to on a hooked command list (off by default; needs
// the list hooks), so a copy out of a game resource names its real StateBefore instead of a guess.
void trackResourceStates(bool on);
// The tracked D3D12_RESOURCE_STATES of an ID3D12Resource; false when no transition of it was seen.
bool resourceState(void* resource, unsigned* state);
// A one-line state summary for the host log
std::string summary();
} // namespace cux
