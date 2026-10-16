#include "diagnostic.h"
#if defined(D4R_NVAPI_COMPAT)
#include "hip_api.h"
#include <memory>
#include <mutex>
#include <atomic>

namespace {
// Public NVAPI ABI: NVIDIA/nvapi nvapi.h and nvapi_interface.h.
// AD100 describes the CUDA/NGX model profile, not the physical HIP target.
struct ArchInfo { unsigned version, architecture, implementation, revision; };
struct LogicalGpuData { unsigned version; void* os_adapter_id; unsigned count; void* physical[64]; unsigned reserved[8]; };
static_assert(sizeof(LogicalGpuData) == 568);
std::unique_ptr<d4r::diag::HipApi> hip;
hipDeviceProp_t properties{};
int device_ordinal = -1;
int physical_handle;
int logical_handle;
std::mutex init_mutex;
std::atomic<bool> runtime_ready{false};
bool valid(void* handle) { return runtime_ready.load(std::memory_order_acquire) && handle == &physical_handle; }
int __cdecl initialize()
{
    try {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!runtime_ready.load(std::memory_order_acquire)) {
            const char* root = std::getenv("HIP_PATH");
            if (!root) return -5;
            auto runtime = std::make_unique<d4r::diag::HipApi>(root);
            device_ordinal = runtime->select_architecture(-1, properties);
            hip = std::move(runtime);
            runtime_ready.store(true, std::memory_order_release);
        }
        std::printf("NVAPI_COMPAT initialize physical=%s CUDA_model_profile=AD100\n", properties.gcnArchName);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "NVAPI_COMPAT init failed: %s\n", e.what()); return -1; }
}
int __cdecl enumerate(void** handles, unsigned* count)
{
    if (!handles || !count) return -5;
    if (!runtime_ready.load(std::memory_order_acquire)) return -4;
    handles[0] = &physical_handle; *count = 1;
    std::printf("NVAPI_COMPAT EnumPhysicalGPUs count=1 HIP_ordinal=%d\n", device_ordinal);
    return 0;
}
int __cdecl architecture(void* handle, ArchInfo* info)
{
    if (!valid(handle) || !info) return -5;
    if (info->version != (sizeof(ArchInfo) | 1u << 16) && info->version != (sizeof(ArchInfo) | 2u << 16)) return -9;
    info->architecture = 0x190; info->implementation = 2; info->revision = 0x11;
    std::printf("NVAPI_COMPAT GetArchInfo profile=AD102 physical=%s\n", properties.gcnArchName);
    return 0;
}
int __cdecl adapter_id(void* handle, void* luid)
{
    if (!valid(handle) || !luid) return -5;
    std::memcpy(luid, properties.luid, sizeof(LUID));
    std::printf("NVAPI_COMPAT GetAdapterIdFromPhysicalGpu matched_HIP_LUID=1\n");
    return 0;
}
int __cdecl logical_from_physical(void* handle, void** logical)
{
    if (!valid(handle) || !logical) return -5;
    *logical = &logical_handle;
    std::printf("NVAPI_COMPAT GetLogicalGPUFromPhysicalGPU count=1\n");
    return 0;
}
int __cdecl logical_info(void* logical, LogicalGpuData* info)
{
    std::printf("NVAPI_COMPAT GetLogicalGpuInfo handle_valid=%d info=%p version=0x%08x os_adapter_buffer=%p\n",
        runtime_ready.load(std::memory_order_acquire) && logical == &logical_handle, static_cast<void*>(info), info ? info->version : 0,
        info ? info->os_adapter_id : nullptr);
    if (!runtime_ready.load(std::memory_order_acquire) || logical != &logical_handle || !info || !info->os_adapter_id) return -5;
    if (info->version != (sizeof(LogicalGpuData) | 1u << 16)) return -9;
    std::memcpy(info->os_adapter_id, properties.luid, sizeof(LUID));
    info->count = 1;
    std::memset(info->physical, 0, sizeof(info->physical));
    std::memset(info->reserved, 0, sizeof(info->reserved));
    info->physical[0] = &physical_handle;
    std::printf("NVAPI_COMPAT GetLogicalGpuInfo matched_HIP_LUID=1\n");
    return 0;
}
}
#endif

// Trace build forwards responses unchanged. The compatibility build adds the
// public topology APIs above; other queries still go to upstream ZLUDA NVAPI.
extern "C" __declspec(dllexport) void* __cdecl nvapi_QueryInterface(unsigned id)
{
    try {
#if defined(D4R_NVAPI_COMPAT)
        switch (id) {
        case 0x0150e828: return reinterpret_cast<void*>(&initialize);
        case 0xe5ac921f: return reinterpret_cast<void*>(&enumerate);
        case 0xd8265d24: return reinterpret_cast<void*>(&architecture);
        case 0x0ff07fde: return reinterpret_cast<void*>(&adapter_id);
        case 0xadd604d1: return reinterpret_cast<void*>(&logical_from_physical);
        case 0x842b066e: return reinterpret_cast<void*>(&logical_info);
        }
#endif
        const char* path = std::getenv("D4R_NVAPI_BACKEND");
        if (!path || !*path) throw std::runtime_error("D4R_NVAPI_BACKEND absolute path required");
        static d4r::diag::Library backend(d4r::diag::wide(path));
        using Query = void* (__cdecl*)(unsigned);
        static Query query = backend.symbol<Query>("nvapi_QueryInterface");
        void* function = query(id);
        std::printf("NVAPI_QUERY id=0x%08x supported=%d\n", id, function != nullptr);
        return function;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "NVAPI_TRACE_ERROR id=0x%08x %s\n", id, e.what());
        return nullptr;
    }
}
