// FSR Ray Regeneration as the native Windows Ray Reconstruction backend (experimental). See fsr_rr_backend.h.
// RDNA4 records conversion, AMD denoising and composition on the game's native D3D12 device.
// The compatibility route described below remains the RDNA3 default.
//
// Data path of one evaluation (every hand-off is CPU-synchronized; no vkd3d-proton patch is needed):
//   game queue : guides -> shared packet (D3D12 copies, as the native DLSS backend), signal, CPU waits
//   vkd3d queue: one command list on a private vkd3d-proton device:
//                  packet -> staging images, fsr_rr_prepare (Vulkan, recorded through vkd3d's interop),
//                  AMD ffxDispatch, fsr_rr_compose (Vulkan), composed image -> packet output; CPU waits
//   game queue : packet output -> the game's output texture, signal, CPU waits
// The conversion and composition follow the Linux adapter (reference 34e6b0fe, tools/d4r_fsr_rr_backend.inl).
#include "fsr_rr_backend.h"
#include "../d4r_fsr_rr.h"
#include "../d4r_vkd3d_interop.h"
#include "fsr_rr_passes.h"
#include "fsr_rr_native_passes.h"
#include "fsr_rr_route.h"
#include "d3d12_command_hooks.h"
#include <volk.h>
#include "rr_bridge_replay.h"
#include "rr_input_formats.h"
#include <vulkan/vulkan_win32.h>
#include <dxgi1_4.h>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <tuple>
#include <vector>
#include <string>

namespace d4r::win::rr {
namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
bool log_enabled() { static const bool on = std::getenv("D4R_FSR_RR_LOG") != nullptr; return on; }
LONG WINAPI crash_report(EXCEPTION_POINTERS* info) {
    const auto code = info->ExceptionRecord->ExceptionCode;
    if (code != 0xC0000005 && code != 0xC000001D && code != 0xC0000094 && code != 0xC00000FD) return EXCEPTION_CONTINUE_SEARCH;
    char name[MAX_PATH] = "?"; HMODULE module = nullptr;
    const void* at = info->ExceptionRecord->ExceptionAddress;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<const char*>(at), &module))
        GetModuleFileNameA(module, name, MAX_PATH);
    std::fprintf(stderr, "FSR_RR crash: code=%08lx at %p (%s +0x%llx) info0=%llu info1=%p\n", code, at, name,
        (unsigned long long)(static_cast<const char*>(at) - reinterpret_cast<const char*>(module)),
        (unsigned long long)info->ExceptionRecord->ExceptionInformation[0], (void*)info->ExceptionRecord->ExceptionInformation[1]);
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
// The denoiser finds amdxc64.dll with GetModuleHandle, so in a process whose game device already loaded AMD's real
// amdxc64.dll (it always has: AMD's D3D12 driver does), the name resolves to AMD's driver component, which crashes
// on a vkd3d device. The loaded copy of AMD's DLL gets its GetModuleHandleA/W imports redirected in memory so that
// the name resolves to d4r's shim; every other lookup is forwarded. The DLL on disk is never modified.
HMODULE g_shimModule = nullptr;
HMODULE WINAPI module_handle_a(LPCSTR name) {
    if (name && g_shimModule && !_stricmp(name, "amdxc64.dll")) return g_shimModule;
    return GetModuleHandleA(name);
}
HMODULE WINAPI module_handle_w(LPCWSTR name) {
    if (name && g_shimModule && !_wcsicmp(name, L"amdxc64.dll")) return g_shimModule;
    return GetModuleHandleW(name);
}
bool redirect_amdxc(HMODULE module, HMODULE shim) {
    g_shimModule = shim;
    auto* base = reinterpret_cast<unsigned char*>(module);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!directory.VirtualAddress) return false;
    unsigned patched = 0;
    for (auto* descriptor = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress); descriptor->Name; ++descriptor) {
        if (_stricmp(reinterpret_cast<const char*>(base + descriptor->Name), "KERNEL32.dll") != 0) continue;
        auto* lookup = reinterpret_cast<IMAGE_THUNK_DATA*>(base + (descriptor->OriginalFirstThunk ? descriptor->OriginalFirstThunk : descriptor->FirstThunk));
        auto* address = reinterpret_cast<void**>(base + descriptor->FirstThunk);
        for (; lookup->u1.AddressOfData; ++lookup, ++address) {
            if (lookup->u1.Ordinal & IMAGE_ORDINAL_FLAG) continue;
            const char* function = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + lookup->u1.AddressOfData)->Name;
            void* replacement = !std::strcmp(function, "GetModuleHandleA") ? reinterpret_cast<void*>(&module_handle_a)
                : !std::strcmp(function, "GetModuleHandleW") ? reinterpret_cast<void*>(&module_handle_w) : nullptr;
            if (!replacement) continue;
            DWORD old = 0;
            if (!VirtualProtect(address, sizeof(void*), PAGE_READWRITE, &old)) return false;
            *address = replacement; ++patched;
            VirtualProtect(address, sizeof(void*), old, &old);
        }
    }
    return patched != 0;
}
void trace(const char* what) { if (log_enabled()) { std::fprintf(stderr, "FSR_RR step: %s\n", what); std::fflush(stderr); } }
std::wstring widen(const char* text) {
    if (!text || !*text) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    std::wstring out(size_t(n > 0 ? n - 1 : 0), L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, text, -1, out.data(), n);
    return out;
}
std::wstring env_wide(const char* name) { return widen(std::getenv(name)); }
std::string narrow(const std::wstring& text) {
    std::string out(size_t(WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr)), '\0');
    if (!out.empty()) { WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, out.data(), int(out.size()), nullptr, nullptr); out.pop_back(); }
    return out;
}
// The vkd3d-proton core interface (include/vkd3d_core_interface.idl). The IDL declares no base interface, so the
// vtable starts at CreateDevice: slot 0, not 3.
const GUID CLSID_Core = {0xed53efad, 0xda21, 0x4d96, {0xa1, 0xbc, 0xe7, 0x34, 0xe0, 0x78, 0x87, 0x9c}};
const GUID IID_CoreInterface = {0x18da885c, 0xa7b5, 0x464e, {0xa1, 0x21, 0xcc, 0xb7, 0x5d, 0x4d, 0xfc, 0x31}};
struct CoreInterface {
    struct Vtbl {
        HRESULT(STDMETHODCALLTYPE* createDevice)(CoreInterface*, IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    };
    const Vtbl* vtbl;
};
// Row-major, row-vector 4x4 helpers (the NGX/FFX convention), in double. From the Linux adapter.
bool invert(const float (&m)[16], double (&out)[16]) {
    double a[16]; for (int i = 0; i < 16; ++i) a[i] = m[i];
    double inv[16];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
    const double det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if (!std::isfinite(det) || std::fabs(det) < 1e-30) return false;
    for (int i = 0; i < 16; ++i) { out[i] = inv[i] / det; if (!std::isfinite(out[i])) return false; }
    return true;
}
void multiply(const double (&a)[16], const double (&b)[16], double (&out)[16]) {
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
        double sum = 0; for (int k = 0; k < 4; ++k) sum += a[r * 4 + k] * b[k * 4 + c]; out[r * 4 + c] = sum;
    }
}
size_t resource_index(size_t plane) { return plane < 7 ? plane : plane + 2; }
// Same sampled-input formats as the native backend (rr_input_formats.h).
rr_bridge::Plane input_format(DXGI_FORMAT format, size_t plane) { return guide_format(format, plane, "FSR RR"); }
DXGI_FORMAT copy_format(ID3D12Device* device, ID3D12Resource* texture, size_t plane) {
    const auto desc = resource_desc(texture);
    if (plane != Snapshot::Depth) return desc.Format;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT rows = 0; UINT64 rowBytes = 0, bytes = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &bytes);
    const auto format = input_format(desc.Format, plane);
    if (!bytes || bytes == UINT64_MAX || rows != desc.Height || rowBytes != desc.Width * format.bytes ||
        footprint.Footprint.Width != desc.Width || footprint.Footprint.Height != desc.Height)
        throw std::runtime_error("Unsupported FSR RR depth-plane footprint");
    return footprint.Footprint.Format;
}
// The Vulkan entry points used on vkd3d-proton's VkDevice. Loaded from the loader by instance/device, never through
// volk's globals, which belong to the native backend's own devices.
struct VkApi {
    PFN_vkGetDeviceProcAddr deviceProc = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties memoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties formatProperties = nullptr;
    PFN_vkCreateBuffer createBuffer = nullptr; PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements bufferRequirements = nullptr;
    PFN_vkAllocateMemory allocate = nullptr; PFN_vkFreeMemory free = nullptr;
    PFN_vkBindBufferMemory bindBuffer = nullptr;
    PFN_vkCreateImage createImage = nullptr; PFN_vkDestroyImage destroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements imageRequirements = nullptr; PFN_vkBindImageMemory bindImage = nullptr;
    PFN_vkCmdCopyBufferToImage copyBufferToImage = nullptr; PFN_vkCmdCopyImageToBuffer copyImageToBuffer = nullptr;
    PFN_vkCmdBlitImage blitImage = nullptr;
    PFN_vkCmdPipelineBarrier barrier = nullptr;
    PFN_vkGetMemoryWin32HandlePropertiesKHR handleProperties = nullptr;
    PFN_vkDeviceWaitIdle waitIdle = nullptr;
};
struct StagingImage {
    VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
};
struct Intermediate {
    ComPtr<ID3D12Resource> resource; D3D12_RESOURCE_DESC desc{};
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    VkImageView view = VK_NULL_HANDLE; VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL; VkImage image = VK_NULL_HANDLE;
};
enum Tex { LinearDepth, MotionT, NormalsT, DiffuseAlbedo, SpecularAlbedo, DiffuseT, SpecularT, DiffuseOut, SpecularOut, Residual, Composed, Widened, TexCount };
std::mutex hostMutex;
}

FsrConfig FsrConfig::from_environment() {
    FsrConfig config;
    config.vkd3dCore = env_wide("D4R_FSR_VKD3D_CORE");
    config.denoiserDll = env_wide("D4R_FSR_RR_DLL");
    config.amdxcShim = env_wide("D4R_FSR_AMDXC_SHIM");
    config.cacheDir = env_wide("D4R_FSR_CACHE_DIR");
    return config;
}
bool FsrConfig::use_native_device() {
    return native_fsr_route(D4R_TARGET_ARCH, std::getenv("D4R_FSR_RR_DEVICE"));
}

struct FsrBackend::Impl {
    bool nativeD3d12 = FsrConfig::use_native_device();
    NativeFsrPasses nativePasses;
    std::array<NativeFsrPasses::Guide, 11> nativeGuides{};
    Creation creation;
    rr_bridge::Layout layout;
    ComPtr<ID3D12Device> d3d;           // the game's device
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Resource> shared;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12CommandAllocator> producerAllocator, consumerAllocator;
    ComPtr<ID3D12GraphicsCommandList> producer, consumer;
    Handle event;
    uint64_t sequence = 0;
    bool pending = false, failed = false;
    std::unique_ptr<Snapshot> retained;
    // --- private vkd3d-proton side ---
    HMODULE core = nullptr;
    ComPtr<ID3D12Device> vdev;
    ComPtr<ID3D12CommandQueue> vqueue;
    ComPtr<ID3D12CommandAllocator> vallocator;
    ComPtr<ID3D12GraphicsCommandList> vlist;
    ComPtr<ID3D12Fence> vfence; Handle vevent; uint64_t vsequence = 0;
    ComPtr<ID3D12DXVKInteropDevice1> interop;
    VkInstance vkInstance = VK_NULL_HANDLE; VkPhysicalDevice vkPhysical = VK_NULL_HANDLE; VkDevice vk = VK_NULL_HANDLE;
    VkQueue vkQueue = VK_NULL_HANDLE; uint32_t vkFamily = 0;
    VkApi api;
    D4rFsrRrPasses passes;
    FsrRrRuntime runtime;
    ffxContext context = nullptr; uint32_t signals = 0;
    VkBuffer packetBuffer = VK_NULL_HANDLE; VkDeviceMemory packetMemory = VK_NULL_HANDLE;
    std::array<StagingImage, 11> staging{};
    std::array<Intermediate, TexCount> textures{};
    float previousView[16] = {}; bool havePrevious = false;
    uint32_t frameIndex = 0;
    FsrConfig config;
    HMODULE shimModule = nullptr;
    FsrBackend::SuperSampling superSampling;
    std::array<unsigned, 4> srRenderLimits{}; // min width/height, max width/height from the SR library
    // D4R_FSR_RR_LOG: GPU timestamps of the vkd3d list (begin, after prepare, after denoise, end)
    ComPtr<ID3D12QueryHeap> stamps; ComPtr<ID3D12Resource> stampReadback; UINT64 stampFrequency = 0;
    void stamp(UINT index) { if (stamps) vlist->EndQuery(stamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index); }
    ComPtr<ID3D12Resource> denoised;  // game-device copy of the denoised colour, the Super Sampling stage's input
    D3D12_RESOURCE_STATES denoisedState = D3D12_RESOURCE_STATE_COMMON;

    Impl(ID3D12Device* device, const Creation& c, const FsrConfig& cfg)
        : creation(c), layout(c.width, c.height, c.outWidth, c.outHeight), d3d(device), config(cfg) {
        // Reserve each guide's largest texel size once, as the native backend does, so format changes keep the packet.
        VkDeviceSize offset = 0;
        for (auto& plane : layout.planes) {
            plane.offset = (offset + 511) & ~VkDeviceSize(511);
            plane.pitch = (VkDeviceSize(layout.rw) * 16 + 255) & ~VkDeviceSize(255);
            offset = plane.offset + plane.pitch * layout.rh;
        }
        layout.outputOffset = (offset + 511) & ~VkDeviceSize(511);
        layout.outputPitch = (VkDeviceSize(layout.ow) * 16 + 255) & ~VkDeviceSize(255);
        layout.bytes = layout.outputOffset + layout.outputPitch * layout.oh;
    }
    ~Impl() {
        if (vk && api.waitIdle) (void)api.waitIdle(vk);
        if (vqueue && vfence) {
            const uint64_t value = ++vsequence;
            if (SUCCEEDED(vqueue->Signal(vfence.Get(), value)) && SUCCEEDED(vfence->SetEventOnCompletion(value, vevent.value)))
                WaitForSingleObject(vevent.value, 10000);
        }
        if (context) runtime.destroyContext(&context, nullptr);
        if (vk) {
            for (auto& s : staging) destroy_staging(s);
            for (auto& t : textures) if (t.view) passes.destroyView(vk, t.view, nullptr);
            passes.destroy();
            if (packetBuffer) api.destroyBuffer(vk, packetBuffer, nullptr);
            if (packetMemory) api.free(vk, packetMemory, nullptr);
        }
        // vkd3d-proton's DLL stays loaded for the process: its device objects are released by their ComPtrs.
    }
    void destroy_staging(StagingImage& s) {
        if (s.view) passes.destroyView(vk, s.view, nullptr);
        if (s.image) api.destroyImage(vk, s.image, nullptr);
        if (s.memory) api.free(vk, s.memory, nullptr);
        s = StagingImage{};
    }
    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
        VkPhysicalDeviceMemoryProperties properties{}; api.memoryProperties(vkPhysical, &properties);
        for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & flags) == flags) return i;
        throw std::runtime_error("No compatible FSR RR Vulkan memory type");
    }
    void check(VkResult r, const char* what) {
        if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " VkResult=" + std::to_string(r));
    }

    void initialize() {
        if (!d3d) throw std::runtime_error("Null FSR RR D3D12 device");
        if ((creation.flags & 3u) != 3u) throw std::runtime_error("FSR RR requires HDR and render-resolution motion vectors");
        if (creation.flags & 4u) throw std::runtime_error("FSR RR does not convert jittered motion vectors");
        if (creation.quality < 0 || creation.quality > 5) throw std::runtime_error("Unsupported RR quality mode");
        if (nativeD3d12) {
            // AMD sees the game's real D3D12 device and its real driver extension.
            // No vkd3d core, amdxc shim, external packet or Vulkan device is loaded.
            std::fprintf(stderr, "D4R_FSR_RR_DEVICE native-d3d12 target=%s cross_api_copies=0\n", D4R_TARGET_ARCH);
            vdev = d3d;
            vevent.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!vevent.value) throw std::runtime_error("FSR RR native event creation failed");
            dx(vdev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&vfence)), "FSR RR native fence");
            dx(vdev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&vallocator)), "FSR RR native allocator");
            dx(vdev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, vallocator.Get(), nullptr, IID_PPV_ARGS(&vlist)), "FSR RR native list");
            dx(vlist->Close(), "FSR RR native initial close");
            if (!runtime.load(config.denoiserDll)) throw std::runtime_error(std::string("FSR RR denoiser DLL: ") + fsr_rr_stage_name(runtime.stage));
            if (!runtime.query_providers(vdev.Get())) throw std::runtime_error(std::string("FSR RR native D3D12 provider: ") + fsr_rr_stage_name(runtime.stage));
            nativePasses.initialize(vdev.Get());
            create_intermediates();
            create_context();
            std::fprintf(stderr, "FSR_RR provider \"%s\" on native D3D12\n", runtime.providerName.c_str());
            return;
        }
        std::fprintf(stderr, "D4R_FSR_RR_DEVICE vkd3d target=%s\n", D4R_TARGET_ARCH);
        if (config.vkd3dCore.empty() || config.denoiserDll.empty() || config.amdxcShim.empty())
            throw std::runtime_error("FSR RR needs D4R_FSR_VKD3D_CORE, D4R_FSR_RR_DLL and D4R_FSR_AMDXC_SHIM");
        // --- the game-device packet, fence and copy lists (as the native backend) ---
        shared = make_buffer(d3d.Get(), layout.bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_STATE_COMMON);
        dx(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "FSR RR fence");
        event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        vevent.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event.value || !vevent.value) throw std::runtime_error("FSR RR event creation failed");
        auto createList = [&](ID3D12Device* device, auto& allocator, auto& list) {
            dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "FSR RR allocator");
            dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "FSR RR command list");
            dx(list->Close(), "FSR RR initial close");
        };
        createList(d3d.Get(), producerAllocator, producer); createList(d3d.Get(), consumerAllocator, consumer);
        Handle memoryHandle;
        dx(d3d->CreateSharedHandle(shared.Get(), nullptr, GENERIC_ALL, nullptr, &memoryHandle.value), "FSR RR packet handle");
        if (log_enabled()) AddVectoredExceptionHandler(1, crash_report);
        trace("before start_vkd3d");
        start_vkd3d();
        trace("before import_packet");
        import_packet(memoryHandle.value);
        trace("before vkd3d list");
        createList(vdev.Get(), vallocator, vlist);
        trace("before create_intermediates");
        create_intermediates();
        trace("before create_context");
        create_context();
    }

    void start_vkd3d() {
        // vkd3d-proton's own environment: the dxil-spirv RDNA3 WMMA/FP8 emulation, and a cache for its shaders.
        // gfx12 uses the native FP8 matrix layout. Applying the gfx11 layout rewrite
        // there also forces FP16 RTZ in dxil-spirv. Preserve explicit diagnostic overrides.
        // Read the Windows environment: the C runtime's getenv snapshot does not
        // reflect SetEnvironmentVariable calls made by other DLLs in this process.
        DWORD configLength = GetEnvironmentVariableA("DXIL_SPIRV_CONFIG", nullptr, 0);
        if (!configLength && GetLastError() == ERROR_ENVVAR_NOT_FOUND &&
            std::string(D4R_TARGET_ARCH).rfind("gfx11", 0) == 0) {
            if (!SetEnvironmentVariableA("DXIL_SPIRV_CONFIG", "wmma_rdna3_workaround"))
                throw std::runtime_error("FSR RR shader compiler configuration failed");
            configLength = GetEnvironmentVariableA("DXIL_SPIRV_CONFIG", nullptr, 0);
        }
        if (log_enabled()) {
            std::vector<char> compilerConfig(configLength ? configLength : 1, '\0');
            GetEnvironmentVariableA("DXIL_SPIRV_CONFIG", compilerConfig.data(), DWORD(compilerConfig.size()));
            std::fprintf(stderr, "FSR_RR compiler: target=%s DXIL_SPIRV_CONFIG=%s\n",
                D4R_TARGET_ARCH, compilerConfig[0] ? compilerConfig.data() : "<default>");
        }
        if (!config.cacheDir.empty() && !std::getenv("VKD3D_SHADER_CACHE_PATH")) {
            CreateDirectoryW(config.cacheDir.c_str(), nullptr);
            SetEnvironmentVariableW(L"VKD3D_SHADER_CACHE_PATH", config.cacheDir.c_str());
        }
        // The denoiser consults amdxc64.dll only when it is already loaded: ours, never AMD's.
        shimModule = LoadLibraryW(config.amdxcShim.c_str());
        if (!shimModule) throw std::runtime_error("FSR RR amdxc64 shim could not be loaded");
        core = LoadLibraryExW(config.vkd3dCore.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!core) throw std::runtime_error("FSR RR vkd3d-proton core could not be loaded");
        using GetInterface = HRESULT(WINAPI*)(REFCLSID, REFIID, void**);
        auto getInterface = reinterpret_cast<GetInterface>(reinterpret_cast<void*>(GetProcAddress(core, "D3D12GetInterface")));
        CoreInterface* iface = nullptr;
        if (!getInterface || FAILED(getInterface(CLSID_Core, IID_CoreInterface, reinterpret_cast<void**>(&iface))) || !iface)
            throw std::runtime_error("FSR RR vkd3d-proton core interface unavailable");
        ComPtr<IDXGIFactory4> factory; dx(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "FSR RR DXGI factory");
        ComPtr<IDXGIAdapter1> adapter; dx(factory->EnumAdapterByLuid(adapter_luid(d3d.Get()), IID_PPV_ARGS(&adapter)), "FSR RR adapter");
        dx(iface->vtbl->createDevice(iface, adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&vdev)), "FSR RR vkd3d device");
        trace("vkd3d device created");
        D3D12_COMMAND_QUEUE_DESC queueDesc{}; queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        dx(vdev->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&vqueue)), "FSR RR vkd3d queue");
        dx(vdev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&vfence)), "FSR RR vkd3d fence");
        if (log_enabled()) {
            D3D12_QUERY_HEAP_DESC heap{}; heap.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; heap.Count = 4;
            if (SUCCEEDED(vdev->CreateQueryHeap(&heap, IID_PPV_ARGS(&stamps))) && SUCCEEDED(vqueue->GetTimestampFrequency(&stampFrequency)) && stampFrequency)
                stampReadback = make_buffer(vdev.Get(), 4 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            else stamps.Reset();
        }
        trace("vkd3d queue and fence created");
        dx(vdev->QueryInterface(IID_PPV_ARGS(&interop)), "FSR RR vkd3d Vulkan interop");
        dx(interop->GetVulkanHandles(&vkInstance, &vkPhysical, &vk), "FSR RR Vulkan handles");
        dx(interop->GetVulkanQueueInfo(vqueue.Get(), &vkQueue, &vkFamily), "FSR RR Vulkan queue info");
        trace("interop and handles ok");
        HMODULE vulkan = GetModuleHandleA("vulkan-1.dll"); if (!vulkan) vulkan = LoadLibraryA("vulkan-1.dll");
        auto instanceProc = vulkan ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(vulkan, "vkGetInstanceProcAddr"))) : nullptr;
        if (!instanceProc) throw std::runtime_error("FSR RR Vulkan loader unavailable");
        api.deviceProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(instanceProc(vkInstance, "vkGetDeviceProcAddr"));
        api.memoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(instanceProc(vkInstance, "vkGetPhysicalDeviceMemoryProperties"));
        api.formatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(instanceProc(vkInstance, "vkGetPhysicalDeviceFormatProperties"));
        auto physicalProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(instanceProc(vkInstance, "vkGetPhysicalDeviceProperties"));
        if (!api.deviceProc || !api.memoryProperties || !api.formatProperties || !physicalProperties) throw std::runtime_error("FSR RR Vulkan entry points missing");
#define D4R_VK(member, name) api.member = reinterpret_cast<decltype(api.member)>(api.deviceProc(vk, "vk" #name)); \
        if (!api.member) throw std::runtime_error("FSR RR Vulkan entry point missing: vk" #name)
        D4R_VK(createBuffer, CreateBuffer); D4R_VK(destroyBuffer, DestroyBuffer); D4R_VK(bufferRequirements, GetBufferMemoryRequirements);
        D4R_VK(allocate, AllocateMemory); D4R_VK(free, FreeMemory); D4R_VK(bindBuffer, BindBufferMemory);
        D4R_VK(createImage, CreateImage); D4R_VK(destroyImage, DestroyImage); D4R_VK(imageRequirements, GetImageMemoryRequirements);
        D4R_VK(bindImage, BindImageMemory); D4R_VK(copyBufferToImage, CmdCopyBufferToImage); D4R_VK(copyImageToBuffer, CmdCopyImageToBuffer);
        D4R_VK(blitImage, CmdBlitImage);
        D4R_VK(barrier, CmdPipelineBarrier); D4R_VK(handleProperties, GetMemoryWin32HandlePropertiesKHR); D4R_VK(waitIdle, DeviceWaitIdle);
#undef D4R_VK
        trace("vulkan entry points loaded");
        VkPhysicalDeviceProperties limits{}; physicalProperties(vkPhysical, &limits);
        if (!passes.init(vk, api.deviceProc, limits.limits.maxPushConstantsSize))
            throw std::runtime_error("FSR RR conversion passes could not be created");
        trace("passes created");
        if (!runtime.load(config.denoiserDll)) throw std::runtime_error(std::string("FSR RR denoiser DLL: ") + fsr_rr_stage_name(runtime.stage));
        if (!redirect_amdxc(runtime.module, shimModule)) throw std::runtime_error("FSR RR could not redirect the denoiser's amdxc64 lookup");
        trace("amdxc64 lookup redirected to the d4r shim");
        trace("before message callback");
        runtime.set_message_callback([](uint32_t type, const wchar_t* message) {
            std::fprintf(stderr, "FSR RR runtime %s: %ls\n", type == FFX_API_MESSAGE_TYPE_WARNING ? "warning" : "error", message);
        }, FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS);
        trace("denoiser dll loaded");
        trace("before query_providers");
        if (!runtime.query_providers(vdev.Get()))
            throw std::runtime_error(std::string("FSR RR provider: ") + fsr_rr_stage_name(runtime.stage));
        trace("providers queried");
        if (log_enabled()) std::fprintf(stderr, "FSR_RR provider \"%s\" on vkd3d-proton\n", runtime.providerName.c_str());
    }

    void import_packet(HANDLE resource) {
        VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external}; bi.size = layout.bytes;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(api.createBuffer(vk, &bi, nullptr, &packetBuffer), "FSR RR packet buffer");
        VkMemoryRequirements req{}; api.bufferRequirements(vk, packetBuffer, &req);
        VkMemoryWin32HandlePropertiesKHR handleProperties{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
        check(api.handleProperties(vk, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, resource, &handleProperties), "FSR RR packet memory types");
        const uint32_t type = memory_type(req.memoryTypeBits & handleProperties.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkImportMemoryWin32HandleInfoKHR imported{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
        imported.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT; imported.handle = resource;
        VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, &imported};
        dedicated.buffer = packetBuffer;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &dedicated};
        ai.allocationSize = req.size; ai.memoryTypeIndex = type;
        check(api.allocate(vk, &ai, nullptr, &packetMemory), "FSR RR packet import");
        check(api.bindBuffer(vk, packetBuffer, packetMemory, 0), "FSR RR packet bind");
    }

    void create_intermediate(int i) {
            auto& t = textures[i];
            t.desc = {}; t.desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            t.desc.Width = creation.width; t.desc.Height = creation.height; t.desc.DepthOrArraySize = 1; t.desc.MipLevels = 1;
            t.desc.Format = i == LinearDepth ? DXGI_FORMAT_R32_FLOAT : i == Widened ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
            t.desc.SampleDesc.Count = 1; t.desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            dx(vdev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &t.desc, t.state, nullptr, IID_PPV_ARGS(&t.resource)), "FSR RR intermediate texture");
            if (nativeD3d12) return;
            UINT64 handle = 0, offset = 0; VkFormat format = VK_FORMAT_UNDEFINED;
            dx(interop->GetVulkanResourceInfo1(t.resource.Get(), &handle, &offset, &format), "FSR RR intermediate Vulkan info");
            if (!handle) throw std::runtime_error("FSR RR intermediate has no Vulkan image");
            t.image = reinterpret_cast<VkImage>(handle);
            interop->GetVulkanImageLayout(t.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &t.layout);
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            info.image = t.image; info.viewType = VK_IMAGE_VIEW_TYPE_2D; info.format = format;
            info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            check(passes.createView(vk, &info, nullptr, &t.view), "FSR RR intermediate view");
    }
    void create_intermediates() {
        // The float output image is needed only for an FSR-only RGBA32F output.
        for (int i = 0; i < Widened; ++i) create_intermediate(i);
    }

    // AMD's tweakable settings (FfxApiConfigureDenoiserKey 1-6). Defaults are logged; D4R_FSR_RR_KEY_<n>=<float>
    // overrides key n (diagnostics) and D4R_FSR_RR_STABILITY_BIAS ([RayReconstruction] FsrStabilityBias) sets key 2.
    // AMD's default stability bias of 1 weights history heavily, which blurs fine detail on some content.
    void apply_denoiser_keys() {
        static const char* const names[7] = {"", "cross bilateral normal strength", "stability bias", "max radiance",
                                             "radiance clip std k", "gaussian kernel relaxation", "disocclusion threshold"};
        std::string defaults;
        for (uint64_t key = 1; key <= 6; ++key) {
            float value = 0.0f;
            ffxQueryDescDenoiserGetDefaultKeyValue query = {};
            query.header.type = FFX_API_QUERY_DESC_TYPE_DENOISER_GET_DEFAULT_KEYVALUE;
            query.key = key; query.count = 1; query.data = &value;
            char text[96];
            if (runtime.query(&context, &query.header) == FFX_API_RETURN_OK)
                std::snprintf(text, sizeof(text), "%s%s %g", defaults.empty() ? "" : ", ", names[key], value);
            else
                std::snprintf(text, sizeof(text), "%s%s ?", defaults.empty() ? "" : ", ", names[key]);
            defaults += text;
            char variable[32];
            std::snprintf(variable, sizeof(variable), "D4R_FSR_RR_KEY_%u", unsigned(key));
            std::string override = std::getenv(variable) ? std::getenv(variable) : "";
            if (override.empty() && key == 2 && std::getenv("D4R_FSR_RR_STABILITY_BIAS") && std::string(std::getenv("D4R_FSR_RR_STABILITY_BIAS")) != "auto")
                override = std::getenv("D4R_FSR_RR_STABILITY_BIAS");
            if (override.empty()) continue;
            char* end = nullptr;
            const float wanted = std::strtof(override.c_str(), &end);
            if (end == override.c_str() || !std::isfinite(wanted)) {
                std::fprintf(stderr, "FSR_RR: %s override '%s' is not a number; ignored\n", names[key], override.c_str());
                continue;
            }
            ffxConfigureDescDenoiserKeyValue configure = {};
            configure.header.type = FFX_API_CONFIGURE_DESC_TYPE_DENOISER_KEYVALUE;
            configure.key = key; configure.count = 1; configure.data = &wanted;
            const ffxReturnCode_t configured = runtime.configure(&context, &configure.header);
            std::fprintf(stderr, "FSR_RR: %s set to %g (%s)\n", names[key], wanted, fsr_rr_return_name(configured));
        }
        if (log_enabled()) std::fprintf(stderr, "FSR_RR denoiser defaults: %s\n", defaults.c_str());
    }
    // [RayReconstruction] FsrDetail (D4R_FSR_RR_DETAIL, 0-1): the share of the colour that bypasses the denoiser through
    // the residual; D4R_FSR_RR_DEMOD_FLOOR (default 0.1): the smallest albedo a signal is divided by; D4R_FSR_RR_HALO_LIMIT
    // (default 1, 0 = off): caps each denoised lobe at that multiple of the brightest noisy input around the texel.
    static float env_float(const char* name, float fallback, float low, float high) {
        const char* text = std::getenv(name);
        if (!text || !*text) return fallback;
        char* end = nullptr; const float value = std::strtof(text, &end);
        if (end == text || !std::isfinite(value) || value < low || value > high)
            throw std::runtime_error(std::string(name) + " must be a number between " + std::to_string(low) + " and " + std::to_string(high));
        return value;
    }

    void create_context() {
        const auto start = Clock::now();
        // Diffuse is a direct signal (the game's diffuse hit distance is not carried); specular is indirect only
        // when the game passes a specular hit distance, which decides the context's signal set once.
        signals = FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE | FFX_DENOISER_SIGNAL_DIRECT_SPECULAR;
        const ffxReturnCode_t created = fsr_rr_create_context(runtime, vdev.Get(), creation.width, creation.height, signals, 0, context);
        if (created != FFX_API_RETURN_OK) throw std::runtime_error(std::string("FSR RR ffxCreateContext failed: ") + fsr_rr_return_name(created));
        apply_denoiser_keys();
        if (log_enabled()) std::fprintf(stderr, "FSR_RR context created in %.0f ms (%ux%u, signals 0x%x)\n", since(start), creation.width, creation.height, signals);
    }

    void validate_frame(const Snapshot& frame) {
        if (failed) throw std::runtime_error("FSR RR backend is poisoned after failed submission");
        if (frame.renderWidth < 64 || frame.renderHeight < 64 || frame.renderWidth > creation.width || frame.renderHeight > creation.height)
            throw std::runtime_error("RR render extent must be at least 64 and within creation geometry");
        // The exposure texture belongs to the Super Sampling stage (it receives it); the denoiser does not use it.
        if (frame.resources[Snapshot::Exposure] && !superSampling) throw std::runtime_error("FSR RR exposure texture needs the Super Sampling stage");
        // The old dynamic fixture shrank below NVIDIA's Quality minimum, triggering BAD00005 and a teardown fault.
        // Query the SR library's actual range and reject unsupported sizes before recording GPU work.
        if (superSampling && (frame.renderWidth < srRenderLimits[0] || frame.renderHeight < srRenderLimits[1] ||
            frame.renderWidth > srRenderLimits[2] || frame.renderHeight > srRenderLimits[3]))
            throw std::runtime_error("FSR RR render size is outside the Super Sampling dynamic render range");
        // Like the Linux FSR backend: the exposure scale goes to the Super Sampling stage (it receives
        // frame.exposureScale), and the NGX guide encoding hints (DLSS.*.Type) are not read.
        if (frame.exposureScale != 1 && !superSampling) throw std::runtime_error("FSR RR exposure scale needs the Super Sampling stage");
        if (!superSampling && (frame.renderWidth != creation.outWidth || frame.renderHeight != creation.outHeight))
            throw std::runtime_error("FSR RR upscaling needs a Super Sampling stage (none is installed) or output size == render size");
        for (size_t i = 0; i < Snapshot::Count; ++i) {
            if (i == Snapshot::Exposure) continue;
            auto* resource = frame.resources[i].Get();
            if (!resource && i >= Snapshot::HitDistance) continue;
            if (!resource) throw std::runtime_error("Missing required FSR RR texture");
            ComPtr<ID3D12Device> owner; dx(resource->GetDevice(IID_PPV_ARGS(&owner)), "FSR RR texture device");
            if (owner.Get() != d3d.Get()) throw std::runtime_error("FSR RR texture belongs to another device");
            const auto desc = resource_desc(resource);
            if (i != Snapshot::Output) (void)input_format(desc.Format, i);
            if (i == Snapshot::Depth) {
                (void)copy_format(d3d.Get(), resource, i);
                if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) &&
                    (frame.origins[i][0] || frame.origins[i][1] || frame.renderWidth != desc.Width || frame.renderHeight != desc.Height))
                    throw std::runtime_error("FSR RR depth-stencil requires a full depth-plane rectangle");
            }
            const uint32_t width = i == Snapshot::Output ? layout.ow : frame.renderWidth;
            const uint32_t height = i == Snapshot::Output ? layout.oh : frame.renderHeight;
            if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || (i == Snapshot::Output && !superSampling &&
                desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT && desc.Format != DXGI_FORMAT_R32G32B32A32_FLOAT) ||
                desc.SampleDesc.Count != 1 || desc.DepthOrArraySize != 1 ||
                frame.origins[i][0] > desc.Width || width > desc.Width - frame.origins[i][0] ||
                frame.origins[i][1] > desc.Height || height > desc.Height - frame.origins[i][1])
                throw std::runtime_error("Unsupported FSR RR texture format or subrect (FSR-only output must be RGBA16F or RGBA32F)");
            if (!nativeD3d12 && i == Snapshot::Output && !superSampling && desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT) {
                VkFormatProperties source{}, destination{};
                api.formatProperties(vkPhysical, VK_FORMAT_R16G16B16A16_SFLOAT, &source);
                api.formatProperties(vkPhysical, VK_FORMAT_R32G32B32A32_SFLOAT, &destination);
                if (!(source.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ||
                    !(destination.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
                    throw std::runtime_error("FSR RR output widening requires half-to-float blit support");
            }
            if (i == Snapshot::Output && !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
                throw std::runtime_error("RR output lacks UAV capability");
        }
        for (size_t i = 0; i < Snapshot::Count; ++i)
            if (i != Snapshot::Output && frame.resources[i] == frame.resources[Snapshot::Output]) throw std::runtime_error("RR input/output alias is unsupported");
    }

    void ensure_staging(size_t plane) {
        auto& s = staging[plane]; const VkFormat wanted = layout.planes[plane].format;
        if (s.image && s.format == wanted) return;
        destroy_staging(s);
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ci.imageType = VK_IMAGE_TYPE_2D; ci.format = wanted;
        ci.extent = {creation.width, creation.height, 1}; ci.mipLevels = ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(api.createImage(vk, &ci, nullptr, &s.image), "FSR RR staging image");
        VkMemoryRequirements req{}; api.imageRequirements(vk, s.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(api.allocate(vk, &ai, nullptr, &s.memory), "FSR RR staging memory");
        check(api.bindImage(vk, s.image, s.memory, 0), "FSR RR staging bind");
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = s.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = wanted; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(passes.createView(vk, &vi, nullptr, &s.view), "FSR RR staging view");
        s.format = wanted;
    }

    void wait_vkd3d() {
        const uint64_t value = ++vsequence;
        dx(vqueue->Signal(vfence.Get(), value), "FSR RR vkd3d signal");
        dx(vfence->SetEventOnCompletion(value, vevent.value), "FSR RR vkd3d completion event");
        if (WaitForSingleObject(vevent.value, 60000) != WAIT_OBJECT_0) throw std::runtime_error("FSR RR vkd3d completion timeout");
        if (vfence->GetCompletedValue() == UINT64_MAX) throw std::runtime_error("FSR RR vkd3d device removed");
    }
    void prepare_native_guides(const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>& access) {
        for (size_t i : {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}) {
            if (!layout.planes[i].enabled) continue;
            const size_t index = resource_index(i);
            nativePasses.copy_guide(vlist.Get(), frame.resources[index].Get(), access[index], index == Snapshot::Depth,
                layout.rw, layout.rh, creation.width, creation.height, frame.origins[index][0], frame.origins[index][1], nativeGuides[i]);
        }
    }
    void wait_game(ID3D12CommandQueue* q, uint64_t value) {
        dx(q->Signal(fence.Get(), value), "FSR RR game signal");
        dx(fence->SetEventOnCompletion(value, event.value), "FSR RR game completion event");
        if (WaitForSingleObject(event.value, 10000) != WAIT_OBJECT_0) throw std::runtime_error("FSR RR D3D12 completion timeout");
        if (fence->GetCompletedValue() == UINT64_MAX) throw std::runtime_error("FSR RR D3D12 device removed");
    }
    void validate(ID3D12CommandQueue* q, const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>& access) {
        if (failed) throw std::runtime_error("FSR RR backend is poisoned after failed submission");
        if (!q) throw std::runtime_error("Null RR submission queue");
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
#if defined(__MINGW32__)
        q->GetDesc(&queueDesc);
#else
        queueDesc = q->GetDesc();
#endif
        if (queueDesc.Type != D3D12_COMMAND_LIST_TYPE_DIRECT) throw std::runtime_error("RR requires a direct queue");
        ComPtr<ID3D12Device> owner; dx(q->GetDevice(IID_PPV_ARGS(&owner)), "RR queue device");
        if (owner.Get() != d3d.Get()) throw std::runtime_error("RR queue belongs to another device");
        for (size_t i = 0; i < access.size(); ++i)
            if (frame.resources[i] && (access[i].pending_split || access[i].inherited)) throw std::runtime_error("Unresolved RR texture state");
        validate_frame(frame);
    }

    void evaluate(ID3D12CommandQueue* q, const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>& access) {
        validate(q, frame, access);
        retained = std::make_unique<Snapshot>(frame);
        failed = true;
        const auto begin = Clock::now();
        layout.rw = frame.renderWidth; layout.rh = frame.renderHeight;
        for (size_t i = 0; i < layout.planes.size(); ++i) {
            const auto resource = resource_index(i);
            layout.planes[i].enabled = frame.resources[resource] != nullptr;
            if (!layout.planes[i].enabled) continue;
            const auto format = input_format(resource_desc(frame.resources[resource].Get()).Format, resource);
            layout.planes[i].format = format.format; layout.planes[i].bytes = format.bytes;
        }
        const bool floatOutput = !superSampling && resource_desc(frame.resources[Snapshot::Output].Get()).Format == DXGI_FORMAT_R32G32B32A32_FLOAT;
        if (floatOutput && !textures[Widened].resource) create_intermediate(Widened);
        layout.outputBytes = floatOutput ? 16 : 8;
        layout.outputFormat = floatOutput ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        const auto outputFormat = floatOutput ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
        commands::InternalScope internal;
        if (nativeD3d12) {
            vqueue = q;
            if (log_enabled() && !stamps) {
                D3D12_QUERY_HEAP_DESC heap{}; heap.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; heap.Count = 4;
                if (SUCCEEDED(vdev->CreateQueryHeap(&heap, IID_PPV_ARGS(&stamps))) &&
                    SUCCEEDED(q->GetTimestampFrequency(&stampFrequency)) && stampFrequency)
                    stampReadback = make_buffer(vdev.Get(), 4 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
                else stamps.Reset();
            }
            pending = true; queue = q;
            run_vkd3d(frame, &access);
            if (superSampling) {
                const auto upscaleStart = Clock::now();
                superSampling(q, textures[Composed].resource.Get(), frame, access);
                if (log_enabled()) std::fprintf(stderr, "FSR_RR super sampling submitted in %.2f ms\n", since(upscaleStart));
            }
            pending = false; failed = false; queue.Reset(); retained.reset();
            if (log_enabled()) std::fprintf(stderr, "FSR_RR native frame %u: %ux%u total %.2f ms\n", frameIndex, layout.rw, layout.rh, since(begin));
            ++frameIndex;
            return;
        }
        // --- 1. the game's guides into the shared packet ---
        dx(producerAllocator->Reset(), "FSR RR producer allocator reset"); dx(producer->Reset(producerAllocator.Get(), nullptr), "FSR RR producer reset");
        transition(producer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        for (size_t i = 0; i < layout.planes.size(); ++i) {
            const auto resource = resource_index(i);
            auto* texture = frame.resources[resource].Get(); const auto& plane = layout.planes[i];
            if (!plane.enabled) continue;
            texture_copy_barrier(producer.Get(), texture, access[resource], false, false);
            auto source = rr_bridge::D3D12Frames::texture(texture);
            auto dest = rr_bridge::D3D12Frames::placed(shared.Get(), plane.offset, plane.pitch, copy_format(d3d.Get(), texture, resource), layout.rw, layout.rh);
            const auto x = frame.origins[resource][0], y = frame.origins[resource][1];
            D3D12_BOX box{x, y, 0, x + layout.rw, y + layout.rh, 1};
            const bool depthStencil = (resource_desc(texture).Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
            producer->CopyTextureRegion(&dest, 0, 0, 0, &source, depthStencil ? nullptr : &box);
            texture_copy_barrier(producer.Get(), texture, access[resource], false, true);
        }
        transition(producer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        dx(producer->Close(), "FSR RR producer close");
        ID3D12CommandList* inputs[] = {producer.Get()}; q->ExecuteCommandLists(1, inputs);
        pending = true; queue = q;
        wait_game(q, ++sequence);
        const double producedMs = since(begin);
        // --- 2. conversion, denoising and composition on the vkd3d device ---
        run_vkd3d(frame);
        const double vkd3dMs = since(begin) - producedMs;
        // --- 3. the packet's output into the game's output texture ---
        dx(consumerAllocator->Reset(), "FSR RR consumer allocator reset"); dx(consumer->Reset(consumerAllocator.Get(), nullptr), "FSR RR consumer reset");
        transition(consumer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto* output = frame.resources[Snapshot::Output].Get();
        if (superSampling) {
            // the denoised colour at render size goes to a game-device texture; the Super Sampling stage upscales it
            if (!denoised) {
                D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = creation.width; desc.Height = creation.height;
                desc.DepthOrArraySize = desc.MipLevels = 1; desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count = 1;
                desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                dx(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, denoisedState, nullptr, IID_PPV_ARGS(&denoised)), "FSR RR denoised texture");
            }
            transition(consumer.Get(), denoised.Get(), denoisedState, D3D12_RESOURCE_STATE_COPY_DEST);
            auto source = rr_bridge::D3D12Frames::placed(shared.Get(), layout.outputOffset, layout.outputPitch, outputFormat, layout.rw, layout.rh);
            auto dest = rr_bridge::D3D12Frames::texture(denoised.Get());
            consumer->CopyTextureRegion(&dest, 0, 0, 0, &source, nullptr);
            denoisedState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            transition(consumer.Get(), denoised.Get(), D3D12_RESOURCE_STATE_COPY_DEST, denoisedState);
        } else {
            texture_copy_barrier(consumer.Get(), output, access[Snapshot::Output], true, false);
            auto source = rr_bridge::D3D12Frames::placed(shared.Get(), layout.outputOffset, layout.outputPitch, outputFormat, layout.ow, layout.oh);
            auto dest = rr_bridge::D3D12Frames::texture(output);
            consumer->CopyTextureRegion(&dest, frame.origins[Snapshot::Output][0], frame.origins[Snapshot::Output][1], 0, &source, nullptr);
            texture_copy_barrier(consumer.Get(), output, access[Snapshot::Output], true, true);
        }
        transition(consumer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        dx(consumer->Close(), "FSR RR consumer close");
        ID3D12CommandList* outputs[] = {consumer.Get()}; q->ExecuteCommandLists(1, outputs);
        wait_game(q, ++sequence);
        if (superSampling) {
            // the stage records/submits its own work on `q`; the denoised texture is in NON_PIXEL_SHADER_RESOURCE
            const auto upscaleStart = Clock::now();
            superSampling(q, denoised.Get(), frame, access);
            if (log_enabled()) std::fprintf(stderr, "FSR_RR super sampling submitted in %.2f ms\n", since(upscaleStart));
        }
        pending = false; failed = false; queue.Reset(); retained.reset();
        if (log_enabled())
            std::fprintf(stderr, "FSR_RR frame %u: %ux%u produce %.2f ms, vkd3d (convert+denoise+compose) %.2f ms, total %.2f ms\n",
                         frameIndex, layout.rw, layout.rh, producedMs, vkd3dMs, since(begin));
        ++frameIndex;
    }

    void run_vkd3d(const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>* access = nullptr) {
        const auto& s = frame.settings;
        const uint32_t rw = layout.rw, rh = layout.rh;
        // which hit distances the game passes decides the context's signals once (as on Linux): a lobe with a
        // ray length is an indirect signal. Layout plane 9 is the Subsurface slot, carrying the diffuse hit.
        const bool specularHit = layout.planes[7].enabled, diffuseHit = frame.diffuseHit && layout.planes[9].enabled;
        const uint32_t wanted = (diffuseHit ? FFX_DENOISER_SIGNAL_INDIRECT_DIFFUSE : FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE) |
            (specularHit ? FFX_DENOISER_SIGNAL_INDIRECT_SPECULAR : FFX_DENOISER_SIGNAL_DIRECT_SPECULAR);
        if (wanted != signals) {
            if (frameIndex == 0) {
                runtime.destroyContext(&context, nullptr); context = nullptr; signals = wanted;
                const ffxReturnCode_t created = fsr_rr_create_context(runtime, vdev.Get(), creation.width, creation.height, signals, 0, context);
                if (created != FFX_API_RETURN_OK) throw std::runtime_error(std::string("FSR RR ffxCreateContext failed: ") + fsr_rr_return_name(created));
                apply_denoiser_keys();
                std::fprintf(stderr, "d4r: FSR RR diffuse %s, specular %s\n",
                    diffuseHit ? (frame.diffuseHitInW ? "indirect (hit distance in ray direction w)" : "indirect (hit distance)") : "direct (no hit distance)",
                    specularHit ? (frame.specularHitInW ? "indirect (hit distance in ray direction w)" : "indirect (hit distance)") : "direct (no hit distance)");
            } else throw std::runtime_error("FSR RR: the game's hit-distance guide changed after the denoiser was created (recreate the feature)");
        }
        if (!nativeD3d12)
            for (size_t plane : {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}) if (layout.planes[plane].enabled) ensure_staging(plane);
        double clipToView[16], viewToWorld[16];
        if (!invert(s.viewToClip, clipToView) || !invert(s.worldToView, viewToWorld)) throw std::runtime_error("FSR RR camera matrix is not invertible");
        const bool reset = s.reset || !havePrevious;
        double previousView[16], viewToPrevious[16];
        for (int i = 0; i < 16; ++i) previousView[i] = reset ? s.worldToView[i] : this->previousView[i];
        multiply(viewToWorld, previousView, viewToPrevious);
        D4rFsrRrPrepareConstants c = {};
        for (int i = 0; i < 16; ++i) { c.clipToView[i] = float(clipToView[i]); c.viewToPreviousView[i] = float(viewToPrevious[i]); }
        c.renderSize[0] = float(rw); c.renderSize[1] = float(rh); c.renderSize[2] = 1.0f / rw; c.renderSize[3] = 1.0f / rh;
        c.motion[0] = s.motionScale[0] / rw; c.motion[1] = s.motionScale[1] / rh; c.motion[2] = 1.0f; c.motion[3] = 1.0f;
        uint32_t flags = 0;
        if (s.hardwareDepth) flags |= kFsrRrHardwareDepth;
        if (s.roughnessPacked) flags |= kFsrRrRoughnessPacked;
        if (specularHit) flags |= kFsrRrSpecularHit | (frame.specularHitInW ? kFsrRrSpecularHitInW : 0u);
        if (diffuseHit) flags |= kFsrRrDiffuseHit | (frame.diffuseHitInW ? kFsrRrDiffuseHitInW : 0u);
        if (unorm_guide(layout.planes[3].format)) flags |= kFsrRrNormalsUnorm;
        if (layout.planes[10].enabled) flags |= kFsrRrBeforeParticles;
        flags |= kFsrRrInvalidNormalPassthrough;
        c.motionBaseFlags[2] = flags;
        const float viewZSign = s.viewToClip[11] < 0.0f ? -1.0f : 1.0f;
        c.depthParams[0] = viewZSign; c.depthParams[1] = s.depthInverted ? 0.0f : 1.0f; c.depthParams[2] = 1.0e6f;
        c.detail[0] = env_float("D4R_FSR_RR_DETAIL", 0.0f, 0.0f, 1.0f);
        c.detail[1] = std::max(env_float("D4R_FSR_RR_DEMOD_FLOOR", 0.1f, 0.0f, 1.0f), 1e-3f);

        dx(vallocator->Reset(), "FSR RR vkd3d allocator reset"); dx(vlist->Reset(vallocator.Get(), nullptr), "FSR RR vkd3d list reset");
        stamp(0);
        auto setState = [&](int index, D3D12_RESOURCE_STATES state) {
            auto& t = textures[index];
            if (t.resource) { transition(vlist.Get(), t.resource.Get(), t.state, state); t.state = state; }
        };
        for (int i = 0; i < TexCount; ++i) setState(i, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        static const int targets[D4rFsrRrPasses::kPrepareOutputs] = {LinearDepth, MotionT, NormalsT, DiffuseAlbedo, SpecularAlbedo, DiffuseT, SpecularT, Residual};
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkBufferMemoryBarrier acquire{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        if (nativeD3d12) {
            prepare_native_guides(frame, *access);
            ID3D12Resource* resources[18]{}; DXGI_FORMAT formats[18]{};
            const size_t planes[10] = {0, 2, 1, 3, 4, 5, 6, specularHit ? 7u : 3u,
                diffuseHit ? 9u : 3u, layout.planes[10].enabled ? 10u : 0u};
            for (size_t i = 0; i < 10; ++i) {
                resources[i] = nativeGuides[planes[i]].resource.Get(); formats[i] = nativeGuides[planes[i]].format;
            }
            for (size_t i = 0; i < 8; ++i) {
                resources[10 + i] = textures[targets[i]].resource.Get(); formats[10 + i] = textures[targets[i]].desc.Format;
            }
            nativePasses.record(vlist.Get(), nativePasses.prepare, &c, resources, formats, rw, rh);
        } else {
            dx(interop->BeginVkCommandBufferInterop(vlist.Get(), &command), "FSR RR Vulkan interop begin");
            // the packet comes in from the game's device
            acquire.buffer = packetBuffer; acquire.size = VK_WHOLE_SIZE;
            acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL; acquire.dstQueueFamilyIndex = vkFamily;
            acquire.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            api.barrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &acquire, 0, nullptr);
            for (size_t plane : {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}) {
                if (!layout.planes[plane].enabled) continue;
                const auto& st = staging[plane];
                VkImageMemoryBarrier toCopy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; toCopy.image = st.image;
                toCopy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; toCopy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toCopy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; toCopy.srcQueueFamilyIndex = toCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toCopy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                api.barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toCopy);
                VkBufferImageCopy region{}; region.bufferOffset = layout.planes[plane].offset;
                region.bufferRowLength = uint32_t(layout.planes[plane].pitch / layout.planes[plane].bytes);
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; region.imageExtent = {rw, rh, 1};
                api.copyBufferToImage(command, packetBuffer, st.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                VkImageMemoryBarrier toRead = toCopy; toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                api.barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toRead);
            }
            auto sampled = [&](size_t plane) { return D4rFsrRrImage{staging[plane].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; };
            const bool hasHit = layout.planes[7].enabled, hasParticles = layout.planes[10].enabled;
            D4rFsrRrImage images[D4rFsrRrPasses::kPrepareInputs + D4rFsrRrPasses::kPrepareOutputs] = {
                sampled(0), sampled(2), sampled(1), sampled(3), sampled(4), sampled(5), sampled(6),
                hasHit ? sampled(7) : sampled(3), diffuseHit ? sampled(9) : sampled(3), hasParticles ? sampled(10) : sampled(0)};
            for (uint32_t i = 0; i < D4rFsrRrPasses::kPrepareOutputs; ++i)
                images[D4rFsrRrPasses::kPrepareInputs + i] = {textures[targets[i]].view, textures[targets[i]].layout};
            passes.record_prepare(command, images, c, rw, rh);
            dx(interop->EndVkCommandBufferInterop(vlist.Get()), "FSR RR Vulkan interop end");
        }
        stamp(1);

        // --- AMD FSR Ray Regeneration (D3D12 on vkd3d-proton) ---
        const D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        for (int index : {LinearDepth, MotionT, NormalsT, DiffuseAlbedo, SpecularAlbedo, DiffuseT, SpecularT}) setState(index, read);
        auto ffx = [&](int index, uint32_t state) {
            const auto& t = textures[index];
            return fsr_rr_texture(t.resource.Get(), t.desc, fsr_rr_surface_format(t.desc.Format), state);
        };
        const uint32_t ffxRead = FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ;
        ffxDispatchDescDenoiser desc = {};
        desc.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER;
        desc.commandList = vlist.Get();
        desc.linearDepth = ffx(LinearDepth, ffxRead); desc.motionVectors = ffx(MotionT, ffxRead); desc.normals = ffx(NormalsT, ffxRead);
        desc.specularAlbedo = ffx(SpecularAlbedo, ffxRead); desc.diffuseAlbedo = ffx(DiffuseAlbedo, ffxRead);
        desc.motionVectorScale = {1.0f, 1.0f, 1.0f};
        desc.jitterOffsets = {s.jitter[0], s.jitter[1]};
        const double position[3] = {viewToWorld[12], viewToWorld[13], viewToWorld[14]};
        double previousPosition[3] = {position[0], position[1], position[2]};
        if (!reset) {
            double previousInverse[16];
            if (invert(this->previousView, previousInverse)) previousPosition[0] = previousInverse[12], previousPosition[1] = previousInverse[13], previousPosition[2] = previousInverse[14];
        }
        desc.cameraPositionDelta = {float(previousPosition[0] - position[0]), float(previousPosition[1] - position[1]), float(previousPosition[2] - position[2])};
        std::memcpy(&desc.view, s.worldToView, sizeof(desc.view)); std::memcpy(&desc.projection, s.viewToClip, sizeof(desc.projection));
        desc.linearDepthBounds = {0.0f, 1.0e6f};
        desc.renderSize = {rw, rh};
        desc.frameIndex = frameIndex;
        desc.flags = uint32_t(FFX_DENOISER_DISPATCH_NON_GAMMA_ALBEDO) | (reset ? uint32_t(FFX_DENOISER_DISPATCH_RESET) : 0u);
        ffxDispatchDescDenoiserDirectDiffuse directDiffuse = {}; ffxDispatchDescDenoiserDirectSpecular directSpecular = {};
        ffxDispatchDescDenoiserIndirectDiffuse indirectDiffuse = {}; ffxDispatchDescDenoiserIndirectSpecular indirectSpecular = {};
        FfxApiDenoiserSignal* diffuseSignal; ffxApiHeader* diffuseHeader;
        if (diffuseHit) {
            indirectDiffuse.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_INDIRECT_DIFFUSE; diffuseSignal = &indirectDiffuse.signal; diffuseHeader = &indirectDiffuse.header;
        } else {
            directDiffuse.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_DIFFUSE; diffuseSignal = &directDiffuse.signal; diffuseHeader = &directDiffuse.header;
        }
        FfxApiDenoiserSignal* specularSignal; ffxApiHeader* specularHeader;
        if (specularHit) {
            indirectSpecular.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_INDIRECT_SPECULAR; specularSignal = &indirectSpecular.signal; specularHeader = &indirectSpecular.header;
        } else {
            directSpecular.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_SPECULAR; specularSignal = &directSpecular.signal; specularHeader = &directSpecular.header;
        }
        diffuseSignal->input = ffx(DiffuseT, ffxRead); diffuseSignal->output = ffx(DiffuseOut, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        specularSignal->input = ffx(SpecularT, ffxRead); specularSignal->output = ffx(SpecularOut, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        desc.header.pNext = diffuseHeader; diffuseHeader->pNext = specularHeader;
        const ffxReturnCode_t dispatched = runtime.dispatch(&context, &desc.header);
        if (dispatched != FFX_API_RETURN_OK) throw std::runtime_error(std::string("FSR RR ffxDispatch failed: ") + fsr_rr_return_name(dispatched));
        D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; vlist->ResourceBarrier(1, &uav);
        stamp(2);

        // --- composition, then the composed image into the packet's output region ---
        for (int i = 0; i < TexCount; ++i) setState(i, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (nativeD3d12) {
            ID3D12Resource* resources[8]{}; DXGI_FORMAT formats[8]{};
            const int indices[8] = {DiffuseOut, SpecularOut, DiffuseAlbedo, SpecularAlbedo, Residual, Composed, DiffuseT, SpecularT};
            for (size_t i = 0; i < 8; ++i) {
                resources[i] = textures[indices[i]].resource.Get(); formats[i] = textures[indices[i]].desc.Format;
            }
            struct { uint32_t width, height; float haloLimit; uint32_t unused; } composeConstants =
                {rw, rh, env_float("D4R_FSR_RR_HALO_LIMIT", 1.0f, 0.0f, 1000.0f), 0};
            nativePasses.record(vlist.Get(), nativePasses.compose, &composeConstants, resources, formats, rw, rh);
            int copiedImage = Composed;
            if (layout.outputBytes == 16) {
                setState(Composed, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                ID3D12Resource* widening[] = {textures[Composed].resource.Get(), textures[Widened].resource.Get()};
                DXGI_FORMAT wideningFormats[] = {textures[Composed].desc.Format, textures[Widened].desc.Format};
                const uint32_t extent[2] = {rw, rh};
                nativePasses.record(vlist.Get(), nativePasses.widen, extent, widening, wideningFormats, rw, rh);
                copiedImage = Widened;
            }
            if (superSampling) {
                setState(Composed, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            } else {
                setState(copiedImage, D3D12_RESOURCE_STATE_COPY_SOURCE);
                auto* output = frame.resources[Snapshot::Output].Get();
                texture_copy_barrier(vlist.Get(), output, (*access)[Snapshot::Output], true, false);
                auto source = rr_bridge::D3D12Frames::texture(textures[copiedImage].resource.Get());
                auto destination = rr_bridge::D3D12Frames::texture(output);
                D3D12_BOX box{0, 0, 0, rw, rh, 1};
                vlist->CopyTextureRegion(&destination, frame.origins[Snapshot::Output][0], frame.origins[Snapshot::Output][1], 0, &source, &box);
                texture_copy_barrier(vlist.Get(), output, (*access)[Snapshot::Output], true, true);
            }
        } else {
            const D4rFsrRrImage compose[D4rFsrRrPasses::kComposeImages] = {
                {textures[DiffuseOut].view, textures[DiffuseOut].layout}, {textures[SpecularOut].view, textures[SpecularOut].layout},
                {textures[DiffuseAlbedo].view, textures[DiffuseAlbedo].layout}, {textures[SpecularAlbedo].view, textures[SpecularAlbedo].layout},
                {textures[Residual].view, textures[Residual].layout}, {textures[Composed].view, textures[Composed].layout},
                {textures[DiffuseT].view, textures[DiffuseT].layout}, {textures[SpecularT].view, textures[SpecularT].layout}};
            dx(interop->BeginVkCommandBufferInterop(vlist.Get(), &command), "FSR RR Vulkan interop begin (compose)");
            passes.record_compose(command, compose, rw, rh, env_float("D4R_FSR_RR_HALO_LIMIT", 1.0f, 0.0f, 1000.0f));
            int copiedImage = Composed;
            if (layout.outputBytes == 16) {
                // Preserve the denoiser/composition's half arithmetic; widen only the completed output on the GPU.
                VkImageBlit region{}; region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.srcOffsets[1] = region.dstOffsets[1] = {int32_t(rw), int32_t(rh), 1};
                api.blitImage(command, textures[Composed].image, textures[Composed].layout,
                    textures[Widened].image, textures[Widened].layout, 1, &region, VK_FILTER_NEAREST);
                passes.full_barrier(command);
                copiedImage = Widened;
            }
            VkBufferImageCopy back{}; back.bufferOffset = layout.outputOffset; back.bufferRowLength = uint32_t(layout.outputPitch / layout.outputBytes);
            back.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; back.imageExtent = {rw, rh, 1};
            api.copyImageToBuffer(command, textures[copiedImage].image, textures[copiedImage].layout, packetBuffer, 1, &back);
            VkBufferMemoryBarrier release = acquire; release.srcQueueFamilyIndex = vkFamily; release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            release.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; release.dstAccessMask = 0;
            api.barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1, &release, 0, nullptr);
            dx(interop->EndVkCommandBufferInterop(vlist.Get()), "FSR RR Vulkan interop end (compose)");
        }
        // Diagnostic (D4R_FSR_RR_DUMP_DIR, first D4R_FSR_RR_DUMP_FRAMES frames, default 3): raw copies of the converted
        // linear depth and motion textures, to check the DLSS -> FFX guide conversion numerically.
        stamp(3);
        if (stamps) vlist->ResolveQueryData(stamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, stampReadback.Get(), 0);
        const char* dumpDir = std::getenv("D4R_FSR_RR_DUMP_DIR");
        const unsigned dumpFrames = std::getenv("D4R_FSR_RR_DUMP_FRAMES") ? unsigned(std::atoi(std::getenv("D4R_FSR_RR_DUMP_FRAMES"))) : 3;
        struct Dump { int index; const char* name; ComPtr<ID3D12Resource> buffer; D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint; UINT rows; UINT bytesPerTexel; };
        std::vector<Dump> dumps;
        if (dumpDir && frameIndex < dumpFrames) {
            for (auto [index, name, texel] : {std::tuple<int, const char*, UINT>{LinearDepth, "linear-depth", 4}, {MotionT, "motion", 8}, {NormalsT, "normals", 8}, {Composed, "composed", 8}}) {
                Dump d{index, name, {}, {}, 0, texel};
                UINT64 rowBytes = 0, total = 0; auto desc = textures[index].desc;
                vdev->GetCopyableFootprints(&desc, 0, 1, 0, &d.footprint, &d.rows, &rowBytes, &total);
                d.buffer = make_buffer(vdev.Get(), total, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
                transition(vlist.Get(), textures[index].resource.Get(), textures[index].state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION destination{}; destination.pResource = d.buffer.Get();
                destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; destination.PlacedFootprint = d.footprint;
                D3D12_TEXTURE_COPY_LOCATION source{}; source.pResource = textures[index].resource.Get(); source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                vlist->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
                transition(vlist.Get(), textures[index].resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, textures[index].state);
                dumps.push_back(std::move(d));
            }
        }
        dx(vlist->Close(), "FSR RR vkd3d list close");
        ID3D12CommandList* lists[] = {vlist.Get()}; vqueue->ExecuteCommandLists(1, lists);
        wait_vkd3d();
        if (stamps) {
            void* mapped = nullptr; D3D12_RANGE range{0, 4 * sizeof(UINT64)};
            if (SUCCEEDED(stampReadback->Map(0, &range, &mapped))) {
                UINT64 t[4]; std::memcpy(t, mapped, sizeof(t)); stampReadback->Unmap(0, nullptr);
                auto ms = [&](int a, int b) { return double(t[b] - t[a]) * 1000.0 / double(stampFrequency); };
                std::fprintf(stderr, "FSR_RR gpu frame %u: prepare %.2f ms, denoise %.2f ms, compose+copy %.2f ms, total %.2f ms\n", frameIndex, ms(0, 1), ms(1, 2), ms(2, 3), ms(0, 3));
            }
        }
        for (auto& d : dumps) {
            void* mapped = nullptr; D3D12_RANGE range{0, size_t(d.footprint.Footprint.RowPitch) * d.rows};
            dx(d.buffer->Map(0, &range, &mapped), "FSR RR dump map");
            const std::string path = std::string(dumpDir) + "/frame" + std::to_string(frameIndex) + "-" + d.name + ".bin";
            if (FILE* out = std::fopen(path.c_str(), "wb")) {
                for (UINT row = 0; row < rw && row < d.rows; ++row) {} // (rows are written below; width = render width)
                for (UINT row = 0; row < rh; ++row)
                    std::fwrite(static_cast<const char*>(mapped) + size_t(row) * d.footprint.Footprint.RowPitch, 1, size_t(rw) * d.bytesPerTexel, out);
                std::fclose(out);
            }
            d.buffer->Unmap(0, nullptr);
        }
        std::memcpy(this->previousView, s.worldToView, sizeof(this->previousView));
        havePrevious = true;
    }
};

FsrBackend::FsrBackend(ID3D12Device* device, const Creation& creation, const FsrConfig& config) {
    std::lock_guard<std::mutex> lock(hostMutex);
    auto candidate = std::make_unique<Impl>(device, creation, config);
    candidate->initialize(); impl = std::move(candidate);
}
FsrBackend::~FsrBackend() {
    std::lock_guard<std::mutex> lock(hostMutex);
    if (impl && impl->pending) {
        std::fprintf(stderr, "D4R_FSR_RR_QUARANTINE incomplete GPU work; retaining backend and guide resources\n");
        (void)impl.release();
    } else impl.reset();
}
namespace {
// AMD's denoiser has no disocclusion or subsurface inputs. Games pass them anyway (Cyberpunk 2077 sends
// both every frame), so they are dropped rather than failing the frame. The shim reuses the Subsurface
// slot for the diffuse hit distance (frame.diffuseHit); a real subsurface guide is dropped here.
Snapshot without_unused_guides(const Snapshot& frame) {
    Snapshot used = frame;
    used.resources[Snapshot::Disocclusion].Reset();
    if (!used.diffuseHit) used.resources[Snapshot::Subsurface].Reset();
    return used;
}
}
void FsrBackend::evaluate(ID3D12CommandQueue* queue, const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>& access) {
    std::lock_guard<std::mutex> lock(hostMutex); impl->evaluate(queue, without_unused_guides(frame), access);
}
void FsrBackend::set_super_sampling(SuperSampling stage, const std::array<unsigned, 4>& renderLimits) {
    std::lock_guard<std::mutex> lock(hostMutex); impl->superSampling = std::move(stage); impl->srRenderLimits = renderLimits;
}
void FsrBackend::validate_frame(const Snapshot& frame) {
    std::lock_guard<std::mutex> lock(hostMutex); impl->validate_frame(without_unused_guides(frame));
}
void FsrBackend::drain() {
    std::lock_guard<std::mutex> lock(hostMutex);
    if (impl->pending || impl->failed) throw std::runtime_error("FSR RR backend cannot retire failed GPU work");
}
}
