// d4r_nvngx_shim_internal.h: what the translation units of d4r_nvngx.dll share. The NGX and CUDA
// declarations, the state this shim keeps of a feature, and the entry points its units call across
// files. What a single unit keeps to itself stays static in that unit instead. Nothing outside
// d4r_nvngx.dll includes this.
#ifndef D4R_NVNGX_SHIM_INTERNAL_H
#define D4R_NVNGX_SHIM_INTERNAL_H

#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <windows.h>
#include <setupapi.h>
#include <tlhelp32.h>
#include "d4r_asi_process.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <vulkan/vulkan_core.h>
#include "d4r_motion_dilation.h"
#include "d4r_native_motion.h"
#include "d4r_wire_history.h"
#include "d4r_input_wait.h"
#include "d4r_vulkan_staging.h"

#include <algorithm>
#include <bit>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "d4r_event_wait.h"
#include "d4r_frame_completion.h"
#include "d4r_render_presets.h"
#include "d4r_win32_wait.h"
#include "d4r_vkd3d_interop.h"
#include <bcrypt.h>
#include "d4r_fsr_rr.h"
#include "d4r_fsr_rr_identity.h"
#include "d4r_fsr4_identity.h"
#include "d4r_engine_compat.h"
#include "d4r_build.h"
#include "d4r_fsr_rr_passes.h"
#include "../engine/game.h"
#include "../engine/game_m.h"
#include "../engine/game_e.h"
#include "../engine/game_rr.h"
#include "../engine/vk_dispatch.h"
#include "../engine/vkhip/vkhip_net.h"
#include "../engine/d3d12/game_d3d12.h"
#include "../engine/d3d12/game_m_d3d12.h"
#include "../engine/d3d12/game_rr_d3d12.h"
#include "../engine/d3d12/runtime_m_rdna3.h"
#include "d4r_d3d12_submit.h"
#include "d4r_d3d12_split.h"
#include "d4r_split_shaders.h"
#ifdef D4R_ENGINE_D3DHIP
#include "../engine/d3d12/hip_net_d3d12.h"
#endif

using NgxResult = unsigned int;
using CudaDevicePtr = unsigned long long;
using CudaArray = void*;
using CudaObject = unsigned long long;
using CudaEvent = void*;
using ProfileClock = std::chrono::steady_clock;

struct NgxHandle;
struct NgxPathListInfo;
struct NgxLoggingInfo;
struct NgxFeatureCommonInfo;
struct NgxFeatureRequirement;
struct CudaArrayDescriptor;
struct CudaArray3DDescriptor;
struct CudaMemcpy2D;
struct CudaResourceDesc;
struct CudaTextureDesc;
struct CudaApi;
struct CoreApi;
struct DenoiserProbe;
struct PortableInstall;
struct IniEntry;
class Worker;
enum class Plane;
struct Global;
struct DenoiserCapability;
enum class RrBackend;
struct ProjectIdentity;
enum class AmdGeneration;
struct VramBuffer;
struct FusedColorInput;
struct VramImage;
struct Staging;
struct CudaImage;
enum class AuxLayout;
struct AuxInput;
struct AuxScalar;
struct FrameParams;
struct FrameTiming;
struct HostPlane;
struct AuxSurface;
struct AuxBinding;
struct InputSlot;
struct AlphaSlot;
struct OutputSlot;
struct EngineBorrowedView;
struct FsrRrFeature;
struct HipNetApi;
struct EngineDxFeature;
struct Feature;
class RowPool;
struct CopyStreams;
struct VulkanInterop;
struct NativeCameraSource;
struct NativeImageSource;
struct NativeVram;
struct InPlaceTexture;
struct InPlaceImport;
struct VramCopy;
struct CapturePool;
struct AlphaRegion;
struct PendingDownload;
class DeferredResources;
class NativeRetirer;
struct EngineBackend;
struct EngineBackendM;
struct EngineNetCode;
struct EngineNetExports;
struct EngineFrameSource;
struct EngineBackendE;
struct EngineBackendR;
struct EngineBackendDx;
struct EngineDxMarks;

// this shim's own functions, the ones its units call across files
std::string ascii_lower(std::string text);
std::vector<IniEntry> parse_ini(const std::string& text);
int ini_flag(const std::vector<IniEntry>& ini, const char* section, const char* key, int fallback);
bool read_whole_file(const std::wstring& path, std::string& text);
bool file_exists(const std::wstring& path);
std::string unix_path(const std::wstring& path);
std::wstring widen(const std::string& text);
void load_portable_config();
void logf(const char* format, ...);
unsigned int env_uint(const char* name, unsigned int fallback);
std::string env_string(const char* name);
bool profile_enabled();
bool profile_log_enabled();
bool profile_log_frame(uint32_t frame);
d4r::GameProfile engine_stage_profile(uint32_t frame, const char* name);
double profile_ms(ProfileClock::time_point start, ProfileClock::time_point end);
bool denoiser_has_preset_f();
double profile_thread_cpu_ms();
bool rr_enabled();
bool aux_array_layout(DXGI_FORMAT format, uint32_t& arrayFormat, uint32_t& channels, AuxLayout& how, uint32_t& texelBytes);
void convert_aux_row(DXGI_FORMAT format, const uint8_t* source, float* destination, UINT width, uint32_t channels);
bool create_buffer(D3D12_HEAP_TYPE heapType, UINT64 size, ID3D12Resource** buffer, uint8_t** mapped);
bool ensure_staging(Staging& staging, ID3D12Resource* resource, D3D12_HEAP_TYPE heapType);
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
void copy_to_staging(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, Staging& staging, D3D12_RESOURCE_STATES state);
CudaTextureDesc input_sampler(UINT width, uint32_t channels, bool linear);
bool ensure_cuda_image(CudaImage& image, UINT width, UINT height, uint32_t format, uint32_t channels, bool surface, bool linear);
int copy_2d(const CudaMemcpy2D& copy);
void copy_rows(RowPool& pool, uint8_t* destination, size_t destinationPitch, const uint8_t* source, size_t sourcePitch, size_t rowBytes, UINT rows);
bool async_copies();
CopyStreams* copy_streams();
void reserve_host(HostPlane& host, size_t bytes);
void release_host(HostPlane& host);
bool ensure_linear(CudaDevicePtr& buffer, size_t& capacity, size_t bytes);
uint32_t plane_format(Plane plane);
uint32_t plane_channels(Plane plane);
bool record_owned_motion_constants(ID3D12GraphicsCommandList* list, Staging& target, ID3D12Resource* motion,uint32_t guidValue,uint64_t expectedBytes);
void record_native_motion_capture(ID3D12GraphicsCommandList* list, InputSlot& slot, ID3D12Resource* motion, uint32_t frame);
void init_vram_interop();
bool vram_interop_available();
bool native_vram_available();
bool create_native_vram_buffer(VramBuffer& target, size_t bytes);
void record_native_inputs(ID3D12GraphicsCommandList* list, InputSlot& slot, ID3D12Resource* const* inputs, const Plane* planes, const D3D12_RESOURCE_STATES* states, int count);
void record_native_output(Feature& feature, unsigned slotIndex, ID3D12GraphicsCommandList* list, ID3D12Resource* output, D3D12_RESOURCE_STATES outputState);
bool create_native_split(Feature& feature);
bool native_split_frame(Feature& feature, ID3D12GraphicsCommandList* list, uint32_t frame, void (*beforeSecond)(void*, ID3D12CommandQueue*) = nullptr, void* beforeOwner = nullptr, ID3D12GraphicsCommandList** back = nullptr);
void release_native_split(Feature& feature);
int device_local_type(uint32_t typeBits);
bool take_pooled_vram_buffer(VramBuffer& target, size_t bytes, uint8_t kind = 0);
bool create_vram_buffer(VramBuffer& target, size_t bytes, bool importNow = true);
void recycle_vram_buffer(VramBuffer& buffer);
void free_vram_pool();
void destroy_vram_image(VramImage& image);
bool create_vram_image(VramImage& image, UINT width, UINT height, VkFormat format);
size_t canonical_texel_bytes(Plane plane);
bool describe_in_place(ID3D12Resource* resource, InPlaceTexture& texture);
bool describe_native_vram(ID3D12Resource* resource, VramCopy& copy, bool output = false);
bool vram_blit_supported(VkFormat source, VkFormat destination);
bool describe_vram_copy(ID3D12Resource* resource, Plane plane, VramCopy& copy, bool output = false);
VkBufferImageCopy full_region(const VramCopy& copy);
bool ensure_fused_color(Feature& feature, int slotIndex, ID3D12Resource* source, const VramCopy& copy, uint32_t frame=0);
bool record_vram_inputs(Feature& feature, ID3D12GraphicsCommandList* list, InputSlot& slot, const VramCopy* copies, int count, bool fusedColor = false);
bool record_feature_vram_output(Feature& feature, unsigned slotIndex, ID3D12GraphicsCommandList* list, ID3D12Resource* output, D3D12_RESOURCE_STATES outputState, const VramCopy& copy, uint32_t frame=0);
void release_ngx_input_wait_locked(Feature& feature, const char* reason, bool disable);
void ngx_input_watchdog();
bool create_split_semaphore(Feature& feature);
bool ensure_vram_buffer(Feature& feature, VramBuffer& buffer, size_t bytes);
bool ensure_conversion_image(Feature& feature, VramImage& image, UINT width, UINT height, VkFormat format);
void record_native_gpu_check(Feature& feature, ID3D12GraphicsCommandList* list, InputSlot& slot, ID3D12Resource* motion);
void release_vram(Feature& feature);
void stage_plane(Plane plane, const Staging& staging, HostPlane& host, FrameTiming* timing, int timingIndex, size_t rowPitch);
void log_motion_stats(const HostPlane& host, const HostPlane& color, uint32_t frame);
bool upload_plane(Plane plane, const HostPlane& host, CudaImage& image, uint32_t frame, FrameTiming* timing, int timingIndex);
void log_output_hash(const HostPlane& host, uint32_t frame);
bool output_dump_path(uint32_t frame, char (&path)[MAX_PATH]);
uint8_t* capture_pinned(const char* path, size_t bytes);
void write_capture_file(uint32_t frame, const char* what, const char* path, const void* data, size_t bytes);
void flush_capture_if_complete(uint32_t frame);
void dump_output(const HostPlane& host, uint32_t frame, const char* path);
bool input_dump_directory(uint32_t frame, char (&directory)[MAX_PATH]);
void dump_input_planes(const HostPlane (&planes)[4], uint32_t frame, const FrameParams& p, const char* directory);
void dump_vram_inputs(const InputSlot& slot, uint32_t frame, const FrameParams& p, int count);
bool download_output(Feature& feature, HostPlane& host, uint32_t frame, FrameTiming* timing);
void write_output(Staging& staging, const HostPlane& host, FrameTiming* timing);
ID3D12Resource* get_resource(void* parameters, const char* name);
void prepare_alpha_vram(Feature& feature, InputSlot& slot, bool vram);
bool capture_alpha_output(Feature& feature, void* parameters, int slotIndex, uint32_t frame);
bool publish_alpha_output(Feature& feature, const InputSlot& slot);
bool eval_sync_enabled();
bool download_alpha_output(Feature& feature, uint32_t frame, const FrameParams& params);
bool record_alpha_copyback(Feature& feature, ID3D12GraphicsCommandList* list, InputSlot& slot, D3D12_RESOURCE_STATES outputState, uint32_t colorFrame, bool split);
void destroy_aux_surface(AuxSurface& surface);
bool gather_aux_inputs(Feature& feature, ID3D12GraphicsCommandList* list, void* parameters, int slotIndex, D3D12_RESOURCE_STATES inputState, uint32_t frame);
void set_evaluation_parameters(Feature& feature, const FrameParams& p);
CudaObject prepare_dilated_motion(Feature& feature, InputSlot& slot, const FrameParams& p);
bool recreate_for_render_size(Feature& feature, FrameParams& params, uint32_t frame);
void prepare_inputs(Feature* feature, int slotIndex, uint32_t frame, FrameParams params, FrameTiming timing);
void drain_pipeline();
bool retain_native(ID3D12GraphicsCommandList* list, IUnknown* resources);
bool engine_k_native_rdna3();
bool engine_k_refused();
void detect_engine_matrices(PFN_vkGetInstanceProcAddr instanceProc, VkInstance instance, VkPhysicalDevice physical);
void init_engine_vulkan();
void init_engine_backend();
void init_engine_backend_ldr();
void prepare_engine_view(Feature& feature, ID3D12Resource* resource, const D3D12_RESOURCE_DESC& desc, d4r::GameImage& image, bool isOutput, bool originSupported, D3D12_RESOURCE_STATES& state);
void init_engine_backend_m_variant(bool presetL, bool ldr);
bool initialize_ngx_gpu_wait(Feature& feature);
bool record_gpu_completion_wait(Feature& feature, ID3D12GraphicsCommandList* list, uint32_t frame);
bool record_ngx_input_ready(Feature& feature, ID3D12GraphicsCommandList* list, int slotIndex, uint32_t frame);
void release_engine_network(Feature& feature);
bool create_engine_network(Feature& feature, const std::string& directory, const char* preset, bool stagesAvailable, unsigned int width, unsigned int height, size_t tokenBytes, size_t headBytes);
void record_engine_input_ready(Feature& feature, VkCommandBuffer command, uint32_t frame);
void release_engine_input_wait(Feature& feature, uint32_t frame, const char* reason, bool disable);
void run_engine_network(Feature* feature, int slotIndex, uint32_t frame);
std::unique_ptr<vkhip::MNetwork> create_engine_vkhip_net(const std::string& directory, uint32_t renderWidth, uint32_t renderHeight, uint64_t bufferBytes);
bool engine_evaluate_m(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color, ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output, const FrameParams& p, uint32_t frame);
bool engine_evaluate_e(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color, ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output, ID3D12Resource* exposure, const FrameParams& p, uint32_t frame);
std::unique_ptr<vkhip::RRStages> create_engine_rr_vkhip(const std::string& directory, d4r::REngine& engine, int slot);
bool engine_evaluate_rr(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color, ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output, void* parameters, const FrameParams& p, uint32_t frame);
std::unique_ptr<vkhip::RRStages> create_engine_rr_vkhip(const std::string& directory, d4r::REngine& engine, int slot);
std::unique_ptr<vkhip::MNetwork> create_engine_vkhip_net(const std::string& directory, uint32_t renderWidth, uint32_t renderHeight, uint64_t bufferBytes);
std::unique_ptr<vkhip::KNetwork> create_engine_vkhip_k(const std::string& directory, uint32_t outWidth, uint32_t outHeight, size_t tokenBytes, size_t headBytes);
bool create_engine_vkhip(Feature& feature, const std::string& directory, size_t tokenBytes, size_t headBytes);
d4r::FrameSettings engine_k_settings(const Feature& feature, const FrameParams& p, bool ldr, bool displayMotion, bool autoExposure, bool first);
void release_engine_dx(Feature& feature);
void engine_dx_mark(EngineDxFeature& state, int slot, ID3D12GraphicsCommandList* list, uint32_t boundary);
bool engine_evaluate(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color, ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output, ID3D12Resource* exposure, void* parameters, const FrameParams& p, uint32_t frame);

// --- NGX types (declared locally: the core's exports differ from the SDK) --
using NgxResult = unsigned int;
inline constexpr NgxResult NGX_SUCCESS = 0x1;
inline constexpr NgxResult NGX_FAIL = 0xBAD00000;
inline constexpr NgxResult NGX_FAIL_FEATURE_NOT_SUPPORTED = NGX_FAIL | 1;
inline constexpr NgxResult NGX_FAIL_PLATFORM_ERROR = NGX_FAIL | 2;
inline constexpr NgxResult NGX_FAIL_INVALID_PARAMETER = NGX_FAIL | 5;
inline constexpr NgxResult NGX_FAIL_NOT_INITIALIZED = NGX_FAIL | 7;
inline constexpr NgxResult NGX_FAIL_UNSUPPORTED_FORMAT = NGX_FAIL | 14;
inline constexpr unsigned int NGX_FEATURE_SUPER_SAMPLING = 1;
// nvngx_dlssd.dll (DLSS-Denoiser / Ray Reconstruction), NVSDK_NGX_Feature.
inline constexpr unsigned int NGX_FEATURE_RAY_RECONSTRUCTION = 13;
struct NgxHandle
{
    unsigned int Id;
};
struct NgxPathListInfo
{
    const wchar_t* const* Path;
    unsigned int Length;
};
struct NgxLoggingInfo
{
    void* LoggingCallback;
    int MinimumLoggingLevel;
    bool DisableOtherLoggingSinks;
};
struct NgxFeatureCommonInfo
{
    NgxPathListInfo PathListInfo;
    void* InternalData;
    NgxLoggingInfo LoggingInfo;
};
struct NgxFeatureRequirement
{
    unsigned int FeatureSupported;
    unsigned int MinHWArchitecture;
    char MinOSVersion[255];
};
// MSVC-ABI parameter accessors (d4r_ngx_param_msvc.cpp, built with clang-cl).
extern "C"
{
    void d4r_ngx_set_ull(void* parameters, const char* name, unsigned long long value);
    void d4r_ngx_set_float(void* parameters, const char* name, float value);
    void d4r_ngx_set_uint(void* parameters, const char* name, unsigned int value);
    void d4r_ngx_set_int(void* parameters, const char* name, int value);
    void d4r_ngx_set_void(void* parameters, const char* name, void* value);
    NgxResult d4r_ngx_get_ull(void* parameters, const char* name, unsigned long long* value);
    NgxResult d4r_ngx_get_float(void* parameters, const char* name, float* value);
    NgxResult d4r_ngx_get_uint(void* parameters, const char* name, unsigned int* value);
    NgxResult d4r_ngx_get_int(void* parameters, const char* name, int* value);
    NgxResult d4r_ngx_get_void(void* parameters, const char* name, void** value);
    NgxResult d4r_ngx_get_d3d12_resource(void* parameters, const char* name, ID3D12Resource** value);
    void d4r_ngx_set_d3d12_resource(void* parameters, const char* name, ID3D12Resource* value);
    void* d4r_ngx_parameters_create();
    void d4r_ngx_parameters_destroy(void* parameters);
}
// --- CUDA driver API (through nvcuda.dll, the Wine bridge to ZLUDA) --------
using CudaDevicePtr = unsigned long long;
using CudaArray = void*;
using CudaObject = unsigned long long;
using CudaEvent = void*;
struct CudaArrayDescriptor
{
    size_t Width;
    size_t Height;
    uint32_t Format;
    uint32_t NumChannels;
};
struct CudaArray3DDescriptor
{
    size_t Width;
    size_t Height;
    size_t Depth;
    uint32_t Format;
    uint32_t NumChannels;
    uint32_t Flags;
};
struct CudaMemcpy2D
{
    size_t srcXInBytes;
    size_t srcY;
    uint32_t srcMemoryType;
    uint32_t srcAlignment;
    const void* srcHost;
    CudaDevicePtr srcDevice;
    CudaArray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    uint32_t dstMemoryType;
    uint32_t dstAlignment;
    void* dstHost;
    CudaDevicePtr dstDevice;
    CudaArray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};
static_assert(sizeof(CudaMemcpy2D) == 128);
struct CudaResourceDesc
{
    uint32_t resType;
    uint32_t alignment;
    union
    {
        struct
        {
            CudaArray hArray;
        } array;
        int reserved[32];
    } res;
    uint32_t flags;
    uint32_t reserved;
};
static_assert(sizeof(CudaResourceDesc) == 144);
struct CudaTextureDesc
{
    uint32_t addressMode[3];
    uint32_t filterMode;
    uint32_t flags;
    uint32_t maxAnisotropy;
    uint32_t mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int32_t reserved[12];
};
static_assert(sizeof(CudaTextureDesc) == 104);
inline constexpr uint32_t CUDA_FORMAT_HALF = 0x10;
inline constexpr uint32_t CUDA_FORMAT_FLOAT = 0x20;
inline constexpr uint32_t CUDA_MEMORY_HOST = 1;
inline constexpr uint32_t CUDA_MEMORY_DEVICE = 2;
inline constexpr uint32_t CUDA_MEMORY_ARRAY = 3;
struct CudaApi
{
    int(WINAPI* memAlloc)(CudaDevicePtr*, size_t) = nullptr;
    int(WINAPI* memFree)(CudaDevicePtr) = nullptr;
    int(WINAPI* arrayCreate)(CudaArray*, const CudaArrayDescriptor*) = nullptr;
    int(WINAPI* array3DCreate)(CudaArray*, const CudaArray3DDescriptor*) = nullptr;
    int(WINAPI* arrayDestroy)(CudaArray) = nullptr;
    int(WINAPI* memcpy2D)(const CudaMemcpy2D*) = nullptr;
    int(WINAPI* texObjectCreate)(CudaObject*, const CudaResourceDesc*, const CudaTextureDesc*, const void*) = nullptr;
    int(WINAPI* texObjectDestroy)(CudaObject) = nullptr;
    int(WINAPI* surfObjectCreate)(CudaObject*, const CudaResourceDesc*) = nullptr;
    int(WINAPI* surfObjectDestroy)(CudaObject) = nullptr;
    int(WINAPI* ctxSynchronize)() = nullptr;
    int(WINAPI* ctxGetDevice)(int*) = nullptr;
    int(WINAPI* eventCreate)(CudaEvent*, unsigned int) = nullptr;
    int(WINAPI* eventRecord)(CudaEvent, void*) = nullptr;
    int(WINAPI* eventQuery)(CudaEvent) = nullptr;
    int(WINAPI* eventSynchronize)(CudaEvent) = nullptr;
    int(WINAPI* eventElapsedTime)(float*, CudaEvent, CudaEvent) = nullptr;
    int(WINAPI* eventDestroy)(CudaEvent) = nullptr;
    // Optional: overlapped staging (see async_copies()).
    int(WINAPI* memAllocHost)(void**, size_t) = nullptr;
    int(WINAPI* memFreeHost)(void*) = nullptr;
    int(WINAPI* memcpyHtoDAsync)(CudaDevicePtr, const void*, size_t, void*) = nullptr;
    int(WINAPI* memcpyDtoHAsync)(void*, CudaDevicePtr, size_t, void*) = nullptr;
    int(WINAPI* streamCreate)(void**, unsigned int) = nullptr;
    int(WINAPI* streamSynchronize)(void*) = nullptr;
    // Optional (d4r nvcuda bridge): device <-> array copies queued on the null stream.
    int(WINAPI* memcpy2DAsync)(const CudaMemcpy2D*, void*) = nullptr;
    // Optional (d4r nvcuda bridge): null-stream wait for a device u32, and a write from another stream.
    int(WINAPI* streamWaitValue32)(CudaDevicePtr, uint32_t) = nullptr;
    int(WINAPI* streamWaitValue32Eq)(CudaDevicePtr, uint32_t) = nullptr;
    int(WINAPI* ngxWriteValue32)(CudaDevicePtr, uint32_t) = nullptr; // D4R_SHIM_GPU_COMPLETION
    int(WINAPI* ngxArmCompletion)(CudaDevicePtr, uint32_t, int) = nullptr; // early word behind the output kernel
    int(WINAPI* ngxCompletionFired)() = nullptr;
    double(WINAPI* threadCpuTimeMs)() = nullptr;
    int(WINAPI* writeValue32)(CudaDevicePtr, uint32_t) = nullptr;
    // Optional (d4r nvcuda bridge): surfaces of an array store to linear memory instead (d4r native kernels).
    int(WINAPI* setArrayRedirect)(CudaArray, CudaDevicePtr, uint32_t) = nullptr;
    int(WINAPI* outputKernelNative)() = nullptr;
    int(WINAPI* outputKernelRgba8)() = nullptr;
    int(WINAPI* timelineCreate)(VkDevice, uint64_t, void**) = nullptr;
    int(WINAPI* timelinePrepare)(void*) = nullptr;
    void(WINAPI* timelineCancel)(void*) = nullptr;
    int(WINAPI* timelineQueue)(void*, uint64_t) = nullptr;
    int(WINAPI* timelineSignal)(void*, uint64_t) = nullptr;
    void(WINAPI* timelineDestroy)(void*) = nullptr;
    int(WINAPI* timelineWait)(void*) = nullptr;
    // Optional (d4r nvcuda bridge): report a pitch-linear texture to NGX as an array of that size/format.
    int(WINAPI* registerLinearTexture)(CudaObject, size_t, size_t, uint32_t, uint32_t) = nullptr;
    int(WINAPI* dilateMotion)(const D4rMotionDilationParams*) = nullptr;
    int(WINAPI* dilateMotionPadded)(const D4rMotionDilationParams*, uint32_t) = nullptr;
    int(WINAPI* dilateMotionThin)(const D4rMotionDilationParams*, uint32_t) = nullptr;
    int(WINAPI* composeNativeMotion)(const D4rNativeMotionParams*) = nullptr;
    int(WINAPI* composeNativeMotionThin)(const D4rNativeMotionThinParams*) = nullptr;
    int(WINAPI* composeNativeMotionAligned)(const D4rNativeMotionThinParams*) = nullptr;
    int(WINAPI* composeNativeMotionHybrid)(const D4rNativeMotionHybridParams*) = nullptr;
    int(WINAPI* composeNativeMotionPrepared)(const D4rNativeMotionPreparedParams*) = nullptr;
    int(WINAPI* stabilizeThinWiresPrepared)(const D4rWireHistoryPreparedParams*) = nullptr;
    int(WINAPI* stabilizeThinWires)(const D4rWireHistoryParams*) = nullptr;
    int(WINAPI* stabilizeThinWiresSparse)(const D4rWireHistorySparseParams*) = nullptr;
};
// --- official NGX core CUDA API --------------------------------------------
struct CoreApi
{
    NgxResult (*init)(unsigned long long, const wchar_t*, const NgxFeatureCommonInfo*, unsigned int) = nullptr;
    NgxResult (*initProjectId)(const char*, int, const char*, const wchar_t*, unsigned int,
                               const NgxFeatureCommonInfo*) = nullptr;
    NgxResult (*shutdown)() = nullptr;
    NgxResult (*getParameters)(void**) = nullptr;
    NgxResult (*allocateParameters)(void**) = nullptr;
    NgxResult (*getCapabilityParameters)(void**) = nullptr;
    NgxResult (*destroyParameters)(void*) = nullptr;
    NgxResult (*createFeature)(unsigned int, void*, NgxHandle**) = nullptr;
    NgxResult (*releaseFeature)(NgxHandle*) = nullptr;
    NgxResult (*evaluateFeature)(const NgxHandle*, void*, void*) = nullptr;
    // The core's CUDA entry point takes a CUdevice ordinal, not an IDXGIAdapter* (its
    // disassembly rejects a negative value in the first argument, which is the ordinal
    // check). The D3D12 export this file implements takes an adapter and must translate.
};
// --- DLSS-Denoiser (Ray Reconstruction) ----------------------------------------
//
// nvngx_dlssd.dll carries the Ray Reconstruction models, and the NGX core loads and
// dispatches it itself: the core is the only thing that knows the denoiser's module name,
// its kernel map and its lifetime. So both features are created, evaluated and released
// through g.ngx, with feature id 13 for Ray Reconstruction, and this file's job is only to
// answer honestly whether that route exists.
//
// Existence is decided by loading the denoiser and looking for its CUDA entry points, the
// only reliable external evidence - builds genuinely differ. The 3.10.7 build exports the
// whole NVSDK_NGX_CUDA_* set; the DLAA-only build exports D3D11/D3D12/Vulkan/DirectSR and
// no CUDA entry points, and cannot serve Ray Reconstruction however it is loaded.
//
// The pointers here are resolved only as that evidence and are never called. In particular
// NVSDK_NGX_CUDA_Init in the 310.7 build is `mov eax, 0xBAD00001; ret` (RVA 0x22b80), a stub
// that only ever reports "not supported"; the core's own init is what brings the denoiser up.
// A single yes/no, not a table of entry points: nothing here is ever called. The denoiser is
// loaded by the NGX core, which is the only thing that may drive it.
struct DenoiserProbe
{
    bool cudaCapable = false;
};
inline HMODULE g_selfModule = nullptr;
// --- portable install ----------------------------------------------------------
// The drag-in release keeps this DLL (as nvngx.dll, explicitly selected by OptiScaler's NvngxPath),
// the nvcuda bridge, ZLUDA and the native kernels in an d4r folder next to the game's executable,
// with that game's settings in d4r\d4r.ini. The file's settings become the environment variables the
// developer launcher sets (config/d4r.ini.default documents them), and a variable that is already set,
// for example in the game's launch options, wins. Settings for the Linux side (ZLUDA, the bridge) are
// handed to the bridge right after it is loaded, before its first CUDA call. Without an d4r.ini next
// to the DLL nothing changes.
struct PortableInstall
{
    bool active = false;
    bool nativeMotionVectors = false;
    std::wstring dir;                                         // the d4r folder
    std::string unixDir;                                      // the same folder as a Linux path
    std::vector<std::pair<std::string, std::string>> unixEnv; // for the bridge's d4rSetEnv
    bool nativeWindows = false; // not under Wine: Windows paths for ZLUDA, no ROCm folder
    std::vector<std::string> notes;                           // logged once the log is open
};
inline PortableInstall g_portable;
inline std::once_flag g_portableOnce;
inline bool g_logFresh = false; // a portable install starts a new log for every launch;
struct IniEntry
{
    std::string section, key, value;
};
using ProfileClock = std::chrono::steady_clock;
// --- CUDA worker thread --------------------------------------------------------
// Every NGX-core and CUDA call runs on this thread so the ZLUDA context that
// the nvcuda bridge creates on first use stays current.
class Worker
{
public:
    void start()
    {
        // Detached: the process may exit without shutting NGX down.
        std::thread([this] { run(); }).detach();
    }

    void post(std::function<void()> job)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        condition_.notify_one();
    }

    size_t pending()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return jobs_.size();
    }

    template <typename Function> auto call(Function function, const char* label = nullptr) -> decltype(function())
    {
        using Result = decltype(function());
        auto task = std::make_shared<std::packaged_task<Result()>>(std::move(function));
        auto future = task->get_future();
        post([task] { (*task)(); });
        const auto started = std::chrono::steady_clock::now();
        while (label && future.wait_for(std::chrono::seconds(15)) == std::future_status::timeout)
            logf("worker wait: %s still pending after %.0f s (resources retained)", label,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        return future.get();
    }

private:
    void run()
    {
        for (;;)
        {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return !jobs_.empty(); });
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::function<void()>> jobs_;
};
// --- half/float and packed format conversions --------------------------------
inline uint16_t float_to_half(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xff)
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    int halfExponent = static_cast<int>(exponent) - 112;
    if (halfExponent >= 31)
        return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t shift = 13;
    if (halfExponent <= 0)
    {
        if (halfExponent < -10)
            return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        shift = static_cast<uint32_t>(14 - halfExponent);
        halfExponent = 0;
    }
    const uint32_t halfMantissa = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    uint32_t result = sign | (static_cast<uint32_t>(halfExponent) << 10);
    result += halfMantissa & (halfExponent == 0 ? 0x7ffu : 0x3ffu);
    if (remainder > halfway || (remainder == halfway && (halfMantissa & 1u)))
        ++result;
    return static_cast<uint16_t>(result);
}
inline float half_to_float(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t result;
    if (exponent == 0)
    {
        float magnitude = static_cast<float>(mantissa) * (1.0f / 16777216.0f);
        std::memcpy(&result, &magnitude, sizeof(result));
        result |= sign;
    }
    else if (exponent == 0x1f)
        result = sign | 0x7f800000u | (mantissa << 13);
    else
        result = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}
// Unsigned small floats (R11G11B10): 5-bit exponent, 6 or 5 bit mantissa.
inline float small_float_to_float(uint32_t bits, int mantissaBits)
{
    const uint32_t exponent = (bits >> mantissaBits) & 0x1fu;
    const uint32_t mantissa = bits & ((1u << mantissaBits) - 1u);
    if (exponent == 0)
        return std::ldexp(static_cast<float>(mantissa), -14 - mantissaBits);
    if (exponent == 0x1f)
        return mantissa ? NAN : INFINITY;
    return std::ldexp(1.0f + static_cast<float>(mantissa) / static_cast<float>(1u << mantissaBits),
                      static_cast<int>(exponent) - 15);
}
inline uint32_t float_to_small_float(float value, int mantissaBits)
{
    if (!(value > 0.0f))
        return 0; // negatives and NaN clamp to zero
    const float maximum = std::ldexp(2.0f - std::ldexp(1.0f, -mantissaBits), 15);
    if (value >= maximum)
        value = maximum;
    int exponent;
    const float fraction = std::frexp(value, &exponent); // value = fraction * 2^exponent
    int biased = exponent - 1 + 15;
    if (biased <= 0)
    {
        const uint32_t mantissa = static_cast<uint32_t>(std::lround(std::ldexp(value, 14 + mantissaBits)));
        return mantissa;
    }
    uint32_t mantissa = static_cast<uint32_t>(std::lround((fraction * 2.0f - 1.0f) * static_cast<float>(1u << mantissaBits)));
    if (mantissa >= (1u << mantissaBits))
    {
        mantissa = 0;
        ++biased;
    }
    if (biased >= 31)
        return (30u << mantissaBits) | ((1u << mantissaBits) - 1u);
    return (static_cast<uint32_t>(biased) << mantissaBits) | mantissa;
}
// Canonical CUDA-side layouts: color RGBA16F, depth R32F, motion RG16F,
// exposure R32F, output RGBA16F.
enum class Plane
{
    Color,
    Depth,
    Motion,
    Exposure
};
inline bool supported_input(Plane plane, DXGI_FORMAT format)
{
    switch (plane)
    {
    case Plane::Color:
        return format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R16G16B16A16_TYPELESS ||
               format == DXGI_FORMAT_R32G32B32A32_FLOAT || format == DXGI_FORMAT_R11G11B10_FLOAT ||
               format == DXGI_FORMAT_R10G10B10A2_UNORM || format == DXGI_FORMAT_R10G10B10A2_TYPELESS ||
               format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ||
               format == DXGI_FORMAT_R8G8B8A8_TYPELESS || format == DXGI_FORMAT_B8G8R8A8_UNORM ||
               format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB || format == DXGI_FORMAT_B8G8R8A8_TYPELESS ||
               format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    case Plane::Depth:
        return format == DXGI_FORMAT_D32_FLOAT || format == DXGI_FORMAT_R32_FLOAT ||
               format == DXGI_FORMAT_R32_TYPELESS || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ||
               format == DXGI_FORMAT_R32G8X24_TYPELESS || format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS ||
               format == DXGI_FORMAT_D24_UNORM_S8_UINT || format == DXGI_FORMAT_R24G8_TYPELESS ||
               format == DXGI_FORMAT_R24_UNORM_X8_TYPELESS || format == DXGI_FORMAT_D16_UNORM ||
               format == DXGI_FORMAT_R16_UNORM || format == DXGI_FORMAT_R16_TYPELESS ||
               format == DXGI_FORMAT_R16_FLOAT;
    case Plane::Motion:
        return format == DXGI_FORMAT_R16G16_FLOAT || format == DXGI_FORMAT_R16G16_TYPELESS ||
               format == DXGI_FORMAT_R32G32_FLOAT || format == DXGI_FORMAT_R32G32_TYPELESS ||
               format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R16G16B16A16_TYPELESS ||
               format == DXGI_FORMAT_R32G32B32A32_FLOAT || format == DXGI_FORMAT_R8G8B8A8_UNORM;
    case Plane::Exposure: // R8G8B8A8_UNORM: Ghost of Tsushima (read normalized, as a sampled texture)
        return format == DXGI_FORMAT_R32_FLOAT || format == DXGI_FORMAT_R32_TYPELESS ||
               format == DXGI_FORMAT_R16_FLOAT || format == DXGI_FORMAT_R16_TYPELESS ||
               format == DXGI_FORMAT_R32G32_FLOAT || format == DXGI_FORMAT_R32G32_TYPELESS ||
               format == DXGI_FORMAT_R16G16_FLOAT || format == DXGI_FORMAT_R16G16_TYPELESS ||
               format == DXGI_FORMAT_R32G32B32A32_FLOAT || format == DXGI_FORMAT_R16G16B16A16_FLOAT ||
               format == DXGI_FORMAT_R8G8B8A8_UNORM;
    }
    return false;
}
inline bool supported_output(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R16G16B16A16_TYPELESS ||
           format == DXGI_FORMAT_R32G32B32A32_FLOAT || format == DXGI_FORMAT_R11G11B10_FLOAT ||
           format == DXGI_FORMAT_R10G10B10A2_UNORM || format == DXGI_FORMAT_R10G10B10A2_TYPELESS ||
           format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_R8G8B8A8_TYPELESS ||
           format == DXGI_FORMAT_B8G8R8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_TYPELESS ||
           format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
}
inline float unorm(uint32_t value, int bits)
{
    return static_cast<float>(value) / static_cast<float>((1u << bits) - 1u);
}
inline uint32_t to_unorm(float value, int bits)
{
    const float clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    return static_cast<uint32_t>(std::lround(clamped * static_cast<float>((1u << bits) - 1u)));
}
// Converts one row of a D3D12 texel format into the canonical layout.
inline void convert_row_in(Plane plane, DXGI_FORMAT format, const uint8_t* source, void* destination, UINT width)
{
    uint16_t* half = static_cast<uint16_t*>(destination);
    float* single = static_cast<float*>(destination);
    for (UINT x = 0; x < width; ++x)
    {
        switch (plane)
        {
        case Plane::Color:
        {
            float rgba[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            switch (format)
            {
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
            case DXGI_FORMAT_R16G16B16A16_TYPELESS:
                std::memcpy(half + x * 4, source + x * 8, 8);
                continue;
            case DXGI_FORMAT_R32G32B32A32_FLOAT:
                std::memcpy(rgba, source + x * 16, 16);
                break;
            case DXGI_FORMAT_R11G11B10_FLOAT:
            {
                uint32_t packed;
                std::memcpy(&packed, source + x * 4, 4);
                rgba[0] = small_float_to_float(packed & 0x7ffu, 6);
                rgba[1] = small_float_to_float((packed >> 11) & 0x7ffu, 6);
                rgba[2] = small_float_to_float((packed >> 22) & 0x3ffu, 5);
                break;
            }
            case DXGI_FORMAT_R10G10B10A2_UNORM:
            case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            {
                uint32_t packed;
                std::memcpy(&packed, source + x * 4, 4);
                rgba[0] = unorm(packed & 0x3ffu, 10);
                rgba[1] = unorm((packed >> 10) & 0x3ffu, 10);
                rgba[2] = unorm((packed >> 20) & 0x3ffu, 10);
                rgba[3] = unorm(packed >> 30, 2);
                break;
            }
            case DXGI_FORMAT_B8G8R8A8_UNORM:
            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            case DXGI_FORMAT_B8G8R8A8_TYPELESS:
                rgba[0] = unorm(source[x * 4 + 2], 8);
                rgba[1] = unorm(source[x * 4 + 1], 8);
                rgba[2] = unorm(source[x * 4 + 0], 8);
                rgba[3] = unorm(source[x * 4 + 3], 8);
                break;
            case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
            {
                // Microsoft spec: c = (p >> offset) & 0x1FF, E = (p >> 27) & 0x1F,
                // v = c * 2^(E - 24) for every E: there is no implicit leading bit.
                uint32_t packed;
                std::memcpy(&packed, source + x * 4, 4);
                uint32_t e = (packed >> 27) & 0x1Fu;
                float scale = std::ldexp(1.0f, (int)e - 24);
                rgba[0] = static_cast<float>((packed >> 0) & 0x1FFu) * scale;
                rgba[1] = static_cast<float>((packed >> 9) & 0x1FFu) * scale;
                rgba[2] = static_cast<float>((packed >> 18) & 0x1FFu) * scale;
                rgba[3] = 1.0f;
                break;
            }
            default: // RGBA8 variants
                for (int c = 0; c < 4; ++c)
                    rgba[c] = unorm(source[x * 4 + c], 8);
                break;
            }
            for (int c = 0; c < 4; ++c)
                half[x * 4 + c] = float_to_half(rgba[c]);
            break;
        }
        case Plane::Depth:
        {
            switch (format)
            {
            case DXGI_FORMAT_D24_UNORM_S8_UINT:
            case DXGI_FORMAT_R24G8_TYPELESS:
            case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
            {
                uint32_t packed;
                std::memcpy(&packed, source + x * 4, 4);
                single[x] = unorm(packed & 0xffffffu, 24);
                break;
            }
            case DXGI_FORMAT_D16_UNORM:
            case DXGI_FORMAT_R16_UNORM:
            case DXGI_FORMAT_R16_TYPELESS:
            {
                uint16_t value;
                std::memcpy(&value, source + x * 2, 2);
                single[x] = unorm(value, 16);
                break;
            }
            case DXGI_FORMAT_R16_FLOAT:
            {
                uint16_t value;
                std::memcpy(&value, source + x * 2, 2);
                single[x] = half_to_float(value);
                break;
            }
            default: // 32-bit float depth (the depth plane of D32S8 copies as 4 bytes)
                std::memcpy(single + x, source + x * 4, 4);
                break;
            }
            break;
        }
        case Plane::Motion:
            switch (format)
            {
            case DXGI_FORMAT_R16G16_FLOAT:
            case DXGI_FORMAT_R16G16_TYPELESS:
                std::memcpy(half + x * 2, source + x * 4, 4);
                break;
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
            case DXGI_FORMAT_R16G16B16A16_TYPELESS:
                std::memcpy(half + x * 2, source + x * 8, 4);
                break;
            case DXGI_FORMAT_R32G32B32A32_FLOAT:
            case DXGI_FORMAT_R32G32_FLOAT:
            case DXGI_FORMAT_R32G32_TYPELESS:
            {
                const size_t stride = format == DXGI_FORMAT_R32G32B32A32_FLOAT ? 16 : 8;
                float mv[2];
                std::memcpy(mv, source + x * stride, 8);
                half[x * 2] = float_to_half(mv[0]);
                half[x * 2 + 1] = float_to_half(mv[1]);
                break;
            }
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            {
                const uint8_t* rgba = source + x * 4;
                half[x * 2] = float_to_half(static_cast<float>(rgba[0]) / 255.0f);
                half[x * 2 + 1] = float_to_half(static_cast<float>(rgba[1]) / 255.0f);
                break;
            }
            default:
                break;
            }
            break;
        case Plane::Exposure:
            switch (format)
            {
            case DXGI_FORMAT_R16G16_FLOAT:
            case DXGI_FORMAT_R16G16_TYPELESS:
            {
                uint16_t value;
                std::memcpy(&value, source + x * 4, 2);
                single[x] = half_to_float(value);
                break;
            }
            case DXGI_FORMAT_R32G32_FLOAT:
            case DXGI_FORMAT_R32G32_TYPELESS:
                // Exposure is the first component; skip the second channel of each texel.
                std::memcpy(single + x, source + x * 8, 4);
                break;
            case DXGI_FORMAT_R16_FLOAT:
            case DXGI_FORMAT_R16_TYPELESS:
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
            {
                uint16_t value;
                std::memcpy(&value, source + x * (format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 2), 2);
                single[x] = half_to_float(value);
                break;
            }
            case DXGI_FORMAT_R8G8B8A8_UNORM:
                single[x] = static_cast<float>(source[x * 4]) / 255.0f;
                break;
            default:
                std::memcpy(single + x, source + x * (format == DXGI_FORMAT_R32G32B32A32_FLOAT ? 16 : 4), 4);
                break;
            }
            break;
        }
    }
}
// Formats whose texels are bit-identical to the canonical CUDA layout (the
// cases convert_row_in handles as a plain copy), so the mapped staging rows can
// be copied straight into the CUDA array with their D3D12 row pitch.
inline bool canonical_input(Plane plane, DXGI_FORMAT format)
{
    switch (plane)
    {
    case Plane::Color:
        return format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R16G16B16A16_TYPELESS;
    case Plane::Depth:
        return format == DXGI_FORMAT_D32_FLOAT || format == DXGI_FORMAT_R32_FLOAT ||
               format == DXGI_FORMAT_R32_TYPELESS || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ||
               format == DXGI_FORMAT_R32G8X24_TYPELESS || format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case Plane::Motion:
        return format == DXGI_FORMAT_R16G16_FLOAT || format == DXGI_FORMAT_R16G16_TYPELESS;
    case Plane::Exposure:
        return format == DXGI_FORMAT_R32_FLOAT || format == DXGI_FORMAT_R32_TYPELESS;
    }
    return false;
}
// R11G11B10 floats share binary16's 5-bit exponent and bias, so each channel
// widens to half exactly by shifting its mantissa; NaN becomes the quiet NaN
// float_to_half produces. Matches convert_row_in bit for bit, without libm.
inline uint16_t small_float_to_half(uint32_t bits, int mantissaBits)
{
    const uint32_t exponent = (bits >> mantissaBits) & 0x1fu;
    const uint32_t mantissa = bits & ((1u << mantissaBits) - 1u);
    if (exponent == 0x1f && mantissa != 0)
        return 0x7e00u;
    return static_cast<uint16_t>((exponent << 10) | (mantissa << (10 - mantissaBits)));
}
inline void convert_r11g11b10_row(const uint8_t* source, uint16_t* half, UINT width)
{
    for (UINT x = 0; x < width; ++x)
    {
        uint32_t packed;
        std::memcpy(&packed, source + x * 4, 4);
        half[x * 4 + 0] = small_float_to_half(packed & 0x7ffu, 6);
        half[x * 4 + 1] = small_float_to_half((packed >> 11) & 0x7ffu, 6);
        half[x * 4 + 2] = small_float_to_half((packed >> 22) & 0x3ffu, 5);
        half[x * 4 + 3] = 0x3c00u; // 1.0
    }
}
// Packs one RGBA16F DLSS texel into the 9-9-9-5 shared-exponent layout.
// Microsoft spec: v = c * 2^(E - 24), c in [0, 511]. The exponent is chosen as
// the smallest E such that all channels fit in 9 bits, then each channel is
// rounded to the nearest representable value (round-half-even, matching
// D3D's conversion semantics). Values above the format's max (65408.0) are
// clamped, like D3D12 does.
inline void rgbe_pack(const float rgba[4], uint8_t* destination)
{
    const float MAX_VALUE = 65408.0f; // 511 * 2^(31 - 24)
    float r = rgba[0] < 0.0f ? 0.0f : (rgba[0] > MAX_VALUE ? MAX_VALUE : rgba[0]);
    float g = rgba[1] < 0.0f ? 0.0f : (rgba[1] > MAX_VALUE ? MAX_VALUE : rgba[1]);
    float b = rgba[2] < 0.0f ? 0.0f : (rgba[2] > MAX_VALUE ? MAX_VALUE : rgba[2]);

    float mx = fmaxf(fmaxf(r, g), b);
    // Smallest scale (i.e. smallest e) such that mx / scale rounds to at most 511. This puts
    // the largest channel in the 9-bit range [256, 511] (lower only at e == 0,
    // which also encodes zero), the maximum precision the format allows.
    int e = 31;
    for (int et = 0; et <= 31; ++et)
    {
        if (mx / std::ldexp(1.0f, et - 24) < 511.5f) { e = et; break; }
    }
    const float scale = std::ldexp(1.0f, e - 24);
    const float inv = 1.0f / scale;

    // Round to nearest, ties to even (D3D round-to-nearest-even semantics).
    auto to_c = [](float v, float inv) -> uint32_t
    {
        double d = std::nearbyint(v * inv); // round-half-even
        if (d < 0.0) d = 0.0;
        if (d > 511.0) d = 511.0;
        return static_cast<uint32_t>(d);
    };

    uint32_t packed = 0;
    packed |= to_c(r, inv) << 0;
    packed |= to_c(g, inv) << 9;
    packed |= to_c(b, inv) << 18;
    packed |= (e & 0x1Fu) << 27;
    std::memcpy(destination, &packed, 4);
}
// Converts one row of RGBA16F DLSS output into the game's output format.
inline void convert_row_out(DXGI_FORMAT format, const uint16_t* source, uint8_t* destination, UINT width)
{
    for (UINT x = 0; x < width; ++x)
    {
        const uint16_t* texel = source + x * 4;
        switch (format)
        {
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            std::memcpy(destination + x * 8, texel, 8);
            break;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            for (int c = 0; c < 4; ++c)
            {
                const float value = half_to_float(texel[c]);
                std::memcpy(destination + x * 16 + c * 4, &value, 4);
            }
            break;
        case DXGI_FORMAT_R11G11B10_FLOAT:
        {
            const uint32_t packed = float_to_small_float(half_to_float(texel[0]), 6) |
                                    (float_to_small_float(half_to_float(texel[1]), 6) << 11) |
                                    (float_to_small_float(half_to_float(texel[2]), 5) << 22);
            std::memcpy(destination + x * 4, &packed, 4);
            break;
        }
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        {
            const uint32_t packed = to_unorm(half_to_float(texel[0]), 10) |
                                    (to_unorm(half_to_float(texel[1]), 10) << 10) |
                                    (to_unorm(half_to_float(texel[2]), 10) << 20) |
                                    (to_unorm(half_to_float(texel[3]), 2) << 30);
            std::memcpy(destination + x * 4, &packed, 4);
            break;
        }
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            destination[x * 4 + 0] = static_cast<uint8_t>(to_unorm(half_to_float(texel[2]), 8));
            destination[x * 4 + 1] = static_cast<uint8_t>(to_unorm(half_to_float(texel[1]), 8));
            destination[x * 4 + 2] = static_cast<uint8_t>(to_unorm(half_to_float(texel[0]), 8));
            destination[x * 4 + 3] = static_cast<uint8_t>(to_unorm(half_to_float(texel[3]), 8));
            break;
        case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
        {
            float rgba[4] = {half_to_float(texel[0]), half_to_float(texel[1]), half_to_float(texel[2]), 1.0f};
            rgbe_pack(rgba, destination + x * 4);
            break;
        }
        default: // RGBA8
            for (int c = 0; c < 4; ++c)
                destination[x * 4 + c] = static_cast<uint8_t>(to_unorm(half_to_float(texel[c]), 8));
            break;
        }
    }
}
// --- global state ----------------------------------------------------------------
struct Global
{
    std::mutex mutex;
    bool started = false;
    Worker worker;        // every NGX-core and CUDA call
    std::thread::id workerThread;
    Worker prep, finish;  // CPU-only pipeline stages around it
    Worker cleanup;       // deferred GPU resource destruction, never under allocator locks
    std::atomic<unsigned int> resourceOwners{0};
    bool shutdownPending = false; // under mutex
    HMODULE core = nullptr;
    CoreApi ngx;
    HMODULE cuda = nullptr;
    CudaApi cu;
    bool ngxInitialized = false;
    bool nativeD3d12Only = false;
    // the native Vulkan frontend (d4r_vulkan_frontend.inl) initialised NGX: its Ray Reconstruction runs on the engine only
    bool nativeVulkan = false;
    ID3D12Device* device = nullptr;
    std::vector<std::wstring> paths;
    std::vector<const wchar_t*> pathPointers;
    NgxFeatureCommonInfo featureInfo = {};
    std::wstring dataPath;
    std::atomic<unsigned int> nextHandleId{0x7200};
    // Ray Reconstruction: nvngx_dlssd.dll, loaded and initialised only when asked for.
    HMODULE denoiserModule = nullptr;
    DenoiserProbe denoiser;
    bool denoiserAttempted = false;
    std::wstring denoiserPath;
    // nvngx_dlssd.dll's file version (major, minor, build, revision); all zero when it has no version resource
    unsigned denoiserVersion[4] = {};
};
inline Global g;
template <typename Function> static bool load_export(HMODULE module, const char* name, Function& function)
{
    function = reinterpret_cast<Function>(reinterpret_cast<void*>(GetProcAddress(module, name)));
    if (function == nullptr)
        logf("missing export %s", name);
    return function != nullptr;
}
// What the NGX core last said about the denoiser it loaded. The core brings the denoiser up
// during its own init and reports the outcome here; a file merely existing on disk proves
// nothing, because the core can come up perfectly well for Super Sampling with a denoiser
// that failed to initialise.
struct DenoiserCapability
{
    bool known = false;
    int available = 0;
    int initResult = NGX_FAIL_NOT_INITIALIZED;
};
// Whether a Ray Reconstruction feature can be created right now: the switch is on, a
// CUDA-capable denoiser build is present, and the core itself reported that denoiser
// initialised. Anything less is reported as unavailable rather than discovered at
// CreateFeature, where a game has already committed to the mode.
// Asks the core what it made of the denoiser. Must run on the worker, like every other core
// call, and must not be called from inside one.
void refresh_denoiser_capability();
// [RayReconstruction] Backend (D4R_RR_BACKEND). Each backend has its own readiness: FSR Ray
// Regeneration needs AMD's DLL, a provider for the game's device and Super Sampling, not
// nvngx_dlssd.dll; DLSS Ray Reconstruction needs the CUDA denoiser. See d4r_fsr_rr_backend.inl.
enum class RrBackend
{
    Dlss,
    Fsr,
    Invalid
};
RrBackend rr_backend();
const char* rr_backend_name(RrBackend backend);
bool fsr_rr_ready();
bool fsr_rr_present();
const char* fsr_rr_denial_reason();
void fsr_rr_prepare_runtime();
d4r::dx::RModel* native_engine_rr(unsigned preset);
std::string rr_model_directory(int slot);
bool engine_requested();
// The render GPU cannot run the engine's shaders (initialize: older than RDNA 3 on native Windows): the CUDA route
// runs as with [Engine] Enabled = false
inline std::atomic<bool> g_engineGpuOff{false};
// A game that initialises NGX with its project ID (engine integrations such as Unreal's) is identified
// to the NGX core the same way.
struct ProjectIdentity
{
    std::string id;
    int engineType = 0;
    std::string engineVersion;
};
void start_engine_preload();
bool native_engine_available(bool ldr);
enum class AmdGeneration { Unknown, Older, Rdna3, Rdna4 }; // ordered: newer compares greater
AmdGeneration amd_render_generation(ID3D12Device* renderDevice);
d4r::dx::MModel* native_engine_m(bool ldr, bool presetL = false);
d4r::dx::rdna3::MModel* native_engine_m_rdna3(bool ldr, bool presetL);
// --- per-feature state ---------------------------------------------------------
inline constexpr int kSlots = 3;
// One more result slot than input slots: a slot is reused only once the GPU
// is past every command list that copies from it (see claim_output_slot).
inline constexpr int kOutputSlots = kSlots + 1;
// A VkBuffer in device-local memory on vkd3d-proton's VkDevice, exported to
// the CUDA side (see "VRAM interop").
struct VramBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    CudaDevicePtr device = 0;
    void* external = nullptr;
    size_t bytes = 0;
    void* mapped = nullptr;       // engine readiness word: CPU-coherent, kept until HIP and Vulkan retire
    // What a pooled buffer can be reused as: 0 the interop transfer buffers (create_vram_buffer), 1 and 2 the
    // engine's device-local and host-coherent storage buffers (create_engine_shared_buffer).
    uint8_t kind = 0;
    ID3D12Resource* d3d = nullptr; // native D3D12 VRAM (no vkd3d-proton): the shared buffer HIP imported
};
// Per-slot descriptors and their source texture live until recorded allocators retire.
struct FusedColorInput {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    D4rVkStagingSet set;
    ID3D12Resource* owner = nullptr;
    uint32_t inputMode = 5;
};
// A temporary image for colour or exposure format conversion.
struct VramImage
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    UINT width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};
struct Staging
{
    ID3D12Resource* buffer = nullptr;
    uint8_t* mapped = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
    UINT rows = 0;
    UINT64 total = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT width = 0;
    UINT height = 0;

    std::vector<ID3D12Resource*> retired;

    void retire()
    {
        if (buffer != nullptr)
            retired.push_back(buffer);
        buffer = nullptr;
        mapped = nullptr;
        width = height = 0;
        format = DXGI_FORMAT_UNKNOWN;
    }

    void release()
    {
        if (buffer != nullptr)
            buffer->Release();
        for (ID3D12Resource* old : retired)
            old->Release();
        *this = Staging{};
    }
};
struct CudaImage
{
    CudaArray array = nullptr;
    CudaObject object = 0; // texture or surface object
    uint32_t format = 0, channels = 0; // the layout the array carries, so a format change
                                      // recreates it rather than only reacting to a resize
    UINT width = 0;
    UINT height = 0;
};
// --- Ray Reconstruction auxiliary inputs --------------------------------------
//
// Super Sampling needs four surfaces (colour, depth, motion vectors, exposure) and the
// file above already moves those. Ray Reconstruction additionally runs a denoiser network
// over the ray-traced G-buffer, so a game that enables it registers a further set of
// surfaces: normals, roughness, diffuse and specular albedo, emissive, the ray hit
// distances and directions, and the effect guides for particles, transparency, fog and
// depth of field.
//
// Those surfaces are read in whatever format the game renders them in, and the denoiser
// samples them through texture objects minted by this file. Rather than force them
// through the canonical planes above - which would re-encode a UNORM normal as half and
// silently change the values the network sees - each is copied with its own format and
// channel count, so the texture object describes exactly what the game drew.

// CUarray_format values beyond the two above (nvsdk: CU_AD_FORMAT_*).
inline constexpr uint32_t CUDA_FORMAT_UNSIGNED_INT8 = 0x01;
inline constexpr uint32_t CUDA_FORMAT_SIGNED_INT8 = 0x08;
inline constexpr uint32_t CUDA_FORMAT_UNSIGNED_INT16 = 0x2;
inline constexpr uint32_t CUDA_FORMAT_SIGNED_INT16 = 0x09;
inline constexpr uint32_t CUDA_FORMAT_UNSIGNED_INT32 = 0x03;
inline constexpr uint32_t CUDA_FORMAT_SIGNED_INT32 = 0x0a;
// How one denoiser input reaches the GPU.
enum class AuxLayout
{
    Verbatim, // the bytes are already exactly what the CUDA array must hold
    Converted // the bytes must be unpacked into floats first (see convert_aux_row)
};
// The texel size an array of `arrayFormat` and `channels` occupies per texel.
size_t aux_texel_bytes(uint32_t arrayFormat, uint32_t channels);
// One auxiliary surface: the name to ask the caller for, and the name the denoiser reads
// it under. They differ for the two albedos, which the SDK and the shipped library spell
// differently, in which case `source` names the caller's parameter and `publish` the
// denoiser's.
struct AuxInput
{
    const char* source;
    const char* publish;
};
// Every auxiliary read name nvngx_dlssd.dll looks up, verified against the names in the
// shipped library and the SDK's nvsdk_ngx_defs_dlssd.h. A caller may bind several of these
// to one texture, so surfaces are keyed on the resource and carry every name bound to it.
//
// The publish column is the name the denoiser reads, taken from the shipped library's own
// lookup table (its .rdata strings, each referenced exactly once from
// NGXDLSSD::EvaluateFeature). It is not always the SDK's spelling: nvngx_dlssd.dll declares
// DLSS.Input.Normals.Subrect.Base.X/Y and DLSS.Input.Roughness.Subrect.Base.X/Y, but it reads
// the normals and roughness textures themselves as GBuffer.Normals and GBuffer.Roughness. It
// does read the two albedos as DLSS.Input.DiffuseAlbedo and DLSS.Input.SpecularAlbedo, which a
// caller may instead supply as GBuffer.DiffuseAlbedo / GBuffer.SpecularAlbedo - the SDK defines
// both spellings, so both are accepted and both publish the name the denoiser asks for. The
// subrect origins in kAuxSubrectNames are a separate matter and keep the SDK's own names.
inline const AuxInput kAuxInputs[] = {
    // Ray-traced G-buffer
    {"GBuffer.Normals", "GBuffer.Normals"},
    {"GBuffer.Roughness", "GBuffer.Roughness"},
    {"DLSS.Input.DiffuseAlbedo", "DLSS.Input.DiffuseAlbedo"},
    {"GBuffer.DiffuseAlbedo", "DLSS.Input.DiffuseAlbedo"},
    {"GBuffer.IndirectAlbedo", "GBuffer.IndirectAlbedo"},
    {"DLSS.Input.SpecularAlbedo", "DLSS.Input.SpecularAlbedo"},
    {"GBuffer.SpecularAlbedo", "DLSS.Input.SpecularAlbedo"},
    {"GBuffer.Emissive", "GBuffer.Emissive"},
    {"GBuffer.SpecularMvec", "GBuffer.SpecularMvec"},
    // Ray-tracing noise hints
    {"RayTracingHitDistance", "RayTracingHitDistance"},
    {"DLSSD.DiffuseHitDistance", "DLSSD.DiffuseHitDistance"},
    {"DLSSD.SpecularHitDistance", "DLSSD.SpecularHitDistance"},
    {"DLSSD.DiffuseRayDirection", "DLSSD.DiffuseRayDirection"},
    {"DLSSD.SpecularRayDirection", "DLSSD.SpecularRayDirection"},
    {"DLSSD.DiffuseRayDirectionHitDistance", "DLSSD.DiffuseRayDirectionHitDistance"},
    {"DLSSD.SpecularRayDirectionHitDistance", "DLSSD.SpecularRayDirectionHitDistance"},
    {"DLSSD.ReflectedAlbedo", "DLSSD.ReflectedAlbedo"},
    // Per-effect guides, read only when the game preserves that effect
    {"DLSSD.ColorBeforeParticles", "DLSSD.ColorBeforeParticles"},
    {"DLSSD.ColorAfterParticles", "DLSSD.ColorAfterParticles"},
    {"DLSSD.ColorBeforeTransparency", "DLSSD.ColorBeforeTransparency"},
    {"DLSSD.ColorAfterTransparency", "DLSSD.ColorAfterTransparency"},
    {"DLSSD.ColorBeforeFog", "DLSSD.ColorBeforeFog"},
    {"DLSSD.ColorAfterFog", "DLSSD.ColorAfterFog"},
    {"DLSSD.DepthOfFieldGuide", "DLSSD.DepthOfFieldGuide"},
    {"DLSSD.ColorBeforeDepthOfField", "DLSSD.ColorBeforeDepthOfField"},
    {"DLSSD.ColorAfterDepthOfField", "DLSSD.ColorAfterDepthOfField"},
    {"DLSSD.ScreenSpaceRefractionGuide", "DLSSD.ScreenSpaceRefractionGuide"},
    {"DLSSD.ColorBeforeScreenSpaceRefraction", "DLSSD.ColorBeforeScreenSpaceRefraction"},
    {"DLSSD.ColorAfterScreenSpaceRefraction", "DLSSD.ColorAfterScreenSpaceRefraction"},
    {"DLSSD.ScreenSpaceSubsurfaceScatteringGuide", "DLSSD.ScreenSpaceSubsurfaceScatteringGuide"},
    {"DLSSD.ColorBeforeScreenSpaceSubsurfaceScattering", "DLSSD.ColorBeforeScreenSpaceSubsurfaceScattering"},
    {"DLSSD.ColorAfterScreenSpaceSubsurfaceScattering", "DLSSD.ColorAfterScreenSpaceSubsurfaceScattering"},
    {"DLSSD.Alpha", "DLSSD.Alpha"},
    // Masks and secondary motion
    {"DLSS.DisocclusionMask", "DLSS.DisocclusionMask"},
    {"DLSS.Input.Reduce.Ghost.Mask", "DLSS.Input.Reduce.Ghost.Mask"},
    {"DLSSD.ResponsivityMask", "DLSSD.ResponsivityMask"},
    {"MotionVectorsReflection", "MotionVectorsReflection"},
    {"TransparencyMask", "TransparencyMask"},
    {"DepthHighRes", "DepthHighRes"},
};
// Scalars the denoiser reads per frame, forwarded from the caller's own parameters.
struct AuxScalar
{
    const char* name;
};
inline const AuxScalar kAuxScalars[] = {
    {"DLSS.Denoise.Mode"},
    {"DLSS.Roughness.Mode"},
    {"DLSS.Use.HW.Depth"},
    {"DLSS.Use.Folded.Network"},
    {"DLSSD.IndicatorLevel"},
    {"DLSS.Translucency.Type"},
    {"DLSS.Subsurface.Type"},
    {"DLSS.Specular.Type"},
};
// Indices into the merged per-frame kAuxScalars snapshot (creation values plus this evaluation's overrides).
inline constexpr size_t AuxRoughnessMode = 1, AuxHardwareDepth = 2;
// The camera matrices the denoiser reprojects with, forwarded as the float[16] the caller
// registered (the SDK passes matrices by pointer, not as sixteen scalars).
inline const char* const kAuxMatrices[] = {"WorldToViewMatrix", "ViewToClipMatrix"};
unsigned int get_uint_or(void* parameters, const char* name, unsigned int fallback);
// The Subrect origins nvngx_dlssd.dll declares for its auxiliary inputs
// (nvsdk_ngx_defs_dlssd.h:60-67 and 99-146). A name here has both a .Subrect.Base.X and a
// .Subrect.Base.Y parameter; the parallel FrameParams array holds one origin per entry, and
// the strings are static so no per-frame allocation is needed to forward them.
inline const char* const kAuxSubrectNames[] = {
    "DLSS.Input.DiffuseAlbedo",       "DLSS.Input.SpecularAlbedo", "DLSS.Input.Normals",
    "DLSS.Input.Roughness",           "DLSSD.Alpha",               "DLSSD.ReflectedAlbedo",
    "DLSSD.ColorBeforeParticles",     "DLSSD.ColorAfterParticles", "DLSSD.ColorBeforeTransparency",
    "DLSSD.ColorAfterTransparency",   "DLSSD.ColorBeforeFog",      "DLSSD.ColorAfterFog",
    "DLSSD.ScreenSpaceRefractionGuide", "DLSSD.ColorBeforeScreenSpaceRefraction",
    "DLSSD.ColorAfterScreenSpaceRefraction", "DLSSD.ScreenSpaceSubsurfaceScatteringGuide",
    "DLSSD.ColorBeforeScreenSpaceSubsurfaceScattering",
    "DLSSD.ColorAfterScreenSpaceSubsurfaceScattering", "DLSSD.DepthOfFieldGuide",
    "DLSSD.ColorBeforeDepthOfField",  "DLSSD.ColorAfterDepthOfField", "DLSSD.DiffuseHitDistance",
    "DLSSD.SpecularHitDistance",      "DLSSD.DiffuseRayDirection", "DLSSD.SpecularRayDirection",
    "DLSSD.DiffuseRayDirectionHitDistance", "DLSSD.SpecularRayDirectionHitDistance",
};
struct FrameParams
{
    float jitterX = 0, jitterY = 0, mvScaleX = 1, mvScaleY = 1, sharpness = 0;
    float preExposure = 1, exposureScale = 1, frameTime = 16.6f;
    int reset = 0, invertX = 0, invertY = 0;
    unsigned int renderWidth = 0, renderHeight = 0;
    unsigned int colorBaseX = 0, colorBaseY = 0, depthBaseX = 0, depthBaseY = 0;
    unsigned int mvBaseX = 0, mvBaseY = 0, outputBaseX = 0, outputBaseY = 0;
    bool hasExposure = false;
    // The exposure texture's own geometry. A game may pass a per-pixel exposure buffer rather
    // than the 1x1 scalar, and a capture that recorded a plane without its size could not say
    // which it was.
    unsigned int exposureWidth = 0, exposureHeight = 0;
    // How this feature was created, copied off it so a capture records what the game asked for
    // rather than what a replay would otherwise have to guess: the model, the render and output
    // dimensions, the quality mode, the HDR/depth/exposure create flags, whether Output.Subrect
    // origins are read, and the render preset in force.
    unsigned int createWidth = 0, createHeight = 0, createOutWidth = 0, createOutHeight = 0;
    int createQuality = 0, createFlags = 0, createSubrects = 0;
    unsigned int createPreset = 0;
    bool vram = false; // inputs and output stay in VRAM (VRAM interop)
    bool gpuInputMarker = false; // this list recorded the exportable per-slot readiness mirror
    bool gpuCompletion = false; // the game's queue polls this frame's completion word instead of a split wait
    bool split = false; // this frame presents its own result (split frames)
    unsigned motionDilation = 0; // actually applied, worker only
    // D4R_SHIM_INPLACE: the game's linear output texture, written without a game-side copy (0 = none)
    CudaDevicePtr outputInPlace = 0;
    uint32_t outputInPlacePitch = 0;
    // D4R_SHIM_OUTPUT_PACKED: the output kernel stores the game's R10G10B10A2 texels itself, so the game-side
    // copy (or the in-place texture) takes them without the RGBA16F conversion image and blit
    bool packedOutput = false;
    bool packedRgba8 = false; // R8G8B8A8_UNORM after the original half rounding
    // Ray Reconstruction: the denoiser's own per-frame settings and camera matrices, copied
    // off the caller's parameter object on the game thread so the worker never reads an
    // object the game may have destroyed by then. A scalar the caller did not set this frame
    // carries the creation-time value, not an invented one and not last frame's override; a
    // matrix the caller did not set is published as absent, because the pointer names this
    // frame's storage and nothing outlives it.
    int auxScalar[sizeof(kAuxScalars) / sizeof(kAuxScalars[0])] = {};
    bool auxScalarSet[sizeof(kAuxScalars) / sizeof(kAuxScalars[0])] = {};
    float auxMatrix[sizeof(kAuxMatrices) / sizeof(kAuxMatrices[0])][16] = {};
    bool auxMatrixSet[sizeof(kAuxMatrices) / sizeof(kAuxMatrices[0])] = {};
    // One Subrect origin per entry of kAuxSubrectNames, snapshotted per frame. Zero means the
    // caller did not register one, and zero is what is forwarded - never the previous frame's.
    unsigned int auxSubrect[sizeof(kAuxSubrectNames) / sizeof(kAuxSubrectNames[0])][2] = {};
};
struct FrameTiming
{
    bool enabled = false;
    ProfileClock::time_point gameStart, queuedAt;
    double slotWait = 0, inputRecord = 0, outputRecord = 0, gameCall = 0, gameInterval = 0;
    double convert[4] = {}, upload[4] = {};
    double download = 0, outputConvert = 0;
    double prep = 0, worker = 0, workerWait = 0, finishWait = 0, markerWait = 0;
    double ngxHost = 0, ctxSync = 0, gpuEval = -1;
    // GPU-input-wait frames split gpu_eval at an event recorded right behind the wait: input_gpu_wait is launch ->
    // inputs released (the game's remaining input work), dlss_gpu is inputs released -> completion (the real DLSS
    // cost). CPU-wait frames have no wait on the GPU: dlss_gpu equals gpu_eval and input_gpu_wait is 0.
    double dlssGpu = -1, inputGpuWait = -1;
    // Game-queue GPU timestamps around this evaluation (D3D12 queries, read 4 frames later, so these describe the
    // frame kOutputSlots earlier): vk_input = shim input commands, vk_dlss_wait = input marker -> second half
    // starts (input release, DLSS, completion signal), vk_output = second-half commands, vk_total = all of it
    // (what an overlay timing the evaluate call on the game's list reports).
    double vkInput = -1, vkDlssWait = -1, vkOutput = -1, vkTotal = -1;
    int vkOutputFused = -1, vkInputFused = -1;
    bool inputEventRecorded = false;
    double motionDilationWall = 0;
    double nativeMotionGpu = -1, wireRepairGpu = -1;
    bool nativeMotionTimed = false, wireRepairTimed = false;
    bool nativeFieldReady = false, nativeSubstituted = false, wireRequested = false;
    bool markerDeferred = false; // the worker waits for the input marker (D4R_SHIM_DEFER_MARKER)
    bool gpuInputWait = false; // GPU events span wait enqueue -> NGX completion, including input readiness
    bool gpuEventsRecorded = false;
    double outputSyncWall = 0, outputSyncCpu = -1;
    bool outputSyncBlocking = false;
    bool outputSyncNative = false;
    double h2dTotal = 0, d2hIssue = 0, d2hWait = 0;
    ProfileClock::time_point prepStart, workerQueued, finishQueued;
    size_t queueDepth = 0;
    uint32_t presentedFrame = 0;
};
// A plane in the canonical CUDA layout, tightly packed, in ordinary host
// memory: filled from the D3D12 staging by one pipeline stage and consumed by
// the next (see prepare_inputs / run_evaluation / publish_output).
struct HostPlane
{
    uint8_t* bytes = nullptr; // page-locked when overlapped staging is on
    size_t capacity = 0;
    bool pinned = false;
    std::vector<uint8_t> fallback;
    UINT width = 0, height = 0;
    size_t rowBytes = 0;
    size_t size() const { return rowBytes * height; }
};
// Ray Reconstruction auxiliary surface: one per distinct D3D12Resource the caller registers,
// keyed on the resource because several names routinely resolve to one texture. The surface
// carries the caller's own DXGI format - it is never converted to a canonical plane - so the
// texture object the denoiser samples describes exactly what was rendered.
//
// The surface outlives the frames that name it: `holds` counts queued frames still referring
// to it, and the resource is AddRef'd so a caller that releases it while a frame is in flight
// cannot pull it out from under the copy recorded on its command list.
struct AuxSurface
{
    ID3D12Resource* resource = nullptr;
    std::vector<const char*> publish;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT width = 0, height = 0;
    uint32_t arrayFormat = 0, channels = 0;
    AuxLayout how = AuxLayout::Verbatim; // whether the bytes reach the array as they are
    uint32_t texelBytes = 0;            // the caller's own texel size, for a capture
    uint32_t lastSeen = 0;  // frame the caller last registered this resource
    unsigned int holds = 0; // queued frames still naming this surface
    bool dropping = false;  // no longer registered; freed once holds reaches zero
    CudaImage image;        // worker only
    CudaObject handle = 0;  // what the parameters point at; its address must stay put
    Staging staging[kSlots];
    HostPlane host[kSlots]; // stable upload bytes for host fallback (worker only)
    VramBuffer vram[kSlots]; // per-frame guide bytes; imported lazily on the CUDA worker
    VramImage conversion;   // packed/BGRA guides converted by Vulkan, not the CPU
    bool gpu[kSlots] = {};
};
// What one frame's slot bound: which surface answered which denoiser name, for that frame
// only. Publish reads this, never the live surface list, so a surface the caller registered
// last frame is never handed to the denoiser for a frame that did not register it.
struct AuxBinding
{
    AuxSurface* surface = nullptr;
    std::vector<const char*> names;   // what the denoiser reads it as
    std::vector<const char*> sources; // what the caller registered it under
    uint32_t frame = 0;
};
struct InputSlot
{
    Staging color, depth, motion, exposure;
    Staging nativeMotionCapture; // opt-in geometry sidecar readback; never an NGX input
    Staging nativeDepthCapture, nativeSceneCapture, nativeOpaqueCapture;
    Staging nativePrimaryDepthCapture, opaqueSceneCapture, opaqueMatrixCapture, originalPrimaryDepthCapture;
    Staging materialSceneCapture;
    bool materialSceneCaptureReady=false;
    Staging glassMaskCapture;
    bool glassMaskCaptureReady=false;
    uint32_t glassMaskFormat=0;
    Staging originalOpaqueCopyCapture;
    bool originalOpaqueCopyCaptureReady=false;
    Staging ownedSceneCapture, ownedMatrixCapture, ownedDepthCapture, ownedValidityCapture, glassValidityCapture;
    bool ownedValidityCaptureReady=false, glassValidityCaptureReady=false;
    ID3D12Resource* nativeValidityOwner=nullptr;
    VramBuffer nativeGpuInputs[4]; // depth, opaque geometry, scene, matrix
    bool nativeGpuCheckReady=false, nativeGpuStageValid=false;
    UINT nativeGpuWidth=0,nativeGpuHeight=0;
    CudaDevicePtr nativeGpuResult=0;
    size_t nativeGpuResultBytes=0;
    // D4R_NATIVE_MV_LIVE (default off): the composed field is handed to NGX as its motion input.
    bool nativeLive=false;              // game thread: this frame recorded live-substitution inputs
    CudaObject nativeGpuTexture=0;      // worker: pitch-linear texture over nativeGpuResult
    UINT nativeTexWidth=0,nativeTexHeight=0;
    size_t nativeTexPitch=0;
    CudaDevicePtr nativeTexPointer=0;
    bool ownedDepthCaptureReady=false;
    bool ownedSceneCaptureReady=false, ownedMatrixCaptureReady=false;
    bool nativePrimaryDepthCaptureReady=false, opaqueSceneCaptureReady=false;
    bool opaqueMatrixCaptureReady=false, originalPrimaryDepthCaptureReady=false;
    bool nativeOpaqueCaptureReady = false;
    bool nativeGeometryCaptureReady = false;
    Staging originalGeometryCapture, originalHardwareDepthCapture;
    bool originalGeometryCaptureReady = false, originalHardwareDepthCaptureReady = false;
    bool nativeDepthCaptureReady = false, nativeSceneCaptureReady = false;
    uint32_t nativeMotionCaptureFrame = 0;
    std::string nativeMotionCapturePath;
    VramBuffer vram[4]; // VRAM interop: the planes in canonical layout
    // D4R_SHIM_LINEAR_INPUTS: pitch-linear texture objects on vram[] handed to NGX (worker only)
    CudaObject linearTexture[4] = {};
    CudaDevicePtr linearPointer[4] = {};
    UINT linearWidth[4] = {}, linearHeight[4] = {};
    size_t linearPitch[4] = {};
    CudaDevicePtr dilatedMotion = 0;
    size_t dilatedBytes = 0;
    CudaObject dilatedTexture = 0;
    UINT dilatedWidth = 0, dilatedHeight = 0;
    size_t dilatedPitch = 0;
    bool dilationEnqueued = false;
    std::vector<AuxBinding> auxBindings; // this slot's denoiser inputs, for one frame
    // The DLSSD.OutputAlpha texture the caller asked for on this frame, held until the frame
    // retires. Present only on a frame that actually registered it.
    ID3D12Resource* alphaResource = nullptr;
    UINT alphaWidth = 0, alphaHeight = 0, alphaBaseX = 0, alphaBaseY = 0;
    UINT alphaExtentW = 0, alphaExtentH = 0;
    DXGI_FORMAT alphaFormat = DXGI_FORMAT_UNKNOWN;
    uint32_t alphaWriteback = 0;
    bool alphaRequested = false;
    // GPU-resident alpha (VRAM interop): the caller's image and whether this frame's result can be
    // blitted into it, so the float32 result never has to come back through host memory. Set with
    // the rest of the alpha capture, on the game thread, before the frame is queued.
    VkImage alphaImageHandle = VK_NULL_HANDLE;
    bool alphaConvert = false;
    bool alphaGpu = false;
    std::atomic<bool> busy{false}; // readback staging owned by the pipeline
    HostPlane host[4];
    std::atomic<bool> hostBusy{false}; // host planes not yet uploaded by the worker
};
// DLSSD.OutputAlpha is an OUTPUT the denoiser writes (nvsdk_ngx_defs_dlssd.h:98, with its own
// Subrect base at 101/102). It is delivered back through its own ring of slots, modelled on
// OutputSlot above, because it has exactly the same lifetime problem as the colour output: a
// result staged for one frame is read by a later frame's command list, so it must not be
// rewritten until every reader has passed.
struct AlphaSlot
{
    Staging staging;                            // upload buffer holding one result
    VramBuffer vram;                            // VRAM interop: the evaluated region as float32
    VramImage conversion;                       // the R32F image that result is blitted from
    bool gpu = false;                           // filled in vram, presented from it; else staging
    std::atomic<uint32_t> lastReadFrame{0};     // reserved/last-read frame, as OutputSlot's
    std::atomic<uint32_t> producedFrame{0};     // frame that filled it
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT baseX = 0, baseY = 0;                  // where this result sits in the staging buffer
    UINT width = 0, height = 0;                 // the region it covers
    uint32_t writeback = 0;                     // destination texel kind (see alpha_output_layout)
};
struct OutputSlot
{
    Staging staging;
    VramBuffer vram; // VRAM interop: the result in RGBA16F
    std::atomic<uint32_t> lastReadFrame{0}; // last frame whose command list copies from it
    std::atomic<uint32_t> producedFrame{0};
};
// Views and their D3D12 image owners survive every command list recorded by this feature.
struct EngineBorrowedView
{
    ID3D12Resource* resource = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
    VkImageUsageFlags usage = 0;
    ~EngineBorrowedView()
    {
        if (view) d4r::vkTable.DestroyImageView(device, view, nullptr);
        if (resource) resource->Release();
    }
};
struct FsrRrFeature;
// d3dhip's bridge (d4r_hip_net.dll, engine/d3d12/hip_net_bridge.cpp): the native HIP network behind a C table,
// loaded on first use. Importing amdhip64 at load time made Ghost of Tsushima's device hang next to OptiScaler's
// driver hooks, before its first frame.
#ifdef D4R_ENGINE_D3DHIP
struct HipNetApi
{
    void* (*create)(ID3D12Device*, const char*, unsigned, unsigned, unsigned long long, unsigned long long, char*,
                    unsigned) = nullptr;
    ID3D12Resource* (*tokens)(void*) = nullptr;
    ID3D12Resource* (*head)(void*) = nullptr;
    int (*run)(void*, char*, unsigned) = nullptr;
    void (*destroy)(void*) = nullptr;
    // optional (older d4r_hip_net.dll has neither): GPU-side ordering through a shared D3D12 fence
    int (*useFence)(void*, ID3D12Fence*, char*, unsigned) = nullptr;
    int (*launchAfter)(void*, unsigned long long, unsigned long long, char*, unsigned) = nullptr;
    // optional (older d4r_hip_net.dll has none): Ray Reconstruction preset F's network (hip_net_rr.h)
    void* (*rrCreate)(ID3D12Device*, const char*, int, unsigned, unsigned, unsigned, unsigned, ID3D12Resource*,
                      unsigned long long, char*, unsigned) = nullptr;
    const void* (*rrExternal)(void*) = nullptr;
    int (*rrRun)(void*, char*, unsigned) = nullptr;
    int (*rrUseFence)(void*, ID3D12Fence*, char*, unsigned) = nullptr;
    int (*rrLaunchAfter)(void*, unsigned long long, unsigned long long, char*, unsigned) = nullptr;
    void (*rrDestroy)(void*) = nullptr;
};
#endif
#ifdef D4R_ENGINE_D3DHIP
inline HipNetApi g_hipApi;
#endif
// The native D3D12 engine's state of one feature (engine_evaluate_d3d12). The submission hooks (d4r_d3d12_submit.h)
// report recordings from the thread that executes or resets the game's list, hence the mutex.
struct EngineDxFeature
{
    std::mutex mutex;
    std::unique_ptr<d4r::dx::GameUpscaler> upscaler;
    std::unique_ptr<d4r::dx::GameDenoiser> denoiser;
    uint32_t rrWidth = 0, rrHeight = 0;
    std::unique_ptr<d4r::dx::GameUpscalerM> upscalerM; // preset M (instead of upscaler)
    // presets M and L on GPUs without FP8 WMMA (engine/d3d12/runtime_m_rdna3.h); the jitter of a recording becomes the
    // previous frame's only once the recording is submitted
    std::unique_ptr<d4r::dx::rdna3::GameUpscalerM> upscalerRdna3;
#ifdef D4R_ENGINE_D3DHIP
    // d3dhip: preset M's network as the model's HIP layers between the D3D12 front and back (d4r_hip_net.dll). The
    // front runs on the game's queue, `front` is signalled after it on the same queue, and the network runs on the CPU
    // between the split list's two parts.
    void* hip = nullptr;
    d4r::dx::ComPtr<ID3D12Resource> hipTokens, hipHead; // the network's shared buffers (retained)
    d4r::dx::ComPtr<ID3D12Fence> front;
    uint64_t frontValue = 0;
    // `front` is shared with HIP: the network waits for the front and the back waits for the network on the GPU
    // (engine_dx_hip_network), so the game's thread never blocks between the parts
    bool hipFenced = false;
    uint32_t hipWidth = 0, hipHeight = 0;
    // Ray Reconstruction preset F instead (d4rHipRr*): the network over the denoiser's shared plane arena; `front`,
    // frontValue and hipFenced as above
    void* hipRr = nullptr;
    ~EngineDxFeature()
    {
        if (hip != nullptr && g_hipApi.destroy != nullptr)
            g_hipApi.destroy(hip);
        if (hipRr != nullptr && g_hipApi.rrDestroy != nullptr)
            g_hipApi.rrDestroy(hipRr);
    }
#endif
    float committedJitter[2]{}, pendingJitter[2]{};
    bool pending = false; // a recording awaits submitted/discarded
    void submitted(ID3D12Fence* fence, uint64_t value)
    {
        if (upscalerRdna3)
        {
            upscalerRdna3->submitted(fence, value);
            committedJitter[0] = pendingJitter[0]; committedJitter[1] = pendingJitter[1];
        }
        else if (denoiser) denoiser->submitted(fence, value);
        else if (upscalerM) upscalerM->submitted(fence, value);
        else upscaler->submitted(fence, value);
    }
    void discarded()
    {
        if (upscalerRdna3) upscalerRdna3->discarded();
        else if (denoiser) denoiser->discarded();
        else if (upscalerM) upscalerM->discarded();
        else upscaler->discarded();
    }
    bool idle() const
    {
        return denoiser ? denoiser->idle() : upscalerRdna3 ? upscalerRdna3->idle() : upscalerM ? upscalerM->idle() : upscaler->idle();
    }
    // D4R_PROFILE: GPU timestamps at the stage boundaries of one evaluation per slot (engine_dx_timing), logged once
    // the submission has retired as the Vulkan engines' D4R_ENGINE_STAGE_PROFILE rows
    static constexpr unsigned TimingSlots = 64, TimingQueries = 32;
    struct Timing {
        d4r::dx::ComPtr<ID3D12Fence> fence;
        uint64_t value = 0, frequency = 0;
        uint32_t frame = 0;
        bool busy = false;
        const char* preset = "";
        const char* const* labels = nullptr; // labels[i]: the stage that ends at boundary i + 1
        uint32_t stages = 0, written = 0;    // written: a bit per boundary that has a timestamp
        uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
        double cpuRecord = -1, gameInterval = -1;
    };
    d4r::dx::ComPtr<ID3D12QueryHeap> queries;
    d4r::dx::ComPtr<ID3D12Resource> readback;
    std::array<Timing, TimingSlots> timings{};
    int pendingTiming = -1;
    std::chrono::steady_clock::time_point profileAt{}, profileRecordAt{}; // the last sampled evaluation's start
    std::chrono::steady_clock::time_point released{};
};
struct Feature
{
    NgxHandle handle = {}; // returned to the caller
    bool nativeVulkan = false; // distinct resource/lifetime contract
    // A D3D12 feature on a device without vkd3d-proton (native Windows): only the native D3D12 engine
    // (engine_evaluate_d3d12) serves it, as the CUDA backend reaches D3D12 resources through vkd3d-proton.
    bool nativeD3d12 = false;
    // [RayReconstruction] Backend = fsr: the feature-13 state of d4r_fsr_rr_backend.inl (owned; freed by
    // fsr_rr_destroy once the GPU is done). `owner`: set on the internal Super Sampling feature such a feature
    // drives, which is released with it and never by the game.
    FsrRrFeature* fsr = nullptr;
    Feature* owner = nullptr;
    std::atomic<bool> retiring{false};
    IUnknown* resources = nullptr; // main reference plus command-allocator references
    NgxHandle* cudaHandle = nullptr;
    void* cudaParams = nullptr;
    CudaDevicePtr scratch = 0;
    unsigned int width = 0, height = 0, outWidth = 0, outHeight = 0;
    unsigned int cudaWidth = 0, cudaHeight = 0; // render size the NGX feature was created with (worker only)
    unsigned int refusedWidth = 0, refusedHeight = 0; // render size NGX would not create a feature for (worker only)
    int quality = 0, flags = 0, subrects = 0;
    int cudaFlags = 0; // flags the NGX CUDA feature currently exists with (worker only; live native MV clears MVLowRes)
    std::array<unsigned int, 6> creationPresets{};
    bool cudaCreationDeferred = false;
    unsigned int preset = 0;
    // DLSS.Enable.Output.Subrects as the caller set it at creation: it decides whether the
    // Output.Subrect origins are read at all, so a replay must create the feature the same way.
    int outputSubrects = 0;
    CudaEvent profileStart = nullptr, profileEnd = nullptr;
    CudaEvent nativePassEvents[4] = {}; // sampled native compose [0,1], wire repair [2,3]
    CudaEvent profileInputsReleased = nullptr; // D4R_PROFILE only: recorded right behind the GPU-side input wait
    // D4R_PROFILE only: 4 game-queue timestamps per output slot (start, input marker, second half, end)
    ID3D12QueryHeap* timestampHeap = nullptr;
    ID3D12Resource* timestampReadback = nullptr;
    volatile uint64_t* timestampValues = nullptr;
    uint8_t timestampOutputFused[kOutputSlots] = {}, timestampInputFused[kOutputSlots] = {};
    bool recordOutputFused = false, recordInputFused = false;
    double timestampTicksPerMs = 0.0;
    bool timestampFailed = false;
    CudaEvent outputReadyEvent = nullptr;
    // output wait pacing (D4R_SHIM_OUTPUT_POLL_US): smoothed time from the event record to its completion, kept
    // separately for each wait site (they wait for very different amounts of GPU work)
    double publishWaitEmaUs = 0.0, evalWaitEmaUs = 0.0;
    // native Vulkan input wait: smoothed time from recording to the input event (waiters may overlap)
    std::atomic<double> vkInputWaitEmaUs{0.0};
    std::atomic<double> markerWaitEmaUs{0.0}; // prep or worker waits; never both for the same frame
    bool profileEventsReady = false;

    CudaImage color, depth, motion, exposure, output;
    CudaObject colorHandle = 0, depthHandle = 0, motionHandle = 0, exposureHandle = 0, outputHandle = 0;

    InputSlot inputs[kSlots];
    OutputSlot outputs[kOutputSlots];
    HostPlane outputHost[2]; // downloaded results waiting for publish_output
    std::atomic<bool> outputHostBusy[2] = {false, false};
    unsigned int nextOutputHost = 0; // worker only
    // Overlapped staging (worker only): linear device buffers the async copies
    // target, moved to/from the CUDA arrays with device-to-device copies.
    CudaDevicePtr planeLinear[4] = {}, outputLinear = 0;
    size_t planeLinearBytes[4] = {}, outputLinearBytes = 0;
    CudaDevicePtr nativeThinMask=0,preparedWireAxes=0;
    size_t nativeThinMaskBytes=0,preparedWireAxesBytes=0;
    uint32_t preparedWireFlags=0;
    bool preparedWireAxesReady=false; // worker only, produced on this frame's NGX stream
    CudaDevicePtr wirePatchMask=0;
    CudaDevicePtr wirePublishBuffer=0;
    bool wirePublishedDirect=false;
    CudaDevicePtr wireBuffers[8] = {}; // raw, filtered output, observed RGB[2], age[2], depth[2]; worker-owned
    UINT wireWidth=0,wireHeight=0;
    uint32_t wireLastFrame=0,wireIndex=0;
    bool wireValid=false;
    bool wireOutputReady=false; // filtered linear RGBA16F is this frame's publish source
    UINT wireRenderWidth=0,wireRenderHeight=0;
    float wirePreExposure=0,wireExposureScale=0;
    // Frames queued but not yet published (or dropped). evaluate() blocks the
    // game at D4R_SHIM_MAX_IN_FLIGHT (default 3) so a backlog cannot build up
    // between pipeline stages and age the presented result.
    std::atomic<int> inFlight{0};
    DXGI_FORMAT outputFormat = DXGI_FORMAT_UNKNOWN;
    UINT outputWidth = 0, outputHeight = 0;
    // VRAM interop: decided on the first evaluation, dropped for good if the
    // formats later stop qualifying.
    bool vram = false, vramDecided = false;
    bool fusedColorAllowed = true;
    FusedColorInput fusedColor[kSlots];
    std::vector<FusedColorInput> retiredFusedColor;
    bool fusedOutputAllowed = true;
    FusedColorInput fusedOutput[kOutputSlots];
    std::vector<FusedColorInput> retiredFusedOutput;
    VramImage colorConversion;
    VramImage depthConversion;
    VramImage motionConversion;
    VramImage exposureConversion;
    VramImage outputConversion;
    // Split frames: the rest of frame N's command list is submitted only once
    // splitSemaphore reaches N (all admitted frames through N have retired).
    std::atomic<bool> split{false};
    VkSemaphore splitSemaphore = VK_NULL_HANDLE;
    // Native D3D12 (d4r_d3d12_split.h): the second part of frame N's list waits for splitFence to reach N; each
    // output slot records into its own second list, reused once its retire fence passes.
    ID3D12Fence* splitFence = nullptr;
    struct SplitPart
    {
        ID3D12CommandAllocator* allocator = nullptr;
        ID3D12GraphicsCommandList* list = nullptr;
        ID3D12Fence* retire = nullptr;
        uint64_t retireValue = 0;
    } splitParts[kOutputSlots];
    std::mutex splitMutex;
    uint64_t splitSignalled = 0; // under splitMutex
    D4rFrameCompletion splitCompletion; // under splitMutex; cancellation can finish out of order
    // GPU input waits are opt-in, with an independent CPU watchdog that can release discarded lists.
    // D4R_SHIM_OUTPUT_DIRECT: this frame's result is stored by the native output kernel straight into its
    // destination buffer (worker only), so the array -> buffer copy is skipped.
    bool outputRedirected = false;
    // The previous frame's output kernel was the native one and honoured the redirect (worker writes, game
    // thread reads): only then is a frame recorded for packed output.
    std::atomic<bool> packedReady{false};
    std::atomic<bool> rgba8Ready{false};
    bool outputDirectAllowed = false; // the forced preset's output kernel honours the redirect
    // D4R_SHIM_LINEAR_INPUTS: NGX samples the interop buffers directly through pitch-linear
    // texture objects (rows padded to 256 bytes), so the buffer -> array copies disappear.
    bool linearInputs = false;
    bool dilationReported = false;
    std::vector<VramBuffer> retiredBuffers; // may still be read by queued command lists
    std::vector<VramImage> retiredImages;
    ID3D12Resource* marker = nullptr;
    volatile uint32_t* markerValue = nullptr; // [0] latest input; [1 + slot] exact input frame
    // The D3D12 READBACK marker has no exportable HIP pointer. Mirror its per-slot tags in a dedicated
    // HOST_VISIBLE | HOST_COHERENT Vulkan allocation imported into HIP, as engineSync is. Keep the original
    // marker as the authority: releasing a wait from the CPU must not turn unsubmitted input into ready input.
    VramBuffer ngxInputSync;
    // D4R_SHIM_GPU_COMPLETION: words kCompletionWord + frame % kOutputSlots of ngxInputSync carry each frame's
    // completion (written by HIP after the output, or by the CPU in frame_retired); the game's queue polls them.
    bool gpuCompletion = false;
    bool completionFired = false; // worker only: the bridge queued this frame's word behind the native output kernel
    D4rVkStagingSet completionSet;
    void* ngxTimeline = nullptr; // native completion and CPU drops share this timeline writer
    bool ngxNativeCompletion = false; // worker only after feature setup
    std::atomic<bool> ngxGpuWait{false};
    bool ngxWaitEventsWarmed = false; // worker only: initialize event storage before the first stream wait
    std::mutex ngxWaitMutex;
    bool ngxWaitActive = false, ngxWaitCancelled = false; // under ngxWaitMutex, armed before calling HIP
    uint32_t ngxWaitFrame = 0;
    int ngxWaitSlot = 0;
    ProfileClock::time_point ngxWaitStart;
    uint32_t frame = 0;
    uint32_t rejectedFormatCount = 0; // game thread; rate-limits format diagnostics
    std::atomic<int> latestOutput{-1};
    std::atomic<uint32_t> completedFrames{0};
    uint32_t engineFrames = 0;    // game thread: frames the engine recorded into the game's lists (no CUDA completion)
    uint32_t engineMisses = 0;    // game thread: native D3D12 frames the engine could not record (no fallback there)
    uint32_t evaluatedFrames = 0; // worker only
    // D4R_ENGINE: d4r's own inference, recorded into the game's command lists (game thread only). Recorded
    // lists retain `resources`, so this is destroyed only once the GPU has finished with them.
    uint64_t engineViewBytes = 0;
    size_t engineSmallViews = 0; // views outside the byte and count budget
    bool engineViewBudgetReported = false;
    std::vector<std::unique_ptr<EngineBorrowedView>> engineViews; // destroyed after engine descriptor pools
    std::unique_ptr<vkhip::KNetwork> engineVkHip; // buffers outlive the adapter's descriptor sets
    std::unique_ptr<d4r::GameUpscaler> engine;
    std::unique_ptr<EngineDxFeature> engineDx; // native D3D12 (no vkd3d-proton): preset K
    std::unique_ptr<vkhip::MNetwork> engineMVkHip; // M's network on vkhip; outlives engineM, which uses its buffers
    std::unique_ptr<d4r::GameUpscalerM> engineM; // preset M/L, original network
    std::unique_ptr<d4r::GameUpscalerM> engineMDynamic; // Vulkan at changing sizes; original HIP resources stay alive
    bool engineMHipFailed = false; // d3dhip could not split the list: this feature's D3D12 engine stays on DXIL
    bool engineKHipFailed = false; // the same for preset K's d3dhip (D4R_D3DHIP_K)
    unsigned int engineMLastWidth = 0, engineMLastHeight = 0;
    unsigned int engineMRenderWidth = 0, engineMRenderHeight = 0; // first evaluation's effective extent
    std::unique_ptr<d4r::GameUpscalerE> engineE; // presets E/F (DLSS 3's CNN): fixed render size, rebuilt on a change
    unsigned int engineERenderWidth = 0, engineERenderHeight = 0;
    std::unique_ptr<vkhip::RRStages> engineRVkHip; // outlives the denoiser and its borrowed stage recorder
    std::unique_ptr<d4r::GameDenoiser> engineR; // Ray Reconstruction (engine/runtime_rr.inc)
    ProfileClock::time_point engineRProfileAt{};
    // Preset M and preset K with the network on the native HIP layers ([Engine] Network): the buffers shared
    // with HIP and the bridge's network. The game's list is split around the network, as for the CUDA backend's
    // split frames. `engineHead` is the network's result (M's expansion / K's final store reads it).
    VramBuffer engineTokens, engineHead;
    // Input readiness for the HIP network: engineSync holds the frame number the engine's front writes (a Vulkan
    // fill) and the network's stream waits for it on the GPU (engineGpuWait), so the layers start without a
    // CPU round trip. engineWaitFrame: the frame a queued stream wait is waiting for (0: none).
    VramBuffer engineSync;
    std::atomic<bool> engineGpuWait{false};
    bool engineNetTimeline = false; // native HIP callback and dropped-frame signals share one timeline lock
    std::atomic<uint32_t> engineWaitFrame{0};
    double engineNetWaitEmaUs = 0.0; // worker only: smoothed launch-to-completion wait, for adaptive polling
    void* engineNet = nullptr;
    bool engineRefused = false;
    bool rrCompileWarned = false; // the "CUDA backend has no preset F" warning was logged for the compile frames
    float engineJitter[2] = {0, 0}; // of the previous engine frame
    // Serializes picking the slot to present (game thread) against picking
    // the slot to overwrite (worker).
    std::mutex outputMutex;
    // Ray Reconstruction: feature 13, which the NGX core dispatches to the denoiser it
    // loaded. Super Sampling features leave this false and differ only in their feature id
    // and their preset names.
    bool rayReconstruction = false;
    // The denoiser's settings as the caller stated them at creation time. A setting the
    // caller varies per frame must fall back to this, not to the previous frame's value:
    // the parameter object outlives the frame, so an untouched one would silently carry
    // over an override that is no longer being asked for.
    int auxScalarBaseline[sizeof(kAuxScalars) / sizeof(kAuxScalars[0])] = {};

    // The denoiser's alpha output. Its CUDA surface is float32 single channel: the vendor's
    // transformer output kernel stores it as a nullable scalar float32 pixel
    // (sust.p.2d.b32.zero at param+248), so that is what the surface is created as.
    CudaImage alphaImage;
    HostPlane alphaHost; // worker: the read-back float32 result
    AlphaSlot alpha[kOutputSlots];
    std::atomic<int> latestAlpha{-1}; // the slot holding the newest unpresented result

    // A list, not a vector or a deque: the parameter object holds &AuxSurface.handle for the
    // duration of the denoiser's evaluate, so every element must keep its address. A vector
    // moves elements on growth and a deque invalidates *all* references when an element is
    // erased from the middle - which is exactly what dropping a surface does while other
    // surfaces are still queued. A list's erase touches only the element removed.
    std::list<AuxSurface> aux; // the caller's registered surfaces, under auxMutex
    // The game thread rewrites this list from the caller's parameter object while the worker
    // reads it, and the parameter object holds &aux.handle for the duration of the
    // denoiser's evaluate, so every aux access is under this lock.
    std::mutex auxMutex;
};
inline std::mutex g_featuresMutex;
inline std::vector<Feature*> g_features;
// D4R_PROFILE: game-queue timestamps for one evaluation. Point 0 is written before the input commands, 1 after the
// input marker, 2 at the start of the split's second half, 3 after the output commands (which also resolves the
// slot). Reading slot frame % kOutputSlots returns the evaluation kOutputSlots frames earlier, whose second half has
// normally run by now; incomplete (zero) values are skipped. Allocation failure disables it for the feature.
struct Feature;
bool profile_timestamps_ready(Feature& feature);
void profile_timestamp(Feature& feature, ID3D12GraphicsCommandList* list, uint32_t frame, int point);
void read_profile_timestamps(Feature& feature, uint32_t frame, FrameTiming& timing);
// Splits row-wise staging work (tens of MB per frame at 1440p, which one core
// moves at ~11 GB/s) across helper threads plus the calling worker thread.
// D4R_SHIM_COPY_THREADS sets the total thread count (default 4, 1 = serial).
class RowPool
{
public:
    void run(UINT rows, size_t bytes, const std::function<void(UINT, UINT)>& task)
    {
        const unsigned int parts = threads();
        if (parts <= 1 || bytes < (2u << 20) || rows < parts * 8)
        {
            task(0, rows);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = &task;
            rows_ = rows;
            parts_ = parts;
            remaining_ = parts - 1;
            ++generation_;
        }
        start_.notify_all();
        task(0, rows / parts);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return remaining_ == 0; });
        task_ = nullptr;
    }

private:
    unsigned int threads()
    {
        std::call_once(once_, [this] {
            count_ = env_uint("D4R_SHIM_COPY_THREADS", 4);
            count_ = count_ < 1 ? 1 : (count_ > 16 ? 16 : count_);
            for (unsigned int index = 1; index < count_; ++index)
                std::thread([this, index] { helper(index); }).detach();
        });
        return count_;
    }

    void helper(unsigned int index)
    {
        uint64_t seen = 0;
        for (;;)
        {
            const std::function<void(UINT, UINT)>* task;
            UINT begin, end;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                task = task_;
                begin = static_cast<UINT>(static_cast<uint64_t>(rows_) * index / parts_);
                end = static_cast<UINT>(static_cast<uint64_t>(rows_) * (index + 1) / parts_);
            }
            (*task)(begin, end);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --remaining_;
            }
            done_.notify_one();
        }
    }

    std::once_flag once_;
    unsigned int count_ = 1;
    std::mutex mutex_;
    std::condition_variable start_, done_;
    const std::function<void(UINT, UINT)>* task_ = nullptr;
    UINT rows_ = 0;
    unsigned int parts_ = 1, remaining_ = 0;
    uint64_t generation_ = 0;
};
// One pool per CPU pipeline stage; a pool serves one caller at a time.
inline RowPool g_inputRows, g_outputRows;
struct CopyStreams
{
    void* up = nullptr;
    void* down = nullptr;
};
// --- VRAM interop ------------------------------------------------------------------
//
// With D4R_SHIM_VRAM_INTEROP=1 the inputs and the result never leave VRAM.
// vkd3d-proton's interop interface hands out the game's VkDevice, the VkImage
// behind each D3D12 texture, and the raw VkCommandBuffer behind the game's
// D3D12 command list. evaluate() records Vulkan copies of the inputs into
// exportable device-local buffers (colour formats other than RGBA16F are
// converted by a blit first), and of the latest result out of one; the Wine
// nvcuda bridge (d4rImportVulkanMemory) maps those buffers into CUDA. The
// worker then only moves data between them and the CUDA arrays on the GPU.
// Frames whose formats do not qualify use the readback/upload path.
struct VulkanInterop
{
    std::once_flag once;
    bool ready = false;
    ID3D12DXVKInteropDevice1* interop = nullptr;
    ID3D12DXVKInteropDeviceD4R2* lifetime = nullptr;
    ID3D12DXVKInteropDeviceD4R* split = nullptr; // only with the d4r vkd3d-proton patch
    ID3D12DXVKInteropDeviceD4R3* linear = nullptr; // experimental: linear exportable game textures
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr deviceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    VkPhysicalDeviceMemoryProperties memory = {};
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    // VK_EXT_shader_float8 (vkd3d-proton enables it when the GPU has it): M's and L's engine layers are FP8
    // cooperative-matrix shaders, which RADV's compiler crashes on without it (RDNA3)
    bool float8 = false;
    // shaderBufferInt64Atomics is enabled on vkd3d-proton's device (it enables it when the GPU has it): Ray
    // Reconstruction preset F then measures the exposure in one pass, as NGX does, instead of two
    bool int64Atomics = false;
    // robustImageAccess2 is enabled on vkd3d-proton's device (it requires it): the Ray Reconstruction stages translated
    // from NVIDIA's kernels then leave the zero of an image load outside the image to the device
    bool robustImages = false;
    bool matrixF16 = false;
    // The engine runs on the game's own Vulkan device (native Vulkan NGX frontend): no vkd3d interop objects.
    bool native = false;
    // VK_KHR_pipeline_binary and VK_KHR_maintenance5 are on that device (the frontend asks the game for them with
    // D4R_ENGINE_VKHIP): vkhip injects the native layers through them on AMD's driver (engine/vkhip/vkhip.h)
    bool pipelineBinary = false;
    PFN_vkGetPhysicalDeviceFormatProperties formatProperties = nullptr;
    PFN_vkCreateBuffer createBuffer = nullptr;
    PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements bufferRequirements = nullptr;
    PFN_vkBindBufferMemory bindBuffer = nullptr;
    PFN_vkCreateImageView createView = nullptr;
    PFN_vkDestroyImageView destroyView = nullptr;
    PFN_vkCreateImage createImage = nullptr;
    PFN_vkDestroyImage destroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements imageRequirements = nullptr;
    PFN_vkBindImageMemory bindImage = nullptr;
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    PFN_vkMapMemory map = nullptr;
    PFN_vkUnmapMemory unmap = nullptr;
    PFN_vkCmdPipelineBarrier barrier = nullptr;
    PFN_vkCmdCopyImageToBuffer copyImageToBuffer = nullptr;
    PFN_vkCmdCopyBuffer copyBuffer = nullptr;
    PFN_vkCmdCopyBufferToImage copyBufferToImage = nullptr;
    PFN_vkCmdBlitImage blit = nullptr;
    PFN_vkCmdFillBuffer fill = nullptr;
    PFN_vkCreateSemaphore createSemaphore = nullptr;
    PFN_vkDestroySemaphore destroySemaphore = nullptr;
    PFN_vkSignalSemaphore signalSemaphore = nullptr;
    int(WINAPI* import)(VkDevice, uint64_t, uint64_t, CudaDevicePtr*, void**) = nullptr;
    int(WINAPI* release)(void*) = nullptr;
};
inline VulkanInterop g_vk;
struct NativeCameraSource
{
    uint64_t buffer, offset, bytes, shader;
    uint32_t width, height;
};
static_assert(sizeof(NativeCameraSource) == 40);
// These handles describe sources already bound by the composition pass. They are
// used only for immediate diagnostic copies on this frame's command list; the
// application's frame fence covers these copies alongside the original reads.
struct NativeImageSource
{
    uint64_t image;
    uint32_t width, height, layout, aspect;
    uint64_t shader;
};
static_assert(sizeof(NativeImageSource) == 32);
// The diagnostic driver exposes the geometry sidecar linked by the game's final
// motion composition pass. This records a separate readback; original inputs stay in use.
void arm_native_capture(uint32_t frame);
bool dump_frame_selected(uint32_t frame, const char* startName, const char* countName, const char* everyName);
inline D4rVkStaging g_fusedColorStage;
// D4R_SHIM_GPU_COMPLETION: the bounded completion-word wait (kernels/support/vulkan_wait.comp)
inline D4rVkStaging g_waitStage;
inline bool g_waitReady = false;
inline constexpr uint32_t kCompletionWord = 16;
// ~1 us per poll of coherent memory: at most about a second, far below the kernel driver's 10 s GPU timeout
inline constexpr uint32_t kCompletionPollLimit = 1u << 20;
inline bool g_fusedColorReady = false;
// --- native D3D12 VRAM (Windows without vkd3d-proton) --------------------------------------------------------------
//
// The VRAM interop route on a plain D3D12 device. Its buffers are shared committed resources HIP imports
// (d4rImportD3D12Memory), and two compute shaders recorded into the game's list (tools/d4r_split_shaders.hlsl) stand
// in for the Vulkan blits: the inputs into their canonical layout, DLSS's result into the output texture. Split frames
// divide the list around DLSS with d4r_d3d12_split.h, the native SplitCommandListForExternalWait.
struct NativeVram
{
    std::once_flag once;
    bool ready = false, split = false;
    // a shared buffer HIP could not import: every later feature stages through host memory (HIP's import does not
    // start working mid-session, and retrying it each frame left the game without a DLSS image)
    std::atomic<bool> importFailed{false};
    using Import = int(WINAPI*)(void*, uint64_t, CudaDevicePtr*, void**);
    Import import = nullptr;
    ID3D12RootSignature* inputRoot = nullptr;
    ID3D12RootSignature* outputRoot = nullptr;
    ID3D12PipelineState* inputPipeline = nullptr;
    ID3D12PipelineState* outputPipeline = nullptr;
    ID3D12DescriptorHeap* heap = nullptr;
    UINT increment = 0;
    std::atomic<uint32_t> next{0};
};
inline NativeVram g_dxv;
// A descriptor ring the GPU never laps: five descriptors a frame.
inline constexpr uint32_t kNativeDescriptors = 1024;
inline constexpr size_t kVramPoolLimit = 64;
// D4R_SHIM_INPLACE (experimental): a game texture that vkd3d-proton created linear and exportable is imported
// into CUDA once, so DLSS writes its texels without a game-side copy. Imports and their source resources
// stay alive for the process lifetime (ROCm does not free mapped external memory, see g_vramPool).
struct InPlaceTexture
{
    CudaDevicePtr texels = 0;
    size_t rowPitch = 0;
};
struct InPlaceImport
{
    uint64_t memory;
    CudaDevicePtr base;
    ID3D12Resource* owner; // retained: prevents reuse of the cached VkDeviceMemory handle
};
// How one input or the output moves between its D3D12 texture and a buffer.
struct VramCopy
{
    ID3D12Resource* resource = nullptr;
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    bool convert = false; // blit to/from the plane's canonical format
    UINT width = 0, height = 0;
};
inline const VkMemoryBarrier kBeforeTransfer = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
                                                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
inline const VkMemoryBarrier kAfterTransfer = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                               VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
// --- Split frames -------------------------------------------------------------------
//
// To present frame N's own result, everything the game records after its DLSS
// call must wait for DLSS. With the d4r vkd3d-proton patch, evaluate() splits
// the game's command list right after the input copies and the frame marker;
// vkd3d-proton submits the second half separately, waiting (at queue submission
// level, so no GPU ring is ever blocked) for splitSemaphore to reach N. The
// worker writes N's result into output slot N % kOutputSlots. Completed and
// dropped frames advance the signal only past earlier retired frames. The
// watchdog reports delays without releasing an unfinished producer.
VkResult signal_split_semaphore(Feature& feature, uint64_t value);
// Pinned staging for memory captures (D4R_SHIM_CAPTURE_POOL_MB, default 3072):
// device readbacks land in it directly, so a capture costs a few ms per frame.
struct CapturePool
{
    uint8_t* base = nullptr;
    size_t size = 0, used = 0;
    bool tried = false;
};
// One frame's evaluated alpha rectangle: where the caller's subrect starts and how much of it
// the denoiser filled. Taken from a published result when a later frame presents it, and from the
// current frame's own capture when a split frame presents the result the worker is about to make.
struct AlphaRegion
{
    UINT baseX = 0, baseY = 0;
    UINT width = 0, height = 0;
};
// Defined with the worker's other synchronization helpers, below; the alpha read-back needs it.
int synchronize_default_stream(Feature& feature, bool* querySleep = nullptr, double* waitEmaUs = nullptr);
// --- Ray Reconstruction: moving the auxiliary surfaces --------------------------

// How long a caller may stop registering a surface before it is treated as dropped. Long
// enough that a game which registers its G-buffer lazily, or only on the frames where it
// changes, keeps it; short enough that a resolution change - which reallocates every resource
// and can reuse a freed address - does not leave a surface bound to somebody else's texture.
inline constexpr uint32_t kAuxStaleFrames = 300;
// DLSS evaluations run as a three-stage pipeline so the CPU copies of one
// frame overlap the GPU work of another:
//   prepare_inputs (prep thread)  waits for the game's input copies and
//                                 converts the readback staging into host planes;
//   run_evaluation (CUDA worker)  uploads them, evaluates, downloads the output;
//   publish_output (finish thread) writes it into an upload staging buffer and
//                                 makes it the result evaluate() presents.
// Every NGX-core and CUDA call stays on the worker. A stage only reuses a
// buffer once the next stage has released it (hostBusy / outputHostBusy).
void publish_output(Feature* feature, int hostIndex, uint32_t frame, FrameParams params, FrameTiming timing);
int claim_output_slot(Feature* feature);
void release_split_frame(Feature* feature, uint32_t frame);
bool complete_ngx_input_wait(Feature& feature, uint32_t frame, bool abort);
// A result whose download is still in flight so the next frame's uploads can
// overlap it. Worker thread only.
struct PendingDownload
{
    Feature* feature = nullptr;
    int hostIndex = 0;
    uint32_t frame = 0;
    FrameParams params;
    FrameTiming timing;
};
bool wait_for_input_marker(Feature* feature, int slotIndex, uint32_t frame);
void release_feature(Feature* feature);
void destroy_recorded_resources(Feature* feature);
void finish_shutdown();
class DeferredResources final : public IUnknown
{
public:
    explicit DeferredResources(Feature* feature) : feature_(feature) { ++g.resourceOwners; }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override
    {
        if (out == nullptr) return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown)) return E_NOINTERFACE;
        *out = static_cast<IUnknown*>(this);
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG remaining = --refs_;
        if (remaining == 0)
            g.cleanup.post([this] {
                destroy_recorded_resources(feature_);
                delete this;
                --g.resourceOwners;
                finish_shutdown();
            });
        return remaining;
    }
private:
    std::atomic<ULONG> refs_{1};
    Feature* feature_;
};
// Native Windows has no vkd3d-proton lifetime interface (RetainExternalResources). The submission hooks
// (d4r_d3d12_submit.h) stand in for it: an object retained for a list is released once d4r's fence passes the
// batch that executed the list, or at once when the list is reset without being executed. A batch the hook could
// not fence is held for a second, far longer than a frame takes.
class NativeRetirer
{
public:
    void retire(IUnknown* object, ID3D12Fence* fence, uint64_t value)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!started_)
            {
                event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                std::thread([this] { run(); }).detach();
                started_ = true;
            }
            queue_.push_back({object, fence, value});
        }
        ready_.notify_one();
    }
private:
    struct Item { IUnknown* object; ID3D12Fence* fence; uint64_t value; };
    void run()
    {
        for (;;)
        {
            Item item;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return !queue_.empty(); });
                item = queue_.front();
                queue_.pop_front();
            }
            // fence values only grow, so waiting in submission order only ever delays a release
            if (item.fence == nullptr)
                Sleep(1000);
            else if (item.fence->GetCompletedValue() < item.value &&
                     SUCCEEDED(item.fence->SetEventOnCompletion(item.value, event_)))
                WaitForSingleObject(event_, INFINITE);
            item.object->Release();
        }
    }
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Item> queue_;
    HANDLE event_ = nullptr;
    bool started_ = false;
};
bool engine_requested();
int start_engine_model_rr(const Feature& feature);
NgxResult fsr_rr_create(Feature& feature, void* parameters);
int rr_engine_slot(uint32_t preset);
// --- Native engine backend (engine/): no NGX evaluation, CUDA, worker or frame split -------------
// [Engine] Enabled in d4r.ini (D4R_ENGINE=1) runs presets K and M with d4r's own shaders inside the game's command
// list. The models (the output of engine/compile_k.py / compile_m.py) are the folders engine\k, engine\k-ldr,
// engine\m and engine\m-ldr next to this DLL, or [Engine] ModelDir (D4R_ENGINE_MODEL_DIR) and the
// D4R_ENGINE_MODEL_{LDR,M,M_LDR}_DIR variables. Anything the engine does not cover falls back to the CUDA backend.
struct EngineBackend
{
    std::once_flag vulkanOnce; // the game's device and the engine's Vulkan entry points, for every preset
    std::once_flag once;
    std::unique_ptr<d4r::Model> model; // shared by all features, kept for the life of the process
    // The same network with NGX's LDR input and output stages, for games that do not set the HDR flag
    // (engine\\k-ldr next to this DLL, or D4R_ENGINE_MODEL_LDR_DIR); loaded on first use.
    std::once_flag ldrOnce;
    std::unique_ptr<d4r::Model> modelLdr;
    std::string directory, directoryLdr; // kept for the native HIP network's hipnet.bin and hip/*.hsaco
};
inline EngineBackend g_engine;
// Presets M and L on the engine (both run d4r::MEngine; the model directory decides which): a model per preset and
// range, engine\\m, engine\\m-ldr, engine\\l and engine\\l-ldr next to this DLL, or D4R_ENGINE_MODEL_{M,L}[_LDR]_DIR.
struct EngineBackendM
{
    std::once_flag once[2][2];                       // [preset L][LDR]
    std::unique_ptr<d4r::MModel> models[2][2];
    std::string directories[2][2];
};
inline EngineBackendM g_engineM;
// Preset M's network on d4r's native HIP layer kernels (bridge: d4rEngineNet*; engine/hip_net.h). HIP runs the
// network about 1.7x faster than the engine's cooperative-matrix shader; the stages around it stay on Vulkan in
// the game's command list, which is split at the network like a CUDA split frame. The ten layers wait on the GPU
// for the front's fill of the frame number (d4rEngineNetInputWaitReady/d4rEngineNetLaunchAt, D4R_SHIM_ENGINE_GPU_WAIT)
// when the bridge supports it; otherwise the worker waits for the CPU-visible marker before launching.
struct EngineNetCode
{
    const char* name;
    const void* data;
    uint64_t bytes;
};
struct EngineNetExports
{
    std::once_flag once;
    int(WINAPI* create)(const uint8_t*, uint64_t, const EngineNetCode*, uint32_t, uint32_t, uint32_t, CudaDevicePtr,
                        CudaDevicePtr, void**) = nullptr;
    int(WINAPI* launch)(void*, uint32_t) = nullptr;
    int(WINAPI* done)(void*, float*) = nullptr;
    void(WINAPI* destroy)(void*) = nullptr;
    // GPU-side readiness: the front fills a u32; the HIP stream waits on it instead of a CPU-visible marker.
    int(WINAPI* inputWaitReady)(void*, CudaDevicePtr, uint32_t*) = nullptr;
    int(WINAPI* launchAt)(void*, uint32_t) = nullptr;
    int(WINAPI* releaseInput)(void*, uint32_t) = nullptr;
    int(WINAPI* setTimeline)(void*, VkDevice, uint64_t) = nullptr;
    int(WINAPI* signalTimeline)(void*, uint64_t) = nullptr;
    bool ready = false;
};
bool wait_for_input_marker(Feature* feature, int slotIndex, uint32_t frame);
void frame_retired(Feature* feature, uint32_t frame);
// Preset K and preset M record a native-network frame the same way: the front half, this frame's marker, the
// split of the game's list, the back half, and the worker that runs the network between the two halves.
// `recordPart(front)` records one half into the game's list; `sources` is {colour, depth, motion, output}, the
// first three sampled through the front. `refuse` (nothing usable recorded; the CUDA backend takes over) and
// `abandon` (recorded but the frame is unusable) belong to the caller so its logging and flags stay preset-specific.
struct EngineFrameSource
{
    ID3D12Resource* resource;
    d4r::GameImage* image;
    unsigned int x, y, width, height;
    D3D12_RESOURCE_STATES state, copyState;
};
template <typename RecordPart, typename Refuse, typename Abandon>
static bool record_engine_network_frame(Feature& feature, ID3D12GraphicsCommandList* list, EngineFrameSource* sources,
                                        uint32_t frame, const char* preset, int reset, float mvScaleX, float mvScaleY,
                                        float jitterX, float jitterY, RecordPart&& recordPart, Refuse&& refuse,
                                        Abandon&& abandon)
{
    // HIP waits on the GPU-published input value between the two Vulkan list parts. Its native completion
    // callback releases the second part. The Wine worker retires resources, not frame timing.
    const int slotIndex = static_cast<int>(frame % kSlots);
    InputSlot& slot = feature.inputs[slotIndex];
    const auto waitStart = std::chrono::steady_clock::now();
    while (slot.busy.load() || feature.inFlight.load() >= 2)
    {
        if (std::chrono::steady_clock::now() - waitStart > std::chrono::seconds(2))
        {
            logf("engine frame %u: %d frames in flight; skipping evaluation", frame, feature.inFlight.load());
            return true;
        }
        d4r_sleep_us(200);
    }
    if (FAILED(g_vk.lifetime->RetainExternalResources(list, feature.resources)))
        return refuse("the command list cannot retain the engine's resources");
    ID3D12GraphicsCommandList2* list2 = nullptr;
    if (FAILED(list->QueryInterface(__uuidof(ID3D12GraphicsCommandList2), reinterpret_cast<void**>(&list2))))
        return refuse("ID3D12GraphicsCommandList2 is unavailable");
    for (int index = 0; index < 3; ++index)
        transition(list, sources[index].resource, sources[index].state, sources[index].copyState);
    const char* failure = recordPart(true);
    if (failure != nullptr)
    {
        for (int index = 0; index < 3; ++index)
            transition(list, sources[index].resource, sources[index].copyState, sources[index].state);
        list2->Release();
        return refuse(failure);
    }
    const D3D12_GPU_VIRTUAL_ADDRESS markerAddress = feature.marker->GetGPUVirtualAddress();
    D3D12_WRITEBUFFERIMMEDIATE_PARAMETER markers[] = {
        {markerAddress + sizeof(uint32_t) * (1 + slotIndex), frame},
        {markerAddress, frame},
    };
    D3D12_WRITEBUFFERIMMEDIATE_MODE modes[] = {
        D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT,
    };
    list2->WriteBufferImmediate(2, markers, modes);
    list2->Release();
    // From here on this call owns the frame: a failure leaves this frame's output undefined and hands the next
    // frame to the CUDA backend (which must not record its own marker and split for this frame).
    if (FAILED(g_vk.split->SplitCommandListForExternalWait(list, reinterpret_cast<UINT64>(feature.splitSemaphore), frame)))
    {
        for (int index = 0; index < 3; ++index)
            transition(list, sources[index].resource, sources[index].copyState, sources[index].state);
        return abandon("the command list could not be split");
    }
    transition(list, sources[3].resource, sources[3].state, sources[3].copyState);
    failure = recordPart(false);
    transition(list, sources[3].resource, sources[3].copyState, sources[3].state);
    // The reconstruction still samples borrowed colour; keep its sampled layout through the back.
    for (int index = 0; index < 3; ++index)
        transition(list, sources[index].resource, sources[index].copyState, sources[index].state);
    feature.split = true;
    slot.busy = true;
    feature.inFlight.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(feature.splitMutex);
        feature.splitCompletion.admit(frame);
    }
    Feature* const owner = &feature;
    g.worker.post([owner, slotIndex, frame] { run_engine_network(owner, slotIndex, frame); });
    if (failure != nullptr)
        return abandon(failure);
    feature.engineJitter[0] = jitterX;
    feature.engineJitter[1] = jitterY;
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded, preset %s with the HIP network (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f)",
             frame, preset, jitterX, jitterY, reset, mvScaleX, mvScaleY);
    return true;
}
// Presets E (5) and F (6), DLSS 3's CNN (engine/runtime_e.inc, built by engine/compile_sr_e.py from
// nvngx_dlss.dll 310.7.0): engine\e and engine\f next to this DLL, or D4R_ENGINE_MODEL_E_DIR / _F_DIR. One model
// serves LDR and HDR, both motion resolutions; its network runs without cooperative matrices (RDNA2 included).
struct EngineBackendE
{
    std::once_flag once[2];
    std::unique_ptr<d4r::EModel> models[2];
};
// Ray Reconstruction models (engine/compile_rr.py; preset F: engine/compile_rr2.py; presets A - C, the CNN of
// nvngx_dlssd.dll 310.3: engine/compile_rr_cnn.py): one per preset, engine\rr-d, rr-e, rr-f, rr-a, rr-b and rr-c next
// to this DLL, or D4R_ENGINE_MODEL_RR_D_DIR / _E_DIR / _F_DIR / _A_DIR / _B_DIR / _C_DIR. Slots: 0 D, 1 E, 2 F, 3 A,
// 4 B, 5 C.
struct EngineBackendR
{
    // The model's pipelines take minutes to compile on a cold shader cache, so each preset's model loads on a thread
    // of its own; until it is ready the CUDA backend serves the frames. state: 0 not started, 1 loading, 2 ready,
    // 3 failed.
    std::once_flag once[6];
    std::atomic<int> state[6] = {};
    std::unique_ptr<d4r::RModel> models[6];
};
inline EngineBackendR g_engineR;
bool engine_rr_vkhip_capable(int slot);
inline const char g_rrLetter[6] = {'D', 'E', 'F', 'A', 'B', 'C'};
// NGX's K configuration against what a K model package supports (either engine's Model): the refusal, or null.
// NVSDK_NGX_DLSS_Feature_Flags: IsHDR 1, MVLowRes 2, MVJittered 4, DepthInverted 8, AutoExposure 0x40.
template <class KModel>
static const char* engine_k_refusal(const KModel& model, const FrameParams& p, bool ldr, bool displayMotion,
                                    bool autoExposure, bool depthInverted)
{
    if (displayMotion && !model.hasDisplayMotion())
        return "display-resolution motion needs a model compiled by the current compile_k.py";
    if (!depthInverted && !model.hasRegularDepth())
        return "regular (non-inverted) depth needs a model compiled by the current compile_k.py";
    // DLSS.Indicator.Invert.X/Y.Axis (p.invertX/Y) only place NGX's debug indicator: NGX's K and M outputs are
    // byte-identical with them set (harness D4R_HARNESS_INDICATOR_INVERT=1), so neither engine path reads them.
    // Two exposure sources. AutoExposure (0x40) measures the frame's luma scaled by 1 / pre-exposure and multiplies
    // the result by the exposure scale. Without it the game's ExposureTexture is the exposure: texel x scale /
    // pre-exposure (read directly, not measured).
    const bool gameExposure = !ldr && !autoExposure;
    if (!ldr && autoExposure && !model.hasAutoExposure())
        return "automatic exposure needs a model with exposure shaders";
    if (gameExposure && (!p.hasExposure || !model.hasGameExposure()))
        return "exposure without AutoExposure needs an exposure texture and a model compiled by the current compile_k.py";
    if (!ldr && !(std::isfinite(p.exposureScale) && p.exposureScale > 0.0f))
        return "the exposure scale is not a positive number";
    if (!ldr && !(std::isfinite(p.preExposure) && p.preExposure > 0.0f))
        return "the pre-exposure is not a positive number";
    return nullptr;
}
// --- Native D3D12 engine (engine/d3d12): preset K on Windows, without vkd3d-proton ----------------------------
// The same model directories as the Vulkan engine (engine\k, engine\k-ldr; their .dxil shaders come from
// compile_k.py on Windows), loaded on the game's own D3D12 device. Evaluation records into the game's command list
// with no CUDA, ZLUDA, HIP or Vulkan; the submission hooks tell the engine when a recording ran.
struct EngineBackendDx
{
    std::once_flag once, ldrOnce, hooksOnce;
    std::unique_ptr<d4r::dx::Model> model, modelLdr;
    std::once_flag mOnce[2][2];
    std::unique_ptr<d4r::dx::MModel> modelM[2][2]; // presets M and L: [L][ldr] (engine\m, engine\m-ldr, engine\l, engine\l-ldr)
    std::once_flag rrOnce[3];
    std::unique_ptr<d4r::dx::RModel> rrModel[3];
    std::unique_ptr<d4r::dx::rdna3::MModel> modelMRdna3[2][2]; // the same packages' rdna3/ shaders, when the FP8 WMMA ones do not load
    bool hooks = false;
    std::mutex retiredMutex;
    std::vector<std::unique_ptr<EngineDxFeature>> retired; // released features whose frames have not retired yet
};
template <size_t N> static constexpr uint32_t stage_count(const char* const (&)[N]) { return uint32_t(N); }
// The engine's boundaries, offset by `base` (the shim's own boundary 0 precedes the adapter's entry barriers)
struct EngineDxMarks
{
    EngineDxFeature* state;
    int slot;
    uint32_t base;
    d4r::dx::FrameProfiler profiler(uint32_t offset)
    {
        base = offset;
        if (slot < 0) return {};
        return {[](ID3D12GraphicsCommandList* list, uint32_t boundary, void* user) {
            auto& m = *static_cast<EngineDxMarks*>(user);
            engine_dx_mark(*m.state, m.slot, list, m.base + boundary);
        }, this};
    }
};
NgxResult evaluate_unlocked(ID3D12GraphicsCommandList* list, const NgxHandle* handle, void* parameters);
void fsr_rr_release(Feature& feature);
void fsr_rr_destroy(Feature& feature);
// The DLL entry point (linked with --entry). At process termination (reserved != NULL) ExitProcess
// has already killed the worker and waiter threads, possibly inside winpthreads' global condition
// variable spinlock, so the CRT's static destructors (mutexes, condition variables) can spin forever
// and the game never exits. Release the GPU waits and skip all CRT teardown then, as Windows advises;
// the system reclaims everything. Every other notification goes through the CRT as usual.
extern "C" BOOL WINAPI DllMainCRTStartup(HANDLE, DWORD, LPVOID);

#endif // D4R_NVNGX_SHIM_INTERNAL_H