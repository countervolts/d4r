#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_4.h>
#include "nvsdk_ngx.h"
#include "ngx_param_abi_probe_shared.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

extern "C" void d4r_ngx_parameter_set_ull(void*, const char*, unsigned long long);
extern "C" void d4r_ngx_parameter_set_float(void*, const char*, float);
extern "C" void d4r_ngx_parameter_set_double(void*, const char*, double);
extern "C" void d4r_ngx_parameter_set_uint(void*, const char*, unsigned int);
extern "C" void d4r_ngx_parameter_set_int(void*, const char*, int);
extern "C" void d4r_ngx_parameter_set_void(void*, const char*, void*);
extern "C" unsigned int d4r_ngx_parameter_get_ull(void*, const char*, unsigned long long*);
extern "C" unsigned int d4r_ngx_parameter_get_float(void*, const char*, float*);
extern "C" unsigned int d4r_ngx_parameter_get_double(void*, const char*, double*);
extern "C" unsigned int d4r_ngx_parameter_get_uint(void*, const char*, unsigned int*);
extern "C" unsigned int d4r_ngx_parameter_get_int(void*, const char*, int*);
extern "C" unsigned int d4r_ngx_parameter_get_void(void*, const char*, void**);
extern "C" unsigned int d4r_ngx_parameter_float_roundtrip(void*, const char*, float, float*);
extern "C" unsigned int d4r_ngx_parameter_uint_roundtrip(void*, const char*, unsigned int, unsigned int*);
extern "C" unsigned int d4r_ngx_parameter_proxy_size();
extern "C" void d4r_ngx_parameter_proxy_construct(void*, void*);
extern "C" unsigned int d4r_ngx_parameter_proxy_count(const void*);
extern "C" unsigned int d4r_ngx_parameter_proxy_dropped(const void*);
extern "C" int d4r_ngx_parameter_proxy_copy_record(
    const void*, unsigned int, D4rNgxParamLogRecord*);

template <typename Target, typename Source> Target copy_function_pointer(Source source)
{
    static_assert(sizeof(Target) == sizeof(Source), "function pointer sizes differ");
    Target target{};
    std::memcpy(&target, &source, sizeof(target));
    return target;
}

template <typename Function> Function load_export(HMODULE module, const char* name)
{
    FARPROC raw = GetProcAddress(module, name);
    return raw ? copy_function_pointer<Function>(raw) : nullptr;
}

static std::wstring utf8_to_wide(const char* input)
{
    const int required = MultiByteToWideChar(CP_UTF8, 0, input, -1, nullptr, 0);
    if (required <= 1)
        return {};
    std::wstring output(static_cast<size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, input, -1, output.data(), required);
    output.resize(static_cast<size_t>(required - 1));
    return output;
}

static const char* ngx_result_name(NVSDK_NGX_Result result)
{
    switch (result)
    {
    case NVSDK_NGX_Result_Success:
        return "Success";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported:
        return "FeatureNotSupported";
    case NVSDK_NGX_Result_FAIL_PlatformError:
        return "PlatformError";
    case NVSDK_NGX_Result_FAIL_InvalidParameter:
        return "InvalidParameter";
    case NVSDK_NGX_Result_FAIL_NotInitialized:
        return "NotInitialized";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:
        return "UnableToInitializeFeature";
    case NVSDK_NGX_Result_FAIL_OutOfDate:
        return "OutOfDate";
    default:
        return "Other";
    }
}

static void print_ngx_result(const char* stage, NVSDK_NGX_Result result)
{
    std::printf("%s: result=0x%08x (%s)\n", stage, static_cast<unsigned int>(result), ngx_result_name(result));
}

using D4rCudaDevicePtr = unsigned long long;
using D4rCudaArray = void*;
using D4rCudaTextureObject = unsigned long long;
using D4rCuMemAlloc = int(WINAPI*)(D4rCudaDevicePtr*, size_t);
using D4rCuMemFree = int(WINAPI*)(D4rCudaDevicePtr);
using D4rCuMemcpyHtoDAsync = int(WINAPI*)(D4rCudaDevicePtr, const void*, size_t, void*);
using D4rCuMemcpyDtoH = int(WINAPI*)(void*, D4rCudaDevicePtr, size_t);
using D4rCuCtxSynchronize = int(WINAPI*)();

struct D4rCudaArrayDescriptor
{
    size_t Width;
    size_t Height;
    uint32_t Format;
    uint32_t NumChannels;
};
static_assert(sizeof(D4rCudaArrayDescriptor) == 24);

struct D4rCudaArray3DDescriptor
{
    size_t Width;
    size_t Height;
    size_t Depth;
    uint32_t Format;
    uint32_t NumChannels;
    uint32_t Flags;
};
static_assert(sizeof(D4rCudaArray3DDescriptor) == 40);

struct D4rCudaMemcpy2D
{
    size_t srcXInBytes;
    size_t srcY;
    uint32_t srcMemoryType;
    uint32_t srcAlignment;
    const void* srcHost;
    D4rCudaDevicePtr srcDevice;
    D4rCudaArray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    uint32_t dstMemoryType;
    uint32_t dstAlignment;
    void* dstHost;
    D4rCudaDevicePtr dstDevice;
    D4rCudaArray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};
static_assert(sizeof(D4rCudaMemcpy2D) == 128);

struct D4rCudaResourceDesc
{
    uint32_t resType;
    uint32_t alignment;
    union
    {
        struct { D4rCudaArray hArray; } array;
        int reserved[32];
    } res;
    uint32_t flags;
    uint32_t reserved;
};
static_assert(sizeof(D4rCudaResourceDesc) == 144);

struct D4rCudaTextureDesc
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
static_assert(sizeof(D4rCudaTextureDesc) == 104);

using D4rCuArrayCreate = int(WINAPI*)(D4rCudaArray*, const D4rCudaArrayDescriptor*);
using D4rCuArray3DCreate = int(WINAPI*)(D4rCudaArray*, const D4rCudaArray3DDescriptor*);
using D4rCuArrayDestroy = int(WINAPI*)(D4rCudaArray);
using D4rCuMemcpy2D = int(WINAPI*)(const D4rCudaMemcpy2D*);
using D4rCuTexObjectCreate = int(WINAPI*)(D4rCudaTextureObject*, const D4rCudaResourceDesc*,
                                          const D4rCudaTextureDesc*, const void*);
using D4rCuTexObjectDestroy = int(WINAPI*)(D4rCudaTextureObject);
using D4rCuTexObjectGetResourceDesc = int(WINAPI*)(D4rCudaResourceDesc*, D4rCudaTextureObject);
using D4rCudaSurfaceObject = unsigned long long;
using D4rCuSurfObjectCreate = int(WINAPI*)(D4rCudaSurfaceObject*, const D4rCudaResourceDesc*);
using D4rCuSurfObjectDestroy = int(WINAPI*)(D4rCudaSurfaceObject);
using D4rCuSurfObjectGetResourceDesc = int(WINAPI*)(D4rCudaResourceDesc*, D4rCudaSurfaceObject);
using D4rNgxGetParameters = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);

struct NgxParameterApi
{
    PFN_NVSDK_NGX_Parameter_SetULL setULL = nullptr;
    PFN_NVSDK_NGX_Parameter_SetF setF = nullptr;
    PFN_NVSDK_NGX_Parameter_SetD setD = nullptr;
    PFN_NVSDK_NGX_Parameter_SetUI setUI = nullptr;
    PFN_NVSDK_NGX_Parameter_SetI setI = nullptr;
    PFN_NVSDK_NGX_Parameter_SetVoidPointer setVoidPointer = nullptr;
    PFN_NVSDK_NGX_Parameter_GetULL getULL = nullptr;
    PFN_NVSDK_NGX_Parameter_GetF getF = nullptr;
    PFN_NVSDK_NGX_Parameter_GetD getD = nullptr;
    PFN_NVSDK_NGX_Parameter_GetUI getUI = nullptr;
    PFN_NVSDK_NGX_Parameter_GetI getI = nullptr;
    PFN_NVSDK_NGX_Parameter_GetVoidPointer getVoidPointer = nullptr;
};

static NgxParameterApi g_ngxParameterApi;

static void ngx_set(NVSDK_NGX_Parameter* parameters, const char* name, unsigned long long value)
{
    if (g_ngxParameterApi.setULL != nullptr)
        g_ngxParameterApi.setULL(parameters, name, value);
    else
        d4r_ngx_parameter_set_ull(parameters, name, value);
}
static void ngx_set(NVSDK_NGX_Parameter* parameters, const char* name, float value)
{
    if (g_ngxParameterApi.setF != nullptr)
        g_ngxParameterApi.setF(parameters, name, value);
    else
        d4r_ngx_parameter_set_float(parameters, name, value);
}
static void ngx_set(NVSDK_NGX_Parameter* parameters, const char* name, double value)
{
    if (g_ngxParameterApi.setD != nullptr)
        g_ngxParameterApi.setD(parameters, name, value);
    else
        d4r_ngx_parameter_set_double(parameters, name, value);
}
static void ngx_set(NVSDK_NGX_Parameter* parameters, const char* name, unsigned int value)
{
    if (g_ngxParameterApi.setUI != nullptr)
        g_ngxParameterApi.setUI(parameters, name, value);
    else
        d4r_ngx_parameter_set_uint(parameters, name, value);
}
static void ngx_set(NVSDK_NGX_Parameter* parameters, const char* name, int value)
{
    if (g_ngxParameterApi.setI != nullptr)
        g_ngxParameterApi.setI(parameters, name, value);
    else
        d4r_ngx_parameter_set_int(parameters, name, value);
}
static void ngx_set(NVSDK_NGX_Parameter* parameters, const char* name, void* value)
{
    if (g_ngxParameterApi.setVoidPointer != nullptr)
        g_ngxParameterApi.setVoidPointer(parameters, name, value);
    else
        d4r_ngx_parameter_set_void(parameters, name, value);
}
static NVSDK_NGX_Result ngx_get_ull(NVSDK_NGX_Parameter* parameters, const char* name,
                                    unsigned long long* value)
{
    if (g_ngxParameterApi.getULL != nullptr)
        return g_ngxParameterApi.getULL(parameters, name, value);
    return static_cast<NVSDK_NGX_Result>(d4r_ngx_parameter_get_ull(parameters, name, value));
}
static NVSDK_NGX_Result ngx_get_float(NVSDK_NGX_Parameter* parameters, const char* name, float* value)
{
    if (g_ngxParameterApi.getF != nullptr)
        return g_ngxParameterApi.getF(parameters, name, value);
    return static_cast<NVSDK_NGX_Result>(d4r_ngx_parameter_get_float(parameters, name, value));
}
static NVSDK_NGX_Result ngx_get_double(NVSDK_NGX_Parameter* parameters, const char* name, double* value)
{
    if (g_ngxParameterApi.getD != nullptr)
        return g_ngxParameterApi.getD(parameters, name, value);
    return static_cast<NVSDK_NGX_Result>(d4r_ngx_parameter_get_double(parameters, name, value));
}
static NVSDK_NGX_Result ngx_get_ui(NVSDK_NGX_Parameter* parameters, const char* name,
                                   unsigned int* value)
{
    if (g_ngxParameterApi.getUI != nullptr)
        return g_ngxParameterApi.getUI(parameters, name, value);
    return static_cast<NVSDK_NGX_Result>(d4r_ngx_parameter_get_uint(parameters, name, value));
}
static NVSDK_NGX_Result ngx_get_i(NVSDK_NGX_Parameter* parameters, const char* name, int* value)
{
    if (g_ngxParameterApi.getI != nullptr)
        return g_ngxParameterApi.getI(parameters, name, value);
    return static_cast<NVSDK_NGX_Result>(d4r_ngx_parameter_get_int(parameters, name, value));
}
static NVSDK_NGX_Result ngx_get_void(NVSDK_NGX_Parameter* parameters, const char* name, void** value)
{
    if (g_ngxParameterApi.getVoidPointer != nullptr)
        return g_ngxParameterApi.getVoidPointer(parameters, name, value);
    return static_cast<NVSDK_NGX_Result>(d4r_ngx_parameter_get_void(parameters, name, value));
}
static void* ngx_pointer_from_handle(uint64_t handle)
{
    return reinterpret_cast<void*>(static_cast<uintptr_t>(handle));
}

static float half_to_float(uint16_t half)
{
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    const uint32_t exponent = (half >> 10) & 0x1fu;
    const uint32_t mantissa = half & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0)
    {
        if (mantissa == 0)
            bits = sign;
        else
        {
            const float subnormal = std::ldexp(static_cast<float>(mantissa), -24);
            return (sign != 0 ? -subnormal : subnormal);
        }
    }
    else if (exponent == 0x1fu)
        bits = sign | 0x7f800000u | (mantissa << 13);
    else
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool write_file(const std::wstring& path, const void* data, size_t bytes)
{
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return false;
    const auto* source = static_cast<const unsigned char*>(data);
    size_t offset = 0;
    bool success = true;
    while (offset < bytes)
    {
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(bytes - offset, 0x7fffffffu));
        DWORD written = 0;
        if (!WriteFile(file, source + offset, chunk, &written, nullptr) || written != chunk)
        {
            success = false;
            break;
        }
        offset += written;
    }
    CloseHandle(file);
    return success;
}

static void evaluate_synthetic_frame(NVSDK_NGX_Parameter* parameters,
                                     const NVSDK_NGX_Handle* featureHandle,
                                     decltype(&NVSDK_NGX_CUDA_EvaluateFeature) evaluateFeature,
                                     decltype(&NVSDK_NGX_CUDA_EvaluateFeature_C) evaluateFeatureC,
                                     bool useCEvaluate,
                                     const std::wstring& appData,
                                     D4rCudaDevicePtr scratchDevice,
                                     size_t scratchBytes)
{
    constexpr unsigned int inputWidth = 640;
    constexpr unsigned int inputHeight = 360;
    constexpr unsigned int outputWidth = 1280;
    constexpr unsigned int outputHeight = 720;
    const size_t inputPixels = static_cast<size_t>(inputWidth) * inputHeight;
    const size_t outputPixels = static_cast<size_t>(outputWidth) * outputHeight;
    const size_t colorBytes = inputPixels * 4 * sizeof(uint16_t);
    const size_t depthBytes = inputPixels * sizeof(float);
    const size_t motionBytes = inputPixels * 2 * sizeof(float);
    const size_t outputBytes = outputPixels * 4 * sizeof(uint16_t);

    HMODULE cuda = LoadLibraryA("nvcuda.dll");
    if (cuda == nullptr)
    {
        std::printf("Synthetic DLSS evaluate: LoadLibraryA(nvcuda.dll) failed: %lu\n", GetLastError());
        return;
    }
    const auto alloc = load_export<D4rCuMemAlloc>(cuda, "cuMemAlloc");
    const auto free = load_export<D4rCuMemFree>(cuda, "cuMemFree");
    const auto copyHtoD = load_export<D4rCuMemcpyHtoDAsync>(cuda, "cuMemcpyHtoDAsync");
    const auto copyDtoH = load_export<D4rCuMemcpyDtoH>(cuda, "cuMemcpyDtoH");
    const auto synchronize = load_export<D4rCuCtxSynchronize>(cuda, "cuCtxSynchronize");
    const auto arrayCreate = load_export<D4rCuArrayCreate>(cuda, "cuArrayCreate");
    const auto arrayDestroy = load_export<D4rCuArrayDestroy>(cuda, "cuArrayDestroy");
    const auto copy2D = load_export<D4rCuMemcpy2D>(cuda, "cuMemcpy2D");
    const auto textureCreate = load_export<D4rCuTexObjectCreate>(cuda, "cuTexObjectCreate");
    const auto textureDestroy = load_export<D4rCuTexObjectDestroy>(cuda, "cuTexObjectDestroy");
    const auto textureGetResourceDesc = load_export<D4rCuTexObjectGetResourceDesc>(
        cuda, "cuTexObjectGetResourceDesc");
    const auto array3DCreate = load_export<D4rCuArray3DCreate>(cuda, "cuArray3DCreate");
    const auto surfaceCreate = load_export<D4rCuSurfObjectCreate>(cuda, "cuSurfObjectCreate");
    const auto surfaceDestroy = load_export<D4rCuSurfObjectDestroy>(cuda, "cuSurfObjectDestroy");
    const auto surfaceGetResourceDesc = load_export<D4rCuSurfObjectGetResourceDesc>(
        cuda, "cuSurfObjectGetResourceDesc");
    const char* resourceMode = std::getenv("D4R_NGX_RESOURCE_MODE");
    const bool textureMode = resourceMode != nullptr &&
        (std::strcmp(resourceMode, "texture") == 0 || std::strcmp(resourceMode, "surface") == 0);
    const bool surfaceMode = resourceMode != nullptr && std::strcmp(resourceMode, "surface") == 0;
    if (alloc == nullptr || free == nullptr || copyHtoD == nullptr || copyDtoH == nullptr || synchronize == nullptr)
    {
        std::printf("Synthetic DLSS evaluate: CUDA driver bridge is missing a required memory export\n");
        FreeLibrary(cuda);
        return;
    }
    if (textureMode && (arrayCreate == nullptr || arrayDestroy == nullptr || copy2D == nullptr ||
                        textureCreate == nullptr || textureDestroy == nullptr))
    {
        std::printf("Synthetic DLSS evaluate: CUDA array/texture bridge exports are missing\n");
        FreeLibrary(cuda);
        return;
    }
    if (surfaceMode && (array3DCreate == nullptr || surfaceCreate == nullptr || surfaceDestroy == nullptr))
    {
        std::printf("Synthetic DLSS evaluate: CUDA surface bridge exports are missing\n");
        FreeLibrary(cuda);
        return;
    }

    std::vector<uint16_t> color(inputPixels * 4);
    std::vector<float> depth(inputPixels, 0.5f);
    std::vector<float> motion(inputPixels * 2, 0.0f);
    std::vector<uint16_t> output(outputPixels * 4, 0);

    constexpr uint16_t dark = 0x2a66u; // half 0.05
    constexpr uint16_t mid = 0x3400u;  // half 0.25
    constexpr uint16_t bright = 0x3a66u; // half 0.8
    constexpr uint16_t white = 0x3c00u; // half 1.0
    for (unsigned int y = 0; y < inputHeight; ++y)
    {
        for (unsigned int x = 0; x < inputWidth; ++x)
        {
            const size_t pixel = static_cast<size_t>(y) * inputWidth + x;
            const bool fence = (x % 24u == 0u) || (y % 24u == 0u);
            const bool checker = (((x / 4u) + (y / 4u)) & 1u) != 0;
            color[pixel * 4 + 0] = fence ? white : (checker ? mid : dark);
            color[pixel * 4 + 1] = ((x + y) % 32u < 2u) ? bright : (checker ? dark : mid);
            color[pixel * 4 + 2] = (((x / 8u) ^ (y / 8u)) & 1u) ? bright : dark;
            color[pixel * 4 + 3] = white;
        }
    }

    D4rCudaArray colorArray = nullptr;
    D4rCudaArray depthArray = nullptr;
    D4rCudaArray motionArray = nullptr;
    D4rCudaArray exposureArray = nullptr;
    D4rCudaArray outputArray = nullptr;
    D4rCudaTextureObject colorTexture = 0;
    D4rCudaTextureObject depthTexture = 0;
    D4rCudaTextureObject motionTexture = 0;
    D4rCudaTextureObject exposureTexture = 0;
    D4rCudaSurfaceObject outputSurface = 0;
    int textureSetupResult = 0;
    if (textureMode)
    {
        auto create_and_upload_array = [&](const char* label, size_t width, size_t height,
                                           uint32_t format, uint32_t channels, const void* source,
                                           size_t pitch, D4rCudaArray& array) {
            const D4rCudaArrayDescriptor descriptor{width, height, format, channels};
            int result = arrayCreate(&array, &descriptor);
            std::printf("Synthetic DLSS eval cuArrayCreate(%s): result=%d array=%p\n", label, result, array);
            if (result != 0)
                return result;
            D4rCudaMemcpy2D copy{};
            copy.srcMemoryType = 1; // CUDA_MEMORYTYPE_HOST
            copy.srcHost = source;
            copy.srcPitch = pitch;
            copy.dstMemoryType = 3; // CUDA_MEMORYTYPE_ARRAY
            copy.dstArray = array;
            copy.WidthInBytes = pitch;
            copy.Height = height;
            result = copy2D(&copy);
            std::printf("Synthetic DLSS eval array upload(%s): result=%d bytesPerRow=%zu\n",
                        label, result, pitch);
            return result;
        };
        auto create_texture = [&](const char* label, D4rCudaArray array, uint32_t filterMode,
                                  D4rCudaTextureObject& texture) {
            D4rCudaResourceDesc resource{};
            resource.resType = 0; // CUDA_RESOURCE_TYPE_ARRAY
            resource.res.array.hArray = array;
            D4rCudaTextureDesc sampler{};
            sampler.addressMode[0] = 1; // clamp
            sampler.addressMode[1] = 1;
            sampler.addressMode[2] = 0;
            sampler.filterMode = filterMode;
            sampler.flags = 2; // CU_TRSF_NORMALIZED_COORDINATES, matching NGX-created textures
            sampler.maxAnisotropy = 0;
            int result = textureCreate(&texture, &resource, &sampler, nullptr);
            std::printf("Synthetic DLSS eval cuTexObjectCreate(%s): result=%d object=0x%llx\n",
                        label, result, texture);
            return result;
        };

        const std::vector<uint16_t> motionHalf(inputPixels * 2, 0);
        const float exposureValue = 1.0f;
        textureSetupResult = create_and_upload_array(
            "Color RGBA16F", inputWidth, inputHeight, 16, 4, color.data(),
            inputWidth * 4 * sizeof(uint16_t), colorArray);
        if (textureSetupResult == 0)
            textureSetupResult = create_and_upload_array(
                "Depth R32F", inputWidth, inputHeight, 32, 1, depth.data(),
                inputWidth * sizeof(float), depthArray);
        if (textureSetupResult == 0)
            textureSetupResult = create_and_upload_array(
                "Motion RG16F", inputWidth, inputHeight, 16, 2, motionHalf.data(),
                inputWidth * 2 * sizeof(uint16_t), motionArray);
        if (textureSetupResult == 0)
            textureSetupResult = create_and_upload_array(
                "Exposure R32F", 1, 1, 32, 1, &exposureValue,
                sizeof(exposureValue), exposureArray);
        if (textureSetupResult == 0)
            textureSetupResult = create_texture("Color", colorArray, 1, colorTexture);
        if (textureSetupResult == 0)
            textureSetupResult = create_texture("Depth", depthArray, 0, depthTexture);
        if (textureSetupResult == 0)
            textureSetupResult = create_texture("MotionVectors", motionArray, 0, motionTexture);
        if (textureSetupResult == 0)
            textureSetupResult = create_texture("Exposure", exposureArray, 0, exposureTexture);
        if (textureSetupResult == 0 && surfaceMode)
        {
            const D4rCudaArray3DDescriptor descriptor{
                outputWidth, outputHeight, 0, 16, 4, 2u // RGBA16F with CUDA_ARRAY3D_SURFACE_LDST
            };
            textureSetupResult = array3DCreate(&outputArray, &descriptor);
            std::printf("Synthetic DLSS eval cuArray3DCreate(Output RGBA16F): result=%d array=%p\n",
                        textureSetupResult, outputArray);
            if (textureSetupResult == 0)
            {
                D4rCudaResourceDesc resource{};
                resource.resType = 0; // CUDA_RESOURCE_TYPE_ARRAY
                resource.res.array.hArray = outputArray;
                textureSetupResult = surfaceCreate(&outputSurface, &resource);
                std::printf("Synthetic DLSS eval cuSurfObjectCreate(Output): result=%d object=0x%llx\n",
                            textureSetupResult, outputSurface);
            }
        }
        if (textureSetupResult == 0 && textureMode)
        {
            D4rCudaResourceDesc descriptor{};
            const int descriptorResult = textureGetResourceDesc != nullptr
                ? textureGetResourceDesc(&descriptor, colorTexture) : 801;
            std::printf("Synthetic DLSS cuTexObjectGetResourceDesc(Color): result=%d type=%u array=%p\n",
                        descriptorResult, descriptor.resType, descriptor.res.array.hArray);
        }
        if (textureSetupResult == 0 && surfaceMode)
        {
            D4rCudaResourceDesc descriptor{};
            const int descriptorResult = surfaceGetResourceDesc != nullptr
                ? surfaceGetResourceDesc(&descriptor, outputSurface) : 801;
            std::printf("Synthetic DLSS cuSurfObjectGetResourceDesc(Output): result=%d type=%u array=%p\n",
                        descriptorResult, descriptor.resType, descriptor.res.array.hArray);
        }
        std::printf("Synthetic DLSS texture resource setup: CUDA result=%d\n", textureSetupResult);
    }

    D4rCudaDevicePtr colorDevice = 0;
    D4rCudaDevicePtr depthDevice = 0;
    D4rCudaDevicePtr motionDevice = 0;
    D4rCudaDevicePtr outputDevice = 0;
    auto allocate_and_upload = [&](const char* label, D4rCudaDevicePtr& device,
                                   const void* source, size_t bytes) {
        int result = alloc(&device, bytes);
        std::printf("Synthetic DLSS eval cuMemAlloc(%s): result=%d ptr=0x%llx bytes=%zu\n",
                    label, result, device, bytes);
        if (result == 0)
            result = copyHtoD(device, source, bytes, nullptr);
        std::printf("Synthetic DLSS eval upload(%s): result=%d\n", label, result);
        return result;
    };

    int cudaResult = textureMode ? textureSetupResult : 0;
    if (cudaResult == 0 && !textureMode)
        cudaResult = allocate_and_upload("Color", colorDevice, color.data(), colorBytes);
    if (cudaResult == 0 && !textureMode)
        cudaResult = allocate_and_upload("Depth", depthDevice, depth.data(), depthBytes);
    if (cudaResult == 0 && !textureMode)
        cudaResult = allocate_and_upload("MotionVectors", motionDevice, motion.data(), motionBytes);
    if (cudaResult == 0 && !surfaceMode)
        cudaResult = allocate_and_upload("Output", outputDevice, output.data(), outputBytes);
    if (cudaResult == 0)
        cudaResult = synchronize();
    std::printf("Synthetic DLSS eval input upload/sync: CUDA result=%d\n", cudaResult);

    if (cudaResult == 0)
    {
        if (scratchDevice != 0 && scratchBytes != 0)
        {
            ngx_set(parameters, NVSDK_NGX_Parameter_Scratch,
                            ngx_pointer_from_handle(scratchDevice));
            ngx_set(parameters, NVSDK_NGX_Parameter_Scratch_SizeInBytes,
                            static_cast<unsigned long long>(scratchBytes));
        }
        ngx_set(parameters, NVSDK_NGX_EParameter_EvaluationNode, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_Width, inputWidth);
        ngx_set(parameters, NVSDK_NGX_Parameter_Height, inputHeight);
        ngx_set(parameters, NVSDK_NGX_Parameter_OutWidth, outputWidth);
        ngx_set(parameters, NVSDK_NGX_Parameter_OutHeight, outputHeight);
        ngx_set(parameters, NVSDK_NGX_Parameter_PerfQualityValue,
                static_cast<int>(NVSDK_NGX_PerfQuality_Value_MaxQuality));
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                static_cast<int>(NVSDK_NGX_DLSS_Feature_Flags_None));
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
        void* colorResource = textureMode
            ? static_cast<void*>(&colorTexture) : static_cast<void*>(&colorDevice);
        void* depthResource = textureMode
            ? static_cast<void*>(&depthTexture) : static_cast<void*>(&depthDevice);
        void* motionResource = textureMode
            ? static_cast<void*>(&motionTexture) : static_cast<void*>(&motionDevice);
        void* exposureResource = textureMode
            ? static_cast<void*>(&exposureTexture) : static_cast<void*>(nullptr);
        void* outputResource = surfaceMode
            ? static_cast<void*>(&outputSurface) : static_cast<void*>(&outputDevice);
        ngx_set(parameters, NVSDK_NGX_Parameter_Color, colorResource);
        ngx_set(parameters, NVSDK_NGX_Parameter_Color_Format, static_cast<int>(NVSDK_NGX_Buffer_Format_RGBA16F));
        ngx_set(parameters, NVSDK_NGX_Parameter_Color_SizeInBytes, static_cast<unsigned long long>(colorBytes));
        ngx_set(parameters, NVSDK_NGX_Parameter_Output, outputResource);
        if (!surfaceMode)
        {
            ngx_set(parameters, NVSDK_NGX_Parameter_Output_Format,
                            static_cast<int>(NVSDK_NGX_Buffer_Format_RGBA16F));
            ngx_set(parameters, NVSDK_NGX_Parameter_Output_SizeInBytes,
                            static_cast<unsigned long long>(outputBytes));
        }
        ngx_set(parameters, NVSDK_NGX_Parameter_Depth, depthResource);
        ngx_set(parameters, NVSDK_NGX_Parameter_MotionVectors, motionResource);
        ngx_set(parameters, NVSDK_NGX_Parameter_Sharpness, 0.0f);
        ngx_set(parameters, NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
        ngx_set(parameters, NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
        ngx_set(parameters, NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
        ngx_set(parameters, NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
        ngx_set(parameters, NVSDK_NGX_Parameter_Reset, 1);
        ngx_set(parameters, NVSDK_NGX_Parameter_TransparencyMask, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_ExposureTexture, exposureResource);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Albedo, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Roughness, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Metallic, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Specular, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Subsurface, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Normals, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_ShadingModelId, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_MaterialId, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Atrrib_8, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Atrrib_9, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Atrrib_10, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Atrrib_11, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Atrrib_12, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Atrrib_13, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Atrrib_14, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_GBuffer_Atrrib_15, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_TonemapperType, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_MotionVectors3D, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_IsParticleMask, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_AnimatedTextureMask, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_DepthHighRes, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_Position_ViewSpace, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_FrameTimeDeltaInMsec, 16.666667f);
        ngx_set(parameters, NVSDK_NGX_Parameter_RayTracingHitDistance, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_MotionVectorsReflection, static_cast<void*>(nullptr));
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_X, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_Y, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_X, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_Y, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, 0u);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, inputWidth);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, inputHeight);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis, 0);
        ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis, 0);

        const char* doubleFloatTest = std::getenv("D4R_NGX_DOUBLE_FLOAT_TEST");
        if (doubleFloatTest != nullptr && std::strcmp(doubleFloatTest, "1") == 0)
            ngx_set(parameters, NVSDK_NGX_Parameter_MV_Scale_X, static_cast<double>(1.0));

        void* checkColor = nullptr;
        unsigned long long checkColorAsUll = 0;
        float checkMvScale = 0.0f;
        double checkMvScaleDouble = 0.0;
        const NVSDK_NGX_Result colorGet = ngx_get_void(
            parameters, NVSDK_NGX_Parameter_Color, &checkColor);
        const NVSDK_NGX_Result colorUllGet = ngx_get_ull(
            parameters, NVSDK_NGX_Parameter_Color, &checkColorAsUll);
        const NVSDK_NGX_Result mvScaleGet = ngx_get_float(
            parameters, NVSDK_NGX_Parameter_MV_Scale_X, &checkMvScale);
        const NVSDK_NGX_Result mvScaleDoubleGet = ngx_get_double(
            parameters, NVSDK_NGX_Parameter_MV_Scale_X, &checkMvScaleDouble);
        std::printf("Synthetic DLSS parameter round-trip: Color=%p result=0x%08x asULL=0x%llx result=0x%08x MV.Scale.X=%g result=0x%08x asDouble=%g result=0x%08x\n",
                    checkColor, static_cast<unsigned int>(colorGet), checkColorAsUll,
                    static_cast<unsigned int>(colorUllGet), checkMvScale,
                    static_cast<unsigned int>(mvScaleGet), checkMvScaleDouble,
                    static_cast<unsigned int>(mvScaleDoubleGet));

        float msvcAbiMvScale = 0.0f;
        unsigned int msvcAbiWidth = 0;
        const unsigned int msvcAbiMvScaleResult = d4r_ngx_parameter_float_roundtrip(
            parameters, NVSDK_NGX_Parameter_MV_Scale_X, 1.0f, &msvcAbiMvScale);
        const unsigned int msvcAbiWidthResult = d4r_ngx_parameter_uint_roundtrip(
            parameters, NVSDK_NGX_Parameter_Width, inputWidth, &msvcAbiWidth);
        std::printf("MSVC ABI parameter round-trip: MV.Scale.X=%g result=0x%08x Width=%u result=0x%08x\n",
                    msvcAbiMvScale, msvcAbiMvScaleResult, msvcAbiWidth, msvcAbiWidthResult);

        const char* proxyMode = std::getenv("D4R_NGX_PARAMETER_PROXY");
        const bool useParameterProxy = proxyMode != nullptr && std::strcmp(proxyMode, "1") == 0;
        NVSDK_NGX_Parameter* evalParameters = parameters;
        std::vector<unsigned long long> proxyStorage;
        if (useParameterProxy)
        {
            const size_t proxyBytes = d4r_ngx_parameter_proxy_size();
            proxyStorage.resize((proxyBytes + sizeof(unsigned long long) - 1) /
                                sizeof(unsigned long long));
            d4r_ngx_parameter_proxy_construct(proxyStorage.data(), parameters);
            evalParameters = reinterpret_cast<NVSDK_NGX_Parameter*>(proxyStorage.data());
            std::printf("Synthetic DLSS parameter proxy: %zu bytes\n", proxyBytes);
        }

        const NVSDK_NGX_Result evalResult = useCEvaluate
            ? evaluateFeatureC(featureHandle, evalParameters, nullptr)
            : evaluateFeature(featureHandle, evalParameters, nullptr);
        print_ngx_result(useCEvaluate
            ? "CUDA_EvaluateFeature_C(DLSS SR; synthetic RGBA16F test pattern)"
            : "CUDA_EvaluateFeature(DLSS SR; synthetic RGBA16F test pattern)", evalResult);
        if (useParameterProxy)
        {
            const unsigned int traceCount = d4r_ngx_parameter_proxy_count(proxyStorage.data());
            const unsigned int droppedCount = d4r_ngx_parameter_proxy_dropped(proxyStorage.data());
            std::printf("Synthetic DLSS parameter proxy trace: %u records, %u dropped\n",
                        traceCount, droppedCount);
            for (unsigned int index = 0; index < traceCount && index < 128; ++index)
            {
                D4rNgxParamLogRecord record{};
                if (d4r_ngx_parameter_proxy_copy_record(proxyStorage.data(), index, &record))
                    std::printf("NGX parameter trace[%u]: %c/%c %s result=0x%08x value=0x%llx\n",
                                index, record.operation, record.valueType, record.name,
                                record.result, record.value);
            }
        }
        if (evalResult == NVSDK_NGX_Result_Success)
        {
            cudaResult = synchronize();
            std::printf("Synthetic DLSS eval cuCtxSynchronize: CUDA result=%d\n", cudaResult);
            if (cudaResult == 0 && surfaceMode)
            {
                D4rCudaMemcpy2D copyBack{};
                copyBack.srcMemoryType = 3; // CUDA_MEMORYTYPE_ARRAY
                copyBack.srcArray = outputArray;
                copyBack.dstMemoryType = 1; // CUDA_MEMORYTYPE_HOST
                copyBack.dstHost = output.data();
                copyBack.dstPitch = outputWidth * 4 * sizeof(uint16_t);
                copyBack.WidthInBytes = outputWidth * 4 * sizeof(uint16_t);
                copyBack.Height = outputHeight;
                cudaResult = copy2D(&copyBack);
            }
            else if (cudaResult == 0)
                cudaResult = copyDtoH(output.data(), outputDevice, outputBytes);
            std::printf("Synthetic DLSS eval output readback: CUDA result=%d bytes=%zu\n",
                        cudaResult, outputBytes);
            if (cudaResult == 0)
            {
                const std::wstring rawPath = appData + L"\\dlss-synthetic-output.rgba16f";
                const std::wstring ppmPath = appData + L"\\dlss-synthetic-output.ppm";
                const bool rawWritten = write_file(rawPath, output.data(), outputBytes);
                char ppmHeader[32]{};
                const int headerLength = std::snprintf(ppmHeader, sizeof(ppmHeader),
                                                       "P6\n%u %u\n255\n", outputWidth, outputHeight);
                std::vector<unsigned char> ppm(static_cast<size_t>(headerLength) + outputPixels * 3);
                std::memcpy(ppm.data(), ppmHeader, static_cast<size_t>(headerLength));
                size_t ppmOffset = static_cast<size_t>(headerLength);
                size_t nonzero = 0;
                double sum = 0.0;
                for (size_t pixel = 0; pixel < outputPixels; ++pixel)
                {
                    for (size_t channel = 0; channel < 3; ++channel)
                    {
                        float value = half_to_float(output[pixel * 4 + channel]);
                        if (!std::isfinite(value))
                            value = 0.0f;
                        if (value > 0.0f)
                            ++nonzero;
                        sum += value;
                        value = std::clamp(value, 0.0f, 1.0f);
                        const float srgb = value <= 0.0031308f ? value * 12.92f
                            : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
                        ppm[ppmOffset++] = static_cast<unsigned char>(std::lround(srgb * 255.0f));
                    }
                }
                const bool ppmWritten = write_file(ppmPath, ppm.data(), ppmOffset);
                std::printf("Synthetic DLSS output: raw=%s preview=%s positiveRGB=%zu/%zu meanRGB=%f\n",
                            rawWritten ? "written" : "failed", ppmWritten ? "written" : "failed",
                            nonzero, outputPixels * 3, sum / static_cast<double>(outputPixels * 3));
            }
        }
    }

    if (textureMode)
    {
        if (outputSurface != 0) surfaceDestroy(outputSurface);
        if (colorTexture != 0) textureDestroy(colorTexture);
        if (depthTexture != 0) textureDestroy(depthTexture);
        if (motionTexture != 0) textureDestroy(motionTexture);
        if (exposureTexture != 0) textureDestroy(exposureTexture);
        if (colorArray != nullptr) arrayDestroy(colorArray);
        if (depthArray != nullptr) arrayDestroy(depthArray);
        if (motionArray != nullptr) arrayDestroy(motionArray);
        if (exposureArray != nullptr) arrayDestroy(exposureArray);
        if (outputArray != nullptr) arrayDestroy(outputArray);
    }

    for (D4rCudaDevicePtr pointer : {outputDevice, motionDevice, depthDevice, colorDevice})
    {
        if (pointer != 0)
            std::printf("Synthetic DLSS eval cuMemFree(0x%llx): result=%d\n", pointer, free(pointer));
    }
    FreeLibrary(cuda);
}

static void set_cuda_luid_from_amd_dxgi_adapter()
{
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr)
    {
        std::printf("DXGI unavailable for CUDA LUID matching: GetLastError=%lu\n", GetLastError());
        return;
    }
    using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
    auto createFactory = load_export<CreateFactoryFn>(dxgi, "CreateDXGIFactory1");
    if (createFactory == nullptr)
    {
        std::printf("CreateDXGIFactory1 unavailable for CUDA LUID matching\n");
        FreeLibrary(dxgi);
        return;
    }

    constexpr GUID iidFactory1 = {0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
    IDXGIFactory1* factory = nullptr;
    HRESULT hr = createFactory(iidFactory1, reinterpret_cast<void**>(&factory));
    if (FAILED(hr) || factory == nullptr)
    {
        std::printf("CreateDXGIFactory1 failed while matching CUDA LUID: HRESULT=0x%08lx\n",
                    static_cast<unsigned long>(hr));
        FreeLibrary(dxgi);
        return;
    }

    bool found = false;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index)
    {
        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(adapter->GetDesc1(&description)) && description.VendorId == 0x1002)
        {
            char low[24]{};
            char high[24]{};
            std::snprintf(low, sizeof(low), "0x%08lx", static_cast<unsigned long>(description.AdapterLuid.LowPart));
            std::snprintf(high, sizeof(high), "0x%08lx", static_cast<unsigned long>(description.AdapterLuid.HighPart));
            SetEnvironmentVariableA("D4R_CUDA_LUID_LOW", low);
            SetEnvironmentVariableA("D4R_CUDA_LUID_HIGH", high);
            SetEnvironmentVariableA("D4R_CUDA_NODE_MASK", "1");
            std::printf("CUDA device LUID override from AMD DXGI adapter: %08lx:%08lx nodeMask=1\n",
                        static_cast<unsigned long>(description.AdapterLuid.HighPart),
                        static_cast<unsigned long>(description.AdapterLuid.LowPart));
            found = true;
            adapter->Release();
            adapter = nullptr;
            break;
        }
        adapter->Release();
        adapter = nullptr;
    }
    if (!found)
        std::printf("No AMD DXGI adapter found for CUDA LUID matching\n");
    factory->Release();
    FreeLibrary(dxgi);
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::fprintf(stderr,
                     "usage: ngx_cuda_probe.exe PATH_TO_NGX_CORE_DLL PATH_TO_NVNGX_DLSS_DLL WRITABLE_APPDATA_DIRECTORY\n");
        return 2;
    }

    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    set_cuda_luid_from_amd_dxgi_adapter();
    const char* corePath = argv[1];
    const char* featureDllPath = argv[2];
    std::wstring appData = utf8_to_wide(argv[3]);
    std::wstring featureDirectory = utf8_to_wide(featureDllPath);
    if (appData.empty() || featureDirectory.empty())
    {
        std::fprintf(stderr, "Invalid UTF-8 feature-DLL path or appdata path argument\n");
        return 2;
    }

    const size_t separator = featureDirectory.find_last_of(L"\\/");
    if (separator == std::wstring::npos)
        featureDirectory = L".";
    else
        featureDirectory.resize(separator);

    const wchar_t* featurePaths[] = {featureDirectory.c_str()};
    NVSDK_NGX_FeatureCommonInfo commonInfo{};
    commonInfo.PathListInfo.Path = featurePaths;
    commonInfo.PathListInfo.Length = 1;

    HMODULE core = LoadLibraryA(corePath);
    if (core == nullptr)
    {
        std::fprintf(stderr, "LoadLibraryA(%s) failed: GetLastError=%lu\n", corePath, GetLastError());
        return 1;
    }
    std::printf("Loaded NGX core: %s\n", corePath);
    std::printf("DLSS SR feature search directory: %ls\n", featureDirectory.c_str());

    auto init = load_export<decltype(&NVSDK_NGX_CUDA_Init)>(core, "NVSDK_NGX_CUDA_Init");
    auto shutdown = load_export<decltype(&NVSDK_NGX_CUDA_Shutdown)>(core, "NVSDK_NGX_CUDA_Shutdown");
    auto getCapabilityParameters = load_export<decltype(&NVSDK_NGX_CUDA_GetCapabilityParameters)>(
        core, "NVSDK_NGX_CUDA_GetCapabilityParameters");
    const auto getLegacyParameters = load_export<D4rNgxGetParameters>(core, "NVSDK_NGX_CUDA_GetParameters");
    auto destroyParameters = load_export<decltype(&NVSDK_NGX_CUDA_DestroyParameters)>(
        core, "NVSDK_NGX_CUDA_DestroyParameters");
    auto getScratchBufferSize = load_export<decltype(&NVSDK_NGX_CUDA_GetScratchBufferSize)>(
        core, "NVSDK_NGX_CUDA_GetScratchBufferSize");
    auto createFeature = load_export<decltype(&NVSDK_NGX_CUDA_CreateFeature)>(core, "NVSDK_NGX_CUDA_CreateFeature");
    auto releaseFeature = load_export<decltype(&NVSDK_NGX_CUDA_ReleaseFeature)>(core, "NVSDK_NGX_CUDA_ReleaseFeature");
    auto evaluateFeature = load_export<decltype(&NVSDK_NGX_CUDA_EvaluateFeature)>(core, "NVSDK_NGX_CUDA_EvaluateFeature");
    auto evaluateFeatureC = load_export<decltype(&NVSDK_NGX_CUDA_EvaluateFeature_C)>(
        core, "NVSDK_NGX_CUDA_EvaluateFeature_C");
    const char* evaluateApiMode = std::getenv("D4R_NGX_USE_C_EVALUATE");
    const bool useCEvaluate = evaluateApiMode != nullptr && std::strcmp(evaluateApiMode, "1") == 0;
    g_ngxParameterApi.setULL = load_export<PFN_NVSDK_NGX_Parameter_SetULL>(core, "NVSDK_NGX_Parameter_SetULL");
    g_ngxParameterApi.setF = load_export<PFN_NVSDK_NGX_Parameter_SetF>(core, "NVSDK_NGX_Parameter_SetF");
    g_ngxParameterApi.setD = load_export<PFN_NVSDK_NGX_Parameter_SetD>(core, "NVSDK_NGX_Parameter_SetD");
    g_ngxParameterApi.setUI = load_export<PFN_NVSDK_NGX_Parameter_SetUI>(core, "NVSDK_NGX_Parameter_SetUI");
    g_ngxParameterApi.setI = load_export<PFN_NVSDK_NGX_Parameter_SetI>(core, "NVSDK_NGX_Parameter_SetI");
    g_ngxParameterApi.setVoidPointer = load_export<PFN_NVSDK_NGX_Parameter_SetVoidPointer>(
        core, "NVSDK_NGX_Parameter_SetVoidPointer");
    g_ngxParameterApi.getULL = load_export<PFN_NVSDK_NGX_Parameter_GetULL>(core, "NVSDK_NGX_Parameter_GetULL");
    g_ngxParameterApi.getF = load_export<PFN_NVSDK_NGX_Parameter_GetF>(core, "NVSDK_NGX_Parameter_GetF");
    g_ngxParameterApi.getD = load_export<PFN_NVSDK_NGX_Parameter_GetD>(core, "NVSDK_NGX_Parameter_GetD");
    g_ngxParameterApi.getUI = load_export<PFN_NVSDK_NGX_Parameter_GetUI>(core, "NVSDK_NGX_Parameter_GetUI");
    g_ngxParameterApi.getI = load_export<PFN_NVSDK_NGX_Parameter_GetI>(core, "NVSDK_NGX_Parameter_GetI");
    g_ngxParameterApi.getVoidPointer = load_export<PFN_NVSDK_NGX_Parameter_GetVoidPointer>(
        core, "NVSDK_NGX_Parameter_GetVoidPointer");
    std::printf("NGX parameter C wrappers: SetULL=%p SetF=%p SetUI=%p SetI=%p SetPtr=%p GetI=%p\n",
                reinterpret_cast<void*>(g_ngxParameterApi.setULL),
                reinterpret_cast<void*>(g_ngxParameterApi.setF),
                reinterpret_cast<void*>(g_ngxParameterApi.setUI),
                reinterpret_cast<void*>(g_ngxParameterApi.setI),
                reinterpret_cast<void*>(g_ngxParameterApi.setVoidPointer),
                reinterpret_cast<void*>(g_ngxParameterApi.getI));
    std::printf("NGX CUDA EvaluateFeature C export: %p\n",
                reinterpret_cast<void*>(evaluateFeatureC));
    const char* parameterMode = std::getenv("D4R_NGX_USE_LEGACY_PARAMETERS");
    const bool useLegacyParameters = parameterMode != nullptr && std::strcmp(parameterMode, "1") == 0;
    if (init == nullptr || (!useLegacyParameters && getCapabilityParameters == nullptr) ||
        (useLegacyParameters && getLegacyParameters == nullptr) || destroyParameters == nullptr ||
        getScratchBufferSize == nullptr || createFeature == nullptr ||
        releaseFeature == nullptr || evaluateFeature == nullptr ||
        (useCEvaluate && evaluateFeatureC == nullptr))
    {
        std::fprintf(stderr, "The NGX core is missing one or more CUDA API exports\n");
        FreeLibrary(core);
        return 1;
    }

    NVSDK_NGX_FeatureDiscoveryInfo discovery{};
    discovery.SDKVersion = static_cast<NVSDK_NGX_Version>(0x13);
    if (const char* apiVersion = std::getenv("D4R_NGX_API_VERSION"))
        discovery.SDKVersion = static_cast<NVSDK_NGX_Version>(std::strtoul(apiVersion, nullptr, 0));
    discovery.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    discovery.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    discovery.Identifier.v.ApplicationId = 241534723ULL;
    discovery.ApplicationDataPath = appData.c_str();
    discovery.FeatureInfo = &commonInfo;

    std::printf("NGX API version passed to core: 0x%08x\n", static_cast<unsigned int>(discovery.SDKVersion));
    NVSDK_NGX_Result result = init(discovery.Identifier.v.ApplicationId, discovery.ApplicationDataPath,
                                   &commonInfo, discovery.SDKVersion);
    print_ngx_result("CUDA_Init(DLSS SR)", result);

    if (result == NVSDK_NGX_Result_Success)
    {
        NVSDK_NGX_Parameter* parameters = nullptr;
        NVSDK_NGX_Result paramsResult = useLegacyParameters
            ? getLegacyParameters(&parameters) : getCapabilityParameters(&parameters);
        print_ngx_result(useLegacyParameters ? "CUDA_GetParameters(legacy)" : "CUDA_GetCapabilityParameters",
                         paramsResult);
        if (paramsResult == NVSDK_NGX_Result_Success && parameters != nullptr)
        {
            int available = -1;
            NVSDK_NGX_Result availabilityResult = ngx_get_i(parameters,
                NVSDK_NGX_Parameter_SuperSampling_Available, &available);
            print_ngx_result("Parameters.Get(SuperSampling.Available)", availabilityResult);
            std::printf("DLSS SR availability value: %d\n", available);

            int featureInitResult = -1;
            NVSDK_NGX_Result initStateResult = ngx_get_i(parameters,
                NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &featureInitResult);
            print_ngx_result("Parameters.Get(SuperSampling.FeatureInitResult)", initStateResult);
            std::printf("DLSS SR recorded feature-init result: 0x%08x\n",
                        static_cast<unsigned int>(featureInitResult));
            int needsUpdatedDriver = -1;
            NVSDK_NGX_Result driverStateResult = ngx_get_i(parameters,
                NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver);
            print_ngx_result("Parameters.Get(SuperSampling.NeedsUpdatedDriver)", driverStateResult);
            std::printf("DLSS SR needs updated driver: %d\n", needsUpdatedDriver);

            ngx_set(parameters, NVSDK_NGX_Parameter_Width, 640u);
            ngx_set(parameters, NVSDK_NGX_Parameter_Height, 360u);
            ngx_set(parameters, NVSDK_NGX_Parameter_OutWidth, 1280u);
            ngx_set(parameters, NVSDK_NGX_Parameter_OutHeight, 720u);
            ngx_set(parameters, NVSDK_NGX_Parameter_PerfQualityValue,
                    static_cast<int>(NVSDK_NGX_PerfQuality_Value_MaxQuality));
            ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                    static_cast<int>(NVSDK_NGX_DLSS_Feature_Flags_None));
            ngx_set(parameters, NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);

            unsigned int widthCheck = 0;
            unsigned int outWidthCheck = 0;
            const NVSDK_NGX_Result widthCheckResult = ngx_get_ui(
                parameters, NVSDK_NGX_Parameter_Width, &widthCheck);
            const NVSDK_NGX_Result outWidthCheckResult = ngx_get_ui(
                parameters, NVSDK_NGX_Parameter_OutWidth, &outWidthCheck);
            std::printf("NGX create parameter round-trip: Width=%u result=0x%08x OutWidth=%u result=0x%08x\n",
                        widthCheck, static_cast<unsigned int>(widthCheckResult), outWidthCheck,
                        static_cast<unsigned int>(outWidthCheckResult));

            size_t scratchSize = 0;
            NVSDK_NGX_Result scratchResult = getScratchBufferSize(
                NVSDK_NGX_Feature_SuperSampling, parameters, &scratchSize);
            print_ngx_result("CUDA_GetScratchBufferSize(DLSS SR)", scratchResult);
            std::printf("DLSS SR scratch size: %llu bytes\n", static_cast<unsigned long long>(scratchSize));

            const char* evaluate = std::getenv("D4R_NGX_EVALUATE");
            const bool evaluateRequested = evaluate != nullptr && std::strcmp(evaluate, "1") == 0;
            HMODULE scratchCuda = nullptr;
            D4rCudaDevicePtr scratchDevice = 0;
            size_t scratchBytes = scratchSize;
            D4rCuMemFree scratchFree = nullptr;
            if (evaluateRequested)
            {
                if (scratchBytes == 0)
                    scratchBytes = 64u * 1024u * 1024u;
                scratchCuda = LoadLibraryA("nvcuda.dll");
                if (scratchCuda != nullptr)
                {
                    const auto scratchAlloc = load_export<D4rCuMemAlloc>(scratchCuda, "cuMemAlloc");
                    scratchFree = load_export<D4rCuMemFree>(scratchCuda, "cuMemFree");
                    const int scratchAllocResult = scratchAlloc != nullptr
                        ? scratchAlloc(&scratchDevice, scratchBytes) : 801;
                    std::printf("Synthetic DLSS scratch allocation before create: CUDA result=%d ptr=0x%llx bytes=%zu\n",
                                scratchAllocResult, scratchDevice, scratchBytes);
                    if (scratchAllocResult == 0)
                    {
                        ngx_set(parameters, NVSDK_NGX_Parameter_Scratch,
                                        reinterpret_cast<void*>(static_cast<uintptr_t>(scratchDevice)));
                        ngx_set(parameters, NVSDK_NGX_Parameter_Scratch_SizeInBytes,
                                        static_cast<unsigned long long>(scratchBytes));
                    }
                    else
                    {
                        scratchDevice = 0;
                        scratchBytes = 0;
                    }
                }
                else
                {
                    std::printf("Synthetic DLSS scratch allocation: LoadLibraryA(nvcuda.dll) failed: %lu\n",
                                GetLastError());
                    scratchBytes = 0;
                }
            }

            NVSDK_NGX_Handle* featureHandle = nullptr;
            NVSDK_NGX_Result createResult = createFeature(
                NVSDK_NGX_Feature_SuperSampling, parameters, &featureHandle);
            print_ngx_result("CUDA_CreateFeature(DLSS SR)", createResult);
            std::printf("DLSS SR feature handle: %s\n", featureHandle != nullptr ? "created" : "null");
            if (createResult == NVSDK_NGX_Result_Success && featureHandle != nullptr)
            {
                if (evaluateRequested)
                    evaluate_synthetic_frame(parameters, featureHandle, evaluateFeature,
                                             evaluateFeatureC, useCEvaluate, appData,
                                             scratchDevice, scratchBytes);
                NVSDK_NGX_Result releaseResult = releaseFeature(featureHandle);
                print_ngx_result("CUDA_ReleaseFeature", releaseResult);
            }

            if (scratchDevice != 0 && scratchFree != nullptr)
                std::printf("Synthetic DLSS scratch free: CUDA result=%d\n", scratchFree(scratchDevice));
            if (scratchCuda != nullptr)
                FreeLibrary(scratchCuda);

            if (useLegacyParameters)
                std::printf("CUDA_GetParameters map lifetime is managed by NGX; skipping DestroyParameters\n");
            else
            {
                NVSDK_NGX_Result destroyResult = destroyParameters(parameters);
                print_ngx_result("CUDA_DestroyParameters", destroyResult);
            }
        }

        if (shutdown != nullptr)
        {
            NVSDK_NGX_Result shutdownResult = shutdown();
            print_ngx_result("CUDA_Shutdown1", shutdownResult);
        }
    }

    FreeLibrary(core);
    return 0;
}
