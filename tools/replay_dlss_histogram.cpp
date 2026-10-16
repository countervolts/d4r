#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <vector>

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = uint64_t;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUarray = void*;
using CUtexObject = uint64_t;

struct CUDA_ARRAY_DESCRIPTOR
{
    size_t Width;
    size_t Height;
    uint32_t Format;
    uint32_t NumChannels;
};
static_assert(sizeof(CUDA_ARRAY_DESCRIPTOR) == 24);

struct CUDA_MEMCPY2D
{
    size_t srcXInBytes;
    size_t srcY;
    uint32_t srcMemoryType;
    uint32_t srcAlignment;
    const void* srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    uint32_t dstMemoryType;
    uint32_t dstAlignment;
    void* dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};
static_assert(sizeof(CUDA_MEMCPY2D) == 128);

struct CUDA_RESOURCE_DESC
{
    uint32_t resType;
    uint32_t alignment;
    union
    {
        struct { CUarray hArray; } array;
        int reserved[32];
    } res;
    uint32_t flags;
    uint32_t reserved;
};
static_assert(sizeof(CUDA_RESOURCE_DESC) == 144);

struct CUDA_TEXTURE_DESC
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
static_assert(sizeof(CUDA_TEXTURE_DESC) == 104);

struct HistogramParams
{
    uint32_t offsetX;
    uint32_t offsetY;
    uint32_t textureWidth;
    uint32_t textureHeight;
    uint32_t maxX;
    uint32_t maxY;
    float valueScale;
    float weightScale;
    float logBias;
    uint32_t alignmentPadding;
    CUdeviceptr histogram;
    CUtexObject texture;
    uint8_t flags[8];
};
static_assert(sizeof(HistogramParams) == 64);
static_assert(offsetof(HistogramParams, histogram) == 40);
static_assert(offsetof(HistogramParams, texture) == 48);
static_assert(offsetof(HistogramParams, flags) == 56);

template <typename Function> static Function load_function(void* library, const char* name)
{
    void* raw = dlsym(library, name);
    Function function{};
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
}

static void print_cuda_error(const char* stage, CUresult result,
                             CUresult (*get_error_string)(CUresult, const char**) = nullptr)
{
    const char* message = nullptr;
    if (get_error_string != nullptr)
        get_error_string(result, &message);
    std::printf("%s: CUDA result=%d", stage, result);
    if (message != nullptr)
        std::printf(" (%s)", message);
    std::putchar('\n');
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: replay_dlss_histogram PATH_TO_LOCALLY_CAPTURED_DLSS_FATBIN\n");
        return 2;
    }

    std::ifstream input(argv[1], std::ios::binary);
    if (!input)
    {
        std::perror("Could not open captured fatbin");
        return 2;
    }
    std::vector<unsigned char> image((std::istreambuf_iterator<char>(input)),
                                     std::istreambuf_iterator<char>());
    if (image.size() < 16 || std::memcmp(image.data(), "\x50\xed\x55\xba", 4) != 0)
    {
        std::fprintf(stderr, "Input is not a CUDA FATBIN_MAGIC v1 payload\n");
        return 2;
    }

    void* library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
    {
        std::fprintf(stderr, "dlopen(libcuda.so) failed: %s\n", dlerror());
        return 1;
    }

    using InitFn = CUresult (*)(unsigned int);
    using DeviceGetFn = CUresult (*)(CUdevice*, int);
    using DeviceGetNameFn = CUresult (*)(char*, int, CUdevice);
    using ContextCreateFn = CUresult (*)(CUcontext*, unsigned int, CUdevice);
    using ContextDestroyFn = CUresult (*)(CUcontext);
    using MemAllocFn = CUresult (*)(CUdeviceptr*, size_t);
    using MemFreeFn = CUresult (*)(CUdeviceptr);
    using CopyHtoDFn = CUresult (*)(CUdeviceptr, const void*, size_t);
    using CopyDtoHFn = CUresult (*)(void*, CUdeviceptr, size_t);
    using ArrayCreateFn = CUresult (*)(CUarray*, const CUDA_ARRAY_DESCRIPTOR*);
    using ArrayDestroyFn = CUresult (*)(CUarray);
    using Copy2DFn = CUresult (*)(const CUDA_MEMCPY2D*);
    using TextureCreateFn = CUresult (*)(CUtexObject*, const CUDA_RESOURCE_DESC*,
                                         const CUDA_TEXTURE_DESC*, const void*);
    using TextureDestroyFn = CUresult (*)(CUtexObject);
    using ModuleLoadDataFn = CUresult (*)(CUmodule*, const void*);
    using ModuleGetFunctionFn = CUresult (*)(CUfunction*, CUmodule, const char*);
    using LaunchKernelFn = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int, unsigned int,
                                        void*, void**, void**);
    using ContextSynchronizeFn = CUresult (*)(void);
    using ModuleUnloadFn = CUresult (*)(CUmodule);
    using GetErrorStringFn = CUresult (*)(CUresult, const char**);

    const InitFn init = load_function<InitFn>(library, "cuInit");
    const DeviceGetFn device_get = load_function<DeviceGetFn>(library, "cuDeviceGet");
    const DeviceGetNameFn device_get_name = load_function<DeviceGetNameFn>(library, "cuDeviceGetName");
    const ContextCreateFn context_create = load_function<ContextCreateFn>(library, "cuCtxCreate_v2");
    const ContextDestroyFn context_destroy = load_function<ContextDestroyFn>(library, "cuCtxDestroy_v2");
    const MemAllocFn mem_alloc = load_function<MemAllocFn>(library, "cuMemAlloc_v2");
    const MemFreeFn mem_free = load_function<MemFreeFn>(library, "cuMemFree_v2");
    const CopyHtoDFn copy_htod = load_function<CopyHtoDFn>(library, "cuMemcpyHtoD_v2");
    const CopyDtoHFn copy_dtoh = load_function<CopyDtoHFn>(library, "cuMemcpyDtoH_v2");
    const ArrayCreateFn array_create = load_function<ArrayCreateFn>(library, "cuArrayCreate_v2");
    const ArrayDestroyFn array_destroy = load_function<ArrayDestroyFn>(library, "cuArrayDestroy");
    const Copy2DFn copy_2d = load_function<Copy2DFn>(library, "cuMemcpy2D_v2");
    const TextureCreateFn texture_create = load_function<TextureCreateFn>(library, "cuTexObjectCreate");
    const TextureDestroyFn texture_destroy = load_function<TextureDestroyFn>(library, "cuTexObjectDestroy");
    const ModuleLoadDataFn module_load = load_function<ModuleLoadDataFn>(library, "cuModuleLoadData");
    const ModuleGetFunctionFn module_get_function =
        load_function<ModuleGetFunctionFn>(library, "cuModuleGetFunction");
    const LaunchKernelFn launch_kernel = load_function<LaunchKernelFn>(library, "cuLaunchKernel");
    const ContextSynchronizeFn context_synchronize =
        load_function<ContextSynchronizeFn>(library, "cuCtxSynchronize");
    const ModuleUnloadFn module_unload = load_function<ModuleUnloadFn>(library, "cuModuleUnload");
    const GetErrorStringFn get_error_string = load_function<GetErrorStringFn>(library, "cuGetErrorString");
    if (init == nullptr || device_get == nullptr || context_create == nullptr || context_destroy == nullptr ||
        mem_alloc == nullptr || mem_free == nullptr || copy_htod == nullptr || copy_dtoh == nullptr ||
        array_create == nullptr || array_destroy == nullptr || copy_2d == nullptr || texture_create == nullptr ||
        texture_destroy == nullptr || module_load == nullptr || module_get_function == nullptr ||
        launch_kernel == nullptr || context_synchronize == nullptr || module_unload == nullptr)
    {
        std::fprintf(stderr, "ZLUDA libcuda.so is missing one or more array/texture/driver exports\n");
        dlclose(library);
        return 1;
    }

    CUresult result = init(0);
    if (result != 0)
    {
        print_cuda_error("cuInit", result, get_error_string);
        dlclose(library);
        return 1;
    }
    CUdevice device = 0;
    result = device_get(&device, 0);
    if (result != 0)
    {
        print_cuda_error("cuDeviceGet", result, get_error_string);
        dlclose(library);
        return 1;
    }
    char device_name[256] = {0};
    if (device_get_name != nullptr && device_get_name(device_name, sizeof(device_name), device) == 0)
        std::printf("ZLUDA CUDA device: %s\n", device_name);

    CUcontext context = nullptr;
    result = context_create(&context, 0, device);
    if (result != 0)
    {
        print_cuda_error("cuCtxCreate_v2", result, get_error_string);
        dlclose(library);
        return 1;
    }

    CUmodule module = nullptr;
    result = module_load(&module, image.data());
    print_cuda_error("cuModuleLoadData(official DLSS histogram fatbin)", result, get_error_string);
    if (result != 0)
    {
        context_destroy(context);
        dlclose(library);
        return 1;
    }
    CUfunction kernel = nullptr;
    result = module_get_function(&kernel, module, "cuda_histogram_kernel");
    print_cuda_error("cuModuleGetFunction(cuda_histogram_kernel)", result, get_error_string);
    if (result != 0)
    {
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }

    constexpr unsigned int width = 32;
    constexpr unsigned int height = 16;
    constexpr size_t pixel_count = width * height;
    std::vector<float> pixels(pixel_count * 4);
    for (size_t i = 0; i < pixel_count; ++i)
    {
        pixels[i * 4 + 0] = 0.5f;
        pixels[i * 4 + 1] = 0.5f;
        pixels[i * 4 + 2] = 0.5f;
        pixels[i * 4 + 3] = 1.0f;
    }

    CUDA_ARRAY_DESCRIPTOR array_desc{};
    array_desc.Width = width;
    array_desc.Height = height;
    array_desc.Format = 32; // CU_AD_FORMAT_FLOAT
    array_desc.NumChannels = 4;
    CUarray array = nullptr;
    result = array_create(&array, &array_desc);
    print_cuda_error("cuArrayCreate_v2(RGBA32F 32x32)", result, get_error_string);
    if (result != 0)
    {
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = 1; // CU_MEMORYTYPE_HOST
    copy.srcHost = pixels.data();
    copy.srcPitch = width * 4 * sizeof(float);
    copy.dstMemoryType = 3; // CU_MEMORYTYPE_ARRAY
    copy.dstArray = array;
    copy.WidthInBytes = width * 4 * sizeof(float);
    copy.Height = height;
    result = copy_2d(&copy);
    if (result != 0)
    {
        print_cuda_error("cuMemcpy2D_v2(texture upload)", result, get_error_string);
        array_destroy(array);
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }
    std::vector<float> array_readback(pixels.size(), 0.0f);
    CUDA_MEMCPY2D copy_back{};
    copy_back.srcMemoryType = 3; // CU_MEMORYTYPE_ARRAY
    copy_back.srcArray = array;
    copy_back.dstMemoryType = 1; // CU_MEMORYTYPE_HOST
    copy_back.dstHost = array_readback.data();
    copy_back.dstPitch = width * 4 * sizeof(float);
    copy_back.WidthInBytes = width * 4 * sizeof(float);
    copy_back.Height = height;
    result = copy_2d(&copy_back);
    const bool array_upload_valid = result == 0 && array_readback == pixels;
    std::printf("RGBA32F texture upload/readback: %s\n", array_upload_valid ? "PASS" : "FAIL");
    if (!array_upload_valid)
    {
        print_cuda_error("cuMemcpy2D_v2(texture readback)", result, get_error_string);
        array_destroy(array);
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }

    CUDA_RESOURCE_DESC resource_desc{};
    resource_desc.resType = 0; // CU_RESOURCE_TYPE_ARRAY
    resource_desc.res.array.hArray = array;
    CUDA_TEXTURE_DESC texture_desc{};
    texture_desc.addressMode[0] = 1; // clamp
    texture_desc.addressMode[1] = 1;
    texture_desc.addressMode[2] = 1;
    texture_desc.filterMode = 0; // point
    texture_desc.maxAnisotropy = 0;
    texture_desc.mipmapFilterMode = 0;

    HIP_RESOURCE_DESC hip_resource_desc{};
    HIP_TEXTURE_DESC hip_texture_desc{};
    static_assert(sizeof(HIP_RESOURCE_DESC) == sizeof(CUDA_RESOURCE_DESC));
    static_assert(sizeof(HIP_TEXTURE_DESC) == sizeof(CUDA_TEXTURE_DESC));
    std::memcpy(&hip_resource_desc, &resource_desc, sizeof(hip_resource_desc));
    std::memcpy(&hip_texture_desc, &texture_desc, sizeof(hip_texture_desc));

    CUtexObject texture = 0;
    result = texture_create(&texture, &resource_desc, &texture_desc, nullptr);
    print_cuda_error("ZLUDA cuTexObjectCreate(RGBA32F test texture)", result, get_error_string);
    bool used_direct_hip_texture = false;
    hipTextureObject_t hip_texture = 0;
    if (result != 0)
    {
        hipError_t direct_hip_texture_result = hipTexObjectCreate(
            &hip_texture, &hip_resource_desc, &hip_texture_desc, nullptr);
        std::printf("Direct HIP texture-object fallback: %s (%d)\n",
                    hipGetErrorName(direct_hip_texture_result), static_cast<int>(direct_hip_texture_result));
        if (direct_hip_texture_result == hipSuccess)
        {
            texture = static_cast<CUtexObject>(reinterpret_cast<uintptr_t>(hip_texture));
            result = 0;
            used_direct_hip_texture = true;
        }
    }
    if (result != 0)
    {
        array_destroy(array);
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }

    const char texture_test_ptx[] =
        ".version 6.5\n"
        ".target sm_30\n"
        ".address_size 64\n"
        ".visible .entry d4r_texture_probe(\n"
        ".param .u64 out_ptr, .param .u64 texobj, .param .s32 coord_x, .param .s32 coord_y) {\n"
        ".reg .u64 %rd<3>; .reg .s32 %ix, %iy; .reg .f32 %f<4>;\n"
        "ld.param.u64 %rd1, [out_ptr]; ld.param.u64 %rd2, [texobj];\n"
        "ld.param.s32 %ix, [coord_x]; ld.param.s32 %iy, [coord_y];\n"
        "tex.2d.v4.f32.s32 {%f0,%f1,%f2,%f3}, [%rd2,{%ix,%iy}];\n"
        "st.global.v4.f32 [%rd1], {%f0,%f1,%f2,%f3}; ret; }\n";
    CUmodule texture_test_module = nullptr;
    CUfunction texture_test_kernel = nullptr;
    CUdeviceptr texture_test_output = 0;
    float sampled_pixel[4] = {};
    CUresult texture_test_result = module_load(&texture_test_module, texture_test_ptx);
    if (texture_test_result == 0)
        texture_test_result = module_get_function(&texture_test_kernel, texture_test_module, "d4r_texture_probe");
    if (texture_test_result == 0)
        texture_test_result = mem_alloc(&texture_test_output, sizeof(sampled_pixel));
    int sample_x = static_cast<int>(width / 2);
    int sample_y = static_cast<int>(height / 2);
    void* texture_test_parameters[] = {&texture_test_output, &texture, &sample_x, &sample_y};
    if (texture_test_result == 0)
        texture_test_result = launch_kernel(texture_test_kernel, 1, 1, 1, 1, 1, 1, 0,
                                            nullptr, texture_test_parameters, nullptr);
    if (texture_test_result == 0)
        texture_test_result = context_synchronize();
    if (texture_test_result == 0)
        texture_test_result = copy_dtoh(sampled_pixel, texture_test_output, sizeof(sampled_pixel));
    const float* expected_pixel = &pixels[(sample_y * width + sample_x) * 4];
    const bool texture_probe_pass = texture_test_result == 0 &&
        std::equal(std::begin(sampled_pixel), std::end(sampled_pixel), expected_pixel,
                   [](float actual, float expected) { return std::abs(actual - expected) < 0.001f; });
    std::printf("ZLUDA texture-object readback: %s (sample=%.3f,%.3f,%.3f,%.3f expected=%.3f,%.3f,%.3f,%.3f)\n",
                texture_probe_pass ? "PASS" : "FAIL", sampled_pixel[0], sampled_pixel[1],
                sampled_pixel[2], sampled_pixel[3], expected_pixel[0], expected_pixel[1],
                expected_pixel[2], expected_pixel[3]);
    if (texture_test_result != 0)
        print_cuda_error("ZLUDA texture-object probe", texture_test_result, get_error_string);
    if (texture_test_output != 0)
        mem_free(texture_test_output);
    if (texture_test_module != nullptr)
        module_unload(texture_test_module);

    std::vector<float> zero_histogram(128, 0.0f);
    CUdeviceptr histogram_device = 0;
    result = mem_alloc(&histogram_device, zero_histogram.size() * sizeof(float));
    if (result == 0)
        result = copy_htod(histogram_device, zero_histogram.data(), zero_histogram.size() * sizeof(float));
    if (result != 0)
    {
        print_cuda_error("allocate/zero histogram output", result, get_error_string);
        if (used_direct_hip_texture)
            (void)hipDestroyTextureObject(hip_texture);
        else
            texture_destroy(texture);
        array_destroy(array);
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }

    HistogramParams params{};
    params.offsetX = 0;
    params.offsetY = 0;
    params.textureWidth = width - 1;
    params.textureHeight = height - 1;
    params.maxX = width - 1;
    params.maxY = height - 1;
    params.valueScale = 1.0f;
    params.weightScale = 1.0f;
    params.logBias = 1.5f;
    params.histogram = histogram_device;
    params.texture = texture;
    void* kernel_parameters[] = {&params};
    result = launch_kernel(kernel, 4, 1, 1, 8, 16, 1, 0, nullptr, kernel_parameters, nullptr);
    print_cuda_error("cuLaunchKernel(cuda_histogram_kernel)", result, get_error_string);
    if (result == 0)
        result = context_synchronize();
    std::vector<float> histogram(128, 0.0f);
    if (result == 0)
        result = copy_dtoh(histogram.data(), histogram_device, histogram.size() * sizeof(float));
    if (result != 0)
    {
        print_cuda_error("synchronize/read histogram output", result, get_error_string);
        mem_free(histogram_device);
        if (used_direct_hip_texture)
            (void)hipDestroyTextureObject(hip_texture);
        else
            texture_destroy(texture);
        array_destroy(array);
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }

    const float total = std::accumulate(histogram.begin(), histogram.end(), 0.0f);
    const size_t nan_count = static_cast<size_t>(std::count_if(histogram.begin(), histogram.end(),
        [](float value) { return std::isnan(value); }));
    const size_t infinite_count = static_cast<size_t>(std::count_if(histogram.begin(), histogram.end(),
        [](float value) { return std::isinf(value); }));
    const auto nonzero = std::count_if(histogram.begin(), histogram.end(), [](float value) {
        return std::isfinite(value) && value != 0.0f;
    });
    auto top = std::max_element(histogram.begin(), histogram.end());
    const size_t top_bin = static_cast<size_t>(top - histogram.begin());
    std::printf("Histogram output: total=%.6f nonzero_bins=%zu NaNs=%zu Infs=%zu top_bin=%zu top_value=%.6f\n",
                total, static_cast<size_t>(nonzero), nan_count, infinite_count, top_bin, *top);

    const bool finite = std::all_of(histogram.begin(), histogram.end(), [](float value) {
        return std::isfinite(value) && value >= 0.0f;
    });
    const float expected_half = static_cast<float>(pixel_count) / 2.0f;
    const bool expected_bins = std::abs(histogram[63] - expected_half) < 0.01f &&
                               std::abs(histogram[64] - expected_half) < 0.01f;
    std::printf("Expected histogram bins 63/64: %.6f / %.6f (expected %.6f each)\n",
                histogram[63], histogram[64], expected_half);
    const bool valid = texture_probe_pass && finite &&
        std::abs(total - static_cast<float>(pixel_count)) < 0.1f && nonzero == 2 && expected_bins;
    std::printf("Official DLSS histogram kernel test: %s (uniform RGBA32F %ux%u synthetic input)\n",
                valid ? "PASS" : "FAIL", width, height);

    mem_free(histogram_device);
    if (used_direct_hip_texture)
        (void)hipDestroyTextureObject(hip_texture);
    else
        texture_destroy(texture);
    array_destroy(array);
    module_unload(module);
    context_destroy(context);
    dlclose(library);
    return valid ? 0 : 1;
}
