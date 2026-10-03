// Modified in this fork for CUDA Ray Reconstruction support and validation (2026).
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUarray = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = uint64_t;
using CUtexObject = uint64_t;
using CUsurfObject = uint64_t;

struct CUDA_ARRAY_DESCRIPTOR
{
    size_t Width;
    size_t Height;
    uint32_t Format;
    uint32_t NumChannels;
};
static_assert(sizeof(CUDA_ARRAY_DESCRIPTOR) == 24);

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
    int reserved[12];
};
static_assert(sizeof(CUDA_TEXTURE_DESC) == 104);

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

template <typename Function> static Function load_function(void* library, const char* name)
{
    void* raw = dlsym(library, name);
    Function function{};
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
}

static CUresult report(const char* stage, CUresult result,
                       CUresult (*get_error_string)(CUresult, const char**) = nullptr)
{
    const char* message = nullptr;
    if (get_error_string != nullptr)
        get_error_string(result, &message);
    std::printf("%s: CUDA result=%d", stage, result);
    if (message != nullptr)
        std::printf(" (%s)", message);
    std::putchar('\n');
    return result;
}

// Integer-coordinate texture reads (tex.base.2d / tex.2d with .s32
// coordinates) must return the exact texel at those indices for every sampler
// configuration, including NGX's: normalized coordinates with linear
// filtering. Every texel holds a distinct value so coordinate mistakes show.
int main()
{
    constexpr unsigned int width = 8;
    constexpr unsigned int height = 8;
    const char* ptx = R"PTX(
.version 8.9
.target sm_89
.address_size 64

.visible .entry d4r_half_texture_sample_probe(
    .param .b64 texture_object,
    .param .b64 output_pointer
)
{
    .reg .b64 %rd<5>;
    .reg .b32 %r<5>;
    .reg .f32 %f<22>;
    .reg .b16 %h<4>;
    ld.param.b64 %rd1, [texture_object];
    ld.param.b64 %rd2, [output_pointer];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %tid.y;
    tex.base.2d.v4.f32.s32 {%f0,%f1,%f2,%f3}, [%rd1, {%r1,%r2}];
    tex.2d.v4.f32.s32 {%f4,%f5,%f6,%f7}, [%rd1, {%r1,%r2}];
    cvt.rn.f32.u32 %f8, %r1;
    cvt.rn.f32.u32 %f9, %r2;
    add.f32 %f8, %f8, 0f3E800000;
    add.f32 %f9, %f9, 0f3F400000;
    mul.f32 %f8, %f8, 0f3E000000;
    mul.f32 %f9, %f9, 0f3E000000;
    tex.base.2d.v4.f32.f32 {%f10,%f11,%f12,%f13}, [%rd1, {%f8,%f9}];
    tex.level.2d.v4.f32.f32 {%f14,%f15,%f16,%f17}, [%rd1, {%f8,%f9}], 0f00000000;
    tex.base.2d.v4.f16.f32 {%h0,%h1,%h2,%h3}, [%rd1, {%f8,%f9}];
    cvt.f32.f16 %f18, %h0;
    cvt.f32.f16 %f19, %h1;
    cvt.f32.f16 %f20, %h2;
    cvt.f32.f16 %f21, %h3;
    mad.lo.u32 %r3, %r2, 8, %r1;
    mul.wide.u32 %rd3, %r3, 80;
    add.s64 %rd4, %rd2, %rd3;
    st.global.v4.f32 [%rd4], {%f0,%f1,%f2,%f3};
    st.global.v4.f32 [%rd4+16], {%f4,%f5,%f6,%f7};
    st.global.v4.f32 [%rd4+32], {%f10,%f11,%f12,%f13};
    st.global.v4.f32 [%rd4+48], {%f14,%f15,%f16,%f17};
    st.global.v4.f32 [%rd4+64], {%f18,%f19,%f20,%f21};
    ret;
}

.visible .entry d4r_half_surface_store_probe(.param .b64 surface_object)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<2>;
    .reg .f32 %f<4>;
    ld.param.b64 %rd0, [surface_object];
    mov.u32 %r0, 3;
    mov.u32 %r1, 4;
    mov.f32 %f0, 0f3E800000;
    mov.f32 %f1, 0f3F000000;
    mov.f32 %f2, 0f3F400000;
    mov.f32 %f3, 0f3F800000;
    sust.p.2d.v4.b32.zero [%rd0, {%r0,%r1}], {%f0,%f1,%f2,%f3};
    ret;
}
)PTX";

    void* library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
    {
        std::fprintf(stderr, "dlopen(libcuda.so) failed: %s\n", dlerror());
        return 1;
    }
    using InitFn = CUresult (*)(unsigned int);
    using DeviceGetFn = CUresult (*)(CUdevice*, int);
    using ContextCreateFn = CUresult (*)(CUcontext*, unsigned int, CUdevice);
    using ContextDestroyFn = CUresult (*)(CUcontext);
    using ArrayCreateFn = CUresult (*)(CUarray*, const CUDA_ARRAY_DESCRIPTOR*);
    using ArrayDestroyFn = CUresult (*)(CUarray);
    using TextureCreateFn = CUresult (*)(CUtexObject*, const CUDA_RESOURCE_DESC*,
                                         const CUDA_TEXTURE_DESC*, const void*);
    using TextureDestroyFn = CUresult (*)(CUtexObject);
    using SurfaceCreateFn = CUresult (*)(CUsurfObject*, const CUDA_RESOURCE_DESC*);
    using SurfaceDestroyFn = CUresult (*)(CUsurfObject);
    using Copy2DFn = CUresult (*)(const CUDA_MEMCPY2D*);
    using MemAllocFn = CUresult (*)(CUdeviceptr*, size_t);
    using MemFreeFn = CUresult (*)(CUdeviceptr);
    using CopyDtoHFn = CUresult (*)(void*, CUdeviceptr, size_t);
    using ModuleLoadFn = CUresult (*)(CUmodule*, const void*);
    using ModuleGetFunctionFn = CUresult (*)(CUfunction*, CUmodule, const char*);
    using LaunchKernelFn = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int, unsigned int,
                                        CUstream, void**, void**);
    using ContextSynchronizeFn = CUresult (*)(void);
    using ModuleUnloadFn = CUresult (*)(CUmodule);
    using GetErrorStringFn = CUresult (*)(CUresult, const char**);

    const auto init = load_function<InitFn>(library, "cuInit");
    const auto deviceGet = load_function<DeviceGetFn>(library, "cuDeviceGet");
    const auto contextCreate = load_function<ContextCreateFn>(library, "cuCtxCreate_v2");
    const auto contextDestroy = load_function<ContextDestroyFn>(library, "cuCtxDestroy_v2");
    const auto arrayCreate = load_function<ArrayCreateFn>(library, "cuArrayCreate_v2");
    const auto arrayDestroy = load_function<ArrayDestroyFn>(library, "cuArrayDestroy");
    const auto textureCreate = load_function<TextureCreateFn>(library, "cuTexObjectCreate");
    const auto textureDestroy = load_function<TextureDestroyFn>(library, "cuTexObjectDestroy");
    const auto surfaceCreate = load_function<SurfaceCreateFn>(library, "cuSurfObjectCreate");
    const auto surfaceDestroy = load_function<SurfaceDestroyFn>(library, "cuSurfObjectDestroy");
    const auto copy2D = load_function<Copy2DFn>(library, "cuMemcpy2D_v2");
    const auto memAlloc = load_function<MemAllocFn>(library, "cuMemAlloc_v2");
    const auto memFree = load_function<MemFreeFn>(library, "cuMemFree_v2");
    const auto copyDtoH = load_function<CopyDtoHFn>(library, "cuMemcpyDtoH_v2");
    const auto moduleLoad = load_function<ModuleLoadFn>(library, "cuModuleLoadData");
    const auto moduleGetFunction = load_function<ModuleGetFunctionFn>(library, "cuModuleGetFunction");
    const auto launchKernel = load_function<LaunchKernelFn>(library, "cuLaunchKernel");
    const auto synchronize = load_function<ContextSynchronizeFn>(library, "cuCtxSynchronize");
    const auto moduleUnload = load_function<ModuleUnloadFn>(library, "cuModuleUnload");
    const auto getErrorString = load_function<GetErrorStringFn>(library, "cuGetErrorString");
    if (init == nullptr || deviceGet == nullptr || contextCreate == nullptr || contextDestroy == nullptr ||
        arrayCreate == nullptr || arrayDestroy == nullptr || textureCreate == nullptr ||
        textureDestroy == nullptr || surfaceCreate == nullptr || surfaceDestroy == nullptr ||
        copy2D == nullptr || memAlloc == nullptr || memFree == nullptr ||
        copyDtoH == nullptr || moduleLoad == nullptr || moduleGetFunction == nullptr ||
        launchKernel == nullptr || synchronize == nullptr || moduleUnload == nullptr)
    {
        std::fprintf(stderr, "ZLUDA libcuda.so is missing one or more texture probe exports\n");
        dlclose(library);
        return 1;
    }

    CUcontext context = nullptr;
    CUarray array = nullptr;
    CUdeviceptr deviceOutput = 0;
    CUmodule module = nullptr;
    CUresult result = report("cuInit", init(0), getErrorString);
    CUdevice device = 0;
    if (result == 0)
        result = report("cuDeviceGet", deviceGet(&device, 0), getErrorString);
    if (result == 0)
        result = report("cuCtxCreate_v2", contextCreate(&context, 0, device), getErrorString);

    // Texel (x, y) = (x/8, y/8, 0.5 + x/16, 1), all exact in f16.
    auto expected_texel = [](unsigned int x, unsigned int y) {
        return std::array<float, 4>{x / 8.0f, y / 8.0f, 0.5f + x / 16.0f, 1.0f};
    };
    auto float_to_half_exact = [](float value) -> uint16_t {
        if (value == 0.0f)
            return 0;
        int exponent = 0;
        const float fraction = std::frexp(value, &exponent); // value = fraction * 2^exponent
        const uint16_t mantissa = static_cast<uint16_t>((fraction * 2.0f - 1.0f) * 1024.0f);
        return static_cast<uint16_t>(((exponent - 1 + 15) << 10) | mantissa);
    };
    const size_t outputFloats = static_cast<size_t>(width) * height * 20;
    if (result == 0)
        result = report("cuMemAlloc_v2(output)", memAlloc(&deviceOutput, outputFloats * sizeof(float)), getErrorString);
    if (result == 0)
        result = report("cuModuleLoadData(texture sample PTX)", moduleLoad(&module, ptx), getErrorString);
    CUfunction function = nullptr;
    if (result == 0)
        result = report("cuModuleGetFunction(d4r_half_texture_sample_probe)",
                        moduleGetFunction(&function, module, "d4r_half_texture_sample_probe"), getErrorString);

    struct SamplerCase
    {
        const char* name;
        uint32_t filterMode;
        uint32_t flags;
    };
    const SamplerCase samplerCases[] = {
        {"point, unnormalized", 0, 0},
        {"linear, normalized coordinates (NGX)", 1, 2 /* CU_TRSF_NORMALIZED_COORDINATES */},
        {"point, normalized coordinates (NGX)", 0, 2},
    };
    // DLSS reads RGBA16F colour, RG16F motion vectors and R32F depth; only
    // the channels a format has are compared.
    struct FormatCase
    {
        const char* name;
        uint32_t format;
        uint32_t channels;
    };
    const FormatCase formatCases[] = {
        {"RGBA16F", 0x10, 4}, {"RG16F", 0x10, 2}, {"R16F", 0x10, 1}, {"R32F", 0x20, 1}, {"RG32F", 0x20, 2},
    };
    bool allPass = true;
    for (const FormatCase& formatCase : formatCases)
    {
        if (result != 0)
            break;
        const size_t elementBytes = formatCase.format == 0x20 ? 4 : 2;
        const size_t rowBytes = width * formatCase.channels * elementBytes;
        std::vector<uint8_t> input(rowBytes * height);
        for (unsigned int y = 0; y < height; ++y)
            for (unsigned int x = 0; x < width; ++x)
                for (unsigned int channel = 0; channel < formatCase.channels; ++channel)
                {
                    const float value = expected_texel(x, y)[channel];
                    uint8_t* at = input.data() + y * rowBytes + (x * formatCase.channels + channel) * elementBytes;
                    if (elementBytes == 2)
                    {
                        const uint16_t half = float_to_half_exact(value);
                        std::memcpy(at, &half, 2);
                    }
                    else
                        std::memcpy(at, &value, 4);
                }
        CUarray formatArray = nullptr;
        const CUDA_ARRAY_DESCRIPTOR descriptor{width, height, formatCase.format, formatCase.channels};
        CUresult status = arrayCreate(&formatArray, &descriptor);
        if (status == 0)
        {
            CUDA_MEMCPY2D upload{};
            upload.srcMemoryType = 1; // CUDA_MEMORYTYPE_HOST
            upload.srcHost = input.data();
            upload.srcPitch = rowBytes;
            upload.dstMemoryType = 3; // CUDA_MEMORYTYPE_ARRAY
            upload.dstArray = formatArray;
            upload.WidthInBytes = rowBytes;
            upload.Height = height;
            status = copy2D(&upload);
        }
        std::vector<uint8_t> readbackBytes(input.size(), 0);
        if (status == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 3;
            readback.srcArray = formatArray;
            readback.dstMemoryType = 1;
            readback.dstHost = readbackBytes.data();
            readback.dstPitch = rowBytes;
            readback.WidthInBytes = rowBytes;
            readback.Height = height;
            status = copy2D(&readback);
        }
        const bool roundTrip = status == 0 && readbackBytes == input;
        std::printf("%s array upload/readback: %s (CUDA result=%d)\n", formatCase.name, roundTrip ? "PASS" : "FAIL", status);
        allPass &= roundTrip;
        for (const SamplerCase& samplerCase : samplerCases)
        {
            if (status != 0)
                break;
            CUtexObject texture = 0;
            CUDA_RESOURCE_DESC resource{};
            resource.resType = 0; // CUDA_RESOURCE_TYPE_ARRAY
            resource.res.array.hArray = formatArray;
            CUDA_TEXTURE_DESC sampler{};
            sampler.addressMode[0] = 1; // clamp
            sampler.addressMode[1] = 1;
            sampler.filterMode = samplerCase.filterMode;
            sampler.flags = samplerCase.flags;
            status = textureCreate(&texture, &resource, &sampler, nullptr);
            if (status == 0)
            {
                uint64_t textureArgument = texture;
                uint64_t outputArgument = deviceOutput;
                void* arguments[] = {&textureArgument, &outputArgument};
                status = launchKernel(function, 1, 1, 1, width, height, 1, 0, nullptr, arguments, nullptr);
            }
            if (status == 0)
                status = synchronize();
            std::vector<float> sampled(outputFloats);
            if (status == 0)
                status = copyDtoH(sampled.data(), deviceOutput, outputFloats * sizeof(float));
            size_t mismatches = 0;
            for (unsigned int y = 0; status == 0 && y < height; ++y)
                for (unsigned int x = 0; x < width; ++x)
                    for (unsigned int form = 0; form < (samplerCase.flags == 2 ? 5u : 2u); ++form)
                        for (unsigned int channel = 0; channel < formatCase.channels; ++channel)
                        {
                            const float got = sampled[(y * width + x) * 20 + form * 4 + channel];
                            float want = expected_texel(x, y)[channel];
                            if (form >= 2 && samplerCase.filterMode == 1)
                            {
                                const auto topLeft = expected_texel(x == 0 ? 0 : x - 1, y)[channel];
                                const auto topRight = expected_texel(x, y)[channel];
                                const auto bottomLeft = expected_texel(x == 0 ? 0 : x - 1, std::min(y + 1, height - 1))[channel];
                                const auto bottomRight = expected_texel(x, std::min(y + 1, height - 1))[channel];
                                want = 0.75f * (0.25f * topLeft + 0.75f * topRight)
                                     + 0.25f * (0.25f * bottomLeft + 0.75f * bottomRight);
                            }
                            if ((!std::isfinite(got) || std::fabs(got - want) > 1e-6f) && mismatches++ < 4)
                                std::printf("  %s %s %s (%u,%u) channel %u: got %g expected %g\n", formatCase.name,
                                            samplerCase.name,
                                            form == 0 ? "tex.base" : form == 1 ? "tex" : form == 2 ? "tex.base normalized" : form == 3 ? "tex.level LOD0" : "tex.base f16 normalized",
                                            x, y, channel, got, want);
                        }
            const bool casePass = status == 0 && mismatches == 0;
            allPass = allPass && casePass;
            std::printf("%s fetch, %s: %s (CUDA result=%d, %zu mismatches)\n", formatCase.name,
                        samplerCase.name, casePass ? "PASS" : "FAIL", status, mismatches);
            if (texture != 0)
                textureDestroy(texture);
        }
        if (formatCase.channels == 4 && formatCase.format == 0x10)
            array = formatArray; // reused by the surface coherence check below
        else if (formatArray != nullptr)
            arrayDestroy(formatArray);
    }
    if (result == 0)
    {
        CUDA_RESOURCE_DESC resource{};
        resource.resType = 0;
        resource.res.array.hArray = array;
        CUDA_TEXTURE_DESC sampler{};
        sampler.addressMode[0] = sampler.addressMode[1] = 1;
        sampler.flags = 2;
        CUsurfObject surface = 0;
        CUtexObject texture = 0;
        CUfunction storeFunction = nullptr;
        CUresult status = surfaceCreate(&surface, &resource);
        if (status == 0)
            status = textureCreate(&texture, &resource, &sampler, nullptr);
        if (status == 0)
            status = moduleGetFunction(&storeFunction, module, "d4r_half_surface_store_probe");
        if (status == 0)
        {
            uint64_t handle = surface;
            void* args[] = {&handle};
            status = launchKernel(storeFunction, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
        }
        if (status == 0)
            status = synchronize();
        if (status == 0)
        {
            uint64_t textureArgument = texture;
            uint64_t outputArgument = deviceOutput;
            void* args[] = {&textureArgument, &outputArgument};
            status = launchKernel(function, 1, 1, 1, width, height, 1, 0, nullptr, args, nullptr);
        }
        if (status == 0)
            status = synchronize();
        std::vector<float> sampled(outputFloats);
        if (status == 0)
            status = copyDtoH(sampled.data(), deviceOutput, outputFloats * sizeof(float));
        const std::array<float, 4> expectedStored{0.25f, 0.5f, 0.75f, 1.0f};
        size_t mismatches = 0;
        for (unsigned int y = 0; status == 0 && y < height; ++y)
            for (unsigned int x = 0; x < width; ++x)
                for (unsigned int form = 0; form < 5; ++form)
                    for (unsigned int channel = 0; channel < 4; ++channel)
                    {
                        const float expected = x == 3 && y == 4 ? expectedStored[channel]
                                                                  : expected_texel(x, y)[channel];
                        const float got = sampled[(y * width + x) * 20 + form * 4 + channel];
                        if (got != expected && mismatches++ < 5)
                            std::printf("  surface->texture (%u,%u) form %u channel %u: got %g expected %g\n",
                                        x, y, form, channel, got, expected);
                    }
        const bool coherencePass = status == 0 && mismatches == 0;
        std::printf("RGBA16F surface store to existing texture: %s (CUDA result=%d, %zu mismatches)\n",
                    coherencePass ? "PASS" : "FAIL", status, mismatches);
        allPass &= coherencePass;
        if (texture != 0)
            textureDestroy(texture);
        if (surface != 0)
            surfaceDestroy(surface);
    }
    if (result == 0)
        std::printf("RGBA16F texture sample smoke test: %s\n", allPass ? "PASS" : "FAIL");
    const int exitCode = result == 0 && allPass ? 0 : 1;

    if (deviceOutput != 0)
        memFree(deviceOutput);
    if (array != nullptr)
        arrayDestroy(array);
    if (module != nullptr)
        moduleUnload(module);
    if (context != nullptr)
        contextDestroy(context);
    dlclose(library);
    return exitCode;
}
