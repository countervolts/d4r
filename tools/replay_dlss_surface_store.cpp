#include <dlfcn.h>

#include <array>
#include <bit>
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
using CUsurfObject = uint64_t;

struct CUDA_ARRAY3D_DESCRIPTOR
{
    size_t Width;
    size_t Height;
    size_t Depth;
    uint32_t Format;
    uint32_t NumChannels;
    uint32_t Flags;
};
static_assert(sizeof(CUDA_ARRAY3D_DESCRIPTOR) == 40);

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

struct CUDA_MEMCPY2D
{
    size_t srcXInBytes;
    size_t srcY;
    uint32_t srcMemoryType;
    uint32_t srcAlignment;
    const void* srcHost;
    uint64_t srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    uint32_t dstMemoryType;
    uint32_t dstAlignment;
    void* dstHost;
    uint64_t dstDevice;
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

int main()
{
    constexpr unsigned int width = 16;
    constexpr unsigned int height = 16;
    constexpr unsigned int x = 5;
    constexpr unsigned int y = 7;
    constexpr uint32_t value = 0x3f800000u; // float 1.0
    constexpr uint32_t outOfBoundsX = width;
    const char* ptx = R"PTX(
.version 8.9
.target sm_89
.address_size 64

.visible .entry d4r_surface_store_probe(
    .param .b64 surface,
    .param .b32 x,
    .param .b32 y,
    .param .b32 value
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<4>;
    ld.param.b64 %rd1, [surface];
    ld.param.b32 %r0, [x];
    ld.param.b32 %r1, [y];
    ld.param.b32 %r2, [value];
    sust.p.2d.b32.zero [%rd1, {%r0,%r1}], {%r2};
    ret;
}

.visible .entry d4r_surface_normalized_probe(
    .param .b64 surface,
    .param .b32 x,
    .param .b32 y,
    .param .f32 red,
    .param .f32 green,
    .param .f32 blue,
    .param .f32 alpha
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<2>;
    .reg .f32 %f<4>;
    ld.param.b64 %rd1, [surface];
    ld.param.b32 %r0, [x];
    ld.param.b32 %r1, [y];
    ld.param.f32 %f0, [red];
    ld.param.f32 %f1, [green];
    ld.param.f32 %f2, [blue];
    ld.param.f32 %f3, [alpha];
    sust.p.2d.v4.b32.zero [%rd1, {%r0,%r1}], {%f0,%f1,%f2,%f3};
    ret;
}

.visible .entry d4r_surface_integer_probe(
    .param .b64 surface,
    .param .b32 x,
    .param .b32 y,
    .param .b32 red,
    .param .b32 green,
    .param .b32 blue,
    .param .b32 alpha
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<6>;
    ld.param.b64 %rd1, [surface];
    ld.param.b32 %r0, [x];
    ld.param.b32 %r1, [y];
    ld.param.b32 %r2, [red];
    ld.param.b32 %r3, [green];
    ld.param.b32 %r4, [blue];
    ld.param.b32 %r5, [alpha];
    sust.p.2d.v4.b32.zero [%rd1, {%r0,%r1}], {%r2,%r3,%r4,%r5};
    ret;
}
)PTX";

    void* library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
    {
        std::fprintf(stderr, "dlopen(libcuda.so) failed: %s\n", dlerror());
        return 1;
    }
    void* hipLibrary = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_LOCAL);
    if (hipLibrary == nullptr)
    {
        std::fprintf(stderr, "dlopen(libamdhip64.so.7) failed: %s\n", dlerror());
        dlclose(library);
        return 1;
    }
    using InitFn = CUresult (*)(unsigned int);
    using DeviceGetFn = CUresult (*)(CUdevice*, int);
    using ContextCreateFn = CUresult (*)(CUcontext*, unsigned int, CUdevice);
    using ContextDestroyFn = CUresult (*)(CUcontext);
    using Array3DCreateFn = CUresult (*)(CUarray*, const CUDA_ARRAY3D_DESCRIPTOR*);
    using ArrayCreateFn = CUresult (*)(CUarray*, const CUDA_ARRAY_DESCRIPTOR*);
    using ArrayDestroyFn = CUresult (*)(CUarray);
    using SurfaceCreateFn = CUresult (*)(CUsurfObject*, const CUDA_RESOURCE_DESC*);
    using SurfaceDestroyFn = CUresult (*)(CUsurfObject);
    using SurfaceGetDescFn = CUresult (*)(CUDA_RESOURCE_DESC*, CUsurfObject);
    using Copy2DFn = CUresult (*)(const CUDA_MEMCPY2D*);
    using MemAllocFn = CUresult (*)(CUdeviceptr*, size_t);
    using MemFreeFn = CUresult (*)(CUdeviceptr);
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
    const auto array3DCreate = load_function<Array3DCreateFn>(library, "cuArray3DCreate_v2");
    const auto arrayCreate = load_function<ArrayCreateFn>(library, "cuArrayCreate_v2");
    const auto arrayDestroy = load_function<ArrayDestroyFn>(library, "cuArrayDestroy");
    const auto surfaceCreate = load_function<SurfaceCreateFn>(library, "cuSurfObjectCreate");
    const auto surfaceDestroy = load_function<SurfaceDestroyFn>(library, "cuSurfObjectDestroy");
    const auto surfaceGetDesc = load_function<SurfaceGetDescFn>(library, "cuSurfObjectGetResourceDesc");
    const auto copy2D = load_function<Copy2DFn>(library, "cuMemcpy2D_v2");
    const auto memAlloc = load_function<MemAllocFn>(library, "cuMemAlloc_v2");
    const auto memFree = load_function<MemFreeFn>(library, "cuMemFree_v2");
    // Inspect ZLUDA's physical RGBA32F backing without invoking the public
    // CUDA packed-format copy contract that this probe validates below.
    const auto backingCopy2D = load_function<Copy2DFn>(hipLibrary, "hipMemcpyParam2D");
    const auto moduleLoad = load_function<ModuleLoadFn>(library, "cuModuleLoadData");
    const auto moduleGetFunction = load_function<ModuleGetFunctionFn>(library, "cuModuleGetFunction");
    const auto launchKernel = load_function<LaunchKernelFn>(library, "cuLaunchKernel");
    const auto synchronize = load_function<ContextSynchronizeFn>(library, "cuCtxSynchronize");
    const auto moduleUnload = load_function<ModuleUnloadFn>(library, "cuModuleUnload");
    const auto getErrorString = load_function<GetErrorStringFn>(library, "cuGetErrorString");
    if (init == nullptr || deviceGet == nullptr || contextCreate == nullptr || contextDestroy == nullptr ||
        array3DCreate == nullptr || arrayCreate == nullptr || arrayDestroy == nullptr || surfaceCreate == nullptr ||
        surfaceDestroy == nullptr || surfaceGetDesc == nullptr || copy2D == nullptr || backingCopy2D == nullptr ||
        memAlloc == nullptr || memFree == nullptr ||
        moduleLoad == nullptr || moduleGetFunction == nullptr || launchKernel == nullptr ||
        synchronize == nullptr || moduleUnload == nullptr)
    {
        std::fprintf(stderr, "ZLUDA libcuda.so is missing one or more surface probe exports\n");
        dlclose(library);
        return 1;
    }

    CUcontext context = nullptr;
    CUarray array = nullptr;
    CUsurfObject surface = 0;
    CUmodule module = nullptr;
    int exitCode = 1;
    CUresult result = report("cuInit", init(0), getErrorString);
    CUdevice device = 0;
    if (result == 0)
        result = report("cuDeviceGet", deviceGet(&device, 0), getErrorString);
    if (result == 0)
        result = report("cuCtxCreate_v2", contextCreate(&context, 0, device), getErrorString);

    const CUDA_ARRAY3D_DESCRIPTOR arrayDescriptor{width, height, 0, 32, 1, 2};
    if (result == 0)
        result = report("cuArray3DCreate_v2(R32F; SURFACE_LDST)",
                        array3DCreate(&array, &arrayDescriptor), getErrorString);
    if (result == 0)
    {
        CUDA_RESOURCE_DESC resource{};
        resource.resType = 0; // CUDA_RESOURCE_TYPE_ARRAY
        resource.res.array.hArray = array;
        result = report("cuSurfObjectCreate", surfaceCreate(&surface, &resource), getErrorString);
    }
    if (result == 0)
    {
        CUDA_RESOURCE_DESC roundTrip{};
        result = report("cuSurfObjectGetResourceDesc",
                        surfaceGetDesc(&roundTrip, surface), getErrorString);
        if (result == 0)
            std::printf("Surface descriptor round-trip: type=%u array=%p expected=%p\n",
                        roundTrip.resType, roundTrip.res.array.hArray, array);
    }

    std::vector<float> pixels(width * height, 0.25f);
    if (result == 0)
    {
        CUDA_MEMCPY2D upload{};
        upload.srcMemoryType = 1; // CUDA_MEMORYTYPE_HOST
        upload.srcHost = pixels.data();
        upload.srcPitch = width * sizeof(float);
        upload.dstMemoryType = 3; // CUDA_MEMORYTYPE_ARRAY
        upload.dstArray = array;
        upload.WidthInBytes = width * sizeof(float);
        upload.Height = height;
        result = report("cuMemcpy2D_v2(upload test surface)", copy2D(&upload), getErrorString);
    }
    if (result == 0)
        result = report("cuModuleLoadData(surface store PTX)", moduleLoad(&module, ptx), getErrorString);
    CUfunction function = nullptr;
    if (result == 0)
        result = report("cuModuleGetFunction(d4r_surface_store_probe)",
                        moduleGetFunction(&function, module, "d4r_surface_store_probe"), getErrorString);

    auto launchStore = [&](unsigned int px, unsigned int py, uint32_t word, const char* label) {
        uint64_t surfaceArgument = surface;
        void* arguments[] = {&surfaceArgument, &px, &py, &word};
        CUresult launchResult = report(label,
            launchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr), getErrorString);
        if (launchResult == 0)
            launchResult = report("cuCtxSynchronize", synchronize(), getErrorString);
        return launchResult;
    };
    if (result == 0)
        result = launchStore(x, y, value, "cuLaunchKernel(surface in-bounds write)");
    if (result == 0)
        result = launchStore(outOfBoundsX, y, value, "cuLaunchKernel(surface .zero out-of-bounds write)");

    if (result == 0)
    {
        CUDA_MEMCPY2D readback{};
        readback.srcMemoryType = 3; // CUDA_MEMORYTYPE_ARRAY
        readback.srcArray = array;
        readback.dstMemoryType = 1; // CUDA_MEMORYTYPE_HOST
        readback.dstHost = pixels.data();
        readback.dstPitch = width * sizeof(float);
        readback.WidthInBytes = width * sizeof(float);
        readback.Height = height;
        result = report("cuMemcpy2D_v2(read back surface)", copy2D(&readback), getErrorString);
    }

    bool valid = result == 0;
    size_t changed = 0;
    for (size_t index = 0; index < pixels.size(); ++index)
    {
        const float expected = index == static_cast<size_t>(y) * width + x ? 1.0f : 0.25f;
        if (pixels[index] != expected)
            valid = false;
        if (pixels[index] != 0.25f)
            ++changed;
    }
    if (result == 0)
        std::printf("Surface store smoke test: %s (changed %zu pixel; expected one changed pixel)\n",
                    valid ? "PASS" : "FAIL", changed);

    // CUDA sust.p interprets input as f32 for UNORM/SNORM surfaces. These
    // deterministic bytes expose a conversion error when a normalized CUDA
    // array is represented by a plain HIP integer array.
    CUfunction normalizedFunction = nullptr;
    CUfunction integerFunction = nullptr;
    if (result == 0)
        result = report("cuModuleGetFunction(d4r_surface_normalized_probe)",
                        moduleGetFunction(&normalizedFunction, module, "d4r_surface_normalized_probe"), getErrorString);
    if (result == 0)
        result = report("cuModuleGetFunction(d4r_surface_integer_probe)",
                        moduleGetFunction(&integerFunction, module, "d4r_surface_integer_probe"), getErrorString);
    auto checkFormatted = [&](uint32_t format, const char* name, std::array<uint32_t, 4> words,
                              std::array<uint8_t, 4> expected, CUfunction storeFunction) {
        constexpr size_t testWidth = 4;
        constexpr size_t testHeight = 4;
        CUarray testArray = nullptr;
        CUsurfObject testSurface = 0;
        const CUDA_ARRAY_DESCRIPTOR descriptor{testWidth, testHeight, format, 4};
        CUresult status = report(name, arrayCreate(&testArray, &descriptor), getErrorString);
        if (status == 0)
        {
            CUDA_RESOURCE_DESC resource{};
            resource.resType = 0;
            resource.res.array.hArray = testArray;
            status = report("cuSurfObjectCreate(normalized)", surfaceCreate(&testSurface, &resource), getErrorString);
        }
        std::array<uint8_t, testWidth * testHeight * 4> bytes{};
        if (status == 0)
        {
            CUDA_MEMCPY2D upload{};
            upload.srcMemoryType = 1;
            upload.srcHost = bytes.data();
            upload.srcPitch = testWidth * 4;
            upload.dstMemoryType = 3;
            upload.dstArray = testArray;
            upload.WidthInBytes = testWidth * 4;
            upload.Height = testHeight;
            status = report("cuMemcpy2D_v2(normalized upload)", copy2D(&upload), getErrorString);
        }
        if (status == 0)
        {
            uint64_t handle = testSurface;
            unsigned int px = 1, py = 2;
            void* arguments[] = {&handle, &px, &py, &words[0], &words[1], &words[2], &words[3]};
            status = report("cuLaunchKernel(formatted store)",
                            launchKernel(storeFunction, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr),
                            getErrorString);
            if (status == 0)
                status = report("cuCtxSynchronize(normalized)", synchronize(), getErrorString);
        }
        if (status == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 3;
            readback.srcArray = testArray;
            readback.dstMemoryType = 1;
            readback.dstHost = bytes.data();
            readback.dstPitch = testWidth * 4;
            readback.WidthInBytes = testWidth * 4;
            readback.Height = testHeight;
            status = report("cuMemcpy2D_v2(normalized readback)", copy2D(&readback), getErrorString);
        }
        const size_t pixelOffset = (2 * testWidth + 1) * 4;
        std::printf("%s: actual=%02x,%02x,%02x,%02x expected=%02x,%02x,%02x,%02x\n", name,
                    bytes[pixelOffset], bytes[pixelOffset + 1], bytes[pixelOffset + 2], bytes[pixelOffset + 3],
                    expected[0], expected[1], expected[2], expected[3]);
        bool caseValid = status == 0;
        for (size_t index = 0; index < bytes.size(); ++index)
        {
            const uint8_t expectedByte = index >= pixelOffset && index < pixelOffset + 4
                ? expected[index - pixelOffset] : 0;
            if (bytes[index] != expectedByte)
                caseValid = false;
        }
        if (testSurface != 0)
            surfaceDestroy(testSurface);
        if (testArray != nullptr)
            arrayDestroy(testArray);
        std::printf("%s formatted store: %s\n", name, caseValid ? "PASS" : "FAIL");
        return caseValid;
    };
    if (result == 0)
    {
        valid &= checkFormatted(194, "RGBA8 UNORM",
                                {std::bit_cast<uint32_t>(0.25f), std::bit_cast<uint32_t>(0.5f),
                                 std::bit_cast<uint32_t>(0.75f), std::bit_cast<uint32_t>(1.0f)},
                                {0x40, 0x80, 0xbf, 0xff}, normalizedFunction);
        valid &= checkFormatted(200, "RGBA8 SNORM",
                                {std::bit_cast<uint32_t>(-1.0f), std::bit_cast<uint32_t>(-0.5f),
                                 std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(1.0f)},
                                {0x81, 0xc0, 0x40, 0x7f}, normalizedFunction);
        valid &= checkFormatted(1, "RGBA8 UINT", {1u, 127u, 200u, 255u},
                                {0x01, 0x7f, 0xc8, 0xff}, integerFunction);
        valid &= checkFormatted(8, "RGBA8 SINT",
                                {uint32_t(-128), uint32_t(-64), 64u, 127u},
                                {0x80, 0xc0, 0x40, 0x7f}, integerFunction);

        // The transformer output writes an RGBA16F CUDA surface and the input
        // kernels write RG16F motion and R16F surfaces with the formatted v4
        // store; only the channels the format has are kept.
        for (uint32_t halfChannels : {4u, 2u, 1u})
        {
            CUarray halfArray = nullptr;
            CUsurfObject halfSurface = 0;
            const CUDA_ARRAY3D_DESCRIPTOR halfDescriptor{4, 4, 0, 16, halfChannels, 2};
            CUresult halfStatus = array3DCreate(&halfArray, &halfDescriptor);
            if (halfStatus == 0)
            {
                CUDA_RESOURCE_DESC resource{};
                resource.resType = 0;
                resource.res.array.hArray = halfArray;
                halfStatus = surfaceCreate(&halfSurface, &resource);
            }
            std::vector<uint16_t> halfPixels(4 * 4 * halfChannels, 0);
            const size_t halfPitch = 4 * halfChannels * sizeof(uint16_t);
            if (halfStatus == 0)
            {
                CUDA_MEMCPY2D upload{};
                upload.srcMemoryType = 1;
                upload.srcHost = halfPixels.data();
                upload.srcPitch = halfPitch;
                upload.dstMemoryType = 3;
                upload.dstArray = halfArray;
                upload.WidthInBytes = halfPitch;
                upload.Height = 4;
                halfStatus = copy2D(&upload);
            }
            if (halfStatus == 0)
            {
                uint64_t handle = halfSurface;
                unsigned int px = 1, py = 2;
                std::array<float, 4> values{0.25f, 0.5f, 0.75f, 1.0f};
                void* arguments[] = {&handle, &px, &py, &values[0], &values[1], &values[2], &values[3]};
                halfStatus = launchKernel(normalizedFunction, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr);
                if (halfStatus == 0)
                    halfStatus = synchronize();
            }
            if (halfStatus == 0)
            {
                CUDA_MEMCPY2D readback{};
                readback.srcMemoryType = 3;
                readback.srcArray = halfArray;
                readback.dstMemoryType = 1;
                readback.dstHost = halfPixels.data();
                readback.dstPitch = halfPitch;
                readback.WidthInBytes = halfPitch;
                readback.Height = 4;
                halfStatus = copy2D(&readback);
            }
            const size_t halfPixelOffset = (2 * 4 + 1) * halfChannels;
            const std::array<uint16_t, 4> halfExpected{0x3400u, 0x3800u, 0x3a00u, 0x3c00u};
            bool halfValid = halfStatus == 0;
            for (size_t index = 0; index < halfPixels.size(); ++index)
            {
                const uint16_t expected = index >= halfPixelOffset && index < halfPixelOffset + halfChannels
                    ? halfExpected[index - halfPixelOffset] : 0;
                if (halfPixels[index] != expected)
                {
                    if (halfValid)
                        std::printf("  %u-channel f16 pixel value %zu: got 0x%04x expected 0x%04x\n", halfChannels,
                                    index, halfPixels[index], expected);
                    halfValid = false;
                }
            }
            std::printf("%s formatted v4 store: %s (CUDA result=%d)\n",
                        halfChannels == 4 ? "RGBA16F" : halfChannels == 2 ? "RG16F" : "R16F",
                        halfValid ? "PASS" : "FAIL", halfStatus);
            valid &= halfValid;
            if (halfSurface != 0)
                surfaceDestroy(halfSurface);
            if (halfArray != nullptr)
                arrayDestroy(halfArray);
        }

        // ZLUDA currently backs CUDA's packed UNORM 10:10:10:2 format 80
        // with RGBA32F. Probe the GPU-side formatted-store conversion in
        // that backing representation; host packed-copy semantics are a
        // separate contract.
        CUarray packedArray = nullptr;
        CUsurfObject packedSurface = 0;
        const CUDA_ARRAY_DESCRIPTOR packedDescriptor{4, 4, 80, 4};
        CUresult packedStatus = report("CUDA format 80 array", arrayCreate(&packedArray, &packedDescriptor),
                                       getErrorString);
        if (packedStatus == 0)
        {
            CUDA_RESOURCE_DESC resource{};
            resource.resType = 0;
            resource.res.array.hArray = packedArray;
            packedStatus = report("cuSurfObjectCreate(format 80)",
                                  surfaceCreate(&packedSurface, &resource), getErrorString);
        }
        std::array<float, 4 * 4 * 4> packedPixels{};
        if (packedStatus == 0)
        {
            CUDA_MEMCPY2D upload{};
            upload.srcMemoryType = 1;
            upload.srcHost = packedPixels.data();
            upload.srcPitch = 4 * 4 * sizeof(float);
            upload.dstMemoryType = 10; // HIP array, bypassing CUDA format conversion
            upload.dstArray = packedArray;
            upload.WidthInBytes = upload.srcPitch;
            upload.Height = 4;
            packedStatus = report("hipMemcpyParam2D(format 80 backing upload)", backingCopy2D(&upload));
        }
        std::array<float, 4> packedInput{0.1234f, 0.501f, 0.8765f, 0.45f};
        if (packedStatus == 0)
        {
            uint64_t handle = packedSurface;
            unsigned int px = 1, py = 2;
            void* arguments[] = {&handle, &px, &py, &packedInput[0], &packedInput[1],
                                 &packedInput[2], &packedInput[3]};
            packedStatus = report("cuLaunchKernel(format 80 formatted store)",
                                  launchKernel(normalizedFunction, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                               arguments, nullptr), getErrorString);
            if (packedStatus == 0)
                packedStatus = report("cuCtxSynchronize(format 80)", synchronize(), getErrorString);
        }
        if (packedStatus == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 10; // HIP array, inspect physical backing
            readback.srcArray = packedArray;
            readback.dstMemoryType = 1;
            readback.dstHost = packedPixels.data();
            readback.dstPitch = 4 * 4 * sizeof(float);
            readback.WidthInBytes = readback.dstPitch;
            readback.Height = 4;
            packedStatus = report("hipMemcpyParam2D(format 80 backing readback)", backingCopy2D(&readback));
        }
        const size_t packedPixelOffset = (2 * 4 + 1) * 4;
        const std::array<float, 4> packedExpected{
            126.0f / 1023.0f, 513.0f / 1023.0f, 897.0f / 1023.0f, 1.0f / 3.0f};
        bool packedValid = packedStatus == 0;
        for (size_t index = 0; index < packedPixels.size(); ++index)
        {
            const float expected = index >= packedPixelOffset && index < packedPixelOffset + 4
                ? packedExpected[index - packedPixelOffset] : 0.0f;
            if (std::fabs(packedPixels[index] - expected) > 1e-6f)
                packedValid = false;
        }
        std::printf("format 80 backing: actual=%g,%g,%g,%g expected=%g,%g,%g,%g: %s\n",
                    packedPixels[packedPixelOffset], packedPixels[packedPixelOffset + 1],
                    packedPixels[packedPixelOffset + 2], packedPixels[packedPixelOffset + 3],
                    packedExpected[0], packedExpected[1], packedExpected[2], packedExpected[3],
                    packedValid ? "PASS" : "FAIL");
        valid &= packedValid;
        // The public CUDA array is four bytes per pixel even though ZLUDA
        // uses sixteen bytes per pixel internally. Test both copy directions
        // with ordinary packed host rows, including their documented pitch.
        std::array<uint32_t, 4 * 4> packedHost{};
        for (size_t index = 0; index < packedHost.size(); ++index)
        {
            const uint32_t red = uint32_t(index * 137u) & 1023u;
            const uint32_t green = uint32_t(index * 257u) & 1023u;
            const uint32_t blue = uint32_t(index * 379u) & 1023u;
            const uint32_t alpha = uint32_t(index) & 3u;
            packedHost[index] = red | (green << 10) | (blue << 20) | (alpha << 30);
        }
        if (packedStatus == 0)
        {
            CUDA_MEMCPY2D upload{};
            upload.srcMemoryType = 1;
            upload.srcHost = packedHost.data();
            upload.srcPitch = 4 * sizeof(uint32_t);
            upload.dstMemoryType = 3;
            upload.dstArray = packedArray;
            upload.WidthInBytes = upload.srcPitch;
            upload.Height = 4;
            packedStatus = report("cuMemcpy2D_v2(format 80 packed host upload)", copy2D(&upload), getErrorString);
        }
        std::array<float, 4 * 4 * 4> copiedBacking{};
        if (packedStatus == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 10;
            readback.srcArray = packedArray;
            readback.dstMemoryType = 1;
            readback.dstHost = copiedBacking.data();
            readback.dstPitch = 4 * 4 * sizeof(float);
            readback.WidthInBytes = readback.dstPitch;
            readback.Height = 4;
            packedStatus = report("hipMemcpyParam2D(format 80 backing after packed upload)", backingCopy2D(&readback));
        }
        bool packedCopyValid = packedStatus == 0;
        if (packedCopyValid)
        {
            for (size_t index = 0; index < packedHost.size(); ++index)
            {
                const uint32_t word = packedHost[index];
                const std::array<float, 4> expected{
                    float(word & 1023u) / 1023.0f,
                    float((word >> 10) & 1023u) / 1023.0f,
                    float((word >> 20) & 1023u) / 1023.0f,
                    float(word >> 30) / 3.0f};
                for (size_t channel = 0; channel < 4; ++channel)
                    packedCopyValid &= std::fabs(copiedBacking[index * 4 + channel] - expected[channel]) < 1e-6f;
            }
        }
        std::array<uint32_t, 4 * 4> packedRoundTrip{};
        if (packedStatus == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 3;
            readback.srcArray = packedArray;
            readback.dstMemoryType = 1;
            readback.dstHost = packedRoundTrip.data();
            readback.dstPitch = 4 * sizeof(uint32_t);
            readback.WidthInBytes = readback.dstPitch;
            readback.Height = 4;
            packedStatus = report("cuMemcpy2D_v2(format 80 packed host readback)", copy2D(&readback), getErrorString);
            packedCopyValid &= packedStatus == 0 && packedRoundTrip == packedHost;
        }
        std::printf("format 80 packed host copy: %s\n", packedCopyValid ? "PASS" : "FAIL");
        valid &= packedCopyValid;
        std::array<uint32_t, 5 * 4> paddedHost{};
        for (size_t index = 0; index < paddedHost.size(); ++index)
            paddedHost[index] = (uint32_t(index * 17u) & 1023u)
                | ((uint32_t(index * 29u) & 1023u) << 10)
                | ((uint32_t(index * 41u) & 1023u) << 20)
                | ((uint32_t(index) & 3u) << 30);
        std::array<uint32_t, 4 * 4> subrectExpected = packedHost;
        for (size_t row = 0; row < 2; ++row)
            for (size_t column = 0; column < 2; ++column)
                subrectExpected[(row + 1) * 4 + column + 1] = paddedHost[(row + 1) * 5 + column + 1];
        if (packedStatus == 0)
        {
            CUDA_MEMCPY2D upload{};
            upload.srcXInBytes = sizeof(uint32_t);
            upload.srcY = 1;
            upload.srcMemoryType = 1;
            upload.srcHost = paddedHost.data();
            upload.srcPitch = 5 * sizeof(uint32_t);
            upload.dstXInBytes = sizeof(uint32_t);
            upload.dstY = 1;
            upload.dstMemoryType = 3;
            upload.dstArray = packedArray;
            upload.WidthInBytes = 2 * sizeof(uint32_t);
            upload.Height = 2;
            packedStatus = report("cuMemcpy2D_v2(format 80 packed subrect upload)", copy2D(&upload), getErrorString);
        }
        if (packedStatus == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 3;
            readback.srcArray = packedArray;
            readback.dstMemoryType = 1;
            readback.dstHost = packedRoundTrip.data();
            readback.dstPitch = 4 * sizeof(uint32_t);
            readback.WidthInBytes = readback.dstPitch;
            readback.Height = 4;
            packedStatus = report("cuMemcpy2D_v2(format 80 after subrect)", copy2D(&readback), getErrorString);
        }
        const bool subrectValid = packedStatus == 0 && packedRoundTrip == subrectExpected;
        std::printf("format 80 packed subrect/pitch: %s\n", subrectValid ? "PASS" : "FAIL");
        valid &= subrectValid;
        if (packedStatus == 0)
        {
            uint64_t handle = packedSurface;
            unsigned int px = 1, py = 2;
            void* arguments[] = {&handle, &px, &py, &packedInput[0], &packedInput[1],
                                 &packedInput[2], &packedInput[3]};
            packedStatus = report("cuLaunchKernel(format 80 store for packed readback)",
                                  launchKernel(normalizedFunction, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                               arguments, nullptr), getErrorString);
            if (packedStatus == 0)
                packedStatus = report("cuCtxSynchronize(format 80 packed readback)", synchronize(), getErrorString);
        }
        if (packedStatus == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 3;
            readback.srcArray = packedArray;
            readback.dstMemoryType = 1;
            readback.dstHost = packedRoundTrip.data();
            readback.dstPitch = 4 * sizeof(uint32_t);
            readback.WidthInBytes = readback.dstPitch;
            readback.Height = 4;
            packedStatus = report("cuMemcpy2D_v2(format 80 GPU store to packed host)", copy2D(&readback), getErrorString);
        }
        subrectExpected[2 * 4 + 1] = 126u | (513u << 10) | (897u << 20) | (1u << 30);
        const bool gpuStoreReadbackValid = packedStatus == 0 && packedRoundTrip == subrectExpected;
        std::printf("format 80 GPU store to packed host: %s\n", gpuStoreReadbackValid ? "PASS" : "FAIL");
        valid &= gpuStoreReadbackValid;
        CUdeviceptr devicePacked = 0;
        CUresult deviceStatus = report("cuMemAlloc_v2(format 80 packed device rows)",
                                       memAlloc(&devicePacked, packedHost.size() * sizeof(uint32_t)), getErrorString);
        if (deviceStatus == 0)
        {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = 1;
            copy.srcHost = packedHost.data();
            copy.srcPitch = 4 * sizeof(uint32_t);
            copy.dstMemoryType = 2;
            copy.dstDevice = devicePacked;
            copy.dstPitch = copy.srcPitch;
            copy.WidthInBytes = copy.srcPitch;
            copy.Height = 4;
            deviceStatus = report("cuMemcpy2D_v2(format 80 host to device staging)", copy2D(&copy), getErrorString);
        }
        if (deviceStatus == 0)
        {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = 2;
            copy.srcDevice = devicePacked;
            copy.srcPitch = 4 * sizeof(uint32_t);
            copy.dstMemoryType = 3;
            copy.dstArray = packedArray;
            copy.WidthInBytes = copy.srcPitch;
            copy.Height = 4;
            deviceStatus = report("cuMemcpy2D_v2(format 80 packed device to array)", copy2D(&copy), getErrorString);
        }
        if (deviceStatus == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 10;
            readback.srcArray = packedArray;
            readback.dstMemoryType = 1;
            readback.dstHost = copiedBacking.data();
            readback.dstPitch = 4 * 4 * sizeof(float);
            readback.WidthInBytes = readback.dstPitch;
            readback.Height = 4;
            deviceStatus = report("hipMemcpyParam2D(format 80 backing after device upload)", backingCopy2D(&readback));
        }
        bool deviceCopyValid = deviceStatus == 0;
        if (deviceCopyValid)
        {
            for (size_t index = 0; index < packedHost.size(); ++index)
            {
                const uint32_t word = packedHost[index];
                const std::array<float, 4> expected{
                    float(word & 1023u) / 1023.0f,
                    float((word >> 10) & 1023u) / 1023.0f,
                    float((word >> 20) & 1023u) / 1023.0f,
                    float(word >> 30) / 3.0f};
                for (size_t channel = 0; channel < 4; ++channel)
                    deviceCopyValid &= std::fabs(copiedBacking[index * 4 + channel] - expected[channel]) < 1e-6f;
            }
        }
        if (deviceStatus == 0)
        {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = 3;
            copy.srcArray = packedArray;
            copy.dstMemoryType = 2;
            copy.dstDevice = devicePacked;
            copy.dstPitch = 4 * sizeof(uint32_t);
            copy.WidthInBytes = copy.dstPitch;
            copy.Height = 4;
            deviceStatus = report("cuMemcpy2D_v2(format 80 array to packed device)", copy2D(&copy), getErrorString);
        }
        if (deviceStatus == 0)
        {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = 2;
            copy.srcDevice = devicePacked;
            copy.srcPitch = 4 * sizeof(uint32_t);
            copy.dstMemoryType = 1;
            copy.dstHost = packedRoundTrip.data();
            copy.dstPitch = copy.srcPitch;
            copy.WidthInBytes = copy.srcPitch;
            copy.Height = 4;
            deviceStatus = report("cuMemcpy2D_v2(format 80 packed device to host)", copy2D(&copy), getErrorString);
            deviceCopyValid &= deviceStatus == 0 && packedRoundTrip == packedHost;
        }
        std::printf("format 80 packed device copy: %s\n", deviceCopyValid ? "PASS" : "FAIL");
        valid &= deviceCopyValid;
        if (devicePacked != 0)
            memFree(devicePacked);
        CUarray otherPackedArray = nullptr;
        CUresult arrayCopyStatus = report("CUDA format 80 second array",
                                          arrayCreate(&otherPackedArray, &packedDescriptor), getErrorString);
        if (arrayCopyStatus == 0)
        {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = 3;
            copy.srcArray = packedArray;
            copy.dstMemoryType = 3;
            copy.dstArray = otherPackedArray;
            copy.WidthInBytes = 4 * sizeof(uint32_t);
            copy.Height = 4;
            arrayCopyStatus = report("cuMemcpy2D_v2(format 80 array to array)", copy2D(&copy), getErrorString);
        }
        if (arrayCopyStatus == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 3;
            readback.srcArray = otherPackedArray;
            readback.dstMemoryType = 1;
            readback.dstHost = packedRoundTrip.data();
            readback.dstPitch = 4 * sizeof(uint32_t);
            readback.WidthInBytes = readback.dstPitch;
            readback.Height = 4;
            arrayCopyStatus = report("cuMemcpy2D_v2(format 80 second array readback)", copy2D(&readback), getErrorString);
        }
        const bool arrayCopyValid = arrayCopyStatus == 0 && packedRoundTrip == packedHost;
        std::printf("format 80 array to array copy: %s\n", arrayCopyValid ? "PASS" : "FAIL");
        valid &= arrayCopyValid;
        if (otherPackedArray != nullptr)
            arrayDestroy(otherPackedArray);
        if (packedSurface != 0)
            surfaceDestroy(packedSurface);
        if (packedArray != nullptr)
            arrayDestroy(packedArray);
    }
    exitCode = valid ? 0 : 1;

    if (surface != 0)
        surfaceDestroy(surface);
    if (array != nullptr)
        arrayDestroy(array);
    if (module != nullptr)
        moduleUnload(module);
    if (context != nullptr)
        contextDestroy(context);
    dlclose(hipLibrary);
    dlclose(library);
    return exitCode;
}
