#include <dlfcn.h>

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

// One unformatted 4-byte store (sust.b.2d.v2.b16.zero, and the same bytes via
// sust.b.2d.b32.zero) into a surface of the given CUDA array format. X is a byte offset (unformatted surface addressing). The expected
// result is the fill pattern with the four stored bytes replaced at X, or the
// untouched fill pattern when X is out of bounds.
struct StoreCase
{
    const char* name;
    uint32_t format;      // CUarray_format
    uint32_t channels;
    uint32_t channelBytes;
    uint32_t x;           // byte offset
    uint32_t y;
    uint16_t value0;
    uint16_t value1;
    uint8_t fill[16];     // one pixel of initial contents
};

int main()
{
    constexpr unsigned int width = 16;
    constexpr unsigned int height = 16;
    const char* ptx = R"PTX(
.version 8.9
.target sm_89
.address_size 64

.visible .entry d4r_surface_half_store_probe(
.param .b64 surface,
.param .b32 x,
.param .b32 y,
.param .b32 packed
)
{
    .reg .b64 %rd<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<4>;
    ld.param.b64 %rd1, [surface];
    ld.param.b32 %r0, [x];
    ld.param.b32 %r1, [y];
    ld.param.b32 %r2, [packed];
    mov.b32 {%rs0, %rs1}, %r2;
    sust.b.2d.v2.b16.zero [%rd1, {%r0,%r1}], {%rs0,%rs1};
    ret;
}

.visible .entry d4r_surface_b32_store_probe(
.param .b64 surface,
.param .b32 x,
.param .b32 y,
.param .b32 packed
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<4>;
    ld.param.b64 %rd1, [surface];
    ld.param.b32 %r0, [x];
    ld.param.b32 %r1, [y];
    ld.param.b32 %r2, [packed];
    sust.b.2d.b32.zero [%rd1, {%r0,%r1}], %r2;
    ret;
}
)PTX";

    // Fill values: 0x3400 is half 0.25, 0x3e800000 is float 0.25.
    const StoreCase cases[] = {
        {"R16F store spans two pixels", 0x10, 1, 2, 4 * 2, 6, 0x3555, 0xc248,
         {0x00, 0x34}},
        {"R16F row start pair", 0x10, 1, 2, 0, 0, 0x8001, 0x7bff,
         {0x00, 0x34}},
        {"R16F row end pair", 0x10, 1, 2, width * 2 - 4, 15, 0xfc00, 0x0400,
         {0x00, 0x34}},
        {"R16 uint store spans two pixels", 0x02, 1, 2, 10 * 2, 3, 0xfffe, 0x1234,
         {0x11, 0x22}},
        {"R8 uint store spans four pixels", 0x01, 1, 1, 8, 5, 0x80ff, 0x017f,
         {0x5a}},
        {"RG16F in-bounds", 0x10, 2, 2, 5 * 4, 7, 0x3555, 0xc248,
         {0x00, 0x34, 0x00, 0x34}},
        {"RG16F subnormal/max", 0x10, 2, 2, 9 * 4, 3, 0x0001, 0x7bff,
         {0x00, 0x34, 0x00, 0x34}},
        {"RG16F negative zero/min normal", 0x10, 2, 2, 0, 0, 0x8000, 0x0400,
         {0x00, 0x34, 0x00, 0x34}},
        {"RG16F .zero out-of-bounds", 0x10, 2, 2, width * 4, 7, 0x3555, 0xc248,
         {0x00, 0x34, 0x00, 0x34}},
        {"RGBA16F upper half-pixel", 0x10, 4, 2, 3 * 8 + 4, 2, 0x3555, 0xc248,
         {0x00, 0x34, 0x00, 0x34, 0x00, 0x34, 0x00, 0x34}},
        {"RGBA16F lower half-pixel", 0x10, 4, 2, 11 * 8, 12, 0xbc00, 0x4900,
         {0x00, 0x34, 0x00, 0x34, 0x00, 0x34, 0x00, 0x34}},
        {"R32F packed halves", 0x20, 1, 4, 6 * 4, 1, 0x3555, 0xc248,
         {0x00, 0x00, 0x80, 0x3e}},
        {"RG16 uint", 0x02, 2, 2, 4 * 4, 9, 0xfffe, 0x1234,
         {0x11, 0x11, 0x22, 0x22}},
        {"R32F denormal bits", 0x20, 1, 4, 2 * 4, 4, 0x0001, 0x0000,
         {0x00, 0x00, 0x80, 0x3e}},
        {"RG16F infinity/quiet NaN", 0x10, 2, 2, 13 * 4, 13, 0xfc00, 0x7e00,
         {0x00, 0x34, 0x00, 0x34}},
    };

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
    using Array3DCreateFn = CUresult (*)(CUarray*, const CUDA_ARRAY3D_DESCRIPTOR*);
    using ArrayDestroyFn = CUresult (*)(CUarray);
    using SurfaceCreateFn = CUresult (*)(CUsurfObject*, const CUDA_RESOURCE_DESC*);
    using SurfaceDestroyFn = CUresult (*)(CUsurfObject);
    using Copy2DFn = CUresult (*)(const CUDA_MEMCPY2D*);
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
    const auto arrayDestroy = load_function<ArrayDestroyFn>(library, "cuArrayDestroy");
    const auto surfaceCreate = load_function<SurfaceCreateFn>(library, "cuSurfObjectCreate");
    const auto surfaceDestroy = load_function<SurfaceDestroyFn>(library, "cuSurfObjectDestroy");
    const auto copy2D = load_function<Copy2DFn>(library, "cuMemcpy2D_v2");
    const auto moduleLoad = load_function<ModuleLoadFn>(library, "cuModuleLoadData");
    const auto moduleGetFunction = load_function<ModuleGetFunctionFn>(library, "cuModuleGetFunction");
    const auto launchKernel = load_function<LaunchKernelFn>(library, "cuLaunchKernel");
    const auto synchronize = load_function<ContextSynchronizeFn>(library, "cuCtxSynchronize");
    const auto moduleUnload = load_function<ModuleUnloadFn>(library, "cuModuleUnload");
    const auto getErrorString = load_function<GetErrorStringFn>(library, "cuGetErrorString");
    if (init == nullptr || deviceGet == nullptr || contextCreate == nullptr || contextDestroy == nullptr ||
        array3DCreate == nullptr || arrayDestroy == nullptr || surfaceCreate == nullptr ||
        surfaceDestroy == nullptr || copy2D == nullptr || moduleLoad == nullptr ||
        moduleGetFunction == nullptr || launchKernel == nullptr || synchronize == nullptr ||
        moduleUnload == nullptr)
    {
        std::fprintf(stderr, "ZLUDA libcuda.so is missing one or more half-surface probe exports\n");
        dlclose(library);
        return 1;
    }

    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction functions[2] = {};
    const char* functionNames[2] = {"sust.b.2d.v2.b16", "sust.b.2d.b32"};
    CUresult result = report("cuInit", init(0), getErrorString);
    CUdevice device = 0;
    if (result == 0)
        result = report("cuDeviceGet", deviceGet(&device, 0), getErrorString);
    if (result == 0)
        result = report("cuCtxCreate_v2", contextCreate(&context, 0, device), getErrorString);
    if (result == 0)
        result = report("cuModuleLoadData(half surface store PTX)", moduleLoad(&module, ptx), getErrorString);
    if (result == 0)
        result = report("cuModuleGetFunction(d4r_surface_half_store_probe)",
                        moduleGetFunction(&functions[0], module, "d4r_surface_half_store_probe"), getErrorString);
    if (result == 0)
        result = report("cuModuleGetFunction(d4r_surface_b32_store_probe)",
                        moduleGetFunction(&functions[1], module, "d4r_surface_b32_store_probe"), getErrorString);

    size_t passed = 0;
    size_t total = 2 * sizeof(cases) / sizeof(cases[0]);
    for (size_t functionIndex = 0; functionIndex < 2; ++functionIndex)
    for (const StoreCase& storeCase : cases)
    {
        CUfunction function = functions[functionIndex];
        if (result != 0)
            break;
        const size_t pixelBytes = static_cast<size_t>(storeCase.channels) * storeCase.channelBytes;
        const size_t rowBytes = width * pixelBytes;
        std::vector<uint8_t> expected(rowBytes * height);
        for (size_t offset = 0; offset < expected.size(); ++offset)
            expected[offset] = storeCase.fill[offset % pixelBytes];
        std::vector<uint8_t> surfaceBytes = expected;
        const uint8_t stored[4] = {
            static_cast<uint8_t>(storeCase.value0), static_cast<uint8_t>(storeCase.value0 >> 8),
            static_cast<uint8_t>(storeCase.value1), static_cast<uint8_t>(storeCase.value1 >> 8)};
        if (storeCase.x + 4 <= rowBytes && storeCase.y < height)
            std::memcpy(expected.data() + storeCase.y * rowBytes + storeCase.x, stored, 4);

        CUarray array = nullptr;
        CUsurfObject surface = 0;
        const CUDA_ARRAY3D_DESCRIPTOR arrayDescriptor{width, height, 0, storeCase.format,
                                                      storeCase.channels, 2 /* SURFACE_LDST */};
        CUresult caseResult = array3DCreate(&array, &arrayDescriptor);
        if (caseResult == 0)
        {
            CUDA_RESOURCE_DESC resource{};
            resource.resType = 0; // CUDA_RESOURCE_TYPE_ARRAY
            resource.res.array.hArray = array;
            caseResult = surfaceCreate(&surface, &resource);
        }
        if (caseResult == 0)
        {
            CUDA_MEMCPY2D upload{};
            upload.srcMemoryType = 1; // CUDA_MEMORYTYPE_HOST
            upload.srcHost = surfaceBytes.data();
            upload.srcPitch = rowBytes;
            upload.dstMemoryType = 3; // CUDA_MEMORYTYPE_ARRAY
            upload.dstArray = array;
            upload.WidthInBytes = rowBytes;
            upload.Height = height;
            caseResult = copy2D(&upload);
        }
        if (caseResult == 0)
        {
            uint64_t surfaceArgument = surface;
            uint32_t x = storeCase.x;
            uint32_t y = storeCase.y;
            uint32_t packed = storeCase.value0 | (static_cast<uint32_t>(storeCase.value1) << 16);
            void* arguments[] = {&surfaceArgument, &x, &y, &packed};
            caseResult = launchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr);
            if (caseResult == 0)
                caseResult = synchronize();
        }
        if (caseResult == 0)
        {
            CUDA_MEMCPY2D readback{};
            readback.srcMemoryType = 3; // CUDA_MEMORYTYPE_ARRAY
            readback.srcArray = array;
            readback.dstMemoryType = 1; // CUDA_MEMORYTYPE_HOST
            readback.dstHost = surfaceBytes.data();
            readback.dstPitch = rowBytes;
            readback.WidthInBytes = rowBytes;
            readback.Height = height;
            caseResult = copy2D(&readback);
        }

        size_t mismatches = 0;
        for (size_t offset = 0; caseResult == 0 && offset < expected.size(); offset += pixelBytes)
        {
            if (std::memcmp(surfaceBytes.data() + offset, expected.data() + offset, pixelBytes) == 0)
                continue;
            ++mismatches;
            std::printf("  pixel (%zu,%zu) got", (offset % rowBytes) / pixelBytes, offset / rowBytes);
            for (size_t b = 0; b < pixelBytes; ++b)
                std::printf(" %02x", surfaceBytes[offset + b]);
            std::printf(" expected");
            for (size_t b = 0; b < pixelBytes; ++b)
                std::printf(" %02x", expected[offset + b]);
            std::putchar('\n');
        }
        const bool casePassed = caseResult == 0 && mismatches == 0;
        passed += casePassed ? 1 : 0;
        std::printf("%s %s: %s (CUDA result=%d, mismatched pixels=%zu)\n", functionNames[functionIndex],
                    storeCase.name, casePassed ? "PASS" : "FAIL", caseResult, mismatches);
        if (surface != 0)
            surfaceDestroy(surface);
        if (array != nullptr)
            arrayDestroy(array);
    }

    if (result == 0)
        std::printf("sust.b.2d unformatted 4-byte surface store tests: %zu/%zu passed\n", passed, total);
    const int exitCode = result == 0 && passed == total ? 0 : 1;
    if (module != nullptr)
        moduleUnload(module);
    if (context != nullptr)
        contextDestroy(context);
    dlclose(library);
    return exitCode;
}
