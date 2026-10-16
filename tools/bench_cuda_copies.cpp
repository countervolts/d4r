// Measures host<->array and host<->linear copy bandwidth through the CUDA
// driver API (ZLUDA), for the staging sizes the NGX shim moves per frame.
// usage: bench_cuda_copies [WIDTH HEIGHT CHANNELS]  (half-float texels)
#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using CUresult = int;
using CUdeviceptr = uint64_t;

struct Memcpy2D
{
    size_t srcXInBytes, srcY;
    uint32_t srcMemoryType, srcAlignment;
    const void* srcHost;
    CUdeviceptr srcDevice;
    void* srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY;
    uint32_t dstMemoryType, dstAlignment;
    void* dstHost;
    CUdeviceptr dstDevice;
    void* dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
};

struct ArrayDescriptor
{
    size_t Width, Height;
    int32_t Format;
    uint32_t NumChannels;
};

template <typename Function> static Function load(void* library, const char* name)
{
    void* raw = dlsym(library, name);
    if (raw == nullptr)
    {
        std::fprintf(stderr, "missing %s\n", name);
        std::exit(1);
    }
    Function function{};
    std::memcpy(&function, &raw, sizeof(function));
    return function;
}

int main(int argc, char** argv)
{
    const size_t width = argc > 3 ? std::strtoul(argv[1], nullptr, 0) : 2560;
    const size_t height = argc > 3 ? std::strtoul(argv[2], nullptr, 0) : 1440;
    const unsigned channels = argc > 3 ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 0)) : 4;
    const size_t rowBytes = width * 2 * channels, bytes = rowBytes * height;
    void* library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
        return 1;
    auto init = load<CUresult (*)(unsigned)>(library, "cuInit");
    auto device_get = load<CUresult (*)(int*, int)>(library, "cuDeviceGet");
    auto context_create = load<CUresult (*)(void**, unsigned, int)>(library, "cuCtxCreate_v2");
    auto alloc_host = load<CUresult (*)(void**, size_t, unsigned)>(library, "cuMemHostAlloc");
    auto alloc = load<CUresult (*)(CUdeviceptr*, size_t)>(library, "cuMemAlloc_v2");
    auto array_create = load<CUresult (*)(void**, const ArrayDescriptor*)>(library, "cuArrayCreate_v2");
    auto copy = load<CUresult (*)(const Memcpy2D*)>(library, "cuMemcpy2D_v2");
    auto synchronize = load<CUresult (*)()>(library, "cuCtxSynchronize");
    int device = 0;
    void* context = nullptr;
    void *pinned = nullptr, *array = nullptr;
    CUdeviceptr linear = 0;
    CUresult result = 0;
#define STEP(call) \
    if ((result = (call)) != 0) \
    { \
        std::fprintf(stderr, "%s failed: %d\n", #call, result); \
        return 1; \
    }
    STEP(init(0));
    STEP(device_get(&device, 0));
    STEP(context_create(&context, 0, device));
    STEP(alloc_host(&pinned, bytes, 0));
    STEP(alloc(&linear, bytes));
    const ArrayDescriptor descriptor{width, height, 0x10 /* HALF */, channels};
    STEP(array_create(&array, &descriptor));
    void* pageable = std::malloc(bytes);
    std::memset(pageable, 1, bytes);
    std::memset(pinned, 1, bytes);
    enum { HOST = 1, DEVICE = 2, ARRAY = 3 };
    auto run = [&](const char* label, uint32_t srcType, uint32_t dstType, void* host) {
        Memcpy2D request{};
        request.srcMemoryType = srcType;
        request.dstMemoryType = dstType;
        request.srcHost = host;
        request.dstHost = host;
        request.srcDevice = linear;
        request.dstDevice = linear;
        request.srcArray = array;
        request.dstArray = array;
        request.srcPitch = request.dstPitch = rowBytes;
        request.WidthInBytes = rowBytes;
        request.Height = height;
        double best = 1e9;
        for (int iteration = 0; iteration < 10; ++iteration)
        {
            synchronize();
            const auto start = std::chrono::steady_clock::now();
            if (copy(&request) || synchronize())
            {
                std::printf("%-26s failed\n", label);
                return;
            }
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            best = ms < best ? ms : best;
        }
        std::printf("%-26s %7.3f ms  %6.2f GB/s\n", label, best, bytes / best / 1e6);
    };
    if (argc > 1 && std::strcmp(argv[1], "overlap") == 0)
    {
        // The shim's per-frame transfers: color 856x480 RGBA16F, depth 856x480
        // R32F and motion 2560x1440 RG16F up; the 2560x1440 RGBA16F result down.
        // Compares issuing them back to back with overlapping the upload and
        // the download on two streams (pinned host memory, async copies).
        auto create_stream = load<CUresult (*)(void**, unsigned)>(library, "cuStreamCreate");
        auto stream_sync = load<CUresult (*)(void*)>(library, "cuStreamSynchronize");
        struct Plane { size_t w, h; unsigned ch; int32_t fmt; size_t texel; void* host; void* array; };
        Plane planes[4] = {{856, 480, 4, 0x10, 8}, {856, 480, 1, 0x20, 4}, {2560, 1440, 2, 0x10, 4},
                           {2560, 1440, 4, 0x10, 8}};
        for (Plane& plane : planes)
        {
            STEP(alloc_host(&plane.host, plane.w * plane.h * plane.texel, 0));
            std::memset(plane.host, 1, plane.w * plane.h * plane.texel);
            const ArrayDescriptor d{plane.w, plane.h, plane.fmt, plane.ch};
            STEP(array_create(&plane.array, &d));
        }
        void *up = nullptr, *down = nullptr;
        STEP(create_stream(&up, 1));   // CU_STREAM_NON_BLOCKING
        STEP(create_stream(&down, 1));
        auto request = [&](Plane& plane, bool upload) {
            Memcpy2D r{};
            r.srcMemoryType = upload ? HOST : ARRAY;
            r.dstMemoryType = upload ? ARRAY : HOST;
            r.srcHost = plane.host;
            r.dstHost = plane.host;
            r.srcArray = r.dstArray = plane.array;
            r.srcPitch = r.dstPitch = plane.w * plane.texel;
            r.WidthInBytes = plane.w * plane.texel;
            r.Height = plane.h;
            return r;
        };
        // Linear device staging per plane: the async DMA runs host<->linear on
        // two non-blocking streams; arrays are filled/read with fast D2D copies.
        auto copy_h2d_async = load<CUresult (*)(CUdeviceptr, const void*, size_t, void*)>(library, "cuMemcpyHtoDAsync_v2");
        auto copy_d2h_async = load<CUresult (*)(void*, CUdeviceptr, size_t, void*)>(library, "cuMemcpyDtoHAsync_v2");
        CUdeviceptr staging[4] = {};
        for (int i = 0; i < 4; ++i)
            STEP(alloc(&staging[i], planes[i].w * planes[i].h * planes[i].texel));
        auto d2d = [&](Plane& plane, CUdeviceptr linear, bool toArray) {
            Memcpy2D r{};
            r.srcMemoryType = toArray ? DEVICE : ARRAY;
            r.dstMemoryType = toArray ? ARRAY : DEVICE;
            r.srcDevice = r.dstDevice = linear;
            r.srcArray = r.dstArray = plane.array;
            r.srcPitch = r.dstPitch = plane.w * plane.texel;
            r.WidthInBytes = plane.w * plane.texel;
            r.Height = plane.h;
            return copy(&r);
        };
        for (int mode = 0; mode < 3; ++mode)
        {
            double best = 1e9;
            for (int iteration = 0; iteration < 20; ++iteration)
            {
                synchronize();
                const auto start = std::chrono::steady_clock::now();
                CUresult status = 0;
                if (mode == 0) // current shim: synchronous 2D copies, one after another
                {
                    for (int i = 0; i < 3; ++i) { Memcpy2D r = request(planes[i], true); status |= copy(&r); }
                    Memcpy2D r = request(planes[3], false);
                    status |= copy(&r);
                }
                else
                {
                    const size_t outBytes = planes[3].w * planes[3].h * planes[3].texel;
                    status |= d2d(planes[3], staging[3], false);
                    status |= copy_d2h_async(planes[3].host, staging[3], outBytes, down);
                    if (mode == 2) // serialized control: wait for the download first
                        status |= stream_sync(down);
                    for (int i = 0; i < 3; ++i)
                        status |= copy_h2d_async(staging[i], planes[i].host, planes[i].w * planes[i].h * planes[i].texel, up);
                    status |= stream_sync(up);
                    for (int i = 0; i < 3; ++i)
                        status |= d2d(planes[i], staging[i], true);
                    status |= stream_sync(down);
                }
                synchronize();
                if (status != 0)
                {
                    std::printf("mode %d failed\n", mode);
                    break;
                }
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                best = ms < best ? ms : best;
            }
            std::printf("%-46s %7.3f ms\n", mode == 0 ? "sequential sync 2D (current shim path)" : mode == 1 ? "linear staging, down || up overlapped" : "linear staging, down then up (no overlap)", best);
        }
        return 0;
    }
    std::printf("%zux%zu x%u half = %.1f MB\n", width, height, channels, bytes / 1e6);
    run("pageable -> array", HOST, ARRAY, pageable);
    run("pinned   -> array", HOST, ARRAY, pinned);
    run("pinned   -> linear", HOST, DEVICE, pinned);
    run("linear   -> array", DEVICE, ARRAY, nullptr);
    run("array    -> pageable", ARRAY, HOST, pageable);
    run("array    -> pinned", ARRAY, HOST, pinned);
    run("array    -> linear", ARRAY, DEVICE, nullptr);
    run("linear   -> pinned", DEVICE, HOST, pinned);
    return 0;
}
