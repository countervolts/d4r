// Replays one captured CUDA launch (see D4R_CUDA_REPLAY_DUMP_DIR in
// tools/wine_nvcuda_bridge.c) through ZLUDA in isolation, to time a single
// DLSS kernel and to compare its output across ZLUDA builds.
//
// usage: bench_dlss_kernel FATBIN REPLAY_DIR [ITERATIONS [OUTPUT_PREFIX]]
//
// Every allocation the launch points into is restored from the capture
// before each launch. After the first launch each allocation is written to
// OUTPUT_PREFIX-alloc-N.bin (if given) and its FNV-1a hash is printed.
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using CUresult = int;
using CUdeviceptr = uint64_t;

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

static std::vector<unsigned char> read_file(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(2);
    }
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

#define CHECK(call) \
    do \
    { \
        const CUresult check_result = (call); \
        if (check_result != 0) \
        { \
            std::fprintf(stderr, "%s failed: %d\n", #call, check_result); \
            return 1; \
        } \
    } while (0)

struct Allocation
{
    size_t bytes = 0;
    std::vector<unsigned char> data;
    CUdeviceptr device = 0;
};

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: bench_dlss_kernel FATBIN REPLAY_DIR [ITERATIONS [OUTPUT_PREFIX]]\n");
        return 2;
    }
    const std::string directory = argv[2];
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 20;
    const char* output_prefix = argc > 4 ? argv[4] : nullptr;

    std::string kernel;
    unsigned int launch[7] = {};
    std::vector<Allocation> allocations;
    struct Pointer { size_t offset, allocation; uint64_t delta; };
    std::vector<Pointer> pointers;
    {
        std::ifstream manifest(directory + "/manifest.txt");
        std::string line;
        while (std::getline(manifest, line))
        {
            std::istringstream fields(line);
            std::string tag;
            fields >> tag;
            if (tag == "kernel")
                fields >> kernel;
            else if (tag == "launch")
                for (unsigned int& value : launch)
                    fields >> value;
            else if (tag == "alloc")
            {
                size_t index = 0, bytes = 0;
                std::string base;
                fields >> index >> base >> bytes;
                if (allocations.size() <= index)
                    allocations.resize(index + 1);
                allocations[index].bytes = bytes;
                allocations[index].data = read_file(directory + "/alloc-" + std::to_string(index) + ".bin");
            }
            else if (tag == "pointer")
            {
                Pointer pointer{};
                fields >> pointer.offset >> pointer.allocation >> pointer.delta;
                pointers.push_back(pointer);
            }
            else if (tag == "surface" || tag == "texture")
            {
                std::fprintf(stderr, "replay of %s arguments is not supported\n", tag.c_str());
                return 2;
            }
        }
    }
    std::vector<unsigned char> arguments = read_file(directory + "/args.bin");
    std::vector<unsigned char> image = read_file(argv[1]);
    image.resize(image.size() + 16, 0);

    void* library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
    {
        std::fprintf(stderr, "dlopen(libcuda.so): %s\n", dlerror());
        return 1;
    }
    auto init = load<CUresult (*)(unsigned int)>(library, "cuInit");
    auto device_get = load<CUresult (*)(int*, int)>(library, "cuDeviceGet");
    auto context_create = load<CUresult (*)(void**, unsigned int, int)>(library, "cuCtxCreate_v2");
    auto module_load = load<CUresult (*)(void**, const void*)>(library, "cuModuleLoadData");
    auto get_function = load<CUresult (*)(void**, void*, const char*)>(library, "cuModuleGetFunction");
    auto mem_alloc = load<CUresult (*)(CUdeviceptr*, size_t)>(library, "cuMemAlloc_v2");
    auto copy_h2d = load<CUresult (*)(CUdeviceptr, const void*, size_t)>(library, "cuMemcpyHtoD_v2");
    auto copy_d2h = load<CUresult (*)(void*, CUdeviceptr, size_t)>(library, "cuMemcpyDtoH_v2");
    auto synchronize = load<CUresult (*)()>(library, "cuCtxSynchronize");
    auto event_create = load<CUresult (*)(void**, unsigned int)>(library, "cuEventCreate");
    auto event_record = load<CUresult (*)(void*, void*)>(library, "cuEventRecord");
    auto event_elapsed = load<CUresult (*)(float*, void*, void*)>(library, "cuEventElapsedTime");
    auto launch_kernel = load<CUresult (*)(void*, unsigned int, unsigned int, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, void*, void**, void**)>(
        library, "cuLaunchKernel");

    int device = 0;
    void* context = nullptr;
    void* module = nullptr;
    void* function = nullptr;
    CHECK(init(0));
    CHECK(device_get(&device, 0));
    CHECK(context_create(&context, 0, device));
    CHECK(module_load(&module, image.data()));
    CHECK(get_function(&function, module, kernel.c_str()));
    for (Allocation& allocation : allocations)
        CHECK(mem_alloc(&allocation.device, allocation.bytes));
    for (const Pointer& pointer : pointers)
    {
        const uint64_t value = allocations.at(pointer.allocation).device + pointer.delta;
        std::memcpy(arguments.data() + pointer.offset, &value, sizeof(value));
    }
    size_t argument_bytes = arguments.size();
    void* extra[] = {reinterpret_cast<void*>(1), arguments.data(), reinterpret_cast<void*>(2), &argument_bytes,
                     nullptr};
    auto restore = [&]() -> CUresult {
        for (Allocation& allocation : allocations)
            if (CUresult result = copy_h2d(allocation.device, allocation.data.data(), allocation.bytes))
                return result;
        return 0;
    };
    auto run = [&]() {
        return launch_kernel(function, launch[0], launch[1], launch[2], launch[3], launch[4], launch[5],
                             launch[6], nullptr, nullptr, extra);
    };

    CHECK(restore());
    CHECK(run());
    CHECK(synchronize());
    for (size_t index = 0; index < allocations.size(); ++index)
    {
        std::vector<unsigned char> result(allocations[index].bytes);
        CHECK(copy_d2h(result.data(), allocations[index].device, result.size()));
        uint64_t hash = 1469598103934665603ull;
        size_t changed = 0;
        for (size_t byte = 0; byte < result.size(); ++byte)
        {
            hash = (hash ^ result[byte]) * 1099511628211ull;
            changed += result[byte] != allocations[index].data[byte];
        }
        std::printf("alloc %zu bytes=%zu changed=%zu hash=%016llx\n", index, result.size(), changed,
                    static_cast<unsigned long long>(hash));
        if (output_prefix != nullptr)
        {
            std::ofstream output(std::string(output_prefix) + "-alloc-" + std::to_string(index) + ".bin",
                                 std::ios::binary);
            output.write(reinterpret_cast<const char*>(result.data()), static_cast<std::streamsize>(result.size()));
        }
    }

    void* start = nullptr;
    void* end = nullptr;
    CHECK(event_create(&start, 0));
    CHECK(event_create(&end, 0));
    std::vector<float> times;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        CHECK(restore());
        CHECK(synchronize());
        CHECK(event_record(start, nullptr));
        CHECK(run());
        CHECK(event_record(end, nullptr));
        CHECK(synchronize());
        float milliseconds = 0;
        CHECK(event_elapsed(&milliseconds, start, end));
        times.push_back(milliseconds);
    }
    std::sort(times.begin(), times.end());
    if (!times.empty())
        std::printf("kernel %s grid=%u,%u,%u block=%u,%u,%u iterations=%d min=%.4f median=%.4f max=%.4f ms\n",
                    kernel.c_str(), launch[0], launch[1], launch[2], launch[3], launch[4], launch[5], iterations,
                    times.front(), times[times.size() / 2], times.back());
    return 0;
}
