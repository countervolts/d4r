// Replays one captured DLSS kernel launch outside Wine with buffer-only
// arguments, using resource dumps from the Wine bridge's
// D4R_CUDA_LAUNCH_DUMP_DIR instrumentation.
//
// usage: replay_dlss_layer MODULE KERNEL GX,GY,GZ BX,BY,BZ SHARED OUTDIR WORD...
//   MODULE  PTX text or CUDA fatbin
//   WORD    one 8-byte packed-argument word: either a hex literal (0x...) or
//           @FILE#0xOFFSET, which uploads FILE into a fresh device buffer (once
//           per distinct FILE) and passes that buffer's address plus OFFSET.
// After the launch every uploaded buffer is written to OUTDIR/<basename>.out.
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = uint64_t;

template <typename Function> static Function load_function(void* library, const char* name)
{
    void* raw = dlsym(library, name);
    Function function{};
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
}

static bool read_file(const std::string& path, std::vector<char>& data)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return false;
    data.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    return true;
}

static bool parse_triple(const char* text, unsigned int values[3])
{
    return std::sscanf(text, "%u,%u,%u", &values[0], &values[1], &values[2]) == 3;
}

struct Buffer
{
    std::string path;
    CUdeviceptr device = 0;
    size_t bytes = 0;
};

int main(int argc, char** argv)
{
    if (argc < 8)
    {
        std::fprintf(stderr, "usage: %s MODULE KERNEL GX,GY,GZ BX,BY,BZ SHARED OUTDIR WORD...\n", argv[0]);
        return 2;
    }
    unsigned int grid[3], block[3];
    if (!parse_triple(argv[3], grid) || !parse_triple(argv[4], block))
    {
        std::fprintf(stderr, "grid and block must be X,Y,Z\n");
        return 2;
    }
    const unsigned int shared = static_cast<unsigned int>(std::strtoul(argv[5], nullptr, 0));
    const std::string outputDirectory = argv[6];

    std::vector<char> module;
    if (!read_file(argv[1], module))
    {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    module.push_back('\0'); // PTX text must be NUL terminated; harmless for fatbins

    void* library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
    {
        std::fprintf(stderr, "dlopen(libcuda.so) failed: %s\n", dlerror());
        return 1;
    }
    const auto init = load_function<CUresult (*)(unsigned int)>(library, "cuInit");
    const auto deviceGet = load_function<CUresult (*)(CUdevice*, int)>(library, "cuDeviceGet");
    const auto contextCreate = load_function<CUresult (*)(CUcontext*, unsigned int, CUdevice)>(library, "cuCtxCreate_v2");
    const auto moduleLoad = load_function<CUresult (*)(CUmodule*, const void*)>(library, "cuModuleLoadData");
    const auto moduleGetFunction = load_function<CUresult (*)(CUfunction*, CUmodule, const char*)>(library, "cuModuleGetFunction");
    const auto alloc = load_function<CUresult (*)(CUdeviceptr*, size_t)>(library, "cuMemAlloc_v2");
    const auto copyHtoD = load_function<CUresult (*)(CUdeviceptr, const void*, size_t)>(library, "cuMemcpyHtoD_v2");
    const auto copyDtoH = load_function<CUresult (*)(void*, CUdeviceptr, size_t)>(library, "cuMemcpyDtoH_v2");
    const auto launch = load_function<CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                                                   unsigned int, unsigned int, unsigned int, CUstream, void**, void**)>(
        library, "cuLaunchKernel");
    const auto synchronize = load_function<CUresult (*)(void)>(library, "cuCtxSynchronize");
    if (!init || !deviceGet || !contextCreate || !moduleLoad || !moduleGetFunction || !alloc ||
        !copyHtoD || !copyDtoH || !launch || !synchronize)
    {
        std::fprintf(stderr, "ZLUDA libcuda.so is missing a required export\n");
        return 1;
    }

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule loaded = nullptr;
    CUfunction function = nullptr;
    CUresult result = init(0);
    if (result == 0)
        result = deviceGet(&device, 0);
    if (result == 0)
        result = contextCreate(&context, 0, device);
    if (result == 0)
        result = moduleLoad(&loaded, module.data());
    if (result == 0)
        result = moduleGetFunction(&function, loaded, argv[2]);
    if (result != 0)
    {
        std::fprintf(stderr, "module setup for %s failed: CUDA result=%d\n", argv[2], result);
        return 1;
    }

    std::vector<Buffer> buffers;
    std::vector<uint64_t> words;
    for (int index = 7; index < argc; ++index)
    {
        const std::string word = argv[index];
        if (word.empty() || word[0] != '@')
        {
            words.push_back(std::strtoull(word.c_str(), nullptr, 0));
            continue;
        }
        const size_t plus = word.rfind('#');
        const std::string path = word.substr(1, plus == std::string::npos ? std::string::npos : plus - 1);
        const uint64_t offset = plus == std::string::npos ? 0 : std::strtoull(word.c_str() + plus + 1, nullptr, 0);
        Buffer* buffer = nullptr;
        for (Buffer& existing : buffers)
            if (existing.path == path)
                buffer = &existing;
        if (buffer == nullptr)
        {
            std::vector<char> contents;
            if (!read_file(path, contents))
            {
                std::fprintf(stderr, "cannot read %s\n", path.c_str());
                return 1;
            }
            Buffer created;
            created.path = path;
            created.bytes = contents.size();
            result = alloc(&created.device, created.bytes);
            if (result == 0)
                result = copyHtoD(created.device, contents.data(), created.bytes);
            if (result != 0)
            {
                std::fprintf(stderr, "upload of %s failed: CUDA result=%d\n", path.c_str(), result);
                return 1;
            }
            buffers.push_back(created);
            buffer = &buffers.back();
        }
        words.push_back(buffer->device + offset);
    }

    size_t argumentBytes = words.size() * sizeof(uint64_t);
    void* extra[] = {reinterpret_cast<void*>(1), words.data(), reinterpret_cast<void*>(2), &argumentBytes,
                     nullptr};
    result = launch(function, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, nullptr,
                    nullptr, extra);
    if (result == 0)
        result = synchronize();
    std::printf("%s grid=%u,%u,%u block=%u,%u,%u: CUDA result=%d\n", argv[2], grid[0], grid[1], grid[2],
                block[0], block[1], block[2], result);
    if (result != 0)
        return 1;

    for (const Buffer& buffer : buffers)
    {
        std::vector<char> contents(buffer.bytes);
        result = copyDtoH(contents.data(), buffer.device, buffer.bytes);
        const size_t slash = buffer.path.rfind('/');
        const std::string name = outputDirectory + "/" +
            (slash == std::string::npos ? buffer.path : buffer.path.substr(slash + 1)) + ".out";
        std::ofstream output(name, std::ios::binary);
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        std::printf("wrote %s (%zu bytes, CUDA result=%d)\n", name.c_str(), buffer.bytes, result);
    }
    return 0;
}
