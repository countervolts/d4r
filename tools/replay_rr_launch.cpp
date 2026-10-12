// Replays one captured Ray Reconstruction (DLSS-D) kernel launch through ZLUDA on Linux, with its textures and
// surfaces: the reference for engine/rr's stage shaders on the GPU the capture came from, and (with debug stores
// added to the PTX) for their intermediates.
//
// usage: replay_rr_launch MODULE.ptx KERNEL MANIFEST
// MANIFEST lines (engine/rr/zluda_replay.py writes them from a capture):
//   args FILE                         the launch's argument block
//   grid GX GY GZ / block BX BY BZ
//   buf OFFSET FILE DELTA [OUTFILE]   upload FILE, store its device address + DELTA at OFFSET, download to OUTFILE
//   tex OFFSET FILE W H CHANNELS FORMAT linear|nearest    CUarray_format FORMAT, clamp, normalized coordinates
//   surf OFFSET W H CHANNELS FORMAT OUTFILE              zero-initialised, downloaded after the launch
#include <dlfcn.h>

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
using CUarray = void*;

struct ArrayDescriptor3D
{
    size_t Width, Height, Depth;
    int Format;
    unsigned NumChannels, Flags;
};
struct Memcpy2D
{
    size_t srcXInBytes, srcY;
    int srcMemoryType;
    const void* srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY;
    int dstMemoryType;
    void* dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
};
struct ResourceDesc
{
    int resType;
    union
    {
        CUarray array;
        int reserved[32];
    } res;
    unsigned flags;
};
struct TextureDesc
{
    int addressMode[3];
    int filterMode;
    unsigned flags, maxAnisotropy;
    int mipmapFilterMode;
    float mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp;
    float borderColor[4];
    int reserved[12];
};

static std::vector<char> slurp(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(1);
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

static void* g_cuda;
template <typename F> static F fn(const char* name)
{
    void* raw = dlsym(g_cuda, name);
    if (raw == nullptr)
    {
        std::fprintf(stderr, "libcuda.so has no %s\n", name);
        std::exit(1);
    }
    F f;
    std::memcpy(&f, &raw, sizeof(f));
    return f;
}
#define CK(expr)                                                                                    \
    do                                                                                              \
    {                                                                                               \
        CUresult r_ = (expr);                                                                       \
        if (r_ != 0)                                                                                \
        {                                                                                           \
            std::fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, r_);               \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

static size_t texelBytes(int format, unsigned channels)
{
    const size_t component = format == 0x10 ? 2 : format == 0x20 ? 4 : 1;
    return format >= 0xb0 ? 4 : component * channels; // the packed int8x4 formats are 4 bytes per texel
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::fprintf(stderr, "usage: %s MODULE.ptx KERNEL MANIFEST\n", argv[0]);
        return 2;
    }
    g_cuda = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (g_cuda == nullptr)
    {
        std::fprintf(stderr, "dlopen(libcuda.so): %s\n", dlerror());
        return 1;
    }
    auto init = fn<CUresult (*)(unsigned)>("cuInit");
    auto ctxCreate = fn<CUresult (*)(void**, unsigned, int)>("cuCtxCreate_v2");
    auto moduleLoad = fn<CUresult (*)(void**, const void*)>("cuModuleLoadData");
    auto getFunction = fn<CUresult (*)(void**, void*, const char*)>("cuModuleGetFunction");
    auto memAlloc = fn<CUresult (*)(CUdeviceptr*, size_t)>("cuMemAlloc_v2");
    auto htod = fn<CUresult (*)(CUdeviceptr, const void*, size_t)>("cuMemcpyHtoD_v2");
    auto dtoh = fn<CUresult (*)(void*, CUdeviceptr, size_t)>("cuMemcpyDtoH_v2");
    auto array3d = fn<CUresult (*)(CUarray*, const ArrayDescriptor3D*)>("cuArray3DCreate_v2");
    auto copy2d = fn<CUresult (*)(const Memcpy2D*)>("cuMemcpy2D_v2");
    auto texCreate = fn<CUresult (*)(uint64_t*, const ResourceDesc*, const TextureDesc*, const void*)>("cuTexObjectCreate");
    auto surfCreate = fn<CUresult (*)(uint64_t*, const ResourceDesc*)>("cuSurfObjectCreate");
    auto launch = fn<CUresult (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*,
                                  void**, void**)>("cuLaunchKernel");
    auto sync = fn<CUresult (*)()>("cuCtxSynchronize");

    void* context = nullptr;
    void* module = nullptr;
    void* function = nullptr;
    CK(init(0));
    CK(ctxCreate(&context, 0, 0));
    auto ptx = slurp(argv[1]);
    ptx.push_back('\0');
    CK(moduleLoad(&module, ptx.data()));
    CK(getFunction(&function, module, argv[2]));

    std::vector<char> args;
    unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1};
    struct Download
    {
        std::string path;
        CUdeviceptr device = 0;
        CUarray array = nullptr;
        size_t width = 0, height = 0, texel = 0;
    };
    std::vector<Download> downloads;
    std::vector<std::pair<std::string, CUdeviceptr>> uploads;
    std::ifstream manifest(argv[3]);
    std::string line;
    while (std::getline(manifest, line))
    {
        std::istringstream in(line);
        std::string key;
        in >> key;
        if (key == "args")
        {
            std::string file;
            in >> file;
            args = slurp(file);
        }
        else if (key == "grid")
            in >> grid[0] >> grid[1] >> grid[2];
        else if (key == "block")
            in >> block[0] >> block[1] >> block[2];
        else if (key == "buf")
        {
            size_t offset;
            std::string file, out;
            int64_t delta;
            in >> offset >> file >> delta >> out;
            // pointers into one allocation name the same FILE: it is uploaded once
            CUdeviceptr device = 0;
            for (const auto& [path, address] : uploads)
                if (path == file)
                    device = address;
            if (device == 0)
            {
                auto data = slurp(file);
                CK(memAlloc(&device, data.size()));
                CK(htod(device, data.data(), data.size()));
                uploads.emplace_back(file, device);
                if (!out.empty())
                    downloads.push_back({out, device, nullptr, data.size(), 1, 1});
            }
            const uint64_t value = device + delta;
            std::memcpy(args.data() + offset, &value, 8);
        }
        else if (key == "tex" || key == "surf")
        {
            size_t offset, width, height;
            unsigned channels;
            int format;
            std::string file, mode, out;
            if (key == "tex")
                in >> offset >> file >> width >> height >> channels >> format >> mode;
            else
                in >> offset >> width >> height >> channels >> format >> out;
            ArrayDescriptor3D desc{width, height, 0, format, channels, key == "surf" ? 2u : 0u};
            CUarray array = nullptr;
            CK(array3d(&array, &desc));
            const size_t texel = texelBytes(format, channels);
            std::vector<char> data = key == "tex" ? slurp(file) : std::vector<char>(width * height * texel, 0);
            Memcpy2D copy{};
            copy.srcMemoryType = 1;
            copy.srcHost = data.data();
            copy.srcPitch = width * texel;
            copy.dstMemoryType = 3;
            copy.dstArray = array;
            copy.WidthInBytes = width * texel;
            copy.Height = height;
            CK(copy2d(&copy));
            ResourceDesc resource{};
            resource.resType = 0;
            resource.res.array = array;
            uint64_t handle = 0;
            if (key == "tex")
            {
                TextureDesc texture{};
                texture.addressMode[0] = texture.addressMode[1] = texture.addressMode[2] = 1; // clamp
                texture.filterMode = mode == "linear" ? 1 : 0;
                texture.flags = 0x2; // normalized coordinates
                CK(texCreate(&handle, &resource, &texture, nullptr));
            }
            else
            {
                CK(surfCreate(&handle, &resource));
                downloads.push_back({out, 0, array, width, height, texel});
            }
            std::memcpy(args.data() + offset, &handle, 8);
        }
    }
    size_t argumentBytes = args.size();
    void* extra[] = {reinterpret_cast<void*>(1), args.data(), reinterpret_cast<void*>(2), &argumentBytes, nullptr};
    CK(launch(function, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, nullptr, nullptr, extra));
    CK(sync());
    for (const Download& d : downloads)
    {
        std::vector<char> data(d.width * d.height * d.texel);
        if (d.array == nullptr)
            CK(dtoh(data.data(), d.device, data.size()));
        else
        {
            Memcpy2D copy{};
            copy.srcMemoryType = 3;
            copy.srcArray = d.array;
            copy.dstMemoryType = 1;
            copy.dstHost = data.data();
            copy.dstPitch = d.width * d.texel;
            copy.WidthInBytes = d.width * d.texel;
            copy.Height = d.height;
            CK(copy2d(&copy));
        }
        std::ofstream(d.path, std::ios::binary).write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    std::printf("%s grid %u,%u,%u block %u,%u,%u: done\n", argv[2], grid[0], grid[1], grid[2], block[0], block[1],
                block[2]);
    return 0;
}
