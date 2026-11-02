// Independent, synchronous CUDA reference. No D3D12 queue, staging ring or
// presentation latency: upload this frame's bytes, evaluate NGX, synchronise,
// and save this frame's result. Usage: rr-zluda APP_DIR CASE_DIR OUT_DIR 4|5|6.
#include "rr_cuda_abi.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <windows.h>
extern "C"
{
    void d4r_ngx_set_ull(void *, const char *, unsigned long long);
    void d4r_ngx_set_float(void *, const char *, float);
    void d4r_ngx_set_uint(void *, const char *, unsigned);
    void d4r_ngx_set_int(void *, const char *, int);
    void d4r_ngx_set_void(void *, const char *, void *);
    void *d4r_host_alloc(unsigned long long size)
    {
        return std::calloc(1, size);
    }
    void d4r_host_free(void *p)
    {
        std::free(p);
    }
}
template <class F> F symbol(HMODULE m, const char *name)
{
    auto p = GetProcAddress(m, name);
    if (!p)
        throw std::runtime_error(std::string("missing ") + name);
    return reinterpret_cast<F>(p);
}
void cuda(int r, const char *name)
{
    if (r)
        throw std::runtime_error(std::string(name) + ": CUDA " + std::to_string(r));
}
void ngx(unsigned r, const char *name)
{
    if (r != 1)
    {
        char s[64];
        std::snprintf(s, sizeof(s), "%s: NGX 0x%08x", name, r);
        throw std::runtime_error(s);
    }
}
struct Paths
{
    const wchar_t *const *paths;
    unsigned count;
};
struct Logging
{
    void *callback;
    int level;
    bool disabled;
};
struct Info
{
    Paths paths;
    void *internal;
    Logging log;
};
struct Handle
{
    unsigned id;
};
struct Driver
{
    int(WINAPI *arrayCreate)(CudaArray *, const CudaArrayDescriptor *);
    int(WINAPI *array3DCreate)(CudaArray *, const CudaArray3DDescriptor *);
    int(WINAPI *copy)(const CudaMemcpy2D *);
    int(WINAPI *texCreate)(CudaObject *, const CudaResourceDesc *, const CudaTextureDesc *, const void *);
    int(WINAPI *surfCreate)(CudaObject *, const CudaResourceDesc *);
    int(WINAPI *sync)();
    int(WINAPI *alloc)(CudaDevicePtr *, size_t);
    explicit Driver(HMODULE m)
    {
        arrayCreate = symbol<decltype(arrayCreate)>(m, "cuArrayCreate");
        array3DCreate = symbol<decltype(array3DCreate)>(m, "cuArray3DCreate");
        copy = symbol<decltype(copy)>(m, "cuMemcpy2D");
        texCreate = symbol<decltype(texCreate)>(m, "cuTexObjectCreate");
        surfCreate = symbol<decltype(surfCreate)>(m, "cuSurfObjectCreate");
        sync = symbol<decltype(sync)>(m, "cuCtxSynchronize");
        alloc = symbol<decltype(alloc)>(m, "cuMemAlloc");
    }
};
std::vector<char> read(const std::filesystem::path &p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot read " + p.string());
    return {std::istreambuf_iterator<char>(f), {}};
}
struct Image
{
    CudaArray array = nullptr;
    CudaObject object = 0;
    unsigned w = 0, h = 0, bytes = 0;
    void create(Driver &d, unsigned width, unsigned height, unsigned format, unsigned channels, bool surface = false,
                bool linear = false)
    {
        w = width;
        h = height;
        bytes = channels * (format == 16 ? 2 : format == 32 ? 4 : 1);
        if (surface)
        {
            CudaArray3DDescriptor a{w, h, 0, format, channels, 2};
            cuda(d.array3DCreate(&array, &a), "array3DCreate");
        }
        else
        {
            CudaArrayDescriptor a{w, h, format, channels};
            cuda(d.arrayCreate(&array, &a), "arrayCreate");
        }
        CudaResourceDesc r{};
        r.res.array.hArray = array;
        if (surface)
            cuda(d.surfCreate(&object, &r), "surfCreate");
        else
        {
            CudaTextureDesc t{};
            t.addressMode[0] = t.addressMode[1] = 1;
            t.filterMode = linear ? 1 : 0;
            t.flags = 2;
            cuda(d.texCreate(&object, &r, &t, nullptr), "texCreate");
        }
    }
    void upload(Driver &d, const std::filesystem::path &path)
    {
        auto data = read(path);
        if (data.size() != size_t(w) * h * bytes)
            throw std::runtime_error("bad input size " + path.string());
        CudaMemcpy2D c{};
        c.srcMemoryType = 1;
        c.srcHost = data.data();
        c.srcPitch = w * bytes;
        c.dstMemoryType = 3;
        c.dstArray = array;
        c.WidthInBytes = w * bytes;
        c.Height = h;
        cuda(d.copy(&c), "upload");
    }
    void save(Driver &d, const std::filesystem::path &path)
    {
        std::vector<char> data(size_t(w) * h * bytes);
        CudaMemcpy2D c{};
        c.srcMemoryType = 3;
        c.srcArray = array;
        c.dstMemoryType = 1;
        c.dstHost = data.data();
        c.dstPitch = w * bytes;
        c.WidthInBytes = w * bytes;
        c.Height = h;
        cuda(d.copy(&c), "download");
        std::ofstream(path, std::ios::binary).write(data.data(), data.size());
    }
};
int main(int argc, char **argv)
try
{
    std::cout.setf(std::ios::unitbuf);
    if (argc != 5)
    {
        std::cerr << "usage: rr-zluda APP CASE OUT 4|5|6\n";
        return 2;
    }
    auto app = std::filesystem::absolute(argv[1]);
    auto input = std::filesystem::absolute(argv[2]);
    auto out = std::filesystem::absolute(argv[3]);
    std::filesystem::create_directories(out);
    int preset = std::atoi(argv[4]);
    if (preset < 4 || preset > 6)
        throw std::runtime_error("requires D, E or F");
    const char *nvapi = std::getenv("D4R_NVAPI");
    if (!nvapi || !LoadLibraryA(nvapi))
        throw std::runtime_error("cannot preload D4R_NVAPI");
    HMODULE bridge = LoadLibraryExW((app / L"nvcuda.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!bridge)
        throw std::runtime_error("cannot load CUDA bridge");
    if (!LoadLibraryExW((app / L"nvngx_dlssd.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH))
        throw std::runtime_error("cannot load DLSSD");
    HMODULE core = LoadLibraryExW((app / L"_nvngx.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!core)
        throw std::runtime_error("cannot load NGX core");
    auto init =
        symbol<unsigned (*)(unsigned long long, const wchar_t *, const Info *, unsigned)>(core, "NVSDK_NGX_CUDA_Init");
    auto initProject = symbol<unsigned (*)(const char *, int, const char *, const wchar_t *, unsigned, const Info *)>(
        core, "NVSDK_NGX_CUDA_Init_ProjectID");
    auto allocate = symbol<unsigned (*)(void **)>(core, "NVSDK_NGX_CUDA_AllocateParameters");
    auto create = symbol<unsigned (*)(unsigned, void *, Handle **)>(core, "NVSDK_NGX_CUDA_CreateFeature");
    auto evaluate = symbol<unsigned (*)(const Handle *, void *, void *)>(core, "NVSDK_NGX_CUDA_EvaluateFeature");
    auto release = symbol<unsigned (*)(Handle *)>(core, "NVSDK_NGX_CUDA_ReleaseFeature");
    auto shutdown = symbol<unsigned (*)()>(core, "NVSDK_NGX_CUDA_Shutdown");
    auto appText = app.wstring();
    const wchar_t *paths[] = {appText.c_str()};
    Info info{{paths, 1}, nullptr, {nullptr, 1, false}};
    auto status = init(241534722, out.wstring().c_str(), &info, 0x15);
    if (status != 1)
        status = initProject("24480451-f00d-face-1304-0308dabad187", 0, "1.0", out.wstring().c_str(), 0x15, &info);
    ngx(status, "Init");
    Driver d(bridge);
    std::ifstream frames(input / "frames.txt", std::ios::binary);
    std::string line, key;
    unsigned rw, rh, ow, oh;
    std::getline(frames, line);
    std::istringstream size(line);
    size >> key >> rw >> rh >> ow >> oh;
    if (key != "size" || !size)
        throw std::runtime_error("invalid case");
    void *params = nullptr;
    ngx(allocate(&params), "AllocateParameters");
    auto ui = [&](const char *name, unsigned v) { d4r_ngx_set_uint(params, name, v); };
    auto si = [&](const char *name, int v) { d4r_ngx_set_int(params, name, v); };
    auto sf = [&](const char *name, float v) { d4r_ngx_set_float(params, name, v); };
    auto ptr = [&](const char *name, void *v) { d4r_ngx_set_void(params, name, v); };
    std::streampos start = frames.tellg();
    std::getline(frames, line);
    std::istringstream first(line);
    int fid, reset, flags;
    float jx, jy, mx, my, pre;
    first >> key >> fid >> jx >> jy >> mx >> my >> pre >> reset >> flags;
    frames.clear();
    frames.seekg(start);
    ui("Width", rw);
    ui("Height", rh);
    ui("OutWidth", ow);
    ui("OutHeight", oh);
    si("PerfQualityValue", 2);
    si("DLSS.Feature.Create.Flags", 3 | ((flags & 1) ? 8 : 0));
    si("DLSS.Enable.Output.Subrects", 0);
    ui("CreationNodeMask", 1);
    ui("VisibilityNodeMask", 1);
    for (const char *quality : {"DLAA", "Quality", "Balanced", "Performance", "UltraPerformance", "UltraQuality"})
    {
        std::string name = std::string("RayReconstruction.Hint.Render.Preset.") + quality;
        ui(name.c_str(), preset);
    }
    si("DLSS.Denoise.Mode", 1);
    CudaDevicePtr scratch = 0;
    cuda(d.alloc(&scratch, 64ull * 1024 * 1024), "scratch");
    ptr("Scratch", reinterpret_cast<void *>(scratch));
    d4r_ngx_set_ull(params, "Scratch.SizeInBytes", 64ull * 1024 * 1024);
    Handle *handle = nullptr;
    ngx(create(13, params, &handle), "CreateFeature");
    std::cout << "RR preset " << preset << " created\n";
    const char *names[] = {"color",    "motion",      "depth",        "normals", "roughness", "diffuse",
                           "specular", "hitdistance", "disocclusion", "sss",     "particles"};
    const char *ngxNames[] = {"Color",
                              "MotionVectors",
                              "Depth",
                              "GBuffer.Normals",
                              "GBuffer.Roughness",
                              "DLSS.Input.DiffuseAlbedo",
                              "DLSS.Input.SpecularAlbedo",
                              "DLSSD.SpecularHitDistance",
                              "DLSS.DisocclusionMask",
                              "DLSSD.ScreenSpaceSubsurfaceScatteringGuide",
                              "DLSSD.ColorBeforeParticles"};
    const unsigned formats[] = {16, 16, 32, 16, 16, 1, 1, 32, 16, 16, 16},
                   channels[] = {4, 2, 1, 4, 4, 4, 4, 1, 1, 1, 4};
    std::array<Image, 11> images;
    for (size_t i = 0; i < images.size(); ++i)
        images[i].create(d, rw, rh, formats[i], channels[i], false, i == 0);
    Image output;
    output.create(d, ow, oh, 16, 4, true);
    ptr("Output", &output.object);
    ptr("ExposureTexture", nullptr);
    si("DLSS.Indicator.Invert.X.Axis", 0);
    si("DLSS.Indicator.Invert.Y.Axis", 0);
    unsigned count = 0;
    while (std::getline(frames, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream f(line);
        std::array<float, 16> view, clip;
        f >> key >> fid >> jx >> jy >> mx >> my >> pre >> reset >> flags;
        for (auto &x : view)
            f >> x;
        for (auto &x : clip)
            f >> x;
        if (key != "frame" || !f)
            throw std::runtime_error("invalid frame: " + line);
        for (size_t i = 0; i < images.size(); ++i)
        {
            auto path = input / (std::to_string(fid) + "-" + names[i] + ".bin");
            if ((i >= 7 || (i == 4 && preset == 6 && (flags & 4))) && !std::filesystem::exists(path))
            {
                ptr(ngxNames[i], nullptr);
                continue;
            }
            images[i].upload(d, path);
            ptr(ngxNames[i], &images[i].object);
        }
        sf("Jitter.Offset.X", jx);
        sf("Jitter.Offset.Y", jy);
        sf("MV.Scale.X", mx);
        sf("MV.Scale.Y", my);
        sf("DLSS.Pre.Exposure", pre);
        sf("DLSS.Exposure.Scale", 1);
        sf("FrameTimeDeltaInMsec", 16.666667f);
        sf("Sharpness", 0);
        si("Reset", reset);
        ui("DLSS.Render.Subrect.Dimensions.Width", rw);
        ui("DLSS.Render.Subrect.Dimensions.Height", rh);
        si("DLSS.Roughness.Mode", (flags & 4) ? 1 : 0);
        si("DLSS.Use.HW.Depth", (flags & 2) ? 1 : 0);
        ptr("WorldToViewMatrix", view.data());
        ptr("ViewToClipMatrix", clip.data());
        cuda(d.sync(), "before Evaluate");
        ngx(evaluate(handle, params, nullptr), "EvaluateFeature");
        cuda(d.sync(), "after Evaluate");
        output.save(d, out / (std::to_string(fid) + "-out.bin"));
        std::cout << "frame " << fid << " saved\n";
        ++count;
    }
    ngx(release(handle), "Release");
    ngx(shutdown(), "Shutdown");
    std::cout << count << " synchronous reference frames\n";
    return 0;
}
catch (const std::exception &e)
{
    std::cerr << "rr-zluda: " << e.what() << "\n";
    return 1;
}
