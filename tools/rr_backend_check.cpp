// Drives d4r_nvngx.dll the way a DLSS Ray Reconstruction game does, with the synthetic ray-traced
// scene of fsr_rr_scene.h, and measures the result against ground truth: the scene's noise-free image
// rendered directly at the output resolution. It exercises whichever Ray Reconstruction backend the
// shim is configured with ([RayReconstruction] Backend / D4R_RR_BACKEND), or Super Sampling alone on
// the same noisy colour (--feature 1) as the no-denoiser baseline.
//
// Per frame the game-side inputs are: the noisy colour (each signal remodulated by its albedo, summed),
// linear or hardware depth, render-resolution motion vectors in pixels (previous - current), world
// normals with the roughness in w (DLSS.Roughness.Mode 1), linear diffuse and specular albedos, the
// unjittered WorldToView/ViewToClip matrices, and Halton(2,3) jitter. The camera holds still, then
// pans, then cuts (Reset) to another view.
//
// It checks: capability and requirement answers, feature creation, every evaluation's result, output
// written at exactly the requested size, finite values, error against ground truth before and after
// the cut, release and shutdown. Timing: CPU time of each EvaluateFeature call and of submit + wait.
//
// usage: rr_backend_check.exe SHIM_DLL OUT_DIR [--render WxH] [--output WxH] [--frames N] [--feature 13|1]
//                             [--hw-depth] [--separate-roughness] [--specular-hit]
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fsr_rr_scene.h"

using namespace fsr_scene;

using NgxResult = unsigned int;
constexpr NgxResult NGX_SUCCESS = 1;
struct NgxHandle
{
    unsigned int Id;
};

extern "C"
{
    void d4r_ngx_set_float(void* parameters, const char* name, float value);
    void d4r_ngx_set_uint(void* parameters, const char* name, unsigned int value);
    void d4r_ngx_set_int(void* parameters, const char* name, int value);
    void d4r_ngx_set_d3d12_resource(void* parameters, const char* name, ID3D12Resource* value);
    void d4r_ngx_set_void(void* parameters, const char* name, void* value);
    NgxResult d4r_ngx_get_int(void* parameters, const char* name, int* value);
    NgxResult d4r_ngx_get_void(void* parameters, const char* name, void** value);
}

using PFN_Init_Ext = NgxResult (*)(unsigned long long, const wchar_t*, ID3D12Device*, unsigned int, const void*);
using PFN_Parameters = NgxResult (*)(void**);
using PFN_CreateFeature = NgxResult (*)(ID3D12GraphicsCommandList*, unsigned int, void*, NgxHandle**);
using PFN_Evaluate = NgxResult (*)(ID3D12GraphicsCommandList*, const NgxHandle*, void*, void*);
using PFN_Release = NgxResult (*)(NgxHandle*);
using PFN_Shutdown = NgxResult (*)();
struct NgxApplicationIdentifier
{
    unsigned int type;
    union
    {
        unsigned long long applicationId;
        struct
        {
            const char* id;
            int engineType;
            const char* engineVersion;
        } project;
    } value;
};
struct NgxFeatureDiscovery
{
    unsigned int sdkVersion;
    unsigned int feature;
    NgxApplicationIdentifier identifier;
    const wchar_t* dataPath;
    const void* featureInfo;
};
struct NgxFeatureRequirement
{
    unsigned int supported;
    unsigned int minArchitecture;
    char minOsVersion[255];
};
using PFN_Requirements = NgxResult (*)(void*, const NgxFeatureDiscovery*, NgxFeatureRequirement*);

namespace
{

ID3D12Device* g_device;
ID3D12CommandQueue* g_queue;
ID3D12CommandAllocator* g_allocator;
ID3D12GraphicsCommandList* g_list;
ID3D12Fence* g_fence;
HANDLE g_event;
UINT64 g_fenceValue;
FILE* g_json;
bool g_jsonFirst = true;

std::string format(const char* pattern, ...)
{
    char buffer[4096];
    va_list args;
    va_start(args, pattern);
    std::vsnprintf(buffer, sizeof(buffer), pattern, args);
    va_end(args);
    return buffer;
}

void json_field(const char* key, const std::string& value)
{
    if (g_json)
        std::fprintf(g_json, "%s\n  \"%s\": %s", g_jsonFirst ? "" : ",", key, value.c_str());
    g_jsonFirst = false;
}

bool g_failed = false;
void check(const char* name, bool ok, const std::string& detail)
{
    std::printf("check=%s result=%s %s\n", name, ok ? "ok" : "FAIL", detail.c_str());
    std::fflush(stdout);
    std::string escaped;
    for (char c : detail)
        if (c == '"' || c == '\\')
            escaped += '\\', escaped += c;
        else if (static_cast<unsigned char>(c) >= 0x20)
            escaped += c;
    json_field((std::string("check_") + name).c_str(),
               format("{\"ok\": %s, \"detail\": \"%s\"}", ok ? "true" : "false", escaped.c_str()));
    g_failed |= !ok;
}

// Process VRAM from the DRM fdinfo of this process (Wine maps Z: to /), as the D3D12 harness does.
unsigned long long process_vram_kib()
{
    std::vector<unsigned long long> clients;
    unsigned long long total = 0;
    for (int fd = 0; fd < 4096; ++fd)
    {
        char path[64];
        std::snprintf(path, sizeof(path), "Z:\\proc\\self\\fdinfo\\%d", fd);
        FILE* file = std::fopen(path, "r");
        if (file == nullptr)
            continue;
        unsigned long long client = 0, vram = 0;
        bool drm = false;
        char line[256];
        while (std::fgets(line, sizeof(line), file) != nullptr)
            if (std::sscanf(line, "drm-client-id: %llu", &client) == 1)
                drm = true;
            else
                std::sscanf(line, "drm-memory-vram: %llu", &vram);
        std::fclose(file);
        if (drm && std::find(clients.begin(), clients.end(), client) == clients.end())
        {
            clients.push_back(client);
            total += vram;
        }
    }
    return total;
}

ID3D12Resource* create_texture(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                               D3D12_RESOURCE_STATES state)
{
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ID3D12Resource* resource = nullptr;
    if (FAILED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                 __uuidof(ID3D12Resource), reinterpret_cast<void**>(&resource))))
        return nullptr;
    return resource;
}

ID3D12Resource* create_buffer(UINT64 bytes, D3D12_HEAP_TYPE type, uint8_t** mapped)
{
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = type;
    ID3D12Resource* buffer = nullptr;
    const D3D12_RESOURCE_STATES state =
        type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST;
    if (FAILED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                 __uuidof(ID3D12Resource), reinterpret_cast<void**>(&buffer))) ||
        FAILED(buffer->Map(0, nullptr, reinterpret_cast<void**>(mapped))))
        return nullptr;
    return buffer;
}

void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    if (before == after)
        return;
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    g_list->ResourceBarrier(1, &barrier);
}

bool submit_and_wait()
{
    if (FAILED(g_list->Close()))
        return false;
    ID3D12CommandList* lists[] = {g_list};
    g_queue->ExecuteCommandLists(1, lists);
    g_queue->Signal(g_fence, ++g_fenceValue);
    g_fence->SetEventOnCompletion(g_fenceValue, g_event);
    const bool done = WaitForSingleObject(g_event, 30000) == WAIT_OBJECT_0;
    g_allocator->Reset();
    g_list->Reset(g_allocator, nullptr);
    return done && SUCCEEDED(g_device->GetDeviceRemovedReason());
}

// Uploads tightly packed rows into a texture (left in `state`).
struct Uploader
{
    ID3D12Resource* staging = nullptr;
    uint8_t* mapped = nullptr;
    UINT64 capacity = 0, offset = 0;
    void upload(ID3D12Resource* texture, UINT width, UINT height, DXGI_FORMAT format, const void* data,
                size_t texelBytes, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
        UINT rows = 0;
        UINT64 rowBytes = 0, total = 0;
        g_device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
        offset = (offset + 511) & ~UINT64(511);
        footprint.Offset = offset;
        for (UINT row = 0; row < rows; ++row)
            std::memcpy(mapped + offset + row * footprint.Footprint.RowPitch,
                        static_cast<const uint8_t*>(data) + row * texelBytes * width, texelBytes * width);
        offset += total;
        transition(texture, state, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = texture;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = staging;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = footprint;
        g_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, state);
    }
};

float halton(uint32_t index, uint32_t base)
{
    float result = 0, fraction = 1;
    while (index > 0)
    {
        fraction /= base;
        result += fraction * (index % base);
        index /= base;
    }
    return result;
}

void write_ppm(const std::string& path, uint32_t width, uint32_t height, const std::vector<V3>& radiance)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
        return;
    std::fprintf(file, "P6\n%u %u\n255\n", width, height);
    std::vector<uint8_t> row(width * 3);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            const V3 c = radiance[static_cast<size_t>(y) * width + x];
            const float rgb[3] = {c.x, c.y, c.z};
            for (int k = 0; k < 3; ++k)
            {
                const float v = std::max(0.0f, rgb[k]);
                row[x * 3 + k] = static_cast<uint8_t>(std::lround(saturate(std::pow(v / (1 + v), 1 / 2.2f)) * 255));
            }
        }
        std::fwrite(row.data(), 1, row.size(), file);
    }
    std::fclose(file);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: rr_backend_check.exe SHIM_DLL OUT_DIR [--render WxH] [--output WxH] [--frames N] "
                    "[--feature 13|1] [--hw-depth] [--separate-roughness] [--specular-hit] [--separate-eval-params] [--water] [--measure-all]\n");
        return 2;
    }
    const std::string outDir = argv[2];
    UINT rw = 960, rh = 540, ow = 1920, oh = 1080, frames = 48, featureId = 13;
    bool hwDepth = false, separateRoughness = false, specularHit = false, separateEvalParams = false, measureAll = false;
    for (int i = 3; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--render" && i + 1 < argc)
            std::sscanf(argv[++i], "%ux%u", &rw, &rh);
        else if (arg == "--output" && i + 1 < argc)
            std::sscanf(argv[++i], "%ux%u", &ow, &oh);
        else if (arg == "--frames" && i + 1 < argc)
            frames = static_cast<UINT>(std::atoi(argv[++i]));
        else if (arg == "--feature" && i + 1 < argc)
            featureId = static_cast<UINT>(std::atoi(argv[++i]));
        else if (arg == "--hw-depth")
            hwDepth = true;
        else if (arg == "--separate-roughness")
            separateRoughness = true;
        else if (arg == "--specular-hit")
            specularHit = true;
        else if (arg == "--separate-eval-params")
            separateEvalParams = true;
        else if (arg == "--water")
            fsr_scene::g_water = true;
        else if (arg == "--measure-all")
            measureAll = true;
    }
    const bool rr = featureId == 13;
    CreateDirectoryA(outDir.c_str(), nullptr);
    g_json = std::fopen((outDir + "\\check.json").c_str(), "w");
    if (g_json)
        std::fputs("{", g_json);
    json_field("feature", format("%u", featureId));
    json_field("render_size", format("[%u, %u]", rw, rh));
    json_field("output_size", format("[%u, %u]", ow, oh));
    json_field("options", format("{\"hw_depth\": %s, \"separate_roughness\": %s, \"specular_hit\": %s, \"separate_eval_params\": %s}",
                                 hwDepth ? "true" : "false", separateRoughness ? "true" : "false",
                                 specularHit ? "true" : "false", separateEvalParams ? "true" : "false"));
    auto finish = [&](int code) {
        if (g_json)
        {
            std::fputs("\n}\n", g_json);
            std::fclose(g_json);
        }
        std::printf("overall=%s\n", code == 0 ? "PASS" : "FAIL");
        return code;
    };

    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&g_device))))
    {
        check("device", false, "D3D12CreateDevice failed");
        return finish(1);
    }
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    g_device->CreateCommandQueue(&queueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&g_queue));
    g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
                                     reinterpret_cast<void**>(&g_allocator));
    g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_allocator, nullptr,
                                __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&g_list));
    g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&g_fence));
    g_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const unsigned long long vramStart = process_vram_kib();

    HMODULE shim = LoadLibraryA(argv[1]);
    if (shim == nullptr)
    {
        check("load", false, format("LoadLibrary(%s) failed: %lu", argv[1], GetLastError()));
        return finish(1);
    }
    auto init = reinterpret_cast<PFN_Init_Ext>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_Init_Ext")));
    auto capability = reinterpret_cast<PFN_Parameters>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_GetCapabilityParameters")));
    auto allocate = reinterpret_cast<PFN_Parameters>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_AllocateParameters")));
    auto createFeature = reinterpret_cast<PFN_CreateFeature>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_CreateFeature")));
    auto evaluate = reinterpret_cast<PFN_Evaluate>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_EvaluateFeature")));
    auto release = reinterpret_cast<PFN_Release>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_ReleaseFeature")));
    auto shutdown = reinterpret_cast<PFN_Shutdown>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_Shutdown")));
    auto requirements = reinterpret_cast<PFN_Requirements>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_GetFeatureRequirements")));
    if (!init || !capability || !allocate || !createFeature || !evaluate || !release || !shutdown || !requirements)
    {
        check("load", false, "missing NGX exports");
        return finish(1);
    }

    // Discovery before Init, as games do.
    wchar_t dataPath[MAX_PATH];
    GetTempPathW(MAX_PATH, dataPath);
    NgxFeatureDiscovery discovery = {};
    discovery.sdkVersion = 0x15;
    discovery.feature = featureId;
    discovery.identifier.value.applicationId = 241534723ULL;
    discovery.dataPath = dataPath;
    NgxFeatureRequirement requirement = {};
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter* adapter = nullptr;
    LUID luid;
    g_device->GetAdapterLuid(&luid);
    CreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory));
    factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter), reinterpret_cast<void**>(&adapter));
    const NgxResult discovered = requirements(adapter, &discovery, &requirement);
    check("requirements_before_init", discovered == NGX_SUCCESS && requirement.supported == 0,
          format("result 0x%08x, FeatureSupported %u", discovered, requirement.supported));
    adapter->Release();
    factory->Release();

    const auto initStart = std::chrono::steady_clock::now();
    const NgxResult initialized = init(241534723ULL, dataPath, g_device, 0x15, nullptr);
    const double initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    check("init", initialized == NGX_SUCCESS, format("0x%08x in %.0f ms", initialized, initMs));
    if (initialized != NGX_SUCCESS)
        return finish(1);
    void* capabilities = nullptr;
    capability(&capabilities);
    int available = -1;
    void* optimal = nullptr;
    const char* key = rr ? "SuperSamplingDenoising.Available" : "SuperSampling.Available";
    d4r_ngx_get_int(capabilities, key, &available);
    d4r_ngx_get_void(capabilities, rr ? "DLSSDOptimalSettingsCallback" : "DLSSOptimalSettingsCallback", &optimal);
    check("capability", available == 1 && optimal != nullptr, format("%s=%d, optimal-settings callback %s", key, available, optimal ? "set" : "absent"));

    // Inputs at render size, output at output size.
    const auto srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ID3D12Resource* color = create_texture(rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE, srv);
    ID3D12Resource* depth = create_texture(rw, rh, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE, srv);
    ID3D12Resource* motion = create_texture(rw, rh, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE, srv);
    ID3D12Resource* normals = create_texture(rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE, srv);
    ID3D12Resource* roughness = create_texture(rw, rh, DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_NONE, srv);
    ID3D12Resource* diffuse = create_texture(rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, srv);
    ID3D12Resource* specular = create_texture(rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, srv);
    ID3D12Resource* hitDistance = create_texture(rw, rh, DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_NONE, srv);
    // The output starts out holding a sentinel, so an unwritten texel is detectable.
    ID3D12Resource* output = create_texture(ow, oh, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Uploader uploader;
    uploader.capacity = 64ull << 20;
    uploader.staging = create_buffer(uploader.capacity, D3D12_HEAP_TYPE_UPLOAD, &uploader.mapped);
    D3D12_RESOURCE_DESC outputDesc = {};
    outputDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    outputDesc.Width = ow;
    outputDesc.Height = oh;
    outputDesc.DepthOrArraySize = 1;
    outputDesc.MipLevels = 1;
    outputDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    outputDesc.SampleDesc.Count = 1;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT outputFootprint = {};
    UINT outputRows = 0;
    UINT64 outputRowBytes = 0, outputTotal = 0;
    g_device->GetCopyableFootprints(&outputDesc, 0, 1, 0, &outputFootprint, &outputRows, &outputRowBytes, &outputTotal);
    uint8_t* readMapped = nullptr;
    ID3D12Resource* readback = create_buffer(outputTotal, D3D12_HEAP_TYPE_READBACK, &readMapped);
    if (!color || !depth || !motion || !normals || !roughness || !diffuse || !specular || !hitDistance || !output ||
        !uploader.staging || !readback)
    {
        check("resources", false, "texture creation");
        return finish(1);
    }

    void* parameters = nullptr;
    allocate(&parameters);
    d4r_ngx_set_uint(parameters, "Width", rw);
    d4r_ngx_set_uint(parameters, "Height", rh);
    d4r_ngx_set_uint(parameters, "OutWidth", ow);
    d4r_ngx_set_uint(parameters, "OutHeight", oh);
    d4r_ngx_set_int(parameters, "PerfQualityValue", 2);
    d4r_ngx_set_int(parameters, "DLSS.Feature.Create.Flags", 1 | 2); // HDR | MVLowRes
    if (rr)
    {
        d4r_ngx_set_int(parameters, "DLSS.Denoise.Mode", 1);
        d4r_ngx_set_int(parameters, "DLSS.Roughness.Mode", separateRoughness ? 0 : 1);
        d4r_ngx_set_int(parameters, "DLSS.Use.HW.Depth", hwDepth ? 1 : 0);
    }
    NgxHandle* feature = nullptr;
    const auto createStart = std::chrono::steady_clock::now();
    const NgxResult created = createFeature(g_list, featureId, parameters, &feature);
    const double createMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - createStart).count();
    check("create", created == NGX_SUCCESS && feature != nullptr, format("0x%08x in %.0f ms", created, createMs));
    json_field("create_ms", format("%.1f", createMs));
    if (created != NGX_SUCCESS)
        return finish(1);

    // Keep the depth/roughness modes only on the creation object, as callers are allowed to do.
    // Frame resources and camera settings below go onto a fresh object without those scalars.
    if (separateEvalParams)
        allocate(&parameters);

    const M4 projection = projection_matrix(1.0f, static_cast<float>(rw) / rh, 0.1f, 1000.0f);
    const uint32_t moveStart = frames / 3, cutFrame = frames * 2 / 3;
    auto camera_at = [&](uint32_t f) -> Camera {
        if (f >= cutFrame)
            return {{3.5f, 1.6f, 1.5f}, {-0.5f, -0.6f, 7.5f}};
        const float t = f >= moveStart ? static_cast<float>(f - moveStart) : 0.0f;
        return {{-0.6f + 0.04f * t, 0.6f, -1.5f}, {0.0f + 0.02f * t, -0.1f, 6.5f}};
    };
    Frame scene, truth;
    Camera previous = camera_at(0);
    std::vector<double> callUs, totalUs;
    std::string log = "[";
    bool allEvaluated = true, sized = true, finite = true;
    double errorBeforeCut = -1, errorFinal = -1, errorAtCut = -1;
    std::vector<uint16_t> hit(static_cast<size_t>(rw) * rh), rough(static_cast<size_t>(rw) * rh);
    std::vector<float> view(16), clip(16);
    for (uint32_t f = 0; f < frames; ++f)
    {
        const Camera camera = camera_at(f);
        const bool reset = f == 0 || f == cutFrame;
        const uint32_t phase = f % 16 + 1;
        const float jx = halton(phase, 2) - 0.5f, jy = halton(phase, 3) - 0.5f;
        render(scene, rw, rh, camera, reset ? camera : previous, projection, f, jx, jy);
        uploader.offset = 0;
        uploader.upload(color, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, scene.color.data(), 8, srv);
        if (hwDepth)
            uploader.upload(depth, rw, rh, DXGI_FORMAT_R32_FLOAT, scene.hardwareDepth.data(), 4, srv);
        else
        {
            std::vector<float> linear(scene.depth);
            for (size_t i = 0; i < linear.size(); ++i)
                linear[i] = scene.valid[i] ? linear[i] : 1.0e7f; // the sky: beyond any geometry
            uploader.upload(depth, rw, rh, DXGI_FORMAT_R32_FLOAT, linear.data(), 4, srv);
        }
        uploader.upload(motion, rw, rh, DXGI_FORMAT_R16G16_FLOAT, scene.motionPixels.data(), 4, srv);
        if (rr)
        {
            uploader.upload(normals, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, scene.normalsRoughness.data(), 8, srv);
            uploader.upload(diffuse, rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM, scene.diffuseAlbedoLinear.data(), 4, srv);
            uploader.upload(specular, rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM, scene.specularAlbedoLinear.data(), 4, srv);
            if (separateRoughness)
            {
                for (size_t i = 0; i < rough.size(); ++i)
                    rough[i] = scene.normalsRoughness[i * 4 + 3];
                uploader.upload(roughness, rw, rh, DXGI_FORMAT_R16_FLOAT, rough.data(), 2, srv);
            }
            if (specularHit)
            {
                for (size_t i = 0; i < hit.size(); ++i)
                    hit[i] = scene.noisy[kIndirectSpecular][i * 4 + 3];
                uploader.upload(hitDistance, rw, rh, DXGI_FORMAT_R16_FLOAT, hit.data(), 2, srv);
            }
        }
        d4r_ngx_set_d3d12_resource(parameters, "Color", color);
        d4r_ngx_set_d3d12_resource(parameters, "Depth", depth);
        d4r_ngx_set_d3d12_resource(parameters, "MotionVectors", motion);
        d4r_ngx_set_d3d12_resource(parameters, "Output", output);
        d4r_ngx_set_float(parameters, "Jitter.Offset.X", -jx); // DLSS: minus the sample offset
        d4r_ngx_set_float(parameters, "Jitter.Offset.Y", -jy);
        d4r_ngx_set_float(parameters, "MV.Scale.X", 1.0f);
        d4r_ngx_set_float(parameters, "MV.Scale.Y", 1.0f);
        d4r_ngx_set_int(parameters, "Reset", reset ? 1 : 0);
        d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Width", rw);
        d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Height", rh);
        if (rr)
        {
            d4r_ngx_set_d3d12_resource(parameters, "GBuffer.Normals", normals);
            d4r_ngx_set_d3d12_resource(parameters, "GBuffer.Roughness", separateRoughness ? roughness : nullptr);
            d4r_ngx_set_d3d12_resource(parameters, "GBuffer.DiffuseAlbedo", diffuse);
            d4r_ngx_set_d3d12_resource(parameters, "GBuffer.SpecularAlbedo", specular);
            d4r_ngx_set_d3d12_resource(parameters, "DLSSD.SpecularHitDistance", specularHit ? hitDistance : nullptr);
            const M4 viewMatrix = view_matrix(camera.eye, camera.target);
            std::memcpy(view.data(), &viewMatrix, 64);
            std::memcpy(clip.data(), &projection, 64);
            d4r_ngx_set_void(parameters, "WorldToViewMatrix", view.data());
            d4r_ngx_set_void(parameters, "ViewToClipMatrix", clip.data());
        }
        // Mark the output with a sentinel (NaN) the backend has to overwrite everywhere.
        {
            std::vector<uint16_t> sentinel(static_cast<size_t>(ow) * oh * 4, 0x7e00u);
            uploader.upload(output, ow, oh, DXGI_FORMAT_R16G16B16A16_FLOAT, sentinel.data(), 8,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        const auto start = std::chrono::steady_clock::now();
        const NgxResult evaluated = evaluate(g_list, feature, parameters, nullptr);
        const auto recorded = std::chrono::steady_clock::now();
        transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = readback;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = outputFootprint;
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = output;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        g_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const bool submitted = submit_and_wait();
        const auto end = std::chrono::steady_clock::now();
        callUs.push_back(std::chrono::duration<double, std::micro>(recorded - start).count());
        totalUs.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        if (evaluated != NGX_SUCCESS || !submitted)
        {
            allEvaluated = false;
            std::printf("frame %u: EvaluateFeature 0x%08x, submit %s\n", f, evaluated, submitted ? "ok" : "FAILED");
            if (!submitted)
                break;
        }

        // Ground truth (the noise-free scene rendered at the output size from the same camera), on the frames
        // that are reported and every 8th.
        const bool measured = measureAll || f % 8 == 7 || f == 0 || f + 1 == moveStart || f + 1 == cutFrame || f == cutFrame ||
                              f + 1 == frames;
        if (!measured)
        {
            previous = camera;
            continue;
        }
        render(truth, ow, oh, camera, camera, projection_matrix(1.0f, static_cast<float>(ow) / oh, 0.1f, 1000.0f), f);
        std::vector<V3> result(static_cast<size_t>(ow) * oh);
        double error = 0, noisyError = 0, waterError = 0;
        size_t unwritten = 0, nonFinite = 0, n = 0, waterN = 0;
        for (UINT y = 0; y < oh; ++y)
            for (UINT x = 0; x < ow; ++x)
            {
                const uint16_t* texel = reinterpret_cast<const uint16_t*>(readMapped + outputFootprint.Offset +
                                                                          y * outputFootprint.Footprint.RowPitch) + x * 4;
                const size_t i = static_cast<size_t>(y) * ow + x;
                const V3 value{from_half(texel[0]), from_half(texel[1]), from_half(texel[2])};
                if (texel[0] == 0x7e00u && texel[1] == 0x7e00u && texel[2] == 0x7e00u)
                {
                    ++unwritten;
                    continue;
                }
                if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
                {
                    ++nonFinite;
                    continue;
                }
                result[i] = value;
                // Compared in a tonemapped space (x / (1 + x)), so highlights do not dominate.
                const V3 t = truth.colorClean[i];
                const float tv[3] = {t.x, t.y, t.z}, ov[3] = {value.x, value.y, value.z};
                for (int c = 0; c < 3; ++c)
                {
                    const double d = ov[c] / (1 + std::max(0.0f, ov[c])) - tv[c] / (1 + tv[c]);
                    error += d * d;
                    ++n;
                    if (from_half(truth.normalsRoughness[i * 4 + 3]) < 0.06f)
                    {
                        waterError += d * d;
                        ++waterN;
                    }
                }
            }
        // The noisy input itself, nearest-upsampled: the error with no denoising and no reconstruction.
        for (UINT y = 0; y < oh; ++y)
            for (UINT x = 0; x < ow; ++x)
            {
                const size_t i = static_cast<size_t>(y) * ow + x;
                const size_t s = static_cast<size_t>(y * rh / oh) * rw + x * rw / ow;
                const V3 t = truth.colorClean[i];
                const float tv[3] = {t.x, t.y, t.z};
                for (int c = 0; c < 3; ++c)
                {
                    const float v = std::max(0.0f, from_half(scene.color[s * 4 + c]));
                    const double d = v / (1 + v) - tv[c] / (1 + tv[c]);
                    noisyError += d * d;
                }
            }
        const double rmse = n ? std::sqrt(error / n) : -1, noisyRmse = std::sqrt(noisyError / (3.0 * ow * oh));
        sized = sized && unwritten == 0;
        finite = finite && nonFinite == 0;
        if (f + 1 == cutFrame)
            errorBeforeCut = rmse;
        if (f == cutFrame)
            errorAtCut = rmse;
        if (f + 1 == frames)
            errorFinal = rmse;
        const double psnr = rmse > 0 ? 20 * std::log10(1.0 / rmse) : 0;
        const double waterRmse = waterN ? std::sqrt(waterError / waterN) : -1;
        if (fsr_scene::g_water)
            std::printf("frame %u: rmse %.5f water %.5f\n", f, rmse, waterRmse);
        log += format("%s\n    {\"frame\": %u, \"reset\": %s, \"result\": %u, \"rmse\": %.5f, \"psnr_db\": %.2f, \"water_rmse\": %.5f, "
                      "\"noisy_input_rmse\": %.5f, \"unwritten\": %zu, \"nonfinite\": %zu, \"call_us\": %.0f, "
                      "\"submit_wait_us\": %.0f}",
                      f ? "," : "", f, reset ? "true" : "false", evaluated, rmse, psnr, waterRmse, noisyRmse, unwritten, nonFinite,
                      callUs.back(), totalUs.back());
        if (f == 0 || f + 1 == moveStart || f + 1 == cutFrame || f == cutFrame || f + 1 == frames)
        {
            write_ppm(format("%s\\frame%03u_output.ppm", outDir.c_str(), f), ow, oh, result);
            write_ppm(format("%s\\frame%03u_truth.ppm", outDir.c_str(), f), ow, oh, truth.colorClean);
            std::vector<V3> input(static_cast<size_t>(rw) * rh);
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = {from_half(scene.color[i * 4]), from_half(scene.color[i * 4 + 1]), from_half(scene.color[i * 4 + 2])};
            write_ppm(format("%s\\frame%03u_input.ppm", outDir.c_str(), f), rw, rh, input);
        }
        previous = camera;
    }
    json_field("frame_log", log + "\n  ]");
    const unsigned long long vramPeak = process_vram_kib();
    std::vector<double> warmCall(callUs.begin() + std::min<size_t>(callUs.size(), 4), callUs.end());
    std::vector<double> warmTotal(totalUs.begin() + std::min<size_t>(totalUs.size(), 4), totalUs.end());
    std::sort(warmCall.begin(), warmCall.end());
    std::sort(warmTotal.begin(), warmTotal.end());
    const double medianCall = warmCall.empty() ? -1 : warmCall[warmCall.size() / 2];
    const double medianTotal = warmTotal.empty() ? -1 : warmTotal[warmTotal.size() / 2];
    json_field("timing_us", format("{\"evaluate_call_median\": %.0f, \"submit_wait_median\": %.0f, \"first_frame_submit_wait\": %.0f}",
                                   medianCall, medianTotal, totalUs.empty() ? -1.0 : totalUs[0]));
    json_field("vram_mib", format("{\"before_init\": %.1f, \"after_frames\": %.1f}", vramStart / 1024.0, vramPeak / 1024.0));
    check("evaluations", allEvaluated, format("%u frames, EvaluateFeature call median %.0f us, call+submit+wait median %.0f us",
                                              frames, medianCall, medianTotal));
    check("output_size", sized, format("every texel of the %ux%u output written on every frame", ow, oh));
    check("output_finite", finite, "no NaN/Inf");
    json_field("rmse", format("{\"before_cut\": %.5f, \"at_cut\": %.5f, \"final\": %.5f}", errorBeforeCut, errorAtCut, errorFinal));
    std::printf("tonemapped RMSE vs ground truth: before cut %.5f, at cut %.5f, final %.5f\n", errorBeforeCut, errorAtCut,
                errorFinal);

    const NgxResult released = release(feature);
    // Let the shim's deferred cleanup see the list retire.
    submit_and_wait();
    const NgxResult shut = shutdown();
    check("release", released == NGX_SUCCESS && shut == NGX_SUCCESS, format("release 0x%08x, shutdown 0x%08x", released, shut));
    return finish(g_failed ? 1 : 0);
}
