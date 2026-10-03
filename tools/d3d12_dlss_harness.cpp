// Modified in this fork for CUDA Ray Reconstruction support and validation (2026).
// Drives d4r_nvngx.dll through the NGX D3D12 API the way a game (or
// OptiScaler's DLSS backend) does: a D3D12 device and command queue, input
// textures in NON_PIXEL_SHADER_RESOURCE, an output UAV, and one
// EvaluateFeature per recorded command list. The inputs reproduce the CUDA
// probe's synthetic frame (ngx_cuda_probe.cpp) so the D3D12 result can be
// compared with the direct CUDA result.
//
// usage: d3d12_dlss_harness.exe SHIM_DLL OUTPUT_RAW [FRAMES [IN_W IN_H OUT_W OUT_H]]
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include "d4r_vkd3d_interop.h"

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

static ID3D12Device* g_device;
static ID3D12CommandQueue* g_queue;
static ID3D12CommandAllocator* g_allocator;
static ID3D12GraphicsCommandList* g_list;
static ID3D12Fence* g_fence;
static HANDLE g_event;
static UINT64 g_fenceValue;

static unsigned long long process_vram_kib()
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

static bool check(HRESULT hr, const char* what)
{
    if (FAILED(hr))
        std::printf("%s failed: 0x%08lx\n", what, hr);
    return SUCCEEDED(hr);
}

static void submit_and_wait()
{
    g_list->Close();
    ID3D12CommandList* lists[] = {g_list};
    g_queue->ExecuteCommandLists(1, lists);
    g_queue->Signal(g_fence, ++g_fenceValue);
    if (g_fence->GetCompletedValue() < g_fenceValue)
    {
        g_fence->SetEventOnCompletion(g_fenceValue, g_event);
        WaitForSingleObject(g_event, INFINITE);
    }
    g_allocator->Reset();
    g_list->Reset(g_allocator, nullptr);
}

class LifetimeProbe final : public IUnknown
{
public:
    explicit LifetimeProbe(std::atomic<bool>& destroyed) : destroyed_(destroyed) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override
    {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown)) return E_NOINTERFACE;
        *out = this; AddRef(); return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        ULONG refs = --refs_;
        if (!refs) { destroyed_ = true; delete this; }
        return refs;
    }
private:
    std::atomic<ULONG> refs_{1};
    std::atomic<bool>& destroyed_;
};

static void lifecycle_report(const char* message)
{
    std::printf("%s\n", message);
    std::fflush(stdout);
    if (const char* path = std::getenv("D4R_HARNESS_TEST_REPORT"))
        if (FILE* file = std::fopen(path, "a"))
        {
            std::fprintf(file, "%s\n", message);
            std::fclose(file);
        }
}

static bool test_external_lifetime()
{
    ID3D12DXVKInteropDeviceD4R2* interop = nullptr;
    if (!check(g_device->QueryInterface(__uuidof(ID3D12DXVKInteropDeviceD4R2),
                                       reinterpret_cast<void**>(&interop)), "lifetime interface"))
        return false;
    for (bool submitted : {false, true})
    {
        std::atomic<bool> destroyed{false};
        auto* owner = new LifetimeProbe(destroyed);
        const HRESULT retained = interop->RetainExternalResources(g_list, owner);
        const HRESULT repeated = interop->RetainExternalResources(g_list, owner);
        owner->Release();
        if (!check(retained, "retain resources") || !check(repeated, "deduplicate resources") || destroyed)
            return false;
        g_list->Close();
        ID3D12Fence* gate = nullptr;
        if (submitted)
        {
            if (!check(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                                            reinterpret_cast<void**>(&gate)), "lifetime gate")) return false;
            g_queue->Wait(gate, 1);
            ID3D12CommandList* lists[] = {g_list};
            g_queue->ExecuteCommandLists(1, lists);
            Sleep(20);
            g_allocator->Reset(); // Pending submissions must keep their external owners.
            if (destroyed) { gate->Signal(1); return false; }
            gate->Signal(1);
            g_queue->Signal(g_fence, ++g_fenceValue);
            if (g_fence->GetCompletedValue() < g_fenceValue)
            {
                g_fence->SetEventOnCompletion(g_fenceValue, g_event);
                if (WaitForSingleObject(g_event, 5000) != WAIT_OBJECT_0) return false;
            }
            gate->Release();
        }
        // The fence callback may take a moment to drop the submission's allocator ref.
        const ULONGLONG start = GetTickCount64();
        do { g_allocator->Reset(); if (!destroyed) Sleep(1); }
        while (!destroyed && GetTickCount64() - start < 5000);
        if (!destroyed) return false;
        if (!check(g_list->Reset(g_allocator, nullptr), "lifetime list reset")) return false;
        lifecycle_report(submitted ? "LIFETIME submitted: retained until safe allocator reset"
                                   : "LIFETIME discarded: retained until safe allocator reset");
    }
    interop->Release();
    return true;
}

static ID3D12Resource* create_buffer(D3D12_HEAP_TYPE type, UINT64 size, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = type;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* buffer = nullptr;
    check(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                            __uuidof(ID3D12Resource), reinterpret_cast<void**>(&buffer)),
          "CreateCommittedResource(buffer)");
    return buffer;
}

static ID3D12Resource* create_texture(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    ID3D12Resource* texture = nullptr;
    check(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                            nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&texture)),
          "CreateCommittedResource(texture)");
    return texture;
}

static void barrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    g_list->ResourceBarrier(1, &b);
}

// Uploads tightly packed rows into a texture, leaving it in finalState.
static void upload(ID3D12Resource* texture, const void* data, UINT rowBytes, D3D12_RESOURCE_STATES finalState)
{
    D3D12_RESOURCE_DESC desc;
    texture->GetDesc(&desc);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
    UINT rows;
    UINT64 rowSize, total;
    g_device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, &rows, &rowSize, &total);
    ID3D12Resource* staging = create_buffer(D3D12_HEAP_TYPE_UPLOAD, total, D3D12_RESOURCE_STATE_GENERIC_READ);
    uint8_t* mapped = nullptr;
    D3D12_RANGE none = {0, 0};
    staging->Map(0, &none, reinterpret_cast<void**>(&mapped));
    for (UINT y = 0; y < rows; ++y)
        std::memcpy(mapped + layout.Offset + static_cast<size_t>(y) * layout.Footprint.RowPitch,
                    static_cast<const uint8_t*>(data) + static_cast<size_t>(y) * rowBytes, rowBytes);
    staging->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION destination = {};
    destination.pResource = texture;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION source = {};
    source.pResource = staging;
    source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    source.PlacedFootprint = layout;
    g_list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    barrier(texture, D3D12_RESOURCE_STATE_COPY_DEST, finalState);
    submit_and_wait();
    staging->Release();
}

static void update_texture(ID3D12Resource* texture, const void* data, UINT rowBytes,
                           D3D12_RESOURCE_STATES currentState)
{
    barrier(texture, currentState, D3D12_RESOURCE_STATE_COPY_DEST);
    submit_and_wait();
    upload(texture, data, rowBytes, currentState);
}

static std::vector<uint8_t> read_back(ID3D12Resource* texture, D3D12_RESOURCE_STATES state, UINT rowBytes)
{
    D3D12_RESOURCE_DESC desc;
    texture->GetDesc(&desc);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
    UINT rows;
    UINT64 rowSize, total;
    g_device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, &rows, &rowSize, &total);
    ID3D12Resource* staging = create_buffer(D3D12_HEAP_TYPE_READBACK, total, D3D12_RESOURCE_STATE_COPY_DEST);
    barrier(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION destination = {};
    destination.pResource = staging;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = layout;
    D3D12_TEXTURE_COPY_LOCATION source = {};
    source.pResource = texture;
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    g_list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    barrier(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    submit_and_wait();
    std::vector<uint8_t> result(static_cast<size_t>(rowBytes) * rows);
    uint8_t* mapped = nullptr;
    D3D12_RANGE range = {0, static_cast<SIZE_T>(total)};
    staging->Map(0, &range, reinterpret_cast<void**>(&mapped));
    for (UINT y = 0; y < rows; ++y)
        std::memcpy(result.data() + static_cast<size_t>(y) * rowBytes,
                    mapped + layout.Offset + static_cast<size_t>(y) * layout.Footprint.RowPitch, rowBytes);
    staging->Unmap(0, nullptr);
    staging->Release();
    return result;
}

static void trace_stage(const char* stage)
{
    const char* path = std::getenv("D4R_HARNESS_TRACE");
    if (path == nullptr || path[0] == '\0')
        return;
    if (FILE* file = std::fopen(path, "a"))
    {
        std::fprintf(file, "%s\n", stage);
        std::fclose(file);
    }
}

// A CPU-rasterized, subpixel-shifted version of the synthetic fence scene.
// Positive jitter moves the image right/down in screen space, so a render
// pixel samples the scene at (pixel - jitter). Four-by-four coverage samples
// make thin-line and checkerboard changes visible to temporal DLSS.
static void render_jittered_background(std::vector<uint16_t>& color, UINT width, UINT height, UINT stride,
                                       float jitterX, float jitterY)
{
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x)
        {
            float sum[3] = {};
            for (int sy = 0; sy < 4; ++sy)
                for (int sx = 0; sx < 4; ++sx)
                {
                    const float sampleX = float(x) + (float(sx) + 0.5f) * 0.25f - 0.5f - jitterX;
                    const float sampleY = float(y) + (float(sy) + 0.5f) * 0.25f - 0.5f - jitterY;
                    const int ix = std::clamp(int(std::floor(sampleX)), 0, int(width) - 1);
                    const int iy = std::clamp(int(std::floor(sampleY)), 0, int(height) - 1);
                    const bool fence = (ix % 24 == 0) || (iy % 24 == 0);
                    const bool checker = (((ix / 4) + (iy / 4)) & 1) != 0;
                    sum[0] += fence ? 1.0f : (checker ? 0.25f : 0.05f);
                    sum[1] += ((ix + iy) % 32 < 2) ? 0.8f : (checker ? 0.05f : 0.25f);
                    sum[2] += (((ix / 8) ^ (iy / 8)) & 1) ? 0.8f : 0.05f;
                }
            const size_t pixel = static_cast<size_t>(y) * stride + x;
            for (size_t channel = 0; channel < 3; ++channel)
                color[pixel * 4 + channel] = std::bit_cast<uint16_t>(static_cast<_Float16>(sum[channel] / 16.0f));
            color[pixel * 4 + 3] = 0x3c00u;
        }
}

// Quality scene (D4R_HARNESS_QUALITY_SCENE=<report file, Windows path>): a static analytic scene with
// detail near and below the render pixel pitch (blinds, thin poles, fine
// diagonals), rendered like a game feeds DLSS: one jittered point sample per
// render pixel. Coordinates are in output pixels.
static float quality_scene(float x, float y, UINT outWidth, UINT outHeight)
{
    const float u = x / outWidth, v = y / outHeight;
    auto frac = [](float value) { return value - std::floor(value); };
    if (u > 0.05f && u < 0.45f && v > 0.1f && v < 0.9f) // blinds: 5 px period, 35% slat
        return frac(y / 5.0f) < 0.35f ? 0.9f : 0.08f;
    if (u > 0.55f && u < 0.95f && v > 0.1f && v < 0.45f) // thin poles: 7 px period, 1.4 px wide
        return frac(x / 7.0f) < 0.2f ? 0.85f : 0.1f;
    if (u > 0.55f && u < 0.95f && v > 0.55f && v < 0.9f) // fine diagonals: 6 px period
        return frac((x + y) / 6.0f) < 0.25f ? 0.8f : 0.12f;
    return 0.3f + 0.1f * std::sin(u * 6.0f) * std::cos(v * 5.0f);
}

// Pole and cable geometry resembling the reported Townfall failure. All
// positions are in output pixels, so both resource-width cases get identical
// active samples and only their three padding columns differ.
static float pole_scene(float x, float y, UINT outWidth, UINT outHeight)
{
    const float u = x / outWidth, v = y / outHeight;
    auto line = [](float px, float py, float ax, float ay, float bx, float by, float width) {
        const float dx = bx - ax, dy = by - ay;
        const float t = std::clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0f, 1.0f);
        const float ex = px - ax - t * dx, ey = py - ay - t * dy;
        return ex * ex + ey * ey < width * width;
    };
    if (std::abs(u - 0.52f) < 0.004f && v > 0.15f)
        return 0.035f;
    for (float height : {0.24f, 0.36f, 0.52f, 0.68f})
        if (line(u, v, 0.49f, height, 0.55f, height - 0.004f, 0.00045f))
            return 0.045f;
    if (line(u, v, 0.0f, 0.37f, 1.0f, 0.21f, 0.00036f) ||
        line(u, v, 0.1f, 0.02f, 0.93f, 0.85f, 0.00030f) ||
        line(u, v, 0.02f, 0.7f, 0.95f, 0.44f, 0.00025f))
        return 0.04f;
    return 0.46f + 0.03f * v;
}

// D4R_HARNESS_REPLAY_DIR=<dir> replays frames captured in a game by the shim
// (D4R_SHIM_INPUT_DUMP_DIR + D4R_SHIM_CAPTURE_COUNT): frame-NNNNNN-color.rgba16f,
// -depth.r32f, -motion.rg16f and -params.txt, starting at D4R_HARNESS_REPLAY_START.
struct ReplayFrame
{
    std::vector<uint8_t> color, depth, motion;
    float jitterX = 0, jitterY = 0, mvScaleX = 1, mvScaleY = 1, sharpness = 0;
    float preExposure = 1, exposureScale = 1, frameTime = 16.666667f;
    int reset = 0, hasExposure = 0;
    unsigned renderWidth = 0, renderHeight = 0;
    unsigned colorBaseX = 0, colorBaseY = 0, depthBaseX = 0, depthBaseY = 0, mvBaseX = 0, mvBaseY = 0;
};

static bool read_file(const std::string& path, std::vector<uint8_t>& bytes, size_t expected)
{
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr)
        return false;
    bytes.resize(expected);
    const size_t got = std::fread(bytes.data(), 1, expected, file);
    const bool extra = std::fgetc(file) != EOF;
    std::fclose(file);
    return got == expected && !extra;
}

static bool load_replay_frame(const char* directory, uint32_t frame, size_t colorBytes, size_t depthBytes,
                              size_t motionBytes, ReplayFrame& out)
{
    char prefix[MAX_PATH];
    std::snprintf(prefix, sizeof(prefix), "%s\\frame-%06u-", directory, frame);
    const std::string base = prefix;
    if (!read_file(base + "color.rgba16f", out.color, colorBytes) ||
        !read_file(base + "depth.r32f", out.depth, depthBytes) ||
        !read_file(base + "motion.rg16f", out.motion, motionBytes))
    {
        std::fprintf(stderr, "replay frame %u: missing or mis-sized plane under %s\n", frame, prefix);
        return false;
    }
    FILE* file = std::fopen((base + "params.txt").c_str(), "r");
    if (file == nullptr)
    {
        std::fprintf(stderr, "replay frame %u: no params\n", frame);
        return false;
    }
    char key[64];
    while (std::fscanf(file, "%63s", key) == 1)
    {
        const std::string name = key;
        if (name == "jitter") std::fscanf(file, "%f %f", &out.jitterX, &out.jitterY);
        else if (name == "mv_scale") std::fscanf(file, "%f %f", &out.mvScaleX, &out.mvScaleY);
        else if (name == "sharpness") std::fscanf(file, "%f", &out.sharpness);
        else if (name == "pre_exposure") std::fscanf(file, "%f", &out.preExposure);
        else if (name == "exposure_scale") std::fscanf(file, "%f", &out.exposureScale);
        else if (name == "frame_time") std::fscanf(file, "%f", &out.frameTime);
        else if (name == "reset") std::fscanf(file, "%d", &out.reset);
        else if (name == "has_exposure") std::fscanf(file, "%d", &out.hasExposure);
        else if (name == "render") std::fscanf(file, "%u %u", &out.renderWidth, &out.renderHeight);
        else if (name == "color_base") std::fscanf(file, "%u %u", &out.colorBaseX, &out.colorBaseY);
        else if (name == "depth_base") std::fscanf(file, "%u %u", &out.depthBaseX, &out.depthBaseY);
        else if (name == "mv_base") std::fscanf(file, "%u %u", &out.mvBaseX, &out.mvBaseY);
        else std::fscanf(file, "%*[^\n]");
    }
    std::fclose(file);
    return true;
}

static float halton(uint32_t index, uint32_t base)
{
    float result = 0.0f, fraction = 1.0f;
    for (; index > 0; index /= base)
    {
        fraction /= static_cast<float>(base);
        result += fraction * static_cast<float>(index % base);
    }
    return result;
}

int main(int argc, char** argv)
{
    trace_stage("entered main");
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s SHIM_DLL OUTPUT_RAW [FRAMES]\n", argv[0]);
        return 2;
    }
    const int frames = argc > 3 ? std::atoi(argv[3]) : 6;
    const UINT inWidth = argc > 7 ? static_cast<UINT>(std::atoi(argv[4])) : 640;
    const UINT inHeight = argc > 7 ? static_cast<UINT>(std::atoi(argv[5])) : 360;
    const UINT outWidth = argc > 7 ? static_cast<UINT>(std::atoi(argv[6])) : 1280;
    const UINT outHeight = argc > 7 ? static_cast<UINT>(std::atoi(argv[7])) : 720;
    const char* resourceWidthSetting = std::getenv("D4R_HARNESS_RESOURCE_WIDTH");
    const UINT resourceWidth = resourceWidthSetting != nullptr
        ? static_cast<UINT>(std::atoi(resourceWidthSetting)) : inWidth;
    if (resourceWidth < inWidth || resourceWidth > inWidth + 256)
    {
        std::fprintf(stderr, "Resource width must be between active width and active width + 256\n");
        return 2;
    }
    const bool motionScene = std::getenv("D4R_HARNESS_MOTION_SCENE") != nullptr;
    const bool jitterScene = std::getenv("D4R_HARNESS_JITTER_SCENE") != nullptr;
    if (motionScene && jitterScene)
    {
        std::fprintf(stderr, "Choose one synthetic temporal scene at a time\n");
        return 2;
    }
    const bool qualityScene = std::getenv("D4R_HARNESS_QUALITY_SCENE") != nullptr;
    const bool poleScene = std::getenv("D4R_HARNESS_POLE_SCENE") != nullptr;
    const bool exposureRGBA32 = std::getenv("D4R_HARNESS_EXPOSURE_RGBA32") != nullptr;
    const char* replayDir = std::getenv("D4R_HARNESS_REPLAY_DIR");
    const bool rgba8 = std::getenv("D4R_HARNESS_RGBA8") != nullptr;
    if (rgba8 && (qualityScene || motionScene || jitterScene || replayDir != nullptr))
    {
        std::fprintf(stderr, "RGBA8 format probe requires the static synthetic scene\n");
        return 2;
    }
    const uint32_t replayStart = std::getenv("D4R_HARNESS_REPLAY_START") != nullptr
        ? static_cast<uint32_t>(std::atoi(std::getenv("D4R_HARNESS_REPLAY_START"))) : 1;
    const bool temporal = std::getenv("D4R_HARNESS_TEMPORAL") != nullptr || motionScene || jitterScene || qualityScene ||
                          replayDir != nullptr;
    const bool forceReset = std::getenv("D4R_HARNESS_FORCE_RESET") != nullptr;
    const bool saveFrames = std::getenv("D4R_HARNESS_SAVE_FRAMES") != nullptr;
    const char* waitSetting = std::getenv("D4R_HARNESS_FRAME_WAIT_MS");
    const int requestedWait = waitSetting != nullptr ? std::atoi(waitSetting) : 200;
    const DWORD frameWaitMs = requestedWait >= 0 && requestedWait <= 10000 ? requestedWait : 200;
    trace_stage(temporal ? "temporal history enabled" : "reset each frame");

    HMODULE shim = nullptr;
    if (std::getenv("D4R_HARNESS_LOAD_NGX_EARLY") != nullptr)
    {
        trace_stage("loading NGX DLL before D3D12 setup");
        shim = LoadLibraryA(argv[1]);
        trace_stage(shim != nullptr ? "early NGX DLL load succeeded" : "early NGX DLL load failed");
        if (shim == nullptr)
            return 1;
    }

    HMODULE d3d12 = LoadLibraryA("d3d12.dll");
    auto createDevice = reinterpret_cast<HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**)>(
        reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")));
    if (createDevice == nullptr ||
        !check(createDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&g_device)),
               "D3D12CreateDevice"))
        return 1;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    check(g_device->CreateCommandQueue(&queueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&g_queue)),
          "CreateCommandQueue");
    check(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
                                           reinterpret_cast<void**>(&g_allocator)),
          "CreateCommandAllocator");
    check(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_allocator, nullptr,
                                      __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&g_list)),
          "CreateCommandList");
    check(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&g_fence)),
          "CreateFence");
    g_event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    std::printf("D3D12 device ready\n");
    trace_stage("D3D12 device ready");

    // Synthetic inputs, identical to ngx_cuda_probe.cpp.
    std::vector<uint16_t> color(static_cast<size_t>(resourceWidth) * inHeight * 4);
    constexpr uint16_t dark = 0x2a66u, mid = 0x3400u, bright = 0x3a66u, white = 0x3c00u;
    for (UINT y = 0; y < inHeight; ++y)
        for (UINT x = 0; x < inWidth; ++x)
        {
            const size_t pixel = static_cast<size_t>(y) * resourceWidth + x;
            const bool fence = (x % 24u == 0u) || (y % 24u == 0u);
            const bool checker = (((x / 4u) + (y / 4u)) & 1u) != 0;
            color[pixel * 4 + 0] = fence ? white : (checker ? mid : dark);
            color[pixel * 4 + 1] = ((x + y) % 32u < 2u) ? bright : (checker ? dark : mid);
            color[pixel * 4 + 2] = (((x / 8u) ^ (y / 8u)) & 1u) ? bright : dark;
            color[pixel * 4 + 3] = white;
        }
    for (UINT y = 0; y < inHeight; ++y)
        for (UINT x = inWidth; x < resourceWidth; ++x)
        {
            const size_t pixel = (static_cast<size_t>(y) * resourceWidth + x) * 4;
            color[pixel] = white;
            color[pixel + 1] = color[pixel + 2] = 0;
            color[pixel + 3] = white;
        }
    const std::vector<uint16_t> backgroundColor = color;
    std::vector<float> depth(static_cast<size_t>(resourceWidth) * inHeight, 0.5f);
    for (UINT y = 0; y < inHeight; ++y)
        for (UINT x = inWidth; x < resourceWidth; ++x)
            depth[static_cast<size_t>(y) * resourceWidth + x] = 0.0f;
    // D4R_HARNESS_MV_HIRES=1 gives DLSS display-resolution motion vectors,
    // what it expects without the MVLowRes create flag (as games pass them).
    const bool motionHighRes = std::getenv("D4R_HARNESS_MV_HIRES") != nullptr;
    const UINT motionWidth = motionHighRes ? outWidth : resourceWidth, motionHeight = motionHighRes ? outHeight : inHeight;
    std::vector<uint16_t> motion(static_cast<size_t>(motionWidth) * motionHeight * 2, 0);
    const float exposureValue = 1.0f;
    const std::array<float, 4> exposureRGBA = {exposureValue, 0.25f, 0.5f, 0.75f};
    std::vector<uint16_t> outputInit(static_cast<size_t>(outWidth) * outHeight * 4, 0);
    std::vector<uint8_t> color8;
    std::vector<uint8_t> outputInit8;
    if (rgba8)
    {
        color8.resize(color.size());
        for (size_t index = 0; index < color.size(); ++index)
            color8[index] = static_cast<uint8_t>(std::lround(
                std::clamp(static_cast<float>(std::bit_cast<_Float16>(color[index])), 0.0f, 1.0f) * 255.0f));
        outputInit8.resize(static_cast<size_t>(outWidth) * outHeight * 4);
    }

    const auto srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const char* featureSetting = std::getenv("D4R_HARNESS_FEATURE_ID");
    const unsigned int featureId = featureSetting ? static_cast<unsigned int>(std::strtoul(featureSetting, nullptr, 0)) : 1;
    const bool reconstruction = featureId == 13;
    std::array<ID3D12Resource*, 4> guides{};
    if (reconstruction)
    {
        // Planar synthetic scene: world-space +Z normals, rough diffuse material.
        // Distinct guides exercise denoising inputs rather than aliasing the noisy colour.
        std::vector<uint16_t> guide(static_cast<size_t>(resourceWidth) * inHeight * 4);
        const std::array<std::array<float, 4>, 4> values{{
            {0.0f, 0.0f, 1.0f, 0.0f}, {0.5f, 0.0f, 0.0f, 0.0f},
            {0.7f, 0.4f, 0.2f, 1.0f}, {0.04f, 0.04f, 0.04f, 1.0f}}};
        for (size_t input = 0; input < guides.size(); ++input)
        {
            // Cyberpunk supplies UNORM8 albedos, while normals remain floating point.
            const bool unorm = input >= 2;
            guides[input] = create_texture(resourceWidth, inHeight,
                                           unorm ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT,
                                           D3D12_RESOURCE_FLAG_NONE);
            if (!guides[input]) return 1;
            for (size_t pixel = 0; pixel < guide.size() / 4; ++pixel)
                for (size_t component = 0; component < 4; ++component)
                    if (unorm)
                        reinterpret_cast<uint8_t*>(guide.data())[pixel * 4 + component] =
                            static_cast<uint8_t>(std::lround(values[input][component] * 255.0f));
                    else
                        guide[pixel * 4 + component] =
                            std::bit_cast<uint16_t>(static_cast<_Float16>(values[input][component]));
            upload(guides[input], guide.data(), resourceWidth * (unorm ? 4 : 8), srv);
        }
    }
    const bool alphaScenario = reconstruction && std::getenv("D4R_HARNESS_RR_ALPHA") != nullptr;
    const UINT alphaBaseX = 4, alphaBaseY = 2;
    const UINT alphaWidth = outWidth + 8, alphaHeight = outHeight + 4;
    ID3D12Resource* alphaInput = nullptr;
    ID3D12Resource* alphaOutput = nullptr;
    if (alphaScenario)
    {
        // A varying coverage guide and a padded destination expose dropped alpha writes and
        // copies that overwrite pixels outside the caller's nonzero output subrect.
        std::vector<float> coverage(static_cast<size_t>(resourceWidth) * inHeight);
        for (UINT y = 0; y < inHeight; ++y)
            for (UINT x = 0; x < resourceWidth; ++x)
                coverage[static_cast<size_t>(y) * resourceWidth + x] =
                    0.25f + 0.5f * static_cast<float>(x) / std::max(1u, resourceWidth - 1);
        alphaInput = create_texture(resourceWidth, inHeight, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE);
        alphaOutput = create_texture(alphaWidth, alphaHeight, DXGI_FORMAT_R32_FLOAT,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (!alphaInput || !alphaOutput) return 1;
        upload(alphaInput, coverage.data(), resourceWidth * sizeof(float), srv);
        std::vector<float> sentinel(static_cast<size_t>(alphaWidth) * alphaHeight, -2.0f);
        upload(alphaOutput, sentinel.data(), alphaWidth * sizeof(float), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    ID3D12Resource* colorTexture = create_texture(resourceWidth, inHeight,
        rgba8 ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* depthTexture = create_texture(resourceWidth, inHeight, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    // D4R_HARNESS_MV_RGBA16=1: motion vectors in an R16G16B16A16_TYPELESS texture (as some games pass them), the
    // unused components filled with junk that must not reach DLSS.
    const bool mvRgba16 = std::getenv("D4R_HARNESS_MV_RGBA16") != nullptr && std::atoi(std::getenv("D4R_HARNESS_MV_RGBA16")) != 0;
    ID3D12Resource* motionTexture = create_texture(motionWidth, motionHeight,
        mvRgba16 ? DXGI_FORMAT_R16G16B16A16_TYPELESS : DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    std::vector<uint16_t> motionWide;
    auto motion_rows = [&](const uint16_t* rg) -> const void* {
        if (!mvRgba16)
            return rg;
        const size_t pixels = static_cast<size_t>(motionWidth) * motionHeight;
        motionWide.resize(pixels * 4);
        for (size_t pixel = 0; pixel < pixels; ++pixel)
        {
            motionWide[pixel * 4 + 0] = rg[pixel * 2 + 0];
            motionWide[pixel * 4 + 1] = rg[pixel * 2 + 1];
            motionWide[pixel * 4 + 2] = 0x5640; // 100.0
            motionWide[pixel * 4 + 3] = 0xd640; // -100.0
        }
        return motionWide.data();
    };
    const UINT motionRowBytes = motionWidth * (mvRgba16 ? 8 : 4);
    ID3D12Resource* exposureTexture = create_texture(1, 1,
        exposureRGBA32 ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* outputTexture = create_texture(outWidth, outHeight,
                                                   rgba8 ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT,
                                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    upload(colorTexture, rgba8 ? static_cast<const void*>(color8.data()) : static_cast<const void*>(color.data()),
           resourceWidth * (rgba8 ? 4 : 8), srv);
    upload(depthTexture, depth.data(), resourceWidth * 4, srv);
    upload(motionTexture, motion_rows(motion.data()), motionRowBytes, srv);
    upload(exposureTexture, exposureRGBA32 ? static_cast<const void*>(exposureRGBA.data())
                                          : static_cast<const void*>(&exposureValue),
           exposureRGBA32 ? 16 : 4, srv);
    upload(outputTexture, rgba8 ? static_cast<const void*>(outputInit8.data()) : static_cast<const void*>(outputInit.data()),
           outWidth * (rgba8 ? 4 : 8), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    std::printf("synthetic inputs uploaded\n");
    trace_stage("synthetic inputs uploaded");

    if (shim == nullptr)
    {
        trace_stage("loading NGX DLL");
        shim = LoadLibraryA(argv[1]);
        trace_stage(shim != nullptr ? "NGX DLL loaded" : "NGX DLL load failed");
    }
    if (shim == nullptr)
    {
        std::printf("LoadLibraryA(%s) failed: %lu\n", argv[1], GetLastError());
        return 1;
    }
    auto init = reinterpret_cast<PFN_Init_Ext>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_Init_Ext")));
    auto capability = reinterpret_cast<PFN_Parameters>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_GetCapabilityParameters")));
    auto allocate = reinterpret_cast<PFN_Parameters>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_AllocateParameters")));
    auto createFeature = reinterpret_cast<PFN_CreateFeature>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_CreateFeature")));
    auto evaluate = reinterpret_cast<PFN_Evaluate>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_EvaluateFeature")));
    auto release = reinterpret_cast<PFN_Release>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_ReleaseFeature")));
    auto shutdown = reinterpret_cast<PFN_Shutdown>(reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_Shutdown")));

    wchar_t dataPath[MAX_PATH];
    GetTempPathW(MAX_PATH, dataPath);
    const char* appIdSetting = std::getenv("D4R_HARNESS_APP_ID");
    const unsigned long long appId = appIdSetting != nullptr ? std::strtoull(appIdSetting, nullptr, 10) : 241534723ULL;
    // Games discover RR before Init; an initialized-only capability answer hides the mode.
    auto requirements = reinterpret_cast<PFN_Requirements>(
        reinterpret_cast<void*>(GetProcAddress(shim, "NVSDK_NGX_D3D12_GetFeatureRequirements")));
    NgxFeatureDiscovery discovery{};
    discovery.sdkVersion = 0x15;
    discovery.feature = reconstruction ? 13 : 1;
    discovery.identifier.value.applicationId = appId;
    discovery.dataPath = dataPath;
    NgxFeatureRequirement requirement{};
    using CreateFactoryFn = HRESULT (*)(REFIID, void**);
    const auto createFactory = reinterpret_cast<CreateFactoryFn>(
        reinterpret_cast<void*>(GetProcAddress(LoadLibraryA("dxgi.dll"), "CreateDXGIFactory1")));
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter* adapter = nullptr;
    LUID adapterLuid{};
    g_device->GetAdapterLuid(&adapterLuid);
    if (FAILED(createFactory(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(adapterLuid, __uuidof(IDXGIAdapter), reinterpret_cast<void**>(&adapter))))
        return 1;
    const NgxResult discoveryResult = requirements(adapter, &discovery, &requirement);
    adapter->Release();
    factory->Release();
    std::printf("GetFeatureRequirements before Init -> 0x%08x, feature=%u supported=%u\n",
                discoveryResult, discovery.feature, requirement.supported);
    if (discoveryResult != NGX_SUCCESS || requirement.supported != 0)
        return 1;
    NgxResult result = init(appId, dataPath, g_device, 0x15, nullptr);
    trace_stage(result == NGX_SUCCESS ? "NGX init succeeded" : "NGX init failed");
    std::printf("NVSDK_NGX_D3D12_Init_Ext -> 0x%08x\n", result);
    if (result != NGX_SUCCESS)
        return 1;
    void* capabilities = nullptr;
    result = capability(&capabilities);
    int available = -1;
    const char* availabilityKey = reconstruction ? "SuperSamplingDenoising.Available" : "SuperSampling.Available";
    if (result == NGX_SUCCESS)
        d4r_ngx_get_int(capabilities, availabilityKey, &available);
    std::printf("GetCapabilityParameters -> 0x%08x, %s=%d\n", result, availabilityKey, available);

    void* parameters = nullptr;
    result = allocate(&parameters);
    std::printf("AllocateParameters -> 0x%08x\n", result);
    // D4R_HARNESS_CREATE_SIZE=WxH creates the feature for another render size than the frames use, as
    // Streamline does for a frame or more after a quality change (PRAGMATA: 1280x720 feature, 854x480 frames).
    unsigned int createWidth = inWidth, createHeight = inHeight;
    if (const char* size = std::getenv("D4R_HARNESS_CREATE_SIZE"))
        std::sscanf(size, "%ux%u", &createWidth, &createHeight);
    d4r_ngx_set_uint(parameters, "Width", createWidth);
    d4r_ngx_set_uint(parameters, "Height", createHeight);
    d4r_ngx_set_uint(parameters, "OutWidth", outWidth);
    d4r_ngx_set_uint(parameters, "OutHeight", outHeight);
    // D4R_HARNESS_QUALITY selects NVSDK_NGX_PerfQuality_Value (default 2,
    // MaxQuality; 0 MaxPerf, 1 Balanced, 3 UltraPerformance, 5 DLAA).
    const char* quality = std::getenv("D4R_HARNESS_QUALITY");
    d4r_ngx_set_int(parameters, "PerfQualityValue", quality != nullptr && *quality != '\0' ? std::atoi(quality) : 2);
    // D4R_HARNESS_CREATE_FLAGS mirrors a game's NVSDK_NGX_DLSS_Feature_Flags
    // (e.g. 0x49 = HDR | DepthInverted | AutoExposure).
    // Transformer RR requires HDR color and low-resolution motion vectors.
    const char* createFlags = std::getenv("D4R_HARNESS_CREATE_FLAGS");
    d4r_ngx_set_int(parameters, "DLSS.Feature.Create.Flags",
                    (createFlags != nullptr ? static_cast<int>(std::strtol(createFlags, nullptr, 0)) :
                                             (reconstruction ? (motionHighRes ? 1 : 3) : 0)) |
                    (alphaScenario ? 0x80 : 0));
    if (alphaScenario)
        d4r_ngx_set_int(parameters, "DLSS.Enable.Output.Subrects", 1);
    std::array<float, 16> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    if (reconstruction)
    {
        d4r_ngx_set_int(parameters, "DLSS.Denoise.Mode", 1);
        d4r_ngx_set_int(parameters, "DLSS.Roughness.Mode", 0);
        d4r_ngx_set_int(parameters, "DLSS.Use.HW.Depth", 0);
        d4r_ngx_set_void(parameters, "WorldToViewMatrix", identity.data());
        d4r_ngx_set_void(parameters, "ViewToClipMatrix", identity.data());
        const char* preset = std::getenv("D4R_HARNESS_RR_PRESET");
        if (preset)
            for (const char* mode : {"DLAA", "Quality", "Balanced", "Performance", "UltraPerformance", "UltraQuality"})
                d4r_ngx_set_uint(parameters, (std::string("RayReconstruction.Hint.Render.Preset.") + mode).c_str(),
                                 static_cast<unsigned int>(std::strtoul(preset, nullptr, 0)));
    }
    NgxHandle* feature = nullptr;
    if (std::getenv("D4R_HARNESS_LIFETIME_TEST") != nullptr && !test_external_lifetime())
        return 1;
    result = createFeature(g_list, featureId, parameters, &feature);
    std::printf("CreateFeature -> 0x%08x handle=%p\n", result, static_cast<void*>(feature));
    if (result != NGX_SUCCESS)
        return 1;

    d4r_ngx_set_d3d12_resource(parameters, "Color", colorTexture);
    d4r_ngx_set_d3d12_resource(parameters, "Depth", depthTexture);
    d4r_ngx_set_d3d12_resource(parameters, "MotionVectors", motionTexture);
    d4r_ngx_set_d3d12_resource(parameters, "Output", outputTexture);
    d4r_ngx_set_d3d12_resource(parameters, "ExposureTexture", exposureTexture);
    if (reconstruction)
    {
        // The SDK's caller-facing names (nvsdk_ngx_defs.h: NVSDK_NGX_Parameter_GBuffer_Normals,
        // _Roughness, _DiffuseAlbedo, _SpecularAlbedo), which is what a game registers. The
        // shim republishes the two albedos under the names the denoiser reads.
        const char* names[] = {"GBuffer.Normals", "GBuffer.Roughness",
                               "GBuffer.DiffuseAlbedo", "GBuffer.SpecularAlbedo"};
        for (size_t input = 0; input < guides.size(); ++input)
            d4r_ngx_set_d3d12_resource(parameters, names[input], guides[input]);
    }
    if (alphaScenario)
    {
        d4r_ngx_set_d3d12_resource(parameters, "DLSSD.Alpha", alphaInput);
        d4r_ngx_set_d3d12_resource(parameters, "DLSSD.OutputAlpha", alphaOutput);
        d4r_ngx_set_uint(parameters, "DLSSD.OutputAlpha.Subrect.Base.X", alphaBaseX);
        d4r_ngx_set_uint(parameters, "DLSSD.OutputAlpha.Subrect.Base.Y", alphaBaseY);
    }
    d4r_ngx_set_float(parameters, "Jitter.Offset.X", 0.0f);
    d4r_ngx_set_float(parameters, "Jitter.Offset.Y", 0.0f);
    d4r_ngx_set_float(parameters, "MV.Scale.X", 1.0f);
    d4r_ngx_set_float(parameters, "MV.Scale.Y", 1.0f);
    d4r_ngx_set_float(parameters, "Sharpness", 0.0f);
    d4r_ngx_set_float(parameters, "DLSS.Pre.Exposure", 1.0f);
    d4r_ngx_set_float(parameters, "DLSS.Exposure.Scale", 1.0f);
    d4r_ngx_set_float(parameters, "FrameTimeDeltaInMsec", 16.666667f);
    d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Width", inWidth);
    d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Height", inHeight);
    // Default to the proven single-frame reference. Temporal probes reset only
    // the first frame and let the same official NGX feature accumulate history.
    d4r_ngx_set_int(parameters, "Reset", 1);

    // Quality scene ground truth (8x8 supersampled output pixels) and metrics.
    // D4R_HARNESS_QUALITY_PAN="vx,vy" scrolls the scene content by whole
    // output pixels per frame (motion vectors report it), so every frame's
    // truth is an exact shift of one canvas that covers the whole scroll.
    int panX = 0, panY = 0;
    if (const char* pan = std::getenv("D4R_HARNESS_QUALITY_PAN"))
        std::sscanf(pan, "%d,%d", &panX, &panY);
    const int canvasX0 = std::min(0, -frames * panX), canvasY0 = std::min(0, -frames * panY);
    const int canvasWidth = static_cast<int>(outWidth) + std::abs(frames * panX);
    const int canvasHeight = static_cast<int>(outHeight) + std::abs(frames * panY);
    std::vector<float> truth;
    std::vector<float> previousError;
    double flickerSum = 0.0;
    int flickerFrames = 0;
    if (qualityScene && !poleScene)
    {
        truth.resize(static_cast<size_t>(canvasWidth) * canvasHeight);
        const int truthSamples = poleScene ? 2 : 8;
        for (int y = 0; y < canvasHeight; ++y)
            for (int x = 0; x < canvasWidth; ++x)
            {
                float sum = 0.0f;
                for (int sy = 0; sy < truthSamples; ++sy)
                    for (int sx = 0; sx < truthSamples; ++sx)
                        sum += poleScene
                            ? pole_scene(x + canvasX0 + (sx + 0.5f) / truthSamples,
                                         y + canvasY0 + (sy + 0.5f) / truthSamples, outWidth, outHeight)
                            : quality_scene(x + canvasX0 + (sx + 0.5f) / truthSamples,
                                            y + canvasY0 + (sy + 0.5f) / truthSamples, outWidth, outHeight);
                truth[static_cast<size_t>(y) * canvasWidth + x] = sum / (truthSamples * truthSamples);
            }
    }
    const float scaleX = static_cast<float>(outWidth) / inWidth, scaleY = static_cast<float>(outHeight) / inHeight;
    const uint32_t phases = static_cast<uint32_t>(std::lround(8.0f * scaleX * scaleY));

    // A discarded recording followed by a submitted frame must not make the
    // discarded inputs appear ready, or contaminate the new feature's history.
    if (std::getenv("D4R_HARNESS_SKIP_RECORDED_INPUT") != nullptr)
    {
        if (evaluate(g_list, feature, parameters, nullptr) != NGX_SUCCESS ||
            !check(g_list->Close(), "skip input list close") ||
            !check(g_allocator->Reset(), "skip input allocator reset") ||
            !check(g_list->Reset(g_allocator, nullptr), "skip input list reset"))
            return 1;
        lifecycle_report("SKIP input: discarded recording before submitting the next frame");
    }

    for (int frame = 1; frame <= frames; ++frame)
    {
        if (qualityScene)
        {
            // Halton(2,3) jitter in render pixels, [-0.5, 0.5); a render pixel
            // samples the scene at its centre plus the jitter.
            const uint32_t index = static_cast<uint32_t>(frame - 1) % phases + 1;
            const float jx = halton(index, 2) - 0.5f, jy = halton(index, 3) - 0.5f;
            // DLSS's Jitter.Offset is minus the sample offset (verified: the other
            // signs lose ~9 dB). D4R_HARNESS_JITTER_SIGNS="sx,sy" overrides it.
            float signX = -1.0f, signY = -1.0f;
            if (const char* signs = std::getenv("D4R_HARNESS_JITTER_SIGNS"))
                std::sscanf(signs, "%f,%f", &signX, &signY);
            // Scene content at frame f sits f * pan output pixels further along.
            const float shiftX = static_cast<float>(frame * panX), shiftY = static_cast<float>(frame * panY);
            for (UINT y = 0; y < inHeight; ++y)
                for (UINT x = 0; x < inWidth; ++x)
                {
                    const float sampleX = (x + 0.5f + jx) * scaleX - shiftX;
                    const float sampleY = (y + 0.5f + jy) * scaleY - shiftY;
                    const float value = poleScene
                        ? pole_scene(sampleX, sampleY, outWidth, outHeight)
                        : quality_scene(sampleX, sampleY, outWidth, outHeight);
                    const size_t pixel = static_cast<size_t>(y) * resourceWidth + x;
                    const uint16_t half = std::bit_cast<uint16_t>(static_cast<_Float16>(value));
                    color[pixel * 4 + 0] = color[pixel * 4 + 1] = color[pixel * 4 + 2] = half;
                    color[pixel * 4 + 3] = 0x3c00u;
                }
            update_texture(colorTexture, color.data(), resourceWidth * 8, srv);
            if (frame == 1 && (panX != 0 || panY != 0))
            {
                // Current position -> previous position. Without MVLowRes DLSS
                // reads display-resolution vectors in output pixels (its input
                // kernel adds them to the output pixel position); render-size
                // vectors are in render pixels. D4R_HARNESS_MV_MULT scales them
                // (convention checks).
                const char* multiplier = std::getenv("D4R_HARNESS_MV_MULT");
                const float mvScale = multiplier != nullptr ? std::strtof(multiplier, nullptr) : 1.0f;
                const float unitX = motionHighRes ? 1.0f : 1.0f / scaleX, unitY = motionHighRes ? 1.0f : 1.0f / scaleY;
                const uint16_t mvX = std::bit_cast<uint16_t>(static_cast<_Float16>(-mvScale * panX * unitX));
                const uint16_t mvY = std::bit_cast<uint16_t>(static_cast<_Float16>(-mvScale * panY * unitY));
                for (size_t pixel = 0; pixel < motion.size() / 2; ++pixel)
                {
                    motion[pixel * 2 + 0] = mvX;
                    motion[pixel * 2 + 1] = mvY;
                }
                update_texture(motionTexture, motion_rows(motion.data()), motionRowBytes, srv);
            }
            d4r_ngx_set_float(parameters, "Jitter.Offset.X", signX * jx);
            d4r_ngx_set_float(parameters, "Jitter.Offset.Y", signY * jy);
        }
        if (jitterScene)
        {
            constexpr std::array<float, 8> jitterX =
                {0.0f, -0.25f, 0.25f, -0.375f, 0.125f, 0.375f, -0.125f, 0.0f};
            constexpr std::array<float, 8> jitterY =
                {0.0f, 0.25f, -0.25f, -0.125f, 0.375f, 0.125f, -0.375f, 0.0f};
            const size_t phase = static_cast<size_t>(frame - 1) % jitterX.size();
            render_jittered_background(color, inWidth, inHeight, resourceWidth, jitterX[phase], jitterY[phase]);
            update_texture(colorTexture, color.data(), resourceWidth * 8, srv);
            d4r_ngx_set_float(parameters, "Jitter.Offset.X", jitterX[phase]);
            d4r_ngx_set_float(parameters, "Jitter.Offset.Y", jitterY[phase]);
        }
        if (motionScene)
        {
            color = backgroundColor;
            std::fill(depth.begin(), depth.end(), 0.5f);
            std::fill(motion.begin(), motion.end(), 0);
            const UINT left = 220u + 4u * static_cast<UINT>(frame - 1);
            for (UINT py = 100; py < 180 && py < inHeight; ++py)
                for (UINT px = left; px < left + 80 && px < inWidth; ++px)
                {
                    const size_t pixel = static_cast<size_t>(py) * resourceWidth + px;
                    color[pixel * 4 + 0] = white;
                    color[pixel * 4 + 1] = dark;
                    color[pixel * 4 + 2] = dark;
                    color[pixel * 4 + 3] = white;
                    depth[pixel] = 0.2f;
                    const size_t motionPixel = static_cast<size_t>(py) * motionWidth + px;
                    motion[motionPixel * 2 + 0] = 0xc400u; // -4 render pixels: current position -> previous
                    motion[motionPixel * 2 + 1] = 0;
                }
            update_texture(colorTexture, color.data(), resourceWidth * 8, srv);
            update_texture(depthTexture, depth.data(), resourceWidth * 4, srv);
            update_texture(motionTexture, motion_rows(motion.data()), motionRowBytes, srv);
        }
        if (replayDir != nullptr)
        {
            ReplayFrame replay;
            if (!load_replay_frame(replayDir, replayStart + static_cast<uint32_t>(frame - 1),
                                   color.size() * sizeof(uint16_t), depth.size() * sizeof(float),
                                   motion.size() * sizeof(uint16_t), replay))
                return 1;
            update_texture(colorTexture, replay.color.data(), resourceWidth * 8, srv);
            update_texture(depthTexture, replay.depth.data(), resourceWidth * 4, srv);
            update_texture(motionTexture, motion_rows(reinterpret_cast<const uint16_t*>(replay.motion.data())), motionRowBytes, srv);
            d4r_ngx_set_float(parameters, "Jitter.Offset.X", replay.jitterX);
            d4r_ngx_set_float(parameters, "Jitter.Offset.Y", replay.jitterY);
            d4r_ngx_set_float(parameters, "MV.Scale.X", replay.mvScaleX);
            d4r_ngx_set_float(parameters, "MV.Scale.Y", replay.mvScaleY);
            d4r_ngx_set_float(parameters, "Sharpness", replay.sharpness);
            d4r_ngx_set_float(parameters, "DLSS.Pre.Exposure", replay.preExposure);
            d4r_ngx_set_float(parameters, "DLSS.Exposure.Scale", replay.exposureScale);
            d4r_ngx_set_float(parameters, "FrameTimeDeltaInMsec", replay.frameTime);
            d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Width", replay.renderWidth);
            d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Height", replay.renderHeight);
            d4r_ngx_set_uint(parameters, "DLSS.Input.Color.Subrect.Base.X", replay.colorBaseX);
            d4r_ngx_set_uint(parameters, "DLSS.Input.Color.Subrect.Base.Y", replay.colorBaseY);
            d4r_ngx_set_uint(parameters, "DLSS.Input.Depth.Subrect.Base.X", replay.depthBaseX);
            d4r_ngx_set_uint(parameters, "DLSS.Input.Depth.Subrect.Base.Y", replay.depthBaseY);
            d4r_ngx_set_uint(parameters, "DLSS.Input.MV.Subrect.Base.X", replay.mvBaseX);
            d4r_ngx_set_uint(parameters, "DLSS.Input.MV.Subrect.Base.Y", replay.mvBaseY);
            d4r_ngx_set_int(parameters, "Reset", frame == 1 || replay.reset != 0 ? 1 : 0);
            // Games with AutoExposure pass no exposure texture; the synthetic scenes pass 1.0.
            d4r_ngx_set_d3d12_resource(parameters, "ExposureTexture", replay.hasExposure ? exposureTexture : nullptr);
        }
        else if (temporal)
            d4r_ngx_set_int(parameters, "Reset", forceReset || frame == 1 ? 1 : 0);
        const DWORD start = GetTickCount();
        result = evaluate(g_list, feature, parameters, nullptr);
        if (const char* delay = std::getenv("D4R_HARNESS_DELAY_SUBMIT_MS"))
            Sleep(static_cast<DWORD>(std::clamp(std::atoi(delay), 0, 4000)));
        submit_and_wait();
        std::printf("frame %d: EvaluateFeature -> 0x%08x (%lu ms incl. submit)\n", frame, result, GetTickCount() - start);
        // NGX rejecting an evaluation outright is reported here; only a failure raised later,
        // on the shim's worker, is invisible in this return value and shows up in the image.
        if (result != NGX_SUCCESS)
        {
            std::fprintf(stderr, "EvaluateFeature rejected frame %d: 0x%08x\n", frame, result);
            return 1;
        }
        Sleep(frameWaitMs); // let the CUDA worker finish before the next presentation
        if (qualityScene && !poleScene && frame > frames - 17)
        {
            // Luma of the presented result vs the ground truth at this frame's
            // scroll position, and the frame-to-frame change of that error
            // (ideally 0; for a static scene it is the change of the output).
            const std::vector<uint8_t> raw = read_back(outputTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outWidth * 8);
            const uint16_t* halves = reinterpret_cast<const uint16_t*>(raw.data());
            const size_t pixels = static_cast<size_t>(outWidth) * outHeight;
            // The shim presents the previous evaluation's result (one frame of
            // pipeline latency), so compare with the truth of that frame.
            static const int presentedAge = std::getenv("D4R_HARNESS_QUALITY_AGE") != nullptr
                ? std::atoi(std::getenv("D4R_HARNESS_QUALITY_AGE")) : 1;
            const int shownFrame = frame - presentedAge;
            std::vector<float> error(pixels);
            double squared = 0.0, blindsSquared = 0.0;
            size_t blindsPixels = 0;
            for (size_t pixel = 0; pixel < pixels; ++pixel)
            {
                float value = 0.0f;
                for (int channel = 0; channel < 3; ++channel)
                    value += static_cast<float>(std::bit_cast<_Float16>(halves[pixel * 4 + channel]));
                const float luma = std::clamp(value / 3.0f, 0.0f, 1.0f);
                // Content coordinates: the screen pixel minus the scroll.
                const int cx = static_cast<int>(pixel % outWidth) - shownFrame * panX;
                const int cy = static_cast<int>(pixel / outWidth) - shownFrame * panY;
                const float target = truth[static_cast<size_t>(cy - canvasY0) * canvasWidth + (cx - canvasX0)];
                error[pixel] = luma - target;
                squared += static_cast<double>(error[pixel]) * error[pixel];
                const float u = static_cast<float>(cx) / outWidth;
                const float v = static_cast<float>(cy) / outHeight;
                if (u > 0.05f && u < 0.45f && v > 0.1f && v < 0.9f)
                {
                    blindsSquared += static_cast<double>(error[pixel]) * error[pixel];
                    ++blindsPixels;
                }
            }
            double change = 0.0;
            if (!previousError.empty())
            {
                for (size_t pixel = 0; pixel < pixels; ++pixel)
                    change += std::fabs(error[pixel] - previousError[pixel]);
                change /= static_cast<double>(pixels);
                flickerSum += change;
                ++flickerFrames;
            }
            previousError = error;
            if (const char* dump = std::getenv("D4R_HARNESS_QUALITY_DUMP"))
            {
                // Every measured frame's raw RGBA16F output, for crops and videos.
                char path[MAX_PATH];
                std::snprintf(path, sizeof(path), "%s.%03d", dump, frame);
                if (FILE* file = std::fopen(path, "wb"))
                {
                    std::fwrite(raw.data(), 1, raw.size(), file);
                    std::fclose(file);
                }
            }
            FILE* report = std::fopen(std::getenv("D4R_HARNESS_QUALITY_SCENE"), "a");
            if (report == nullptr)
                report = stdout;
            std::fprintf(report, "QUALITY frame=%d psnr=%.2f blinds_psnr=%.2f flicker=%.5f\n", frame,
                        10.0 * std::log10(1.0 / (squared / pixels)),
                        10.0 * std::log10(1.0 / (blindsSquared / blindsPixels)), change);
            if (frame == frames)
                std::fprintf(report, "QUALITY summary psnr=%.2f blinds_psnr=%.2f mean_flicker=%.5f\n",
                             10.0 * std::log10(1.0 / (squared / pixels)),
                             10.0 * std::log10(1.0 / (blindsSquared / blindsPixels)),
                             flickerFrames != 0 ? flickerSum / flickerFrames : 0.0);
            if (report != stdout)
                std::fclose(report);
        }
        if (saveFrames)
        {
            const std::vector<uint8_t> frameOutput =
                read_back(outputTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outWidth * (rgba8 ? 4 : 8));
            const std::string framePath = std::string(argv[2]) + ".frame" + std::to_string(frame);
            if (FILE* frameFile = std::fopen(framePath.c_str(), "wb"))
            {
                std::fwrite(frameOutput.data(), 1, frameOutput.size(), frameFile);
                std::fclose(frameFile);
            }
        }
    }

    const std::vector<uint8_t> output = read_back(outputTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                  outWidth * (rgba8 ? 4 : 8));
    FILE* file = std::fopen(argv[2], "wb");
    if (file != nullptr)
    {
        std::fwrite(output.data(), 1, output.size(), file);
        std::fclose(file);
    }
    size_t nonzero = 0;
    for (uint8_t value : output)
        nonzero += value != 0;
    std::printf("output read back: %zu of %zu bytes nonzero, written to %s\n", nonzero, output.size(), argv[2]);
    // The shim queues each evaluation on its own worker, so a failure inside the denoiser is
    // invisible in this thread's EvaluateFeature return value: the frame is accepted and the
    // output texture is simply never written. The image is the only observable. On the static
    // scene the denoiser reconstructs, so the output must carry that scene's structure:
    // compare block-averaged luma against the same average of the input. Averaging first makes
    // this about scene structure rather than per-pixel detail a denoiser is free to change. An
    // untouched output buffer (the initial fill) has no variance and cannot correlate.
    // Temporal scenes move content between the input and the frame presented, and replay
    // frames come from a capture, so neither is checked here.
    if (reconstruction && !rgba8 && replayDir == nullptr && !motionScene && !jitterScene && !qualityScene)
    {
        const uint16_t* const halves = reinterpret_cast<const uint16_t*>(output.data());
        constexpr UINT block = 8;
        const UINT blocksX = (outWidth + block - 1) / block, blocksY = (outHeight + block - 1) / block;
        std::vector<double> outBlocks, inBlocks;
        outBlocks.reserve(static_cast<size_t>(blocksX) * blocksY);
        inBlocks.reserve(static_cast<size_t>(blocksX) * blocksY);
        for (UINT by = 0; by < blocksY; ++by)
            for (UINT bx = 0; bx < blocksX; ++bx)
            {
                double outSum = 0.0, inSum = 0.0;
                size_t count = 0;
                for (UINT y = by * block; y < std::min(by * block + block, outHeight); ++y)
                    for (UINT x = bx * block; x < std::min(bx * block + block, outWidth); ++x)
                    {
                        const size_t outPixel = (static_cast<size_t>(y) * outWidth + x) * 4;
                        // The same output block mapped back onto the render grid: the denoiser
                        // upsamples exactly this scene, so no filter assumption is needed.
                        const UINT sourceY = static_cast<UINT>(static_cast<uint64_t>(y) * inHeight / outHeight);
                        const UINT sourceX = static_cast<UINT>(static_cast<uint64_t>(x) * inWidth / outWidth);
                        const size_t inPixel = (static_cast<size_t>(sourceY) * resourceWidth + sourceX) * 4;
                        for (int channel = 0; channel < 3; ++channel)
                        {
                            outSum += static_cast<double>(std::bit_cast<_Float16>(halves[outPixel + channel]));
                            inSum += static_cast<double>(std::bit_cast<_Float16>(color[inPixel + channel]));
                        }
                        ++count;
                    }
                outBlocks.push_back(outSum / (3.0 * static_cast<double>(count)));
                inBlocks.push_back(inSum / (3.0 * static_cast<double>(count)));
            }
        const double n = static_cast<double>(outBlocks.size());
        double sumOut = 0.0, sumIn = 0.0, sumOutOut = 0.0, sumInIn = 0.0, sumProduct = 0.0;
        for (size_t index = 0; index < outBlocks.size(); ++index)
        {
            sumOut += outBlocks[index];
            sumIn += inBlocks[index];
            sumOutOut += outBlocks[index] * outBlocks[index];
            sumInIn += inBlocks[index] * inBlocks[index];
            sumProduct += outBlocks[index] * inBlocks[index];
        }
        const double outVariance = sumOutOut - sumOut * sumOut / n;
        const double inVariance = sumInIn - sumIn * sumIn / n;
        const double correlation = (sumProduct - sumOut * sumIn / n) /
                                   (std::sqrt(outVariance) * std::sqrt(inVariance));
        std::printf("RR output block correlation with the input scene: %.6f over %zu blocks "
                    "(mean luma %.6f vs %.6f)\n",
                    correlation, outBlocks.size(), sumOut / n, sumIn / n);
        // A scene-structure floor, not a quality comparison with NVIDIA output. Unwritten
        // output has zero variance and produces NaN here; nonfinite or unrelated output also
        // fails. Denoising may change individual pixels without losing the scene's structure.
        if (!(correlation >= 0.5))
        {
            std::fprintf(stderr,
                         "Ray Reconstruction produced no usable image: block correlation with the "
                         "input scene is %.6f (unwritten, unrelated, or nonfinite output)\n",
                         correlation);
            return 1;
        }
    }
    if (alphaScenario)
    {
        const std::vector<uint8_t> raw =
            read_back(alphaOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, alphaWidth * sizeof(float));
        const float* values = reinterpret_cast<const float*>(raw.data());
        size_t invalid = 0, overwritten = 0, leftCount = 0, rightCount = 0;
        double left = 0, right = 0;
        for (UINT y = 0; y < alphaHeight; ++y)
            for (UINT x = 0; x < alphaWidth; ++x)
            {
                const float value = values[static_cast<size_t>(y) * alphaWidth + x];
                const bool inside = x >= alphaBaseX && x < alphaBaseX + outWidth &&
                                    y >= alphaBaseY && y < alphaBaseY + outHeight;
                if (!inside)
                {
                    overwritten += value != -2.0f;
                    continue;
                }
                invalid += !std::isfinite(value) || value < -0.001f || value > 1.001f;
                const UINT localX = x - alphaBaseX;
                if (localX < outWidth / 4)
                {
                    left += value;
                    ++leftCount;
                }
                if (localX >= outWidth * 3 / 4)
                {
                    right += value;
                    ++rightCount;
                }
            }
        const double leftMean = leftCount ? left / leftCount : 0;
        const double rightMean = rightCount ? right / rightCount : 0;
        std::printf("RR alpha: invalid=%zu outside-overwritten=%zu left=%.6f right=%.6f\n",
                    invalid, overwritten, leftMean, rightMean);
        if (FILE* alphaFile = std::fopen((std::string(argv[2]) + ".alpha.raw").c_str(), "wb"))
        {
            std::fwrite(raw.data(), 1, raw.size(), alphaFile);
            std::fclose(alphaFile);
        }
        if (invalid || overwritten || !leftCount || !rightCount || !(rightMean - leftMean > 0.1))
        {
            std::fprintf(stderr, "RR alpha coverage or subrect preservation failed\n");
            return 1;
        }
    }

    if (std::getenv("D4R_HARNESS_DISCARD_EVALUATION") != nullptr)
    {
        const NgxResult queued = evaluate(g_list, feature, parameters, nullptr);
        const ULONGLONG start = GetTickCount64();
        const NgxResult retired = release(feature);
        feature = nullptr;
        if (queued != NGX_SUCCESS || retired != NGX_SUCCESS || GetTickCount64() - start > 2000)
        {
            std::fprintf(stderr, "DISCARD evaluation release failed or blocked\n");
            return 1;
        }
        g_list->Close();
        g_allocator->Reset();
        g_list->Reset(g_allocator, nullptr);
        result = createFeature(g_list, featureId, parameters, &feature);
        if (result != NGX_SUCCESS) return 1;
        lifecycle_report("DISCARD evaluation: release completed without submitting inputs");
    }

    // D4R_HARNESS_RECREATE=N: N more release/create cycles at alternating render sizes (as when a game's DLSS
    // quality setting changes), a few evaluations each, logging this process's VRAM to find leaks per cycle.
    const int recreateCycles = std::getenv("D4R_HARNESS_RECREATE") != nullptr ? std::atoi(std::getenv("D4R_HARNESS_RECREATE")) : 0;
    const bool verifyRecreation = std::getenv("D4R_HARNESS_VERIFY_RECREATION") != nullptr;
    std::vector<uint8_t> recreationReference[2];
    for (int cycle = 1; cycle <= recreateCycles; ++cycle)
    {
        if (release(feature) != NGX_SUCCESS) return 1;
        feature = nullptr;
        const UINT width = cycle % 2 != 0 ? inWidth * 10 / 13 : inWidth;
        const UINT height = cycle % 2 != 0 ? inHeight * 10 / 13 : inHeight;
        d4r_ngx_set_uint(parameters, "Width", width);
        d4r_ngx_set_uint(parameters, "Height", height);
        d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Width", width);
        d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Height", height);
        result = createFeature(g_list, featureId, parameters, &feature);
        if (result != NGX_SUCCESS)
        {
            std::printf("RECREATE cycle %d: CreateFeature -> 0x%08x\n", cycle, result);
            return 1;
        }
        for (int frame = 0; frame < 4; ++frame)
        {
            d4r_ngx_set_int(parameters, "Reset", frame == 0 ? 1 : 0);
            if (evaluate(g_list, feature, parameters, nullptr) != NGX_SUCCESS) return 1;
            submit_and_wait();
            Sleep(frameWaitMs);
        }
        const std::vector<uint8_t> cycleOutput = read_back(outputTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                           outWidth * (rgba8 ? 4 : 8));
        if (verifyRecreation)
        {
            auto& reference = recreationReference[cycle % 2];
            if (reference.empty())
                reference = cycleOutput;
            else if (reference != cycleOutput)
            {
                lifecycle_report("RECREATE coherence FAILED: identical inputs and render size changed output");
                return 1;
            }
        }
        if (FILE* cycleFile = std::fopen((std::string(argv[2]) + ".cycle" + std::to_string(cycle)).c_str(), "wb"))
        {
            std::fwrite(cycleOutput.data(), 1, cycleOutput.size(), cycleFile);
            std::fclose(cycleFile);
        }
        // Also to the quality report file, which survives when Proton drops stdout.
        const char* reportPath = std::getenv("D4R_HARNESS_QUALITY_SCENE");
        FILE* report = reportPath != nullptr ? std::fopen(reportPath, "a") : nullptr;
        for (FILE* out : {stdout, report})
            if (out != nullptr)
                std::fprintf(out, "RECREATE cycle %d (%ux%u): process VRAM %llu KiB\n", cycle, width, height,
                             process_vram_kib());
        if (report != nullptr)
            std::fclose(report);
    }
    if (verifyRecreation && recreateCycles >= 4)
        lifecycle_report("RECREATE coherence: repeated render sizes match byte-for-byte");

    release(feature);
    shutdown();
    for (ID3D12Resource* guide : guides)
        if (guide) guide->Release();
    if (alphaInput) alphaInput->Release();
    if (alphaOutput) alphaOutput->Release();
    return 0;
}
