// Standalone check that AMD's FSR Ray Regeneration runtime runs on this D3D12 device, separate from
// any game and from the NGX shim. It:
//   1. creates a D3D12 device and reports the adapter identity the runtime will see through DXGI,
//   2. loads the given amd_fidelityfx_denoiser_dx12.dll (unmodified, by path) and enumerates its
//      denoiser providers against that device (FFX_API_QUERY_DESC_TYPE_GET_VERSIONS),
//   3. creates a context (this is where the runtime builds its shader pipelines) and asks it which
//      provider it got,
//   4. dispatches a synthetic ray-traced scene with per-frame white noise: a static camera, then a
//      moving one (with exact 2.5D motion vectors), then a camera cut with FFX_DENOISER_DISPATCH_RESET,
//   5. reads every output back and compares input and output against the noise-free signals, so a
//      passthrough or a stale history cannot pass for denoising.
// Every stage prints one "stage=... result=..." line, and OUT_DIR/probe.json collects the lot.
//
// usage: fsr_rr_probe.exe DENOISER_DLL OUT_DIR [--size WxH] [--frames N] [--signals HEX] [--query-only]
//                         [--no-driver-extension]
//   --no-driver-extension: do not load Proton's amdxc64.dll for an AMD GPU hidden as NVIDIA
//                          (see d4r_fsr_rr_identity.h); reproduces the "no provider" outcome
//   --signals: FfxApiDenoiserSignalFlags of the radiance signals to run (default 0x36: direct and
//              indirect diffuse and specular, AMD's four-signal configuration; 0x30 = indirect only)
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

#include "d4r_fsr_rr.h"
#include "windows/fsr_rr_identity.h"
#include "fsr_rr_scene.h"

// Per-dispatch GPU timing of AMD's denoiser (D4R_FSR_RR_DISPATCH_PROFILE=1), shared with the Linux shim.
inline void logf(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
}
inline std::string env_string(const char* name) { const char* v = std::getenv(name); return v ? v : ""; }
inline uint32_t env_uint(const char* name, uint32_t fallback)
{
    const char* v = std::getenv(name);
    return v && *v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : fallback;
}
inline uint32_t profile_log_every() { return env_uint("D4R_PROFILE_LOG_EVERY", 1000000); }
#include "d4r_fsr_rr_dispatch_profile.h"

namespace
{
using namespace fsr_scene;

FILE* g_json = nullptr;
bool g_jsonFirst = true;

void json_raw(const char* format, ...)
{
    if (g_json == nullptr)
        return;
    va_list args;
    va_start(args, format);
    std::vfprintf(g_json, format, args);
    va_end(args);
}

// One top-level "key": value pair; value is preformatted JSON.
void json_field(const char* key, const std::string& value)
{
    json_raw("%s\n  \"%s\": %s", g_jsonFirst ? "" : ",", key, value.c_str());
    g_jsonFirst = false;
}

std::string quoted(const std::string& text)
{
    std::string out = "\"";
    for (char c : text)
    {
        if (c == '"' || c == '\\')
            out += '\\';
        if (static_cast<unsigned char>(c) < 0x20)
            continue;
        out += c;
    }
    return out + "\"";
}

std::string narrow(const wchar_t* text)
{
    char buffer[1024] = {};
    WideCharToMultiByte(CP_UTF8, 0, text, -1, buffer, sizeof(buffer), nullptr, nullptr);
    return buffer;
}

std::string format(const char* pattern, ...)
{
    char buffer[2048];
    va_list args;
    va_start(args, pattern);
    std::vsnprintf(buffer, sizeof(buffer), pattern, args);
    va_end(args);
    return buffer;
}

void stage(const char* name, bool ok, const std::string& detail)
{
    std::printf("stage=%s result=%s %s\n", name, ok ? "ok" : "FAIL", detail.c_str());
    std::fflush(stdout);
    json_field((std::string("stage_") + name).c_str(),
               format("{\"ok\": %s, \"detail\": %s}", ok ? "true" : "false", quoted(detail).c_str()));
}

std::vector<std::string> g_messages;

void ffx_message(uint32_t type, const wchar_t* message)
{
    const std::string text = narrow(message);
    std::printf("ffx %s: %s\n", type == FFX_API_MESSAGE_TYPE_WARNING ? "warning" : "error", text.c_str());
    std::fflush(stdout);
    if (g_messages.size() < 64)
        g_messages.push_back(text);
}

// --- D3D12 plumbing -------------------------------------------------------------------

#define CHECK_HR(expr)                                                                                               \
    do                                                                                                               \
    {                                                                                                                \
        const HRESULT hr_ = (expr);                                                                                  \
        if (FAILED(hr_))                                                                                             \
        {                                                                                                            \
            std::printf("%s failed: 0x%08lx (line %d)\n", #expr, static_cast<unsigned long>(hr_), __LINE__);         \
            return false;                                                                                            \
        }                                                                                                            \
    } while (0)

struct Texture
{
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_DESC desc = {};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    UINT rows = 0;
    UINT64 rowBytes = 0, total = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

ID3D12Device* g_device = nullptr;

bool make_texture(Texture& texture, uint32_t width, uint32_t height, DXGI_FORMAT format, bool uav,
                  D3D12_RESOURCE_STATES state)
{
    texture.desc = {};
    texture.desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.desc.Width = width;
    texture.desc.Height = height;
    texture.desc.DepthOrArraySize = 1;
    texture.desc.MipLevels = 1;
    texture.desc.Format = format;
    texture.desc.SampleDesc.Count = 1;
    texture.desc.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    CHECK_HR(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture.desc, state, nullptr,
                                               IID_PPV_ARGS(&texture.resource)));
    texture.state = state;
    g_device->GetCopyableFootprints(&texture.desc, 0, 1, 0, &texture.footprint, &texture.rows, &texture.rowBytes,
                                    &texture.total);
    return true;
}

bool make_buffer(ID3D12Resource*& buffer, UINT64 bytes, D3D12_HEAP_TYPE type, uint8_t*& mapped)
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
    const D3D12_RESOURCE_STATES state =
        type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST;
    CHECK_HR(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&buffer)));
    CHECK_HR(buffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped)));
    return true;
}

void barrier(ID3D12GraphicsCommandList* list, Texture& texture, D3D12_RESOURCE_STATES after)
{
    if (texture.state == after)
        return;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = texture.resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = texture.state;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
    texture.state = after;
}

// Copies `texelBytes`-sized texels from `source` (tightly packed rows) into the upload buffer at
// `offset` with the texture's footprint, and records the copy.
void upload(ID3D12GraphicsCommandList* list, Texture& texture, ID3D12Resource* staging, uint8_t* mapped,
            UINT64& offset, const void* source, size_t texelBytes)
{
    offset = (offset + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) & ~UINT64(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = texture.footprint;
    footprint.Offset = offset;
    const size_t rowBytes = texelBytes * texture.desc.Width;
    for (UINT row = 0; row < texture.rows; ++row)
        std::memcpy(mapped + offset + row * footprint.Footprint.RowPitch,
                    static_cast<const uint8_t*>(source) + row * rowBytes, rowBytes);
    offset += texture.total;
    barrier(list, texture, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = texture.resource;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = staging;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

void readback(ID3D12GraphicsCommandList* list, Texture& texture, ID3D12Resource* buffer, UINT64 offset)
{
    barrier(list, texture, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = texture.footprint;
    footprint.Offset = offset;
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = buffer;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = texture.resource;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

// --- metrics and captures -------------------------------------------------------------

struct Error
{
    double input = 0, output = 0, outputVsPrevious = 0, change = 0;
    size_t nonFinite = 0;
};

Error measure(const Frame& frame, const Frame* previous, const uint8_t* rows, UINT pitch, uint32_t width,
              uint32_t height, int signal)
{
    Error e;
    size_t n = 0;
    for (uint32_t y = 2; y + 2 < height; ++y)
        for (uint32_t x = 2; x + 2 < width; ++x)
        {
            const size_t i = static_cast<size_t>(y) * width + x;
            const uint16_t* out = reinterpret_cast<const uint16_t*>(rows + static_cast<size_t>(y) * pitch) + x * 4;
            for (int c = 0; c < 3; ++c)
            {
                const float o = from_half(out[c]);
                if (!std::isfinite(o))
                {
                    ++e.nonFinite;
                    continue;
                }
                if (!frame.valid[i])
                    continue;
                const float clean = frame.clean[signal][i * 3 + c];
                const float in = from_half(frame.noisy[signal][i * 4 + c]);
                e.input += (in - clean) * (in - clean);
                e.output += (o - clean) * (o - clean);
                e.change += std::fabs(o - in);
                if (previous != nullptr && previous->valid[i])
                {
                    const float old = previous->clean[signal][i * 3 + c];
                    e.outputVsPrevious += (o - old) * (o - old);
                }
                ++n;
            }
        }
    if (n > 0)
    {
        e.input = std::sqrt(e.input / n);
        e.output = std::sqrt(e.output / n);
        e.outputVsPrevious = std::sqrt(e.outputVsPrevious / n);
        e.change /= n;
    }
    return e;
}

// Tonemapped composite (diffuse signals x diffuse albedo + specular signals x specular albedo).
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
                const float mapped = std::pow(std::max(0.0f, rgb[k]) / (1.0f + std::max(0.0f, rgb[k])), 1.0f / 2.2f);
                row[x * 3 + k] = static_cast<uint8_t>(std::lround(saturate(mapped) * 255.0f));
            }
        }
        std::fwrite(row.data(), 1, row.size(), file);
    }
    std::fclose(file);
}

} // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc < 3)
    {
        std::printf("usage: fsr_rr_probe.exe DENOISER_DLL OUT_DIR [--size WxH] [--frames N] [--signals HEX] [--query-only]\n");
        return 2;
    }
    const std::wstring dllPath = argv[1];
    const std::string outDir = narrow(argv[2]);
    uint32_t width = 960, height = 540, frames = 48, signals = 0x36;
    bool queryOnly = false, driverExtension = true;
    for (int i = 3; i < argc; ++i)
    {
        const std::wstring arg = argv[i];
        if (arg == L"--size" && i + 1 < argc)
            std::swscanf(argv[++i], L"%ux%u", &width, &height);
        else if (arg == L"--frames" && i + 1 < argc)
            frames = static_cast<uint32_t>(std::wcstoul(argv[++i], nullptr, 0));
        else if (arg == L"--signals" && i + 1 < argc)
            signals = static_cast<uint32_t>(std::wcstoul(argv[++i], nullptr, 16));
        else if (arg == L"--query-only")
            queryOnly = true;
        else if (arg == L"--no-driver-extension")
            driverExtension = false;
    }
    signals &= 0x36;
    if (signals == 0 || frames < 12)
    {
        std::printf("need at least one radiance signal and 12 frames\n");
        return 2;
    }
    CreateDirectoryA(outDir.c_str(), nullptr);
    g_json = std::fopen((outDir + "\\probe.json").c_str(), "w");
    json_raw("{");
    json_field("denoiser_dll", quoted(narrow(dllPath.c_str())));
    json_field("render_size", format("[%u, %u]", width, height));
    json_field("signal_flags", format("%u", signals));
    json_field("frames", format("%u", frames));
    int exitCode = 1;
    auto finish = [&](int code) {
        json_raw("\n}\n");
        if (g_json)
            std::fclose(g_json);
        g_json = nullptr;
        return code;
    };

    // 1. Device, and the adapter identity DXGI reports (what the runtime can see).
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumAdapters1(0, &adapter)))
    {
        stage("device", false, "no DXGI adapter");
        return finish(1);
    }
    DXGI_ADAPTER_DESC1 adapterDesc = {};
    adapter->GetDesc1(&adapterDesc);
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g_device))))
    {
        stage("device", false, "D3D12CreateDevice failed");
        return finish(1);
    }
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = {D3D_SHADER_MODEL_6_6};
    g_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel));
    stage("device", true,
          format("adapter=\"%s\" vendor=0x%04x device=0x%04x luid=%08lx:%08lx shader_model=0x%x",
                 narrow(adapterDesc.Description).c_str(), adapterDesc.VendorId, adapterDesc.DeviceId,
                 static_cast<unsigned long>(adapterDesc.AdapterLuid.HighPart),
                 static_cast<unsigned long>(adapterDesc.AdapterLuid.LowPart),
                 static_cast<unsigned>(shaderModel.HighestShaderModel)));
    json_field("adapter", format("{\"name\": %s, \"vendor\": %u, \"device\": %u}",
                                 quoted(narrow(adapterDesc.Description)).c_str(), adapterDesc.VendorId,
                                 adapterDesc.DeviceId));

    // 2. Runtime and provider.
    FsrRrRuntime runtime;
    if (!runtime.load(dllPath))
    {
        stage("load", false, format("%s (error %lu)", fsr_rr_stage_name(runtime.stage), GetLastError()));
        return finish(1);
    }
    stage("load", true, "mapped=" + narrow(runtime.mappedPath.c_str()));
    if (driverExtension)
    {
        std::string detail;
        const bool loaded = d4r_fsr_rr_driver_extension(g_device, detail);
        stage("driver_extension", loaded, detail);
    }
    const ffxReturnCode_t debug = runtime.set_message_callback(&ffx_message, FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE);
    std::printf("ffx global debug callback: %s\n", fsr_rr_return_name(debug));
    const bool provider = runtime.query_providers(g_device);
    {
        std::string list = "[";
        for (size_t i = 0; i < runtime.versionIds.size(); ++i)
            list += format("%s{\"id\": \"0x%016llx\", \"name\": %s}", i ? ", " : "",
                           static_cast<unsigned long long>(runtime.versionIds[i]),
                           quoted(runtime.versionNames[i]).c_str());
        json_field("providers", list + "]");
    }
    stage("provider", provider,
          provider ? format("count=%zu selected=0x%016llx name=\"%s\"", runtime.versionIds.size(),
                            static_cast<unsigned long long>(runtime.providerId), runtime.providerName.c_str())
                   : format("%s (query %s)", fsr_rr_stage_name(runtime.stage), fsr_rr_return_name(runtime.queryResult)));
    if (!provider || queryOnly)
        return finish(provider ? 0 : 1);

    FfxApiEffectMemoryUsage memory = {};
    const ffxReturnCode_t memoryResult = fsr_rr_query_memory(runtime, g_device, width, height, signals, 0, memory);
    json_field("gpu_memory", format("{\"result\": %u, \"total_mb\": %.2f, \"aliasable_mb\": %.2f}", memoryResult,
                                    memory.totalUsageInBytes / 1048576.0, memory.aliasableUsageInBytes / 1048576.0));
    std::printf("gpu memory query: %s total=%.2f MB aliasable=%.2f MB\n", fsr_rr_return_name(memoryResult),
                memory.totalUsageInBytes / 1048576.0, memory.aliasableUsageInBytes / 1048576.0);

    // 3. Context (shader pipelines are created here: the slow and the most likely to fail part).
    ffxContext context = nullptr;
    const auto createStart = std::chrono::steady_clock::now();
    fsr_rr_dp::hook_device(g_device);
    const ffxReturnCode_t created = fsr_rr_create_context(runtime, g_device, width, height, signals, 0, context);
    const double createMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - createStart).count();
    json_field("context_create_ms", format("%.1f", createMs));
    if (created != FFX_API_RETURN_OK)
    {
        stage("context", false, format("ffxCreateContext -> %s after %.1f ms", fsr_rr_return_name(created), createMs));
        return finish(1);
    }
    uint64_t actualId = 0;
    const std::string actual = fsr_rr_context_provider(runtime, context, &actualId);
    stage("context", true, format("%.1f ms, provider in use 0x%016llx \"%s\"", createMs,
                                  static_cast<unsigned long long>(actualId), actual.c_str()));
    json_field("provider_in_use", quoted(actual));

    // 4. Resources and frames.
    ID3D12CommandQueue* queue = nullptr;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    if (FAILED(g_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue))) ||
        FAILED(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
    {
        stage("dispatch", false, "command objects");
        return finish(1);
    }
    list->Close();
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    uint64_t fenceValue = 0;
    const D3D12_RESOURCE_STATES readState =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    Texture depth, motion, normals, diffuseAlbedo, specularAlbedo, input[kSignalCount], output[kSignalCount];
    bool ok = make_texture(depth, width, height, DXGI_FORMAT_R32_FLOAT, false, readState) &&
              make_texture(motion, width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, false, readState) &&
              make_texture(normals, width, height, DXGI_FORMAT_R10G10B10A2_UNORM, false, readState) &&
              make_texture(diffuseAlbedo, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, false, readState) &&
              make_texture(specularAlbedo, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, false, readState);
    for (int s = 0; s < kSignalCount && ok; ++s)
        ok = make_texture(input[s], width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, false, readState) &&
             make_texture(output[s], width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, true,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource *staging = nullptr, *readbackBuffer = nullptr, *timestampBuffer = nullptr;
    uint8_t *stagingMapped = nullptr, *readbackMapped = nullptr, *timestampMapped = nullptr;
    const UINT64 planeBytes = output[0].total;
    ok = ok && make_buffer(staging, 16 * (planeBytes + 65536), D3D12_HEAP_TYPE_UPLOAD, stagingMapped) &&
         make_buffer(readbackBuffer, kSignalCount * planeBytes, D3D12_HEAP_TYPE_READBACK, readbackMapped) &&
         make_buffer(timestampBuffer, 16, D3D12_HEAP_TYPE_READBACK, timestampMapped);
    ID3D12QueryHeap* timestamps = nullptr;
    D3D12_QUERY_HEAP_DESC queryDesc = {};
    queryDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    queryDesc.Count = 2;
    ok = ok && SUCCEEDED(g_device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&timestamps)));
    UINT64 ticksPerSecond = 0;
    queue->GetTimestampFrequency(&ticksPerSecond);
    if (!ok)
    {
        stage("dispatch", false, "resource creation");
        return finish(1);
    }

    const float aspect = static_cast<float>(width) / height;
    const M4 projection = projection_matrix(1.0f, aspect, 0.1f, 1000.0f);
    const uint32_t cutFrame = frames * 2 / 3, moveStart = frames / 3;
    auto camera_at = [&](uint32_t f) -> Camera {
        if (f >= cutFrame)
            return {{3.5f, 1.6f, 1.5f}, {-0.5f, -0.6f, 7.5f}};
        const float t = f >= moveStart ? static_cast<float>(f - moveStart) : 0.0f;
        return {{-0.6f + 0.04f * t, 0.6f, -1.5f}, {0.0f + 0.02f * t, -0.1f, 6.5f}};
    };

    Frame frameData, previousFrame;
    std::vector<double> gpuMs, cpuMs;
    std::string frameLog = "[";
    bool dispatchOk = true, converged = true, resetOk = true, finite = true, notPassthrough = true;
    Camera previousCamera = camera_at(0);
    for (uint32_t f = 0; f < frames; ++f)
    {
        const Camera camera = camera_at(f);
        const bool reset = f == 0 || f == cutFrame;
        std::swap(previousFrame, frameData);
        render(frameData, width, height, camera, reset ? camera : previousCamera, projection, f);

        allocator->Reset();
        list->Reset(allocator, nullptr);
        UINT64 offset = 0;
        upload(list, depth, staging, stagingMapped, offset, frameData.depth.data(), 4);
        upload(list, motion, staging, stagingMapped, offset, frameData.motion.data(), 8);
        upload(list, normals, staging, stagingMapped, offset, frameData.normals.data(), 4);
        upload(list, diffuseAlbedo, staging, stagingMapped, offset, frameData.diffuseAlbedo.data(), 4);
        upload(list, specularAlbedo, staging, stagingMapped, offset, frameData.specularAlbedo.data(), 4);
        for (int s = 0; s < kSignalCount; ++s)
            if (signals & kSignalFlags[s])
                upload(list, input[s], staging, stagingMapped, offset, frameData.noisy[s].data(), 8);
        Texture* all[] = {&depth, &motion, &normals, &diffuseAlbedo, &specularAlbedo};
        for (Texture* t : all)
            barrier(list, *t, readState);
        for (int s = 0; s < kSignalCount; ++s)
        {
            barrier(list, input[s], readState);
            barrier(list, output[s], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        ffxDispatchDescDenoiser desc = {};
        desc.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER;
        desc.commandList = list;
        const uint32_t readFfx = FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ;
        desc.linearDepth = fsr_rr_texture(depth.resource, depth.desc, FFX_API_SURFACE_FORMAT_R32_FLOAT, readFfx);
        desc.motionVectors = fsr_rr_texture(motion.resource, motion.desc, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, readFfx);
        desc.normals = fsr_rr_texture(normals.resource, normals.desc, FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM, readFfx);
        desc.specularAlbedo = fsr_rr_texture(specularAlbedo.resource, specularAlbedo.desc, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, readFfx);
        desc.diffuseAlbedo = fsr_rr_texture(diffuseAlbedo.resource, diffuseAlbedo.desc, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, readFfx);
        desc.motionVectorScale = {1.0f, 1.0f, 1.0f};
        desc.jitterOffsets = {0.0f, 0.0f};
        const V3 delta = previousCamera.eye - camera.eye;
        desc.cameraPositionDelta = reset ? FfxApiFloatCoords3D{0, 0, 0} : FfxApiFloatCoords3D{delta.x, delta.y, delta.z};
        const M4 view = view_matrix(camera.eye, camera.target);
        std::memcpy(&desc.view, &view, sizeof(desc.view));
        std::memcpy(&desc.projection, &projection, sizeof(desc.projection));
        desc.linearDepthBounds = {0.0f, 1000.0f};
        desc.renderSize = {width, height};
        desc.frameIndex = f;
        desc.flags = reset ? FFX_DENOISER_DISPATCH_RESET : 0;
        ffxDispatchDescDenoiserDirectDiffuse dd = {};
        ffxDispatchDescDenoiserDirectSpecular ds = {};
        ffxDispatchDescDenoiserIndirectDiffuse id = {};
        ffxDispatchDescDenoiserIndirectSpecular is = {};
        dd.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_DIFFUSE;
        ds.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_SPECULAR;
        id.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_INDIRECT_DIFFUSE;
        is.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_INDIRECT_SPECULAR;
        FfxApiDenoiserSignal* signalDescs[kSignalCount] = {&dd.signal, &ds.signal, &id.signal, &is.signal};
        ffxApiHeader* headers[kSignalCount] = {&dd.header, &ds.header, &id.header, &is.header};
        ffxApiHeader* tail = &desc.header;
        for (int s = 0; s < kSignalCount; ++s)
        {
            if (!(signals & kSignalFlags[s]))
                continue;
            signalDescs[s]->input = fsr_rr_texture(input[s].resource, input[s].desc, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, readFfx);
            signalDescs[s]->output = fsr_rr_texture(output[s].resource, output[s].desc, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                                                    FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            tail->pNext = headers[s];
            tail = headers[s];
        }
        list->EndQuery(timestamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        const auto cpuStart = std::chrono::steady_clock::now();
        fsr_rr_dp::begin(g_device, list, f);
        const ffxReturnCode_t dispatched = runtime.dispatch(&context, &desc.header);
        fsr_rr_dp::end(list);
        cpuMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count());
        list->EndQuery(timestamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        list->ResolveQueryData(timestamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, timestampBuffer, 0);
        if (dispatched != FFX_API_RETURN_OK)
        {
            stage("dispatch", false, format("frame %u: ffxDispatch -> %s", f, fsr_rr_return_name(dispatched)));
            dispatchOk = false;
            list->Close();
            break;
        }
        for (int s = 0; s < kSignalCount; ++s)
            if (signals & kSignalFlags[s])
                readback(list, output[s], readbackBuffer, s * planeBytes);
        if (FAILED(list->Close()))
        {
            stage("dispatch", false, format("frame %u: Close failed", f));
            dispatchOk = false;
            break;
        }
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, ++fenceValue);
        fence->SetEventOnCompletion(fenceValue, event);
        if (WaitForSingleObject(event, 20000) != WAIT_OBJECT_0)
        {
            stage("dispatch", false, format("frame %u: GPU did not finish within 20 s", f));
            dispatchOk = false;
            break;
        }
        const HRESULT removed = g_device->GetDeviceRemovedReason();
        if (FAILED(removed))
        {
            stage("dispatch", false, format("frame %u: device removed 0x%08lx", f, static_cast<unsigned long>(removed)));
            dispatchOk = false;
            break;
        }
        uint64_t stamps[2];
        std::memcpy(stamps, timestampMapped, sizeof(stamps));
        const double gpu = ticksPerSecond ? (stamps[1] - stamps[0]) * 1000.0 / ticksPerSecond : -1.0;
        gpuMs.push_back(gpu);

        // 5. Compare against the clean signals.
        frameLog += format("%s\n    {\"frame\": %u, \"reset\": %s, \"gpu_ms\": %.3f, \"cpu_ms\": %.3f, \"signals\": {",
                           f ? "," : "", f, reset ? "true" : "false", gpu, cpuMs.back());
        bool firstSignal = true;
        std::vector<V3> composeOut(static_cast<size_t>(width) * height), composeIn(composeOut.size()),
            composeClean(composeOut.size());
        for (int s = 0; s < kSignalCount; ++s)
        {
            if (!(signals & kSignalFlags[s]))
                continue;
            const uint8_t* rows = readbackMapped + s * planeBytes;
            const Error e = measure(frameData, f == cutFrame ? &previousFrame : nullptr, rows,
                                    output[s].footprint.Footprint.RowPitch, width, height, s);
            frameLog += format("%s\"%s\": {\"rmse_in\": %.5f, \"rmse_out\": %.5f, \"mean_abs_change\": %.5f, "
                               "\"rmse_out_vs_previous_scene\": %.5f, \"nonfinite\": %zu}",
                               firstSignal ? "" : ", ", kSignalNames[s], e.input, e.output, e.change,
                               e.outputVsPrevious, e.nonFinite);
            firstSignal = false;
            finite = finite && e.nonFinite == 0;
            notPassthrough = notPassthrough && (f < 2 || e.change > 1e-3 * (e.input + 1e-6));
            // Converged before the cut: at least 8 static or slowly moving frames into history.
            if (f + 1 == cutFrame && !(e.output < 0.5 * e.input))
                converged = false;
            if (f == cutFrame && !(e.output < e.outputVsPrevious))
                resetOk = false;
            if (f + 1 == frames && !(e.output < 0.6 * e.input))
                resetOk = false;
            const bool diffuse = s == kDirectDiffuse || s == kIndirectDiffuse;
            for (uint32_t y = 0; y < height; ++y)
                for (uint32_t x = 0; x < width; ++x)
                {
                    const size_t i = static_cast<size_t>(y) * width + x;
                    const V3 albedo = diffuse ? frameData.diffuseLinear[i] : frameData.specularLinear[i];
                    const uint16_t* o = reinterpret_cast<const uint16_t*>(rows + static_cast<size_t>(y) * output[s].footprint.Footprint.RowPitch) + x * 4;
                    composeOut[i] = composeOut[i] + mul(V3{from_half(o[0]), from_half(o[1]), from_half(o[2])}, albedo);
                    composeIn[i] = composeIn[i] + mul(V3{from_half(frameData.noisy[s][i * 4]), from_half(frameData.noisy[s][i * 4 + 1]),
                                                         from_half(frameData.noisy[s][i * 4 + 2])}, albedo);
                    composeClean[i] = composeClean[i] + mul(V3{frameData.clean[s][i * 3], frameData.clean[s][i * 3 + 1],
                                                               frameData.clean[s][i * 3 + 2]}, albedo);
                }
        }
        frameLog += "}}";
        if (f == 0 || f + 1 == moveStart || f + 1 == cutFrame || f == cutFrame || f == cutFrame + 1 || f + 1 == frames)
        {
            write_ppm(format("%s\\frame%03u_noisy.ppm", outDir.c_str(), f), width, height, composeIn);
            write_ppm(format("%s\\frame%03u_denoised.ppm", outDir.c_str(), f), width, height, composeOut);
            write_ppm(format("%s\\frame%03u_clean.ppm", outDir.c_str(), f), width, height, composeClean);
        }
        previousCamera = camera;
    }
    if (fsr_rr_dp::enabled())
        fsr_rr_dp::log_stats();
    json_field("frame_log", frameLog +"\n  ]");
    if (dispatchOk)
    {
        std::vector<double> warm(gpuMs.begin() + std::min<size_t>(gpuMs.size(), 3), gpuMs.end());
        std::sort(warm.begin(), warm.end());
        const double median = warm.empty() ? -1 : warm[warm.size() / 2];
        std::vector<double> cpuWarm(cpuMs.begin() + std::min<size_t>(cpuMs.size(), 3), cpuMs.end());
        std::sort(cpuWarm.begin(), cpuWarm.end());
        stage("dispatch", true, format("%u frames, GPU median %.3f ms (first frame %.3f ms), CPU record median %.3f ms",
                                       frames, median, gpuMs.empty() ? -1.0 : gpuMs[0],
                                       cpuWarm.empty() ? -1.0 : cpuWarm[cpuWarm.size() / 2]));
        json_field("gpu_ms_median", format("%.4f", median));
        stage("output_finite", finite, finite ? "no NaN/Inf in any output" : "non-finite output values");
        stage("not_passthrough", notPassthrough, "outputs differ from the noisy inputs");
        stage("denoised", converged, format("before the cut (frame %u) every signal's RMSE vs clean is < 0.5x the input's",
                                            cutFrame - 1));
        stage("reset", resetOk, format("frame %u (reset) is closer to the new scene than the old one; final frame < 0.6x input",
                                       cutFrame));
        exitCode = finite && notPassthrough && converged && resetOk ? 0 : 1;
    }
    std::string messages = "[";
    for (size_t i = 0; i < g_messages.size(); ++i)
        messages += (i ? ", " : "") + quoted(g_messages[i]);
    json_field("ffx_messages", messages + "]");
    runtime.destroyContext(&context, nullptr);
    return finish(exitCode);
}
