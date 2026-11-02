// Standalone FSR4 upscaling probe for instruction-mix comparisons with preset L.
// Uses the signed AMD runtime through its documented API. Refuses an analytical
// FSR2/3 provider. D3D12 timestamps bracket only dispatch; initialization, uploads
// and final readback are excluded. Saves probe.log and output.rgba16f in OUT_DIR.
// Usage: fsr_upscale_probe.exe UPSCALER_DLL OUT_DIR [RW RH OW OH FRAMES]
// Defaults: 1920 1080 3840 2160 140 (40 warmup + 100 measured evaluations).
// OUT_DIR must exist. For RGP, set MESA_VK_TRACE=rgp and
// MESA_VK_TRACE_PER_SUBMIT=1 on the Proton process and run only 2-3 frames;
// select a later submit, and never use traced durations as benchmark results.
// The shared FFX loader/descriptor helpers are provided by d4r_fsr_rr.h.
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include "../third_party/fidelityfx/upscalers/include/ffx_upscale.h"
#include "d4r_fsr_rr.h"
#include "d4r_fsr_rr_identity.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <string>
#include <vector>
#include <windows.h>
namespace
{
std::string narrow(const wchar_t *text)
{
    char buffer[1024] = {};
    WideCharToMultiByte(CP_UTF8, 0, text, -1, buffer, sizeof(buffer), nullptr, nullptr);
    return buffer;
}

void ffx_message(uint32_t type, const wchar_t *message)
{
    printf("ffx message %u: %s\n", type, narrow(message).c_str());
}
#define CHECK_HR(expr)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        const HRESULT hr_ = (expr);                                                                                    \
        if (FAILED(hr_))                                                                                               \
        {                                                                                                              \
            std::printf("%s failed: 0x%08lx (line %d)\n", #expr, static_cast<unsigned long>(hr_), __LINE__);           \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

struct Texture
{
    ID3D12Resource *resource = nullptr;
    D3D12_RESOURCE_DESC desc = {};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    UINT rows = 0;
    UINT64 rowBytes = 0, total = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

ID3D12Device *g_device = nullptr;

bool make_texture(Texture &texture, uint32_t width, uint32_t height, DXGI_FORMAT format, bool uav,
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

bool make_buffer(ID3D12Resource *&buffer, UINT64 bytes, D3D12_HEAP_TYPE type, uint8_t *&mapped)
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
    CHECK_HR(
        g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&buffer)));
    CHECK_HR(buffer->Map(0, nullptr, reinterpret_cast<void **>(&mapped)));
    return true;
}

void barrier(ID3D12GraphicsCommandList *list, Texture &texture, D3D12_RESOURCE_STATES after)
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
void upload(ID3D12GraphicsCommandList *list, Texture &texture, ID3D12Resource *staging, uint8_t *mapped, UINT64 &offset,
            const void *source, size_t texelBytes)
{
    offset =
        (offset + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) & ~UINT64(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = texture.footprint;
    footprint.Offset = offset;
    const size_t rowBytes = texelBytes * texture.desc.Width;
    for (UINT row = 0; row < texture.rows; ++row)
        std::memcpy(mapped + offset + row * footprint.Footprint.RowPitch,
                    static_cast<const uint8_t *>(source) + row * rowBytes, rowBytes);
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

void readback(ID3D12GraphicsCommandList *list, Texture &texture, ID3D12Resource *buffer, UINT64 offset)
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

} // namespace

int wmain(int argc, wchar_t **argv)
{
    if (argc < 3)
        return 2;
    const std::string logPath = narrow(argv[2]) + "/probe.log";
    if (!freopen(logPath.c_str(), "w", stdout))
        return 2;
    setvbuf(stdout, nullptr, _IONBF, 0);
    uint32_t rw = argc > 3 ? wcstoul(argv[3], 0, 10) : 1920, rh = argc > 4 ? wcstoul(argv[4], 0, 10) : 1080;
    uint32_t ow = argc > 5 ? wcstoul(argv[5], 0, 10) : 3840, oh = argc > 6 ? wcstoul(argv[6], 0, 10) : 2160;
    int frames = argc > 7 ? wcstol(argv[7], 0, 10) : 140;
    if (!rw || !rh || !ow || !oh || frames < 0)
        return 2;
    std::string out = narrow(argv[2]);
    IDXGIFactory4 *factory = nullptr;
    IDXGIAdapter1 *adapter = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumAdapters1(0, &adapter)))
        return 3;
    DXGI_ADAPTER_DESC1 ad = {};
    adapter->GetDesc1(&ad);
    printf("adapter=%s vendor=%04x device=%04x\n", narrow(ad.Description).c_str(), ad.VendorId, ad.DeviceId);
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g_device))))
        return 4;
    std::string detail;
    printf("driver_extension=%d ", d4r_fsr_rr_driver_extension(g_device, detail));
    puts(detail.c_str());
    FsrRrRuntime rt;
    if (!rt.load(argv[1]))
    {
        printf("load failed %lu\n", GetLastError());
        return 5;
    }
    printf("mapped=%s\n", narrow(rt.mappedPath.c_str()).c_str());
    uint64_t count = 0;
    ffxQueryDescGetVersions ver = {};
    ver.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    ver.createDescType = FFX_API_EFFECT_ID_UPSCALE;
    ver.device = g_device;
    ver.outputCount = &count;
    auto rc = rt.query(nullptr, &ver.header);
    printf("query=%u providers=%llu\n", rc, (unsigned long long)count);
    if (rc || !count)
        return 6;
    std::vector<uint64_t> ids(count);
    std::vector<const char *> names(count);
    ver.versionIds = ids.data();
    ver.versionNames = names.data();
    rc = rt.query(nullptr, &ver.header);
    if (rc)
        return 6;
    int selected = -1;
    for (size_t i = 0; i < count; i++)
    {
        printf("provider=%016llx %s\n", (unsigned long long)ids[i], names[i]);
        if (selected < 0 && names[i] && strncmp(names[i], "4.", 2) == 0)
            selected = (int)i;
    }
    if (selected < 0)
    {
        puts("REFUSING non-FSR4 provider");
        return 7;
    }
    rt.providerId = ids[selected];
    if (frames == 0)
        return 0;
    ffxCreateContextDescUpscale create = {};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    create.maxRenderSize = {rw, rh};
    create.maxUpscaleSize = {ow, oh};
    create.fpMessage = ffx_message;
    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = g_device;
    ffxOverrideVersion pin = {};
    pin.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    pin.versionId = ids[selected];
    create.header.pNext = &backend.header;
    backend.header.pNext = &pin.header;
    ffxContext ctx = nullptr;
    rc = rt.createContext(&ctx, &create.header, nullptr);
    printf("create=%u\n", rc);
    if (rc)
        return 8;
    const std::string actualProvider = fsr_rr_context_provider(rt, ctx);
    printf("actual_provider=%s\n", actualProvider.c_str());
    if (actualProvider.rfind("4.", 0) != 0)
    {
        puts("REFUSING a context without a verified FSR4 provider");
        return 7;
    }
    for (const wchar_t *name : {L"amdxc64.dll", L"amdxcffx64.dll", L"d3d12.dll", L"d3d12core.dll"})
    {
        if (HMODULE module = GetModuleHandleW(name))
        {
            wchar_t path[MAX_PATH] = {};
            GetModuleFileNameW(module, path, MAX_PATH);
            printf("module=%s path=%s\n", narrow(name).c_str(), narrow(path).c_str());
        }
    }
    ID3D12CommandQueue *q = nullptr;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandAllocator *alloc = nullptr;
    ID3D12GraphicsCommandList *cmd = nullptr;
    ID3D12Fence *fence = nullptr;
    if (FAILED(g_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&q))) ||
        FAILED(g_device->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&alloc))) ||
        FAILED(g_device->CreateCommandList(0, qd.Type, alloc, nullptr, IID_PPV_ARGS(&cmd))) ||
        FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return 9;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event)
        return 9;
    UINT64 seq = 0;
    cmd->Close();
    auto submit = [&]() {
        cmd->Close();
        ID3D12CommandList *lists[] = {cmd};
        q->ExecuteCommandLists(1, lists);
        q->Signal(fence, ++seq);
        fence->SetEventOnCompletion(seq, event);
        WaitForSingleObject(event, INFINITE);
    };
    Texture color, depth, motion, output;
    auto read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (!make_texture(color, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, false, read) ||
        !make_texture(depth, rw, rh, DXGI_FORMAT_R32_FLOAT, false, read) ||
        !make_texture(motion, rw, rh, DXGI_FORMAT_R16G16_FLOAT, false, read) ||
        !make_texture(output, ow, oh, DXGI_FORMAT_R16G16B16A16_FLOAT, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
        return 10;
    ID3D12Resource *stage = nullptr, *rb = nullptr, *trb = nullptr;
    uint8_t *sp = nullptr, *rp = nullptr, *tp = nullptr;
    if (!make_buffer(stage, color.total + depth.total + motion.total + 65536, D3D12_HEAP_TYPE_UPLOAD, sp) ||
        !make_buffer(rb, output.total, D3D12_HEAP_TYPE_READBACK, rp) ||
        !make_buffer(trb, 16, D3D12_HEAP_TYPE_READBACK, tp))
        return 10;
    std::vector<uint16_t> colors((size_t)rw * rh * 4), mvs((size_t)rw * rh * 2, 0);
    std::vector<float> ds((size_t)rw * rh, .5f);
    for (uint32_t y = 0; y < rh; y++)
        for (uint32_t x = 0; x < rw; x++)
        {
            size_t a = ((size_t)y * rw + x) * 4;
            colors[a] = 0x3400 + ((x / 8 + y / 8) % 8) * 128;
            colors[a + 1] = 0x3800;
            colors[a + 2] = 0x3000;
            colors[a + 3] = 0x3c00;
        }
    alloc->Reset();
    cmd->Reset(alloc, nullptr);
    UINT64 at = 0;
    upload(cmd, color, stage, sp, at, colors.data(), 8);
    upload(cmd, depth, stage, sp, at, ds.data(), 4);
    upload(cmd, motion, stage, sp, at, mvs.data(), 4);
    barrier(cmd, color, read);
    barrier(cmd, depth, read);
    barrier(cmd, motion, read);
    submit();
    ID3D12QueryHeap *queries = nullptr;
    D3D12_QUERY_HEAP_DESC qh = {};
    qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qh.Count = 2;
    if (FAILED(g_device->CreateQueryHeap(&qh, IID_PPV_ARGS(&queries))))
        return 11;
    UINT64 hz = 0;
    if (FAILED(q->GetTimestampFrequency(&hz)) || !hz)
        return 11;
    std::vector<double> times;
    const float jitter[4][2] = {{-.25f, -.25f}, {.25f, -.25f}, {.25f, .25f}, {-.25f, .25f}};
    for (int f = 0; f < frames; f++)
    {
        alloc->Reset();
        cmd->Reset(alloc, nullptr);
        barrier(cmd, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ffxDispatchDescUpscale d = {};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = cmd;
        d.color = fsr_rr_texture(color.resource, color.desc, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                                 FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.depth = fsr_rr_texture(depth.resource, depth.desc, FFX_API_SURFACE_FORMAT_R32_FLOAT,
                                 FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.motionVectors = fsr_rr_texture(motion.resource, motion.desc, FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
                                         FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.output = fsr_rr_texture(output.resource, output.desc, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                                  FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset = {jitter[f % 4][0], jitter[f % 4][1]};
        d.motionVectorScale = {(float)rw, (float)rh};
        d.renderSize = {rw, rh};
        d.upscaleSize = {ow, oh};
        d.frameTimeDelta = 16.6667f;
        d.preExposure = 1;
        d.reset = f == 0;
        d.cameraNear = .1f;
        d.cameraFar = 1000;
        d.cameraFovAngleVertical = 1;
        d.viewSpaceToMetersFactor = 1;
        cmd->EndQuery(queries, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        rc = rt.dispatch(&ctx, &d.header);
        if (rc)
        {
            printf("dispatch=%u frame=%d\n", rc, f);
            return 12;
        }
        cmd->EndQuery(queries, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        cmd->ResolveQueryData(queries, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, trb, 0);
        if (f == frames - 1)
            readback(cmd, output, rb, 0);
        submit();
        UINT64 t[2];
        memcpy(t, tp, 16);
        double us = double(t[1] - t[0]) * 1e6 / hz;
        printf("frame=%d gpu_us=%.3f\n", f, us);
        if (f >= 40)
            times.push_back(us);
    }
    FILE *raw = fopen((out + "/output.rgba16f").c_str(), "wb");
    if (!raw)
        return 14;
    size_t nonFinite = 0, zeroPixels = 0;
    for (uint32_t y = 0; y < oh; ++y)
    {
        const uint8_t *row = rp + y * output.footprint.Footprint.RowPitch;
        if (fwrite(row, 8, ow, raw) != ow)
            return 14;
        for (uint32_t x = 0; x < ow; ++x)
        {
            uint16_t rgba[4];
            memcpy(rgba, row + 8 * x, sizeof(rgba));
            bool zero = true;
            for (int k = 0; k < 3; ++k)
            {
                nonFinite += (rgba[k] & 0x7c00u) == 0x7c00u;
                zero &= (rgba[k] & 0x7fffu) == 0;
            }
            zeroPixels += zero;
        }
    }
    if (fclose(raw))
        return 14;
    printf("output_pixels=%zu nonfinite_rgb=%zu zero_rgb_pixels=%zu\n", size_t(ow) * oh, nonFinite, zeroPixels);
    if (!times.empty())
    {
        std::sort(times.begin(), times.end());
        const double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) * 0.5;
        printf("median_us=%.3f measured=%zu\n", median, times.size());
    }
    rt.destroyContext(&ctx, nullptr);
    return nonFinite || zeroPixels == size_t(ow) * oh ? 13 : 0;
}
