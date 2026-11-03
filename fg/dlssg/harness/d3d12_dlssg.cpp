// d3d12_dlssg: drives NVIDIA's DLSS Frame Generation through the D3D12 NGX path under Proton, the way Streamline
// does in a game: an NGX core (our nvngx.dll_dlssg_host.dll in standalone mode) hosting nvngx_dlssg.dll, a D3D12
// device from vkd3d-proton, dxvk-nvapi's cubin calls, and the nvxrt Vulkan layer executing the kernels.
// Synthetic scene as in the native harness (dlssg_harness.cpp): a gradient square moving over a panning checkerboard.
//
// usage: d3d12_dlssg.exe HOST_DLL OUT_DIR [--w W --h H --rw RW --rh RH --frames N --mfc K --bench --direct|--compute]
//        [--replay DIR [--replay-ui N] [--replay-repeat N] [--replay-pipe N]] [--lite] [--copyout]
//   writes OUT_DIR\f<N>_g<k>.ppm (generated), f<N>_real.ppm, f<N>_gt<k>.ppm (ground truth) for the last frame
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using NgxResult = unsigned int;
struct NgxHandle { unsigned int Id; };
static bool failed(NgxResult r) { return (r & 0xFFF00000u) == 0xBAD00000u; }

// MSVC layout of NVSDK_NGX_Parameter (see dlssg_host.cpp)
namespace ms {
template <typename F> F slot(void* obj, int i) { return reinterpret_cast<F>((*reinterpret_cast<void***>(obj))[i]); }
void setPtr(void* p, const char* n, void* v) { slot<void (*)(void*, const char*, void*)>(p, 0)(p, n, v); }
void setI(void* p, const char* n, int v) { slot<void (*)(void*, const char*, int)>(p, 3)(p, n, v); }
void setU(void* p, const char* n, unsigned v) { slot<void (*)(void*, const char*, unsigned)>(p, 4)(p, n, v); }
void setF(void* p, const char* n, float v) { slot<void (*)(void*, const char*, float)>(p, 6)(p, n, v); }
NgxResult getI(void* p, const char* n, int* v) { return slot<NgxResult (*)(void*, const char*, int*)>(p, 11)(p, n, v); }
NgxResult getU(void* p, const char* n, unsigned* v) { return slot<NgxResult (*)(void*, const char*, unsigned*)>(p, 12)(p, n, v); }
} // namespace ms

#define CHK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { fprintf(stderr, "%s failed: 0x%08lx (line %d)\n", #x, (unsigned long)hr_, __LINE__); exit(1); } } while (0)

static float g_pan = 6.0f, g_sqvx = 23.0f, g_sqvy = 11.0f;
struct Scene {
    uint32_t W, H, RW, RH;
    float pan = g_pan, sqx0 = 400, sqy0 = 300, sqvx = g_sqvx, sqvy = g_sqvy, sq = 160;
    void square_pos(float t, float& x, float& y) const { x = sqx0 + sqvx * t; y = sqy0 + sqvy * t; }
    void shade(float t, float px, float py, uint8_t* o) const
    {
        float sx, sy;
        square_pos(t, sx, sy);
        if (px >= sx && px < sx + sq && py >= sy && py < sy + sq) {
            o[0] = 230; o[1] = (uint8_t)(40 + (px - sx) / sq * 180); o[2] = (uint8_t)(40 + (py - sy) / sq * 180); o[3] = 255;
            return;
        }
        float bx = px + pan * t;
        int cx = (int)floorf(bx / 64.f), cy = (int)floorf(py / 64.f);
        uint8_t v = ((cx + cy) & 1) ? 200 : 60;
        o[0] = v; o[1] = (uint8_t)(v * 0.9f); o[2] = (uint8_t)(v * 0.7f); o[3] = 255;
        if (((int)floorf(bx) & 15) == 0) { o[0] = 255; o[1] = 255; o[2] = 255; }
    }
    void render(float t, std::vector<uint8_t>& rgba) const
    {
        rgba.resize((size_t)W * H * 4);
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++) shade(t, x + 0.5f, y + 0.5f, &rgba[4 * ((size_t)y * W + x)]);
    }
    void aux(float t, std::vector<float>& depth, std::vector<float>& mv) const
    {
        depth.resize((size_t)RW * RH);
        mv.resize((size_t)RW * RH * 2);
        float sx = (float)W / RW, sy = (float)H / RH, qx, qy;
        square_pos(t, qx, qy);
        for (uint32_t y = 0; y < RH; y++)
            for (uint32_t x = 0; x < RW; x++) {
                float px = (x + 0.5f) * sx, py = (y + 0.5f) * sy;
                bool insq = px >= qx && px < qx + sq && py >= qy && py < qy + sq;
                size_t i = (size_t)y * RW + x;
                depth[i] = insq ? 0.1f : 0.9f;
                float dx = insq ? -sqvx : pan, dy = insq ? -sqvy : 0.0f;
                mv[2 * i] = dx / sx;
                mv[2 * i + 1] = dy / sy;
            }
    }
};

static void write_ppm(const std::string& path, const uint8_t* rgba, uint32_t W, uint32_t H, uint32_t pitch)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    fprintf(f, "P6\n%u %u\n255\n", W, H);
    std::vector<uint8_t> row(W * 3);
    for (uint32_t y = 0; y < H; y++) {
        const uint8_t* s = rgba + (size_t)y * pitch;
        for (uint32_t x = 0; x < W; x++) { row[3 * x] = s[4 * x]; row[3 * x + 1] = s[4 * x + 1]; row[3 * x + 2] = s[4 * x + 2]; }
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
}

struct Tex {
    ID3D12Resource* res = nullptr;
    DXGI_FORMAT fmt;
    uint32_t w, h, bpp;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total = 0;
    D3D12_RESOURCE_STATES state;
};

static Tex make_tex(ID3D12Device* dev, uint32_t w, uint32_t h, DXGI_FORMAT fmt, uint32_t bpp, bool uav,
                    D3D12_RESOURCE_STATES initial)
{
    Tex t;
    t.fmt = fmt; t.w = w; t.h = h; t.bpp = bpp; t.state = initial;
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w; d.Height = h; d.DepthOrArraySize = 1; d.MipLevels = 1; d.Format = fmt;
    d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    CHK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, initial, nullptr, IID_ID3D12Resource, (void**)&t.res));
    UINT rows; UINT64 rowsize;
    dev->GetCopyableFootprints(&d, 0, 1, 0, &t.fp, &rows, &rowsize, &t.total);
    return t;
}

static ID3D12Resource* make_buffer(ID3D12Device* dev, UINT64 size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES st)
{
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = type;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = size; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    CHK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_ID3D12Resource, (void**)&r));
    return r;
}

static void transition(ID3D12GraphicsCommandList* cl, Tex& t, D3D12_RESOURCE_STATES to)
{
    if (t.state == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = t.res; b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = t.state; b.Transition.StateAfter = to;
    cl->ResourceBarrier(1, &b);
    t.state = to;
}

// ------------------------------------------------------------------------------------------------ replay
// --replay-pipe N (with --replay): every captured real frame's inputs uploaded once, then N real frames cycling through
// them with two command lists in flight (the GPU never idles, as in the game); REPLAY_MFC=1/2/3 picks 2x/3x/4x.
// Prints the median GPU ms per real frame (frames 4..N) and the median per evaluate.
static unsigned g_replay_pipe = 0;
static unsigned g_replay_ui = 0;   // --replay-ui N: DLSSG.UserInterfaceRecompositionEnabled at create (W3: 1)
// --replay DIR: real game frames captured by the host (dlssg_host.ini DumpDir/DumpReal): per real frame r the input
// resources (res_rRR_NAME_WxH_fmtF_pitchP.bin) and every evaluate's parameters (params_rRR_eK.txt). Runs the same
// evaluates and writes the generated frames as rRR_gK.ppm (8-bit RGB), plus the real frames (rRR_real.ppm).
struct ReplayRes { std::string name; uint32_t w, h, pitch; DXGI_FORMAT fmt; std::string file; };
struct ReplayEval { std::vector<std::pair<std::string, float>> f; std::vector<std::pair<std::string, unsigned>> u;
                    std::vector<std::pair<std::string, std::vector<float>>> m; std::vector<std::string> r; };

static bool list_dir(const std::string& dir, std::vector<std::string>& out)
{
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    do { out.push_back(fd.cFileName); } while (FindNextFileA(h, &fd));
    FindClose(h);
    return true;
}

static void to_rgb8(const uint8_t* src, uint32_t W, uint32_t H, uint32_t pitch, DXGI_FORMAT fmt, std::vector<uint8_t>& rgba)
{
    rgba.resize((size_t)W * H * 4);
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            uint8_t* o = &rgba[4 * ((size_t)y * W + x)];
            const uint8_t* i = src + (size_t)y * pitch;
            if (fmt == DXGI_FORMAT_R10G10B10A2_UNORM || fmt == DXGI_FORMAT_R10G10B10A2_TYPELESS) {
                uint32_t v; memcpy(&v, i + 4 * x, 4);
                o[0] = (uint8_t)((v & 1023) >> 2); o[1] = (uint8_t)(((v >> 10) & 1023) >> 2); o[2] = (uint8_t)(((v >> 20) & 1023) >> 2); o[3] = 255;
            } else if (fmt == DXGI_FORMAT_B8G8R8A8_UNORM || fmt == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB || fmt == DXGI_FORMAT_B8G8R8A8_TYPELESS) {
                o[0] = i[4 * x + 2]; o[1] = i[4 * x + 1]; o[2] = i[4 * x]; o[3] = 255;
            } else {
                memcpy(o, i + 4 * x, 4);
            }
        }
}

static int replay(const std::string& out, const std::string& dir, ID3D12Device* dev, ID3D12CommandQueue* q,
                  ID3D12CommandAllocator* ca, ID3D12GraphicsCommandList* cl, ID3D12Fence* fence, HANDLE ev,
                  void* (*alloc)(), unsigned (*create)(ID3D12GraphicsCommandList*, void*, void**),
                  unsigned (*eval)(ID3D12GraphicsCommandList*, void*, void*), bool lite, unsigned repeat)
{
    std::vector<std::string> files;
    if (!list_dir(dir, files)) { fprintf(stderr, "replay: cannot list %s\n", dir.c_str()); return 1; }
    std::map<int, std::vector<ReplayRes>> res;      // real frame -> resources
    std::map<std::pair<int, int>, ReplayEval> evs;  // (real frame, index) -> parameters
    for (auto& f : files) {
        int r, k; char name[128]; unsigned w, h, fmt, pitch;
        if (sscanf(f.c_str(), "res_r%d_%127[^_]_%ux%u_fmt%u_pitch%u.bin", &r, name, &w, &h, &fmt, &pitch) == 6)
            res[r].push_back({name, w, h, pitch, (DXGI_FORMAT)fmt, dir + "\\" + f});
        else if (sscanf(f.c_str(), "params_r%d_e%d.txt", &r, &k) == 2) {
            FILE* fp = fopen((dir + "\\" + f).c_str(), "r");
            if (!fp) continue;
            ReplayEval e; char line[1024];
            while (fgets(line, sizeof line, fp)) {
                char t[8], key[160];
                if (sscanf(line, "%7s %159s", t, key) != 2) continue;
                const char* v = strstr(line, key) + strlen(key);
                if (!strcmp(t, "f")) e.f.push_back({key, (float)atof(v)});
                else if (!strcmp(t, "u")) e.u.push_back({key, (unsigned)strtoul(v, nullptr, 10)});
                else if (!strcmp(t, "m")) {
                    std::vector<float> m(16); char* pp = (char*)v;
                    for (int i = 0; i < 16; i++) m[i] = strtof(pp, &pp);
                    e.m.push_back({key, m});
                } else if (!strcmp(t, "r")) e.r.push_back(key);
            }
            fclose(fp);
            evs[{r, k}] = e;
        }
    }
    if (res.empty() || evs.empty()) { fprintf(stderr, "replay: no capture in %s\n", dir.c_str()); return 1; }
    UINT64 fv = 100;
    auto submit_wait = [&]() {
        CHK(cl->Close());
        ID3D12CommandList* lists[] = {cl};
        q->ExecuteCommandLists(1, lists);
        CHK(q->Signal(fence, ++fv));
        if (fence->GetCompletedValue() < fv) { fence->SetEventOnCompletion(fv, ev); WaitForSingleObject(ev, 20000); }
        if (fence->GetCompletedValue() < fv) { fprintf(stderr, "GPU timeout\n"); exit(3); }
        CHK(ca->Reset());
        CHK(cl->Reset(ca, nullptr));
    };
    // textures for the inputs (same formats as in the game), outputs in the back buffer format
    const auto& r0 = res.begin()->second;
    std::map<std::string, Tex> tex;
    UINT64 upsize = 0;
    for (auto& rr : r0) {
        // depth: the capture holds the 32-bit depth plane (subresource 0); the harness gives DLSS-G an R32_FLOAT texture
        bool d32 = rr.fmt == DXGI_FORMAT_D32_FLOAT || rr.fmt == DXGI_FORMAT_R32_TYPELESS ||
                   rr.fmt == DXGI_FORMAT_R32G8X24_TYPELESS || rr.fmt == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ||
                   rr.fmt == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        DXGI_FORMAT f = d32 ? DXGI_FORMAT_R32_FLOAT : rr.fmt;
        if (getenv("REPLAY_MV32") && rr.name == "MVecs" && rr.fmt == DXGI_FORMAT_R16G16_FLOAT) f = DXGI_FORMAT_R32G32_FLOAT;
        if (rr.fmt == DXGI_FORMAT_D24_UNORM_S8_UINT || rr.fmt == DXGI_FORMAT_R24G8_TYPELESS) { fprintf(stderr, "replay: depth format %u not handled\n", rr.fmt); return 1; }
        tex[rr.name] = make_tex(dev, rr.w, rr.h, f, 0, false, D3D12_RESOURCE_STATE_COPY_DEST);
        upsize += (tex[rr.name].total + 511) & ~511ull;
        fprintf(stderr, "replay: %s %ux%u fmt %u (pitch %u)\n", rr.name.c_str(), rr.w, rr.h, (unsigned)rr.fmt, rr.pitch);
    }
    unsigned W = 0, H = 0, bbfmt = 0, ui = g_replay_ui, mfc = 1;
    for (auto& kv : evs.begin()->second.u) {
        if (kv.first == "DLSSG.Width") W = kv.second;
        if (kv.first == "DLSSG.Height") H = kv.second;
        if (kv.first == "DLSSG.BackbufferFormat") bbfmt = kv.second;
        // (UserInterfaceRecompositionEnabled: create-time, taken from --replay-ui, not from the capture)
        if (kv.first == "DLSSG.MultiFrameCount") mfc = kv.second;
    }
    if (!W) { W = tex["Backbuffer"].w; H = tex["Backbuffer"].h; }
    if (!bbfmt) bbfmt = tex["Backbuffer"].fmt;
    std::vector<Tex> outs;
    for (unsigned k = 0; k < mfc; k++)
        outs.push_back(make_tex(dev, W, H, (DXGI_FORMAT)bbfmt, 4, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    Tex outReal = make_tex(dev, W, H, (DXGI_FORMAT)bbfmt, 4, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* up = make_buffer(dev, upsize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    UINT64 rbsize = outs[0].total * mfc;
    ID3D12Resource* rb = make_buffer(dev, rbsize, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12QueryHeap* qh = nullptr;
    D3D12_QUERY_HEAP_DESC qhd{}; qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qhd.Count = 16;
    CHK(dev->CreateQueryHeap(&qhd, IID_ID3D12QueryHeap, (void**)&qh));
    ID3D12Resource* tsb = make_buffer(dev, 4096, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    UINT64 tsfreq = 1; q->GetTimestampFrequency(&tsfreq);

    void* p = alloc();
    ms::setU(p, "CreationNodeMask", 1);
    ms::setU(p, "VisibilityNodeMask", 1);
    ms::setU(p, "DLSSG.Width", W);
    ms::setU(p, "DLSSG.Height", H);
    ms::setU(p, "DLSSG.BackbufferFormat", bbfmt);
    if (ui) ms::setU(p, "DLSSG.UserInterfaceRecompositionEnabled", ui);
    if (lite) ms::setU(p, "DLSSG.LiteCandidateBlendingEnabled", 1);
    if (getenv("REPLAY_RENDERWH") && tex.count("MVecs")) {
        ms::setU(p, "DLSSG.RenderWidth", tex["MVecs"].w);
        ms::setU(p, "DLSSG.RenderHeight", tex["MVecs"].h);
    }
    void* h = nullptr;
    unsigned cr = create(cl, p, &h);
    fprintf(stderr, "replay: CreateFeature(%ux%u fmt %u, UI recomposition %u, lite %d) -> 0x%x\n", W, H, bbfmt, ui, (int)lite, cr);
    if (failed(cr) || !h) return 1;
    submit_wait();
    std::vector<std::vector<float>> mats;
    mats.reserve(64);
    std::vector<double> times;
    if (g_replay_pipe) {
        // every captured real frame gets its own input textures, uploaded once; then g_replay_pipe real
        // frames cycle through them with two command lists in flight (the GPU never idles, as in the game)
        std::vector<int> order;
        std::map<int, std::map<std::string, Tex>> ftex;
        for (auto& rf : res) {
            int r = rf.first;
            order.push_back(r);
            uint8_t* m = nullptr; D3D12_RANGE none{0, 0};
            CHK(up->Map(0, &none, (void**)&m));
            UINT64 off = 0;
            std::map<std::string, UINT64> offs;
            for (auto& rr : rf.second) {
                Tex& t0 = tex[rr.name];
                Tex t = make_tex(dev, t0.w, t0.h, t0.fmt, 0, false, D3D12_RESOURCE_STATE_COPY_DEST);
                FILE* fp = fopen(rr.file.c_str(), "rb");
                std::vector<uint8_t> data((size_t)rr.pitch * rr.h);
                if (fp) { size_t n = fread(data.data(), 1, data.size(), fp); (void)n; fclose(fp); }
                for (uint32_t y = 0; y < t.fp.Footprint.Height; y++)
                    memcpy(m + off + t.fp.Offset + (size_t)y * t.fp.Footprint.RowPitch, data.data() + (size_t)y * rr.pitch,
                           std::min<size_t>(rr.pitch, t.fp.Footprint.RowPitch));
                offs[rr.name] = off;
                off += (t.total + 511) & ~511ull;
                ftex[r][rr.name] = t;
            }
            up->Unmap(0, nullptr);
            for (auto& rr : rf.second) {
                Tex& t = ftex[r][rr.name];
                D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
                dst.pResource = t.res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
                src.pResource = up; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = t.fp;
                src.PlacedFootprint.Offset += offs[rr.name];
                cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                transition(cl, t, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            submit_wait();
        }
        ID3D12CommandAllocator* ca2 = nullptr; ID3D12GraphicsCommandList* cl2 = nullptr;
        CHK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator, (void**)&ca2));
        CHK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ca2, nullptr, IID_ID3D12GraphicsCommandList, (void**)&cl2));
        CHK(cl2->Close());
        ID3D12CommandAllocator* cas[2] = {ca, ca2};
        ID3D12GraphicsCommandList* cls[2] = {cl, cl2};
        ID3D12Resource* tsbs[2] = {tsb, make_buffer(dev, 4096, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST)};
        D3D12_QUERY_HEAP_DESC qhd2{}; qhd2.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qhd2.Count = 32;
        ID3D12QueryHeap* qh2 = nullptr;
        CHK(dev->CreateQueryHeap(&qhd2, IID_ID3D12QueryHeap, (void**)&qh2));
        UINT64 setfv[2] = {0, 0};
        UINT64 fv2 = 1000000;
        std::vector<std::vector<double>> eval_ms(16);
        auto collect = [&](unsigned fi, int set) {
            if (fence->GetCompletedValue() < setfv[set]) { fence->SetEventOnCompletion(setfv[set], ev); WaitForSingleObject(ev, 20000); }
            if (fence->GetCompletedValue() < setfv[set]) { fprintf(stderr, "GPU timeout\n"); exit(3); }
            UINT64* ts = nullptr; D3D12_RANGE all{0, 4096};
            CHK(tsbs[set]->Map(0, &all, (void**)&ts));
            if (fi >= 4) {
                times.push_back((double)(ts[mfc] - ts[0]) * 1000.0 / tsfreq);
                for (unsigned k = 1; k <= mfc; k++) eval_ms[k - 1].push_back((double)(ts[k] - ts[k - 1]) * 1000.0 / tsfreq);
            }
            tsbs[set]->Unmap(0, nullptr);
        };
        // REPLAY_MFC=N: run the captured frames as N generated frames per real frame (2x: 1, 3x: 2, 4x: 3)
        if (getenv("REPLAY_MFC")) mfc = std::min<unsigned>(mfc, (unsigned)atoi(getenv("REPLAY_MFC")));
        CHK(cl->Close());
        for (unsigned f = 0; f < g_replay_pipe; f++) {
            int set = f & 1;
            if (f >= 2) collect(f - 2, set);
            ID3D12GraphicsCommandList* c = cls[set];
            CHK(cas[set]->Reset());
            CHK(c->Reset(cas[set], nullptr));
            int r = order[f % order.size()];
            unsigned qb = 16u * (unsigned)set;
            c->EndQuery(qh2, D3D12_QUERY_TYPE_TIMESTAMP, qb);
            for (unsigned k = 1; k <= mfc; k++) {
                auto it = evs.find({r, (int)k});
                if (it == evs.end()) continue;
                auto& e = it->second;
                for (auto& kv : e.f) ms::setF(p, kv.first.c_str(), kv.second);
                for (auto& kv : e.u)
                    if (kv.first != "DLSSG.OutputDisableInterpolation" && kv.first != "DLSSG.UserInterfaceRecompositionEnabled")
                        ms::setU(p, kv.first.c_str(), kv.second);
                for (auto& kv : e.m) { mats.push_back(kv.second); ms::setPtr(p, kv.first.c_str(), mats.back().data()); if (mats.size() > 60) mats.erase(mats.begin()); }
                // as in the game: OutputReal only where the captured evaluate has it (Streamline sets it at index 1)
                bool has_real = false;
                for (auto& n : e.r) if (n.substr(6) == "OutputReal") has_real = true;
                if (!has_real) ms::setPtr(p, "DLSSG.OutputReal", nullptr);
                for (auto& n : e.r) {
                    std::string nm = n.substr(6);
                    if (nm == "OutputInterpolated") ms::setPtr(p, n.c_str(), outs[k - 1].res);
                    else if (nm == "OutputReal") ms::setPtr(p, n.c_str(), outReal.res);
                    else if (ftex[r].count(nm)) ms::setPtr(p, n.c_str(), ftex[r][nm].res);
                }
                ms::setU(p, "DLSSG.Reset", f == 0 ? 1u : 0u);
                ms::setU(p, "DLSSG.MultiFrameCount", mfc);
                unsigned er = eval(c, h, p);
                if (failed(er)) fprintf(stderr, "replay-pipe: evaluate f%u r%d e%u -> 0x%x\n", f, r, k, er);
                c->EndQuery(qh2, D3D12_QUERY_TYPE_TIMESTAMP, qb + k);
            }
            c->ResolveQueryData(qh2, D3D12_QUERY_TYPE_TIMESTAMP, qb, mfc + 1, tsbs[set], 0);
            CHK(c->Close());
            ID3D12CommandList* lists[] = {c};
            q->ExecuteCommandLists(1, lists);
            CHK(q->Signal(fence, ++fv2));
            setfv[set] = fv2;
        }
        for (unsigned f = g_replay_pipe >= 2 ? g_replay_pipe - 2 : 0; f < g_replay_pipe; f++) collect(f, f & 1);
        std::sort(times.begin(), times.end());
        std::string per;
        for (unsigned k = 0; k < mfc; k++) {
            auto& v = eval_ms[k];
            std::sort(v.begin(), v.end());
            char b[32]; snprintf(b, sizeof b, " %.3f", v.empty() ? 0.0 : v[v.size() / 2]);
            per += b;
        }
        fprintf(stderr, "replay-pipe: GPU time per real frame median %.3f ms min %.3f over %zu (%zu captured frames)\n",
                times.empty() ? 0.0 : times[times.size() / 2], times.empty() ? 0.0 : times[0], times.size(), order.size());
        fprintf(stderr, "replay-pipe: median per evaluate:%s ms\n", per.c_str());
        return 0;
    }
    // REPLAY_MFC=N (serial replay too): N generated frames per real frame (2x: 1, 3x: 2, 4x: 3); the captured
    // evaluates 1..N are run with DLSSG.MultiFrameCount = N (index k of N: t = k / (N + 1))
    unsigned mfc_override = 0;
    if (getenv("REPLAY_MFC")) { mfc_override = std::min<unsigned>(mfc, (unsigned)atoi(getenv("REPLAY_MFC"))); mfc = mfc_override; }
    for (unsigned rep = 0; rep < repeat; rep++)
    for (auto& rf : res) {
        int r = rf.first;
        // upload this real frame's inputs
        uint8_t* m = nullptr; D3D12_RANGE none{0, 0};
        CHK(up->Map(0, &none, (void**)&m));
        UINT64 off = 0;
        std::map<std::string, UINT64> offs;
        for (auto& rr : rf.second) {
            Tex& t = tex[rr.name];
            FILE* fp = fopen(rr.file.c_str(), "rb");
            std::vector<uint8_t> data((size_t)rr.pitch * rr.h);
            if (fp) { size_t n = fread(data.data(), 1, data.size(), fp); (void)n; fclose(fp); }
            uint32_t pitch_in = rr.pitch;
            if (t.fmt == DXGI_FORMAT_R32G32_FLOAT && rr.fmt == DXGI_FORMAT_R16G16_FLOAT) {
                // REPLAY_MV32: half -> float
                std::vector<uint8_t> conv((size_t)rr.w * 8 * rr.h);
                for (uint32_t y = 0; y < rr.h; y++)
                    for (uint32_t x = 0; x < rr.w * 2; x++) {
                        uint16_t hv; memcpy(&hv, data.data() + (size_t)y * rr.pitch + 2 * x, 2);
                        uint32_t sgn = (hv >> 15) & 1, ex = (hv >> 10) & 31, mt = hv & 1023, fv;
                        if (ex == 0) { if (!mt) fv = sgn << 31; else { int e2 = -1; do { e2++; mt <<= 1; } while (!(mt & 1024)); mt &= 1023; fv = (sgn << 31) | ((uint32_t)(127 - 15 - e2) << 23) | (mt << 13); } }
                        else if (ex == 31) fv = (sgn << 31) | 0x7f800000u | (mt << 13);
                        else fv = (sgn << 31) | ((ex - 15 + 127) << 23) | (mt << 13);
                        memcpy(conv.data() + (size_t)y * rr.w * 8 + 4 * x, &fv, 4);
                    }
                data.swap(conv);
                pitch_in = rr.w * 8;
            }
            for (uint32_t y = 0; y < t.fp.Footprint.Height; y++)
                memcpy(m + off + t.fp.Offset + (size_t)y * t.fp.Footprint.RowPitch, data.data() + (size_t)y * pitch_in,
                       std::min<size_t>(pitch_in, t.fp.Footprint.RowPitch));
            offs[rr.name] = off;
            off += (t.total + 511) & ~511ull;
            if (rr.name == "Backbuffer" && rep == 0) {
                std::vector<uint8_t> rgba;
                to_rgb8(data.data(), rr.w, rr.h, rr.pitch, rr.fmt, rgba);
                char nm[64]; snprintf(nm, sizeof nm, "\\r%02d_real.ppm", r);
                write_ppm(out + nm, rgba.data(), rr.w, rr.h, rr.w * 4);
            }
        }
        up->Unmap(0, nullptr);
        for (auto& rr : rf.second) {
            Tex& t = tex[rr.name];
            transition(cl, t, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            dst.pResource = t.res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.pResource = up; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = t.fp;
            src.PlacedFootprint.Offset += offs[rr.name];
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            transition(cl, t, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        cl->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        unsigned nk = 0;
        for (unsigned k = 1; k <= mfc; k++) {
            auto it = evs.find({r, (int)k});
            if (it == evs.end()) continue;
            auto& e = it->second;
            // REPLAY_SKIP=name,name: parameters not set (debugging)
            static std::string skip = getenv("REPLAY_SKIP") ? std::string(",") + getenv("REPLAY_SKIP") + "," : "";
            auto skipped = [&](const std::string& n) { return !skip.empty() && skip.find("," + n.substr(6) + ",") != std::string::npos; };
            for (auto& kv : e.f) if (!skipped(kv.first)) ms::setF(p, kv.first.c_str(), kv.second);
            for (auto& kv : e.u)
                if (!skipped(kv.first) && kv.first != "DLSSG.OutputDisableInterpolation" &&   // an output buffer, not a value
                    (kv.first != "DLSSG.UserInterfaceRecompositionEnabled" || getenv("REPLAY_EVAL_UI")))   // create-time (--replay-ui)
                    ms::setU(p, kv.first.c_str(), kv.second);
            if (mfc_override) ms::setU(p, "DLSSG.MultiFrameCount", mfc_override);
            for (auto& kv : e.m) { mats.push_back(kv.second); ms::setPtr(p, kv.first.c_str(), mats.back().data()); if (mats.size() > 60) mats.erase(mats.begin()); }
            // as in the game: OutputReal only where the captured evaluate has it (Streamline sets it at index 1); the
            // parameter map is reused, so an earlier evaluate's OutputReal must be cleared (DLSS-G would otherwise copy
            // the back buffer to it on every evaluate)
            bool has_real = false;
            for (auto& n : e.r) if (n.substr(6) == "OutputReal" && !skipped(n)) has_real = true;
            if (!has_real) ms::setPtr(p, "DLSSG.OutputReal", nullptr);
            for (auto& n : e.r) {
                std::string nm = n.substr(6);
                if (skipped(n)) continue;
                if (nm == "OutputInterpolated") ms::setPtr(p, n.c_str(), outs[k - 1].res);
                else if (nm == "OutputReal") ms::setPtr(p, n.c_str(), outReal.res);
                else if (tex.count(nm)) ms::setPtr(p, n.c_str(), tex[nm].res);
            }
            if (rep > 0) ms::setU(p, "DLSSG.Reset", 0);
            unsigned er = eval(cl, h, p);
            if (failed(er)) fprintf(stderr, "replay: evaluate r%d e%u -> 0x%x\n", r, k, er);
            cl->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, k);
            nk = k;
        }
        cl->ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, nk + 1, tsb, 0);
        for (unsigned k = 0; k < nk; k++) {
            transition(cl, outs[k], D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            src.pResource = outs[k].res; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.pResource = rb; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = outs[k].fp;
            dst.PlacedFootprint.Offset = outs[0].total * k;
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            transition(cl, outs[k], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        submit_wait();
        UINT64* ts = nullptr; D3D12_RANGE all{0, 4096};
        CHK(tsb->Map(0, &all, (void**)&ts));
        double tot = (double)(ts[nk] - ts[0]) * 1000.0 / tsfreq;
        tsb->Unmap(0, nullptr);
        if (rep > 0 || repeat == 1) times.push_back(tot);
        if (rep + 1 == repeat) {
            uint8_t* rm = nullptr; D3D12_RANGE rall{0, (SIZE_T)rbsize};
            CHK(rb->Map(0, &rall, (void**)&rm));
            for (unsigned k = 0; k < nk; k++) {
                std::vector<uint8_t> rgba;
                to_rgb8(rm + outs[0].total * k + outs[k].fp.Offset, W, H, outs[k].fp.Footprint.RowPitch, (DXGI_FORMAT)bbfmt, rgba);
                char nm[64]; snprintf(nm, sizeof nm, "\\r%02d_g%u.ppm", r, k + 1);
                write_ppm(out + nm, rgba.data(), W, H, W * 4);
            }
            rb->Unmap(0, nullptr);
        }
        fprintf(stderr, "replay: real frame %d: %u evaluates, GPU %.3f ms\n", r, nk, tot);
    }
    if (!times.empty()) {
        std::sort(times.begin(), times.end());
        fprintf(stderr, "replay: GPU time per real frame median %.3f ms over %zu\n", times[times.size() / 2], times.size());
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s HOST_DLL OUT_DIR [options]\n", argv[0]); return 2; }
    std::string host = argv[1], out = argv[2];
    // Proton does not hand a GUI-less process's console output back to the terminal: log to OUT_DIR\harness.log
    freopen((out + "\\harness.log").c_str(), "w", stderr);
    setvbuf(stderr, nullptr, _IONBF, 0);
    uint32_t W = 2560, H = 1440, RW = 1280, RH = 720, frames = 8, mfc = 1;
    bool bench = false, compute = false, pipe = false, copyout = false, lite = false;
    std::string replay_dir;
    unsigned replay_repeat = 1;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(argv[++i]); };
        if (a == "--w") W = std::stoul(nxt());
        else if (a == "--h") H = std::stoul(nxt());
        else if (a == "--rw") RW = std::stoul(nxt());
        else if (a == "--rh") RH = std::stoul(nxt());
        else if (a == "--frames") frames = std::stoul(nxt());
        else if (a == "--mfc") mfc = std::stoul(nxt());
        else if (a == "--bench") bench = true;
        else if (a == "--pipe") pipe = true;   // bench: two frames in flight (the GPU never idles between frames)
        else if (a == "--copyout") copyout = true; // after each evaluate copy the output into a back-buffer-like texture
        else if (a == "--replay") replay_dir = nxt();
        else if (a == "--replay-repeat") replay_repeat = std::stoul(nxt());
        else if (a == "--replay-ui") g_replay_ui = std::stoul(nxt());
        else if (a == "--replay-pipe") g_replay_pipe = std::stoul(nxt());
        else if (a == "--lite") lite = true;     // DLSSG.LiteCandidateBlendingEnabled = 1 at create
        else if (a == "--compute") compute = true;
        else if (a == "--direct") compute = false;
    }
    HMODULE ngx = LoadLibraryA(host.c_str());
    if (!ngx) { fprintf(stderr, "LoadLibrary %s failed %lu\n", host.c_str(), GetLastError()); return 1; }
    auto Init = (NgxResult (*)(unsigned long long, const wchar_t*, ID3D12Device*, unsigned, const void*))GetProcAddress(ngx, "NVSDK_NGX_D3D12_Init_Ext");
    auto Caps = (NgxResult (*)(void**))GetProcAddress(ngx, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    auto Alloc = (NgxResult (*)(void**))GetProcAddress(ngx, "NVSDK_NGX_D3D12_AllocateParameters");
    auto Create = (NgxResult (*)(ID3D12GraphicsCommandList*, unsigned, void*, NgxHandle**))GetProcAddress(ngx, "NVSDK_NGX_D3D12_CreateFeature");
    auto Eval = (NgxResult (*)(ID3D12GraphicsCommandList*, const NgxHandle*, void*, void*))GetProcAddress(ngx, "NVSDK_NGX_D3D12_EvaluateFeature");
    auto Release = (NgxResult (*)(NgxHandle*))GetProcAddress(ngx, "NVSDK_NGX_D3D12_ReleaseFeature");
    auto Shutdown1 = (NgxResult (*)(ID3D12Device*))GetProcAddress(ngx, "NVSDK_NGX_D3D12_Shutdown1");
    auto GetReq = (NgxResult (*)(IDXGIAdapter*, const void*, void*))GetProcAddress(ngx, "NVSDK_NGX_D3D12_GetFeatureRequirements");
    if (!Init || !Caps || !Alloc || !Create || !Eval) { fprintf(stderr, "missing NGX exports\n"); return 1; }

    ID3D12Device* dev = nullptr;
    CHK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_ID3D12Device, (void**)&dev));
    D3D12_COMMAND_LIST_TYPE lt = compute ? D3D12_COMMAND_LIST_TYPE_COMPUTE : D3D12_COMMAND_LIST_TYPE_DIRECT;
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = lt;
    ID3D12CommandQueue* q = nullptr;
    CHK(dev->CreateCommandQueue(&qd, IID_ID3D12CommandQueue, (void**)&q));
    ID3D12CommandAllocator* ca = nullptr;
    CHK(dev->CreateCommandAllocator(lt, IID_ID3D12CommandAllocator, (void**)&ca));
    ID3D12GraphicsCommandList* cl = nullptr;
    CHK(dev->CreateCommandList(0, lt, ca, nullptr, IID_ID3D12GraphicsCommandList, (void**)&cl));
    ID3D12CommandAllocator* ca2 = nullptr;
    ID3D12GraphicsCommandList* cl2 = nullptr;
    CHK(dev->CreateCommandAllocator(lt, IID_ID3D12CommandAllocator, (void**)&ca2));
    CHK(dev->CreateCommandList(0, lt, ca2, nullptr, IID_ID3D12GraphicsCommandList, (void**)&cl2));
    CHK(cl2->Close());
    ID3D12Fence* fence = nullptr;
    CHK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence, (void**)&fence));
    HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;
    auto submit_wait = [&]() {
        CHK(cl->Close());
        ID3D12CommandList* lists[] = {cl};
        q->ExecuteCommandLists(1, lists);
        CHK(q->Signal(fence, ++fv));
        if (fence->GetCompletedValue() < fv) { fence->SetEventOnCompletion(fv, ev); WaitForSingleObject(ev, 20000); }
        if (fence->GetCompletedValue() < fv) { fprintf(stderr, "GPU timeout (frame did not complete in 20 s)\n"); exit(3); }
        CHK(ca->Reset());
        CHK(cl->Reset(ca, nullptr));
    };

    if (GetReq) {
        unsigned disc[16] = {}; disc[0] = 0x15; disc[1] = 11; // SDKVersion, FeatureID
        struct { int supported; unsigned minArch; char os[255]; } req{};
        NgxResult r = GetReq(nullptr, disc, &req);
        fprintf(stderr, "GetFeatureRequirements(FG) -> 0x%x supported %d minArch 0x%x\n", r, req.supported, req.minArch);
    }
    NgxResult r = Init(0x1234, L"C:\\users\\steamuser\\Temp", dev, 0x15, nullptr);
    fprintf(stderr, "Init_Ext -> 0x%x\n", r);
    if (failed(r)) return 1;
    void* caps = nullptr;
    r = Caps(&caps);
    int avail = -1; unsigned mfcMax = 0, model = 0;
    ms::getI(caps, "FrameGeneration.Available", &avail);
    ms::getU(caps, "DLSSG.MultiFrameCountMax", &mfcMax);
    ms::getU(caps, "DLSSG.ModelVersion", &model);
    fprintf(stderr, "caps -> 0x%x: FrameGeneration.Available %d, MultiFrameCountMax %u, ModelVersion %u\n", r, avail, mfcMax, model);

    if (!replay_dir.empty()) {
        static decltype(Alloc) s_alloc; static decltype(Create) s_create; static decltype(Eval) s_eval;
        s_alloc = Alloc; s_create = Create; s_eval = Eval;
        int rc = replay(out, replay_dir, dev, q, ca, cl, fence, ev,
            []() -> void* { void* pp = nullptr; s_alloc(&pp); return pp; },
            [](ID3D12GraphicsCommandList* c, void* pp, void** hh) -> unsigned { NgxHandle* x = nullptr; unsigned r = s_create(c, 11, pp, &x); *hh = x; return r; },
            [](ID3D12GraphicsCommandList* c, void* hh, void* pp) -> unsigned { return s_eval(c, (NgxHandle*)hh, pp, nullptr); },
            lite, replay_repeat);
        if (Shutdown1) fprintf(stderr, "Shutdown1 -> 0x%x\n", Shutdown1(dev));
        return rc;
    }
    Tex back = make_tex(dev, W, H, DXGI_FORMAT_R8G8B8A8_UNORM, 4, false, D3D12_RESOURCE_STATE_COPY_DEST);
    Tex hudless = make_tex(dev, W, H, DXGI_FORMAT_R8G8B8A8_UNORM, 4, false, D3D12_RESOURCE_STATE_COPY_DEST);
    Tex depth = make_tex(dev, RW, RH, DXGI_FORMAT_R32_FLOAT, 4, false, D3D12_RESOURCE_STATE_COPY_DEST);
    Tex mvec = make_tex(dev, RW, RH, DXGI_FORMAT_R32G32_FLOAT, 8, false, D3D12_RESOURCE_STATE_COPY_DEST);
    std::vector<Tex> outs;
    for (uint32_t k = 0; k < mfc; k++)
        outs.push_back(make_tex(dev, W, H, DXGI_FORMAT_R8G8B8A8_UNORM, 4, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    Tex* ins[4] = {&back, &hudless, &depth, &mvec};
    // --copyout: what Streamline does per shown frame (copy the generated frame into a swap chain back buffer)
    std::vector<Tex> bbs;
    if (copyout)
        for (uint32_t k = 0; k < mfc; k++) {
            Tex t;
            t.fmt = DXGI_FORMAT_R8G8B8A8_UNORM; t.w = W; t.h = H; t.bpp = 4; t.state = D3D12_RESOURCE_STATE_COPY_DEST;
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC d{};
            d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = W; d.Height = H; d.DepthOrArraySize = 1;
            d.MipLevels = 1; d.Format = t.fmt; d.SampleDesc.Count = 1; d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            CHK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, t.state, nullptr, IID_ID3D12Resource, (void**)&t.res));
            bbs.push_back(t);
        }
    UINT64 upsize = 0, upoff[4];
    for (int i = 0; i < 4; i++) { upoff[i] = upsize; upsize += (ins[i]->total + 511) & ~511ull; }
    ID3D12Resource* up = make_buffer(dev, upsize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    UINT64 rbsize = outs[0].total * mfc + 4096;
    ID3D12Resource* rb = make_buffer(dev, rbsize, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    // --pipe: a second upload / timestamp readback set for the frame in flight
    ID3D12Resource* ups[2] = {up, pipe ? make_buffer(dev, upsize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ) : up};
    ID3D12Resource* rbs[2] = {rb, pipe ? make_buffer(dev, 4096, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST) : rb};
    ID3D12CommandAllocator* cas[2] = {ca, ca2};
    ID3D12GraphicsCommandList* cls[2] = {cl, cl2};
    UINT64 setfv[2] = {0, 0};
    ID3D12QueryHeap* qh = nullptr;
    D3D12_QUERY_HEAP_DESC qhd{}; qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qhd.Count = 32;
    CHK(dev->CreateQueryHeap(&qhd, IID_ID3D12QueryHeap, (void**)&qh));
    UINT64 tsfreq = 1;
    q->GetTimestampFrequency(&tsfreq);

    void* p = nullptr;
    Alloc(&p);
    ms::setU(p, "CreationNodeMask", 1);
    ms::setU(p, "VisibilityNodeMask", 1);
    ms::setU(p, "DLSSG.Width", W);
    ms::setU(p, "DLSSG.Height", H);
    ms::setU(p, "DLSSG.BackbufferFormat", (unsigned)DXGI_FORMAT_R8G8B8A8_UNORM);
    ms::setU(p, "DLSSG.RenderWidth", RW);
    ms::setU(p, "DLSSG.RenderHeight", RH);
    if (lite) ms::setU(p, "DLSSG.LiteCandidateBlendingEnabled", 1);
    if (g_replay_ui) ms::setU(p, "DLSSG.UserInterfaceRecompositionEnabled", g_replay_ui);
    NgxHandle* h = nullptr;
    r = Create(cl, 11, p, &h);
    fprintf(stderr, "CreateFeature -> 0x%x handle %p\n", r, (void*)h);
    if (failed(r) || !h) return 1;
    submit_wait();

    Scene sc{W, H, RW, RH};
    std::vector<std::vector<uint8_t>> rgba_c(4);
    std::vector<std::vector<float>> dep_c(4), mv_c(4);
    std::vector<uint8_t> rgba;
    std::vector<float> dep, mv;
    std::vector<double> gpu_ms;
    std::vector<std::vector<double>> eval_ms(16);
    if (bench)
        for (int f = 0; f < 4; f++) { sc.render((float)f, rgba_c[f]); sc.aux((float)f, dep_c[f], mv_c[f]); }
    float ident[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    if (pipe && bench) {
        // two frames in flight: record frame f while frame f-1 runs; frame f-2's timestamps are read when its set
        // is reused (the GPU never idles between frames, as in a game)
        auto collect = [&](uint32_t fi, int set) {
            if (fence->GetCompletedValue() < setfv[set]) { fence->SetEventOnCompletion(setfv[set], ev); WaitForSingleObject(ev, 20000); }
            if (fence->GetCompletedValue() < setfv[set]) { fprintf(stderr, "GPU timeout (frame did not complete in 20 s)\n"); exit(3); }
            uint8_t* rm = nullptr;
            D3D12_RANGE all{0, 4096};
            CHK(rbs[set]->Map(0, &all, (void**)&rm));
            UINT64 ts[16];
            memcpy(ts, rm, sizeof(UINT64) * (mfc + 1));
            rbs[set]->Unmap(0, nullptr);
            double tot = (double)(ts[mfc] - ts[0]) * 1000.0 / tsfreq;
            if (fi >= 2) {
                gpu_ms.push_back(tot);
                for (uint32_t k = 1; k <= mfc; k++)
                    eval_ms[k - 1].push_back((double)(ts[k] - ts[k - 1]) * 1000.0 / tsfreq);
            }
        };
        CHK(cl->Close());                       // left open (empty) by the last submit_wait
        for (uint32_t f = 0; f < frames; f++) {
            int set = f & 1;
            uint32_t qb = 16u * (uint32_t)set;
            if (f >= 2) collect(f - 2, set);
            ca = cas[set]; cl = cls[set]; up = ups[set]; rb = rbs[set];
            CHK(ca->Reset());
            CHK(cl->Reset(ca, nullptr));
        if (bench) { rgba = rgba_c[f % 4]; dep = dep_c[f % 4]; mv = mv_c[f % 4]; }
        else { sc.render((float)f, rgba); sc.aux((float)f, dep, mv); }
        uint8_t* m = nullptr;
        D3D12_RANGE none{0, 0};
        CHK(up->Map(0, &none, (void**)&m));
        const void* srcs[4] = {rgba.data(), rgba.data(), dep.data(), mv.data()};
        for (int i = 0; i < 4; i++) {
            Tex& t = *ins[i];
            for (uint32_t y = 0; y < t.h; y++)
                memcpy(m + upoff[i] + t.fp.Offset + (size_t)y * t.fp.Footprint.RowPitch,
                       (const uint8_t*)srcs[i] + (size_t)y * t.w * t.bpp, (size_t)t.w * t.bpp);
        }
        up->Unmap(0, nullptr);
        for (int i = 0; i < 4; i++) {
            Tex& t = *ins[i];
            transition(cl, t, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            dst.pResource = t.res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
            src.pResource = up; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = t.fp;
            src.PlacedFootprint.Offset += upoff[i];
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            transition(cl, t, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        cl->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, qb);
        for (uint32_t k = 1; k <= mfc; k++) {
            ms::setPtr(p, "DLSSG.Backbuffer", back.res);
            ms::setPtr(p, "DLSSG.HUDLess", hudless.res);
            ms::setPtr(p, "DLSSG.Depth", depth.res);
            ms::setPtr(p, "DLSSG.MVecs", mvec.res);
            ms::setPtr(p, "DLSSG.OutputInterpolated", outs[k - 1].res);
            ms::setU(p, "DLSSG.MultiFrameCount", mfc);
            ms::setU(p, "DLSSG.MultiFrameIndex", k);
            ms::setU(p, "DLSSG.Reset", f == 0 ? 1u : 0u);
            ms::setPtr(p, "DLSSG.CameraViewToClip", ident);
            ms::setPtr(p, "DLSSG.ClipToCameraView", ident);
            ms::setPtr(p, "DLSSG.ClipToLensClip", ident);
            ms::setPtr(p, "DLSSG.ClipToPrevClip", ident);
            ms::setPtr(p, "DLSSG.PrevClipToClip", ident);
            ms::setF(p, "DLSSG.MvecScaleX", 1.0f / RW);
            ms::setF(p, "DLSSG.MvecScaleY", 1.0f / RH);
            ms::setF(p, "DLSSG.JitterOffsetX", 0.0f);
            ms::setF(p, "DLSSG.JitterOffsetY", 0.0f);
            ms::setF(p, "DLSSG.CameraNear", 0.1f);
            ms::setF(p, "DLSSG.CameraFar", 1000.0f);
            ms::setF(p, "DLSSG.CameraFOV", 1.2f);
            ms::setF(p, "DLSSG.CameraAspectRatio", (float)W / H);
            ms::setU(p, "DLSSG.ColorBuffersHDR", 0);
            ms::setU(p, "DLSSG.DepthInverted", 0);
            ms::setU(p, "DLSSG.CameraMotionIncluded", 1);
            ms::setU(p, "DLSSG.NotRenderingGameFrames", 0);
            ms::setU(p, "DLSSG.OrthoProjection", 0);
            ms::setU(p, "DLSSG.MvecDilated", 0);
            ms::setU(p, "DLSSG.MenuDetectionEnabled", 0);
            ms::setU(p, "DLSSG.MVecsSubrectWidth", RW);
            ms::setU(p, "DLSSG.MVecsSubrectHeight", RH);
            ms::setU(p, "DLSSG.DepthSubrectWidth", RW);
            ms::setU(p, "DLSSG.DepthSubrectHeight", RH);
            ms::setU(p, "DLSSG.HUDLessSubrectWidth", W);
            ms::setU(p, "DLSSG.HUDLessSubrectHeight", H);
            ms::setU(p, "DLSSG.InputBackbufferSubrectWidth", W);
            ms::setU(p, "DLSSG.InputBackbufferSubrectHeight", H);
            ms::setU(p, "DLSSG.OutputInterpolatedSubrectWidth", W);
            ms::setU(p, "DLSSG.OutputInterpolatedSubrectHeight", H);
            r = Eval(cl, h, p, nullptr);
            if (failed(r)) fprintf(stderr, "Evaluate(frame %u, %u/%u) -> 0x%x\n", f, k, mfc, r);
            if (copyout) {
                transition(cl, outs[k - 1], D3D12_RESOURCE_STATE_COPY_SOURCE);
                transition(cl, bbs[k - 1], D3D12_RESOURCE_STATE_COPY_DEST);
                cl->CopyResource(bbs[k - 1].res, outs[k - 1].res);
                transition(cl, bbs[k - 1], D3D12_RESOURCE_STATE_PRESENT);
                transition(cl, outs[k - 1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            cl->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, qb + k);
        }
            cl->ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, qb, mfc + 1, rb, 0);
            CHK(cl->Close());
            ID3D12CommandList* lists[] = {cl};
            q->ExecuteCommandLists(1, lists);
            CHK(q->Signal(fence, ++fv));
            setfv[set] = fv;
        }
        for (uint32_t f = frames >= 2 ? frames - 2 : 0; f < frames; f++) collect(f, f & 1);
        ca = cas[0]; cl = cls[0]; up = ups[0]; rb = rbs[0];
        CHK(ca->Reset());
        CHK(cl->Reset(ca, nullptr));
    } else
    for (uint32_t f = 0; f < frames; f++) {
        if (bench) { rgba = rgba_c[f % 4]; dep = dep_c[f % 4]; mv = mv_c[f % 4]; }
        else { sc.render((float)f, rgba); sc.aux((float)f, dep, mv); }
        uint8_t* m = nullptr;
        D3D12_RANGE none{0, 0};
        CHK(up->Map(0, &none, (void**)&m));
        const void* srcs[4] = {rgba.data(), rgba.data(), dep.data(), mv.data()};
        for (int i = 0; i < 4; i++) {
            Tex& t = *ins[i];
            for (uint32_t y = 0; y < t.h; y++)
                memcpy(m + upoff[i] + t.fp.Offset + (size_t)y * t.fp.Footprint.RowPitch,
                       (const uint8_t*)srcs[i] + (size_t)y * t.w * t.bpp, (size_t)t.w * t.bpp);
        }
        up->Unmap(0, nullptr);
        for (int i = 0; i < 4; i++) {
            Tex& t = *ins[i];
            transition(cl, t, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            dst.pResource = t.res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
            src.pResource = up; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = t.fp;
            src.PlacedFootprint.Offset += upoff[i];
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            transition(cl, t, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        cl->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        for (uint32_t k = 1; k <= mfc; k++) {
            ms::setPtr(p, "DLSSG.Backbuffer", back.res);
            ms::setPtr(p, "DLSSG.HUDLess", hudless.res);
            ms::setPtr(p, "DLSSG.Depth", depth.res);
            ms::setPtr(p, "DLSSG.MVecs", mvec.res);
            ms::setPtr(p, "DLSSG.OutputInterpolated", outs[k - 1].res);
            ms::setU(p, "DLSSG.MultiFrameCount", mfc);
            ms::setU(p, "DLSSG.MultiFrameIndex", k);
            ms::setU(p, "DLSSG.Reset", f == 0 ? 1u : 0u);
            ms::setPtr(p, "DLSSG.CameraViewToClip", ident);
            ms::setPtr(p, "DLSSG.ClipToCameraView", ident);
            ms::setPtr(p, "DLSSG.ClipToLensClip", ident);
            ms::setPtr(p, "DLSSG.ClipToPrevClip", ident);
            ms::setPtr(p, "DLSSG.PrevClipToClip", ident);
            ms::setF(p, "DLSSG.MvecScaleX", 1.0f / RW);
            ms::setF(p, "DLSSG.MvecScaleY", 1.0f / RH);
            ms::setF(p, "DLSSG.JitterOffsetX", 0.0f);
            ms::setF(p, "DLSSG.JitterOffsetY", 0.0f);
            ms::setF(p, "DLSSG.CameraNear", 0.1f);
            ms::setF(p, "DLSSG.CameraFar", 1000.0f);
            ms::setF(p, "DLSSG.CameraFOV", 1.2f);
            ms::setF(p, "DLSSG.CameraAspectRatio", (float)W / H);
            ms::setU(p, "DLSSG.ColorBuffersHDR", 0);
            ms::setU(p, "DLSSG.DepthInverted", 0);
            ms::setU(p, "DLSSG.CameraMotionIncluded", 1);
            ms::setU(p, "DLSSG.NotRenderingGameFrames", 0);
            ms::setU(p, "DLSSG.OrthoProjection", 0);
            ms::setU(p, "DLSSG.MvecDilated", 0);
            ms::setU(p, "DLSSG.MenuDetectionEnabled", 0);
            ms::setU(p, "DLSSG.MVecsSubrectWidth", RW);
            ms::setU(p, "DLSSG.MVecsSubrectHeight", RH);
            ms::setU(p, "DLSSG.DepthSubrectWidth", RW);
            ms::setU(p, "DLSSG.DepthSubrectHeight", RH);
            ms::setU(p, "DLSSG.HUDLessSubrectWidth", W);
            ms::setU(p, "DLSSG.HUDLessSubrectHeight", H);
            ms::setU(p, "DLSSG.InputBackbufferSubrectWidth", W);
            ms::setU(p, "DLSSG.InputBackbufferSubrectHeight", H);
            ms::setU(p, "DLSSG.OutputInterpolatedSubrectWidth", W);
            ms::setU(p, "DLSSG.OutputInterpolatedSubrectHeight", H);
            r = Eval(cl, h, p, nullptr);
            if (failed(r)) fprintf(stderr, "Evaluate(frame %u, %u/%u) -> 0x%x\n", f, k, mfc, r);
            cl->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, k);
        }
        bool dump = !bench && f + 1 == frames;
        if (dump)
            for (uint32_t k = 0; k < mfc; k++) {
                transition(cl, outs[k], D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
                src.pResource = outs[k].res; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                dst.pResource = rb; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = outs[k].fp;
                dst.PlacedFootprint.Offset = 4096 + outs[0].total * k;
                cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                transition(cl, outs[k], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
        cl->ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, mfc + 1, rb, 0);
        DWORD t0 = GetTickCount();
        submit_wait();
        DWORD cpu = GetTickCount() - t0;
        uint8_t* rm = nullptr;
        D3D12_RANGE all{0, (SIZE_T)rbsize};
        CHK(rb->Map(0, &all, (void**)&rm));
        UINT64 ts[16];
        memcpy(ts, rm, sizeof(UINT64) * (mfc + 1));
        double tot = (double)(ts[mfc] - ts[0]) * 1000.0 / tsfreq;
        std::string per;
        for (uint32_t k = 1; k <= mfc; k++) {
            char b[32];
            snprintf(b, sizeof b, " %.3f", (double)(ts[k] - ts[k - 1]) * 1000.0 / tsfreq);
            per += b;
        }
        if (f >= 2) {
            gpu_ms.push_back(tot);
            for (uint32_t k = 1; k <= mfc; k++)
                eval_ms[k - 1].push_back((double)(ts[k] - ts[k - 1]) * 1000.0 / tsfreq);
        }
        if (!bench || f + 1 == frames)
            fprintf(stderr, "frame %u: GPU evaluates %.3f ms [per evaluate:%s] (submit+wait %lu ms)\n", f, tot, per.c_str(), cpu);
        if (dump) {
            for (uint32_t k = 0; k < mfc; k++)
                write_ppm(out + "\\f" + std::to_string(f) + "_g" + std::to_string(k + 1) + ".ppm",
                          rm + 4096 + outs[0].total * k + outs[k].fp.Offset, W, H, outs[k].fp.Footprint.RowPitch);
            write_ppm(out + "\\f" + std::to_string(f) + "_real.ppm", rgba.data(), W, H, W * 4);
            for (uint32_t k = 1; k <= mfc; k++) {
                std::vector<uint8_t> gt;
                sc.render((float)f - 1.0f + (float)k / (mfc + 1), gt);
                write_ppm(out + "\\f" + std::to_string(f) + "_gt" + std::to_string(k) + ".ppm", gt.data(), W, H, W * 4);
            }
        }
        rb->Unmap(0, nullptr);
    }
    if (!gpu_ms.empty()) {
        std::sort(gpu_ms.begin(), gpu_ms.end());
        fprintf(stderr, "GPU time per real frame (all %u evaluates): median %.3f ms min %.3f ms over %zu frames\n", mfc,
                gpu_ms[gpu_ms.size() / 2], gpu_ms[0], gpu_ms.size());
        std::string per;
        for (uint32_t k = 0; k < mfc; k++) {
            std::sort(eval_ms[k].begin(), eval_ms[k].end());
            char b[32];
            snprintf(b, sizeof b, " %.3f", eval_ms[k][eval_ms[k].size() / 2]);
            per += b;
        }
        fprintf(stderr, "median per evaluate:%s ms\n", per.c_str());
    }
    if (Release) fprintf(stderr, "Release -> 0x%x\n", Release(h));
    if (Shutdown1) fprintf(stderr, "Shutdown1 -> 0x%x\n", Shutdown1(dev));
    return 0;
}
