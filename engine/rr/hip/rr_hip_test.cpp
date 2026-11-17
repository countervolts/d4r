// Checks a HIP Ray Reconstruction stage (engine/rr/hip/rr_swin.hip) against the Vulkan engine's own run of the same
// stage, and times both forms of it. The reference is engine/replay_rr.cpp (d4r-rr) run with
// D4R_ENGINE_RR_ARENA_SNAP=1 and D4R_ENGINE_RR_DUMP=REF into OUT=REF: per frame F it leaves the parameter blocks
// (F-STAGE.bin), the arena after every network stage (F-arena-N.bin) and every image (F-NAME.bin, RGBA16F). The GLSL
// stages there are bit-identical to NVIDIA's kernels through ZLUDA.
//
// usage: rr_hip_test HSACO WEIGHTS REF STAGE FRAME RW RH [REPEAT]
//   HSACO    the stage's code object (kernel rr_swin), built against the model's swin_stage.h and W_WIDE
//   WEIGHTS  the model's weights-STAGEm.bin (with the widened copy)
//   RW RH    the render size of the reference run
// Prints the bytes of the arena and of each surface that differ from the reference, then the kernel's time.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); } while (0)

static std::vector<uint8_t> readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<uint8_t> d(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(d.data()), static_cast<std::streamsize>(d.size()));
    return d;
}

struct Image {
    hipArray_t array = nullptr;
    uint32_t w = 0, h = 0;
    hipTextureObject_t tex = 0;
    hipSurfaceObject_t surf = 0;
    uint32_t texel = 8;
};
static Image makeImage(uint32_t w, uint32_t h, const std::vector<uint8_t>* data, bool linear, bool snorm8 = false)
{
    Image im;
    im.w = w; im.h = h; im.texel = snorm8 ? 4 : 8;
    hipChannelFormatDesc fd = snorm8 ? hipCreateChannelDesc(8, 8, 8, 8, hipChannelFormatKindSigned)
                                     : hipCreateChannelDesc(16, 16, 16, 16, hipChannelFormatKindFloat);
    CK(hipMallocArray(&im.array, &fd, w, h, hipArraySurfaceLoadStore));
    std::vector<uint8_t> zero;
    if (!data) { zero.assign(size_t(w) * h * im.texel, 0); data = &zero; }
    if (data->size() != size_t(w) * h * im.texel) throw std::runtime_error("image size");
    CK(hipMemcpy2DToArray(im.array, 0, 0, data->data(), w * im.texel, w * im.texel, h, hipMemcpyHostToDevice));
    hipResourceDesc rd{};
    rd.resType = hipResourceTypeArray;
    rd.res.array.array = im.array;
    hipTextureDesc td{};
    td.addressMode[0] = td.addressMode[1] = hipAddressModeClamp;
    td.filterMode = linear ? hipFilterModeLinear : hipFilterModePoint;
    td.readMode = snorm8 ? hipReadModeNormalizedFloat : hipReadModeElementType;
    td.normalizedCoords = 1;
    CK(hipCreateTextureObject(&im.tex, &rd, &td, nullptr));
    CK(hipCreateSurfaceObject(&im.surf, &rd));
    return im;
}
static std::vector<uint8_t> readImage(const Image& im)
{
    std::vector<uint8_t> d(size_t(im.w) * im.h * im.texel);
    CK(hipMemcpy2DFromArray(d.data(), im.w * im.texel, im.array, 0, 0, im.w * im.texel, im.h, hipMemcpyDeviceToHost));
    return d;
}
static size_t diffBytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t& first)
{
    size_t n = 0;
    first = SIZE_MAX;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (a[i] != b[i]) { if (!n) first = i; ++n; }
    return n + (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}
// the surfaces' RGB only (the stages store 0 in alpha; the reference's undefined alpha is not compared)
static size_t diffRGB(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t& first)
{
    size_t n = 0;
    first = SIZE_MAX;
    for (size_t i = 0; i + 8 <= a.size() && i + 8 <= b.size(); i += 8)
        for (size_t k = 0; k < 6; k += 2)
            if (std::memcmp(&a[i + k], &b[i + k], 2) != 0) { if (!n) first = (i + k) / 2; ++n; }
    return n;
}

struct RrArgs {
    const uint32_t* param;
    const void* weights;
    uint32_t* arena;
    const uint32_t* desc;
    uint32_t weightBytes, arenaBytes;
    const uint32_t* aux;   // encoder 0: the gather stage's per-window words; unused elsewhere
    uint32_t auxBytes, pad;
};

int main(int argc, char** argv) try
{
    if (argc < 8) {
        std::fprintf(stderr, "usage: rr_hip_test HSACO WEIGHTS REF STAGE FRAME RW RH [REPEAT]\n");
        return 2;
    }
    const std::string hsaco = argv[1], weightsPath = argv[2], ref = argv[3], stage = argv[4], frame = argv[5];
    const uint32_t rw = std::atoi(argv[6]), rh = std::atoi(argv[7]);
    const int repeat = argc > 8 ? std::atoi(argv[8]) : 20;
    static const char* names[] = {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec4", "dec3", "dec2", "dec1", "dec0"};
    static const int levels[] = {0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0};
    int si = -1;
    for (int i = 0; i < 11; ++i) if (stage == names[i]) si = i;
    if (si < 1 || si > 10) throw std::runtime_error("stage must be enc1..enc5 or dec4..dec0");
    const int l = levels[si];
    const bool dec = si >= 6;
    const uint32_t w0 = (rw + 31) & ~31u, h0 = (rh + 31) & ~31u;
    auto lw = [&](int lv) { return w0 >> lv; };
    auto lh = [&](int lv) { return h0 >> lv; };
    auto img = [&](const std::string& n) { return readFile(ref + "/" + frame + "-" + n + ".bin"); };

    const auto param = readFile(ref + "/" + frame + "-" + stage + ".bin");
    const auto weights = readFile(weightsPath);
    const auto before = readFile(ref + "/" + frame + "-arena-" + std::to_string(si - 1) + ".bin");
    const auto after = readFile(ref + "/" + frame + "-arena-" + std::to_string(si) + ".bin");
    uint32_t p[36];
    std::memcpy(p, param.data(), sizeof(p));
    const uint32_t shift = p[20], sx = shift & 0xffffu, sy = shift >> 16;

    // textures and surfaces in the engine's wiring (engine/runtime_rr.inc REngine::Impl::resolve)
    Image t88, t96, t104, s112, s120, s128;
    std::vector<uint8_t> ref112, ref120, ref128;
    if (!dec) {
        const auto d88 = img("enc" + std::to_string(l - 1) + "-128");
        t88 = makeImage(lw(l), lh(l), &d88, true);
        s112 = makeImage(lw(l), lh(l), nullptr, false); ref112 = img("enc" + std::to_string(l) + "-112");
        s120 = makeImage(lw(l), lh(l), nullptr, false); ref120 = img("enc" + std::to_string(l) + "-120");
        if (l < 5) { s128 = makeImage(lw(l + 1), lh(l + 1), nullptr, false); ref128 = img("enc" + std::to_string(l) + "-128"); }
        else s128 = makeImage(1, 1, nullptr, false);
        t96 = t104 = makeImage(1, 1, nullptr, false);
    } else if (si == 10) {   // dec0: surface 144 (RGBA8 SNORM, r0 / r1 by frame parity) in the S120 slot
        const auto d88 = img("dec1-112");
        t88 = makeImage(lw(1), lh(1), &d88, true);
        const auto d96 = img("enc0-112"), d104 = img("enc0-120");
        t96 = makeImage(lw(0), lh(0), &d96, true);
        t104 = makeImage(lw(0), lh(0), &d104, true);
        s112 = makeImage(lw(0), lh(0), nullptr, false); ref112 = img("dec0-112");
        s120 = makeImage(rw, rh, nullptr, false, true); ref120 = img(std::string("r") + char('0' + ((std::atoi(frame.c_str()) - 1) & 1)));
        s128 = makeImage(1, 1, nullptr, false);
    } else {
        const auto d88 = img(l == 4 ? "enc5-112" : "dec" + std::to_string(l + 1) + "-112");
        t88 = makeImage(lw(l + 1), lh(l + 1), &d88, true);
        const auto d96 = img("enc" + std::to_string(l) + "-112"), d104 = img("enc" + std::to_string(l) + "-120");
        t96 = makeImage(lw(l), lh(l), &d96, true);
        t104 = makeImage(lw(l), lh(l), &d104, true);
        s112 = makeImage(lw(l), lh(l), nullptr, false); ref112 = img("dec" + std::to_string(l) + "-112");
        s120 = s128 = makeImage(1, 1, nullptr, false);
    }
    // the descriptor table: T88 T96 T104 S112 S120 S128 (8 dwords each), the linear sampler (4)
    std::vector<uint32_t> desc(56, 0);
    auto copySrd = [&](const void* obj, uint32_t at, uint32_t dwords, uint32_t from) {
        CK(hipMemcpy(&desc[at], static_cast<const uint32_t*>(obj) + from, dwords * 4, hipMemcpyDeviceToHost));
    };
    copySrd(t88.tex, 0, 8, 0); copySrd(t96.tex, 8, 8, 0); copySrd(t104.tex, 16, 8, 0);
    copySrd(s112.surf, 24, 8, 0); copySrd(s120.surf, 32, 8, 0); copySrd(s128.surf, 40, 8, 0);
    copySrd(t88.tex, 48, 4, HIP_SAMPLER_OBJECT_OFFSET_DWORD);

    // buffers: each with a guard tail (the kernel's own accesses are bounds-checked by buffer descriptors)
    void *dParam, *dWeights, *dArena, *dDesc;
    CK(hipMalloc(&dParam, 4096));
    CK(hipMemcpy(dParam, param.data(), param.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dWeights, weights.size() + 4096));
    CK(hipMemcpy(dWeights, weights.data(), weights.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dArena, before.size() + 4096));
    CK(hipMemcpy(dArena, before.data(), before.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dDesc, desc.size() * 4));
    CK(hipMemcpy(dDesc, desc.data(), desc.size() * 4, hipMemcpyHostToDevice));

    hipModule_t module;
    hipFunction_t fn;
    CK(hipModuleLoad(&module, hsaco.c_str()));
    uint32_t block = 512;   // rr_swin_wide.hip's sixteen-wave kernel, else rr_swin.hip's / rr_dec0.hip's four waves
    if (hipModuleGetFunction(&fn, module, "rr_swin_w16") != hipSuccess) {
        block = 128;
        if (hipModuleGetFunction(&fn, module, "rr_dec0") != hipSuccess) CK(hipModuleGetFunction(&fn, module, "rr_swin"));
    }
    RrArgs args{static_cast<const uint32_t*>(dParam), dWeights, static_cast<uint32_t*>(dArena), static_cast<const uint32_t*>(dDesc),
                static_cast<uint32_t>(weights.size()), static_cast<uint32_t>(before.size()), nullptr, 0, 0};
    size_t argBytes = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &argBytes, HIP_LAUNCH_PARAM_END};
    const uint32_t gx = (lw(l) + sx + 7) / 8, gy = (lh(l) + sy + 7) / 8;
    auto launch = [&] { CK(hipModuleLaunchKernel(fn, gx, gy, 1, block, 1, 1, 0, nullptr, nullptr, config)); };
    launch();
    CK(hipDeviceSynchronize());

    std::vector<uint8_t> arena(before.size());
    CK(hipMemcpy(arena.data(), dArena, arena.size(), hipMemcpyDeviceToHost));
    if (const char* d = std::getenv("RR_HIP_DUMP"))
        std::ofstream(std::string(d) + "/" + stage + "-arena.bin", std::ios::binary)
            .write(reinterpret_cast<const char*>(arena.data()), static_cast<std::streamsize>(arena.size()));
    if (std::getenv("RR_HIP_STAMPS")) {   // a timing build's s_memrealtime stamps (100 MHz), window (0, 0)
        const char* names[8] = {"start", "prologue", "attention", "norm", "mlp", "outputs", "end", "pre"};
        uint64_t st[8];
        std::memcpy(st, arena.data() + arena.size() - 128, sizeof(st));
        std::printf("stamps (us from start):");
        for (int i = 1; i < 8; ++i) if (st[i]) std::printf(" %s %.1f", names[i], (st[i] - st[0]) / 100.0);
        std::printf("\n");
        uint64_t gs[8];
        std::memcpy(gs, arena.data() + arena.size() - 64, sizeof(gs));
        std::printf("mlp groups (us from start):");
        for (int i = 0; i < 8; ++i) if (gs[i]) std::printf(" %.1f", (gs[i] - st[0]) / 100.0);
        std::printf("\n");
    }
    size_t first;
    size_t n = diffBytes(arena, after, first);
    std::printf("%s frame %s grid %ux%u: arena %zu bytes differ", stage.c_str(), frame.c_str(), gx, gy, n);
    if (n) std::printf(" (first at %zu: %02x vs %02x)", first, arena[first], after[first]);
    std::printf("\n");
    // RR_HIP_DUMP=DIR: the HIP run's surfaces as DIR/STAGE-NAME.bin
    const char* dumpDir = std::getenv("RR_HIP_DUMP");
    auto check = [&](const char* name, const Image& im, const std::vector<uint8_t>& want) {
        if (want.empty()) return;
        const auto got = readImage(im);
        if (im.texel == 4) {   // RGBA8: every byte
            const size_t d = diffBytes(got, want, first);
            std::printf("  surface %s (rgba8): %zu of %zu bytes differ\n", si == 10 ? "144" : name, d, got.size());
            return;
        }
        if (dumpDir)
            std::ofstream(std::string(dumpDir) + "/" + stage + "-" + name + ".bin", std::ios::binary)
                .write(reinterpret_cast<const char*>(got.data()), static_cast<std::streamsize>(got.size()));
        const size_t d = diffRGB(got, want, first);
        std::printf("  surface %s: %zu of %zu values differ", name, d, got.size() / 8 * 3);
        if (d) {
            uint16_t g, w;
            std::memcpy(&g, &got[first * 2], 2); std::memcpy(&w, &want[first * 2], 2);
            std::printf(" (first: texel %zu ch %zu: %04x vs %04x)", first / 4, first % 4, g, w);
        }
        std::printf("\n");
    };
    check("112", s112, ref112);
    check("120", s120, ref120);
    check("128", s128, ref128);

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    float best = 1e30f;
    // RR_HIP_COLD=1: 128 MiB written before every timed launch (the stage's weights leave the caches, as between the
    // engine's stages)
    void* flush = nullptr;
    if (std::getenv("RR_HIP_COLD")) CK(hipMalloc(&flush, 128u << 20));
    for (int r = 0; r < repeat; ++r) {
        if (flush) CK(hipMemsetAsync(flush, r & 0xff, 128u << 20, nullptr));
        CK(hipEventRecord(e0));
        launch();
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms;
        CK(hipEventElapsedTime(&ms, e0, e1));
        if (ms < best) best = ms;
    }
    std::printf("  time: %.1f us (best of %d)\n", best * 1000.0f, repeat);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "rr_hip_test: %s\n", e.what());
    return 1;
}
