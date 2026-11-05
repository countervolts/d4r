// d3dhip: the Ray Reconstruction network on HIP. See hip_net_rr.h.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include "hip_net_rr.h"
#include "hip_net_d3d12.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <hip/hip_runtime_api.h>
#include <wrl/client.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace d4r::dx {
namespace {
void hipCheck(hipError_t error, const char* what) {
    if (error != hipSuccess) throw std::runtime_error(std::string("HIP: ") + what + ": " + hipGetErrorString(error));
}

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// The stage kernels' parameter block, by value: kernels/rr/rrvk_swin.hip's Args (shared with the Vulkan vkhip route,
// which passes the address of this block as a push constant).
struct RrArgs {
    uint64_t arena, w;
    int32_t ex, ey;
    uint32_t inOff, skOff, outOff, mrgOff, shift;
    float iw, ih;
    uint32_t biasF;
    alignas(32) int32_t t88[8];
    int32_t t96[8], t104[8], s112[8], s120[8], s128[8];
    alignas(16) int32_t linear[4];
};
static_assert(offsetof(RrArgs, t88) == 64 && offsetof(RrArgs, linear) == 256 && sizeof(RrArgs) == 288, "rr stage ABI");

// The online weights: the model's packed stage weights with the V tiles transposed in place and the attention bias
// converted to f32 [head][query][key] after them (as engine/vkhip/vkhip_net.cpp does for vkhip, from
// kernels/rr/rrvk_swin.hip's layout global rrvk_layout: C, ..., NH, NW, wv0, bias0, head).
uint32_t halfToFloatBits(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 31u, m = h & 1023u;
    float v = e == 0 ? std::ldexp(float(m), -24) : (e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(m | 1024u), int(e) - 25));
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return bits | sign;
}

constexpr uint32_t SamplerClamp = 0x00000049u;   // CLAMP on all three axes
constexpr uint32_t SamplerLinear = 0x00100000u;  // gfx12's FILTER_MODE = bilinear in the sampler descriptor's word 2

constexpr uint64_t AllocationGranularity = 65536;
uint64_t alignedBytes(uint64_t bytes) {
    return std::max<uint64_t>((bytes + AllocationGranularity - 1) / AllocationGranularity * AllocationGranularity, AllocationGranularity);
}

// A pitch-linear image in HIP memory with the hardware descriptor the stage kernels load, sample and store it with.
// HIP builds the descriptor for a pitched 2D resource; the model's kernels take it raw (llvm.amdgcn.image.*), so the
// 8 dwords of its texture object are copied out. Format: RGBA16F or RGBA8_SNORM.
struct Image {
    hipTextureObject_t object = nullptr;
    void* memory = nullptr;
    uint64_t bytes = 0;
    uint32_t width = 0, height = 0, pitch = 0, pixelBytes = 0;
    uint32_t t[8]{};
    ~Image() {
        if (object) (void)hipTexObjectDestroy(object);
        if (memory) (void)hipFree(memory);
    }
    Image() = default;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
};

uint32_t pitchFor(uint32_t width, uint32_t pixelBytes) { return (width * pixelBytes + 255) / 256 * 256; }

void createImage(Image& image, uint32_t width, uint32_t height, hipArray_Format format, uint32_t channels) {
    image.width = width;
    image.height = height;
    image.pixelBytes = channels * (format == HIP_AD_FORMAT_HALF ? 2 : 1);
    image.pitch = pitchFor(width, image.pixelBytes);
    image.bytes = uint64_t(image.pitch) * height;
    hipCheck(hipMalloc(&image.memory, size_t(image.bytes)), "hipMalloc(image)");
    HIP_RESOURCE_DESC desc{};
    desc.resType = HIP_RESOURCE_TYPE_PITCH2D;
    desc.res.pitch2D.devPtr = image.memory;
    desc.res.pitch2D.format = format;
    desc.res.pitch2D.numChannels = channels;
    desc.res.pitch2D.width = width;
    desc.res.pitch2D.height = height;
    desc.res.pitch2D.pitchInBytes = image.pitch;
    HIP_TEXTURE_DESC texture{};
    texture.addressMode[0] = texture.addressMode[1] = texture.addressMode[2] = HIP_TR_ADDRESS_MODE_CLAMP;
    texture.filterMode = HIP_TR_FILTER_MODE_POINT;
    texture.mipmapFilterMode = HIP_TR_FILTER_MODE_POINT;
    hipCheck(hipTexObjectCreate(&image.object, &desc, &texture, nullptr), "hipTexObjectCreate(image)");
    // the first 8 dwords of the object are the image resource descriptor (the sampler follows at 12)
    std::memcpy(image.t, image.object, sizeof(image.t));
}

constexpr uint32_t RGBA16FChannels = 4;

// A D3D12 buffer shared with the network as an opaque NT handle and the HIP image it stages into.
struct Staging {
    ComPtr<ID3D12Resource> resource;
    void* imported = nullptr;
    void* external = nullptr;
    Image image;
    uint32_t width = 0, height = 0, pitch = 0, pixelBytes = 0;
};

void createStaging(ID3D12Device* device, Staging& staging, uint32_t width, uint32_t height, hipArray_Format format,
                   uint32_t channels) {
    staging.width = width;
    staging.height = height;
    staging.pixelBytes = channels * (format == HIP_AD_FORMAT_HALF ? 2 : 1);
    staging.pitch = pitchFor(width, staging.pixelBytes);
    staging.resource = createSharedBuffer(device, uint64_t(staging.pitch) * height);
    staging.imported = importSharedBuffer(device, staging.resource.Get(), uint64_t(staging.pitch) * height, &staging.external);
    createImage(staging.image, width, height, format, channels);
    if (staging.image.pitch != staging.pitch)
        throw std::runtime_error("the staged image's row pitch differs from the shared buffer's");
}

// The presets D and E network's stages in network order (engine/rr_frame_parameters.h RNames' enums, the code objects
// kernels/rr/build_vkhip.sh builds): encoder 0 and the output stage stay on the engine.
constexpr const char* DeStages[] = {"enc1", "enc2", "enc3", "enc4", "enc5", "dec4", "dec3", "dec2", "dec1", "dec0"};
// Preset F's (engine/runtime_rr2.inc R2Names indices 1..14, kernels/rr/build_vkhip_f.sh).
constexpr const char* FStages[] = {"enc1", "enc2", "enc3", "enc4", "enc5", "dec4_0", "dec4_1", "dec3_0", "dec3_1",
                                   "dec2_0", "dec2_1", "dec1_0", "dec1_1", "dec0_0"};
constexpr unsigned DeCount = sizeof(DeStages) / sizeof(DeStages[0]);
constexpr unsigned FCount = sizeof(FStages) / sizeof(FStages[0]);

// "enc3" and "dec4_1" name their level (the network's RLevel / R2Level)
int levelOf(const std::string& name) {
    const size_t digits = name.find_first_of("0123456789");
    if (digits == std::string::npos) return -1;
    return std::atoi(name.c_str() + digits);
}
bool isEncoder(const std::string& name) { return name.compare(0, 3, "enc") == 0; }
} // namespace

struct HipNetworkRr::Impl {
    bool presetF = false;
    uint32_t w0 = 0, h0 = 0;
    uint32_t frame = 0;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    float milliseconds = 0;
    std::vector<hipModule_t> modules;
    void* arena = nullptr;
    void* arenaExternal = nullptr;

    // presets D and E: the images (the three the engine stages in and the three it takes back are separate)
    std::vector<std::unique_ptr<Image>> owned;
    Image* s112[6]{};
    Image* s120[6]{};
    Image* s128[5]{};
    Image* d112[5]{};
    Image* d120[5]{};
    Staging staging[6];   // 0..2 the front's (surface 128/112/120 level 0), 3..5 the back's (d112[0], r[0], r[1])
    struct DeStage {
        std::string name;
        unsigned index = 0;
        int level = 0;
        bool encoder = false, decoder0 = false;
        hipFunction_t function = nullptr;
        uint32_t blockZ = 1;
        void* weights = nullptr;
        uint32_t biasF = 0;
        uint32_t ex = 0, ey = 0;             // this frame's level dimensions (from the engine's block)
        uint64_t in = 0, skip = 0, out = 0, merge = 0;
        uint32_t gx = 0, gy = 0, shift = 0;
        const Image* t88 = nullptr;
        const Image* t96 = nullptr;
        const Image* t104 = nullptr;
        const Image* surface112 = nullptr;
        const Image* surface120[2]{};        // decoder 0's surface 144 per frame parity
        const Image* surface128 = nullptr;
    };
    std::vector<DeStage> de;

    // preset F: one kernel per network stage, the stage's parameter block in HIP memory, and its weights
    struct FStage {
        std::string name;
        unsigned index = 0;
        hipFunction_t function = nullptr;
        uint32_t threads = 32;
        void* weights = nullptr;
    };
    std::vector<FStage> f;
    // preset F's parameter areas and graphs, one per frame parity (see hip_net_rr.h)
    struct FSlot {
        void* device = nullptr;              // the stages' blocks at `offsets`
        void* host = nullptr;                // pinned staging for the copy into `device`
        size_t bytes = 0;
        std::vector<uint8_t> held;           // what `device` holds (or is about to)
        std::vector<uint64_t> key;           // per stage: index, grid, block offset (the graph's launches)
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        hipEvent_t done = nullptr;           // after the last launch reading `host`
        bool pending = false;
    };
    FSlot slots[2];
    uint32_t slot = 0;
    hipExternalSemaphore_t fence = nullptr;
    ID3D12Device* device = nullptr;

    struct Queued {
        unsigned index = 0;
        uint32_t gx = 0, gy = 0, shift = 0;
        uint32_t ex = 0, ey = 0;
        uint64_t in = 0, skip = 0, out = 0, merge = 0;
        std::vector<uint8_t> block;
    };
    std::vector<Queued> queued;

    ~Impl() {
        if (stream) (void)hipStreamSynchronize(stream);
        for (auto& s : slots) {
            if (s.exec) (void)hipGraphExecDestroy(s.exec);
            if (s.graph) (void)hipGraphDestroy(s.graph);
            if (s.device) (void)hipFree(s.device);
            if (s.host) (void)hipHostFree(s.host);
            if (s.done) (void)hipEventDestroy(s.done);
        }
        if (fence) (void)hipDestroyExternalSemaphore(fence);
        for (auto& image : owned) image.reset();
        for (auto& stage : de)
            if (stage.weights) (void)hipFree(stage.weights);
        for (auto& stage : f) {
            if (stage.weights) (void)hipFree(stage.weights);
        }
        for (auto module : modules) (void)hipModuleUnload(module);
        if (arenaExternal) (void)hipDestroyExternalMemory(arenaExternal);
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
        if (stream) (void)hipStreamDestroy(stream);
    }

    Image& image(uint32_t width, uint32_t height, hipArray_Format format, uint32_t channels) {
        owned.push_back(std::make_unique<Image>());
        createImage(*owned.back(), width, height, format, channels);
        return *owned.back();
    }

    void launch(const DeStage& stage, const Queued& q) {
        RrArgs args{};
        args.arena = uint64_t(reinterpret_cast<uintptr_t>(arena));
        args.w = uint64_t(reinterpret_cast<uintptr_t>(stage.weights));
        args.ex = int32_t(q.ex);
        args.ey = int32_t(q.ey);
        args.inOff = uint32_t(q.in);
        args.skOff = uint32_t(q.skip);
        args.outOff = uint32_t(q.out);
        args.mrgOff = uint32_t(q.merge);
        args.shift = q.shift;
        args.iw = q.ex ? 1.0f / float(q.ex) : 0.0f;
        args.ih = q.ey ? 1.0f / float(q.ey) : 0.0f;
        args.biasF = stage.biasF;
        auto set = [](int32_t (&dst)[8], const Image* from) {
            if (from) std::memcpy(dst, from->t, sizeof(dst));
            else std::memset(dst, 0, sizeof(dst));
        };
        set(args.t88, stage.t88);
        set(args.t96, stage.t96);
        set(args.t104, stage.t104);
        set(args.s112, stage.surface112);
        // decoder 0's surface 144 is the frame parity's: it takes the odd shift on odd frames
        set(args.s120, stage.surface120[stage.decoder0 && q.shift == 0x00060002u ? 1 : 0]);
        set(args.s128, stage.surface128);
        args.linear[0] = int32_t(SamplerClamp);
        args.linear[1] = 0;
        args.linear[2] = int32_t(SamplerLinear);
        args.linear[3] = 0;
        void* arguments[] = {&args};
        hipCheck(hipModuleLaunchKernel(stage.function, q.gx, q.gy, 1, 32, 1, stage.blockZ, 0, stream, arguments, nullptr),
                 "hipModuleLaunchKernel(rr)");
    }

    // The queued stages' blocks into this frame parity's area (copied on the stream only when they changed) and its
    // graph (recaptured when the launches changed). Returns the graph to launch on `stream`.
    hipGraphExec_t prepareF() {
        FSlot& s = slots[slot];
        slot ^= 1;
        std::vector<uint8_t> staged;
        std::vector<uint64_t> key;
        for (const auto& q : queued) {
            const uint64_t offset = staged.size();
            staged.insert(staged.end(), q.block.begin(), q.block.end());
            staged.resize((staged.size() + 255) / 256 * 256);
            key.insert(key.end(), {q.index, q.gx, q.gy, offset});
        }
        if (staged.size() > s.bytes) {
            if (s.pending) hipCheck(hipEventSynchronize(s.done), "hipEventSynchronize(param area)");
            hipCheck(hipStreamSynchronize(stream), "hipStreamSynchronize(param area)");
            if (s.device) (void)hipFree(s.device);
            if (s.host) (void)hipHostFree(s.host);
            s.device = s.host = nullptr;
            s.bytes = staged.size();
            hipCheck(hipMalloc(&s.device, s.bytes), "hipMalloc(param area)");
            hipCheck(hipHostMalloc(&s.host, s.bytes, hipHostMallocDefault), "hipHostMalloc(param area)");
            if (!s.done) hipCheck(hipEventCreateWithFlags(&s.done, hipEventDisableTiming), "hipEventCreate(param area)");
            s.held.clear();
            s.key.clear();
        }
        if (staged != s.held) {
            if (s.pending) hipCheck(hipEventSynchronize(s.done), "hipEventSynchronize(param area)");
            std::memcpy(s.host, staged.data(), staged.size());
            hipCheck(hipMemcpyAsync(s.device, s.host, staged.size(), hipMemcpyHostToDevice, stream), "hipMemcpyAsync(param area)");
            hipCheck(hipEventRecord(s.done, stream), "hipEventRecord(param area)");
            s.pending = true;
            s.held = std::move(staged);
        }
        if (key != s.key) {
            if (s.exec) (void)hipGraphExecDestroy(s.exec);
            if (s.graph) (void)hipGraphDestroy(s.graph);
            s.exec = nullptr;
            s.graph = nullptr;
            hipCheck(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "hipStreamBeginCapture");
            for (size_t i = 0; i < queued.size(); ++i)
                launchF(f[queued[i].index - 1], queued[i], static_cast<uint8_t*>(s.device) + key[4 * i + 3]);
            hipCheck(hipStreamEndCapture(stream, &s.graph), "hipStreamEndCapture");
            hipCheck(hipGraphInstantiate(&s.exec, s.graph, nullptr, nullptr, 0), "hipGraphInstantiate");
            s.key = std::move(key);
        }
        return s.exec;
    }

    void launchF(FStage& stage, const Queued& q, void* params) {
        // the kernel's one by-value argument: the parameter block, its weights and the arena (vkhip's kernarg layout)
        struct { uint64_t param, weights, arena; } args{uint64_t(reinterpret_cast<uintptr_t>(params)),
                                                       uint64_t(reinterpret_cast<uintptr_t>(stage.weights)),
                                                       uint64_t(reinterpret_cast<uintptr_t>(arena))};
        void* arguments[] = {&args};
        hipCheck(hipModuleLaunchKernel(stage.function, q.gx, q.gy, 1, stage.threads, 1, 1, 0, stream, arguments, nullptr),
                 "hipModuleLaunchKernel(rr2)");
    }
};

HipNetworkRr::HipNetworkRr(ID3D12Device* device, const std::string& modelDirectory, bool presetF, uint32_t renderWidth,
                           uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight, ID3D12Resource* arena,
                           uint64_t arenaBytes)
    : impl(std::make_unique<Impl>()) {
    auto& n = *impl;
    if (!device) throw std::runtime_error("no D3D12 device");
    if (!renderWidth || !renderHeight || renderWidth > outputWidth || renderHeight > outputHeight)
        throw std::runtime_error("unsupported Ray Reconstruction dimensions");
    if (!arena || !arenaBytes) throw std::runtime_error("the Ray Reconstruction network needs the engine's arena");
    const std::string kernels = modelDirectory + "/hip";
    if (!std::filesystem::is_directory(kernels)) throw std::runtime_error("the model has no native HIP network (" + kernels + ")");
    checkHipAdapter(device);

    n.presetF = presetF;
    n.device = device;
    n.w0 = (renderWidth + 31) / 32 * 32;
    n.h0 = (renderHeight + 31) / 32 * 32;
    hipCheck(hipInit(0), "hipInit");
    int least = 0, greatest = 0;
    hipCheck(hipDeviceGetStreamPriorityRange(&least, &greatest), "hipDeviceGetStreamPriorityRange");
    hipCheck(hipStreamCreateWithPriority(&n.stream, hipStreamNonBlocking, greatest), "hipStreamCreateWithPriority");
    hipCheck(hipEventCreate(&n.start), "hipEventCreate");
    hipCheck(hipEventCreate(&n.stop), "hipEventCreate");
    n.arena = importSharedBuffer(device, arena, arenaBytes, &n.arenaExternal);

    const auto lw = [&](int l) { return n.w0 >> l; };
    const auto lh = [&](int l) { return n.h0 >> l; };

    if (!presetF) {
        // The images of the ten stages. The three the engine fills are staged into HIP images; so are the three it
        // takes back. Level l is (w0 >> l) x (h0 >> l) texels; surface 128 holds the 2x2 merge of level l + 1.
        const uint32_t half4 = RGBA16FChannels;
        for (int i = 0; i < 3; ++i) {
            const uint32_t w = i == 0 ? lw(1) : lw(0), h = i == 0 ? lh(1) : lh(0);
            createStaging(device, n.staging[i], w, h, HIP_AD_FORMAT_HALF, half4);
        }
        createStaging(device, n.staging[3], lw(0), lh(0), HIP_AD_FORMAT_HALF, half4);
        // decoder 0's surface 144 is the engine's r[par] texture: the render size, not the 32-aligned one
        for (int i = 4; i < 6; ++i)
            createStaging(device, n.staging[i], renderWidth, renderHeight, HIP_AD_FORMAT_SIGNED_INT8, 4);
        // the engine's images become HIP images through the staging copies
        n.s112[0] = &n.staging[1].image;
        n.s120[0] = &n.staging[2].image;
        n.s128[0] = &n.staging[0].image;
        n.d112[0] = &n.staging[3].image;
        for (int l = 1; l < 6; ++l) {
            n.s112[l] = &n.image(lw(l), lh(l), HIP_AD_FORMAT_HALF, half4);
            n.s120[l] = &n.image(lw(l), lh(l), HIP_AD_FORMAT_HALF, half4);
        }
        for (int l = 1; l < 5; ++l) {
            n.s128[l] = &n.image(lw(l + 1), lh(l + 1), HIP_AD_FORMAT_HALF, half4);
            n.d112[l] = &n.image(lw(l), lh(l), HIP_AD_FORMAT_HALF, half4);
            n.d120[l] = &n.image(lw(l), lh(l), HIP_AD_FORMAT_HALF, half4);
        }
        // the stages: kernels + weights (weights-<stage>m.bin, rearranged as vkhip does)
        n.de.reserve(DeCount);
        for (unsigned i = 0; i < DeCount; ++i) {
            Impl::DeStage stage;
            stage.name = DeStages[i];
            stage.index = i + 1;
            stage.level = levelOf(stage.name);
            stage.encoder = isEncoder(stage.name);
            stage.decoder0 = stage.name == "dec0";
            if (!stage.encoder && stage.level < 0) throw std::runtime_error("invalid stage " + stage.name);
            const std::string file = kernels + "/rr_" + stage.name + ".hsaco";
            auto code = readFile(file);
            hipModule_t module = nullptr;
            hipCheck(hipModuleLoadData(&module, code.data()), ("hipModuleLoadData(" + stage.name + ")").c_str());
            n.modules.push_back(module);
            hipCheck(hipModuleGetFunction(&stage.function, module, ("rr_" + stage.name).c_str()),
                     ("hipModuleGetFunction(" + stage.name + ")").c_str());
            uint32_t layout[8]{};
            hipDeviceptr_t global = nullptr;
            size_t bytes = 0;
            if (hipModuleGetGlobal(&global, &bytes, module, "rrvk_layout") != hipSuccess || bytes != sizeof(layout))
                throw std::runtime_error("rr_" + stage.name + ".hsaco has no rrvk_layout");
            hipCheck(hipMemcpyDtoH(layout, global, sizeof(layout)), "hipMemcpyDtoH(rrvk_layout)");
            stage.blockZ = layout[4] ? layout[4] : 1;
            auto weights = readFile(modelDirectory + "/weights-" + stage.name + "m.bin");
            const uint32_t C = layout[1], NH = layout[3], wv0 = layout[5], bias0 = layout[6], head = layout[7];
            if (!C || !NH || !head || !layout[4] ||
                uint64_t(bias0) + uint64_t(head) * (NH - 1) + 96 * 64 * 2 > weights.size() ||
                uint64_t(wv0) + uint64_t(head) * (NH - 1) + 256 * (C / 8) > weights.size())
                throw std::runtime_error("rr_" + stage.name + ": weights do not match its layout");
            for (uint32_t hd = 0; hd < NH; ++hd) {
                uint8_t* v = weights.data() + wv0 + head * hd;
                for (uint32_t t = 0; t < (C / 16) * 2; ++t)
                    for (int r = 0; r < 16; ++r)
                        for (int c = 0; c < r; ++c) std::swap(v[256 * t + 16 * r + c], v[256 * t + 16 * c + r]);
            }
            stage.biasF = uint32_t((weights.size() + 255) / 256 * 256);
            weights.resize(size_t(stage.biasF) + size_t(NH) * 96 * 64 * 4);
            for (uint32_t hd = 0; hd < NH; ++hd) {
                uint16_t bias[96 * 64];
                std::memcpy(bias, weights.data() + bias0 + head * hd, sizeof(bias));
                uint32_t* out = reinterpret_cast<uint32_t*>(weights.data() + stage.biasF) + hd * 96 * 64;
                for (int key = 0; key < 96; ++key)
                    for (int q = 0; q < 64; ++q) out[q * 96 + key] = halfToFloatBits(bias[key * 64 + q]);
            }
            hipCheck(hipMalloc(&stage.weights, weights.size()), "hipMalloc(weights)");
            hipCheck(hipMemcpy(stage.weights, weights.data(), weights.size(), hipMemcpyHostToDevice), "hipMemcpy(weights)");
            // (the module stays loaded: its functions are what the stages launch)
            // the stage's images, by the engine's wiring (runtime_rr.inc REngine::resolve)
            const int l = stage.level;
            if (stage.encoder) {
                stage.t88 = n.s128[l - 1];
                stage.surface112 = n.s112[l];
                stage.surface120[0] = stage.surface120[1] = n.s120[l];
                stage.surface128 = l < 5 ? n.s128[l] : nullptr;
            } else if (stage.name == "dec4") {
                stage.t88 = n.s112[5];
                stage.t96 = n.s112[l];
                stage.t104 = n.s120[l];
                stage.surface112 = n.d112[l];
                stage.surface120[0] = stage.surface120[1] = n.d120[l];
            } else if (stage.name == "dec0") {
                stage.t88 = n.d112[1];
                stage.t96 = n.s112[0];
                stage.t104 = n.s120[0];
                stage.surface112 = n.d112[0];
                stage.surface120[0] = &n.staging[4].image;
                stage.surface120[1] = &n.staging[5].image;
            } else {
                stage.t88 = n.d112[l + 1];
                stage.t96 = n.s112[l];
                stage.t104 = n.s120[l];
                stage.surface112 = n.d112[l];
                stage.surface120[0] = stage.surface120[1] = n.d120[l];
            }
            n.de.push_back(std::move(stage));
        }
    } else {
        // preset F: the fourteen kernels and their weights (the V tiles transposed first, as vkhip's F route)
        n.f.reserve(FCount);
        for (unsigned i = 0; i < FCount; ++i) {
            Impl::FStage stage;
            stage.name = FStages[i];
            stage.index = i + 1;
            const std::string code_name = "rr2_" + stage.name;
            auto code = readFile(kernels + "/" + code_name + ".hsaco");
            hipModule_t module = nullptr;
            hipCheck(hipModuleLoadData(&module, code.data()), ("hipModuleLoadData(" + code_name + ")").c_str());
            n.modules.push_back(module);
            hipCheck(hipModuleGetFunction(&stage.function, module, code_name.c_str()),
                     ("hipModuleGetFunction(" + code_name + ")").c_str());
            auto weights = readFile(modelDirectory + "/weights-" + stage.name + "m.bin");
            uint32_t layout[4]{};
            hipDeviceptr_t global = nullptr;
            size_t bytes = 0;
            if (hipModuleGetGlobal(&global, &bytes, module, "rr2_v_layout") == hipSuccess && bytes == sizeof(layout)) {
                hipCheck(hipMemcpyDtoH(layout, global, sizeof(layout)), "hipMemcpyDtoH(rr2_v_layout)");
                const uint32_t count = layout[0], heads = layout[1];
                if (!count || count % 16 || !heads ||
                    uint64_t(layout[2]) + uint64_t(layout[3]) * (heads - 1) + uint64_t(count / 16) * 512 > weights.size())
                    throw std::runtime_error(code_name + ": weights do not match its V layout");
                for (uint32_t h = 0; h < heads; ++h)
                    for (uint32_t t = 0; t < count / 16 * 2; ++t) {
                        uint8_t* tile = weights.data() + layout[2] + layout[3] * h + 256 * t;
                        for (int r = 0; r < 16; ++r)
                            for (int c = 0; c < r; ++c) std::swap(tile[16 * r + c], tile[16 * c + r]);
                    }
            }
            // the kernel's own weight offsets are compile-time constants: keep a zero tail past the file, as vkhip's
            // carve did for these stages
            const size_t tail = 65536;
            hipCheck(hipMalloc(&stage.weights, weights.size() + tail), "hipMalloc(weights)");
            hipCheck(hipMemset(stage.weights, 0, weights.size() + tail), "hipMemset(weights)");
            hipCheck(hipMemcpy(stage.weights, weights.data(), weights.size(), hipMemcpyHostToDevice), "hipMemcpy(weights)");
            uint32_t threads = 32;
            if (hipModuleGetGlobal(&global, &bytes, module, "rr2_threads") == hipSuccess && bytes == 4)
                hipCheck(hipMemcpyDtoH(&threads, global, 4), "hipMemcpyDtoH(rr2_threads)");
            if (threads < 32 || threads > 1024 || threads % 32) throw std::runtime_error(code_name + ": invalid workgroup size");
            stage.threads = threads;
            n.f.push_back(std::move(stage));
        }
    }

    // the engine's half of the interface
    link.network = this;
    if (!presetF) {
        for (int i = 0; i < 3; ++i) {
            link.front[i] = {n.staging[i].resource.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT, n.staging[i].width,
                                  n.staging[i].height, n.staging[i].pitch};
            const int back = i + 3;
            link.back[i] = {n.staging[back].resource.Get(),
                                 i == 0 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_SNORM,
                                 n.staging[back].width, n.staging[back].height, n.staging[back].pitch};
        }
    }
}
HipNetworkRr::~HipNetworkRr() = default;

const ExternalNetworkRr& HipNetworkRr::external() const { return link; }
bool HipNetworkRr::runs(unsigned index) const {
    return impl->presetF ? index >= 1 && index <= FCount : index >= 1 && index <= DeCount;
}
void HipNetworkRr::stage(unsigned index, const void* block, size_t bytes, uint32_t gx, uint32_t gy, uint32_t shift) {
    auto& n = *impl;
    if (!runs(index)) throw std::runtime_error("the network has no stage " + std::to_string(index));
    if (!block || !bytes) throw std::runtime_error("a network stage needs its parameter block");
    Impl::Queued q;
    q.index = index;
    q.gx = gx;
    q.gy = gy;
    q.shift = shift;
    if (n.presetF) {
        q.block.assign(static_cast<const uint8_t*>(block), static_cast<const uint8_t*>(block) + bytes);
    } else {
        // the engine's stage block: the level's dimensions and the arena's input, skip, output and merge regions
        if (bytes < 64) throw std::runtime_error("unexpected Ray Reconstruction stage block");
        const uint8_t* b = static_cast<const uint8_t*>(block);
        std::memcpy(&q.ex, b + 0, 4);
        std::memcpy(&q.ey, b + 4, 4);
        std::memcpy(&q.in, b + 8, 8);
        std::memcpy(&q.skip, b + 24, 8);
        std::memcpy(&q.out, b + 48, 8);
        std::memcpy(&q.merge, b + 56, 8);
    }
    n.queued.push_back(std::move(q));
}
void HipNetworkRr::run() {
    auto& n = *impl;
    if (n.queued.empty()) throw std::runtime_error("no Ray Reconstruction stage was recorded");
    hipCheck(hipEventRecord(n.start, n.stream), "hipEventRecord");
    if (n.presetF) {
        hipCheck(hipGraphLaunch(n.prepareF(), n.stream), "hipGraphLaunch(rr2)");
    } else {
        // the engine's three staged images into HIP's
        for (int i = 0; i < 3; ++i) {
            const Staging& s = n.staging[i];
            hipCheck(hipMemcpy2DAsync(s.image.memory, s.image.pitch, s.imported, s.pitch, uint64_t(s.width) * s.pixelBytes,
                                      s.height, hipMemcpyDeviceToDevice, n.stream),
                     "hipMemcpy2DAsync(front)");
        }
        for (const auto& q : n.queued) n.launch(n.de[q.index - 1], q);
        // and the results back into the engine's buffers
        for (int i = 3; i < 6; ++i) {
            const Staging& s = n.staging[i];
            hipCheck(hipMemcpy2DAsync(s.imported, s.pitch, s.image.memory, s.image.pitch, uint64_t(s.width) * s.pixelBytes,
                                      s.height, hipMemcpyDeviceToDevice, n.stream),
                     "hipMemcpy2DAsync(back)");
        }
    }
    hipCheck(hipEventRecord(n.stop, n.stream), "hipEventRecord");
    hipCheck(hipStreamSynchronize(n.stream), "hipStreamSynchronize");
    float milliseconds = 0;
    hipCheck(hipEventElapsedTime(&milliseconds, n.start, n.stop), "hipEventElapsedTime");
    n.milliseconds = milliseconds;
    ++n.frame;
    n.queued.clear();
}
void HipNetworkRr::useFence(ID3D12Fence* fence) {
    if (!impl->presetF) throw std::runtime_error("the shared fence is preset F's");
    if (!fence) throw std::runtime_error("no shared fence");
    HANDLE handle = nullptr;
    if (FAILED(impl->device->CreateSharedHandle(fence, nullptr, GENERIC_ALL, nullptr, &handle)))
        throw std::runtime_error("CreateSharedHandle failed for the network's fence (create it with D3D12_FENCE_FLAG_SHARED)");
    hipExternalSemaphoreHandleDesc desc{};
    desc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.handle = handle;
    hipExternalSemaphore_t imported = nullptr;
    const hipError_t result = hipImportExternalSemaphore(&imported, &desc);
    CloseHandle(handle);   // the import holds its own reference
    hipCheck(result, "hipImportExternalSemaphore (D3D12 fence)");
    if (impl->fence) (void)hipDestroyExternalSemaphore(impl->fence);
    impl->fence = imported;
}
void HipNetworkRr::launchAfter(uint64_t wait, uint64_t signal) {
    auto& n = *impl;
    if (!n.fence) throw std::runtime_error("launchAfter() needs useFence() first");
    if (n.queued.empty()) throw std::runtime_error("no Ray Reconstruction stage was recorded");
    // the parameter copy (when there is one) goes ahead of the wait: the blocks are the network's own memory
    const hipGraphExec_t graph = n.prepareF();
    hipExternalSemaphoreWaitParams waitParams{};
    waitParams.params.fence.value = wait;
    hipCheck(hipWaitExternalSemaphoresAsync(&n.fence, &waitParams, 1, n.stream), "hipWaitExternalSemaphoresAsync");
    hipCheck(hipEventRecord(n.start, n.stream), "hipEventRecord");
    hipCheck(hipGraphLaunch(graph, n.stream), "hipGraphLaunch(rr2)");
    hipCheck(hipEventRecord(n.stop, n.stream), "hipEventRecord");
    hipExternalSemaphoreSignalParams signalParams{};
    signalParams.params.fence.value = signal;
    hipCheck(hipSignalExternalSemaphoresAsync(&n.fence, &signalParams, 1, n.stream), "hipSignalExternalSemaphoresAsync");
    ++n.frame;
    n.queued.clear();
}
void HipNetworkRr::finish() {
    auto& n = *impl;
    hipCheck(hipStreamSynchronize(n.stream), "hipStreamSynchronize");
    if (n.frame) {
        float milliseconds = 0;
        if (hipEventElapsedTime(&milliseconds, n.start, n.stop) == hipSuccess) n.milliseconds = milliseconds;
    }
}
float HipNetworkRr::lastMilliseconds() { return impl->milliseconds; }
uint32_t HipNetworkRr::frames() const { return impl->frame; }
} // namespace d4r::dx
