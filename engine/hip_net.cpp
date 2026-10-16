// d4r engine: the DLSS network on d4r's native HIP layer kernels. See hip_net.h.
//
// The backend is shared by the two presets and only the plan differs: preset M runs ten layers with the
// kernels/m parameter block (kernels/m/swin_common.h) in two window-alignment phases, preset K runs eleven
// layers with kernels/k/pwin_common.h's PwinParams (the bottleneck layer as its four phase kernels) in the
// eight-frame window cycle of makeKWindows. Everything else - the stream, the immutable captured graphs, the
// events and the per-evaluation frame counter - is the same code.
#define __HIP_PLATFORM_AMD__
#include "hip_net.h"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace d4r {
namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("HIP: ") + what + " failed: " + hipGetErrorString(e));
}
#define HIPCHECK(x) check((x), #x)
std::vector<char> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}
// preset M's kernels' parameter block (kernels/m/swin_common.h: CommonParams, TubeParams). Both blocks are
// plain aggregates: every parameter set is value-initialized (all bytes zero) before its fields are filled.
struct LayerParamsM {
    void* w; int32_t sx, sy, tw, th; void* in; void* skip; void* out; void* merged;
    int32_t r69, r70, r71, r72; void* inFlags; void* outFlags;
};
static_assert(sizeof(LayerParamsM) == 88 && offsetof(LayerParamsM, in) == 24, "M param block");
// preset K's kernels' parameter block (kernels/k/pwin_layer.h: PwinParams)
struct LayerParamsK {
    int32_t W, H; void* in; void* skip; void* out24; void* out32; uint64_t pad40, pad48; int32_t sx, sy; void* w;
    uint8_t rest[104];
};
static_assert(sizeof(LayerParamsK) == 176 && offsetof(LayerParamsK, in) == 8 && offsetof(LayerParamsK, w) == 64, "K param block");
union Params {
    LayerParamsM m;
    LayerParamsK k;
    uint8_t raw[176];
};
// one code object of a layer: the main entry, the weight-image preparation, and the K bottleneck's post phases
struct Kernel {
    hipModule_t module = nullptr;
    hipFunction_t main = nullptr, prep = nullptr;
    hipFunction_t post[8] = {};
    uint32_t prepBlocks = 0, blockZ = 4, gridX = 0, gridY = 0;
    uint32_t postGridX[8] = {}, postGridY[8] = {}, postBlockX[8] = {}, postBlockY[8] = {}, postBlockZ[8] = {};
    int postCount = 0;
};
// one immutable launch: entry point, grid, block and the (frozen) parameter block
struct Node {
    hipFunction_t function = nullptr;
    uint32_t gx = 1, gy = 1, bx = 32, by = 1, bz = 1;
    Params params{};
};
uint32_t global32(hipModule_t module, const char* name, uint32_t fallback) {
    hipDeviceptr_t at = nullptr; size_t bytes = 0; uint32_t v = fallback;
    if (hipModuleGetGlobal(&at, &bytes, module, name) == hipSuccess && bytes == 4) {
        if (hipMemcpyDtoH(&v, at, 4) != hipSuccess || v == 0) v = fallback;
    }
    return v;
}
void* import(int fd, uint64_t bytes, hipExternalMemory_t& memory) {
    hipExternalMemoryHandleDesc desc{}; desc.type = hipExternalMemoryHandleTypeOpaqueFd; desc.handle.fd = fd; desc.size = bytes;
    HIPCHECK(hipImportExternalMemory(&memory, &desc));
    hipExternalMemoryBufferDesc buffer{}; buffer.offset = 0; buffer.size = bytes;
    void* at = nullptr; HIPCHECK(hipExternalMemoryGetMappedBuffer(&at, memory, &buffer));
    return at;
}
// makeKWindows (engine/runtime.cpp): dec5 moves once per frame around an eight-frame cycle and every
// shallower layer follows the one below it as (2 * shift + 4) mod 8. Role 0 = dec5, 1 = enc4/dec4,
// 2 = enc3/dec3; every other layer keeps shift 4.
constexpr int32_t KCycle[8][2] = {{0, 2}, {5, 6}, {4, 0}, {1, 4}, {7, 5}, {2, 1}, {3, 7}, {6, 3}};
void kShift(int role, uint32_t frame, int32_t& sx, int32_t& sy) {
    if (role < 0) { sx = sy = 4; return; }
    int32_t x = KCycle[frame % 8][0], y = KCycle[frame % 8][1];
    for (int depth = 0; depth < role; ++depth) { x = (2 * x + 4) & 7; y = (2 * y + 4) & 7; }
    sx = x; sy = y;
}
// preset K's eleven layers in network order: token-grid divisor, input channels, core channels, the channels
// written to out24 (the patch-merged skip for encoders, the block output for the decoders, the 40-channel head
// for dec0) and the window-schedule role.
struct KLayer {
    const char* name; uint32_t divisor, cin, c, out24ch; int role;
};
constexpr KLayer KPlan[11] = {
    {"enc0", 1, 16, 32, 64, -1}, {"enc1", 2, 64, 64, 64, -1}, {"enc2", 4, 64, 64, 96, -1},
    {"enc3", 8, 96, 96, 128, 2}, {"enc4", 16, 128, 128, 160, 1}, {"dec5", 32, 160, 160, 160, 0},
    {"dec4", 16, 160, 128, 128, 1}, {"dec3", 8, 128, 96, 96, 2}, {"dec2", 4, 96, 64, 64, -1},
    {"dec1", 2, 64, 64, 64, -1}, {"dec0", 1, 64, 32, 40, -1},
};
constexpr uint32_t KLayers = 11, MLayers = 10, MaxGraphs = 8;
} // namespace

struct HipNetwork::Impl {
    std::vector<Kernel> kernels;
    std::vector<Node> preps;
    std::vector<Node> nodes[MaxGraphs];
    std::vector<void*> owned;
    hipExternalMemory_t shared[2]{};
    hipStream_t stream = nullptr;
    hipGraph_t graph[MaxGraphs]{};
    hipGraphExec_t executable[MaxGraphs]{};
    uint32_t graphs = 2, frame = 0;
    hipEvent_t start = nullptr, stop = nullptr;
    bool pending = false;
    ~Impl() {
        if (stream) (void)hipStreamSynchronize(stream);
        for (auto e : executable) if (e) (void)hipGraphExecDestroy(e);
        for (auto g : graph) if (g) (void)hipGraphDestroy(g);
        for (void* p : owned) (void)hipFree(p);
        for (auto& k : kernels) if (k.module) (void)hipModuleUnload(k.module);
        for (auto m : shared) if (m) (void)hipDestroyExternalMemory(m);
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
        if (stream) (void)hipStreamDestroy(stream);
    }
    void run(const Node& n) {
        void* args[] = {const_cast<Params*>(&n.params)};
        HIPCHECK(hipModuleLaunchKernel(n.function, n.gx, n.gy, 1, n.bx, n.by, n.bz, 0, stream, args, nullptr));
    }
};

HipNetwork::HipNetwork(const std::string& directory, uint32_t width, uint32_t height, int tokensFd, int outputFd,
                       uint64_t tokensBytes, uint64_t outputBytes) : impl(std::make_unique<Impl>()) {
    auto& n = *impl;
    HIPCHECK(hipInit(0));
    int leastPriority = 0, greatestPriority = 0;
    HIPCHECK(hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    HIPCHECK(hipStreamCreateWithPriority(&n.stream, hipStreamNonBlocking, greatestPriority));
    HIPCHECK(hipEventCreate(&n.start)); HIPCHECK(hipEventCreate(&n.stop));
    const auto file = slurp(directory + "/hipnet.bin");
    if (file.size() < 16) throw std::runtime_error("incompatible hipnet.bin");
    const bool kPreset = std::memcmp(file.data(), "D4RKHIP1", 8) == 0;
    if (!kPreset && std::memcmp(file.data(), "D4RMHIP1", 8)) throw std::runtime_error("incompatible hipnet.bin");
    uint32_t count = 0;
    std::memcpy(&count, file.data() + 8, 4);
    const uint32_t layers = kPreset ? KLayers : MLayers;
    if (count != layers || file.size() < 16 + 64ull * count) throw std::runtime_error("invalid hipnet.bin");
    void* tokens = import(tokensFd, tokensBytes, n.shared[0]);
    void* output = import(outputFd, outputBytes, n.shared[1]);
    // Both dimensions describe the layer-0 token grid: preset M's render dimensions, preset K's output
    // dimensions (makeKShape: max 256, align 32 of the output quarter resolution).
    uint32_t tw = 0, th = 0;
    if (kPreset) {
        if (width < 256 || height < 256 || width > 8192 || height > 8192)
            throw std::runtime_error("standard K output dimensions must be 256..8192");
        tw = std::max(256u, ((width + 3) / 4 + 31) / 32 * 32);
        th = std::max(256u, ((height + 3) / 4 + 31) / 32 * 32);
    } else {
        tw = (width + 31) / 32 * 16; th = (height + 31) / 32 * 16;
    }
    // Kernel lookup by name; the M tube shares one code object across six layers, the K layer names are unique.
    std::vector<std::string> names;
    auto kernelOf = [&](const char* name) -> uint32_t {
        size_t k = 0;
        while (k < names.size() && names[k] != name) ++k;
        if (k < names.size()) return uint32_t(k);
        names.emplace_back(name); n.kernels.emplace_back();
        auto& kernel = n.kernels.back();
        const auto code = slurp(directory + "/hip/" + name + ".hsaco");
        HIPCHECK(hipModuleLoadData(&kernel.module, code.data()));
        HIPCHECK(hipModuleGetFunction(&kernel.main, kernel.module, name));
        const std::string base(name);
        if (hipModuleGetFunction(&kernel.prep, kernel.module, (base + "_prep").c_str()) != hipSuccess) kernel.prep = nullptr;
        kernel.prepBlocks = global32(kernel.module, "d4r_prep_blocks", 0);
        kernel.blockZ = global32(kernel.module, "d4r_block_z", 4);
        kernel.gridX = global32(kernel.module, "d4r_grid_x", 0);
        kernel.gridY = global32(kernel.module, "d4r_grid_y", 0);
        if (kernel.prep && !kernel.prepBlocks) throw std::runtime_error(base + ": prep kernel without d4r_prep_blocks");
        // The deep layers' code objects also carry their post phases; their grids are compile-time.
        for (int j = 1; j <= 8; ++j) {
            const std::string tag = base + "_post" + std::to_string(j);
            if (hipModuleGetFunction(&kernel.post[kernel.postCount], kernel.module, tag.c_str()) != hipSuccess) break;
            const std::string suffix = "_post" + std::to_string(j);
            kernel.postGridX[kernel.postCount] = global32(kernel.module, ("d4r" + suffix + "_grid_x").c_str(), 0);
            kernel.postGridY[kernel.postCount] = global32(kernel.module, ("d4r" + suffix + "_grid_y").c_str(), 0);
            kernel.postBlockX[kernel.postCount] = global32(kernel.module, ("d4r" + suffix + "_block_x").c_str(), 32);
            kernel.postBlockY[kernel.postCount] = global32(kernel.module, ("d4r" + suffix + "_block_y").c_str(), 1);
            kernel.postBlockZ[kernel.postCount] = global32(kernel.module, ("d4r" + suffix + "_block_z").c_str(), 1);
            if (!kernel.postGridX[kernel.postCount]) kernel.postGridX[kernel.postCount] = kernel.gridX;
            ++kernel.postCount;
        }
        // a module keeps one prepared weight image per weights pointer while its slots last; the M tube needs six
        if (global32(kernel.module, "d4r_prep_key_slots", 0) < 6 && base.find("tube") != std::string::npos)
            throw std::runtime_error(base + ": fewer than six weight image slots");
        return uint32_t(names.size() - 1);
    };
    auto allocate = [&](uint64_t bytes) {
        void* p = nullptr; HIPCHECK(hipMalloc(&p, size_t(bytes) + 4096)); n.owned.push_back(p); return p;
    };
    std::vector<uint32_t> layerKernel(layers), divisor(layers);
    std::vector<int32_t> shiftX(layers), shiftY(layers);
    std::vector<void*> weights(layers);
    for (uint32_t i = 0; i < count; ++i) {
        const char* e = file.data() + 16 + 64 * i;
        char name[33]{}; std::memcpy(name, e, 32);
        int32_t sx, sy; uint32_t scale; uint64_t at, bytes;
        std::memcpy(&sx, e + 32, 4); std::memcpy(&sy, e + 36, 4); std::memcpy(&scale, e + 40, 4);
        std::memcpy(&at, e + 48, 8); std::memcpy(&bytes, e + 56, 8);
        if (!scale || at > file.size() || bytes > file.size() - at) throw std::runtime_error("invalid hipnet.bin layer");
        if (kPreset && std::strcmp(name, ("dltss_pwin_" + std::string(KPlan[i].name) + "_layer").c_str()))
            throw std::runtime_error(std::string("unexpected K layer order: ") + name);
        layerKernel[i] = kernelOf(name);
        void* w = allocate(bytes);
        HIPCHECK(hipMemcpyHtoD(w, file.data() + at, bytes));
        weights[i] = w; divisor[i] = scale; shiftX[i] = sx; shiftY[i] = sy;
    }
    // The layer plan: activation buffers and the frozen parameter block of every layer at frame/phase 0.
    std::vector<Params> base(layers);
    if (kPreset) {
        // NGX packs the encoder skips, the merged outputs and the decoder outputs into one scratch allocation;
        // separate buffers only have to respect the channel counts and the wiring, not the original aliasing.
        std::vector<void*> act(layers), full(5, nullptr);
        for (uint32_t i = 0; i < layers; ++i) {
            const auto& p = KPlan[i];
            const uint32_t w = tw / p.divisor, h = th / p.divisor;
            // an encoder's out24 is the patch-merged 2x2 output, half the resolution of its input
            const uint32_t ow = i < 5 ? tw / (2 * p.divisor) : w, oh = i < 5 ? th / (2 * p.divisor) : h;
            // dec0 writes the head straight into the caller's output buffer
            if (i + 1 != layers) act[i] = allocate(uint64_t(ow) * oh * p.out24ch * 2);
            if (i < 5) full[i] = allocate(uint64_t(w) * h * p.c * 2);
        }
        for (uint32_t i = 0; i < layers; ++i) {
            const auto& p = KPlan[i];
            auto& q = base[i].k;
            q.W = int32_t(tw / p.divisor); q.H = int32_t(th / p.divisor);
            q.in = i == 0 ? tokens : act[i - 1];
            q.skip = i >= 6 ? full[10 - i] : nullptr;
            q.out24 = i + 1 == layers ? output : act[i];
            q.out32 = i < 5 ? full[i] : nullptr;
            kShift(p.role, 0, q.sx, q.sy);
            q.w = weights[i];
        }
    } else {
        void* skip1 = allocate(uint64_t(tw) * th * 64);
        void* merged1 = allocate(uint64_t(tw / 2) * (th / 2) * 96);
        void* skip2 = allocate(uint64_t(tw / 2) * (th / 2) * 96);
        void* merged2 = allocate(uint64_t(tw / 4) * (th / 4) * 128);
        void* tubeA = allocate(uint64_t(tw / 4) * (th / 4) * 128);
        void* tubeB = allocate(uint64_t(tw / 4) * (th / 4) * 128);
        void* dec2 = allocate(uint64_t(tw / 2) * (th / 2) * 96);
        // tokens -> enc1 -> enc2 -> six tube blocks (ping-pong) -> dec2 -> dec1
        void* in[MLayers] = {tokens, merged1, merged2, tubeA, tubeB, tubeA, tubeB, tubeA, tubeB, dec2};
        void* skip[MLayers] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, skip2, skip1};
        void* out[MLayers] = {skip1, skip2, tubeA, tubeB, tubeA, tubeB, tubeA, tubeB, dec2, output};
        void* merged[MLayers] = {merged1, merged2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        for (uint32_t i = 0; i < layers; ++i) {
            auto& q = base[i].m;
            q.w = weights[i]; q.sx = shiftX[i]; q.sy = shiftY[i];
            q.tw = int32_t(tw / divisor[i]); q.th = int32_t(th / divisor[i]);
            q.in = in[i]; q.skip = skip[i]; q.out = out[i]; q.merged = merged[i];
        }
    }
    // Immutable graphs: preset M has two horizontal window-alignment phases, preset K the eight-frame cycle.
    n.graphs = kPreset ? MaxGraphs : 2;
    for (uint32_t phase = 0; phase < n.graphs; ++phase) {
        auto& nodes = n.nodes[phase];
        for (uint32_t i = 0; i < layers; ++i) {
            const auto& kernel = n.kernels[layerKernel[i]];
            Params params = base[i];
            if (kPreset) kShift(KPlan[i].role, phase, params.k.sx, params.k.sy);
            else params.m.sx = shiftX[i] ^ int32_t(phase * 2); // NGX alternates horizontal alignment every evaluation
            if (kernel.postCount) {
                Node node; node.function = kernel.main; node.params = params;
                node.gx = kernel.gridX; node.gy = kernel.gridY ? kernel.gridY : 1; node.bz = kernel.blockZ;
                nodes.push_back(node);
                for (int j = 0; j < kernel.postCount; ++j) {
                    Node post; post.function = kernel.post[j]; post.params = params;
                    post.gx = kernel.postGridX[j]; post.gy = kernel.postGridY[j] ? kernel.postGridY[j] : 1;
                    post.bx = kernel.postBlockX[j]; post.by = kernel.postBlockY[j]; post.bz = kernel.postBlockZ[j];
                    nodes.push_back(post);
                }
            } else {
                Node node; node.function = kernel.main; node.params = params;
                if (kPreset) {
                    node.gx = (params.k.W + params.k.sx + 7) / 8; node.gy = (params.k.H + params.k.sy + 7) / 8;
                } else {
                    node.gx = (params.m.tw + params.m.sx + 7) / 8; node.gy = (params.m.th + params.m.sy + 7) / 8;
                }
                node.bz = kernel.blockZ;
                nodes.push_back(node);
            }
        }
    }
    // The weights never change: every layer's weight image is prepared once, here, and found by its pointer.
    // A shared code object (the M tube) gets one preparation per layer, i.e. per distinct weights pointer.
    for (uint32_t i = 0; i < layers; ++i) {
        const auto& kernel = n.kernels[layerKernel[i]];
        if (!kernel.prep) continue;
        Node prep; prep.function = kernel.prep; prep.gx = kernel.prepBlocks; prep.bx = 128; prep.params = base[i];
        n.preps.push_back(prep);
    }
    for (const auto& prep : n.preps) n.run(prep);
    HIPCHECK(hipStreamSynchronize(n.stream));
    for (uint32_t phase = 0; phase < n.graphs; ++phase) {
        HIPCHECK(hipStreamBeginCapture(n.stream, hipStreamCaptureModeRelaxed));
        try {
            for (const auto& node : n.nodes[phase]) n.run(node);
        } catch (...) {
            (void)hipStreamEndCapture(n.stream, &n.graph[phase]);
            throw;
        }
        HIPCHECK(hipStreamEndCapture(n.stream, &n.graph[phase]));
        HIPCHECK(hipGraphInstantiate(&n.executable[phase], n.graph[phase], nullptr, nullptr, 0));
        HIPCHECK(hipGraphUpload(n.executable[phase], n.stream));
    }
    HIPCHECK(hipStreamSynchronize(n.stream));
}
HipNetwork::~HipNetwork() = default;
void HipNetwork::launch() {
    auto& n = *impl;
    HIPCHECK(hipEventRecord(n.start, n.stream));
    HIPCHECK(hipGraphLaunch(n.executable[n.frame % n.graphs], n.stream));
    ++n.frame;
    HIPCHECK(hipEventRecord(n.stop, n.stream));
    n.pending = true;
}
bool HipNetwork::done() {
    auto& n = *impl;
    if (!n.pending) return true;
    const hipError_t e = hipEventQuery(n.stop);
    if (e == hipErrorNotReady) return false;
    check(e, "hipEventQuery"); n.pending = false; return true;
}
void HipNetwork::wait() { HIPCHECK(hipStreamSynchronize(impl->stream)); impl->pending = false; }
float HipNetwork::lastMilliseconds() { float ms = 0; HIPCHECK(hipEventElapsedTime(&ms, impl->start, impl->stop)); return ms; }
} // namespace d4r
