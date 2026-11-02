// Preset M on D3D12: runs raw frames through the model's DXIL (compile_m.py --dxil) and writes the output images.
//   d4r-m-d3d12 MODEL_DIR RENDER_W RENDER_H OUT_W OUT_H OUTPUT_PREFIX [repeats] -- COLOR MOTION DEPTH JX JY ...
// The same frame as the Vulkan runtime (../runtime_m.inc: parameters, bindings, dispatches, the per-frame window
// shift and the two history sets), on AMD's driver: the network's matrices are WMMA through AGS's reserved register
// space (d3d12/m_wmma.hlsli), the other stages are their GLSL through hlsl_build.py. Binding N is register tN / uN
// (sampler sN) as in the D3D12 K runtime. COLOR: RGBA16F, MOTION: RG16F, DEPTH: R32F at render size, tightly packed.
// The first frame resets. D4R_ENGINE_M_LAYER_STAMPS=1 also times every layer; D4R_M_DUMP=PREFIX writes the first
// frame's intermediates (D4R_M_DUMP_FRAME=N: frame N's) to PREFIX-NAME-N.bin; D4R_D3D12_DUMP_DIR=DIR writes the root
// signatures to DIR/SHADER.rootsig.
//
// A trailing `hip` argument (after [repeats], before --) runs the network on the model's native HIP layers between
// the input and expansion stages (d3d12/hip_net_d3d12.h): front submit, CPU fence wait, HIP graph, back submit. A
// model with hipnet.bin and hip/ takes that route by default; D4R_DX_NET=dxil forces the DXIL network.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef D4R_ENGINE_D3DHIP
#include "hip_net_d3d12.h"
#endif
using Microsoft::WRL::ComPtr;
#define CK(x) do { HRESULT h_ = (x); if (FAILED(h_)) { char b_[256]; std::snprintf(b_, sizeof(b_), "%s: 0x%08lx", #x, (unsigned long)h_); throw std::runtime_error(b_); } } while (0)

namespace {
constexpr UINT AgsSpace = 0x7fff0ade;
constexpr unsigned MLayers = 10;
constexpr const char* MShader[MLayers] = {"m_enc1", "m_enc2", "m_tube", "m_tube", "m_tube", "m_tube", "m_tube", "m_tube", "m_dec2", "m_dec1"};
constexpr int32_t MShift[MLayers] = {2, 2, 2, 0, 2, 0, 2, 0, 0, 0};
constexpr unsigned MScale[MLayers] = {1, 2, 4, 4, 4, 4, 4, 4, 2, 1};
// the optional two-window tube for large dispatches, as runtime_m_d3d12.cpp (MTubeW2Groups)
constexpr uint32_t MTubeW2Groups = 128;
struct MEnc0Params {
    int32_t size[2], shift[2], tok[2];
    float jitter[2], prevJitter[2], mvScale[2], mvOffs[2], rnd2hi[2], hi2rnd[2], hiSize[2];
    float changeScale, changeGain;
    uint32_t depthFlag, reset, frame;
    float expo, expoPrev;
    float halfInv[2], renderInv[2];
    uint32_t displayMv; int32_t mvMax[2]; float mvPix[2];
};
struct MDec0Params { int32_t size[2], tokW; uint32_t reset; float rndOffs[2], rnd2hi[2], hiSize[2], histOffs[2], sigma, rho, renderInv[2]; };
struct MPostParams {
    int32_t colOrigin[2], hiSize[2], rndSize[2]; float hi2rnd[2], rndOffs[2], invRnd[2], gain; uint32_t reset; float expo, expoPrev;
    int32_t mvSize[2]; float mvScale[2], mvOffs[2];
};
struct MDownParams { int32_t srcSize[2], outOrigin[2], colOrigin[2], colSize[2]; float out2rnd[2], rndOffs[2], expo; };
struct MLayerParams { int32_t w, h, sx, sy; uint32_t o[72]; };
struct ExposureParams { int32_t origin[2], size[2]; float invCount, scale, key; uint32_t flags, blocks, blocksX; int32_t sampleSize[2];
                        float gameScale = 1, invPre = 1; };
static_assert(sizeof(MEnc0Params) == 144 && sizeof(MDec0Params) == 64 && sizeof(MPostParams) == 88 && sizeof(MDownParams) == 52 &&
              sizeof(MLayerParams) == 304 && sizeof(ExposureParams) == 56, "the stages' parameter blocks");
uint32_t exposureGrid(uint32_t render, uint32_t output) { return output / 2 <= render ? output / 2 : std::max(render / 2, 256u); }

#include "replay_resources.h"

// ---------------------------------------------------------------- stages
// A descriptor table slot: binding N of a stage.
enum class Kind { None, Tex, Raw, RwTex, RwStruct, RwRaw };
struct Slot {
    Kind kind = Kind::None;
    Res* res = nullptr;
    bool linear = false;        // Tex: the sampler sN filters
    UINT stride = 0;            // RwStruct
    UINT64 offset = 0;          // Raw: byte offset of the view
    bool fixed = false;         // an external (HIP) resource: in COMMON, never transitioned
};
Slot tex(Res& r, bool linear = false) { return {Kind::Tex, &r, linear}; }
Slot raw(Res& r, UINT64 offset = 0) { return {Kind::Raw, &r, false, 0, offset}; }
Slot rwTex(Res& r) { return {Kind::RwTex, &r}; }
Slot rwStruct(Res& r, UINT stride) { return {Kind::RwStruct, &r, false, stride}; }
Slot rwRaw(Res& r) { return {Kind::RwRaw, &r}; }
Slot fixedRaw(Res& r) { Slot s{Kind::RwRaw, &r}; s.fixed = true; return s; }
Slot fixedSrv(Res& r) { Slot s{Kind::Raw, &r}; s.fixed = true; return s; }

struct Pipeline {
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> state;
    // the table's layout (kinds and samplers) is fixed by the first table it is created with
    void create(ID3D12Device* dev, const std::string& path, const std::vector<Slot>& table) {
        const auto dxil = slurp(path);
        std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
        std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
        for (UINT i = 0; i < table.size(); ++i) {
            const Kind k = table[i].kind;
            if (k == Kind::None) continue;
            const bool srv = k == Kind::Tex || k == Kind::Raw;
            ranges.push_back({srv ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, i, 0, i});
            if (k == Kind::Tex) {
                D3D12_STATIC_SAMPLER_DESC s{};
                s.Filter = table[i].linear ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
                s.AddressU = s.AddressV = s.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                s.MaxLOD = D3D12_FLOAT32_MAX; s.ShaderRegister = i; s.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                s.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
                samplers.push_back(s);
            }
        }
        D3D12_ROOT_PARAMETER p[2]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[0].DescriptorTable = {UINT(ranges.size()), ranges.data()};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;     // the AGS intrinsic UAV (unused by the plain stages)
        p[1].Descriptor = {0, AgsSpace};
        D3D12_ROOT_SIGNATURE_DESC desc{2, p, UINT(samplers.size()), samplers.data(), D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob, error;
        if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error)))
            throw std::runtime_error(std::string("root signature: ") + (error ? static_cast<const char*>(error->GetBufferPointer()) : "?"));
        CK(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        // D4R_D3D12_DUMP_DIR: the root signatures as files (RGA's --rs-bin), as the K runtime writes them
        if (const char* directory = std::getenv("D4R_D3D12_DUMP_DIR"); directory && *directory) {
            std::filesystem::create_directories(directory);
            std::ofstream(std::filesystem::path(directory) / (std::filesystem::path(path).stem().string() + ".rootsig"), std::ios::binary)
                .write(static_cast<const char*>(blob->GetBufferPointer()), std::streamsize(blob->GetBufferSize()));
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get(); pso.CS = {dxil.data(), dxil.size()};
        if (FAILED(dev->CreateComputePipelineState(&pso, IID_PPV_ARGS(&state))))
            throw std::runtime_error("cannot create the pipeline of " + path);
    }
};

// One dispatch's bindings: a range of the shader-visible heap.
struct Table { std::vector<Slot> slots; UINT first = 0; };
struct Heap {
    ComPtr<ID3D12DescriptorHeap> heap;
    UINT increment = 0, used = 0;
    void create(ID3D12Device* dev, UINT count) {
        D3D12_DESCRIPTOR_HEAP_DESC d{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        CK(dev->CreateDescriptorHeap(&d, IID_PPV_ARGS(&heap)));
        increment = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    Table table(ID3D12Device* dev, std::vector<Slot> slots) {
        Table t{std::move(slots), used};
        used += UINT(t.slots.size());
        if (used > heap->GetDesc().NumDescriptors) throw std::runtime_error("descriptor heap too small");
        for (UINT i = 0; i < t.slots.size(); ++i) {
            const Slot& s = t.slots[i];
            auto h = heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(t.first + i) * increment;
            if (s.kind == Kind::Tex) {
                D3D12_SHADER_RESOURCE_VIEW_DESC v{}; v.Format = s.res->format; v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; v.Texture2D.MipLevels = 1;
                dev->CreateShaderResourceView(s.res->get(), &v, h);
            } else if (s.kind == Kind::Raw) {
                D3D12_SHADER_RESOURCE_VIEW_DESC v{}; v.Format = DXGI_FORMAT_R32_TYPELESS; v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                v.Buffer = {s.offset / 4, UINT((s.res->bytes - s.offset) / 4), 0, D3D12_BUFFER_SRV_FLAG_RAW};
                dev->CreateShaderResourceView(s.res->get(), &v, h);
            } else if (s.kind == Kind::RwTex) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC v{}; v.Format = s.res->format; v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                dev->CreateUnorderedAccessView(s.res->get(), nullptr, &v, h);
            } else if (s.kind == Kind::RwStruct) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC v{}; v.Format = DXGI_FORMAT_UNKNOWN; v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                v.Buffer = {0, UINT(s.res->bytes / s.stride), s.stride, 0, D3D12_BUFFER_UAV_FLAG_NONE};
                dev->CreateUnorderedAccessView(s.res->get(), nullptr, &v, h);
            } else if (s.kind == Kind::RwRaw) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC v{}; v.Format = DXGI_FORMAT_R32_TYPELESS; v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                v.Buffer = {0, UINT(s.res->bytes / 4), 0, 0, D3D12_BUFFER_UAV_FLAG_RAW};
                dev->CreateUnorderedAccessView(s.res->get(), nullptr, &v, h);
            }
        }
        return t;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT i) const { auto h = heap->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(i) * increment; return h; }
};
} // namespace

int main(int argc, char** argv) try {
    int sep = 0;
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--")) sep = i;
    if (sep < 7 || (argc - sep - 1) % 5 || argc - sep - 1 == 0) {
        std::fprintf(stderr, "usage: d4r-m-d3d12 MODEL_DIR RW RH OW OH OUTPUT_PREFIX [repeats] -- COLOR MOTION DEPTH JX JY ...\n");
        return 2;
    }
    const std::string model = argv[1], prefix = argv[6];
    const uint32_t rw = std::atoi(argv[2]), rh = std::atoi(argv[3]), ow = std::atoi(argv[4]), oh = std::atoi(argv[5]);
    // a trailing `hip` after [repeats] (before --): the network runs on the model's native HIP layers
    const bool hipArg = sep >= 8 && !std::strcmp(argv[sep - 1], "hip");
    const int repeats = sep - int(hipArg) > 7 ? std::atoi(argv[7]) : 0;
    const int frames = (argc - sep - 1) / 5;
    if (rw < 64 || rh < 64 || ow % 2 || oh % 2 || rw > ow || rh > oh) throw std::runtime_error("unsupported preset M dimensions");

    Device d;
    ComPtr<IDXGIFactory6> factory; CK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter; CK(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
    CK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d.dev)));
    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    CK(d.dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&d.queue)));
    CK(d.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d.alloc)));
    CK(d.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d.alloc.Get(), nullptr, IID_PPV_ARGS(&d.list)));
    CK(d.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d.fence)));
    ID3D12Device* dev = d.dev.Get();
    auto* cl = d.list.Get();

    // ---------------------------------------------------------------- the model
    // preset L (D4RL0002, compile_m.py compile_l): the dither table, L's five constants, then the input stage's block,
    // the ten layers and the expansion stage's block, each a layer's (FP8 offset, f16 offset, o[72])
    const auto head = slurp(model + "/offsets.bin");
    const bool L = head.size() == 40 + (MLayers + 2) * 74 * 4 && !std::memcmp(head.data(), "D4RL0002", 8);
    if (!L && (head.size() != 16 + 5 * 4 + MLayers * 74 * 4 || std::memcmp(head.data(), "D4RM000", 7)))
        throw std::runtime_error("not a preset M or L model (offsets.bin)");
    uint32_t n8, n16, enc0Off[3] = {}, dec0Off[2] = {}, dither16 = 0;
    float lConst[5] = {9.045065879821777f, 0.22356846928596497f, 8.f, 0.07863261550664902f, 2.7530112f};    // M's
    std::array<std::array<uint32_t, 74>, MLayers> layers;
    std::array<uint32_t, 74> lEnc0{}, lDec0{};
    std::memcpy(&n8, head.data() + 8, 4); std::memcpy(&n16, head.data() + 12, 4);
    if (L) {
        std::memcpy(&dither16, head.data() + 16, 4); std::memcpy(lConst, head.data() + 20, sizeof(lConst));
        std::memcpy(lEnc0.data(), head.data() + 40, sizeof(lEnc0));
        std::memcpy(layers.data(), head.data() + 40 + sizeof(lEnc0), sizeof(layers));
        std::memcpy(lDec0.data(), head.data() + 40 + sizeof(lEnc0) + sizeof(layers), sizeof(lDec0));
    } else {
        std::memcpy(enc0Off, head.data() + 16, 12); std::memcpy(dec0Off, head.data() + 28, 8);
        std::memcpy(layers.data(), head.data() + 36, sizeof(layers));
    }
    auto b8 = slurp(model + "/weights8.bin");
    const auto b16 = slurp(model + "/weights16.bin");
    if (b8.size() != n8 || b16.size() != n16 || (!L && (enc0Off[0] + 128 * 64 > n8 || dec0Off[0] + 64 * 80 > n8)))
        throw std::runtime_error("invalid M model sizes");
    // The FP8 matrices of the input and expansion stages rearranged after the package's, column-major, so that a lane's
    // WMMA operand bytes are adjacent (the package's row-major layout takes a byte load per element: input stage 143 ->
    // 79 us at 854x480 -> 2560x1440). Input stage, [128][64]: the bytes of channel c are [2][4][8], half h, k32 step s,
    // k = 32 s + 16 j + 8 h + i at c * 128 + 64 h + 16 s + 8 j + i (a k32 step's A operands are one 16-byte load).
    // Expansion, [64][80]: the bytes of column c are [2][4][8], k = 16 t + 8 h + i at c * 64 + 32 h + 8 t + i.
    const size_t enc0W = (b8.size() + 255) / 256 * 256, dec0W = enc0W + 128 * 64;
    if (!L) b8.resize(dec0W + 64 * 80);
    if (!L) for (uint32_t k = 0; k < 128; ++k)
        for (uint32_t c = 0; c < 64; ++c)
            b8[enc0W + c * 128 + 64 * (k >> 3 & 1) + 16 * (k >> 5) + 8 * (k >> 4 & 1) + (k & 7)] = b8[enc0Off[0] + k * 64 + c];
    if (!L) for (uint32_t k = 0; k < 64; ++k)
        for (uint32_t c = 0; c < 80; ++c)
            b8[dec0W + c * 64 + 32 * (k >> 3 & 1) + 8 * (k >> 4) + (k & 7)] = b8[dec0Off[0] + k * 80 + c];
    const bool ldr = std::filesystem::exists(model + "/ldr");
    // L: the input stage's feature slots (enc0_m.comp L_FEATURES, enc0_lf), its block (layer_m.hlsl KIND 3), the
    // expansion (KIND 4) and the reconstruction's L tiers
    const std::string inName = L ? "enc0_l" : "enc0_m", exName = L ? "dec0_l" : "dec0_m", po = L ? "post_l" : "post_m";
    for (const std::string& f : {std::string("m_enc1"), std::string("m_enc2"), std::string("m_tube"), std::string("m_dec2"),
                                 std::string("m_dec1"), inName, exName, po, po + "_q", po + "_c", std::string("down_m"),
                                 std::string("exposure_m0"), std::string("exposure_m1"), L ? std::string("enc0_lf") : inName})
        if (!std::filesystem::exists(model + "/" + f + ".dxil"))
            throw std::runtime_error(model + " has no " + f + ".dxil: build it with compile_m.py --dxil yes");

    // ---------------------------------------------------------------- resources (runtime_m.inc MEngine)
    const uint32_t hw = ow / 2 * 3, hh = oh / 2 * 3;
    const uint32_t gw = (rw + 31) / 32 * 32, gh = (rh + 31) / 32 * 32, tw = gw / 2, th = gh / 2;
    const bool quality = uint64_t(hw) < 3ull * rw || uint64_t(hh) < 3ull * rh;
    const bool compactTier = quality && 20ull * rw <= 9ull * hw && 20ull * rh <= 9ull * hh;
    Res w8 = d.buffer(b8.size()), w16 = d.buffer(b16.size());
    Res enc0P = d.buffer(sizeof(MEnc0Params)), dec0P = d.buffer(sizeof(MDec0Params)), postP = d.buffer(sizeof(MPostParams));
    Res downP = d.buffer(sizeof(MDownParams)), expP = d.buffer(sizeof(ExposureParams));
    Res layerP = d.buffer(512 * (MLayers + 2));     // L: slots 10 and 11 are the input and expansion stages' blocks
    Res expSums = d.buffer(UINT64((std::max(ow / 2, 256u) + 31) / 32) * ((std::max(oh / 2, 256u) + 31) / 32) * 4);
    Res expState = d.buffer(8), dummyIn = d.buffer(256), dummyOut = d.buffer(256), ags = d.buffer(256);
    Res packed = d.buffer(UINT64(hw) * hh * 4);
    auto tok = [&](unsigned scale, unsigned channels) { return d.buffer(UINT64(tw / scale) * (th / scale) * channels + 64); };
    Res tokens = tok(1, 64), skip1 = tok(1, 64), merged1 = tok(2, 96), skip2 = tok(2, 96), merged2 = tok(4, 128);
    Res tubeA = tok(4, 128), tubeB = tok(4, 128), dec2Out = tok(2, 96), dec1Out = tok(1, 64);
    // L: the input stage's pixel-resolution block output (the expansion's skip) and its feature slots, per 8x8 block of
    // its grid (32 bytes a pixel)
    const uint32_t lbx = (gw + 9) / 8, lby = (gh + 9) / 8;
    Res skipL = d.buffer(L ? UINT64(gw) * gh * 32 + 64 : 256), featsL = d.buffer(L ? UINT64(lbx) * lby * 64 * 32 : 256);
    Res dummyOut2 = d.buffer(256);
    // The native HIP network between the input and expansion stages (the same route d4r-m's `hip` argument takes):
    // its two shared buffers stand in for `tokens` and `dec1Out`. A model with hipnet.bin and hip/ defaults to it;
    // D4R_DX_NET=dxil keeps the DXIL network.
    std::unique_ptr<d4r::dx::HipNetworkDx> hipNetwork;
    Res hipTokens, hipHead;
    const uint64_t tokenBytes = UINT64(tw) * th * 64 + 64;
    const char* netMode = std::getenv("D4R_DX_NET");
    const bool dxilForced = netMode && !std::strcmp(netMode, "dxil");
    const bool hipDefault = !dxilForced && std::filesystem::exists(model + "/hipnet.bin") &&
                            std::filesystem::is_directory(model + "/hip");
    if (!dxilForced && (hipArg || hipDefault)) {
#ifdef D4R_ENGINE_D3DHIP
        try {
            // L's input and expansion stages have plane-layout builds of their own (layer_m.hlsl PLANES, KIND 3 and 4)
            const std::string inPlanes = std::string(L ? "enc0_l" : "enc0_m") + "_planes";
            const std::string exPlanes = std::string(L ? "dec0_l" : "dec0_m") + "_planes";
            if (!std::filesystem::exists(model + "/" + inPlanes + ".dxil") || !std::filesystem::exists(model + "/" + exPlanes + ".dxil"))
                throw std::runtime_error("the model has no plane-layout DXIL stages (compile_m.py --dxil yes)");
            hipNetwork = std::make_unique<d4r::dx::HipNetworkDx>(dev, model, rw, rh, tokenBytes, tokenBytes);
        } catch (const std::exception& error) {
            std::printf("network: hip unavailable (%s); using the DXIL network\n", error.what());
        }
#else
        std::printf("network: hip unavailable (built without the HIP SDK: set D4R_ROCM_DIR); using the DXIL network\n");
#endif
    }
    if (hipNetwork) {
        hipTokens.r = hipNetwork->tokens(); hipTokens.bytes = tokenBytes; hipTokens.state = D3D12_RESOURCE_STATE_COMMON;
        hipHead.r = hipNetwork->head(); hipHead.bytes = tokenBytes; hipHead.state = D3D12_RESOURCE_STATE_COMMON;
        std::printf(hipArg ? "network: hip (native layers)\n"
                           : "network: hip (default; D4R_DX_NET=dxil to force the DXIL network)\n");
    }
    std::array<Res, 2> mvA{d.texture(rw, rh, DXGI_FORMAT_R16G16_FLOAT), d.texture(rw, rh, DXGI_FORMAT_R16G16_FLOAT)};
    Res mvB = d.texture(rw, rh, DXGI_FORMAT_R16G16_FLOAT), mvC = d.texture(rw, rh, DXGI_FORMAT_R16G16_FLOAT);
    std::array<Res, 2> feat{d.texture(rw, rh, DXGI_FORMAT_R8G8B8A8_SNORM), d.texture(rw, rh, DXGI_FORMAT_R8G8B8A8_SNORM)};
    Res gates = d.texture(rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM), mix = d.texture(rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM);
    Res cov = d.texture(rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    std::array<Res, 2> hist{d.texture(hw, hh, DXGI_FORMAT_R16G16B16A16_FLOAT), d.texture(hw, hh, DXGI_FORMAT_R16G16B16A16_FLOAT)};
    std::array<Res, 2> half{d.texture(hw / 2, hh / 2, DXGI_FORMAT_R32G32B32A32_FLOAT), d.texture(hw / 2, hh / 2, DXGI_FORMAT_R32G32B32A32_FLOAT)};
    Res output = d.texture(ow, oh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    // the game's inputs (render-resolution motion)
    Res color = d.texture(rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT), motion = d.texture(rw, rh, DXGI_FORMAT_R16G16_FLOAT);
    Res depth = d.texture(rw, rh, DXGI_FORMAT_R32_FLOAT);

    // uploads: the weights once, the inputs and parameters every frame
    auto pitch = [](UINT64 row) { return (row + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) / D3D12_TEXTURE_DATA_PITCH_ALIGNMENT * D3D12_TEXTURE_DATA_PITCH_ALIGNMENT; };
    const UINT64 inPitch[3] = {pitch(UINT64(rw) * 8), pitch(UINT64(rw) * 4), pitch(UINT64(rw) * 4)};
    const UINT64 inBytes[3] = {UINT64(rw) * 8, UINT64(rw) * 4, UINT64(rw) * 4};
    UINT64 inOffset[3], upBytes = 0;
    for (int k = 0; k < 3; ++k) { inOffset[k] = upBytes; upBytes += inPitch[k] * rh; upBytes = (upBytes + 511) / 512 * 512; }
    const UINT64 paramOffset = upBytes, layerParamOffset = paramOffset + 1024;
    upBytes = layerParamOffset + 512 * (MLayers + 2);
    Res upload = d.buffer(std::max<UINT64>(upBytes, b8.size() + b16.size() + 256), D3D12_HEAP_TYPE_UPLOAD);
    char* up; CK(upload.r->Map(0, nullptr, reinterpret_cast<void**>(&up)));
    const UINT64 outPitch = pitch(UINT64(ow) * 8);
    Res readback = d.buffer(outPitch * oh, D3D12_HEAP_TYPE_READBACK);
    std::memcpy(up, b8.data(), b8.size()); std::memcpy(up + (b8.size() + 255) / 256 * 256, b16.data(), b16.size());
    d.use(w8, D3D12_RESOURCE_STATE_COPY_DEST); d.use(w16, D3D12_RESOURCE_STATE_COPY_DEST); d.flushBarriers();
    cl->CopyBufferRegion(w8.get(), 0, upload.get(), 0, b8.size());
    cl->CopyBufferRegion(w16.get(), 0, upload.get(), (b8.size() + 255) / 256 * 256, b16.size());
    d.use(w8, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); d.use(w16, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    d.use(ags, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    d.submit();

    // ---------------------------------------------------------------- pipelines and their bindings
    Heap heap; heap.create(dev, 256);
    // the network: tokens -> enc1 -> enc2 -> six tube blocks (ping-pong) -> dec2 -> dec1
    Res* in[MLayers] = {&tokens, &merged1, &merged2, &tubeA, &tubeB, &tubeA, &tubeB, &tubeA, &tubeB, &dec2Out};
    Res* skip[MLayers] = {&dummyIn, &dummyIn, &dummyIn, &dummyIn, &dummyIn, &dummyIn, &dummyIn, &dummyIn, &skip2, &skip1};
    Res* out[MLayers] = {&skip1, &skip2, &tubeA, &tubeB, &tubeA, &tubeB, &tubeA, &tubeB, &dec2Out, &dec1Out};
    Res* merged[MLayers] = {&merged1, &merged2, &dummyOut, &dummyOut, &dummyOut, &dummyOut, &dummyOut, &dummyOut, &dummyOut, &dummyOut};
    std::map<std::string, Pipeline> shaders;
    std::array<Table, MLayers> layerT;
    for (unsigned l = 0; l < MLayers; ++l) {
        layerT[l] = heap.table(dev, {raw(layerP, 512 * l), raw(w8, layers[l][0]), raw(w16, layers[l][1]), raw(*in[l]), raw(*skip[l]),
                                     rwRaw(*out[l]), rwRaw(*merged[l])});
        if (!shaders.count(MShader[l])) shaders[MShader[l]].create(dev, model + "/" + MShader[l] + ".dxil", layerT[l].slots);
    }
    const bool tubeW2 = std::filesystem::exists(model + "/m_tube_w2.dxil");
    if (tubeW2) shaders["m_tube_w2"].create(dev, model + "/m_tube_w2.dxil", layerT[2].slots);
    const char* tubeW2Env = std::getenv("D4R_ENGINE_M_TUBE_W2_GROUPS");
    const uint32_t tubeW2Groups = tubeW2Env ? uint32_t(std::strtoul(tubeW2Env, nullptr, 10)) : MTubeW2Groups;
    std::array<Table, 2> enc0T, dec0T, postT, enc0LXT, enc0PlaneT, dec0PlaneT, enc0LPlaneT, dec0LPlaneT;
    // L in one dispatch (enc0_lx, vkd3d-proton packages: layer_m.hlsl KIND 3 L_FUSED): the input stage's slots, then its
    // layer block's
    const bool lx = L && std::filesystem::exists(model + "/enc0_lx.dxil");
    Table enc0LT;
    if (L) enc0LT = heap.table(dev, {raw(layerP, 512 * MLayers), raw(w8, lEnc0[0]), raw(w16, lEnc0[1]), raw(featsL), raw(dummyIn),
                                     rwRaw(skipL), rwRaw(tokens)});
    for (unsigned i = 0; i < 2; ++i) {     // i: the index of this frame's history outputs
        const unsigned prev = 1 - i;
        if (L) {
            enc0T[i] = heap.table(dev, {raw(enc0P), tex(color), tex(motion), tex(depth), tex(mvA[prev], true), tex(half[prev], true),
                                        tex(feat[prev], true), rwTex(mvA[i]), rwTex(mvB), rwTex(mvC), raw(w8), raw(w16),
                                        raw(w16, dither16), rwRaw(featsL)});
            if (lx)
                enc0LXT[i] = heap.table(dev, {raw(enc0P), tex(color), tex(motion), tex(depth), tex(mvA[prev], true), tex(half[prev], true),
                                              tex(feat[prev], true), rwTex(mvA[i]), rwTex(mvB), rwTex(mvC), raw(w8), raw(w16),
                                              raw(w16, dither16), rwRaw(featsL), raw(layerP, 512 * MLayers), raw(w8, lEnc0[0]),
                                              raw(w16, lEnc0[1]), raw(featsL), raw(dummyIn), rwRaw(skipL), rwRaw(tokens)});
            dec0T[i] = heap.table(dev, {raw(layerP, 512 * (MLayers + 1)), raw(w8, lDec0[0]), raw(w16, lDec0[1]), raw(dec1Out), raw(skipL),
                                        rwRaw(dummyOut), rwRaw(dummyOut2), raw(dec0P), rwTex(feat[i]), rwTex(gates), rwTex(mix), rwTex(cov)});
            if (hipNetwork) {
                // the same two blocks in the plane-layout DXIL, with the network's shared buffers where the internal
                // tokens (the merge's output) and head (the expansion's input) are
                enc0LPlaneT[i] = heap.table(dev, {raw(layerP, 512 * MLayers), raw(w8, lEnc0[0]), raw(w16, lEnc0[1]), raw(featsL), raw(dummyIn),
                                                  rwRaw(skipL), fixedRaw(hipTokens)});
                dec0LPlaneT[i] = heap.table(dev, {raw(layerP, 512 * (MLayers + 1)), raw(w8, lDec0[0]), raw(w16, lDec0[1]), fixedSrv(hipHead), raw(skipL),
                                                  rwRaw(dummyOut), rwRaw(dummyOut2), raw(dec0P), rwTex(feat[i]), rwTex(gates), rwTex(mix), rwTex(cov)});
            }
        } else {
            enc0T[i] = heap.table(dev, {raw(enc0P), tex(color), tex(motion), tex(depth), tex(mvA[prev], true), tex(half[prev], true),
                                        tex(feat[prev], true), rwTex(mvA[i]), rwTex(mvB), rwTex(mvC), raw(w8, enc0W), raw(w16, enc0Off[1]),
                                        raw(w16, enc0Off[2]), rwRaw(tokens)});
            dec0T[i] = heap.table(dev, {raw(dec0P), tex(mvA[i]), tex(feat[prev], true), rwTex(feat[i]), rwTex(gates), rwTex(mix),
                                        rwTex(cov), raw(dec1Out), raw(w8, dec0W), raw(w16, dec0Off[1])});
            if (hipNetwork) {
                // the plane-layout stages with the network's shared buffers where the internal tokens/head are
                enc0PlaneT[i] = heap.table(dev, {raw(enc0P), tex(color), tex(motion), tex(depth), tex(mvA[prev], true), tex(half[prev], true),
                                                 tex(feat[prev], true), rwTex(mvA[i]), rwTex(mvB), rwTex(mvC), raw(w8, enc0W),
                                                 raw(w16, enc0Off[1]), raw(w16, enc0Off[2]), fixedRaw(hipTokens)});
                dec0PlaneT[i] = heap.table(dev, {raw(dec0P), tex(mvA[i]), tex(feat[prev], true), rwTex(feat[i]), rwTex(gates), rwTex(mix),
                                                 rwTex(cov), fixedSrv(hipHead), raw(w8, dec0W), raw(w16, dec0Off[1])});
            }
        }
        postT[i] = heap.table(dev, {raw(postP), tex(color), tex(mix), tex(gates), tex(cov), tex(hist[prev], true), tex(mvA[i]),
                                    tex(mvB), tex(mvC, true), rwTex(hist[i]), rwTex(half[i]), rwStruct(packed, 4), tex(motion)});
    }
    const Table exp0T = heap.table(dev, {rwStruct(expP, sizeof(ExposureParams)), tex(color), rwStruct(expSums, 4)});
    const Table exp1T = heap.table(dev, {rwStruct(expP, sizeof(ExposureParams)), rwStruct(expSums, 4), rwStruct(expState, 8),
                                         rwStruct(enc0P, 4), rwStruct(postP, 4), rwStruct(downP, 4)});
    const Table downT = heap.table(dev, {raw(downP), tex(color), rwTex(output), rwStruct(packed, 4)});
    Pipeline enc0, enc0L, enc0LX, dec0, post, postQ, postC, down, exp0, exp1, enc0Planes, dec0Planes, enc0LPlanes, dec0LPlanes;
    enc0.create(dev, model + "/" + (L ? std::string("enc0_lf") : inName) + ".dxil", enc0T[0].slots);
    dec0.create(dev, model + "/" + exName + ".dxil", dec0T[0].slots);
    if (hipNetwork && L) {
        enc0LPlanes.create(dev, model + "/enc0_l_planes.dxil", enc0LPlaneT[0].slots);
        dec0LPlanes.create(dev, model + "/dec0_l_planes.dxil", dec0LPlaneT[0].slots);
    } else if (hipNetwork) {
        enc0Planes.create(dev, model + "/enc0_m_planes.dxil", enc0PlaneT[0].slots);
        dec0Planes.create(dev, model + "/dec0_m_planes.dxil", dec0PlaneT[0].slots);
    }
    if (L) enc0L.create(dev, model + "/enc0_l.dxil", enc0LT.slots);
    if (lx) enc0LX.create(dev, model + "/enc0_lx.dxil", enc0LXT[0].slots);
    post.create(dev, model + "/" + po + ".dxil", postT[0].slots); postQ.create(dev, model + "/" + po + "_q.dxil", postT[0].slots);
    postC.create(dev, model + "/" + po + "_c.dxil", postT[0].slots);
    down.create(dev, model + "/down_m.dxil", downT.slots);
    exp0.create(dev, model + "/exposure_m0.dxil", exp0T.slots); exp1.create(dev, model + "/exposure_m1.dxil", exp1T.slots);

    // a dispatch: its resources in the states the table needs, then the barrier after it
    auto dispatch = [&](Pipeline& p, const Table& t, UINT x, UINT y) {
        for (const Slot& s : t.slots) {
            if (s.fixed) continue;      // the external network's resources stay in COMMON
            if (s.kind == Kind::Tex || s.kind == Kind::Raw) d.use(*s.res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            else if (s.kind != Kind::None) d.use(*s.res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        d.flushBarriers();
        cl->SetComputeRootSignature(p.root.Get()); cl->SetPipelineState(p.state.Get());
        cl->SetComputeRootDescriptorTable(0, heap.gpu(t.first));
        cl->SetComputeRootUnorderedAccessView(1, ags.r->GetGPUVirtualAddress());
        cl->Dispatch(x, y, 1);
        d.uavBarrier();
    };

    // ---------------------------------------------------------------- frames
    std::vector<std::vector<char>> inputs;
    for (int f = 0; f < frames; ++f)
        for (int k = 0; k < 3; ++k) {
            inputs.push_back(slurp(argv[sep + 1 + 5 * f + k]));
            if (inputs.back().size() != inBytes[k] * rh) throw std::runtime_error("wrong input size");
        }
    D3D12_QUERY_HEAP_DESC qhd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 17};
    ComPtr<ID3D12QueryHeap> queries; CK(dev->CreateQueryHeap(&qhd, IID_PPV_ARGS(&queries)));
    Res stampBuffer = d.buffer(17 * 8, D3D12_HEAP_TYPE_READBACK);
    UINT64 frequency; CK(d.queue->GetTimestampFrequency(&frequency));
    static const bool layerStamps = std::getenv("D4R_ENGINE_M_LAYER_STAMPS") != nullptr;
    const char* dump = std::getenv("D4R_M_DUMP");
    const int dumpFrame = std::getenv("D4R_M_DUMP_FRAME") ? std::atoi(std::getenv("D4R_M_DUMP_FRAME")) : 0;
    float prevJitter[2] = {0, 0};
    unsigned current = 0, frameIndex = 0;
    std::vector<double> times, walls, networkMs; std::vector<std::vector<double>> stage(6), layer(MLayers);
    const auto now = [] { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    for (int pass = 0; pass <= repeats; ++pass)
        for (int f = 0; f < frames; ++f) {
            char** a = argv + sep + 1 + 5 * f;
            const float jitter[2] = {float(std::atof(a[3])), float(std::atof(a[4]))};
            const bool reset = pass == 0 && f == 0;
            const unsigned i = current;
            // the parameters (runtime_m.inc setFrame / recordFront)
            const float fx = float(hw) / float(rw), fy = float(hh) / float(rh);
            // L's input stage alternates its horizontal window shift every frame, (2, 2) / (0, 2); its expansion (0, 0) / (2, 0)
            const int32_t lsx = 2 ^ int32_t((frameIndex & 1u) * 2u);
            const MEnc0Params e0{{int32_t(rw), int32_t(rh)}, {L ? lsx : 2, 2}, {int32_t(tw), int32_t(th)}, {jitter[0], jitter[1]},
                                 {prevJitter[0], prevJitter[1]}, {fx, fy}, {0, 0}, {fx, fy}, {1.f / fx, 1.f / fy}, {float(hw), float(hh)},
                                 lConst[0], lConst[1],
                                 1u, reset ? 1u : 0u, frameIndex & 7u, 1.f, 0.f, {1.f / float(hw), 1.f / float(hh)}, {1.f / float(rw), 1.f / float(rh)},
                                 0u, {int32_t(ow) - 1, int32_t(oh) - 1}, {float(ow) / float(rw), float(oh) / float(rh)}};
            const MDec0Params d0{{int32_t(rw), int32_t(rh)}, int32_t(tw), reset ? 1u : 0u, {jitter[0], jitter[1]}, {fx, fy}, {float(hw), float(hh)},
                                 {prevJitter[0], prevJitter[1]}, lConst[2], lConst[3], {1.f / float(rw), 1.f / float(rh)}};
            const MPostParams po{{0, 0}, {int32_t(hw), int32_t(hh)}, {int32_t(rw), int32_t(rh)}, {1.f / fx, 1.f / fy}, {jitter[0], jitter[1]},
                                 {1.f / float(rw), 1.f / float(rh)}, lConst[4], reset ? 1u : 0u, 1.f, 0.f, {0, 0}, {fx, fy}, {0, 0}};
            const MDownParams dn{{int32_t(hw), int32_t(hh)}, {0, 0}, {0, 0}, {int32_t(rw), int32_t(rh)}, {float(rw) / float(ow), float(rh) / float(oh)},
                                 {jitter[0], jitter[1]}, 1.f};
            const uint32_t sw = exposureGrid(rw, ow), sh = exposureGrid(rh, oh), bx = (sw + 31) / 32, by = (sh + 31) / 32;
            const ExposureParams ex{{0, 0}, {int32_t(rw), int32_t(rh)}, 1.f / (float(sw) * float(sh)), 1.f, 0.23547087609767914f,
                                    1u | (reset ? 2u : 0u), bx * by, bx, {int32_t(sw), int32_t(sh)}};
            prevJitter[0] = jitter[0]; prevJitter[1] = jitter[1];
            char* pp = up + paramOffset;
            std::memcpy(pp, &e0, sizeof(e0)); std::memcpy(pp + 256, &d0, sizeof(d0)); std::memcpy(pp + 384, &po, sizeof(po));
            std::memcpy(pp + 512, &dn, sizeof(dn)); std::memcpy(pp + 640, &ex, sizeof(ex));
            for (unsigned l = 0; l < MLayers; ++l) {
                MLayerParams lp{int32_t(tw / MScale[l]), int32_t(th / MScale[l]), MShift[l] ^ int32_t((frameIndex & 1u) * 2u), MShift[l], {}};
                std::memcpy(lp.o, layers[l].data() + 2, sizeof(lp.o));
                std::memcpy(up + layerParamOffset + 512 * l, &lp, sizeof(lp));
            }
            if (L) {
                MLayerParams lp{int32_t(gw), int32_t(gh), lsx, 2, {}};
                std::memcpy(lp.o, lEnc0.data() + 2, sizeof(lp.o)); std::memcpy(up + layerParamOffset + 512 * MLayers, &lp, sizeof(lp));
                lp = {int32_t(gw), int32_t(gh), int32_t((frameIndex & 1u) * 2u), 0, {}};
                std::memcpy(lp.o, lDec0.data() + 2, sizeof(lp.o)); std::memcpy(up + layerParamOffset + 512 * (MLayers + 1), &lp, sizeof(lp));
            }
            for (int k = 0; k < 3; ++k)
                for (uint32_t y = 0; y < rh; ++y)
                    std::memcpy(up + inOffset[k] + inPitch[k] * y, inputs[3 * f + k].data() + inBytes[k] * y, inBytes[k]);
            const double t0 = now();
            Res* params[5] = {&enc0P, &dec0P, &postP, &downP, &expP};
            const UINT64 sizes[5] = {sizeof(e0), sizeof(d0), sizeof(po), sizeof(dn), sizeof(ex)}, offs[5] = {0, 256, 384, 512, 640};
            for (Res* r : params) d.use(*r, D3D12_RESOURCE_STATE_COPY_DEST);
            d.use(layerP, D3D12_RESOURCE_STATE_COPY_DEST);
            Res* textures[3] = {&color, &motion, &depth};
            for (Res* r : textures) d.use(*r, D3D12_RESOURCE_STATE_COPY_DEST);
            d.flushBarriers();
            for (int k = 0; k < 5; ++k) cl->CopyBufferRegion(params[k]->get(), 0, upload.get(), paramOffset + offs[k], sizes[k]);
            cl->CopyBufferRegion(layerP.get(), 0, upload.get(), layerParamOffset, 512 * (MLayers + 2));
            for (int k = 0; k < 3; ++k) {
                D3D12_TEXTURE_COPY_LOCATION dst{textures[k]->get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}}, src{upload.get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                dst.SubresourceIndex = 0;
                src.PlacedFootprint = {inOffset[k], {textures[k]->format, rw, rh, 1, UINT(inPitch[k])}};
                cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            ID3D12DescriptorHeap* heaps[] = {heap.heap.Get()};
            cl->SetDescriptorHeaps(1, heaps);
            UINT query = 0;
            auto stamp = [&](UINT q) { d.flushBarriers(); cl->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q); };
            stamp(query++);
            if (!ldr) {
                dispatch(exp0, exp0T, bx, by);
                dispatch(exp1, exp1T, 1, 1);
            }
            stamp(query++);
            const bool compact = compactTier && jitter[0] >= -0.5f && jitter[0] <= 0.5f && jitter[1] >= -0.5f && jitter[1] <= 0.5f;
            if (hipNetwork) {
                // front: the input stage writes the network's tokens; submit and wait, then run the HIP graph, then the
                // expansion reads its head (the CPU-serialized path of d3d12/hip_net_d3d12.h)
                if (L) {
                    // L's front is two stages: the feature slots (enc0_lf) and the block and merge, which write the tokens
                    dispatch(enc0, enc0T[i], (gw + 2 + 7) / 8, (gh + 2 + 7) / 8);
                    dispatch(enc0LPlanes, enc0LPlaneT[i], lbx, lby);
                } else
                    dispatch(enc0Planes, enc0PlaneT[i], (gw + 2 + 7) / 8, (gh + 2 + 7) / 8);
                stamp(query++);
                d.submit();
                hipNetwork->run();
                stamp(query++);     // after the network: its own milliseconds are from HIP's events
                if (L) dispatch(dec0LPlanes, dec0LPlaneT[i], (rw + (frameIndex & 1u) * 2u + 7) / 8, (rh + 7) / 8);
                else dispatch(dec0Planes, dec0PlaneT[i], gw / 8, gh / 8);
                stamp(query++);
                dispatch(compact ? postC : quality ? postQ : post, postT[i], (hw + 15) / 16, (hh + 15) / 16);
                stamp(query++);
                dispatch(down, downT, (ow + 7) / 8, (oh + 63) / 64);
                stamp(query++);
            } else {
                if (lx) dispatch(enc0LX, enc0LXT[i], lbx, lby);
                else {
                    dispatch(enc0, enc0T[i], (gw + 2 + 7) / 8, (gh + 2 + 7) / 8);
                    if (L) dispatch(enc0L, enc0LT, lbx, lby);
                }
                stamp(query++);
                for (unsigned l = 0; l < MLayers; ++l) {
                    const uint32_t w = tw / MScale[l], h = th / MScale[l];
                    const uint32_t gx = (w + (MShift[l] ^ ((frameIndex & 1u) * 2u)) + 7) / 8, gy = (h + MShift[l] + 7) / 8;
                    dispatch(shaders[tubeW2 && MScale[l] == 4 && gx * gy >= tubeW2Groups ? "m_tube_w2" : MShader[l]], layerT[l], gx, gy);
                    if (layerStamps) stamp(7 + l);
                }
                stamp(query++);
                // L: the blocks with a pixel of the render size, one more in x on frames whose expansion shift is 2
                if (L) dispatch(dec0, dec0T[i], (rw + (frameIndex & 1u) * 2u + 7) / 8, (rh + 7) / 8);
                else dispatch(dec0, dec0T[i], gw / 8, gh / 8);
                stamp(query++);
                dispatch(compact ? postC : quality ? postQ : post, postT[i], (hw + 15) / 16, (hh + 15) / 16);
                stamp(query++);
                dispatch(down, downT, (ow + 7) / 8, (oh + 63) / 64);
                stamp(query++);
            }
            cl->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 17, stampBuffer.get(), 0);
            if (pass == 0) {
                d.use(output, D3D12_RESOURCE_STATE_COPY_SOURCE); d.flushBarriers();
                D3D12_TEXTURE_COPY_LOCATION src{output.get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}}, dst{readback.get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                src.SubresourceIndex = 0;
                dst.PlacedFootprint = {0, {output.format, ow, oh, 1, UINT(outPitch)}};
                cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            d.submit();
            if (pass) walls.push_back(now() - t0);
            UINT64* t; CK(stampBuffer.r->Map(0, nullptr, reinterpret_cast<void**>(&t)));
            const double tick = 1e6 / double(frequency);
            double st[6]; for (int k = 0; k < 6; ++k) st[k] = double(t[k + 1] - t[k]) * tick;
            const double us = double(t[6] - t[0]) * tick;
            if (pass) for (int k = 0; k < 6; ++k) stage[k].push_back(st[k]);
            if (pass && layerStamps && !hipNetwork) for (unsigned l = 0; l < MLayers; ++l) layer[l].push_back(double(t[7 + l] - (l ? t[6 + l] : t[2])) * tick);
            stampBuffer.r->Unmap(0, nullptr);
            if (pass == 0 && dump && f == dumpFrame) {
                // development: the frame's intermediates (buffers whole, images as tightly packed rows)
                const std::pair<const char*, Res*> named[] = {{"tokens", hipNetwork ? &hipTokens : &tokens}, {"skip1", &skip1},
                    {"merged1", &merged1}, {"skip2", &skip2}, {"merged2", &merged2}, {"tubeA", &tubeA}, {"tubeB", &tubeB},
                    {"dec2", &dec2Out}, {"head", hipNetwork ? &hipHead : &dec1Out}, {"packed", &packed},
                    {"mvA", &mvA[i]}, {"mvB", &mvB}, {"mvC", &mvC}, {"feat", &feat[i]}, {"gates", &gates}, {"mix", &mix}, {"cov", &cov},
                    {"hist", &hist[i]}, {"half", &half[i]}};
                for (const auto& [name, r] : named) {
                    const auto desc = r->r->GetDesc();
                    const bool image = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}; UINT rows = 0; UINT64 rowBytes = 0, total = r->bytes;
                    if (image) dev->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &rows, &rowBytes, &total);
                    Res rb = d.buffer(total, D3D12_HEAP_TYPE_READBACK);
                    d.use(*r, D3D12_RESOURCE_STATE_COPY_SOURCE); d.flushBarriers();
                    if (image) {
                        D3D12_TEXTURE_COPY_LOCATION src{r->get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}}, dst{rb.get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                        dst.PlacedFootprint = fp;
                        cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                    } else cl->CopyBufferRegion(rb.get(), 0, r->get(), 0, r->bytes);
                    d.submit();
                    char* m; CK(rb.r->Map(0, nullptr, reinterpret_cast<void**>(&m)));
                    std::ofstream o(std::string(dump) + "-" + name + "-" + std::to_string(f) + ".bin", std::ios::binary);
                    if (image) for (UINT y = 0; y < rows; ++y) o.write(m + fp.Footprint.RowPitch * y, std::streamsize(rowBytes));
                    else o.write(m, std::streamsize(r->bytes));
                    rb.r->Unmap(0, nullptr);
                }
            }
            if (pass == 0) {
                char* m; CK(readback.r->Map(0, nullptr, reinterpret_cast<void**>(&m)));
                std::ofstream o(prefix + "-" + std::to_string(f) + ".rgba16f", std::ios::binary);
                for (uint32_t y = 0; y < oh; ++y) o.write(m + outPitch * y, std::streamsize(ow) * 8);
                readback.r->Unmap(0, nullptr);
                if (hipNetwork)
                    std::printf("frame %d: %.1f us GPU (front %.1f + network %.2f ms hip + back %.1f; %.1f us serialization)\n",
                                f, us, st[0] + st[1], hipNetwork->lastMilliseconds(), st[3] + st[4] + st[5],
                                us - (st[0] + st[1]) - st[3] - st[4] - st[5] - hipNetwork->lastMilliseconds());
                else
                    std::printf("frame %d: %.1f us GPU\n", f, us);
            } else { times.push_back(us); if (hipNetwork) networkMs.push_back(hipNetwork->lastMilliseconds()); }
            current = 1 - current; ++frameIndex;
        }
    if (!times.empty()) {
        std::sort(times.begin(), times.end()); std::printf("%zu timed frames: median %.1f us, min %.1f us GPU\n", times.size(), times[times.size() / 2], times[0]);
        std::sort(walls.begin(), walls.end()); std::printf("  wall clock, record to completion: median %.1f us, min %.1f us\n", walls[walls.size() / 2], walls[0]);
        const char* names[6] = {"exposure", "input", "network", "expansion", "reconstruction", "downsample"};
        for (int k = 0; k < 6; ++k) { std::sort(stage[k].begin(), stage[k].end()); std::printf("  %-14s median %.1f us\n", names[k], stage[k][stage[k].size() / 2]); }
        if (!networkMs.empty()) { std::sort(networkMs.begin(), networkMs.end()); std::printf("  network (hip)  median %.2f ms GPU, %u launches\n", networkMs[networkMs.size() / 2], hipNetwork->launches()); }
        if (!layer[0].empty()) {
            std::printf("  layers (us):");
            for (auto& v : layer) { std::sort(v.begin(), v.end()); std::printf(" %.1f", v[v.size() / 2]); }
            std::printf("\n");
        }
    }
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "d4r-m-d3d12: %s\n", e.what()); return 1; }
