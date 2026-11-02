// d4r-rr-d3d12 MODEL INPUT_DIR OUTPUT_DIR [repeats] [hip]. Same frames.txt/raw inputs
// as d4r-rr. Saves all histories/stages with D4R_RR_DUMP=1, otherwise final
// RGB.
//
// A trailing `hip` argument (after [repeats]) runs the model's native HIP rr/rr2 stages between the engine's front
// (parameters, exposure and encoder 0) and its back (the output stage, or decoder 0's second block and the output
// passes for preset F): submit the front, wait on its fence on the CPU, run the HIP network, record and submit the
// back (d3d12/hip_net_rr.h). A model with hip/rr_*.hsaco (presets D/E) or hip/rr2_*.hsaco (preset F) takes that route
// by default; D4R_RR_DX_NET=dxil keeps the model's DXIL network.
#include "runtime_rr_d3d12.h"
#ifdef D4R_ENGINE_D3DHIP
#include "hip_net_rr.h"
#endif
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <dxgi1_6.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
using Microsoft::WRL::ComPtr;
#define CK(x)                                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        HRESULT hr_ = (x);                                                                                             \
        if (FAILED(hr_))                                                                                               \
            throw std::runtime_error(#x + std::string(": ") + std::to_string(hr_));                                    \
    } while (0)
namespace
{
#include "replay_resources.h"
UINT pixelBytes(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return 16;
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return 8;
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
        return 2;
    default:
        return 4;
    }
}
void upload(Device &d, Res &dst, const std::vector<char> &data)
{
    auto desc = dst.r->GetDesc();
    UINT64 size;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    d.dev->GetCopyableFootprints(&desc, 0, 1, 0, &fp, nullptr, nullptr, &size);
    UINT bytes = UINT(desc.Width) * pixelBytes(desc.Format);
    if (data.size() != size_t(bytes) * desc.Height)
        throw std::runtime_error("wrong RR input size");
    auto up = d.buffer(size, D3D12_HEAP_TYPE_UPLOAD);
    char *ptr;
    CK(up.r->Map(0, nullptr, reinterpret_cast<void **>(&ptr)));
    for (UINT y = 0; y < desc.Height; ++y)
        std::memcpy(ptr + fp.Footprint.RowPitch * y, data.data() + bytes * y, bytes);
    up.r->Unmap(0, nullptr);
    d.use(dst, D3D12_RESOURCE_STATE_COPY_DEST);
    d.flushBarriers();
    D3D12_TEXTURE_COPY_LOCATION src{up.get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}},
        target{dst.get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    src.PlacedFootprint = fp;
    d.list->CopyTextureRegion(&target, 0, 0, 0, &src, nullptr);
    d.use(dst, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    d.submit();
}
void save(Device &d, ID3D12Resource *image, D3D12_RESOURCE_STATES state, const std::string &file)
{
    auto desc = image->GetDesc();
    UINT64 size;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    d.dev->GetCopyableFootprints(&desc, 0, 1, 0, &fp, nullptr, nullptr, &size);
    auto rb = d.buffer(size, D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {image, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state, D3D12_RESOURCE_STATE_COPY_SOURCE};
    d.list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION src{image, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}},
        target{rb.get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    target.PlacedFootprint = fp;
    d.list->CopyTextureRegion(&target, 0, 0, 0, &src, nullptr);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    d.list->ResourceBarrier(1, &b);
    d.submit();
    char *ptr;
    CK(rb.r->Map(0, nullptr, reinterpret_cast<void **>(&ptr)));
    std::ofstream out(file, std::ios::binary);
    UINT bytes = UINT(desc.Width) * pixelBytes(desc.Format);
    for (UINT y = 0; y < desc.Height; ++y)
        out.write(ptr + fp.Footprint.RowPitch * y, bytes);
    rb.r->Unmap(0, nullptr);
}
} // namespace
int main(int argc, char **argv)
try
{
    // a trailing `hip` after [repeats]: the native HIP network between the engine's stages
    const bool hipArg = argc > 1 && !std::strcmp(argv[argc - 1], "hip");
    const int positional = argc - int(hipArg);
    if (positional < 4 || positional > 5)
    {
        std::cerr << "usage: d4r-rr-d3d12 MODEL INPUT_DIR OUTPUT_DIR [repeats] [hip]\n";
        return 2;
    }
    std::string input = argv[2], out = argv[3];
    int repeats = positional == 5 ? std::atoi(argv[4]) : 1;
    if (repeats < 1)
        throw std::runtime_error("repeats must be positive");
    std::filesystem::create_directories(out);
    std::ifstream frames(input + "/frames.txt");
    std::string line, key;
    uint32_t rw, rh, ow, oh;
    if (!std::getline(frames, line))
        throw std::runtime_error("missing frames.txt");
    std::istringstream dims(line);
    dims >> key >> rw >> rh >> ow >> oh;
    if (key != "size" || !dims)
        throw std::runtime_error("invalid RR size");
    Device d;
    ComPtr<IDXGIFactory6> factory;
    CK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter;
    CK(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
    if (std::getenv("D4R_D3D12_DEBUG"))
    {
        ComPtr<ID3D12Debug> debug;
        CK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
        debug->EnableDebugLayer();
    }
    CK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d.dev)));
    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    CK(d.dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&d.queue)));
    CK(d.dev->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&d.alloc)));
    CK(d.dev->CreateCommandList(0, qd.Type, d.alloc.Get(), nullptr, IID_PPV_ARGS(&d.list)));
    CK(d.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d.fence)));
    d4r::dx::RModel model(d.dev.Get(), argv[1]);
    // The native HIP network between the engine's stages (d3d12/hip_net_rr.h). A model with the hip code objects takes
    // that route by default; D4R_RR_DX_NET=dxil keeps the DXIL network, and any hip setup failure falls back to it.
    const char *netMode = std::getenv("D4R_RR_DX_NET");
    const bool dxilForced = netMode && !std::strcmp(netMode, "dxil");
    const bool hipModel =
        std::filesystem::exists(argv[1] + std::string("/hip/rr_enc1.hsaco")) ||
        std::filesystem::exists(argv[1] + std::string("/hip/rr2_enc1.hsaco"));
    std::unique_ptr<d4r::dx::ExternalNetworkRr> externalNetwork;
    std::unique_ptr<d4r::dx::REngine> enginePtr;
#ifdef D4R_ENGINE_D3DHIP
    std::unique_ptr<d4r::dx::HipNetworkRr> hipNetwork;
#endif
    if (!dxilForced && (hipArg || hipModel))
    {
#ifdef D4R_ENGINE_D3DHIP
        try
        {
            // the engine's arena is the network's too, so it must be shareable
            enginePtr = std::make_unique<d4r::dx::REngine>(model, rw, rh, ow, oh, true);
            hipNetwork = std::make_unique<d4r::dx::HipNetworkRr>(d.dev.Get(), argv[1], model.preset() == 6, rw, rh, ow,
                                                                 oh, enginePtr->arena(), enginePtr->arenaBytes());
            externalNetwork = std::make_unique<d4r::dx::ExternalNetworkRr>(hipNetwork->external());
            enginePtr->setExternalNetwork(externalNetwork.get());
            std::printf(hipArg ? "network: hip (native layers)\n"
                               : "network: hip (default; D4R_RR_DX_NET=dxil to force the DXIL network)\n");
        }
        catch (const std::exception &e)
        {
            hipNetwork.reset();
            externalNetwork.reset();
            enginePtr.reset();
            std::printf("network: hip unavailable (%s); using the DXIL network\n", e.what());
        }
#else
        std::printf("network: hip unavailable (built without the HIP SDK: set D4R_ROCM_DIR); using the DXIL network\n");
#endif
    }
    if (!enginePtr)
        enginePtr = std::make_unique<d4r::dx::REngine>(model, rw, rh, ow, oh);
    d4r::dx::REngine &engine = *enginePtr;
    const char *names[] = {"color",    "motion",      "depth",        "normals", "roughness", "diffuse",
                           "specular", "hitdistance", "disocclusion", "sss",     "particles"};
    DXGI_FORMAT formats[] = {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16_FLOAT,       DXGI_FORMAT_R32_FLOAT,
                             DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
                             DXGI_FORMAT_R8G8B8A8_UNORM,     DXGI_FORMAT_R32_FLOAT,          DXGI_FORMAT_R16_FLOAT,
                             DXGI_FORMAT_R16_FLOAT,          DXGI_FORMAT_R16G16B16A16_FLOAT};
    if (std::getenv("D4R_RR_TMP_MOTION4")) // TMP: the game's RGBA16F motion
        formats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
    std::vector<Res> textureSets[4]; // TMP: D4R_RR_TMP_ROTATE input sets, as a game's ping-pong targets
    const int rotate = std::getenv("D4R_RR_TMP_ROTATE") ? std::atoi(std::getenv("D4R_RR_TMP_ROTATE")) : 1;
    for (int k = 0; k < rotate; ++k)
        for (auto f : formats)
            textureSets[k].push_back(d.texture(rw, rh, f));
    auto output = d.texture(ow, oh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    d.use(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    d.submit();
    ComPtr<ID3D12QueryHeap> queries;
    D3D12_QUERY_HEAP_DESC qh{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 20, 0}; // TMP 20
    CK(d.dev->CreateQueryHeap(&qh, IID_PPV_ARGS(&queries)));
    auto stamps = d.buffer(20 * 8, D3D12_HEAP_TYPE_READBACK);
    UINT64 frequency;
    CK(d.queue->GetTimestampFrequency(&frequency));
    struct Stamp
    {
        ID3D12QueryHeap *heap;
        Device *device;
        bool split;
        std::string gate;
        uint32_t gateBoundary, frame = 0;
    };
    const bool split = std::getenv("D4R_RR_PROFILE_SPLIT") != nullptr;
    if (split)
        std::cerr << "RR profiling: separate stage submissions; timestamps include CPU waits and are not throughput measurements\n";
    const char *gate = std::getenv("D4R_RR_PROFILE_GATE");
    const char *boundary = std::getenv("D4R_RR_PROFILE_BOUNDARY");
    const int gateBoundary = boundary ? std::atoi(boundary) : 0;
    if (gate && (!split || gateBoundary < 1 || gateBoundary > 12))
        throw std::runtime_error("RR profile gate requires split submissions and boundary 1..12");
    Stamp stamp{queries.Get(), &d, split, gate ? gate : "", uint32_t(gateBoundary)};
    d4r::dx::FrameProfiler profiler{[](ID3D12GraphicsCommandList *list, uint32_t i, void *u) {
                                        auto &s = *static_cast<Stamp *>(u);
                                        list->EndQuery(s.heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
                                        if (i == 0)
                                            ++s.frame;
                                        if (s.split && i > 0 && i < 13)
                                            s.device->submit();
                                        // The third replay has warm data. Pause after the preceding
                                        // stage retires so RGP captures exactly the next submission.
                                        if (!s.gate.empty() && s.frame == 3 && i == s.gateBoundary)
                                        {
                                            std::ofstream(s.gate + ".ready") << i;
                                            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                                            while (!std::filesystem::exists(s.gate + ".go"))
                                            {
                                                if (std::chrono::steady_clock::now() >= deadline)
                                                    throw std::runtime_error("RR profile gate timed out");
                                                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                                            }
                                        }
                                    },
                                    &stamp};
    while (std::getline(frames, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream f(line);
        int id, reset, flags;
        d4r::dx::RFrameSettings s;
        f >> key >> id >> s.jitter[0] >> s.jitter[1] >> s.motionScale[0] >> s.motionScale[1] >> s.preExposure >>
            reset >> flags;
        for (float &x : s.worldToView)
            f >> x;
        for (float &x : s.viewToClip)
            f >> x;
        if (!f || key != "frame")
            throw std::runtime_error("invalid RR frame");
        s.reset = reset != 0;
        s.depthInverted = (flags & 1) != 0;
        s.hardwareDepth = (flags & 2) != 0;
        s.roughnessPacked = (flags & 4) != 0;
        auto &textures = textureSets[id % rotate];
        d4r::dx::RFrameImages images;
        d4r::dx::FrameTexture *slots[] = {&images.color,
                                          &images.motion,
                                          &images.depth,
                                          &images.normals,
                                          &images.roughness,
                                          &images.diffuseAlbedo,
                                          &images.specularAlbedo,
                                          &images.specularHitDistance,
                                          &images.disocclusionMask,
                                          &images.subsurfaceGuide,
                                          &images.colorBeforeParticles};
        for (size_t i = 0; i < textures.size(); ++i)
        {
            std::string path = input + "/" + std::to_string(id) + "-" + names[i] + ".bin";
            if ((i >= 7 || (i == 4 && model.preset() == 6 && s.roughnessPacked)) && !std::filesystem::exists(path))
                continue;
            upload(d, textures[i], slurp(path));
            *slots[i] = {textures[i].get(), formats[i]};
        }
        for (int rep = 0; rep < repeats; ++rep)
        {
            // Exercise rollback at every temporal frame, including camera/jitter history (not with an external
            // network: its front has already been submitted when it runs).
            if (std::getenv("D4R_RR_TEST_DISCARD") && !engine.externalNetwork())
            {
                engine.setFrame(images, s, {output.get(), output.format});
                engine.recordFrame(d.list.Get());
                CK(d.list->Close());
                CK(d.alloc->Reset());
                CK(d.list->Reset(d.alloc.Get(), nullptr));
                engine.discarded();
            }
            engine.setFrame(images, s, {output.get(), output.format});
            if (const d4r::dx::ExternalNetworkRr *net = engine.externalNetwork())
            {
                // d3dhip: the front retires on the CPU, then the network runs, then the back
                const auto started = std::chrono::steady_clock::now();
                engine.recordFront(d.list.Get());
                d.submit();                     // the front's fence retires on the CPU before the network runs
                net->network->run();
                engine.recordBack(d.list.Get());
                d.submit();
                engine.submitted(d.fence.Get(), d.fenceValue);
                const double elapsed =
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
                std::cout << "frame " << id << " repeat " << rep << " us " << elapsed << " serialized";
#ifdef D4R_ENGINE_D3DHIP
                std::cout << " hip " << hipNetwork->lastMilliseconds() * 1000.0;
#endif
                std::cout << "\n";
            }
            else
            {
                if (split)
                    engine.recordFrameForProfiling(d.list.Get(), profiler);
                else if (std::getenv("D4R_RR_TMP_NOPROF")) // TMP: record as the game does
                    engine.recordFrame(d.list.Get());
                else
                    engine.recordFrame(d.list.Get(), profiler);
                d.list->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, std::getenv("D4R_RR_TMP_FINE") ? 20 : 14, stamps.get(), 0); // TMP
                d.submit();
                engine.submitted(d.fence.Get(), d.fenceValue);
                UINT64 *t;
                CK(stamps.r->Map(0, nullptr, reinterpret_cast<void **>(&t)));
                double us = 1e6 / double(frequency);
                std::cout << "frame " << id << " repeat " << rep << " us " << (t[13] - t[0]) * us << " stages";
                for (int i = 1; i < 14; ++i)
                    std::cout << " " << (t[i] - t[i - 1]) * us;
                if (std::getenv("D4R_RR_TMP_FINE")) // TMP: gather, then each decoder's first kernel
                {
                    std::cout << " fine " << (t[14] - t[1]) * us;
                    for (int k = 0; k < 5; ++k)
                        std::cout << " " << (t[15 + k] - t[7 + k]) * us;
                }
                std::cout << "\n";
                stamps.r->Unmap(0, nullptr);
            }
        }
        save(d, output.get(), output.state, out + "/" + std::to_string(id) + "-out.bin");
        if (std::getenv("D4R_RR_DUMP") && id >= std::atoi(std::getenv("D4R_RR_DUMP"))) // TMP: from frame N
            for (auto &image : engine.debugImages())
                save(d, image.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     out + "/" + std::to_string(id) + "-" + image.name + ".bin");
    }
    if (std::getenv("D4R_D3D12_DEBUG"))
    {
        ComPtr<ID3D12InfoQueue> info;
        CK(d.dev.As(&info));
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i)
        {
            SIZE_T bytes = 0;
            CK(info->GetMessage(i, nullptr, &bytes));
            std::vector<char> storage(bytes);
            auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
            CK(info->GetMessage(i, message, &bytes));
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
                throw std::runtime_error(std::string("D3D12 debug layer: ") + message->pDescription);
        }
    }
    return 0;
}
catch (const std::exception &e)
{
    std::cerr << "d4r-rr-d3d12: " << e.what() << "\n";
    return 1;
}
