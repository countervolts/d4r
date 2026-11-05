// d3dhip's bridge DLL: HipNetworkDx (hip_net_d3d12.h) behind a C table, so the NGX shim does not import the HIP
// runtime at load time. Loading amdhip64 into Ghost of Tsushima next to OptiScaler's driver hooks hung the GPU
// before the game's first frame; the shim loads this DLL on the first preset M frame whose model has the HIP
// network instead.
//
// Built as d4r_hip_net.dll beside nvngx.dll. Every entry catches its exceptions: the shim logs the message and
// keeps its DXIL network.
#include "hip_net_d3d12.h"
#include "hip_net_rr.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstring>
#include <memory>
#include <string>

namespace {
struct Net {
    std::unique_ptr<d4r::dx::HipNetworkDx> network;
};
struct RrNet {
    std::unique_ptr<d4r::dx::HipNetworkRr> network;
};

void setError(char* error, unsigned bytes, const char* text) {
    if (error == nullptr || bytes == 0) return;
    std::strncpy(error, text, bytes - 1);
    error[bytes - 1] = '\0';
}
// Runs `body` on an RR network, turning exceptions into *error: 0 done, -1 failed.
template <typename F> int rrCall(void* net, char* error, unsigned errorBytes, F body) {
    if (net == nullptr) {
        setError(error, errorBytes, "the network is not created");
        return -1;
    }
    try {
        body(*static_cast<RrNet*>(net)->network);
        return 0;
    } catch (const std::exception& e) {
        setError(error, errorBytes, e.what());
        return -1;
    }
}
} // namespace

extern "C" {
// The shared token/head resources and the network over them. Null: *error says why (missing files, another
// adapter, import failure).
__declspec(dllexport) void* d4rHipNetCreate(ID3D12Device* device, const char* directory, unsigned width,
                                            unsigned height, unsigned long long tokenBytes, unsigned long long headBytes,
                                            char* error, unsigned errorBytes) {
    try {
        auto net = std::make_unique<Net>();
        net->network = std::make_unique<d4r::dx::HipNetworkDx>(device, directory, width, height, tokenBytes, headBytes);
        return net.release();
    } catch (const std::exception& e) {
        setError(error, errorBytes, e.what());
        return nullptr;
    }
}

__declspec(dllexport) ID3D12Resource* d4rHipNetTokens(void* net) {
    return net != nullptr ? static_cast<Net*>(net)->network->tokens() : nullptr;
}

__declspec(dllexport) ID3D12Resource* d4rHipNetHead(void* net) {
    return net != nullptr ? static_cast<Net*>(net)->network->head() : nullptr;
}

// Launches the graph and blocks until it has finished. 0: done; otherwise *error says what failed.
__declspec(dllexport) int d4rHipNetRun(void* net, char* error, unsigned errorBytes) {
    if (net == nullptr) {
        setError(error, errorBytes, "the network is not created");
        return -1;
    }
    try {
        static_cast<Net*>(net)->network->run();
        return 0;
    } catch (const std::exception& e) {
        setError(error, errorBytes, e.what());
        return -1;
    }
}

// GPU-side ordering: imports `fence` (D3D12_FENCE_FLAG_SHARED) into HIP. 0: d4rHipNetLaunchAfter may be used;
// otherwise *error says why and d4rHipNetRun's CPU-serialized handoff remains.
__declspec(dllexport) int d4rHipNetUseFence(void* net, ID3D12Fence* fence, char* error, unsigned errorBytes) {
    if (net == nullptr) {
        setError(error, errorBytes, "the network is not created");
        return -1;
    }
    try {
        static_cast<Net*>(net)->network->useFence(fence);
        return 0;
    } catch (const std::exception& e) {
        setError(error, errorBytes, e.what());
        return -1;
    }
}
// Queues without blocking: HIP waits for the fence to reach `wait`, runs the network, signals `signal`.
__declspec(dllexport) int d4rHipNetLaunchAfter(void* net, unsigned long long wait, unsigned long long signal, char* error,
                                               unsigned errorBytes) {
    if (net == nullptr) {
        setError(error, errorBytes, "the network is not created");
        return -1;
    }
    try {
        static_cast<Net*>(net)->network->launchAfter(wait, signal);
        return 0;
    } catch (const std::exception& e) {
        setError(error, errorBytes, e.what());
        return -1;
    }
}
__declspec(dllexport) float d4rHipNetMilliseconds(void* net) {
    return net != nullptr ? static_cast<Net*>(net)->network->lastMilliseconds() : 0.0f;
}

__declspec(dllexport) void d4rHipNetDestroy(void* net) {
    delete static_cast<Net*>(net);
}
// Ray Reconstruction (hip_net_rr.h): the network over the engine's shared plane arena. presetF: a D4RR0002 package.
// Null: *error says why.
__declspec(dllexport) void* d4rHipRrCreate(ID3D12Device* device, const char* directory, int presetF, unsigned renderWidth,
                                           unsigned renderHeight, unsigned outputWidth, unsigned outputHeight,
                                           ID3D12Resource* arena, unsigned long long arenaBytes, char* error,
                                           unsigned errorBytes) {
    try {
        auto net = std::make_unique<RrNet>();
        net->network = std::make_unique<d4r::dx::HipNetworkRr>(device, directory, presetF != 0, renderWidth, renderHeight,
                                                               outputWidth, outputHeight, arena, arenaBytes);
        return net.release();
    } catch (const std::exception& e) {
        setError(error, errorBytes, e.what());
        return nullptr;
    }
}
// The engine's half (d4r::dx::ExternalNetworkRr, runtime_rr_d3d12.h), valid while the network lives.
__declspec(dllexport) const void* d4rHipRrExternal(void* net) {
    return net != nullptr ? &static_cast<RrNet*>(net)->network->external() : nullptr;
}
__declspec(dllexport) int d4rHipRrRun(void* net, char* error, unsigned errorBytes) {
    return rrCall(net, error, errorBytes, [](d4r::dx::HipNetworkRr& n) { n.run(); });
}
__declspec(dllexport) int d4rHipRrUseFence(void* net, ID3D12Fence* fence, char* error, unsigned errorBytes) {
    return rrCall(net, error, errorBytes, [fence](d4r::dx::HipNetworkRr& n) { n.useFence(fence); });
}
__declspec(dllexport) int d4rHipRrLaunchAfter(void* net, unsigned long long wait, unsigned long long signal, char* error,
                                              unsigned errorBytes) {
    return rrCall(net, error, errorBytes, [wait, signal](d4r::dx::HipNetworkRr& n) { n.launchAfter(wait, signal); });
}
__declspec(dllexport) void d4rHipRrDestroy(void* net) {
    delete static_cast<RrNet*>(net);
}
}
