#pragma once
#include "d3d12_external.h"
#include "rr_ngx_snapshot.h"
#include <functional>
#include <memory>
#include <string>

namespace d4r::win::rr {
// Where the experimental FSR Ray Regeneration backend finds its pieces. All paths are explicit: nothing is
// searched for, and AMD's amdxc64.dll is never loaded.
struct FsrConfig {
    std::wstring vkd3dCore;    // d4r's vkd3d-proton d3d12core.dll, renamed so it cannot collide with the system's
    std::wstring denoiserDll;  // AMD's amd_fidelityfx_denoiser_dx12.dll (FidelityFX SDK 2.3), unmodified
    std::wstring amdxcShim;    // d4r's amdxc64 shim (tools/windows/amdxc64_shim)
    std::wstring cacheDir;     // vkd3d-proton's shader cache: the denoiser's shaders are compiled once
    static FsrConfig from_environment();
};

// FSR Ray Regeneration as the Ray Reconstruction backend on native Windows. The game's guides cross to a private
// vkd3d-proton device (D3D12 over Vulkan, dxil-spirv FP8 emulation) through the same shared packet the native
// DLSS backend uses; AMD's denoiser runs there. Evaluation is synchronous, as the native backend's is.
// The denoiser works at render resolution. When the output is larger (or a Super Sampling stage is wanted at all),
// the owner installs one with set_super_sampling(): the denoised colour is then copied into a texture on the game's
// device and handed to that callback together with the frame, which must record/submit the upscale into the game's
// output on `queue`. Without it an output larger than the render size is refused, never returned noisy.
class FsrBackend {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    FsrBackend(ID3D12Device* device, const Creation& creation, const FsrConfig& config);
    ~FsrBackend();
    FsrBackend(const FsrBackend&) = delete;
    FsrBackend& operator=(const FsrBackend&) = delete;
    using SuperSampling = std::function<void(ID3D12CommandQueue*, ID3D12Resource* denoised, const Snapshot&,
        const std::array<ResourceAccess, Snapshot::Count>&)>;
    void set_super_sampling(SuperSampling stage, const std::array<unsigned, 4>& renderLimits);
    void validate_frame(const Snapshot& frame);
    void evaluate(ID3D12CommandQueue* queue, const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>& access);
    void drain();
};
}
