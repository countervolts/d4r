#pragma once
#include "d3d12_external.h"
#include "rr_ngx_snapshot.h"
#include <memory>

namespace d4r::win::rr {
// Native RR execution on borrowed game resources. No fixture files, synthetic
// guide creation, child process, NVIDIA runtime, or CPU tensor transfer.
class Backend {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    Backend(ID3D12Device* device, const Creation& creation, const std::string& modelDirectory);
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    void validate_frame(const Snapshot& frame);
    void evaluate(ID3D12CommandQueue* queue, const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>& access);
    void drain();
};
}
