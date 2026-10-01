#pragma once
#include "d3d12_external.h"
#include <functional>
#include <string>

namespace d4r::win::commands {
// Intercept only documented D3D12 COM methods on this device's runtime.
// A boundary becomes prefix -> callback -> suffix at ExecuteCommandLists,
// preserving preceding lists in the caller's batch and current-frame output.
using Boundary = std::function<void(ID3D12CommandQueue*)>;
void install(ID3D12Device* device);
void record_boundary(ID3D12GraphicsCommandList* list, Boundary callback);
std::string error();
size_t live_recordings();
D3D12_RESOURCE_STATES resource_state(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES fallback);
ResourceAccess resource_access(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, ResourceAccess fallback);
ResourceAccess submitted_resource_access(ID3D12Resource* resource, ResourceAccess fallback);
// Internal copies and callback submissions must bypass logical list routing.
struct InternalScope {
    bool previous;
    InternalScope();
    ~InternalScope();
};
}
