// d3dhip: D3D12 resource sharing and adapter matching for the native HIP network. See hip_net_d3d12.h.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include "hip_net_d3d12.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi.h>
#include <devguid.h>
#include <setupapi.h>
#include <wrl/client.h>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include "../hip_net.h"

using Microsoft::WRL::ComPtr;

namespace d4r::dx {
namespace {
void hipCheck(hipError_t error, const char* what) {
    if (error != hipSuccess) throw std::runtime_error(std::string("HIP: ") + what + ": " + hipGetErrorString(error));
}

struct PciFunction {
    uint32_t domain = 0, bus = 0, device = 0, function = 0;
};
// hipDeviceGetPCIBusId's "dddd:bb:dd.f"
PciFunction parsePciBusId(const char* text) {
    PciFunction p;
    unsigned domain = 0, bus = 0, device = 0, function = 0;
    if (std::sscanf(text, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4)
        throw std::runtime_error(std::string("HIP: cannot parse the PCI bus id \"") + text + "\"");
    p.domain = domain; p.bus = bus; p.device = device; p.function = function;
    return p;
}
// The PCI identity of one bus/device/function, from the PnP device tree (Windows does not expose it through DXGI or
// HIP): vendor, device, subsystem and subsystem vendor, as a card's config space reports them.
struct PciIdentity {
    uint16_t vendor = 0, device = 0, subsystem = 0, subsystemVendor = 0;
};
bool pciIdentityOf(const PciFunction& at, PciIdentity& out) {
    const HDEVINFO set = SetupDiGetClassDevsW(nullptr, L"PCI", nullptr, DIGCF_PRESENT | DIGCF_ALLCLASSES);
    if (set == INVALID_HANDLE_VALUE) throw std::runtime_error("SetupDiGetClassDevs failed for the PCI enumerator");
    bool found = false;
    SP_DEVINFO_DATA data{};
    data.cbSize = sizeof(data);
    for (DWORD index = 0; !found && SetupDiEnumDeviceInfo(set, index, &data); ++index) {
        DWORD bus = 0, address = 0;
        if (!SetupDiGetDeviceRegistryPropertyW(set, &data, SPDRP_BUSNUMBER, nullptr, reinterpret_cast<PBYTE>(&bus), sizeof(bus), nullptr) ||
            !SetupDiGetDeviceRegistryPropertyW(set, &data, SPDRP_ADDRESS, nullptr, reinterpret_cast<PBYTE>(&address), sizeof(address), nullptr))
            continue;
        // SPDRP_ADDRESS: PCI device number in the high word, function number in the low word
        if (bus != at.bus || ((address >> 16) & 0xffffu) != at.device || (address & 0xffffu) != at.function) continue;
        wchar_t ids[512] = {};
        if (!SetupDiGetDeviceRegistryPropertyW(set, &data, SPDRP_HARDWAREID, nullptr, reinterpret_cast<PBYTE>(ids),
                                               sizeof(ids), nullptr))
            continue;
        // PCI\VEN_1002&DEV_73BF&SUBSYS_24011DA2&REV_C1 (SUBSYS: <subsystem id><subsystem vendor id>)
        unsigned vendor = 0, device = 0, subsystem = 0, subsystemVendor = 0, revision = 0;
        if (std::swscanf(ids, L"PCI\\VEN_%4x&DEV_%4x&SUBSYS_%4x%4x&REV_%x", &vendor, &device, &subsystem, &subsystemVendor,
                         &revision) != 5)
            continue;
        out.vendor = uint16_t(vendor);
        out.device = uint16_t(device);
        out.subsystem = uint16_t(subsystem);
        out.subsystemVendor = uint16_t(subsystemVendor);
        found = true;
    }
    SetupDiDestroyDeviceInfoList(set);
    return found;
}

bool sameLuid(const LUID& a, const LUID& b) { return a.LowPart == b.LowPart && a.HighPart == b.HighPart; }

// Present display adapters with AMD's PCI vendor, from the PnP device tree (DXGI's descriptions may be spoofed).
unsigned amdDisplayAdapters() {
    const HDEVINFO set = SetupDiGetClassDevsW(&GUID_DEVCLASS_DISPLAY, nullptr, nullptr, DIGCF_PRESENT);
    if (set == INVALID_HANDLE_VALUE) return 0;
    unsigned count = 0;
    SP_DEVINFO_DATA data{};
    data.cbSize = sizeof(data);
    for (DWORD index = 0; SetupDiEnumDeviceInfo(set, index, &data); ++index) {
        wchar_t ids[512] = {};
        if (!SetupDiGetDeviceRegistryPropertyW(set, &data, SPDRP_HARDWAREID, nullptr,
                                               reinterpret_cast<PBYTE>(ids), sizeof(ids), nullptr))
            continue;
        unsigned vendor = 0;
        if (std::swscanf(ids, L"PCI\\VEN_%4x", &vendor) == 1 && vendor == 0x1002) ++count;
    }
    SetupDiDestroyDeviceInfoList(set);
    return count;
}

// The HIP device must be the D3D12 device's adapter. HIP on Windows does not report the adapter LUID
// (hipDeviceAttributeLuid is unsupported on ROCm 7.14), so the match uses DXGI's description of the adapter the
// device names by LUID plus the card's PCI identity. OptiScaler spoofs that description (an NVIDIA card for the
// game): then the D3D12 device is still the enumerated adapter, and the match is accepted only when the HIP device
// is AMD and the machine has a single AMD display adapter, so no other card can be meant.
void checkAdapter(ID3D12Device* device) {
    const LUID luid = device->GetAdapterLuid();
    ComPtr<IDXGIFactory1> factory;
    const HRESULT created = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(created)) throw std::runtime_error("CreateDXGIFactory1 failed");
    DXGI_ADAPTER_DESC1 selected{};
    bool found = false;
    unsigned amd = 0;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> candidate;
        if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc{};
        candidate->GetDesc1(&desc);
        if (sameLuid(desc.AdapterLuid, luid)) {
            selected = desc;
            found = true;
        }
        if (desc.VendorId == 0x1002) ++amd;
    }
    if (!found) throw std::runtime_error("the D3D12 device's adapter is not one DXGI enumerates");
    hipCheck(hipInit(0), "hipInit");
    int devices = 0;
    hipCheck(hipGetDeviceCount(&devices), "hipGetDeviceCount");
    if (devices < 1) throw std::runtime_error("HIP reports no device");
    char busId[64] = {};
    hipCheck(hipDeviceGetPCIBusId(busId, sizeof(busId), 0), "hipDeviceGetPCIBusId");
    const PciFunction at = parsePciBusId(busId);
    PciIdentity pci;
    if (!pciIdentityOf(at, pci))
        throw std::runtime_error("cannot read the PCI identity of HIP device 0 at " + std::string(busId));
    if (pci.vendor != 0x1002)
        throw std::runtime_error("HIP device 0 is not an AMD adapter (PCI vendor " + std::to_string(pci.vendor) + "): AMD-only");
    const auto id = [](const PciIdentity& p) {
        char text[64];
        std::snprintf(text, sizeof(text), "%04x:%04x (subsystem %04x:%04x)", p.vendor, p.device, p.subsystemVendor, p.subsystem);
        return std::string(text);
    };
    if (selected.VendorId != 0x1002) {
        // a description OptiScaler replaced: only a machine with one AMD card makes HIP device 0 unambiguous
        const unsigned present = amdDisplayAdapters();
        if (amd > 1 || present > 1)
            throw std::runtime_error("the D3D12 adapter's description is spoofed (vendor " +
                                     std::to_string(selected.VendorId) + ") and this machine has " +
                                     std::to_string(std::max(amd, present)) +
                                     " AMD display adapters: cannot prove HIP device 0 is the D3D12 adapter's card");
        return;
    }
    // DXGI_ADAPTER_DESC1::SubSysId holds the card's subsystem and subsystem vendor IDs; accept either word order
    const uint32_t subsystemFirst = (uint32_t(pci.subsystem) << 16) | pci.subsystemVendor;
    const uint32_t vendorFirst = (uint32_t(pci.subsystemVendor) << 16) | pci.subsystem;
    const auto matches = [&](const DXGI_ADAPTER_DESC1& d) {
        return d.VendorId == pci.vendor && d.DeviceId == pci.device && (d.SubSysId == subsystemFirst || d.SubSysId == vendorFirst);
    };
    if (!matches(selected))
        throw std::runtime_error("HIP device 0's PCI identity " + id(pci) + " is not the D3D12 adapter's (" +
                                 std::to_string(selected.VendorId) + ":" + std::to_string(selected.DeviceId) + ")");
    // more than one AMD adapter with this card's identity: which one HIP device 0 is would be ambiguous
    unsigned matches_ = 0;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> candidate;
        if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc{};
        candidate->GetDesc1(&desc);
        if (matches(desc)) ++matches_;
    }
    if (matches_ > 1)
        throw std::runtime_error("more than one AMD adapter matches HIP device 0's PCI identity " + id(pci));
}

constexpr uint64_t AllocationGranularity = 65536;
uint64_t alignedBytes(uint64_t bytes) { return std::max<uint64_t>((bytes + AllocationGranularity - 1) / AllocationGranularity * AllocationGranularity, AllocationGranularity); }

ComPtr<ID3D12Resource> sharedBuffer(ID3D12Device* device, uint64_t bytes) {
    D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_SHARED, &desc,
                                               D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource))))
        throw std::runtime_error("cannot create the shared token/head resource");
    return resource;
}
// An NT handle for HIP: a duplicate, because hip_net's Windows import closes the handle it is given (hip_net.cpp),
// and these resources may need sharing again if construction fails.
HANDLE duplicateHandle(HANDLE handle) {
    HANDLE copy = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), handle, GetCurrentProcess(), &copy, 0, FALSE, DUPLICATE_SAME_ACCESS))
        throw std::runtime_error("DuplicateHandle failed for a shared GPU resource");
    return copy;
}
} // namespace

// ---------------------------------------------------------------- the parts the native HIP networks share
void checkHipAdapter(ID3D12Device* device) {
    if (!device) throw std::runtime_error("no D3D12 device");
    checkAdapter(device);
}
ID3D12Resource* createSharedBuffer(ID3D12Device* device, uint64_t bytes) {
    if (!device || !bytes) throw std::runtime_error("a shared network buffer needs a size");
    return sharedBuffer(device, alignedBytes(bytes)).Detach();
}
void* importSharedBuffer(ID3D12Device* device, ID3D12Resource* resource, uint64_t bytes, void** memory) {
    if (!device || !resource || !bytes || !memory) throw std::runtime_error("a shared network buffer needs a resource");
    HANDLE handle = nullptr;
    if (FAILED(device->CreateSharedHandle(resource, nullptr, GENERIC_ALL, nullptr, &handle)))
        throw std::runtime_error("CreateSharedHandle failed for a shared network buffer");
    const HANDLE forHip = duplicateHandle(handle);
    CloseHandle(handle);
    hipExternalMemoryHandleDesc desc{};
    desc.size = alignedBytes(bytes);
    desc.type = hipExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = forHip;
    hipExternalMemory_t imported = nullptr;
    const hipError_t result = hipImportExternalMemory(&imported, &desc);
    // hip_net's Windows import closes the handle it is given; keep the D3D12 side's own handle a duplicate so that a
    // failed import never closes someone else's
    CloseHandle(forHip);
    hipCheck(result, "hipImportExternalMemory");
    hipExternalMemoryBufferDesc buffer{};
    buffer.offset = 0;
    buffer.size = alignedBytes(bytes);
    void* at = nullptr;
    const hipError_t mapped = hipExternalMemoryGetMappedBuffer(&at, imported, &buffer);
    if (mapped != hipSuccess) {
        (void)hipDestroyExternalMemory(imported);
        hipCheck(mapped, "hipExternalMemoryGetMappedBuffer");
    }
    *memory = imported;
    return at;
}

struct HipNetworkDx::Impl {
    ComPtr<ID3D12Resource> tokens, head;
    std::unique_ptr<d4r::HipNetwork> network;
    ID3D12Device* device = nullptr;
    hipExternalSemaphore_t fence = nullptr;   // useFence(): the shared D3D12 fence, imported
    uint32_t launches = 0;
    float milliseconds = 0;
    ~Impl() {
        if (network) network->wait();         // nothing queued may outlive the semaphore or the buffers
        if (fence) (void)hipDestroyExternalSemaphore(fence);
    }
};

HipNetworkDx::HipNetworkDx(ID3D12Device* device, const std::string& modelDirectory, uint32_t width, uint32_t height,
                           uint64_t tokensBytes, uint64_t headBytes)
    : impl(std::make_unique<Impl>()) {
    if (!device) throw std::runtime_error("no D3D12 device");
    if (!width || !height || !tokensBytes || !headBytes) throw std::runtime_error("the network's dimensions and buffer sizes must be nonzero");
    const std::string hipnet = modelDirectory + "/hipnet.bin", kernels = modelDirectory + "/hip";
    if (!std::filesystem::exists(hipnet) || !std::filesystem::is_directory(kernels) ||
        std::filesystem::is_empty(kernels))
        throw std::runtime_error("the model has no native HIP network (" + hipnet + " and " + kernels + "/*.hsaco)");
    checkAdapter(device);
    impl->device = device;
    const uint64_t tokensAllocation = alignedBytes(tokensBytes), headAllocation = alignedBytes(headBytes);
    impl->tokens = sharedBuffer(device, tokensAllocation);
    impl->head = sharedBuffer(device, headAllocation);
    HANDLE tokens = nullptr, head = nullptr;
    if (FAILED(device->CreateSharedHandle(impl->tokens.Get(), nullptr, GENERIC_ALL, nullptr, &tokens)) ||
        FAILED(device->CreateSharedHandle(impl->head.Get(), nullptr, GENERIC_ALL, nullptr, &head)))
        throw std::runtime_error("CreateSharedHandle failed for the token/head resources");
    const HANDLE tokensForHip = duplicateHandle(tokens), headForHip = duplicateHandle(head);
    try {
        impl->network = std::make_unique<d4r::HipNetwork>(modelDirectory, width, height, tokensForHip, headForHip,
                                                         tokensAllocation, headAllocation);
    } catch (...) {
        CloseHandle(tokens);
        CloseHandle(head);
        throw;
    }
    // hip_net closed its duplicates; ours are the D3D12 side's share handles and are not needed any more
    CloseHandle(tokens);
    CloseHandle(head);
}
HipNetworkDx::~HipNetworkDx() = default;

ID3D12Resource* HipNetworkDx::tokens() const { return impl->tokens.Get(); }
ID3D12Resource* HipNetworkDx::head() const { return impl->head.Get(); }
void HipNetworkDx::run() {
    impl->network->launch();
    impl->network->wait();
    impl->milliseconds = impl->network->lastMilliseconds();
    ++impl->launches;
}
void HipNetworkDx::useFence(ID3D12Fence* fence) {
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
void HipNetworkDx::launchAfter(uint64_t wait, uint64_t signal) {
    if (!impl->fence) throw std::runtime_error("launchAfter() needs useFence() first");
    const auto stream = static_cast<hipStream_t>(impl->network->stream());
    hipExternalSemaphoreWaitParams waitParams{};
    waitParams.params.fence.value = wait;
    hipCheck(hipWaitExternalSemaphoresAsync(&impl->fence, &waitParams, 1, stream), "hipWaitExternalSemaphoresAsync");
    impl->network->launch();
    hipExternalSemaphoreSignalParams signalParams{};
    signalParams.params.fence.value = signal;
    hipCheck(hipSignalExternalSemaphoresAsync(&impl->fence, &signalParams, 1, stream), "hipSignalExternalSemaphoresAsync");
    ++impl->launches;
}
void HipNetworkDx::finish() {
    impl->network->wait();
    impl->milliseconds = impl->network->lastMilliseconds();
}
float HipNetworkDx::lastMilliseconds() { return impl->milliseconds; }
std::vector<float> HipNetworkDx::debugNodeMicros(uint32_t phase, uint32_t trials) {
    impl->network->wait();
    return impl->network->debugNodeMicros(phase, trials);
}
std::vector<std::pair<float, float>> HipNetworkDx::debugPhaseMicros(uint32_t trials) {
    impl->network->wait();
    std::vector<std::pair<float, float>> out;
    for (uint32_t phase = 0; phase < impl->network->phases(); ++phase) {
        float nodes = 0;
        for (float us : impl->network->debugNodeMicros(phase, trials)) nodes += us;
        out.emplace_back(impl->network->debugGraphMicros(phase, trials), nodes);
    }
    return out;
}
uint32_t HipNetworkDx::launches() const { return impl->launches; }
} // namespace d4r::dx
