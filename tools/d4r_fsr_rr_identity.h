// AMD's FSR Ray Regeneration runtime in a process that reports an NVIDIA GPU to the game.
//
// d4r's DLSS games run with WINE_HIDE_AMD_GPU (PROTON_FORCE_NVAPI): Vulkan, and so DXGI and
// vkd3d-proton, report vendor 0x10de, which is what lets the game offer DLSS at all. vkd3d-proton
// loads Proton's amdxc64.dll - the shim standing in for AMD's D3D12 driver-extension DLL, which
// forwards to vkd3d-proton's own extensions - only when Vulkan reports vendor 0x1002
// (d3d12_device_init_vendor_hacks), so under the spoof it never does. AMD's runtimes look that module
// up by name, and without it the unmodified denoiser 1.2.0 DLL enumerates no provider for the same
// RX 7700 XT it serves without the spoof (tools/fsr_rr_probe.cpp, 2026-10-13).
//
// For a GPU that sysfs says is AMD's, the module is loaded here exactly as vkd3d-proton itself would
// have. The game's view of the GPU is not changed, and nothing of AMD's is patched. The real PCI ids
// are read from sysfs (Z:\sys\bus\pci\devices\...) at the address Vulkan reports for vkd3d-proton's
// physical device, which the vendor-id spoof does not touch: nothing is assumed without that evidence.
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "d4r_vkd3d_interop.h"

struct D4rPciIdentity
{
    uint32_t vendor = 0, device = 0;
    bool known = false;
};

// The identity the D3D12 device's own DXGI adapter reports (spoofed or not).
inline D4rPciIdentity d4r_dxgi_identity(ID3D12Device* device)
{
    D4rPciIdentity identity;
    LUID luid;
    device->GetAdapterLuid(&luid);
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory))))
        return identity;
    IDXGIAdapter1* adapter = nullptr;
    if (SUCCEEDED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&adapter))))
    {
        DXGI_ADAPTER_DESC1 desc = {};
        if (SUCCEEDED(adapter->GetDesc1(&desc)))
            identity = {desc.VendorId, desc.DeviceId, true};
        adapter->Release();
    }
    factory->Release();
    return identity;
}

inline bool d4r_read_hex_file(const char* path, uint32_t& value)
{
    FILE* file = std::fopen(path, "rb");
    if (file == nullptr)
        return false;
    char text[32] = {};
    const size_t length = std::fread(text, 1, sizeof(text) - 1, file);
    std::fclose(file);
    if (length == 0)
        return false;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 16);
    if (end == text)
        return false;
    value = static_cast<uint32_t>(parsed);
    return true;
}

// The real PCI identity of the GPU behind a vkd3d-proton device: Vulkan's PCI bus address, then that
// device's sysfs vendor/device files.
inline D4rPciIdentity d4r_true_identity(ID3D12Device* device, std::string& detail, std::string* address = nullptr)
{
    D4rPciIdentity identity;
    ID3D12DXVKInteropDevice1* interop = nullptr;
    if (FAILED(device->QueryInterface(__uuidof(ID3D12DXVKInteropDevice1), reinterpret_cast<void**>(&interop))))
    {
        detail = "not a vkd3d-proton device";
        return identity;
    }
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice vkDevice = VK_NULL_HANDLE;
    const HRESULT handles = interop->GetVulkanHandles(&instance, &physical, &vkDevice);
    interop->Release();
    HMODULE vulkan = GetModuleHandleA("vulkan-1.dll");
    if (vulkan == nullptr)
        vulkan = LoadLibraryA("vulkan-1.dll");
    auto instanceProc = vulkan != nullptr ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                                                reinterpret_cast<void*>(GetProcAddress(vulkan, "vkGetInstanceProcAddr")))
                                          : nullptr;
    auto properties2 = instanceProc != nullptr && SUCCEEDED(handles)
                           ? reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                                 instanceProc(instance, "vkGetPhysicalDeviceProperties2"))
                           : nullptr;
    if (properties2 == nullptr)
    {
        detail = "no Vulkan physical-device query";
        return identity;
    }
    VkPhysicalDevicePCIBusInfoPropertiesEXT bus = {};
    bus.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
    VkPhysicalDeviceDriverProperties driver = {};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    driver.pNext = &bus;
    VkPhysicalDeviceProperties2 props = {};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &driver;
    properties2(physical, &props);
    if (bus.pciBus == 0 && bus.pciDevice == 0 && bus.pciDomain == 0 && bus.pciFunction == 0)
    {
        detail = "Vulkan reports no PCI bus address";
        return identity;
    }
    char base[96];
    std::snprintf(base, sizeof(base), "Z:\\sys\\bus\\pci\\devices\\%04x:%02x:%02x.%x\\", bus.pciDomain, bus.pciBus,
                  bus.pciDevice, bus.pciFunction);
    const std::string vendorPath = std::string(base) + "vendor", devicePath = std::string(base) + "device";
    if (!d4r_read_hex_file(vendorPath.c_str(), identity.vendor) || !d4r_read_hex_file(devicePath.c_str(), identity.device))
    {
        detail = "cannot read " + vendorPath;
        return identity;
    }
    identity.known = true;
    if (address != nullptr)
    {
        char pci[32];
        std::snprintf(pci, sizeof(pci), "%04x:%02x:%02x.%x", bus.pciDomain, bus.pciBus, bus.pciDevice, bus.pciFunction);
        *address = pci;
    }
    char text[256];
    std::snprintf(text, sizeof(text), "PCI %04x:%02x:%02x.%x is %04x:%04x (Vulkan reports %04x:%04x, driver %s)",
                  bus.pciDomain, bus.pciBus, bus.pciDevice, bus.pciFunction, identity.vendor, identity.device,
                  props.properties.vendorID, props.properties.deviceID, driver.driverName);
    detail = text;
    return identity;
}

// Makes Proton's amdxc64.dll available to AMD's runtime when the GPU really is AMD's. Returns false
// only when it should be there and could not be loaded; `detail` says what was found either way.
inline bool d4r_fsr_rr_driver_extension(ID3D12Device* device, std::string& detail)
{
    const D4rPciIdentity reported = d4r_dxgi_identity(device);
    std::string pci;
    const D4rPciIdentity real = d4r_true_identity(device, pci);
    char prefix[512];
    std::snprintf(prefix, sizeof(prefix), "DXGI reports %04x:%04x; %s", reported.vendor, reported.device,
                  real.known ? pci.c_str() : ("real PCI identity unknown (" + pci + ")").c_str());
    detail = prefix;
    if (!real.known || real.vendor != 0x1002)
    {
        detail += "; not an AMD GPU by sysfs: amdxc64.dll left alone";
        return true;
    }
    if (GetModuleHandleA("winevulkan.dll") == nullptr)
    {
        // A native Windows driver's amdxc64.dll is AMD's own and is not loaded from here.
        detail += "; not running on winevulkan: amdxc64.dll left alone";
        return true;
    }
    HMODULE extension = GetModuleHandleA("amdxc64.dll");
    const bool already = extension != nullptr;
    if (extension == nullptr)
        extension = LoadLibraryA("amdxc64.dll");
    if (extension == nullptr)
    {
        detail += "; amdxc64.dll (Proton's AMD driver-extension shim) is not available";
        return false;
    }
    char path[MAX_PATH] = {};
    GetModuleFileNameA(extension, path, MAX_PATH);
    detail += std::string(already ? "; amdxc64.dll already loaded (" : "; loaded amdxc64.dll (") + path + ")";
    return true;
}
