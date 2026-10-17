#pragma once

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// Native Windows uses the installed driver's normal D3D12 path. Do not import
// Proton's amdxc64 shim or Linux PCI/sysfs discovery into this process.
inline bool d4r_fsr_rr_driver_extension(ID3D12Device* device, std::string& detail)
{
    // Experimental: the denoiser only consults amdxc64.dll when it is already loaded in the process. With
    // D4R_FSR_AMDXC_SHIM=path, preload d4r's own shim (tools/windows/amdxc64_shim) for a vkd3d-proton device.
    // AMD's real amdxc64.dll is never loaded by this helper.
    if (const char* shim = std::getenv("D4R_FSR_AMDXC_SHIM")) {
        if (!LoadLibraryA(shim)) {
            detail = "D4R_FSR_AMDXC_SHIM could not be loaded";
            return false;
        }
    }
    LUID luid{};
    device->GetAdapterLuid(&luid);
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        detail = "native Windows: DXGI factory query failed";
        return false;
    }
    IDXGIAdapter1* adapter = nullptr;
    const HRESULT result = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
    factory->Release();
    if (FAILED(result)) {
        detail = "native Windows: device adapter query failed";
        return false;
    }
    DXGI_ADAPTER_DESC1 desc{};
    const HRESULT described = adapter->GetDesc1(&desc);
    adapter->Release();
    if (FAILED(described)) {
        detail = "native Windows: adapter description failed";
        return false;
    }
    char text[160];
    std::snprintf(text, sizeof(text), "native Windows DXGI %04x:%04x; installed D3D12 driver path",
                  desc.VendorId, desc.DeviceId);
    detail = text;
    return true;
}
