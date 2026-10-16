#pragma once

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <string>

// Native Windows uses the installed driver's normal D3D12 path. Do not import
// Proton's amdxc64 shim or Linux PCI/sysfs discovery into this process.
inline bool d4r_fsr_rr_driver_extension(ID3D12Device* device, std::string& detail)
{
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
