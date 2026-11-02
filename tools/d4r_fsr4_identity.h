// FSR 4 from OptiScaler's upscaler menu while the game sees an NVIDIA GPU.
//
// d4r's games run with WINE_HIDE_AMD_GPU (PROTON_FORCE_NVAPI), so every DXGI adapter reports vendor 0x10de. Three
// parties then refuse FSR 4 on the AMD GPU that runs it: OptiScaler's own GPU check (RDNA3/RDNA4 -> FSR 4 update,
// INT8 or FP8 model), AMD's amd_fidelityfx_upscaler_dx12.dll and the FSR 4 provider amdxcffx64.dll, each of which
// reads the adapter's vendor before using the driver extension (2026-10-17: with the identity hidden, FSR 4.1.1
// produced FSR 3.1's output byte for byte; amdxcffx64.dll never loaded unless Fsr4Update was forced, and then the
// upscaler still chose FSR 3.1).
//
// The adapter's GetDesc family is wrapped so that these callers alone - OptiScaler (found by its PE export name,
// whatever its loader file is called) and AMD's FidelityFX modules - read the GPU's real PCI vendor and device ids,
// from sysfs. The game, vkd3d-proton, NVIDIA's NGX libraries and everything else keep the NVIDIA identity, and only
// the adapter behind the game's D3D12 device is changed.
#pragma once

#include <windows.h>
#include <dxgi1_6.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "d4r_fsr_rr_identity.h"

namespace d4r_fsr4_identity
{
inline D4rPciIdentity g_real;
inline LUID g_luid = {};
// FSR Ray Regeneration's Proton provider needs the normal identity throughout discovery,
// including nested OptiScaler/driver-extension calls. Keep its scope local to the calling
// thread so the independent FSR upscaler still receives the real GPU identity.
inline thread_local unsigned g_originalIdentityDepth = 0;
struct OriginalIdentityScope
{
    OriginalIdentityScope() { ++g_originalIdentityDepth; }
    ~OriginalIdentityScope() { --g_originalIdentityDepth; }
    OriginalIdentityScope(const OriginalIdentityScope&) = delete;
    OriginalIdentityScope& operator=(const OriginalIdentityScope&) = delete;
};
using GetDescFn = HRESULT(STDMETHODCALLTYPE*)(IDXGIAdapter*, DXGI_ADAPTER_DESC*);
using GetDesc1Fn = HRESULT(STDMETHODCALLTYPE*)(IDXGIAdapter1*, DXGI_ADAPTER_DESC1*);
using GetDesc2Fn = HRESULT(STDMETHODCALLTYPE*)(IDXGIAdapter2*, DXGI_ADAPTER_DESC2*);
using GetDesc3Fn = HRESULT(STDMETHODCALLTYPE*)(IDXGIAdapter4*, DXGI_ADAPTER_DESC3*);
inline GetDescFn g_getDesc = nullptr;
inline GetDesc1Fn g_getDesc1 = nullptr;
inline GetDesc2Fn g_getDesc2 = nullptr;
inline GetDesc3Fn g_getDesc3 = nullptr;

// The DLL name in a module's PE export directory ("OptiScaler.dll" for OptiScaler under any file name).
inline bool export_name_is(HMODULE module, const char* expected)
{
    auto* base = reinterpret_cast<const unsigned char*>(module);
    auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return false;
    auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE)
        return false;
    const IMAGE_DATA_DIRECTORY& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (directory.VirtualAddress == 0 || directory.Size == 0)
        return false;
    auto* exports = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + directory.VirtualAddress);
    return exports->Name != 0 && _stricmp(reinterpret_cast<const char*>(base + exports->Name), expected) == 0;
}

// Whether the code at `address` belongs to OptiScaler or to one of AMD's FidelityFX/FSR 4 modules.
inline bool caller_sees_real(void* address)
{
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<const char*>(address), &module) || module == nullptr)
        return false;
    char path[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameA(module, path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH)
        return false;
    const char* slash = std::strrchr(path, '\\');
    const char* name = slash != nullptr ? slash + 1 : path;
    if (_strnicmp(name, "amd_fidelityfx_", 15) == 0 || _stricmp(name, "amdxcffx64.dll") == 0 ||
        _stricmp(name, "amdxc64.dll") == 0)
        return true;
    return export_name_is(module, "OptiScaler.dll");
}

template <typename Desc> inline void correct(void* caller, Desc* out)
{
    if (g_originalIdentityDepth == 0 && out->AdapterLuid.LowPart == g_luid.LowPart && out->AdapterLuid.HighPart == g_luid.HighPart &&
        caller_sees_real(caller))
    {
        out->VendorId = g_real.vendor;
        out->DeviceId = g_real.device;
    }
}

inline HRESULT STDMETHODCALLTYPE get_desc(IDXGIAdapter* self, DXGI_ADAPTER_DESC* desc)
{
    const HRESULT result = g_getDesc(self, desc);
    if (SUCCEEDED(result) && desc != nullptr)
        correct(__builtin_return_address(0), desc);
    return result;
}
inline HRESULT STDMETHODCALLTYPE get_desc1(IDXGIAdapter1* self, DXGI_ADAPTER_DESC1* desc)
{
    const HRESULT result = g_getDesc1(self, desc);
    if (SUCCEEDED(result) && desc != nullptr)
        correct(__builtin_return_address(0), desc);
    return result;
}
inline HRESULT STDMETHODCALLTYPE get_desc2(IDXGIAdapter2* self, DXGI_ADAPTER_DESC2* desc)
{
    const HRESULT result = g_getDesc2(self, desc);
    if (SUCCEEDED(result) && desc != nullptr)
        correct(__builtin_return_address(0), desc);
    return result;
}
inline HRESULT STDMETHODCALLTYPE get_desc3(IDXGIAdapter4* self, DXGI_ADAPTER_DESC3* desc)
{
    const HRESULT result = g_getDesc3(self, desc);
    if (SUCCEEDED(result) && desc != nullptr)
        correct(__builtin_return_address(0), desc);
    return result;
}

template <typename Fn> inline void patch_slot(void** vtable, int slot, Fn hook, Fn& original)
{
    if (vtable[slot] == reinterpret_cast<void*>(hook))
        return;
    DWORD protect = 0;
    if (!VirtualProtect(&vtable[slot], sizeof(void*), PAGE_READWRITE, &protect))
        return;
    original = reinterpret_cast<Fn>(vtable[slot]);
    vtable[slot] = reinterpret_cast<void*>(hook);
    VirtualProtect(&vtable[slot], sizeof(void*), protect, &protect);
}

// Installs the wrapper for the adapter of `device` when the GPU is really AMD's and DXGI reports someone else.
// Once per process; `detail` says what was found and done.
inline bool install(ID3D12Device* device, std::string& detail)
{
    const D4rPciIdentity reported = d4r_dxgi_identity(device);
    std::string pci;
    const D4rPciIdentity real = d4r_true_identity(device, pci);
    char text[640];
    std::snprintf(text, sizeof(text), "DXGI reports %04x:%04x; %s", reported.vendor, reported.device,
                  real.known ? pci.c_str() : ("real PCI identity unknown (" + pci + ")").c_str());
    detail = text;
    if (!real.known || real.vendor != 0x1002)
    {
        detail += "; not an AMD GPU: nothing to do";
        return false;
    }
    if (reported.known && reported.vendor == real.vendor)
    {
        detail += "; DXGI already reports the real vendor: nothing to do";
        return false;
    }
    g_real = real;
    device->GetAdapterLuid(&g_luid);
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory))))
    {
        detail += "; no DXGI factory";
        return false;
    }
    IDXGIAdapter4* adapter = nullptr;
    const HRESULT found = factory->EnumAdapterByLuid(g_luid, __uuidof(IDXGIAdapter4), reinterpret_cast<void**>(&adapter));
    factory->Release();
    if (FAILED(found) || adapter == nullptr)
    {
        detail += "; the device's adapter has no IDXGIAdapter4";
        return false;
    }
    // IDXGIAdapter4's vtable: GetDesc 8, GetDesc1 10, GetDesc2 11, GetDesc3 18. All adapters of the
    // implementation share it; the LUID check above keeps other adapters unchanged.
    void** vtable = *reinterpret_cast<void***>(adapter);
    patch_slot(vtable, 8, &get_desc, g_getDesc);
    patch_slot(vtable, 10, &get_desc1, g_getDesc1);
    patch_slot(vtable, 11, &get_desc2, g_getDesc2);
    patch_slot(vtable, 18, &get_desc3, g_getDesc3);
    adapter->Release();
    detail += "; OptiScaler and AMD's FidelityFX modules read the real ids (FSR 4)";
    return g_getDesc != nullptr && g_getDesc1 != nullptr && g_getDesc2 != nullptr && g_getDesc3 != nullptr;
}
} // namespace d4r_fsr4_identity
