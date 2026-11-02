// Native K startup/capability regression. Link with d4r_ngx_param_msvc.obj,
// d4r_ngx_param_host.cpp and d3d12. No NVIDIA binaries are needed.
#include <windows.h>
#include <d3d12.h>
#include <cstdio>
#include <stdexcept>
#include <string>

extern "C" {
unsigned d4r_ngx_get_int(void*, const char*, int*);
unsigned d4r_ngx_get_uint(void*, const char*, unsigned*);
unsigned d4r_ngx_get_void(void*, const char*, void**);
void d4r_ngx_set_int(void*, const char*, int);
void d4r_ngx_set_uint(void*, const char*, unsigned);
}

static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> static T symbol(HMODULE module, const char* name) {
    auto p = GetProcAddress(module, name); require(p != nullptr, name);
    return reinterpret_cast<T>(p);
}

int main(int argc, char** argv) {
    try {
        require(argc == 4, "usage: test-native-d3d12-shim SHIM HDR_MODEL LDR_MODEL");
        SetEnvironmentVariableA("D4R_ENGINE", "1");
        SetEnvironmentVariableA("D4R_DLSS_PRESET", nullptr);
        SetEnvironmentVariableA("D4R_ENGINE_MODEL_DIR", argv[2]);
        SetEnvironmentVariableA("D4R_ENGINE_MODEL_LDR_DIR", argv[3]);
        // These would fail if native startup accidentally touched CUDA/NGX.
        SetEnvironmentVariableA("D4R_NGX_CORE", "C:\\missing-d4r-test\\_nvngx.dll");
        SetEnvironmentVariableA("D4R_CUDA_DLL", "C:\\missing-d4r-test\\nvcuda.dll");
        auto module = LoadLibraryA(argv[1]); require(module != nullptr, "load shim");
        ID3D12Device* device = nullptr;
        require(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))), "create device");
        require(symbol<unsigned(*)()>(module, "d4r_WindowsBackendVersion")() == 1, "Windows backend handshake");
        require(symbol<unsigned(*)(ID3D12Device*)>(module, "d4r_D3D12_InstallCommandBackend")(device) == 1, "early hooks");
        auto init = symbol<unsigned(*)(unsigned long long, const wchar_t*, ID3D12Device*, unsigned, const void*)>(module, "NVSDK_NGX_D3D12_Init_Ext");
        require(init(42, L".", device, 0x15, nullptr) == 1, "native init without NVIDIA runtime");
        auto caps = symbol<unsigned(*)(void**)>(module, "NVSDK_NGX_D3D12_GetCapabilityParameters");
        auto destroy = symbol<unsigned(*)(void*)>(module, "NVSDK_NGX_D3D12_DestroyParameters");
        void* p = nullptr; require(caps(&p) == 1 && p, "capabilities");
        int available = -1;
        require(d4r_ngx_get_int(p, "SuperSampling.Available", &available) == 1 && available == 1, "SR available");
        require(d4r_ngx_get_int(p, "SuperSamplingDenoising.Available", &available) == 1 && available == 0, "RR unavailable");
        void* callback = nullptr;
        require(d4r_ngx_get_void(p, "DLSSOptimalSettingsCallback", &callback) == 1 && callback, "optimal callback");
        auto optimal = reinterpret_cast<unsigned(*)(void*)>(callback);
        d4r_ngx_set_uint(p, "Width", 2560); d4r_ngx_set_uint(p, "Height", 1440);
        d4r_ngx_set_int(p, "PerfQualityValue", 2);
        require(optimal(p) == 1, "quality optimal settings");
        unsigned w = 0, h = 0;
        d4r_ngx_get_uint(p, "OutWidth", &w); d4r_ngx_get_uint(p, "OutHeight", &h);
        require(w == 1707 && h == 960, "quality render dimensions");
        d4r_ngx_set_int(p, "PerfQualityValue", 99);
        require(optimal(p) != 1, "reject invalid quality");
        destroy(p);
        auto allocate = symbol<unsigned(*)(void**)>(module, "NVSDK_NGX_D3D12_AllocateParameters");
        require(allocate(&p) == 1, "parameters");
        d4r_ngx_set_uint(p, "Width", 640); d4r_ngx_set_uint(p, "Height", 360);
        d4r_ngx_set_uint(p, "OutWidth", 1280); d4r_ngx_set_uint(p, "OutHeight", 720);
        d4r_ngx_set_int(p, "PerfQualityValue", 2); d4r_ngx_set_int(p, "DLSS.Feature.Create.Flags", 3);
        auto create = symbol<unsigned(*)(void*, unsigned, void*, void**)>(module, "NVSDK_NGX_D3D12_CreateFeature");
        void* handle = nullptr;
        d4r_ngx_set_uint(p, "DLSS.Hint.Render.Preset.Quality", 13);
        require(create(nullptr, 1, p, &handle) != 1 && !handle, "reject M before evaluation");
        d4r_ngx_set_uint(p, "DLSS.Hint.Render.Preset.Quality", 0);
        d4r_ngx_set_int(p, "DLSS.Feature.Create.Flags", 2);
        const bool haveLdr = GetFileAttributesA((std::string(argv[3]) + "/offsets.bin").c_str()) != INVALID_FILE_ATTRIBUTES;
        const auto ldrResult = create(nullptr, 1, p, &handle);
        require(haveLdr ? ldrResult == 1 && handle : ldrResult != 1 && !handle, "LDR package availability at creation");
        auto release = symbol<unsigned(*)(void*)>(module, "NVSDK_NGX_D3D12_ReleaseFeature");
        if (handle) { require(release(handle) == 1, "release LDR"); handle = nullptr; }
        d4r_ngx_set_int(p, "DLSS.Feature.Create.Flags", 3);
        require(create(nullptr, 1, p, &handle) == 1 && handle, "default selects native K");
        require(release(handle) == 1, "release");
        destroy(p);
        require(symbol<unsigned(*)()>(module, "NVSDK_NGX_D3D12_Shutdown")() == 1, "native shutdown");
        require(GetModuleHandleA("nvcuda.dll") == nullptr, "CUDA was loaded");
        device->Release();
        std::puts("native init, capabilities, optimal settings, preset rejection, default K and shutdown passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
