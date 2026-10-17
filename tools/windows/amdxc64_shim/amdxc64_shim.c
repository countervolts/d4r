// Experimental app-local amdxc64.dll for running AMD's FSR Ray Regeneration denoiser on a vkd3d-proton device
// (D3D12 over Vulkan) on native Windows RDNA3. AMD's real amdxc64.dll expects AMD's own ID3D12Device and must not
// see a vkd3d device. This shim only answers the capability queries the denoiser makes (recovered from the call
// sites in amd_fidelityfx_denoiser_dx12.dll 1.2.0.2740): it advertises the FP8 wave-matrix shape the denoiser's
// shaders use. The shaders themselves are executed by dxil-spirv's FP8 emulation inside vkd3d-proton, not by
// hardware FP8. It does not forward anything to AMD's driver.
#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
static void trace(const char* what, const GUID* g) {
    const char* path = getenv("D4R_AMDXC_LOG"); if (!path) return;
    FILE* f = fopen(path, "a"); if (!f) return;
    if (g) fprintf(f, "%s {%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}\n", what, g->Data1, g->Data2, g->Data3, g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6], g->Data4[7]);
    else fprintf(f, "%s\n", what);
    fclose(f);
}

static const GUID IID_Factory = {0x014937ec, 0x9288, 0x446f, {0xa9, 0xac, 0xd7, 0x5a, 0x8e, 0x3a, 0x98, 0x4f}};
static const GUID IID_WaveMatrix = {0xf714e11a, 0xb54e, 0x4e0f, {0xab, 0xc5, 0xdf, 0x58, 0xb1, 0x81, 0x33, 0xd1}};
static const GUID IID_Intrinsics = {0xba019d53, 0xccab, 0x4cbd, {0xb5, 0x6a, 0x72, 0x30, 0xed, 0x43, 0x30, 0xad}};
static const GUID IID_DeviceInfo = {0xb58d6601, 0x7401, 0x4234, {0x81, 0x80, 0x6f, 0xeb, 0xfc, 0x0e, 0x48, 0x4c}};

typedef struct Obj { void** vtbl; LONG refs; } Obj;
static ULONG WINAPI AddRef_(void* self) { return (ULONG)InterlockedIncrement(&((Obj*)self)->refs); }
static ULONG WINAPI Release_(void* self) { return (ULONG)InterlockedDecrement(&((Obj*)self)->refs); } // statics: never freed
static HRESULT WINAPI QI_(void* self, const GUID* iid, void** out) { (void)iid; *out = self; AddRef_(self); return S_OK; }
static HRESULT WINAPI Unused_(void* self) { (void)self; return (HRESULT)0x80004001; } // E_NOTIMPL
static HRESULT WINAPI Ok_(void* self) { (void)self; return S_OK; }
static HRESULT WINAPI OkOpcode_(void* self, uint32_t opcode) { (void)self; (void)opcode; trace("EnableOpcode", NULL); return S_OK; }

// IID_WaveMatrix slot 16: (count in/out, properties); entry = M, N, K (u64), then 0xb, 0xb, 1, 1 (u32), flag (u8)
static HRESULT WINAPI WaveMatrixProperties_(void* self, uint64_t* count, uint8_t* props) {
    (void)self; trace("WaveMatrixProperties", NULL);
    if (*count < 1) { *count = 1; return S_OK; }
    memset(props, 0, 0x30);
    uint64_t shape[3] = {16, 16, 16}; memcpy(props, shape, sizeof(shape));
    uint32_t types[4] = {0xb, 0xb, 1, 1}; memcpy(props + 0x18, types, sizeof(types));
    props[0x28] = 0;
    *count = 1;
    return S_OK;
}
// IID_DeviceInfo slot 3: (request[0x50], size). Command 2 fills 0x40 nonzero bytes at +0x10.
static HRESULT WINAPI DeviceInfo_(void* self, uint8_t* request, uint64_t size) {
    (void)self; trace("DeviceInfo", NULL);
    if (size < 0x50) return (HRESULT)0x80070057;
    uint32_t command; memcpy(&command, request, 4);
    if (command != 2) return (HRESULT)0x80004001;
    uint64_t info[8] = {0x1101, 1, 0x20, 0x2000, 64, 1, 1, 1}; // placeholders, see docs/windows-fsr-rr.md
    memcpy(request + 0x10, info, sizeof(info));
    return S_OK;
}

#define VT 24
static void* vt_factory[VT], *vt_wave[VT], *vt_intr[VT], *vt_info[VT];
static Obj factory = {vt_factory, 1}, wave = {vt_wave, 1}, intr = {vt_intr, 1}, info = {vt_info, 1};

static HRESULT WINAPI CreateInterface_(void* self, void* device, const GUID* iid, void** out) {
    (void)self; (void)device; *out = NULL; trace("CreateInterface", iid);
    Obj* o = NULL;
    if (IsEqualGUID(iid, &IID_WaveMatrix)) o = &wave;
    else if (IsEqualGUID(iid, &IID_Intrinsics)) o = &intr;
    else if (IsEqualGUID(iid, &IID_DeviceInfo)) o = &info;
    if (!o) return (HRESULT)0x80004002; // E_NOINTERFACE
    AddRef_(o); *out = o; return S_OK;
}
static void init_tables(void) {
    void** all[4] = {vt_factory, vt_wave, vt_intr, vt_info};
    for (int t = 0; t < 4; ++t) {
        all[t][0] = (void*)QI_; all[t][1] = (void*)AddRef_; all[t][2] = (void*)Release_;
        for (int i = 3; i < VT; ++i) all[t][i] = (void*)Unused_;
    }
    vt_factory[3] = (void*)CreateInterface_;
    vt_wave[16] = (void*)WaveMatrixProperties_;
    vt_intr[3] = (void*)Ok_; vt_intr[4] = (void*)OkOpcode_; vt_intr[5] = (void*)Ok_;
    vt_info[3] = (void*)DeviceInfo_;
}
__declspec(dllexport) HRESULT WINAPI AmdExtD3DCreateInterface(void* device, const GUID* iid, void** out) {
    static volatile LONG ready = 0; (void)device; *out = NULL; trace("AmdExtD3DCreateInterface", iid);
    if (!ready) { init_tables(); InterlockedExchange(&ready, 1); } // idempotent: concurrent init writes the same values
    if (!IsEqualGUID(iid, &IID_Factory)) return (HRESULT)0x80004002; // E_NOINTERFACE
    AddRef_(&factory); *out = &factory; return S_OK;
}
