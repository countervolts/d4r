// Experimental D4R-only Linux CUDA ABI over HIP, without ZLUDA's runtime or PTX translator.
// Modules come from an offline pack (tools/microcuda/make_pack.py: the code objects ZLUDA compiled
// for one exact DLSS DLL, ZLUDA build and switch set); anything outside it fails closed. Native
// kernel overrides, launch handling, arrays and texture/surface objects follow the patched ZLUDA
// runtime exactly (its module.rs, function.rs, memory.rs, array.rs, surf.rs) so that its code
// objects and d4r's native kernels see the same ABI. See ZLUDA_REPLACEMENT.md.
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <openssl/sha.h>

extern char** environ;

using CUresult = int;
using CUdeviceptr = uint64_t;
constexpr int SUCCESS=0, INVALID=1, OOM=2, NOT_INIT=3, INVALID_IMAGE=200,
    INVALID_HANDLE=400, NOT_FOUND=500, NOT_READY=600, UNSUPPORTED=801, UNKNOWN=999;

static int error(hipError_t e) {
    // HIP errors are not globally interchangeable with CUDA driver errors.
    switch (e) {
    case hipSuccess: return SUCCESS;
    case hipErrorInvalidValue: case hipErrorInvalidConfiguration:
    case hipErrorInvalidDevicePointer: return INVALID;
    case hipErrorOutOfMemory: return OOM;
    case hipErrorNotInitialized: return NOT_INIT;
    case hipErrorInvalidDevice: return 101;
    case hipErrorNoDevice: return 100;
    case hipErrorInvalidImage: return INVALID_IMAGE;
    case hipErrorInvalidContext: return 201;
    case hipErrorNoBinaryForGpu: return 209;
    case hipErrorInvalidHandle: return INVALID_HANDLE;
    case hipErrorNotFound: case hipErrorInvalidDeviceFunction: return NOT_FOUND;
    case hipErrorNotReady: return NOT_READY;
    case hipErrorNotSupported: return UNSUPPORTED;
    case hipErrorIllegalAddress: return 700;
    case hipErrorLaunchOutOfResources: return 701;
    default: return UNKNOWN;
    }
}
static std::atomic<bool> initialized{false};
static uint64_t fnv(const unsigned char* p, size_t size) {
    uint64_t h=0xcbf29ce484222325ULL;
    for(size_t i=0;i<size;++i) h=(h^p[i])*0x100000001b3ULL;
    return h;
}
static void note(const char* format,const char* a,const char* b="") {
    std::fprintf(stderr,"[d4r-microcuda] "); std::fprintf(stderr,format,a,b); std::fputc('\n',stderr);
}

// Allocation identity for prep reuse (d4r_prep_identity_at): every cuMemAlloc gets a fresh generation, every
// host-API write overlapping it bumps its epoch, and every free bumps free_generation (so slot tables keyed by
// address are reset: NGX recycles addresses after feature recreation). Device-side writes by kernels are not
// tracked: only weights uploaded by host APIs (NGX's model weights) may use identity reuse.
struct Allocation { size_t bytes; uint64_t generation; uint64_t epoch; };
static std::mutex allocations_lock;
static std::map<uint64_t,Allocation> allocations;
static uint64_t next_generation=1;
static std::atomic<uint64_t> free_generation{0};
static void track_alloc(uint64_t base,size_t bytes) {
    std::lock_guard<std::mutex> lock(allocations_lock); allocations[base]={bytes,next_generation++,0};
}
static void track_free(uint64_t base) {
    std::lock_guard<std::mutex> lock(allocations_lock); allocations.erase(base);
    free_generation.fetch_add(1,std::memory_order_release);
}
static void track_write(uint64_t dst,size_t bytes) {
    if(!bytes) return;
    std::lock_guard<std::mutex> lock(allocations_lock);
    auto it=allocations.upper_bound(dst);
    if(it!=allocations.begin()) --it;
    for(;it!=allocations.end()&&it->first<dst+bytes;++it)
        if(it->first+it->second.bytes>dst) ++it->second.epoch;
}
// -> false if p is not inside a tracked allocation
static bool identity_of(uint64_t p,uint64_t& generation,uint64_t& epoch) {
    std::lock_guard<std::mutex> lock(allocations_lock);
    auto it=allocations.upper_bound(p);
    if(it==allocations.begin()) return false;
    --it;
    if(p-it->first>=it->second.bytes) return false;
    generation=it->second.generation; epoch=it->second.epoch; return true;
}

extern "C" {
int cuInit(unsigned flags) {
    if(flags) return INVALID;
    int r=error(hipInit(0)); if(!r) initialized.store(true,std::memory_order_release); return r;
}
int cuDeviceGetCount(int* n) { return n?error(hipGetDeviceCount(n)):INVALID; }
int cuDeviceGet(int* d,int ordinal) { return d?error(hipDeviceGet(d,ordinal)):INVALID; }
// Contexts are bookkeeping over HIP's per-device state, as in ZLUDA (context.rs): no hipCtx API
// (deprecated in HIP), just hipSetDevice and a per-thread current-context stack.
struct Context { int device; };
static thread_local std::vector<Context*> context_stack;
static int make_current(Context* c) { return c?error(hipSetDevice(c->device)):SUCCESS; }
int cuCtxCreate_v2(Context** c,unsigned,int d) {
    if(!c) return INVALID;
    int count=0; if(int r=error(hipGetDeviceCount(&count))) return r;
    if(d<0||d>=count) return 101;
    *c=new Context{d};
    if(int r=make_current(*c)) { delete *c; *c=nullptr; return r; }
    context_stack.push_back(*c); return SUCCESS;
}
int cuCtxDestroy_v2(Context* c) {
    if(!c) return INVALID;
    context_stack.erase(std::remove(context_stack.begin(),context_stack.end(),c),context_stack.end());
    delete c; return SUCCESS;
}
int cuCtxPushCurrent_v2(Context* c) { if(int r=make_current(c)) return r; context_stack.push_back(c); return SUCCESS; }
int cuCtxPopCurrent_v2(Context** c) {
    if(context_stack.empty()) return 201;
    if(c) *c=context_stack.back();
    context_stack.pop_back();
    return context_stack.empty()?SUCCESS:make_current(context_stack.back());
}
int cuCtxSetCurrent(Context* c) {
    if(int r=make_current(c)) return r;
    if(context_stack.empty()) context_stack.push_back(c); else context_stack.back()=c;
    if(!c) context_stack.pop_back();
    return SUCCESS;
}
int cuCtxGetCurrent(Context** c) { if(!c) return INVALID; *c=context_stack.empty()?nullptr:context_stack.back(); return SUCCESS; }
int cuCtxGetDevice(int* d) {
    if(!d) return INVALID;
    if(context_stack.empty()||!context_stack.back()) return 201;
    *d=context_stack.back()->device; return SUCCESS;
}
int cuCtxSynchronize() { return error(hipDeviceSynchronize()); }
int cuDeviceGetName(char* p,int n,int d) { return p&&n>0?error(hipDeviceGetName(p,n,d)):INVALID; }
int cuDeviceGetUuid(hipUUID* uuid,int d) { return uuid?error(hipDeviceGetUuid(uuid,d)):INVALID; }
int cuDeviceGetLuid(char* luid,unsigned* mask,int d) {
    if(!luid||!mask) return INVALID;
    hipDeviceProp_t props{};
    int r=error(hipGetDeviceProperties(&props,d));
    if(!r) { std::memcpy(luid,props.luid,8); *mask=props.luidDeviceNodeMask; }
    return r;
}
int cuDeviceTotalMem_v2(size_t* bytes,int d) { return bytes?error(hipDeviceTotalMem(bytes,d)):INVALID; }
int cuDeviceComputeCapability(int* major,int* minor,int d) {
    int ignored; if(!major||!minor) return INVALID;
    int r=error(hipDeviceGetAttribute(&ignored,hipDeviceAttributeMultiprocessorCount,d));
    if(!r) { *major=8; *minor=9; } return r; // same NGX sm_89 selection as D4R reference
}
int cuDriverGetVersion(int* v) { if(!v) return INVALID; *v=12080; return SUCCESS; }
int cuDeviceGetAttribute(int* out,int attribute,int d) {
    if(!out) return INVALID;
    if(attribute==75||attribute==76) { int ma,mi; int r=cuDeviceComputeCapability(&ma,&mi,d);
        if(!r) *out=attribute==75?ma:mi;
        return r; }
    // Explicit CUDA->HIP enum map. Only the currently audited subset.
    hipDeviceAttribute_t h;
    switch(attribute) {
    case 1: h=hipDeviceAttributeMaxThreadsPerBlock; break;
    case 2: h=hipDeviceAttributeMaxBlockDimX; break;
    case 3: h=hipDeviceAttributeMaxBlockDimY; break;
    case 4: h=hipDeviceAttributeMaxBlockDimZ; break;
    case 5: h=hipDeviceAttributeMaxGridDimX; break;
    case 6: h=hipDeviceAttributeMaxGridDimY; break;
    case 7: h=hipDeviceAttributeMaxGridDimZ; break;
    case 8: h=hipDeviceAttributeMaxSharedMemoryPerBlock; break;
    case 10: { int v; int r=error(hipDeviceGetAttribute(&v,hipDeviceAttributeWarpSize,d));
        if(!r) *out=32;
        return r; }
    case 13: h=hipDeviceAttributeClockRate; break;
    case 16: h=hipDeviceAttributeMultiprocessorCount; break;
    case 22: h=hipDeviceAttributeMaxTexture2DWidth; break;
    case 23: h=hipDeviceAttributeMaxTexture2DHeight; break;
    case 35: { int ignored; int r=error(hipDeviceGetAttribute(&ignored,
        hipDeviceAttributeMultiprocessorCount,d)); if(!r) *out=0; return r; } // TCC_DRIVER
    default: return UNSUPPORTED;
    }
    return error(hipDeviceGetAttribute(out,h,d));
}
// cuMemAlloc zero-fills like ZLUDA (memory.rs alloc_v2): NGX output must not depend on stale VRAM.
int cuMemAlloc_v2(CUdeviceptr* p,size_t bytes) {
    if(!p) return INVALID;
    void* raw=nullptr; int r=error(hipMalloc(&raw,bytes)); if(r) return r;
    if((r=error(hipMemset(raw,0,bytes)))) { auto ignored=hipFree(raw); (void)ignored; return r; }
    *p=reinterpret_cast<CUdeviceptr>(raw); track_alloc(*p,bytes); return SUCCESS;
}
int cuMemFree_v2(CUdeviceptr p) { track_free(p); return error(hipFree(reinterpret_cast<void*>(p))); }
int cuMemHostAlloc(void** p,size_t bytes,unsigned flags) { return p?error(hipHostMalloc(p,bytes,flags)):INVALID; }
int cuMemAllocHost_v2(void** p,size_t bytes) { return cuMemHostAlloc(p,bytes,0); }
int cuMemFreeHost(void* p) { return error(hipHostFree(p)); }
int cuMemGetInfo_v2(size_t* free,size_t* total) { return free&&total?error(hipMemGetInfo(free,total)):INVALID; }
int cuMemcpyHtoD_v2(CUdeviceptr d,const void* s,size_t n) { track_write(d,n); return error(hipMemcpyHtoD(reinterpret_cast<void*>(d),const_cast<void*>(s),n)); }
int cuMemcpyDtoH_v2(void* d,CUdeviceptr s,size_t n) { return error(hipMemcpyDtoH(d,reinterpret_cast<void*>(s),n)); }
int cuMemcpyDtoD_v2(CUdeviceptr d,CUdeviceptr s,size_t n) { track_write(d,n); return error(hipMemcpyDtoD(reinterpret_cast<void*>(d),reinterpret_cast<void*>(s),n)); }
int cuMemcpyHtoDAsync_v2(CUdeviceptr d,const void* s,size_t n,hipStream_t st) { track_write(d,n); return error(hipMemcpyHtoDAsync(reinterpret_cast<void*>(d),const_cast<void*>(s),n,st)); }
int cuMemcpyDtoHAsync_v2(void* d,CUdeviceptr s,size_t n,hipStream_t st) { return error(hipMemcpyDtoHAsync(d,reinterpret_cast<void*>(s),n,st)); }
int cuMemcpyDtoDAsync_v2(CUdeviceptr d,CUdeviceptr s,size_t n,hipStream_t st) { track_write(d,n); return error(hipMemcpyDtoDAsync(reinterpret_cast<void*>(d),reinterpret_cast<void*>(s),n,st)); }
int cuMemsetD8_v2(CUdeviceptr d,unsigned char v,size_t n) { track_write(d,n); return error(hipMemset(reinterpret_cast<void*>(d),v,n)); }
int cuEventCreate(hipEvent_t* e,unsigned flags) { return e?error(hipEventCreateWithFlags(e,flags)):INVALID; }
int cuEventDestroy_v2(hipEvent_t e) { return error(hipEventDestroy(e)); }
int cuEventRecord(hipEvent_t e,hipStream_t s) { return error(hipEventRecord(e,s)); }
int cuEventQuery(hipEvent_t e) { return error(hipEventQuery(e)); }
int cuEventSynchronize(hipEvent_t e) { return error(hipEventSynchronize(e)); }
int cuEventElapsedTime(float* ms,hipEvent_t a,hipEvent_t b) { return ms?error(hipEventElapsedTime(ms,a,b)):INVALID; }
int cuStreamCreate(hipStream_t* s,unsigned flags) { return s?error(hipStreamCreateWithFlags(s,flags)):INVALID; }
int cuStreamDestroy_v2(hipStream_t s) { return error(hipStreamDestroy(s)); }
int cuStreamSynchronize(hipStream_t s) { return error(hipStreamSynchronize(s)); }
int cuStreamQuery(hipStream_t s) { return error(hipStreamQuery(s)); }
int cuStreamWaitEvent(hipStream_t s,hipEvent_t e,unsigned flags) { return error(hipStreamWaitEvent(s,e,flags)); }
// Unimplemented in ZLUDA, which returns NOT_SUPPORTED; the bridge falls back to its own records on that
// error (descriptors it stored at creation, linear-texture stand-ins), so these must fail the same way.
int cuDestroyExternalMemory(void*) { return UNSUPPORTED; }
int cuMipmappedArrayDestroy(void*) { return UNSUPPORTED; }
int cuGetErrorString(int e,const char** p) {
    if(!p) return INVALID;
    switch(e) {
    case 0: *p="success"; break; case 1: *p="invalid value"; break;
    case 200: *p="module outside the MicroCUDA pack, or a pack/native file that fails verification"; break;
    case 801: *p="outside the MicroCUDA subset"; break;
    default: *p="MicroCUDA error (see numeric CUDA driver code)";
    } return SUCCESS;
}
}

// ---- arrays, 2D copies, texture and surface objects (ZLUDA array.rs, memory.rs, surf.rs) ----

// CUDA formats HIP's legacy descriptor lacks get equivalent backing storage; the original format is
// remembered per array for surface descriptors and packed copies.
static std::mutex resources_lock;
static std::unordered_map<hipArray_t,uint32_t> original_formats;
static uint32_t original_format(hipArray_t a) {
    std::lock_guard<std::mutex> lock(resources_lock);
    auto it=original_formats.find(a); return it==original_formats.end()?0:it->second;
}
extern "C" int cuArrayCreate_v2(hipArray_t* handle,const HIP_ARRAY_DESCRIPTOR* desc) {
    if(!handle||!desc) return INVALID;
    int format,channels;
    switch(static_cast<int>(desc->Format)) {
    case 80: if(desc->NumChannels!=4) return error(hipArrayCreate(handle,desc));
             format=32; channels=4; break;                 // UNORM 10:10:10:2 -> RGBA32F backing
    case 192: format=1; channels=1; break; case 193: format=1; channels=2; break;   // UNORM_INT8
    case 194: format=1; channels=4; break;
    case 195: format=2; channels=1; break; case 196: format=2; channels=2; break;   // UNORM_INT16
    case 197: format=2; channels=4; break;
    case 198: format=8; channels=1; break; case 199: format=8; channels=2; break;   // SNORM_INT8
    case 200: format=8; channels=4; break;
    case 201: format=9; channels=1; break; case 202: format=9; channels=2; break;   // SNORM_INT16
    case 203: format=9; channels=4; break;
    default: return error(hipArrayCreate(handle,desc));
    }
    HIP_ARRAY_DESCRIPTOR backing=*desc;
    backing.Format=static_cast<hipArray_Format>(format); backing.NumChannels=channels;
    int r=error(hipArrayCreate(handle,&backing)); if(r) return r;
    std::lock_guard<std::mutex> lock(resources_lock);
    original_formats[*handle]=static_cast<uint32_t>(desc->Format); return SUCCESS;
}
extern "C" int cuArray3DCreate_v2(hipArray_t* handle,const HIP_ARRAY3D_DESCRIPTOR* desc) {
    return handle&&desc?error(hipArray3DCreate(handle,desc)):INVALID;
}
extern "C" int cuArrayDestroy(hipArray_t a) {
    int r=error(hipArrayDestroy(a)); if(r) return r;
    std::lock_guard<std::mutex> lock(resources_lock); original_formats.erase(a); return SUCCESS;
}
extern "C" int cuArrayGetDescriptor_v2(void*,hipArray_t) { return UNSUPPORTED; } // as ZLUDA (see above)

static uint32_t packed_component(float v,uint32_t scale) {
    float c=std::isnan(v)?0.f:std::clamp(v,0.f,1.f);
    return static_cast<uint32_t>(std::floor(c*static_cast<float>(scale)+0.5f));
}
// Packed 10:10:10:2 pixels between public 4-byte words and the RGBA32F backing array.
static int copy_2d_packed(const hip_Memcpy2D& m,bool src_packed,bool dst_packed) {
    if(m.WidthInBytes%4||(src_packed&&m.srcXInBytes%4)||(dst_packed&&m.dstXInBytes%4)) return INVALID;
    const size_t backing_width=m.WidthInBytes*4;
    if(src_packed&&dst_packed) {
        hip_Memcpy2D b=m; b.srcXInBytes*=4; b.dstXInBytes*=4; b.WidthInBytes=backing_width;
        return error(hipMemcpyParam2D(&b));
    }
    if((src_packed&&m.dstMemoryType==hipMemoryTypeArray)||(dst_packed&&m.srcMemoryType==hipMemoryTypeArray))
        return INVALID;
    const size_t width=m.WidthInBytes;
    std::vector<float> backing(backing_width/4*m.Height);
    std::vector<uint8_t> packed(width*m.Height);
    if(dst_packed) {
        // external (host or device) packed words -> host staging
        if(m.srcMemoryType==hipMemoryTypeHost) {
            if(!m.srcHost||m.srcXInBytes+width>m.srcPitch) return INVALID;
            for(size_t row=0;row<m.Height;++row)
                std::memcpy(packed.data()+row*width,static_cast<const uint8_t*>(m.srcHost)+(m.srcY+row)*m.srcPitch+m.srcXInBytes,width);
        } else if(m.srcMemoryType==hipMemoryTypeDevice||m.srcMemoryType==hipMemoryTypeUnified) {
            hip_Memcpy2D s=m; s.dstMemoryType=hipMemoryTypeHost; s.dstHost=packed.data();
            s.dstXInBytes=0; s.dstY=0; s.dstPitch=width;
            if(int r=error(hipMemcpyParam2D(&s))) return r;
        } else return UNSUPPORTED;
        for(size_t p=0;p<packed.size()/4;++p) {
            uint32_t bits; std::memcpy(&bits,packed.data()+4*p,4);
            backing[4*p]=static_cast<float>(bits&1023)/1023.f;
            backing[4*p+1]=static_cast<float>((bits>>10)&1023)/1023.f;
            backing[4*p+2]=static_cast<float>((bits>>20)&1023)/1023.f;
            backing[4*p+3]=static_cast<float>(bits>>30)/3.f;
        }
        hip_Memcpy2D t=m; t.srcMemoryType=hipMemoryTypeHost; t.srcHost=backing.data();
        t.srcXInBytes=0; t.srcY=0; t.srcPitch=backing_width; t.dstXInBytes=m.dstXInBytes*4; t.WidthInBytes=backing_width;
        return error(hipMemcpyParam2D(&t));
    }
    hip_Memcpy2D t=m; t.srcXInBytes=m.srcXInBytes*4; t.dstMemoryType=hipMemoryTypeHost; t.dstHost=backing.data();
    t.dstXInBytes=0; t.dstY=0; t.dstPitch=backing_width; t.WidthInBytes=backing_width;
    if(int r=error(hipMemcpyParam2D(&t))) return r;
    for(size_t p=0;p<packed.size()/4;++p) {
        const float* c=backing.data()+4*p;
        uint32_t bits=packed_component(c[0],1023)|packed_component(c[1],1023)<<10|
                      packed_component(c[2],1023)<<20|packed_component(c[3],3)<<30;
        std::memcpy(packed.data()+4*p,&bits,4);
    }
    if(m.dstMemoryType==hipMemoryTypeHost) {
        if(!m.dstHost||m.dstXInBytes+width>m.dstPitch) return INVALID;
        for(size_t row=0;row<m.Height;++row)
            std::memcpy(static_cast<uint8_t*>(m.dstHost)+(m.dstY+row)*m.dstPitch+m.dstXInBytes,packed.data()+row*width,width);
        return SUCCESS;
    }
    if(m.dstMemoryType!=hipMemoryTypeDevice&&m.dstMemoryType!=hipMemoryTypeUnified) return UNSUPPORTED;
    hip_Memcpy2D s=m; s.srcMemoryType=hipMemoryTypeHost; s.srcHost=packed.data();
    s.srcXInBytes=0; s.srcY=0; s.srcPitch=width;
    return error(hipMemcpyParam2D(&s));
}
// CUDA_MEMCPY2D has hip_Memcpy2D's layout, but CUDA numbers memory types HOST 1, DEVICE 2, ARRAY 3,
// UNIFIED 4 where HIP uses 1, 2, 10, 11.
static bool memory_type(int cuda,hipMemoryType& hip) {
    switch(cuda) {
    case 1: hip=hipMemoryTypeHost; return true; case 2: hip=hipMemoryTypeDevice; return true;
    case 3: hip=hipMemoryTypeArray; return true; case 4: hip=hipMemoryTypeUnified; return true;
    default: return false;
    }
}
extern "C" int cuMemcpy2D_v2(const hip_Memcpy2D* cuda) {
    if(!cuda) return INVALID;
    hip_Memcpy2D m=*cuda;
    if(!memory_type(static_cast<int>(cuda->srcMemoryType),m.srcMemoryType)||
       !memory_type(static_cast<int>(cuda->dstMemoryType),m.dstMemoryType)) return UNSUPPORTED;
    if(m.dstMemoryType==hipMemoryTypeDevice||m.dstMemoryType==hipMemoryTypeUnified)
        track_write(reinterpret_cast<uint64_t>(m.dstDevice)+m.dstY*m.dstPitch+m.dstXInBytes,
                    m.Height?(m.Height-1)*m.dstPitch+m.WidthInBytes:0);
    bool src_packed=m.srcMemoryType==hipMemoryTypeArray&&original_format(m.srcArray)==80;
    bool dst_packed=m.dstMemoryType==hipMemoryTypeArray&&original_format(m.dstArray)==80;
    if(!src_packed&&!dst_packed) return error(hipMemcpyParam2D(&m));
    return copy_2d_packed(m,src_packed,dst_packed);
}

extern "C" int cuTexObjectCreate(hipTextureObject_t* object,const HIP_RESOURCE_DESC* resource,
    const HIP_TEXTURE_DESC* texture,const HIP_RESOURCE_VIEW_DESC* view) {
    return object&&resource&&texture?error(hipTexObjectCreate(object,resource,texture,view)):INVALID;
}
extern "C" int cuTexObjectDestroy(hipTextureObject_t object) { return error(hipTexObjectDestroy(object)); }
extern "C" int cuTexObjectGetResourceDesc(void*,hipTextureObject_t) { return UNSUPPORTED; } // as ZLUDA

// A surface handle is device memory: a copy of HIP's 48-byte image object (so the handle is also a
// valid texture object for texel fetches), a zeroed 32-byte sampler slot, then the CUDA format word
// and padding the bridge's output redirect writes into (bytes 84..95).
constexpr size_t IMAGE_OBJECT_BYTES=48, SAMPLER_OBJECT_BYTES=32,
    FORMAT_OFFSET=IMAGE_OBJECT_BYTES+SAMPLER_OBJECT_BYTES, SURFACE_HANDLE_BYTES=FORMAT_OFFSET+16;
struct Surface { hipSurfaceObject_t native; HIP_RESOURCE_DESC desc; };
static std::unordered_map<uint64_t,Surface> surfaces;
extern "C" int cuSurfObjectCreate(uint64_t* object,const HIP_RESOURCE_DESC* resource) {
    if(!object||!resource) return INVALID;
    uint32_t format=resource->resType==HIP_RESOURCE_TYPE_ARRAY?original_format(resource->res.array.hArray):0;
    hipSurfaceObject_t native{};
    int r=error(hipCreateSurfaceObject(&native,reinterpret_cast<const hipResourceDesc*>(resource))); if(r) return r;
    void* handle=nullptr;
    if((r=error(hipMalloc(&handle,SURFACE_HANDLE_BYTES)))) { auto ignored=hipDestroySurfaceObject(native); (void)ignored; return r; }
    uint32_t tail[(SURFACE_HANDLE_BYTES-IMAGE_OBJECT_BYTES)/4]{};
    tail[(FORMAT_OFFSET-IMAGE_OBJECT_BYTES)/4]=format;
    r=error(hipMemcpyDtoD(handle,reinterpret_cast<void*>(native),IMAGE_OBJECT_BYTES));
    if(!r) r=error(hipMemcpyHtoD(static_cast<uint8_t*>(handle)+IMAGE_OBJECT_BYTES,tail,sizeof(tail)));
    if(r) { auto a=hipFree(handle); auto b=hipDestroySurfaceObject(native); (void)a; (void)b; return r; }
    std::lock_guard<std::mutex> lock(resources_lock);
    surfaces[reinterpret_cast<uint64_t>(handle)]={native,*resource};
    *object=reinterpret_cast<uint64_t>(handle); return SUCCESS;
}
extern "C" int cuSurfObjectDestroy(uint64_t object) {
    Surface s;
    { std::lock_guard<std::mutex> lock(resources_lock);
      auto it=surfaces.find(object); if(it==surfaces.end()) return INVALID; s=it->second; surfaces.erase(it); }
    int native=error(hipDestroySurfaceObject(s.native));
    int freed=error(hipFree(reinterpret_cast<void*>(object)));
    return native?native:freed;
}
extern "C" int cuSurfObjectGetResourceDesc(HIP_RESOURCE_DESC* desc,uint64_t object) {
    if(!desc) return INVALID;
    std::lock_guard<std::mutex> lock(resources_lock);
    auto it=surfaces.find(object); if(it==surfaces.end()) return INVALID;
    *desc=it->second.desc; return SUCCESS;
}

// ---- module pack ----

struct PackEntry { uint64_t fatbin_hash; size_t fatbin_bytes; uint64_t ptx_hash; size_t ptx_bytes;
                   std::string sha256, file; bool empty=false; };
struct Pack { bool loaded=false; int status=NOT_FOUND; std::string dir, arch, switches; int fp8native=0;
              std::vector<PackEntry> entries; };
static std::mutex pack_lock;
static Pack pack;

// The code-generation switch set ZLUDA would key its cache with (module.rs zluda_build_version),
// minus settings that do not change generated code. Pack and process must agree.
static std::string process_switches() {
    static const char* ignored[]={"D4R_ZLUDA_NATIVE_DIR","D4R_ZLUDA_DIR","D4R_ZLUDA_DUMP_DIR",
        "D4R_ZLUDA_DUMP_PTX_DIR","D4R_ZLUDA_LIBCUDA","D4R_ZLUDA_WMMA_FP8_NATIVE","D4R_ZLUDA_CACHE_HOME"};
    std::vector<std::string> out;
    for(char** e=environ;*e;++e) {
        std::string kv=*e; if(kv.rfind("D4R_ZLUDA_",0)) continue;
        size_t eq=kv.find('='); if(eq==std::string::npos) continue;
        std::string key=kv.substr(0,eq), value=kv.substr(eq+1);
        if(std::any_of(std::begin(ignored),std::end(ignored),[&](const char* i) { return key==i; })) continue;
        size_t b=value.find_first_not_of(" \t\r\n"), e2=value.find_last_not_of(" \t\r\n");
        out.push_back(key+"="+(b==std::string::npos?"":value.substr(b,e2-b+1)));
    }
    std::sort(out.begin(),out.end());
    std::string joined; for(auto& s:out) { if(!joined.empty()) joined+=','; joined+=s; }
    return joined;
}
static int load_pack() {
    std::lock_guard<std::mutex> lock(pack_lock);
    if(pack.loaded) return pack.status;
    pack.loaded=true;
    const char* dir=std::getenv("D4R_MICROCUDA_PACK");
    if(!dir||!*dir) { note("%s","D4R_MICROCUDA_PACK is not set; no module can be loaded"); return pack.status=NOT_FOUND; }
    pack.dir=dir;
    FILE* f=std::fopen((pack.dir+"/microcuda-pack.txt").c_str(),"r");
    if(!f) {
        // a directory of packs, one per GPU target (<dir>/gfx1101/microcuda-pack.txt, ...): take this GPU's
        hipDeviceProp_t props{}; int device=0;
        if(hipGetDevice(&device)==hipSuccess&&hipGetDeviceProperties(&props,device)==hipSuccess) {
            std::string arch=props.gcnArchName; arch=arch.substr(0,arch.find(':'));
            pack.dir=std::string(dir)+"/"+arch;
            f=std::fopen((pack.dir+"/microcuda-pack.txt").c_str(),"r");
        }
    }
    if(!f) { note("cannot read %s/microcuda-pack.txt",dir); return pack.status=NOT_FOUND; }
    char line[1024];
    while(std::fgets(line,sizeof(line),f)) {
        std::string s=line; while(!s.empty()&&(s.back()=='\n'||s.back()=='\r')) s.pop_back();
        if(s.empty()||s[0]=='#') continue;
        if(s.rfind("arch ",0)==0) { pack.arch=s.substr(5); continue; }
        if(s.rfind("switches",0)==0) { pack.switches=s.size()>9?s.substr(9):""; continue; }
        if(s.rfind("fp8native ",0)==0) { pack.fp8native=std::atoi(s.c_str()+10); continue; }
        PackEntry e; char sha[65], file[256]; unsigned long long fh,ph; size_t fb,pb;
        if(std::sscanf(line,"empty %llx %zu",&fh,&fb)==2) {
            e.fatbin_hash=fh; e.fatbin_bytes=fb; e.ptx_hash=0; e.ptx_bytes=SIZE_MAX; e.empty=true;
            pack.entries.push_back(e); continue;
        }
        if(std::sscanf(line,"module %llx %zu %llx %zu %64s %255s",&fh,&fb,&ph,&pb,sha,file)!=6) continue;
        e.fatbin_hash=fh; e.fatbin_bytes=fb; e.ptx_hash=ph; e.ptx_bytes=pb; e.sha256=sha; e.file=file;
        pack.entries.push_back(e);
    }
    std::fclose(f);
    std::string current=process_switches();
    if(current!=pack.switches) {
        note("pack was compiled with ZLUDA switches '%s', this process has '%s'; refusing the pack",
             pack.switches.c_str(),current.c_str());
        return pack.status=INVALID_IMAGE;
    }
    return pack.status=pack.entries.empty()?NOT_FOUND:SUCCESS;
}
// RDNA4 objects differ with native FP8 WMMA (ZLUDA keys that by effect: gfx12 with WMMA and NativeFp8 on).
static bool fp8_matches(const std::string& arch) {
    auto on=[](const char* k) { const char* v=std::getenv(k); return v&&std::strcmp(v,"1")==0; };
    bool expected=arch.rfind("gfx12",0)==0&&on("D4R_ZLUDA_WMMA")&&on("D4R_ZLUDA_WMMA_FP8_NATIVE");
    return expected==(pack.fp8native!=0);
}
static bool read_file(const std::string& path,std::vector<unsigned char>& bytes) {
    FILE* file=std::fopen(path.c_str(),"rb"); if(!file) return false;
    unsigned char buffer[65536]; size_t count;
    while((count=std::fread(buffer,1,sizeof(buffer),file))) bytes.insert(bytes.end(),buffer,buffer+count);
    bool failed=std::ferror(file); std::fclose(file); return !failed;
}
static bool digest_matches(const std::vector<unsigned char>& data,const std::string& expected) {
    unsigned char digest[SHA256_DIGEST_LENGTH]; SHA256(data.data(),data.size(),digest);
    char hex[65]; for(size_t i=0;i<32;++i) std::sprintf(hex+2*i,"%02x",digest[i]);
    return expected==hex;
}

// ---- modules and functions (ZLUDA module.rs / function.rs) ----

struct Post { hipFunction_t function; uint32_t gx, gy, bx, by, bz; };
struct Function {
    hipFunction_t main=nullptr, prep=nullptr;
    uint32_t prep_blocks=0, block_z=0, grid_x=0, prep_slots=0;
    bool has_prep_key=false; uint32_t prep_key=0;
    std::atomic<uint64_t> prep_last{0};
    std::mutex seen_lock; std::vector<uint64_t> seen;
    std::vector<Post> post;
    bool native=false;
    std::atomic<uint64_t> prep_runs{0}, prep_skips{0};
    // identity policy (d4r_prep_identity_at): prepared images per weights pointer, valid while the pointer's
    // allocation generation and write epoch are unchanged and no allocation has been freed since the slot
    // table was last reset. Only on one stream; another stream turns the policy off for this function.
    uint32_t identity_at=0;
    hipDeviceptr_t prep_keys=nullptr; size_t prep_keys_bytes=0;
    std::mutex identity_lock;
    struct Prepared { uint64_t key, generation, epoch; };
    std::vector<Prepared> prepared;
    uint64_t prepared_free_generation=0;
    hipStream_t identity_stream=nullptr; bool identity_stream_set=false, identity_off=false;
};
struct NativeImage { hipModule_t module; std::vector<unsigned char> bytes; };
struct Module {
    hipModule_t hip=nullptr;
    std::vector<unsigned char> image; // retained until HIP module unload
    std::mutex lock;
    std::map<std::string,std::unique_ptr<Function>> functions;
    std::vector<std::unique_ptr<NativeImage>> natives;
};
static std::mutex modules_lock;
static std::vector<Module*> modules;
static bool registered(Module* m) { return std::find(modules.begin(),modules.end(),m)!=modules.end(); }

static bool global_u32(hipModule_t m,const char* name,uint32_t& value) {
    hipDeviceptr_t ptr=nullptr; size_t bytes=0;
    return hipModuleGetGlobal(&ptr,&bytes,m,name)==hipSuccess&&bytes==4&&hipMemcpyDtoH(&value,ptr,4)==hipSuccess;
}
// d4r native replacement served by the bridge as DIR/NAME.hsaco (only for PTX it verified); same
// metadata contract as ZLUDA's native_override. Returns false to use the pack's compile of NAME.
static bool native_override(Module* m,const std::string& name,Function& f) {
    const char* dir=std::getenv("D4R_MICROCUDA_NATIVE_DIR"); if(!dir||!*dir) return false;
    auto image=std::make_unique<NativeImage>();
    std::string path=std::string(dir)+"/"+name+".hsaco";
    if(!read_file(path,image->bytes)||image->bytes.empty()) return false;
    if(hipModuleLoadData(&image->module,image->bytes.data())!=hipSuccess) { note("failed to load %s",path.c_str()); return false; }
    hipModule_t nm=image->module;
    auto reject=[&](const char* why) { note("%s: %s; using the pack's kernel",path.c_str(),why);
        auto ignored=hipModuleUnload(nm); (void)ignored; return false; };
    if(hipModuleGetFunction(&f.main,nm,name.c_str())!=hipSuccess) return reject("no kernel of that name");
    if(hipModuleGetFunction(&f.prep,nm,(name+"_prep").c_str())==hipSuccess) {
        if(!global_u32(nm,"d4r_prep_blocks",f.prep_blocks)||!f.prep_blocks) return reject("prep kernel without d4r_prep_blocks");
    } else f.prep=nullptr;
    global_u32(nm,"d4r_block_z",f.block_z);
    global_u32(nm,"d4r_grid_x",f.grid_x);
    // d4r_prep_key_at holds offset + 1; the older d4r_prep_key_offset uses 0 for "no key"
    uint32_t at=0, offset=0;
    if(global_u32(nm,"d4r_prep_key_at",at)&&at>0) { f.has_prep_key=true; f.prep_key=at-1; }
    else if(global_u32(nm,"d4r_prep_key_offset",offset)&&offset) { f.has_prep_key=true; f.prep_key=offset; }
    global_u32(nm,"d4r_prep_key_slots",f.prep_slots);
    if(f.prep&&global_u32(nm,"d4r_prep_identity_at",f.identity_at)&&f.identity_at) {
        // the slot table the prep claims into (keys only ever go from 0 to a pointer until reset)
        if(hipModuleGetGlobal(&f.prep_keys,&f.prep_keys_bytes,nm,"g_prep_keys")!=hipSuccess||
           f.prep_keys_bytes!=(f.prep_slots+1)*sizeof(uint64_t)||!f.prep_slots) {
            note("%s: d4r_prep_identity_at without a matching g_prep_keys slot table; preparing every launch",path.c_str());
            f.identity_at=0;
        }
    }
    for(unsigned k=1;k<=8;++k) {
        Post p{};
        if(hipModuleGetFunction(&p.function,nm,(name+"_post"+std::to_string(k)).c_str())!=hipSuccess) break;
        auto field=[&](const char* s) { uint32_t v=0; global_u32(nm,("d4r_post"+std::to_string(k)+"_"+s).c_str(),v); return v; };
        p.gx=field("grid_x"); p.gy=field("grid_y");
        p.bx=field("block_x"); p.by=field("block_y"); p.bz=field("block_z");
        if(!p.bx) p.bx=32;
        if(!p.by) p.by=1;
        if(!p.bz) p.bz=1;
        f.post.push_back(p);
    }
    f.native=true;
    m->natives.push_back(std::move(image));
    return true;
}

static int load_module(Module** out,const void* image) {
    if(!out||!image) return INVALID;
    *out=nullptr;
    if(!initialized.load(std::memory_order_acquire)) return NOT_INIT;
    if(int r=load_pack()) return r;
    const auto* p=static_cast<const unsigned char*>(image);
    uint32_t magic; std::memcpy(&magic,p,4);
    const PackEntry* entry=nullptr;
    if(magic==0xba55ed50U) {
        uint16_t version,header; uint64_t payload;
        std::memcpy(&version,p+4,2); std::memcpy(&header,p+6,2); std::memcpy(&payload,p+8,8);
        if(version!=1||header!=16||payload>256*1024*1024) return INVALID_IMAGE;
        size_t n=header+payload; uint64_t h=0; bool hashed=false;
        for(const auto& e:pack.entries) if(n==e.fatbin_bytes) {
            if(!hashed) { h=fnv(p,n); hashed=true; }
            if(h==e.fatbin_hash) { entry=&e; break; }
        }
    } else {
        // PTX text: identity of the text without trailing NULs, as the bridge's native manifest
        size_t n=strnlen(reinterpret_cast<const char*>(p),256*1024*1024); uint64_t h=fnv(p,n);
        for(const auto& e:pack.entries) if(n==e.ptx_bytes&&h==e.ptx_hash) { entry=&e; break; }
    }
    if(!entry) return INVALID_IMAGE;
    hipDeviceProp_t props{}; int device;
    if(int r=error(hipGetDevice(&device))) return r;
    if(int r=error(hipGetDeviceProperties(&props,device))) return r;
    std::string arch=props.gcnArchName; arch=arch.substr(0,arch.find(':'));
    if(arch!=pack.arch) return 209;
    if(!fp8_matches(arch)) { note("%s pack was built with NativeFp8=%s; the process setting differs",arch.c_str(),
                                  pack.fp8native?"1":"0"); return INVALID_IMAGE; }
    auto m=std::make_unique<Module>();
    if(entry->empty) { // fatbin without PTX: a module without functions, as ZLUDA loads it
        std::lock_guard<std::mutex> lock(modules_lock); modules.push_back(m.get()); *out=m.release(); return SUCCESS;
    }
    if(!read_file(pack.dir+"/modules/"+entry->file,m->image)||!digest_matches(m->image,entry->sha256)) {
        note("%s/modules/%s is missing or fails its digest",pack.dir.c_str(),entry->file.c_str());
        return INVALID_IMAGE;
    }
    if(int r=error(hipModuleLoadData(&m->hip,m->image.data()))) return r;
    std::lock_guard<std::mutex> lock(modules_lock); modules.push_back(m.get()); *out=m.release(); return SUCCESS;
}
extern "C" int cuModuleLoadData(Module** out,const void* image) { return load_module(out,image); }
extern "C" int cuModuleLoadDataEx(Module** m,const void* image,unsigned,int*,void**) {
    return load_module(m,image); // JIT options are irrelevant to precompiled objects (ZLUDA ignores them too)
}
extern "C" int cuModuleLoad(Module** m,const char* path) {
    if(!m||!path) return INVALID;
    std::vector<unsigned char> bytes; if(!read_file(path,bytes)) return INVALID;
    bytes.push_back(0);
    return load_module(m,bytes.data());
}
extern "C" int cuModuleGetFunction(Function** out,Module* m,const char* name) {
    if(!out||!name) return INVALID;
    *out=nullptr;
    { std::lock_guard<std::mutex> lock(modules_lock); if(!registered(m)) return INVALID_HANDLE; }
    if(!m->hip) return NOT_FOUND; // empty module
    std::lock_guard<std::mutex> lock(m->lock);
    auto& slot=m->functions[name];
    if(!slot) {
        auto f=std::make_unique<Function>();
        if(!native_override(m,name,*f)) {
            f=std::make_unique<Function>();
            if(int r=error(hipModuleGetFunction(&f->main,m->hip,name))) { m->functions.erase(name); return r; }
        }
        slot=std::move(f);
    }
    *out=slot.get(); return SUCCESS;
}
extern "C" int cuModuleUnload(Module* m) {
    { std::lock_guard<std::mutex> lock(modules_lock);
      if(!registered(m)) return INVALID_HANDLE;
      modules.erase(std::find(modules.begin(),modules.end(),m)); }
    static const bool stats=std::getenv("D4R_MICROCUDA_STATS")!=nullptr;
    if(stats) for(auto& [name,f]:m->functions) if(f->prep)
        std::fprintf(stderr,"[d4r-microcuda] %s: prep runs %llu, skips %llu%s\n",name.c_str(),
                     (unsigned long long)f->prep_runs.load(),(unsigned long long)f->prep_skips.load(),
                     f->identity_at?(f->identity_off?" (identity reuse, turned off: several streams)":" (identity reuse)"):"");
    int r=m->hip?error(hipModuleUnload(m->hip)):SUCCESS;
    for(auto& n:m->natives) { int nr=error(hipModuleUnload(n->module)); if(!r) r=nr; }
    delete m; return r;
}
extern "C" int d4rMicrocudaGetDispatchStats(Function* f,uint64_t* prep,uint64_t* skips) {
    if(!f||!prep||!skips) return INVALID;
    *prep=f->prep_runs.load(std::memory_order_relaxed);
    *skips=f->prep_skips.load(std::memory_order_relaxed); return SUCCESS;
}

// Prep under the identity policy; launch(h, grid..., block..., shared) submits on the caller's stream.
template <class Launch>
static int identity_prep(Function* f,const unsigned char* block,hipStream_t stream,Launch& launch) {
    std::lock_guard<std::mutex> lock(f->identity_lock);
    auto prep=[&] {
        int r=launch(f->prep,f->prep_blocks,1,1,128,1,1,0);
        if(!r) f->prep_runs.fetch_add(1,std::memory_order_relaxed);
        return r;
    };
    if(f->identity_off||!block) return prep();
    if(!f->identity_stream_set) { f->identity_stream=stream; f->identity_stream_set=true; }
    else if(f->identity_stream!=stream) {
        // a second stream: slot resets could race with the other stream's reads. Prepare every launch from now on.
        f->identity_off=true; f->prepared.clear(); return prep();
    }
    uint64_t key=0, generation=0, epoch=0;
    std::memcpy(&key,block+f->identity_at-1,sizeof(key));
    const uint64_t frees=free_generation.load(std::memory_order_acquire);
    if(frees!=f->prepared_free_generation) {
        // an allocation was freed (feature release/recreation): addresses may be recycled. Forget every image
        // and clear the device slot table on this stream, so no key can match an old image.
        if(int r=error(hipMemsetAsync(f->prep_keys,0,f->prep_keys_bytes,stream))) return r;
        f->prepared.clear(); f->prepared_free_generation=frees;
    }
    if(!key||!identity_of(key,generation,epoch)) return prep(); // untracked weights: no reuse
    for(auto& p:f->prepared) if(p.key==key) {
        if(p.generation==generation&&p.epoch==epoch) { f->prep_skips.fetch_add(1,std::memory_order_relaxed); return SUCCESS; }
        p.generation=generation; p.epoch=epoch; return prep(); // same slot (claimed by key), new contents
    }
    // a new key claims the next free slot; beyond the slots it shares the overflow slot and prepares every launch
    if(f->prepared.size()<f->prep_slots) f->prepared.push_back({key,generation,epoch});
    return prep();
}

// CUDA's packed-argument tokens are POINTER=1, SIZE=2, END=0; HIP's END is 3.
extern "C" int cuLaunchKernel(Function* f,unsigned gx,unsigned gy,unsigned gz,
    unsigned bx,unsigned by,unsigned bz,unsigned shared,hipStream_t stream,void** params,void** extra) {
    if(!f) return INVALID_HANDLE;
    void* translated[8]{}; void** hip_extra=nullptr; const unsigned char* block=nullptr;
    if(extra) {
        size_t i=0, o=0; bool ended=false;
        while(i<8&&o<8) {
            uintptr_t token=reinterpret_cast<uintptr_t>(extra[i]);
            if(token==0) { translated[o]=reinterpret_cast<void*>(uintptr_t{3}); ended=true; break; }
            if(token!=1&&token!=2) return UNSUPPORTED;
            if(i+1>=8||o+1>=8) return INVALID;
            if(token==1) block=static_cast<const unsigned char*>(extra[i+1]);
            translated[o]=extra[i]; translated[o+1]=extra[i+1]; i+=2; o+=2;
        }
        if(!ended) return INVALID;
        hip_extra=translated;
    } else if(params) block=static_cast<const unsigned char*>(params[0]); // single-struct kernels
    auto launch=[&](hipFunction_t h,unsigned x,unsigned y,unsigned z,unsigned tx,unsigned ty,unsigned tz,unsigned sm) {
        return error(hipModuleLaunchKernel(h,x,y,z,tx,ty,tz,sm,stream,params,hip_extra));
    };
    if(f->prep&&f->identity_at) {
        if(int r=identity_prep(f,block,stream,launch)) return r;
    } else if(f->prep) {
        uint64_t key=0;
        if(f->has_prep_key&&block) std::memcpy(&key,block+f->prep_key,sizeof(key));
        bool run=true;
        if(key) {
            if(!f->prep_slots) run=f->prep_last.load(std::memory_order_relaxed)!=key;
            else {
                // one prepared image per key while slots last; keys beyond them prepare every launch
                std::lock_guard<std::mutex> lock(f->seen_lock);
                if(std::find(f->seen.begin(),f->seen.end(),key)!=f->seen.end()) run=false;
                else if(f->seen.size()<f->prep_slots) f->seen.push_back(key);
            }
        }
        if(run) {
            if(int r=launch(f->prep,f->prep_blocks,1,1,128,1,1,0)) return r;
            f->prep_last.store(key,std::memory_order_relaxed);
            f->prep_runs.fetch_add(1,std::memory_order_relaxed);
        } else f->prep_skips.fetch_add(1,std::memory_order_relaxed);
    }
    if(f->block_z) bz=f->block_z;
    if(f->grid_x) { gx=f->grid_x; gy=1; gz=1; }
    if(int r=launch(f->main,gx,gy,gz,bx,by,bz,shared)) return r;
    for(const auto& p:f->post)
        if(int r=launch(p.function,p.gx?p.gx:gx,p.gy?p.gy:gy,(p.gx||p.gy)?1:gz,p.bx,p.by,p.bz,0)) return r;
    return SUCCESS;
}

extern "C" int cuGetProcAddress_v2(const char* name,void** out,int version,uint64_t flags,int* status) {
    if(!out||!name) return INVALID;
    *out=nullptr; if(status) *status=1;
    // No per-thread default-stream ABI or unversioned 32-bit size_t APIs.
    if(flags>1||version<3020||version>12080) return UNSUPPORTED;
    struct Symbol { const char* name; void* address; };
#define SYMBOL(n) {#n,reinterpret_cast<void*>(&n)}
    static const Symbol symbols[]={
        SYMBOL(cuInit),SYMBOL(cuDeviceGetCount),SYMBOL(cuDeviceGet),
        SYMBOL(cuCtxCreate_v2),SYMBOL(cuCtxDestroy_v2),SYMBOL(cuCtxPushCurrent_v2),
        SYMBOL(cuCtxPopCurrent_v2),SYMBOL(cuCtxSetCurrent),SYMBOL(cuCtxGetCurrent),
        SYMBOL(cuCtxGetDevice),SYMBOL(cuCtxSynchronize),SYMBOL(cuDriverGetVersion),
        SYMBOL(cuDeviceGetAttribute),SYMBOL(cuDeviceGetName),SYMBOL(cuDeviceGetUuid),
        SYMBOL(cuDeviceGetLuid),SYMBOL(cuDeviceTotalMem_v2),SYMBOL(cuDeviceComputeCapability),
        SYMBOL(cuMemAlloc_v2),SYMBOL(cuMemFree_v2),SYMBOL(cuMemAllocHost_v2),
        SYMBOL(cuMemHostAlloc),SYMBOL(cuMemFreeHost),SYMBOL(cuMemGetInfo_v2),
        SYMBOL(cuMemcpyHtoD_v2),SYMBOL(cuMemcpyDtoH_v2),SYMBOL(cuMemcpyDtoD_v2),
        SYMBOL(cuMemcpyHtoDAsync_v2),SYMBOL(cuMemcpyDtoHAsync_v2),SYMBOL(cuMemcpyDtoDAsync_v2),
        SYMBOL(cuMemsetD8_v2),SYMBOL(cuMemcpy2D_v2),SYMBOL(cuEventCreate),SYMBOL(cuEventDestroy_v2),
        SYMBOL(cuEventRecord),SYMBOL(cuEventQuery),SYMBOL(cuEventSynchronize),SYMBOL(cuEventElapsedTime),
        SYMBOL(cuStreamCreate),SYMBOL(cuStreamDestroy_v2),SYMBOL(cuStreamSynchronize),
        SYMBOL(cuStreamQuery),SYMBOL(cuStreamWaitEvent),SYMBOL(cuGetErrorString),
        SYMBOL(cuArrayCreate_v2),SYMBOL(cuArray3DCreate_v2),SYMBOL(cuArrayDestroy),
        SYMBOL(cuArrayGetDescriptor_v2),SYMBOL(cuMipmappedArrayDestroy),SYMBOL(cuDestroyExternalMemory),
        SYMBOL(cuTexObjectCreate),SYMBOL(cuTexObjectDestroy),SYMBOL(cuTexObjectGetResourceDesc),
        SYMBOL(cuSurfObjectCreate),SYMBOL(cuSurfObjectDestroy),SYMBOL(cuSurfObjectGetResourceDesc),
        SYMBOL(cuModuleLoad),SYMBOL(cuModuleLoadData),SYMBOL(cuModuleLoadDataEx),SYMBOL(cuModuleGetFunction),
        SYMBOL(cuModuleUnload),SYMBOL(cuLaunchKernel)
    };
#undef SYMBOL
    for(const auto& s:symbols) {
        size_t n=std::strlen(s.name);
        bool match=std::strcmp(name,s.name)==0;
        // GetProcAddress receives CUDA base names and selects the modern ABI.
        if(n>3&&std::strcmp(s.name+n-3,"_v2")==0)
            match=match||(std::strlen(name)==n-3&&std::strncmp(name,s.name,n-3)==0);
        if(match) { *out=s.address; if(status) *status=0; return SUCCESS; }
    }
    return NOT_FOUND;
}
extern "C" int cuGetProcAddress(const char* name,void** out,int version,uint64_t flags) {
    return cuGetProcAddress_v2(name,out,version,flags,nullptr);
}
