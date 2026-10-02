#pragma once
#include "cuda_api.h"

namespace d4r::cuda {
// Public CUDA Driver API layouts, kept independent of NVIDIA SDK headers.
using Array = void*;
using Texture = uint64_t;
struct ArrayDescriptor { size_t width, height; uint32_t format, channels; };
struct Array3DDescriptor {
    size_t width, height, depth;
    uint32_t format, channels, flags;
};
struct Copy2D {
    size_t srcX, srcY;
    uint32_t srcType, srcAlignment;
    const void* srcHost;
    CUdeviceptr srcDevice;
    Array srcArray;
    size_t srcPitch, dstX, dstY;
    uint32_t dstType, dstAlignment;
    void* dstHost;
    CUdeviceptr dstDevice;
    Array dstArray;
    size_t dstPitch, widthBytes, height;
};
struct ResourceDescriptor {
    uint32_t type, alignment;
    union { Array array; int reserved[32]; } resource;
    uint32_t flags, reserved;
};
struct TextureDescriptor {
    uint32_t addressMode[3], filterMode, flags, maxAnisotropy, mipmapFilterMode;
    float mipmapBias, minMipmapClamp, maxMipmapClamp, borderColor[4];
    int reserved[12];
};
static_assert(sizeof(Array3DDescriptor) == 40);
static_assert(sizeof(ArrayDescriptor) == 24);
static_assert(sizeof(Copy2D) == 128);
static_assert(sizeof(ResourceDescriptor) == 144);
static_assert(sizeof(TextureDescriptor) == 104);

struct ImageApi {
    CudaApi& cuda;
#define D4R_IMAGE(name, ...) using name##_fn = int(WINAPI*)(__VA_ARGS__); name##_fn name = cuda.library.symbol<name##_fn>(#name)
    D4R_IMAGE(cuArray3DCreate_v2, Array*, const Array3DDescriptor*);
    D4R_IMAGE(cuArrayGetDescriptor_v2, ArrayDescriptor*, Array);
    D4R_IMAGE(cuArray3DGetDescriptor_v2, Array3DDescriptor*, Array);
    D4R_IMAGE(cuArrayDestroy, Array);
    D4R_IMAGE(cuMemcpy2D_v2, const Copy2D*);
    D4R_IMAGE(cuTexObjectCreate, Texture*, const ResourceDescriptor*, const TextureDescriptor*, const void*);
    D4R_IMAGE(cuTexObjectDestroy, Texture);
    D4R_IMAGE(cuTexObjectGetResourceDesc, ResourceDescriptor*, Texture);
    D4R_IMAGE(cuSurfObjectCreate, Texture*, const ResourceDescriptor*);
    D4R_IMAGE(cuSurfObjectDestroy, Texture);
    D4R_IMAGE(cuSurfObjectGetResourceDesc, ResourceDescriptor*, Texture);
#undef D4R_IMAGE
    explicit ImageApi(CudaApi& api) : cuda(api) {}
};

class Image {
    ImageApi& api_;
    void release() noexcept {
        if (object) {
            const int result = surface_ ? api_.cuSurfObjectDestroy(object) : api_.cuTexObjectDestroy(object);
            if (result) std::fprintf(stderr, "IMAGE_DESTROY object result=%d\n", result);
            object = 0;
        }
        if (array_) {
            const int result = api_.cuArrayDestroy(array_);
            if (result) std::fprintf(stderr, "IMAGE_DESTROY array result=%d\n", result);
            array_ = nullptr;
        }
    }
    Array array_ = nullptr;
    bool surface_;
    size_t rowBytes_, height_;
public:
    Texture object = 0;
    Image(ImageApi& api, unsigned width, unsigned height, uint32_t format,
          unsigned channels, bool surface, unsigned filter = 0)
        : api_(api), surface_(surface), rowBytes_(size_t(width) * channels * (format == 16 ? 2 : 4)), height_(height) {
        try {
            const Array3DDescriptor descriptor{width, height, 0, format, channels, surface ? 2u : 0u};
            api_.cuda.check(api_.cuArray3DCreate_v2(&array_, &descriptor), "cuArray3DCreate_v2");
            ResourceDescriptor resource{};
            resource.resource.array = array_;
            if (surface) api_.cuda.check(api_.cuSurfObjectCreate(&object, &resource), "cuSurfObjectCreate");
            else {
                TextureDescriptor texture{};
                texture.addressMode[0] = texture.addressMode[1] = 1; // clamp
                texture.filterMode = filter;
                texture.flags = 2; // normalized coordinates
                api_.cuda.check(api_.cuTexObjectCreate(&object, &resource, &texture, nullptr), "cuTexObjectCreate");
            }
            ResourceDescriptor roundtrip{};
            api_.cuda.check(surface ? api_.cuSurfObjectGetResourceDesc(&roundtrip, object) :
                api_.cuTexObjectGetResourceDesc(&roundtrip, object), "CUDA object resource descriptor roundtrip");
            if (roundtrip.type != 0 || roundtrip.resource.array != array_ || roundtrip.flags != 0)
                throw std::runtime_error("CUDA object descriptor does not match its array");
            ArrayDescriptor twoD{};
            Array3DDescriptor threeD{};
            api_.cuda.check(api_.cuArrayGetDescriptor_v2(&twoD, array_), "cuArrayGetDescriptor_v2");
            api_.cuda.check(api_.cuArray3DGetDescriptor_v2(&threeD, array_), "cuArray3DGetDescriptor_v2");
            if (twoD.width != width || twoD.height != height || twoD.format != format || twoD.channels != channels ||
                threeD.width != width || threeD.height != height || threeD.depth != 0 || threeD.format != format ||
                threeD.channels != channels || threeD.flags != descriptor.flags)
                throw std::runtime_error("CUDA array descriptor geometry or format mismatch");
        } catch (...) { release(); throw; }
    }
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    ~Image() { release(); }
    void verify_wrong_object_kind() {
        ResourceDescriptor wrong{};
        const int result = surface_ ? api_.cuTexObjectGetResourceDesc(&wrong, object) :
            api_.cuSurfObjectGetResourceDesc(&wrong, object);
        if (!result) throw std::runtime_error("Wrong CUDA object kind was accepted as a valid resource");
        std::printf("CUDA wrong-kind descriptor safely rejected result=%d\n", result);
    }
    void upload(const void* source) {
        Copy2D copy{};
        copy.srcType = 1; copy.srcHost = source; copy.srcPitch = rowBytes_;
        copy.dstType = 3; copy.dstArray = array_;
        copy.widthBytes = rowBytes_; copy.height = height_;
        api_.cuda.check(api_.cuMemcpy2D_v2(&copy), "cuMemcpy2D_v2(host to diagnostic array)");
    }
    void download(void* destination) {
        Copy2D copy{};
        copy.srcType = 3; copy.srcArray = array_;
        copy.dstType = 1; copy.dstHost = destination; copy.dstPitch = rowBytes_;
        copy.widthBytes = rowBytes_; copy.height = height_;
        api_.cuda.check(api_.cuMemcpy2D_v2(&copy), "cuMemcpy2D_v2(diagnostic array to host)");
    }
    void upload_device(CUdeviceptr source, size_t pitch) {
        if (pitch < rowBytes_) throw std::runtime_error("Device input pitch is smaller than its row");
        Copy2D copy{};
        copy.srcType = 2; copy.srcDevice = source; copy.srcPitch = pitch;
        copy.dstType = 3; copy.dstArray = array_;
        copy.widthBytes = rowBytes_; copy.height = height_;
        api_.cuda.check(api_.cuMemcpy2D_v2(&copy), "cuMemcpy2D_v2(external VRAM to array)");
    }
    void download_device(CUdeviceptr destination, size_t pitch) {
        if (pitch < rowBytes_) throw std::runtime_error("Device output pitch is smaller than its row");
        Copy2D copy{};
        copy.srcType = 3; copy.srcArray = array_;
        copy.dstType = 2; copy.dstDevice = destination; copy.dstPitch = pitch;
        copy.widthBytes = rowBytes_; copy.height = height_;
        api_.cuda.check(api_.cuMemcpy2D_v2(&copy), "cuMemcpy2D_v2(array to external VRAM)");
    }
};
}
