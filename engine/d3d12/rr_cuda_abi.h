// CUDA driver structures used by the independent RR reference runner.
#pragma once
#include <cstddef>
#include <cstdint>
using CudaDevicePtr = unsigned long long;
using CudaArray = void *;
using CudaObject = unsigned long long;
using CudaEvent = void *;

struct CudaArrayDescriptor
{
    size_t Width;
    size_t Height;
    uint32_t Format;
    uint32_t NumChannels;
};

struct CudaArray3DDescriptor
{
    size_t Width;
    size_t Height;
    size_t Depth;
    uint32_t Format;
    uint32_t NumChannels;
    uint32_t Flags;
};

struct CudaMemcpy2D
{
    size_t srcXInBytes;
    size_t srcY;
    uint32_t srcMemoryType;
    uint32_t srcAlignment;
    const void *srcHost;
    CudaDevicePtr srcDevice;
    CudaArray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    uint32_t dstMemoryType;
    uint32_t dstAlignment;
    void *dstHost;
    CudaDevicePtr dstDevice;
    CudaArray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};
static_assert(sizeof(CudaMemcpy2D) == 128);

struct CudaResourceDesc
{
    uint32_t resType;
    uint32_t alignment;
    union {
        struct
        {
            CudaArray hArray;
        } array;
        int reserved[32];
    } res;
    uint32_t flags;
    uint32_t reserved;
};
static_assert(sizeof(CudaResourceDesc) == 144);

struct CudaTextureDesc
{
    uint32_t addressMode[3];
    uint32_t filterMode;
    uint32_t flags;
    uint32_t maxAnisotropy;
    uint32_t mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int32_t reserved[12];
};
static_assert(sizeof(CudaTextureDesc) == 104);
