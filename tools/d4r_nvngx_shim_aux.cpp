// d4r_nvngx_shim_aux.cpp: the surfaces the denoiser needs beyond Super Sampling's four, and the CUDA-side
// resources that carry them: each auxiliary surface's own format and channel count, the texture objects
// minted over them, the staging buffers, and d4r_hip_net.dll's bridge for the native HIP network.

// the command-list stubs live in d4r_nvngx_shim.cpp, which assembles them once
#define D4R_D3D12_SPLIT_DECL 1
#include "d4r_nvngx_shim_internal.h"

// The CUDA array a denoiser input uses, plus the texel size the caller's own resource
// occupies. The two differ for every converted format (B8G8R8A8_UNORM is four bytes to
// D3D12 and sixteen to the float array), and a capture records the caller's bytes, so a
// replay runs them through the same conversion again.
//
// Only formats that are byte-for-byte correct go down the verbatim path, and that is a
// shorter list than it looks. A CUDA array stores element_size * channels bytes per texel
// and only ever has 1, 2 or 4 channels, so a packed format has no verbatim layout at all:
// R10G10B10A2_UNORM is 4 bytes per texel, but a 4-channel uint array is 16, and copying
// "4 channels' worth" would read four times past the end of every row. R11G11B10_FLOAT has
// three channels, which no CUDA array has.
//
// Nor is memory layout the only question. Sampling an integer-typed CUDA array normalizes it,
// which is right for a UNORM or SNORM surface and wrong for a UINT one, where a D3D12 shader
// reads the raw integer. And B8G8R8A8_UNORM has RGBA bytes in BGRA order, so a verbatim copy
// would hand the denoiser a texture whose red and blue channels are swapped.
//
// Everything that is not in this table is converted to a float array instead, which is
// correct for all of those cases at the cost of a per-row unpack.

bool aux_array_layout(DXGI_FORMAT format, uint32_t& arrayFormat, uint32_t& channels, AuxLayout& how,
                             uint32_t& texelBytes)
{
    struct Entry
    {
        DXGI_FORMAT format;
        uint32_t arrayFormat;
        uint32_t channels;
    };
    // UNORM and SNORM map onto signed/unsigned integer arrays of the same width: sampling
    // normalizes them, which is exactly how a D3D12 shader reads the resource. Their texel
    // size is the array's own, so it is derived rather than listed twice.
    static const Entry entries[] = {
        {DXGI_FORMAT_R8_UNORM, CUDA_FORMAT_UNSIGNED_INT8, 1},
        {DXGI_FORMAT_R8_SNORM, CUDA_FORMAT_SIGNED_INT8, 1},
        {DXGI_FORMAT_R8G8_UNORM, CUDA_FORMAT_UNSIGNED_INT8, 2},
        {DXGI_FORMAT_R8G8_SNORM, CUDA_FORMAT_SIGNED_INT8, 2},
        {DXGI_FORMAT_R8G8B8A8_UNORM, CUDA_FORMAT_UNSIGNED_INT8, 4},
        {DXGI_FORMAT_R8G8B8A8_SNORM, CUDA_FORMAT_SIGNED_INT8, 4},
        {DXGI_FORMAT_R16_UNORM, CUDA_FORMAT_UNSIGNED_INT16, 1},
        {DXGI_FORMAT_R16_SNORM, CUDA_FORMAT_SIGNED_INT16, 1},
        {DXGI_FORMAT_R16_FLOAT, CUDA_FORMAT_HALF, 1},
        {DXGI_FORMAT_R16G16_UNORM, CUDA_FORMAT_UNSIGNED_INT16, 2},
        {DXGI_FORMAT_R16G16_SNORM, CUDA_FORMAT_SIGNED_INT16, 2},
        {DXGI_FORMAT_R16G16_FLOAT, CUDA_FORMAT_HALF, 2},
        {DXGI_FORMAT_R16G16B16A16_UNORM, CUDA_FORMAT_UNSIGNED_INT16, 4},
        {DXGI_FORMAT_R16G16B16A16_SNORM, CUDA_FORMAT_SIGNED_INT16, 4},
        {DXGI_FORMAT_R16G16B16A16_FLOAT, CUDA_FORMAT_HALF, 4},
        {DXGI_FORMAT_R32_FLOAT, CUDA_FORMAT_FLOAT, 1},
        {DXGI_FORMAT_R32G32_FLOAT, CUDA_FORMAT_FLOAT, 2},
        {DXGI_FORMAT_R32G32B32A32_FLOAT, CUDA_FORMAT_FLOAT, 4},
    };
    for (const Entry& entry : entries)
        if (entry.format == format)
        {
            arrayFormat = entry.arrayFormat;
            channels = entry.channels;
            how = AuxLayout::Verbatim;
            texelBytes = static_cast<uint32_t>(aux_texel_bytes(entry.arrayFormat, entry.channels));
            return true;
        }
    // Converted formats all end up as float arrays, so a shader read of the denoiser's
    // texture returns the same numbers a D3D12 read of the caller's resource would. Their
    // texel size is the caller's own, which is what a capture records.
    struct Converted
    {
        DXGI_FORMAT format;
        uint32_t channels;
        uint32_t texelBytes;
    };
    static const Converted converted[] = {
        {DXGI_FORMAT_R8_UINT, 1, 1},
        {DXGI_FORMAT_R8G8_UINT, 2, 2},
        {DXGI_FORMAT_R8G8B8A8_UINT, 4, 4},
        {DXGI_FORMAT_R16_UINT, 1, 2},
        {DXGI_FORMAT_R32_UINT, 1, 4},
        {DXGI_FORMAT_R32_SINT, 1, 4},
        {DXGI_FORMAT_R32G32_UINT, 2, 8},
        {DXGI_FORMAT_R32G32_SINT, 2, 8},
        {DXGI_FORMAT_R32G32B32A32_UINT, 4, 16},
        {DXGI_FORMAT_B8G8R8A8_UNORM, 4, 4},
        {DXGI_FORMAT_R10G10B10A2_UNORM, 4, 4},
        {DXGI_FORMAT_R11G11B10_FLOAT, 4, 4},
    };
    for (const Converted& entry : converted)
        if (entry.format == format)
        {
            arrayFormat = CUDA_FORMAT_FLOAT;
            channels = entry.channels;
            how = AuxLayout::Converted;
            texelBytes = entry.texelBytes;
            return true;
        }
    return false;
}

// Unpacks one row of a converted denoiser input into `channels` floats per texel. Every
// branch reads exactly the bytes of one texel and no more.

void convert_aux_row(DXGI_FORMAT format, const uint8_t* source, float* destination, UINT width,
                            uint32_t channels)
{
    switch (format)
    {
    case DXGI_FORMAT_R8_UINT:
        for (UINT x = 0; x < width; ++x)
            destination[x] = static_cast<float>(source[x]);
        return;
    case DXGI_FORMAT_R8G8_UINT:
        for (UINT x = 0; x < width; ++x)
        {
            destination[x * 2] = static_cast<float>(source[x * 2]);
            destination[x * 2 + 1] = static_cast<float>(source[x * 2 + 1]);
        }
        return;
    case DXGI_FORMAT_R8G8B8A8_UINT:
        for (UINT x = 0; x < width; ++x)
            for (uint32_t c = 0; c < 4; ++c)
                destination[x * 4 + c] = static_cast<float>(source[x * 4 + c]);
        return;
    case DXGI_FORMAT_R16_UINT:
        for (UINT x = 0; x < width; ++x)
        {
            uint16_t value;
            std::memcpy(&value, source + x * 2, sizeof(value));
            destination[x] = static_cast<float>(value);
        }
        return;
    case DXGI_FORMAT_R32_UINT:
        for (UINT x = 0; x < width; ++x)
        {
            uint32_t value;
            std::memcpy(&value, source + x * 4, sizeof(value));
            destination[x] = static_cast<float>(value);
        }
        return;
    case DXGI_FORMAT_R32_SINT:
        for (UINT x = 0; x < width; ++x)
        {
            int32_t value;
            std::memcpy(&value, source + x * 4, sizeof(value));
            destination[x] = static_cast<float>(value);
        }
        return;
    case DXGI_FORMAT_R32G32_UINT:
        for (UINT x = 0; x < width; ++x)
        {
            uint32_t value[2];
            std::memcpy(value, source + x * 8, sizeof(value));
            destination[x * 2] = static_cast<float>(value[0]);
            destination[x * 2 + 1] = static_cast<float>(value[1]);
        }
        return;
    case DXGI_FORMAT_R32G32_SINT:
        for (UINT x = 0; x < width; ++x)
        {
            int32_t value[2];
            std::memcpy(value, source + x * 8, sizeof(value));
            destination[x * 2] = static_cast<float>(value[0]);
            destination[x * 2 + 1] = static_cast<float>(value[1]);
        }
        return;
    case DXGI_FORMAT_R32G32B32A32_UINT:
        for (UINT x = 0; x < width; ++x)
            for (uint32_t c = 0; c < 4; ++c)
            {
                uint32_t value;
                std::memcpy(&value, source + x * 16 + c * 4, sizeof(value));
                destination[x * 4 + c] = static_cast<float>(value);
            }
        return;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        // Stored B, G, R, A; the denoiser's texture is read as R, G, B, A.
        for (UINT x = 0; x < width; ++x)
        {
            const uint8_t* texel = source + x * 4;
            destination[x * 4] = unorm(texel[2], 8);
            destination[x * 4 + 1] = unorm(texel[1], 8);
            destination[x * 4 + 2] = unorm(texel[0], 8);
            destination[x * 4 + 3] = unorm(texel[3], 8);
        }
        return;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        for (UINT x = 0; x < width; ++x)
        {
            uint32_t packed;
            std::memcpy(&packed, source + x * 4, sizeof(packed));
            destination[x * 4] = unorm(packed & 0x3FF, 10);
            destination[x * 4 + 1] = unorm((packed >> 10) & 0x3FF, 10);
            destination[x * 4 + 2] = unorm((packed >> 20) & 0x3FF, 10);
            destination[x * 4 + 3] = unorm((packed >> 30) & 0x3, 2);
        }
        return;
    case DXGI_FORMAT_R11G11B10_FLOAT:
        for (UINT x = 0; x < width; ++x)
        {
            uint32_t packed;
            std::memcpy(&packed, source + x * 4, sizeof(packed));
            destination[x * 4] = small_float_to_float(packed & 0x7FF, 6);
            destination[x * 4 + 1] = small_float_to_float((packed >> 11) & 0x7FF, 6);
            destination[x * 4 + 2] = small_float_to_float((packed >> 22) & 0x3FF, 5);
            destination[x * 4 + 3] = 1.0f; // missing alpha follows shader texture reads
        }
        return;
    default:
        break;
    }
    // Unreachable: aux_array_layout only marks a format Converted when this switch handles it.
    for (UINT x = 0; x < width * channels; ++x)
        destination[x] = 0.0f;
}

// The texel size `format` occupies, for the pitch of an interop buffer holding it verbatim.

size_t aux_texel_bytes(uint32_t arrayFormat, uint32_t channels)
{
    const size_t element = arrayFormat == CUDA_FORMAT_FLOAT || arrayFormat == CUDA_FORMAT_UNSIGNED_INT32 ||
                                   arrayFormat == CUDA_FORMAT_SIGNED_INT32
                               ? 4
                               : (arrayFormat == CUDA_FORMAT_HALF || arrayFormat == CUDA_FORMAT_UNSIGNED_INT16 ||
                                  arrayFormat == CUDA_FORMAT_SIGNED_INT16)
                                     ? 2
                                     : 1;
    return element * channels;
}

bool create_buffer(D3D12_HEAP_TYPE heapType, UINT64 size, ID3D12Resource** buffer, uint8_t** mapped)
{
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = heapType;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_RESOURCE_STATES state =
        heapType == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ;
    HRESULT hr = g.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                   __uuidof(ID3D12Resource), reinterpret_cast<void**>(buffer));
    if (FAILED(hr))
    {
        logf("CreateCommittedResource(heap=%d, %llu bytes) failed: 0x%08lx", heapType,
             static_cast<unsigned long long>(size), hr);
        return false;
    }
    D3D12_RANGE readRange = {0, heapType == D3D12_HEAP_TYPE_READBACK ? static_cast<SIZE_T>(size) : 0};
    hr = (*buffer)->Map(0, &readRange, reinterpret_cast<void**>(mapped));
    if (FAILED(hr))
    {
        logf("Map failed: 0x%08lx", hr);
        (*buffer)->Release();
        *buffer = nullptr;
        return false;
    }
    return true;
}

// Ensures staging matches the resource's first subresource; heap selects
// readback (inputs) or upload (output).

bool ensure_staging(Staging& staging, ID3D12Resource* resource, D3D12_HEAP_TYPE heapType)
{
    D3D12_RESOURCE_DESC desc;
    resource->GetDesc(&desc);
    if (staging.buffer != nullptr && staging.format == desc.Format && staging.width == desc.Width &&
        staging.height == desc.Height)
        return true;
    staging.retire();
    UINT64 rowSize = 0;
    g.device->GetCopyableFootprints(&desc, 0, 1, 0, &staging.layout, &staging.rows, &rowSize, &staging.total);
    if (!create_buffer(heapType, staging.total, &staging.buffer, &staging.mapped))
        return false;
    staging.format = desc.Format;
    staging.width = static_cast<UINT>(desc.Width);
    staging.height = desc.Height;
    logf("staging %s %ux%u format=%d rowPitch=%u total=%llu", heapType == D3D12_HEAP_TYPE_READBACK ? "readback" : "upload",
         staging.width, staging.height, staging.format, staging.layout.Footprint.RowPitch,
         static_cast<unsigned long long>(staging.total));
    return true;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                       D3D12_RESOURCE_STATES after)
{
    // A null pResource invalidates the whole command list in vkd3d-proton ("A resource pointer is NULL.")
    if (resource == nullptr || before == after)
        return;
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

void copy_to_staging(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, Staging& staging,
                            D3D12_RESOURCE_STATES state)
{
    transition(list, resource, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION destination = {};
    destination.pResource = staging.buffer;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = staging.layout;
    D3D12_TEXTURE_COPY_LOCATION source = {};
    source.pResource = resource;
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source.SubresourceIndex = 0;
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    transition(list, resource, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
}

// --- CUDA-side resources (worker thread) -----------------------------------------

// Sampler of an input texture: clamp addressing, normalized coordinates (as NGX's own textures); bilinear
// for the colour input (D4R_SHIM_COLOR_LINEAR), optionally for motion / depth (experiments).

CudaTextureDesc input_sampler(UINT width, uint32_t channels, bool linear)
{
    CudaTextureDesc sampler = {};
    sampler.addressMode[0] = 1;
    sampler.addressMode[1] = 1;
    static const bool colorLinear = env_uint("D4R_SHIM_COLOR_LINEAR", 1) != 0;
    sampler.filterMode = linear && colorLinear ? 1 : 0;
    static const bool motionLinear = env_uint("D4R_SHIM_MOTION_LINEAR", 0) != 0;
    static const bool depthLinear = env_uint("D4R_SHIM_DEPTH_LINEAR", 0) != 0;
    if ((motionLinear && channels == 2) || (depthLinear && channels == 1 && width > 1))
        sampler.filterMode = 1;
    sampler.flags = 2; // CU_TRSF_NORMALIZED_COORDINATES
    return sampler;
}

bool ensure_cuda_image(CudaImage& image, UINT width, UINT height, uint32_t format, uint32_t channels,
                              bool surface, bool linear)
{
    if (image.array != nullptr && image.width == width && image.height == height &&
        image.format == format && image.channels == channels)
        return true;
    if (image.object != 0)
        surface ? g.cu.surfObjectDestroy(image.object) : g.cu.texObjectDestroy(image.object);
    if (image.array != nullptr)
        g.cu.arrayDestroy(image.array);
    image = CudaImage{};
    int result;
    if (surface)
    {
        const CudaArray3DDescriptor descriptor{width, height, 0, format, channels, 2u /* SURFACE_LDST */};
        result = g.cu.array3DCreate(&image.array, &descriptor);
    }
    else
    {
        const CudaArrayDescriptor descriptor{width, height, format, channels};
        result = g.cu.arrayCreate(&image.array, &descriptor);
    }
    if (result != 0)
    {
        logf("CUDA array create %ux%u fmt=0x%x ch=%u failed: %d", width, height, format, channels, result);
        return false;
    }
    CudaResourceDesc resource = {};
    resource.resType = 0; // CU_RESOURCE_TYPE_ARRAY
    resource.res.array.hArray = image.array;
    if (surface)
        result = g.cu.surfObjectCreate(&image.object, &resource);
    else
    {
        // Matches the texture objects NGX itself creates: clamp addressing,
        // normalized coordinates.
        CudaTextureDesc sampler = {};
        sampler.addressMode[0] = 1;
        sampler.addressMode[1] = 1;
        // D4R_SHIM_COLOR_LINEAR=0 samples the colour input with point filtering (experiment).
        static const bool colorLinear = env_uint("D4R_SHIM_COLOR_LINEAR", 1) != 0;
        sampler.filterMode = linear && colorLinear ? 1 : 0;
        // D4R_SHIM_MOTION_LINEAR / D4R_SHIM_DEPTH_LINEAR=1: bilinear filtering for
        // those planes (identified by format; experiment).
        static const bool motionLinear = env_uint("D4R_SHIM_MOTION_LINEAR", 0) != 0;
        static const bool depthLinear = env_uint("D4R_SHIM_DEPTH_LINEAR", 0) != 0;
        if ((motionLinear && channels == 2) || (depthLinear && channels == 1 && width > 1))
            sampler.filterMode = 1;
        sampler.flags = 2; // CU_TRSF_NORMALIZED_COORDINATES
        result = g.cu.texObjectCreate(&image.object, &resource, &sampler, nullptr);
    }
    if (result != 0)
    {
        logf("CUDA %s object create failed: %d", surface ? "surface" : "texture", result);
        return false;
    }
    image.width = width;
    image.height = height;
    image.format = format;
    image.channels = channels;
    return true;
}

// Device <-> array copy. Queued on the null stream, where NGX evaluates, when the bridge provides
// it (D4R_SHIM_ASYNC_2D=0: ZLUDA's cuMemcpy2D); callers synchronize before the data is used.

int copy_2d(const CudaMemcpy2D& copy)
{
    static const bool async = env_uint("D4R_SHIM_ASYNC_2D", 1) != 0;
    if (async && g.cu.memcpy2DAsync != nullptr && g.cu.memcpy2DAsync(&copy, nullptr) == 0)
        return 0;
    return g.cu.memcpy2D(&copy);
}

// Copies rows between pitched buffers.

void copy_rows(RowPool& pool, uint8_t* destination, size_t destinationPitch, const uint8_t* source,
                      size_t sourcePitch, size_t rowBytes, UINT rows)
{
    pool.run(rows, rowBytes * rows, [&](UINT begin, UINT end) {
        if (destinationPitch == rowBytes && sourcePitch == rowBytes)
            std::memcpy(destination + begin * rowBytes, source + begin * rowBytes, rowBytes * (end - begin));
        else
            for (UINT y = begin; y < end; ++y)
                std::memcpy(destination + y * destinationPitch, source + y * sourcePitch, rowBytes);
    });
}

// Overlapped staging: frame N's uploads run while frame N-1's result downloads
// (PCIe is full duplex), as async copies between page-locked host memory and
// linear device buffers on two non-blocking streams. D4R_SHIM_ASYNC_COPIES=0
// keeps the synchronous host<->array copies.

bool async_copies()
{
    static const bool enabled = env_uint("D4R_SHIM_ASYNC_COPIES", 1) != 0;
    return enabled && g.cu.memAllocHost != nullptr;
}

// Worker thread only.

CopyStreams* copy_streams()
{
    static CopyStreams streams;
    static bool attempted = false;
    if (!attempted)
    {
        attempted = true;
        constexpr unsigned int kNonBlocking = 1; // CU_STREAM_NON_BLOCKING
        if (g.cu.streamCreate(&streams.up, kNonBlocking) != 0 || g.cu.streamCreate(&streams.down, kNonBlocking) != 0)
        {
            logf("cuStreamCreate failed; using synchronous copies");
            streams = {};
        }
    }
    return streams.up != nullptr ? &streams : nullptr;
}

// Grows a host plane's buffer. Page-locked memory is allocated on the CUDA
// worker thread (the only one with a current context).

void reserve_host(HostPlane& host, size_t bytes)
{
    if (bytes <= host.capacity)
        return;
    auto allocate = [&host, bytes] {
        if (host.pinned)
            g.cu.memFreeHost(host.bytes);
        host = HostPlane{};
        void* pinned = nullptr;
        if (async_copies() && g.cu.memAllocHost(&pinned, bytes) == 0)
        {
            host.bytes = static_cast<uint8_t*>(pinned);
            host.pinned = true;
        }
        else
        {
            host.fallback.resize(bytes);
            host.bytes = host.fallback.data();
        }
        host.capacity = bytes;
        return 0;
    };
    if (async_copies() && std::this_thread::get_id() != g.workerThread)
        g.worker.call(allocate);
    else
        allocate();
}

void release_host(HostPlane& host)
{
    if (host.pinned)
        g.cu.memFreeHost(host.bytes);
    host = HostPlane{};
}

// Worker thread only.

bool ensure_linear(CudaDevicePtr& buffer, size_t& capacity, size_t bytes)
{
    if (bytes <= capacity)
        return true;
    if (buffer != 0)
        g.cu.memFree(buffer);
    buffer = 0;
    capacity = 0;
    if (g.cu.memAlloc(&buffer, bytes) != 0)
    {
        buffer = 0;
        logf("cuMemAlloc(%zu) for staging failed", bytes);
        return false;
    }
    capacity = bytes;
    return true;
}

uint32_t plane_format(Plane plane)
{
    return (plane == Plane::Color || plane == Plane::Motion) ? CUDA_FORMAT_HALF : CUDA_FORMAT_FLOAT;
}

uint32_t plane_channels(Plane plane)
{
    return plane == Plane::Color ? 4 : (plane == Plane::Motion ? 2 : 1);
}
