#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include "../../engine/runtime.h"
#include "binary16.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <algorithm>
#include <set>
#include <cstdio>
#include <cstring>
#include <string>

// Canonical-input replay through the native D3D12/Vulkan shared packet. The
// packet is GPU memory at the API boundary; CPU access is fixture upload/QA only.
namespace rr_bridge {
inline void check(VkResult r, const char* operation) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(operation) + " VkResult=" + std::to_string(r));
}
inline std::vector<char> read(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot read RR fixture: " + path.string());
    std::vector<char> result(size_t(f.tellg())); f.seekg(0); f.read(result.data(), std::streamsize(result.size()));
    if (!f) throw std::runtime_error("Incomplete RR fixture read");
    return result;
}
struct Frame { uint32_t id; d4r::RFrameSettings settings; std::array<bool, 4> optional{}; };
struct Plane {
    const char* name; VkFormat format; uint32_t bytes;
    VkDeviceSize offset = 0, pitch = 0;
    bool enabled = true;
};
struct Layout {
    uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
    std::array<Plane, 11> planes{{
        {"color", VK_FORMAT_R16G16B16A16_SFLOAT, 8}, {"motion", VK_FORMAT_R16G16_SFLOAT, 4},
        {"depth", VK_FORMAT_R32_SFLOAT, 4}, {"normals", VK_FORMAT_R16G16B16A16_SFLOAT, 8},
        {"roughness", VK_FORMAT_R16G16B16A16_SFLOAT, 8}, {"diffuse", VK_FORMAT_R8G8B8A8_UNORM, 4},
        {"specular", VK_FORMAT_R8G8B8A8_UNORM, 4},
        {"hitdistance", VK_FORMAT_R32_SFLOAT, 4, 0, 0, false},
        {"disocclusion", VK_FORMAT_R16_SFLOAT, 2, 0, 0, false},
        {"sss", VK_FORMAT_R16_SFLOAT, 2, 0, 0, false},
        {"particles", VK_FORMAT_R16G16B16A16_SFLOAT, 8, 0, 0, false}}};
    VkDeviceSize bytes = 0, outputOffset = 0, outputPitch = 0;
    VkFormat outputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    uint32_t outputBytes = 8;
    std::filesystem::path inputs, output, reference;
    std::vector<Frame> frames;
    // Runtime layout construction performs no fixture I/O. Both callers use
    // the same aligned packet contract for native texture transfers.
    Layout(uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight)
        : rw(renderWidth), rh(renderHeight), ow(outputWidth), oh(outputHeight) {
        if (!rw || !rh || !ow || !oh || rw > 16384 || rh > 16384 || ow > 16384 || oh > 16384)
            throw std::runtime_error("Invalid RR packet geometry");
        initializePacket();
    }
    void initializePacket() {
        bytes = 0;
        for (auto& plane : planes) {
            bytes = (bytes + 511) & ~VkDeviceSize(511); plane.offset = bytes;
            plane.pitch = (VkDeviceSize(rw) * plane.bytes + 255) & ~VkDeviceSize(255);
            bytes += plane.pitch * rh;
        }
        outputOffset = (bytes + 511) & ~VkDeviceSize(511);
        outputPitch = (VkDeviceSize(ow) * outputBytes + 255) & ~VkDeviceSize(255);
        bytes = outputOffset + outputPitch * oh;
    }
    Layout(const std::string& inputDir, const std::string& outputDir, const std::string& referenceDir)
        : inputs(inputDir), output(outputDir), reference(referenceDir) {
        std::ifstream f(inputs / "frames.txt"); std::string token;
        if (!(f >> token >> rw >> rh >> ow >> oh) || token != "size" || rw < 32 || rh < 32 ||
            rw > 2560 || rh > 1440 || !ow || !oh || ow > 2560 || oh > 1440)
            throw std::runtime_error("Invalid bounded RR replay geometry");
        while (f >> token) {
            Frame frame{}; int reset, mode;
            auto& s = frame.settings;
            if (token != "frame" || !(f >> frame.id >> s.jitter[0] >> s.jitter[1] >> s.motionScale[0] >>
                s.motionScale[1] >> s.preExposure >> reset >> mode)) throw std::runtime_error("Invalid RR frame header");
            for (auto& value : s.worldToView) if (!(f >> value)) throw std::runtime_error("Invalid RR view matrix");
            for (auto& value : s.viewToClip) if (!(f >> value)) throw std::runtime_error("Invalid RR projection matrix");
            s.reset = reset != 0; s.depthInverted = (mode & 1) != 0; s.hardwareDepth = (mode & 2) != 0;
            s.roughnessPacked = (mode & 4) != 0;
            if (!std::isfinite(s.preExposure) || s.preExposure <= 0 ||
                !std::isfinite(s.jitter[0]) || !std::isfinite(s.jitter[1]) ||
                !std::isfinite(s.motionScale[0]) || !std::isfinite(s.motionScale[1]))
                throw std::runtime_error("Nonfinite or invalid RR frame settings");
            for (float value : s.worldToView) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite RR view matrix");
            for (float value : s.viewToClip) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite RR projection matrix");
            frames.push_back(frame);
            if (frames.size() > 32) throw std::runtime_error("RR replay is limited to 32 frames");
        }
        if (frames.empty()) throw std::runtime_error("RR fixture has no frames");
        std::set<uint32_t> ids;
        for (auto& frame : frames) {
            if (!ids.insert(frame.id).second) throw std::runtime_error("Duplicate RR frame ID");
            for (size_t i = 7; i < planes.size(); ++i) {
                frame.optional[i - 7] = std::filesystem::exists(inputs / (std::to_string(frame.id) + "-" + planes[i].name + ".bin"));
                planes[i].enabled = planes[i].enabled || frame.optional[i - 7];
            }
        }
        initializePacket();
        std::filesystem::create_directories(output);
        for (const auto& frame : frames)
            if (std::filesystem::exists(output / (std::to_string(frame.id) + "-out.bin")))
                throw std::runtime_error("Use a fresh RR output directory");
    }
    void upload(void* destination, unsigned frame) const {
        std::memset(destination, 0, size_t(bytes));
        for (const auto& p : planes) {
            if (!p.enabled) continue;
            const auto index = size_t(&p - planes.data());
            if (index >= 7 && !frames.at(frame).optional[index - 7]) continue;
            auto data = read(inputs / (std::to_string(frames.at(frame).id) + "-" + p.name + ".bin"));
            if (data.size() != size_t(rw) * rh * p.bytes) throw std::runtime_error("RR input size mismatch");
            for (uint32_t y = 0; y < rh; ++y)
                std::memcpy(static_cast<char*>(destination) + p.offset + y * p.pitch,
                            data.data() + size_t(y) * rw * p.bytes, size_t(rw) * p.bytes);
        }
    }
    void validate(const void* source, unsigned frame) const {
        std::vector<char> packed(size_t(ow) * oh * outputBytes);
        for (uint32_t y = 0; y < oh; ++y)
            std::memcpy(packed.data() + size_t(y) * ow * outputBytes,
                        static_cast<const char*>(source) + outputOffset + y * outputPitch, size_t(ow) * outputBytes);
        double low = std::numeric_limits<double>::infinity(), high = -low;
        for (size_t i = 0; i < size_t(ow) * oh * 4; ++i) {
            double value;
            if (outputBytes == 16) { float v; std::memcpy(&v, packed.data() + i * 4, 4); value = v; }
            else { uint16_t bits; std::memcpy(&bits, packed.data() + i * 2, 2); value = d4r::diag::binary16::half_value(bits); }
            if (!std::isfinite(value)) throw std::runtime_error("RR output contains NaN/Inf");
            if (i % 4 != 3) { low = std::min(low, value); high = std::max(high, value); }
        }
        if (!(high - low > 1e-5)) throw std::runtime_error("RR output is uniform");
        const auto name = std::to_string(frames.at(frame).id) + "-out.bin";
        if (!reference.empty()) {
            auto expected = read(reference / name);
            if (outputBytes == 16) {
                if (expected.size() != size_t(ow) * oh * 8) throw std::runtime_error("Invalid half output reference size");
                std::vector<char> widened(packed.size());
                for (size_t i = 0; i < expected.size() / 2; ++i) {
                    uint16_t bits; std::memcpy(&bits, expected.data() + i * 2, 2);
                    const float value = float(d4r::diag::binary16::half_value(bits));
                    std::memcpy(widened.data() + i * 4, &value, 4);
                }
                expected = std::move(widened);
            }
            if (expected != packed) throw std::runtime_error("Bridged RR output differs from standalone replay");
        }
        std::ofstream file(output / name, std::ios::binary); file.write(packed.data(), std::streamsize(packed.size()));
        if (!file) throw std::runtime_error("RR output write failed");
        std::printf("RR_BRIDGED_FRAME id=%u finite=1 nonuniform=1 standalone_exact=%u output=%ux%u\n",
                    frames[frame].id, !reference.empty(), ow, oh);
    }
};
struct D3D12Frames {
    const Layout& layout;
    std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, 11> inputs;
    Microsoft::WRL::ComPtr<ID3D12Resource> output;
    static void checkD3D(HRESULT r) { if (FAILED(r)) throw std::runtime_error("RR D3D12 texture operation failed: " + std::to_string(r)); }
    static DXGI_FORMAT format(VkFormat f) {
        switch (f) {
        case VK_FORMAT_R16G16B16A16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case VK_FORMAT_R16G16_SFLOAT: return DXGI_FORMAT_R16G16_FLOAT;
        case VK_FORMAT_R32_SFLOAT: return DXGI_FORMAT_R32_FLOAT;
        case VK_FORMAT_R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case VK_FORMAT_R32G32_SFLOAT: return DXGI_FORMAT_R32G32_FLOAT;
        case VK_FORMAT_R16_SFLOAT: return DXGI_FORMAT_R16_FLOAT;
        case VK_FORMAT_R8_UNORM: return DXGI_FORMAT_R8_UNORM;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return DXGI_FORMAT_R10G10B10A2_UNORM;
        default: throw std::runtime_error("Unsupported canonical RR D3D12 format");
        }
    }
    static void barrier(ID3D12GraphicsCommandList* cb, ID3D12Resource* image, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {image, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to}; cb->ResourceBarrier(1, &b);
    }
    static D3D12_TEXTURE_COPY_LOCATION placed(ID3D12Resource* resource, VkDeviceSize offset, VkDeviceSize pitch,
                                             DXGI_FORMAT f, uint32_t width, uint32_t height) {
        D3D12_TEXTURE_COPY_LOCATION result{}; result.pResource = resource;
        result.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; result.PlacedFootprint.Offset = offset;
        result.PlacedFootprint.Footprint = {f, width, height, 1, UINT(pitch)}; return result;
    }
    static D3D12_TEXTURE_COPY_LOCATION texture(ID3D12Resource* resource) {
        D3D12_TEXTURE_COPY_LOCATION result{}; result.pResource = resource;
        result.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; return result;
    }
    D3D12Frames(const Layout& l, ID3D12Device* device, DXGI_FORMAT depthFormat = DXGI_FORMAT_UNKNOWN) : layout(l) {
        auto create = [&](DXGI_FORMAT f, uint32_t width, uint32_t height, bool isOutput, ID3D12Resource** resource) {
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Format = f;
            desc.Width = width; desc.Height = height; desc.DepthOrArraySize = desc.MipLevels = 1; desc.SampleDesc.Count = 1;
            if (isOutput) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            if (f == DXGI_FORMAT_D32_FLOAT || f == DXGI_FORMAT_R32G8X24_TYPELESS || f == DXGI_FORMAT_D32_FLOAT_S8X24_UINT)
                desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            checkD3D(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                isOutput ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                nullptr, IID_PPV_ARGS(resource)));
        };
        for (size_t i = 0; i < inputs.size(); ++i) if (l.planes[i].enabled)
            create(i == 2 && depthFormat != DXGI_FORMAT_UNKNOWN ? depthFormat : format(l.planes[i].format), l.rw, l.rh, false, inputs[i].GetAddressOf());
        create(format(l.outputFormat), l.ow, l.oh, true, output.GetAddressOf());
    }
    void recordInputs(ID3D12GraphicsCommandList* cb, ID3D12Resource* upload, ID3D12Resource* shared, uint32_t optionalMask = 15) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i]) continue;
            if (i >= 7 && !(optionalMask & (1u << (i - 7)))) continue;
            auto* input = inputs[i].Get(); const auto& plane = layout.planes[i];
            auto copyFormat = format(plane.format);
            if (i == 2) {
                Microsoft::WRL::ComPtr<ID3D12Device> device; checkD3D(input->GetDevice(IID_PPV_ARGS(&device)));
                D3D12_RESOURCE_DESC desc{};
#if defined(__MINGW32__)
                input->GetDesc(&desc);
#else
                desc = input->GetDesc();
#endif
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
                device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
                copyFormat = footprint.Footprint.Format;
            }
            auto src = placed(upload, plane.offset, plane.pitch, copyFormat, layout.rw, layout.rh);
            auto image = texture(input);
            barrier(cb, input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            cb->CopyTextureRegion(&image, 0, 0, 0, &src, nullptr);
            barrier(cb, input, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            auto dst = placed(shared, plane.offset, plane.pitch, copyFormat, layout.rw, layout.rh);
            // A dynamic subrect can occupy a creation-sized fixture texture.
            // Only depth-stencil resources require whole-subresource copies.
            Microsoft::WRL::ComPtr<ID3D12Device> owner; checkD3D(input->GetDevice(IID_PPV_ARGS(&owner)));
            D3D12_RESOURCE_DESC description{};
#if defined(__MINGW32__)
            input->GetDesc(&description);
#else
            description = input->GetDesc();
#endif
            D3D12_BOX box{0, 0, 0, layout.rw, layout.rh, 1};
            cb->CopyTextureRegion(&dst, 0, 0, 0, &image,
                (description.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) ? nullptr : &box);
            barrier(cb, input, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
    }
    void recordOutput(ID3D12GraphicsCommandList* cb, ID3D12Resource* shared, ID3D12Resource* readback) {
        auto src = placed(shared, layout.outputOffset, layout.outputPitch, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.ow, layout.oh);
        auto image = texture(output.Get());
        barrier(cb, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        cb->CopyTextureRegion(&image, 0, 0, 0, &src, nullptr);
        barrier(cb, output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto dst = placed(readback, layout.outputOffset, layout.outputPitch, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.ow, layout.oh);
        cb->CopyTextureRegion(&dst, 0, 0, 0, &image, nullptr);
        barrier(cb, output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
};
struct Image {
    VkFormat format = VK_FORMAT_UNDEFINED; bool initialized = false;
    VkDevice device; VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE;
    ~Image() { if (view) vkDestroyImageView(device, view, nullptr); if (image) vkDestroyImage(device, image, nullptr); if (memory) vkFreeMemory(device, memory, nullptr); }
};
struct Replay {
    const Layout& layout;
    VkPhysicalDevice physical; VkDevice device;
    std::array<std::unique_ptr<Image>, 11> images;
    std::array<std::vector<std::unique_ptr<Image>>, 11> alternatives;
    std::unique_ptr<d4r::RModel> model;
    std::unique_ptr<d4r::REngine> engine;
    std::unique_ptr<Image> convertedOutput;
    bool first = true;
    bool modelUploaded = false;
    Replay(const Layout& l, VkPhysicalDevice physical, VkDevice device, const std::string& directory)
        : layout(l), physical(physical), device(device) {
        const auto data = read(std::filesystem::path(directory) / "rr.txt");
        const std::string manifest(data.begin(), data.end());
        if (manifest.rfind("D4RR0001", 0) != 0 || manifest.find("arithmetic gfx11") == std::string::npos)
            throw std::runtime_error("This canonical RR bridge gate requires a D/E gfx11 model");
        for (size_t i = 0; i < images.size(); ++i) if (l.planes[i].enabled) configure(i, l.planes[i].format);
        model = std::make_unique<d4r::RModel>(physical, device, directory);
        engine = std::make_unique<d4r::REngine>(*model, l.rw, l.rh, l.ow, l.oh);
    }
    // Caller retires all submissions before changing layout extents. Pipelines
    // and uploaded weights survive; descriptors/views and temporal state do not.
    void resize() {
        engine.reset();
        for (auto& image : images) image.reset();
        for (auto& cached : alternatives) cached.clear();
        for (size_t i = 0; i < images.size(); ++i)
            if (layout.planes[i].enabled) configure(i, layout.planes[i].format);
        engine = std::make_unique<d4r::REngine>(*model, layout.rw, layout.rh, layout.ow, layout.oh);
        first = true;
    }
    void configureOutput() {
        if (layout.outputFormat == VK_FORMAT_R16G16B16A16_SFLOAT) return;
        if (layout.outputFormat != VK_FORMAT_R32G32B32A32_SFLOAT)
            throw std::runtime_error("Unsupported RR output conversion");
        if (convertedOutput) return;
        VkFormatProperties source{}, target{};
        vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_R16G16B16A16_SFLOAT, &source);
        vkGetPhysicalDeviceFormatProperties(physical, layout.outputFormat, &target);
        if (!(source.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ||
            (target.optimalTilingFeatures & (VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT)) !=
                (VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT))
            throw std::runtime_error("RR float32 output blit unsupported");
        auto image = std::make_unique<Image>(); image->device = device; image->format = layout.outputFormat;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = image->format; ci.extent = {layout.ow, layout.oh, 1}; ci.mipLevels = ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        check(vkCreateImage(device, &ci, nullptr, &image->image), "RR converted output image");
        VkMemoryRequirements req; vkGetImageMemoryRequirements(device, image->image, &req);
        VkPhysicalDeviceMemoryProperties memory; vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        uint32_t type = memory.memoryTypeCount;
        for (uint32_t n = 0; n < memory.memoryTypeCount; ++n)
            if ((req.memoryTypeBits & (1u << n)) && (memory.memoryTypes[n].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { type = n; break; }
        if (type == memory.memoryTypeCount) throw std::runtime_error("No converted RR output memory type");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = type;
        check(vkAllocateMemory(device, &ai, nullptr, &image->memory), "RR converted output memory");
        check(vkBindImageMemory(device, image->image, image->memory, 0), "RR converted output bind");
        convertedOutput = std::move(image);
    }
    // Cache views by format: REngine caches descriptors referencing old views,
    // so they must remain alive until the engine is destroyed.
    void configure(size_t i, VkFormat format) {
        if (images.at(i) && images[i]->format == format) return;
        for (auto& previous : alternatives[i]) if (previous->format == format) {
            images[i].swap(previous); return;
        }
        VkFormatProperties properties{}; vkGetPhysicalDeviceFormatProperties(physical, format, &properties);
        const auto required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((properties.optimalTilingFeatures & required) != required)
            throw std::runtime_error("RR input format lacks Vulkan sampled/transfer support");
        VkPhysicalDeviceMemoryProperties memory; vkGetPhysicalDeviceMemoryProperties(physical, &memory);
            auto image = std::make_unique<Image>(); image->device = device; image->format = format;
            VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ci.imageType = VK_IMAGE_TYPE_2D;
            ci.format = format; ci.extent = {layout.rw, layout.rh, 1}; ci.mipLevels = ci.arrayLayers = 1;
            ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            check(vkCreateImage(device, &ci, nullptr, &image->image), "RR input image");
            VkMemoryRequirements req; vkGetImageMemoryRequirements(device, image->image, &req);
            uint32_t type = memory.memoryTypeCount;
            for (uint32_t n = 0; n < memory.memoryTypeCount; ++n)
                if ((req.memoryTypeBits & (1u << n)) && (memory.memoryTypes[n].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { type = n; break; }
            if (type == memory.memoryTypeCount) throw std::runtime_error("No RR image memory type");
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = type;
            check(vkAllocateMemory(device, &ai, nullptr, &image->memory), "RR image memory");
            check(vkBindImageMemory(device, image->image, image->memory, 0), "Bind RR image");
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = image->image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = ci.format; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            check(vkCreateImageView(device, &vi, nullptr, &image->view), "RR input view");
            if (images[i]) alternatives[i].push_back(std::move(images[i]));
            images[i] = std::move(image);
    }
    void record(VkCommandBuffer cb, VkBuffer shared, unsigned frame) {
        uint32_t mask = 0;
        for (unsigned i = 0; i < 4; ++i) if (layout.frames.at(frame).optional[i]) mask |= 1u << i;
        record(cb, shared, layout.frames.at(frame).settings, mask);
    }
    // Own settings at the NGX submission boundary; never retain game pointers.
    void record(VkCommandBuffer cb, VkBuffer shared, const d4r::RFrameSettings& settings, uint32_t optionalMask = 15) {
        for (size_t i = 0; i < images.size(); ++i) {
            if (!layout.planes[i].enabled) continue;
            if (i >= 7 && !(optionalMask & (1u << (i - 7)))) continue;
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.image = images[i]->image;
            b.oldLayout = !images[i]->initialized ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b.srcAccessMask = !images[i]->initialized ? 0 : VK_ACCESS_SHADER_READ_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            const auto& plane = layout.planes[i];
            VkBufferImageCopy copy{}; copy.bufferOffset = plane.offset; copy.bufferRowLength = uint32_t(plane.pitch / plane.bytes);
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = {layout.rw, layout.rh, 1};
            vkCmdCopyBufferToImage(cb, shared, images[i]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            images[i]->initialized = true;
            b.oldLayout = b.newLayout; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        }
        d4r::RFrameImages inputs{}; inputs.color = images[0]->view; inputs.motion = images[1]->view; inputs.depth = images[2]->view;
        inputs.normals = images[3]->view; inputs.roughness = images[4]->view; inputs.diffuseAlbedo = images[5]->view; inputs.specularAlbedo = images[6]->view;
        VkImageView* optional[] = {&inputs.specularHitDistance, &inputs.disocclusionMask, &inputs.subsurfaceGuide, &inputs.colorBeforeParticles};
        for (size_t i = 7; i < images.size(); ++i) if (layout.planes[i].enabled && (optionalMask & (1u << (i - 7)))) *optional[i - 7] = images[i]->view;
        engine->setFrame(inputs, settings);
        if (first) {
            if (!modelUploaded) { model->recordUpload(cb); modelUploaded = true; }
            engine->recordInitialize(cb); first = false;
        }
        engine->recordFrame(cb);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.image = engine->outputImage();
        b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkImage output = engine->outputImage();
        if (layout.outputFormat != VK_FORMAT_R16G16B16A16_SFLOAT) {
            if (!convertedOutput) throw std::runtime_error("RR converted output was not configured");
            VkImageMemoryBarrier target = b; target.image = convertedOutput->image;
            target.oldLayout = convertedOutput->initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
            target.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            target.srcAccessMask = convertedOutput->initialized ? VK_ACCESS_TRANSFER_READ_BIT : 0;
            target.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &target);
            VkImageBlit region{}; region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.srcOffsets[1] = region.dstOffsets[1] = {int32_t(layout.ow), int32_t(layout.oh), 1};
            vkCmdBlitImage(cb, output, VK_IMAGE_LAYOUT_GENERAL, convertedOutput->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region, VK_FILTER_NEAREST);
            target.oldLayout = target.newLayout; target.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; target.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &target);
            convertedOutput->initialized = true; output = convertedOutput->image;
        }
        VkBufferImageCopy copy{}; copy.bufferOffset = layout.outputOffset; copy.bufferRowLength = uint32_t(layout.outputPitch / layout.outputBytes);
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = {layout.ow, layout.oh, 1};
        vkCmdCopyImageToBuffer(cb, output, VK_IMAGE_LAYOUT_GENERAL, shared, 1, &copy);
    }
};
}
