// Opt-in, fence-retired snapshots of a live K evaluation. Included by the runtime
// after its resource helpers; no readback or filesystem polling unless enabled.
struct CaptureReadback {
    ComPtr<ID3D12Resource> staging;
    std::string path;
    UINT64 bytes = 0, rowBytes = 0;
    UINT rows = 1;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    CaptureReadback(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                    ID3D12Resource* source, D3D12_RESOURCE_STATES state, std::string file)
        : path(std::move(file)) {
        const auto desc = source->GetDesc();
        const bool image = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        if (image) device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &bytes);
        else bytes = rowBytes = desc.Width;
        staging = buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto before = transition(source, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &before);
        if (image) {
            D3D12_TEXTURE_COPY_LOCATION src{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION dst{staging.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            dst.PlacedFootprint = footprint;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        } else list->CopyBufferRegion(staging.Get(), 0, source, 0, bytes);
        auto after = transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        list->ResourceBarrier(1, &after);
    }
    void save() const {
        void* mapped = nullptr; D3D12_RANGE range{0, SIZE_T(bytes)};
        DXCHECK(staging->Map(0, &range, &mapped));
        std::ofstream out(path, std::ios::binary);
        const auto* data = static_cast<const char*>(mapped);
        for (UINT y = 0; y < rows; ++y)
            out.write(data + footprint.Offset + UINT64(y) * (rows == 1 ? 0 : footprint.Footprint.RowPitch), std::streamsize(rowBytes));
        D3D12_RANGE none{0, 0}; staging->Unmap(0, &none);
        if (!out) throw std::runtime_error("cannot save D3D12 capture: " + path);
    }
};

struct FrameCapture {
    std::vector<CaptureReadback> data;
    ComPtr<ID3D12Fence> fence;
    uint64_t value = 0;
    bool pending = true;
    void add(ID3D12Device* device, ID3D12GraphicsCommandList* list, ID3D12Resource* source,
             D3D12_RESOURCE_STATES state, const std::string& path) {
        data.emplace_back(device, list, source, state, path);
    }
};
