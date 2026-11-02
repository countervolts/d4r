// Resource upload and submission helpers shared by standalone D3D12 replay tools.
std::vector<char> slurp(const std::string &p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot read " + p);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

// A resource and the state it is in (explicit transitions only).
struct Res
{
    ComPtr<ID3D12Resource> r;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT64 bytes = 0;
    ID3D12Resource *get() const
    {
        return r.Get();
    }
};
struct Device
{
    ComPtr<ID3D12Device> dev;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    UINT64 fenceValue = 0;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    std::vector<D3D12_RESOURCE_BARRIER> pending;
    void submit()
    {
        flushBarriers();
        CK(list->Close());
        ID3D12CommandList *lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        CK(queue->Signal(fence.Get(), ++fenceValue));
        CK(fence->SetEventOnCompletion(fenceValue, event));
        WaitForSingleObject(event, INFINITE);
        CK(dev->GetDeviceRemovedReason());
        CK(alloc->Reset());
        CK(list->Reset(alloc.Get(), nullptr));
    }
    void use(Res &r, D3D12_RESOURCE_STATES state)
    {
        if (r.state == state)
            return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {r.get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, r.state, state};
        pending.push_back(b);
        r.state = state;
    }
    void uavBarrier()
    {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        pending.push_back(b);
    }
    void flushBarriers()
    {
        if (!pending.empty())
            list->ResourceBarrier(UINT(pending.size()), pending.data());
        pending.clear();
    }
    Res buffer(UINT64 bytes, D3D12_HEAP_TYPE heap = D3D12_HEAP_TYPE_DEFAULT)
    {
        Res res;
        res.bytes = std::max<UINT64>((bytes + 255) / 256 * 256, 256);
        D3D12_HEAP_PROPERTIES hp{heap};
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = res.bytes;
        d.Height = d.DepthOrArraySize = d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        d.Flags =
            heap == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        res.state = heap == D3D12_HEAP_TYPE_UPLOAD     ? D3D12_RESOURCE_STATE_GENERIC_READ
                    : heap == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                       : D3D12_RESOURCE_STATE_COMMON;
        CK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, res.state, nullptr, IID_PPV_ARGS(&res.r)));
        return res;
    }
    // committed resources start zeroed: the history needs no clear
    Res texture(UINT w, UINT h, DXGI_FORMAT format)
    {
        Res res;
        res.format = format;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = w;
        d.Height = h;
        d.DepthOrArraySize = d.MipLevels = 1;
        d.Format = format;
        d.SampleDesc.Count = 1;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        res.state = D3D12_RESOURCE_STATE_COMMON;
        CK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, res.state, nullptr, IID_PPV_ARGS(&res.r)));
        return res;
    }
};
