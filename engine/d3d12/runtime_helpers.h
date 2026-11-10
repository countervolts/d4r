// Internal D3D12 resource and pipeline helpers shared by M/L and RR.
// Include inside an anonymous namespace.
void check(HRESULT result, const char *operation)
{
    if (FAILED(result))
    {
        char text[32];
        std::snprintf(text, sizeof(text), "0x%08lx", static_cast<unsigned long>(result));
        throw std::runtime_error(std::string(operation) + ": HRESULT " + text);
    }
}
#define DXCHECK(call) check((call), #call)

constexpr UINT AgsSpace = 0x7fff0ade;
std::vector<char> slurp(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot read " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}
ComPtr<ID3D12Resource> buffer(ID3D12Device *device, UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES properties{heap};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::max<UINT64>((bytes + 255) / 256 * 256, 256);
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags =
        heap == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> resource;
    DXCHECK(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                            IID_PPV_ARGS(&resource)));
    return resource;
}
// committed resources start zeroed: the history needs no clear (the first frame resets anyway)
ComPtr<ID3D12Resource> texture(ID3D12Device *device, UINT width, UINT height, DXGI_FORMAT format)
{
    D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> resource;
    DXCHECK(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc,
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                            IID_PPV_ARGS(&resource)));
    return resource;
}
D3D12_RESOURCE_BARRIER transition(ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    return barrier;
}
DXGI_FORMAT readFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24G8_TYPELESS:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_TYPELESS:
        return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return DXGI_FORMAT_R11G11B10_FLOAT;
    default:
        return format;
    }
}
DXGI_FORMAT viewFormat(const FrameTexture &t)
{
    return t.format != DXGI_FORMAT_UNKNOWN ? t.format : readFormat(t.resource->GetDesc().Format);
}
// A Ray Reconstruction output the output stage can write: RGBA16F (NGX's own), or another colour format Super Resolution
// writes too, when the device stores it through a typed UAV (the stage's float4 store converts). Resident Evil Requiem
// hands RR an R9G9B9E5 output (AMD's driver stores it typed on RDNA3 and RDNA4); it was refused before.
bool rrOutputFormat(const FrameTexture &t)
{
    const DXGI_FORMAT f = viewFormat(t);
    switch (f)
    {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return true;
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        break;
    default:
        return false;
    }
    ComPtr<ID3D12Device> device;
    if (FAILED(t.resource->GetDevice(IID_PPV_ARGS(&device))))
        return false;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{f};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
           (support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) &&
           (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
}

// ---------------------------------------------------------------- pipelines
// A stage's binding N is register tN or uN, slot N of its one descriptor table; a texture's sampler is sN.
enum class Kind
{
    None,
    Tex,
    Raw,
    RwTex,
    RwStruct,
    RwRaw
};
struct Binding
{
    Kind kind = Kind::None;
    bool linear = false;
    UINT stride = 0;
};
const Binding T{Kind::Tex}, TL{Kind::Tex, true}, R{Kind::Raw}, UT{Kind::RwTex}, UR{Kind::RwRaw};
Binding US(UINT stride)
{
    return {Kind::RwStruct, false, stride};
}
struct Pipeline
{
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> state;
    const std::vector<Binding> *bindings = nullptr;
    void create(ID3D12Device *device, const std::string &path, const std::vector<Binding> &table)
    {
        bindings = &table;
        const auto dxil = slurp(path);
        std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
        std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
        for (UINT i = 0; i < table.size(); ++i)
        {
            const Kind k = table[i].kind;
            if (k == Kind::None)
                continue;
            const bool srv = k == Kind::Tex || k == Kind::Raw;
            ranges.push_back({srv ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, i, 0, i});
            if (k == Kind::Tex)
            {
                D3D12_STATIC_SAMPLER_DESC s{};
                s.Filter = table[i].linear ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
                s.AddressU = s.AddressV = s.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                s.MaxLOD = D3D12_FLOAT32_MAX;
                s.ShaderRegister = i;
                s.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                s.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
                samplers.push_back(s);
            }
        }
        D3D12_ROOT_PARAMETER p[2]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[0].DescriptorTable = {UINT(ranges.size()), ranges.data()};
        p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // the AGS intrinsic UAV (unused by the plain stages)
        p[1].Descriptor = {0, AgsSpace};
        p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC desc{2, p, UINT(samplers.size()), samplers.data(), D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob, error;
        if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error)))
            throw std::runtime_error(std::string("root signature: ") +
                                     (error ? static_cast<const char *>(error->GetBufferPointer()) : "?"));
        DXCHECK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
        pso.CS = {dxil.data(), dxil.size()};
        if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&state))))
            throw std::runtime_error("cannot create the pipeline of " + path +
                                     " (AMD WMMA through AGS needs an AMD driver)");
    }
    explicit operator bool() const
    {
        return state != nullptr;
    }
};
