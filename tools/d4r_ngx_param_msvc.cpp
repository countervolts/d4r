// Microsoft-ABI access to NVSDK_NGX_Parameter objects for MinGW-built code.
// NVSDK_NGX_Parameter is a C++ interface with overloaded virtual Set/Get
// methods; MSVC orders overloaded virtuals differently from the Itanium ABI,
// so calls must be compiled with clang-cl. Build freestanding (/Zl /GS-).
using D4rNgxResult = unsigned int;
struct ID3D11Resource;
struct ID3D12Resource;

struct D4rNgxParameterAbi
{
    virtual void Set(const char* name, unsigned long long value) = 0;
    virtual void Set(const char* name, float value) = 0;
    virtual void Set(const char* name, double value) = 0;
    virtual void Set(const char* name, unsigned int value) = 0;
    virtual void Set(const char* name, int value) = 0;
    virtual void Set(const char* name, ID3D11Resource* value) = 0;
    virtual void Set(const char* name, ID3D12Resource* value) = 0;
    virtual void Set(const char* name, void* value) = 0;

    virtual D4rNgxResult Get(const char* name, unsigned long long* value) const = 0;
    virtual D4rNgxResult Get(const char* name, float* value) const = 0;
    virtual D4rNgxResult Get(const char* name, double* value) const = 0;
    virtual D4rNgxResult Get(const char* name, unsigned int* value) const = 0;
    virtual D4rNgxResult Get(const char* name, int* value) const = 0;
    virtual D4rNgxResult Get(const char* name, ID3D11Resource** value) const = 0;
    virtual D4rNgxResult Get(const char* name, ID3D12Resource** value) const = 0;
    virtual D4rNgxResult Get(const char* name, void** value) const = 0;
    virtual void Reset() = 0;
};

static D4rNgxParameterAbi* parameters(void* raw)
{
    return static_cast<D4rNgxParameterAbi*>(raw);
}

extern "C" void d4r_ngx_set_ull(void* raw, const char* name, unsigned long long value) { parameters(raw)->Set(name, value); }
extern "C" void d4r_ngx_set_float(void* raw, const char* name, float value) { parameters(raw)->Set(name, value); }
extern "C" void d4r_ngx_set_uint(void* raw, const char* name, unsigned int value) { parameters(raw)->Set(name, value); }
extern "C" void d4r_ngx_set_int(void* raw, const char* name, int value) { parameters(raw)->Set(name, value); }
extern "C" void d4r_ngx_set_void(void* raw, const char* name, void* value) { parameters(raw)->Set(name, value); }

extern "C" D4rNgxResult d4r_ngx_get_ull(void* raw, const char* name, unsigned long long* value) { return parameters(raw)->Get(name, value); }
extern "C" D4rNgxResult d4r_ngx_get_float(void* raw, const char* name, float* value) { return parameters(raw)->Get(name, value); }
extern "C" D4rNgxResult d4r_ngx_get_uint(void* raw, const char* name, unsigned int* value) { return parameters(raw)->Get(name, value); }
extern "C" D4rNgxResult d4r_ngx_get_int(void* raw, const char* name, int* value) { return parameters(raw)->Get(name, value); }
extern "C" D4rNgxResult d4r_ngx_get_void(void* raw, const char* name, void** value) { return parameters(raw)->Get(name, value); }
extern "C" D4rNgxResult d4r_ngx_get_d3d12_resource(void* raw, const char* name, ID3D12Resource** value) { return parameters(raw)->Get(name, value); }
extern "C" void d4r_ngx_set_d3d12_resource(void* raw, const char* name, ID3D12Resource* value) { parameters(raw)->Set(name, value); }

// A complete NVSDK_NGX_Parameter implementation handed out by the shim's
// Allocate/Get/GetCapabilityParameters. It stores D3D12 resources (which the
// CUDA-path parameter objects of the official core do not) and converts
// between numeric types on Get, like the official implementations.
inline void* operator new(decltype(sizeof(0)), void* place) noexcept { return place; }
extern "C" void* d4r_host_alloc(unsigned long long size); // zero-filled, provided by the shim
extern "C" void d4r_host_free(void* pointer);

namespace
{
constexpr D4rNgxResult kSuccess = 0x1;
constexpr D4rNgxResult kUnsupportedParameter = 0xBAD00010;

enum ValueType : unsigned
{
    TypeULL = 1,
    TypeFloat,
    TypeDouble,
    TypeUInt,
    TypeInt,
    TypePointer
};

struct Entry
{
    char name[96];
    unsigned type;
    union
    {
        unsigned long long u;
        long long i;
        double d;
        void* p;
    } value;
};

bool same_name(const char* a, const char* b)
{
    for (; *a != 0 && *a == *b; ++a, ++b)
    {
    }
    return *a == *b;
}

class Parameters final : public D4rNgxParameterAbi
{
public:
    void Set(const char* name, unsigned long long value) override { store(name, TypeULL)->value.u = value; }
    void Set(const char* name, float value) override { store(name, TypeFloat)->value.d = value; }
    void Set(const char* name, double value) override { store(name, TypeDouble)->value.d = value; }
    void Set(const char* name, unsigned int value) override { store(name, TypeUInt)->value.u = value; }
    void Set(const char* name, int value) override { store(name, TypeInt)->value.i = value; }
    void Set(const char* name, ID3D11Resource* value) override { store(name, TypePointer)->value.p = value; }
    void Set(const char* name, ID3D12Resource* value) override { store(name, TypePointer)->value.p = value; }
    void Set(const char* name, void* value) override { store(name, TypePointer)->value.p = value; }

    D4rNgxResult Get(const char* name, unsigned long long* value) const override { return number(name, value); }
    D4rNgxResult Get(const char* name, float* value) const override { return number(name, value); }
    D4rNgxResult Get(const char* name, double* value) const override { return number(name, value); }
    D4rNgxResult Get(const char* name, unsigned int* value) const override { return number(name, value); }
    D4rNgxResult Get(const char* name, int* value) const override { return number(name, value); }
    D4rNgxResult Get(const char* name, ID3D11Resource** value) const override { return pointer(name, value); }
    D4rNgxResult Get(const char* name, ID3D12Resource** value) const override { return pointer(name, value); }
    D4rNgxResult Get(const char* name, void** value) const override { return pointer(name, value); }
    void Reset() override { count_ = 0; }

    void CopyTo(D4rNgxParameterAbi* dst, void (*logger)(const char* name, int type, unsigned long long val)) const
    {
        for (unsigned index = 0; index < count_; ++index)
        {
            const Entry& e = entries_[index];
            if (logger != nullptr)
                logger(e.name, static_cast<int>(e.type), e.value.u);

            switch (e.type)
            {
            case TypeULL:
                dst->Set(e.name, e.value.u);
                break;
            case TypeFloat:
                dst->Set(e.name, static_cast<float>(e.value.d));
                break;
            case TypeDouble:
                dst->Set(e.name, e.value.d);
                break;
            case TypeUInt:
                dst->Set(e.name, static_cast<unsigned int>(e.value.u));
                break;
            case TypeInt:
                dst->Set(e.name, static_cast<int>(e.value.i));
                break;
            case TypePointer:
                // Указатели на ресурсы D3D12 в CUDA-параметры не копируем
                break;
            }
        }
    }

private:
    const Entry* find(const char* name) const
    {
        for (unsigned index = 0; index < count_; ++index)
            if (same_name(entries_[index].name, name))
                return &entries_[index];
        return nullptr;
    }

    Entry* store(const char* name, unsigned type)
    {
        Entry* entry = const_cast<Entry*>(find(name));
        if (entry == nullptr)
        {
            // Past capacity the last slot is reused rather than failing.
            entry = &entries_[count_ < kCapacity ? count_++ : kCapacity - 1];
            unsigned length = 0;
            for (; name[length] != 0 && length + 1 < sizeof(entry->name); ++length)
                entry->name[length] = name[length];
            entry->name[length] = 0;
        }
        entry->type = type;
        entry->value.u = 0;
        return entry;
    }

    template <typename T> D4rNgxResult number(const char* name, T* value) const
    {
        const Entry* entry = find(name);
        if (entry == nullptr || value == nullptr)
            return kUnsupportedParameter;
        switch (entry->type)
        {
        case TypeFloat:
        case TypeDouble:
            *value = static_cast<T>(entry->value.d);
            break;
        case TypeInt:
            *value = static_cast<T>(entry->value.i);
            break;
        case TypePointer:
            *value = static_cast<T>(reinterpret_cast<unsigned long long>(entry->value.p));
            break;
        default:
            *value = static_cast<T>(entry->value.u);
            break;
        }
        return kSuccess;
    }

    template <typename T> D4rNgxResult pointer(const char* name, T** value) const
    {
        const Entry* entry = find(name);
        if (entry == nullptr || value == nullptr)
            return kUnsupportedParameter;
        *value = entry->type == TypePointer ? static_cast<T*>(entry->value.p)
                                            : reinterpret_cast<T*>(entry->value.u);
        return kSuccess;
    }

    static constexpr unsigned kCapacity = 256;
    unsigned count_ = 0;
    Entry entries_[kCapacity];
};
} // namespace

extern "C" void* d4r_ngx_parameters_create()
{
    void* memory = d4r_host_alloc(sizeof(Parameters));
    return memory != nullptr ? static_cast<D4rNgxParameterAbi*>(new (memory) Parameters()) : nullptr;
}

extern "C" void d4r_ngx_parameters_destroy(void* parameters)
{
    d4r_host_free(parameters);
}

extern "C" void d4r_ngx_parameters_copy_all(void* src_raw, void* dst_raw, void (*logger)(const char*, int, unsigned long long))
{
    if (src_raw == nullptr || dst_raw == nullptr)
        return;
    auto* src = static_cast<Parameters*>(src_raw);
    auto* dst = static_cast<D4rNgxParameterAbi*>(dst_raw);
    src->CopyTo(dst, logger);
}
