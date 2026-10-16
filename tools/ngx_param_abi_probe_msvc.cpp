#include "ngx_param_abi_probe_shared.h"
inline void* operator new(__SIZE_TYPE__, void* place) noexcept { return place; }
inline void operator delete(void*, void*) noexcept {}

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

extern "C" void d4r_ngx_parameter_set_ull(
    void* rawParameters, const char* name, unsigned long long value)
{
    static_cast<D4rNgxParameterAbi*>(rawParameters)->Set(name, value);
}
extern "C" void d4r_ngx_parameter_set_float(
    void* rawParameters, const char* name, float value)
{
    static_cast<D4rNgxParameterAbi*>(rawParameters)->Set(name, value);
}
extern "C" void d4r_ngx_parameter_set_double(
    void* rawParameters, const char* name, double value)
{
    static_cast<D4rNgxParameterAbi*>(rawParameters)->Set(name, value);
}
extern "C" void d4r_ngx_parameter_set_uint(
    void* rawParameters, const char* name, unsigned int value)
{
    static_cast<D4rNgxParameterAbi*>(rawParameters)->Set(name, value);
}
extern "C" void d4r_ngx_parameter_set_int(
    void* rawParameters, const char* name, int value)
{
    static_cast<D4rNgxParameterAbi*>(rawParameters)->Set(name, value);
}
extern "C" void d4r_ngx_parameter_set_void(
    void* rawParameters, const char* name, void* value)
{
    static_cast<D4rNgxParameterAbi*>(rawParameters)->Set(name, value);
}

extern "C" D4rNgxResult d4r_ngx_parameter_get_ull(
    void* rawParameters, const char* name, unsigned long long* value)
{
    return static_cast<D4rNgxParameterAbi*>(rawParameters)->Get(name, value);
}
extern "C" D4rNgxResult d4r_ngx_parameter_get_float(
    void* rawParameters, const char* name, float* value)
{
    return static_cast<D4rNgxParameterAbi*>(rawParameters)->Get(name, value);
}
extern "C" D4rNgxResult d4r_ngx_parameter_get_double(
    void* rawParameters, const char* name, double* value)
{
    return static_cast<D4rNgxParameterAbi*>(rawParameters)->Get(name, value);
}
extern "C" D4rNgxResult d4r_ngx_parameter_get_uint(
    void* rawParameters, const char* name, unsigned int* value)
{
    return static_cast<D4rNgxParameterAbi*>(rawParameters)->Get(name, value);
}
extern "C" D4rNgxResult d4r_ngx_parameter_get_int(
    void* rawParameters, const char* name, int* value)
{
    return static_cast<D4rNgxParameterAbi*>(rawParameters)->Get(name, value);
}
extern "C" D4rNgxResult d4r_ngx_parameter_get_void(
    void* rawParameters, const char* name, void** value)
{
    return static_cast<D4rNgxParameterAbi*>(rawParameters)->Get(name, value);
}

extern "C" D4rNgxResult d4r_ngx_parameter_float_roundtrip(
    void* rawParameters, const char* name, float value, float* outValue)
{
    auto* parameters = static_cast<D4rNgxParameterAbi*>(rawParameters);
    parameters->Set(name, value);
    *outValue = 0.0f;
    return parameters->Get(name, outValue);
}

extern "C" D4rNgxResult d4r_ngx_parameter_uint_roundtrip(
    void* rawParameters, const char* name, unsigned int value, unsigned int* outValue)
{
    auto* parameters = static_cast<D4rNgxParameterAbi*>(rawParameters);
    parameters->Set(name, value);
    *outValue = 0;
    return parameters->Get(name, outValue);
}

struct D4rNgxParameterProxy final : D4rNgxParameterAbi
{
    explicit D4rNgxParameterProxy(void* rawOriginal)
        : original(static_cast<D4rNgxParameterAbi*>(rawOriginal)), recordCount(0), droppedRecords(0)
    {
    }

    void Set(const char* name, unsigned long long value) override
    {
        original->Set(name, value);
        append('S', 'q', name, 0xffffffffu);
    }
    void Set(const char* name, float value) override
    {
        original->Set(name, value);
        append('S', 'f', name, 0xffffffffu);
    }
    void Set(const char* name, double value) override
    {
        original->Set(name, value);
        append('S', 'd', name, 0xffffffffu);
    }
    void Set(const char* name, unsigned int value) override
    {
        original->Set(name, value);
        append('S', 'u', name, 0xffffffffu);
    }
    void Set(const char* name, int value) override
    {
        original->Set(name, value);
        append('S', 'i', name, 0xffffffffu);
    }
    void Set(const char* name, ID3D11Resource* value) override
    {
        original->Set(name, value);
        append('S', '1', name, 0xffffffffu);
    }
    void Set(const char* name, ID3D12Resource* value) override
    {
        original->Set(name, value);
        append('S', '2', name, 0xffffffffu);
    }
    void Set(const char* name, void* value) override
    {
        original->Set(name, value);
        append('S', 'p', name, 0xffffffffu);
    }

    D4rNgxResult Get(const char* name, unsigned long long* value) const override
    {
        const D4rNgxResult result = original->Get(name, value);
        append('G', 'q', name, result, result == 1 ? *value : 0);
        return result;
    }
    D4rNgxResult Get(const char* name, float* value) const override
    {
        const D4rNgxResult result = original->Get(name, value);
        append('G', 'f', name, result, result == 1 ? static_cast<unsigned long long>(*value) : 0);
        return result;
    }
    D4rNgxResult Get(const char* name, double* value) const override
    {
        const D4rNgxResult result = original->Get(name, value);
        append('G', 'd', name, result, result == 1 ? static_cast<unsigned long long>(*value) : 0);
        return result;
    }
    D4rNgxResult Get(const char* name, unsigned int* value) const override
    {
        const D4rNgxResult result = original->Get(name, value);
        append('G', 'u', name, result, result == 1 ? *value : 0);
        return result;
    }
    D4rNgxResult Get(const char* name, int* value) const override
    {
        const D4rNgxResult result = original->Get(name, value);
        append('G', 'i', name, result, result == 1 ? static_cast<unsigned int>(*value) : 0);
        return result;
    }
    D4rNgxResult Get(const char* name, ID3D11Resource** value) const override
    {
        const D4rNgxResult result = original->Get(name, value);
        append('G', '1', name, result, result == 1 ? reinterpret_cast<unsigned long long>(*value) : 0);
        return result;
    }
    D4rNgxResult Get(const char* name, ID3D12Resource** value) const override
    {
        const D4rNgxResult result = original->Get(name, value);
        append('G', '2', name, result, result == 1 ? reinterpret_cast<unsigned long long>(*value) : 0);
        return result;
    }
    D4rNgxResult Get(const char* name, void** value) const override
    {
        const D4rNgxResult result = original->Get(name, value);
        append('G', 'p', name, result, result == 1 ? reinterpret_cast<unsigned long long>(*value) : 0);
        return result;
    }
    void Reset() override
    {
        original->Reset();
        append('S', 'R', "Reset", 0xffffffffu);
    }

    void append(char operation, char valueType, const char* name, D4rNgxResult result,
                unsigned long long value = 0) const
    {
        if (recordCount >= 512)
        {
            ++droppedRecords;
            return;
        }
        D4rNgxParamLogRecord& record = records[recordCount++];
        record.operation = static_cast<unsigned char>(operation);
        record.valueType = static_cast<unsigned char>(valueType);
        record.reserved = 0;
        record.result = result;
        record.value = value;
        unsigned int index = 0;
        if (name != nullptr)
        {
            while (index < 63 && name[index] != '\0')
            {
                record.name[index] = name[index];
                ++index;
            }
        }
        record.name[index] = '\0';
    }

    D4rNgxParameterAbi* original;
    mutable unsigned int recordCount;
    mutable unsigned int droppedRecords;
    mutable D4rNgxParamLogRecord records[512];
};

extern "C" unsigned int d4r_ngx_parameter_proxy_size()
{
    return static_cast<unsigned int>(sizeof(D4rNgxParameterProxy));
}
extern "C" void d4r_ngx_parameter_proxy_construct(void* storage, void* original)
{
    ::new (storage) D4rNgxParameterProxy(original);
}
extern "C" unsigned int d4r_ngx_parameter_proxy_count(const void* storage)
{
    return static_cast<const D4rNgxParameterProxy*>(storage)->recordCount;
}
extern "C" unsigned int d4r_ngx_parameter_proxy_dropped(const void* storage)
{
    return static_cast<const D4rNgxParameterProxy*>(storage)->droppedRecords;
}
extern "C" int d4r_ngx_parameter_proxy_copy_record(
    const void* storage, unsigned int index, D4rNgxParamLogRecord* outRecord)
{
    const auto* proxy = static_cast<const D4rNgxParameterProxy*>(storage);
    if (index >= proxy->recordCount)
        return 0;
    *outRecord = proxy->records[index];
    return 1;
}
