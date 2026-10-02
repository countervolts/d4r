#pragma once
#include "d3d12_external.h"
#include <cstring>

namespace d4r::win {
// Diagnostic timestamps, read only after the copy slot's existing completion
// fence. No extra queue/CPU wait, and no image readback. The interval across
// the external fence includes HIP execution and scheduling, not just kernels.
class BoundaryTiming {
    ComPtr<ID3D12QueryHeap> queries_;
    ComPtr<ID3D12Resource> readback_;
    uint64_t ticket_ = 0;
public:
    explicit BoundaryTiming(ID3D12Device* device) {
        D3D12_QUERY_HEAP_DESC query{};
        query.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; query.Count = 4;
        dx(device->CreateQueryHeap(&query, IID_PPV_ARGS(queries_.GetAddressOf())), "Boundary timestamp heap");
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = 4 * sizeof(uint64_t);
        desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
        desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        dx(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(readback_.GetAddressOf())), "Boundary timestamp readback");
    }
    void input_begin(ID3D12GraphicsCommandList* list) { list->EndQuery(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0); }
    void input_end(ID3D12GraphicsCommandList* list) { list->EndQuery(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1); }
    void output_begin(ID3D12GraphicsCommandList* list) { list->EndQuery(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2); }
    void output_end(ID3D12GraphicsCommandList* list) {
        list->EndQuery(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
        list->ResolveQueryData(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, readback_.Get(), 0);
    }
    void submitted(uint64_t ticket) { ticket_ = ticket; }
    // Caller has observed the slot's completion fence, including ResolveQueryData.
    void report(ID3D12CommandQueue* queue, const void* feature) {
        if (!ticket_) return;
        uint64_t ticks[4]{}, frequency = 0;
        D3D12_RANGE read{0, sizeof(ticks)}, written{0, 0};
        void* mapped = nullptr;
        dx(readback_->Map(0, &read, &mapped), "Map completed boundary timestamps");
        std::memcpy(ticks, mapped, sizeof(ticks)); readback_->Unmap(0, &written);
        dx(queue->GetTimestampFrequency(&frequency), "Boundary timestamp frequency");
        if (!frequency || !ticks[0] || ticks[0] > ticks[1] || ticks[1] > ticks[2] || ticks[2] > ticks[3])
            throw std::runtime_error("Invalid completed boundary timestamps");
        const double ms = 1000. / double(frequency);
        std::printf("D4R_GPU_BOUNDARY feature=%p ticket=%llu input_copy_ms=%.6f external_span_ms=%.6f output_copy_ms=%.6f total_ms=%.6f input_start_ticks=%llu input_end_ticks=%llu output_start_ticks=%llu output_end_ticks=%llu frequency=%llu diagnostics_cpu_bytes=32 serializing=0\n",
            feature, (unsigned long long)ticket_, double(ticks[1] - ticks[0]) * ms,
            double(ticks[2] - ticks[1]) * ms, double(ticks[3] - ticks[2]) * ms,
            double(ticks[3] - ticks[0]) * ms, (unsigned long long)ticks[0],
            (unsigned long long)ticks[1], (unsigned long long)ticks[2],
            (unsigned long long)ticks[3], (unsigned long long)frequency);
        ticket_ = 0;
    }
};
}
