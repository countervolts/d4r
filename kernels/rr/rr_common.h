#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

typedef _Float16 half_t;
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));

struct [[gnu::packed, gnu::aligned(8)]] DldnParams {
    uint8_t payload[512];
};

struct [[gnu::packed, gnu::aligned(8)]] HKPNTailParams {
    uint64_t in_radiance_ptr;   // Смещение 0: тензор радиации
    uint64_t out_surface_obj;   // Смещение 8: дескриптор выходной поверхности
    uint32_t width;             // Смещение 16: ширина
    uint32_t height;            // Смещение 20: высота
    uint8_t  rest[456];
};