#include "../../kernels/common/hip_device_minimal.h"
#include "pixel_format.h"

namespace {
using namespace d4r::pixel;
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned;
using u64 = unsigned long long;
__device__ inline float as_float(u32 bits) { return __builtin_bit_cast(float, bits); }
__device__ inline u32 as_uint(float value) { return __builtin_bit_cast(u32, value); }
__device__ inline float from_half(u16 bits) {
    u32 sign = u32(bits & 0x8000u) << 16, exp = (bits >> 10) & 31, m = bits & 1023;
    if (!exp) return as_float(as_uint(float(m) * 0x1p-24f) | sign);
    return as_float(sign | (exp == 31 ? 0x7f800000u : (exp + 112u) << 23) | (m << 13));
}
__device__ inline u16 to_half(float value) {
    u32 bits = as_uint(value), sign = (bits >> 16) & 0x8000u, exponent = (bits >> 23) & 255u;
    u32 mantissa = bits & 0x7fffffu;
    if (exponent == 255) return u16(sign | 0x7c00u | (mantissa ? 0x200u : 0));
    int e = int(exponent) - 112;
    if (e >= 31) return u16(sign | 0x7c00u);
    unsigned shift = 13;
    if (e <= 0) {
        if (e < -10) return u16(sign);
        mantissa |= 0x800000u; shift = unsigned(14 - e); e = 0;
    }
    u32 kept = mantissa >> shift, remainder = mantissa & ((1u << shift) - 1), midpoint = 1u << (shift - 1);
    u32 result = sign | (u32(e) << 10);
    result += kept & (e == 0 ? 0x7ffu : 0x3ffu);
    return u16(result + (remainder > midpoint || (remainder == midpoint && (kept & 1))));
}
__device__ inline u32 quantize(float value, unsigned maximum) {
    if (!(value > 0)) return 0;
    if (value >= 1) return maximum;
    float scaled = value * float(maximum);
    u32 integer = u32(scaled); float remainder = scaled - float(integer);
    return integer + (remainder > .5f || (remainder == .5f && (integer & 1)));
}
__device__ inline u32 half_to_unsigned(u16 h, unsigned mantissa_bits) {
    if ((h & 0x7c00u) == 0x7c00u && (h & 1023u)) return (31u << mantissa_bits) | (1u << (mantissa_bits - 1));
    if (h & 0x8000u) return 0;
    // D3D floating-point format narrowing truncates. RNE here differs from
    // the driver's typed UAV store and can turn finite HDR values into Inf.
    return h >> (10 - mantissa_bits);
}
__device__ inline unsigned pixel_bytes(Storage fmt) {
    switch (fmt) {
        case rgba32f: return 16;
        case rgba16f: case rg32f: return 8;
        case r16f: case r16unorm: return 2;
        default: return 4;
    }
}
__device__ inline void decode(const u8* p, Storage fmt, float* value) {
    const auto* halves = reinterpret_cast<const u16*>(p);
    const auto* singles = reinterpret_cast<const float*>(p);
    value[0] = value[1] = value[2] = 0; value[3] = 1;
    switch (fmt) {
        case rgba16f: for (unsigned i = 0; i < 4; ++i) value[i] = from_half(halves[i]); break;
        case rgba32f: for (unsigned i = 0; i < 4; ++i) value[i] = singles[i]; break;
        case rgba8: case bgra8:
            for (unsigned i = 0; i < 4; ++i) value[i] = float(p[i]) / 255.f;
            if (fmt == bgra8) { float swap = value[0]; value[0] = value[2]; value[2] = swap; } break;
        case r11g11b10f: {
            u32 packed = *reinterpret_cast<const u32*>(p);
            value[0] = from_half(u16((packed & 2047) << 4));
            value[1] = from_half(u16(((packed >> 11) & 2047) << 4));
            value[2] = from_half(u16((packed >> 22) << 5)); break;
        }
        case rgb10a2: {
            u32 packed = *reinterpret_cast<const u32*>(p);
            for (unsigned i = 0; i < 3; ++i) value[i] = float((packed >> (10 * i)) & 1023) / 1023.f;
            value[3] = float(packed >> 30) / 3.f; break;
        }
        case rg16f: value[0] = from_half(halves[0]); value[1] = from_half(halves[1]); break;
        case rg32f: value[0] = singles[0]; value[1] = singles[1]; break;
        case rg16snorm:
            for (unsigned i = 0; i < 2; ++i) {
                int v = reinterpret_cast<const short*>(p)[i];
                value[i] = v == -32768 ? -1.f : float(v) / 32767.f;
            } break;
        case rg16unorm: for (unsigned i = 0; i < 2; ++i) value[i] = float(halves[i]) / 65535.f; break;
        case r16f: value[0] = from_half(halves[0]); break;
        case r32f: value[0] = singles[0]; break;
        case depth24: value[0] = float(*reinterpret_cast<const u32*>(p) & 0xffffffu) / 16777215.f; break;
        case r16unorm: value[0] = float(halves[0]) / 65535.f; break;
        default: break; // Host rejects formats which have no tested decoder.
    }
}
__device__ inline void encode(u8* p, Storage fmt, const float* value) {
    auto* halves = reinterpret_cast<u16*>(p);
    auto* singles = reinterpret_cast<float*>(p);
    switch (fmt) {
        case rgba16f: for (unsigned i = 0; i < 4; ++i) halves[i] = to_half(value[i]); break;
        case rgba32f: for (unsigned i = 0; i < 4; ++i) singles[i] = value[i]; break;
        case rgba8: case bgra8:
            for (unsigned i = 0; i < 4; ++i) p[i] = u8(quantize(value[i], 255));
            if (fmt == bgra8) { u8 swap = p[0]; p[0] = p[2]; p[2] = swap; } break;
        case r11g11b10f:
            *reinterpret_cast<u32*>(p) = half_to_unsigned(to_half(value[0]), 6) |
                (half_to_unsigned(to_half(value[1]), 6) << 11) | (half_to_unsigned(to_half(value[2]), 5) << 22); break;
        case rgb10a2:
            *reinterpret_cast<u32*>(p) = quantize(value[0], 1023) | (quantize(value[1], 1023) << 10) |
                (quantize(value[2], 1023) << 20) | (quantize(value[3], 3) << 30); break;
        default: break; // Only color/output formats are encoded in the runtime.
    }
}
}

extern "C" __global__ void d4r_pixel_decode(const u8* source, u64 source_pitch, u8* dest, u64 dest_pitch,
    unsigned width, unsigned height, unsigned format, unsigned plane) {
    unsigned i = blockIdx.x * 256u + threadIdx.x;
    if (u64(i) >= u64(width) * height) return;
    unsigned x = i % width, y = i / width;
    float value[4]; decode(source + u64(y) * source_pitch + u64(x) * pixel_bytes(Storage(format)), Storage(format), value);
    if (plane == depth || plane == exposure) reinterpret_cast<float*>(dest + u64(y) * dest_pitch)[x] = value[0];
    else {
        unsigned channels = plane == motion ? 2 : 4;
        auto* halves = reinterpret_cast<u16*>(dest + u64(y) * dest_pitch) + x * channels;
        for (unsigned c = 0; c < channels; ++c) halves[c] = to_half(value[c]);
    }
}

extern "C" __global__ void d4r_pixel_encode(const u8* source, u64 source_pitch, u8* dest, u64 dest_pitch,
    unsigned width, unsigned height, unsigned format, unsigned plane) {
    unsigned i = blockIdx.x * 256u + threadIdx.x;
    if (u64(i) >= u64(width) * height) return;
    unsigned x = i % width, y = i / width;
    const auto* halves = reinterpret_cast<const u16*>(source + u64(y) * source_pitch) + x * 4;
    float value[4]; for (unsigned c = 0; c < 4; ++c) value[c] = from_half(halves[c]);
    encode(dest + u64(y) * dest_pitch + u64(x) * pixel_bytes(Storage(format)), Storage(format), value);
}
