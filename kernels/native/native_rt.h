// Runtime for natively compiled DLSS texture kernels: PTX semantics (ptx_rt.h) plus surface stores
// (tex/tex_common.h, with ZLUDA's generic raw-store fallback ported here, since no ZLUDA bitcode is linked).
#pragma once
#include "ptx_rt.h"
#include "../tex/tex_common.h"

extern "C" __attribute__((device)) v4f __ockl_image_load_2D(tsharp_t* image, v2i coord);

namespace zsurf {
// zluda_ptx_impl.cpp surface_* helpers (raw sust.b stores), verbatim semantics
RT int channel_bytes(int dt)
{
    switch (dt)
    {
    case 0: case 2: case 8: case 11: return 1;
    case 1: case 3: case 9: case 12: case 14: return 2;
    case 10: case 13: case 15: return 4;
    default: return 0;
    }
}
RT int channel_count(int order)
{
    switch (order)
    {
    case 3: case 4: case 5: return 2;
    case 6: case 12: return 3;
    case 7: case 8: case 9: case 10: case 11: case 13: case 14: case 15: return 4;
    default: return 1;
    }
}
RT int memory_channel_lane(int order, int channel)
{
    switch (order)
    {
    case 0: return 3;
    case 5: return channel == 0 ? 0 : 3;
    case 9: case 15: return channel == 0 ? 2 : channel == 2 ? 0 : channel;
    case 10: return channel == 0 ? 3 : channel - 1;
    case 11: return 3 - channel;
    default: return channel;
    }
}
RT uint32_t float_to_half_bits(float value)
{
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    uint32_t sign = (bits >> 16) & 0x8000u, exponent = (bits >> 23) & 0xffu, mantissa = bits & 0x7fffffu;
    if (exponent == 0xff)
        return sign | 0x7c00u | (mantissa != 0 ? 0x200u | (mantissa >> 13) : 0u);
    int half_exponent = int(exponent) - 112;
    if (half_exponent >= 31)
        return sign | 0x7c00u;
    uint32_t shift = 13;
    if (half_exponent <= 0)
    {
        if (half_exponent < -10)
            return sign;
        mantissa |= 0x800000u;
        shift = uint32_t(14 - half_exponent);
        half_exponent = 0;
    }
    uint32_t half_mantissa = mantissa >> shift, remainder = mantissa & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
    uint32_t result = sign | (uint32_t(half_exponent) << 10);
    result += half_mantissa & (half_exponent == 0 ? 0x7ffu : 0x3ffu);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u)))
        result++;
    return result;
}
RT int32_t clampi(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }
RT float decode(uint32_t bits, int dt)
{
    switch (dt)
    {
    case 0: return float(clampi(int32_t(int8_t(bits)), -127, 127)) / 127.0f;
    case 1: return float(clampi(int32_t(int16_t(bits)), -32767, 32767)) / 32767.0f;
    case 2: return float(int32_t(bits & 0xffu)) / 255.0f;
    case 3: return float(int32_t(bits & 0xffffu)) / 65535.0f;
    case 8: return __builtin_bit_cast(float, int32_t(int8_t(bits)));
    case 9: return __builtin_bit_cast(float, int32_t(int16_t(bits)));
    case 14: return half_bits_to_float(bits);
    default: return __builtin_bit_cast(float, bits);
    }
}
RT uint32_t encode(float value, int dt)
{
    switch (dt)
    {
    case 0: return uint32_t(clampi(int32_t(__builtin_floorf(value * 127.0f + 0.5f)), -127, 127)) & 0xffu;
    case 1: return uint32_t(clampi(int32_t(__builtin_floorf(value * 32767.0f + 0.5f)), -32767, 32767)) & 0xffffu;
    case 2: return uint32_t(clampi(int32_t(__builtin_floorf(value * 255.0f + 0.5f)), 0, 255));
    case 3: return uint32_t(clampi(int32_t(__builtin_floorf(value * 65535.0f + 0.5f)), 0, 65535));
    case 14: return float_to_half_bits(value);
    default: return __builtin_bit_cast(uint32_t, value);
    }
}
RT int lane_channel(int order, int lane, int channels)
{
    switch (order)
    {
    case 0: return lane == 3 ? 0 : -1;
    case 5: return lane == 0 ? 0 : (lane == 3 ? 1 : -1);
    case 9: case 15: return lane == 0 ? 2 : (lane == 2 ? 0 : lane);
    case 10: return lane == 3 ? 0 : lane + 1;
    case 11: return 3 - lane;
    default: return lane < channels ? lane : -1;
    }
}
RT uint32_t sel_channel(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, int ch)
{
    return ch == 0 ? c0 : (ch == 1 ? c1 : (ch == 2 ? c2 : c3));
}
RT float sel_lane(v4f v, int lane) { return lane == 0 ? v.x : (lane == 1 ? v.y : (lane == 2 ? v.z : v.w)); }
RT uint32_t merge_channel(uint32_t bits, int channel, int cb, uint32_t data, int byte_offset, int byte_count)
{
    for (int b = 0; b < 4; b++)
    {
        int relative = channel * cb + b - byte_offset;
        if (b < cb && relative >= 0 && relative < byte_count)
        {
            uint32_t byte = (data >> (8 * relative)) & 0xffu;
            bits = (bits & ~(0xffu << (8 * b))) | (byte << (8 * b));
        }
    }
    return bits;
}
RT float lane_value(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, int order, int lane, int channels, int dt)
{
    int ch = lane_channel(order, lane, channels);
    return ch < 0 ? 0.0f : decode(sel_channel(c0, c1, c2, c3, ch), dt);
}
__attribute__((device, noinline)) static void store_raw_2d(tsharp_t* image, int32_t cx, int32_t cy, uint32_t data, int byte_count)
{
    int dt = __ockl_image_channel_data_type_2D(image), order = __ockl_image_channel_order_2D(image);
    int cb = channel_bytes(dt);
    if (cb == 0 || cx < 0)
        return;
    int channels = channel_count(order), pixel_bytes = cb * channels;
    for (int consumed = 0; consumed < byte_count;)
    {
        int byte_x = cx + consumed, pixel_x = byte_x / pixel_bytes, byte_offset = byte_x - pixel_x * pixel_bytes;
        int chunk = pixel_bytes - byte_offset;
        if (chunk > byte_count - consumed)
            chunk = byte_count - consumed;
        uint32_t fragment = data >> (8 * consumed);
        uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        if (byte_offset != 0 || chunk < pixel_bytes)
        {
            v4f loaded = __ockl_image_load_2D(image, (v2i){pixel_x, cy});
            c0 = encode(sel_lane(loaded, memory_channel_lane(order, 0)), dt);
            c1 = encode(sel_lane(loaded, memory_channel_lane(order, 1)), dt);
            c2 = encode(sel_lane(loaded, memory_channel_lane(order, 2)), dt);
            c3 = encode(sel_lane(loaded, memory_channel_lane(order, 3)), dt);
        }
        c0 = merge_channel(c0, 0, cb, fragment, byte_offset, chunk);
        c1 = merge_channel(c1, 1, cb, fragment, byte_offset, chunk);
        c2 = merge_channel(c2, 2, cb, fragment, byte_offset, chunk);
        c3 = merge_channel(c3, 3, cb, fragment, byte_offset, chunk);
        __ockl_image_store_2D(image, (v2i){pixel_x, cy},
                              (v4f){lane_value(c0, c1, c2, c3, order, 0, channels, dt), lane_value(c0, c1, c2, c3, order, 1, channels, dt),
                                    lane_value(c0, c1, c2, c3, order, 2, channels, dt), lane_value(c0, c1, c2, c3, order, 3, channels, dt)});
        consumed += chunk;
    }
}
} // namespace zsurf

extern "C" __attribute__((device)) void __zluda_ptx_impl_surfobj_b_2d_b32_zero(uint64_t surface, v2i coord, uint32_t value)
{
    zsurf::store_raw_2d((tsharp_t*)(uintptr_t)surface, coord.x, coord.y, value, 4);
}
extern "C" __attribute__((device)) void __zluda_ptx_impl_surfobj_b_2d_v2_b16_zero(uint64_t surface, v2i coord, v2s value)
{
    zsurf::store_raw_2d((tsharp_t*)(uintptr_t)surface, coord.x, coord.y, uint32_t(uint16_t(value.x)) | (uint32_t(uint16_t(value.y)) << 16), 4);
}

// A sample that every lane takes with the same handle, coordinates and lod (e.g. a 1x1 exposure texture at (0.5,
// 0.5)): only the first active lane samples and the result is broadcast with readfirstlane. Same values, but the
// texture unit processes one lane instead of the whole wave.
RT f4 tex_level_uniform(uint64_t t, float x, float y, float lod)
{
    f4 r = (f4){0.0f, 0.0f, 0.0f, 0.0f};
    if (wave_lane() == (uint32_t)__builtin_ctzll(__builtin_amdgcn_read_exec()))
        r = tex_level(t, x, y, lod);
    return (f4){asf(__builtin_amdgcn_readfirstlane(asu(r[0]))), asf(__builtin_amdgcn_readfirstlane(asu(r[1]))),
                asf(__builtin_amdgcn_readfirstlane(asu(r[2]))), asf(__builtin_amdgcn_readfirstlane(asu(r[3])))};
}

