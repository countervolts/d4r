// Shared device code for native tails of the rrlite texture kernels (DLSS 310.7.0, preset M).
// These are linked into ZLUDA's own compile of a hand-edited PTX kernel (D4R_ZLUDA_EXTRA_BC), so
// only plain clang builtins are used (compiled with -nogpuinc -nogpulib).
#pragma once
#include <stdint.h>
#include "../common/wmma_layout.h"
#if defined(D4R_TEX_FP8) && D4R_WMMA_LAYOUT != 12 && defined(__HIP_DEVICE_COMPILE__)
#error "D4R_TEX_FP8 needs the gfx12 WMMA layout"
#endif

#pragma clang fp contract(off)

#define DEV extern "C" __attribute__((device))
// Kernels that ZLUDA compiles in strict-FP mode (constrained intrinsics) cannot take inlined FP code from
// here (the AMDGPU backend fails on some constrained operations): build them with -DD4R_SUST_OUTLINE,
// which keeps the FP bodies of the surface stores in noinline leaf functions (no calls, no stack).
#if defined(D4R_SUST_OUTLINE) || defined(D4R_ACCURACY)
#define FP_BODY __attribute__((device, noinline)) static
#else
#define FP_BODY __attribute__((device, always_inline)) static inline
#endif
#define KERNEL extern "C" __attribute__((global))

typedef _Float16 half_t;
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t uint2_t __attribute__((ext_vector_type(2)));
typedef uint16_t u16x2 __attribute__((ext_vector_type(2)));
typedef int16_t i16x2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) uint8_t lds_u8;

__attribute__((device)) static inline uint32_t lane_id()
{
    return __builtin_amdgcn_mbcnt_lo(~0u, 0u);
}

// ZLUDA's e4m3x4_to_scaled_f16x2x2 (zluda_ptx_impl.cpp): e4m3 bytes -> f16 pairs holding value * 2^-8,
// {bytes 0,1} and {bytes 2,3}.
__attribute__((device)) static inline void e4m3x4_scaled(uint32_t packed, uint32_t& low, uint32_t& high)
{
    uint32_t magnitude = packed & 0x7f7f7f7fu;
    magnitude |= (magnitude + 0x01010101u) & 0x80808080u;
    uint32_t sign = packed & 0x80808080u;
    low = (__builtin_amdgcn_perm(0u, magnitude, 0x0c010c00u) << 7) | __builtin_amdgcn_perm(0u, sign, 0x010c000cu);
    high = (__builtin_amdgcn_perm(0u, magnitude, 0x0c030c02u) << 7) | __builtin_amdgcn_perm(0u, sign, 0x030c020cu);
}

// A k16 WMMA operand from 16 consecutive e4m3 bytes of NVIDIA's k order (4 dwords): register j holds
// k 4j, 4j+1 and register 4 + j holds k 4j+2, 4j+3 -- ZLUDA's slot order (e4m3_a_k16_fragment).
__attribute__((device)) static inline u8v operand16(u4v d)
{
    u8v v;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        uint32_t lo, hi;
        e4m3x4_scaled(d[j], lo, hi);
        v[j] = lo;
        v[4 + j] = hi;
    }
    return v;
}

__attribute__((device)) static inline f8v wmma(u8v a, u8v b, f8v c)
{
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b), c);
}

// ---------------------------------------------------------------- gfx12 m16n8k32 e4m3 steps
// On gfx12 the GEMM tails do a k32 step as ZLUDA's gfx12 lowering of NVIDIA's m16n8k32 e4m3 MMA does, so they
// stay bit-identical to ZLUDA's compile of the kernel they replace: a lane's k16 operand holds this half's
// e4m3 k 8h .. 8h + 7 (ZLUDA's gfx12 slot order), either widened as above with C * 2^-16 inside the f32
// WMMAs (d4r_wmma12_*), or with D4R_TEX_FP8 (ZLUDA's D4R_ZLUDA_WMMA_FP8_NATIVE) as the bytes of a native FP8
// WMMA with no scaling (d4r_wmma12f8_*). (The gfx11 tails use operand16 and wmma above.)
#if D4R_WMMA_LAYOUT == 12
// 8 e4m3 bytes (k in order) widened to value * 2^-8 f16 pairs, k in order
__attribute__((device)) static inline u4v widen8(uint2_t d)
{
    uint32_t a0, a1, b0, b1;
    e4m3x4_scaled(d.x, a0, a1);
    e4m3x4_scaled(d.y, b0, b1);
    return (u4v){a0, a1, b0, b1};
}
#ifdef D4R_TEX_FP8
typedef uint2_t kop;
typedef u4v kslot; // weight image slot: the 16 e4m3 bytes of one k16 operand row
__attribute__((device)) static inline kop kop_from8(uint2_t d)
{
    return d;
}
__attribute__((device)) static inline kslot kslot_from16(u4v d)
{
    return d;
}
#else
typedef u4v kop;
typedef u8v kslot; // weight image slot: the 16 widened values of one k16 operand row, k in order
__attribute__((device)) static inline kop kop_from8(uint2_t d)
{
    return widen8(d);
}
__attribute__((device)) static inline kslot kslot_from16(u4v d)
{
    const u4v lo = widen8((uint2_t){d[0], d[1]}), hi = widen8((uint2_t){d[2], d[3]});
    return (u8v){lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
}
#endif
// this half's k16 operand from 16 consecutive e4m3 bytes of NVIDIA's k order
__attribute__((device)) static inline kop kop_from16(u4v d)
{
    return kop_from8(lane_id() >= 16 ? (uint2_t){d[2], d[3]} : (uint2_t){d[0], d[1]});
}
// this half's operand from slot idx of a weight image
__attribute__((device)) static inline kop kop_image(const kslot* img, int idx)
{
    return ((const kop*)img)[2 * idx + (lane_id() >> 4)];
}
// f32 C (true scale) + one k32 step
__attribute__((device)) static inline f8v k32_e4m3(f8v c, const kop& a0, const kop& b0, const kop& a1, const kop& b1)
{
#if defined(D4R_TEX_FP8) && defined(__GFX12__)
    typedef int i2v __attribute__((ext_vector_type(2)));
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(i2v, a0), __builtin_bit_cast(i2v, b0), c);
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(i2v, a1), __builtin_bit_cast(i2v, b1), c);
#elif defined(D4R_TEX_FP8)
    // layout shim, like ZLUDA's: each FP8 WMMA as the scaled widening (exact powers of two)
    c = wm_mma_zluda(widen8(a0), widen8(b0), c * 0x1p-16f) * 0x1p16f;
    return wm_mma_zluda(widen8(a1), widen8(b1), c * 0x1p-16f) * 0x1p16f;
#else
    c = c * 0x1p-16f;
    c = wm_mma_zluda(a0, b0, c);
    c = wm_mma_zluda(a1, b1, c);
    return c * 0x1p16f;
#endif
}
#endif

// RDNA4 native FP8: the tails encode with the hardware conversion. e4m3x4_f32 gives the codes of four f32 values
// that hold f16 values already clamped to the finite e4m3 range (the instruction encodes overflow as NaN), in the
// bytes of one word. With the clamp it equals e4m3x2 for every non-NaN f16 (all 65536 patterns checked on an
// RX 9070 XT). -DD4R_TEX_SOFT_E4M3 keeps e4m3x2.
#if defined(D4R_TEX_FP8) && defined(__GFX12__) && !defined(D4R_TEX_SOFT_E4M3)
#define D4R_TEX_HW_E4M3 1
__attribute__((device)) static inline uint32_t e4m3x4_f32(float a, float b, float c, float d)
{
    const uint32_t w = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    return (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(c, d, (int)w, true);
}
#endif

// cvt.rn.f16x2.e4m3x2 for the kernels whose PTX make_ptx.py points here (dec0's tail on the native-FP8 build):
// two e4m3 codes -> an f16 pair. The hardware conversion equals ZLUDA's e4m3_to_f16_bits for every code except
// the NaN codes 0x7f / 0xff (a different NaN), which the head never stages: it clamps to +-448 before encoding.
DEV __attribute__((always_inline)) uint32_t d4r_dec_e4m3x2(uint16_t codes)
{
#ifdef D4R_TEX_HW_E4M3
    typedef float dec_f2v __attribute__((ext_vector_type(2)));
    typedef _Float16 dec_h2v __attribute__((ext_vector_type(2)));
    const dec_f2v f = __builtin_amdgcn_cvt_pk_f32_fp8((int)codes, false);
    return __builtin_bit_cast(uint32_t, (dec_h2v){(_Float16)f[0], (_Float16)f[1]});
#else
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        const uint32_t code = (codes >> (8 * i)) & 0xffu, magnitude = code & 0x7fu;
        const uint32_t normal = (magnitude << 7) + 0x2000u;
        const uint32_t subnormal = __builtin_bit_cast(uint16_t, (_Float16)((float)(magnitude & 7u) * 0x1p-9f));
        uint32_t half = magnitude >= 8u ? normal : subnormal;
        half = magnitude == 0x7fu ? 0x7fffu : half;
        out |= (((code & 0x80u) << 8) | half) << (16 * i);
    }
    return out;
#endif
}

// ZLUDA's f16x2_to_e4m3x2_satfinite_bits: RNE satfinite e4m3 of both halves, low half -> low byte.
__attribute__((device)) static inline uint32_t e4m3x2(uint32_t bits)
{
    u16x2 half = __builtin_bit_cast(u16x2, bits);
    u16x2 magnitude = half & (uint16_t)0x7fffu;
    u16x2 normal = (magnitude + ((magnitude >> (uint16_t)7) & (uint16_t)1) + (uint16_t)0xe03fu) >> (uint16_t)7;
    u16x2 exponent = magnitude >> (uint16_t)10;
    u16x2 clamped = __builtin_elementwise_max(__builtin_elementwise_min(exponent, (u16x2)8), (u16x2)1);
    u16x2 shift = (uint16_t)16 - clamped;
    u16x2 significand = (magnitude & (uint16_t)0x3ffu) |
                        (__builtin_elementwise_min(magnitude, (u16x2)0x400) & (uint16_t)0x400u);
    u16x2 subnormal = (significand + (((u16x2)1 << (shift - (uint16_t)1)) - (uint16_t)1) +
                       ((significand >> shift) & (uint16_t)1)) >> shift;
    u16x2 below = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)(magnitude - (uint16_t)0x2400u)) >> (int16_t)15));
    u16x2 code = (subnormal & below) | (normal & ~below);
    code = __builtin_elementwise_min(code, (u16x2)0x7e);
    u16x2 nan = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)((uint16_t)0x7c00u - magnitude)) >> (int16_t)15));
    code |= nan & (uint16_t)0x7fu;
    code |= (half >> (uint16_t)8) & (uint16_t)0x80u;
    return uint32_t(code.x) | (uint32_t(code.y) << 8);
}

__attribute__((device)) static inline uint32_t pack2(half_t a, half_t b)
{
    return __builtin_bit_cast(uint32_t, (hv2){a, b});
}

// value from the lane with the same index in the other half of the wave
__attribute__((device)) static inline uint32_t other_half(uint32_t v)
{
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
}

// ---------------------------------------------------------------- surface stores
typedef int32_t v2i __attribute__((ext_vector_type(2)));
typedef int16_t v2s __attribute__((ext_vector_type(2)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef int32_t v4i_t __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(4))) unsigned int tsharp_t;
extern "C" __attribute__((device)) int __ockl_image_channel_data_type_2D(tsharp_t* image);
extern "C" __attribute__((device)) int __ockl_image_channel_order_2D(tsharp_t* image);
extern "C" __attribute__((device)) void __ockl_image_store_2D(tsharp_t* image, v2i coord, v4f value);
extern "C" __attribute__((device)) void __zluda_ptx_impl_surfobj_b_2d_v2_b16_zero(uint64_t surface, v2i coord, v2s value);

// ZLUDA's surface_half_bits_to_float (exact, independent of the f16 denormal mode)
__attribute__((device)) static inline float half_bits_to_float(uint32_t bits)
{
    uint32_t sign = (bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x3ffu;
    if (exponent == 0)
    {
        float magnitude = float(mantissa) * 0x1p-24f;
        return __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, magnitude) | sign);
    }
    if (exponent == 0x1f)
        return __builtin_bit_cast(float, sign | 0x7f800000u | (mantissa << 13));
    return __builtin_bit_cast(float, sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

FP_BODY void sust_rg16f_body(tsharp_t* image, int32_t px, int32_t y, uint16_t a, uint16_t b)
{
    __ockl_image_store_2D(image, (v2i){px, y}, (v4f){half_bits_to_float(a), half_bits_to_float(b), 0.0f, 0.0f});
}

// sust.b.2d.v2.b16.zero: a whole RG16F pixel is one image store of the decoded halves (what ZLUDA's
// surface_store_raw_2d does for that format); anything else goes through ZLUDA's generic helper.
__attribute__((device, noinline)) static void sust_v2b16_generic(uint64_t surface, int32_t x, int32_t y, uint16_t a, uint16_t b)
{
    __zluda_ptx_impl_surfobj_b_2d_v2_b16_zero(surface, (v2i){x, y}, (v2s){(int16_t)a, (int16_t)b});
}

DEV __attribute__((always_inline)) void d4r_sust_v2b16(uint64_t surface, int32_t x, int32_t y, uint16_t a, uint16_t b)
{
    tsharp_t* image = (tsharp_t*)surface;
    if (__ockl_image_channel_data_type_2D(image) == 14 && __ockl_image_channel_order_2D(image) == 3 && (x & 3) == 0)
    {
        if (x >= 0)
            sust_rg16f_body(image, x >> 2, y, a, b);
    }
    else
        sust_v2b16_generic(surface, x, y, a, b);
}


// f32 comparisons on the bits (no fcmp: kernels ZLUDA compiles in strict-FP mode cannot lower
// constrained fcmp); same results as the float compares, -0.0 included
__attribute__((device)) static inline bool f_isnan(uint32_t u) { return (u & 0x7fffffffu) > 0x7f800000u; }
__attribute__((device)) static inline bool f_lt0(uint32_t u) { return (int32_t)u < 0 && (u & 0x7fffffffu) != 0 && !f_isnan(u); }
// value > 1.0f (value not NaN)
__attribute__((device)) static inline bool f_gt1(uint32_t u) { return (int32_t)u > 0x3f800000; }
// value < -1.0f (value not NaN)
__attribute__((device)) static inline bool f_ltm1(uint32_t u) { return (int32_t)u < 0 && (u & 0x7fffffffu) > 0x3f800000u; }

// ZLUDA's surface_formatted_store_bits (zluda_ptx_impl.cpp), inlined: normalized CUDA formats convert
// the f32 value, everything else stores the bits unchanged.
__attribute__((device)) static inline uint32_t formatted_store_bits(uint32_t bits, uint32_t cuda_format, int channel)
{
    if (cuda_format == 80)
    {
        uint32_t u = bits;
        if (f_isnan(u) || f_lt0(u))
            u = 0u;
        if (f_gt1(u))
            u = 0x3f800000u;
        const float value = __builtin_bit_cast(float, u);
        const float scale = channel == 3 ? 3.0f : 1023.0f;
        // float(int(floor(x))) == floor(x) here (0 <= x <= 1023.5): no int <-> float conversions, which
        // strict-FP kernels cannot lower as constrained signed conversions
        return __builtin_bit_cast(uint32_t, __builtin_floorf(value * scale + 0.5f) / scale);
    }
    float scale = 0.0f;
    bool signed_normalized = false;
    if (cuda_format >= 192 && cuda_format <= 194)
        scale = 255.0f;
    else if (cuda_format >= 195 && cuda_format <= 197)
        scale = 65535.0f;
    else if (cuda_format >= 198 && cuda_format <= 200)
    {
        scale = 127.0f;
        signed_normalized = true;
    }
    else if (cuda_format >= 201 && cuda_format <= 203)
    {
        scale = 32767.0f;
        signed_normalized = true;
    }
    if (!(cuda_format >= 192 && cuda_format <= 203))
        return bits;
    uint32_t u = bits;
    if (f_isnan(u))
        u = 0u;
    if (signed_normalized ? f_ltm1(u) : f_lt0(u))
        u = signed_normalized ? 0xbf800000u : 0u;
    if (f_gt1(u))
        u = 0x3f800000u;
    const float scaled = __builtin_bit_cast(float, u) * scale;
    const uint32_t su = __builtin_bit_cast(uint32_t, scaled);
    int32_t integer = f_lt0(su) ? -int32_t(__builtin_floorf(-scaled + 0.5f)) : int32_t(__builtin_floorf(scaled + 0.5f));
    return uint32_t(integer);
}

// sust.p.2d.v4.b32.zero: ZLUDA stores the raw bits unless the original CUDA format was normalized
// (surface_formatted_store_bits; format word after the image and sampler descriptors).
FP_BODY void sust_p_body(uint64_t surface, int32_t x, int32_t y, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    const uint32_t format = ((const __attribute__((address_space(4))) uint32_t*)surface)[20];
    __ockl_image_store_2D((tsharp_t*)surface, (v2i){x, y},
                          (v4f){__builtin_bit_cast(float, formatted_store_bits(a, format, 0)),
                                __builtin_bit_cast(float, formatted_store_bits(b, format, 1)),
                                __builtin_bit_cast(float, formatted_store_bits(c, format, 2)),
                                __builtin_bit_cast(float, formatted_store_bits(d, format, 3))});
}

// d4r output redirect (wine_nvcuda_bridge.c d4rSetArrayRedirect): a tagged row pitch at byte 84 of the
// ZLUDA surface object (zero padding otherwise) sends stores of that surface to the linear memory whose
// address is at byte 88, rows `pitch` bytes apart. Out-of-range coordinates are dropped as by .zero stores.
extern "C" __attribute__((device)) int __ockl_image_width_2D(tsharp_t* image);
extern "C" __attribute__((device)) int __ockl_image_height_2D(tsharp_t* image);
__attribute__((device)) static inline uint32_t redirect_pitch(uint64_t surface)
{
    // 'R2' tag in the high half (other handles may have data there), row pitch / 8 in the low half
    const uint32_t word = ((const __attribute__((address_space(4))) uint32_t*)surface)[21];
    return (word >> 16) == 0x5232u ? (word & 0xffffu) << 3 : 0u;
}
// D4R_AS1_STORES: redirect stores through address-space-1 pointers (global_store instead of flat_store; the
// redirect target is linear device memory)
#ifdef D4R_AS1_STORES
#define D4R_REDIRECT_PTR(T, address) ((__attribute__((address_space(1))) T*)(uintptr_t)(address))
#else
#define D4R_REDIRECT_PTR(T, address) ((T*)(address))
#endif
__attribute__((device)) static inline uint8_t* redirect_row(uint64_t surface, int32_t y)
{
    const uint64_t base = *(const __attribute__((address_space(4))) uint64_t*)(surface + 88);
    return (uint8_t*)(base + (uint64_t)(uint32_t)y * redirect_pitch(surface));
}
__attribute__((device)) static inline bool redirect_inside(uint64_t surface, int32_t x, int32_t y)
{
    tsharp_t* image = (tsharp_t*)surface;
    return x >= 0 && y >= 0 && x < __ockl_image_width_2D(image) && y < __ockl_image_height_2D(image);
}

DEV __attribute__((always_inline)) void d4r_sust_p_v4b32(uint64_t surface, int32_t x, int32_t y, uint32_t a, uint32_t b,
                                                        uint32_t c, uint32_t d)
{
    const uint32_t format0 = ((const __attribute__((address_space(4))) uint32_t*)surface)[20];
    if (redirect_pitch(surface) != 0 && format0 != 80 && !(format0 >= 192 && format0 <= 203))
    {
        // RGBA16F: the f32 values as halves, rounded toward zero like the image store's format conversion
        if (redirect_inside(surface, x, y))
        {
            const uint32_t lo = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(__builtin_bit_cast(float, a), __builtin_bit_cast(float, b)));
            const uint32_t hi = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(__builtin_bit_cast(float, c), __builtin_bit_cast(float, d)));
            *D4R_REDIRECT_PTR(uint2_t, redirect_row(surface, y) + 8 * x) = (uint2_t){lo, hi};
        }
        return;
    }
    // formatted_store_bits keeps the bits unless the format is normalized: one uniform test per store
    const uint32_t format = ((const __attribute__((address_space(4))) uint32_t*)surface)[20];
    if (format != 80 && !(format >= 192 && format <= 203))
        __ockl_image_store_2D((tsharp_t*)surface, (v2i){x, y},
                              (v4f){__builtin_bit_cast(float, a), __builtin_bit_cast(float, b), __builtin_bit_cast(float, c),
                                    __builtin_bit_cast(float, d)});
    else
        sust_p_body(surface, x, y, a, b, c, d);
}

// ---------------------------------------------------------------- raw 32-bit surface stores
extern "C" __attribute__((device)) void __zluda_ptx_impl_surfobj_b_2d_b32_zero(uint64_t surface, v2i coord, uint32_t value);

// exact int -> float for |v| < 2^24 without a signed conversion (see formatted_store_bits)
__attribute__((device)) static inline float i2f(int32_t v)
{
    return v < 0 ? -(float)(uint32_t)(-v) : (float)(uint32_t)v;
}

// ZLUDA's surface_decode_channel (bits of one memory channel -> the float4 lane value the store converts back)
__attribute__((device)) static inline float decode_channel(uint32_t bits, int data_type)
{
    switch (data_type)
    {
    case 0:
        return i2f(__builtin_elementwise_max(__builtin_elementwise_min(int32_t(int8_t(bits)), 127), -127)) / 127.0f;
    case 1:
        return i2f(__builtin_elementwise_max(__builtin_elementwise_min(int32_t(int16_t(bits)), 32767), -32767)) / 32767.0f;
    case 2:
        return (float)(bits & 0xffu) / 255.0f;
    case 3:
        return (float)(bits & 0xffffu) / 65535.0f;
    case 8:
        return __builtin_bit_cast(float, int32_t(int8_t(bits)));
    case 9:
        return __builtin_bit_cast(float, int32_t(int16_t(bits)));
    case 14:
        return half_bits_to_float(bits);
    default:
        return __builtin_bit_cast(float, bits);
    }
}

// out of line: inlining ZLUDA's generic raw store into large kernels crashes AMDGPU instruction
// selection (see zluda_ptx_impl.cpp); it makes no calls, so the wrapper needs no stack
__attribute__((device, noinline)) static void sust_b32_generic(uint64_t surface, int32_t x, int32_t y, uint32_t data)
{
    __zluda_ptx_impl_surfobj_b_2d_b32_zero(surface, (v2i){x, y}, data);
}

// sust.b.2d.b32.zero (x is a byte offset): whole 4-byte pixels and pairs of 2-byte pixels in R/RG/RGBA
// channel order need no read-modify-write (ZLUDA's surface_store_raw_2d, same conversions); anything
// else goes through ZLUDA's generic helper.
FP_BODY void sust_b32_fast(tsharp_t* image, int32_t x, int32_t y, uint32_t data, int dt, int cb, int pb)
{
    if (pb == 4)
    {
        v4f v = {0.0f, 0.0f, 0.0f, 0.0f};
        if (cb == 4)
            v[0] = decode_channel(data, dt);
        else if (cb == 2)
        {
            v[0] = decode_channel(data & 0xffffu, dt);
            v[1] = decode_channel(data >> 16, dt);
        }
        else
        {
            v[0] = decode_channel(data & 0xffu, dt);
            v[1] = decode_channel((data >> 8) & 0xffu, dt);
            v[2] = decode_channel((data >> 16) & 0xffu, dt);
            v[3] = decode_channel(data >> 24, dt);
        }
        __ockl_image_store_2D(image, (v2i){x >> 2, y}, v);
    }
    else
    {
        v4f v0 = {0.0f, 0.0f, 0.0f, 0.0f}, v1 = {0.0f, 0.0f, 0.0f, 0.0f};
        if (cb == 2)
        {
            v0[0] = decode_channel(data & 0xffffu, dt);
            v1[0] = decode_channel(data >> 16, dt);
        }
        else
        {
            v0[0] = decode_channel(data & 0xffu, dt);
            v0[1] = decode_channel((data >> 8) & 0xffu, dt);
            v1[0] = decode_channel((data >> 16) & 0xffu, dt);
            v1[1] = decode_channel(data >> 24, dt);
        }
        __ockl_image_store_2D(image, (v2i){x >> 1, y}, v0);
        __ockl_image_store_2D(image, (v2i){(x >> 1) + 1, y}, v1);
    }
}

DEV __attribute__((always_inline)) void d4r_sust_b32(uint64_t surface, int32_t x, int32_t y, uint32_t data)
{
    tsharp_t* image = (tsharp_t*)surface;
    if (redirect_pitch(surface) != 0)
    {
        // raw bits at byte offset x (RGBA16F rows are 8 bytes per pixel)
        if (x >= 0 && (x & 3) == 0 && redirect_inside(surface, x >> 3, y))
            *D4R_REDIRECT_PTR(uint32_t, redirect_row(surface, y) + x) = data;
        return;
    }
    const int dt = __ockl_image_channel_data_type_2D(image), order = __ockl_image_channel_order_2D(image);
    const int cb = (dt == 0 || dt == 2 || dt == 8 || dt == 11) ? 1 : (dt == 1 || dt == 3 || dt == 9 || dt == 12 || dt == 14) ? 2
                   : (dt == 10 || dt == 13 || dt == 15) ? 4 : 0;
    const int n = order == 1 ? 1 : order == 3 ? 2 : order == 8 ? 4 : 0; // R, RG, RGBA (identity lane order)
    const int pb = cb * n;
    if ((pb == 4 && (x & 3) == 0) || (pb == 2 && (x & 1) == 0))
    {
        if (x >= 0)
            sust_b32_fast(image, x, y, data, dt, cb, pb);
    }
    else
        sust_b32_generic(surface, x, y, data);
}
