// Native runtime for PTX kernels translated to HIP by ptx2hip.py (no ZLUDA compile involved).
// Every helper reproduces ZLUDA's lowering of the PTX instruction (ptx/src/pass/llvm/emit.rs and
// ptx/lib/zluda_ptx_impl.cpp, patches/zluda), so a translated kernel computes the same values as
// ZLUDA's own compile of NVIDIA's PTX. Texture and surface objects are ZLUDA handles: the address of
// a HIP image descriptor (8 dwords), its sampler (4 dwords at +48 bytes) and the CUDA format (+80).
// Compile flags (see build_native.sh): -ffp-contract=off, f32 denormals flushed (ZLUDA's
// "denormal-fp-math-f32"="preserve-sign"), f16 denormals kept.
#pragma once
#include <stdint.h>

#pragma clang fp contract(off)

#define RT static __attribute__((device, always_inline)) inline
#define AS4 __attribute__((address_space(4)))
#define AS3 __attribute__((address_space(3)))
#define AS1 __attribute__((address_space(1)))
#define AS5 __attribute__((address_space(5)))

typedef _Float16 h1;
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef float f2 __attribute__((ext_vector_type(2)));
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i8v __attribute__((ext_vector_type(8)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef uint32_t u4 __attribute__((ext_vector_type(4)));
typedef uint32_t u2v __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------- bit casts
RT float asf(uint32_t u) { return __builtin_bit_cast(float, u); }
RT uint32_t asu(float f) { return __builtin_bit_cast(uint32_t, f); }
RT h1 ash(uint16_t u) { return __builtin_bit_cast(h1, u); }
RT uint16_t ashu(h1 h) { return __builtin_bit_cast(uint16_t, h); }
RT h2 ash2(uint32_t u) { return __builtin_bit_cast(h2, u); }
RT uint32_t ash2u(h2 h) { return __builtin_bit_cast(uint32_t, h); }
RT uint32_t pack16(uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); }

// ---------------------------------------------------------------- f32
RT float canon(float x) { return __builtin_canonicalizef(x); }
RT float fmaxn(float a, float b) { return __builtin_fmaximum_numf(a, b); }
RT float fminn(float a, float b) { return __builtin_fminimum_numf(a, b); }
RT float sat(float x) { return fminn(fmaxn(x, 0.0f), 1.0f); }
RT float sat_ftz(float x) { return sat(canon(x)); }                     // cvt.ftz.sat.f32.f32
RT float abs_ftz(float x) { return __builtin_fabsf(canon(x)); }
RT float neg_ftz(float x) { return -canon(x); }
RT float rcp_a(float x) { return __builtin_amdgcn_rcpf(x); }             // rcp.approx(.ftz).f32
RT float div_a(float a, float b) { return a * __builtin_amdgcn_rcpf(b); } // div.approx.ftz.f32 (fdiv arcp afn)
RT float ex2_a(float x) { return __builtin_amdgcn_exp2f(x); }
RT float lg2_a(float x) { return __builtin_amdgcn_logf(x); }
RT float rsq_a(float x) { return __builtin_amdgcn_rsqf(x); }
// cvt.rzi.s32.f32: saturating, NaN -> 0 (llvm.fptosi.sat, v_cvt_i32_f32); spelled out, a plain C cast is UB out of range
// Branch-free: the operand is clamped into int range first (fmaxf(NaN, lo) = lo), so the cast is always defined and
// the compiler can use selects (the guarded form compiled to a branch per conversion, splitting basic blocks).
RT int32_t f2i_rz(float x)
{
#ifdef D4R_NATIVE_F2I
    // AMD V_CVT_I32_F32 truncates, saturates out-of-range inputs and maps NaN to zero.
    // Explicit instruction avoids redundant clamps/selects emitted for a C saturating cast.
    int32_t r;
    __asm__("v_cvt_i32_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
#else
    const float c = __builtin_fminf(__builtin_fmaxf(x, -2147483648.0f), 2147483520.0f);
    int32_t r = (int32_t)c;
    r = x >= 2147483648.0f ? 2147483647 : r;
    return x != x ? 0 : r;
#endif
}

// cvt.rzi.u32.f32: ZLUDA emits a plain fptoui (frozen) and relies on V_CVT_U32_F32 (truncates; negative and NaN -> 0,
// too large -> 0xffffffff), so the native helper is that instruction
RT uint32_t f2u_rz(float x)
{
    uint32_t r;
    __asm__("v_cvt_u32_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

// ---------------------------------------------------------------- f16 (denormals kept)
// cvt.rn.f16.f32: the f32 value rounded to f16. The empty asm keeps LLVM from folding the conversion into the
// f32 operation that produced the value (fptrunc(fma(fpext ...)) -> v_fma_mixlo/mixhi_f16), which rounds the exact
// result to f16 once instead of to f32 and then to f16: on an RX 7700 XT that differed from the two-step result for
// 1 in ~29,000 random f16 operand triples, and in the hiluma output kernel it left single 1-ulp seeds that the
// temporal history then spread (ZLUDA keeps both roundings).
RT h1 f2h_rn(float x)
{
    __asm__ volatile("" : "+v"(x));
    return (h1)x;
}
RT h1 hfma(h1 a, h1 b, h1 c) { return __builtin_fmaf16(a, b, c); }
RT h2 h2fma(h2 a, h2 b, h2 c) { return __builtin_elementwise_fma(a, b, c); }
RT h1 hmaxn(h1 a, h1 b) { return __builtin_fmaximum_numf16(a, b); }
RT h1 hminn(h1 a, h1 b) { return __builtin_fminimum_numf16(a, b); }
RT h1 hsat(h1 x) { return hminn(hmaxn(x, (h1)0.0f), (h1)1.0f); }
RT h2 h2sat(h2 x) { return (h2){hsat(x.x), hsat(x.y)}; }
RT h2 h2abs(h2 x) { return __builtin_elementwise_abs(x); }
// ex2.approx.f16x2: v_exp_f32 per half (zluda_ptx_impl.cpp)
RT h2 h2ex2(h2 a) { return (h2){(h1)__builtin_amdgcn_exp2f((float)a.x), (h1)__builtin_amdgcn_exp2f((float)a.y)}; }
// cvt.rni/rmi.s32.f16: llvm.fptosi.sat.i32.f16 after the rounding (|f16| < 2^31, NaN -> 0)
RT int32_t h2i_sat(h1 h)
{
#ifdef D4R_NATIVE_H2I
    return f2i_rz((float)h);
#else
    return h != h ? 0 : (int32_t)(float)h;
#endif
}
RT h1 h_roundeven(h1 h) { return __builtin_elementwise_roundeven(h); }
RT h1 h_floor(h1 h) { return __builtin_elementwise_floor(h); }

// ---------------------------------------------------------------- threads (wave64: two CUDA warps per wave)
RT uint32_t wave_lane() { return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)); }
RT uint32_t warp_base()
{
#if __AMDGCN_WAVEFRONT_SIZE == 64
    return wave_lane() & 32u;
#else
    return 0u;
#endif
}
RT uint32_t laneid() { return wave_lane() & 31u; }
// shfl.sync.bfly.b32 d|p, a, delta, opts, mask (ZLUDA's SHFL_SYNC_IMPL)
RT uint32_t shfl_bfly(uint32_t input, int32_t delta, uint32_t opts, bool* in_bounds)
{
    int32_t section_mask = (opts >> 8) & 31, warp_end = opts & 31;
    delta &= 31;
    int32_t self = (int32_t)laneid();
    int32_t subsection = section_mask & self;
    int32_t subsection_end = subsection | (~section_mask & warp_end);
    int32_t idx = self ^ delta;
    bool oob = idx > subsection_end;
    if (oob)
        idx = self;
    if (in_bounds)
        *in_bounds = !oob;
    return (uint32_t)__builtin_amdgcn_ds_bpermute((int32_t)(warp_base() | (uint32_t)idx) << 2, (int32_t)input);
}
RT uint32_t bfi32(uint32_t insert, uint32_t base, uint32_t pos_32, uint32_t len_32)
{
    uint32_t pos = pos_32 & 0xffu, len = len_32 & 0xffu;
    if (pos >= 32)
        return base;
    uint32_t mask = len >= 32 ? (0xffffffffu << pos) : (((len >= 32 ? 0u : (1u << len)) - 1u) << pos);
    return (~mask & base) | (mask & (insert << pos));
}

// ---------------------------------------------------------------- textures (ZLUDA handles)
__attribute__((device)) static uint32_t d4r_null_texture[32];
RT uint64_t tex_or_null(uint64_t t) { return t != 0 ? t : (uint64_t)(uintptr_t)d4r_null_texture; }
RT i8v tdesc(uint64_t t) { return *(const AS4 i8v*)(uintptr_t)tex_or_null(t); }
RT i4v sdesc(uint64_t t) { return *(const AS4 i4v*)(uintptr_t)(tex_or_null(t) + 48); }

__attribute__((device)) f4 llvm_image_load_mip_2d(uint32_t, int32_t, int32_t, int32_t, i8v, int, int)
    __asm("llvm.amdgcn.image.load.mip.2d.v4f32.i32");
__attribute__((device)) f4 llvm_image_sample_l_2d(uint32_t, float, float, float, i8v, i4v, bool, int, int)
    __asm("llvm.amdgcn.image.sample.l.2d.v4f32.f32");
extern "C" __attribute__((device)) f4 __ockl_image_gather4r_2D(const AS4 uint32_t* image, const AS4 uint32_t* sampler, f2 coord);

// tex.base.2d.v4.f32.s32: texel load, mip 0
RT f4 tex_fetch(uint64_t t, int32_t x, int32_t y) { return llvm_image_load_mip_2d(0xf, x, y, 0, tdesc(t), 0, 0); }
// tex.level.2d.v4.f32.f32
RT f4 tex_level(uint64_t t, float x, float y, float lod) { return llvm_image_sample_l_2d(0xf, x, y, lod, tdesc(t), sdesc(t), false, 0, 0); }
// tld4.r.2d.v4.f32.f32
RT f4 tex_gather_r(uint64_t t, float x, float y)
{
    const AS4 uint32_t* image = (const AS4 uint32_t*)(uintptr_t)tex_or_null(t);
    return __ockl_image_gather4r_2D(image, image + 12, (f2){x, y});
}

// bar.sync 0 (HIP's __syncthreads)
RT void bar_sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
