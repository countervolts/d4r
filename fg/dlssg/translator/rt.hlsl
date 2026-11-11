// nvxrt T0 runtime for D3D12 (DXIL): rt.glsl's helpers with PTX semantics for the kernels glsl2hlsl.py converts, run as
// compute shaders in the game's own D3D12 command list (cubin_d3d12.cpp, the DXIL route). Contains no NVIDIA code;
// semantics follow the public PTX ISA and rt.glsl.
//
// Bindings (root signature below; descriptors indexed directly, SM 6.6):
//   b0  NvxCB: ext (x, y: GPU VA of the pointer heap's span buffer; z: an opaque 0), then the kernel's parameter block
//   u0  the pointer heap's span buffer: a CUDA pointer p is byte (p - span VA) of it (DLSS-G's buffers live there)
//   t0  the executor's info words (sampler info, per-slot image classes; rt.glsl's NvxInfo)
//   ResourceDescriptorHeap[slot] textures, [NVX_UAV_BASE + slot] surfaces; SamplerDescriptorHeap[slot] samplers,
//   [NVX_SMPN_BASE + slot] their NEAREST companions
// Compile: dxc -T cs_6_6 -E main -enable-16bit-types -HV 2021 -DNVX_BX=.. -DNVX_BY=.. -DNVX_BZ=.. (-denorm ftz)
#define NVX_UAV_BASE 65536u
#define NVX_SMPN_BASE 1024u
#define NVX_RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED), CBV(b0), UAV(u0), SRV(t0), UAV(u0, space = 2147420894)"

// ---------------------------------------------------------------- GLSL spellings
typedef float2 vec2; typedef float3 vec3; typedef float4 vec4;
typedef int2 ivec2; typedef int3 ivec3; typedef int4 ivec4;
typedef uint2 uvec2; typedef uint3 uvec3; typedef uint4 uvec4;
typedef bool2 bvec2; typedef bool3 bvec3; typedef bool4 bvec4;
typedef float16_t2 f16vec2; typedef float16_t3 f16vec3; typedef float16_t4 f16vec4;
typedef uint16_t2 u16vec2; typedef uint16_t4 u16vec4; typedef int16_t2 i16vec2;
typedef uint64_t2 u64vec2;
typedef uint uint8_t;   // DXIL has no 8-bit type: 8-bit values live in uint (masked where they are made)
typedef int int8_t;
#define uintBitsToFloat asfloat
#define floatBitsToUint asuint
#define intBitsToFloat asfloat
#define floatBitsToInt asint
#define uint16BitsToFloat16 asfloat16
#define float16BitsToUint16 asuint16
#define fract frac
#define roundEven round   // DXIL Round_ne
#define inversesqrt rsqrt
#define mix lerp
#define atan(y, x) atan2(y, x)
#define findMSB(x) int(firstbithigh(x))
#define findLSB(x) int(firstbitlow(x))
#define bitCount countbits
#define bitfieldReverse reversebits
// GLSL constructors: one argument splats (vec4(0.0)) or converts, several build the vector (the type names above stay
// usable in declarations: these macros only expand before a parenthesis)
#define NVX_ARGN(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define NVX_NARGS(...) NVX_ARGN(__VA_ARGS__, M, M, M, M, M, M, M, 1, 0)
#define NVX_CAT(a, b) NVX_CAT_(a, b)
#define NVX_CAT_(a, b) a##b
#define NVX_CTOR(T, ...) NVX_CAT(NVX_CTOR_, NVX_NARGS(__VA_ARGS__))(T, __VA_ARGS__)
#define NVX_CTOR_1(T, x) ((T)(x))
#define NVX_CTOR_M(T, ...) T(__VA_ARGS__)
#define vec2(...) NVX_CTOR(float2, __VA_ARGS__)
#define vec3(...) NVX_CTOR(float3, __VA_ARGS__)
#define vec4(...) NVX_CTOR(float4, __VA_ARGS__)
#define ivec2(...) NVX_CTOR(int2, __VA_ARGS__)
#define ivec3(...) NVX_CTOR(int3, __VA_ARGS__)
#define ivec4(...) NVX_CTOR(int4, __VA_ARGS__)
#define uvec2(...) NVX_CTOR(uint2, __VA_ARGS__)
#define uvec3(...) NVX_CTOR(uint3, __VA_ARGS__)
#define uvec4(...) NVX_CTOR(uint4, __VA_ARGS__)
#define u16vec2(...) NVX_CTOR(uint16_t2, __VA_ARGS__)
#define u16vec4(...) NVX_CTOR(uint16_t4, __VA_ARGS__)
#define u64vec2(...) NVX_CTOR(uint64_t2, __VA_ARGS__)
uint64_t packUint2x32(uint2 v) { return uint64_t(v.x) | (uint64_t(v.y) << 32); }
uint2 unpackUint2x32(uint64_t v) { return uint2(uint(v), uint(v >> 32)); }
// fused multiply-add: HLSL's mad may be either; the kernels' fma are PTX fma.rn
#define fma(a, b, c) mad(a, b, c)

// ---------------------------------------------------------------- resources
struct NvxExt { uint memLo, memHi, zero, memLast, gx, gy, gz, pad2; };   // memLast: heap bytes - 16
cbuffer NvxCB : register(b0) { NvxExt nvx_ext; uint4 nvx_pcw[256]; };
// The pointer heap: a root UAV (no bounds, but AMD's fastest loads and stores; a bounded view costs ~0.6 ms per real
// frame at 4x). Offsets are clamped to the heap (nvx_off), so a stray address stays inside it: the executor sends launches
// whose pointers lie outside the heap to the Vulkan route, the clamp only keeps a bug from reaching other GPU memory.
RWByteAddressBuffer nvx_mem : register(u0);
ByteAddressBuffer nvx_infob : register(t0);

static uint3 nvx_gid, nvx_lid, nvx_ngroups;
static const uint3 NVX_BLOCK = uint3(NVX_BX, NVX_BY, NVX_BZ);
#define gl_WorkGroupID nvx_gid
#define gl_LocalInvocationID nvx_lid
#define gl_WorkGroupSize NVX_BLOCK
#define gl_NumWorkGroups nvx_ngroups
#define gl_SubgroupInvocationID WaveGetLaneIndex()
#define gl_SubgroupSize 32u
#define gl_LocalInvocationIndex (nvx_lid.x + NVX_BX * (nvx_lid.y + NVX_BY * nvx_lid.z))
#define gl_SubgroupID (gl_LocalInvocationIndex / 32u)   // waves are consecutive in the flat group (glsl2hlsl.py)
#define gl_NumSubgroups NVX_NSUB
#define NVX_MAXC(a, b) ((a) > (b) ? (a) : (b))   // a constant expression (array sizes); max() is not

uint nvx_opaque_zero() { return nvx_ext.zero; }
uint nvx_pword(uint i) { return nvx_pcw[i >> 2][i & 3u]; }
uint nvx_p32(uint off) { return nvx_pword(off >> 2); }
uint nvx_p16(uint off) { return (nvx_pword(off >> 2) >> ((off & 2u) * 8u)) & 0xffffu; }
uint nvx_p8(uint off) { return (nvx_pword(off >> 2) >> ((off & 3u) * 8u)) & 0xffu; }
uint64_t nvx_p64(uint off) { return packUint2x32(uint2(nvx_pword(off >> 2), nvx_pword((off >> 2) + 1u))); }

// ---------------------------------------------------------------- memory through CUDA pointers
uint nvx_off(uint64_t a) { return uint(min(a - packUint2x32(uint2(nvx_ext.memLo, nvx_ext.memHi)), uint64_t(nvx_ext.memLast))); }
uint nvx_ld_NvxU8(uint64_t a) { uint o = nvx_off(a); return (nvx_mem.Load(o & ~3u) >> ((o & 3u) * 8u)) & 0xffu; }
int nvx_ld_NvxI8(uint64_t a) { uint o = nvx_off(a); return int(nvx_mem.Load(o & ~3u) << (24u - (o & 3u) * 8u)) >> 24; }
uint16_t nvx_ld_NvxU16(uint64_t a) { return nvx_mem.Load<uint16_t>(nvx_off(a)); }
u16vec2 nvx_ld_NvxU16x2(uint64_t a) { return nvx_mem.Load<u16vec2>(nvx_off(a)); }
uint nvx_ld_NvxU32(uint64_t a) { return nvx_mem.Load(nvx_off(a)); }
uint2 nvx_ld_NvxU32x2(uint64_t a) { return nvx_mem.Load2(nvx_off(a)); }
uint4 nvx_ld_NvxU32x4(uint64_t a) { return nvx_mem.Load4(nvx_off(a)); }
uint4 nvx_ld_NvxU32x4a4(uint64_t a) { return nvx_mem.Load4(nvx_off(a)); }
uint64_t nvx_ld_NvxU64(uint64_t a) { return nvx_mem.Load<uint64_t>(nvx_off(a)); }
u64vec2 nvx_ld_NvxU64x2(uint64_t a) { return nvx_mem.Load<u64vec2>(nvx_off(a)); }
int64_t nvx_ld_NvxI64A(uint64_t a) { return nvx_mem.Load<int64_t>(nvx_off(a)); }
uint64_t nvx_ld_NvxU64A(uint64_t a) { return nvx_mem.Load<uint64_t>(nvx_off(a)); }
uint nvx_ld_NvxU32A(uint64_t a) { return nvx_mem.Load(nvx_off(a)); }
int nvx_ld_NvxI32A(uint64_t a) { return asint(nvx_mem.Load(nvx_off(a))); }
// a byte store touches only its own byte: two atomics on the word (other lanes may store the neighbouring bytes)
void nvx_st_NvxU8(uint64_t a, uint v)
{
    uint o = nvx_off(a), sh = (o & 3u) * 8u, w = o & ~3u, old;
    nvx_mem.InterlockedAnd(w, ~(0xffu << sh), old);
    nvx_mem.InterlockedOr(w, (v & 0xffu) << sh, old);
}
void nvx_st_NvxI8(uint64_t a, int v) { nvx_st_NvxU8(a, uint(v)); }
void nvx_st_NvxU16(uint64_t a, uint16_t v) { nvx_mem.Store<uint16_t>(nvx_off(a), v); }
void nvx_st_NvxU16x2(uint64_t a, u16vec2 v) { nvx_mem.Store<u16vec2>(nvx_off(a), v); }
void nvx_st_NvxU32(uint64_t a, uint v) { nvx_mem.Store(nvx_off(a), v); }
void nvx_st_NvxU32x2(uint64_t a, uint2 v) { nvx_mem.Store2(nvx_off(a), v); }
void nvx_st_NvxU32x4(uint64_t a, uint4 v) { nvx_mem.Store4(nvx_off(a), v); }
void nvx_st_NvxU32x4a4(uint64_t a, uint4 v) { nvx_mem.Store4(nvx_off(a), v); }
void nvx_st_NvxU64(uint64_t a, uint64_t v) { nvx_mem.Store<uint64_t>(nvx_off(a), v); }
void nvx_st_NvxU64x2(uint64_t a, u64vec2 v) { nvx_mem.Store<u64vec2>(nvx_off(a), v); }
void nvx_st_NvxI64A(uint64_t a, int64_t v) { nvx_mem.Store<int64_t>(nvx_off(a), v); }
void nvx_st_NvxU64A(uint64_t a, uint64_t v) { nvx_mem.Store<uint64_t>(nvx_off(a), v); }
void nvx_st_NvxU32A(uint64_t a, uint v) { nvx_mem.Store(nvx_off(a), v); }
void nvx_st_NvxI32A(uint64_t a, int v) { nvx_mem.Store(nvx_off(a), asuint(v)); }
// atomics through pointers: GLSL atomicOp(NvxT(a).v, x) returns the old value
#define NVX_ATOM32(op, T, VT) \
    VT nvx_atom##op##_##T(uint64_t a, VT x) { uint o; nvx_mem.Interlocked##op(nvx_off(a), asuint(x), o); return (VT)o; }
NVX_ATOM32(Add, NvxU32A, uint) NVX_ATOM32(Min, NvxU32A, uint) NVX_ATOM32(Max, NvxU32A, uint)
NVX_ATOM32(And, NvxU32A, uint) NVX_ATOM32(Or, NvxU32A, uint) NVX_ATOM32(Xor, NvxU32A, uint)
NVX_ATOM32(Exchange, NvxU32A, uint)
int nvx_atomAdd_NvxI32A(uint64_t a, int x) { int o; nvx_mem.InterlockedAdd(nvx_off(a), x, o); return o; }
int nvx_atomMin_NvxI32A(uint64_t a, int x) { int o; nvx_mem.InterlockedMin(nvx_off(a), x, o); return o; }
int nvx_atomMax_NvxI32A(uint64_t a, int x) { int o; nvx_mem.InterlockedMax(nvx_off(a), x, o); return o; }
int64_t nvx_atomMin_NvxI64A(uint64_t a, int64_t x) { int64_t o; nvx_mem.InterlockedMin64(nvx_off(a), x, o); return o; }
int64_t nvx_atomMax_NvxI64A(uint64_t a, int64_t x) { int64_t o; nvx_mem.InterlockedMax64(nvx_off(a), x, o); return o; }
uint64_t nvx_atomAdd_NvxU64A(uint64_t a, uint64_t x) { uint64_t o; nvx_mem.InterlockedAdd64(nvx_off(a), x, o); return o; }
uint64_t nvx_atomMin_NvxU64A(uint64_t a, uint64_t x) { uint64_t o; nvx_mem.InterlockedMin64(nvx_off(a), x, o); return o; }
uint64_t nvx_atomMax_NvxU64A(uint64_t a, uint64_t x) { uint64_t o; nvx_mem.InterlockedMax64(nvx_off(a), x, o); return o; }
uint nvx_atomCompSwap_NvxU32A(uint64_t a, uint c, uint x) { uint o; nvx_mem.InterlockedCompareExchange(nvx_off(a), c, x, o); return o; }

// info words: [0, 4096) uvec4 sampler info, then simg_class[65536], timg_class[65536]
uint4 nvx_smpinfo(uint s) { return nvx_infob.Load4(s * 16u); }
uint nvx_simg_class(uint s) { return nvx_infob.Load(65536u + s * 4u); }
uint nvx_timg_class(uint s) { return nvx_infob.Load(65536u + 262144u + s * 4u); }

// ---------------------------------------------------------------- integer helpers (PTX shift/bitfield semantics)
uint nvx_shl32(uint a, uint b) { return b >= 32u ? 0u : (a << b); }
uint nvx_shr32(uint a, uint b) { return b >= 32u ? 0u : (a >> b); }
int nvx_sar32(int a, uint b) { return b >= 32u ? (a >> 31) : (a >> b); }
uint64_t nvx_shl64(uint64_t a, uint b) { return b >= 64u ? 0ull : (a << b); }
uint64_t nvx_shr64(uint64_t a, uint b) { return b >= 64u ? 0ull : (a >> b); }
int64_t nvx_sar64(int64_t a, uint b) { return b >= 64u ? (a >> 63) : (a >> b); }
uint16_t nvx_shl16(uint16_t a, uint b) { return b >= 16u ? uint16_t(0) : uint16_t(uint(a) << b); }
uint16_t nvx_shr16(uint16_t a, uint b) { return b >= 16u ? uint16_t(0) : uint16_t(uint(a) >> b); }
int16_t nvx_sar16(int16_t a, uint b) { return int16_t(int(a) >> min(b, 15u)); }
int nvx_rem32(int a, int b) { return a - (a / b) * b; }
uint nvx_bfe_u32(uint a, uint b, uint c)
{
    uint pos = b & 0xffu, len = c & 0xffu;
    if (len == 0u || pos >= 32u) return 0u;
    uint r = a >> pos;
    return len >= 32u ? r : (r & ((1u << len) - 1u));
}
int nvx_bfe_s32(int a, uint b, uint c)
{
    uint pos = b & 0xffu, len = c & 0xffu;
    if (len == 0u) return 0;
    if (pos >= 32u) return a >> 31;
    uint msb = min(pos + len - 1u, 31u);
    return int(uint(a) << (31u - msb)) >> (31u - msb + pos);
}
uint nvx_bfi32(uint ins, uint base, uint b, uint c)
{
    uint pos = b & 0xffu, len = c & 0xffu;
    if (len == 0u || pos >= 32u) return base;
    uint mask = (len >= 32u ? 0xffffffffu : ((1u << len) - 1u)) << pos;
    return (base & ~mask) | ((ins << pos) & mask);
}
uint64_t nvx_bfi64(uint64_t ins, uint64_t base, uint b, uint c)
{
    uint pos = b & 0xffu, len = c & 0xffu;
    if (len == 0u || pos >= 64u) return base;
    uint64_t mask = (len >= 64u ? 0xffffffffffffffffull : ((1ull << len) - 1ull)) << pos;
    return (base & ~mask) | ((ins << pos) & mask);
}
uint nvx_bfind_u32(uint a, bool shiftamt)
{
    if (a == 0u) return 0xffffffffu;
    uint m = firstbithigh(a);
    return shiftamt ? 31u - m : m;
}
uint nvx_mulhi_u32(uint a, uint b) { return uint((uint64_t(a) * uint64_t(b)) >> 32); }
int nvx_mulhi_s32(int a, int b) { return int((int64_t(a) * int64_t(b)) >> 32); }

// ---------------------------------------------------------------- float helpers
uint nvx_fbits(float x) { return asuint(x); }
float nvx_next_up(float p) { uint b = asuint(p); if ((isinf(p) && p > 0.0) || isnan(p)) return p; if (p == 0.0) return asfloat(0x00800000u); return asfloat(p > 0.0 ? b + 1u : b - 1u); }
float nvx_next_down(float p) { uint b = asuint(p); if ((isinf(p) && p < 0.0) || isnan(p)) return p; if (p == 0.0) return asfloat(0x80800000u); return asfloat(p > 0.0 ? b - 1u : b + 1u); }
float nvx_mul_rp(float a, float b) { precise float p = a * b; precise float e = fma(a, b, -p); return e > 0.0 ? nvx_next_up(p) : p; }
float nvx_mul_rm(float a, float b) { precise float p = a * b; precise float e = fma(a, b, -p); return e < 0.0 ? nvx_next_down(p) : p; }
float nvx_mul_rz(float a, float b) { precise float p = a * b; precise float e = fma(a, b, -p); return (p > 0.0 && e < 0.0) ? nvx_next_down(p) : ((p < 0.0 && e > 0.0) ? nvx_next_up(p) : p); }
float nvx_add_rz(float a, float b)
{
    precise float s = a + b;
    precise float bb = s - a;
    precise float e = (a - (s - bb)) + (b - bb);
    if (s > 0.0 && e < 0.0) return nvx_next_down(s);
    if (s < 0.0 && e > 0.0) return nvx_next_up(s);
    return s;
}
float nvx_add_rm(float a, float b) { precise float s = a + b; precise float bb = s - a; precise float e = (a - (s - bb)) + (b - bb); return e < 0.0 ? nvx_next_down(s) : s; }
float nvx_add_rp(float a, float b) { precise float s = a + b; precise float bb = s - a; precise float e = (a - (s - bb)) + (b - bb); return e > 0.0 ? nvx_next_up(s) : s; }
// PTX min/max.f32: a NaN operand yields the other operand (DXIL FMin / FMax are IEEE minNum / maxNum)
float nvx_fmin(float a, float b) { return min(a, b); }
float nvx_fmax(float a, float b) { return max(a, b); }
float nvx_sat(float x) { return x > 0.0 ? (x < 1.0 ? x : 1.0) : 0.0; }
float nvx_rcp(float x) { precise float r = 1.0 / x; return r; }
float nvx_div_approx(float a, float b) { precise float r = 1.0 / b; precise float q = a * r; return q; }
float nvx_ex2(float x) { return exp2(x); }
float nvx_copysign(float a, float b) { return asfloat((asuint(b) & 0x7fffffffu) | (asuint(a) & 0x80000000u)); }
int nvx_f2i(float x) { if (isnan(x)) return 0; if (x >= 2147483648.0) return 2147483647; if (x <= -2147483648.0) return int(0x80000000u); return int(x); }
uint nvx_f2u(float x) { if (isnan(x) || x <= 0.0) return 0u; if (x >= 4294967296.0) return 0xffffffffu; return uint(x); }

// ---------------------------------------------------------------- f16
// f32 -> f16 round to nearest even from the bits (AMD's D3D12 compiler converts toward zero; PTX cvt.rn wants RNE).
// Subnormal results are rounded on the f16 subnormal grid; overflow goes to infinity; NaN stays NaN (quiet).
uint16_t nvx_f32_to_f16_rne(float f)
{
    uint x = asuint(f), s = (x >> 16) & 0x8000u, e = (x >> 23) & 0xffu, m = x & 0x7fffffu;
    if (e == 0xffu) return uint16_t(s | 0x7c00u | (m != 0u ? (0x200u | (m >> 13)) : 0u));
    int ue = int(e) - 127 + 15;
    if (ue >= 31) return uint16_t(s | 0x7c00u);
    if (ue <= 0) {
        if (ue < -10) return uint16_t(s);
        uint mm = m | 0x800000u, sh = uint(14 - ue);
        uint r = mm >> sh, rem = mm & ((1u << sh) - 1u), half = 1u << (sh - 1u);
        if (rem > half || (rem == half && (r & 1u) != 0u)) r += 1u;
        return uint16_t(s | r);
    }
    uint r = (uint(ue) << 10) | (m >> 13), rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u) != 0u)) r += 1u;   // a carry into the exponent is correct rounding
    return uint16_t(s | r);
}
float16_t nvx_cvt_f16(float f) { return asfloat16(nvx_f32_to_f16_rne(f)); }
float16_t nvx_cvt_f16(float16_t h) { return h; }
float16_t nvx_cvt_f16(double d) { return nvx_cvt_f16(float(d)); }
f16vec2 nvx_mk_f16v2(float a, float b) { return f16vec2(nvx_cvt_f16(a), nvx_cvt_f16(b)); }
f16vec2 nvx_mk_f16v2(float16_t a, float16_t b) { return f16vec2(a, b); }
f16vec2 nvx_mk_f16v2(float2 v) { return f16vec2(nvx_cvt_f16(v.x), nvx_cvt_f16(v.y)); }
f16vec2 nvx_mk_f16v2(f16vec2 v) { return v; }
f16vec2 nvx_mk_f16v2(float16_t a) { return f16vec2(a, a); }
f16vec4 nvx_mk_f16v4(float a, float b, float c, float d) { return f16vec4(nvx_cvt_f16(a), nvx_cvt_f16(b), nvx_cvt_f16(c), nvx_cvt_f16(d)); }
f16vec4 nvx_mk_f16v4(float16_t a, float16_t b, float16_t c, float16_t d) { return f16vec4(a, b, c, d); }
f16vec4 nvx_mk_f16v4(float4 v) { return f16vec4(nvx_cvt_f16(v.x), nvx_cvt_f16(v.y), nvx_cvt_f16(v.z), nvx_cvt_f16(v.w)); }
f16vec4 nvx_mk_f16v4(f16vec4 v) { return v; }
float16_t nvx_h(uint16_t b) { return asfloat16(b); }
f16vec2 unpackFloat2x16(uint v) { return f16vec2(asfloat16(uint16_t(v & 0xffffu)), asfloat16(uint16_t(v >> 16))); }
uint packFloat2x16(f16vec2 v) { return uint(asuint16(v.x)) | (uint(asuint16(v.y)) << 16); }
uint16_t nvx_hb(float16_t h) { return asuint16(h); }
float16_t nvx_hmax(float16_t a, float16_t b) { return isnan(a) ? b : (isnan(b) ? a : max(a, b)); }
float16_t nvx_hmin(float16_t a, float16_t b) { return isnan(a) ? b : (isnan(b) ? a : min(a, b)); }
int nvx_h2i(float16_t h) { return nvx_f2i(float(h)); }
float nvx_d2f(uint64_t b)
{
    uint hi = uint(b >> 32), lo = uint(b), s = hi & 0x80000000u, mh = hi & 0xfffffu;
    int e = int((hi >> 20) & 0x7ffu);
    if (e == 0x7ff)
        return asfloat((mh | lo) != 0u ? 0x7fffffffu : (s | 0x7f800000u));
    int fe = e - 1023 + 127;
    if (fe >= 255)
        return asfloat(s | 0x7f800000u);
    uint m = (mh << 3) | (lo >> 29), rem = lo & 0x1fffffffu;
    if (fe <= 0) {
        bool up = fe == 0 && m == 0x7fffffu;
        return asfloat(up ? (s | 0x00800000u) : s);
    }
    uint r = (uint(fe) << 23) | m;
    if (rem > 0x10000000u || (rem == 0x10000000u && (m & 1u) != 0u))
        r += 1u;
    return asfloat(s | r);
}
uint16_t nvx_f2h(float x) { float y = asfloat(asuint(x) ^ nvx_opaque_zero()); return nvx_f32_to_f16_rne(y); }
uint packHalf2x16(float2 v) { return uint(nvx_f32_to_f16_rne(v.x)) | (uint(nvx_f32_to_f16_rne(v.y)) << 16); }
float2 unpackHalf2x16(uint v) { return float2(float(asfloat16(uint16_t(v & 0xffffu))), float(asfloat16(uint16_t(v >> 16)))); }

// ---------------------------------------------------------------- textures
uint nvx_ti(uint64_t h) { return uint(h) & 0xfffffu; }
uint nvx_si(uint64_t h) { return (uint(h) >> 20) & 0xfffu; }
Texture2D<float4> nvx_tex(uint ti) { return ResourceDescriptorHeap[NonUniformResourceIndex(ti)]; }
Texture2D<uint4> nvx_utex(uint ti) { return ResourceDescriptorHeap[NonUniformResourceIndex(ti)]; }
RWTexture2D<float4> nvx_img(uint s) { return ResourceDescriptorHeap[NonUniformResourceIndex(NVX_UAV_BASE + s)]; }
RWTexture2D<uint4> nvx_uimg(uint s) { return ResourceDescriptorHeap[NonUniformResourceIndex(NVX_UAV_BASE + s)]; }
RWTexture2D<int4> nvx_iimg(uint s) { return ResourceDescriptorHeap[NonUniformResourceIndex(NVX_UAV_BASE + s)]; }
SamplerState nvx_smp(uint si) { return SamplerDescriptorHeap[NonUniformResourceIndex(si)]; }
int2 nvx_tsize(uint ti) { uint w, h; nvx_tex(ti).GetDimensions(w, h); return int2(w, h); }
int2 nvx_utsize(uint ti) { uint w, h; nvx_utex(ti).GetDimensions(w, h); return int2(w, h); }
int2 nvx_isize(uint s) { uint w, h; nvx_img(s).GetDimensions(w, h); return int2(w, h); }
int2 nvx_clamp_coord(int2 c, int2 sz, uint smp) { return clamp(c, int2(0, 0), sz - 1); }
bool nvx_border(uint smp) { return nvx_smpinfo(smp).y == 3u; }
// exact texel fetch at integer coordinates (rt.glsl's NVX_SW_CLAMP path: D3D12 samplers have no unnormalized coordinates)
float4 nvx_fetch_n(uint ti, uint si, int2 c)
{
    int2 sz = nvx_tsize(ti);
    if (nvx_border(si) && (any(c < 0) || any(c >= sz))) return float4(0.0, 0.0, 0.0, 0.0);
    return nvx_tex(ti).Load(int3(nvx_clamp_coord(c, sz, si), 0));
}
uint4 nvx_ufetch_n(uint ti, uint si, int2 c)
{
    int2 sz = nvx_utsize(ti);
    if (nvx_border(si) && (any(c < 0) || any(c >= sz))) return uint4(0u, 0u, 0u, 0u);
    return nvx_utex(ti).Load(int3(nvx_clamp_coord(c, sz, si), 0));
}
float4 nvx_tex_fetch_i(uint64_t h, int x, int y) { return nvx_fetch_n(nvx_ti(h), nvx_si(h), int2(x, y)); }
uint4 nvx_utex_fetch_i(uint64_t h, int x, int y) { return nvx_ufetch_n(nvx_ti(h), nvx_si(h), int2(x, y)); }
#ifdef NVX_HW_FILTER
float4 nvx_tex2d(uint64_t h, float u, float v)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    uint filt = nvx_smpinfo(si).x & 0xfu;
    if (filt == 0u) {
        int2 sz = nvx_tsize(ti);
        precise float xu = u * float(sz.x);
        precise float yv = v * float(sz.y);
        return nvx_fetch_n(ti, si, int2(int(floor(xu)), int(floor(yv))));
    }
    return nvx_tex(ti).SampleLevel(nvx_smp(si), float2(u, v), 0.0);
}
float4 nvx_tex2d_m(uint64_t h, float u, float v, uint m) { return nvx_tex2d(h, u, v); }
#else
// filtered fetch with normalized coordinates and CUDA's 8-bit fixed-point bilinear weights
float4 nvx_tex2d(uint64_t h, float u, float v)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    int2 sz = nvx_tsize(ti);
    uint filt = nvx_smpinfo(si).x & 0xfu;
    precise float xu = u * float(sz.x);
    precise float yv = v * float(sz.y);
    if (filt == 0u)
        return nvx_fetch_n(ti, si, int2(int(floor(xu)), int(floor(yv))));
    precise float xb = xu - 0.5;
    precise float yb = yv - 0.5;
    precise float xq = floor(xb * 256.0 + 0.5);
    precise float yq = floor(yb * 256.0 + 0.5);
    precise float x0f = floor(xq * (1.0 / 256.0));
    precise float y0f = floor(yq * (1.0 / 256.0));
    precise float a = (xq - x0f * 256.0) * (1.0 / 256.0);
    precise float b = (yq - y0f * 256.0) * (1.0 / 256.0);
    int2 c0 = int2(int(x0f), int(y0f));
    int2 c1 = c0 + 1;
    float4 t00 = nvx_fetch_n(ti, si, c0);
    float4 t10 = nvx_fetch_n(ti, si, int2(c1.x, c0.y));
    float4 t01 = nvx_fetch_n(ti, si, int2(c0.x, c1.y));
    float4 t11 = nvx_fetch_n(ti, si, c1);
    precise float4 top = fma(t10 - t00, float4(a, a, a, a), t00);
    precise float4 bot = fma(t11 - t01, float4(a, a, a, a), t01);
    precise float4 r = fma(bot - top, float4(b, b, b, b), top);
    return r;
}
float4 nvx_gather(uint ti, uint si, float2 g, uint k)
{
    Texture2D<float4> t = nvx_tex(ti);
    SamplerState s = nvx_smp(si);
    if (k == 0u) return t.GatherRed(s, g);
    if (k == 1u) return t.GatherGreen(s, g);
    if (k == 2u) return t.GatherBlue(s, g);
    return t.GatherAlpha(s, g);
}
// nvx_tex2d for the channels in mask m only: one gather of the 2x2 footprint per used channel
float4 nvx_tex2d_m(uint64_t h, float u, float v, uint m)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    int2 sz = nvx_tsize(ti);
    uint filt = nvx_smpinfo(si).x & 0xfu;
    precise float xu = u * float(sz.x);
    precise float yv = v * float(sz.y);
    if (filt == 0u)
        return nvx_fetch_n(ti, si, int2(int(floor(xu)), int(floor(yv))));
    precise float xb = xu - 0.5;
    precise float yb = yv - 0.5;
    precise float xq = floor(xb * 256.0 + 0.5);
    precise float yq = floor(yb * 256.0 + 0.5);
    precise float x0f = floor(xq * (1.0 / 256.0));
    precise float y0f = floor(yq * (1.0 / 256.0));
    precise float a = (xq - x0f * 256.0) * (1.0 / 256.0);
    precise float b = (yq - y0f * 256.0) * (1.0 / 256.0);
    float2 g = (float2(x0f, y0f) + 1.0) / float2(sz);
    float4 r = float4(0.0, 0.0, 0.0, 0.0);
    [unroll] for (uint k = 0u; k < 4u; k++) {
        if ((m & (1u << k)) == 0u) continue;
        float4 q = nvx_gather(ti, si, g, k);
        precise float top_ = fma(q.z - q.w, a, q.w);
        precise float bot_ = fma(q.y - q.x, a, q.x);
        precise float r_ = fma(bot_ - top_, b, top_);
        r[k] = r_;
    }
    return r;
}
#endif
uint4 nvx_utex2d(uint64_t h, float u, float v)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    int2 sz = nvx_utsize(ti);
    precise float xu = u * float(sz.x);
    precise float yv = v * float(sz.y);
    return nvx_ufetch_n(ti, si, int2(int(floor(xu)), int(floor(yv))));
}
float4 nvx_tex2d_level(uint64_t h, float u, float v, float lod)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    if (lod == 0.0) return nvx_tex2d(h, u, v);
    return nvx_tex(ti).SampleLevel(nvx_smp(si), float2(u, v), lod);
}

// ---------------------------------------------------------------- surfaces (out-of-range accesses dropped / read 0)
uint nvx_sclass(uint64_t h) { return nvx_simg_class(uint(h) & 0xffffu) & 0xffu; }
void nvx_sust_p(uint64_t h, int x, int y, uint4 d)
{
    uint s = uint(h) & 0xfffffu;
    uint cls = nvx_sclass(h);
    int2 sz = nvx_isize(s);
    if (x < 0 || y < 0 || x >= sz.x || y >= sz.y) return;
    if (cls == 1u) nvx_uimg(s)[int2(x, y)] = d;
    else if (cls == 2u) nvx_iimg(s)[int2(x, y)] = int4(d);
    else nvx_img(s)[int2(x, y)] = asfloat(d);
}
uint4 nvx_suld_b_v4(uint64_t h, int xb, int y)
{
    uint s = uint(h) & 0xfffffu;
    uint cls = nvx_sclass(h);
    int x = xb >> 4;
    int2 sz = nvx_isize(s);
    if (x < 0 || y < 0 || x >= sz.x || y >= sz.y) return uint4(0u, 0u, 0u, 0u);
    if (cls == 1u) return nvx_uimg(s)[int2(x, y)];
    return asuint(nvx_img(s)[int2(x, y)]);
}
uint4 nvx_suld_b(uint64_t h, int xb, int y, uint esz, bool clampc)
{
    uint s = uint(h) & 0xfffffu;
    uint info = nvx_simg_class(uint(h) & 0xffffu);
    uint cls = info & 0xffu, kind = (info >> 8) & 0xfu, cb = max((info >> 12) & 0xfu, 1u), nc = max((info >> 16) & 0xfu, 1u);
    int x = xb / int(cb * nc);
    int2 sz = nvx_isize(s);
    if (clampc) {
        x = clamp(x, 0, sz.x - 1);
        y = clamp(y, 0, sz.y - 1);
    } else if (x < 0 || y < 0 || x >= sz.x || y >= sz.y) {
        return uint4(0u, 0u, 0u, 0u);
    }
    uint mask = cb >= 4u ? 0xffffffffu : (1u << (8u * cb)) - 1u;
    uint4 r;
    if (cls == 1u) {
        r = nvx_uimg(s)[int2(x, y)] & mask;
    } else if (cls == 2u) {
        r = uint4(nvx_iimg(s)[int2(x, y)]) & mask;
    } else {
        float4 v = nvx_img(s)[int2(x, y)];
        if (kind == 1u) {
            r = uint4(round(v * float(mask)));
        } else if (kind == 2u) {
            r = uint4(int4(round(v * float(mask >> 1)))) & mask;
        } else if (cb == 2u) {
            uint lo = packHalf2x16(v.xy), hi = packHalf2x16(v.zw);
            r = uint4(lo & 0xffffu, lo >> 16, hi & 0xffffu, hi >> 16);
        } else {
            r = asuint(v);
        }
    }
    if (((info >> 20) & 1u) != 0u)
        r = r.zyxw;
    return r;
}
void nvx_sust_b(uint64_t h, int xb, int y, uint4 d, uint esz, uint n, bool clampc)
{
    uint s = uint(h) & 0xfffffu;
    uint info = nvx_simg_class(uint(h) & 0xffffu);
    uint cls = info & 0xffu, kind = (info >> 8) & 0xfu, cb = max((info >> 12) & 0xfu, 1u), nc = max((info >> 16) & 0xfu, 1u);
    int x = xb / int(cb * nc);
    int2 sz = nvx_isize(s);
    if (clampc) {
        x = clamp(x, 0, sz.x - 1);
        y = clamp(y, 0, sz.y - 1);
    } else if (x < 0 || y < 0 || x >= sz.x || y >= sz.y) {
        return;
    }
    if (((info >> 20) & 1u) != 0u)
        d = d.zyxw;
    uint mask = cb >= 4u ? 0xffffffffu : (1u << (8u * cb)) - 1u;
    if (cls == 1u) {
        nvx_uimg(s)[int2(x, y)] = d;
    } else if (cls == 2u) {
        uint sh = 32u - 8u * cb;
        nvx_iimg(s)[int2(x, y)] = int4(d << sh) >> int(sh);
    } else {
        float4 v;
        if (kind == 1u) {
            v = float4(d) / float(mask);
        } else if (kind == 2u) {
            uint sh = 32u - 8u * cb;
            v = max(float4(int4(d << sh) >> int(sh)) / float(mask >> 1), float4(-1.0, -1.0, -1.0, -1.0));
        } else if (cb == 2u) {
            v = float4(unpackHalf2x16(d.x | (d.y << 16)), unpackHalf2x16(d.z | (d.w << 16)));
        } else {
            v = asfloat(d);
        }
        nvx_img(s)[int2(x, y)] = v;
    }
}

// ---------------------------------------------------------------- warp (wave32)
uint nvx_shfl_down(uint a, uint b, uint c, out bool p)
{
    uint lane = WaveGetLaneIndex();
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    uint j = lane + (b & 31u);
    p = j <= maxl;
    if (!p) j = lane;
    return WaveReadLaneAt(a, j);
}
uint nvx_shfl_bfly(uint a, uint b, uint c, out bool p)
{
    uint lane = WaveGetLaneIndex();
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    uint j = lane ^ (b & 31u);
    p = j <= maxl;
    if (!p) j = lane;
    return WaveReadLaneAt(a, j);
}
uint nvx_shfl_up(uint a, uint b, uint c, out bool p)
{
    uint lane = WaveGetLaneIndex();
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    int j = int(lane) - int(b & 31u);
    p = j >= int(maxl);
    if (!p) j = int(lane);
    return WaveReadLaneAt(a, uint(j));
}
uint nvx_shfl_idx(uint a, uint b, uint c, out bool p)
{
    uint lane = WaveGetLaneIndex();
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    uint minl = lane & segmask;
    uint j = minl | (b & 31u & ~segmask);
    p = j <= maxl;
    if (!p) j = lane;
    return WaveReadLaneAt(a, j);
}
#define nvx_shfl_down_k nvx_shfl_down
#define nvx_shfl_bfly_k nvx_shfl_bfly
#define nvx_shfl_up_k nvx_shfl_up
