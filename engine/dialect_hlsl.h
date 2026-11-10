// d4r engine: the GLSL of the engine's compute shaders, read as HLSL (DXC, Shader Model 6.6, HLSL 2021,
// -enable-16bit-types). engine/hlsl_build.py removes the GLSL-only directives (#version, #extension, the workgroup
// size layout) and includes this header first; everything the two languages spell differently is either a macro
// here or an #ifdef D4R_HLSL block in the shader itself (resource declarations, shared memory, matrix operations).
// glslang never sees this file, so the SPIR-V the Vulkan backend runs is not affected by it.
//
// Arithmetic follows the GLSL meaning: fma maps to mad, which AMD's compilers fuse (DXIL has no single-precision
// fma; the cross-backend replay checks confirm the results), and precise keeps its meaning in both languages.
// Conversions to binary16 round to nearest even, as in the Vulkan backend: AMD's D3D12 compiler converts with
// v_cvt_pkrtz (toward zero), which D3D allows, so every conversion here first rounds the value in f32 (d4rRne16),
// and the hardware conversion is then exact. Packaged D3D12 kernels select
// D4R_NATIVE_HALF to use native conversion rounding; the emulation remains for
// isolated cross-backend correctness tests.
#ifndef D4R_DIALECT_HLSL
#define D4R_DIALECT_HLSL
#define D4R_HLSL 1

// ---------------------------------------------------------------- types and constructors
// GLSL constructors with one argument splat a scalar or convert a vector; HLSL writes both as a cast. Constructors
// with more arguments are the same in both languages. A type name not followed by '(' is the typedef.
#define D4R_ARGN(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, N, ...) N
#define D4R_NARGS(...) D4R_ARGN(__VA_ARGS__, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, 1, 0)
#define D4R_CAT(a, b) D4R_CAT_(a, b)
#define D4R_CAT_(a, b) a##b
#define D4R_CTOR(T, ...) D4R_CAT(D4R_CTOR_, D4R_NARGS(__VA_ARGS__))(T, __VA_ARGS__)
#define D4R_CTOR_1(T, x) ((T)(x))
#define D4R_CTOR_M(T, ...) T(__VA_ARGS__)
#define D4R_TYPE(G, H) typedef H G;
// v rounded to the nearest binary16 value (ties to even), as an f32 that any conversion then keeps exactly. Normal
// values round in the integer bits (bias and mask at mantissa bit 13); binary16 subnormals, at their fixed spacing
// of 2^-24, through an f32 addition (precise: x + 0.75 has an ulp of 2^-24). Infinity and NaN stay as they are; a
// value that rounds past 65504 gives 65536, which AMD's conversion (toward zero) turns into 65504 where Vulkan gives
// infinity. About four instructions: the general form with the overflow and NaN selects cost the layers 2.3 ms at
// 1080p -> 4K (RX 9070 XT, 2026-10-22).
// D4R_RNE16_EXTERNAL: the including model provides d4rRne16 for float and float2 (rr_wmma.hlsli: the AGS matrix
// conversion, which rounds to nearest even in one instruction per pair; D3D12 casts and f32tof16 truncate on AMD).
// Not for vkd3d-proton, whose casts already round to nearest even (below).
#if D4R_RNE16_EXTERNAL && D4R_RNE16_AGS_GFX11 && !D4R_VKD3D
#include "d3d12/rne16_ags.hlsli"
#elif D4R_RNE16_EXTERNAL && !D4R_VKD3D
float d4rRne16(float v);
float2 d4rRne16(float2 v);
#else
float d4rRne16(float v)
{
#if D4R_VKD3D
    // vkd3d-proton (compile_m.py --d3d12-target vkd3d): a binary16 cast is SPIR-V's OpFConvert, which RADV rounds
    // to nearest even; precise keeps the f32 -> f16 -> f32 pair (a following cast to binary16 folds it away)
    precise float r = (float)(float16_t)v;
    return r;
#elif D4R_NATIVE_HALF
    return f16tof32(f32tof16(v));
#elif D4R_RNE16_EXACT
    // RDNA 3 builds (m_d3d12_build.py, compile_rr_dxil.py gfx11): every f32 rounded as NVIDIA's cvt.rn.f16.f32 does.
    // The form below loses binary16 subnormals on AMD's D3D12 compiler, which folds the precise (v + 0.75) - 0.75 to
    // v (RX 7700 XT, 26.10.1: half of the inputs under 2^-14 truncated); here they round on round(), DXIL's round_ne,
    // at their spacing 2^-24 (v * 2^24 is exact), and a result past 65504 becomes the infinity of its sign. Equal to
    // round-to-nearest-even for all of 2^20 f32 inputs over binary16's range on that GPU.
    const uint bits = asuint(v);
    const float subnormal = round(v * 16777216.0) * 5.9604644775390625e-08;
    const float normal = asfloat((bits + 4095u + ((bits >> 13u) & 1u)) & 0xffffe000u);
    const float r = abs(v) < 6.103515625e-05 ? subnormal : normal;
    return abs(r) > 65504.0 && !isnan(r) ? asfloat((bits & 0x80000000u) | 0x7f800000u) : r;
#else
    const uint bits = asuint(v);
    precise float subnormal = (v + 0.75) - 0.75;
    return abs(v) < 6.103515625e-05 ? subnormal : asfloat((bits + 4095u + ((bits >> 13u) & 1u)) & 0xffffe000u);
#endif
}
float2 d4rRne16(float2 v) { return float2(d4rRne16(v.x), d4rRne16(v.y)); }
#endif
float3 d4rRne16(float3 v) { return float3(d4rRne16(v.xy), d4rRne16(v.z)); }
float4 d4rRne16(float4 v) { return float4(d4rRne16(v.xy), d4rRne16(v.zw)); }
D4R_TYPE(vec2, float2) D4R_TYPE(vec3, float3) D4R_TYPE(vec4, float4)
D4R_TYPE(ivec2, int2) D4R_TYPE(ivec3, int3) D4R_TYPE(ivec4, int4)
D4R_TYPE(uvec2, uint2) D4R_TYPE(uvec3, uint3) D4R_TYPE(uvec4, uint4)
D4R_TYPE(bvec2, bool2) D4R_TYPE(bvec3, bool3) D4R_TYPE(bvec4, bool4)
D4R_TYPE(f16vec2, float16_t2) D4R_TYPE(f16vec3, float16_t3) D4R_TYPE(f16vec4, float16_t4)
D4R_TYPE(u16vec2, uint16_t2) D4R_TYPE(u16vec4, uint16_t4)
#define vec2(...) D4R_CTOR(float2, __VA_ARGS__)
#define vec3(...) D4R_CTOR(float3, __VA_ARGS__)
#define vec4(...) D4R_CTOR(float4, __VA_ARGS__)
#define ivec2(...) D4R_CTOR(int2, __VA_ARGS__)
#define ivec3(...) D4R_CTOR(int3, __VA_ARGS__)
#define ivec4(...) D4R_CTOR(int4, __VA_ARGS__)
#define uvec2(...) D4R_CTOR(uint2, __VA_ARGS__)
#define uvec3(...) D4R_CTOR(uint3, __VA_ARGS__)
#define uvec4(...) D4R_CTOR(uint4, __VA_ARGS__)
#define bvec4(...) D4R_CTOR(bool4, __VA_ARGS__)
// binary16 constructors: every argument through d4rH, which rounds f32 values (d4rRne16) and integers (through f32)
// and passes binary16 values as they are (a binary16 argument rounded again in the integer bits would cost the
// emulation's instructions for nothing: post_m's Gaussian splats, 2026-10-23)
// (an untyped literal matches the float and the binary16 overload alike: d4rL, a template, gives it the type float)
template <typename T> T d4rL(T v) { return v; }
#if D4R_VKD3D
// vkd3d-proton: the cast itself rounds to nearest even (one conversion; d4rRne16's precise pair would add two).
// Not precise, as in the Vulkan source: unfused, RR D and F gain 0.04 and 0.2 dB against the Vulkan build, E loses 0.3
#define D4R_H(H, F, I, U) \
    H d4rH(F v) { return (H)v; } \
    H d4rH(H v) { return v; } \
    H d4rH(I v) { return (H)(F)v; } \
    H d4rH(U v) { return (H)(F)v; }
#else
#define D4R_H(H, F, I, U) \
    H d4rH(F v) { return (H)d4rRne16(v); } \
    H d4rH(H v) { return v; } \
    H d4rH(I v) { return (H)d4rRne16((F)v); } \
    H d4rH(U v) { return (H)d4rRne16((F)v); }
#endif
D4R_H(float16_t, float, int, uint) D4R_H(float16_t2, float2, int2, uint2)
D4R_H(float16_t3, float3, int3, uint3) D4R_H(float16_t4, float4, int4, uint4)
#define D4R_NUM(...) D4R_NUM_(__VA_ARGS__, 4, 3, 2, 1, 0)
#define D4R_NUM_(_1, _2, _3, _4, N, ...) N
#define D4R_HV(T, ...) D4R_CAT(D4R_HV_, D4R_NUM(__VA_ARGS__))(T, __VA_ARGS__)
#define D4R_HV_1(T, a) ((T)d4rH(d4rL(a)))
#define D4R_HV_2(T, a, b) T(d4rH(d4rL(a)), d4rH(d4rL(b)))
#define D4R_HV_3(T, a, b, c) T(d4rH(d4rL(a)), d4rH(d4rL(b)), d4rH(d4rL(c)))
#define D4R_HV_4(T, a, b, c, d) T(d4rH(d4rL(a)), d4rH(d4rL(b)), d4rH(d4rL(c)), d4rH(d4rL(d)))
#define float16_t(x) d4rH(d4rL(x))
#define f16vec2(...) D4R_HV(float16_t2, __VA_ARGS__)
#define f16vec3(...) D4R_HV(float16_t3, __VA_ARGS__)
#define f16vec4(...) D4R_HV(float16_t4, __VA_ARGS__)
#define u16vec2(...) D4R_CTOR(uint16_t2, __VA_ARGS__)
#define u16vec4(...) D4R_CTOR(uint16_t4, __VA_ARGS__)

// ---------------------------------------------------------------- built-in functions
#define fma mad
#define inversesqrt rsqrt
#define roundEven round
#define fract frac
#define mix(a, b, c) select((c), (b), (a))       // the engine only mixes with boolean selectors
#define lessThan(a, b) ((a) < (b))
#define greaterThan(a, b) ((a) > (b))
#define lessThanEqual(a, b) ((a) <= (b))
#define greaterThanEqual(a, b) ((a) >= (b))
#define not(a) (!(a))
#define equal(a, b) ((a) == (b))
#define notEqual(a, b) ((a) != (b))
#define uintBitsToFloat asfloat
#define floatBitsToUint asuint
#define intBitsToFloat asfloat
#define floatBitsToInt asint
#define uint16BitsToFloat16 asfloat16
#define float16BitsToUint16 asuint16
#if D4R_VKD3D
uint packHalf2x16(float2 v) { const uint16_t2 b = asuint16((float16_t2)v); return uint(b.x) | (uint(b.y) << 16); }
#else
uint packHalf2x16(float2 v) { v = d4rRne16(v); return f32tof16(v.x) | (f32tof16(v.y) << 16); }
#endif
float2 unpackHalf2x16(uint v) { return float2(f16tof32(v & 0xffffu), f16tof32(v >> 16)); }
uint packFloat2x16(float16_t2 v) { const uint16_t2 b = asuint16(v); return uint(b.x) | (uint(b.y) << 16); }
float16_t2 unpackFloat2x16(uint v) { return asfloat16(uint16_t2(uint16_t(v & 0xffffu), uint16_t(v >> 16))); }
uint16_t2 unpack16(uint v) { return uint16_t2(uint16_t(v & 0xffffu), uint16_t(v >> 16)); }
// Packed operations the RDNA 3 software-FP8 M/L stages use (m_hlsl_build.py)
#define dotPacked4x8EXT(a, b) dot4add_i8packed((a), (b), 0)
#define swizzleInvocationsMaskedAMD(v, mask) WaveReadLaneAt((v), (WaveGetLaneIndex() & (mask).x) | (mask).y)
#define subgroupShuffleXor(x, mask) WaveReadLaneAt((x), WaveGetLaneIndex() ^ (mask))
#define subgroupAll(x) WaveActiveAllTrue(x)
#define memoryBarrierShared()
#define barrier() GroupMemoryBarrierWithGroupSync()
#define shared groupshared
// GLSL identifiers that are HLSL keywords
#define line d4r_line
#define point d4r_point
#define triangle d4r_triangle
#define sample d4r_sample
#define linear d4r_linear

// ---------------------------------------------------------------- textures and images
// A GLSL sampler2D NAME is declared as Texture2D NAME and SamplerState NAME_s (D4R_SAMPLER).
#define D4R_SAMPLER(name, t, s) Texture2D<float4> name : register(t); SamplerState name##_s : register(s);
#define texelFetch(t, c, l) (t).Load(int3((c), (l)))
#define textureLod(t, uv, l) (t).SampleLevel(t##_s, (uv), (l))
#define textureGather(t, uv, c) (t).GatherRed(t##_s, (uv))
#define imageStore(i, c, v) ((i)[(c)] = (v))
int2 d4rTextureSize(Texture2D<float4> t) { uint w, h; t.GetDimensions(w, h); return int2(w, h); }
#define textureSize(t, l) d4rTextureSize(t)

// ---------------------------------------------------------------- the GLSL built-in inputs and the entry point
static uint3 gl_GlobalInvocationID, gl_LocalInvocationID, gl_WorkGroupID;
static uint gl_LocalInvocationIndex;
// hlsl_build.py appends D4R_ENTRY(x, y, z) with the shader's workgroup size; D4R_HLSL_INIT (optional, defined by
// the shader) loads the parameter blocks.
#ifndef D4R_WAVE_SIZE
#define D4R_WAVE_SIZE 64
#endif
#define D4R_ENTRY(X, Y, Z) \
    [WaveSize(D4R_WAVE_SIZE)] [numthreads(X, Y, Z)] \
    void d4r_entry(uint3 global : SV_DispatchThreadID, uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID, \
                   uint index : SV_GroupIndex) \
    { \
        gl_GlobalInvocationID = global; gl_LocalInvocationID = local; gl_WorkGroupID = group; \
        gl_LocalInvocationIndex = index; \
        D4R_HLSL_INIT_CALL \
        main(); \
    }
#endif
