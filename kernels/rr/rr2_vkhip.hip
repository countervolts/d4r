// RDNA4 preset F cooperative tiles and packed FP8 operations; native blocks in rr2_native_body.h.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "rr2_stage.h"
#pragma clang fp contract(off)
using uint = uint32_t;
using float16_t = _Float16;
using vec2 = float __attribute__((ext_vector_type(2)));
using f16vec2 = _Float16 __attribute__((ext_vector_type(2)));
using h8 = _Float16 __attribute__((ext_vector_type(8)));
using i2 = int __attribute__((ext_vector_type(2)));
using f8 = float __attribute__((ext_vector_type(8)));
#define D __device__ __forceinline__
struct Args { uint64_t param, weights, arena; };
D float satfp8(float v) {
#ifdef RRVK_SAT_MODE
    return v;
#else
    return __builtin_isnan(v) ? v : __builtin_fminf(__builtin_fmaxf(v,-448.f),448.f);
#endif
}
struct fp8 {
    uint8_t b;
    D fp8() = default;
    D fp8(float v) { b = __builtin_amdgcn_cvt_pk_fp8_f32(satfp8(v),0,0,false); }
    D operator float() const { return __builtin_amdgcn_cvt_f32_fp8(b,0); }
    D static fp8 bits(uint8_t b) { fp8 r; r.b=b; return r; }
};
using floate4m3_t = fp8;
struct fe4m3vec2 { fp8 x,y; };
D uint pack4f(float a,float b,float c,float d) {
    a=satfp8(a); b=satfp8(b); c=satfp8(c); d=satfp8(d);
    const int lo=__builtin_amdgcn_cvt_pk_fp8_f32(a,b,0,false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(c,d,lo,true);
}
D float finitefp8(float v) {
#ifdef RRVK_SAT_MODE
    return v; // the native block's f32 sums of finite FP8 products cannot overflow
#else
    return __builtin_isinf(v) ? __builtin_nanf("") : v;
#endif
}
D uint packfinite4f(float a,float b,float c,float d) { return pack4f(finitefp8(a),finitefp8(b),finitefp8(c),finitefp8(d)); }
D vec2 finitepair(vec2 v) { return {finitefp8(v.x),finitefp8(v.y)}; }
D fe4m3vec2 quant(vec2 v) { uint w=pack4f(v.x,v.y,0,0); return {fp8::bits(w),fp8::bits(w>>8)}; }
D fe4m3vec2 quant(f16vec2 v) { return quant((vec2){float(v.x),float(v.y)}); }
D f16vec2 fp8pair(uint w,uint h) { return {(_Float16)float(fp8::bits(w >> (16*h))),(_Float16)float(fp8::bits(w >> (16*h+8)))}; }
D uint floatBitsToUint(float v) { return __builtin_bit_cast(uint,v); }
D float uintBitsToFloat(uint v) { return __builtin_bit_cast(float,v); }
D uint code(fp8 v) { return v.b; }
D _Float16 hb(uint b) { return __builtin_bit_cast(_Float16,uint16_t(b)); }
D uint bh(_Float16 h) { return __builtin_bit_cast(uint16_t,h); }
D uint inter(uint j) { return (j & ~31u) | (((j & 31u) >> 2) << 1) | (j & 1u) | (((j >> 1) & 1u) << 4); }
D int mirror(int v,int n) { v=v<0?-v:v; int m=2*n-2-v;return v<m?v:m; }
D uint packFloat2x16(f16vec2 v) { return __builtin_bit_cast(uint,v); }
D f16vec2 unpackFloat2x16(uint v) { return __builtin_bit_cast(f16vec2,v); }
D uint subgroupShuffleXor(uint v,uint mask) { return __shfl_xor(v,mask); }
D void subgroupMemoryBarrierShared() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL,"wavefront","local"); }
D void memoryBarrierShared() { subgroupMemoryBarrierShared(); }
D void barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
D vec2 clamp(vec2 a,vec2 b,vec2 c) { return __builtin_elementwise_min(__builtin_elementwise_max(a,b),c); }
D vec2 abs(vec2 a) { return __builtin_elementwise_abs(a); }
D vec2 fma(vec2 a,vec2 b,vec2 c) { return __builtin_elementwise_fma(a,b,c); }
D vec2 makevec(float a) { return {a,a}; }
D vec2 makevec(float a,float b) { return {a,b}; }
D f16vec2 makehalf(_Float16 a) { return {a,a}; }
D f16vec2 makehalf(_Float16 a,_Float16 b) { return {a,b}; }
D f16vec2 makehalf(f16vec2 a) { return a; }
#define vec2(...) makevec(__VA_ARGS__)
#define f16vec2(...) makehalf(__VA_ARGS__)
template<class T,int role> struct Tile {
    T v[8];
    D Tile() = default;
    D explicit Tile(float x) { for(int i=0;i<8;++i)v[i]=T(x); }
    D T& operator[](int i) { return v[i]; }
    D const T& operator[](int i) const { return v[i]; }
};
template<int role> struct Tile<fp8,role> {
    uint64_t packed = 0;
    struct Ref {
        uint64_t& packed; int i;
        D operator fp8() const { return fp8::bits(packed >> (8*i)); }
        D Ref& operator=(fp8 f) { const uint64_t mask=uint64_t(255)<<(8*i); packed=(packed&~mask)|(uint64_t(f.b)<<(8*i)); return *this; }
    };
    D Ref operator[](int i) { return {packed,i}; }
    D fp8 operator[](int i) const { return fp8::bits(packed>>(8*i)); }
};
using ACC=Tile<float,2>;using HAC=Tile<_Float16,2>;using A8=Tile<fp8,0>;using B8=Tile<fp8,1>;
constexpr int RM=0,CMJ=1;
template<class T,int role,class P> D void coopMatLoad(Tile<T,role>& dst,const P* src,uint offset,uint stride,int layout) {
    const uint lane=threadIdx.x&31, col=lane&15, hl=lane>>4;
    #pragma unroll
    for(uint i=0;i<8;++i) {
        const uint row=role==0?col:8*hl+i, column=role==0?8*hl+i:col;
        // GLSL strides count source elements, not bytes.
        const uint element=layout==RM?row*stride:column*stride;
        const char* p=(const char*)src+(offset+element)*sizeof(P)+(layout==RM?column:row)*sizeof(T);
        dst[i]=*(const T*)p;
    }
}
template<class T,int role,class P> D void coopMatStore(const Tile<T,role>& src,P* dst,uint offset,uint stride,int layout) {
    const uint lane=threadIdx.x&31, col=lane&15, hl=lane>>4;
    #pragma unroll
    for(uint i=0;i<8;++i) {
        const uint row=role==0?col:8*hl+i, column=role==0?8*hl+i:col;
        char* p=(char*)dst+(offset+(layout==RM?row:column)*stride)*sizeof(P)+(layout==RM?column:row)*sizeof(T);
        *(T*)p=src[i];
    }
}
// FP8 operand tiles are eight consecutive bytes per lane in their native layout.
// Keep them packed through memory and WMMA instead of issuing eight byte loads.
template<int role,class P> D void coopMatLoad(Tile<fp8,role>& dst,const P* src,uint offset,uint stride,int layout) {
    const uint lane=threadIdx.x&31, col=lane&15, hl=lane>>4;
    if ((role==0 && layout==RM) || (role==1 && layout==CMJ)) {
        const char* p=(const char*)src+(offset+col*stride)*sizeof(P)+8*hl;
        dst.packed=*(const uint64_t*)p;
    } else {
        uint64_t bits=0;
        #pragma unroll
        for(uint i=0;i<8;++i) {
            const char* p=(const char*)src+(offset+(8*hl+i)*stride)*sizeof(P)+col;
            bits|=uint64_t(*(const uint8_t*)p)<<(8*i);
        }
        dst.packed=bits;
    }
}
template<int role,class P> D void coopMatStore(const Tile<fp8,role>& src,P* dst,uint offset,uint stride,int layout) {
    const uint lane=threadIdx.x&31, col=lane&15, hl=lane>>4;
    if ((role==0 && layout==RM) || (role==1 && layout==CMJ)) {
        char* p=(char*)dst+(offset+col*stride)*sizeof(P)+8*hl;
        *(uint64_t*)p=src.packed;
    } else {
        #pragma unroll
        for(uint i=0;i<8;++i) {
            char* p=(char*)dst+(offset+(8*hl+i)*stride)*sizeof(P)+col;
            *(uint8_t*)p=src.packed>>(8*i);
        }
    }
}
D ACC coopMatMulAdd(A8 a,B8 b,ACC c) {
    ACC r;
    const auto out=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(i2,a),__builtin_bit_cast(i2,b),__builtin_bit_cast(f8,c));
    return __builtin_bit_cast(ACC,out);
}
#ifndef RRVK_HS
#define RRVK_HS 2
#endif
#ifndef RRVK_NAME
#define RRVK_NAME rr2_dec0_0
#endif
extern "C" {
__device__ __constant__ uint32_t rr2_threads = 128 * RRVK_HS;
__device__ __constant__ uint32_t rr2_v_layout[4] = {C, NH, W_WV0, W_WQK1-W_WQK0};
}
using uvec2 = uint __attribute__((ext_vector_type(2)));
using f16vec3 = _Float16 __attribute__((ext_vector_type(3)));
using f16vec4 = _Float16 __attribute__((ext_vector_type(4)));
D f16vec3 makehalf3(_Float16 a,_Float16 b,_Float16 c) { return {a,b,c}; }
D f16vec4 makehalf4(_Float16 a) { return {a,a,a,a}; }
D f16vec4 makehalf4(f16vec3 a,_Float16 b) { return {a.x,a.y,a.z,b}; }
D f16vec4 min(f16vec4 a,f16vec4 b) { return __builtin_elementwise_min(a,b); }
D _Float16 sig(_Float16 v) { return (_Float16)(1.0f / (__builtin_amdgcn_exp2f(float(v)*-1.4426950216293335f)+1.0f)); }
D uvec2 makeuint2(uint a,uint b) { return {a,b}; }
D f16vec4 makehalf4(_Float16 a,_Float16 b,_Float16 c,_Float16 d) { return {a,b,c,d}; }
D vec2 makevecpair(uint w,uint h) { auto v=fp8pair(w,h); return {float(v.x),float(v.y)}; }
#define uvec2(...) makeuint2(__VA_ARGS__)
#define f16vec3(...) makehalf3(__VA_ARGS__)
#define f16vec4(...) makehalf4(__VA_ARGS__)
#include "rr2_native_body.h"
