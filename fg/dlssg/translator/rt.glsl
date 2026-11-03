// nvxrt T0 runtime: helpers with PTX semantics, included by every translated kernel.
// Contains no NVIDIA code; semantics follow the public PTX ISA.

#define NVX_SHMEM_STATIC_WORDS_DEFAULT 1
#ifdef NVX_SHADOW8
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_spirv_intrinsics : require
#endif

layout(push_constant) uniform NvxPC { uint w[64]; } pc;
layout(set = 0, binding = 0) uniform texture2D nvx_tex[];
layout(set = 0, binding = 0) uniform utexture2D nvx_utex[];
layout(set = 0, binding = 0) uniform itexture2D nvx_itex[];
layout(set = 0, binding = 1) uniform image2D nvx_img[];
layout(set = 0, binding = 1) uniform uimage2D nvx_uimg[];
layout(set = 0, binding = 1) uniform iimage2D nvx_iimg[];
layout(set = 0, binding = 2) uniform sampler nvx_smp[];
// per sampler slot: NEAREST, unnormalized coordinates, same clamp-to-edge / clamp-to-border(0) rule (set by the layer)
layout(set = 0, binding = 4) uniform sampler nvx_smpn[];
// [0, 4096) sampler info {filter, addrU, addrV, border}; then per storage-image slot format class
layout(set = 0, binding = 3, std430) readonly buffer NvxInfo { uvec4 smp[4096]; uint simg_class[65536]; uint timg_class[65536]; } nvx_info;
#ifdef NVX_SHADOW8
// flags the kernels raise for the layer (host-visible): [0] != 0: a value stored into an RGBA8 shadow of class 3 was not
// an 8-bit quantity (the layer then stops shadowing; see nvx_sust_p)
layout(set = 0, binding = 5, std430) coherent buffer NvxFlags { uint w[]; } nvx_flags;
#endif

layout(buffer_reference, buffer_reference_align = 1, scalar) buffer NvxU8 { uint8_t v; };
layout(buffer_reference, buffer_reference_align = 1, scalar) buffer NvxI8 { int8_t v; };
layout(buffer_reference, buffer_reference_align = 2, scalar) buffer NvxU16 { uint16_t v; };
layout(buffer_reference, buffer_reference_align = 4, scalar) buffer NvxU16x2 { u16vec2 v; };
layout(buffer_reference, buffer_reference_align = 4, scalar) buffer NvxU32 { uint v; };
layout(buffer_reference, buffer_reference_align = 8, scalar) buffer NvxU32x2 { uvec2 v; };
layout(buffer_reference, buffer_reference_align = 16, scalar) buffer NvxU32x4 { uvec4 v; };
layout(buffer_reference, buffer_reference_align = 4, scalar) buffer NvxU32x4a4 { uvec4 v; };
layout(buffer_reference, buffer_reference_align = 8, scalar) buffer NvxU64 { uint64_t v; };
layout(buffer_reference, buffer_reference_align = 16, scalar) buffer NvxU64x2 { u64vec2 v; };
layout(buffer_reference, buffer_reference_align = 8) buffer NvxI64A { int64_t v; };
layout(buffer_reference, buffer_reference_align = 8) buffer NvxU64A { uint64_t v; };
layout(buffer_reference, buffer_reference_align = 4) buffer NvxU32A { uint v; };
layout(buffer_reference, buffer_reference_align = 4) buffer NvxI32A { int v; };

uint nvx_opaque_zero() { return pc.w[63]; }

// ---------------------------------------------------------------- params
#ifdef NVX_BIG_PARAMS
// parameter blocks above 248 bytes: the layer writes the block into a host-visible buffer, address in pc.w[60..61]
layout(buffer_reference, buffer_reference_align = 4) readonly buffer NvxParamWords { uint w[]; };
uint nvx_pword(uint i) { return NvxParamWords(packUint2x32(uvec2(pc.w[60], pc.w[61]))).w[i]; }
uint nvx_p32(uint off) { return nvx_pword(off >> 2); }
uint nvx_p16(uint off) { return (nvx_pword(off >> 2) >> ((off & 2u) * 8u)) & 0xffffu; }
uint nvx_p8(uint off) { return (nvx_pword(off >> 2) >> ((off & 3u) * 8u)) & 0xffu; }
uint64_t nvx_p64(uint off) { return packUint2x32(uvec2(nvx_pword(off >> 2), nvx_pword((off >> 2) + 1u))); }
#else
uint nvx_p32(uint off) { return pc.w[off >> 2]; }
uint nvx_p16(uint off) { return (pc.w[off >> 2] >> ((off & 2u) * 8u)) & 0xffffu; }
uint nvx_p8(uint off) { return (pc.w[off >> 2] >> ((off & 3u) * 8u)) & 0xffu; }
uint64_t nvx_p64(uint off) { return packUint2x32(uvec2(pc.w[off >> 2], pc.w[(off >> 2) + 1u])); }
#endif

// ---------------------------------------------------------------- integer helpers (PTX shift/bitfield semantics)
uint nvx_shl32(uint a, uint b) { return b >= 32u ? 0u : (a << b); }
uint nvx_shr32(uint a, uint b) { return b >= 32u ? 0u : (a >> b); }
int nvx_sar32(int a, uint b) { return b >= 32u ? (a >> 31) : (a >> b); }
uint64_t nvx_shl64(uint64_t a, uint b) { return b >= 64u ? 0ul : (a << b); }
uint64_t nvx_shr64(uint64_t a, uint b) { return b >= 64u ? 0ul : (a >> b); }
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
    int r = int(uint(a) << (31u - msb)) >> (31u - msb + pos);
    return r;
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
    uint64_t mask = (len >= 64u ? 0xfffffffffffffffful : ((1ul << len) - 1ul)) << pos;
    return (base & ~mask) | ((ins << pos) & mask);
}
uint nvx_bfind_u32(uint a, bool shiftamt)
{
    if (a == 0u) return 0xffffffffu;
    uint m = uint(findMSB(a));
    return shiftamt ? 31u - m : m;
}
uint nvx_mulhi_u32(uint a, uint b) { uint hi, lo; umulExtended(a, b, hi, lo); return hi; }
int nvx_mulhi_s32(int a, int b) { int hi, lo; imulExtended(a, b, hi, lo); return hi; }

// ---------------------------------------------------------------- float helpers
uint nvx_fbits(float x) { return floatBitsToUint(x); }
float nvx_next_up(float p) { uint b = floatBitsToUint(p); if (isinf(p) && p > 0.0 || isnan(p)) return p; if (p == 0.0) return uintBitsToFloat(0x00800000u); return uintBitsToFloat(p > 0.0 ? b + 1u : b - 1u); }
float nvx_next_down(float p) { uint b = floatBitsToUint(p); if (isinf(p) && p < 0.0 || isnan(p)) return p; if (p == 0.0) return uintBitsToFloat(0x80800000u); return uintBitsToFloat(p > 0.0 ? b - 1u : b + 1u); }
// directed-rounding multiply/add from the RNE result and its exact error
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
// PTX min/max: if one operand is NaN the other is returned
// PTX min/max.f32: a NaN operand yields the other operand - GLSL.std.450 NMin/NMax exactly (one VALU op instead of
// two NaN tests and selects)
spirv_instruction(set = "GLSL.std.450", id = 79) float nvx_nmin_(float a, float b);
spirv_instruction(set = "GLSL.std.450", id = 80) float nvx_nmax_(float a, float b);
float nvx_fmin(float a, float b) { return nvx_nmin_(a, b); }
float nvx_fmax(float a, float b) { return nvx_nmax_(a, b); }
float nvx_sat(float x) { return x > 0.0 ? (x < 1.0 ? x : 1.0) : 0.0; }
float nvx_rcp(float x) { precise float r = 1.0 / x; return r; }
float nvx_div_approx(float a, float b) { precise float r = 1.0 / b; precise float q = a * r; return q; }
float nvx_ex2(float x) { return exp2(x); }
float nvx_copysign(float a, float b) { return uintBitsToFloat((floatBitsToUint(b) & 0x7fffffffu) | (floatBitsToUint(a) & 0x80000000u)); }
// float -> int conversions: saturating, NaN -> 0 (PTX cvt.*i)
int nvx_f2i(float x) { if (isnan(x)) return 0; if (x >= 2147483648.0) return 2147483647; if (x <= -2147483648.0) return int(0x80000000u); return int(x); }
uint nvx_f2u(float x) { if (isnan(x) || x <= 0.0) return 0u; if (x >= 4294967296.0) return 0xffffffffu; return uint(x); }
int nvx_h2i(float16_t h) { return nvx_f2i(float(h)); }

// ---------------------------------------------------------------- f16
float16_t nvx_h(uint16_t b) { return uint16BitsToFloat16(b); }
uint16_t nvx_hb(float16_t h) { return float16BitsToUint16(h); }
float16_t nvx_hmax(float16_t a, float16_t b) { return isnan(a) ? b : (isnan(b) ? a : max(a, b)); }
float16_t nvx_hmin(float16_t a, float16_t b) { return isnan(a) ? b : (isnan(b) ? a : min(a, b)); }
// f32 -> f16 round-to-nearest-even in a form the compiler cannot fuse into the producing operation
// cvt.rn.ftz.f32.f64 from the double's bits: round to nearest even, results below the f32 normal range flushed to
// signed zero (ftz), overflow to infinity, NaN -> CUDA's canonical 0x7fffffff
float nvx_d2f(uint64_t b)
{
    uint hi = uint(b >> 32), lo = uint(b), s = hi & 0x80000000u, mh = hi & 0xfffffu;
    int e = int((hi >> 20) & 0x7ffu);
    if (e == 0x7ff)
        return uintBitsToFloat((mh | lo) != 0u ? 0x7fffffffu : (s | 0x7f800000u));
    int fe = e - 1023 + 127;
    if (fe >= 255)
        return uintBitsToFloat(s | 0x7f800000u);
    uint m = (mh << 3) | (lo >> 29), rem = lo & 0x1fffffffu;
    if (fe <= 0) {
        // f32 rounds below 2^-126 on the subnormal grid (2^-149): with fe == 0 and m == 0x7fffff the value is at least
        // 2^-126 - 2^-150 and rounds (ties to even) to the smallest normal, which survives ftz; everything else flushes
        bool up = fe == 0 && m == 0x7fffffu;
        return uintBitsToFloat(up ? (s | 0x00800000u) : s);
    }
    uint r = (uint(fe) << 23) | m;
    if (rem > 0x10000000u || (rem == 0x10000000u && (m & 1u) != 0u))
        r += 1u;                                   // a carry into the exponent is correct rounding (up to infinity)
    return uintBitsToFloat(s | r);
}
uint16_t nvx_f2h(float x) { float y = uintBitsToFloat(floatBitsToUint(x) ^ nvx_opaque_zero()); return float16BitsToUint16(float16_t(y)); }

// ---------------------------------------------------------------- textures
uint nvx_ti(uint64_t h) { return uint(h) & 0xfffffu; }
uint nvx_si(uint64_t h) { return (uint(h) >> 20) & 0xfffu; }
ivec2 nvx_clamp_coord(ivec2 c, ivec2 sz, uint smp)
{
    // CLAMP_TO_EDGE (2) is what NGX samplers use; REPEAT/MIRROR fall back to clamp as well (logged by the layer)
    return clamp(c, ivec2(0), sz - 1);
}
bool nvx_border(uint smp) { return nvx_info.smp[smp].y == 3u; }

// UI-copy shadows, NVXRT_SHADOW=c (compiled into the kernels that read or write those images: NVX_SHADOW8, see ptxc.SHADOW8_ENTRIES):
// images the layer keeps as RGBA8 shadows of DLSS-G data that is an 8-bit value stored to RGBA16F
// (formatted stores round f32 -> f16 toward zero on RDNA). Sampled slots of such shadows have class 3: a fetched texel
// k becomes the fp16 value the original image would hold (RZ of the f32 value k/255), returned as f32: reads are exact.
// f32 -> f16 round toward zero, as f32, for |x| in the f16 normal range (and 0): the low 13 mantissa bits cleared. (Not
// via a float16_t round trip: the compiler may fold f32(f16(x)) back to x.)
float nvx_rz16n(float x) { return uintBitsToFloat(floatBitsToUint(x) & 0xffffe000u); }
// a fetched texel t of an RGBA8 image -> the fp16 RZ value of k/255: the texture unit may return unorm8 at reduced
// precision, so k is recovered first (|t - k/255| < 2^-12 for every k, far below half a step of 1/255); k/255 lies
// >= 32 f32 ulps away from every f16 grid point, so an approximate division truncates to the same value
float nvx_u8rz16(float t)
{
    precise float k = floor(t * 255.0 + 0.5);
    precise float y = k / 255.0;
    return nvx_rz16n(y);
}
vec4 nvx_rz16(vec4 v) { return vec4(nvx_u8rz16(v.x), nvx_u8rz16(v.y), nvx_u8rz16(v.z), nvx_u8rz16(v.w)); }
bool nvx_shadow8(uint ti) { return nvx_info.timg_class[ti & 0xffffu] == 3u; }
// true when the RGBA8 shadow reproduces what an RGBA16F store of x reads back: x rounds to some k/255 and
// RZ_f16(x) == RZ_f16(k/255), compared bitwise (-0, NaN, values outside [0, 1] and tiny non-zero values fail)
bool nvx_is8(float x)
{
    precise float k = floor(clamp(x, 0.0, 1.0) * 255.0 + 0.5);
    precise float y = k / 255.0;
    return floatBitsToUint(nvx_rz16n(y)) == floatBitsToUint(nvx_rz16n(x));
}

#ifndef NVX_SW_CLAMP
// Exact texel fetch at integer coordinates: the texture unit applies the sampler's out-of-range rule (clamp to edge,
// or border 0) to the texel centre c + 0.5 of a NEAREST unnormalized-coordinate sampler.
vec4 nvx_fetch_n(uint ti, uint si, ivec2 c)
{
#ifdef NVX_SHADOW8
    vec4 r = textureLod(sampler2D(nvx_tex[nonuniformEXT(ti)], nvx_smpn[nonuniformEXT(si)]), vec2(c) + 0.5, 0.0);
    [[dont_flatten]] if (nvx_shadow8(ti)) r = nvx_rz16(r);
    return r;
#else
    return textureLod(sampler2D(nvx_tex[nonuniformEXT(ti)], nvx_smpn[nonuniformEXT(si)]), vec2(c) + 0.5, 0.0);
#endif
}
uvec4 nvx_ufetch_n(uint ti, uint si, ivec2 c)
{
    return textureLod(usampler2D(nvx_utex[nonuniformEXT(ti)], nvx_smpn[nonuniformEXT(si)]), vec2(c) + 0.5, 0.0);
}
#else
vec4 nvx_fetch_n(uint ti, uint si, ivec2 c)
{
    ivec2 sz = textureSize(nvx_tex[nonuniformEXT(ti)], 0);
    if (nvx_border(si) && (any(lessThan(c, ivec2(0))) || any(greaterThanEqual(c, sz)))) return vec4(0.0);
#ifdef NVX_SHADOW8
    vec4 r = texelFetch(nvx_tex[nonuniformEXT(ti)], nvx_clamp_coord(c, sz, si), 0);
    [[dont_flatten]] if (nvx_shadow8(ti)) r = nvx_rz16(r);
    return r;
#else
    return texelFetch(nvx_tex[nonuniformEXT(ti)], nvx_clamp_coord(c, sz, si), 0);
#endif
}
uvec4 nvx_ufetch_n(uint ti, uint si, ivec2 c)
{
    ivec2 sz = textureSize(nvx_utex[nonuniformEXT(ti)], 0);
    if (nvx_border(si) && (any(lessThan(c, ivec2(0))) || any(greaterThanEqual(c, sz)))) return uvec4(0u);
    return texelFetch(nvx_utex[nonuniformEXT(ti)], nvx_clamp_coord(c, sz, si), 0);
}
#endif
vec4 nvx_tex_fetch_i(uint64_t h, int x, int y) { return nvx_fetch_n(nvx_ti(h), nvx_si(h), ivec2(x, y)); }
uvec4 nvx_utex_fetch_i(uint64_t h, int x, int y) { return nvx_ufetch_n(nvx_ti(h), nvx_si(h), ivec2(x, y)); }
#ifdef NVX_HW_FILTER
// Fast mode (NVXRT_TEX=hw; not the exact model): linear-filtered fetches go to the texture unit
// (RDNA's own bilinear weights instead of CUDA's 8-bit fixed-point ones). Point samplers keep the exact path.
vec4 nvx_tex2d(uint64_t h, float u, float v)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    uint filt = nvx_info.smp[si].x & 0xfu;
    if (filt == 0u) {
        ivec2 sz = textureSize(nvx_tex[nonuniformEXT(ti)], 0);
        precise float xu = u * float(sz.x);
        precise float yv = v * float(sz.y);
        return nvx_fetch_n(ti, si, ivec2(int(floor(xu)), int(floor(yv))));
    }
    return textureLod(sampler2D(nvx_tex[nonuniformEXT(ti)], nvx_smp[nonuniformEXT(si)]), vec2(u, v), 0.0);
}
vec4 nvx_tex2d_m(uint64_t h, float u, float v, uint m) { return nvx_tex2d(h, u, v); }
#else
// Filtered fetch with normalized coordinates and CUDA's 8-bit fixed-point bilinear weights.
vec4 nvx_tex2d(uint64_t h, float u, float v)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    ivec2 sz = textureSize(nvx_tex[nonuniformEXT(ti)], 0);
    uint filt = nvx_info.smp[si].x & 0xfu;
    precise float xu = u * float(sz.x);
    precise float yv = v * float(sz.y);
    if (filt == 0u)
        return nvx_fetch_n(ti, si, ivec2(int(floor(xu)), int(floor(yv))));
    precise float xb = xu - 0.5;
    precise float yb = yv - 0.5;
    precise float xq = floor(xb * 256.0 + 0.5);
    precise float yq = floor(yb * 256.0 + 0.5);
    precise float x0f = floor(xq * (1.0 / 256.0));
    precise float y0f = floor(yq * (1.0 / 256.0));
    precise float a = (xq - x0f * 256.0) * (1.0 / 256.0);
    precise float b = (yq - y0f * 256.0) * (1.0 / 256.0);
    ivec2 c0 = ivec2(int(x0f), int(y0f));
    ivec2 c1 = c0 + 1;
    vec4 t00 = nvx_fetch_n(ti, si, c0);
    vec4 t10 = nvx_fetch_n(ti, si, ivec2(c1.x, c0.y));
    vec4 t01 = nvx_fetch_n(ti, si, ivec2(c0.x, c1.y));
    vec4 t11 = nvx_fetch_n(ti, si, c1);
    precise vec4 top = fma(t10 - t00, vec4(a), t00);
    precise vec4 bot = fma(t11 - t01, vec4(a), t01);
    precise vec4 r = fma(bot - top, vec4(b), top);
    return r;
}
#ifdef NVX_SHADOW8
#define NVX_SHADOW8_FIX(q) [[dont_flatten]] if (nvx_shadow8(ti)) q = nvx_rz16(q);
#else
#define NVX_SHADOW8_FIX(q)
#endif
// nvx_tex2d for the channels in mask m only: per used channel one textureGather of the 2x2 footprint
// (texel centre of the footprint as coordinate, so the hardware picks exactly c0..c0+1), then the same arithmetic
vec4 nvx_tex2d_m(uint64_t h, float u, float v, uint m)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    ivec2 sz = textureSize(nvx_tex[nonuniformEXT(ti)], 0);
    uint filt = nvx_info.smp[si].x & 0xfu;
    precise float xu = u * float(sz.x);
    precise float yv = v * float(sz.y);
    if (filt == 0u)
        return nvx_fetch_n(ti, si, ivec2(int(floor(xu)), int(floor(yv))));
    precise float xb = xu - 0.5;
    precise float yb = yv - 0.5;
    precise float xq = floor(xb * 256.0 + 0.5);
    precise float yq = floor(yb * 256.0 + 0.5);
    precise float x0f = floor(xq * (1.0 / 256.0));
    precise float y0f = floor(yq * (1.0 / 256.0));
    precise float a = (xq - x0f * 256.0) * (1.0 / 256.0);
    precise float b = (yq - y0f * 256.0) * (1.0 / 256.0);
    vec2 g = (vec2(x0f, y0f) + 1.0) / vec2(sz);
    vec4 r = vec4(0.0);
    vec4 q;
#define NVX_GATHER_CH(k, c) \
    if ((m & (1u << k)) != 0u) { \
        q = textureGather(sampler2D(nvx_tex[nonuniformEXT(ti)], nvx_smp[nonuniformEXT(si)]), g, k); \
        NVX_SHADOW8_FIX(q) \
        precise float top_ = fma(q.z - q.w, a, q.w); \
        precise float bot_ = fma(q.y - q.x, a, q.x); \
        precise float r_ = fma(bot_ - top_, b, top_); \
        r.c = r_; \
    }
    NVX_GATHER_CH(0, x)
    NVX_GATHER_CH(1, y)
    NVX_GATHER_CH(2, z)
    NVX_GATHER_CH(3, w)
#undef NVX_GATHER_CH
    return r;
}
#endif
uvec4 nvx_utex2d(uint64_t h, float u, float v)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    ivec2 sz = textureSize(nvx_utex[nonuniformEXT(ti)], 0);
    precise float xu = u * float(sz.x);
    precise float yv = v * float(sz.y);
    return nvx_ufetch_n(ti, si, ivec2(int(floor(xu)), int(floor(yv))));
}
vec4 nvx_tex2d_level(uint64_t h, float u, float v, float lod)
{
    uint ti = nvx_ti(h), si = nvx_si(h);
    if (lod == 0.0) return nvx_tex2d(h, u, v);
    return textureLod(sampler2D(nvx_tex[nonuniformEXT(ti)], nvx_smp[nonuniformEXT(si)]), vec2(u, v), lod);
}

// ---------------------------------------------------------------- surfaces (".zero": out-of-range accesses are dropped / read 0)
uint nvx_sclass(uint64_t h) { return nvx_info.simg_class[uint(h) & 0xffffu] & 0xffu; }
void nvx_sust_p(uint64_t h, int x, int y, uvec4 d)
{
    uint s = uint(h) & 0xfffffu;
    uint cls = nvx_sclass(h);
    if (cls == 1u) {
        ivec2 sz = imageSize(nvx_uimg[nonuniformEXT(s)]);
        if (x >= 0 && y >= 0 && x < sz.x && y < sz.y) imageStore(nvx_uimg[nonuniformEXT(s)], ivec2(x, y), d);
    } else if (cls == 2u) {
        ivec2 sz = imageSize(nvx_iimg[nonuniformEXT(s)]);
        if (x >= 0 && y >= 0 && x < sz.x && y < sz.y) imageStore(nvx_iimg[nonuniformEXT(s)], ivec2(x, y), ivec4(d));
    } else {
        ivec2 sz = imageSize(nvx_img[nonuniformEXT(s)]);
#ifdef NVX_SHADOW8
        if (x >= 0 && y >= 0 && x < sz.x && y < sz.y) {
            vec4 v = uintBitsToFloat(d);
            // class 3: RGBA8 shadow (UI copies); a value that is not an 8-bit quantity raises the flag
            [[dont_flatten]] if (cls == 3u && !(nvx_is8(v.x) && nvx_is8(v.y) && nvx_is8(v.z) && nvx_is8(v.w))) atomicOr(nvx_flags.w[0], 1u);
            imageStore(nvx_img[nonuniformEXT(s)], ivec2(x, y), v);
        }
#else
        if (x >= 0 && y >= 0 && x < sz.x && y < sz.y) imageStore(nvx_img[nonuniformEXT(s)], ivec2(x, y), uintBitsToFloat(d));
#endif
    }
}
// raw load of 16 bytes at byte x-coordinate (only 16-byte texel formats are supported)
uvec4 nvx_suld_b_v4(uint64_t h, int xb, int y)
{
    uint s = uint(h) & 0xfffffu;
    uint cls = nvx_sclass(h);
    int x = xb >> 4;
    if (cls == 1u) {
        ivec2 sz = imageSize(nvx_uimg[nonuniformEXT(s)]);
        if (x >= 0 && y >= 0 && x < sz.x && y < sz.y) return imageLoad(nvx_uimg[nonuniformEXT(s)], ivec2(x, y));
        return uvec4(0u);
    }
    ivec2 sz = imageSize(nvx_img[nonuniformEXT(s)]);
    if (x >= 0 && y >= 0 && x < sz.x && y < sz.y) return floatBitsToUint(imageLoad(nvx_img[nonuniformEXT(s)], ivec2(x, y)));
    return uvec4(0u);
}

// raw load of one texel at byte x-coordinate for any 8/16/32-bit component format: the layer stores the storage view's
// format in bits 8.. of its class word (kind 1 unorm, 2 snorm, 3 float, 4 uint, 5 sint; bytes per component; number of
// components; BGRA order), and the raw bits are rebuilt from the typed load (exact: every such value converts back).
// clampc: .clamp (coordinates clamped to the surface), else out-of-range loads read 0 (.zero / .trap)
uvec4 nvx_suld_b(uint64_t h, int xb, int y, uint esz, bool clampc)
{
    uint s = uint(h) & 0xfffffu;
    uint info = nvx_info.simg_class[uint(h) & 0xffffu];
    uint cls = info & 0xffu, kind = (info >> 8) & 0xfu, cb = max((info >> 12) & 0xfu, 1u), nc = max((info >> 16) & 0xfu, 1u);
    int x = xb / int(cb * nc);
    ivec2 sz = cls == 1u ? imageSize(nvx_uimg[nonuniformEXT(s)]) :
               cls == 2u ? imageSize(nvx_iimg[nonuniformEXT(s)]) : imageSize(nvx_img[nonuniformEXT(s)]);
    if (clampc) {
        x = clamp(x, 0, sz.x - 1);
        y = clamp(y, 0, sz.y - 1);
    } else if (x < 0 || y < 0 || x >= sz.x || y >= sz.y) {
        return uvec4(0u);
    }
    uint mask = cb >= 4u ? 0xffffffffu : (1u << (8u * cb)) - 1u;
    uvec4 r;
    if (cls == 1u) {
        r = imageLoad(nvx_uimg[nonuniformEXT(s)], ivec2(x, y)) & mask;
    } else if (cls == 2u) {
        r = uvec4(imageLoad(nvx_iimg[nonuniformEXT(s)], ivec2(x, y))) & mask;
    } else {
        vec4 v = imageLoad(nvx_img[nonuniformEXT(s)], ivec2(x, y));
        if (kind == 1u) {
            r = uvec4(round(v * float(mask)));
        } else if (kind == 2u) {
            r = uvec4(ivec4(round(v * float(mask >> 1)))) & mask;
        } else if (cb == 2u) {
            r = uvec4(packHalf2x16(v.xy) & 0xffffu, packHalf2x16(v.xy) >> 16, packHalf2x16(v.zw) & 0xffffu, packHalf2x16(v.zw) >> 16);
        } else {
            r = floatBitsToUint(v);
        }
    }
    if (((info >> 20) & 1u) != 0u)
        r = r.zyxw;   // BGRA: memory order B, G, R, A
    return r;
}

// raw store of one texel of n components at byte x-coordinate (inverse of nvx_suld_b; the view's format from the layer)
void nvx_sust_b(uint64_t h, int xb, int y, uvec4 d, uint esz, uint n, bool clampc)
{
    uint s = uint(h) & 0xfffffu;
    uint info = nvx_info.simg_class[uint(h) & 0xffffu];
    uint cls = info & 0xffu, kind = (info >> 8) & 0xfu, cb = max((info >> 12) & 0xfu, 1u), nc = max((info >> 16) & 0xfu, 1u);
    int x = xb / int(cb * nc);
    ivec2 sz = cls == 1u ? imageSize(nvx_uimg[nonuniformEXT(s)]) :
               cls == 2u ? imageSize(nvx_iimg[nonuniformEXT(s)]) : imageSize(nvx_img[nonuniformEXT(s)]);
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
        imageStore(nvx_uimg[nonuniformEXT(s)], ivec2(x, y), d);
    } else if (cls == 2u) {
        uint sh = 32u - 8u * cb;
        imageStore(nvx_iimg[nonuniformEXT(s)], ivec2(x, y), ivec4(d << sh) >> int(sh));
    } else {
        vec4 v;
        if (kind == 1u) {
            v = vec4(d) / float(mask);
        } else if (kind == 2u) {
            uint sh = 32u - 8u * cb;
            v = max(vec4(ivec4(d << sh) >> int(sh)) / float(mask >> 1), vec4(-1.0));
        } else if (cb == 2u) {
            v = vec4(unpackHalf2x16(d.x | (d.y << 16)), unpackHalf2x16(d.z | (d.w << 16)));
        } else {
            v = uintBitsToFloat(d);
        }
        imageStore(nvx_img[nonuniformEXT(s)], ivec2(x, y), v);
    }
}

// ---------------------------------------------------------------- warp
uint nvx_shfl_down(uint a, uint b, uint c, out bool p)
{
    uint lane = gl_SubgroupInvocationID;
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    uint j = lane + (b & 31u);
    p = j <= maxl;
    if (!p) j = lane;
    return subgroupShuffle(a, j);
}
uint nvx_shfl_bfly(uint a, uint b, uint c, out bool p)
{
    uint lane = gl_SubgroupInvocationID;
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    uint j = lane ^ (b & 31u);
    p = j <= maxl;
    if (!p) j = lane;
    return subgroupShuffle(a, j);
}
uint nvx_shfl_up(uint a, uint b, uint c, out bool p)
{
    uint lane = gl_SubgroupInvocationID;
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    int j = int(lane) - int(b & 31u);
    p = j >= int(maxl);
    if (!p) j = int(lane);
    return subgroupShuffle(a, uint(j));
}
// Constant-delta forms (the translator uses them when the PTX delta is an immediate): relative shuffles let ACO use
// DPP / swizzle / permlane instead of ds_bpermute. Identical results: when p holds the source lane is lane+b
// (lane-b, lane^b) inside the segment, which the relative shuffle reads; otherwise the lane keeps its own value.
uint nvx_shfl_down_k(uint a, uint b, uint c, out bool p)
{
    uint lane = gl_SubgroupInvocationID;
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    uint bb = b & 31u;
    p = lane + bb <= maxl;
    uint r = subgroupShuffleDown(a, bb);
    return p ? r : a;
}
uint nvx_shfl_bfly_k(uint a, uint b, uint c, out bool p)
{
    uint lane = gl_SubgroupInvocationID;
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    uint bb = b & 31u;
    p = (lane ^ bb) <= maxl;
    uint r = subgroupShuffleXor(a, bb);
    return p ? r : a;
}
uint nvx_shfl_up_k(uint a, uint b, uint c, out bool p)
{
    uint lane = gl_SubgroupInvocationID;
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    uint bb = b & 31u;
    p = int(lane) - int(bb) >= int(maxl);
    uint r = subgroupShuffleUp(a, bb);
    return p ? r : a;
}
uint nvx_shfl_idx(uint a, uint b, uint c, out bool p)
{
    uint lane = gl_SubgroupInvocationID;
    uint segmask = (c >> 8) & 31u, cval = c & 31u;
    uint maxl = (lane & segmask) | (cval & ~segmask);
    uint minl = lane & segmask;
    uint j = minl | (b & 31u & ~segmask);
    p = j <= maxl;
    if (!p) j = lane;
    return subgroupShuffle(a, j);
}
