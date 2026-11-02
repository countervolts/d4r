// d4r engine, preset M (rrlite) on AMD's D3D12 driver: one Swin block over 4x4-token windows, register-resident.
// The D3D12 build of ../layer_m.comp in the arithmetic compile_m.py packages for Vulkan (M_RELAXED, M_RESIDUAL_F32,
// TEXL; WPW windows per wave, one by default here), for the network layers (KIND 0 plain, 1 encoder with the 2x2 patch merge, 2 decoder
// with the patch expand and skip). See layer_m.comp for the algorithm; this file only differs in how it reaches the
// matrices.
//
// AGS WMMA (ags_wmma.hlsli) hands the matrices over as registers, and on gfx1201 they have the layout layer_m.comp's
// GLSL relies on (measured with the AGS probes on AMD's D3D12 driver 32.0.32015, 2026-10-23; the same as RADV's):
//   f32 accumulator, one element per register:      element i of lane l is (row 8*(l>>4) + i, col l & 15)
//   f16 accumulator and B operand, two per register: the same elements, (2j, 2j+1) in register j
//   FP8 B operand, four per register (registers 0, 1): the same elements
//   FP8 / f16 A operand:                             element i of lane l is (row l & 15, col 8*(l>>4) + i)
// so a product's result is the next product's B operand element for element, and q's registers read as an A operand
// are q transposed (the scores' Q^T Q). Every rounding goes through AGS conversions, which are the hardware's:
// f32 -> f16 accumulator to nearest even, f32 or f16 -> FP8 B to nearest even with saturation at +-448 (NaN stays
// NaN), i.e. SAT2_ / saturatedConvertEXT. HLSL's own f32 -> f16 conversion truncates on this driver, so the few scalar
// conversions use rne16.
//
// Bindings (raw buffers): t0 the layer's parameter block (W, H, sx, sy, o[72]), t1 the FP8 weights (test_m.pack8
// tiles), t2 the f16 vectors, t3 the input tokens, t4 the decoders' skip tokens, u5 the block output, u6 the encoders'
// merged tokens. One workgroup is an 8x8-token block of four windows; a wave handles WPW of them.
#ifndef C
#define C 64
#endif
#ifndef NH
#define NH 2
#endif
#ifndef KIND
#define KIND 0
#endif
#ifndef XC
#define XC 16
#endif
// one window per wave (as K on Windows): the windows of a wave share its weight-tile loads, but with two the f32 residual
// alone takes 128 registers, and the dispatches of the small layers leave a single wave per SIMD to hide the loads
// (RX 9070 XT, 1280x720 capture: the ten layers 555 -> 385 us)
#ifndef WPW
#define WPW 1
#endif
#if KIND > 4 || (WPW != 1 && WPW != 2 && WPW != 4) || (KIND >= 3 && WPW > 2)
#error layer_m.hlsl: KIND 0-4, one, two or four windows per wave (L's stages: at most two, their 64 pixels one per invocation)
#endif
#define NT (C / 16)
// weight tiles requested per group (layer.comp PREFETCH_K / PREFETCH_N): PF k32 steps (pairs of K tiles, both output
// tiles), PFN output tiles of the projections. RX 9070 XT, the ten layers at 854x480 -> 2560x1440 (1280x720): PF 2,
// PFN 4 594 us (458); PF 4, PFN 8 515 (380), no spills (the tube 180 registers); 8 / 8 or 4 / 16 no faster
#ifndef PF
#define PF 4
#endif
#ifndef PFN
#define PFN 8
#endif
#define S4 (4 * C + 8)              // row stride of the 4C rows (patch expand / merge), bytes
#define SLO (XC + 8)                // row stride of the low-resolution token image, bytes
#define MERGE (KIND == 1 || KIND == 3)
#define EXPAND (KIND == 2 || KIND == 4)
// PLANES (KIND 3 and 4, preset L's input and expansion stages): the half-resolution tokens in NGX's layout, planes of
// 32 channels ([XC/32][H/2][W/2][32]), for a network that runs outside the engine (the native HIP layers, as
// layer_m.comp PLANES and enc0_m_planes.dxil's token buffer)
#ifndef PLANES
#define PLANES 0
#endif
// shared memory regions, in words. VSH: per wave and head the V tiles (512 bytes a tile), in a buffer that is dead
// during the attention (layer_m.comp)
#if KIND == 0
#define SH_WORDS 1024
#define VSH_W 0
#elif MERGE
#define NB_W 0                      // 16 rows of slack, then the 16 merged rows (FP8)
#define VSH_W 0
#define SH_WORDS (32 * S4 / 4)
#if SH_WORDS < 1024
#error layer_m.hlsl: the V tiles no longer fit in NB
#endif
#else
#define NLI_W 0                     // the 16 low-resolution tokens (FP8)
#define NEX_W (16 * SLO / 4)        // their expanded 4C columns (FP8)
#define VSH_W NEX_W
#if KIND == 4
// L's expansion: the V tiles past NEX (C 32: NEX is smaller than they are), then the head's input HO in their place
#define HS 40u                      // HO's row stride, binary16 values
#define SH_WORDS (NEX_W + 1024)
#if SH_WORDS < 64 * 40 / 2
#error layer_m.hlsl: HO no longer fits
#endif
#else
#define SH_WORDS ((16 * SLO + 16 * S4) / 4)
#endif
#endif
#define D4R_WMMA_SHARED_WORDS SH_WORDS
#include "ags_wmma.hlsli"

// L_FUSED (KIND 3, vkd3d-proton): L's input stage with its feature slots gathered here (enc0_l_fused.hlsli), whose
// bindings take registers 0-13; the layer's follow
#ifndef L_FUSED
#define L_FUSED 0
#endif
#if L_FUSED && (KIND != 3 || WPW != 2)
#error layer_m.hlsl: L_FUSED is preset L's input stage (KIND 3) with its 64 pixels one per invocation (WPW 2)
#endif
#if L_FUSED
ByteAddressBuffer prm : register(t14);
ByteAddressBuffer w8 : register(t15);
ByteAddressBuffer w16 : register(t16);
ByteAddressBuffer inb : register(t17);
ByteAddressBuffer skb : register(t18);
RWByteAddressBuffer out0 : register(u19);
RWByteAddressBuffer out1 : register(u20);
#else
ByteAddressBuffer prm : register(t0);
ByteAddressBuffer w8 : register(t1);
ByteAddressBuffer w16 : register(t2);
ByteAddressBuffer inb : register(t3);
ByteAddressBuffer skb : register(t4);
RWByteAddressBuffer out0 : register(u5);
RWByteAddressBuffer out1 : register(u6);
#endif
#if KIND == 4
// L's expansion head: t7 the expansion parameters (MDec0Params: render size at 0, sigma at 48, rho at 52), u8-u11 the
// feature (SNORM8x4), gates, mix (UNORM8x4) and covariance (RGBA16F) of every render pixel
ByteAddressBuffer dp : register(t7);
RWTexture2D<float4> featOut : register(u8);
RWTexture2D<float4> gateOut : register(u9);
RWTexture2D<float4> mixOut : register(u10);
RWTexture2D<float4> covOut : register(u11);
#endif

uint po(uint i) { return prm.Load(16u + 4u * i); }
// A binary16 value that reaches an AGS conversion (to f32 or FP8) must keep its binary16 rounding, but the driver
// folds such a pair (a binary16 add or multiply, or the f32 -> binary16 conversion of halfOf, then the conversion) into
// one operation without it, precise or not (measured: the residual x0 + b_o entered the output projection as the f32
// sum, and q8 of a binary16 accumulator was the f32 sum's FP8 code). halfOf's registers pass an OR with a zero the
// driver cannot prove (parameter word o[71], never assigned); binary16 arithmetic that feeds a conversion is done in
// f32 and rounded with rne16 instead (a sum or product of two binary16 values is exact in f32, or rounds to the same
// binary16 value), which leaves nothing to fold. (The same OR on binary16 arithmetic made the driver miscompile the
// 128-channel blocks with two windows per wave.)
#define KEEP_ZERO po(71)
int mirror(int v, int n) { v = abs(v); return min(v, 2 * n - 2 - v); }

// ---------------------------------------------------------------- matrices
#include "m_wmma.hlsli"
// the packed fixed-16 FP8 weight tiles (test_m.pack8): tile (ks, moff) of W[K][N] read as the A operand
uint2 lda8(uint base, uint ks, uint stride, uint moff)
{
    const Matrix16 m = matrixLoad(MatrixA8, w8, base + 256u * (ks * (stride / 16u) + moff / 16u), 16u);
    return uint2(m.reg[0], m.reg[1]);
}
// a tiled f16 vector (16 equal columns) as a binary16 accumulator whose rows are its channels
uint4 bias16(uint off, uint stride)
{
    const Matrix16 m = matrixLoad(MatrixAcc16 | 0x80, w16, off * 2u, stride * 2u);
    return uint4(m.reg[0], m.reg[1], m.reg[2], m.reg[3]);
}

// M_ACC32: the products' accumulators (Q, V, the MLP's hidden layer, the patch merge and expand; ACC32_QV, _MLP, _PATCH
// one at a time, ACC32_OUT the block output) stay f32 through all their 32-deep steps and become FP8 in one conversion,
// as RADV compiles layer_m.comp's relaxed build (it folds the register-only f32 -> binary16 -> f32 conversions after
// every step away; layer_m.comp M_RELAXED). Rounding them to binary16 after every step, as the M_EXACT build must, is
// 16 conversions per two products on AMD's D3D12 driver. The relaxed build's default for the layers it speeds up:
// RX 9070 XT, 854x480 -> 2560x1440, enc1 108 -> 97 us, dec2 64 -> 58, dec1 103 -> 93; enc2 69 -> 87 and every tube
// block 28 -> 35 (the QV and MLP parts each), so those keep the binary16 steps
#ifndef M_ACC32
#define M_ACC32 (!M_EXACT && (C == 64 || KIND == 2 || KIND >= 3))
#endif
#ifndef ACC32_QV
#define ACC32_QV M_ACC32
#endif
#ifndef ACC32_MLP
#define ACC32_MLP M_ACC32
#endif
#ifndef ACC32_PATCH
#define ACC32_PATCH M_ACC32
#endif
#ifndef ACC32_OUT
#define ACC32_OUT M_ACC32
#endif
// either accumulator: f32 (Matrix16) or binary16 (uint4)
Matrix16 accInit(Matrix16 z, uint4 h) { return f32of(h); }
uint4 accInit(uint4 z, uint4 h) { return h; }
Matrix16 k32(Matrix16 d, uint2 a0, uint2 b0, uint2 a1, uint2 b1) { return mma8(a1, b1, mma8(a0, b0, d)); }
uint4 k32(uint4 d, uint2 a0, uint2 b0, uint2 a1, uint2 b1) { return k32h(d, a0, b0, a1, b1); }
float accAt(Matrix16 m, uint i) { return asfloat(m.reg[i]); }
float accAt(uint4 m, uint i) { return float(asfloat16(uint16_t((m[i >> 1] >> (16u * (i & 1u))) & 0xffffu))); }
uint4 accHalf(Matrix16 m) { return halfOf(m); }
uint4 accHalf(uint4 m) { return m; }
#if ACC32_QV
#define ACC_QV Matrix16
#else
#define ACC_QV uint4
#endif
#if ACC32_MLP
#define ACC_MLP Matrix16
#else
#define ACC_MLP uint4
#endif
#if ACC32_PATCH
#define ACC_PATCH Matrix16
#else
#define ACC_PATCH uint4
#endif
#define ACC_ZERO(T) ((T)0)

// ---------------------------------------------------------------- scalars
// v rounded to the nearest binary16 (ties to even), as dialect_hlsl.h d4rRne16; the conversion is then exact
float16_t rne16(float v)
{
    const uint bits = asuint(v);
    precise float subnormal = (v + 0.75) - 0.75;
    return (float16_t)(abs(v) < 6.103515625e-05 ? subnormal : asfloat((bits + 4095u + ((bits >> 13u) & 1u)) & 0xffffe000u));
}
// binary16 arithmetic through f32 and rne16, packed (the conversion of a rounded value is exact)
float2 r16(float2 v) { return float2(rne16(v.x), rne16(v.y)); }
uint pk16(float2 v) { return f32tof16(float(rne16(v.x))) | (f32tof16(float(rne16(v.y))) << 16); }
half2 h2(uint u) { return asfloat16(uint16_t2(uint16_t(u & 0xffffu), uint16_t(u >> 16))); }
uint u2(half2 h) { const uint16_t2 b = asuint16(h); return uint(b.x) | (uint(b.y) << 16); }
uint sh16(uint u) { return WaveReadLaneAt(u, WaveGetLaneIndex() ^ 16u); }
half2 sh16(half2 h) { return h2(sh16(u2(h))); }
// four FP8 codes (a word) as binary16: the code's bits moved to binary16's, then scaled by 2^8 (exact). vkd3d-proton:
// the AGS conversion of the word (patch 0010: RADV's hardware FP8 conversion, as the Vulkan build's), the same values
// for every code but NaN, which the network's saturating conversions never produce
#if D4R_VKD3D
void fp8x4(uint v, out half2 lo, out half2 hi)
{
    const Matrix16 r = convert(MatrixB8, MatrixAcc16, 0, m16(uint2(v, 0u)));
    lo = h2(r.reg[0]); hi = h2(r.reg[1]);
}
#else
void fp8x4(uint v, out half2 lo, out half2 hi)
{
    const uint a = ((v & 0x7fu) << 7) | ((v & 0x80u) << 8) | ((v & 0x7f00u) << 15) | ((v & 0x8000u) << 16);
    const uint b = (((v >> 16) & 0x7fu) << 7) | (((v >> 16) & 0x80u) << 8) | ((v >> 1) & 0x3f800000u) | (v & 0x80000000u);
    lo = h2(a) * half(256.0); hi = h2(b) * half(256.0);
}
#endif
// a lane's eight channels of a tile: four binary16 pairs
struct X8 { half2 p[4]; };
X8 x8(uint2 v) { X8 x; fp8x4(v.x, x.p[0], x.p[1]); fp8x4(v.y, x.p[2], x.p[3]); return x; }
X8 x8(uint4 h) { X8 x; x.p[0] = h2(h.x); x.p[1] = h2(h.y); x.p[2] = h2(h.z); x.p[3] = h2(h.w); return x; }
// NVIDIA's softmax numerator, two neighbouring columns in one 32-bit word (layer_m.comp softmax2)
uint softmax2(uint s)
{
    const half lim = half(0.55615234375), cs = half(0.0972222164273262 / 5.656854152679443);
    const half2 t = clamp(h2(s) * cs, -lim, lim);
    const half2 p = mad(t, mad(t, -t, half(0.92730712890625)), half(1.375));
    return (u2(p) << 5) + 0x7FF88000u;
}
#if NT == 2
#define WH(P, w) P[w]
#elif NT == 4
#define WH(P, w) (P[w] + P[2 + w])
#elif NT == 6
#define WH(P, w) ((P[w] + P[2 + w]) + P[4 + w])
#else
#define WH(P, w) ((P[w] + P[2 + w]) + (P[4 + w] + P[6 + w]))
#endif
// TEXL's tok_rsq: the lane's channels in natural order; per g the tiles of each parity, the two parities, g, halves
float16_t rsqTree(X8 x[NT])
{
#if M_EXACT
    // in f32 with explicit binary16 roundings (r16): the driver drops the rounding of binary16 arithmetic here,
    // precise or not, which shows on the full-precision residual (norm2) though not on FP8 inputs (norm1)
    float2 P0[NT], P1[NT];
    [unroll] for (uint t = 0; t < NT; ++t)
    {
        const float2 a0 = float2(x[t].p[0]), a1 = float2(x[t].p[1]), a2 = float2(x[t].p[2]), a3 = float2(x[t].p[3]);
        P0[t] = r16(r16(a0 * a0) + r16(a1 * a1));
        P1[t] = r16(r16(a2 * a2) + r16(a3 * a3));
    }
#if NT == 2
#define WS(P, w) P[w]
#elif NT == 4
#define WS(P, w) r16(P[w] + P[2 + w])
#elif NT == 6
#define WS(P, w) r16(r16(P[w] + P[2 + w]) + P[4 + w])
#else
#define WS(P, w) r16(r16(P[w] + P[2 + w]) + r16(P[4 + w] + P[6 + w]))
#endif
    const float2 U0 = r16(WS(P0, 0) + WS(P0, 1)), U1 = r16(WS(P1, 0) + WS(P1, 1));
    const float2 A = r16(U0 + U1);
    const uint Ap = pk16(A);
    const float2 S = r16(A + float2(h2(sh16(Ap))));
    const float s = rne16(S.x + S.y);
    const float biased = rne16(s + 0.0001220703125);
    return rne16(rsqrt(biased));     // rsqrt.approx as ZLUDA lowers it (v_rsq_f32)
#else
    // M_RELAXED: binary16 arithmetic as the driver compiles it (layer_m.comp RSQ_TREE)
    half2 P0[NT], P1[NT];
    [unroll] for (uint t = 0; t < NT; ++t)
    {
        P0[t] = x[t].p[0] * x[t].p[0] + x[t].p[1] * x[t].p[1];
        P1[t] = x[t].p[2] * x[t].p[2] + x[t].p[3] * x[t].p[3];
    }
    const half2 A = (WH(P0, 0) + WH(P0, 1)) + (WH(P1, 0) + WH(P1, 1));
    const half2 S = A + sh16(A);
    return rne16(rsqrt(float(S.x + S.y + half(0.0001220703125))));
#endif
}
// the normalised token's tile into FP8: x * (r * g), in f32, saturating (layer_m.comp H1N). M_EXACT: NVIDIA's
// roundings instead (binary16 products), for the byte-identical check against the CUDA path
#ifndef M_EXACT
#define M_EXACT 0
#endif
uint2 h1n(X8 x, float r, uint gw)
{
    const uint4 g = w16.Load4(gw);
#if M_EXACT
    const half rh = half(r);
    uint4 hv;
    [unroll] for (uint i = 0; i < 4; ++i) { const float2 sc = float2(h2(pk16(float(rh) * float2(h2(g[i]))))); hv[i] = pk16(float2(x.p[i]) * sc); }
    return fp8Of(hv);
#endif
    float v[8];
    [unroll] for (uint i = 0; i < 4; ++i)
    {
        const float2 gg = float2(h2(g[i]));
        const float2 xx = float2(x.p[i]);
        v[2 * i] = xx.x * (r * gg.x); v[2 * i + 1] = xx.y * (r * gg.y);
    }
    return fp8Of(f32m(v));
}
// x + d in binary16, x's pairs and the f16 bias vector's eight channels at byte gw. precise: the driver otherwise
// folds a binary16 sum that only reaches an f32 conversion (f32of, the residual entering a product as C) into an f32
// sum, losing the binary16 rounding (as RADV does with register-only conversions, layer_m.comp R16)
uint4 addBias(X8 x, uint gw)
{
    const uint4 d = w16.Load4(gw);
    uint4 r;
#if M_EXACT
    [unroll] for (uint i = 0; i < 4; ++i) r[i] = pk16(float2(x.p[i]) + float2(h2(d[i])));
#else
    [unroll] for (uint i = 0; i < 4; ++i) r[i] = u2(x.p[i] + h2(d[i]));
#endif
    return r;
}
// ZLUDA's key order of the P V product (layer_m.comp ZK): pairs exchanged with the other half-wave
uint4 zk(uint4 m, uint hf)
{
    const bool up = hf != 0u;
    const uint r0 = sh16(up ? m.x : m.z), r1 = sh16(up ? m.y : m.w);
    return uint4(up ? r0 : m.x, up ? m.z : r0, up ? r1 : m.y, up ? m.w : r1);
}

#if KIND == 4
// L's expansion head (layer_m.comp KIND 4): NVIDIA's f32 activations, a sigmoid rounded to binary16 and a tanh that is
// a polynomial below 0.6 and 1 - 2 / (e^2x + 1) above; the head's inputs in natural channel order (L_NOTES.md)
float sigmoidL(float v) { return float(rne16(1.0 / (exp2(v * -1.442695) + 1.0))); }
float tanhL(float t)
{
    const float m = abs(t), s = t * t;
    const float big = m >= 9.0109138 ? 1.0 : mad(1.0 / (exp2(m * 2.88539) + 1.0), -2.0, 1.0);
    return m >= 0.6 ? (t < 0.0 ? -big : big) :
        mad(mad(mad(mad(mad(0.01573968306183815, s, -0.05230396240949631), s, 0.13315297663211823), s, -0.33332768082618713), s, 0.0), t, t);
}
// exact normalized codes, rounded half away from zero as CUDA's formatted stores (normalized_store.glsl)
float4 unorm8Value(float4 v) { v = clamp(select(isnan(v), 0.0, v), 0.0, 1.0); precise float4 scaled = v * 255.0; return floor(scaled + 0.5) * (1.0 / 255.0); }
float4 snorm8Value(float4 v)
{
    v = clamp(select(isnan(v), 0.0, v), -1.0, 1.0);
    precise float4 scaled = abs(v) * 127.0;
    return sign(v) * floor(scaled + 0.5) * (1.0 / 127.0);
}
// binary16 value c of HO
float headIn(uint c) { return float(asfloat16(uint16_t((matrixShared[c >> 1] >> (16u * (c & 1u))) & 0xffffu))); }
#define LH_GATE0 0
#define LH_GATE1 1
#define LH_GATE2 12
#define LH_MIX0 13
#define LH_MIX1 16
#define LH_MIX2 17
#define LH_COVA 20
#define LH_COVB 21
#define LH_RHO 24
#define LH_FEAT0 4
#define LH_FEAT1 5
#define LH_FEAT2 8
#define LH_FEAT3 9
#endif
#define FORW [unroll] for (uint w = 0; w < WPW; ++w)
#ifndef DBGP
#define DBGP 0
#endif
#ifdef DBG
#define DBG_OUT(V, STAGE) if (DBG == STAGE) { FORW { WINV if (inside) { const uint dst = uint(ty * PW + tx) * C + ch8;     [unroll] for (uint n = 0; n < NT; ++n) out0.Store2(dst + 16u * n, V); } } return; }
#else
#define DBG_OUT(V, STAGE)
#endif
#if L_FUSED
#include "enc0_l_fused.hlsli"
#endif
[WaveSize(32)] [numthreads(128 / WPW, 1, 1)]
void main(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint mt0 = (lid.x >> 5) * WPW, lane = lid.x & 31u;
#if L_FUSED
    gatherBlockFeatures(lid.x, gid.xy);
    GroupMemoryBarrierWithGroupSync();
#endif

    const uint row = lane & 15u, hf = lane >> 4, ch8 = 8u * hf;
    const int PW = asint(prm.Load(0)), PH = asint(prm.Load(4));
    const int X0 = 8 * int(gid.x) - asint(prm.Load(8)), Y0 = 8 * int(gid.y) - asint(prm.Load(12));
#define WINV const uint mt = mt0 + w; \
    const uint bxr = 4u * (mt & 1u) + (row & 3u), byr = 4u * (mt >> 1) + (row >> 2); \
    const int tx = X0 + int(bxr), ty = Y0 + int(byr); \
    const bool inside = ty >= 0 && ty < PH && tx >= 0 && tx < PW; \
    const uint iy = uint(mirror(ty, PH)), ix = uint(mirror(tx, PW));
    uint4 res[WPW][NT];      // the residual stream (binary16): rows are channels, columns are tokens
    uint2 h1[WPW][NT];       // the normalised activation (FP8 B operand)
    // ------------------------------------------------------------ patch expand: x0 = q8(E) + skip
#if EXPAND
    FORW {
        WINV
        const uint W2 = uint(PW) / 2u, H2 = uint(PH) / 2u, part = XC / 8u, c0 = (mt * 2u + hf) * part;
        const uint lt = uint(mirror((Y0 >> 1) + int(row >> 2), int(H2)) * int(W2) + mirror((X0 >> 1) + int(row & 3u), PW / 2));
        [unroll] for (uint c = 0; c < part; c += 4u)
#if PLANES
            // NGX's layout: the 32-channel plane, the token, then the channel; a lane's four-byte word stays in its plane
            matrixShared[NLI_W + (row * SLO + c0 + c) / 4u] = inb.Load((((c0 + c) >> 5) * H2 * W2 + lt) * 32u + ((c0 + c) & 31u));
#else
            matrixShared[NLI_W + (row * SLO + c0 + c) / 4u] = inb.Load(lt * XC + c0 + c);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    FORW {
        const uint mt = mt0 + w;
        ACC_PATCH ef[NT];
        [unroll] for (uint n = 0; n < NT; ++n) ef[n] = accInit(ACC_ZERO(ACC_PATCH), bias16(po(69) + 16u * (mt * NT + n), 4u * C));
        [unroll] for (uint kt = 0; kt < XC / 16u; kt += 2u)
        {
            const uint2 b0 = ldsB8(NLI_W * 4u + kt * 16u, SLO), b1 = ldsB8(NLI_W * 4u + (kt + 1u) * 16u, SLO);
            [unroll] for (uint n = 0; n < NT; ++n)
                ef[n] = k32(ef[n], lda8(po(68), kt, 4u * C, 16u * (mt * NT + n)), b0, lda8(po(68), kt + 1u, 4u * C, 16u * (mt * NT + n)), b1);
        }
        [unroll] for (uint n = 0; n < NT; ++n) stsA8(NEX_W * 4u + mt * C + 16u * n, S4, fp8Of(ef[n]));
    }
    GroupMemoryBarrierWithGroupSync();
#endif
    // ------------------------------------------------------------ norm 1: sum of squares, residual and h1
    FORW {
        WINV
        X8 x[NT];
#if EXPAND
        const uint lb0 = ((byr >> 1) * 4u + (bxr >> 1)) * S4 + ((byr & 1u) * 2u + (bxr & 1u)) * C + ch8;
        const uint sb0 = (iy * uint(PW) + ix) * C + ch8;
        [unroll] for (uint t = 0; t < NT; ++t)
        {
            const uint lw = NEX_W + (lb0 + 16u * t) / 4u;
            const X8 e = x8(uint2(matrixShared[lw], matrixShared[lw + 1u])), s = x8(skb.Load2(sb0 + 16u * t));
#if M_EXACT
            [unroll] for (uint i = 0; i < 4; ++i) x[t].p[i] = h2(pk16(float2(e.p[i]) + float2(s.p[i])));
#else
            [unroll] for (uint i = 0; i < 4; ++i) x[t].p[i] = e.p[i] + s.p[i];
#endif
        }
#elif KIND == 3
        // L's input: the embedding, an MLP chunk's shape (32 features -> 32 hidden, ReLU, FP8 -> C), of the pixel's FP8
        // feature slots (enc0_m.comp L_FEATURES: per block, 32 bytes a pixel) straight into the block's input x0. The
        // product's register layout is x8's (a lane's eight channels of one pixel), so x0 needs no shared memory.
#if L_FUSED
        const uint fw = (byr * 8u + bxr) * 8u + ch8 / 4u;
        const uint2 f0 = uint2(LF[fw], LF[fw + 1u]), f1 = uint2(LF[fw + 4u], LF[fw + 5u]);
#else
        const uint fb = ((gid.y * uint((PW + 9) / 8) + gid.x) * 64u + byr * 8u + bxr) * 32u + ch8;
        const uint2 f0 = inb.Load2(fb), f1 = inb.Load2(fb + 16u);
#endif
        uint4 z[2];
        [unroll] for (uint m = 0; m < 2u; ++m)
        {
            z[m] = k32h(bias16(po(61) + 16u * m, 32u), lda8(po(60), 0u, 32u, 16u * m), f0, lda8(po(60), 1u, 32u, 16u * m), f1);
            [unroll] for (uint i = 0; i < 4; ++i) z[m][i] = u2(max(h2(z[m][i]), half2(0.0, 0.0)));
        }
        const uint2 g0 = fp8Of(z[0]), g1 = fp8Of(z[1]);
        [unroll] for (uint t = 0; t < NT; ++t)
            x[t] = x8(k32h(bias16(po(63) + 16u * t, C), lda8(po(62), 0u, C, 16u * t), g0, lda8(po(62), 1u, C, 16u * t), g1));
#else
        const uint ib0 = (iy * uint(PW) + ix) * C + ch8;
        [unroll] for (uint t = 0; t < NT; ++t) x[t] = x8(inb.Load2(ib0 + 16u * t));
#endif
        const float rn = float(rsqTree(x));
        [unroll] for (uint n = 0; n < NT; ++n)
        {
            res[w][n] = addBias(x[n], (po(1) + 16u * n + ch8) * 2u);
            h1[w][n] = h1n(x[n], rn, (po(0) + 16u * n + ch8) * 2u);
        }
    }
    DBG_OUT(h1[w][n], 1)
    DBG_OUT(fp8Of(res[w][n]), 2)
#ifdef DBG
    if (DBG == 6) { FORW { WINV if (inside) { const uint dst = uint(ty * PW + tx) * C + 16u * hf;
        [unroll] for (uint n = 0; n < NT / 2; ++n) out0.Store4(dst + 32u * n, res[w][n]); } } return; }
#endif
#if EXPAND
    GroupMemoryBarrierWithGroupSync();      // every wave has read the block's input (NEX): the V tiles take its place
#endif
    // ------------------------------------------------------------ attention, one head at a time
#ifndef NH0
#define NH0 0
#endif
#ifndef NHRUN
#define NHRUN NH
#endif
    [loop] for (uint hd = NH0; hd < NHRUN; ++hd)
    {
        const uint oq = po(4u + 4u * hd), ov = po(5u + 4u * hd), ob = po(6u + 4u * hd), ow = po(7u + 4u * hd);
        // Q of every window, then V of every window: each weight tile is loaded once for the wave's windows
        uint2 QA[WPW][2];
        {
            ACC_QV q[WPW][2];
            FORW { q[w][0] = ACC_ZERO(ACC_QV); q[w][1] = ACC_ZERO(ACC_QV); }
            [unroll] for (uint k0 = 0; k0 < NT; k0 += 2u * PF)
            {
                uint2 a[PF][2][2];      // weight tiles requested together, before their products (layer.comp PREFETCH_K)
                [unroll] for (uint i = 0; i < PF; ++i) [unroll] for (uint m = 0; m < 2u; ++m)
                    if (k0 + 2u * i < NT) { a[i][m][0] = lda8(oq, k0 + 2u * i, 32u, 16u * m); a[i][m][1] = lda8(oq, k0 + 2u * i + 1u, 32u, 16u * m); }
                [unroll] for (uint i = 0; i < PF; ++i) [unroll] for (uint m = 0; m < 2u; ++m)
                    if (k0 + 2u * i < NT) { const uint kt = k0 + 2u * i; FORW q[w][m] = k32(q[w][m], a[i][m][0], h1[w][kt], a[i][m][1], h1[w][kt + 1u]); }
            }
            FORW { QA[w][0] = fp8Of(q[w][0]); QA[w][1] = fp8Of(q[w][1]); }
        }
        {
            ACC_QV v[WPW][2];
            FORW { v[w][0] = ACC_ZERO(ACC_QV); v[w][1] = ACC_ZERO(ACC_QV); }
            [unroll] for (uint k0 = 0; k0 < NT; k0 += 2u * PF)
            {
                uint2 a[PF][2][2];      // weight tiles requested together, before their products (layer.comp PREFETCH_K)
                [unroll] for (uint i = 0; i < PF; ++i) [unroll] for (uint m = 0; m < 2u; ++m)
                    if (k0 + 2u * i < NT) { a[i][m][0] = lda8(ov, k0 + 2u * i, 32u, 16u * m); a[i][m][1] = lda8(ov, k0 + 2u * i + 1u, 32u, 16u * m); }
                [unroll] for (uint i = 0; i < PF; ++i) [unroll] for (uint m = 0; m < 2u; ++m)
                    if (k0 + 2u * i < NT) { const uint kt = k0 + 2u * i; FORW v[w][m] = k32(v[w][m], a[i][m][0], h1[w][kt], a[i][m][1], h1[w][kt + 1u]); }
            }
            // V of every window to its shared tiles now, so it does not occupy registers during the scores
            FORW {
                const uint mt = mt0 + w;
                matrixSharedStore(MatrixAcc16, VSH_W + (mt * 2u + 0u) * 128u, 8u, m16(accHalf(v[w][0])));
                matrixSharedStore(MatrixAcc16, VSH_W + (mt * 2u + 1u) * 128u, 8u, m16(accHalf(v[w][1])));
            }
            GroupMemoryBarrier();       // the wave's lanes read one another's V through VSH below
        }
        // scores: q8(Q)^T q8(Q) plus the relative position bias (the accumulator holds S^T: this lane holds the keys
        // 8h..8h+7 of one query)
        const uint4 sbias = bias16(ob, 16u);
        uint2 OB[WPW][2];
        FORW {
            const uint mt = mt0 + w;
            const uint4 s = k32h(sbias, QA[w][0], QA[w][0], QA[w][1], QA[w][1]);
            half2 G[4];
            [unroll] for (uint i = 0; i < 4; ++i) G[i] = h2(softmax2(s[i]));
            precise half2 V0 = G[0] + sh16(G[0]), V1 = G[1] + sh16(G[1]);
            precise half2 V2 = G[2] + sh16(G[2]), V3 = G[3] + sh16(G[3]);
            precise half2 VV = (V0 + V1) + (V2 + V3);
            precise half vs = VV.x + VV.y;
            const half rc = rne16(1.0 / float(vs));
            uint4 pb;
            [unroll] for (uint i = 0; i < 4; ++i) pb[i] = u2(G[i] * rc);
            // V through this wave's shared tiles, then O^T = V^T x P^T: rows are the head channels, columns the tokens
            Matrix16 vt0 = matrixSharedLoad(MatrixA16, VSH_W + (mt * 2u + 0u) * 128u, 8u);
            Matrix16 vt1 = matrixSharedLoad(MatrixA16, VSH_W + (mt * 2u + 1u) * 128u, 8u);
            // ZLUDA's lowering of m16n8k16 puts key pair (j & 1) * 4 + 2 h + (j >> 1) in K dword j of half h
            pb = zk(pb, hf);
            const uint4 va = zk(uint4(vt0.reg[0], vt0.reg[1], vt0.reg[2], vt0.reg[3]), hf);
            const uint4 vb = zk(uint4(vt1.reg[0], vt1.reg[1], vt1.reg[2], vt1.reg[3]), hf);
#if M_EXACT
            OB[w][0] = fp8Of(halfOf(mma16(va, pb, zero32())));
            OB[w][1] = fp8Of(halfOf(mma16(vb, pb, zero32())));
#else
            OB[w][0] = fp8Of(mma16(va, pb, zero32()));
            OB[w][1] = fp8Of(mma16(vb, pb, zero32()));
#endif
        }
#ifdef DBG
        if (DBG == 7) { FORW { WINV if (inside) { const uint dst = uint(ty * PW + tx) * C + ch8;
            out0.Store2(dst, OB[w][0]); out0.Store2(dst + 16u, OB[w][1]); } } return; }
        if (DBG == 8) { FORW { WINV if (inside) { const uint dst = uint(ty * PW + tx) * C + ch8;
            out0.Store2(dst, QA[w][0]); out0.Store2(dst + 16u, QA[w][1]); } } return; }
#endif
#ifdef DBG
        if (DBG == 9) { FORW { WINV const uint2 a0 = lda8(ow, 0u, C, 0u), a1 = lda8(ow, 1u, C, 0u);
            const Matrix16 p = mma8(a1, OB[w][1], mma8(a0, OB[w][0], f32of(res[w][0])));
#if DBGP == 1
            const Matrix16 p0 = mma8(a0, OB[w][0], zero32());
#elif DBGP == 2
            const Matrix16 p0 = mma8(a1, OB[w][1], zero32());
#elif DBGP == 3
            const Matrix16 p0 = mma8(a0, OB[w][0], f32of(res[w][0]));
#else
            const Matrix16 p0 = f32of(res[w][0]);
#endif
            if (inside) { const uint dst = uint(ty * PW + tx) * C + 32u * hf;
            [unroll] for (uint i = 0; i < 8; ++i) out0.Store(dst + 4u * i, (DBGP ? p0.reg[i] : p.reg[i])); } } return; }
#endif
        // the output projection of this head's 32 attention channels, straight into the residual stream
        [unroll] for (uint n0 = 0; n0 < NT; n0 += PFN)
        {
            uint2 a[PFN][2];
            [unroll] for (uint i = 0; i < PFN; ++i) if (n0 + i < NT) { a[i][0] = lda8(ow, 0u, C, 16u * (n0 + i)); a[i][1] = lda8(ow, 1u, C, 16u * (n0 + i)); }
            [unroll] for (uint i = 0; i < PFN; ++i) if (n0 + i < NT) { const uint n = n0 + i; FORW res[w][n] = k32h(res[w][n], a[i][0], OB[w][0], a[i][1], OB[w][1]); }
        }
        GroupMemoryBarrier();       // this head's VSH reads before the next head's stores
    }
    DBG_OUT(fp8Of(res[w][n]), 3)
#ifdef DBG
    if (DBG == 5) { FORW { WINV if (inside) { const uint dst = uint(ty * PW + tx) * C + 16u * hf;
        [unroll] for (uint n = 0; n < NT / 2; ++n) out0.Store4(dst + 32u * n, res[w][n]); } } return; }
#endif
    // ------------------------------------------------------------ norm 2
    FORW {
        X8 x[NT];
        [unroll] for (uint t = 0; t < NT; ++t) x[t] = x8(res[w][t]);
        const float rn2 = float(rsqTree(x));
        [unroll] for (uint n = 0; n < NT; ++n)
        {
            h1[w][n] = h1n(x[n], rn2, (po(2) + 16u * n + ch8) * 2u);
            res[w][n] = addBias(x[n], (po(3) + 16u * n + ch8) * 2u);
        }
    }
    DBG_OUT(h1[w][n], 4)
    // ------------------------------------------------------------ the MLP, 32 hidden channels at a time
    // the residual stays f32 across the chunks (M_RESIDUAL_F32) and is rounded to binary16 once after them
    uint2 y8[WPW][NT];
    {
#if !M_EXACT
        Matrix16 rf[WPW][NT];
        FORW { [unroll] for (uint n = 0; n < NT; ++n) rf[w][n] = f32of(res[w][n]); }
#endif
        [loop] for (uint c = 0; c < C / 8u; ++c)
        {
            const uint ow1 = po(20u + 3u * c), ob1 = po(21u + 3u * c), ow2 = po(22u + 3u * c);
            ACC_MLP hm[WPW][2];
            { const ACC_MLP b0 = accInit(ACC_ZERO(ACC_MLP), bias16(ob1, 32u)), b1 = accInit(ACC_ZERO(ACC_MLP), bias16(ob1 + 16u, 32u)); FORW { hm[w][0] = b0; hm[w][1] = b1; } }
            [unroll] for (uint k0 = 0; k0 < NT; k0 += 2u * PF)
            {
                uint2 a[PF][2][2];
                [unroll] for (uint i = 0; i < PF; ++i) [unroll] for (uint m = 0; m < 2u; ++m)
                    if (k0 + 2u * i < NT) { a[i][m][0] = lda8(ow1, k0 + 2u * i, 32u, 16u * m); a[i][m][1] = lda8(ow1, k0 + 2u * i + 1u, 32u, 16u * m); }
                [unroll] for (uint i = 0; i < PF; ++i) [unroll] for (uint m = 0; m < 2u; ++m)
                    if (k0 + 2u * i < NT) { const uint kt = k0 + 2u * i; FORW hm[w][m] = k32(hm[w][m], a[i][m][0], h1[w][kt], a[i][m][1], h1[w][kt + 1u]); }
            }
            // w2's first tiles are requested before the activation, so they arrive while it runs (layer.comp)
            uint2 b2[PFN][2];
            [unroll] for (uint i = 0; i < PFN; ++i) if (i < NT) { b2[i][0] = lda8(ow2, 0u, C, 16u * i); b2[i][1] = lda8(ow2, 1u, C, 16u * i); }
            // the activation in f32, then FP8 (saturating)
            uint2 G[WPW][2];
            FORW {
                [unroll] for (uint m = 0; m < 2u; ++m)
                {
#if M_EXACT
                    const X8 hx = x8(hm[w][m]);
                    // gelu1 in binary16, unfused
                    uint4 gv;
                    [unroll] for (uint i = 0; i < 4; ++i)
                    {
                        // every operation rounded to binary16, in f32 (see KEEP_ZERO)
                        const float2 xx = float2(hx.p[i]), cc = clamp(xx, -2.0, 2.0);
                        const float2 pr = r16(float(half(0.08108133)) * abs(cc));
                        const float2 inner = r16(float(half(0.41216981)) - pr);
                        const float2 outer = r16(0.5 + r16(cc * inner));
                        gv[i] = pk16(xx * outer);
                    }
                    G[w][m] = fp8Of(gv);
                    continue;
#endif
                    float v[8];
                    [unroll] for (uint i = 0; i < 8; ++i)
                    {
                        const float xx = accAt(hm[w][m], i), y = clamp(xx, -2.0, 2.0);
                        v[i] = xx * mad(y, mad(-abs(y), 0.08108133, 0.41216981), 0.5);
                    }
                    G[w][m] = fp8Of(f32m(v));
                }
            }
            [unroll] for (uint n0 = 0; n0 < NT; n0 += PFN)
            {
                if (n0 != 0u)
                    [unroll] for (uint i = 0; i < PFN; ++i) if (n0 + i < NT) { b2[i][0] = lda8(ow2, 0u, C, 16u * (n0 + i)); b2[i][1] = lda8(ow2, 1u, C, 16u * (n0 + i)); }
                [unroll] for (uint i = 0; i < PFN; ++i) if (n0 + i < NT)
                {
                    const uint n = n0 + i;
#if M_EXACT
                    FORW res[w][n] = k32h(res[w][n], b2[i][0], G[w][0], b2[i][1], G[w][1]);
#else
                    FORW rf[w][n] = mma8(b2[i][1], G[w][1], mma8(b2[i][0], G[w][0], rf[w][n]));
#endif
                }
            }
        }
    // the block output in FP8 (with M_ACC32 straight from the f32 residual, as RADV compiles it); L's expansion keeps
    // it binary16 for its head
#if KIND == 4
#if !M_EXACT
        FORW { [unroll] for (uint n = 0; n < NT; ++n) res[w][n] = halfOf(rf[w][n]); }
#endif
#elif ACC32_OUT && !M_EXACT
        FORW { [unroll] for (uint n = 0; n < NT; ++n) y8[w][n] = fp8Of(rf[w][n]); }
#else
#if !M_EXACT
        FORW { [unroll] for (uint n = 0; n < NT; ++n) res[w][n] = halfOf(rf[w][n]); }
#endif
        FORW { [unroll] for (uint n = 0; n < NT; ++n) y8[w][n] = fp8Of(res[w][n]); }
#endif
    }
#if KIND == 4
    // ------------------------------------------------------------ L's expansion head: per-pixel reconstruction inputs
    GroupMemoryBarrierWithGroupSync();      // every wave is done with the V tiles
    FORW {
        WINV
        const uint hb = ((mt * 16u + row) * HS + ch8) / 2u;
        [unroll] for (uint n = 0; n < NT; ++n)
            [unroll] for (uint j = 0; j < 4; ++j) matrixShared[hb + 8u * n + j] = res[w][n][j];
    }
    GroupMemoryBarrierWithGroupSync();
    {
        const uint li = lid.x, px = li & 7u, py = li >> 3;
        const int2 at = int2(X0 + int(px), Y0 + int(py)), size = asint(dp.Load2(0));
        if (li < 64u && at.x >= 0 && at.y >= 0 && at.x < size.x && at.y < size.y)
        {
            const uint hp = (((py >> 2) * 2u + (px >> 2)) * 16u + (py & 3u) * 4u + (px & 3u)) * HS;
            const float sigma = asfloat(dp.Load(48)), rho = asfloat(dp.Load(52));
#define V(c) headIn(hp + (c))
            gateOut[at] = unorm8Value(float4(sigmoidL(V(LH_GATE0)), sigmoidL(V(LH_GATE1)), sigmoidL(V(LH_GATE2)), 0.0));
            const float3 m = float3(V(LH_MIX0), V(LH_MIX1), V(LH_MIX2)), ex = exp2((m - max(max(m.x, m.y), m.z)) * 1.442695);
            mixOut[at] = unorm8Value(float4(ex * (1.0 / (ex.z + (ex.x + ex.y))), 0.0));
            const float a = sigma * sigmoidL(clamp(V(LH_COVA), -20.0, 20.0)), b = sigma * sigmoidL(clamp(V(LH_COVB), -20.0, 20.0));
            covOut[at] = float4(a * a, b * b, a * b * tanhL(V(LH_RHO) * rho), 0.0);
            featOut[at] = snorm8Value(float4(tanhL(V(LH_FEAT0)), tanhL(V(LH_FEAT1)), tanhL(V(LH_FEAT2)), tanhL(V(LH_FEAT3))));
#undef V
        }
    }
#else
    // ------------------------------------------------------------ the block output, FP8
    FORW {
        WINV
        // every lane stores (AGS conversions run on the whole wave; see the patch merge), outside the image out of bounds
        const uint dst = inside ? uint(ty * PW + tx) * C + ch8 : 0xffff0000u;
        [unroll] for (uint n = 0; n < NT; ++n) out0.Store2(dst + 16u * n, y8[w][n]);
    }
#endif
    // ------------------------------------------------------------ patch merge: 2x2 tokens (4C) -> XC, half resolution
#if MERGE
    GroupMemoryBarrierWithGroupSync();      // every wave is done with the V tiles in NB
    FORW {
        WINV
        // the 16 merged rows are gathered from the four windows: the row is the merged token of the lane's pixel
        const uint mrow = (byr >> 1) * 4u + (bxr >> 1), qq = (byr & 1u) * 2u + (bxr & 1u);
        const uint at = NB_W * 4u + 16u * S4 + mrow * S4 + qq * C + ch8;
        [unroll] for (uint n = 0; n < NT; ++n)
        {
            const uint2 y = y8[w][n];
            matrixShared[(at + 16u * n) / 4u] = y.x; matrixShared[(at + 16u * n) / 4u + 1u] = y.y;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    FORW {
        const uint mt = mt0 + w;
        const uint MT = XC / 16u, NI = (MT + 3u) / 4u;
        const int ym = (Y0 >> 1) + int(row >> 2), xm = (X0 >> 1) + int(row & 3u);
        const bool mok = ym >= 0 && ym < PH / 2 && xm >= 0 && xm < PW / 2;
#if PLANES
        // NGX's layout, as layer_m.comp PLANES: the plane, the token, then the channel (16 u + ch8 is a multiple of 8,
        // so a lane's eight-byte store stays in its plane)
        const uint mtok = uint(ym * (PW / 2) + xm), mplane = uint(PW / 2) * uint(PH / 2) * 32u;
#define MADDR(C) (((C) >> 5) * mplane + mtok * 32u + ((C) & 31u))
#else
        const uint md = uint(ym * (PW / 2) + xm) * XC;
#endif
        [unroll] for (uint i = 0; i < NI; ++i)
        {
            const uint u = mt + 4u * i;
            if (u < MT)
            {
                ACC_PATCH y = accInit(ACC_ZERO(ACC_PATCH), bias16(po(69) + 16u * u, XC));
                [unroll] for (uint kt = 0; kt < 4u * C / 16u; kt += 2u)
                {
                    const uint2 b0 = ldsB8(NB_W * 4u + 16u * S4 + kt * 16u, S4), b1 = ldsB8(NB_W * 4u + 16u * S4 + (kt + 1u) * 16u, S4);
                    y = k32(y, lda8(po(68), kt, XC, 16u * u), b0, lda8(po(68), kt + 1u, XC, 16u * u), b1);
                }
                // NVIDIA's encoders clamp the merged result to +-2 pi (as f16) before its e4m3 encoding. The conversion
                // is an AGS operation and has to run on the whole wave: the driver moves it into a branch around the
                // store, so every lane stores, the ones outside the image out of bounds (dropped: descriptor views)
                const half mcl = half(6.2831855);
#if ACC32_PATCH
                [unroll] for (uint j = 0; j < 8; ++j) y.reg[j] = asuint(clamp(asfloat(y.reg[j]), -float(mcl), float(mcl)));
#if PLANES
                out1.Store2(mok ? MADDR(16u * u + ch8) : 0xfffffff0u, fp8Of(y));
#else
                out1.Store2(mok ? md + 16u * u + ch8 : 0xfffffff0u, fp8Of(y));
#endif
#else
                uint4 cl;
                [unroll] for (uint j = 0; j < 4; ++j) cl[j] = u2(clamp(h2(y[j]), -mcl, mcl));
#if PLANES
                out1.Store2(mok ? MADDR(16u * u + ch8) : 0xfffffff0u, fp8Of(cl));
#else
                out1.Store2(mok ? md + 16u * u + ch8 : 0xfffffff0u, fp8Of(cl));
#endif
#endif
            }
        }
    }
#if PLANES
#undef MADDR
#endif
#endif
}
