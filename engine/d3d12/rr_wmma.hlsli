// RR cooperative matrices on the same AGS interface as K/M/L. Scalar FP8
// values are floats; codes are converted by the hardware (AGS conversions).
// The element order is gfx12's: lane = column, elements = rows i + 8 * half-wave
// (accumulators/B), or lane = row, elements = columns i + 8 * half-wave (A).
#define D4R_WMMA_SHARED_WORDS 1
#include "ags_wmma.hlsli"
#include "m_wmma.hlsli"
typedef float RRFloat[8];
typedef uint2 RRByte;
// vkd3d-proton (D4R_VKD3D): dialect_hlsl.h's cast instead, which RADV rounds to
// nearest even (this conversion would go through the AGS scalar-register adapter)
#if D4R_RNE16_EXTERNAL && !D4R_VKD3D
// dialect_hlsl.h's binary16 rounding (ties to even), as the Vulkan conversions
// round: AMD's D3D12 casts and f32tof16 truncate, the emulation costs about six
// instructions per value, the AGS matrix conversion one per pair.
float2 d4rRne16(float2 v) {
  Matrix16 m = zero32();
  m.reg[0] = asuint(v.x);
  m.reg[1] = asuint(v.y);
  const Matrix16 r = convert(MatrixAcc32, MatrixAcc16, 0, m);
#if RR_GFX11
  // gfx11's f16 accumulator holds element i in the low half of register i (RX 7700 XT, AGS on 26.10.1); gfx12 packs
  // the pair into register 0
  return float2(f16tof32(r.reg[0]), f16tof32(r.reg[1]));
#else
  const uint h = r.reg[0];
  return float2(f16tof32(h), f16tof32(h >> 16));
#endif
}
float d4rRne16(float v) { return d4rRne16(float2(v, 0.0)).x; }
#endif
float rrDecode(uint c);
uint rrCode(float x);
float rrGet(RRByte v, uint i) { return rrDecode((v[i >> 2] >> (8u * (i & 3u))) & 255u); }
void rrSet(inout RRByte v, uint i, float value) {
  uint b=rrCode(value);
  uint shift = 8u * (i & 3u);
  v[i >> 2] = (v[i >> 2] & ~(255u << shift)) | ((b & 255u) << shift);
}
int2 imageSize(RWTexture2D<float4> t) {
  uint w, h;
  t.GetDimensions(w, h);
  return int2(w, h);
}
typedef float16_t RRHalf[8];
RRFloat rrFloat(float x) {
  RRFloat r;
  [unroll] for (uint i = 0; i < 8; ++i) r[i] = x;
  return r;
}
RRFloat rrFloat(RRFloat x) { return x; }
RRFloat rrFloat(RRHalf x) {
  RRFloat r;
  [unroll] for (uint i = 0; i < 8; ++i) r[i] = float(x[i]);
  return r;
}
RRHalf rrHalf(float x) {
  RRHalf r;
  [unroll] for (uint i = 0; i < 8; ++i) r[i] = float16_t(x);
  return r;
}
RRHalf rrHalf(RRFloat x) {
  RRHalf r;
  [unroll] for (uint i = 0; i < 8; ++i) r[i] = float16_t(x[i]);
  return r;
}
RRHalf rrHalf(RRHalf x) { return x; }
// E4M3 codecs (round to nearest even, saturating at 448, NaN kept). Decoding
// is the hardware conversion of the code's register (exact; one instruction
// per pair, where the bit arithmetic took about ten per code).
float rrDecode(uint c) {
  const Matrix16 r = convert(MatrixB8, MatrixAcc32, 0, m16(uint2(c & 255u, 0u)));
  return asfloat(r.reg[0]);
}
// Two consecutive FP8 codes (unpack8's low or high half) decoded by one
// matrix conversion. The codes occupy the low two bytes of one DWORD; the
// B8->Acc16 result's first register holds both as binary16 (the scalar
// rrDecodeH mapping, two lanes at once).
float2 rrDecode2(uint2 c) {
  const uint word = (c.x & 255u) | ((c.y & 255u) << 8u);
  const Matrix16 r = convert(MatrixB8, MatrixAcc16, 0, m16(uint2(word, 0u)));
  return float2(f16tof32(r.reg[0] & 0xffffu), f16tof32(r.reg[0] >> 16u));
}
// unpack8(w): the four bytes of w as FP8 codes, low pair then high pair
struct RRBytes8 { uint2 xy, zw; };
RRBytes8 unpack8(uint w) {
  RRBytes8 r;
  r.xy = uint2(w & 255u, (w >> 8u) & 255u);
  r.zw = uint2((w >> 16u) & 255u, w >> 24u);
  return r;
}
float rrE4(uint c) { return rrDecode(c); }
float2 rrE4(uint2 c) { return rrDecode2(c); }
// The low (low true) or high byte pair of w (rr_hlsl.py's translation of
// uintBitsToFloate4m3EXT(low ? unpack8(w).xy : unpack8(w).zw); rrE4PairAt, element i
// of it, for the decode of one byte of that pair). vkd3d-proton:
// B8 -> Acc32 of the whole word, element i in register i, one conversion for
// both halves after CSE and no byte repacking; through rrDecode2 (Acc16 and
// back) RADV kept a binary16 conversion per value and a mode switch per pair.
#if D4R_VKD3D
float2 rrE4Pair(uint w, bool low) {
  const Matrix16 r = convert(MatrixB8, MatrixAcc32, 0, m16(uint2(w, 0u)));
  return low ? asfloat(uint2(r.reg[0], r.reg[1])) : asfloat(uint2(r.reg[2], r.reg[3]));
}
#define rrE4PairAt(w, low, i) rrE4Pair(w, low)[i]
#else
#define rrE4Pair(w, low) rrE4((low) ? unpack8(w).xy : unpack8(w).zw)
#define rrE4PairAt(w, low, i) rrE4(((low) ? unpack8(w).xy : unpack8(w).zw)[i])
#endif
#if D4R_VKD3D
// vkd3d-proton: the hardware conversion (dxil-spirv's saturating FP8
// conversion, per lane on RADV, also in divergent code), as the Vulkan build's
// floate4m3_t: the bit arithmetic below cost encoder 0's feature gather about
// 80 instructions per value (rrCode of rrQuant). NaN stays NaN, values beyond
// 448 saturate; infinity becomes NaN, as the hardware's (and Vulkan's) does.
uint rrCode(float x) {
  Matrix16 m = zero32();
  m.reg[0] = asuint(x);
  return convert(MatrixAcc32, MatrixB8, 1, m).reg[0] & 255u;
}
#else
// Branch-free: the early returns compiled to divergent exec masks with a VALU
// to SALU hazard wait per test, most of encoder 0's translated gather time.
// (The AGS conversion of one value saves only ~2% of encoder 0 and would run
// in divergent code, where the driver's lowering of AGS results is unsafe.)
uint rrCode(float x) {
  const uint u = asuint(x), s = (u >> 24u) & 128u, a = u & 0x7fffffffu, t = a - 0x3c000000u;
  uint v = a < asuint(0.015625) ? uint(round(asfloat(a) * 512.0)) : (t + 0x7ffffu + ((t >> 20u) & 1u)) >> 20u;
  v = a >= asuint(448.0) ? 126u : v;
  v = a > 0x7f800000u ? 127u : v;
  return s | v;
}
#endif
float16_t rrDecodeH(uint c) {
  const Matrix16 r = convert(MatrixB8, MatrixAcc16, 0, m16(uint2(c & 255u, 0u)));
  return asfloat16(uint16_t(r.reg[0] & 0xffffu));
}
// Byte b of w (rr_hlsl.py's translation of e4(w, b)). vkd3d-proton: the
// conversion of the whole word to f32, element b in register b, so the four
// bytes' decodes are one conversion after CSE; the cast to binary16 then folds
// into an f32 consumer, where an Acc16 result kept a conversion per value
#if D4R_VKD3D
float16_t rrDecodeByteH(uint w, uint b) {
  const Matrix16 r = convert(MatrixB8, MatrixAcc32, 0, m16(uint2(w, 0u)));
  return float16_t(asfloat(r.reg[b]));
}
#else
float16_t rrDecodeByteH(uint w, uint b) { return rrDecodeH((w >> (8u * b)) & 0xffu); }
#endif
#if D4R_VKD3D
float rrQuant(float x) { return float(rrDecodeH(rrCode(x))); }
#else
// rrDecode(rrCode(x)) without leaving the f32 pattern.
float rrQuant(float x) {
  uint u = asuint(x), a = u & 0x7fffffffu;
  float sub = round(asfloat(a) * 512.0) * 0.001953125;
  uint n = (a + 0x7ffffu + ((a >> 20u) & 1u)) & 0xfff00000u;
  float v = a < asuint(0.015625) ? sub : asfloat(n);
  v = a >= asuint(448.0) ? 448.0 : v;
  v = a > 0x7f800000u ? asfloat(0x7fc00000u) : v;
  return asfloat(asuint(v) | (u & 0x80000000u));
}
#endif
float2 rrQuant(float2 x) { return float2(rrQuant(x.x), rrQuant(x.y)); }
uint2 rrCode(float2 x) { return uint2(rrCode(x.x), rrCode(x.y)); }
uint2 rrPacked(RRByte x) { return x; }
// Eight registers of a lane rounded by the hardware conversion. As M's relaxed
// fp8Of: an f32 clamp and the plain conversion, not the driver's saturating
// one (two compares and two selects per element). Unclamped, a value beyond
// 448 becomes a NaN code that the next stage's products spread. Relaxed
// (RR_QUANT_CLAMPONLY): the clamp alone, which turns NaN into a bound. Exact:
// fma(0, x, clamp(x)) restores NaN (0 x is 0 for a finite x, NaN for NaN or
// infinity, which becomes NaN as with q8s). vkd3d-proton (D4R_VKD3D): the
// saturating conversion unclamped, as the Vulkan build's saturatedConvertEXT
// (dxil-spirv's SaturatedToLargestFloat8NormalConversionEXT, one instruction per
// pair on RADV): NaN stays NaN and, as on the gfx12 hardware, infinity becomes
// NaN, which also makes rrInfNan's test redundant.
RRByte rrQuantSet(RRFloat v) {
  Matrix16 m;
#if D4R_VKD3D
  [unroll] for (uint i = 0; i < 8; ++i) m.reg[i] = asuint(v[i]);
  const Matrix16 r = convert(MatrixAcc32, MatrixB8, 1, m);
  return uint2(r.reg[0], r.reg[1]);
#else
  [unroll] for (uint i = 0; i < 8; ++i) {
#if RR_QUANT_NONE // TMP experiment
    m.reg[i] = asuint(v[i]);
#elif RR_QUANT_CLAMPONLY
    m.reg[i] = asuint(clamp(v[i], -448.0, 448.0));
#else
    precise float x = fma(0.0, v[i], clamp(v[i], -448.0, 448.0));
    m.reg[i] = asuint(x);
#endif
  }
  const Matrix16 r = convert(MatrixAcc32, MatrixB8, 0, m);
  return uint2(r.reg[0], r.reg[1]);
#endif
}
struct RRHalf8 { vector<float16_t, 4> lo, hi; };
RRHalf8 rrHalf8(vector<float16_t, 4> lo, vector<float16_t, 4> hi) { RRHalf8 r; r.lo = lo; r.hi = hi; return r; }
float16_t rrHalfAt(RRHalf8 v, uint i) { return i < 4u ? v.lo[i] : v.hi[i - 4u]; }
#if D4R_VKD3D
float rrInfNan(float x) { return x; }   // for rrQuantSet only, whose conversion makes infinity NaN
#else
float rrInfNan(float x) { return isinf(x) ? asfloat(0x7fc00000u) : x; }
#endif
// From binary16, packed halves to the plain conversion without widening each to
// f32. NaN stays NaN; unlike rrCode, infinity and halves beyond 448 become NaN
// too (the saturating conversion costs as much as the f32 path, and the
// normalised values converted here stay in range: no PSNR change).
RRByte rrQuantSetH(RRHalf v) {
  uint4 h;
  [unroll] for (uint i = 0; i < 4; ++i) h[i] = uint(asuint16(v[2 * i])) | (uint(asuint16(v[2 * i + 1])) << 16);
  const Matrix16 r = convert(MatrixAcc16, MatrixB8, 0, m16(h));
  return uint2(r.reg[0], r.reg[1]);
}
// The driver moves an AGS conversion used only in divergent code into that
// branch, where its lowering reads inactive lanes. An unconditional store of
// the result keeps the conversion in uniform flow (one VALU, one LDS write).
groupshared uint rrPinned[D4R_WAVE_SIZE];
void rrPin(RRByte v) { rrPinned[WaveGetLaneIndex()] = v.x | v.y; }
uint rrPack4(float16_t a, float16_t b, float16_t c, float16_t d) {
  RRHalf v = {a, b, c, d, 0.0h, 0.0h, 0.0h, 0.0h};
  return rrQuantSetH(v).x;
}
uint rrPack4(float a, float b, float c, float d) {
  RRFloat v = {a, b, c, d, 0.0, 0.0, 0.0, 0.0};
  return rrQuantSet(v).x;
}
RRByte rrQuantMatrix(RRFloat v, bool activation) {
  [unroll] for (uint i = 0; i < 8; ++i) {
    if (activation) {
      float y = clamp(v[i], -2.0, 2.0);
      v[i] = v[i] * mad(y, mad(-abs(y), 0.0810546875, 0.412109375), 0.5);
    }
  }
  return rrQuantSet(v);
}
RRByte rrQuantMatrix(RRHalf v, bool activation) {
  return rrQuantMatrix(rrFloat(v), activation);
}
RRFloat rrMma(RRByte a, RRByte b, RRFloat c) {
  Matrix16 r = mma8(rrPacked(a), rrPacked(b), f32m(c));
  RRFloat v;
  [unroll] for (uint i = 0; i < 8; ++i) v[i] = asfloat(r.reg[i]);
  return v;
}
// Used by the generated gather's reference helpers (dead in a native model).
RRFloat rrMma(RRHalf a, RRHalf b, RRFloat c) {
  uint4 aa = 0, bb = 0;
  [unroll] for (uint i = 0; i < 8; ++i) {
    aa[i / 2] |= uint(asuint16(a[i])) << (16 * (i & 1));
    bb[i / 2] |= uint(asuint16(b[i])) << (16 * (i & 1));
  }
  Matrix16 r = mma16(aa, bb, f32m(c));
  RRFloat v;
  [unroll] for (uint i = 0; i < 8; ++i) v[i] = asfloat(r.reg[i]);
  return v;
}
#if RR_GFX11
// RDNA3 (rr_hlsl.translate gfx11): an f16 A or B operand is a lane's 16 elements (k 0..15 of its row / column), the
// eight registers of gfx11's v_wmma_f32_16x16x16_f16; AMD's D3D12 driver passes AGS's matrices in that hardware
// layout. Only f16 operands: no FP8 matrix or conversion may reach an RDNA3 pipeline (it removes the device).
typedef float16_t RRHalf16[16];
RRHalf16 rrHalf16(float x) {
  RRHalf16 r;
  [unroll] for (uint i = 0; i < 16; ++i) r[i] = float16_t(x);
  return r;
}
RRHalf16 rrHalf16(RRHalf16 x) { return x; }
Matrix16 rrPack16(RRHalf16 a) {
  Matrix16 m;
  [unroll] for (uint i = 0; i < 8; ++i) m.reg[i] = uint(asuint16(a[2 * i])) | (uint(asuint16(a[2 * i + 1])) << 16);
  return m;
}
RRFloat rrMma(RRHalf16 a, RRHalf16 b, RRFloat c) {
  Matrix16 m;
  [unroll] for (uint i = 0; i < 8; ++i) m.reg[i] = asuint(c[i]);
  const Matrix16 r = mmaOp(5, rrPack16(a), rrPack16(b), m);   // f32_16x16x16_f16
  RRFloat v;
  [unroll] for (uint i = 0; i < 8; ++i) v[i] = asfloat(r.reg[i]);
  return v;
}
#endif
#define gl_SubgroupInvocationID WaveGetLaneIndex()
#define gl_SubgroupID (gl_LocalInvocationIndex / D4R_WAVE_SIZE)
#define gl_NumSubgroups (D4R_THREADS / D4R_WAVE_SIZE)
#define gl_NumWorkGroups uint3(1, 1, 1)
#define subgroupBarrier() GroupMemoryBarrier()
#define subgroupMemoryBarrier() DeviceMemoryBarrier()
#define subgroupMemoryBarrierShared() GroupMemoryBarrier()
#define subgroupMemoryBarrierBuffer() DeviceMemoryBarrier()
#define subgroupMemoryBarrierImage() DeviceMemoryBarrier()
#if RR_SPILL_WORDS
#undef barrier
#define barrier() AllMemoryBarrierWithGroupSync()
#undef memoryBarrierShared
#define memoryBarrierShared() DeviceMemoryBarrier()
#undef subgroupBarrier
#define subgroupBarrier() AllMemoryBarrier()
#endif
#define subgroupAny WaveActiveAnyTrue
#define subgroupAll WaveActiveAllTrue
#define subgroupBallot WaveActiveBallot
#define subgroupShuffle(x, l) WaveReadLaneAt(x, l)
#define subgroupBroadcast(x, l) WaveReadLaneAt(x, l)
#define subgroupAdd WaveActiveSum
#define subgroupMax WaveActiveMax
#define subgroupMin WaveActiveMin
#define bitCount countbits
#define findLSB firstbitlow
#define findMSB firstbithigh
#define uint8_t uint
#define fe4m3vec2 float2
typedef float floate4m3_t;
#define floate4m3_t(x) rrQuant(float(x))
#define uintBitsToFloate4m3EXT(x) rrE4((x))
#define floate4m3BitsToUintEXT(x) rrCode(x)
#define saturatedConvertEXT(d, s) d = rrQuant(s)
#define sampler2D Texture2D<float4>
