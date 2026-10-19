#!/usr/bin/env python3
"""Generate a GLSL compute shader for one Ray Reconstruction (DLSS-D) PTX kernel.

    python3 engine/rr/ptx2rr.py MODULE.ptx KERNEL OUT.comp [--block X,Y,Z]

The translation core is vulkan/ptx2glsl.py. This module subclasses its translator and
implements the stage contract of engine/rr/README.md: binding 0 = parameter block,
binding 1 = weight buffer (argument 40), binding 2 = plane arena, then textures and
surfaces in ascending argument offset. One PTX block is one workgroup of X*Y*Z invocations.

The generated shader is derived from the user's own DLL and is model data: write it to the
model/build directory only; do not commit or redistribute it.
"""
import argparse
import importlib.util
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('ptx2glsl', ROOT / 'vulkan' / 'ptx2glsl.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

# Workgroup shapes of the 15 Swin/HKPN kernels, from the captured launch manifests.
BLOCKS = {
    'enc0': (32, 1, 1), 'enc1': (32, 1, 1), 'enc2': (32, 1, 4), 'enc3': (32, 1, 4), 'enc4': (32, 1, 8),
    'enc5': (32, 1, 8), 'dec0': (32, 1, 1), 'dec1': (32, 1, 2), 'dec2': (32, 1, 4), 'dec3': (32, 1, 4),
    'dec4': (32, 1, 8), 'hkpn': (16, 16, 1),
}


def default_block(kernel):
    m = re.search(r'swin_(enc|dec)(\d)', kernel)
    key = f'{m.group(1)}{m.group(2)}' if m else ('hkpn' if 'hkpn_output' in kernel else None)
    if key not in BLOCKS:
        raise SystemExit(f'no default block shape for {kernel}; pass --block X,Y,Z')
    return BLOCKS[key]


def braces(tok):
    s = tok.strip()
    if not (s.startswith('{') and s.endswith('}')):
        raise base.Err(f'expected a {{...}} vector operand, got {tok}')
    return [x.strip() for x in s[1:-1].split(',')]


def mem_shape(o):
    """(bytes per element, element count) of ld/st.global modifiers o."""
    count = 2 if 'v2' in o else 4 if 'v4' in o else 1
    size = int(re.sub(r'\D', '', o[-1])) // 8
    if size not in (2, 4):
        raise base.Err(f'unsupported global access width in {".".join(o)}')
    return size, count


# --exact-convert (the module is passed through engine/rr/spv_exact_convert.py, which keeps every conversion): a
# cvt.rn.f16.f32 only needs the opaque zero on its f32 bits, against the fused multiply-convert (see cvt)
CVT_GUARD = '--exact-convert' not in sys.argv
RR_RUNTIME = r'''
float16_t H(uint u) { return uint16BitsToFloat16(uint16_t(u)); }
uint HU(float16_t h) { return uint(float16BitsToUint16(h)); }
f16vec2 H2(uint u) { return uint16BitsToFloat16(u16vec2(uint16_t(u & 0xffffu), uint16_t(u >> 16))); }
uint HU2_(u16vec2 u) { return uint(u.x) | (uint(u.y) << 16); }
uint H2U(f16vec2 h) { return HU2_(float16BitsToUint16(h)); }
int s16(uint u) { return (int(u) << 16) >> 16; }
float rcp_a(float x) { return 1.0 / x; }
// add.rz.f32: the sum rounded toward zero (round to nearest, then one ulp back when that went away from zero; the
// error term is exact, TwoSum)
float add_rz(float a, float b) { precise float s = a + b; precise float bb = s - a; precise float e = (a - (s - bb)) + (b - bb);
    return (e != 0.0 && ((e < 0.0) != (s < 0.0)) && s != 0.0) ? uintBitsToFloat(floatBitsToUint(s) - 1u) : s; }
float lg2_ftz(float x) { return log2(abs(x) < 1.17549435e-38 ? 0.0 : x); }
// PTX .relu: negative results become +0, NaN stays NaN
f16vec2 relu2(f16vec2 v) { return f16vec2(v.x < 0.0hf ? 0.0hf : v.x, v.y < 0.0hf ? 0.0hf : v.y); }
float16_t relu1(float16_t v) { return v < 0.0hf ? 0.0hf : v; }
// An integer fetch outside the image returns zero (one unsigned comparison per axis: a negative coordinate is above
// every size). With robustImageAccess2 the image load returns zero there itself, except for a w component the format
// lacks, which it returns as one: the runtime sets RR_ROBUST_IMAGES when the device has it enabled (vkd3d-proton
// always does), and only a fetch whose w is used still needs the image's size.
layout(constant_id = 0) const bool RR_ROBUST_IMAGES = false;
vec4 fetch0(sampler2D t, ivec2 c)
{
    const bool out_ = any(greaterThanEqual(uvec2(c), uvec2(textureSize(t, 0))));
    if (RR_ROBUST_IMAGES) { const vec4 v = texelFetch(t, c, 0); return vec4(v.xyz, out_ ? 0.0 : v.w); }
    return out_ ? vec4(0.0) : texelFetch(t, c, 0);
}
// x + copysign(0.5, x) rounded toward zero and then truncated: round half away from zero, exactly
float round_half_away(float x) { float t = trunc(x); return abs(x - t) >= 0.5 ? t + sign(x) : t; }
uint PARAM32(uint o) { return param.w[o >> 2]; }
uint64_t PARAM64(uint o) { return uint64_t(param.w[o >> 2]) | (uint64_t(param.w[(o >> 2) + 1u]) << 32); }
uint PARAM16(uint o) { return (param.w[o >> 2] >> ((o & 2u) * 8u)) & 0xffffu; }
uint PARAM8(uint o) { return (param.w[o >> 2] >> ((o & 3u) * 8u)) & 0xffu; }
uint SH32(uint a) { return sh[a >> 2]; }
uint SH16(uint a) { return (sh[a >> 2] >> ((a & 2u) * 8u)) & 0xffffu; }
uint bfi32(uint ins, uint base, uint pos, uint len)
{
    pos &= 0xffu; len &= 0xffu;
    if (pos >= 32u) return base;
    uint mask = len >= 32u ? (0xffffffffu << pos) : (((1u << len) - 1u) << pos);
    return (~mask & base) | (mask & (ins << pos));
}

// Global memory: weight buffer (binding 1, read-only) and plane arena (binding 2). Addresses are byte offsets.
#ifdef OOB_REPORT
// debugging (ptx2rr.py --oob-report): per buffer (0 weights read, 1 arena read, 2 arena write) a count and the lowest
// and highest out-of-range byte address, in dbg.d[3k .. 3k+2]
void OOB(uint k, uint a, uint size) { if (a + 2u > size) { atomicAdd(dbg.d[3u * k], 1u); atomicMin(dbg.d[3u * k + 1u], a); atomicMax(dbg.d[3u * k + 2u], a); } }
#define CHECKW(a) OOB(0u, a, uint(wbuf.wh.length()) * 2u)
#define CHECKR(a) OOB(1u, a, uint(abuf.ah.length()) * 2u)
#define CHECKS(a) OOB(2u, a, uint(abuf.ah.length()) * 2u)
#else
#define CHECKW(a)
#define CHECKR(a)
#define CHECKS(a)
#endif
uint GW16(uint a) { CHECKW(a); return uint(wbuf.wh[a >> 1]); }
// a 32-bit access is 4-byte aligned (PTX ISA): one word of the same buffer's word view
uint GW32(uint a) { CHECKW(a + 2u); return wbuf32.ww[a >> 2]; }
uint GA16(uint a) { CHECKR(a); return uint(abuf.ah[a >> 1]); }
uint GA32(uint a) { CHECKR(a + 2u);
#ifdef D4R_WORD_ARENA
    return abuf32.aw[a >> 2];
#else
    return uint(abuf.ah[a >> 1]) | (uint(abuf.ah[(a >> 1) + 1u]) << 16);
#endif
}
void SA16(uint a, uint v) { CHECKS(a); abuf.ah[a >> 1] = uint16_t(v & 0xffffu); }
void SA32(uint a, uint v) { CHECKS(a + 2u);
#ifdef D4R_WORD_ARENA
    abuf32.aw[a >> 2] = v;
#else
    abuf.ah[a >> 1] = uint16_t(v & 0xffffu); abuf.ah[(a >> 1) + 1u] = uint16_t(v >> 16);
#endif
}

// Surface stores with .zero out-of-bounds semantics.
// Expand stores at the call site: passing a formatted image to an unformatted image2D
// parameter makes glslang emit mismatched SPIR-V image types.
#define sust_p4(S, X, Y, V) { ivec2 sn_ = imageSize(S); ivec2 sc_ = ivec2(X, Y); if (uint(sc_.x) < uint(sn_.x) && uint(sc_.y) < uint(sn_.y)) imageStore(S, sc_, V); }
#define sust_p1(S, X, Y, V) sust_p4(S, X, Y, vec4(V, 0.0, 0.0, 0.0))
// Formatted stores to normalized surfaces, as the reference converts them (ZLUDA's surface_formatted_store_bits): NaN
// to zero, clamp, round half away from zero in f32. The quotient is what the image store then converts back exactly.
// UNORM 10:10:10:2 surfaces (the pooled histories) are RGBA32F images holding the quantised values, like ZLUDA's.
// tanh.approx.f32 as the reference computes it: ROCm's OCML tanh (device-libs ocml/src/tanhF.cl) with LLVM's AMDGPU
// lowering of exp (lowerFEXP, fast-FMA path): the same fused operations, hardware exp2, ldexp and reciprocal.
// Up to 2026-10-19 every fused operation went through f64 (the product exact there, the sum rounded to 53 bits, then
// to f32), which this GPU runs at a small fraction of the f32 rate. They are the driver's f32 fma now, with two
// exceptions that keep every result bit for bit what it was, on all 2^32 arguments (RX 9070 XT, RADV; NaN payloads
// included):
// - the polynomial stays in f64: the driver drops the f32 rounding between nested f64 operations (it folds
//   f2f64(f2f32(x)) to x; the conversions are not NoContraction), so its four nested steps always ran in f64 with one
//   rounding at the end. In f32 (true fma), 24,836 arguments round one f32 step differently, 2 of them in binary16.
// - exp's error term fma(w, c, -ph) next to ph = w * c: the driver makes that the product it already has and a sum,
//   exactly 0, even with every operand precise or opaque. It is Dekker's exact product error: w split into two halves
//   of 12 bits (w * 4097 = w * (2^12 + 1)), log2(e) into 1.4423828125 + 0.0003122091293334961 (11 and 12 bits), so
//   every partial product is exact and so is each sum.
// A NaN argument keeps its payload, quieted and positive, as the f64 conversion left it.
float tanh_ocml(float x)
{
    const float y = abs(x);
    precise float z;
    if (y < 0.625) {
        precise float y2 = y * y;
        precise double y2d = double(y2);
        precise double pd = (((y2d * -0.005700020585209131 + 0.02063407190144062) * y2d - 0.053737930953502655) * y2d +
                             0.13331416249275208) * y2d - 0.3333328068256378;
        precise float p = float(pd);
        precise float yp = y * p;
        z = fma(y2, yp, y);
    } else {
        const float w = 2.0 * y;
        precise float ph = w * 1.4426950216293335;
        precise float wt = w * 4097.0;
        precise float wh = wt - (wt - w);
        precise float wl = w - wh;
        precise float f1 = wh * 1.4423828125 - ph;
        precise float f2 = f1 + wh * 0.0003122091293334961;
        precise float f3 = f2 + wl * 1.4423828125;
        precise float f0 = f3 + wl * 0.0003122091293334961;
        precise float pl = fma(w, 1.925962855864327e-08, f0);
        const float e = roundEven(ph);
        precise float d = ph - e;
        precise float a = d + pl;
        float t = ldexp(exp2(a), int(e));
        if (w > 88.72283172607422) t = uintBitsToFloat(0x7f800000u);
        precise float s1 = t + 1.0;
        z = fma(-2.0, rcp_a(s1), 1.0);
        if (isnan(z)) z = uintBitsToFloat(floatBitsToUint(y) | 0x00400000u);
    }
    return x < 0.0 || floatBitsToUint(x) == 0x80000000u ? -z : z;
}
// tanh.approx.f16: from a table of the correctly rounded values indexed by the argument's bits (RR_TANH_TABLE: its
// byte offset in the weight buffer; engine/rr/rr2_pack.py) where a stage has one, else through f32
#ifdef RR_TANH_TABLE
float16_t tanh16(float16_t x) { return uint16BitsToFloat16(wbuf.wh[RR_TANH_TABLE / 2u + uint(float16BitsToUint16(x))]); }
#else
float16_t tanh16(float16_t x) { return float16_t(tanh(float(x))); }
#endif
vec4 q_snorm8(vec4 v)
{
    v = clamp(mix(v, vec4(0.0), isnan(v)), vec4(-1.0), vec4(1.0));
    precise vec4 s = v * 127.0;
    precise vec4 a = abs(s) + 0.5;
    return sign(s) * floor(a) / 127.0;
}
vec4 q_unorm8(vec4 v)
{
    v = clamp(mix(v, vec4(0.0), isnan(v)), vec4(0.0), vec4(1.0));
    precise vec4 a = v * 255.0 + 0.5;
    return floor(a) / 255.0;
}
// (the quotient q / 1023 must be the correctly rounded f32 the reference stores: the driver divides by multiplying
// with a reciprocal, one step off for some q, so it comes from a table)
const float Q1023[1024] = float[1024](Q1023_VALUES);
vec4 q_unorm1010102(vec4 v)
{
    v = clamp(mix(v, vec4(0.0), isnan(v)), vec4(0.0), vec4(1.0));
    precise vec4 a = v * vec4(1023.0, 1023.0, 1023.0, 3.0) + 0.5;
    const ivec4 q = ivec4(floor(a));
    const float third[4] = float[4](0.0, 0.3333333432674408, 0.6666666865348816, 1.0);
    return vec4(Q1023[q.x], Q1023[q.y], Q1023[q.z], third[q.w]);
}
// for a real UNORM 10:10:10:2 image: the reference's rounding (half away from zero) here, and any value within half a
// step of q / 1023 for the store, which the hardware quantises back to q
vec4 q_unorm1010102_hw(vec4 v)
{
    v = clamp(mix(v, vec4(0.0), isnan(v)), vec4(0.0), vec4(1.0));
    precise vec4 a = v * vec4(1023.0, 1023.0, 1023.0, 3.0) + 0.5;
    return floor(a) * vec4(1.0 / 1023.0, 1.0 / 1023.0, 1.0 / 1023.0, 1.0 / 3.0);
}
// Binary stores write the halves' bits. RADV folds float(float16_t(x)) back to x when the half came from a
// cvt.rn.f16.f32 in a register, and the image store then truncates x instead (the output stage's history surface
// was one binary16 ulp low in half its values, which the next frame's encoder 0 amplifies): OR-ing in a zero the
// compiler cannot see (the dispatch's z count is 1) keeps the conversion's rounding.
#define SUST_Z (gl_NumWorkGroups.z - 1u)
#define sust_b2(S, X, Y, A, B) sust_p4(S, (X) >> 2, Y, vec4(float(H((A) | SUST_Z)), float(H((B) | SUST_Z)), 0.0, 0.0))
#define sust_b4(S, X, Y, A, B, C, D) sust_p4(S, (X) >> 3, Y, vec4(float(H((A) | SUST_Z)), float(H((B) | SUST_Z)), float(H((C) | SUST_Z)), float(H((D) | SUST_Z))))

// shfl.sync.up/down within a 32-lane warp (ZLUDA's SHFL_SYNC_IMPL). in_bounds is the predicate output.
uint shfl_dir(uint v, uint delta, uint opts, bool down, out bool in_bounds)
{
    int section_mask = int((opts >> 8) & 31u), warp_end = int(opts & 31u);
    int self = int(gl_SubgroupInvocationID & 31u);
    int d = int(delta & 31u);
    int idx = down ? self + d : self - d;
    int sub_end = (section_mask & self) | (~section_mask & (down ? warp_end : 0));
    bool oob = down ? idx > sub_end : idx < sub_end;
    if (oob) idx = self;
    in_bounds = !oob;
    return subgroupShuffle(v, (gl_SubgroupInvocationID & 32u) | (uint(idx) & 31u));   // the warp is this half of a 64-wide wave
}

// shfl.sync.idx (PTX ISA): j = minLane | (b & ~segmask), in bounds while j <= maxLane, otherwise the lane's own value
uint shfl_idx(uint v, uint b, uint opts, out bool in_bounds)
{
    uint mask = (opts >> 8) & 31u, lane = gl_SubgroupInvocationID & 31u;
    uint maxLane = (lane & mask) | (opts & 31u & ~mask), j = (lane & mask) | (b & 31u & ~mask);
    in_bounds = j <= maxLane;
    return subgroupShuffle(v, (gl_SubgroupInvocationID & 32u) | (in_bounds ? j : lane));
}
// movmatrix.sync.trans.aligned.m8n8.b16 on one b32 fragment register per lane: lane (g = lane>>2, t = lane&3)
// holds row g, columns 2t and 2t+1. The transpose gathers (row 2t, col g) and (row 2t+1, col g).
uint movm_trans(uint src)
{
    // through the subgroup's mma scratch rather than two shuffles: RADV miscompiled the shuffle form in the large
    // stages (enc1's movmatrix results were wrong unless a store followed them; ZLUDA replay, 2026-10-11)
    const uint base = gl_SubgroupID * 384u + 256u;
    uint lane = gl_SubgroupInvocationID & 31u;
    uint g = lane >> 2u, t = lane & 3u;
    // order this operation's writes after every lane's reads of the previous one's results (the same scratch)
    memoryBarrierShared(); subgroupBarrier();
    mms[base + lane] = src;
    memoryBarrierShared(); subgroupBarrier();
    uint w1 = mms[base + 8u * t + (g >> 1u)];
    uint w2 = mms[base + 8u * t + 4u + (g >> 1u)];
    uint s = 16u * (g & 1u);
    return ((w1 >> s) & 0xffffu) | (((w2 >> s) & 0xffffu) << 16);
}

// PTX min/max: a NaN operand yields the other one. GLSL's min/max leave NaN undefined (the result depended on
// codegen); SPIR-V's NMin/NMax (GLSL.std.450 79/80) are defined that way, and compile to the bare v_min/v_max
// (checked on RADV / RDNA4 for f16, f16x2 and f32 with NaN, infinity and signed zero operands, 2026-10-15)
spirv_instruction(set = "GLSL.std.450", id = 79) float16_t pmin(float16_t a, float16_t b);
spirv_instruction(set = "GLSL.std.450", id = 80) float16_t pmax(float16_t a, float16_t b);
spirv_instruction(set = "GLSL.std.450", id = 79) f16vec2 pmin2(f16vec2 a, f16vec2 b);
spirv_instruction(set = "GLSL.std.450", id = 80) f16vec2 pmax2(f16vec2 a, f16vec2 b);
spirv_instruction(set = "GLSL.std.450", id = 79) float pminf(float a, float b);
spirv_instruction(set = "GLSL.std.450", id = 80) float pmaxf(float a, float b);
// e4m3 <-> f16 (PTX cvt semantics; RNE, satfinite; NaN -> 0x7f e4m3 / 0x7e00 f16).
uint e4m3_to_h(uint b)
{
    uint e = (b >> 3) & 15u, m = b & 7u, s = (b & 0x80u) << 8;
    if (e == 15u && m == 7u) return s | 0x7e00u;
    float f = e == 0u ? ldexp(float(m), -9) : ldexp(float(8u + m), int(max(e, 1u)) - 10);
    return s | HU(float16_t(f));
}
#if defined(RR_GFX11)
// RDNA3 (engine/rr/rr_gfx11.glsl): ZLUDA's gfx11 conversions exactly. e4m3 -> f16: NaN -> 0x7fff with the sign
uint g11_e4m3_to_h(uint c)
{
    const uint m = c & 0x7fu;
    uint h = m >= 8u ? (m << 7u) + 0x2000u : uint(float16BitsToUint16(float16_t(float(m & 7u) * 0.001953125)));
    h = m == 0x7fu ? 0x7fffu : h;
    return ((c & 0x80u) << 8u) | h;
}
f16vec2 e4m3x2_to_f16x2(uint v) { return H2(g11_e4m3_to_h(v & 0xffu) | (g11_e4m3_to_h((v >> 8) & 0xffu) << 16)); }
#elif defined(SOFTWARE_E4M3)
f16vec2 e4m3x2_to_f16x2(uint v) { return H2(e4m3_to_h(v & 0xffu) | (e4m3_to_h((v >> 8) & 0xffu) << 16)); }
#else
// the hardware conversion: exact for every non-NaN code (2026-10-12); NaN keeps e4m3_to_h's sign | 0x7e00
uint e4m3_to_h_hw(uint b)
{
    if ((b & 0x7fu) == 0x7fu) return ((b & 0x80u) << 8) | 0x7e00u;
    return uint(float16BitsToUint16(float16_t(uintBitsToFloate4m3EXT(uint8_t(b)))));
}
f16vec2 e4m3x2_to_f16x2(uint v) { return H2(e4m3_to_h_hw(v & 0xffu) | (e4m3_to_h_hw((v >> 8) & 0xffu) << 16)); }
#endif
uint h2e4m3(float f, bool relu)
{
    if (isnan(f)) return 0x7fu;
    if (relu && !(f > 0.0)) return 0u;
    uint s = f < 0.0 ? 0x80u : 0u;
    float a = abs(f);
    if (a >= 448.0) return s | 0x7eu;
    if (a == 0.0) return s;
    int ex;
    frexp(a, ex);
    int qe = max(ex - 4, -9);
    uint n = uint(roundEven(ldexp(a, -qe)));
    uint bits;
    if (n < 8u) bits = n;
    else { int p = findMSB(n); bits = (uint(qe + p + 7) << 3) | ((n >> uint(p - 3)) & 7u); }
    return s | min(bits, 0x7eu);
}
#if defined(RR_GFX11)
// f16 -> e4m3 as ZLUDA computes it on gfx11 (HIP's satfinite conversion, infinities clamped first): nearest even,
// |x| > 448 -> 0x7e, NaN -> 0x7f, the sign kept; .relu first zeroes negative non-NaN halves
uint g11_h2e4m3(uint h)
{
    const uint s = (h >> 8u) & 0x80u, a = h & 0x7fffu;
    if (a > 0x7c00u) return s | 0x7fu;
    uint c;
    if (a >= 0x2400u) {
        const uint t = a - 0x2000u;
        c = min((t + 0x3fu + ((t >> 7u) & 1u)) >> 7u, 0x7eu);
    } else {
        c = uint(roundEven(float(uint16BitsToFloat16(uint16_t(a))) * 512.0));
    }
    return s | c;
}
#define G11_HAVE_H2E4M3
uint g11_relu(uint h) { return (h & 0x8000u) != 0u && (h & 0x7fffu) <= 0x7c00u ? 0u : h; }
uint f16x2_to_e4m3x2(f16vec2 h, bool relu)
{
    uint x = HU(h.x), y = HU(h.y);
    if (relu) { x = g11_relu(x); y = g11_relu(y); }
    return g11_h2e4m3(x) | (g11_h2e4m3(y) << 8);
}
#elif defined(SOFTWARE_E4M3)
uint f16x2_to_e4m3x2(f16vec2 h, bool relu) { return h2e4m3(float(h.x), relu) | (h2e4m3(float(h.y), relu) << 8); }
#else
// cvt.rn.satfinite(.relu).e4m3x2.f16x2 on the hardware conversion: round to nearest even after clamping to +-448
// (satfinite), NaN to 0x7f. Equal to the software h2e4m3 for all 65536 f16 inputs (2026-10-12) and about two thirds
// of a stage's time faster (dec0: 2.85 -> 1.03 ms).
uint h2e4m3_hw(float16_t h, bool relu)
{
    if (isnan(h)) return 0x7fu;
    if (relu) h = h > float16_t(0.0) ? h : float16_t(0.0);
    return uint(floate4m3BitsToUintEXT(floate4m3_t(clamp(h, float16_t(-448.0), float16_t(448.0)))));
}
uint f16x2_to_e4m3x2(f16vec2 h, bool relu) { return h2e4m3_hw(h.x, relu) | (h2e4m3_hw(h.y, relu) << 8); }
#endif

// mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 on the FP8 cooperative matrices, with ZLUDA's lowering of
// it (engine/layer_m.comp's TEXL): C enters the first of two 16x16x16 products, the sum stays f32 and is rounded to
// binary16 once. The operands pass through the subgroup's own scratch (MMS): NVIDIA's fragments are written in
// place (A row-major 16x32, B column-major 32x8 in a 16-column tile, C/D row-major f16 16x16), then loaded as
// whole matrices. Lane l = 4g + t holds A rows g and g+8 (k 4t.. and 16+4t..), B column g (same k), C/D row g
// and g+8, columns 2t and 2t+1.
#define MMF coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define MMH coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#ifdef RR_GFX11
// RDNA3: f16 operands holding ZLUDA's widening (value * 2^-8, k positions 2q + j <- 4q + j, 8 + 2q + j <- 4q + 2 + j
// in each K half), C scaled by 2^-16 into the two products and back (engine/rr/rr_gfx11.glsl)
#define MMA coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MMB coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
uvec2 g11_sc4(uint packed)
{
    uint mag = packed & 0x7f7f7f7fu;
    mag |= (mag + 0x01010101u) & 0x80808080u;
    const uint sign = packed & 0x80808080u;
    return uvec2(((mag & 0xffu) << 7u) | ((mag & 0xff00u) << 15u) | ((sign & 0x80u) << 8u) | ((sign & 0x8000u) << 16u),
                 (((mag >> 16u) & 0xffu) << 7u) | ((mag >> 24u) << 23u) | ((sign & 0x800000u) >> 8u) | (sign & 0x80000000u));
}
#define G11_FILL(m, w0, w1, w2, w3) { \
    const uvec2 s0_ = g11_sc4(w0), s1_ = g11_sc4(w1), s2_ = g11_sc4(w2), s3_ = g11_sc4(w3); f16vec2 l_; \
    l_ = unpackFloat2x16(s0_.x); (m)[0] = l_.x; (m)[1] = l_.y; l_ = unpackFloat2x16(s0_.y); (m)[8] = l_.x; (m)[9] = l_.y; \
    l_ = unpackFloat2x16(s1_.x); (m)[2] = l_.x; (m)[3] = l_.y; l_ = unpackFloat2x16(s1_.y); (m)[10] = l_.x; (m)[11] = l_.y; \
    l_ = unpackFloat2x16(s2_.x); (m)[4] = l_.x; (m)[5] = l_.y; l_ = unpackFloat2x16(s2_.y); (m)[12] = l_.x; (m)[13] = l_.y; \
    l_ = unpackFloat2x16(s3_.x); (m)[6] = l_.x; (m)[7] = l_.y; l_ = unpackFloat2x16(s3_.y); (m)[14] = l_.x; (m)[15] = l_.y; }
#define G11_HAVE_FILL
#define coopMatLoad8(m, arr, off, stride) { const uint b_ = (off) + (gl_SubgroupInvocationID & 15u) * (stride); \
    G11_FILL(m, arr[b_], arr[b_ + 1u], arr[b_ + 2u], arr[b_ + 3u]) }
#define MM8(x1, y1, x0, y0, c) MMF acc_ = MMF(c); for (int i_ = 0; i_ < acc_.length(); ++i_) acc_[i_] *= 1.52587890625e-05; \
    acc_ = coopMatMulAdd(x1, y1, coopMatMulAdd(x0, y0, acc_)); for (int i_ = 0; i_ < acc_.length(); ++i_) acc_[i_] *= 65536.0;
#else
#define MMA coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MMB coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define coopMatLoad8(m, arr, off, stride) coopMatLoad(m, arr, off, stride, gl_CooperativeMatrixLayoutRowMajor)
#define MM8(x1, y1, x0, y0, c) MMF acc_ = coopMatMulAdd(x1, y1, coopMatMulAdd(x0, y0, MMF(c)));
#endif
void mma_e4m3_f16(uint a0, uint a1, uint a2, uint a3, uint b0, uint b1, uint c0, uint c1, out uint d0, out uint d1)
{
    const uint base = gl_SubgroupID * 384u, A = base, B = base + 128u, C = base + 256u;
    const uint lane = gl_SubgroupInvocationID & 31u, g = lane >> 2u, t = lane & 3u;
    // order this operation's writes after every lane's reads of the previous one's results (the same scratch)
    memoryBarrierShared(); subgroupBarrier();
    mms[A + g * 8u + t] = a0; mms[A + (g + 8u) * 8u + t] = a1;
    mms[A + g * 8u + 4u + t] = a2; mms[A + (g + 8u) * 8u + 4u + t] = a3;
    mms[B + g * 8u + t] = b0; mms[B + g * 8u + 4u + t] = b1;
    mms[B + (g + 8u) * 8u + t] = 0u; mms[B + (g + 8u) * 8u + 4u + t] = 0u;
    mms[C + g * 8u + t] = c0; mms[C + (g + 8u) * 8u + t] = c1;
    mms[C + g * 8u + 4u + t] = 0u; mms[C + (g + 8u) * 8u + 4u + t] = 0u;
    memoryBarrierShared(); subgroupBarrier();
    MMA x0, x1; MMB y0, y1; MMH c;
#ifdef RR_GFX11
    coopMatLoad8(x0, mms, A, 8u);
    coopMatLoad8(x1, mms, A + 4u, 8u);
    coopMatLoad8(y0, mms, B, 8u);
    coopMatLoad8(y1, mms, B + 4u, 8u);
    coopMatLoad(c, mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
    MM8(x1, y1, x0, y0, c)
    memoryBarrierShared(); subgroupBarrier();
    coopMatStore(MMH(acc_), mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
#else
    coopMatLoad(x0, mms, A, 8u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(x1, mms, A + 4u, 8u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(y0, mms, B, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(y1, mms, B + 4u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(c, mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
    MMF acc = coopMatMulAdd(x1, y1, coopMatMulAdd(x0, y0, MMF(c)));
    memoryBarrierShared(); subgroupBarrier();
    coopMatStore(MMH(acc), mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
#endif
    memoryBarrierShared(); subgroupBarrier();
    d0 = mms[C + g * 8u + t]; d1 = mms[C + (g + 8u) * 8u + t];
}
// Two m16n8k32 products with the same A fragment (NVIDIA issues every one of them in such adjacent pairs): one 16x16
// product, the first's B, C and D in columns 0-7, the second's in columns 8-15.
void mma_e4m3_f16_pair(uint a0, uint a1, uint a2, uint a3, uint b0, uint b1, uint c0, uint c1,
                       uint e0, uint e1, uint f0, uint f1, out uint d0, out uint d1, out uint h0, out uint h1)
{
    const uint base = gl_SubgroupID * 384u, A = base, B = base + 128u, C = base + 256u;
    const uint lane = gl_SubgroupInvocationID & 31u, g = lane >> 2u, t = lane & 3u;
    memoryBarrierShared(); subgroupBarrier();
    mms[A + g * 8u + t] = a0; mms[A + (g + 8u) * 8u + t] = a1;
    mms[A + g * 8u + 4u + t] = a2; mms[A + (g + 8u) * 8u + 4u + t] = a3;
    mms[B + g * 8u + t] = b0; mms[B + g * 8u + 4u + t] = b1;
    mms[B + (g + 8u) * 8u + t] = e0; mms[B + (g + 8u) * 8u + 4u + t] = e1;
    mms[C + g * 8u + t] = c0; mms[C + (g + 8u) * 8u + t] = c1;
    mms[C + g * 8u + 4u + t] = f0; mms[C + (g + 8u) * 8u + 4u + t] = f1;
    memoryBarrierShared(); subgroupBarrier();
    MMA x0, x1; MMB y0, y1; MMH c;
#ifdef RR_GFX11
    coopMatLoad8(x0, mms, A, 8u);
    coopMatLoad8(x1, mms, A + 4u, 8u);
    coopMatLoad8(y0, mms, B, 8u);
    coopMatLoad8(y1, mms, B + 4u, 8u);
    coopMatLoad(c, mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
    MM8(x1, y1, x0, y0, c)
    memoryBarrierShared(); subgroupBarrier();
    coopMatStore(MMH(acc_), mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
#else
    coopMatLoad(x0, mms, A, 8u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(x1, mms, A + 4u, 8u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(y0, mms, B, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(y1, mms, B + 4u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(c, mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
    MMF acc = coopMatMulAdd(x1, y1, coopMatMulAdd(x0, y0, MMF(c)));
    memoryBarrierShared(); subgroupBarrier();
    coopMatStore(MMH(acc), mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
#endif
    memoryBarrierShared(); subgroupBarrier();
    d0 = mms[C + g * 8u + t]; d1 = mms[C + (g + 8u) * 8u + t];
    h0 = mms[C + g * 8u + 4u + t]; h1 = mms[C + (g + 8u) * 8u + 4u + t];
}
'''


def statement_lines(ptx, kernel):
    """the source line of every statement of KERNEL, in the order vulkan/ptx2glsl.py's tokenizer yields them"""
    text = re.sub(r'//[^\n]*', '', ptx.replace('\r', ''))
    m = re.search(r'\.entry\s+' + re.escape(kernel) + r'\s*\((.*?)\)', text, re.S)
    start = text.index('{', m.end())
    depth, k = 0, start
    while True:
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                break
        k += 1
    lines, i, line = [], start + 1, text.count('\n', 0, start + 1) + 1
    while i < k:
        c = text[i]
        if c.isspace():
            line += c == '\n'
            i += 1
            continue
        if c in '{}':
            i += 1
            continue
        lm = re.match(r'(\$?[A-Za-z_][\w$]*):', text[i:i + 200])
        if lm:
            i += lm.end()
            continue
        j = text.index(';', i)
        lines.append(line)
        line += text.count('\n', i, j)
        i = j + 1
    return lines


def base_helper(name):
    """one helper function of vulkan/ptx2glsl.py's runtime, with its leading comment"""
    lines = base.RUNTIME.split('\n')
    start = next(i for i, l in enumerate(lines) if re.match(r'\w+ ' + name + r'\(', l))
    while lines[start - 1].startswith('//'):
        start -= 1
    end = start
    if not lines[start].rstrip().endswith('}'):    # one-line helpers end on their own line
        while lines[end] != '}':
            end += 1
    return '\n'.join(lines[start:end + 1]) + '\n'


RR_RUNTIME = RR_RUNTIME.replace('Q1023_VALUES', ', '.join(repr(float(np.float32(q) / np.float32(1023))) for q in range(1024)))
RR_RUNTIME += base_helper('prmt') + base_helper('shfl_bfly').replace(
    'return subgroupShuffle(v, (gl_SubgroupInvocationID & 32u) | uint(idx));',
    'return subgroupShuffle(v, (gl_SubgroupInvocationID & 32u) | (uint(idx) & 31u));')


class RRT(base.T):
    def __init__(self, ptx, kernel, block, weights_arg=40, buffers=()):
        self.weights_arg = weights_arg
        self.buffers = sorted(buffers)   # argument offsets of further buffers, each its own binding
        self.written_buffers = set()
        self.unified_arena = False
        self.snorm16 = set()             # texture arguments of --snorm16
        bx, by, bz = block
        if bx * by * bz % 32:
            raise base.Err('block size must be a multiple of the 32-lane subgroup')
        self.bx, self.by, self.bz = bx, by, bz
        self.sreg = {
            '%tid.x': f'(gl_LocalInvocationIndex % {bx}u)',
            '%tid.y': f'((gl_LocalInvocationIndex / {bx}u) % {by}u)',
            '%tid.z': f'(gl_LocalInvocationIndex / {bx * by}u)',
            '%ntid.x': f'{bx}u', '%ntid.y': f'{by}u', '%ntid.z': f'{bz}u',
            '%ctaid.x': 'gl_WorkGroupID.x', '%ctaid.y': 'gl_WorkGroupID.y', '%ctaid.z': 'gl_WorkGroupID.z',
            '%nctaid.x': 'gl_NumWorkGroups.x', '%nctaid.y': 'gl_NumWorkGroups.y', '%nctaid.z': 'gl_NumWorkGroups.z',
            '%laneid': 'gl_SubgroupInvocationID',
            'WARP_SZ': '32u',
        }
        self.origin = {}
        self.f64 = {}   # %fd registers: the kernels only load f64 immediates into them, to round them to f16
        self.probes = {}         # debugging: PTX line -> registers stored to the debug buffer after it executes
        self.probe_block = (0, 0)
        self.probe_slots = 0
        self.oob = False
        # every register precise, the integer ones too: they carry f16x2 and f32 bit patterns between operations, and
        # without it the compiler may contract a product and a sum across them depending on scheduling (results then
        # changed with unrelated code and with the device's robustness features)
        self.precise_all = True
        # module globals with an initialiser (the output filter's bicubic phase table): constants, laid out one after
        # another in a blob that is the stage's weight buffer (binding 1); `mov.u64 %rd, NAME` is its offset there
        self.globals, self.globals_blob = {}, b''
        for gm in re.finditer(r'\.(?:global|const)\s+\.align\s+\d+\s+\.b8\s+(\w+)\[(\d+)\]\s*=\s*\{([^}]*)\}', ptx):
            self.globals[gm.group(1)] = len(self.globals_blob)
            data = bytes(int(x) for x in gm.group(3).split(','))
            self.globals_blob += data + bytes(-len(data) % 16)
        self.stmt_lines = statement_lines(ptx, kernel)
        self.stmt_index = 0
        super().__init__(ptx, kernel)
        if not os.environ.get('D4R_PTX2RR_NO_PAIR'):
            self.pair_mma()
        self.origins()
        self.low32 = self.low32_regs()

    def low32_regs(self):
        """The %rd registers whose value only matters modulo 2^32: every use is the address of a global load or store
        (the translation takes its low 32 bits, gbyte), a truncation to 32 bits, or an operand of an add, subtract,
        low multiply, move or left shift whose result is such a register too. A wide multiply or a 64-bit left shift
        into one is translated in 32 bits (op): the same low 32 bits, without the 64-bit arithmetic the driver keeps."""
        uses = {}   # register -> [(kind, destination)]: 'ok' a truncating use, 'dep' fine if the destination is low32
        regs = set()
        dep_ops = ('add.s64', 'add.u64', 'sub.s64', 'sub.u64', 'mul.lo.s64', 'mul.lo.u64', 'mov.b64', 'mov.u64',
                   'mov.s64', 'cvta.to.global.u64', 'cvta.global.u64')
        for kind, text in self.toks:
            if kind != 'stmt' or text.startswith('.'):
                continue
            text = re.sub(r'^@!?%p\d+\s+', '', text)
            parts = text.split(None, 1)
            if len(parts) < 2:
                continue
            op, a = parts[0], base.split_operands(parts[1])
            found = [(i, r) for i, t in enumerate(a) for r in re.findall(r'%rd\d+', t)]
            regs.update(r for _, r in found)
            o = op.split('.')
            for i, r in found:
                if i == 0 and not a[0].startswith('['):
                    continue   # a definition
                if o[0] in ('ld', 'st') and 'global' in o and a[1 if o[0] == 'ld' else 0].strip().startswith('[') \
                        and i == (1 if o[0] == 'ld' else 0):
                    uses.setdefault(r, []).append(('ok', None))
                elif op in ('cvt.u32.u64', 'cvt.s32.s64'):
                    uses.setdefault(r, []).append(('ok', None))
                elif op in dep_ops or (op == 'shl.b64' and i == 1):
                    uses.setdefault(r, []).append(('dep', a[0].strip()))
                else:
                    uses.setdefault(r, []).append(('bad', None))
        low = set(regs)
        changed = True
        while changed:
            changed = False
            for r in list(low):
                if any(k == 'bad' or (k == 'dep' and d not in low) for k, d in uses.get(r, [])):
                    low.discard(r); changed = True
        return low

    def stmt(self, s):
        line = self.stmt_lines[self.stmt_index] if self.stmt_index < len(self.stmt_lines) else None
        self.stmt_index += 1
        super().stmt(s)
        for reg in self.probes.get(line, ()):
            # the debug buffer: 32-bit words [slot][local invocation], the last execution of the statement
            n = self.reg(reg)
            g = self.regs[n]
            v = (f'floatBitsToUint({n})' if g == 'float' else f'uint({n})' if g in ('uint', 'uint64_t', 'bool') else
                 f'HU({n})' if g == 'float16_t' else f'H2U({n})')
            bx, by = self.probe_block
            self.E(f'if (gl_WorkGroupID.xy == uvec2({bx}u, {by}u)) dbg.d[{self.probe_slots}u * (gl_WorkGroupSize.x * gl_WorkGroupSize.y) + '
                   f'gl_LocalInvocationIndex] = {v};')
            self.probe_slots += 1

    def pair_mma(self):
        """merge adjacent mma statements with the same A fragment and independent operands into one paired product;
        the second statement becomes a no-op so statement numbering (probes) is unchanged"""
        stmts = [k for k, (kind, _) in enumerate(self.toks) if kind == 'stmt']
        n = 0
        while n + 1 < len(stmts):
            i, j = stmts[n], stmts[n + 1]
            x, y = self.toks[i][1], self.toks[j][1]
            if x.startswith('mma.sync.aligned.m16n8k32') and y.startswith('mma.sync.aligned.m16n8k32'):
                gx = [braces(t) for t in base.split_operands(x.split(None, 1)[1])]
                gy = [braces(t) for t in base.split_operands(y.split(None, 1)[1])]
                if gx[1] == gy[1] and not set(gx[0]) & set(gy[2] + gy[3] + gy[1]):
                    ops = ', '.join('{' + ', '.join(g) + '}' for g in (gx[0], gy[0], gx[1], gx[2], gx[3], gy[2], gy[3]))
                    self.toks[i] = ('stmt', 'mma.pair.e4m3 ' + ops)
                    self.toks[j] = ('stmt', 'mma.paired.noop')
                    n += 2
                    continue
            n += 1

    # ------------------------------------------------ SREG remap onto the flat workgroup
    def is_reg(self, tok):
        return tok.strip() not in self.sreg and super().is_reg(tok)

    def val(self, tok, t):
        tok = tok.strip()
        if re.fullmatch(r'(0[xX][0-9a-fA-F]+|-?\d+)[uU]', tok):
            tok = tok[:-1]
        tok = re.sub(r'^0X', '0x', tok)
        if tok in self.sreg:
            e = self.sreg[tok]
            return f'int({e})' if t == 's32' else f'uint64_t({e})' if t in ('u64', 'b64') else f'uint({e})'
        return super().val(tok, t)

    # ------------------------------------------------ global memory origin: weight buffer or arena
    def origins(self):
        stmts = []
        for kind, s in self.toks:
            if kind != 'stmt' or s.startswith('.'):
                continue
            s = re.sub(r'^@!?%p\d+\s+', '', s)
            parts = s.split(None, 1)
            stmts.append((parts[0], base.split_operands(parts[1]) if len(parts) > 1 else []))
        org = self.origin
        changed = True
        while changed:
            changed = False
            for op, a in stmts:
                if not a or not re.fullmatch(r'%rd\d+', a[0]) or op.startswith(('ld.', 'st.', 'bra')):
                    if op != 'ld.param.u64':
                        continue
                if op.startswith('mov') and len(a) > 1 and a[1].strip() in self.globals:
                    new = {'w'}
                elif op == 'ld.param.u64':
                    # the argument's true offset (the load may go through a register based inside the block)
                    off_ = self.param_off(a[1])
                    new = {'w' if off_ == self.weights_arg else f'x{off_}' if off_ in self.buffers else 'a'}
                else:
                    new = set()
                    for x in a[1:]:
                        if re.fullmatch(r'%rd\d+', x):
                            new |= org.get(x, set())
                cur = org.setdefault(a[0], set())
                if new - cur:
                    cur |= new
                    changed = True

    def gorigin(self, reg):
        s = self.origin.get(reg, set())
        if self.unified_arena and s and 'w' not in s:
            return 'a'
        if len(s) != 1:
            raise base.Err(f'global pointer {reg} has origin {sorted(s)} (need exactly one of w/a)')
        return next(iter(s))

    def gfn(self, reg, store=False):
        o = self.gorigin(reg)
        if o == 'w':
            return 'GW'
        if o == 'a':
            return 'SA' if store else 'GA'
        return ('SX' if store else 'GX') + o[1:] + '_'

    def gaddr(self, tok):
        m = re.fullmatch(r'\[\s*(%rd\d+)\s*(?:\+\s*(-?\d+))?\s*\]', tok.strip())
        if not m:
            raise base.Err(f'unsupported global address {tok}')
        return m.group(1), int(m.group(2) or 0)

    def gbyte(self, reg, off):
        return f'uint({self.val(reg, "u64")}) + {off}u' if off >= 0 else f'uint({self.val(reg, "u64")}) - {-off}u'

    def ld(self, o, a):
        if 'const' in o:
            m = re.fullmatch(r'\[\s*(\w+)\s*(?:\+\s*(\d+))?\s*\]', a[1].strip())
            if not m or m.group(1) not in self.globals:
                raise base.Err(f'unsupported constant address {a[1]}')
            size, count = mem_shape(o)
            at = self.globals[m.group(1)] + int(m.group(2) or 0)
            dsts = braces(a[0]) if count > 1 else [a[0]]
            for i, d in enumerate(dsts):
                self.A(d, f'GW{size * 8}({at + i * size}u)', 'u16' if size == 2 else 'u32')
            return
        if 'global' not in o:
            return super().ld(o, a)
        size, count = mem_shape(o)
        reg, off = self.gaddr(a[1])
        fn = self.gfn(reg)
        dsts = braces(a[0]) if count > 1 else [a[0]]
        for i, d in enumerate(dsts):
            self.A(d, f'{fn}{size * 8}({self.gbyte(reg, off + i * size)})', 'u16' if size == 2 else 'u32')

    def st(self, o, a):
        if 'shared' in o and o[-1] in ('u16', 'b16', 'f16') and not {'v2', 'v4'} & set(o):
            reg, off = self.addr(a[0])
            value = self.val(a[1], 'u16')
            self.E('{ uint at_ = ' + self.val(reg, 'u32') + f' + {off}u;')
            self.E('uint shift_ = (at_ & 2u) * 8u;')
            self.E('atomicAnd(sh[at_ >> 2], ~(0xffffu << shift_));')
            self.E(f'atomicOr(sh[at_ >> 2], (uint({value}) & 0xffffu) << shift_); }}')
            return
        if 'global' not in o:
            return super().st(o, a)
        size, count = mem_shape(o)
        reg, off = self.gaddr(a[0])
        if self.gorigin(reg) == 'w':
            raise base.Err(f'store through the read-only weight pointer: {a}')
        for origin in self.origin.get(reg, ()):
            if origin.startswith('x'):
                self.written_buffers.add(int(origin[1:]))
        fn = self.gfn(reg, True)
        vals = braces(a[1]) if count > 1 else [a[1]]
        for i, v in enumerate(vals):
            if size == 2:
                self.E(f'{fn}16({self.gbyte(reg, off + i * size)}, {self.val(v, "u16")});')
            else:
                self.E(f'{fn}32({self.gbyte(reg, off + i * size)}, {self.val(v, "u32")});')

    # ------------------------------------------------ textures and surfaces
    def handle_off(self, h):
        off = self.handle.get(h)
        if off is None:
            raise base.Err(f'{h} is not a texture or surface handle loaded from the parameter block')
        return off

    def set_surface(self, off, fmt):
        old = self.surfaces.get(off)
        if old is not None and old != fmt:
            raise base.Err(f'surface {off} used with formats {old} and {fmt}')
        self.surfaces[off] = fmt

    def tex(self, op, a):
        o = op.split('.')[1:]
        dst = braces(a[0])
        h, c = self.texref(a[1])
        off = self.handle_off(h)
        self.textures.add(off)
        if 'base' in o and o[-1] == 's32':
            # an integer fetch outside the image: CUDA applies the texture's address mode (clamp for every denoiser
            # texture), but the reference this engine is checked against is ZLUDA, whose image load returns zero there
            # (zluda_ptx_impl.cpp fetch_2D). The output stage's specular reprojection reads such neighbours at the image
            # border: with clamp its surfaces were 99.92% identical to the capture, with zero 99.997% (capDX, 2026-10-12)
            fetch = f'fetch0(T{off}, ivec2({self.val(c[0], "s32")}, {self.val(c[1], "s32")}))'
        elif ('base' in o or o[0] == '2d') and o[-1] == 'f32':   # tex.2d: no derivatives in a kernel, level 0
            fetch = f'textureLod(T{off}, vec2({self.val(c[0], "f32")}, {self.val(c[1], "f32")}), 0.0)'
        elif 'level' in o and o[-1] == 'f32':
            fetch = (f'textureLod(T{off}, vec2({self.val(c[0], "f32")}, {self.val(c[1], "f32")}), '
                     f'{self.val(a[2], "f32")})')
        else:
            raise base.Err(f'unsupported texture op {op}')
        f16 = 'f16' in o
        self.E(f'{{ vec4 t_ = {fetch};')
        if off in self.snorm16 and o[-1] == 'f32':
            # --snorm16: a filtered fetch of an 8-bit SNORM texture on an NVIDIA card is this card's value (8-bit weights)
            # rounded to a 16-bit step (measured on enc0's reprojected normals, build/rr/f/nv; D4R_RR_FUSE models)
            self.E('{ precise vec4 q_ = roundEven(t_ * 32767.0); precise vec4 d_ = q_ / 32767.0; t_ = d_; }')
        for i, d in enumerate(dst):
            # a binary16 result is the texel rounded to binary16: through H(HU(...) | SUST_Z) so that the driver cannot
            # fold the rounding away when the value is widened again (RR_RUNTIME's note on sust_b)
            self.A(d, (f'H(HU(float16_t(t_[{i}])) | SUST_Z)' if CVT_GUARD else f'float16_t(t_[{i}])') if f16 else f't_[{i}]', 'f16' if f16 else 'float')
        self.E('}')

    def sust(self, op, a):
        h, c = self.texref(a[0])
        off = self.handle_off(h)
        v = braces(a[1])
        x, y = self.val(c[0], 's32'), self.val(c[1], 's32')
        S = f'S{off}'
        op = op.replace('.cs.', '.')   # the streaming cache policy: a hint, the store is the same
        if op == 'sust.p.2d.v4.b32.zero':
            self.set_surface(off, 'rgba16f')
            self.E(f'sust_p4({S}, {x}, {y}, SUSTQ_{off}(vec4({", ".join(self.val(q, "f32") for q in v)})));')
        elif op == 'sust.p.2d.b32.zero':
            self.set_surface(off, 'r16f')
            self.E(f'sust_p1({S}, {x}, {y}, {self.val(v[0], "f32")});')
        elif op == 'sust.b.2d.v2.b16.zero':
            self.set_surface(off, 'rg16f')
            self.E(f'sust_b2({S}, {x}, {y}, {", ".join(self.val(q, "u16") for q in v)});')
        elif op == 'sust.b.2d.v4.b16.zero':
            self.set_surface(off, 'rgba16f')
            self.E(f'sust_b4({S}, {x}, {y}, {", ".join(self.val(q, "u16") for q in v)});')
        else:
            raise base.Err(f'unsupported surface op {op}')

    # ------------------------------------------------ warp-level ops
    def mma(self, a):
        d, x, b, c = (braces(t) for t in a)
        if not (len(d) == 2 and len(x) == 4 and len(b) == 2 and len(c) == 2):
            raise base.Err('mma operand shape')
        args = ', '.join(self.val(r, 'u32') for r in x + b + c)
        self.E('{ uint md0_, md1_;')
        self.E(f'mma_e4m3_f16({args}, md0_, md1_);')
        self.A(d[0], 'md0_', 'u32')
        self.A(d[1], 'md1_', 'u32')
        self.E('}')

    def shfl_dir(self, op, a):
        down = op.split('.')[2] == 'down'
        d = a[0].split('|')
        self.E('{ bool ib_;')
        self.A(d[0], f'shfl_dir({self.val(a[1], "u32")}, {self.val(a[2], "u32")}, {self.val(a[3], "u32")}, '
                     f'{"true" if down else "false"}, ib_)', 'u32')
        if len(d) > 1:
            self.A(d[1], 'ib_', 'bool')
        self.E('}')

    def cvt_to_e4m3(self, op, a):
        relu = 'relu' in op.split('.')
        self.A(a[0], f'f16x2_to_e4m3x2({self.val(a[1], "f16x2")}, {"true" if relu else "false"})', 'u16')

    def cvt(self, op, o, a):
        types = [t for t in o if re.match(r'^(f|s|u|b)\d+(x2)?$', t)]
        if types[:2] in (['u32', 'u64'], ['s32', 's64'], ['u32', 's64'], ['s32', 'u64']):
            return self.A(a[0], f'uint({self.val(a[1], "u64")})', 'u32')
        if types[:2] == ['f16', 'f32'] and 'rn' in o and 'satfinite' not in o:
            # Two things the driver must not do. It folds float(float16_t(x)) to x when the result is widened again
            # (RR_RUNTIME's note on sust_b): the OR on the binary16 bits. And it fuses a product or sum with this
            # conversion into v_fma_mix, one rounding of the exact value, where the kernel rounds to f32 first: an f32
            # result that is exactly a binary16 midpoint then rounds the other way (the output stages' rare one-step
            # differences, e.g. log2(x) * c1 * c2; found 2026-10-13): the OR on the f32 bits keeps the f32 value.
            # With --exact-convert the conversion itself is decorated NoContraction (spv_exact_convert.py), which keeps the
            # first: only the f32 OR is needed (checked in the ISA: the same v_cvt_f16_f32 / v_cvt_f32_f16 counts, no
            # v_fma_mix*_f16; without the f32 OR, ACO fuses some products into v_fma_mixlo_f16 despite `precise`)
            if not CVT_GUARD:
                return self.A(a[0], f'float16_t(uintBitsToFloat(floatBitsToUint({self.val(a[1], "f32")}) | SUST_Z))', 'f16')
            return self.A(a[0], f'H(HU(float16_t(uintBitsToFloat(floatBitsToUint({self.val(a[1], "f32")}) | SUST_Z))) | SUST_Z)', 'f16')
        return super().cvt(op, o, a)

    def f64_op(self, op, a):
        if op == 'mov.f64' and re.fullmatch(r'0[dD][0-9A-Fa-f]{16}', a[1].strip()):
            self.f64[a[0].strip()] = struct.unpack('<d', int(a[1].strip()[2:], 16).to_bytes(8, 'little'))[0]
            return True
        if op == 'cvt.rn.f16.f64':
            v = self.f64.get(a[1].strip())
            if v is None:
                raise base.Err(f'{a[1]} is not a known f64 constant')
            bits = int(np.float16(v).view(np.uint16))
            self.A(a[0], f'{bits}u', 'u16')
            return True
        return False

    def op(self, op, a):
        parts = op.split('.')
        if '%fd' in ','.join(a) and self.f64_op(op, a):
            return
        if op.startswith('mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16'):
            return self.mma(a)
        if op in ('mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16',
                  'mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16'):
            d, x, b, c = (braces(t) for t in a)
            depth = 8 if '.m16n8k8.' in op else 16
            if (len(d), len(x), len(b), len(c)) != (2, depth // 4, depth // 8, 2):
                raise base.Err(f'invalid fragments for {op}')
            operands = ', '.join(self.val(r, 'u32') for r in x + b + c)
            self.E('{ uint md0_, md1_;')
            self.E(f'mma_f16_m16n8k{depth}({operands}, md0_, md1_);')
            self.A(d[0], 'md0_', 'u32')
            self.A(d[1], 'md1_', 'u32')
            return self.E('}')
        if op.startswith('mma.sync.'):
            raise base.Err(f'unsupported matrix operation {op}')
        if op == 'mma.pair.e4m3':
            d, h, x, b, c, e, f = (braces(t) for t in a)
            args = ', '.join(self.val(r, 'u32') for r in x + b + c + e + f)
            self.E('{ uint md0_, md1_, mh0_, mh1_;')
            self.E(f'mma_e4m3_f16_pair({args}, md0_, md1_, mh0_, mh1_);')
            for reg, v in zip(d + h, ('md0_', 'md1_', 'mh0_', 'mh1_')):
                self.A(reg, v, 'u32')
            return self.E('}')
        if op == 'mma.paired.noop':
            return
        if parts[0] == 'mov' and len(a) > 1 and a[1].strip() in self.globals:
            return self.A(a[0], f'uint64_t({self.globals[a[1].strip()]}u)', 'u64')
        # cross-kernel progress counters (NGX runs the output passes concurrently and has each wait for the rows it
        # needs): plain loads and adds here. The engine runs the passes one after another and leaves the counter
        # pointers null, which the kernels test, so these never execute.
        if parts[0] == 'ld' and ('relaxed' in parts or 'acquire' in parts) and 'global' in parts:
            return self.ld([q for q in parts[1:] if q not in ('relaxed', 'acquire', 'gpu') and '::' not in q and not q.startswith('L1')], a)
        if parts[0] == 'red' and parts[-2] == 'add':
            reg, off = self.gaddr(a[0])
            fn, sfn = self.gfn(reg), self.gfn(reg, True)
            return self.E(f'{sfn}32({self.gbyte(reg, off)}, {fn}32({self.gbyte(reg, off)}) + {self.val(a[1], "u32")});')
        if op == 'membar.gl':
            return
        # a wide multiply or a 64-bit left shift whose result only matters modulo 2^32 (low32_regs), in 32 bits
        if op in ('mul.wide.u32', 'mul.wide.s32') and a[0].strip() in self.low32:
            return self.A(a[0], f'uint64_t({self.val(a[1], "u32")} * {self.val(a[2], "u32")})', 'u64')
        if op == 'shl.b64' and a[0].strip() in self.low32:
            n = self.val(a[2], 'u32')
            return self.A(a[0], f'uint64_t(({n} & 63u) >= 32u ? 0u : uint({self.val(a[1], "u64")}) << ({n} & 63u))', 'u64')
        # ... and the sums and products of such registers (in 32 bits too, so that the products above stay in 32 bits)
        if op in ('add.s64', 'add.u64', 'sub.s64', 'sub.u64', 'mul.lo.s64', 'mul.lo.u64') and a[0].strip() in self.low32:
            x, y = (f'uint({self.val(t, "u64")})' for t in a[1:3])
            return self.A(a[0], f'uint64_t({x} {"+" if parts[0] == "add" else "-" if parts[0] == "sub" else "*"} {y})', 'u64')
        if op == 'activemask.b32':
            # the warp's lanes: this half of a 64-wide wave
            return self.A(a[0], '((gl_SubgroupInvocationID & 32u) != 0u ? subgroupBallot(true).y : subgroupBallot(true).x)', 'u32')
        if op == 'movmatrix.sync.trans.aligned.m8n8.b16':
            return self.A(a[0], f'movm_trans({self.val(a[1], "u32")})', 'u32')
        # PTX clamps shift amounts at the register width (shl/shr.u give 0, shr.s the sign); GLSL leaves them undefined
        if parts[0] in ('shl', 'shr') and len(parts) == 2 and parts[1] in ('b32', 'u32', 's32'):
            x, n = self.val(a[1], 's32' if parts[1] == 's32' else 'u32'), self.val(a[2], 'u32')
            if parts[0] == 'shl':
                return self.A(a[0], f'(({n}) >= 32u ? 0u : ({x}) << ({n}))', 'u32')
            if parts[1] == 's32':
                return self.A(a[0], f'uint(({x}) >> min({n}, 31u))', 'u32')
            return self.A(a[0], f'(({n}) >= 32u ? 0u : ({x}) >> ({n}))', 'u32')
        if op == 'shfl.sync.idx.b32':
            d = a[0].split('|')
            self.E('{ bool ib_;')
            self.A(d[0], f'shfl_idx({self.val(a[1], "u32")}, {self.val(a[2], "u32")}, {self.val(a[3], "u32")}, ib_)', 'u32')
            if len(d) > 1:
                self.A(d[1], 'ib_', 'bool')
            return self.E('}')
        if op in ('shfl.sync.down.b32', 'shfl.sync.up.b32'):
            return self.shfl_dir(op, a)
        if parts[0] == 'sust':
            return self.sust(op, a)
        if parts[0] == 'tex':
            return self.tex(op, a)
        if parts[0] == 'cvt' and [t for t in parts[1:] if t in ('f16x2', 'e4m3x2')] == ['e4m3x2', 'f16x2']:
            return self.cvt_to_e4m3(op, a)
        if op == 'copysign.f32':   # PTX: the sign of the first source on the value of the second
            x, y = self.val(a[1], 'f32'), self.val(a[2], 'f32')
            return self.A(a[0], f'uintBitsToFloat((floatBitsToUint({y}) & 0x7fffffffu) | (floatBitsToUint({x}) & 0x80000000u))', 'float')
        if op == 'tanh.approx.f16':
            return self.A(a[0], f'tanh16({self.val(a[1], "f16")})', 'f16')
        if op == 'tanh.approx.f32':
            return self.A(a[0], f'tanh_ocml({self.val(a[1], "f32")})', 'float')
        if op == 'ex2.approx.f16':
            # v_exp_f32 of the widened value, rounded to binary16; the ORs keep that rounding (see cvt)
            if not CVT_GUARD:
                return self.A(a[0], f'float16_t(uintBitsToFloat(floatBitsToUint(exp2(float({self.val(a[1], "f16")}))) | SUST_Z))', 'f16')
            return self.A(a[0], f'H(HU(float16_t(uintBitsToFloat(floatBitsToUint(exp2(float({self.val(a[1], "f16")}))) | SUST_Z))) | SUST_Z)', 'f16')
        if op == 'neg.f16':
            return self.A(a[0], f'-{self.val(a[1], "f16")}', 'f16')
        if parts[0] in ('min', 'max') and 'NaN' not in parts and any(t in parts for t in ('f16', 'f16x2', 'f32')):
            t = 'f16x2' if 'f16x2' in parts else 'f16' if 'f16' in parts else 'f32'
            fn = {'f16x2': 'p{}2', 'f16': 'p{}', 'f32': 'p{}f'}[t].format(parts[0])
            x, y = self.val(a[1], t), self.val(a[2], t)
            return self.A(a[0], f'{fn}({x}, {y})', 'float' if t == 'f32' else t)
        if op == 'neg.f16x2':
            return self.A(a[0], f'-{self.val(a[1], "f16x2")}', 'f16x2')
        if op == 'abs.f16x2':
            return self.A(a[0], f'abs({self.val(a[1], "f16x2")})', 'f16x2')
        if parts[0] == 'set' and parts[2] == 'f16':
            c = ('(isnan({0}) || isnan({1}))' if parts[1] == 'nan' else
                 '!(isnan({0}) || isnan({1}))' if parts[1] == 'num' else self.cmp(set(parts), True))
            return self.A(a[0], f'(({c.format(self.val(a[1], "f16"), self.val(a[2], "f16"))}) ? 1.0hf : 0.0hf)', 'f16')
        if op == 'popc.b32':
            return self.A(a[0], f'uint(bitCount({self.val(a[1], "u32")}))', 'u32')
        if op in ('dp2a.lo.u32.u32', 'dp2a.hi.u32.u32'):
            x, y, c = (self.val(t, 'u32') for t in a[1:])
            shift = 0 if parts[1] == 'lo' else 16
            return self.A(a[0], f'({c} + ({x} & 0xffffu) * (({y} >> {shift}u) & 0xffu) + '
                          f'({x} >> 16u) * (({y} >> {shift + 8}u) & 0xffu))', 'u32')
        if op.startswith('ldmatrix.sync.aligned.m8n8.'):
            regs = braces(a[0])
            count = 4 if 'x4' in parts else 2 if 'x2' in parts else 1
            if len(regs) != count or parts[-1] != 'b16':
                raise base.Err(f'unsupported matrix load {op}')
            src, off = self.addr(a[1])
            addr = f'({self.val(src, "u32")} + {off}u)'
            self.E('{ uint lane_ = gl_SubgroupInvocationID;')
            for i, dst in enumerate(regs):
                if 'trans' in parts:
                    self.E(f'uint lo_{i} = subgroupShuffle({addr}, {8*i}u + 2u * (lane_ & 3u)) + (lane_ >> 2u) * 2u;')
                    self.E(f'uint hi_{i} = subgroupShuffle({addr}, {8*i + 1}u + 2u * (lane_ & 3u)) + (lane_ >> 2u) * 2u;')
                    self.A(dst, f'(SH16(lo_{i}) | (SH16(hi_{i}) << 16u))', 'u32')
                else:
                    self.E(f'uint row_{i} = subgroupShuffle({addr}, (lane_ >> 2u) + {8*i}u);')
                    self.A(dst, f'SH32(row_{i} + (lane_ & 3u) * 4u)', 'u32')
            return self.E('}')
        if op.startswith('cp.async'):
            if op == 'cp.async.commit_group' or op.startswith('cp.async.wait_group'):
                return  # Copies below execute synchronously in program order.
            if not op.startswith(('cp.async.cg.shared.global', 'cp.async.ca.shared.global')):
                raise base.Err(f'unsupported asynchronous copy {op}')
            dst, doff = self.addr(a[0])
            src, soff = self.gaddr(a[1])
            size = int(a[2])
            if size not in (4, 8, 16):
                raise base.Err(f'unsupported asynchronous copy size {size}')
            valid = self.val(a[3], 'u32') if len(a) > 3 else f'{size}u'
            fn = self.gfn(src)
            self.E('{ uint dst_ = ' + self.val(dst, 'u32') + f' + {doff}u;')
            self.E(f'uint bytes_ = {valid};')
            for i in range(size // 4):
                self.E(f'uint word_{i} = 0u;')
                self.E(f'if (bytes_ > {4*i}u) {{')
                self.E(f'uint n_ = min(bytes_ - {4*i}u, 4u);')
                self.E(f'word_{i} = {fn}32({self.gbyte(src, soff + 4*i)}) & (0xffffffffu >> ((4u - n_) * 8u)); }}')
                self.E(f'sh[(dst_ + {4*i}u) >> 2] = word_{i};')
            return self.E('}')
        return super().op(op, a)

    # ------------------------------------------------ stage contract
    def preamble(self, w, tex_filter):
        w('#version 460')
        w('#extension GL_EXT_shader_explicit_arithmetic_types : require')
        w('#extension GL_EXT_shader_16bit_storage : require')
        w('#extension GL_EXT_control_flow_attributes : require')
        w('#extension GL_KHR_shader_subgroup_basic : require')
        w('#extension GL_KHR_shader_subgroup_shuffle : require')
        w('#extension GL_KHR_shader_subgroup_ballot : require')
        w('#extension GL_KHR_cooperative_matrix : require')
        w('#extension GL_KHR_memory_scope_semantics : require')
        w('#ifndef RR_GFX11')
        w('#extension GL_EXT_float_e4m3 : require')
        w('#endif')
        w('#extension GL_EXT_spirv_intrinsics : require')
        w(f'// Generated by engine/rr/ptx2rr.py from {self.kernel}. Model data derived from the user\'s DLL; do not commit.')
        w(f'layout(local_size_x = {self.bx * self.by * self.bz}, local_size_y = 1, local_size_z = 1) in;')
        w('layout(set = 0, binding = 0, std430) readonly buffer ParamBlock { uint w[]; } param;')
        w('layout(set = 0, binding = 1, std430) readonly buffer WeightBuf { uint16_t wh[]; } wbuf;')
        w('layout(set = 0, binding = 1, std430) readonly buffer WeightBuf32 { uint ww[]; } wbuf32;')
        w('layout(set = 0, binding = 2, std430) buffer ArenaBuf { uint16_t ah[]; } abuf;')
        w('#ifdef D4R_WORD_ARENA')
        w('layout(set = 0, binding = 2, std430) buffer ArenaBuf32 { uint aw[]; } abuf32;')
        w('#endif')
        n = 3
        for off in sorted(self.textures):
            w(f'layout(set = 0, binding = {n}) uniform sampler2D T{off};')
            n += 1
        for off in sorted(self.surfaces):
            w(f'layout(set = 0, binding = {n}, {self.surfaces[off]}) uniform writeonly image2D S{off};')
            q = {'rgba8_snorm': 'q_snorm8', 'rgba8': 'q_unorm8', 'rgba32f': 'q_unorm1010102', 'rgb10_a2': 'q_unorm1010102_hw'}.get(self.surfaces[off], '')
            w(f'#define SUSTQ_{off}(v) {q}(v)')
            n += 1
        for off in (() if self.unified_arena else self.buffers):
            w(f'layout(set = 0, binding = {n}, std430) buffer XBuf{off} {{ uint16_t h[]; }} xb{off};')
            w(f'uint GX{off}_16(uint a) {{ return uint(xb{off}.h[a >> 1]); }}')
            w(f'uint GX{off}_32(uint a) {{ return uint(xb{off}.h[a >> 1]) | (uint(xb{off}.h[(a >> 1) + 1u]) << 16); }}')
            w(f'void SX{off}_16(uint a, uint v) {{ xb{off}.h[a >> 1] = uint16_t(v & 0xffffu); }}')
            w(f'void SX{off}_32(uint a, uint v) {{ xb{off}.h[a >> 1] = uint16_t(v & 0xffffu); xb{off}.h[(a >> 1) + 1u] = uint16_t(v >> 16); }}')
            n += 1
        w(f'shared uint sh[{max(1, (self.shared_bytes + 3) // 4)}];')
        if self.probe_slots or self.oob:
            w(f'layout(set = 0, binding = {n}, std430) buffer Debug {{ uint d[]; }} dbg;')
        if self.oob:
            w('#define OOB_REPORT 1')
        w(f'shared uint mms[{384 * (self.bx * self.by * self.bz // 32)}];   // mma_e4m3_f16 scratch, per subgroup')

    def runtime(self):
        runtime = RR_RUNTIME
        if getattr(self, 'unguarded', False):   # --unguarded: no opaque zero
            assert '#define SUST_Z (gl_NumWorkGroups.z - 1u)' in RR_RUNTIME
            runtime = RR_RUNTIME.replace('#define SUST_Z (gl_NumWorkGroups.z - 1u)', '#define SUST_Z 0u')
        return (runtime + (ROOT / 'engine' / 'round_half.glsl').read_text() +
                (ROOT / 'engine' / 'cnn_sr' / 'mma_f16.glsl').read_text())


def main():
    ap = argparse.ArgumentParser(description='PTX kernel -> GLSL compute shader (Ray Reconstruction stage contract)')
    ap.add_argument('module')
    ap.add_argument('kernel')
    ap.add_argument('out')
    ap.add_argument('--block', help='workgroup shape X,Y,Z (default: from the kernel table)')
    ap.add_argument('--surface', action='append', default=[], metavar='OFFSET=FORMAT',
                    help="the GLSL image format of a surface whose stores are formatted (sust.p), when it is not the "
                         "store's own guess, e.g. 144=rgba8_snorm (repeatable)")
    ap.add_argument('--snorm16', action='append', type=int, default=[], metavar='OFFSET',
                    help='texture argument whose filtered fetches are rounded to 16-bit steps, as an NVIDIA card returns them')
    ap.add_argument('--weights-arg', type=int, default=40,
                    help='argument offset of the weight pointer (40 for the DLSS-D kernels, 24 for rr2_*)')
    ap.add_argument('--buffer', action='append', type=int, default=[], metavar='OFFSET',
                    help='a pointer argument that is neither the weights nor the arena: its own storage buffer, bound '
                         'after the surfaces in ascending argument offset (repeatable)')
    ap.add_argument('--unified-arena', action='store_true',
                    help='place all --buffer pointer arguments in binding 2; retain their write metadata')
    ap.add_argument('--probe', action='append', default=[], metavar='LINE:%REG,...',
                    help='debugging: store the registers after the statement at that module line executes (last '
                         'execution) for workgroup --probe-block, into a debug buffer bound after the surfaces')
    ap.add_argument('--probe-block', default='0,0')
    ap.add_argument('--oob-report', action='store_true', help='debugging: count out-of-range buffer accesses')
    ap.add_argument('--exact-convert', action='store_true',
                    help='the module goes through spv_exact_convert.py: lighter guards on binary16 conversions (CVT_GUARD)')
    ap.add_argument('--unguarded', action='store_true',
                    help="none of the exactness guards: no 'precise' registers and no opaque zero OR-ed into binary16 "
                         "stores and conversions, so the driver may contract, fold and fuse (compile_rr2.py D4R_RR2_UNGUARDED)")
    ns = ap.parse_args()
    block = tuple(int(x) for x in ns.block.split(',')) if ns.block else default_block(ns.kernel)
    formats = {int(k): v for k, v in (x.split('=') for x in ns.surface)}
    src = open(ns.module).read()
    probes = {}
    for spec in ns.probe:
        line, regs = spec.split(':', 1)
        probes[int(line)] = [r.strip() for r in regs.split(',')]

    def make():
        t = RRT(src, ns.kernel, block, ns.weights_arg, ns.buffer)
        t.unified_arena = ns.unified_arena
        t.snorm16 = set(ns.snorm16)
        t.probes = probes
        t.probe_block = tuple(int(x) for x in ns.probe_block.split(','))
        t.oob = ns.oob_report
        t.unguarded = ns.unguarded
        return t
    t = base.build(make, False, ns.unguarded, False)
    for off, fmt in formats.items():
        if off in t.surfaces:
            t.surfaces[off] = fmt
    open(ns.out, 'w').write(t.emit({}))
    if t.globals_blob:
        Path(ns.out).with_suffix('.globals.bin').write_bytes(t.globals_blob)
    # the binding order (stage contract): textures, then surfaces, in ascending argument offset
    Path(ns.out).with_suffix('.json').write_text(json.dumps(
        {'kernel': ns.kernel, 'block': block, 'textures': sorted(t.textures),
         'surfaces': [[off, t.surfaces[off]] for off in sorted(t.surfaces)],
         'buffers': [] if t.unified_arena else t.buffers, 'shared': t.shared_bytes,
         'written_buffers': sorted(t.written_buffers),
         'probes': [[line, reg] for line in sorted(probes) for reg in probes[line]], 'oob': ns.oob_report}) + '\n')
    sys.stderr.write(f'block={block} textures={sorted(t.textures)} surfaces={t.surfaces} shared={t.shared_bytes}\n')


if __name__ == '__main__':
    main()
