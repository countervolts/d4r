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


RR_RUNTIME = r'''
float16_t H(uint u) { return uint16BitsToFloat16(uint16_t(u)); }
uint HU(float16_t h) { return uint(float16BitsToUint16(h)); }
f16vec2 H2(uint u) { return uint16BitsToFloat16(u16vec2(uint16_t(u & 0xffffu), uint16_t(u >> 16))); }
uint HU2_(u16vec2 u) { return uint(u.x) | (uint(u.y) << 16); }
uint H2U(f16vec2 h) { return HU2_(float16BitsToUint16(h)); }
int s16(uint u) { return (int(u) << 16) >> 16; }
float rcp_a(float x) { return 1.0 / x; }
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
uint GW32(uint a) { CHECKW(a + 2u); return uint(wbuf.wh[a >> 1]) | (uint(wbuf.wh[(a >> 1) + 1u]) << 16); }
uint GA16(uint a) { CHECKR(a); return uint(abuf.ah[a >> 1]); }
uint GA32(uint a) { CHECKR(a + 2u); return uint(abuf.ah[a >> 1]) | (uint(abuf.ah[(a >> 1) + 1u]) << 16); }
void SA16(uint a, uint v) { CHECKS(a); abuf.ah[a >> 1] = uint16_t(v & 0xffffu); }
void SA32(uint a, uint v) { CHECKS(a + 2u); abuf.ah[a >> 1] = uint16_t(v & 0xffffu); abuf.ah[(a >> 1) + 1u] = uint16_t(v >> 16); }

// Surface stores with .zero out-of-bounds semantics.
void sust_p4(writeonly image2D s, int x, int y, vec4 v)
{
    ivec2 n = imageSize(s);
    if (uint(x) < uint(n.x) && uint(y) < uint(n.y)) imageStore(s, ivec2(x, y), v);
}
void sust_p1(writeonly image2D s, int x, int y, float v) { sust_p4(s, x, y, vec4(v, 0.0, 0.0, 0.0)); }
// Binary stores write the halves' bits. RADV folds float(float16_t(x)) back to x when the half came from a
// cvt.rn.f16.f32 in a register, and the image store then truncates x instead (the output stage's history surface
// was one binary16 ulp low in half its values, which the next frame's encoder 0 amplifies): OR-ing in a zero the
// compiler cannot see (the dispatch's z count is 1) keeps the conversion's rounding.
#define SUST_Z (gl_NumWorkGroups.z - 1u)
void sust_b2(writeonly image2D s, int x, int y, uint a, uint b)
{
    sust_p4(s, x >> 2, y, vec4(float(H(a | SUST_Z)), float(H(b | SUST_Z)), 0.0, 0.0));
}
void sust_b4(writeonly image2D s, int x, int y, uint a, uint b, uint c, uint d)
{
    sust_p4(s, x >> 3, y, vec4(float(H(a | SUST_Z)), float(H(b | SUST_Z)), float(H(c | SUST_Z)), float(H(d | SUST_Z))));
}

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
    return subgroupShuffle(v, uint(idx) & 31u);
}

// shfl.sync.idx (PTX ISA): j = minLane | (b & ~segmask), in bounds while j <= maxLane, otherwise the lane's own value
uint shfl_idx(uint v, uint b, uint opts, out bool in_bounds)
{
    uint mask = (opts >> 8) & 31u, lane = gl_SubgroupInvocationID & 31u;
    uint maxLane = (lane & mask) | (opts & 31u & ~mask), j = (lane & mask) | (b & 31u & ~mask);
    in_bounds = j <= maxLane;
    return subgroupShuffle(v, in_bounds ? j : lane);
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

// PTX min/max: a NaN operand yields the other one (GLSL leaves NaN undefined, so the result depended on codegen)
float16_t pmin(float16_t a, float16_t b) { return isnan(a) ? b : isnan(b) ? a : min(a, b); }
float16_t pmax(float16_t a, float16_t b) { return isnan(a) ? b : isnan(b) ? a : max(a, b); }
f16vec2 pmin2(f16vec2 a, f16vec2 b) { return f16vec2(pmin(a.x, b.x), pmin(a.y, b.y)); }
f16vec2 pmax2(f16vec2 a, f16vec2 b) { return f16vec2(pmax(a.x, b.x), pmax(a.y, b.y)); }
float pminf(float a, float b) { return isnan(a) ? b : isnan(b) ? a : min(a, b); }
float pmaxf(float a, float b) { return isnan(a) ? b : isnan(b) ? a : max(a, b); }
// e4m3 <-> f16 (PTX cvt semantics; RNE, satfinite; NaN -> 0x7f e4m3 / 0x7e00 f16).
uint e4m3_to_h(uint b)
{
    uint e = (b >> 3) & 15u, m = b & 7u, s = (b & 0x80u) << 8;
    if (e == 15u && m == 7u) return s | 0x7e00u;
    float f = e == 0u ? ldexp(float(m), -9) : ldexp(float(8u + m), int(max(e, 1u)) - 10);
    return s | HU(float16_t(f));
}
#ifdef SOFTWARE_E4M3
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
#ifdef SOFTWARE_E4M3
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
#define MMA coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MMB coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define MMF coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define MMH coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
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
    coopMatLoad(x0, mms, A, 8u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(x1, mms, A + 4u, 8u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(y0, mms, B, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(y1, mms, B + 4u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(c, mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
    MMF acc = coopMatMulAdd(x1, y1, coopMatMulAdd(x0, y0, MMF(c)));
    memoryBarrierShared(); subgroupBarrier();
    coopMatStore(MMH(acc), mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
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
    coopMatLoad(x0, mms, A, 8u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(x1, mms, A + 4u, 8u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(y0, mms, B, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(y1, mms, B + 4u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(c, mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
    MMF acc = coopMatMulAdd(x1, y1, coopMatMulAdd(x0, y0, MMF(c)));
    memoryBarrierShared(); subgroupBarrier();
    coopMatStore(MMH(acc), mms, C, 8u, gl_CooperativeMatrixLayoutRowMajor);
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


RR_RUNTIME += base_helper('prmt') + base_helper('shfl_bfly').replace(
    'return subgroupShuffle(v, (gl_SubgroupInvocationID & 32u) | uint(idx));', 'return subgroupShuffle(v, uint(idx) & 31u);')


class RRT(base.T):
    def __init__(self, ptx, kernel, block):
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
        self.stmt_lines = statement_lines(ptx, kernel)
        self.stmt_index = 0
        super().__init__(ptx, kernel)
        self.pair_mma()
        self.origins()

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
                if op == 'ld.param.u64':
                    # the argument's true offset (the load may go through a register based inside the block)
                    new = {'w' if self.param_off(a[1]) == 40 else 'a'}
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
        if len(s) != 1:
            raise base.Err(f'global pointer {reg} has origin {sorted(s)} (need exactly one of w/a)')
        return next(iter(s))

    def gaddr(self, tok):
        m = re.fullmatch(r'\[\s*(%rd\d+)\s*(?:\+\s*(\d+))?\s*\]', tok.strip())
        if not m:
            raise base.Err(f'unsupported global address {tok}')
        return m.group(1), int(m.group(2) or 0)

    def gbyte(self, reg, off):
        return f'uint({self.val(reg, "u64")}) + {off}u'

    def ld(self, o, a):
        if 'global' not in o:
            return super().ld(o, a)
        size, count = mem_shape(o)
        reg, off = self.gaddr(a[1])
        fn = 'GW' if self.gorigin(reg) == 'w' else 'GA'
        dsts = braces(a[0]) if count > 1 else [a[0]]
        for i, d in enumerate(dsts):
            self.A(d, f'{fn}{size * 8}({self.gbyte(reg, off + i * size)})', 'u16' if size == 2 else 'u32')

    def st(self, o, a):
        if 'global' not in o:
            return super().st(o, a)
        size, count = mem_shape(o)
        reg, off = self.gaddr(a[0])
        if self.gorigin(reg) != 'a':
            raise base.Err('store through the read-only weight pointer')
        vals = braces(a[1]) if count > 1 else [a[1]]
        for i, v in enumerate(vals):
            if size == 2:
                self.E(f'SA16({self.gbyte(reg, off + i * size)}, {self.val(v, "u16")});')
            else:
                self.E(f'SA32({self.gbyte(reg, off + i * size)}, {self.val(v, "u32")});')

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
            # integer coordinates still go through the texture's address mode in CUDA (clamp for every denoiser
            # texture); texelFetch does not clamp, and what it returns out of range depends on the device's robustness
            fetch = (f'texelFetch(T{off}, clamp(ivec2({self.val(c[0], "s32")}, {self.val(c[1], "s32")}), ivec2(0), '
                     f'textureSize(T{off}, 0) - 1), 0)')
        elif 'base' in o and o[-1] == 'f32':
            fetch = f'textureLod(T{off}, vec2({self.val(c[0], "f32")}, {self.val(c[1], "f32")}), 0.0)'
        elif 'level' in o and o[-1] == 'f32':
            fetch = (f'textureLod(T{off}, vec2({self.val(c[0], "f32")}, {self.val(c[1], "f32")}), '
                     f'{self.val(a[2], "f32")})')
        else:
            raise base.Err(f'unsupported texture op {op}')
        f16 = 'f16' in o
        self.E(f'{{ vec4 t_ = {fetch};')
        for i, d in enumerate(dst):
            self.A(d, f'float16_t(t_[{i}])' if f16 else f't_[{i}]', 'f16' if f16 else 'float')
        self.E('}')

    def sust(self, op, a):
        h, c = self.texref(a[0])
        off = self.handle_off(h)
        v = braces(a[1])
        x, y = self.val(c[0], 's32'), self.val(c[1], 's32')
        S = f'S{off}'
        if op == 'sust.p.2d.v4.b32.zero':
            self.set_surface(off, 'rgba16f')
            self.E(f'sust_p4({S}, {x}, {y}, vec4({", ".join(self.val(q, "f32") for q in v)}));')
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
        return super().op(op, a)

    # ------------------------------------------------ stage contract
    def preamble(self, w, tex_filter):
        w('#version 460')
        w('#extension GL_EXT_shader_explicit_arithmetic_types : require')
        w('#extension GL_EXT_shader_16bit_storage : require')
        w('#extension GL_KHR_shader_subgroup_basic : require')
        w('#extension GL_KHR_shader_subgroup_shuffle : require')
        w('#extension GL_KHR_cooperative_matrix : require')
        w('#extension GL_KHR_memory_scope_semantics : require')
        w('#extension GL_EXT_float_e4m3 : require')
        w(f'// Generated by engine/rr/ptx2rr.py from {self.kernel}. Model data derived from the user\'s DLL; do not commit.')
        w(f'layout(local_size_x = {self.bx * self.by * self.bz}, local_size_y = 1, local_size_z = 1) in;')
        w('layout(set = 0, binding = 0, std430) readonly buffer ParamBlock { uint w[]; } param;')
        w('layout(set = 0, binding = 1, std430) readonly buffer WeightBuf { uint16_t wh[]; } wbuf;')
        w('layout(set = 0, binding = 2, std430) buffer ArenaBuf { uint16_t ah[]; } abuf;')
        n = 3
        for off in sorted(self.textures):
            w(f'layout(set = 0, binding = {n}) uniform sampler2D T{off};')
            n += 1
        for off in sorted(self.surfaces):
            w(f'layout(set = 0, binding = {n}, {self.surfaces[off]}) uniform writeonly image2D S{off};')
            n += 1
        w(f'shared uint sh[{max(1, (self.shared_bytes + 3) // 4)}];')
        if self.probe_slots or self.oob:
            w(f'layout(set = 0, binding = {n}, std430) buffer Debug {{ uint d[]; }} dbg;')
        if self.oob:
            w('#define OOB_REPORT 1')
        w(f'shared uint mms[{384 * (self.bx * self.by * self.bz // 32)}];   // mma_e4m3_f16 scratch, per subgroup')

    def runtime(self):
        return RR_RUNTIME


def main():
    ap = argparse.ArgumentParser(description='PTX kernel -> GLSL compute shader (Ray Reconstruction stage contract)')
    ap.add_argument('module')
    ap.add_argument('kernel')
    ap.add_argument('out')
    ap.add_argument('--block', help='workgroup shape X,Y,Z (default: from the kernel table)')
    ap.add_argument('--surface', action='append', default=[], metavar='OFFSET=FORMAT',
                    help="the GLSL image format of a surface whose stores are formatted (sust.p), when it is not the "
                         "store's own guess, e.g. 144=rgba8_snorm (repeatable)")
    ap.add_argument('--probe', action='append', default=[], metavar='LINE:%REG,...',
                    help='debugging: store the registers after the statement at that module line executes (last '
                         'execution) for workgroup --probe-block, into a debug buffer bound after the surfaces')
    ap.add_argument('--probe-block', default='0,0')
    ap.add_argument('--oob-report', action='store_true', help='debugging: count out-of-range buffer accesses')
    ns = ap.parse_args()
    block = tuple(int(x) for x in ns.block.split(',')) if ns.block else default_block(ns.kernel)
    formats = {int(k): v for k, v in (x.split('=') for x in ns.surface)}
    src = open(ns.module).read()
    probes = {}
    for spec in ns.probe:
        line, regs = spec.split(':', 1)
        probes[int(line)] = [r.strip() for r in regs.split(',')]

    def make():
        t = RRT(src, ns.kernel, block)
        t.probes = probes
        t.probe_block = tuple(int(x) for x in ns.probe_block.split(','))
        t.oob = ns.oob_report
        return t
    t = base.build(make, False, False, False)
    for off, fmt in formats.items():
        if off in t.surfaces:
            t.surfaces[off] = fmt
    open(ns.out, 'w').write(t.emit({}))
    # the binding order (stage contract): textures, then surfaces, in ascending argument offset
    Path(ns.out).with_suffix('.json').write_text(json.dumps(
        {'kernel': ns.kernel, 'block': block, 'textures': sorted(t.textures),
         'surfaces': [[off, t.surfaces[off]] for off in sorted(t.surfaces)], 'shared': t.shared_bytes,
         'probes': [[line, reg] for line in sorted(probes) for reg in probes[line]], 'oob': ns.oob_report}) + '\n')
    sys.stderr.write(f'block={block} textures={sorted(t.textures)} surfaces={t.surfaces} shared={t.shared_bytes}\n')


if __name__ == '__main__':
    main()
