// Shared pieces of the CNN Ray Reconstruction stages (engine/rr/cnn_enc.comp, cnn_dec.comp): nvngx_dlssd.dll 310.3's
// presets A-C, bit-exact to NVIDIA's kernels run through ZLUDA.
// With -DD4R_EMU_CM the cooperative-matrix ops become scalar code for GPUs without
// VK_KHR_cooperative_matrix (RADV on RDNA2 and older): same algorithm, finite results, no longer bit-exact.
//
// Binding 0 is NVIDIA's argument block byte for byte, its pointers device addresses (buffer_reference).
//
// Arithmetic: every 16-deep step of a convolution is one f16 x f16 -> f32 WMMA whose accumulator enters and leaves
// as binary16, which is how ZLUDA lowers NVIDIA's f16-accumulating mma.m16n8k16. The WMMA's sum depends on the order
// of its products, so K slot s of a 16-block holds channel P16(s) as in ZLUDA's lowering (engine/rr/cnn_pack.py
// packs the weights that way, rows and columns); a conv's output column s of a 16-block is then channel P16(s) too.
// Activations between stages are stored in 8 bits: the binary16 value's bits 14..7 (unsigned, 5 exponent and 3
// mantissa bits), rounded half up through fma(h & 0xfc00, 2^-4, h). Output channel n of a convolution
// (n = 8 j + 2 t + e for NVIDIA's n-tile j and lane t) is stored at slot 2 (T t + j) + e, T = n-tiles per store;
// biases are read the same way with T = 4.
#ifndef D4R_EMU_CM
#extension GL_KHR_cooperative_matrix : require
#else
#extension GL_KHR_shader_subgroup_shuffle : require
#endif
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_control_flow_attributes : require

layout(std430, binding = 0) readonly buffer Params { uint p[]; };

layout(buffer_reference, std430, buffer_reference_align = 2) buffer H16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer U8 { uint8_t v[]; };

#ifdef D4R_EMU_CM
// Scalar replacement of the 16x16 cooperative-matrix ops: a block is the eight elements its lanes hold;
// internally to the emulation, lane l slot i is the block's row-major element (row, col) = ((l * 8 + i) / 16,
// (l * 8 + i) % 16). The loads and stores below map elements to their logical LDS / global matrix positions,
// so the surrounding tile code cannot tell the two flavors apart.
struct M16H { float16_t e[8]; };   // one lane's binary16: A, B and the entering accumulator share this
struct M16F { float e[8]; };       // the same eight in f32 (the WMMA's accumulator)
#define ACC M16F
#define HAC M16H
#define AH M16H
#define BH M16H
// RowMajor load of a block at `idx` with row stride `stride`: element (row, col) lives at idx + row * stride + col
#define LOAD_A(m, buf, idx, stride) { const uint _l = gl_LocalInvocationIndex & 31u; \
    for (uint _i = 0u; _i < 8u; ++_i) { const uint _e = _l * 8u + _i; m.e[_i] = buf[(idx) + (_e >> 4u) * (stride) + (_e & 15u)]; } }
#define STORE(m, buf, idx, stride) { const uint _l = gl_LocalInvocationIndex & 31u; \
    for (uint _i = 0u; _i < 8u; ++_i) { const uint _e = _l * 8u + _i; buf[(idx) + (_e >> 4u) * (stride) + (_e & 15u)] = m.e[_i]; } }
#else
#define ACC coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define HAC coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define AH coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define BH coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#endif

uint64_t ptr(uint off) { return packUint2x32(uvec2(p[off / 4], p[off / 4 + 1])); }
int pi(uint off) { return int(p[off / 4]); }

// RADV folds a register-only float(float16_t(x)) back to x: binary16 roundings go through a conversion with an
// opaque zero OR-ed in (set in main from something the compiler cannot evaluate)
uint16_t RR_ZERO16;
float16_t h16(float x) { return uint16BitsToFloat16(float16BitsToUint16(float16_t(x)) | RR_ZERO16); }
float16_t hb(uint b) { return uint16BitsToFloat16(uint16_t(b)); }

// K slot / output column s of a 16-block holds channel P16(s) (cnn_pack.py)
uint P16(uint s) { return (s & 1u) | (((s >> 1u) & 1u) << 3u) | ((s >> 2u) << 1u); }
uint perm(uint c) { return (c & ~15u) | P16(c & 15u); }
// the stored slot of logical channel n with T n-tiles per store
uint slot(uint n, uint T) { uint j = n >> 3u, t = (n & 7u) >> 1u; return (j / T) * 8u * T + 2u * (T * t + j % T) + (n & 1u); }

float16_t dec8(uint b) { return uint16BitsToFloat16(uint16_t(b << 7u)); }
uint enc8(float16_t h)
{
    const float16_t e = uint16BitsToFloat16(float16BitsToUint16(h) & uint16_t(0xfc00u));
    const float16_t r = fma(e, float16_t(0.0625), h);
    return (uint(float16BitsToUint16(r)) >> 7u) & 0xffu;
}

// the accumulator back to binary16, nearest even element by element (a whole-matrix conversion rounds toward zero)
#ifdef D4R_EMU_CM
HAC tohalf(ACC f) { HAC h; for (int i = 0; i < 8; ++i) h.e[i] = h16(f.e[i]); return h; }
HAC addh(HAC a, HAC b) { HAC h; for (int i = 0; i < 8; ++i) h.e[i] = a.e[i] + b.e[i]; return h; }
HAC hac0() { HAC h; for (int i = 0; i < 8; ++i) h.e[i] = float16_t(0.0); return h; }

// One 16-deep conv step in binary16 pair code: lane l computes the block's row l / 2, columns (l & 1) * 8 .. +7
// of C = A x B + acc. Every v_pk_fma_f16 covers the column pair (2j, 2j + 1), so one FMA handles two outputs and
// the accumulator is four f16vec2. Lane l's A slots hold the k of its own half (splat into the pair operand) and
// the partner lane's half arrives as packed pairs over a subgroupShuffleXor. The B tile goes through LDS in
// column-pair order (per wave 16 rows of 8 binary16 pairs): each lane stages the pairs of its B row (the k) by
// loading the [output][K] weights column-wise and stores them as f16vec2, and every read is an f16vec2 LDS load
// feeding an FMA directly - measured 1.9x the products per cycle of the f32 scalar loop (gfx1035 microbench: the
// f32 route spends one VALU slot per product plus one per unpackHalf2x16). Accumulating 16 deep in binary16
// costs ~0.5% precision; bit-exactness with the WMMA flavor is off for this flavor by design.
shared f16vec2 emuH2[4u * 128u];
HAC mma(HAC acc, AH a, H16 w, uint base)
{
    const uint l = gl_LocalInvocationIndex & 31u, h = l & 1u, eb = (gl_LocalInvocationIndex >> 5u) * 128u + (l >> 1u) * 8u + h * 4u;
    [[unroll]] for (uint j = 0u; j < 4u; ++j)
        emuH2[eb + j] = f16vec2(w.v[base + (h * 8u + 2u * j) * 16u + (l >> 1u)], w.v[base + (h * 8u + 2u * j + 1u) * 16u + (l >> 1u)]);
    barrier();
    // the partner lane's half of A's row: its eight binary16 in four words, four shuffles
    const uint q0 = packHalf2x16(vec2(float(a.e[0]), float(a.e[1]))), q1 = packHalf2x16(vec2(float(a.e[2]), float(a.e[3]))),
                 q2 = packHalf2x16(vec2(float(a.e[4]), float(a.e[5]))), q3 = packHalf2x16(vec2(float(a.e[6]), float(a.e[7])));
    const vec2 p01 = unpackHalf2x16(subgroupShuffleXor(q0, 1u)), p23 = unpackHalf2x16(subgroupShuffleXor(q1, 1u)),
                 p45 = unpackHalf2x16(subgroupShuffleXor(q2, 1u)), p67 = unpackHalf2x16(subgroupShuffleXor(q3, 1u));
    f16vec2 c0 = f16vec2(acc.e[0], acc.e[1]), c1 = f16vec2(acc.e[2], acc.e[3]),
               c2 = f16vec2(acc.e[4], acc.e[5]), c3 = f16vec2(acc.e[6], acc.e[7]);
    // every k of the block's own half first, then the partner's: the four column pairs of row k from LDS
    #define EMU_H(K, AV) { const f16vec2 _a = (AV); const uint _w = eb + ((K) - (l >> 1u)) * 8u; \
        c0 = fma(_a, emuH2[_w], c0); c1 = fma(_a, emuH2[_w + 1u], c1); c2 = fma(_a, emuH2[_w + 2u], c2); c3 = fma(_a, emuH2[_w + 3u], c3); }
    EMU_H(h * 8u + 0u, f16vec2(a.e[0], a.e[0])) EMU_H(h * 8u + 1u, f16vec2(a.e[1], a.e[1]))
    EMU_H(h * 8u + 2u, f16vec2(a.e[2], a.e[2])) EMU_H(h * 8u + 3u, f16vec2(a.e[3], a.e[3]))
    EMU_H(h * 8u + 4u, f16vec2(a.e[4], a.e[4])) EMU_H(h * 8u + 5u, f16vec2(a.e[5], a.e[5]))
    EMU_H(h * 8u + 6u, f16vec2(a.e[6], a.e[6])) EMU_H(h * 8u + 7u, f16vec2(a.e[7], a.e[7]))
    EMU_H(8u - h * 8u + 0u, f16vec2(float16_t(p01.x), float16_t(p01.x))) EMU_H(8u - h * 8u + 1u, f16vec2(float16_t(p01.y), float16_t(p01.y)))
    EMU_H(8u - h * 8u + 2u, f16vec2(float16_t(p23.x), float16_t(p23.x))) EMU_H(8u - h * 8u + 3u, f16vec2(float16_t(p23.y), float16_t(p23.y)))
    EMU_H(8u - h * 8u + 4u, f16vec2(float16_t(p45.x), float16_t(p45.x))) EMU_H(8u - h * 8u + 5u, f16vec2(float16_t(p45.y), float16_t(p45.y)))
    EMU_H(8u - h * 8u + 6u, f16vec2(float16_t(p67.x), float16_t(p67.x))) EMU_H(8u - h * 8u + 7u, f16vec2(float16_t(p67.y), float16_t(p67.y)))
    #undef EMU_H
    HAC r;
    r.e[0] = c0.x; r.e[1] = c0.y; r.e[2] = c1.x; r.e[3] = c1.y;
    r.e[4] = c2.x; r.e[5] = c2.y; r.e[6] = c3.x; r.e[7] = c3.y;
    return r;
}

// The same step with m = 32: the wave's two tile rows share the B tile, so one staged binary16 pair operand
// feeds two v_pk_fma_f16. Staging, the barrier and the partner shuffles run once per pair of m-tiles, which
// halves the LDS bytes per output (the LDS stream is the binding constraint on RDNA2: each 16x16 tile costs
// 8 KB of LDS reads for only 4096 products at m = 16) and quarters the barriers at m = 32 depth.
void mma2(inout HAC accA, inout HAC accB, AH aA, AH aB, H16 w, uint base)
{
    const uint l = gl_LocalInvocationIndex & 31u, h = l & 1u, eb = (gl_LocalInvocationIndex >> 5u) * 128u + (l >> 1u) * 8u + h * 4u;
    [[unroll]] for (uint j = 0u; j < 4u; ++j)
        emuH2[eb + j] = f16vec2(w.v[base + (h * 8u + 2u * j) * 16u + (l >> 1u)], w.v[base + (h * 8u + 2u * j + 1u) * 16u + (l >> 1u)]);
    barrier();
    // both rows' partner halves: eight packed pairs, eight shuffles
    const uint q0 = packHalf2x16(vec2(float(aA.e[0]), float(aA.e[1]))), q1 = packHalf2x16(vec2(float(aA.e[2]), float(aA.e[3]))),
                 q2 = packHalf2x16(vec2(float(aA.e[4]), float(aA.e[5]))), q3 = packHalf2x16(vec2(float(aA.e[6]), float(aA.e[7])));
    const uint u0 = packHalf2x16(vec2(float(aB.e[0]), float(aB.e[1]))), u1 = packHalf2x16(vec2(float(aB.e[2]), float(aB.e[3]))),
                 u2 = packHalf2x16(vec2(float(aB.e[4]), float(aB.e[5]))), u3 = packHalf2x16(vec2(float(aB.e[6]), float(aB.e[7])));
    const vec2 p01 = unpackHalf2x16(subgroupShuffleXor(q0, 1u)), p23 = unpackHalf2x16(subgroupShuffleXor(q1, 1u)),
                 p45 = unpackHalf2x16(subgroupShuffleXor(q2, 1u)), p67 = unpackHalf2x16(subgroupShuffleXor(q3, 1u));
    const vec2 s01 = unpackHalf2x16(subgroupShuffleXor(u0, 1u)), s23 = unpackHalf2x16(subgroupShuffleXor(u1, 1u)),
                 s45 = unpackHalf2x16(subgroupShuffleXor(u2, 1u)), s67 = unpackHalf2x16(subgroupShuffleXor(u3, 1u));
    f16vec2 cA0 = f16vec2(accA.e[0], accA.e[1]), cA1 = f16vec2(accA.e[2], accA.e[3]),
               cA2 = f16vec2(accA.e[4], accA.e[5]), cA3 = f16vec2(accA.e[6], accA.e[7]);
    f16vec2 cB0 = f16vec2(accB.e[0], accB.e[1]), cB1 = f16vec2(accB.e[2], accB.e[3]),
               cB2 = f16vec2(accB.e[4], accB.e[5]), cB3 = f16vec2(accB.e[6], accB.e[7]);
    // every k of the block's own half first, then the partner's: row k's four column pairs from LDS, each used twice
    #define EMU_H2(K, AVA, AVB) { const uint _w = eb + ((K) - (l >> 1u)) * 8u; \
        const f16vec2 _b0 = emuH2[_w], _b1 = emuH2[_w + 1u], _b2 = emuH2[_w + 2u], _b3 = emuH2[_w + 3u]; \
        cA0 = fma(AVA, _b0, cA0); cB0 = fma(AVB, _b0, cB0); cA1 = fma(AVA, _b1, cA1); cB1 = fma(AVB, _b1, cB1); \
        cA2 = fma(AVA, _b2, cA2); cB2 = fma(AVB, _b2, cB2); cA3 = fma(AVA, _b3, cA3); cB3 = fma(AVB, _b3, cB3); }
    EMU_H2(h * 8u + 0u, f16vec2(aA.e[0], aA.e[0]), f16vec2(aB.e[0], aB.e[0]))
    EMU_H2(h * 8u + 1u, f16vec2(aA.e[1], aA.e[1]), f16vec2(aB.e[1], aB.e[1]))
    EMU_H2(h * 8u + 2u, f16vec2(aA.e[2], aA.e[2]), f16vec2(aB.e[2], aB.e[2]))
    EMU_H2(h * 8u + 3u, f16vec2(aA.e[3], aA.e[3]), f16vec2(aB.e[3], aB.e[3]))
    EMU_H2(h * 8u + 4u, f16vec2(aA.e[4], aA.e[4]), f16vec2(aB.e[4], aB.e[4]))
    EMU_H2(h * 8u + 5u, f16vec2(aA.e[5], aA.e[5]), f16vec2(aB.e[5], aB.e[5]))
    EMU_H2(h * 8u + 6u, f16vec2(aA.e[6], aA.e[6]), f16vec2(aB.e[6], aB.e[6]))
    EMU_H2(h * 8u + 7u, f16vec2(aA.e[7], aA.e[7]), f16vec2(aB.e[7], aB.e[7]))
    EMU_H2(8u - h * 8u + 0u, f16vec2(float16_t(p01.x), float16_t(p01.x)), f16vec2(float16_t(s01.x), float16_t(s01.x)))
    EMU_H2(8u - h * 8u + 1u, f16vec2(float16_t(p01.y), float16_t(p01.y)), f16vec2(float16_t(s01.y), float16_t(s01.y)))
    EMU_H2(8u - h * 8u + 2u, f16vec2(float16_t(p23.x), float16_t(p23.x)), f16vec2(float16_t(s23.x), float16_t(s23.x)))
    EMU_H2(8u - h * 8u + 3u, f16vec2(float16_t(p23.y), float16_t(p23.y)), f16vec2(float16_t(s23.y), float16_t(s23.y)))
    EMU_H2(8u - h * 8u + 4u, f16vec2(float16_t(p45.x), float16_t(p45.x)), f16vec2(float16_t(s45.x), float16_t(s45.x)))
    EMU_H2(8u - h * 8u + 5u, f16vec2(float16_t(p45.y), float16_t(p45.y)), f16vec2(float16_t(s45.y), float16_t(s45.y)))
    EMU_H2(8u - h * 8u + 6u, f16vec2(float16_t(p67.x), float16_t(p67.x)), f16vec2(float16_t(s67.x), float16_t(s67.x)))
    EMU_H2(8u - h * 8u + 7u, f16vec2(float16_t(p67.y), float16_t(p67.y)), f16vec2(float16_t(s67.y), float16_t(s67.y)))
    #undef EMU_H2
    accA.e[0] = cA0.x; accA.e[1] = cA0.y; accA.e[2] = cA1.x; accA.e[3] = cA1.y;
    accA.e[4] = cA2.x; accA.e[5] = cA2.y; accA.e[6] = cA3.x; accA.e[7] = cA3.y;
    accB.e[0] = cB0.x; accB.e[1] = cB0.y; accB.e[2] = cB1.x; accB.e[3] = cB1.y;
    accB.e[4] = cB2.x; accB.e[5] = cB2.y; accB.e[6] = cB3.x; accB.e[7] = cB3.y;
}
#else
HAC tohalf(ACC f) { HAC h; for (int i = 0; i < f.length(); ++i) h[i] = h16(f[i]); return h; }
HAC addh(HAC a, HAC b) { HAC h; for (int i = 0; i < a.length(); ++i) h[i] = a[i] + b[i]; return h; }
#endif

// NVIDIA's hexp (cuda_fp16): ex2.approx of x log2(e) in f32, rounded to binary16, with its four corrections
float16_t hexp(float16_t h)
{
    precise float f = fma(float(h), uintBitsToFloat(0x3fb8aa3bu), -0.0);
    float16_t r = h16(exp2(f));
    const uint b = uint(float16BitsToUint16(h));
    if (b == 0x1f79u || b == 0x25cfu) r = fma(float16_t(1.0), hb(0x9400u), r);
    if (b == 0xc13bu) r = fma(float16_t(1.0), hb(0x0400u), r);
    if (b == 0xc1efu) r = fma(float16_t(1.0), hb(0x0200u), r);
    return r;
}
// __hdiv: rcp.approx quotient, refined when it is a tiny nonzero
float16_t hdiv(float16_t a, float16_t b)
{
    const float fa = float(a), fb = float(b);
    const float rc = 1.0 / fb;
    precise float q = fa * rc;
    float16_t r = h16(q);
    const float16_t m = abs(r);
    if (m < hb(0x008fu) && float16_t(0.0) < m) {
        precise float e = fma(-fb, q, fa);
        precise float q2 = fma(rc, e, q);
        r = h16(q2);
    }
    return r;
}
float16_t sigm(float16_t x) { return hdiv(float16_t(1.0), float16_t(1.0) + hexp(-x)); }

// the softmax of the first nine logits, NVIDIA's way: binary16 max, hexp, sum in order, hdiv per tap
void softmax9(float16_t lg[12], out float16_t w[9])
{
    float16_t m = lg[0];
    [[unroll]] for (uint i = 1u; i < 9u; ++i) m = h16(max(float(m), float(lg[i])));
    float16_t e[9];
    float16_t s = float16_t(0.0);
    [[unroll]] for (uint i = 0u; i < 9u; ++i) { e[i] = hexp(lg[i] - m); s = s + e[i]; }
    [[unroll]] for (uint i = 0u; i < 9u; ++i) w[i] = hdiv(e[i], s);
}
// the 3x3 neighbourhood, mirrored at the borders as NVIDIA's kernels do (rows 0 / H - 1 take 1 / H - 2)
void taps3(int v, int n, out int r[3]) { r[0] = v < 1 ? 1 : v - 1; r[1] = v; r[2] = v + 2 > n ? n - 2 : v + 1; }
