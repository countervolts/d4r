// RDNA3 (gfx11, RR_GFX11) forms of the native Ray Reconstruction stages' FP8 operations, included by rr_exact.glsl.
// gfx11 has no FP8 matrix instruction and RADV no VK_EXT_shader_float8, so the reference is ZLUDA's gfx11 lowering of
// mma.m16n8k32.f16.e4m3.e4m3.f16 (D4R_ZLUDA_WMMA_FP8, patches/zluda 0004), which these reproduce bit for bit:
//   - each e4m3 byte is widened on its bits to the binary16 of value * 2^-8 (magnitude at bits 7..14, the sign at 15,
//     the NaN code to a NaN), so the products carry 2^-16;
//   - the accumulator enters scaled by 2^-16, the two K halves (k 0..15, 16..31) run as two chained
//     v_wmma_f32_16x16x16_f16, the sum is scaled back and rounded to binary16 once;
//   - inside each K half the operand positions hold k in ZLUDA's order: position 2q + j holds k 4q + j and position
//     8 + 2q + j holds k 4q + 2 + j (q < 4, j < 2). The instruction's f32 sum depends on that order (tests on an
//     RX 7700 XT: 29% of the outputs change when the same products are permuted), so it is part of the arithmetic.
// FP8 values in registers are their codes (uint); an operand matrix is coopmat<float16_t> holding the widened values.
// RADV's gfx11 layouts: an A or B operand lane l holds row / column l & 15, element i = k i (both half-waves alike);
// an accumulator lane holds column l & 15, element i = row 2i + (l >> 4). The stages' elementwise code is written for
// gfx12's accumulator layout (element i = row i + 8 (l >> 4)), so every A operand is loaded with its rows permuted:
// lane l takes row g11_row(l & 15), and a product's physical row 2i + h is then row i + 8h, i.e. the gfx12 layout.
// That only relabels the outputs (each is its own dot product), so it is exact. Accumulators loaded or built
// elementwise therefore use the gfx12 positions too.
#define A8 coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define B8 coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>

// cvt.rn.satfinite.e4m3 of binary16 bits as ZLUDA computes it on gfx11 (HIP's conversion after clamping infinities
// to 448): nearest even, |x| > 448 -> 0x7e, NaN -> 0x7f, the sign kept. Equal to it for all 65536 inputs.
#ifndef G11_HAVE_H2E4M3  // ptx2rr.py's preamble (a translated head before this file) defines the same
uint g11_h2e4m3(uint h)
{
    const uint s = (h >> 8u) & 0x80u, a = h & 0x7fffu;
    if (a > 0x7c00u) return s | 0x7fu;
    uint c;
    if (a >= 0x2400u) {   // |x| >= 2^-6: an e4m3 normal; rebias 15 -> 7 and round 10 -> 3 mantissa bits
        const uint t = a - 0x2000u;
        c = min((t + 0x3fu + ((t >> 7u) & 1u)) >> 7u, 0x7eu);
    } else {              // e4m3 subnormals: steps of 2^-9 (x * 2^9 is exact in f32)
        c = uint(roundEven(float(uint16BitsToFloat16(uint16_t(a))) * 512.0));
    }
    return s | c;
}
#endif
uint g11_code(float16_t h) { return g11_h2e4m3(uint(float16BitsToUint16(h))); }
// an f32 that holds a binary16 value
uint g11_codef(float x) { return g11_code(float16_t(x)); }
// the exact binary16 value of a code (ZLUDA's e4m3_to_f16_bits; NaN -> 0x7fff with the sign)
float16_t g11_e4h(uint c)
{
    const uint m = c & 0x7fu;
    uint h = m >= 8u ? (m << 7u) + 0x2000u : uint(float16BitsToUint16(float16_t(float(m & 7u) * 0.001953125)));
    h = m == 0x7fu ? 0x7fffu : h;
    return uint16BitsToFloat16(uint16_t(((c & 0x80u) << 8u) | h));
}

#ifndef G11_HAVE_FILL
// four codes -> the scaled binary16 pairs (bytes 0, 1) and (bytes 2, 3), ZLUDA's e4m3x4_to_scaled_f16x2x2
uvec2 g11_sc4(uint packed)
{
    uint mag = packed & 0x7f7f7f7fu;
    mag |= (mag + 0x01010101u) & 0x80808080u;
    const uint sign = packed & 0x80808080u;
    const uint lo = ((mag & 0xffu) << 7u) | ((mag & 0xff00u) << 15u) | ((sign & 0x80u) << 8u) | ((sign & 0x8000u) << 16u);
    const uint hi = (((mag >> 16u) & 0xffu) << 7u) | ((mag >> 24u) << 23u) | ((sign & 0x800000u) >> 8u) | (sign & 0x80000000u);
    return uvec2(lo, hi);
}
// an operand from its lane's 16 codes in k order (four words): positions in ZLUDA's order
#define G11_FILL(m, w0, w1, w2, w3) { \
    const uvec2 s0_ = g11_sc4(w0), s1_ = g11_sc4(w1), s2_ = g11_sc4(w2), s3_ = g11_sc4(w3); f16vec2 l_; \
    l_ = unpackFloat2x16(s0_.x); (m)[0] = l_.x; (m)[1] = l_.y; l_ = unpackFloat2x16(s0_.y); (m)[8] = l_.x; (m)[9] = l_.y; \
    l_ = unpackFloat2x16(s1_.x); (m)[2] = l_.x; (m)[3] = l_.y; l_ = unpackFloat2x16(s1_.y); (m)[10] = l_.x; (m)[11] = l_.y; \
    l_ = unpackFloat2x16(s2_.x); (m)[4] = l_.x; (m)[5] = l_.y; l_ = unpackFloat2x16(s2_.y); (m)[12] = l_.x; (m)[13] = l_.y; \
    l_ = unpackFloat2x16(s3_.x); (m)[6] = l_.x; (m)[7] = l_.y; l_ = unpackFloat2x16(s3_.y); (m)[14] = l_.x; (m)[15] = l_.y; }
#endif
// the A operand row a lane holds: physical row r carries row (r >> 1) + 8 (r & 1)
uint g11_row(uint r) { return (r >> 1u) | ((r & 1u) << 3u); }
// an A operand row-major (rows permuted) or a B operand column-major from a shared byte image (word offset and
// stride, as coopMatLoad takes them for an FP8 matrix): the lane's row / column is 16 consecutive bytes
#define LD8A(m, ARR, off, stride) { const uint b_ = (off) + g11_row(gl_SubgroupInvocationID & 15u) * (stride); \
    G11_FILL(m, ARR[b_], ARR[b_ + 1u], ARR[b_ + 2u], ARR[b_ + 3u]) }
#define LD8B(m, ARR, off, stride) { const uint b_ = (off) + (gl_SubgroupInvocationID & 15u) * (stride); \
    G11_FILL(m, ARR[b_], ARR[b_ + 1u], ARR[b_ + 2u], ARR[b_ + 3u]) }
// natural k order (a matrix loaded from the widened weights) -> ZLUDA's positions
#define G11_PERM(d, s) { (d)[0] = (s)[0]; (d)[1] = (s)[1]; (d)[2] = (s)[4]; (d)[3] = (s)[5]; (d)[4] = (s)[8]; \
    (d)[5] = (s)[9]; (d)[6] = (s)[12]; (d)[7] = (s)[13]; (d)[8] = (s)[2]; (d)[9] = (s)[3]; (d)[10] = (s)[6]; \
    (d)[11] = (s)[7]; (d)[12] = (s)[10]; (d)[13] = (s)[11]; (d)[14] = (s)[14]; (d)[15] = (s)[15]; }

// FP8 vectors in the accumulator layout: element i of a lane (row i + 8 half) as byte i & 3 of word i >> 2
#define F8V uvec2
// the codes of an accumulator that holds binary16 values
#define G11_CODES(dst, src) { (dst) = uvec2(0u); [[unroll]] for (int i_ = 0; i_ < 8; ++i_) \
    (dst)[i_ >> 2] |= g11_codef((src)[i_]) << (8u * uint(i_ & 3)); }
// the B operand that is the accumulator-layout codes c (the gfx12 elementwise conversion of an accumulator into B):
// rows 0..7 are half-wave 0's codes, 8..15 half-wave 1's
#define G11_B(dst, c) { const uvec2 o_ = uvec2(subgroupShuffleXor((c).x, 16u), subgroupShuffleXor((c).y, 16u)); \
    const bool h_ = gl_SubgroupInvocationID >= 16u; const uvec2 e0_ = h_ ? o_ : (c), e1_ = h_ ? (c) : o_; \
    G11_FILL(dst, e0_.x, e0_.y, e1_.x, e1_.y) }
// store the codes c column-major as bytes (column j at word off + j * stride): what an FP8 B stored column-major or
// an FP8 A (the accumulator's transpose) stored row-major leaves; each half-wave writes its 8 rows
#define G11_ST(c, ARR, off, stride) { const uint b_ = (off) + (gl_SubgroupInvocationID & 15u) * (stride) + \
    (gl_SubgroupInvocationID >= 16u ? 2u : 0u); ARR[b_] = (c).x; ARR[b_ + 1u] = (c).y; }
