// RR_GFX11: the weight loads and the 32-deep step of the native stages (rr_gfx11.glsl explains the arithmetic).
// Included after ACC / HAC / RM / CM and the binding-1 views. W_WIDE (a -D of the build) is the f16 element at which
// the weight file carries its widened copy: element W_WIDE + j is byte j of the image widened like an FP8 operand
// (value * 2^-8), so a stage's byte offsets address it unchanged.
layout(set = 0, binding = 1, std430) readonly buffer W128 { uvec4 w[]; } w128;
#ifndef W_WIDE
#error RR_GFX11 needs W_WIDE (the widened weight copy, engine/rr/widen_weights.py)
#endif
// A: the lane's (permuted) row is 16 consecutive f16, two 16-byte loads (base and tiles are multiples of 16 bytes)
#define LDA(dst, base, k, mt, kt) { const uint e_ = W_WIDE + (base) + 256u * ((mt) * ((k) / 16u) + (kt)) + \
    16u * g11_row(gl_SubgroupInvocationID & 15u); const uvec4 u0_ = w128.w[e_ >> 3u], u1_ = w128.w[(e_ >> 3u) + 1u]; \
    f16vec2 l_; l_ = unpackFloat2x16(u0_.x); (dst)[0] = l_.x; (dst)[1] = l_.y; l_ = unpackFloat2x16(u0_.y); (dst)[8] = l_.x; (dst)[9] = l_.y; \
    l_ = unpackFloat2x16(u0_.z); (dst)[2] = l_.x; (dst)[3] = l_.y; l_ = unpackFloat2x16(u0_.w); (dst)[10] = l_.x; (dst)[11] = l_.y; \
    l_ = unpackFloat2x16(u1_.x); (dst)[4] = l_.x; (dst)[5] = l_.y; l_ = unpackFloat2x16(u1_.y); (dst)[12] = l_.x; (dst)[13] = l_.y; \
    l_ = unpackFloat2x16(u1_.z); (dst)[6] = l_.x; (dst)[7] = l_.y; l_ = unpackFloat2x16(u1_.w); (dst)[14] = l_.x; (dst)[15] = l_.y; }
#define LDB(dst, base, n, kt, nt) { B8 t_; coopMatLoad(t_, w16.w, W_WIDE + (base) + 256u * ((kt) * ((n) / 16u) + (nt)), 16u, RM); G11_PERM(dst, t_) }
#define RND(m) [[unroll]] for (int i_ = 0; i_ < 8; ++i_) (m)[i_] = rnd16((m)[i_]);
#if defined(RR_MIX) || defined(RR_RELAXED)
#error RR_GFX11 builds the exact arithmetic only
#endif
#define RAC ACC
#define RV(x) float(x)
// C * 2^-16 (exact), the two halves, * 2^16 (exact), one rounding to binary16
#define K32(acc, a0, b0, a1, b1) { [[unroll]] for (int i_ = 0; i_ < 8; ++i_) (acc)[i_] *= 1.52587890625e-05; \
    (acc) = coopMatMulAdd((a1), (b1), coopMatMulAdd((a0), (b0), (acc))); \
    [[unroll]] for (int i_ = 0; i_ < 8; ++i_) (acc)[i_] = rnd16((acc)[i_] * 65536.0); }
#define K32H(acc, a0, b0, a1, b1) { ACC p_ = ACC(acc); [[unroll]] for (int i_ = 0; i_ < 8; ++i_) p_[i_] *= 1.52587890625e-05; \
    p_ = coopMatMulAdd((a1), (b1), coopMatMulAdd((a0), (b0), p_)); \
    [[unroll]] for (int i_ = 0; i_ < 8; ++i_) (acc)[i_] = float16_t(p_[i_] * 65536.0); }
