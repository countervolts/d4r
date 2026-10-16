// The attention core shared by the 32-channel-head stages (engine/rr/enc1_m.comp, dec0_m.comp), included inside the
// head loop: this wave's 16 query slots against the window's 96 key slots. Expects KS (K, token slot x 32 FP8
// channels in k order) and VS (V^T, 32 channels x 96 token slots) filled for head hd, the bias at W_BIAS0 + HEAD * hd
// (enc1_pack: 96 x 64, [key slot][query slot]) and the cubic table at W_CUBIC. Leaves Ob[2]: O^T (32 channels x
// this wave's queries) as the out-projection's B operand. RR_KS / RR_VS (+ word offsets RR_KS0 / RR_VS0) name
// the arrays when they are not KS and VS.
#ifndef RR_KS
#define RR_KS KS
#define RR_KS0 0u
#endif
#ifndef RR_VS
#define RR_VS VS
#define RR_VS0 0u
#endif
        B8 Qb[2];
        [[unroll]] for (uint c = 0u; c < 2u; ++c) LDS8B(Qb[c], RR_KS, RR_KS0 + wv * 128u + 4u * c, 8u);
        // scores S^T (keys x this wave's queries) + bias, then the cubic softmax down each column
        B8 Pb[6];
        {
            float16_t e[6][8];
            [[unroll]] for (uint kt = 0u; kt < 6u; ++kt) {
                HAC c;
#ifdef RR_GFX11
                // the accumulator layout the stages are written for (rr_gfx11.glsl): element i is row i + 8 hl
                [[unroll]] for (int i = 0; i < 8; ++i)
                    c[i] = w16.w[(W_BIAS0 + HEAD * hd) / 2u + kt * 1024u + 16u * wv + (8u * hl + uint(i)) * 64u + col];
#else
                coopMatLoad(c, w16.w, (W_BIAS0 + HEAD * hd) / 2u + kt * 1024u + 16u * wv, 64u, RM);
#endif
                A8 k0, k1;
                LDS8A(k0, RR_KS, RR_KS0 + kt * 128u, 8u);
                LDS8A(k1, RR_KS, RR_KS0 + kt * 128u + 4u, 8u);
                K32H(c, k0, Qb[0], k1, Qb[1])
                // the cubic numerator from enc1_pack.cubic_table, indexed by the score's binary16 bits
                [[unroll]] for (int i = 0; i < 8; ++i) e[kt][i] = hb(uint(w16u.w[W_CUBIC / 2u + bh(c[i])]));
            }
            // key slot 16kt + 8hl + i is NVIDIA's column 8j + 2tt + half with j = 4(kt >> 1) + (kt & 1) + 2((i >> 1) & 1),
            // tt = 2hl + (i >> 2), half = i & 1 (enc1_ref.softmax_cubic's tree)
            // packed: component h of a pair is column parity h (elements i = 4tq + h, i + 2), so the tree runs on f16x2
            f16vec2 ep[6][4];
            [[unroll]] for (uint kt = 0u; kt < 6u; ++kt)
                [[unroll]] for (uint q = 0u; q < 4u; ++q) ep[kt][q] = f16vec2(e[kt][2u * q], e[kt][2u * q + 1u]);
            precise f16vec2 st[2];
            [[unroll]] for (uint tq = 0u; tq < 2u; ++tq) {   // pairs q = 2tq (i0) and 2tq + 1 (i1 = i0 + 2)
                const uint q0 = 2u * tq, q1 = q0 + 1u;
                precise f16vec2 ev = ((ep[0][q0] + ep[0][q1]) + (ep[2][q0] + ep[2][q1])) + (ep[4][q0] + ep[4][q1]);
                precise f16vec2 od = ((ep[1][q0] + ep[1][q1]) + (ep[3][q0] + ep[3][q1])) + (ep[5][q0] + ep[5][q1]);
                st[tq] = ev + od;
            }
            precise f16vec2 loc = st[0] + st[1];
            const uint other = subgroupShuffleXor(packFloat2x16(loc), 16u);
            precise f16vec2 o = loc + unpackFloat2x16(other);
            precise float16_t rs = o.x + o.y;
            // f16(1 / rs) for a binary16 rs: x * m = 1 + d with |d| >= 2^-22 for every binary16 midpoint m, so the f32
            // reciprocal (within an ulp) already rounds correctly
            const float16_t r = float16_t(1.0 / float(rs));
            const f16vec2 r2 = f16vec2(r, r);
            [[unroll]] for (uint kt = 0u; kt < 6u; ++kt) {
#ifdef RR_GFX11
                F8V pc_ = uvec2(0u);
#endif
                [[unroll]] for (uint q = 0u; q < 4u; ++q) {
                    precise f16vec2 pv = ep[kt][q] * r2;
#ifdef RR_GFX11
                    pc_[q >> 1u] |= (q8h(pv.x) | (q8h(pv.y) << 8u)) << (16u * (q & 1u));
#else
                    // a probability is in [0, 1]: the plain conversion, no saturation clamp, both in one instruction
                    { const fe4m3vec2 t_ = fe4m3vec2(pv); Pb[kt][2u * q] = t_.x; Pb[kt][2u * q + 1u] = t_.y; }
#endif
                }
#ifdef RR_GFX11
                G11_B(Pb[kt], pc_)
#endif
            }
        }
        // O^T = V^T P^T, then the out-projection onto the residual
        B8 Ob[2];
        [[unroll]] for (uint c = 0u; c < 2u; ++c) {
            RAC o = RAC(RV(float16_t(0.0)));
            [[unroll]] for (uint ks = 0u; ks < 3u; ++ks) {
                A8 v0, v1;
                LDS8A(v0, RR_VS, RR_VS0 + c * 384u + 8u * ks, 24u);
                LDS8A(v1, RR_VS, RR_VS0 + c * 384u + 8u * ks + 4u, 24u);
                K32(o, v0, Pb[2u * ks], v1, Pb[2u * ks + 1u])
            }
#ifdef RR_GFX11
            { F8V oc_; G11_CODES(oc_, o) G11_B(Ob[c], oc_) }
#else
            Q8(Ob[c], o)
#endif
        }
