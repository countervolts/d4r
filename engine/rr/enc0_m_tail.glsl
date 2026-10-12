    // ======================================================================== native part (engine/rr/enc0_ref.py)
    memoryBarrierShared(); barrier();   // wave 0's translated prologue filled sh (features, noise) and wrote 304 / 1016
    const uint tid = gl_LocalInvocationIndex, wv = gl_SubgroupID, lane = gl_SubgroupInvocationID;
    const uint hl = lane >> 4u, col = lane & 15u;
    RR_ZERO16 = uint16_t(gl_NumWorkGroups.z - 1u);
    const int ex = int(param.w[0]), ey = int(param.w[1]);
    const uint outOff = param.w[12], mrgOff = param.w[14], shift = param.w[20];
    const int x0 = 8 * int(gl_WorkGroupID.x) - int(shift & 0xffffu), y0 = 8 * int(gl_WorkGroupID.y) - int(shift >> 16u);
    const uint lplane = uint((ex / 2) * (ey / 2)) * 32u;
    // the key slots' features, from the 10 x 10 grid (origin x0 - 1, y0 - 1)
    if (tid < 96u) {
        const ivec2 k = keyXY(inter(tid), x0, y0);
        const uint g = uint((k.y - y0 + 1) * 10 + (k.x - x0 + 1));
        [[unroll]] for (uint wd = 0u; wd < 4u; ++wd) {
            XF[tid * 8u + wd] = sh[(800u + g * 16u) / 4u + wd];
            XF[tid * 8u + 4u + wd] = sh[(2400u + g * 16u) / 4u + wd];
        }
    }
    barrier();
    // the embedding x = W2 q8(relu(W1 f + b1)) + b2 for token tiles n = wv (this wave's queries: x is also their
    // residual) and n = wv + 4 (the border keys, waves 0 and 1), then each key's norm in registers (dec0_ref.norm_plane
    // on the accumulator layout, as dec0_m.comp's MLP norm) to XT
    ACC R[2];
    for (uint n = wv; n < 6u; n += 4u) {
        B8 fb0, fb1, hb_[2];
        coopMatLoad(fb0, XF, n * 128u, 8u, CM);
        coopMatLoad(fb1, XF, n * 128u + 4u, 8u, CM);
        [[unroll]] for (uint m = 0u; m < 2u; ++m) {
            ACC a;
            [[unroll]] for (int i = 0; i < 8; ++i) a[i] = float(w16.w[W_B1 / 2u + 16u * m + 8u * hl + uint(i)]);
            A8 a0, a1;
            LDA(a0, W_W1, 32u, m, 0u); LDA(a1, W_W1, 32u, m, 1u);
            K32(a, a0, fb0, a1, fb1)
            [[unroll]] for (int i = 0; i < 8; ++i) hb_[m][i] = q8s(max(a[i], 0.0));
        }
        float16_t xv[2][8];
        [[unroll]] for (uint m = 0u; m < 2u; ++m) {
            ACC a;
            [[unroll]] for (int i = 0; i < 8; ++i) a[i] = float(w16.w[W_B2 / 2u + 16u * m + 8u * hl + uint(i)]);
            A8 a0, a1;
            LDA(a0, W_W2, 32u, m, 0u); LDA(a1, W_W2, 32u, m, 1u);
            K32(a, a0, hb_[0], a1, hb_[1])
            [[unroll]] for (int i = 0; i < 8; ++i) xv[m][i] = float16_t(a[i]);
            if (n == wv) {   // the residual: x + table 0
                [[unroll]] for (int i = 0; i < 8; ++i) {
                    precise float16_t c = xv[m][i] + w16.w[W_TABLE0 / 2u + 16u * m + 8u * hl + uint(i)];
                    R[m][i] = float(c);
                }
            }
        }
        float16_t h1[2];
        [[unroll]] for (uint c = 0u; c < 2u; ++c) {
            float16_t ht[2];
            [[unroll]] for (uint tq = 0u; tq < 2u; ++tq) {
                const uint i0 = 4u * tq + c, i1 = i0 + 2u;
                precise float16_t p0 = xv[0][i0] * xv[0][i0] + xv[0][i1] * xv[0][i1];
                precise float16_t p1 = xv[1][i0] * xv[1][i0] + xv[1][i1] * xv[1][i1];
                precise float16_t hv = p0 + p1;
                ht[tq] = hv;
            }
            precise float16_t hh = ht[0] + ht[1];
            h1[c] = hh;
        }
        const uint other = subgroupShuffleXor(bh(h1[0]) | (bh(h1[1]) << 16u), 16u);
        precise float16_t g0 = h1[0] + hb(other & 0xffffu), g1 = h1[1] + hb(other >> 16u);
        precise float16_t s = g0 + g1;
        precise float16_t se = s + float16_t(0.0001220703125);
        const float16_t r = rsq16(se);
        [[unroll]] for (uint m = 0u; m < 2u; ++m) {
            B8 xq;
            [[unroll]] for (int i = 0; i < 8; ++i) {
                precise float16_t g = r * w16.w[W_GAIN0 / 2u + inter(16u * m + 8u * hl + uint(i))];
                precise float16_t nn = xv[m][i] * g;
                xq[i] = q8h(nn);
            }
            coopMatStore(xq, XT, n * 128u + 4u * m, 8u, CM);
        }
    }
    barrier();   // XT complete; XF is read (the attention reuses its memory)
    // two heads
    [[unroll]] for (uint hd = 0u; hd < 2u; ++hd) {
        if (hd == 1u) barrier();
        for (uint j = 0u; j < 3u; ++j) {
            const uint job = wv + 4u * j, n = job >> 1u, c = job & 1u;
            A8 wk[2], xa[2];
            B8 wvb[2], xb[2];
            [[unroll]] for (uint kt = 0u; kt < 2u; ++kt) {
                coopMatLoad(xb[kt], XT, n * 128u + 4u * kt, 8u, CM);
                coopMatLoad(xa[kt], XT, n * 128u + 4u * kt, 8u, RM);
                LDA(wk[kt], W_WK0 + HEAD * hd, 32u, c, kt);
                LDB(wvb[kt], W_WV0 + HEAD * hd, 32u, kt, c);
            }
            ACC a = ACC(0.0);
            K32(a, wk[0], xb[0], wk[1], xb[1])
            A8 ka;
            Q8(ka, a)
            coopMatStore(ka, KVS, n * 128u + 4u * c, 8u, RM);
            ACC vv = ACC(0.0);
            K32(vv, xa[0], wvb[0], xa[1], wvb[1])
            A8 va;
            Q8(va, vv)
            coopMatStore(va, KVS, 768u + c * 384u + 4u * n, 24u, RM);
        }
        barrier();
#include "rr_attn.glsl"
        [[unroll]] for (uint m = 0u; m < 2u; ++m) {
            A8 a0, a1;
            LDA(a0, W_WO0 + HEAD * hd, 32u, m, 0u); LDA(a1, W_WO0 + HEAD * hd, 32u, m, 1u);
            K32(R[m], a0, Ob[0], a1, Ob[1])
        }
    }
    // MLP (dec0_m.comp's)
    B8 Nb[2];
    {
        float16_t rv[2][8];
        [[unroll]] for (uint m = 0u; m < 2u; ++m) [[unroll]] for (int i = 0; i < 8; ++i) rv[m][i] = float16_t(R[m][i]);
        float16_t h1[2];
        [[unroll]] for (uint c = 0u; c < 2u; ++c) {
            float16_t ht[2];
            [[unroll]] for (uint tq = 0u; tq < 2u; ++tq) {
                const uint i0 = 4u * tq + c, i1 = i0 + 2u;
                precise float16_t p0 = rv[0][i0] * rv[0][i0] + rv[0][i1] * rv[0][i1];
                precise float16_t p1 = rv[1][i0] * rv[1][i0] + rv[1][i1] * rv[1][i1];
                precise float16_t hv = p0 + p1;
                ht[tq] = hv;
            }
            precise float16_t hh = ht[0] + ht[1];
            h1[c] = hh;
        }
        const uint other = subgroupShuffleXor(bh(h1[0]) | (bh(h1[1]) << 16u), 16u);
        precise float16_t g0 = h1[0] + hb(other & 0xffffu), g1 = h1[1] + hb(other >> 16u);
        precise float16_t s = g0 + g1;
        precise float16_t se = s + float16_t(0.0001220703125);
        const float16_t r = rsq16(se);
        [[unroll]] for (uint m = 0u; m < 2u; ++m)
            [[unroll]] for (int i = 0; i < 8; ++i) {
                const uint k = 16u * m + 8u * hl + uint(i);
                precise float16_t g = r * w16.w[W_GAIN1 / 2u + k];
                precise float16_t nn = rv[m][i] * g;
                Nb[m][i] = q8h(nn);
                precise float16_t a = rv[m][i] + w16.w[W_TABLE1 / 2u + k];
                R[m][i] = float(a);
            }
    }
    for (uint br = 0u; br < 4u; ++br) {
        const uint w1 = W_W1_0 + BRANCH * br, b1 = W_B1_0 + BRANCH * br, w2 = W_W2_0 + BRANCH * br;
        B8 Hb[2];
        [[unroll]] for (uint c = 0u; c < 2u; ++c) {
            ACC h;
            [[unroll]] for (int i = 0; i < 8; ++i) h[i] = float(w16.w[b1 / 2u + 16u * c + 8u * hl + uint(i)]);
            A8 a0, a1;
            LDA(a0, w1, 32u, c, 0u); LDA(a1, w1, 32u, c, 1u);
            K32(h, a0, Nb[0], a1, Nb[1])
            [[unroll]] for (int i = 0; i < 8; ++i) Hb[c][i] = ACTQ(float16_t(h[i]));
        }
        [[unroll]] for (uint m = 0u; m < 2u; ++m) {
            A8 a0, a1;
            LDA(a0, w2, 32u, m, 0u); LDA(a1, w2, 32u, m, 1u);
            K32(R[m], a0, Hb[0], a1, Hb[1])
        }
    }
    // outputs: the tokens (one plane, dec0's skip) and the merge rows
    const uint t = inter(16u * wv + col);
    const int tx = x0 + int(t & 7u), ty = y0 + int(t >> 3u);
    const bool inside = tx >= 0 && ty >= 0 && tx < ex && ty < ey;
    B8 Tb[2];
    {
        const uint ty2 = t >> 3u, tx2 = t & 7u;
        const uint mr = (ty2 >> 1u) < 2u ? 2u * (tx2 >> 1u) + (ty2 >> 1u) : 8u + 2u * (tx2 >> 1u) + (ty2 >> 1u) - 2u;
        const uint blk = 2u * (ty2 & 1u) + (tx2 & 1u);
        const uint o = (outOff + uint(ty * ex + tx) * 32u) >> 2u;
        barrier();   // XT is read; the merge rows reuse it
        [[unroll]] for (uint m = 0u; m < 2u; ++m) {
            Q8(Tb[m], R[m])
            const uint q0 = pack4(Tb[m][0], Tb[m][1], Tb[m][2], Tb[m][3]), q1 = pack4(Tb[m][4], Tb[m][5], Tb[m][6], Tb[m][7]);
            const uint at = 4u * m + 2u * hl;
            if (inside) { arena.w[o + at] = q0; arena.w[o + at + 1u] = q1; }
            const uint gm = mr * 32u + blk * 8u + 4u * m + 2u * hl;
            MG[gm] = q0; MG[gm + 1u] = q1;
        }
    }
    {   // the 32 -> 16 head and encoder 0's filter of the noise grid (two f16 fma chains, taps even / odd)
        ACC hv;
        [[unroll]] for (int i = 0; i < 8; ++i) hv[i] = float(w16.w[W_BH / 2u + 8u * hl + uint(i)]);
        A8 a0, a1;
        LDA(a0, W_WH, 32u, 0u, 0u); LDA(a1, W_WH, 32u, 0u, 1u);
        K32(hv, a0, Tb[0], a1, Tb[1])
        float16_t l[12];
        [[unroll]] for (int i = 0; i < 8; ++i) l[i] = float16_t(hv[i]);
        const uint u0 = subgroupShuffleXor(bh(l[0]) | (bh(l[1]) << 16u), 16u), u1 = subgroupShuffleXor(bh(l[2]) | (bh(l[3]) << 16u), 16u);
        l[8] = hb(u0 & 0xffffu); l[9] = hb(u0 >> 16u); l[10] = hb(u1 & 0xffffu); l[11] = hb(u1 >> 16u);
        const float16_t ma = max(max(max(l[0], l[2]), l[4]), l[6]), mb = max(max(max(l[1], l[3]), l[5]), l[7]);
        const float16_t mx = max(max(ma, mb), l[8]);
        float e[9];
        [[unroll]] for (uint i = 0u; i < 9u; ++i) { precise float16_t d = l[i] - mx; e[i] = exp2(float(d) * 1.4426950216293335); }
        precise float s = (e[0] + e[1]) + 0.0;
        s = s + (e[2] + e[3]);
        s = s + (e[4] + e[5]);
        s = s + (e[6] + e[7]);
        s = s + e[8];
        const float16_t rr = rcp16(s);
        float16_t wg[9];
        [[unroll]] for (uint i = 0u; i < 9u; ++i) wg[i] = h16(r16(e[i]) * float(rr));
        const uint gx = t & 7u, gy = t >> 3u;   // the token's 3 x 3 taps in the grid: (gx .. gx + 2, gy .. gy + 2)
        float16_t o112[3], o120[3];
        [[unroll]] for (uint c = 0u; c < 3u; ++c) {
            float16_t lo = 0.0hf, hi = 0.0hf;
            [[unroll]] for (uint k = 0u; k < 8u; k += 2u) {
                lo = fma16(noiseAt(gx + k % 3u, gy + k / 3u, c), wg[k], lo);
                hi = fma16(noiseAt(gx + (k + 1u) % 3u, gy + (k + 1u) / 3u, c), wg[k + 1u], hi);
            }
            precise float16_t sum = lo + hi;
            const float16_t f = fma16(noiseAt(gx + 2u, gy + 2u, c), wg[8], sum);
            precise float16_t a = sig(l[9u + c]) * f;
            precise float16_t b = f - a;
            o120[c] = a; o112[c] = b;
        }
        if (hl == 0u) {
            if (inside) {
                imageStore(S112, ivec2(tx, ty), vec4(o112[0], o112[1], o112[2], 0.0));
                imageStore(S120, ivec2(tx, ty), vec4(o120[0], o120[1], o120[2], 0.0));
            }
            [[unroll]] for (uint c = 0u; c < 3u; ++c) O112[t * 4u + c] = o112[c];
        }
    }
    barrier();
    {   // the 2x2 merge (128 -> 64): wave w computes channel tile w
        const uvec2 mxy = mergeXY(col);
        const int X = x0 / 2 + int(mxy.x), Y = y0 / 2 + int(mxy.y), hx = ex / 2, hy = ey / 2;
        const uint m = wv;
        ACC a;
        [[unroll]] for (int i = 0; i < 8; ++i) a[i] = float(w16.w[W_BM / 2u + 16u * m + 8u * hl + uint(i)]);
        [[unroll]] for (uint ks = 0u; ks < 4u; ++ks) {
            B8 g0, g1; A8 w0, w1;
            coopMatLoad(g0, MG, 8u * ks, 32u, CM);
            coopMatLoad(g1, MG, 8u * ks + 4u, 32u, CM);
            LDA(w0, W_WM, 128u, m, 2u * ks); LDA(w1, W_WM, 128u, m, 2u * ks + 1u);
            K32(a, w0, g0, w1, g1)
        }
        B8 q;
        Q8(q, a)
        if (X >= 0 && Y >= 0 && X < hx && Y < hy) {
            const uint at = (mrgOff + (m >> 1u) * lplane + uint(Y * hx + X) * 32u) / 4u + 4u * (m & 1u) + 2u * hl;
            arena.w[at] = pack4(q[0], q[1], q[2], q[3]);
            arena.w[at + 1u] = pack4(q[4], q[5], q[6], q[7]);
        }
    }
    if (wv == 3u && lane < 16u) {   // surface 128: 2x2 sums of 0.25 * surface 112, rightwards first, then downwards
        const uint mx = lane & 3u, my = lane >> 2u;
        const int x = x0 + 2 * int(mx), y = y0 + 2 * int(my);
        if (x >= 0 && y >= 0 && x < ex && y < ey) {
            const uint t00 = (2u * my) * 8u + 2u * mx;
            vec4 out_ = vec4(0.0);
            [[unroll]] for (uint c = 0u; c < 3u; ++c) {
                precise float16_t a00 = O112[t00 * 4u + c] * float16_t(0.25), a01 = O112[(t00 + 1u) * 4u + c] * float16_t(0.25);
                precise float16_t a10 = O112[(t00 + 8u) * 4u + c] * float16_t(0.25), a11 = O112[(t00 + 9u) * 4u + c] * float16_t(0.25);
                precise float16_t b0 = a00 + a01, b1 = a10 + a11;
                precise float16_t cc = b0 + b1;
                out_[c] = float(cc);
            }
            imageStore(S128, ivec2(x >> 1, y >> 1), out_);
        }
    }
