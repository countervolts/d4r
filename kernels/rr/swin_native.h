// A complete Ray Reconstruction Swin encoder stage as one native HIP kernel (gfx12, wave32, native FP8 WMMA): the port
// of engine/rr/swin_m.comp's KIND 0 path (tokens from the arena, attention, MLP, kernel-prediction filter, 2x2 merge),
// with nothing of NVIDIA's PTX in it. The arithmetic, slot orders and weight layout are that shader's (see its header
// and engine/rr/swin_pack.py); a line here corresponds to a line there.
//
// Differences from the shader's inputs, all made by the host (kernels/tools/rr_swin_native_test.cpp):
//   - the V weights' tiles are stored [n16][k16] and the attention bias [query][key], so every operand of every
//     product is 8 contiguous bytes per lane (the shader's row-major B loads are strided gathers);
//   - texture 88 and the three surfaces are plain RGBA16F buffers.
//
// The including file defines the stage's shape and weight offsets first (the swin_stage.h that swin_pack.py writes).
// Block (32, 1, 4): one 8x8-token window, wave w = threadIdx.z owns query slots 16w .. 16w + 15.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#if KIND != 0 || CX == 0
#error "swin_native.h covers the encoder stages with a merge (enc1 - enc4)"
#endif
#define SN_KC (C / 32)
#define SN_MT (C / 16)
#define SN_NB (C / 8)
#define SN_HEAD (W_WK1 - W_WK0)
#define SN_BRANCH (W_W1_1 - W_W1_0)
// profiling only: fewer attention heads / MLP branches than the stage has (the output is then wrong)
#ifndef SN_HEADS
#define SN_HEADS NH
#endif
#ifndef SN_BRANCHES
#define SN_BRANCHES SN_NB
#endif

#pragma clang fp contract(off)

namespace swin_native
{
typedef _Float16 hf;
typedef float f8v __attribute__((ext_vector_type(8)));
typedef int i2v __attribute__((ext_vector_type(2)));
struct h2 { hf x, y; };
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
#define SN_DEV __device__ __forceinline__

SN_DEV h2 operator+(h2 a, h2 b) { return {(hf)(a.x + b.x), (hf)(a.y + b.y)}; }
SN_DEV h2 operator*(h2 a, h2 b) { return {(hf)(a.x * b.x), (hf)(a.y * b.y)}; }
SN_DEV uint32_t inter(uint32_t j) { return (j & ~31u) | (((j & 31u) >> 2) << 1) | (j & 1u) | (((j >> 1) & 1u) << 4); }
SN_DEV int mirror(int v, int n) { v = v < 0 ? -v : v; const int m = 2 * n - 2 - v; return v < m ? v : m; }
SN_DEV hf hb(uint32_t b) { return __builtin_bit_cast(hf, (uint16_t)b); }
SN_DEV uint32_t bh(hf h) { return __builtin_bit_cast(uint16_t, h); }
SN_DEV float r16(float x) { return (float)(hf)x; }
SN_DEV uint32_t other_half(uint32_t v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false); }
// byte b of w as an e4m3 value
SN_DEV hf e4(uint32_t w, int b) { return (hf)__builtin_amdgcn_cvt_f32_fp8((int)(w >> (8 * b)), 0); }
SN_DEV float sat(float x) { return __builtin_fminf(__builtin_fmaxf(x, -448.0f), 448.0f); }
// four values -> four e4m3 codes in one word (no saturation: the caller clamps where the value can exceed 448)
SN_DEV uint32_t pk4(float a, float b, float c, float d)
{
    const int w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    return (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
}
SN_DEV uint32_t q4(float a, float b, float c, float d) { return pk4(sat(a), sat(b), sat(c), sat(d)); }
// an accumulator (rows 8h + i of this lane's column) as the operand of the next product (the rows become K)
// A binary16 accumulator: rows 8h + i of this lane's column, two per register
struct A16
{
    hv2 p[4];
    SN_DEV hf get(int i) const { return p[i >> 1][i & 1]; }
    SN_DEV void set(int i, hf v) { p[i >> 1][i & 1] = v; }
};
SN_DEV A16 zero16() { A16 r; r.p[0] = r.p[1] = r.p[2] = r.p[3] = (hv2){0, 0}; return r; }
SN_DEV A16 load16(const uint8_t* q) { return *(const A16*)q; }   // 8 contiguous binary16 values
SN_DEV i2v q8(const A16& a)
{
    return (i2v){(int)q4((float)a.get(0), (float)a.get(1), (float)a.get(2), (float)a.get(3)),
                 (int)q4((float)a.get(4), (float)a.get(5), (float)a.get(6), (float)a.get(7))};
}
// c = f16(P + c) with one rounding (v_fma_mix: P is f32, the addend and the result binary16), in place: mixlo writes
// only the low half and mixhi reads only the high half of the addend. The reference adds c inside the f32 product
// and rounds that to binary16 (two roundings); kernels/m/swin_common.h found the two equal on its launches, and the
// stage test compares every byte against the capture.
SN_DEV void mix(A16& c, f8v P)
{
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        asm("v_fma_mixlo_f16 %0, %1, 1.0, %0 op_sel_hi:[0,0,1]" : "+v"(c.p[j]) : "v"(P[2 * j]));
        asm("v_fma_mixhi_f16 %0, %1, 1.0, %0 op_sel:[0,0,1] op_sel_hi:[0,0,1]" : "+v"(c.p[j]) : "v"(P[2 * j + 1]));
    }
}
SN_DEV void k32(A16& c, i2v a0, i2v b0, i2v a1, i2v b1)
{
    f8v P = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, b0, (f8v){0, 0, 0, 0, 0, 0, 0, 0});
    P = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, b1, P);
    mix(c, P);
}
// this lane's 8 K values of row / column `col` of a 16-byte-per-row image
SN_DEV i2v op(const uint8_t* p) { return *(const i2v*)p; }
SN_DEV i2v tile(const uint8_t* w, uint32_t base, uint32_t t, uint32_t col, uint32_t hl) { return op(w + base + 256u * t + 16u * col + 8u * hl); }
SN_DEV hf h16at(const uint8_t* w, uint32_t byte) { return *(const hf*)(w + byte); }

SN_DEV int cmp1(float m, float x)
{
    const float t = m * x;
    const float e = __builtin_fmaf(m, x, -t);
    return t > 1.0f ? 1 : (t < 1.0f ? -1 : (e > 0.0f ? 1 : (e < 0.0f ? -1 : 0)));
}
// binary16 rounding of 1 / sqrt(s) (s > 0), as engine/rr/rr_exact.glsl
SN_DEV hf rsq16(hf s16)
{
    const float s = (float)s16;
    hf q = (hf)__builtin_amdgcn_rsqf(s);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t b = bh(q);
        const float lo = ((float)q + (float)hb(b - 1u)) * 0.5f, hi = ((float)q + (float)hb(b + 1u)) * 0.5f;
        const int cl = cmp1(lo * lo, s), ch = cmp1(hi * hi, s);
        if (cl > 0 || (cl == 0 && (b & 1u) != 0u)) q = hb(b - 1u);
        else if (ch < 0 || (ch == 0 && (b & 1u) != 0u)) q = hb(b + 1u);
    }
    return q;
}
SN_DEV hf rcp16(float x) { return (hf)__builtin_amdgcn_rcpf(x); }
SN_DEV hf sig(hf v) { return rcp16(__builtin_amdgcn_exp2f((float)v * -1.4426950216293335f) + 1.0f); }
template <typename T> SN_DEV T tree(const T* v)
{
#if SN_KC == 1
    return v[0];
#elif SN_KC == 2
    return v[0] + v[1];
#elif SN_KC == 3
    return (v[0] + v[1]) + v[2];
#elif SN_KC == 4
    return (v[0] + v[1]) + (v[2] + v[3]);
#else
    return ((v[0] + v[1]) + (v[2] + v[3])) + v[4];
#endif
}
SN_DEV void key_xy(uint32_t t, int x0, int y0, int& x, int& y)
{
    if (t < 64u) { x = x0 + (int)(t & 7u); y = y0 + (int)(t >> 3); return; }
    const int i = (int)(t & 7u), side = (int)((t - 64u) >> 3);
    x = side < 2 ? x0 + i : (side == 2 ? x0 - 1 : x0 + 8);
    y = side == 0 ? y0 - 1 : (side == 1 ? y0 + 8 : y0 + i);
}
SN_DEV const hf* texel(const hf* t88, int ex, int x, int y) { return t88 + 4 * (y * ex + x); }

// param: the stage's parameter block (engine/rr/test_layer.stage_params); t88 and s112 / s120 are ex x ey, s128 half
__global__ void swin_stage(const uint32_t* __restrict__ param, const uint8_t* __restrict__ w, uint8_t* __restrict__ arena_b,
                           const hf* __restrict__ t88, hf* __restrict__ s112, hf* __restrict__ s120, hf* __restrict__ s128)
{
    __shared__ __attribute__((aligned(16))) uint8_t XT[96 * C];     // the normalised key tokens, then the merge rows
    __shared__ __attribute__((aligned(16))) uint8_t KVS[2 * 3072];  // the current head's K, then V^T
    __shared__ hf O112[64 * 4];
    uint8_t* const KS = KVS;
    uint8_t* const VS = KVS + 3072;
    uint32_t* const arena = (uint32_t*)arena_b;

    const uint32_t lane = threadIdx.x, wv = threadIdx.z, tid = 32u * wv + lane;
    const uint32_t hl = lane >> 4, col = lane & 15u;
    const int ex = (int)param[0], ey = (int)param[1], hx = ex / 2, hy = ey / 2;
    const uint32_t inOff = param[2], outOff = param[12], mrgOff = param[14], shift = param[20];
    const int x0 = 8 * (int)blockIdx.x - (int)(shift & 0xffffu), y0 = 8 * (int)blockIdx.y - (int)(shift >> 16);
    const uint32_t plane = (uint32_t)(ex * ey) * 32u, lplane = (uint32_t)(hx * hy) * 32u;

    // ---------------------------------------------------------------- prologue: one key token per thread
    if (tid < 96u) {
        int kx, ky;
        key_xy(inter(tid), x0, y0, kx, ky);
        const int xx = mirror(kx, ex), yy = mirror(ky, ey);
        const uint32_t tb = (inOff + (uint32_t)(yy * ex + xx) * 32u) / 4u;
        hf v[C];
#pragma unroll
        for (uint32_t wd = 0; wd < C / 4; ++wd) {
            const uint32_t t = arena[tb + (wd >> 3) * (plane >> 2) + (wd & 7u)];
#pragma unroll
            for (int b = 0; b < 4; ++b) v[4u * wd + b] = e4(t, b);
        }
        h2 h[4];
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) {
            h2 hh[2];
#pragma unroll
            for (uint32_t half_ = 0; half_ < 2; ++half_) {
                h2 pr[SN_KC];
#pragma unroll
                for (uint32_t p = 0; p < SN_KC; ++p) {
                    const uint32_t o = 32u * p + 4u * (q + 4u * half_);
                    const h2 a = h2{v[o], v[o + 1]} * h2{v[o], v[o + 1]};
                    const h2 c = h2{v[o + 2], v[o + 3]} * h2{v[o + 2], v[o + 3]};
                    pr[p] = a + c;
                }
                hh[half_] = tree(pr);
            }
            h[q] = hh[0] + hh[1];
        }
        const h2 h0 = h[0] + h[1], h2_ = h[2] + h[3];
        const h2 hs = h0 + h2_;
        const hf sm = hs.x + hs.y;
        const hf se = sm + (hf)0.0001220703125f;
        const hf rr = rsq16(se);
#pragma unroll
        for (uint32_t wd = 0; wd < C / 4; ++wd) {
            float n[4];
#pragma unroll
            for (uint32_t b = 0; b < 4; ++b) {
                const uint32_t j = 4u * wd + b;
                const hf g = rr * h16at(w, W_GAIN0 + 2u * inter(j));
                n[b] = (float)(hf)(v[j] * g);
            }
            *(uint32_t*)(XT + tid * C + 4u * wd) = q4(n[0], n[1], n[2], n[3]);
        }
    }
    __syncthreads();

    // ---------------------------------------------------------------- the residual: the window token + table 7
    A16 R[SN_MT];
    {
        const uint32_t tq = inter(16u * wv + col);
        const int qx = x0 + (int)(tq & 7u), qy = y0 + (int)(tq >> 3);
        const uint32_t tb = (inOff + (uint32_t)(mirror(qy, ey) * ex + mirror(qx, ex)) * 32u) / 4u;
#pragma unroll
        for (uint32_t m = 0; m < SN_MT; ++m) {
            const uint32_t ta = tb + (m >> 1) * (plane >> 2) + 4u * (m & 1u) + 2u * hl;
            const uint32_t t[2] = {arena[ta], arena[ta + 1]};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const hf c = e4(t[i >> 2], i & 3) + h16at(w, W_TABLE7 + 2u * (16u * m + 8u * hl + (uint32_t)i));
                R[m].set(i, c);
            }
        }
    }

    // ---------------------------------------------------------------- attention, NH heads
#pragma nounroll
    for (uint32_t hd = 0; hd < SN_HEADS; ++hd) {
        __syncthreads();
        const uint32_t wk = W_WK0 + SN_HEAD * hd, wvb = W_WV0 + SN_HEAD * hd, wo = W_WO0 + SN_HEAD * hd;
        for (uint32_t j = 0; j < 3; ++j) {   // 12 jobs (token tile n, channel tile c)
            const uint32_t job = wv + 4u * j, n = job >> 1, c = job & 1u;
            A16 a = zero16(), v = zero16();
#pragma unroll
            for (uint32_t kt = 0; kt < 2 * SN_KC; kt += 2) {
                // token tile n's rows are the A operand of V and the B operand of K: the same 8 bytes
                const i2v x0_ = op(XT + (16u * n + col) * C + 16u * kt + 8u * hl), x1_ = op(XT + (16u * n + col) * C + 16u * kt + 16u + 8u * hl);
                k32(a, tile(w, wk, c * (C / 16) + kt, col, hl), x0_, tile(w, wk, c * (C / 16) + kt + 1, col, hl), x1_);
                k32(v, x0_, tile(w, wvb, kt * 2 + c, col, hl), x1_, tile(w, wvb, (kt + 1) * 2 + c, col, hl));
            }
            *(i2v*)(KS + (16u * n + col) * 32u + 16u * c + 8u * hl) = q8(a);
            *(i2v*)(VS + (16u * c + col) * 96u + 16u * n + 8u * hl) = q8(v);
        }
        __syncthreads();
        i2v Qb[2], Pb[6];
#pragma unroll
        for (uint32_t c = 0; c < 2; ++c) Qb[c] = op(KS + (16u * wv + col) * 32u + 16u * c + 8u * hl);
        {
            hf e[6][8];
            const uint8_t* bias = w + W_BIAS0 + SN_HEAD * hd + (16u * wv + col) * 192u + 16u * hl;   // [query][key] f16
#pragma unroll
            for (uint32_t kt = 0; kt < 6; ++kt) {
                A16 s = load16(bias + 32u * kt);
                k32(s, op(KS + (16u * kt + col) * 32u + 8u * hl), Qb[0], op(KS + (16u * kt + col) * 32u + 16u + 8u * hl), Qb[1]);
#pragma unroll
                for (int i = 0; i < 8; ++i) e[kt][i] = hb(*(const uint16_t*)(w + W_CUBIC + 2u * bh(s.get(i))));
            }
            h2 ep[6][4];
#pragma unroll
            for (uint32_t kt = 0; kt < 6; ++kt)
#pragma unroll
                for (uint32_t q = 0; q < 4; ++q) ep[kt][q] = h2{e[kt][2 * q], e[kt][2 * q + 1]};
            h2 st[2];
#pragma unroll
            for (uint32_t tq = 0; tq < 2; ++tq) {
                const uint32_t q0 = 2u * tq, q1 = q0 + 1u;
                const h2 ev = ((ep[0][q0] + ep[0][q1]) + (ep[2][q0] + ep[2][q1])) + (ep[4][q0] + ep[4][q1]);
                const h2 od = ((ep[1][q0] + ep[1][q1]) + (ep[3][q0] + ep[3][q1])) + (ep[5][q0] + ep[5][q1]);
                st[tq] = ev + od;
            }
            const h2 loc = st[0] + st[1];
            const uint32_t other = other_half(bh(loc.x) | (bh(loc.y) << 16));
            const h2 o = loc + h2{hb(other & 0xffffu), hb(other >> 16)};
            const hf rs = o.x + o.y;
            const hf r = (hf)__builtin_amdgcn_rcpf((float)rs);
#pragma unroll
            for (uint32_t kt = 0; kt < 6; ++kt) {
                float p[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) p[i] = (float)(hf)(e[kt][i] * r);
                Pb[kt] = (i2v){(int)pk4(p[0], p[1], p[2], p[3]), (int)pk4(p[4], p[5], p[6], p[7])};
            }
        }
        i2v Ob[2];
#pragma unroll
        for (uint32_t c = 0; c < 2; ++c) {
            A16 o = zero16();
#pragma unroll
            for (uint32_t ks = 0; ks < 3; ++ks) {
                const uint8_t* vp = VS + (16u * c + col) * 96u + 32u * ks + 8u * hl;
                k32(o, op(vp), Pb[2 * ks], op(vp + 16), Pb[2 * ks + 1]);
            }
            Ob[c] = q8(o);
        }
#pragma unroll
        for (uint32_t m = 0; m < SN_MT; ++m)
            k32(R[m], tile(w, wo, 2 * m, col, hl), Ob[0], tile(w, wo, 2 * m + 1, col, hl), Ob[1]);
    }

    // ---------------------------------------------------------------- MLP
    i2v Nb[SN_MT];
    {
        hf rv[SN_MT][8];
#pragma unroll
        for (uint32_t m = 0; m < SN_MT; ++m)
#pragma unroll
            for (int i = 0; i < 8; ++i) rv[m][i] = R[m].get(i);
        hf h1[2];
#pragma unroll
        for (uint32_t c = 0; c < 2; ++c) {
            hf ht[2];
#pragma unroll
            for (uint32_t tq = 0; tq < 2; ++tq) {
                const uint32_t i0 = 4u * tq + c, i1 = i0 + 2u;
                hf par[2];
#pragma unroll
                for (uint32_t pa = 0; pa < 2; ++pa) {
                    hf pr[SN_KC];
#pragma unroll
                    for (uint32_t b = 0; b < SN_KC; ++b) {
                        const hf s0 = rv[2 * b + pa][i0] * rv[2 * b + pa][i0], s1 = rv[2 * b + pa][i1] * rv[2 * b + pa][i1];
                        pr[b] = s0 + s1;
                    }
                    par[pa] = tree(pr);
                }
                ht[tq] = par[0] + par[1];
            }
            h1[c] = ht[0] + ht[1];
        }
        const uint32_t other = other_half(bh(h1[0]) | (bh(h1[1]) << 16));
        const hf g0 = h1[0] + hb(other & 0xffffu), g1 = h1[1] + hb(other >> 16);
        const hf s = g0 + g1;
        const hf se = s + (hf)0.0001220703125f;
        const hf r = rsq16(se);
#pragma unroll
        for (uint32_t m = 0; m < SN_MT; ++m) {
            float n[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t k = 16u * m + 8u * hl + (uint32_t)i;
                const hf g = r * h16at(w, W_GAIN14 + 2u * k);
                n[i] = (float)(hf)(rv[m][i] * g);
                const hf a = rv[m][i] + h16at(w, W_TABLE16 + 2u * k);
                R[m].set(i, a);
            }
            Nb[m] = (i2v){(int)q4(n[0], n[1], n[2], n[3]), (int)q4(n[4], n[5], n[6], n[7])};
        }
    }
#pragma nounroll
    for (uint32_t br = 0; br < SN_BRANCHES; ++br) {
        const uint32_t w1 = W_W1_0 + SN_BRANCH * br, b1 = W_B1_0 + SN_BRANCH * br, w2 = W_W2_0 + SN_BRANCH * br;
        i2v Hb[2];
#pragma unroll
        for (uint32_t c = 0; c < 2; ++c) {
            A16 h = load16(w + b1 + 2u * (16u * c + 8u * hl));
#pragma unroll
            for (uint32_t kt = 0; kt < SN_MT; kt += 2)
                k32(h, tile(w, w1, c * (C / 16) + kt, col, hl), Nb[kt], tile(w, w1, c * (C / 16) + kt + 1, col, hl), Nb[kt + 1]);
            uint32_t b[2] = {0, 0};
#pragma unroll
            for (int i = 0; i < 8; ++i) b[i >> 2] |= (uint32_t)w[W_ACT + bh(h.get(i))] << (8 * (i & 3));
            Hb[c] = (i2v){(int)b[0], (int)b[1]};
        }
#pragma unroll
        for (uint32_t m = 0; m < SN_MT; ++m)
            k32(R[m], tile(w, w2, 2 * m, col, hl), Hb[0], tile(w, w2, 2 * m + 1, col, hl), Hb[1]);
    }

    // ---------------------------------------------------------------- outputs
    const uint32_t t = inter(16u * wv + col);
    const int tx = x0 + (int)(t & 7u), ty = y0 + (int)(t >> 3);
    const bool inside = tx >= 0 && ty >= 0 && tx < ex && ty < ey;
    i2v Tb[SN_MT];
    {
        const uint32_t o = (outOff + (uint32_t)(ty * ex + tx) * 32u) >> 2;
        const uint32_t ty2 = t >> 3, tx2 = t & 7u;
        const uint32_t mr = (ty2 >> 1) < 2u ? 2u * (tx2 >> 1) + (ty2 >> 1) : 8u + 2u * (tx2 >> 1) + (ty2 >> 1) - 2u;
        const uint32_t blk = 2u * (ty2 & 1u) + (tx2 & 1u);
        __syncthreads();   // every wave is done with XT (the merge rows reuse it)
#pragma unroll
        for (uint32_t m = 0; m < SN_MT; ++m) {
            Tb[m] = q8(R[m]);
            const uint32_t at = 4u * (m & 1u) + 2u * hl;
            if (inside) { arena[o + (m >> 1) * (plane >> 2) + at] = (uint32_t)Tb[m][0]; arena[o + (m >> 1) * (plane >> 2) + at + 1u] = (uint32_t)Tb[m][1]; }
            *(i2v*)(XT + mr * 4u * C + blk * C + 16u * m + 8u * hl) = Tb[m];
        }
    }
    // the C -> 16 head
    A16 hv = load16(w + W_BH + 16u * hl);
#pragma unroll
    for (uint32_t kt = 0; kt < SN_MT; kt += 2)
        k32(hv, tile(w, W_WH, kt, col, hl), Tb[kt], tile(w, W_WH, kt + 1, col, hl), Tb[kt + 1]);
    {   // the kernel-prediction filter (lanes 0-15; 16-31 hold head channels 8-15)
        hf l[12];
#pragma unroll
        for (int i = 0; i < 8; ++i) l[i] = hv.get(i);
        const uint32_t u0 = other_half(bh(l[0]) | (bh(l[1]) << 16)), u1 = other_half(bh(l[2]) | (bh(l[3]) << 16));
        l[8] = hb(u0 & 0xffffu); l[9] = hb(u0 >> 16); l[10] = hb(u1 & 0xffffu); l[11] = hb(u1 >> 16);
#define SN_MAX(a, b) __builtin_fmaxf16((a), (b))
        const hf ma = SN_MAX(SN_MAX(SN_MAX(l[0], l[2]), l[4]), l[6]), mb = SN_MAX(SN_MAX(SN_MAX(l[1], l[3]), l[5]), l[7]);
        const hf mx = SN_MAX(SN_MAX(ma, mb), l[8]);
        float e[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) { const hf d = l[i] - mx; e[i] = __builtin_amdgcn_exp2f((float)d * 1.4426950216293335f); }
        float s = (e[0] + e[1]) + 0.0f;
        s = s + (e[2] + e[3]);
        s = s + (e[4] + e[5]);
        s = s + (e[6] + e[7]);
        s = s + e[8];
        const hf rr = rcp16(s);
        float wg[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) wg[i] = r16(r16(e[i]) * (float)rr);
        // a token outside the image stores nothing: its taps only have to stay inside the texture buffer
        const int x = inside ? tx : 0, y = inside ? ty : 0;
        const int xm = x - 1 < 0 ? 1 : x - 1, xc = x < 0 ? 1 : (x >= ex ? ex - 2 : x), xp = x + 1 >= ex ? ex - 2 : x + 1;
        const int ym = y - 1 < 0 ? 1 : y - 1, yc = y < 0 ? 1 : (y >= ey ? ey - 2 : y), yp = y + 1 >= ey ? ey - 2 : y + 1;
        const int px[9] = {xm, xc, xp, xm, xc, xp, xm, xc, xp}, py[9] = {ym, ym, ym, yc, yc, yc, yp, yp, yp};
        float acc[3][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const hf* tv = texel(t88, ex, px[k], py[k]);
#pragma unroll
            for (int c = 0; c < 3; ++c) { const float f = (float)tv[c] * wg[k] + acc[c][k & 1]; acc[c][k & 1] = r16(f); }
        }
        const hf* t8 = texel(t88, ex, px[8], py[8]);
        float o112[3], o120[3];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float f0 = acc[c][0] + acc[c][1];
            const float f1 = (float)t8[c] * wg[8] + r16(f0);
            const float f = r16(f1);
            const float a = (float)sig(l[9 + c]) * f;
            const float ar = r16(a);
            const float b = f - ar;
            o120[c] = ar; o112[c] = r16(b);
        }
        if (hl == 0u && inside) {
            hf* p = s112 + 4 * (y * ex + x);
            hf* q = s120 + 4 * (y * ex + x);
#pragma unroll
            for (int c = 0; c < 3; ++c) { p[c] = (hf)o112[c]; q[c] = (hf)o120[c]; }
            p[3] = (hf)0.0f; q[3] = (hf)0.0f;
        }
        if (hl == 0u)
#pragma unroll
            for (int c = 0; c < 3; ++c) O112[t * 4u + c] = (hf)o112[c];
    }
    __syncthreads();
    {   // the 2x2 merge: merged^T (CX x 16 rows), channel tiles m = wv, wv + 4, ...
        const uint32_t mxx = col < 8u ? col >> 1 : (col - 8u) >> 1, mxy = col < 8u ? col & 1u : 2u + (col & 1u);
        const int X = x0 / 2 + (int)mxx, Y = y0 / 2 + (int)mxy;
        const uint32_t o = (mrgOff + (uint32_t)(Y * hx + X) * 32u) >> 2;
        for (uint32_t m = wv; m < CX / 16u; m += 4u) {
            A16 a = load16(w + W_BM + 2u * (16u * m + 8u * hl));
            for (uint32_t ks = 0; ks < C / 8u; ++ks) {
                const uint8_t* g = XT + col * 4u * C + 32u * ks + 8u * hl;
                k32(a, tile(w, W_WM, m * (4u * C / 16u) + 2u * ks, col, hl), op(g), tile(w, W_WM, m * (4u * C / 16u) + 2u * ks + 1u, col, hl), op(g + 16));
            }
            const i2v q = q8(a);
            if (X >= 0 && Y >= 0 && X < hx && Y < hy) {
                const uint32_t at = o + (m >> 1) * (lplane >> 2) + 4u * (m & 1u) + 2u * hl;
                arena[at] = (uint32_t)q[0];
                arena[at + 1u] = (uint32_t)q[1];
            }
        }
        if (wv == 3u && lane < 16u) {   // surface 128: 2x2 sums of 0.25 * surface 112, rightwards first, then downwards
            const uint32_t mx_ = lane & 3u, my_ = lane >> 2;
            const int x = x0 + 2 * (int)mx_, y = y0 + 2 * (int)my_;
            if (x >= 0 && y >= 0 && x < ex && y < ey) {
                const uint32_t t00 = (2u * my_) * 8u + 2u * mx_;
                hf* p = s128 + 4 * ((y >> 1) * hx + (x >> 1));
#pragma unroll
                for (uint32_t c = 0; c < 3; ++c) {
                    const hf a00 = O112[t00 * 4u + c] * (hf)0.25f, a01 = O112[(t00 + 1u) * 4u + c] * (hf)0.25f;
                    const hf a10 = O112[(t00 + 8u) * 4u + c] * (hf)0.25f, a11 = O112[(t00 + 9u) * 4u + c] * (hf)0.25f;
                    const hf b0 = a00 + a01, b1 = a10 + a11;
                    p[c] = b0 + b1;
                }
                p[3] = (hf)0.0f;
            }
        }
    }
}
} // namespace swin_native
