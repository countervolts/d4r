// Ray Reconstruction presets A - C (the CNN of nvngx_dlssd.dll 310.3) on RDNA4 through vkhip: one network stage
// (Enc0 - Enc4, Dec4 - Dec0, PDL) per code object, in place of engine/rr/cnn_enc.comp and cnn_dec.comp. The shaders'
// operations (K in their order, the 8-bit activations, NVIDIA's softmax, sigmoid and filters, engine/rr/cnn_common.glsl),
// every 16-deep conv step one binary16-accumulator WMMA (55 - 59 dB against ZLUDA's frames; RR3_EXACT: the shaders'
// f32 one, rounded to binary16 after every step, bit-exact). What differs is the schedule:
//   - a workgroup is 16 x TH pixels; its waves split the convolutions' (row, 16-channel tile) pairs R rows by NTW
//     tiles each, so every wave keeps R x NTW independent WMMA chains in flight and every weight fragment (one
//     16-byte load per lane) feeds R of them; the deep levels' few pixels get small tiles and many waves
//   - activations load 16 channels per lane (16-byte loads, the channel permutation in registers) and are stored
//     from the accumulators, two codes per 16-bit store (the encoders' max-pool from the LDS)
//   - the convolutions run transposed (conv1x1), so that results leave the accumulators 16 bytes at a time, bias
//     and ReLU applied on the way
// The kernel takes the address of the stage's parameter block (R3Engine's: NVIDIA's argument block, pointers as device
// addresses) and exports its workgroup size and tile height (rr3_threads, rr3_rows) for engine/vkhip/vkhip_net.cpp.
//
// Built by kernels/rr/build_vkhip_cnn.sh: -DRR3_NAME -DRR3_DEC (else an encoder) -DCIN -DC, the stage's variant
// (IN8, SPLITK, UPS, PDL, TS as for the shaders) and its schedule (TH, R, NTW).
#include <hip/hip_runtime.h>
#include <stdint.h>
#pragma clang fp contract(off)

typedef _Float16 half;
typedef half h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef uint32_t u4 __attribute__((ext_vector_type(4)));
#define D __device__ __forceinline__

#ifndef TS
#define TS 4
#endif
#ifndef R
#define R 2
#endif
#ifndef NTW
#define NTW 1
#endif
constexpr uint32_t NT = C / 16, NK = CIN / 16, NK2 = C / 16;
constexpr uint32_t TW = 16, PX = TW * TH, HW = TW + 2, HH = TH + 2, CINP = CIN + 8, CP = C + 8;
constexpr uint32_t ROWGROUPS = TH / R, WAVES = ROWGROUPS * (NT / NTW), THREADS = 32 * WAVES;
static_assert(TH % R == 0 && NT % NTW == 0, "the schedule must tile the stage");
static_assert(THREADS <= 1024, "too many waves");
#ifndef RR3_DEC
#endif
#ifdef PDL
constexpr uint32_t NL = C;
#define OUT 216
#else
constexpr uint32_t NL = 16;
#define OUT 192
#endif
constexpr uint32_t SIN_SIZE = HH * HW * CINP, SA_SIZE = PX * CP, S0_SIZE = SIN_SIZE > SA_SIZE ? SIN_SIZE : SA_SIZE;
constexpr uint32_t SL0 = S0_SIZE, SF0 = S0_SIZE + PX * NL, SM_SIZE = S0_SIZE + PX * NL + PX * 4;
static_assert(SM_SIZE * 2 <= 65536, "the tile does not fit the LDS");

extern "C" __device__ __constant__ uint32_t rr3_threads = THREADS;
extern "C" __device__ __constant__ uint32_t rr3_rows = TH;

template <class T> D T* G(uint64_t a) { return (T*)(__attribute__((address_space(1))) T*)a; }
struct Params {
    const uint32_t* p;
    D int i(uint32_t o) const { return int(p[o / 4]); }
    D uint64_t q(uint32_t o) const { return uint64_t(p[o / 4]) | (uint64_t(p[o / 4 + 1]) << 32); }
};

D half hb(uint32_t b) { return __builtin_bit_cast(half, uint16_t(b)); }
D uint32_t bh(half h) { return __builtin_bit_cast(uint16_t, h); }
D half hmax(half a, half b) { return __builtin_fmaxf16(a, b); }
D half hfma(half a, half b, half c) { return __builtin_fmaf16(a, b, c); }
// K slot / output column s of a 16-block holds channel P16(s) (engine/rr/cnn_pack.py); Q16 is its inverse
D uint32_t P16(uint32_t s) { return (s & 1u) | (((s >> 1u) & 1u) << 3u) | ((s >> 2u) << 1u); }
D uint32_t Q16(uint32_t v) { return (v & 1u) | (((v >> 3u) & 1u) << 1u) | (((v >> 1u) & 3u) << 2u); }
D uint32_t perm(uint32_t c) { return (c & ~15u) | P16(c & 15u); }
D uint32_t slot(uint32_t n, uint32_t T) { uint32_t j = n >> 3u, t = (n & 7u) >> 1u; return (j / T) * 8u * T + 2u * (T * t + j % T) + (n & 1u); }
D half dec8(uint32_t b) { return hb(b << 7u); }
typedef half h2 __attribute__((ext_vector_type(2)));
typedef half h4 __attribute__((ext_vector_type(4)));
// the pair of activations in bytes k, k + 1 (k even) of a 16-byte chunk, as two binary16
D h2 dec8x2(u4 v, uint32_t k) {
    const uint32_t w = v[k >> 2];
    const uint32_t b = __builtin_amdgcn_perm(0u, w, (k & 2u) ? 0x0c030c02u : 0x0c010c00u);
    return __builtin_bit_cast(h2, b << 7u);
}
D uint32_t enc8(half h) {
    const half e = hb(bh(h) & 0xfc00u);
    return (bh(hfma(e, half(0.0625), h)) >> 7u) & 0xffu;
}
// NVIDIA's hexp and __hdiv as engine/rr/cnn_common.glsl spells them
D half hexp(half h) {
    const float f = __builtin_fmaf(float(h), __builtin_bit_cast(float, 0x3fb8aa3bu), -0.0f);
    half r = half(__builtin_amdgcn_exp2f(f));
    const uint32_t b = bh(h);
    if (b == 0x1f79u || b == 0x25cfu) r = hfma(half(1.0), hb(0x9400u), r);
    if (b == 0xc13bu) r = hfma(half(1.0), hb(0x0400u), r);
    if (b == 0xc1efu) r = hfma(half(1.0), hb(0x0200u), r);
    return r;
}
D half hdiv(half a, half b) {
    const float fa = float(a), fb = float(b), rc = __builtin_amdgcn_rcpf(fb), q = fa * rc;
    half r = half(q);
    const half m = __builtin_fabsf16(r);
    if (m < hb(0x008fu) && half(0.0) < m) {
        const float e = __builtin_fmaf(-fb, q, fa), q2 = __builtin_fmaf(rc, e, q);
        r = half(q2);
    }
    return r;
}
D half sigm(half x) { return hdiv(half(1.0), half(1.0) + hexp(-x)); }
D void softmax9(const half* lg, half* w) {
    half m = lg[0];
#pragma unroll
    for (int i = 1; i < 9; ++i) m = half(__builtin_fmaxf(float(m), float(lg[i])));
    half e[9], s = half(0.0);
#pragma unroll
    for (int i = 0; i < 9; ++i) { e[i] = hexp(lg[i] - m); s = s + e[i]; }
#pragma unroll
    for (int i = 0; i < 9; ++i) w[i] = hdiv(e[i], s);
}
D void taps3(int v, int n, int* r) { r[0] = v < 1 ? 1 : v - 1; r[1] = v; r[2] = v + 2 > n ? n - 2 : v + 1; }

__shared__ __attribute__((aligned(16))) half sm[SM_SIZE];

D h8 lds8(uint32_t at) { return *(const h8*)&sm[at]; }
// the weights through a buffer descriptor: the per-lane offset stays in a VGPR, the rest of each address is an SGPR
// offset or an immediate (flat 64-bit addresses cost two VALU adds per load)
typedef __amdgpu_buffer_rsrc_t Rsrc;
D Rsrc rsrc(const half* p) { return __builtin_amdgcn_make_buffer_rsrc((void*)p, 0, 0x7fffffff, 0x31004000); }
D h8 wload(Rsrc r, uint32_t lane16, uint32_t element) {
    return __builtin_bit_cast(h8, __builtin_amdgcn_raw_buffer_load_b128(r, lane16, element * 2u, 0));
}
D h8 mma(h8 acc, h8 a, h8 b) {
#ifndef RR3_EXACT
    // the binary16 accumulator WMMA: no conversions around every step. Not bit-exact to NVIDIA's (its sum is rounded
    // once, not through f32): 55 - 59 dB against ZLUDA's frames; RR3_EXACT: the shaders' f32 accumulator, bit-exact
    return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, acc);
#else
    f8 c;
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = float(acc[i]);
    const f8 r = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    h8 o;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = half(r[i]);
    return o;
#endif
}
D void barrier() { __syncthreads(); }

// The convolutions are computed transposed: the weights ([K-chunk][..][cout][16] as cnn_pack.py stores them, an output
// channel's 16 k contiguous) are the A operand and a row's 16 pixels (16 channels contiguous in the LDS) the B
// operand, so that a result's lane holds eight consecutive output channels of one pixel: lane l, pixel l & 15,
// channels 8 (l >> 4) + i of the tile.
//
// A 1x1 conv over sA for the rows row0 .. row0 + RR - 1 and the tiles nt0 .. nt0 + NN - 1. SPLITK: one chain per
// 32-channel slice from zero, the slices added in binary16.
template <uint32_t RR, uint32_t NN>
D void conv1x1(h8 (&out)[RR][NN], const half* w, uint32_t cout, uint32_t row0, uint32_t nt0) {
    const uint32_t lane = threadIdx.x & 31u, n = lane & 15u, hl = lane >> 4, lane16 = (n * 16u + 8u * hl) * 2u;
    const Rsrc wr = rsrc(w);
#ifdef SPLITK
    constexpr uint32_t SLICE = 2;
#else
    constexpr uint32_t SLICE = NK2;
#endif
#pragma unroll
    for (uint32_t sl = 0; sl < NK2 / SLICE; ++sl) {
        h8 acc[RR][NN];
#pragma unroll
        for (uint32_t r = 0; r < RR; ++r)
#pragma unroll
            for (uint32_t j = 0; j < NN; ++j) acc[r][j] = h8(half(0.0));
#pragma unroll
        for (uint32_t kc = sl * SLICE; kc < sl * SLICE + SLICE; ++kc) {
            h8 a[RR], b[NN];
#pragma unroll
            for (uint32_t r = 0; r < RR; ++r) a[r] = lds8(((row0 + r) * TW + n) * CP + kc * 16u + 8u * hl);
#pragma unroll
            for (uint32_t j = 0; j < NN; ++j) b[j] = wload(wr, lane16, (kc * cout + (nt0 + j) * 16u) * 16u);
#pragma unroll
            for (uint32_t r = 0; r < RR; ++r)
#pragma unroll
                for (uint32_t j = 0; j < NN; ++j) acc[r][j] = mma(acc[r][j], b[j], a[r]);
        }
#pragma unroll
        for (uint32_t r = 0; r < RR; ++r)
#pragma unroll
            for (uint32_t j = 0; j < NN; ++j) out[r][j] = sl == 0 ? acc[r][j] : out[r][j] + acc[r][j];
    }
}

// the biases of a lane's eight channels: column c of the LDS is logical channel perm(c), whose bias is at slot 4
D h8 bias8(const half* bias, uint32_t col0) {
    h8 b;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) b[i] = bias[slot(perm(col0 + i), 4u)];
    return b;
}
// An 8-bit tensor (its pointer null: none): H W, then channel and row strides in bytes
struct Bytes { uint8_t* p; int H, W, cs, rs; };
// two activations (LDS columns c, c + 1, c even) as their 8-bit codes, the first in the low byte
D uint32_t enc8x2(h2 h) {
    const h2 e = __builtin_bit_cast(h2, __builtin_bit_cast(uint32_t, h) & 0xfc00fc00u);
    const uint32_t r = __builtin_bit_cast(uint32_t, __builtin_elementwise_fma(e, h2(half(0.0625)), h)) >> 7;
    return (r & 0xffu) | ((r >> 8) & 0xff00u);
}
// a lane's eight values of one pixel to a tensor: columns c, c + 1 (c even) are logical channels n, n + 1 (P16 keeps
// pairs together), whose slots are adjacent too, so two codes per 16-bit store
D void put8(const Bytes& t, int y, int x, uint32_t col0, h8 v, uint32_t T) {
    if (y >= t.H || x >= t.W) return;
    uint8_t* at = t.p + y * t.rs + x * t.cs;
#pragma unroll
    for (uint32_t i = 0; i < 8; i += 2)
        *(uint16_t*)&at[slot(perm(col0 + i), T)] = uint16_t(enc8x2(h2{v[i], v[i + 1]}));
}
// The wave's results (rows row0 + r, tiles nt0 + j) to sA, bias added and (relu) ReLU'd, one 16-byte store each, and
// to `out` as 8-bit codes
D void storeA(const h8 (&v)[R][NTW], const half* bias, uint32_t row0, uint32_t nt0, bool relu, int y0, int x0,
              const Bytes& out = {}) {
    const uint32_t lane = threadIdx.x & 31u, n = lane & 15u, hl = lane >> 4;
#pragma unroll
    for (uint32_t j = 0; j < NTW; ++j) {
        const uint32_t col0 = (nt0 + j) * 16u + 8u * hl;
        const h8 b = bias8(bias, col0);
        h8 x[R];
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) {
            x[r] = v[r][j] + b;
            if (relu) x[r] = __builtin_elementwise_max(x[r], h8(half(0.0)));
            *(h8*)&sm[((row0 + r) * TW + n) * CP + col0] = x[r];
            if (out.p) put8(out, y0 + int(row0 + r), x0 + int(n), col0, x[r], TS);
        }
    }
}

// a C -> C 1x1 conv of sA in place: every wave's results first, then the stores
D void conv1x1All(const half* w, const half* bias, bool relu, uint32_t row0, uint32_t nt0, int y0, int x0, const Bytes& out = {}) {
    h8 v[R][NTW];
    conv1x1<R, NTW>(v, w, C, row0, nt0);
    barrier();
    storeA(v, bias, row0, nt0, relu, y0, x0, out);
    barrier();
}

// the head (16 outputs) of every row, into sm[SL0 + pixel * 16 + column], without its bias
D void head16(const half* w, uint32_t wave) {
    const uint32_t lane = threadIdx.x & 31u, n = lane & 15u, hl = lane >> 4;
    for (uint32_t row = wave; row < TH; row += WAVES) {
        h8 v[1][1];
        conv1x1<1, 1>(v, w, 16u, row, 0u);
        *(h8*)&sm[SL0 + (row * TW + n) * 16u + 8u * hl] = v[0][0];
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) RR3_NAME(const uint32_t* param) {
    const Params P{param};
    const uint32_t tid = threadIdx.x, wave = __builtin_amdgcn_readfirstlane(tid >> 5), lane = tid & 31u;
    const uint32_t n = lane & 15u, hl = lane >> 4;
    const int y0 = int(blockIdx.y) * int(TH), x0 = int(blockIdx.x) * int(TW);
    // this wave's share of the convolutions: rows row0 .. row0 + R - 1, tiles nt0 .. nt0 + NTW - 1
    const uint32_t row0 = (wave % ROWGROUPS) * R, nt0 = (wave / ROWGROUPS) * NTW;
#ifdef RR3_DEC
    const int H = P.i(OUT + 8), W = P.i(OUT + 12);
#else
    const int H = P.i(8), W = P.i(12);
#endif

    // ---- the input tile with its halo, zero outside: 16 channels per lane, LDS slot s from stored position P16(s)
    {
        const int INC = P.i(16), INRS = P.i(20);
#if defined(RR3_DEC) && defined(UPS)
        const int LH = P.i(8), LW = P.i(12);
        const uint8_t* low = G<const uint8_t>(P.q(0));
        const uint8_t* skp = G<const uint8_t>(P.q(32));
        const int SC = P.i(48), SRS = P.i(52);
#elif defined(IN8) || defined(RR3_DEC)
        const uint8_t* in = G<const uint8_t>(P.q(0));
#else
        const half* in = G<const half>(P.q(0));
#endif
        for (uint32_t i = tid; i < HH * HW * NK; i += THREADS) {
            const uint32_t chunk = i % NK, q = i / NK, hx = q % HW, hy = q / HW;
            const int y = y0 - 1 + int(hy), x = x0 - 1 + int(hx);
            h8 lo = h8(half(0.0)), hi = h8(half(0.0));
            if (y >= 0 && y < H && x >= 0 && x < W) {
                half v[16];
#if defined(RR3_DEC) && defined(UPS)
                const int iy = y >> 1, ix = x >> 1;
                const int ny = min(max(iy + ((y & 1) == 0 ? -1 : 1), 0), LH - 1);
                const int nx = min(max(ix + ((x & 1) == 0 ? -1 : 1), 0), LW - 1);
                const int c0 = int(chunk * 16u);
                const u4 A = *(const u4*)&low[iy * INRS + ix * INC + c0], B = *(const u4*)&low[iy * INRS + nx * INC + c0];
                const u4 A2 = *(const u4*)&low[ny * INRS + ix * INC + c0], B2 = *(const u4*)&low[ny * INRS + nx * INC + c0];
                const u4 S = *(const u4*)&skp[y * SRS + x * SC + c0];
                // slots s, s + 1 are the stored bytes P16(s), P16(s) + 1: two channels per packed binary16 operation
                const h2 q = h2(half(0.75)), o = h2(half(0.25));
#pragma unroll
                for (uint32_t s = 0; s < 16; s += 2) {
                    const uint32_t k = P16(s);
                    const h2 m0 = q * dec8x2(A, k), m1 = q * dec8x2(A2, k);
                    const h2 h0 = __builtin_elementwise_fma(o, dec8x2(B, k), m0), h1 = __builtin_elementwise_fma(o, dec8x2(B2, k), m1);
                    const h2 up = __builtin_elementwise_fma(o, h1, q * h0);
                    const h2 sum = dec8x2(S, k) + up;
                    v[s] = sum[0]; v[s + 1] = sum[1];
                }
#elif defined(IN8) || defined(RR3_DEC)
                const u4 A = *(const u4*)&in[y * INRS + x * INC + int(chunk * 16u)];
#pragma unroll
                for (uint32_t s = 0; s < 16; s += 2) { const h2 p = dec8x2(A, P16(s)); v[s] = p[0]; v[s + 1] = p[1]; }
#else
                const h8 A = *(const h8*)&in[y * INRS + x * INC + int(chunk * 16u)];
                const h8 B = *(const h8*)&in[y * INRS + x * INC + int(chunk * 16u) + 8];
#pragma unroll
                for (uint32_t s = 0; s < 16; ++s) { const uint32_t k = P16(s); v[s] = k < 8 ? A[k] : B[k - 8]; }
#endif
#pragma unroll
                for (uint32_t s = 0; s < 8; ++s) { lo[s] = v[s]; hi[s] = v[s + 8]; }
            }
            const uint32_t at = (hy * HW + hx) * CINP + chunk * 16u;
            *(h8*)&sm[at] = lo;
            *(h8*)&sm[at + 8] = hi;
        }
    }
    barrier();

    // ---- conv 3x3: K over the chunks, within a chunk tap by tap (the shaders' order)
    {
#ifdef RR3_DEC
        const half* w1 = G<const half>(P.q(64));
        const half* b1 = G<const half>(P.q(128));
#else
        const half* w1 = G<const half>(P.q(24));
        const half* b1 = G<const half>(P.q(72));
#endif
        const Rsrc wr = rsrc(w1);
        const uint32_t lane16 = (n * 16u + 8u * hl) * 2u;
        h8 acc[R][NTW];
#pragma unroll
        for (uint32_t r = 0; r < R; ++r)
#pragma unroll
            for (uint32_t j = 0; j < NTW; ++j) acc[r][j] = h8(half(0.0));
#pragma unroll 1
        for (uint32_t kc = 0; kc < NK; ++kc) {   // rolled: unrolled, LLVM hoists every chunk's weights and spills
#pragma unroll
            for (uint32_t t = 0; t < 9; ++t) {
                const uint32_t dy = t / 3u, dx = t % 3u;
                h8 a[R], b[NTW];
#pragma unroll
                for (uint32_t r = 0; r < R; ++r) a[r] = lds8(((row0 + r + dy) * HW + n + dx) * CINP + kc * 16u + 8u * hl);
#pragma unroll
                for (uint32_t j = 0; j < NTW; ++j) b[j] = wload(wr, lane16, ((kc * 9u + t) * C + (nt0 + j) * 16u) * 16u);
#pragma unroll
                for (uint32_t r = 0; r < R; ++r)
#pragma unroll
                    for (uint32_t j = 0; j < NTW; ++j) acc[r][j] = mma(acc[r][j], b[j], a[r]);
            }
        }
        barrier();   // sA overlays the input tile
#ifdef RR3_DEC
        storeA(acc, b1, row0, nt0, true, y0, x0);
#else
        // with the skip tensor
        storeA(acc, b1, row0, nt0, true, y0, x0, Bytes{G<uint8_t>(P.q(120)), P.i(128), P.i(132), P.i(136), P.i(140)});
#endif
        barrier();
#ifndef RR3_DEC
        {   // the skip tensor's 2x2 max-pool, from the LDS as cnn_enc.comp
            uint8_t* pool = G<uint8_t>(P.q(152));
            const int PH = P.i(160), PW = P.i(164), PC = P.i(168), PRS = P.i(172);
            for (uint32_t i = tid; i < (PX / 4u) * C; i += THREADS) {
                const uint32_t col = i % C, q = i / C, qx = q % (TW / 2u), qy = q / (TW / 2u), px = qy * 2u * TW + qx * 2u;
                const uint32_t m = max(max(enc8(sm[px * CP + col]), enc8(sm[(px + 1u) * CP + col])),
                                       max(enc8(sm[(px + TW) * CP + col]), enc8(sm[(px + TW + 1u) * CP + col])));
                const int y = y0 / 2 + int(qy), x = x0 / 2 + int(qx);
                if (y < PH && x < PW) pool[y * PRS + x * PC + int(slot(perm(col), TS))] = uint8_t(m);
            }
        }
#endif
    }

#ifndef RR3_DEC
    // ---- conv 1x1 C -> C, then the head
    conv1x1All(G<const half>(P.q(40)), G<const half>(P.q(88)), true, row0, nt0, y0, x0);
    head16(G<const half>(P.q(56)), wave);
    barrier();

    // ---- the filter, one pixel per thread
    const int IH = P.i(208), IW = P.i(212), IS = P.i(216);
    const bool down = P.q(192) != 0;
    const half* b3 = G<const half>(P.q(104));
    for (uint32_t px = tid; px < PX; px += THREADS) {
        const int row = y0 + int(px / TW), col = x0 + int(px % TW);
        const bool inside = row < IH && col < IW;
        half lg[12];
#pragma unroll
        for (uint32_t c = 0; c < 16; ++c) {
            const uint32_t s = slot(P16(c), 2u);
            if (s < 12u) lg[s] = sm[SL0 + px * 16u + c] + b3[s];
        }
        half fr = half(0.0), fg = half(0.0), fb = half(0.0);
        if (inside) {
            half w[9];
            softmax9(lg, w);
            const half* img = G<const half>(P.q(200));
            int rr[3], cc[3];
            taps3(row, IH, rr);
            taps3(col, IW, cc);
#pragma unroll
            for (uint32_t t = 0; t < 9; ++t) {
                const h4 p = *(const h4*)&img[(rr[t / 3u] * IW + cc[t % 3u]) * IS];
                fr = hfma(p[0], w[t], fr);
                fg = hfma(p[1], w[t], fg);
                fb = hfma(p[2], w[t], fb);
            }
            half* dst = G<half>(P.q(184));
            if (down) {
                const half f3[3] = {fr, fg, fb};
                half hp[3], lo[3];
#pragma unroll
                for (uint32_t c = 0; c < 3; ++c) {
                    hp[c] = sigm(lg[9u + c]) * f3[c];
                    lo[c] = f3[c] + hp[c] * half(-1.0);
                }
                const int o = (row * IW + col) * 8;
                dst[o] = lo[0]; dst[o + 1] = lo[1]; dst[o + 2] = lo[2]; dst[o + 3] = hp[0]; dst[o + 4] = hp[1]; dst[o + 5] = hp[2];
                fr = lo[0]; fg = lo[1]; fb = lo[2];
            } else {
                const int o = (row * IW + col) * 4;
                dst[o] = fr; dst[o + 1] = fg; dst[o + 2] = fb; dst[o + 3] = half(0.0);
            }
        }
        if (down) {
            sm[SF0 + px * 4u] = fr * half(0.25);
            sm[SF0 + px * 4u + 1u] = fg * half(0.25);
            sm[SF0 + px * 4u + 2u] = fb * half(0.25);
        }
    }
    barrier();
    if (down) {
        half* nxt = G<half>(P.q(192));
        for (uint32_t px = tid; px < PX; px += THREADS) {
            const int row = y0 + int(px / TW), col = x0 + int(px % TW);
            if (!(row < IH && col < IW) || (px & 1u) || ((px / TW) & 1u)) continue;
            const int o = ((IW / 2) * (row / 2) + col / 2) * 4;
#pragma unroll
            for (uint32_t c = 0; c < 3; ++c) {
                const half a = sm[SF0 + px * 4u + c] + sm[SF0 + (px + 1u) * 4u + c];
                const half b = sm[SF0 + (px + TW) * 4u + c] + sm[SF0 + (px + TW + 1u) * 4u + c];
                nxt[o + int(c)] = a + b;
            }
            nxt[o + 3] = half(0.0);
        }
    }
#else
    // ---- conv 1x1, the level's features (PDL: not stored)
#ifdef PDL
    conv1x1All(G<const half>(P.q(80)), G<const half>(P.q(144)), true, row0, nt0, y0, x0);
#else
    conv1x1All(G<const half>(P.q(80)), G<const half>(P.q(144)), true, row0, nt0, y0, x0,
               Bytes{G<uint8_t>(P.q(OUT)), H, W, P.i(OUT + 16), P.i(OUT + 20)});
#endif
#ifdef PDL
    // the third conv is the head, its bias added, no ReLU, straight from the accumulators: stored slots 0 - 11 are
    // the logits, 16 - 23 go to 216 (8 halves per pixel)
    {
        h8 v[R][NTW];
        conv1x1<R, NTW>(v, G<const half>(P.q(96)), C, row0, nt0);
        const half* bias = G<const half>(P.q(160));
        half* dst = G<half>(P.q(216));
#pragma unroll
        for (uint32_t j = 0; j < NTW; ++j)
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t s = slot(perm((nt0 + j) * 16u + 8u * hl + i), 4u);
                const half b = bias[s];
#pragma unroll
                for (uint32_t r = 0; r < R; ++r) {
                    const uint32_t px = (row0 + r) * TW + n;
                    const int y = y0 + int(row0 + r), x = x0 + int(n);
                    const half o = v[r][j][i] + b;
                    if (s < 16u) sm[SL0 + px * NL + s] = o;
                    else if (s < 24u && y < H && x < W) dst[(y * W + x) * 8 + int(s) - 16] = o;
                }
            }
    }
    barrier();
#else
    conv1x1All(G<const half>(P.q(96)), G<const half>(P.q(160)), true, row0, nt0, y0, x0);
    head16(G<const half>(P.q(112)), wave);
    barrier();
#endif

    // ---- the filter, one pixel per thread
    const int IH = P.i(272), IW = P.i(276), IS = P.i(280);
#ifndef UPS
    // Dec4: the encoders' filter (of the image at 256, split when 248 is set)
    const bool down = P.q(248) != 0;
    const half* b4 = G<const half>(P.q(176));
    for (uint32_t px = tid; px < PX; px += THREADS) {
        const int row = y0 + int(px / TW), col = x0 + int(px % TW);
        const bool inside = row < IH && col < IW;
        half lg[12];
#pragma unroll
        for (uint32_t c = 0; c < 16; ++c) {
            const uint32_t s = slot(P16(c), 2u);
            if (s < 12u) lg[s] = sm[SL0 + px * NL + c] + b4[s];
        }
        half fr = half(0.0), fg = half(0.0), fb = half(0.0);
        if (inside) {
            half w[9];
            softmax9(lg, w);
            const half* img = G<const half>(P.q(256));
            int rr[3], cc[3];
            taps3(row, IH, rr);
            taps3(col, IW, cc);
#pragma unroll
            for (uint32_t t = 0; t < 9; ++t) {
                const h4 p = *(const h4*)&img[(rr[t / 3u] * IW + cc[t % 3u]) * IS];
                fr = hfma(p[0], w[t], fr);
                fg = hfma(p[1], w[t], fg);
                fb = hfma(p[2], w[t], fb);
            }
            half* dst = G<half>(P.q(240));
            if (down) {
                const half f3[3] = {fr, fg, fb};
                half hp[3], lo[3];
#pragma unroll
                for (uint32_t c = 0; c < 3; ++c) {
                    hp[c] = sigm(lg[9u + c]) * f3[c];
                    lo[c] = f3[c] + hp[c] * half(-1.0);
                }
                const int o = (row * IW + col) * 8;
                dst[o] = lo[0]; dst[o + 1] = lo[1]; dst[o + 2] = lo[2]; dst[o + 3] = hp[0]; dst[o + 4] = hp[1]; dst[o + 5] = hp[2];
                fr = lo[0]; fg = lo[1]; fb = lo[2];
            } else {
                const int o = (row * IW + col) * 4;
                dst[o] = fr; dst[o + 1] = fg; dst[o + 2] = fb; dst[o + 3] = half(0.0);
            }
        }
        if (down) {
            sm[SF0 + px * 4u] = fr * half(0.25);
            sm[SF0 + px * 4u + 1u] = fg * half(0.25);
            sm[SF0 + px * 4u + 2u] = fb * half(0.25);
        }
    }
    barrier();
    if (down) {
        half* nxt = G<half>(P.q(248));
        for (uint32_t px = tid; px < PX; px += THREADS) {
            const int row = y0 + int(px / TW), col = x0 + int(px % TW);
            if (!(row < IH && col < IW) || (px & 1u) || ((px / TW) & 1u)) continue;
            const int o = ((IW / 2) * (row / 2) + col / 2) * 4;
#pragma unroll
            for (uint32_t c = 0; c < 3; ++c) {
                const half a = sm[SF0 + px * 4u + c] + sm[SF0 + (px + 1u) * 4u + c];
                const half b = sm[SF0 + (px + TW) * 4u + c] + sm[SF0 + (px + TW + 1u) * 4u + c];
                nxt[o + int(c)] = a + b;
            }
            nxt[o + 3] = half(0.0);
        }
    }
#else
    const half* low = G<const half>(P.q(256));
    const half* det = G<const half>(P.q(264));
    half* dst = G<half>(P.q(240));
#ifndef PDL
    const half* b4 = G<const half>(P.q(176));
#endif
    const int LW = IW / 2;
    for (uint32_t px = tid; px < PX; px += THREADS) {
        const int row = y0 + int(px / TW), col = x0 + int(px % TW);
        if (!(row < IH && col < IW)) continue;
        half lg[12];
#ifdef PDL
#pragma unroll
        for (uint32_t s = 0; s < 12; ++s) lg[s] = sm[SL0 + px * NL + s];
#else
#pragma unroll
        for (uint32_t c = 0; c < 16; ++c) {
            const uint32_t s = slot(P16(c), 2u);
            if (s < 12u) lg[s] = sm[SL0 + px * NL + c] + b4[s];
        }
#endif
        half w[9];
        softmax9(lg, w);
        int rr[3], cc[3];
        taps3(row, IH, rr);
        taps3(col, IW, cc);
        half f[3] = {half(0.0), half(0.0), half(0.0)};
#pragma unroll
        for (uint32_t t = 0; t < 9; ++t) {
            // the level below's image upsampled bilinearly at the tap
            const int r = rr[t / 3u], c = cc[t % 3u];
            const int ri0 = (r - 1) / 2, ci0 = (c - 1) / 2;
            const int ri1 = ri0 + ((r != 0 && r != IH - 1) ? 1 : 0), ci1 = ci0 + ((c != 0 && c != IW - 1) ? 1 : 0);
            const half wx0 = (c % 2) == 1 ? half(0.75) : half(0.25), wx1 = (c % 2) == 1 ? half(0.25) : half(0.75);
            const half wy0 = (r % 2) == 1 ? half(0.75) : half(0.25), wy1 = (r % 2) == 1 ? half(0.25) : half(0.75);
            const half wA = wx0 * wy0, wB = wx1 * wy0, wC = wx0 * wy1, wD = wx1 * wy1;
            const h4 pA = *(const h4*)&low[(ri0 * LW + ci0) * 4], pB = *(const h4*)&low[(ri0 * LW + ci1) * 4];
            const h4 pC = *(const h4*)&low[(ri1 * LW + ci0) * 4], pD = *(const h4*)&low[(ri1 * LW + ci1) * 4];
#pragma unroll
            for (uint32_t ch = 0; ch < 3; ++ch) {
                const half u0 = pD[ch] * wD;
                const half u1 = hfma(pC[ch], wC, u0);
                const half u2 = hfma(pB[ch], wB, u1);
                const half u3 = hfma(pA[ch], wA, u2);
                f[ch] = hfma(u3, w[t], f[ch]);
            }
        }
        const int od = (row * IW + col) * 8;
        const half al = sigm(lg[9]);
        const half om = hfma(half(-1.0), al, half(1.0));
        const int o = (row * IW + col) * 4;
#pragma unroll
        for (uint32_t ch = 0; ch < 3; ++ch) {
            const half fa = f[ch] * al;
            const half v = hfma(det[od + int(ch)], om, fa);
            const half hi = det[od + 3 + int(ch)];
            const half vh = v + hi;
            dst[o + int(ch)] = __builtin_isnan(hi) ? v : vh;
        }
        dst[o + 3] = half(0.0);
    }
#endif
#endif
}
