// Vector-ALU issue rate: independent chains per thread, so the result is throughput, not latency.
// Reported as warp/wave instructions per second (one instruction for 32 lanes); same source for HIP and CUDA.
#define CH 8
#define BODY(T, OP) T v[CH]; for (int k = 0; k < CH; ++k) v[k] = (T)(threadIdx.x + k); \
    for (int i = 0; i < it; ++i) { _Pragma("unroll") for (int k = 0; k < CH; ++k) OP; } \
    T s = 0; for (int k = 0; k < CH; ++k) s += v[k]; if (s == (T)1234) o[threadIdx.x] = (float)s;
KG void k_fma(float* o, int it, float a, float b) { BODY(float, v[k] = fmaf(v[k], a, b)) }
KG void k_iadd(float* o, int it, float a, float b) { const unsigned x = (unsigned)(a * 1000.f), y = (unsigned)(b * 1e6f); BODY(unsigned, v[k] = (v[k] ^ x) + (v[k] >> 3) + y) }
KG void k_h2fma(float* o, int it, float a, float b) {
    half2 v[CH]; half2 ha = __floats2half2_rn(a, a), hb = __floats2half2_rn(b, b);
    for (int k = 0; k < CH; ++k) v[k] = __floats2half2_rn((float)threadIdx.x, (float)k);
    for (int i = 0; i < it; ++i) { _Pragma("unroll") for (int k = 0; k < CH; ++k) v[k] = __hfma2(v[k], ha, hb); }
    float s = 0; for (int k = 0; k < CH; ++k) s += __low2float(v[k]); if (s == 1234.5f) o[threadIdx.x] = s; }
