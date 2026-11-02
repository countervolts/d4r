// Shared K rounding/activation rules for dot2 and WMMA. Keep f16 operation
// boundaries: contracting GELU changes the reference's binary16 results.
#ifndef D4R_K_MATH
#define D4R_K_MATH
float16_t2 unpack(uint v) { return float16_t2(f16tof32(v & 65535), f16tof32(v >> 16)); }
uint pack(float16_t2 v) { return f32tof16(float(v.x)) | (f32tof16(float(v.y)) << 16); }
int mirror(int v, int n) { return min(abs(v), 2 * n - 2 - abs(v)); }
float16_t2 softmax_exp(float16_t2 s)
{
    const float16_t2 t = clamp(s, float16_t2(f16tof32(0xcd69), f16tof32(0xcd69)),
                                  float16_t2(f16tof32(0x4d69), f16tof32(0x4d69)));
    const float16_t2 v = mad(t * t, float16_t2(f16tof32(0x91d5), f16tof32(0x91d5)), float16_t2(1, 1));
    return float16_t2(exp2(float2(mad(t, v, float16_t2(f16tof32(0xc50d), f16tof32(0xc50d))))));
}
float16_t2 gelu(float16_t2 x)
{
    const float16_t2 c = clamp(x, float16_t2(-2, -2), float16_t2(2, 2));
    precise float16_t2 u = float16_t2(0.41216981, 0.41216981) - float16_t2(0.08108133, 0.08108133) * abs(c);
    precise float16_t2 y = float16_t2(0.5, 0.5) + c * u;
    return x * y;
}
#endif
