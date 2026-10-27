// The CUDA backend gives NGX's kernels canonical binary16 colour and motion: the game's textures blitted to
// RGBA16F / RG16F, a conversion RADV's blit rounds toward zero. The engine reads the game's textures directly, so
// it applies the same conversion to every colour and motion texel it fetches. Values already exact in binary16
// (RGBA16F, RG16F, R11G11B10, unchanged) pass through. Integer and power-of-two operations only: a float
// f32 -> f16 -> f32 round trip is folded away by the compiler (see round_half.glsl).
#ifndef D4R_INPUT_QUANT_GL
#define D4R_INPUT_QUANT_GL
float canonicalHalf(float v) {
    const uint bits = floatBitsToUint(v), magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x47800000u)       // at least 65536: the largest finite half, toward zero; Inf and NaN unchanged
        return magnitude >= 0x7f800000u ? v : uintBitsToFloat((bits & 0x80000000u) | 0x477fe000u);
    if (magnitude < 0x38800000u)        // below 2^-14: a binary16 subnormal, a multiple of 2^-24
        return trunc(v * 16777216.0) * (1.0 / 16777216.0);
    return uintBitsToFloat(bits & 0xffffe000u);
}
vec2 canonicalHalf(vec2 v) { return vec2(canonicalHalf(v.x), canonicalHalf(v.y)); }
vec3 canonicalHalf(vec3 v) { return vec3(canonicalHalf(v.x), canonicalHalf(v.y), canonicalHalf(v.z)); }
vec4 canonicalHalf(vec4 v) { return vec4(canonicalHalf(v.x), canonicalHalf(v.y), canonicalHalf(v.z), canonicalHalf(v.w)); }
#endif
