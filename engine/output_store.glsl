// The direct final stores. OUTPUT_STORE_F32 (K's default stage) hands the f32 result to the format's conversion, as
// NVIDIA's formatted store does.
// Otherwise they keep the CUDA backend's half store + blit result, where RADV's rgba16f image stores truncate before
// the blit's conversion: they round to binary16 toward zero and leave the format conversion to the same hardware. For
// the UNORM formats (RGB10A2, RGBA8) values below binary16's normal range quantize to zero and overflow saturates, so
// a normal-range mantissa mask suffices; R11G11B10 keeps small values (and its 6/5-bit mantissas round the half), so
// it takes the full conversion.
// K's copy path (Engine::setFrame) stores to an image of the destination's own bit layout and copies it raw, because a
// blit narrows to half whatever its source format. The formats below that the hardware cannot store take a layout
// with the same bits: BGRA8 is RGBA8 with the channels swapped, and E5B9G9R9 is encoded here into R32UI.
#ifndef OUTPUT_RGB10
#define OUTPUT_RGB10 0
#endif
#ifndef OUTPUT_RGBA8
#define OUTPUT_RGBA8 0
#endif
#ifndef OUTPUT_R11G11B10
#define OUTPUT_R11G11B10 0
#endif
#ifndef OUTPUT_RGBA32F
#define OUTPUT_RGBA32F 0
#endif
#ifndef OUTPUT_BGRA8
#define OUTPUT_BGRA8 0
#endif
#ifndef OUTPUT_E5B9G9R9
#define OUTPUT_E5B9G9R9 0
#endif
#define FINAL_OUTPUT_IMAGE image2D
#if OUTPUT_RGB10 || OUTPUT_RGBA8
#if OUTPUT_RGB10
#define FINAL_OUTPUT_FORMAT rgb10_a2
#else
#define FINAL_OUTPUT_FORMAT rgba8
#endif
#ifdef OUTPUT_STORE_F32
#define finalOutputValue(value) (value)
#else
vec4 finalOutputValue(vec4 value) {
    vec4 rounded = uintBitsToFloat(floatBitsToUint(value) & 0xffffe000u);
    return mix(rounded, value, isnan(value));
}
#endif
#elif OUTPUT_R11G11B10
#include "input_quant.glsl"
#define FINAL_OUTPUT_FORMAT r11f_g11f_b10f
// The blit turns every NaN into the format's canonical NaN; a direct store would keep the payload's top bits.
vec4 finalOutputValue(vec4 value) {
#ifdef OUTPUT_STORE_F32
    return mix(value, vec4(uintBitsToFloat(0x7fc00000u)), isnan(value));
#else
    return mix(canonicalHalf(value), vec4(uintBitsToFloat(0x7fc00000u)), isnan(value));
#endif
}
#elif OUTPUT_RGBA32F
#define FINAL_OUTPUT_FORMAT rgba32f
#define finalOutputValue(value) (value)
#elif OUTPUT_BGRA8
#define FINAL_OUTPUT_FORMAT rgba8
#define finalOutputValue(value) ((value).bgra)
#elif OUTPUT_E5B9G9R9
#define FINAL_OUTPUT_FORMAT r32ui
#undef FINAL_OUTPUT_IMAGE
#define FINAL_OUTPUT_IMAGE uimage2D
// The Vulkan specification's encoding, rounding to nearest. NaN and negative values store zero, overflow the maximum.
uvec4 finalOutputValue(vec4 value) {
    const float maxValue = float(511u << 16u); // (2^9 - 1) / 2^9 * 2^(31 - 15)
    const vec3 c = min(mix(max(value.rgb, vec3(0.0)), vec3(0.0), isnan(value.rgb)), vec3(maxValue));
    const float top = max(c.r, max(c.g, c.b));
    // floor(log2(top)), from the exponent bits so it has no rounding of its own; smaller values use the lowest exponent
    const int floorLog2 = top > 0.0 ? int((floatBitsToUint(top) >> 23u) & 255u) - 127 : -127;
    int e = max(-16, floorLog2) + 16;
    float scale = ldexp(1.0, 9 + 15 - e);
    if (uint(floor(top * scale + 0.5)) == 512u) { e += 1; scale *= 0.5; }
    const uvec3 m = uvec3(floor(c * scale + 0.5));
    return uvec4(m.r | (m.g << 9u) | (m.b << 18u) | (uint(e) << 27u), 0u, 0u, 0u);
}
#else
#define FINAL_OUTPUT_FORMAT rgba16f
#define finalOutputValue(value) (value)
#endif
