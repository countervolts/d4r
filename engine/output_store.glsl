// RADV's rgba16f image stores truncate before the copy path's conversion (a blit to the game's texture).
// The direct stores keep that half-store + blit result: they round to binary16 toward zero and leave the
// format conversion to the same hardware. For the UNORM formats (RGB10A2, RGBA8) values below binary16's
// normal range quantize to zero and overflow saturates, so a normal-range mantissa mask suffices; R11G11B10
// keeps small values (and its 6/5-bit mantissas round the half), so it takes the full conversion.
#ifndef OUTPUT_RGB10
#define OUTPUT_RGB10 0
#endif
#ifndef OUTPUT_RGBA8
#define OUTPUT_RGBA8 0
#endif
#ifndef OUTPUT_R11G11B10
#define OUTPUT_R11G11B10 0
#endif
#if OUTPUT_RGB10 || OUTPUT_RGBA8
#if OUTPUT_RGB10
#define FINAL_OUTPUT_FORMAT rgb10_a2
#else
#define FINAL_OUTPUT_FORMAT rgba8
#endif
vec4 finalOutputValue(vec4 value) {
    vec4 rounded = uintBitsToFloat(floatBitsToUint(value) & 0xffffe000u);
    return mix(rounded, value, isnan(value));
}
#elif OUTPUT_R11G11B10
#include "input_quant.glsl"
#define FINAL_OUTPUT_FORMAT r11f_g11f_b10f
// The blit turns every NaN into the format's canonical NaN; a direct store would keep the payload's top bits.
vec4 finalOutputValue(vec4 value) {
    return mix(canonicalHalf(value), vec4(uintBitsToFloat(0x7fc00000u)), isnan(value));
}
#else
#define FINAL_OUTPUT_FORMAT rgba16f
#define finalOutputValue(value) (value)
#endif
