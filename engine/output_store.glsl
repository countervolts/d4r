// RADV's rgba16f image stores truncate before the copy path's UNORM conversion.
// Preserve that half-store + blit result when writing RGB10A2 directly. Below
// binary16's normal range both 10-bit RGB and 2-bit alpha quantize to zero, and
// overflow saturates, so a normal-range mantissa mask suffices for this format.
#ifndef OUTPUT_RGB10
#define OUTPUT_RGB10 0
#endif
#if OUTPUT_RGB10
#define FINAL_OUTPUT_FORMAT rgb10_a2
vec4 finalOutputValue(vec4 value) {
    vec4 rounded = uintBitsToFloat(floatBitsToUint(value) & 0xffffe000u);
    return mix(rounded, value, isnan(value));
}
#else
#define FINAL_OUTPUT_FORMAT rgba16f
#define finalOutputValue(value) (value)
#endif
