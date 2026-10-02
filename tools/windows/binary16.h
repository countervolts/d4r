#pragma once
// Adapted from realdody/d4r ae530b0 (MIT), realdody <dodobozicek@gmail.com>.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>

namespace d4r::diag::binary16 {
// MSVC has no _Float16 extension. Convert double<->IEEE binary16 explicitly with
// round-to-nearest-even, matching a native _Float16 cast. This probe compares the
// f16-accumulator MMA against a double sum rounded once to half, so the rounding
// must match IEEE 754 binary16 exactly.
inline uint16_t half_bits(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 63) << 15);
    int32_t exponent = static_cast<int32_t>((bits >> 52) & 0x7ff);
    uint64_t mantissa = bits & 0x000fffffffffffffULL;
    if (exponent == 0x7ff) return static_cast<uint16_t>(sign | (mantissa ? 0x7e00 : 0x7c00));
    if (exponent == 0) return sign;                       // double subnormal rounds to half zero
    int32_t unbiased = exponent - 1023;
    if (unbiased > 15) return static_cast<uint16_t>(sign | 0x7c00);
    mantissa |= 0x0010000000000000ULL;                    // implicit leading one (53-bit significand)
    const int shift = unbiased >= -14 ? 42 : 28 - unbiased;
    uint64_t rounded = shift >= 64 ? 0 : (mantissa >> shift);
    if (shift < 64) {
        const uint64_t remainder = mantissa & ((1ULL << shift) - 1);
        const uint64_t halfway = 1ULL << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
    }
    if (unbiased >= -14) {
        if (rounded >= 0x800) {                            // rounding overflowed into the exponent
            rounded = 0x400;
            if (++unbiased > 15) return static_cast<uint16_t>(sign | 0x7c00);
        }
        return static_cast<uint16_t>(sign | ((unbiased + 15) << 10) | static_cast<uint16_t>(rounded & 0x3ff));
    }
    if (rounded >= 0x400) return static_cast<uint16_t>(sign | 0x0400);  // smallest normal half
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(rounded));
}
inline double half_value(uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    double magnitude;
    if (exponent == 0) magnitude = std::ldexp(static_cast<double>(mantissa), -24);
    else if (exponent == 0x1fu) magnitude = mantissa ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    else magnitude = std::ldexp(static_cast<double>(mantissa) / 1024.0 + 1.0, static_cast<int>(exponent) - 15);
    return negative ? -magnitude : magnitude;
}
}
