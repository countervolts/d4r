#include "binary16.h"
#include <cstdio>
#include <initializer_list>
#include <stdexcept>

static void check(double value, uint16_t expected) {
    const auto actual = d4r::diag::binary16::half_bits(value);
    if (actual != expected) throw std::runtime_error("binary16 rounding mismatch");
}
int main() {
    using namespace d4r::diag::binary16;
    try {
        unsigned encodings = 0, boundaries = 0;
        for (unsigned bits = 0; bits < 65536; ++bits) {
            if ((bits & 0x7c00) == 0x7c00 && (bits & 0x3ff)) continue;
            check(half_value(uint16_t(bits)), uint16_t(bits));
            ++encodings;
        }
        for (unsigned bits = 0; bits < 0x7bff; ++bits) {
            const double low = half_value(uint16_t(bits)), high = half_value(uint16_t(bits + 1));
            const double midpoint = (low + high) * 0.5;
            const uint16_t tie = uint16_t(bits + (bits & 1));
            check(midpoint, tie); check(-midpoint, uint16_t(tie | 0x8000));
            check(std::nextafter(midpoint, low), uint16_t(bits));
            check(std::nextafter(midpoint, high), uint16_t(bits + 1));
#if !defined(_MSC_VER)
            for (double value : {midpoint, -midpoint, std::nextafter(midpoint, low), std::nextafter(midpoint, high)}) {
                const _Float16 native = static_cast<_Float16>(value);
                uint16_t expected; std::memcpy(&expected, &native, sizeof(expected));
                check(value, expected);
            }
#endif
            ++boundaries;
        }
        check(65519.0, 0x7bff); check(65520.0, 0x7c00);
        check(-65520.0, 0xfc00);
        check(std::numeric_limits<double>::denorm_min(), 0);
        check(-std::numeric_limits<double>::denorm_min(), 0x8000);
        check(std::numeric_limits<double>::quiet_NaN(), 0x7e00);
        std::printf("PASS BINARY16 encodings=%u rounding_boundaries=%u\n", encodings, boundaries);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL BINARY16 %s\n", e.what()); return 5;
    }
}
