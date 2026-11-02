#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace d4r {
// Keep borrowed descriptors alive for recorded GPU work. Once their budget is exhausted, the adapter copies every
// input and output through stable images. Four optional inputs leave at most 16 further descriptor combinations.
inline constexpr size_t kRRDirectBindings = 16;
inline constexpr size_t kRROptionalBindings = 1u << 4;
inline constexpr size_t kRRMaxBindings = kRRDirectBindings + kRROptionalBindings;

template<class View> class RRBindingCache {
public:
    using Views = std::array<View, 12>; // colour, motion, depth, eight guides, output; null means copied/absent
    bool useCopies(const Views& views, uint8_t optionalMask) {
        if (copyAll) return true;
        for (size_t i = 0; i < count; ++i)
            if (combinations[i].views == views && combinations[i].optionalMask == optionalMask) return false;
        if (count == combinations.size()) {
            copyAll = true;
            return true;
        }
        combinations[count++] = {views, optionalMask};
        return false;
    }
private:
    struct Combination { Views views{}; uint8_t optionalMask = 0; };
    std::array<Combination, kRRDirectBindings> combinations{};
    size_t count = 0;
    bool copyAll = false;
};
} // namespace d4r
