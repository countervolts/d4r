"""RR descriptor budgeting under independently rotating inputs, outputs and optional guides."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class RRBindingCacheTest(unittest.TestCase):
    def test_descriptor_budget_and_stable_copy_transition(self):
        source = r'''
#include "engine/rr_binding_cache.h"
#include <cassert>
#include <set>
#include <utility>
using Cache = d4r::RRBindingCache<uint64_t>;
using Key = std::pair<Cache::Views, uint8_t>;

Cache::Views effective(Cache& cache, Cache::Views views, uint8_t mask) {
    if (cache.useCopies(views, mask)) views.fill(0);
    // Stable adapter planes replace copied inputs. Optional guides stay null when absent; output has its own plane.
    for (size_t i = 0; i < views.size(); ++i) {
        const bool present = i < 7 || i == 11 || (mask & (1u << (i - 7)));
        if (!present) views[i] = 0;
        else if (!views[i]) views[i] = 1000 + i;
    }
    return views;
}
int main() {
    Cache repeated;
    Cache::Views views{};
    views.fill(1);
    for (unsigned i = 0; i < 10000; ++i) assert(!repeated.useCopies(views, 15));

    // Cover output-only rotation, which the previous per-input guard missed.
    Cache output;
    for (unsigned i = 0; i < d4r::kRRDirectBindings; ++i) {
        views[11] = 10 + i;
        assert(!output.useCopies(views, 15));
    }
    views[11] = 99;
    assert(output.useCopies(views, 15));
    views[11] = 10; // Once copies are selected, even an old borrowed combination stays on stable planes.
    assert(output.useCopies(views, 15));

    // No input needs more than three views to exceed the old 16-combination limit.
    Cache rotating;
    std::set<Key> bindings;
    for (unsigned frame = 0; frame < 10000; ++frame) {
        unsigned ring = frame;
        for (size_t i = 0; i < views.size(); ++i) {
            views[i] = 10 + 4 * i + ring % 3;
            ring /= 3;
        }
        const uint8_t mask = frame < d4r::kRRDirectBindings ? 15 : frame % 16;
        bindings.insert({effective(rotating, views, mask), mask});
        assert(bindings.size() <= d4r::kRRMaxBindings);
    }
    assert(bindings.size() == d4r::kRRMaxBindings);

    // Absent and copied optional guides share null candidate views but must have different identities.
    Cache optional;
    views.fill(0);
    for (uint8_t mask = 0; mask < 16; ++mask) assert(!optional.useCopies(views, mask));
    for (uint8_t mask = 0; mask < 16; ++mask) assert(!optional.useCopies(views, mask));
    views[11] = 9;
    assert(optional.useCopies(views, 0));
}
'''
        with tempfile.TemporaryDirectory() as temp:
            runner = pathlib.Path(temp) / "rr_binding_cache"
            subprocess.run(["c++", "-x", "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT), "-o", str(runner), "-"], input=source, text=True, check=True)
            subprocess.run([str(runner)], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
