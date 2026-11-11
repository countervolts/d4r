"""Timeline safety when prep cancellation overtakes a CUDA producer."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from shim_source import shim_source

ROOT = Path(__file__).resolve().parents[1]


class FrameCompletionTests(unittest.TestCase):
    def test_shim_does_not_signal_past_running_producer(self):
        shim = shim_source(ROOT)
        start = shim.index('void release_split_frame(Feature* feature, uint32_t frame)\n{')
        end = shim.index('\nstatic void split_watchdog()', start)
        # release_split_frame signals through this helper, defined after the engine network's exports
        helper = shim.index('\nVkResult signal_split_semaphore(Feature& feature, uint64_t value)\n{')
        helper = shim[helper:shim.index('\n}\n', helper) + 3]
        source = r'''
#include "d4r_frame_completion.h"
#include <cassert>
#include <mutex>
#include <cstdarg>
using VkResult = int;
constexpr int VK_SUCCESS = 0, VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO = 1;
constexpr int VK_ERROR_DEVICE_LOST = -1;
struct VkSemaphoreSignalInfo { int type; int semaphore = 0; uint64_t value = 0; };
constexpr bool SUCCEEDED(int code) { return code == 0; }
struct D4rSplitFence { int Signal(uint64_t) { return 0; } }; // native D3D12 only: this test's path stays on nullptr
struct Feature {
    std::mutex splitMutex;
    D4rFrameCompletion splitCompletion;
    uint64_t splitSignalled = 0;
    int splitSemaphore = 1;
    D4rSplitFence* splitFence = nullptr; // the native D3D12 timeline: null here, so the Vulkan paths below run
    bool engineNetTimeline = false; // the Vulkan timeline of this test, not the native network's
    int engineNet = 0;
    void* ngxTimeline = nullptr;
};
static struct { int (*signalTimeline)(int, uint64_t) = nullptr; } g_engineNet;
static struct { struct { int (*timelineSignal)(void*, uint64_t) = nullptr; } cu; } g;
static uint64_t gpuTimeline = 0;
static bool failSignal = false;
static int signal_semaphore(int, const VkSemaphoreSignalInfo* info) { // not signal(): MinGW declares C's
    if (failSignal) return -1;
    assert(info->value > gpuTimeline);
    gpuTimeline = info->value;
    return VK_SUCCESS;
}
static struct { int device = 1; decltype(&signal_semaphore) signalSemaphore = signal_semaphore; } g_vk;
static void logf(const char*, ...) {}
''' + helper + shim[start:end] + r'''
int main() {
    Feature feature;
    feature.splitCompletion.admit(1); // still running on the CUDA worker
    feature.splitCompletion.admit(2); // cancelled by the prep worker
    release_split_frame(&feature, 2);
    assert(gpuTimeline == 0 && feature.splitSignalled == 0);
    release_split_frame(&feature, 1);
    assert(gpuTimeline == 2 && feature.splitSignalled == 2);
    feature.splitCompletion.admit(4); // 3 was rejected before admission
    failSignal = true;
    release_split_frame(&feature, 4);
    assert(gpuTimeline == 2 && feature.splitSignalled == 2);
    failSignal = false;
    feature.splitCompletion.admit(5);
    release_split_frame(&feature, 5);
    assert(gpuTimeline == 5 && feature.splitSignalled == 5);
}
'''
        self.run_cpp(source)

    def test_out_of_order_retirement(self):
        source = r'''
#include "d4r_frame_completion.h"
#include <algorithm>
#include <array>
#include <cassert>

int main() {
    // Exhaust every completion/cancellation order of six admitted frames.
    // Gaps represent rejected evaluations, which never have a producer.
    const std::array<unsigned, 6> frames = {1, 2, 4, 7, 8, 12};
    std::array<unsigned, 6> order = {0, 1, 2, 3, 4, 5};
    do {
        D4rFrameCompletion completion;
        for (auto frame : frames) completion.admit(frame);
        std::array<bool, 6> done = {};
        for (auto index : order) {
            done[index] = true;
            unsigned expected = 0;
            for (unsigned i = 0; i < frames.size() && done[i]; ++i)
                expected = frames[i];
            assert(completion.retire(frames[index]) == expected);
        }
        assert(completion.retire(12) == 12);
        completion.admit(15);
        completion.admit(16);
        assert(completion.retire(16) == 12);
        assert(completion.retire(999) == 12);
        assert(completion.retire(15) == 16);
    } while (std::next_permutation(order.begin(), order.end()));
    // A retiring prep job must not release a still-running earlier writer.
    D4rFrameCompletion completion;
    completion.admit(1345);
    completion.admit(1346);
    assert(completion.retire(1346) == 0);
    assert(completion.retire(1345) == 1346);
}
'''
        self.run_cpp(source)

    def run_cpp(self, source):
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / 'frame-completion'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT / 'tools'), '-x', 'c++', '-', '-o', str(runner)],
                           input=source, text=True, capture_output=True, check=True)
            subprocess.run([str(runner)], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
