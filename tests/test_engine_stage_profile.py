"""Asynchronous engine profiling: queued resets must never expose stale query results."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('g++') and Path('/usr/include/vulkan/vulkan.h').exists(), 'needs C++ and Vulkan headers')
class EngineStageProfileTests(unittest.TestCase):
    def test_deferred_submission_sampling_and_timestamp_wrap(self):
        source = r'''
#define D4R_ENGINE_VK_TABLE
#include "engine/game_profile.h"
#include <cassert>
#include <functional>
#include <cstring>

namespace d4r { VkTable vkTable; }
static std::vector<std::function<void()>> commands;
static std::vector<std::string> logs;
static std::array<uint64_t, 1024> ticks{};
static std::array<bool, 64> events{};
static uint64_t clockTick = 250;
static unsigned eventCount = 0, queryReads = 0, resets = 0, poolCreates = 0;
static void submit() { for (auto& c : commands) c(); commands.clear(); }
static void sink(const char* line) { logs.emplace_back(line); }
int main() {
    using namespace d4r;
    vkTable.GetPhysicalDeviceProperties = [](VkPhysicalDevice, VkPhysicalDeviceProperties* p) {
        *p = {}; p->limits.timestampPeriod = 1000000;
    };
    vkTable.GetPhysicalDeviceQueueFamilyProperties = [](VkPhysicalDevice, uint32_t* n, VkQueueFamilyProperties* p) {
        *n = 1; if (p) { *p = {}; p->queueFlags = VK_QUEUE_COMPUTE_BIT; p->timestampValidBits = 8; }
    };
    vkTable.CreateQueryPool = [](VkDevice, const VkQueryPoolCreateInfo* i, const VkAllocationCallbacks*, VkQueryPool* p) {
        ++poolCreates; assert(i->queryCount == ticks.size()); *p = (VkQueryPool)1; return VK_SUCCESS;
    };
    vkTable.DestroyQueryPool = [](VkDevice, VkQueryPool, const VkAllocationCallbacks*) {};
    vkTable.CreateEvent = [](VkDevice, const VkEventCreateInfo*, const VkAllocationCallbacks*, VkEvent* e) {
        *e = (VkEvent)(uintptr_t)(++eventCount); return VK_SUCCESS;
    };
    vkTable.DestroyEvent = [](VkDevice, VkEvent, const VkAllocationCallbacks*) {};
    vkTable.ResetEvent = [](VkDevice, VkEvent e) { events[(uintptr_t)e] = false; return VK_SUCCESS; };
    vkTable.GetEventStatus = [](VkDevice, VkEvent e) { return events[(uintptr_t)e] ? VK_EVENT_SET : VK_EVENT_RESET; };
    vkTable.CmdResetQueryPool = [](VkCommandBuffer, VkQueryPool, uint32_t first, uint32_t count) {
        ++resets; commands.push_back([=] { for (unsigned i = 0; i < count; ++i) ticks[first + i] = 0; });
    };
    vkTable.CmdWriteTimestamp = [](VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t i) {
        commands.push_back([=] { ticks[i] = clockTick & 255; clockTick += 10; });
    };
    vkTable.CmdSetEvent = [](VkCommandBuffer, VkEvent e, VkPipelineStageFlags) {
        commands.push_back([=] { events[(uintptr_t)e] = true; });
    };
    vkTable.GetQueryPoolResults = [](VkDevice, VkQueryPool, uint32_t first, uint32_t count, size_t bytes,
                                    void* data, VkDeviceSize stride, VkQueryResultFlags flags) {
        ++queryReads; assert(flags == VK_QUERY_RESULT_64_BIT); // no WAIT_BIT
        assert(stride == 8 && bytes == count * 8);
        std::memcpy(data, ticks.data() + first, bytes); return VK_SUCCESS;
    };
    {
        game_detail::StageProfile p((VkPhysicalDevice)1, (VkDevice)1);
        p.begin({}, {}, 1, 1, 2, 2, {"input", "network"}); p.end({});
        assert(poolCreates == 0 && commands.empty()); // disabled means no Vulkan work
        auto begin = [&](unsigned frame, bool sampled) {
            p.begin({}, {sink, "L", frame, sampled}, 640, 360, 1920, 1080, {"input", "network"});
        };
        begin(1, true); p.mark({}, 1); p.end({});
        assert(logs.empty()); submit();
        begin(2, false); p.mark({}, 1); p.end({});
        assert(logs.size() == 1 && logs[0].find("gpu_frame=1") != std::string::npos);
        assert(logs[0].find("gpu_eval=20.000 input=10.000 network=10.000") != std::string::npos);
        // Slot 0 still holds old available queries. Its new reset has not executed yet.
        // Polling must not report those values as frame 2 or reset/reuse its live slot.
        begin(3, true); p.mark({}, 1); p.end({});
        assert(logs.size() == 1 && queryReads == 1);
        submit();
        begin(4, true); p.mark({}, 1); p.end({});
        assert(logs.size() == 2 && logs[1].find("gpu_frame=3") != std::string::npos);
        // Exhaust the remaining slots without submission. Profiling skips rather than waiting.
        for (unsigned i = 5; i < 20; ++i) { begin(i, true); p.mark({}, 1); p.end({}); }
        unsigned before = resets;
        begin(20, true); p.mark({}, 1); p.end({});
        assert(resets == before && !p.pool());
        submit();
        begin(21, true); p.mark({}, 1); p.end({}); submit();
        assert(logs.size() == 18); // 1, 3, 4..19 (frame 2 unsampled, 20 skipped)
    }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp) / 'profile.cpp'
            binary = Path(tmp) / 'profile'
            cpp.write_text(source)
            subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++20', '-Wall', '-Wextra', '-Werror',
                            '-Wno-missing-field-initializers', '-I', str(ROOT), str(cpp), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
