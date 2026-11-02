// GPU stage timing for all game adapters. Never waits for the queue or reuses a pending slot.
#pragma once
#include "game.h"
#include "vk_dispatch.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

namespace d4r::game_detail {
class StageProfile {
    static constexpr unsigned Slots = 16, Queries = 64;
    struct Slot {
        VkEvent done = VK_NULL_HANDLE;
        GameProfile settings;
        std::vector<const char*> labels;
        uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
        bool pending = false;
    };
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueryPool queries = VK_NULL_HANDLE;
    std::array<Slot, Slots> slots{};
    int active = -1;
    bool failed = false;
    double nsPerTick = 0;
    uint64_t mask = ~uint64_t(0);

    void collect() {
        for (unsigned i = 0; i < Slots; ++i) {
            auto& s = slots[i];
            // Availability alone is insufficient: a new vkCmdResetQueryPool might still be queued.
            // Reset the event on the host before recording; only this submission can signal it again.
            if (!s.pending || vkGetEventStatus(device, s.done) != VK_EVENT_SET) continue;
            const uint32_t count = uint32_t(s.labels.size()) + 1;
            std::array<uint64_t, Queries> ticks{};
            if (vkGetQueryPoolResults(device, queries, i * Queries, count, count * sizeof(uint64_t),
                                      ticks.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) continue;
            s.pending = false;
            if (!s.settings.sampled) continue;
            auto ms = [&](unsigned a, unsigned b) { return double((ticks[b] - ticks[a]) & mask) * nsPerTick / 1e6; };
            char line[256];
            std::snprintf(line, sizeof(line), "D4R_ENGINE_STAGE_PROFILE preset=%s gpu_frame=%u render=%ux%u output=%ux%u gpu_eval=%.3f",
                          s.settings.name, s.settings.frame, s.rw, s.rh, s.ow, s.oh, ms(0, count - 1));
            std::string text(line);
            for (unsigned j = 0; j < s.labels.size(); ++j) {
                std::snprintf(line, sizeof(line), " %s=%.3f", s.labels[j], ms(j, j + 1));
                text += line;
            }
            s.settings.log(text.c_str());
        }
    }
    bool initialize(const GameProfile& settings) {
        if (failed) return false;
        if (queries) return true;
        failed = true;
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physical, &properties);
        nsPerTick = properties.limits.timestampPeriod;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
        unsigned bits = 64;
        for (const auto& family : families) {
            if (family.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
                bits = std::min(bits, family.timestampValidBits);
        }
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = Slots * Queries;
        if (!bits || nsPerTick <= 0 || vkCreateQueryPool(device, &qi, nullptr, &queries) != VK_SUCCESS) {
            settings.log("D4R_ENGINE_STAGE_PROFILE: GPU timestamps unavailable; inference continues without stage profiling");
            return false;
        }
        mask = bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
        failed = false;
        return true;
    }
public:
    StageProfile(VkPhysicalDevice p, VkDevice d) : physical(p), device(d) {}
    ~StageProfile() {
        for (auto& s : slots) if (s.done) vkDestroyEvent(device, s.done, nullptr);
        if (queries) vkDestroyQueryPool(device, queries, nullptr);
    }
    StageProfile(const StageProfile&) = delete;
    StageProfile& operator=(const StageProfile&) = delete;
    void begin(VkCommandBuffer cb, const GameProfile& settings, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh,
               std::initializer_list<const char*> labels) {
        active = -1;
        if (!settings.log || !initialize(settings)) return;
        collect();
        for (unsigned i = 0; i < Slots; ++i) {
            auto& s = slots[i];
            if (s.pending) continue;
            if (!s.done) {
                VkEventCreateInfo ei{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
                if (vkCreateEvent(device, &ei, nullptr, &s.done) != VK_SUCCESS) {
                    failed = true;
                    settings.log("D4R_ENGINE_STAGE_PROFILE: event allocation failed; skipping GPU stage sample");
                    return;
                }
            }
            if (vkResetEvent(device, s.done) != VK_SUCCESS) return;
            s.settings = settings; s.labels = labels;
            s.rw = rw; s.rh = rh; s.ow = ow; s.oh = oh;
            s.pending = true;
            active = int(i);
            vkCmdResetQueryPool(cb, queries, i * Queries, Queries);
            mark(cb, 0);
            return;
        }
        // All slots are in flight: skip instrumentation rather than waiting or resetting live queries.
    }
    VkQueryPool pool() const { return active < 0 ? VK_NULL_HANDLE : queries; }
    uint32_t first(uint32_t offset = 1) const { return active < 0 ? 0 : uint32_t(active) * Queries + offset; }
    void mark(VkCommandBuffer cb, uint32_t boundary) {
        if (active >= 0) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queries, first(boundary));
    }
    FrameProfiler callback() {
        return {[](VkCommandBuffer cb, uint32_t boundary, void* user) {
            static_cast<StageProfile*>(user)->mark(cb, boundary + 1);
        }, this};
    }
    void end(VkCommandBuffer cb) {
        if (active < 0) return;
        auto& s = slots[active];
        mark(cb, uint32_t(s.labels.size()));
        vkCmdSetEvent(cb, s.done, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        active = -1;
    }
};
} // namespace d4r::game_detail
