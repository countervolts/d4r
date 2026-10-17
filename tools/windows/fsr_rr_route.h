#pragma once
#include <stdexcept>
#include <string_view>

namespace d4r::win::rr {
inline bool native_fsr_route(std::string_view architecture, const char* overrideMode = nullptr) {
    const std::string_view mode = overrideMode ? overrideMode : "auto";
    if (mode == "native") return true;
    if (mode == "vkd3d") return false;
    if (mode.empty() || mode == "auto") return architecture.rfind("gfx12", 0) == 0;
    throw std::runtime_error("D4R_FSR_RR_DEVICE must be auto, native or vkd3d");
}
}
