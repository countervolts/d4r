#pragma once

#include <cstdlib>

// Platform-specific environment reads stay in the caller. SR and RR deliberately
// use separate settings, even when both are configured for the same feature call.
inline constexpr const char* d4r_render_preset_variable(bool rayReconstruction)
{
    return rayReconstruction ? "D4R_RR_PRESET" : "D4R_DLSS_PRESET";
}

// An absent/empty override leaves every game-selected quality preset unchanged.
// An explicit zero is an override too: it requests NGX's default preset.
struct D4rRenderPresetOverride
{
    bool enabled;
    unsigned int value;

    void apply(unsigned int (&presets)[6]) const
    {
        if (enabled)
            for (unsigned int& preset : presets)
                preset = value;
    }
};

inline D4rRenderPresetOverride d4r_parse_render_preset_override(const char* overrideValue)
{
    if (overrideValue == nullptr || *overrideValue == '\0')
        return {false, 0};
    return {true, static_cast<unsigned int>(std::strtoul(overrideValue, nullptr, 0))};
}
