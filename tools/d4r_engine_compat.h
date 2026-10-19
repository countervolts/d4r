#pragma once
#include <vulkan/vulkan_core.h>
#include <sstream>
#include <string>

// Inspect metadata before constructing any pipeline. Old default D/E and F models require FP8;
// only the explicitly marked gfx11 D/E models use widened f16 cooperative matrices.
inline bool d4r_rr_model_compatible(const std::string& metadata, bool f16Matrices, bool fp8Matrices)
{
    std::istringstream lines(metadata);
    std::string line;
    if (!std::getline(lines, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "D4RR0002") return fp8Matrices;
    if (line != "D4RR0001") return false;
    bool widened = false;
    while (std::getline(lines, line))
    {
        std::istringstream fields(line);
        std::string key, value; fields >> key >> value;
        if (key == "arithmetic")
        {
            if (value != "gfx11") return false;
            widened = true;
        }
    }
    return widened ? f16Matrices : fp8Matrices;
}

inline bool d4r_matrix_tuple(const VkCooperativeMatrixPropertiesKHR& p, VkComponentTypeKHR operand)
{
    return p.MSize == 16 && p.NSize == 16 && p.KSize == 16 && p.scope == VK_SCOPE_SUBGROUP_KHR &&
        p.AType == operand && p.BType == operand && p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && !p.saturatingAccumulation;
}
