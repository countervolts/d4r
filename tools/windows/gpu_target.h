#pragma once
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include "gpu_targets.generated.h"

#ifndef D4R_TARGET_ARCH
#define D4R_TARGET_ARCH "gfx1201"
#endif

namespace d4r::diag {
inline bool supported_gpu_target(const std::string& target) {
    for (const auto& row : gpu_targets) if (target == row.architecture) return true;
    return false;
}
inline std::string gpu_architecture(const std::string& name) {
    return name.substr(0, name.find(':'));
}
inline void require_gpu_target(const std::string& actual, const std::string& expected) {
    if (!supported_gpu_target(expected) || gpu_architecture(actual) != expected)
        throw std::runtime_error("GPU architecture mismatch: detected " + actual + ", package requires " + expected);
}
inline std::string code_object_target(const std::filesystem::path& path) {
    // AMDGPU ELF e_flags machine IDs from LLVM's documented code-object ABI.
    // Never infer compatibility from a filename or rewrite the ELF target.
    std::ifstream file(path, std::ios::binary);
    std::array<unsigned char, 64> header{};
    if (!file.read(reinterpret_cast<char*>(header.data()), header.size()) ||
        header[0] != 0x7f || header[1] != 'E' || header[2] != 'L' || header[3] != 'F' ||
        header[4] != 2 || header[5] != 1 || header[18] != 0xe0 || header[19] != 0)
        throw std::runtime_error("Invalid AMDGPU ELF64 code object: " + path.string());
    for (const auto& row : gpu_targets) if (header[48] == row.elf_machine) return row.architecture;
    throw std::runtime_error("Unsupported AMDGPU code-object target: " + path.string());
}
inline void require_code_object_target(const std::filesystem::path& path, const std::string& expected) {
    require_gpu_target(code_object_target(path), expected);
}
}
