#pragma once
// GPU-architecture selection rules shared by the loader, the NGX shim and the
// NVAPI compatibility DLL. Pure logic: no HIP, no Windows headers, so the same
// code is unit-tested without a GPU (see player_config_probe.cpp).
//
// One physical adapter decides everything: its HIP architecture picks the
// _nvngx_<arch>.dll shim, the nvapi\<arch>\nvapi64.dll and the native kernels,
// and its LUID picks the device when several GPUs share an architecture.
#include "gpu_target.h"
#include <array>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace d4r::diag {
using AdapterLuid = std::array<unsigned char, 8>; // Memory layout of LUID: LowPart, then HighPart.

struct GpuCandidate { std::string name, architecture; AdapterLuid luid{}; };

inline std::string supported_gpu_list() {
    std::string list;
    for (const auto& target : gpu_targets) list += (list.empty() ? "" : ", ") + std::string(target.architecture);
    return list;
}
inline std::string describe_gpus(const std::vector<GpuCandidate>& devices) {
    std::string text;
    for (const auto& device : devices) text += (text.empty() ? "" : ", ") + device.architecture + " (" + device.name + ")";
    return text.empty() ? "none" : text;
}

// "HHHHHHHH:LLLLLLLL", the form the logs already use for adapter LUIDs.
inline std::string format_luid(const AdapterLuid& luid) {
    unsigned low = 0, high = 0;
    for (int i = 0; i < 4; ++i) { low |= unsigned(luid[i]) << (8 * i); high |= unsigned(luid[4 + i]) << (8 * i); }
    char text[32]; std::snprintf(text, sizeof(text), "%08x:%08x", high, low);
    return text;
}
inline bool parse_luid(const char* text, AdapterLuid& luid) {
    const std::string value = text ? text : "";
    if (value.size() != 17 || value[8] != ':' || value.find_first_not_of("0123456789abcdefABCDEF:") != std::string::npos ||
        value.find(':', 9) != std::string::npos) return false;
    const unsigned high = std::stoul(value.substr(0, 8), nullptr, 16), low = std::stoul(value.substr(9), nullptr, 16);
    for (int i = 0; i < 4; ++i) { luid[i] = static_cast<unsigned char>(low >> (8 * i)); luid[4 + i] = static_cast<unsigned char>(high >> (8 * i)); }
    return true;
}

// D4R_GPU_ARCH, when valid, names the architecture this process runs for; the
// build's own target applies when it is absent. A set but unsupported value is
// reported and ignored: it must not silently pick a different GPU target.
struct TargetResolution { std::string architecture, source, warning; };
inline TargetResolution resolve_target_arch(const char* configured, const char* built_in) {
    TargetResolution result{built_in, "build", {}};
    if (!configured || !*configured) return result;
    const std::string arch = gpu_architecture(configured); // "gfx1101:xnack-" -> "gfx1101"
    if (supported_gpu_target(arch)) { result.architecture = arch; result.source = "D4R_GPU_ARCH"; return result; }
    result.warning = std::string("D4R_GPU_ARCH='") + configured + "' is not a supported target (supported: " +
        supported_gpu_list() + "); using the build's target " + built_in;
    return result;
}

inline int find_gpu_device(const std::vector<GpuCandidate>& devices, const std::string& target, int requested = -1,
    const AdapterLuid* luid = nullptr) {
    for (size_t i = 0; i < devices.size(); ++i)
        if (devices[i].architecture == target && (requested < 0 || requested == int(i)) && (!luid || devices[i].luid == *luid))
            return int(i);
    return -1;
}

// Why no device matched, in terms a player can act on: unsupported hardware,
// a file built for another GPU, or an adapter HIP cannot see.
inline std::string no_device_message(const std::vector<GpuCandidate>& devices, const std::string& target,
    const AdapterLuid* luid = nullptr) {
    const std::string prefix = "No selected " + target + " device for this adapter. ";
    if (devices.empty()) return prefix + "The AMD HIP runtime reports no GPU; update the AMD graphics driver.";
    const GpuCandidate* actual = nullptr;
    if (luid) {
        for (const auto& device : devices) if (device.luid == *luid) actual = &device;
        if (!actual) return prefix + "No HIP device has the game's adapter LUID " + format_luid(*luid) + ". HIP sees: " + describe_gpus(devices) + ".";
    } else {
        for (const auto& device : devices) if (supported_gpu_target(device.architecture)) { actual = &device; break; }
    }
    if (!actual) return prefix + "Detected: " + describe_gpus(devices) + ", none supported by d4r (supported: " + supported_gpu_list() + ").";
    if (!supported_gpu_target(actual->architecture))
        return prefix + "The GPU is " + actual->architecture + " (" + actual->name + "), which d4r does not support (supported: " + supported_gpu_list() + ").";
    return prefix + "The GPU is " + actual->architecture + " (" + actual->name + ") but this file is built for " + target +
        ". A file for another GPU was loaded (nvapi\\" + actual->architecture + "\\nvapi64.dll, _nvngx_" + actual->architecture +
        ".dll, or a stale D4R_GPU_ARCH); extract the d4r ZIP again.";
}

// An NVAPI (or any per-target file) that names another target than the selected GPU's is a mixed install.
inline std::string nvapi_target_error(const std::string& built, const std::string& wanted, const std::string& path) {
    if (built == wanted) return {};
    return "nvapi64.dll (" + path + ") is built for " + (built.empty() ? std::string("an unknown target") : built) +
        " but the selected GPU is " + wanted + "; use d4r\\nvapi\\" + wanted + "\\nvapi64.dll and extract the d4r ZIP again if files are mixed up";
}

// Loader: the HIP architecture of the game's adapter (matched by LUID), or,
// before any adapter is known, the only architecture that has a build.
struct LoaderChoice { std::string architecture, error; };
inline LoaderChoice choose_loader_architecture(const std::vector<GpuCandidate>& devices, const AdapterLuid* luid,
    const std::function<bool(const std::string&)>& has_build, const std::string& available) {
    LoaderChoice choice;
    for (const auto& device : devices) {
        if (luid && device.luid != *luid) continue;
        if (!luid && !has_build(device.architecture)) continue;
        if (!choice.architecture.empty() && choice.architecture != device.architecture)
            return {{}, "several GPUs with different architectures; start the game once DLSS selects one"};
        choice.architecture = device.architecture;
    }
    if (choice.architecture.empty())
        choice.error = "no supported AMD GPU for this game's adapter. Detected: " + describe_gpus(devices) +
            ". This release supports: " + available + ".";
    else if (!supported_gpu_target(choice.architecture))
        choice.error = "this GPU (" + choice.architecture + ") is not supported by d4r. This release supports: " + available + ".";
    else if (!has_build(choice.architecture))
        choice.error = "this release has no build for your GPU (" + choice.architecture + "). It supports: " + available + ".";
    if (!choice.error.empty()) choice.architecture.clear();
    return choice;
}
}
