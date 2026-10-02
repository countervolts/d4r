#pragma once

// Numeric storage descriptions shared by the native HIP kernel and Windows
// host. These are independent of Win32, Wine, KFD and NVIDIA SDK headers.
namespace d4r::pixel {
enum Storage : unsigned {
    rgba16f, rgba32f, rgba8, bgra8, r11g11b10f, rgb10a2,
    rg16f, rg32f, rg16snorm, rg16unorm, r16f, r32f,
    depth24, r16unorm, rgba8srgb, bgra8srgb
};
enum Plane : unsigned { color, depth, motion, output, exposure };
}
