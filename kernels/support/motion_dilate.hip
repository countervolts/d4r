#include "../../tools/d4r_motion_dilation.h"
#pragma clang fp contract(off)

__attribute__((device)) static int clamp_index(int x, uint32_t limit)
{
    return x < 0 ? 0 : x >= (int)limit ? (int)limit - 1 : x;
}

extern "C" __attribute__((global, amdgpu_flat_work_group_size(128, 128)))
void d4r_motion_dilate(D4rMotionDilationParams p)
{
    const uint32_t id = __builtin_amdgcn_workgroup_id_x() * 128 + __builtin_amdgcn_workitem_id_x();
    if ((uint64_t)id >= (uint64_t)p.motion_width * p.motion_height)
        return;
    const uint32_t x = id % p.motion_width, y = id / p.motion_width;
    const int cx = clamp_index((int)__builtin_floorf((x + 0.5f) * p.render_width / p.motion_width + p.jitter_x), p.render_width);
    const int cy = clamp_index((int)__builtin_floorf((y + 0.5f) * p.render_height / p.motion_height + p.jitter_y), p.render_height);
    const uint8_t* depth = (const uint8_t*)p.depth;
    const uint8_t* motion = (const uint8_t*)p.motion;
    uint8_t* output = (uint8_t*)p.output;
    const float current = *(const float*)(depth + (uint64_t)cy * p.depth_pitch + 4 * cx);
    float best = current;
    int bx = cx, by = cy;
    for (int dy = -(int)p.radius; dy <= (int)p.radius; ++dy)
        for (int dx = -(int)p.radius; dx <= (int)p.radius; ++dx)
        {
            const int nx = clamp_index(cx + dx, p.render_width), ny = clamp_index(cy + dy, p.render_height);
            const float candidate = *(const float*)(depth + (uint64_t)ny * p.depth_pitch + 4 * nx);
            if (p.reversed_depth ? candidate > best : candidate < best)
                best = candidate, bx = nx, by = ny;
        }
    // Sky gaps need wider coverage than opaque edges. Select the closest
    // foreground pixel within eight render pixels, rather than the closest
    // depth (which can borrow a distant roof/chimney's velocity).
    const bool sky = p.reversed_depth ? current <= 1.e-7f : current >= 1.0f - 1.e-7f;
    bool sky_filled = false;
    if (sky)
    {
        int best_distance = 65;
        for (int dy = -8; dy <= 8; ++dy)
            for (int dx = -8; dx <= 8; ++dx)
            {
                const int nx = cx + dx, ny = cy + dy;
                const int distance = dx * dx + dy * dy;
                if (nx < 0 || ny < 0 || nx >= (int)p.render_width || ny >= (int)p.render_height ||
                    distance > 64 || distance > best_distance)
                    continue;
                const float candidate = *(const float*)(depth + (uint64_t)ny * p.depth_pitch + 4 * nx);
                const bool foreground = p.reversed_depth ? candidate > 1.e-7f : candidate < 1.0f - 1.e-7f;
                // Stable coordinate tie-break matches nearest-foreground fill.
                if (foreground && (distance < best_distance || nx < bx || (nx == bx && ny < by)))
                    best_distance = distance, best = candidate, bx = nx, by = ny, sky_filled = true;
            }
    }
    uint32_t value = *(const uint32_t*)(motion + (uint64_t)y * p.motion_pitch + 4 * x);
    const bool nearer = p.reversed_depth ? best > current * 1.02f + 1.e-7f : best < current * 0.98f - 1.e-7f;
    if (sky_filled || nearer)
    {
        const int mx = clamp_index((int)__builtin_floorf((bx + 0.5f - p.jitter_x) * p.motion_width / p.render_width), p.motion_width);
        const int my = clamp_index((int)__builtin_floorf((by + 0.5f - p.jitter_y) * p.motion_height / p.render_height), p.motion_height);
        value = *(const uint32_t*)(motion + (uint64_t)my * p.motion_pitch + 4 * mx);
    }
    *(uint32_t*)(output + (uint64_t)y * p.motion_pitch + 4 * x) = value;
}
