// Exact low-res, unjittered specialization. Compile as wave32 only.
#include "../../tools/d4r_motion_dilation.h"
#pragma clang fp contract(off)
#define D4R_LDS_SYNC() do{__builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");}while(0)
#define LDS __attribute__((shared))
__attribute__((device)) static inline int clampi(int x, int n) { return x < 0 ? 0 : x >= n ? n - 1 : x; }
template<bool ThinSky>
__attribute__((device, always_inline)) static inline
void motion_dilate_lds(D4rMotionDilationParams p, uint32_t allocation_height)
{
    LDS float tile[32][33];
    LDS unsigned rowmask[32];
    LDS unsigned anyfg;
    LDS uint64_t extended[ThinSky ? 38 : 1];
    const int t = __builtin_amdgcn_workitem_id_x();
    const int tx = t & 15, ty = t >> 4;
    const int ox = __builtin_amdgcn_workgroup_id_x() * 16 - 8, oy = __builtin_amdgcn_workgroup_id_y() * 16 - 8;
    const int W = p.render_width, H = p.render_height;
    const bool rev = p.reversed_depth;
    const uint8_t* depth = (const uint8_t*)p.depth;
    const uint8_t* motion = (const uint8_t*)p.motion;
    uint8_t* output = (uint8_t*)p.output;
    // Extra tiles and inactive lanes copy allocation padding, including pitch bytes.
    // The bridge launches over pitch/4 by allocation_height, not the active grid.
    const int x = ox + 8 + tx, y = oy + 8 + ty;
    const bool allocated = (uint64_t)x < p.motion_pitch / 4 && y < (int)allocation_height;
    if (allocated && (x >= W || y >= H))
        *(uint32_t*)(output + (uint64_t)y * p.motion_pitch + 4 * x) =
            *(const uint32_t*)(motion + (uint64_t)y * p.motion_pitch + 4 * x);
    // This exit is uniform across the workgroup; partial tiles reach both barriers.
    if (ox + 8 >= W || oy + 8 >= H) return;
    if (t == 0) anyfg = 0;
    #pragma unroll
    for (int j = 0; j < 2; ++j)
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const int ly = ty + 16 * j, lx = tx + 16 * i;
            tile[ly][lx] = *(const float*)(depth + (uint64_t)clampi(oy + ly, H) * p.depth_pitch + 4 * clampi(ox + lx, W));
        }
    D4R_LDS_SYNC();
    if constexpr (ThinSky)
    {
        // Source candidates reach the edge of the 32x32 tile. Three extra
        // foreground bits on each side prove whether a candidate belongs to a
        // run of four, without truncating runs at a workgroup boundary.
        const int lane = t & 31, wave = t >> 5;
        for (int r = wave; r < 38; r += 8)
        {
            const int row = r - 3, gy = oy + row;
            bool fg[2];
            #pragma unroll
            for (int part = 0; part < 2; ++part)
            {
                const int col = lane - 3 + part * 32, gx = ox + col;
                const bool valid = (part == 0 || lane < 6) && gx >= 0 && gx < W && gy >= 0 && gy < H;
                float d = 0;
                if (valid)
                    d = row >= 0 && row < 32 && col >= 0 && col < 32 ? tile[row][col] :
                        *(const float*)(depth + (uint64_t)gy * p.depth_pitch + 4 * gx);
                fg[part] = valid && (rev ? d > 1.e-7f : d < 1.0f - 1.e-7f);
            }
            const unsigned low = __builtin_amdgcn_ballot_w32(fg[0]);
            const unsigned high = __builtin_amdgcn_ballot_w32(fg[1]);
            if (lane == 0) extended[r] = low | ((uint64_t)high << 32);
        }
        D4R_LDS_SYNC();
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            if (lane == 0)
            {
                const int row = wave * 4 + r + 3;
                const uint64_t m = extended[row];
                const uint64_t heads = m & (m >> 1) & (m >> 2) & (m >> 3);
                const uint64_t horizontal = heads | (heads << 1) | (heads << 2) | (heads << 3);
                uint64_t vertical = 0;
                #pragma unroll
                for (int k = -3; k <= 0; ++k)
                    vertical |= extended[row+k] & extended[row+k+1] & extended[row+k+2] & extended[row+k+3];
                // Preserve long wires/poles: short in either cross-section.
                const unsigned thin = (unsigned)((m & ~(horizontal & vertical)) >> 3);
                rowmask[row-3] = thin;
                if (thin) __atomic_fetch_or(&anyfg, 1u, __ATOMIC_RELAXED);
            }
    }
    else
    {
        // Each wave32 handles four rows, one ballot per row. Out-of-image halo
        // values are clamped for the opaque scan but excluded from the sky masks.
        {
            const int lane = t & 31, wave = t >> 5;
            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                const int row = wave * 4 + r, gy = oy + row, gx = ox + lane;
                const float d = tile[row][lane];
                const bool fg = (rev ? d > 1.e-7f : d < 1.0f - 1.e-7f) && gy >= 0 && gy < H && gx >= 0 && gx < W;
                const unsigned m = __builtin_amdgcn_ballot_w32(fg);
                if (lane == 0) { rowmask[row] = m; if (m) __atomic_fetch_or(&anyfg, 1u, __ATOMIC_RELAXED); }
            }
        }
    }
    D4R_LDS_SYNC();
    if (x >= W || y >= H) return;
    const int lx = tx + 8, ly = ty + 8;
    const float current = tile[ly][lx];
    float best = current;
    int bx = x, by = y;
    const int R = (int)p.radius;
    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy)
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx)
        {
            if (dy < -R || dy > R || dx < -R || dx > R) continue;
            const float c = tile[ly + dy][lx + dx];
            if (rev ? c > best : c < best)
                best = c, bx = x + dx, by = y + dy;
        }
    bx = clampi(bx, W); by = clampi(by, H);
    const bool sky = rev ? current <= 1.e-7f : current >= 1.0f - 1.e-7f;
    bool sky_filled = false;
    if (sky && anyfg)
    {
        const int wd[9] = {8, 7, 7, 7, 6, 6, 5, 3, 0};
        int bd = 65, fx = 0, fy = 0;
        #pragma unroll
        for (int dy = -8; dy <= 8; ++dy)
        {
            const int w = wd[dy < 0 ? -dy : dy];
            unsigned m = rowmask[ly + dy];
            const int lo = lx - w, hi = lx + w;
            m &= (hi >= 31 ? 0xffffffffu : ((1u << (hi + 1)) - 1u)) & (lo <= 0 ? 0xffffffffu : ~((1u << lo) - 1u));
            if (!m) continue;
            const unsigned left = m & ((2u << lx) - 1u), right = m & ~((1u << lx) - 1u);
            int cdx = 99;
            if (left) cdx = (31 - __builtin_clz(left)) - lx;
            if (right) { const int b = __builtin_ctz(right) - lx; if (cdx == 99 || b < -cdx) cdx = b; }
            const int d2 = cdx * cdx + dy * dy, nx = x + cdx;
            if (d2 < bd || (d2 == bd && nx < fx)) bd = d2, fx = nx, fy = y + dy;
        }
        if (bd <= 64) bx = fx, by = fy, sky_filled = true;
    }
    uint32_t value = *(const uint32_t*)(motion + (uint64_t)y * p.motion_pitch + 4 * x);
    const bool nearer = rev ? best > current * 1.02f + 1.e-7f : best < current * 0.98f - 1.e-7f;
    if (sky_filled || nearer)
        value = *(const uint32_t*)(motion + (uint64_t)by * p.motion_pitch + 4 * bx);
    *(uint32_t*)(output + (uint64_t)y * p.motion_pitch + 4 * x) = value;
}

extern "C" __attribute__((global, amdgpu_flat_work_group_size(256, 256)))
void d4r_motion_dilate_lds(D4rMotionDilationParams p, uint32_t allocation_height)
{
    motion_dilate_lds<false>(p, allocation_height);
}

extern "C" __attribute__((global, amdgpu_flat_work_group_size(256, 256)))
void d4r_motion_dilate_lds_thin(D4rMotionDilationParams p, uint32_t allocation_height)
{
    motion_dilate_lds<true>(p, allocation_height);
}

#undef LDS
#undef D4R_LDS_SYNC
