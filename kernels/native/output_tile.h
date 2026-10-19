// Structured path-B tile fill. Same color/depth values and 24-column LDS layout
// as the original kernel; work-items traverse x/y directly without integer division.
#pragma once

#ifndef D4R_OUTPUT_TILE_PITCH
#define D4R_OUTPUT_TILE_PITCH 24
#endif
static_assert(D4R_OUTPUT_TILE_PITCH >= 24 && D4R_OUTPUT_TILE_PITCH <= 32, "LDS tile pitch capacity");

RT void output_color_pack(f4 color, float exposure, uint32_t* a, uint32_t* b)
{
    const float r = fmaxn(color[0], 0.0f), g = fmaxn(color[1], 0.0f), blue = fmaxn(color[2], 0.0f);
    const float qr = r * 0.25f, hg = g * 0.5f, qb = blue * 0.25f;
    const float y = exposure * ((qr + hg) + qb);
    const float co = __builtin_fmaf(r, 0.5f, g * 0.0f) - blue * 0.5f;
    const float cg = (hg - qr) - qb;
    float weight;
    if (y < 1000.0f)
        weight = rcp_a(y + 1.0f);
    else
    {
        const float z = __builtin_fmaf(y, 9.0f, 1.0f);
        const float ln = lg2_a(z) * asf(0x3F317218u);
        const float q = div_a(ln, 9.0f) + asf(0xBC4FAE29u);
        weight = rcp_a(y) * q;
    }
    // f2h_rn: rounded to f32 and then to f16, as ZLUDA's compile does (no v_fma_mixlo of the product)
    *a = pack16(ashu(f2h_rn(y * weight)), ashu(f2h_rn((exposure * co) * weight)));
    *b = pack16(ashu(f2h_rn((exposure * cg) * weight)), ashu(f2h_rn(color[3])));
}

RT int output_tile_coord(int relative, int lo, int hi)
{
    return relative > 0 ? __builtin_elementwise_min(lo + relative, hi) : lo;
}

RT void output_tile_fill(const AS4 uint8_t* args, int x0, int y0, int width, int height,
                         float exposure, AS3 uint8_t* color_tile, AS3 uint8_t* depth_tile)
{
    const uint64_t color = *(const AS4 uint64_t*)(args + 200);
    const uint64_t depth = *(const AS4 uint64_t*)(args + 248);
    const int cx0 = *(const AS4 int32_t*)(args + 208), cy0 = *(const AS4 int32_t*)(args + 212);
    const int cx1 = *(const AS4 int32_t*)(args + 216), cy1 = *(const AS4 int32_t*)(args + 220);
    const int dx0 = *(const AS4 int32_t*)(args + 96), dy0 = *(const AS4 int32_t*)(args + 100);
    const int dx1 = *(const AS4 int32_t*)(args + 104), dy1 = *(const AS4 int32_t*)(args + 108);
    bool fill_depth = true;
#ifdef D4R_K_NO_DEPTH_TILE
    // Pinned K LUT has |slope|<=1. Its 4x3 color stencil stays in the four-texel
    // halo for positive upsampling scale<=1, so no color read aliases depth LDS.
    // Keep the original fill outside the proven geometry. Input depth is unchanged.
    const float scale_x = *(const AS4 float*)(args + 40), scale_y = *(const AS4 float*)(args + 44);
    fill_depth = !(scale_x > 0.0f && scale_x <= 1.0f && scale_y > 0.0f && scale_y <= 1.0f);
#endif
    // Path B selects a footprint at most 24 x 24. At each iteration 0<=x<width,
    // 0<=y<height: color accesses stay within 4608 bytes, depth within 2304 bytes.
    for (int y = __builtin_amdgcn_workitem_id_y(); y < height; y += 16)
        for (int x = __builtin_amdgcn_workitem_id_x(); x < width; x += 16)
        {
            const int sx = x0 + x, sy = y0 + y;
            const f4 c = tex_fetch(color, output_tile_coord(sx, cx0, cx1), output_tile_coord(sy, cy0, cy1));
            uint32_t a, b;
            output_color_pack(c, exposure, &a, &b);
            AS3 uint32_t* dst = (AS3 uint32_t*)(color_tile + y * (8 * D4R_OUTPUT_TILE_PITCH) + x * 8);
            dst[0] = a;
            dst[1] = b;
            if (fill_depth)
            {
                const f4 d = tex_fetch(depth, output_tile_coord(sx, dx0, dx1), output_tile_coord(sy, dy0, dy1));
                *(AS3 float*)(depth_tile + y * (4 * D4R_OUTPUT_TILE_PITCH) + x * 4) = d[0];
            }
        }
}
