#pragma once
#include <stdint.h>

/* Shared Wine/HIP ABI. Pitches are bytes; motion pixels are packed RG16F. */
typedef struct D4rMotionDilationParams
{
    uint64_t depth, motion, output;
    uint64_t depth_pitch, motion_pitch;
    uint32_t render_width, render_height, motion_width, motion_height;
    float jitter_x, jitter_y;
    uint32_t radius, reversed_depth;
} D4rMotionDilationParams;

/* Low-resolution velocity and depth share the render raster grid. Allocation
   padding is not part of that grid; retain the original NGX flags and MV scale.
   Display-resolution velocity still needs the jittered depth-grid mapping. */
static inline int d4r_motion_dilation_geometry(D4rMotionDilationParams* p, uint32_t flags,
    uint32_t output_width, uint32_t output_height)
{
    if ((flags & 4u) != 0 || p->render_width == 0 || p->render_height == 0)
        return 0; /* Jittered motion values remain unsupported. */
    if ((flags & 2u) != 0)
    {
        if (p->motion_width < p->render_width || p->motion_height < p->render_height)
            return 0;
        p->motion_width = p->render_width;
        p->motion_height = p->render_height;
        p->jitter_x = p->jitter_y = 0;
        return 1;
    }
    return p->motion_width == output_width && p->motion_height == output_height;
}
#ifdef __cplusplus
static_assert(sizeof(D4rMotionDilationParams) == 72, "motion dilation ABI");
#else
_Static_assert(sizeof(D4rMotionDilationParams) == 72, "motion dilation ABI");
#endif
