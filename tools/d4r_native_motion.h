#pragma once
#include <stdint.h>
typedef struct D4rNativeMotionParams
{
    uint64_t depth,geometry,scene,matrix,output;
    uint64_t depth_pitch,geometry_pitch,output_pitch;
    uint32_t width,height,render_width,render_height;
} D4rNativeMotionParams;
#ifdef __cplusplus
static_assert(sizeof(D4rNativeMotionParams)==80);
#endif

/* Native AW2 hardware-depth contract: immutable input planes; disjoint output. */
static inline int d4r_native_motion_parameters_valid(const D4rNativeMotionParams* p)
{
    uint64_t pointers[5],sizes[5],pitches[3],row;
    unsigned int i;
    if(!p || !p->width || !p->height || p->width>4096 || p->height>4096 ||
        !p->render_width || !p->render_height || p->render_width>p->width || p->render_height>p->height) return 0;
    row=(uint64_t)p->width*4;
    pitches[0]=p->depth_pitch;pitches[1]=p->geometry_pitch;pitches[2]=p->output_pitch;
    for(i=0;i<3;i++)
    {
        if(pitches[i]<row || pitches[i]%4 || (p->height>1 && pitches[i]>(UINT64_MAX-row)/(p->height-1))) return 0;
    }
    pointers[0]=p->depth;pointers[1]=p->geometry;pointers[2]=p->scene;pointers[3]=p->matrix;pointers[4]=p->output;
    sizes[0]=(p->height-1)*p->depth_pitch+row;sizes[1]=(p->height-1)*p->geometry_pitch+row;
    sizes[2]=71*16;sizes[3]=4*16;sizes[4]=(p->height-1)*p->output_pitch+row;
    for(i=0;i<5;i++) if(!pointers[i] || pointers[i]%4 || pointers[i]>UINT64_MAX-sizes[i]) return 0;
    for(i=0;i<4;i++)
        if(pointers[4]<pointers[i]+sizes[i] && pointers[i]<pointers[4]+sizes[4]) return 0;
    return 1;
}

// Additive thin-field ABI: depth coverage is jittered, output velocity grid is not.
typedef struct D4rNativeMotionThinParams
{
    D4rNativeMotionParams base;
    float jitter_x, jitter_y;
} D4rNativeMotionThinParams;
#ifdef __cplusplus
static_assert(sizeof(D4rNativeMotionThinParams)==88);
#endif

// Preserve the game's vectors outside thin native-silhouette footprints.
typedef struct D4rNativeMotionHybridParams
{
    D4rNativeMotionThinParams native;
    uint64_t game_motion,game_motion_pitch;
} D4rNativeMotionHybridParams;
#ifdef __cplusplus
static_assert(sizeof(D4rNativeMotionHybridParams)==104);
#endif
static inline int d4r_native_motion_hybrid_parameters_valid(const D4rNativeMotionHybridParams* p)
{
    if(!p || !d4r_native_motion_parameters_valid(&p->native.base) || !p->game_motion || p->game_motion%4 ||
        !(p->native.jitter_x>=-1 && p->native.jitter_x<=1 && p->native.jitter_y>=-1 && p->native.jitter_y<=1)) return 0;
    const D4rNativeMotionParams* b=&p->native.base;
    const uint64_t row=(uint64_t)b->render_width*4;
    if(p->game_motion_pitch<row || p->game_motion_pitch%4 ||
        (b->render_height>1 && p->game_motion_pitch>(UINT64_MAX-row)/(b->render_height-1))) return 0;
    const uint64_t size=(b->render_height-1)*p->game_motion_pitch+row;
    const uint64_t output_size=(b->height-1)*b->output_pitch+(uint64_t)b->width*4;
    if(p->game_motion>UINT64_MAX-size || (p->game_motion<b->output+output_size && b->output<p->game_motion+size)) return 0;
    return 1;
}

// Packed silhouette mask and per-pixel wire axes, prepared once for this frame.
typedef struct D4rNativeMotionPreparedParams
{
    D4rNativeMotionThinParams native;
    uint64_t thin_mask,wire_axes;
    uint32_t wire_flags,reserved;
} D4rNativeMotionPreparedParams;
#ifdef __cplusplus
static_assert(sizeof(D4rNativeMotionPreparedParams)==112);
#endif
static inline int d4r_native_motion_prepared_parameters_valid(const D4rNativeMotionPreparedParams* p)
{
    if(!p || !d4r_native_motion_parameters_valid(&p->native.base) || !p->thin_mask || p->thin_mask%4 ||
        !p->wire_axes || p->wire_axes%4 || (p->wire_flags & ~3u) || p->reserved ||
        !(p->native.jitter_x>=-1 && p->native.jitter_x<=1 && p->native.jitter_y>=-1 && p->native.jitter_y<=1)) return 0;
    const D4rNativeMotionParams* b=&p->native.base;
    const uint64_t mask_size=(uint64_t)((b->width+31)/32)*b->height*4,axes_size=(uint64_t)b->width*b->height;
    if(p->thin_mask>UINT64_MAX-mask_size || p->wire_axes>UINT64_MAX-axes_size ||
        (p->thin_mask<p->wire_axes+axes_size && p->wire_axes<p->thin_mask+mask_size)) return 0;
    const uint64_t pointers[5]={b->depth,b->geometry,b->scene,b->matrix,b->output};
    const uint64_t sizes[5]={(b->height-1)*b->depth_pitch+(uint64_t)b->width*4,
        (b->height-1)*b->geometry_pitch+(uint64_t)b->width*4,1136,64,
        (b->height-1)*b->output_pitch+(uint64_t)b->width*4};
    for(unsigned i=0;i<5;i++) if((p->thin_mask<pointers[i]+sizes[i] && pointers[i]<p->thin_mask+mask_size) ||
        (p->wire_axes<pointers[i]+sizes[i] && pointers[i]<p->wire_axes+axes_size)) return 0;
    return 1;
}
