#pragma once
#include <stdint.h>
#define D4R_WIRE_HISTORY_CONSERVATIVE 1u
#define D4R_WIRE_HISTORY_SKY_ONLY 2u
typedef struct D4rWireHistoryParams
{
    uint64_t current, color, native_depth, motion;
    uint64_t previous, previous_age, previous_depth;
    uint64_t output, output_age, output_depth, input_depth, output_history;
    uint64_t color_pitch, depth_pitch, motion_pitch, output_pitch, input_depth_pitch;
    uint32_t width, height, render_width, render_height, history_valid;
    float jitter_x, jitter_y;
    uint32_t reserved;
} D4rWireHistoryParams;
#ifdef __cplusplus
static_assert(sizeof(D4rWireHistoryParams)==168);
#endif
static inline int d4r_wire_history_parameters_valid(const D4rWireHistoryParams* p)
{
    if(!p || !p->width || !p->height || p->width>4096 || p->height>4096 ||
        p->render_width<12 || p->render_height<12 || p->render_width>p->width || p->render_height>p->height ||
        p->history_valid>1 || !(p->jitter_x>=-1 && p->jitter_x<=1 && p->jitter_y>=-1 && p->jitter_y<=1) ||
        !p->current || !p->color || !p->native_depth || !p->motion || !p->previous || !p->previous_age ||
        !p->previous_depth || !p->output || !p->output_age || !p->output_depth || !p->input_depth || !p->output_history ||
        p->color_pitch<(uint64_t)p->render_width*8 || p->input_depth_pitch<(uint64_t)p->render_width*4 || p->depth_pitch<(uint64_t)p->width*4 ||
        p->motion_pitch<(uint64_t)p->width*4 || p->output_pitch<(uint64_t)p->width*8 ||
        p->current==p->output || p->previous==p->output || p->previous_age==p->output_age ||
        p->previous_depth==p->output_depth || p->output_history==p->output ||
        p->output_history==p->current || p->output_history==p->previous) return 0;
    return 1;
}

// Sparse publication keeps current immutable until the separate apply dispatch.
// Scratch RGB uses the same pitch; patch_mask has one byte per output pixel.
typedef struct D4rWireHistorySparseParams
{
    D4rWireHistoryParams base;
    uint64_t patch_mask;
} D4rWireHistorySparseParams;
#ifdef __cplusplus
static_assert(sizeof(D4rWireHistorySparseParams)==176);
#endif
static inline int d4r_wire_history_sparse_parameters_valid(const D4rWireHistorySparseParams* p)
{
    uint64_t pointers[12],rows[12],pitches[12],heights[12],bytes,end;
    unsigned i;
    if(!p || !d4r_wire_history_parameters_valid(&p->base) || !p->patch_mask) return 0;
    const D4rWireHistoryParams* q=&p->base;
    bytes=(uint64_t)q->width*q->height;
    if(p->patch_mask>UINT64_MAX-bytes) return 0;
    end=p->patch_mask+bytes;
    pointers[0]=q->current;pointers[1]=q->color;pointers[2]=q->native_depth;pointers[3]=q->motion;
    pointers[4]=q->previous;pointers[5]=q->previous_age;pointers[6]=q->previous_depth;
    pointers[7]=q->output;pointers[8]=q->output_age;pointers[9]=q->output_depth;
    pointers[10]=q->input_depth;pointers[11]=q->output_history;
    for(i=0;i<12;i++) { rows[i]=(uint64_t)q->width*8;pitches[i]=q->output_pitch;heights[i]=q->height; }
    rows[1]=(uint64_t)q->render_width*8;pitches[1]=q->color_pitch;heights[1]=q->render_height;
    rows[2]=(uint64_t)q->width*4;pitches[2]=q->depth_pitch;
    rows[3]=(uint64_t)q->width*4;pitches[3]=q->motion_pitch;
    rows[5]=rows[8]=q->width;pitches[5]=pitches[8]=q->width;
    rows[6]=rows[9]=(uint64_t)q->width*4;pitches[6]=pitches[9]=(uint64_t)q->width*4;
    rows[10]=(uint64_t)q->render_width*4;pitches[10]=q->input_depth_pitch;heights[10]=q->render_height;
    for(i=0;i<12;i++)
    {
        if(heights[i]>1 && pitches[i]>(UINT64_MAX-rows[i])/(heights[i]-1)) return 0;
        bytes=(heights[i]-1)*pitches[i]+rows[i];
        if(pointers[i]>UINT64_MAX-bytes || (p->patch_mask<pointers[i]+bytes && pointers[i]<end)) return 0;
    }
    // current becomes the apply destination: every input and history output must
    // be disjoint from it, as must the correction scratch allocation.
    const uint64_t current_bytes=(q->height-1)*q->output_pitch+(uint64_t)q->width*8;
    const uint64_t scratch_bytes=current_bytes;
    for(i=1;i<12;i++)
    {
        bytes=(heights[i]-1)*pitches[i]+rows[i];
        if(q->current<pointers[i]+bytes && pointers[i]<q->current+current_bytes) return 0;
        if(i!=7 && q->output<pointers[i]+bytes && pointers[i]<q->output+scratch_bytes) return 0;
    }
    return 1;
}

// Prepared repair preserves rendered output and live history exactly. RGB/depth
// history entries with age 255 are unspecified and must never be read.
typedef struct D4rWireHistoryPreparedParams
{
    D4rWireHistorySparseParams sparse;
    uint64_t axes;
} D4rWireHistoryPreparedParams;
#ifdef __cplusplus
static_assert(sizeof(D4rWireHistoryPreparedParams)==184);
#endif
static inline int d4r_wire_history_prepared_parameters_valid(const D4rWireHistoryPreparedParams* p)
{
    if(!p || !d4r_wire_history_sparse_parameters_valid(&p->sparse) || !p->axes || (p->sparse.base.reserved & ~3u)) return 0;
    const D4rWireHistoryParams* b=&p->sparse.base;
    const uint64_t size=(uint64_t)b->width*b->height;
    if(p->axes>UINT64_MAX-size || (p->axes<p->sparse.patch_mask+size && p->sparse.patch_mask<p->axes+size)) return 0;
    const uint64_t pointers[12]={b->current,b->color,b->native_depth,b->motion,b->previous,b->previous_age,
        b->previous_depth,b->output,b->output_age,b->output_depth,b->input_depth,b->output_history};
    const uint64_t sizes[12]={(b->height-1)*b->output_pitch+(uint64_t)b->width*8,
        (b->render_height-1)*b->color_pitch+(uint64_t)b->render_width*8,
        (b->height-1)*b->depth_pitch+(uint64_t)b->width*4,(b->height-1)*b->motion_pitch+(uint64_t)b->width*4,
        (b->height-1)*b->output_pitch+(uint64_t)b->width*8,size,size*4,
        (b->height-1)*b->output_pitch+(uint64_t)b->width*8,size,size*4,
        (b->render_height-1)*b->input_depth_pitch+(uint64_t)b->render_width*4,
        (b->height-1)*b->output_pitch+(uint64_t)b->width*8};
    for(unsigned i=0;i<12;i++) if(p->axes<pointers[i]+sizes[i] && pointers[i]<p->axes+size) return 0;
    return 1;
}
