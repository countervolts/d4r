// Geometry-confirmed short-lived color history for locally separated thin silhouettes.
#include "../../tools/d4r_wire_history.h"
#pragma clang fp contract(off)
#define WH_SYNC() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup"); } while(0)
__attribute__((device,always_inline)) static inline int wh_clamp(int v,int lo,int hi)
{ return v<lo ? lo : v>hi ? hi : v; }
__attribute__((device,always_inline)) static inline float wh_depth(const D4rWireHistoryParams& p,int x,int y)
{ x=wh_clamp(x,0,p.width-1);y=wh_clamp(y,0,p.height-1);return *(const float*)(p.native_depth+(uint64_t)y*p.depth_pitch+4*x); }
template<class Depth>
__attribute__((device,always_inline)) static inline bool wh_thin_axis(const D4rWireHistoryParams& p,int x,int y,int axis,Depth depth)
{
    float center=depth(x,y);if(!(center>=0 && center<.99999f)) return false;
    if((p.reserved & D4R_WIRE_HISTORY_SKY_ONLY) &&
        (!(depth(x-4*(axis==0),y-4*(axis==1))>=.99999f) || !(depth(x+4*(axis==0),y+4*(axis==1))>=.99999f))) return false;
    float epsilon=.25f*(1-center);if(epsilon<2e-6f) epsilon=2e-6f;
    float limit=center+epsilon;
    int run=1;
    for(int sign=-1;sign<=1;sign+=2) for(int step=1;step<5 && run<5;step++)
    {
        int tx=x+(axis==0 ? sign*step : 0),ty=y+(axis==1 ? sign*step : 0);
        if(tx<0 || ty<0 || tx>=(int)p.width || ty>=(int)p.height || !(depth(tx,ty)<limit)) break;
        run++;
    }
    if(run>=5) return false;
    float gap=.5f*(1-center);if(gap<2e-6f) gap=2e-6f;
    bool sides[2]={false,false};
    for(int sign=-1;sign<=1;sign+=2) for(int step=1;step<=4;step++)
        sides[(sign+1)/2]|=depth(x+sign*step*(axis==0),y+sign*step*(axis==1))>center+gap;
    return sides[0] && sides[1];
}
// Interior specialization: fixed eight depth loads, bit scans for the first
// non-near neighbour on each side, and the original strict separation test.
// Edge samples keep the bounds-aware reference above.
template<class Depth>
__attribute__((device,always_inline)) static inline bool wh_thin_interior(const D4rWireHistoryParams& p,int x,int y,int axis,Depth depth)
{
    if(x<4 || y<4 || x>=(int)p.width-4 || y>=(int)p.height-4)
        return wh_thin_axis(p,x,y,axis,depth);
    const float center=depth(x,y);
    if(!(center>=0 && center<.99999f)) return false;
    if((p.reserved & D4R_WIRE_HISTORY_SKY_ONLY) &&
        (!(depth(x-4*(axis==0),y-4*(axis==1))>=.99999f) || !(depth(x+4*(axis==0),y+4*(axis==1))>=.99999f))) return false;
    float epsilon=.25f*(1-center);if(epsilon<2e-6f) epsilon=2e-6f;
    float gap=.5f*(1-center);if(gap<2e-6f) gap=2e-6f;
    const float limit=center+epsilon,far_limit=center+gap;
    unsigned negative=0,positive=0;
    bool far_negative=false,far_positive=false;
    #pragma unroll
    for(int step=1;step<=4;step++)
    {
        const float n=depth(x-(axis==0 ? step : 0),y-(axis==1 ? step : 0));
        negative|=(unsigned)(!(n<limit))<<(step-1);
        far_negative|=n>far_limit;
    }
    // Four contiguous near neighbours already prove a run of at least five.
    // No positive-side depths or separation checks can make it thin.
    if(!negative) return false;
    #pragma unroll
    for(int step=1;step<=4;step++)
    {
        const float v=depth(x+(axis==0 ? step : 0),y+(axis==1 ? step : 0));
        positive|=(unsigned)(!(v<limit))<<(step-1);
        far_positive|=v>far_limit;
    }
    if(!positive) return false;
    return __builtin_ctz(negative|16)+__builtin_ctz(positive|16)<4 && far_negative && far_positive;
}
__attribute__((device,always_inline)) static inline float wh_luma(uint64_t rgb,uint64_t pitch,int x,int y)
{
    const _Float16* c=(const _Float16*)(rgb+(uint64_t)y*pitch+8*x);
    float l=(float)c[0]*.2126f+(float)c[1]*.7152f+(float)c[2]*.0722f;
    return l>1e-5f ? l : 1e-5f;
}
// Median in log space is the geometric mean of the two central luminances.
__attribute__((device,always_inline)) static inline float wh_background(uint64_t rgb,uint64_t pitch,int x,int y,int ax,int ay,bool input)
{
    float v[6];int count=input ? 4 : 6;
    const int offsets[6]={-12,-10,-8,8,10,12};
    const int small[4]={-5,-4,4,5};
    for(int i=0;i<count;i++) { int d=input ? small[i] : offsets[i];v[i]=wh_luma(rgb,pitch,x+d*ax,y+d*ay); }
    for(int i=1;i<count;i++) for(int j=i;j>0 && v[j]<v[j-1];j--) {float a=v[j];v[j]=v[j-1];v[j-1]=a;}
    return __builtin_sqrtf(v[count/2-1]*v[count/2]);
}
template<bool Sparse,bool CachedAxes=false>
__attribute__((device,always_inline)) static inline void wh_run(D4rWireHistoryParams p,uint64_t patch_mask,uint64_t prepared_axes=0)
{
    uint32_t t=__builtin_amdgcn_workitem_id_x();
    int x=__builtin_amdgcn_workgroup_id_x()*16+(t&15),y=__builtin_amdgcn_workgroup_id_y()*16+(t>>4);
    // Classify each nearby depth texel once, shared by the four bilinear samples
    // of adjacent output pixels. All lanes reach both barriers, including partial tiles.
    constexpr int span=26,margin=5;
    __attribute__((shared)) float tile[span*span];
    __attribute__((shared)) uint8_t axes[span*span];
    const int ox=(int)__builtin_amdgcn_workgroup_id_x()*16+(int)__builtin_floorf(p.jitter_x*p.width/p.render_width)-margin;
    const int oy=(int)__builtin_amdgcn_workgroup_id_y()*16+(int)__builtin_floorf(p.jitter_y*p.height/p.render_height)-margin;
    if constexpr(!CachedAxes) {
        for(int i=t;i<span*span;i+=256) tile[i]=wh_depth(p,ox+i%span,oy+i/span);
        WH_SYNC();
    }
    auto depth=[&](int tx,int ty)->float {
        int lx=tx-ox,ly=ty-oy;
        return (unsigned)lx<span && (unsigned)ly<span ? tile[ly*span+lx] : wh_depth(p,tx,ty);
    };
    auto tile_depth=[&](int tx,int ty)->float { return tile[(ty-oy)*span+tx-ox]; };
    if constexpr(!CachedAxes) {
    for(int i=t;i<span*span;i+=256)
    {
        int lx=i%span,ly=i/span;
        uint8_t mask=0;
        if(lx>=4 && lx<span-4 && ly>=4 && ly<span-4)
        {
            if(wh_thin_interior(p,ox+lx,oy+ly,0,tile_depth)) mask|=1;
            if(wh_thin_interior(p,ox+lx,oy+ly,1,tile_depth)) mask|=2;
        }
        axes[i]=mask;
    }
    WH_SYNC();
    }
    auto thin_axes=[&](int tx,int ty)->uint8_t {
        if constexpr(CachedAxes) return *(const uint8_t*)(prepared_axes+(uint64_t)ty*p.width+tx);
        else {
        int lx=tx-ox,ly=ty-oy;
        if(lx>=4 && lx<span-4 && ly>=4 && ly<span-4) return axes[ly*span+lx];
        return (wh_thin_axis(p,tx,ty,0,depth) ? 1 : 0) | (wh_thin_axis(p,tx,ty,1,depth) ? 2 : 0);
        }
    };
    if(x>=(int)p.width || y>=(int)p.height) return;
    uint64_t offset=(uint64_t)y*p.output_pitch+8*x,index=(uint64_t)y*p.width+x;
    if constexpr(Sparse) *(uint8_t*)(patch_mask+index)=0;
    else *(uint64_t*)(p.output+offset)=*(const uint64_t*)(p.current+offset);
    // Age 255 makes this entry unreadable by the next frame. The prepared
    // path need not copy raw RGB or initialize depth for an unobserved entry.
    if constexpr(!CachedAxes) *(uint64_t*)(p.output_history+offset)=*(const uint64_t*)(p.current+offset);
    *(uint8_t*)(p.output_age+index)=255;
    if constexpr(!CachedAxes) *(float*)(p.output_depth+4*index)=1.f;
    if(x<16 || y<16 || x>=(int)p.width-16 || y>=(int)p.height-16) return;
    float nx=x+p.jitter_x*p.width/p.render_width,ny=y+p.jitter_y*p.height/p.render_height;
    int fx=(int)__builtin_floorf(nx),fy=(int)__builtin_floorf(ny),sx=0,sy=0;
    float coverage=0,best_weight=-1;
    for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
    {
        int tx=wh_clamp(fx+dx,0,p.width-1),ty=wh_clamp(fy+dy,0,p.height-1);
        if(!thin_axes(tx,ty)) continue;
        float weight=(1-__builtin_fabsf(nx-tx))*(1-__builtin_fabsf(ny-ty));coverage+=weight;
        if(weight>best_weight) { best_weight=weight;sx=tx;sy=ty; }
    }
    if(coverage<=.05f) return;
    int ay=(thin_axes(sx,sy)&2) ? 1 : 0,ax=1-ay;
    const bool conservative=(p.reserved & D4R_WIRE_HISTORY_CONSERVATIVE)!=0;
    const int radius=conservative ? 1 : 2,age_limit=conservative ? 4 : 16;
    float current_z=wh_depth(p,sx,sy);
    float tolerance=(conservative ? .01f : .02f)*(1-current_z);
    const float depth_floor=conservative ? 2e-7f : 2e-6f;
    if(tolerance<depth_floor) tolerance=depth_floor;
    if(p.reserved & D4R_WIRE_HISTORY_SKY_ONLY)
    {
        // Fractional wire coverage may overlap unrelated foreground geometry.
        // Preserve that surface rather than dragging wire color onto it. Sky
        // and depth-matched wire centers remain eligible; NaN depth is rejected.
        const float nearest_z=wh_depth(p,(int)__builtin_floorf(nx+.5f),(int)__builtin_floorf(ny+.5f));
        if(!(nearest_z>=.99999f) && !(__builtin_fabsf(nearest_z-current_z)<tolerance)) return;
    }
    *(float*)(p.output_depth+4*index)=current_z;
    int ix=wh_clamp((int)__builtin_floorf((x+.5f)*p.render_width/p.width+p.jitter_x),5,p.render_width-6);
    int iy=wh_clamp((int)__builtin_floorf((y+.5f)*p.render_height/p.height+p.jitter_y),5,p.render_height-6);
    float source_x=(x+.5f)*p.render_width/p.width+p.jitter_x;
    float source_y=(y+.5f)*p.render_height/p.height+p.jitter_y;

    float best_distance=1e30f;int source_ix=ix,source_iy=iy;
    for(int dy=-radius;dy<=radius;dy++) for(int dx=-radius;dx<=radius;dx++)
    {
        int tx=ix+dx,ty=iy+dy;
        float value=*(const float*)(p.input_depth+(uint64_t)ty*p.input_depth_pitch+4*tx);
        if(!(__builtin_fabsf(value-current_z)<tolerance)) continue;
        float xx=tx+.5f-source_x,yy=ty+.5f-source_y,distance=xx*xx+yy*yy;
        if(distance<best_distance) {best_distance=distance;source_ix=tx;source_iy=ty;}
    }
    uint64_t target=0;
    if(best_distance!=1e30f)
    {
        target=p.color+(uint64_t)source_iy*p.color_pitch+8*source_ix;
        *(uint64_t*)(p.output_history+offset)=*(const uint64_t*)target;
        *(uint8_t*)(p.output_age+index)=0;
    }
    else if(p.history_valid)
    {
        const _Float16* motion=(const _Float16*)(p.motion+(uint64_t)y*p.motion_pitch+4*x);
        float px=x+(float)motion[0]*p.width,py=y+(float)motion[1]*p.height;
        if(!(px>=16 && py>=16 && px<p.width-16 && py<p.height-16)) return;
        float best=1e30f;int hx=0,hy=0;
        for(int dy=-radius;dy<=radius;dy++) for(int dx=-radius;dx<=radius;dx++)
        {
            int tx=wh_clamp((int)__builtin_rintf(px+dx),16,p.width-17),ty=wh_clamp((int)__builtin_rintf(py+dy),16,p.height-17);
            uint64_t previous_index=(uint64_t)ty*p.width+tx;
            if(*(const uint8_t*)(p.previous_age+previous_index)>=age_limit) continue;
            float previous_z=*(const float*)(p.previous_depth+4*previous_index),limit=(conservative ? .01f : .05f)*(1-current_z);if(limit<depth_floor) limit=depth_floor;
            if(!(__builtin_fabsf(previous_z-current_z)<limit)) continue;
            // Brightness is not correspondence: selecting the darkest neighbour can
            // drag a previous wire sample sideways along a camera pan.
            const float ddx=tx-px,ddy=ty-py;
            float value=conservative ? ddx*ddx+ddy*ddy : wh_luma(p.previous,p.output_pitch,tx,ty);
            if(value<best) {best=value;hx=tx;hy=ty;}
        }
        if(best!=1e30f)
        {
            uint64_t hi=(uint64_t)hy*p.width+hx;
            target=p.previous+(uint64_t)hy*p.output_pitch+8*hx;
            *(uint64_t*)(p.output_history+offset)=*(const uint64_t*)target;
            *(uint8_t*)(p.output_age+index)=*(const uint8_t*)(p.previous_age+hi)+1;
        }
    }
    if(!target) return;
    float background=wh_background(p.current,p.output_pitch,x,y,ax,ay,false);
    float current=wh_luma(p.current,p.output_pitch,x,y);
    float value=wh_luma(target,8,0,0),ratio=value/background;
    float threshold=background*__builtin_sqrtf(ratio*__builtin_sqrtf(ratio));
    if(!((ratio<.8408964153f && current>threshold) || (ratio>1.189207115f && current<threshold))) return;
    if constexpr(Sparse)
    {
        *(uint64_t*)(p.output+offset)=*(const uint64_t*)(p.current+offset);
        *(uint8_t*)(patch_mask+index)=1;
    }
    if(coverage>1) coverage=1;
    const _Float16* observed=(const _Float16*)target;
    const _Float16* now=(_Float16*)(p.current+offset);_Float16* out=(_Float16*)(p.output+offset);
    for(int c=0;c<3;c++)
    {
        float result=(float)now[c]*(1-coverage)+(float)observed[c]*coverage;
        if(result<0) result=0;if(result>65504) result=65504;out[c]=(_Float16)result;
    }
}

extern "C" __attribute__((global, amdgpu_flat_work_group_size(256,256))) void d4r_wire_history(D4rWireHistoryParams p)
{ wh_run<false>(p,0); }
extern "C" __attribute__((global, amdgpu_flat_work_group_size(256,256))) void d4r_wire_history_sparse(D4rWireHistorySparseParams p)
{ wh_run<true>(p.base,p.patch_mask); }
extern "C" __attribute__((global, amdgpu_flat_work_group_size(256,256))) void d4r_wire_history_apply(D4rWireHistorySparseParams p)
{
    const unsigned t=__builtin_amdgcn_workitem_id_x();
    const unsigned x=__builtin_amdgcn_workgroup_id_x()*16+(t&15),y=__builtin_amdgcn_workgroup_id_y()*16+(t>>4);
    if(x>=p.base.width || y>=p.base.height || !*(const uint8_t*)(p.patch_mask+(uint64_t)y*p.base.width+x)) return;
    const uint64_t offset=(uint64_t)y*p.base.output_pitch+8*x;
    *(uint64_t*)(p.base.current+offset)=*(const uint64_t*)(p.base.output+offset);
}
#undef WH_SYNC

extern "C" __attribute__((global,amdgpu_flat_work_group_size(256,256))) void d4r_wire_history_preclassified(D4rWireHistoryPreparedParams p)
{ wh_run<true,true>(p.sparse.base,p.sparse.patch_mask,p.axes); }
