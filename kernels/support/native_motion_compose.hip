// Complete AW2 native geometry + camera composition and output-grid dilation.
// Private prototype; input buffers are immutable and no NGX routing is changed.
#include "../../tools/d4r_native_motion.h"
#pragma clang fp contract(off)
__attribute__((device,always_inline)) static inline float d4r_native_project(const float* m,int c,float x,float y,float z)
{
    float value=m[c]*x;
    value=__builtin_fmaf(m[4+c],y,value);
    value=__builtin_fmaf(m[8+c],z,value);
    return value+m[12+c];
}
__attribute__((device,always_inline)) static inline void d4r_native_motion_finish(const D4rNativeMotionParams& p,
    uint32_t x,uint32_t y,uint32_t bx,uint32_t by,float best)
{
    const uint32_t geometry=*(const uint32_t*)((const uint8_t*)p.geometry+(uint64_t)by*p.geometry_pitch+4*bx);
    const uint16_t lo=geometry,hi=geometry>>16;
    const bool lo_nan=(lo&0x7c00)==0x7c00 && (lo&0x3ff);
    const bool hi_nan=(hi&0x7c00)==0x7c00 && (hi&0x3ff);
    uint32_t result=geometry;
    if(geometry==UINT32_C(0xfbff7bff) || (lo_nan && hi_nan))
    {
        const float* scene=(const float*)p.scene;
        const float* matrix=(const float*)p.matrix;
        const float px=(bx+.5f)*((float)p.render_width/p.width);
        const float py=(by+.5f)*((float)p.render_height/p.height);
        const float cx=d4r_native_project(matrix,0,px,py,best),cy=d4r_native_project(matrix,1,px,py,best);
        const float cw=d4r_native_project(matrix,3,px,py,best);
        const float reciprocal=cw!=0.f ? 1.f/cw : 0.f;
        const _Float16 mx=(_Float16)(cx*reciprocal-scene[6]*px);
        const _Float16 my=(_Float16)(cy*reciprocal-scene[7]*py);
        result=(uint32_t)__builtin_bit_cast(uint16_t,mx) | (uint32_t)__builtin_bit_cast(uint16_t,my)<<16;
    }
    *(uint32_t*)((uint8_t*)p.output+(uint64_t)y*p.output_pitch+4*x)=result;
}

__attribute__((device,always_inline)) static inline void d4r_native_motion_compose_impl(D4rNativeMotionParams p,float jitter_x=0,float jitter_y=0)
{
    const uint32_t t=__builtin_amdgcn_workitem_id_x();
    const uint32_t x=__builtin_amdgcn_workgroup_id_x()*16+(t&15);
    const uint32_t y=__builtin_amdgcn_workgroup_id_y()*16+(t>>4);
    if(x>=p.width || y>=p.height) return;
    const uint8_t* depth=(const uint8_t*)p.depth;
    const int cx=clamp_index((int)__builtin_floorf(x+.5f+jitter_x*p.width/p.render_width),p.width);
    const int cy=clamp_index((int)__builtin_floorf(y+.5f+jitter_y*p.height/p.render_height),p.height);
    uint32_t bx=cx,by=cy;
    float best=*(const float*)(depth+(uint64_t)cy*p.depth_pitch+4*cx);
    // Same center-preserving, row-major strict-nearer tie rule as the CPU reference.
    for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++)
    {
        const int nx=cx+dx,ny=cy+dy;
        if(nx<0 || ny<0 || nx>=(int)p.width || ny>=(int)p.height) continue;
        const float z=*(const float*)(depth+(uint64_t)ny*p.depth_pitch+4*(uint32_t)nx);
        if(z<best) { best=z;bx=nx;by=ny; }
    }
    d4r_native_motion_finish(p,x,y,bx,by,best);
}

// Thin-silhouette halo: shared depth plus row/column bit masks replace repeated
// foreground run scans. Unusual footprints retain the direct depth fallback.
#define D4R_LDS_SYNC() do{__builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");}while(0)
#define D4R_LDS __attribute__((shared))
template<bool Hybrid=false>
__attribute__((device,always_inline)) static inline void d4r_native_motion_compose_thin_impl(D4rNativeMotionParams p,float jitter_x,float jitter_y,uint64_t game_motion=0,uint64_t game_pitch=0)
{
    // Align the tile to the jittered grid; halo and run checks reach at most ten pixels.
    constexpr int margin=Hybrid ? 11 : 10,span=16+2*margin;
    D4R_LDS float tile[span*span];
    D4R_LDS uint64_t candidate_row[span]; // one bit per eligible thin foreground texel
    D4R_LDS uint64_t foreground_row[span],foreground_column[span];
    D4R_LDS uint64_t wave_candidates[Hybrid ? 8 : 1];
    const uint32_t t=__builtin_amdgcn_workitem_id_x();
    const int x0=(int)__builtin_amdgcn_workgroup_id_x()*16,y0=(int)__builtin_amdgcn_workgroup_id_y()*16;
    const int gx0=x0+(int)__builtin_floorf(.5f+jitter_x*p.width/p.render_width)-margin;
    const int gy0=y0+(int)__builtin_floorf(.5f+jitter_y*p.height/p.render_height)-margin;
    const uint8_t* depth=(const uint8_t*)p.depth;
    const float outside=__builtin_inff(); // behaves as a missing neighbour in every comparison below
    for(int i=t;i<span*span;i+=256)
    {
        const int gx=gx0+i%span,gy=gy0+i/span;
        const float z=gx>=0 && gy>=0 && gx<(int)p.width && gy<(int)p.height ?
            *(const float*)(depth+(uint64_t)gy*p.depth_pitch+4*(uint32_t)gx) : outside;
        tile[i]=z;
    }
    D4R_LDS_SYNC();
    const int lane=t&31,wave=t>>5;
    // Pack the binary foreground test once in both directions. A foreground
    // center is thick on an axis iff its nine-bit neighbourhood contains a
    // contiguous run of five (all five possible runs include the center).
    for(int row=wave;row<span;row+=8)
    {
        const unsigned r0=__builtin_amdgcn_ballot_w32(tile[row*span+lane]<.99999f);
        const unsigned c0=__builtin_amdgcn_ballot_w32(tile[lane*span+row]<.99999f);
        const unsigned r1=__builtin_amdgcn_ballot_w32(lane<span-32 && tile[row*span+32+lane]<.99999f);
        const unsigned c1=__builtin_amdgcn_ballot_w32(lane<span-32 && tile[(32+lane)*span+row]<.99999f);
        if(lane==0)
        {
            foreground_row[row]=(uint64_t)r0 | (uint64_t)r1<<32;
            foreground_column[row]=(uint64_t)c0 | (uint64_t)c1<<32;
        }
    }
    D4R_LDS_SYNC();
    uint64_t nearby_candidates=0;
    for(int row=wave;row<span;row+=8)
    {
        uint64_t bits=0;
        for(int part=0;part<2;part++)
        {
            const int col=lane+32*part;
            bool thin=false;
            if(col>=4 && col<span-4 && row>=4 && row<span-4 && (foreground_row[row]>>col&1))
            {
                const unsigned horizontal=(foreground_row[row]>>(col-4))&511;
                const unsigned vertical=(foreground_column[col]>>(row-4))&511;
                const bool thick_x=(horizontal & (horizontal>>1) & (horizontal>>2) & (horizontal>>3) & (horizontal>>4))!=0;
                const bool thick_y=(vertical & (vertical>>1) & (vertical>>2) & (vertical>>3) & (vertical>>4))!=0;
                thin=!thick_x || !thick_y;
            }
            const unsigned mask=__builtin_amdgcn_ballot_w32(thin);
            bits|=(uint64_t)mask<<(32*part);
        }
        if(lane==0) candidate_row[row]=bits;
        if constexpr(Hybrid) nearby_candidates|=bits;
    }
    if constexpr(Hybrid) if(lane==0) wave_candidates[wave]=nearby_candidates;
    D4R_LDS_SYNC();
    const uint32_t x=x0+(t&15),y=y0+(t>>4);
    if(x>=p.width || y>=p.height) return;
    auto depth_at=[&](int nx,int ny)->float
    {
        const int lx=nx-gx0,ly=ny-gy0;
        if((unsigned)lx<(unsigned)span && (unsigned)ly<(unsigned)span) return tile[ly*span+lx];
        if(nx<0 || ny<0 || nx>=(int)p.width || ny>=(int)p.height) return outside;
        return *(const float*)(depth+(uint64_t)ny*p.depth_pitch+4*(uint32_t)nx);
    };
    const int cx=clamp_index((int)__builtin_floorf(x+.5f+jitter_x*p.width/p.render_width),p.width);
    const int cy=clamp_index((int)__builtin_floorf(y+.5f+jitter_y*p.height/p.render_height),p.height);
    if constexpr(Hybrid)
    {
        uint64_t any=0;for(int i=0;i<8;i++) any|=wave_candidates[i];
        bool keep_native=false;
        const bool roi_fits=cx-gx0>=10 && cx-gx0<span-10 && cy-gy0>=10 && cy-gy0<span-10;
        if(!roi_fits) keep_native=true; // unusual clamped footprints keep the complete native path
        else if(any)
        {
            const uint64_t window=UINT64_C(8191)<<(cx-gx0-6);
            for(int dy=-6;dy<=6;dy++) keep_native|=(candidate_row[cy+dy-gy0]&window)!=0;
        }
        if(!keep_native)
        {
            const unsigned mx=(unsigned)((x+.5f)*p.render_width/p.width);
            const unsigned my=(unsigned)((y+.5f)*p.render_height/p.height);
            *(uint32_t*)((uint8_t*)p.output+(uint64_t)y*p.output_pitch+4*x)=
                *(const uint32_t*)(game_motion+(uint64_t)my*game_pitch+4*mx);
            return;
        }
    }
    uint32_t bx=cx,by=cy;
    const float centre=depth_at(cx,cy);
    float best=centre;
    for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++)
    {
        const int nx=cx+dx,ny=cy+dy;
        if(nx<0 || ny<0 || nx>=(int)p.width || ny>=(int)p.height) continue;
        const float z=depth_at(nx,ny);
        if(z<best) { best=z;bx=nx;by=ny; }
    }
    // Cover the render filter footprint only around thin silhouettes against
    // sky. Long runs on both axes belong to thick surfaces and stay excluded.
    const bool tile_reaches=cx-gx0>=10 && cx-gx0<span-10 && cy-gy0>=10 && cy-gy0<span-10;
    if(centre>=.99999f)
    {
        const int sx=(p.width+p.render_width/2)/p.render_width;
        const int sy=(p.height+p.render_height/2)/p.render_height;
        const int radius=2*(sx>sy ? sx : sy)>6 ? 6 : 2*(sx>sy ? sx : sy);
        if(tile_reaches)
        {
            // Row-major, strictly nearer wins, exactly the scan below but visiting only candidate pixels.
            const int left=cx-radius-gx0;
            const uint64_t window=((uint64_t)1<<(2*radius+1))-1;
            for(int dy=-radius;dy<=radius;dy++)
            {
                const int row=cy+dy-gy0;
                uint64_t m=(candidate_row[row]>>left)&window;
                if(dy>=-1 && dy<=1) m&=~(((uint64_t)7)<<(radius-1));
                while(m)
                {
                    const int bit=__builtin_ctzll(m);
                    m&=m-1;
                    const float z=tile[row*span+left+bit];
                    if(z<best) { best=z;bx=cx-radius+bit;by=cy+dy; }
                }
            }
        }
        else
        for(int dy=-radius;dy<=radius;dy++) for(int dx=-radius;dx<=radius;dx++)
        {
            if(dx>=-1 && dx<=1 && dy>=-1 && dy<=1) continue;
            const int nx=cx+dx,ny=cy+dy;
            if(nx<0 || ny<0 || nx>=(int)p.width || ny>=(int)p.height) continue;
            const float z=depth_at(nx,ny);
            if(!(z<best && z<.99999f)) continue;
            bool thin=false;
            for(int axis=0;axis<2 && !thin;axis++)
            {
                int run=1;
                for(int sign=-1;sign<=1;sign+=2)
                    for(int step=1;step<5 && run<5;step++)
                    {
                        const int tx=nx+(axis==0 ? sign*step : 0),ty=ny+(axis==1 ? sign*step : 0);
                        if(tx<0 || ty<0 || tx>=(int)p.width || ty>=(int)p.height ||
                            !(depth_at(tx,ty)<.99999f)) break;
                        run++;
                    }
                thin=run<5;
            }
            if(thin) { best=z;bx=nx;by=ny; }
        }
    }
    d4r_native_motion_finish(p,x,y,bx,by,best);
}

extern "C" __attribute__((global)) void d4r_native_motion_compose(D4rNativeMotionParams p)
{ d4r_native_motion_compose_impl(p); }
extern "C" __attribute__((global, amdgpu_flat_work_group_size(256, 256))) void d4r_native_motion_compose_thin(D4rNativeMotionThinParams p)
{ d4r_native_motion_compose_thin_impl(p.base,p.jitter_x,p.jitter_y); }

// Alignment without the extra silhouette halo; wire-color repair is independent.
extern "C" __attribute__((global, amdgpu_flat_work_group_size(256,256))) void d4r_native_motion_compose_aligned(D4rNativeMotionThinParams p)
{ d4r_native_motion_compose_impl(p.base,p.jitter_x,p.jitter_y); }

extern "C" __attribute__((global, amdgpu_flat_work_group_size(256,256))) void d4r_native_motion_compose_hybrid(D4rNativeMotionHybridParams p)
{ d4r_native_motion_compose_thin_impl<true>(p.native.base,p.native.jitter_x,p.native.jitter_y,p.game_motion,p.game_motion_pitch); }
