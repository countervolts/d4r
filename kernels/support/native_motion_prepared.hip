// Exact native composition with shared wire geometry; wave64 compose, wave32 preparation.
#pragma clang fp contract(off)
#define NMP_SYNC() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup"); } while(0)
__attribute__((device,always_inline,target("wavefrontsize64"))) static inline float d4r_nmp_project64(const float* m,int c,float x,float y,float z)
{
    float value=m[c]*x;
    value=__builtin_fmaf(m[4+c],y,value);
    value=__builtin_fmaf(m[8+c],z,value);
    return value+m[12+c];
}
__attribute__((device,always_inline,target("wavefrontsize64"))) static inline void d4r_nmp_finish64(const D4rNativeMotionParams& p,
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
        const float cx=d4r_nmp_project64(matrix,0,px,py,best),cy=d4r_nmp_project64(matrix,1,px,py,best);
        const float cw=d4r_nmp_project64(matrix,3,px,py,best);
        const float reciprocal=cw!=0.f ? 1.f/cw : 0.f;
        const _Float16 mx=(_Float16)(cx*reciprocal-scene[6]*px);
        const _Float16 my=(_Float16)(cy*reciprocal-scene[7]*py);
        result=(uint32_t)__builtin_bit_cast(uint16_t,mx) | (uint32_t)__builtin_bit_cast(uint16_t,my)<<16;
    }
    *(uint32_t*)((uint8_t*)p.output+(uint64_t)y*p.output_pitch+4*x)=result;
}

__attribute__((device,always_inline,target("wavefrontsize64"))) static inline unsigned d4r_nmp_bits64(const unsigned* data,int words,int left)
{
    int word=left>=0 ? left/32 : -1,shift=left&31;
    unsigned lo=word>=0 && word<words ? data[word] : 0;
    unsigned hi=word+1>=0 && word+1<words ? data[word+1] : 0;
    return (unsigned)(((uint64_t)lo | (uint64_t)hi<<32)>>shift);
}
__attribute__((device,always_inline,target("wavefrontsize64"))) static inline int d4r_nmp_clamp64(int x,unsigned n) {return x<0 ? 0 : x>=(int)n ? (int)n-1 : x;}
extern "C" __attribute__((global,amdgpu_flat_work_group_size(256,256),target("wavefrontsize64"))) void d4r_native_motion_prepared(D4rNativeMotionPreparedParams a)
{
    const auto& p=a.native.base;
    unsigned t=__builtin_amdgcn_workitem_id_x();
    int x0=__builtin_amdgcn_workgroup_id_x()*16,y0=__builtin_amdgcn_workgroup_id_y()*16;
    int left=x0+(int)__builtin_floorf(.5f+a.native.jitter_x*p.width/p.render_width)-8;
    int top=y0+(int)__builtin_floorf(.5f+a.native.jitter_y*p.height/p.render_height)-8;
    const int rw=(p.width+31)/32;
    __attribute__((shared)) unsigned rows[32],any;
    if(t<32) {
        int y=top+t;unsigned bits=y>=0 && y<(int)p.height ? d4r_nmp_bits64((const unsigned*)a.thin_mask+(uint64_t)y*rw,rw,left) : 0;
        rows[t]=bits;unsigned nonempty=(unsigned)__builtin_amdgcn_ballot_w64(bits!=0);if(t==0) any=nonempty;
    }
    NMP_SYNC();
    unsigned x=x0+(t&15),y=y0+(t>>4);
    if(x>=p.width || y>=p.height) return;
    const int cx=d4r_nmp_clamp64((int)__builtin_floorf(x+.5f+a.native.jitter_x*p.width/p.render_width),p.width);
    const int cy=d4r_nmp_clamp64((int)__builtin_floorf(y+.5f+a.native.jitter_y*p.height/p.render_height),p.height);
    const uint8_t* depth=(const uint8_t*)p.depth;
    const float center=*(const float*)(depth+(uint64_t)cy*p.depth_pitch+4*cx);
    float best=center;unsigned bx=cx,by=cy;
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
        int nx=cx+dx,ny=cy+dy;if(nx<0 || ny<0 || nx>=(int)p.width || ny>=(int)p.height)continue;
        float z=*(const float*)(depth+(uint64_t)ny*p.depth_pitch+4*nx);
        if(z<best){best=z;bx=nx;by=ny;}
    }
    if(center>=.99999f) {
        int sx=(p.width+p.render_width/2)/p.render_width,sy=(p.height+p.render_height/2)/p.render_height;
        int radius=2*(sx>sy ? sx : sy);if(radius>6)radius=6;
        bool cached=cx-radius>=left && cx+radius<left+32 && cy-radius>=top && cy+radius<top+32;
        if(any || !cached)for(int dy=-radius;dy<=radius;dy++) {
            int ny=cy+dy;if(ny<0 || ny>=(int)p.height)continue;
            unsigned m=(cached ? rows[ny-top]>>(cx-radius-left) : d4r_nmp_bits64((const unsigned*)a.thin_mask+(uint64_t)ny*rw,rw,cx-radius)) & ((1u<<(2*radius+1))-1);
            if(dy>=-1 && dy<=1)m&=~(7u<<(radius-1));
            while(m) {int bit=__builtin_ctz(m);m&=m-1;int nx=cx-radius+bit;
                float z=*(const float*)(depth+(uint64_t)ny*p.depth_pitch+4*nx);
                if(z<best){best=z;bx=nx;by=ny;}
            }
        }
    }
    d4r_nmp_finish64(p,x,y,bx,by,best);
}

extern "C" __attribute__((global,amdgpu_flat_work_group_size(256,256))) void d4r_native_motion_prepare(D4rNativeMotionPreparedParams a)
{
    const auto& p=a.native.base;
    __attribute__((shared)) float z[40][41];
    __attribute__((shared)) uint64_t row_bits[40],col_bits[40];
    unsigned t=__builtin_amdgcn_workitem_id_x(),lane=t&31,wave=t>>5;
    const int x0=(int)__builtin_amdgcn_workgroup_id_x()*32,y0=(int)__builtin_amdgcn_workgroup_id_y()*32;
    for(int i=t;i<1600;i+=256) {
        int ly=i/40,lx=i%40,gx=clamp_index(x0+lx-4,p.width),gy=clamp_index(y0+ly-4,p.height);
        z[ly][lx]=*(const float*)(p.depth+(uint64_t)gy*p.depth_pitch+4*gx);
    }
    NMP_SYNC();
    for(int r=wave;r<40;r+=8) {
        uint64_t rb=0,cb=0;
        for(int part=0;part<2;part++) {
            int c=lane+32*part;
            bool rf=c<40 && x0+c-4>=0 && x0+c-4<(int)p.width && y0+r-4>=0 && y0+r-4<(int)p.height && z[r][c]<.99999f;
            bool cf=c<40 && y0+c-4>=0 && y0+c-4<(int)p.height && x0+r-4>=0 && x0+r-4<(int)p.width && z[c][r]<.99999f;
            rb|=(uint64_t)__builtin_amdgcn_ballot_w32(rf)<<(part*32);cb|=(uint64_t)__builtin_amdgcn_ballot_w32(cf)<<(part*32);
        }
        if(lane==0){row_bits[r]=rb;col_bits[r]=cb;}
    }
    NMP_SYNC();
    D4rWireHistoryParams wp={};wp.width=p.width;wp.height=p.height;wp.native_depth=p.depth;wp.depth_pitch=p.depth_pitch;wp.reserved=a.wire_flags;
    auto depth=[&](int gx,int gy)->float {return z[gy-y0+4][gx-x0+4];};
    for(int r=wave;r<32;r+=8) {
        const int gx=x0+lane,gy=y0+r,lx=lane+4,ly=r+4;
        unsigned h=(row_bits[ly]>>lane)&511,v=(col_bits[lx]>>r)&511;
        bool hx=(h&(h>>1)&(h>>2)&(h>>3)&(h>>4))!=0,hy=(v&(v>>1)&(v>>2)&(v>>3)&(v>>4))!=0;
        bool thin=(h&16)!=0 && (!hx || !hy);
        unsigned bits=__builtin_amdgcn_ballot_w32(thin);
        if(lane==0 && gy<(int)p.height) ((unsigned*)a.thin_mask)[(uint64_t)gy*((p.width+31)/32)+x0/32]=bits;
        if(a.wire_axes && gx<(int)p.width && gy<(int)p.height) {
            // Existing foreground bits reject sky centers and non-sky endpoints
            // without repeating LDS depth reads. The reference still handles NaNs,
            // edge clamping and relative-depth separation for every surviving axis.
            unsigned axes=0;
            if(h&16) {
                const bool sky_only=(a.wire_flags & D4R_WIRE_HISTORY_SKY_ONLY)!=0;
                if(!sky_only || !(h&257)) axes|=wh_thin_interior(wp,gx,gy,0,depth)?1u:0u;
                if(!sky_only || !(v&257)) axes|=wh_thin_interior(wp,gx,gy,1,depth)?2u:0u;
            }
            ((uint8_t*)a.wire_axes)[(uint64_t)gy*p.width+gx]=axes;
        }
    }
}

#undef NMP_SYNC
