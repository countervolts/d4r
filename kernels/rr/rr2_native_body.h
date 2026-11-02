// Preset F blocks with heads and MLP branches spread over groups of four waves.
// Every lane retains only its channel tiles; heads and branches accumulate in order.
#ifndef RRVK_HS
#define RRVK_HS 2
#endif
#define HS RRVK_HS
static_assert(HS == 1 || HS == 2 || HS == 4, "F blocks support 1, 2 or 4 head groups");
#define MT (C/16)
#define NW (4*HS)
#define MP ((MT+HS-1)/HS)
#define XW (C/4+2)
#define EW (C+2)
#define KW 10
#define VW 18
#define OW 10
#define KVW (64*KW+32*VW)
#define SLOTW (KVW+64*OW)
#define NEEDW ((16*EW>2176)?16*EW:2176)
#define POOL_WORDS ((NEEDW > HS*SLOTW) ? NEEDW : HS*SLOTW)
extern "C" __global__ __launch_bounds__(NW*32) void RRVK_NAME(Args args) {
#ifdef RRVK_SAT_MODE
    __builtin_amdgcn_s_setreg(1 | (23 << 6), 1);
#endif
    const uint* __restrict__ p=(const uint*)(const __attribute__((address_space(1))) uint*)args.param;
    const uint8_t* __restrict__ w=(const uint8_t*)(const __attribute__((address_space(1))) uint8_t*)args.weights;
    auto prm=[&] __device__ (uint i) { return __builtin_amdgcn_readfirstlane(p[i]); };
    const auto* wf=(const float*)w;
    const auto* wh=(const uint16_t*)w;
    uint* __restrict__ arena=(uint*)(__attribute__((address_space(1))) uint*)args.arena;
    const uint lane=threadIdx.x&31, wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
    const uint tid=32*wave+lane,q=wave&3,group=wave>>2,col=lane&15,hl=lane>>4;
    const uint ex=prm(0), ey=prm(1), hx=ex/2, hy=ey/2, shift=prm(20);
    const int x0=8*int(blockIdx.x)-int(shift&65535), y0=8*int(blockIdx.y)-int(shift>>16);
    const uint plane=ex*ey*32, lowplane=hx*hy*32;
    const uint t=inter(16*q+col);
    const int tx=x0+int(t&7), ty=y0+int(t>>3);
    const bool inside=tx>=0 && ty>=0 && tx<int(ex) && ty<int(ey);
    const _Float16 s0=hb(prm(23)&65535), s1=hb(prm(23)>>16), s2=hb(prm(24)&65535), s3=hb(prm(24)>>16);
    __shared__ uint mem[64*XW+POOL_WORDS+128];
    uint* X=mem;
    uint* pool=mem+64*XW;
    uint* ob=pool+HS*KVW;
    _Float16* guides=(_Float16*)(pool+POOL_WORDS);
    h8 xr[MP];
    auto quant8=[] __device__ (f8 v) {
        B8 b; b.packed=uint64_t(packfinite4f(v[0],v[1],v[2],v[3])) | (uint64_t(packfinite4f(v[4],v[5],v[6],v[7]))<<32); return b;
    };
    auto mma=[] __device__ (ACC& c,A8 a0,B8 b0,A8 a1,B8 b1) {
        c=coopMatMulAdd(a1,b1,coopMatMulAdd(a0,b0,c));
    };
    auto bias=[&] __device__ (uint off) {
        ACC a; const f8 v=*(const f8*)(wf+off/4+8*hl);
        return __builtin_bit_cast(ACC,v);
    };
    auto weight=[&] __device__ (uint off,uint k,uint m,uint kt) {
        const fp8* base=(const fp8*)(w+__builtin_amdgcn_readfirstlane(off+256*(m*(k/16)+kt)));
        A8 a; coopMatLoad(a,base,0,16,RM);return a;
    };
#if KIND==0 || KIND==3
    if(tid<64) {
        uint k=inter(tid);
        uint at=(prm(2)+uint(mirror(y0+int(k>>3),ey)*ex+mirror(x0+int(k&7),ex))*32)/4;
        for(uint wd=0;wd<C/4;++wd) X[tid*XW+wd]=arena[at+(wd>>3)*(plane/4)+(wd&7)];
    }
    barrier();
#elif KIND==1
    if(tid<64) {
        uint k=inter(tid); int xx=mirror(x0+int(k&7),ex), yy=mirror(y0+int(k>>3),ey);
        for(uint sub=0;sub<4;++sub) {
            uint at=(prm(2)+uint((2*yy+int(sub>>1))*(2*ex)+2*xx+int(sub&1))*32)/4;
            for(uint wd=0;wd<8;++wd) pool[tid*34+sub*8+wd]=arena[at+wd];
        }
    }
    barrier();
    #pragma unroll
    for(uint j=0;j<MP;++j) {
        uint m=group+HS*j;
        if(m<MT) {
            ACC a=bias(W_B1_F+64*m);
            #pragma unroll
            for(uint kt=0;kt<8;kt+=2) {
                B8 b0,b1;
                coopMatLoad(b0,pool,q*16*34+4*kt,34,CMJ);
                coopMatLoad(b1,pool,q*16*34+4*kt+4,34,CMJ);
                mma(a,weight(W_W1,128,m,kt),b0,weight(W_W1,128,m,kt+1),b1);
            }
            B8 b=quant8(__builtin_bit_cast(f8,a)); coopMatStore(b,X,q*16*XW+4*m,XW,CMJ);
        }
    }
    barrier();
#elif KIND==2
    const int X0=x0>>1,Y0=y0>>1;
    for(uint i=tid;i<4*CL;i+=NW*32) {
        uint row=i/(CL/4),wd=i%(CL/4);
        int xx=mirror(X0+int(row&3),hx),yy=mirror(Y0+int(row>>2),hy);
        X[row*(CL/4+2)+wd]=arena[(prm(2)+(wd>>3)*lowplane+uint(yy*hx+xx)*32)/4+(wd&7)];
    }
    barrier();
    for(uint m=wave;m<4*MT;m+=NW) {
        ACC a=bias(W_BE_F+64*m);
        #pragma unroll
        for(uint kt=0;kt<CL/16;kt+=2) {
            B8 b0,b1; coopMatLoad(b0,X,4*kt,CL/4+2,CMJ);coopMatLoad(b1,X,4*kt+4,CL/4+2,CMJ);
            mma(a,weight(W_WE,CL,m,kt),b0,weight(W_WE,CL,m,kt+1),b1);
        }
        B8 b=quant8(__builtin_bit_cast(f8,a));coopMatStore(b,pool,4*m,EW,CMJ);
    }
    barrier();
    #pragma unroll
    for(uint j=0;j<MP;++j) {
        uint m=group+HS*j;
        if(m<MT) {
            uint at=(prm(4)+uint(mirror(ty,ey)*ex+mirror(tx,ex))*32)/4;
            uint row=uint(((ty>>1)-Y0)*4+((tx>>1)-X0)),sub=uint(2*(ty&1)+(tx&1));
            #pragma unroll
            for(uint h=0;h<2;++h) {
                uint wd=4*m+2*hl+h;
                uint sk=arena[at+(wd>>3)*(plane/4)+(wd&7)], e=pool[row*EW+sub*(C/4)+wd];
                #pragma unroll
                for(uint pair=0;pair<2;++pair) {
                    f16vec2 prod=fp8pair(sk,pair)*f16vec2(s3);
                    f16vec2 val=fp8pair(e,pair)+prod;
                    xr[j][4*h+2*pair]=val.x;xr[j][4*h+2*pair+1]=val.y;
                }
            }
        }
    }
    // Every wave has consumed LOW in X before X becomes the expanded tokens.
    barrier();
    #pragma unroll
    for(uint j=0;j<MP;++j) {
        uint m=group+HS*j;
        if(m<MT) {
            f8 v;for(int i=0;i<8;++i)v[i]=float(xr[j][i]);
            B8 b=quant8(v);coopMatStore(b,X,q*16*XW+4*m,XW,CMJ);
        }
    }
    barrier();
#else
#error unsupported F block kind
#endif
    ACC at[MP];
    #pragma unroll
    for(uint j=0;j<MP;++j) if(group+HS*j<MT) at[j]=bias(W_BO_F+64*(group+HS*j));
    const uint HEAD=W_WQK1-W_WQK0,BRANCH=W_WD1-W_WD0;
    for(uint base=0;base<NH;base+=HS) {
        uint hd=base+group;
        uint* kv=pool+group*KVW;
        if(hd<NH) {
            for(uint job=q;job<8;job+=4) {
                uint n=job>>1,c=job&1;
                ACC k(0.f),v(0.f);
                #pragma unroll
                for(uint kt=0;kt<MT;kt+=2) {
                    A8 xa0,xa1;B8 xb0,xb1,vw0,vw1;
                    coopMatLoad(xa0,X,n*16*XW+4*kt,XW,RM);coopMatLoad(xa1,X,n*16*XW+4*kt+4,XW,RM);
                    coopMatLoad(xb0,X,n*16*XW+4*kt,XW,CMJ);coopMatLoad(xb1,X,n*16*XW+4*kt+4,XW,CMJ);
                    vw0.packed=weight(W_WV0+HEAD*hd,32,kt,c).packed;
                    vw1.packed=weight(W_WV0+HEAD*hd,32,kt+1,c).packed;
                    mma(k,weight(W_WQK0+HEAD*hd,C,c,kt),xb0,weight(W_WQK0+HEAD*hd,C,c,kt+1),xb1);
                    mma(v,xa0,vw0,xa1,vw1);
                }
                B8 kb=quant8(__builtin_bit_cast(f8,k)),vb=quant8(__builtin_bit_cast(f8,v));
                A8 ka,va;ka.packed=kb.packed;va.packed=vb.packed;
                coopMatStore(ka,kv,n*16*KW+4*c,KW,RM);
                coopMatStore(va,kv,64*KW+c*16*VW+4*n,VW,RM);
            }
        }
        barrier();
        if(hd<NH) {
            B8 qb[2],pb[4];
            coopMatLoad(qb[0],kv,q*16*KW,KW,CMJ);coopMatLoad(qb[1],kv,q*16*KW+4,KW,CMJ);
            f16vec2 ep[4][4];
            #pragma unroll
            for(uint kt=0;kt<4;++kt) {
                A8 a0,a1;ACC c;
                coopMatLoad(a0,kv,kt*16*KW,KW,RM);coopMatLoad(a1,kv,kt*16*KW+4,KW,RM);
                coopMatLoad(c,wf,(W_BIAS0_F+HEAD*hd)/4+(kt*4+q)*256,16,CMJ);
                mma(c,a0,qb[0],a1,qb[1]);
                #pragma unroll
                for(uint i=0;i<4;++i) ep[kt][i]=f16vec2(hb(wh[W_CUBIC/2+bh((_Float16)c[2*i])]),hb(wh[W_CUBIC/2+bh((_Float16)c[2*i+1])]));
            }
            f16vec2 st[2];
            #pragma unroll
            for(uint tq=0;tq<2;++tq) {
                uint q0=2*tq,q1=q0+1;
                f16vec2 even=(ep[0][q0]+ep[0][q1])+(ep[2][q0]+ep[2][q1]);
                f16vec2 odd=(ep[1][q0]+ep[1][q1])+(ep[3][q0]+ep[3][q1]);
                st[tq]=even+odd;
            }
            f16vec2 loc=st[0]+st[1];uint other=subgroupShuffleXor(packFloat2x16(loc),16);
            f16vec2 sums=hl==0?loc+unpackFloat2x16(other):unpackFloat2x16(other)+loc;
            _Float16 rs=sums.x+sums.y,r=(_Float16)__builtin_amdgcn_rcpf(float(rs));
            #pragma unroll
            for(uint kt=0;kt<4;++kt) {
                uint64_t bits=0;
                #pragma unroll
                for(uint i=0;i<4;++i) {
                    f16vec2 z=ep[kt][i]*f16vec2(r);
                    fe4m3vec2 b=quant(z); bits|=uint64_t(code(b.x)|(code(b.y)<<8))<<(16*i);
                }
                pb[kt].packed=bits;
            }
            #pragma unroll
            for(uint c=0;c<2;++c) {
                ACC o(0.f);
                #pragma unroll
                for(uint ks=0;ks<2;++ks) {
                    A8 v0,v1;coopMatLoad(v0,kv,64*KW+c*16*VW+8*ks,VW,RM);coopMatLoad(v1,kv,64*KW+c*16*VW+8*ks+4,VW,RM);
                    mma(o,v0,pb[2*ks],v1,pb[2*ks+1]);
                }
                B8 b=quant8(__builtin_bit_cast(f8,o));coopMatStore(b,ob,group*64*OW+q*16*OW+4*c,OW,CMJ);
            }
        }
        barrier();
        for(uint h=0;h<HS && base+h<NH;++h) {
            B8 o0,o1;coopMatLoad(o0,ob,h*64*OW+q*16*OW,OW,CMJ);coopMatLoad(o1,ob,h*64*OW+q*16*OW+4,OW,CMJ);
            #pragma unroll
            for(uint j=0;j<MP;++j) {
                uint m=group+HS*j;
                if(m<MT) mma(at[j],weight(W_WO0+HEAD*(base+h),32,m,0),o0,weight(W_WO0+HEAD*(base+h),32,m,1),o1);
            }
        }
        barrier();
    }
    f8 rv[MP];
    #pragma unroll
    for(uint j=0;j<MP;++j) {
        uint m=group+HS*j;
        if(m<MT) {
            #pragma unroll
            for(uint i=0;i<8;++i) {
#if KIND==2
                float x=float(xr[j][i]);
#else
                uint wd=X[(16*q+col)*XW+4*m+2*hl+(i>>2)];
                float x=float(fp8::bits(wd>>(8*(i&3))));
#endif
                rv[j][i]=__builtin_fmaf(at[j][i],float(s1),x);
            }
        }
    }
    // All X reads are complete before the distributed residual overwrites X.
    barrier();
    #pragma unroll
    for(uint j=0;j<MP;++j) if(group+HS*j<MT) {
        B8 b=quant8(rv[j]);coopMatStore(b,X,q*16*XW+4*(group+HS*j),XW,CMJ);
    }
    barrier();
    ACC ml[MP];
    #pragma unroll
    for(uint j=0;j<MP;++j) if(group+HS*j<MT) ml[j]=bias(W_BU_F+64*(group+HS*j));
    for(uint base=0;base<NB;base+=HS) {
        uint br=base+group;
        if(br<NB) {
            #pragma unroll
            for(uint c=0;c<2;++c) {
                ACC h=bias(W_BD0_F+BRANCH*br+64*c);
                #pragma unroll
                for(uint kt=0;kt<MT;kt+=2) {
                    B8 b0,b1;coopMatLoad(b0,X,q*16*XW+4*kt,XW,CMJ);coopMatLoad(b1,X,q*16*XW+4*kt+4,XW,CMJ);
                    mma(h,weight(W_WD0+BRANCH*br,C,c,kt),b0,weight(W_WD0+BRANCH*br,C,c,kt+1),b1);
                }
                f8 a;
                #pragma unroll
                for(uint i=0;i<8;++i) {
                    float x=h[i],y=__builtin_fminf(__builtin_fmaxf(x,-2.f),2.f);
                    float g=__builtin_fmaf(y,__builtin_fmaf(-__builtin_fabsf(y),.0810546875f,.412109375f),.5f);
                    a[i]=x*g;
                }
                B8 b=quant8(a);coopMatStore(b,ob,group*64*OW+q*16*OW+4*c,OW,CMJ);
            }
        }
        barrier();
        for(uint b=0;b<HS && base+b<NB;++b) {
            B8 h0,h1;coopMatLoad(h0,ob,b*64*OW+q*16*OW,OW,CMJ);coopMatLoad(h1,ob,b*64*OW+q*16*OW+4,OW,CMJ);
            #pragma unroll
            for(uint j=0;j<MP;++j) {
                uint m=group+HS*j;
                if(m<MT) mma(ml[j],weight(W_WU0+BRANCH*(base+b),32,m,0),h0,weight(W_WU0+BRANCH*(base+b),32,m,1),h1);
            }
        }
        barrier();
    }
    #pragma unroll
    for(uint j=0;j<MP;++j) {
        uint m=group+HS*j;
        if(m<MT) {
            f8 v;for(uint i=0;i<8;++i)v[i]=__builtin_fmaf(ml[j][i],float(s2),rv[j][i]);
            B8 b=quant8(v);coopMatStore(b,X,q*16*XW+4*m,XW,CMJ);
            if(inside) {
                uint o=(prm(8)+uint(ty*int(ex)+tx)*32)/4+(m>>1)*(plane/4)+4*(m&1)+2*hl;
                arena[o]=uint(b.packed);arena[o+1]=uint(b.packed>>32);
            }
        }
    }
    barrier();
#if GUIDE || KIND==3
    if(wave<4) {
        ACC h=bias(W_BH_F);
        #pragma unroll
        for(uint kt=0;kt<MT;kt+=2) {
            B8 b0,b1;coopMatLoad(b0,X,q*16*XW+4*kt,XW,CMJ);coopMatLoad(b1,X,q*16*XW+4*kt+4,XW,CMJ);
            mma(h,weight(W_WH,C,0,kt),b0,weight(W_WH,C,0,kt+1),b1);
        }
        if(hl==0) {
            f16vec4 out=f16vec4(0.f);
            if(inside) {
                auto guide=[&] __device__ (uint off,int x,int y,int width) {
                    uint a=(off+uint(y*width+x)*8)/4;
                    uint b0=arena[a],b1=arena[a+1];return f16vec4(hb(b0),hb(b0>>16),hb(b1),hb(b1>>16));
                };
                f16vec3 gate=f16vec3(sig((_Float16)h[0]),sig((_Float16)h[1]),sig((_Float16)h[2]))*s0;
#if KIND==3
                int bx=tx==0?0:(tx-1)>>1,by=ty==0?0:(ty-1)>>1;
                _Float16 wx=tx==0?(_Float16)0:((tx&1)?(_Float16).25:(_Float16).75);
                _Float16 wy=ty==0?(_Float16)0:((ty&1)?(_Float16).25:(_Float16).75);
                _Float16 ux=(_Float16)1-wx,uy=(_Float16)1-wy;
                int xx=min(bx+1,int(hx)-1),yy=min(by+1,int(hy)-1);
                f16vec4 p00=guide(prm(12),bx,by,hx)*ux,p10=guide(prm(12),xx,by,hx)*wx;
                f16vec4 p01=guide(prm(12),bx,yy,hx)*ux,p11=guide(prm(12),xx,yy,hx)*wx;
                f16vec4 a10=p10*uy,u0=p00*uy+a10,u1=p01*wy+u0,up=p11*wy+u1;
                _Float16 al=sig((_Float16)h[3]),ia=(_Float16)1-al;
                f16vec4 gt=up*f16vec4(gate,(_Float16)0),pg=gt*al;
                f16vec4 mix=guide(prm(14),tx,ty,ex)*ia+pg;
                out=min(mix,f16vec4(65504.f));
#else
                out=min(guide(prm(12),tx,ty,ex)*f16vec4(gate,(_Float16)0),f16vec4(65504.f));
#endif
                uint off=(prm(16)+uint(ty*int(ex)+tx)*8)/4;
                arena[off]=bh(out.x)|(bh(out.y)<<16);arena[off+1]=bh(out.z)|(bh(out.w)<<16);
            }
            for(uint c=0;c<4;++c) guides[t*4+c]=out[c];
        }
    }
#endif
    barrier();
#if CM
    if(tid<64) {
        uint tt=inter(tid),yy=tt>>3,xx=tt&7;
        uint row=(yy>>1)<2?2*(xx>>1)+(yy>>1):8+2*(xx>>1)+(yy>>1)-2;
        uint sub=2*(yy&1)+(xx&1);
        for(uint wd=0;wd<C/4;++wd) pool[row*EW+sub*(C/4)+wd]=X[tid*XW+wd];
    }
    barrier();
    for(uint m=wave;m<CM/16;m+=NW) {
        ACC a=bias(W_BM_F+64*m);
        for(uint ks=0;ks<C/8;++ks) {
            B8 b0,b1;coopMatLoad(b0,pool,8*ks,EW,CMJ);coopMatLoad(b1,pool,8*ks+4,EW,CMJ);
            mma(a,weight(W_WM,4*C,m,2*ks),b0,weight(W_WM,4*C,m,2*ks+1),b1);
        }
        B8 b=quant8(__builtin_bit_cast(f8,a));
        uint mx=col<8?col>>1:(col-8)>>1,my=col<8?col&1:2+(col&1);
        int xx=x0/2+int(mx),yy=y0/2+int(my);
        if(xx>=0&&yy>=0&&xx<int(hx)&&yy<int(hy)) {
            uint off=(prm(10)+uint(yy*int(hx)+xx)*32)/4+(m>>1)*(lowplane/4)+4*(m&1)+2*hl;
            arena[off]=uint(b.packed);arena[off+1]=uint(b.packed>>32);
        }
    }
#if GUIDE
    if(wave==0 && lane<16) {
        uint mx=lane&3,my=lane>>2;int xx=x0+2*mx,yy=y0+2*my;
        if(xx>=0&&yy>=0&&xx<int(ex)&&yy<int(ey)) {
            uint t00=2*my*8+2*mx;f16vec4 v[4];
            for(uint d=0;d<4;++d) {
                uint tt=t00+(d&1)+8*(d>>1);
                v[d]=f16vec4(guides[tt*4],guides[tt*4+1],guides[tt*4+2],guides[tt*4+3])*(_Float16).25;
            }
            f16vec4 h0=v[0]+v[1],h1=v[2]+v[3],hm=h0+h1;
            uint off=(prm(18)+uint((yy/2)*int(hx)+xx/2)*8)/4;
            arena[off]=bh(hm.x)|(bh(hm.y)<<16);arena[off+1]=bh(hm.z)|(bh(hm.w)<<16);
        }
    }
#endif
#endif
}
