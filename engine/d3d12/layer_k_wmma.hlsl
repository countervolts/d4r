// Plain preset K block, using the same 16x16 matrix tiles as ../layer.comp.
// Row-major extracted weights, existing 448-byte parameters; no repacking.
// AGS reserved-space protocol works on an ordinary AMD D3D12 device.
#ifndef C
#define C 160
#endif
#ifndef NH
#define NH 8
#endif
#if C < 64 || C > 160 || C % 32 != 0 || NH < 2 || NH > 8
#error Unsupported plain matrix K shape
#endif
// Keep the measured baseline as the default until Linux/RADV is revalidated.
#ifndef WMMA_WAVE_SYNC
#define WMMA_WAVE_SYNC 0
#endif
#ifndef WMMA_PREFETCH
#define WMMA_PREFETCH 0
#endif
#define SX (C + 8)
#define SQ 40
#define BF_WORDS (64 * SX / 2)
#define D4R_WMMA_SHARED_WORDS (BF_WORDS + 64 * SQ)
#include "ags_wmma.hlsli"
#include "k_math.hlsli"
ByteAddressBuffer parameters : register(t0);
ByteAddressBuffer weights : register(t1);
ByteAddressBuffer inputs : register(t2);
RWByteAddressBuffer output : register(u0);
groupshared uint waveCounter;
uint offset(uint i) { return parameters.Load(16 + 4*i); }
Matrix16 weight(uint type, uint halfOffset, uint halfStride)
{ return matrixLoad(type, weights, halfOffset*2, halfStride*2); }
Matrix16 bias(uint halfOffset, uint halfStride)
{ return matrixConvert(MatrixAcc16, MatrixAcc32, weight(MatrixAcc16, halfOffset, halfStride)); }
Matrix16 bload(uint type, uint row, uint col)
{ return matrixSharedLoad(type, (row*SX+col)/2, SX/2); }
void bstore(Matrix16 m, uint row, uint col)
{ matrixSharedStore(MatrixAcc16, (row*SX+col)/2, SX/2, matrixConvert(MatrixAcc32, MatrixAcc16, m)); }
Matrix16 tload(uint type, uint row, uint col)
{ return matrixSharedLoad(type, BF_WORDS+(row*SQ+col)/2, SQ/2); }
void tstore(Matrix16 m, uint row, uint col)
{ matrixSharedStore(MatrixAcc16, BF_WORDS+(row*SQ+col)/2, SQ/2, matrixConvert(MatrixAcc32, MatrixAcc16, m)); }
void waveMemoryFence()
{
#if WMMA_WAVE_SYNC
    // Fixed wave32, converged execution: the producer/consumer addresses here
    // belong only to this wave's sixteen token rows. Keep LDS memory ordering,
    // but do not wait for other waves. Q/K/V sharing still uses group sync.
    GroupMemoryBarrier();
#else
    GroupMemoryBarrierWithGroupSync();
#endif
}
[WaveSize(32)]
[numthreads(128,1,1)]
void main(uint thread : SV_GroupIndex, uint3 group : SV_GroupID)
{
    if(thread==0) waveCounter=0;
    GroupMemoryBarrierWithGroupSync();
    uint index=0;
    if(WaveIsFirstLane()) InterlockedAdd(waveCounter,1,index);
    const uint wave=WaveReadLaneFirst(index), lane=WaveGetLaneIndex();
    const uint tok=wave*16+(lane&15), halfChannel=(lane>>4)*(C/2);
    const int W=asint(parameters.Load(0)), H=asint(parameters.Load(4));
    const int X0=8*int(group.x)-asint(parameters.Load(8)), Y0=8*int(group.y)-asint(parameters.Load(12));
    const int x=X0+int(tok&7), y=Y0+int(tok>>3);
    const uint src=uint(mirror(y,H)*W+mirror(x,W))*C+halfChannel;
    uint residual[C/4];
    float2 ss=float2(0,0);
    [unroll] for(uint d=0;d<C/4;++d)
    {
        const uint v=inputs.Load(2*(src+2*d)); residual[d]=v;
        const float2 f=float2(unpack(v)); ss+=f*f;
    }
    float sum=ss.x+ss.y; sum+=WaveReadLaneAt(sum,lane^16);
    const float16_t inv=float16_t(rsqrt(float(float16_t(sum))));
    [unroll] for(uint d=0;d<C/4;++d)
        matrixShared[(tok*SX+halfChannel)/2+d]=pack(unpack(residual[d])*inv);
    waveMemoryFence(); // each wave loads only its own normalized rows
    Matrix16 xn[C/16], acc[C/16];
    [unroll] for(uint k=0;k<C/16;++k)
    { xn[k]=bload(MatrixA16,wave*16,k*16); acc[k]=bias(offset(1)+k*16,C); }
    [loop] for(uint h=0;h<NH;++h)
    {
        // Q, K and V. Each wave owns sixteen token rows.
        [unroll] for(uint proj=0;proj<3;++proj)
        {
            Matrix16 p0=matrixFill(MatrixAcc32,0), p1=matrixFill(MatrixAcc32,0);
#if WMMA_PREFETCH
            [unroll] for(uint k=0;k<C/16;k+=2)
            {
                const uint ow=offset(4+5*h+proj)+k*16*32;
                Matrix16 b0=weight(MatrixB16,ow,32), b1=weight(MatrixB16,ow+16,32);
                Matrix16 b2=weight(MatrixB16,ow+512,32), b3=weight(MatrixB16,ow+528,32);
                p0=matrixMultiplyAdd(xn[k],b0,p0); p1=matrixMultiplyAdd(xn[k],b1,p1);
                p0=matrixMultiplyAdd(xn[k+1],b2,p0); p1=matrixMultiplyAdd(xn[k+1],b3,p1);
            }
#else
            [unroll] for(uint k=0;k<C/16;++k)
            {
                const uint ow=offset(4+5*h+proj)+k*16*32;
                p0=matrixMultiplyAdd(xn[k],weight(MatrixB16,ow,32),p0);
                p1=matrixMultiplyAdd(xn[k],weight(MatrixB16,ow+16,32),p1);
            }
#endif
            if(proj==0) { bstore(p0,wave*16,0); bstore(p1,wave*16,16); }
            else { const uint row=(proj==2 ? 64 : 0)+wave*16; tstore(p0,row,0); tstore(p1,row,16); }
        }
        GroupMemoryBarrierWithGroupSync(); // all waves consume all K/V rows
        Matrix16 scores[4], q0=bload(MatrixA16,wave*16,0), q1=bload(MatrixA16,wave*16,16);
        [unroll] for(uint n=0;n<4;++n)
        {
            scores[n]=bias(offset(7+5*h)+wave*16*64+n*16,64);
            scores[n]=matrixMultiplyAdd(q0,tload(MatrixB16|0x80,n*16,0),scores[n]);
            scores[n]=matrixMultiplyAdd(q1,tload(MatrixB16|0x80,n*16,16),scores[n]);
            bstore(scores[n],wave*16,n*16);
        }
        waveMemoryFence(); // scores and softmax cover this wave's own rows
        float16_t2 e[16]; float2 esum=float2(0,0);
        const uint scoreWord=(tok*SX+(lane>>4)*32)/2;
        [unroll] for(uint j=0;j<16;++j) { e[j]=softmax_exp(unpack(matrixShared[scoreWord+j])); esum+=float2(e[j]); }
        float total=esum.x+esum.y; total+=WaveReadLaneAt(total,lane^16);
        const float16_t r=float16_t(1.0/float(float16_t(total)));
        [unroll] for(uint j=0;j<16;++j) matrixShared[scoreWord+j]=pack(e[j]*r);
        waveMemoryFence(); // probabilities stay in this wave's own rows
        Matrix16 o0=matrixFill(MatrixAcc32,0), o1=matrixFill(MatrixAcc32,0);
        [unroll] for(uint k=0;k<4;++k)
        {
            Matrix16 p=bload(MatrixA16,wave*16,k*16);
            o0=matrixMultiplyAdd(p,tload(MatrixB16,64+k*16,0),o0);
            o1=matrixMultiplyAdd(p,tload(MatrixB16,64+k*16,16),o1);
        }
        bstore(o0,wave*16,0); bstore(o1,wave*16,16);
        waveMemoryFence();
        Matrix16 oa0=bload(MatrixA16,wave*16,0), oa1=bload(MatrixA16,wave*16,16);
#if WMMA_PREFETCH
        [unroll] for(uint n=0;n<C/16;n+=2)
        {
            const uint ow=offset(8+5*h)+n*16;
            Matrix16 b0=weight(MatrixB16,ow,C), b1=weight(MatrixB16,ow+16*C,C);
            Matrix16 b2=weight(MatrixB16,ow+16,C), b3=weight(MatrixB16,ow+16*C+16,C);
            acc[n]=matrixMultiplyAdd(oa0,b0,acc[n]); acc[n]=matrixMultiplyAdd(oa1,b1,acc[n]);
            acc[n+1]=matrixMultiplyAdd(oa0,b2,acc[n+1]); acc[n+1]=matrixMultiplyAdd(oa1,b3,acc[n+1]);
        }
#else
        [unroll] for(uint n=0;n<C/16;++n)
        {
            const uint ow=offset(8+5*h)+n*16;
            acc[n]=matrixMultiplyAdd(oa0,weight(MatrixB16,ow,C),acc[n]);
            acc[n]=matrixMultiplyAdd(oa1,weight(MatrixB16,ow+16*C,C),acc[n]);
        }
#endif
        GroupMemoryBarrierWithGroupSync(); // finish every V reader before reuse
    }
    [unroll] for(uint n=0;n<C/16;++n) bstore(acc[n],wave*16,n*16);
    waveMemoryFence();
    [unroll] for(uint d=0;d<C/4;++d)
    {
        const uint i=(tok*SX+halfChannel)/2+d;
        matrixShared[i]=pack(unpack(matrixShared[i])+unpack(residual[d]));
    }
    waveMemoryFence();
    [unroll] for(uint k=0;k<C/16;++k) xn[k]=bload(MatrixA16,wave*16,k*16);
    // Keep the f16 MLP input in xn, then initialize its residual plus b2.
    [unroll] for(uint d=0;d<C/4;++d)
    {
        const uint i=(tok*SX+halfChannel)/2+d;
        matrixShared[i]=pack(unpack(matrixShared[i])+unpack(weights.Load(2*(offset(3)+halfChannel+2*d))));
    }
    waveMemoryFence();
    [unroll] for(uint k=0;k<C/16;++k)
        acc[k]=matrixConvert(MatrixAcc16,MatrixAcc32,bload(MatrixAcc16,wave*16,k*16));
    [loop] for(uint ch=0;ch<C/8;++ch)
    {
        const uint w1=offset(44+3*ch), b1=offset(45+3*ch), w2=offset(46+3*ch);
        Matrix16 h0=bias(b1,32), h1=bias(b1+16,32);
#if WMMA_PREFETCH
        [unroll] for(uint k=0;k<C/16;k+=2)
        {
            const uint ow=w1+k*16*32;
            Matrix16 b0=weight(MatrixB16,ow,32), b1_=weight(MatrixB16,ow+16,32);
            Matrix16 b2=weight(MatrixB16,ow+512,32), b3=weight(MatrixB16,ow+528,32);
            h0=matrixMultiplyAdd(xn[k],b0,h0); h1=matrixMultiplyAdd(xn[k],b1_,h1);
            h0=matrixMultiplyAdd(xn[k+1],b2,h0); h1=matrixMultiplyAdd(xn[k+1],b3,h1);
        }
#else
        [unroll] for(uint k=0;k<C/16;++k)
        {
            h0=matrixMultiplyAdd(xn[k],weight(MatrixB16,w1+k*16*32,32),h0);
            h1=matrixMultiplyAdd(xn[k],weight(MatrixB16,w1+k*16*32+16,32),h1);
        }
#endif
        tstore(h0,wave*16,0); tstore(h1,wave*16,16);
#if WMMA_PREFETCH
        // Request the first output tiles before GELU, as in layer.comp.
        Matrix16 w20=weight(MatrixB16,w2,C), w21=weight(MatrixB16,w2+16*C,C);
        Matrix16 w22=weight(MatrixB16,w2+16,C), w23=weight(MatrixB16,w2+16*C+16,C);
#endif
        waveMemoryFence(); // each wave owns its MLP hidden rows
        [unroll] for(uint j=0;j<8;++j)
        {
            const uint i=BF_WORDS+(tok*SQ+(lane>>4)*16)/2+j;
            matrixShared[i]=pack(gelu(unpack(matrixShared[i])));
        }
        waveMemoryFence();
        Matrix16 a0=tload(MatrixA16,wave*16,0), a1=tload(MatrixA16,wave*16,16);
#if WMMA_PREFETCH
        acc[0]=matrixMultiplyAdd(a0,w20,acc[0]); acc[0]=matrixMultiplyAdd(a1,w21,acc[0]);
        acc[1]=matrixMultiplyAdd(a0,w22,acc[1]); acc[1]=matrixMultiplyAdd(a1,w23,acc[1]);
        [unroll] for(uint n=2;n<C/16;n+=2)
        {
            Matrix16 b0=weight(MatrixB16,w2+n*16,C), b1_=weight(MatrixB16,w2+16*C+n*16,C);
            Matrix16 b2=weight(MatrixB16,w2+(n+1)*16,C), b3=weight(MatrixB16,w2+16*C+(n+1)*16,C);
            acc[n]=matrixMultiplyAdd(a0,b0,acc[n]); acc[n]=matrixMultiplyAdd(a1,b1_,acc[n]);
            acc[n+1]=matrixMultiplyAdd(a0,b2,acc[n+1]); acc[n+1]=matrixMultiplyAdd(a1,b3,acc[n+1]);
        }
#else
        [unroll] for(uint n=0;n<C/16;++n)
        {
            acc[n]=matrixMultiplyAdd(a0,weight(MatrixB16,w2+n*16,C),acc[n]);
            acc[n]=matrixMultiplyAdd(a1,weight(MatrixB16,w2+16*C+n*16,C),acc[n]);
        }
#endif
    }
    [unroll] for(uint n=0;n<C/16;++n) bstore(acc[n],wave*16,n*16);
    waveMemoryFence();
    if(x>=0 && x<W && y>=0 && y<H)
        [unroll] for(uint d=0;d<C/4;++d)
            output.Store(2*(uint(y*W+x)*C+halfChannel+2*d),matrixShared[(tok*SX+halfChannel)/2+d]);
}
