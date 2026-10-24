// HIP GPU regression for gfx11 L's FP8 boundary and texture-PTX QK grouping.
// Requires gfx11; no NVIDIA DLL/data. Build with cached ROCm's HIP clang.
#define SWIN_GFX11_ROW
#define SWIN_GFX11_SCALED
#define SWIN_EXACT
#define SWIN_TOK_NATURAL
#include "../m/swin_common.h"
#include "../m/swin_tok.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) { std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e)); std::exit(1); } } while(0)
__global__ void qk_layout(uint16_t* output) {
    const int lane=threadIdx.x, hf=lane>>4;
    h8 a,b;
#pragma unroll
    for(int i=0;i<8;++i) {
        a[i]=(half_t)((lane&15)*32+8*hf+i);
        b[i]=(half_t)((lane&15)*32+16+8*hf+i);
    }
#pragma unroll
    for(int step=0;step<2;++step) {
        const h8 q=tok_qk_operand(a,b,step);
#pragma unroll
        for(int i=0;i<8;++i) output[(step*32+lane)*8+i]=(uint16_t)q[i];
    }
}
__global__ void decode_boundary(uint16_t* output) {
    const unsigned code=threadIdx.x;
    const uint32_t packed=code*0x01010101u;
    const h4 decoded=tok_decode4(packed);
    typedef uint16_t u4s __attribute__((ext_vector_type(4)));
    const u4s d=__builtin_bit_cast(u4s,decoded);
    const h8 operand=tok_operand(packed,packed);
    typedef uint16_t u8s __attribute__((ext_vector_type(8)));
    const u8s o=__builtin_bit_cast(u8s,operand);
#pragma unroll
    for(int i=0;i<4;++i) output[code*12+i]=d[i];
#pragma unroll
    for(int i=0;i<8;++i) output[code*12+4+i]=o[i];
}
__global__ void nan_canonicalization(uint16_t* output) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=2046) return;
    const uint16_t input=uint16_t(0x7c01u+i%1023u+(i>=1023 ? 0x8000u : 0));
    typedef uint16_t u8s __attribute__((ext_vector_type(8)));
    const gop h=__builtin_bit_cast(gop,(u8s){input,input,input,input,input,input,input,input});
    gop quantized;
#pragma unroll
    for(int k=0;k<4;++k) {
        const hv2 v=q8x2_exact<0x7f80>((hv2){h[2*k],h[2*k+1]});
        quantized[2*k]=v[0]; quantized[2*k+1]=v[1];
    }
    const u8s result=__builtin_bit_cast(u8s,tok_scaled_operand(quantized));
#pragma unroll
    for(int j=0;j<8;++j) output[8*i+j]=result[j];
}
int main() {
    uint16_t* d; HIP(hipMalloc(&d,2046*8*sizeof(uint16_t)));
    qk_layout<<<1,32>>>(d); HIP(hipGetLastError());
    std::vector<uint16_t> q(2*32*8); HIP(hipMemcpy(q.data(),d,q.size()*2,hipMemcpyDeviceToHost));
    for(int step=0;step<2;++step) for(int lane=0;lane<32;++lane) for(int i=0;i<8;++i) {
        // Original m16n8k32 texture fragments interleave the low/high 16
        // pair-order Q columns, with columns 8..15 in the second k16.
        const int half=lane>>4, k=8*half+i;
        const int column=8*step+2*(k/4)+(k&1)+16*((k>>1)&1);
        const int expected=(lane&15)*32+column;
        if(q[(step*32+lane)*8+i]!=expected) {
            std::fprintf(stderr,"QK step=%d lane=%d slot=%d got=%u expected=%d\n",step,lane,i,q[(step*32+lane)*8+i],expected);return 1;
        }
    }
    decode_boundary<<<1,256>>>(d); HIP(hipGetLastError());
    std::vector<uint16_t> b(256*12); HIP(hipMemcpy(b.data(),d,b.size()*2,hipMemcpyDeviceToHost));
    for(unsigned code=0;code<256;++code) {
        const unsigned sign=(code&128u)<<8, mag=code&127u;
        const uint16_t scaled=uint16_t(sign | ((mag | ((mag+1)&128u))<<7));
        const float v=mag<8 ? mag/512.0f : __builtin_ldexpf(float(8+(mag&7)),int(mag/8)-10);
        const _Float16 h=(_Float16)(code&128u ? -v : v);
        const uint16_t decoded=__builtin_bit_cast(uint16_t,h);
        for(int i=0;i<12;++i) {
            const uint16_t got=b[code*12+i],want=i<4 ? decoded : scaled;
            const uint16_t expected = mag==127 && i<4 ? uint16_t(sign|0x7fff) : want;
            if(got!=expected) {
                std::fprintf(stderr,"FP8 code=%02x field=%d got=%04x expected=%04x\n",code,i,got,expected);return 1;
            }
        }
    }
    nan_canonicalization<<<8,256>>>(d); HIP(hipGetLastError());
    std::vector<uint16_t> n(2046*8); HIP(hipMemcpy(n.data(),d,n.size()*2,hipMemcpyDeviceToHost));
    for(unsigned i=0;i<2046;++i) for(int j=0;j<8;++j)
        if(n[8*i+j]!=(i>=1023 ? 0xff80 : 0x7f80)) return 1;
    HIP(hipFree(d));
    std::puts("512 QK slots, all 256 FP8 codes and 2046 NaN payloads match the texture contracts");
}
