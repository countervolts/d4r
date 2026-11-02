#include <cstdio>
#include <cuda_runtime.h>
// FP8 e4m3 mma.sync m16n8k32: 16*8*32*2 = 8192 FLOP per warp-instruction
template<int ACC16> __global__ void mma8(float* o,int it){
  unsigned a0=threadIdx.x,a1=1,a2=2,a3=3,b0=4,b1=threadIdx.x;
  float c[8][4]={}; unsigned h[8][2]={};
  for(int i=0;i<it;i++){
#pragma unroll
    for(int k=0;k<8;k++){
      if(ACC16) asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
        :"+r"(h[k][0]),"+r"(h[k][1]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
      else asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
  }
  float s=0; for(int k=0;k<8;k++){s+=c[k][0]+c[k][1]+c[k][2]+c[k][3]+h[k][0]+h[k][1];} if(s==1234.f) o[threadIdx.x]=s;
}
// FP16 m16n8k16, f32 accumulate: 4096 FLOP
__global__ void mma16(float* o,int it){
  unsigned a0=threadIdx.x,a1=1,a2=2,a3=3,b0=4,b1=threadIdx.x; float c[8][4]={};
  for(int i=0;i<it;i++){
#pragma unroll
    for(int k=0;k<8;k++) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  float s=0; for(int k=0;k<8;k++) s+=c[k][0]+c[k][1]+c[k][2]+c[k][3]; if(s==1234.f) o[threadIdx.x]=s;
}
__global__ void cp(const float4* __restrict__ p,float4* q,size_t n){
  for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) q[i]=p[i];
}
__global__ void rd(const float4* __restrict__ p,float* o,size_t n){
  float4 s={}; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){float4 v=p[i];s.x+=v.x;s.y+=v.y;s.z+=v.z;s.w+=v.w;}
  if(s.x==1234.f) o[0]=s.y;
}
int main(){
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0); printf("%s SMs=%d\n",pr.name,pr.multiProcessorCount);
  float* o; cudaMalloc(&o,4096); cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;
  int blocks=pr.multiProcessorCount*8, thr=256, it=20000;
  auto T=[&](const char* n,auto k,double flop_per){ for(int r=0;r<5;r++){ cudaEventRecord(e0); k<<<blocks,thr>>>(o,it); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
      printf("%s: %.1f TFLOPS\n",n,(double)blocks*(thr/32)*it*8*flop_per/ms/1e9);} };
  T("FP8 e4m3 mma, f32 acc",mma8<0>,8192); T("FP8 e4m3 mma, f16 acc",mma8<1>,8192); T("FP16 mma, f32 acc",mma16,4096);
  size_t n=(size_t)1<<26; float4 *p,*q; cudaMalloc(&p,n*16); cudaMalloc(&q,n*16); cudaMemset(p,0,n*16);
  for(int r=0;r<5;r++){ cudaEventRecord(e0); rd<<<pr.multiProcessorCount*16,512>>>(p,o,n); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1); printf("read: %.0f GB/s\n",n*16/ms/1e6);}
  for(int r=0;r<5;r++){ cudaEventRecord(e0); cp<<<pr.multiProcessorCount*16,512>>>(p,q,n); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1); printf("copy: %.0f GB/s\n",2*n*16/ms/1e6);}
  printf("err: %s\n",cudaGetErrorString(cudaGetLastError()));
}
