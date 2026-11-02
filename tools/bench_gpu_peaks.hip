#include <hip/hip_runtime.h>
#include <cstdio>
typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
__global__ void wmma(float* o,int it){
  v2i a={(int)threadIdx.x,1},b={2,(int)threadIdx.x}; v8f c0={},c1={},c2={},c3={},c4={},c5={},c6={},c7={};
  for(int i=0;i<it;i++){
#define W(c) c=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c);
    W(c0)W(c1)W(c2)W(c3)W(c4)W(c5)W(c6)W(c7)
  }
  v8f s=c0+c1+c2+c3+c4+c5+c6+c7; if(s[0]==1234.f) o[threadIdx.x]=s[1];
}
__global__ void rd(const float4* __restrict__ p,float* o,size_t n){
  float4 s={}; for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){float4 v=p[i];s.x+=v.x;s.y+=v.y;s.z+=v.z;s.w+=v.w;}
  if(s.x==1234.f) o[0]=s.y;
}
__global__ void cp(const float4* __restrict__ p,float4* q,size_t n){
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) q[i]=p[i];
}
int main(){
  float* o; hipMalloc(&o,4096); hipEvent_t e0,e1; hipEventCreate(&e0);hipEventCreate(&e1); float ms;
  int it=20000, blocks=64*64, thr=256;
  for(int r=0;r<3;r++){ hipEventRecord(e0); wmma<<<blocks,thr>>>(o,it); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&ms,e0,e1);
    double fl=(double)blocks*(thr/32)*it*8*16*16*16*2; printf("FP8 WMMA dense: %.1f TFLOPS\n",fl/ms/1e9);}
  size_t n=(size_t)1<<26; float4 *p,*q; hipMalloc(&p,n*16); hipMalloc(&q,n*16); hipMemset(p,0,n*16);
  for(int r=0;r<3;r++){ hipEventRecord(e0); rd<<<64*32,256>>>(p,o,n); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&ms,e0,e1); printf("read: %.0f GB/s\n",n*16/ms/1e6);}
  for(int r=0;r<3;r++){ hipEventRecord(e0); cp<<<64*32,256>>>(p,q,n); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&ms,e0,e1); printf("copy: %.0f GB/s\n",2*n*16/ms/1e6);}
}
