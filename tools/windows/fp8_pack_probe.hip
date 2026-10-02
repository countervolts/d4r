// Diagnostic only: all finite OCP e4m3 encodings, including signed zero.
#include "../../kernels/m/swin_common.h"
struct PackParams { const half_t* input; uint32_t* output; };
extern "C" __global__ void fp8_pack_probe(PackParams p) {
    const unsigned lane = blockIdx.x * 32 + threadIdx.x;
    if (lane >= 32) return;
    const auto packed = to_fp8(*(const sop*)(p.input + lane * 8));
    p.output[lane * 2] = packed[0]; p.output[lane * 2 + 1] = packed[1];
}
