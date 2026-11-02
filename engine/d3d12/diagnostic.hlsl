// Diagnostic stand-in for inference (Windows support plan, milestone 1): a temporal accumulation that exercises
// borrowed input/output textures, an output subrectangle and feature-owned history, with no model.
//   output[origin + p] = history'[p] = color[p] * scale + (reset ? 0 : history[p])
struct Frame { uint2 origin; uint2 size; float scale; uint reset; uint2 unused; };
ConstantBuffer<Frame> frame : register(b0);
Texture2D<float4> color : register(t0);
RWTexture2D<float4> output : register(u0);
RWTexture2D<float4> history : register(u1);      // last committed history
RWTexture2D<float4> nextHistory : register(u2);  // committed only when the host reports the recording submitted

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= frame.size)) return;
    const float4 previous = frame.reset ? float4(0, 0, 0, 0) : history[id.xy];
    const float4 value = color[id.xy] * frame.scale + previous;
    output[frame.origin + id.xy] = value;
    nextHistory[id.xy] = value;
}
