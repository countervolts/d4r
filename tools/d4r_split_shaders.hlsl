// The native D3D12 VRAM path of d4r_nvngx_shim.cpp (Windows without vkd3d-proton): the stand-ins for the Vulkan
// blits and copies of the vkd3d-proton route. to_canonical writes a game input into its plane's canonical layout
// (tightly packed rows of rowBytes: colour f16x4, motion f16x2, depth and exposure f32) in a shared buffer HIP reads;
// from_canonical writes DLSS's result (f16x4 rows) into the game's output texture. Built into the shim with dxc
// (scripts/build_d4r_nvngx_shim.sh).
#define INPUT_RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0)), UAV(u0)"
#define OUTPUT_RS "RootConstants(num32BitConstants=4, b0), SRV(t0), DescriptorTable(UAV(u0))"

cbuffer Plane : register(b0) { uint width, height, rowBytes, layout; };

Texture2D<float4> source : register(t0);
RWByteAddressBuffer canonical : register(u0);

uint pack_half2(float a, float b) { return f32tof16(a) | (f32tof16(b) << 16); }

[RootSignature(INPUT_RS)]
[numthreads(8, 8, 1)]
void to_canonical(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height)
        return;
    const float4 v = source.Load(int3(id.xy, 0));
    const uint row = id.y * rowBytes;
    if (layout == 0)        // colour
        canonical.Store2(row + id.x * 8, uint2(pack_half2(v.x, v.y), pack_half2(v.z, v.w)));
    else if (layout == 1)   // motion
        canonical.Store(row + id.x * 4, pack_half2(v.x, v.y));
    else                    // depth, exposure
        canonical.Store(row + id.x * 4, asuint(v.x));
}

ByteAddressBuffer result : register(t0);
RWTexture2D<float4> output : register(u0);

[RootSignature(OUTPUT_RS)]
[numthreads(8, 8, 1)]
void from_canonical(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height)
        return;
    const uint2 h = result.Load2(id.y * rowBytes + id.x * 8);
    output[id.xy] = float4(f16tof32(h.x), f16tof32(h.x >> 16), f16tof32(h.y), f16tof32(h.y >> 16));
}
