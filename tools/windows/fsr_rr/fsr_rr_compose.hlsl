// Native D3D12 counterpart of fsr_rr_compose.comp. Keep conversion semantics aligned.
// FSR Ray Regeneration backend, stage 3: the denoised diffuse and specular signals remodulated by
// their albedos, plus the residual the denoiser never saw (particles, what demodulation could not
// carry, sky). The result is render-resolution colour for the Super Sampling stage.

RWTexture2D<float4> denoisedDiffuse : register(u0);
RWTexture2D<float4> denoisedSpecular : register(u1);
RWTexture2D<float4> diffuseAlbedo : register(u2);
RWTexture2D<float4> specularAlbedo : register(u3);
RWTexture2D<float4> residual : register(u4);
RWTexture2D<float4> composed : register(u5);
RWTexture2D<float4> noisyDiffuse : register(u6);
RWTexture2D<float4> noisySpecular : register(u7);

struct FsrRrConstants
{
    uint2 renderSize;
    float haloLimit; // > 0: each denoised lobe at most this multiple of the brightest noisy input of that lobe around it
    uint unused;
};
cbuffer Constants : register(b0) { FsrRrConstants pc; }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const int2 px = int2(tid.xy);
    if (px.x >= int(pc.renderSize.x) || px.y >= int(pc.renderSize.y))
        return;
    const float3 da = diffuseAlbedo[px].rgb, sa = specularAlbedo[px].rgb;
    float3 diffuse = denoisedDiffuse[px].rgb * da;
    float3 specular = denoisedSpecular[px].rgb * sa;
    if (pc.haloLimit > 0.0)
    {
        // The denoiser can carry a thin object's large demodulated signal (dark albedo) across its edge, where this
        // texel's albedo turns it into a bright fringe. A denoised value above everything around it in the noisy
        // input is such a leak: capped, by luminance (the hue is kept).
        const float3 w = float3(0.2126, 0.7152, 0.0722);
        float diffuseMax = 0.0, specularMax = 0.0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int2 q = clamp(px + int2(dx, dy), int2(0, 0), int2(pc.renderSize) - 1);
                // what the lobe contributes there (the neighbour's own albedo): a leaked demodulated signal
                // remodulated by this texel's albedo exceeds all of these
                diffuseMax = max(diffuseMax, dot(noisyDiffuse[q].rgb * diffuseAlbedo[q].rgb, w));
                specularMax = max(specularMax, dot(noisySpecular[q].rgb * specularAlbedo[q].rgb, w));
            }
        const float dl = dot(diffuse, w), sl = dot(specular, w);
        if (dl > pc.haloLimit * diffuseMax)
            diffuse *= pc.haloLimit * diffuseMax / dl;
        if (sl > pc.haloLimit * specularMax)
            specular *= pc.haloLimit * specularMax / sl;
    }
    const float3 color = diffuse + specular + residual[px].rgb;
    composed[px] = float4(clamp(color, 0.0, 65000.0), 1.0);
}
