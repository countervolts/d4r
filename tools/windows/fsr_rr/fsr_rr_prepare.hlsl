// Native D3D12 counterpart of fsr_rr_prepare.comp. Keep conversion semantics aligned.
// FSR Ray Regeneration backend, stage 1: a game's DLSS Ray Reconstruction inputs -> the inputs of
// AMD's FidelityFX denoiser 1.2 (ffx_denoiser.h), at render resolution.
//
// The split of the game's single noisy colour into a diffuse and a specular signal follows the
// DLSS-RR -> FSR-RR adapter of Zach Hembree's OptiScaler branch ffx-denoise-experimental (GPL-3.0,
// 18ac686, FSRDInputConv.hlsl, "Mode 2") and burak113's continuation (5a6adb1): each lobe gets the
// colour in proportion to its albedo and is demodulated by it. That is an approximation - one
// colour cannot be split exactly - and what does not survive demodulation goes to the residual,
// which bypasses the denoiser. Unlike that reference this writes FSR-RR 1.2's contract: signed
// linear depth reconstructed from hardware or linear depth, motion as previous-minus-current UV,
// linear (non-gamma) albedo and the colour before particles as the denoised base.

Texture2D<float4> inColor : register(t0);
Texture2D<float4> inDepth : register(t1);
Texture2D<float4> inMotion : register(t2);
Texture2D<float4> inNormals : register(t3);
Texture2D<float4> inRoughness : register(t4);       // the normals again when roughness is packed
Texture2D<float4> inDiffuseAlbedo : register(t5);
Texture2D<float4> inSpecularAlbedo : register(t6);
Texture2D<float4> inSpecularHit : register(t7);     // bound to the normals when absent
Texture2D<float4> inDiffuseHit : register(t8);      // bound to the normals when absent
Texture2D<float4> inBeforeParticles : register(t9); // bound to the colour when absent

RWTexture2D<float> outLinearDepth : register(u0);
RWTexture2D<float4> outMotion : register(u1);
RWTexture2D<float4> outNormals : register(u2);
RWTexture2D<float4> outDiffuseAlbedo : register(u3);
RWTexture2D<float4> outSpecularAlbedo : register(u4);
RWTexture2D<float4> outDiffuse : register(u5);
RWTexture2D<float4> outSpecular : register(u6);
RWTexture2D<float4> outResidual : register(u7);

// Matrices are uploaded as the row-major, row-vector (v' = vM) float[16] NGX and FFX use. Read as a
// GLSL (column-major) column_major float4x4 that is the transpose, so M * v here is the row-vector product v * M.
struct FsrRrConstants
{
    column_major float4x4 clipToView;          // inverse(ViewToClipMatrix)
    column_major float4x4 viewToPreviousView;  // inverse(WorldToView) * previous WorldToView
    float4 renderSize;          // width, height, 1 / width, 1 / height
    float4 motion;              // xy: MV texel -> UV delta (MV.Scale / MV resolution); zw: MV texel per render pixel
    uint4 colorDepthBase;     // colour subrect origin xy, depth subrect origin xy
    uint4 motionBaseFlags;    // motion subrect origin xy, flags, unused
    float4 depthParams;         // x: sign of view z in front of the camera, y: sky depth (hardware), z: far marker
    float4 detail;              // x: share of the colour that bypasses the denoiser (0: none); y: smallest albedo a
                              // signal is divided by
};
cbuffer Constants : register(b0) { FsrRrConstants pc; }

static const uint kHardwareDepth = 1u;
static const uint kRoughnessPacked = 2u;
static const uint kSpecularHit = 4u;
static const uint kSpecularHitInW = 8u;   // DLSSD.SpecularRayDirectionHitDistance: the distance is .w
static const uint kDiffuseHit = 16u;
static const uint kDiffuseHitInW = 32u;
static const uint kNormalsUnorm = 64u;    // [0,1]-encoded normals (UNORM formats)
static const uint kBeforeParticles = 128u;
static const uint kInvalidNormalPassthrough = 256u;

static const float kHalfMax = 65000.0;

float3 safe(float3 v)
{
    // NaN fails both comparisons and becomes 0
    return float3(v.x >= 0.0 ? min(v.x, kHalfMax) : 0.0, v.y >= 0.0 ? min(v.y, kHalfMax) : 0.0,
                v.z >= 0.0 ? min(v.z, kHalfMax) : 0.0);
}

// AMD's NormalToOctahedronUv (FidelityFX denoising.md 5.3.1)
float2 octahedron(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 k = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    float2 xy = n.z < 0.0 ? (1.0 - abs(n.yx)) * k : n.xy;
    return xy * 0.5 + 0.5;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const int2 px = int2(tid.xy);
    if (px.x >= int(pc.renderSize.x) || px.y >= int(pc.renderSize.y))
        return;
    const uint flags = pc.motionBaseFlags.z;
    const float2 uv = (float2(px) + 0.5) * pc.renderSize.zw;
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // Signed view-space position. Hardware depth goes through the inverse projection, which carries
    // the game's handedness and (reversed) depth range; linear depth is the distance along the view
    // axis, given its sign by the projection.
    const float depth = inDepth.Load(int3(int2(pc.colorDepthBase.zw) + px, 0)).r;
    float3 viewPos;
    bool sky;
    if ((flags & kHardwareDepth) != 0u)
    {
        const float4 p = mul(pc.clipToView, float4(ndc, depth, 1.0));
        viewPos = p.xyz / p.w;
        sky = depth == pc.depthParams.y || !(abs(p.w) > 0.0);
    }
    else
    {
        const float4 p = mul(pc.clipToView, float4(ndc, 0.5, 1.0));
        const float3 ray = p.xyz / p.w;
        const float z = depth * pc.depthParams.x;
        viewPos = ray * (z / ray.z);
        sky = !(depth > 0.0) || depth >= pc.depthParams.z;
    }

    const float3 color = safe(inColor.Load(int3(int2(pc.colorDepthBase.xy) + px, 0)).rgb);
    const float3 base = (flags & kBeforeParticles) != 0u ? safe(inBeforeParticles.Load(int3(px, 0)).rgb) : color;
    // Particles (colour after minus before) stay out of the denoiser; signed, so smoke can darken.
    const float3 particles = color - base;

    float3 diffuseAlbedo = clamp(inDiffuseAlbedo.Load(int3(px, 0)).rgb, 0.0, 1.0);
    float3 specularAlbedo = clamp(inSpecularAlbedo.Load(int3(px, 0)).rgb, 0.0, 1.0);
    // Energy above 1 is taken from the diffuse lobe first (reference: albedo overshoot clamp).
    diffuseAlbedo = max(diffuseAlbedo - max(diffuseAlbedo + specularAlbedo - 1.0, 0.0), 0.0);
    const float albedoSum = dot(diffuseAlbedo + specularAlbedo, ((float3)1.0));
    sky = sky || !(albedoSum > 3e-3) || any(isnan(viewPos)) || any(isinf(viewPos));
    const float4 normalTexel = inNormals.Load(int3(px, 0));
    float3 normal = (flags & kNormalsUnorm) != 0u ? normalTexel.xyz * 2.0 - 1.0 : normalTexel.xyz;
    const float normalLength = dot(normal, normal);
    // A zero/invalid normal is no surface, not an invented world-Z-facing mirror. Alan Wake 2
    // supplies zero normals for its sky with finite FP16-max depth, so the far bound alone misses it.
    if ((flags & kInvalidNormalPassthrough) != 0u)
        sky = sky || !(normalLength > 1e-8) || isnan(normalLength) || isinf(normalLength);

    if (sky)
    {
        // Outside linearDepthBounds: the denoiser passes these texels through, and the residual carries
        // the whole colour.
        outLinearDepth[px] = float(pc.depthParams.z * 10.0 * pc.depthParams.x);
        outMotion[px] = ((float4)0.0);
        outNormals[px] = float4(0.5, 0.5, 1.0, 0.0);
        outDiffuseAlbedo[px] = ((float4)0.0);
        outSpecularAlbedo[px] = ((float4)0.0);
        outDiffuse[px] = ((float4)0.0);
        outSpecular[px] = ((float4)0.0);
        outResidual[px] = float4(color, 0.0);
        return;
    }
    outLinearDepth[px] = float(viewPos.z);

    // 2.5D motion: DLSS motion vectors point from this pixel to where it was (previous - current), in
    // the MV texture's pixels. The depth delta is the camera's: the previous view's z of this point
    // (moving geometry is not reprojected; there is no previous depth for it).
    const int2 mvTexel = int2(pc.motionBaseFlags.xy) + int2(float2(px) * pc.motion.zw);
    const float2 mv = inMotion.Load(int3(mvTexel, 0)).xy * pc.motion.xy;
    const float4 previous = mul(pc.viewToPreviousView, float4(viewPos, 1.0));
    outMotion[px] = float4(mv, previous.z / previous.w - viewPos.z, 0.0);

    normal = dot(normal, normal) > 1e-8 ? normalize(normal) : float3(0.0, 0.0, 1.0);
    const float roughness = clamp((flags & kRoughnessPacked) != 0u ? normalTexel.w : inRoughness.Load(int3(px, 0)).r,
                                  0.0, 1.0);
    // material type 0 (the default material) everywhere
    outNormals[px] = float4(octahedron(normal), roughness, 0.0);
    outDiffuseAlbedo[px] = float4(diffuseAlbedo, 0.0);
    outSpecularAlbedo[px] = float4(specularAlbedo, 0.0);

    // Split by albedo and demodulate (reference "Mode 2"). Per channel, so a tinted metal keeps its tint.
    const float3 total = diffuseAlbedo + specularAlbedo;
    const float3 specularShare = specularAlbedo / max(total, ((float3)1e-4));
    const float3 specularColor = base * specularShare;
    const float3 diffuseColor = base - specularColor;
    // A dark albedo would turn light it did not shape (fog, sky light, the background at a thin object's edge) into
    // a huge signal that the denoiser clips (fog lost) or spreads (bright fringes): no albedo below the floor
    // divides a signal, and what the division cannot carry stays in the residual.
    float3 diffuse = safe(diffuseColor / max(diffuseAlbedo, ((float3)pc.detail.y)));
    float3 specular = safe(specularColor / max(specularAlbedo, ((float3)pc.detail.y)));
    if (pc.detail.x > 0.0)
    {
        // Fine detail the denoiser treats as noise (hair, water ripples): part of the colour goes around it,
        // through the residual.
        const float denoised = 1.0 - clamp(pc.detail.x, 0.0, 1.0);
        diffuse *= denoised;
        specular *= denoised;
    }
    const float3 remodulated = diffuse * diffuseAlbedo + specular * specularAlbedo;

    // A: the ray hit distance for an indirect signal (ignored, but non-negative, for a direct one)
    float specularHit = 0.0, diffuseHit = 0.0;
    if ((flags & kSpecularHit) != 0u)
    {
        const float4 hit = inSpecularHit.Load(int3(px, 0));
        specularHit = clamp((flags & kSpecularHitInW) != 0u ? hit.w : hit.x, 0.0, kHalfMax);
    }
    if ((flags & kDiffuseHit) != 0u)
    {
        const float4 hit = inDiffuseHit.Load(int3(px, 0));
        diffuseHit = clamp((flags & kDiffuseHitInW) != 0u ? hit.w : hit.x, 0.0, kHalfMax);
    }
    outDiffuse[px] = float4(diffuse, diffuseHit);
    outSpecular[px] = float4(specular, specularHit);
    outResidual[px] = float4(base - remodulated + particles, 0.0);
}
