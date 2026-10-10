// d4r engine, presets M and L: the input stage's per-pixel features, shared by enc0_m.comp (M) and layer_m.comp
// KIND 3 (L, whose input stage gathers the same 32 feature slots). The including shader declares the parameter
// block (EP) and the bindings colorTex, motionTex, depthTex, prevMvTex, halfTex, featTex, mvAOut, mvBOut, mvCOut and
// noise. EP names the parameter block (default p).
#ifndef EP
#define EP p
#endif
const float L10 = 3.3219281;
// LDR (NGX's *_ldr_* kernels, the game did not set the HDR flag): display-referred colour, no exposure and no
// range compression anywhere.
#ifndef LDR
#define LDR 0
#endif
#if LDR
float compress1(float v) { return v; }
#else
float compress1(float v) {
    precise float mapped = fma(v, 100.0, 1.0);
    precise float lg = log2(max(mapped, 1e-4));
    precise float base10 = lg * 0.30103;
    precise float result = base10 * 0.36;
    return result;
}
#endif
vec3 compress(vec3 v) { return vec3(compress1(v.x), compress1(v.y), compress1(v.z)); }
#if LDR
float colour1(float v) { return isinf(v) || isnan(v) ? 0.0 : min(v, 32752.0); }
#else
float colour1(float v) { return compress1(isinf(v) || isnan(v) ? 0.0 : min(v, 32752.0)); }
#endif
ivec2 texel(ivec2 t) { return ivec2(t.x > 0 ? min(t.x, EP.size.x - 1) : 0, t.y > 0 ? min(t.y, EP.size.y - 1) : 0); }
vec2 motion(ivec2 t) { return EP.mvScale * (EP.mvOffs + texelFetch(motionTex, texel(t), 0).xy); }
// NGX's dithered truncation of an f16 value to the FP8 grid
floate4m3_t dither(float v, uint n)
{
    const float16_t h = float16_t(isinf(v) || isnan(v) ? 0.0 : min(v, 1024.0));
    const uint e = uint(float16BitsToUint16(h)) & 0xFC00u;
    precise float16_t added = h + uint16BitsToFloat16(uint16_t(e | n));
    precise float16_t t = added - uint16BitsToFloat16(uint16_t(e));
    const float16_t r = uint16BitsToFloat16(uint16_t(uint(float16BitsToUint16(t)) & 0xFF80u));
    return floate4m3_t(clamp(r, float16_t(-448.0), float16_t(448.0)));
}
// The 32 dithered FP8 feature slots of block pixel pi (16 per plane) and the motion candidates of the pixel.
void gatherFeatures(uint pi, ivec2 block, float e1, float e2, bool fusedFourth, out floate4m3_t fe[32])
{
    const uint px = pi & 7u, py = pi >> 3;
    const ivec2 raw = block + ivec2(px, py), a = abs(raw);
    const ivec2 m = ivec2(a.x < EP.size.x ? a.x : 2 * EP.size.x - a.x - 2, a.y < EP.size.y ? a.y : 2 * EP.size.y - a.y - 2);
    const bool outside = raw.x < 0 || raw.y < 0 || a.x >= EP.size.x || a.y >= EP.size.y;
    // the neighbours with the smallest and the largest depth
    ivec2 lo = m, hi = m;
    float dlo = texelFetch(depthTex, texel(m), 0).r, dhi = dlo;
    const ivec2 offs[8] = ivec2[8](ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0), ivec2(1, 0), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));
    for (int k = 0; k < 8; ++k)
    {
        const float d = texelFetch(depthTex, texel(m + offs[k]), 0).r;
        if (d < dlo) { dlo = d; lo = m + offs[k]; }
        if (d > dhi) { dhi = d; hi = m + offs[k]; }
    }
    const vec2 mvA = motion(EP.depthFlag != 0u ? lo : hi), mvB = motion(EP.depthFlag != 0u ? hi : lo), mvC = motion(m);
    if (!outside)
    {
        imageStore(mvAOut, m, vec4(mvA, 0.0, 0.0)); imageStore(mvBOut, m, vec4(mvB, 0.0, 0.0)); imageStore(mvCOut, m, vec4(mvC, 0.0, 0.0));
    }
    const vec3 own = texelFetch(colorTex, texel(m), 0).rgb;
    const vec2 centre = vec2(m) + 0.5, f = centre - EP.jitter, inv = 1.0 / vec2(EP.size);
    // how much the motion changed since the previous frame
    const vec2 prev = textureLod(prevMvTex, clamp(((f + EP.prevJitter) + mvA * EP.hi2rnd) * vec2(EP.renderInvX, EP.renderInvY), vec2(0.0), vec2(EP.size) * vec2(EP.renderInvX, EP.renderInvY)), 0.0).xy;
    const vec2 dm = (EP.changeScale * 540.0 / float(EP.size.y)) * (mvA * EP.hi2rnd - EP.hi2rnd * prev);
    const float d2 = fma(dm.x, dm.x, dm.y * dm.y);
    const float change = EP.changeGain * (log2(d2 > 0.0 ? sqrt(d2) + 1.0 : 1.0) * 0.6931472);
    // the previous result along the candidates
    precise vec2 q[4];
    q[0] = mvA + f * EP.rnd2hi; q[1] = mvC + f * EP.rnd2hi; q[2] = mvB + f * EP.rnd2hi;
    // The captured PTX fuses the first pixel's fourth coordinate, but reuses a rounded
    // centre*scale product for the second pixel. A single FMA for both changes FP8 dither.
    q[3] = fusedFourth ? fma(centre, EP.rnd2hi, mvC) : centre * EP.rnd2hi + mvC;
    vec3 h[4];
    for (int k = 0; k < 4; ++k)
    {
        precise vec3 s = own;
        if (!(q[k].x < 0.0 || q[k].y < 0.0 || q[k].x > EP.hiSize.x || q[k].y > EP.hiSize.y || EP.reset != 0u))
#if LDR
            s = textureLod(halfTex, q[k] * vec2(EP.halfInvX, EP.halfInvY), 0.0).rgb;
#else
        {
            precise vec3 packedHistory = textureLod(halfTex, q[k] * vec2(EP.halfInvX, EP.halfInvY), 0.0).rgb * 2.45;
            precise vec3 scaled = clamp(packedHistory, 1e-4, 3.0) * (1.0 / 0.36);
            precise vec3 exponent = scaled * L10;
            precise vec3 linear = (exp2(exponent) - 1.0) * (1.0 / 100.0);
            s = linear * (1.0 / e2);
        }
#endif
        h[k] = compress(e1 * s);
    }
    vec4 feat = vec4(0.0);
    float mf = 0.0;
    if (EP.reset == 0u)
    {
        mf = change;
        if (!(q[0].x < 0.0 || q[0].y < 0.0 || q[0].x > EP.hiSize.x || q[0].y > EP.hiSize.y))
        {
            precise vec2 normalized = fma(q[0], 1.0 / EP.hiSize, EP.prevJitter * inv);
            precise vec2 texels = normalized * vec2(EP.size);
            precise vec2 uv = texels * vec2(EP.renderInvX, EP.renderInvY);
            feat = textureLod(featTex, uv, 0.0);
        }
    }
    const vec3 c = e1 * own;
    const vec3 col = vec3(colour1(c.r), colour1(c.g), colour1(c.b));
    const uint n = uint(noise.n[(EP.frame << 8) | uint((a.y & 15) << 4) | uint(a.x & 15)]) >> 9;
    // 32 slots: 16 per plane
    float v[32];
    for (int k = 0; k < 32; ++k) v[k] = 0.0;
    v[0] = col.r; v[1] = col.g; v[2] = feat.x; v[3] = feat.y; v[4] = col.b; v[5] = h[0].r; v[6] = feat.z; v[7] = feat.w;
    v[8] = h[0].g; v[9] = h[0].b; v[12] = h[1].r; v[13] = h[1].g;
    v[16] = h[1].b; v[17] = h[2].r; v[20] = h[2].g; v[21] = h[2].b; v[24] = h[3].r; v[25] = h[3].g; v[28] = h[3].b; v[29] = mf;
    for (uint k = 0u; k < 32u; ++k) fe[k] = dither(v[k], n);
}
