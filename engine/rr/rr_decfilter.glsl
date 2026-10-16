// The decoders' output filter (engine/rr/dec0_ref.outputs' surface 112), shared by dec1_m.comp and swin_m.comp: included
// after the head has been written to OUTF (16 f16 per window token, raster order) and a barrier. Texture 88 (one level
// down, bilinear) at the window's 10 x 10 tap positions into TAP; then two threads per token: the softmax of channels
// 0-8 over the 3x3 taps, one f16 fma chain per colour channel, blended with texture 96 by the gate (channel 9) plus
// texture 104, to surface 112.
    // texture 88 (one level down) at the window's 10 x 10 full-resolution tap positions (clamped like the filter's
    // coordinates), through the bilinear sampler at NVIDIA's coordinates: the reference's texture unit does not round
    // like f32 arithmetic (computing the 1/4, 3/4 blend by hand left ~0.2% of surface 112 one ulp off, capDT 2026-10-12)
    if (tid < 100u) {
        const int xi = clampTap(x0 - 1 + int(tid % 10u), ex), yi = clampTap(y0 - 1 + int(tid / 10u), ey);
        const float iw = uintBitsToFloat(param.p[34]), ih = uintBitsToFloat(param.p[35]);
        precise float cu = fma(iw, float(xi), iw * 0.5), cv = fma(ih, float(yi), ih * 0.5);
        const vec4 bl = textureLod(T88, vec2(cu, cv), 0.0);
        [[unroll]] for (uint c = 0u; c < 3u; ++c) TAP[tid * 4u + c] = float16_t(bl[c]);
    }
    barrier();
    // two threads per token: half 0 filters channels 0 and 1, half 1 channel 2
    {
        const uint t = tid >> 1u, hf = tid & 1u;
        const int x = x0 + int(t & 7u), y = y0 + int(t >> 3u);
        if (x < 0 || y < 0 || x >= ex || y >= ey) return;
        float16_t a[10];
        [[unroll]] for (uint c = 0u; c < 10u; ++c) a[c] = OUTF[t * 16u + c];
        // surface 112: softmax of channels 0-8 over the 3x3 taps, one f16 fma chain per channel; the gate (channel 9)
        // blends it with texture 96; + texture 104
        const float16_t ma = max(max(max(a[0], a[2]), a[4]), a[6]), mb = max(max(max(a[1], a[3]), a[5]), a[7]);
        const float16_t mx = max(max(ma, mb), a[8]);
        float e[9];
        [[unroll]] for (uint i = 0u; i < 9u; ++i) { precise float16_t d = a[i] - mx; e[i] = exp2(float(d) * 1.4426950216293335); }
        precise float s = (e[0] + e[1]) + 0.0;
        s = s + (e[2] + e[3]);
        s = s + (e[4] + e[5]);
        s = s + (e[6] + e[7]);
        s = s + e[8];
        const float16_t rr = rcp16(s);
        float16_t wg[9];
        [[unroll]] for (uint i = 0u; i < 9u; ++i) wg[i] = h16(r16(e[i]) * float(rr));
        const uint tx = uint(x - x0), ty = uint(y - y0), c0 = 2u * hf, nc = 2u - hf;
        float16_t f[2] = float16_t[2](0.0hf, 0.0hf);
        bool mid = false;
        [[unroll]] for (uint k = 0u; k < 9u; ++k) {
            const uint at = ((ty + k / 3u) * 10u + tx + k % 3u) * 4u + c0;
            [[unroll]] for (uint c = 0u; c < 2u; ++c) f[c] = fma16q(TAP[at + c], wg[k], f[c], mid);
        }
        if (mid) {   // an f32 result on a binary16 midpoint: redo the chain with the exact fma (rare)
            f[0] = 0.0hf; f[1] = 0.0hf;
            for (uint k = 0u; k < 9u; ++k) {
                const uint at = ((ty + k / 3u) * 10u + tx + k % 3u) * 4u + c0;
                for (uint c = 0u; c < 2u; ++c) f[c] = fma16(TAP[at + c], wg[k], f[c]);
            }
        }
        const float16_t g = sig(a[9]);
        precise float16_t om = float16_t(1.0) - g;
        const vec4 t96 = texelFetch(T96, ivec2(x, y), 0), t104 = texelFetch(T104, ivec2(x, y), 0);
        float16_t o[2];
        [[unroll]] for (uint c = 0u; c < 2u; ++c) {
            precise float16_t fg = f[c] * g;
            float16_t v = fma16(float16_t(t96[c0 + c]), om, fg);
            const float16_t t4 = float16_t(t104[c0 + c]);
            if (!isnan(t4)) { precise float16_t vv = v + t4; v = vv; }
            o[c] = v;
        }
        const uint other = subgroupShuffleXor(bh(o[0]), 1u);
        if (hf == 0u) imageStore(S112, ivec2(x, y), vec4(o[0], o[1], hb(other), 0.0));
    }
