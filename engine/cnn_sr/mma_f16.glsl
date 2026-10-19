// Extension-free f16 MMA for wave32. PTX m16n8k8 / m16n8k16 row.col fragments:
// A: (row=g+8*(word&1), k=2*t+8*(word/2)+half).
// B: (k=2*t+8*word+half, col=g). C/D: (row=g+8*word, col=2*t+half).
// g=lane/4, t=lane%4. Shuffle sources stay within the current subgroup;
// there is no workgroup scratch or cross-wave synchronization.
// Include engine/round_half.glsl first: RADV's packHalf2x16 truncates instead of rounding to nearest.
#ifndef D4R_MMA_F16_INCLUDED
#define D4R_MMA_F16_INCLUDED

#ifdef D4R_MMA_PACKED
// The SR reconstruction head uses the same packed-half accumulation as e_net.comp.
// Each lane owns two rows and two columns; one packed FMA updates a row's two columns.
// Keep the fragment mapping and K order, rounding after each FMA instead of each MMA.
f16vec4 mma_packed_step8(uint a0, uint a1, uint b0, f16vec4 acc)
{
    const uint lane = gl_SubgroupInvocationID, row = lane >> 2u, col = (lane & 3u) * 2u;
    [[unroll]] for (uint k = 0u; k < 4u; ++k) {
        const f16vec2 x0 = unpackFloat2x16(subgroupShuffle(a0, row * 4u + k));
        const f16vec2 x1 = unpackFloat2x16(subgroupShuffle(a1, row * 4u + k));
        const f16vec2 y0 = unpackFloat2x16(subgroupShuffle(b0, col * 4u + k));
        const f16vec2 y1 = unpackFloat2x16(subgroupShuffle(b0, (col + 1u) * 4u + k));
        acc.xy = fma(f16vec2(x0.x), f16vec2(y0.x, y1.x), acc.xy);
        acc.xy = fma(f16vec2(x0.y), f16vec2(y0.y, y1.y), acc.xy);
        acc.zw = fma(f16vec2(x1.x), f16vec2(y0.x, y1.x), acc.zw);
        acc.zw = fma(f16vec2(x1.y), f16vec2(y0.y, y1.y), acc.zw);
    }
    return acc;
}
void mma_f16_m16n8k8(uint a0, uint a1, uint b0, uint c0, uint c1, out uint d0, out uint d1)
{
    const f16vec4 acc = mma_packed_step8(a0, a1, b0, f16vec4(unpackFloat2x16(c0), unpackFloat2x16(c1)));
    d0 = packFloat2x16(acc.xy); d1 = packFloat2x16(acc.zw);
}
void mma_f16_m16n8k16(uint a0, uint a1, uint a2, uint a3, uint b0, uint b1,
                     uint c0, uint c1, out uint d0, out uint d1)
{
    f16vec4 acc = mma_packed_step8(a0, a1, b0, f16vec4(unpackFloat2x16(c0), unpackFloat2x16(c1)));
    acc = mma_packed_step8(a2, a3, b1, acc);
    d0 = packFloat2x16(acc.xy); d1 = packFloat2x16(acc.zw);
}
#else
vec4 mma_f16_step8(uint a0, uint a1, uint b0, vec4 acc)
{
    const uint lane = gl_SubgroupInvocationID, row = lane >> 2u, col = (lane & 3u) * 2u;
    [[unroll]] for (uint k = 0u; k < 4u; ++k) {
        const vec2 x0 = unpackHalf2x16(subgroupShuffle(a0, row * 4u + k));
        const vec2 x1 = unpackHalf2x16(subgroupShuffle(a1, row * 4u + k));
        const vec2 y0 = unpackHalf2x16(subgroupShuffle(b0, col * 4u + k));
        const vec2 y1 = unpackHalf2x16(subgroupShuffle(b0, (col + 1u) * 4u + k));
        acc.x = fma(x0.x, y0.x, acc.x); acc.x = fma(x0.y, y0.y, acc.x);
        acc.y = fma(x0.x, y1.x, acc.y); acc.y = fma(x0.y, y1.y, acc.y);
        acc.z = fma(x1.x, y0.x, acc.z); acc.z = fma(x1.y, y0.y, acc.z);
        acc.w = fma(x1.x, y1.x, acc.w); acc.w = fma(x1.y, y1.y, acc.w);
    }
    return acc;
}
void mma_f16_m16n8k8(uint a0, uint a1, uint b0, uint c0, uint c1, out uint d0, out uint d1)
{
    const vec4 acc = mma_f16_step8(a0, a1, b0, vec4(unpackHalf2x16(c0), unpackHalf2x16(c1)));
    d0 = packHalf2x16(roundHalfValue(acc).xy); d1 = packHalf2x16(roundHalfValue(acc).zw);
}
void mma_f16_m16n8k16(uint a0, uint a1, uint a2, uint a3, uint b0, uint b1,
                     uint c0, uint c1, out uint d0, out uint d1)
{
    vec4 acc = mma_f16_step8(a0, a1, b0, vec4(unpackHalf2x16(c0), unpackHalf2x16(c1)));
    acc = mma_f16_step8(a2, a3, b1, acc);
    d0 = packHalf2x16(roundHalfValue(acc).xy); d1 = packHalf2x16(roundHalfValue(acc).zw);
}
#endif // D4R_MMA_PACKED
#endif
