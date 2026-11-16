// d3dhip on RDNA 3 (runtime_m_rdna3.inc): the RDNA 3 M/L engine keeps its tokens token-major, [tok.y][tok.x][64] FP8
// codes (the network's input `tokens` and output `dec1`), while the native HIP network (hip_net) reads and writes NGX's
// plane layout, [2][tok.y][tok.x][32] (enc0_m.comp / dec0_m.comp PLANES). The bytes are the same; only their order
// differs: dword d of token t is dword d % 8 of token t in plane d / 8. toPlanes: tokens -> the network's input;
// otherwise: the network's head -> dec1. Bindings: u0 the parameters, t1 the source, u2 the target (raw buffers).
struct PlanesParams { uint tokens; uint toPlanes; uint pad0; uint pad1; };
RWStructuredBuffer<PlanesParams> params : register(u0);
ByteAddressBuffer source : register(t1);
RWByteAddressBuffer target : register(u2);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const PlanesParams p = params[0];
    const uint i = id.x;
    if (i >= p.tokens * 16u)
        return;
    const uint t = i >> 4, d = i & 15u;
    const uint plane = ((d >> 3) * p.tokens + t) * 8u + (d & 7u);
    if (p.toPlanes != 0u)
        target.Store(plane * 4u, source.Load(i * 4u));
    else
        target.Store(i * 4u, source.Load(plane * 4u));
}
