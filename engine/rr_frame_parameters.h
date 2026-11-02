// Shared RR D/E CPU parameter ABI for Vulkan and D3D12. Include inside the
// backend's anonymous namespace after its exposure type. Camera order is
// retained.
template <class T> bool rrPresent(T handle)
{
    return bool(handle);
}
constexpr unsigned RStages = 12;
enum : unsigned
{
    REnc0,
    REnc1,
    REnc2,
    REnc3,
    REnc4,
    REnc5,
    RDec4,
    RDec3,
    RDec2,
    RDec1,
    RDec0,
    RHkpn
};
constexpr const char *RNames[RStages] = {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5",
                                         "dec4", "dec3", "dec2", "dec1", "dec0", "hkpn"};
constexpr uint32_t RParamBytes[RStages] = {1032, 144, 144, 144, 144, 144, 144, 144, 144, 144, 152, 480};
constexpr int RLevel[RStages] = {0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, -1};
// The plane arena: 70 * P0 bytes, P0 = the render size rounded up to 32 on both
// axes (W0 x H0). Every region starts at a multiple of P0 / 32 (NGX's layout,
// the same at 640x360 and 800x450): per stage, the arena offset of its input
// (argument 8), skip (24), output (48) and merged output (56) in units of P0 /
// 32, -1 where unused.
constexpr uint32_t RArenaUnits = 70 * 32;
constexpr int RArena[RStages][4] = {{-1, -1, 0, 1024}, // enc0: 32-channel tokens (dec0's skip), the 2x2 merge for enc1
                                    {1024, -1, 1536, 2048}, // enc1
                                    {2048, -1, 1024, 1216}, // enc2
                                    {1216, -1, 1280, 1344}, // enc3
                                    {1344, -1, 1216, 1236}, // enc4
                                    {1236, -1, 1241, -1},   // enc5, the bottleneck
                                    {1241, 1216, 1344, -1}, // dec4
                                    {1344, 1280, 1216, -1}, // dec3
                                    {1216, 1024, 2048, -1}, // dec2
                                    {2048, 1536, 1024, -1}, // dec1
                                    {1024, 0, 1536, -1},    // dec0: its output is the output stage's argument 256
                                    {-1, -1, -1, -1}};
constexpr uint32_t RHkpnArena = 1536;
// NGX's window shift of a stage, packed (x | y << 16) at argument 80: it
// alternates every frame, the decoders opposite to the encoders; the bottleneck
// has none
uint32_t rShift(unsigned stage, uint32_t frame)
{
    if (stage == REnc5 || stage == RHkpn)
        return 0;
    const bool even = ((frame & 1u) == 0) == (stage <= REnc5);
    return even ? 0x00060002u : 0x00020006u;
}
void put32(std::vector<uint8_t> &p, uint32_t offset, uint32_t value)
{
    std::memcpy(p.data() + offset, &value, 4);
}
void putF(std::vector<uint8_t> &p, uint32_t offset, float value)
{
    std::memcpy(p.data() + offset, &value, 4);
}
void put64(std::vector<uint8_t> &p, uint32_t offset, uint64_t value)
{
    std::memcpy(p.data() + offset, &value, 8);
}
using Mat4 = std::array<double, 16>;
Mat4 mul(const Mat4 &a, const Mat4 &b)
{
    Mat4 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
    return r;
}
Mat4 inverse(const Mat4 &m)
{
    // Gauss-Jordan with partial pivoting; a singular camera falls back to the
    // identity
    Mat4 a = m, r{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (int c = 0; c < 4; ++c)
    {
        int pivot = c;
        for (int i = c + 1; i < 4; ++i)
            if (std::fabs(a[i * 4 + c]) > std::fabs(a[pivot * 4 + c]))
                pivot = i;
        if (std::fabs(a[pivot * 4 + c]) < 1e-30)
            return Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        for (int j = 0; j < 4; ++j)
        {
            std::swap(a[c * 4 + j], a[pivot * 4 + j]);
            std::swap(r[c * 4 + j], r[pivot * 4 + j]);
        }
        const double d = a[c * 4 + c];
        for (int j = 0; j < 4; ++j)
        {
            a[c * 4 + j] /= d;
            r[c * 4 + j] /= d;
        }
        for (int i = 0; i < 4; ++i)
        {
            if (i == c)
                continue;
            const double f = a[i * 4 + c];
            for (int j = 0; j < 4; ++j)
            {
                a[i * 4 + j] -= f * a[c * 4 + j];
                r[i * 4 + j] -= f * r[c * 4 + j];
            }
        }
    }
    return r;
}
void putMat(std::vector<uint8_t> &p, uint32_t offset, const Mat4 &m)
{
    for (int i = 0; i < 16; ++i)
        putF(p, offset + 4 * i, static_cast<float>(m[i]));
}

template <class E, class I, class S> void setRRParameters(E &e, const I &images, const S &settings)
{
    const bool reset = settings.reset || e.frame == 0;
    const uint32_t f = e.frame;
    const float rw = static_cast<float>(e.rw), rh = static_cast<float>(e.rh), ow = static_cast<float>(e.ow),
                oh = static_cast<float>(e.oh);
    const float prevJitter[2] = {reset ? 0.0f : e.prevJitter[0], reset ? 0.0f : e.prevJitter[1]};
    // the network layers: only the window shift changes
    for (unsigned stage = REnc1; stage <= RDec0; ++stage)
        put32(e.blocks[stage], 80, rShift(stage, f));
    // enc0 (the argument block's constants from the model; offsets as observed on
    // harness captures, 2026-10-11)
    auto &p = e.blocks[REnc0];
    p = e.model.enc0Template;
    put32(p, 0, e.w0);
    put32(p, 4, e.h0);
    put64(p, 48, e.arenaOffset(RArena[REnc0][2]));
    put64(p, 56, e.arenaOffset(RArena[REnc0][3]));
    put32(p, 80, rShift(REnc0, f));
    putF(p, 136, 1.0f / static_cast<float>(e.w0));
    putF(p, 140, 1.0f / static_cast<float>(e.h0));
    put32(p, 144, e.rw);
    put32(p, 148, e.rh);
    put32(p, 152, e.w0);
    put32(p, 156, e.h0);
    putF(p, 312, prevJitter[0]);
    putF(p, 316, prevJitter[1]);
    putF(p, 320, settings.jitter[0]);
    putF(p, 324, settings.jitter[1]);
    putF(p, 328, rw / ow);
    putF(p, 332, rh / oh);
    putF(p, 336, ow / rw);
    putF(p, 340, oh / rh);
    putF(p, 344, settings.motionScale[0] * (ow / rw));
    putF(p, 348, settings.motionScale[1] * (oh / rh));
    // byte 365 (and the output stage's byte 440): the reset flag. Temporal
    // harness captures (capDT, capDfr, capDst, 2026-10-12) show it follows Reset
    // alone, not the jitter: set with reset and jitter, clear without both.
    put32(p, 364, reset ? 256u : 0u);
    // byte 366: DLSS.Roughness.Mode packed. enc0 then takes the roughness from
    // the normals' w and never reads the roughness texture (argument 192, bound
    // to the normals as well)
    p[366] = settings.roughnessPacked ? 1u : 0u;
    // bytes 536 / 539: DLSSD.SpecularHitDistance (texture 248) /
    // DLSS.DisocclusionMask (texture 296) are bound. The hit distance also sets
    // the output stage's byte 160 (harness captures capDXhit, capDXmask, capDX,
    // 2026-10-12)
    p[536] = rrPresent(images.specularHitDistance) ? 1u : 0u;
    p[539] = rrPresent(images.disocclusionMask) ? 1u : 0u;
    put32(p, 432, e.rw - 1);
    put32(p, 436, e.rh - 1);
    put32(p, 472, e.ow);
    put32(p, 476, e.oh);
    put32(p, 528, settings.depthInverted ? 1u : 0u);
    // byte 980 (and the output stage's byte 443): DLSS.Use.HW.Depth (capDXhw,
    // 2026-10-12)
    p[980] = settings.hardwareDepth ? 1u : 0u;
    // the camera: the view's rotation (its translation row dropped) times the
    // projection, its inverse, and both composed with the render viewport (NDC
    // <-> render pixels, y down)
    Mat4 view, clip;
    for (int i = 0; i < 16; ++i)
    {
        view[i] = settings.worldToView[i];
        clip[i] = settings.viewToClip[i];
    }
    view[12] = view[13] = view[14] = 0.0;
    const Mat4 rc = mul(view, clip), inv = inverse(rc);
    const Mat4 toNdc{2.0 / e.rw, 0, 0, 0, 0, -2.0 / e.rh, 0, 0, 0, 0, 1, 0, -1, 1, 0, 1};
    const Mat4 toPixels{e.rw / 2.0, 0, 0, 0, 0, -(e.rh / 2.0), 0, 0, 0, 0, 1, 0, e.rw / 2.0, e.rh / 2.0, 0, 1};
    putMat(p, 596, rc);
    putMat(p, 660, clip);
    putMat(p, 724, clip);
    putMat(p, 788, inv);
    putMat(p, 852, mul(toNdc, inv));
    putMat(p, 916, mul(rc, toPixels));
    putF(p, 756, settings.jitter[0] * (2.0f / rw));
    putF(p, 760, -settings.jitter[1] * (2.0f / rh));
    if (e.model.gfx11)
    {
        rr_camera::Matrix v, c;
        for (int i = 0; i < 16; ++i)
        {
            v[i] = settings.worldToView[i];
            c[i] = settings.viewToClip[i];
        }
        const auto camera = rr_camera::transforms(v, c, reset ? v : e.previousView, reset ? c : e.previousClip, rw, rh);
        auto store = [&](uint32_t offset, const rr_camera::Matrix &matrix) {
            for (int i = 0; i < 16; ++i)
                putF(p, offset + 4 * i, matrix[i]);
        };
        store(596, camera.previousClip);
        store(788, camera.inverseClip);
        store(852, camera.inversePixels);
        store(916, camera.previousPixels);
        putF(p, 756, c[8] + (settings.jitter[0] + settings.jitter[0]) / rw);
        putF(p, 760, c[9] - (settings.jitter[1] * 2.0f) / rh);
        e.previousView = v;
        e.previousClip = c;
    }
    put32(p, 984, f);
    put32(p, 1024, e.rw);
    put32(p, 1028, e.rh);
    // the output stage
    auto &o = e.blocks[RHkpn];
    o = e.model.hkpnTemplate;
    put32(o, 0, e.ow);
    put32(o, 4, e.oh);
    put32(o, 8, e.ow);
    put32(o, 12, e.oh);
    put32(o, 16, e.rw);
    put32(o, 20, e.rh);
    putF(o, 24, rw / ow);
    putF(o, 28, rh / oh);
    putF(o, 32, ow / rw);
    putF(o, 36, oh / rh);
    putF(o, 40, settings.jitter[0]);
    putF(o, 44, settings.jitter[1]);
    put32(o, 56, e.w0);
    put32(o, 60, e.h0);
    put32(o, 140, e.rw - 1);
    put32(o, 144, e.rh - 1);
    putF(o, 152, settings.frameTimeMs);
    o[160] = rrPresent(images.specularHitDistance) ? 1u : 0u;
    o[432] = rrPresent(images.colorBeforeParticles) ? 1u : 0u; // the particle layer is composited (textures 392 / 416,
                                                               // surface 384)
    put64(o, 256, e.arenaOffset(RHkpnArena));
    putF(o, 336, settings.motionScale[0] * (ow / rw));
    putF(o, 340, settings.motionScale[1] * (oh / rh));
    put32(o, 440,
          (reset ? 1u : 0u) | (settings.depthInverted ? 0x10000u : 0u) | (settings.hardwareDepth ? 0x1000000u : 0u));
    // the exposure: NGX's MODE 1 measurement (engine/exposure_k.comp) on 1 /
    // pre-exposure scaled colour
    const uint32_t gx = exposureGrid(e.rw, e.ow), gy = exposureGrid(e.rh, e.oh);
    e.exposure.origin[0] = e.exposure.origin[1] = 0;
    e.exposure.size[0] = static_cast<int32_t>(e.rw);
    e.exposure.size[1] = static_cast<int32_t>(e.rh);
    e.exposure.sampleSize[0] = static_cast<int32_t>(gx);
    e.exposure.sampleSize[1] = static_cast<int32_t>(gy);
    e.exposure.invCount = 1.0f / static_cast<float>(gx * gy);
    e.exposure.scale = 1.0f / (settings.preExposure > 0.0f ? settings.preExposure : 1.0f);
    // cuda_dldn_engine_auto_exposure_copy_kernel divides a constant of its own
    // (0d3FCE23E8E0000000), not its argument 0 (0.2195...), by max(mean luma,
    // 0.005): the capture's 0.919 = 0.23547 / 0.2562
    e.exposure.key = 0.23547087609767914f;
    // NGX's reset keeps the exposure state, the recurrent feature and the output
    // history and only clears the previous frame's enc0 feature q (a harness run
    // resetting every frame: enc0's textures 208, 488 and 496 carry the previous
    // frame from frame 2 on, 224 stays zero; capture capE3, 2026-10-11). The
    // first frame starts from zero entirely.
    e.exposure.flags = 1u | (e.frame == 0 ? 2u : 0u);
    e.reset = reset;
}
