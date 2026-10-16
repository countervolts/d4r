// The synthetic ray-traced scene shared by tools/fsr_rr_probe.cpp (AMD's denoiser alone) and
// tools/rr_backend_check.cpp (a game's NGX Ray Reconstruction request through the shim): a camera over
// a checkered floor, three spheres and a wall, with smooth noise-free lighting signals and one-sample
// white-noise versions of them, every guide the denoisers read, and exact 2.5D motion. Because the
// noise-free signals are known, outputs are compared against ground truth rather than another
// denoiser.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace fsr_scene
{

// --- small math -----------------------------------------------------------------

struct V3
{
    float x, y, z;
};
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline V3 mul(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline V3 normalize(V3 a) { return a * (1.0f / std::sqrt(dot(a, a))); }
inline float saturate(float v) { return std::min(1.0f, std::max(0.0f, v)); }

inline uint16_t to_half(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (((bits >> 23) & 0xff) == 0xff)
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    if (exponent >= 31)
        return static_cast<uint16_t>(sign | 0x7bffu); // clamp to the largest finite half
    if (exponent <= 0)
    {
        if (exponent < -10)
            return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t half = mantissa >> shift;
        if ((mantissa >> (shift - 1)) & 1u)
            ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
    if (mantissa & 0x1000u)
        ++half;
    return static_cast<uint16_t>(half);
}

inline float from_half(uint16_t half)
{
    const uint32_t sign = (half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x3ffu;
    uint32_t bits;
    if (exponent == 0)
    {
        if (mantissa == 0)
            bits = sign;
        else
        {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0)
            {
                mantissa <<= 1;
                --exponent;
            }
            bits = sign | (exponent << 23) | ((mantissa & 0x3ffu) << 13);
        }
    }
    else if (exponent == 31)
        bits = sign | 0x7f800000u | (mantissa << 13);
    else
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}

// Row-major, row-vector (v' = vM) 4x4, the convention ffx_denoiser.h asks for.
struct M4
{
    float m[4][4];
};

inline M4 view_matrix(V3 eye, V3 target)
{
    const V3 forward = normalize(target - eye);
    const V3 right = normalize(cross({0, 1, 0}, forward));
    const V3 up = cross(forward, right);
    M4 v = {};
    v.m[0][0] = right.x, v.m[0][1] = up.x, v.m[0][2] = forward.x;
    v.m[1][0] = right.y, v.m[1][1] = up.y, v.m[1][2] = forward.y;
    v.m[2][0] = right.z, v.m[2][1] = up.z, v.m[2][2] = forward.z;
    v.m[3][0] = -dot(eye, right), v.m[3][1] = -dot(eye, up), v.m[3][2] = -dot(eye, forward), v.m[3][3] = 1;
    return v;
}

inline M4 projection_matrix(float fovY, float aspect, float zNear, float zFar)
{
    const float ys = 1.0f / std::tan(fovY * 0.5f);
    M4 p = {};
    p.m[0][0] = ys / aspect;
    p.m[1][1] = ys;
    p.m[2][2] = zFar / (zFar - zNear);
    p.m[2][3] = 1;
    p.m[3][2] = -zNear * zFar / (zFar - zNear);
    return p;
}

inline void transform(const M4& m, V3 v, float out[4])
{
    for (int c = 0; c < 4; ++c)
        out[c] = v.x * m.m[0][c] + v.y * m.m[1][c] + v.z * m.m[2][c] + m.m[3][c];
}

// --- the synthetic scene ----------------------------------------------------------

struct Camera
{
    V3 eye, target;
};

struct Sphere
{
    V3 center;
    float radius;
    V3 base;
    float roughness, metal;
};

inline const Sphere kSpheres[] = {
    {{0.0f, 0.2f, 6.0f}, 1.2f, {0.8f, 0.3f, 0.2f}, 0.35f, 0.0f},
    {{2.4f, -0.3f, 7.5f}, 0.7f, {0.95f, 0.9f, 0.8f}, 0.15f, 1.0f},
    {{-2.2f, -0.4f, 5.0f}, 0.6f, {0.2f, 0.5f, 0.85f}, 0.6f, 0.0f},
};
inline constexpr float kGroundY = -1.0f;
inline constexpr float kWallZ = 14.0f;
// Optional near-mirror pond in the floor (rr_backend_check --water): the reflection is the traced scene, with
// guides as Alan Wake 2 hands them for its water (flat diffuse albedo 0.5, specular albedo ~0.96, roughness ~0.047).
inline bool g_water = false;
inline constexpr float kWaterRoughness = 0.047f;
inline constexpr float kWaterMinX = -4.0f, kWaterMaxX = 4.5f, kWaterMinZ = 0.5f, kWaterMaxZ = 4.8f;
inline const V3 kLight = normalize({0.45f, 0.9f, -0.35f});

struct Hit
{
    float t = 1e30f;
    V3 position{}, normal{}, base{};
    float roughness = 1, metal = 0;
    bool valid = false, water = false;
};

inline bool hit_sphere(V3 origin, V3 direction, const Sphere& s, float& t)
{
    const V3 oc = origin - s.center;
    const float b = dot(oc, direction);
    const float c = dot(oc, oc) - s.radius * s.radius;
    const float h = b * b - c;
    if (h < 0)
        return false;
    const float root = -b - std::sqrt(h);
    if (root <= 1e-3f)
        return false;
    t = root;
    return true;
}

inline Hit trace(V3 origin, V3 direction)
{
    Hit hit;
    float t;
    for (const Sphere& s : kSpheres)
        if (hit_sphere(origin, direction, s, t) && t < hit.t)
        {
            hit.t = t;
            hit.position = origin + direction * t;
            hit.normal = normalize(hit.position - s.center);
            hit.base = s.base;
            hit.roughness = s.roughness;
            hit.metal = s.metal;
            hit.valid = true;
        }
    if (direction.y < -1e-4f)
    {
        t = (kGroundY - origin.y) / direction.y;
        if (t > 1e-3f && t < hit.t)
        {
            hit.t = t;
            hit.position = origin + direction * t;
            hit.normal = {0, 1, 0};
            const int checker = (static_cast<int>(std::floor(hit.position.x)) + static_cast<int>(std::floor(hit.position.z))) & 1;
            hit.base = checker ? V3{0.75f, 0.75f, 0.72f} : V3{0.22f, 0.24f, 0.25f};
            hit.roughness = 0.55f;
            hit.metal = 0;
            hit.valid = true;
            const V3 q = hit.position;
            if (g_water && q.x > kWaterMinX && q.x < kWaterMaxX && q.z > kWaterMinZ && q.z < kWaterMaxZ)
            {
                // small world-space ripples, so the reflection has fine detail a denoiser can lose
                hit.normal = normalize({0.035f * std::sin(5.1f * q.x + 1.3f * q.z) + 0.02f * std::sin(11.7f * q.z),
                                        1.0f,
                                        0.035f * std::cos(4.3f * q.z - 0.7f * q.x) + 0.02f * std::cos(13.1f * q.x)});
                hit.base = {0.5f, 0.5f, 0.5f};
                hit.roughness = kWaterRoughness;
                hit.water = true;
            }
        }
    }
    if (direction.z > 1e-4f)
    {
        t = (kWallZ - origin.z) / direction.z;
        if (t > 1e-3f && t < hit.t)
        {
            hit.t = t;
            hit.position = origin + direction * t;
            hit.normal = {0, 0, -1};
            hit.base = {0.55f, 0.5f, 0.45f};
            hit.roughness = 0.85f;
            hit.metal = 0;
            hit.valid = true;
        }
    }
    return hit;
}

inline bool shadowed(V3 position, V3 normal)
{
    float t;
    const V3 origin = position + normal * 2e-3f;
    for (const Sphere& s : kSpheres)
        if (hit_sphere(origin, kLight, s, t))
            return true;
    return false;
}

// AMD's ApproximateBRDF (denoising.md 5.5.2, after Ray Tracing Gems ch. 32).
inline V3 approximate_brdf(V3 f0, float alpha, float nov)
{
    nov = std::fabs(nov);
    const float x[4] = {1, nov, nov * nov, nov * nov * nov};
    const float y[4] = {1, alpha, alpha * alpha, alpha * alpha * alpha};
    const float m1[2][2] = {{0.99044f, -1.28514f}, {1.29678f, -0.755907f}};
    const float m2[3][3] = {{1.0f, 2.92338f, 59.4188f}, {20.3225f, -27.0302f, 222.592f}, {121.563f, 626.13f, 316.627f}};
    const float m3[2][2] = {{0.0365463f, 3.32707f}, {9.0632f, -9.04756f}};
    const float m4[3][3] = {{1.0f, 3.59685f, -1.36772f}, {9.04401f, -16.3174f, 9.22949f}, {5.56589f, 19.7886f, -20.2123f}};
    // HLSL mul(M, v) with M row-major: (M v)_i = sum_j M[i][j] v_j
    const float a0 = m1[0][0] * x[0] + m1[0][1] * x[1], a1 = m1[1][0] * x[0] + m1[1][1] * x[1];
    const float xw[3] = {x[0], x[1], x[3]}, yw[3] = {y[0], y[1], y[3]};
    float b[3];
    for (int i = 0; i < 3; ++i)
        b[i] = m2[i][0] * xw[0] + m2[i][1] * xw[1] + m2[i][2] * xw[2];
    const float c0 = m3[0][0] * x[0] + m3[0][1] * x[1], c1 = m3[1][0] * x[0] + m3[1][1] * x[1];
    const float xz[3] = {x[0], x[2], x[3]};
    float d[3];
    for (int i = 0; i < 3; ++i)
        d[i] = m4[i][0] * xz[0] + m4[i][1] * xz[1] + m4[i][2] * xz[2];
    float bias = (a0 * y[0] + a1 * y[1]) / (b[0] * yw[0] + b[1] * yw[1] + b[2] * yw[2]);
    const float scale = (c0 * y[0] + c1 * y[1]) / (d[0] * yw[0] + d[1] * yw[1] + d[2] * yw[2]);
    bias *= saturate(f0.y * 50);
    const float s = std::max(0.0f, scale), bb = std::max(0.0f, bias);
    return {f0.x * s + bb, f0.y * s + bb, f0.z * s + bb};
}

inline V3 reflect_dir(V3 direction, V3 normal) { return direction - normal * (2 * dot(direction, normal)); }

inline V3 environment(V3 direction)
{
    const float up = saturate(direction.y * 0.5f + 0.5f);
    return V3{0.35f, 0.45f, 0.6f} * (0.4f + 1.2f * up);
}

inline uint32_t hash32(uint32_t x)
{
    // lowbias32, as AMD's guide recommends for white noise
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline float uniform(uint32_t pixel, uint32_t frame, uint32_t stream)
{
    return (hash32(pixel * 0x9e3779b9u ^ hash32(frame * 4u + stream + 0x1234567u)) >> 8) * (1.0f / 16777216.0f);
}

enum Signal
{
    kDirectDiffuse,
    kDirectSpecular,
    kIndirectDiffuse,
    kIndirectSpecular,
    kSignalCount
};
// FfxApiDenoiserSignalFlags of each signal (DIRECT_DIFFUSE 2, DIRECT_SPECULAR 4, INDIRECT_DIFFUSE 16,
// INDIRECT_SPECULAR 32)
inline const uint32_t kSignalFlags[kSignalCount] = {1u << 1, 1u << 2, 1u << 4, 1u << 5};
inline const char* const kSignalNames[kSignalCount] = {"direct_diffuse", "direct_specular", "indirect_diffuse",
                                                       "indirect_specular"};

struct Frame
{
    // FSR Ray Regeneration's own inputs (fsr_rr_probe)
    std::vector<float> depth;       // signed linear (view z)
    std::vector<uint16_t> motion;   // RGBA16F: prevUV - curUV, prev z - cur z
    std::vector<uint32_t> normals;  // R10G10B10A2: oct normal, roughness, material 0
    std::vector<uint32_t> diffuseAlbedo, specularAlbedo; // RGBA8, sqrt encoded
    std::vector<float> clean[kSignalCount]; // RGB
    std::vector<uint16_t> noisy[kSignalCount]; // RGBA16F
    std::vector<uint8_t> valid;
    std::vector<V3> diffuseLinear, specularLinear;
    // What a DLSS Ray Reconstruction game hands NGX (rr_backend_check)
    std::vector<uint16_t> color;              // RGBA16F: the noisy signals remodulated and summed, sky added
    std::vector<V3> colorClean;               // the same from the noise-free signals: the ground truth
    std::vector<float> hardwareDepth;         // D32: NDC z of the projection (1 for the sky)
    std::vector<uint16_t> motionPixels;       // RG16F: (previous - current) position in render pixels
    std::vector<uint16_t> normalsRoughness;   // RGBA16F: signed world normal, w = linear roughness
    std::vector<uint32_t> diffuseAlbedoLinear, specularAlbedoLinear; // RGBA8 UNORM, linear
};

inline uint32_t pack_unorm8(V3 v)
{
    auto q = [](float f) { return static_cast<uint32_t>(std::lround(saturate(f) * 255.0f)); };
    return q(v.x) | (q(v.y) << 8) | (q(v.z) << 16) | (255u << 24);
}

inline uint32_t pack_normal(V3 n, float roughness)
{
    // AMD's NormalToOctahedronUv (denoising.md 5.3.1)
    const float l1 = std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z);
    float x = n.x / l1, y = n.y / l1;
    if (n.z < 0)
    {
        const float ox = (1.0f - std::fabs(y)) * (x >= 0 ? 1.0f : -1.0f);
        const float oy = (1.0f - std::fabs(x)) * (y >= 0 ? 1.0f : -1.0f);
        x = ox, y = oy;
    }
    auto q10 = [](float f) { return static_cast<uint32_t>(std::lround(saturate(f) * 1023.0f)); };
    return q10(x * 0.5f + 0.5f) | (q10(y * 0.5f + 0.5f) << 10) | (q10(roughness) << 20) | (0u << 30);
}

// Renders one frame. (jitterX, jitterY): the sample offset from each pixel's centre in pixels (DLSS's
// Jitter.Offset is its negation). Motion is that of the sampled surface point.
inline void render(Frame& frame, uint32_t width, uint32_t height, const Camera& camera, const Camera& previous,
                   const M4& projection, uint32_t frameIndex, float jitterX = 0.0f, float jitterY = 0.0f)
{
    const size_t count = static_cast<size_t>(width) * height;
    frame.depth.assign(count, 0);
    frame.motion.assign(count * 4, 0);
    frame.normals.assign(count, 0);
    frame.diffuseAlbedo.assign(count, 0);
    frame.specularAlbedo.assign(count, 0);
    frame.valid.assign(count, 0);
    frame.diffuseLinear.assign(count, V3{});
    frame.specularLinear.assign(count, V3{});
    frame.color.assign(count * 4, 0);
    frame.colorClean.assign(count, V3{});
    frame.hardwareDepth.assign(count, 1.0f);
    frame.motionPixels.assign(count * 2, 0);
    frame.normalsRoughness.assign(count * 4, 0);
    frame.diffuseAlbedoLinear.assign(count, 0);
    frame.specularAlbedoLinear.assign(count, 0);
    for (int s = 0; s < kSignalCount; ++s)
    {
        frame.clean[s].assign(count * 3, 0);
        frame.noisy[s].assign(count * 4, 0);
    }
    const M4 view = view_matrix(camera.eye, camera.target);
    const M4 previousView = view_matrix(previous.eye, previous.target);
    const V3 forward = normalize(camera.target - camera.eye);
    const V3 right = normalize(cross({0, 1, 0}, forward));
    const V3 up = cross(forward, right);
    const float tanX = 1.0f / projection.m[0][0], tanY = 1.0f / projection.m[1][1];
    for (uint32_t py = 0; py < height; ++py)
        for (uint32_t px = 0; px < width; ++px)
        {
            const size_t i = static_cast<size_t>(py) * width + px;
            const float u = (px + 0.5f + jitterX) / width, v = (py + 0.5f + jitterY) / height;
            const float ndcX = u * 2 - 1, ndcY = 1 - v * 2;
            const V3 direction = normalize(forward + right * (ndcX * tanX) + up * (ndcY * tanY));
            const Hit hit = trace(camera.eye, direction);
            const uint32_t pixelId = static_cast<uint32_t>(i);
            frame.color[i * 4 + 3] = to_half(1.0f);
            frame.normalsRoughness[i * 4 + 2] = to_half(-1.0f);
            frame.normalsRoughness[i * 4 + 3] = to_half(1.0f);
            if (!hit.valid)
            {
                // Sky: outside linearDepthBounds, so the runtime passes it through.
                frame.depth[i] = 5000.0f;
                frame.normals[i] = pack_normal({0, 0, -1}, 1);
                const V3 sky = environment(direction);
                frame.colorClean[i] = sky;
                frame.color[i * 4 + 0] = to_half(sky.x);
                frame.color[i * 4 + 1] = to_half(sky.y);
                frame.color[i * 4 + 2] = to_half(sky.z);
                continue;
            }
            frame.valid[i] = 1;
            float clip[4];
            transform(view, hit.position, clip);
            frame.depth[i] = clip[2];
            float projected[4];
            transform(projection, {clip[0], clip[1], clip[2]}, projected);
            frame.hardwareDepth[i] = projected[2] / projected[3];
            float prevView[4], prevClip[4];
            transform(previousView, hit.position, prevView);
            const M4& p = projection;
            const V3 pv = {prevView[0], prevView[1], prevView[2]};
            transform(p, pv, prevClip);
            const float prevU = (prevClip[0] / prevClip[3]) * 0.5f + 0.5f;
            const float prevV = 0.5f - (prevClip[1] / prevClip[3]) * 0.5f;
            frame.motion[i * 4 + 0] = to_half(prevU - u);
            frame.motion[i * 4 + 1] = to_half(prevV - v);
            frame.motion[i * 4 + 2] = to_half(prevView[2] - clip[2]);
            frame.motion[i * 4 + 3] = to_half(0);
            frame.motionPixels[i * 2 + 0] = to_half((prevU - u) * width);
            frame.motionPixels[i * 2 + 1] = to_half((prevV - v) * height);

            const V3 diffuse = hit.base * (1.0f - hit.metal);
            const V3 f0 = V3{0.04f, 0.04f, 0.04f} * (1.0f - hit.metal) + hit.base * hit.metal;
            const V3 toEye = normalize(camera.eye - hit.position);
            const float nov = std::max(dot(hit.normal, toEye), 1e-3f);
            const V3 specular = hit.water ? V3{0.96f, 0.96f, 0.96f} : approximate_brdf(f0, hit.roughness * hit.roughness, nov);
            frame.diffuseLinear[i] = diffuse;
            frame.specularLinear[i] = specular;
            frame.diffuseAlbedo[i] = pack_unorm8({std::sqrt(diffuse.x), std::sqrt(diffuse.y), std::sqrt(diffuse.z)});
            frame.specularAlbedo[i] = pack_unorm8({std::sqrt(specular.x), std::sqrt(specular.y), std::sqrt(specular.z)});
            frame.diffuseAlbedoLinear[i] = pack_unorm8(diffuse);
            frame.specularAlbedoLinear[i] = pack_unorm8(specular);
            frame.normals[i] = pack_normal(hit.normal, hit.roughness);
            frame.normalsRoughness[i * 4 + 0] = to_half(hit.normal.x);
            frame.normalsRoughness[i * 4 + 1] = to_half(hit.normal.y);
            frame.normalsRoughness[i * 4 + 2] = to_half(hit.normal.z);
            frame.normalsRoughness[i * 4 + 3] = to_half(hit.roughness);

            // Demodulated (albedo-free) clean signals, smooth in world space.
            const float visibility = shadowed(hit.position, hit.normal) ? 0.0f : 1.0f;
            const float ndl = std::max(dot(hit.normal, kLight), 0.0f);
            const V3 reflected = direction - hit.normal * (2 * dot(direction, hit.normal));
            const float rdl = std::max(dot(reflected, kLight), 0.0f);
            const float glossExponent = 2.0f / (hit.roughness * hit.roughness * hit.roughness * hit.roughness + 0.02f);
            V3 clean[kSignalCount];
            clean[kDirectDiffuse] = V3{1.0f, 0.95f, 0.85f} * (2.5f * ndl * visibility);
            clean[kDirectSpecular] = V3{1.0f, 0.95f, 0.85f} * std::min(8.0f, 3.0f * std::pow(rdl, glossExponent) * visibility * (ndl > 0));
            const float wobble = 0.12f * std::sin(1.7f * hit.position.x) * std::sin(1.3f * hit.position.z);
            clean[kIndirectDiffuse] = environment(hit.normal) * (0.6f + wobble);
            clean[kIndirectSpecular] = environment(reflected) * (0.8f + 0.2f * (1 - hit.roughness));
            if (hit.water)
            {
                // The pond's light is its reflection: the traced scene, lit noise-free. Its diffuse lobe is
                // nearly dark despite the 0.5 guide albedo.
                clean[kDirectDiffuse] = clean[kDirectDiffuse] * 0.01f;
                clean[kIndirectDiffuse] = clean[kIndirectDiffuse] * 0.01f;
                const Hit bounce = trace(hit.position + hit.normal * 2e-3f, reflected);
                if (bounce.valid && !bounce.water)
                {
                    const float bounceVisibility = shadowed(bounce.position, bounce.normal) ? 0.0f : 1.0f;
                    const float bounceNdl = std::max(dot(bounce.normal, kLight), 0.0f);
                    const V3 light = V3{1.0f, 0.95f, 0.85f} * (2.5f * bounceNdl * bounceVisibility) +
                                     environment(bounce.normal) * 0.6f;
                    clean[kIndirectSpecular] = mul(bounce.base * (1.0f - bounce.metal), light) +
                                               environment(reflect_dir(reflected, bounce.normal)) * (0.04f + 0.9f * bounce.metal);
                }
                else
                    clean[kIndirectSpecular] = environment(reflected);
            }
            V3 combined{}, combinedClean{};
            for (int s = 0; s < kSignalCount; ++s)
            {
                // One-sample-per-pixel white noise: a mean-one exponential weight shared by RGB,
                // like a single path sample's throughput.
                float sample = -std::log(std::max(1e-6f, 1.0f - uniform(pixelId, frameIndex, s)));
                if (hit.water && s == kIndirectSpecular)
                    sample = 1.0f + 0.35f * (sample - 1.0f); // a mirror's reflection ray is far less noisy
                frame.clean[s][i * 3 + 0] = clean[s].x;
                frame.clean[s][i * 3 + 1] = clean[s].y;
                frame.clean[s][i * 3 + 2] = clean[s].z;
                frame.noisy[s][i * 4 + 0] = to_half(clean[s].x * sample);
                frame.noisy[s][i * 4 + 1] = to_half(clean[s].y * sample);
                frame.noisy[s][i * 4 + 2] = to_half(clean[s].z * sample);
                // A: ray hit distance for the indirect signals, any non-negative value for direct.
                float alpha = 0.0f;
                if (s == kIndirectDiffuse)
                    alpha = 0.5f + 3.0f * uniform(pixelId, frameIndex, 10 + s);
                else if (s == kIndirectSpecular)
                {
                    const Hit bounce = trace(hit.position + hit.normal * 2e-3f, reflected);
                    alpha = bounce.valid ? std::min(bounce.t, 100.0f) : 100.0f;
                }
                frame.noisy[s][i * 4 + 3] = to_half(alpha);
                const V3 albedo = s == kDirectDiffuse || s == kIndirectDiffuse ? diffuse : specular;
                combined = combined + mul(clean[s] * sample, albedo);
                combinedClean = combinedClean + mul(clean[s], albedo);
            }
            frame.colorClean[i] = combinedClean;
            frame.color[i * 4 + 0] = to_half(combined.x);
            frame.color[i * 4 + 1] = to_half(combined.y);
            frame.color[i * 4 + 2] = to_half(combined.z);
        }
}

} // namespace fsr_scene
