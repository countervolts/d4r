// CPU-only checks of the NGX RR snapshot contract. No model or GPU dispatch.
#include <volk.h>
#include "rr_ngx_snapshot.h"
#include <cstdio>
#include <limits>

namespace {
struct Parameters {
    void* value = d4r_ngx_parameters_create();
    Parameters() { if (!value) throw std::runtime_error("Parameter allocation failed"); }
    ~Parameters() { if (value) d4r_ngx_parameters_destroy(value); }
};
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F operation, const char* message) {
    try { operation(); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error(message);
}
}
int main() try {
    using namespace d4r::win::rr;
    Parameters parameters;
    void* p = parameters.value;
    // RR callers can publish dimensions as signed NGX integers.
    d4r_ngx_set_int(p, "Width", 64); d4r_ngx_set_uint(p, "Height", 48);
    d4r_ngx_set_uint(p, "OutWidth", 128); d4r_ngx_set_uint(p, "OutHeight", 96);
    d4r_ngx_set_int(p, "DLSS.Feature.Create.Flags", 8);
    d4r_ngx_set_int(p, "DLSS.Use.HW.Depth", 1);
    d4r_ngx_set_int(p, "DLSS.Roughness.Mode", 1);
    d4r_ngx_set_uint(p, "RayReconstruction.Hint.Render.Preset.Quality", 4);
    Creation creation(p);
    require(creation.width == 64 && creation.presets[1] == 4, "RR creation values lost");
    float view[16]{}, clip[16]{};
    for (unsigned i = 0; i < 16; ++i) { view[i] = float(i + 1); clip[i] = float(i + 32); }
    d4r_ngx_set_void(p, "WorldToViewMatrix", view); d4r_ngx_set_void(p, "ViewToClipMatrix", clip);
    d4r_ngx_set_float(p, "Jitter.Offset.X", .25f);
    d4r_ngx_set_float(p, "DLSS.Pre.Exposure", 2.f);
    d4r_ngx_set_uint(p, "DLSS.Input.Normals.Subrect.Base.X", 7);
    d4r_ngx_set_uint(p, "DLSSD.SpecularHitDistance.Subrect.Base.X", 5);
    d4r_ngx_set_uint(p, "DLSSD.ColorBeforeParticles.Subrect.Base.Y", 9);
    d4r_ngx_set_uint(p, "DLSS.Use.HW.Depth", 0);
    Snapshot frame(creation, p);
    view[0] = -91; clip[15] = -93;
    d4r_ngx_set_float(p, "DLSS.Pre.Exposure", 3.f);
    d4r_ngx_set_uint(p, "Width", 777);
    d4r_ngx_set_uint(p, "RayReconstruction.Hint.Render.Preset.Quality", 5);
    require(frame.settings.worldToView[0] == 1 && frame.settings.viewToClip[15] == 47, "RR matrices were not copied");
    require(frame.settings.preExposure == 2 && frame.settings.jitter[0] == .25f, "RR scalars were not copied");
    require(creation.width == 64 && creation.presets[1] == 4, "RR creation depends on mutable parameters");
    require(frame.settings.depthInverted && !frame.settings.hardwareDepth, "RR frame override lost");
    require(frame.renderWidth == 64 && frame.renderHeight == 48, "RR render dimensions lost creation fallback");
    require(frame.settings.roughnessPacked && frame.origins[Snapshot::Roughness][0] == 7, "RR packed roughness origin lost");
    require(frame.origins[Snapshot::HitDistance][0] == 5 && frame.origins[Snapshot::BeforeParticles][1] == 9,
        "RR optional guide origins lost");
    auto copied = frame;
    require(copied.settings.worldToView[0] == 1, "Copied RR snapshot lost matrix ownership");
    {
        Parameters frameParameters;
        d4r_ngx_set_void(frameParameters.value, "WorldToViewMatrix", view);
        d4r_ngx_set_void(frameParameters.value, "ViewToClipMatrix", clip);
        Snapshot baseline(creation, frameParameters.value);
        require(baseline.settings.hardwareDepth && baseline.settings.roughnessPacked,
            "RR frame settings lost creation baseline");
        require(baseline.settings.preExposure == 1 && baseline.settings.motionScale[0] == 1,
            "RR frame scalar defaults incorrect");
    }
    view[3] = std::numeric_limits<float>::infinity();
    rejects([&] { Snapshot bad(creation, p); }, "Nonfinite RR matrix accepted");
    view[3] = 4;
    d4r_ngx_set_float(p, "DLSS.Pre.Exposure", 0);
    rejects([&] { Snapshot bad(creation, p); }, "Zero RR pre-exposure accepted");
    d4r_ngx_set_float(p, "DLSS.Pre.Exposure", 1);
    d4r_ngx_set_float(p, "Jitter.Offset.Y", std::numeric_limits<float>::quiet_NaN());
    rejects([&] { Snapshot bad(creation, p); }, "Nonfinite RR jitter accepted");
    d4r_ngx_set_float(p, "Jitter.Offset.Y", 0);
    d4r_ngx_set_void(p, "ViewToClipMatrix", nullptr);
    rejects([&] { Snapshot bad(creation, p); }, "Missing RR matrix accepted");
    d4r_ngx_set_uint(p, "Width", 0);
    rejects([&] { Creation bad(p); }, "Zero RR creation width accepted");
    // The completed snapshot must still be valid after the parameter object dies.
    d4r_ngx_parameters_destroy(parameters.value); parameters.value = nullptr;
    require(copied.settings.viewToClip[15] == 47 && copied.settings.preExposure == 2, "RR snapshot retained parameter storage");
    std::puts("PASS RR_SNAPSHOT cpu_only=1 matrices_owned=1 scalars_owned=1 creation_fallback=1 invalid_rejected=1");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL RR_SNAPSHOT %s\n", error.what()); return 1;
}
