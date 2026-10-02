#pragma once
#include <cstdint>
struct ID3D12Resource;
extern "C" {
void d4r_ngx_set_ull(void*, const char*, unsigned long long);
void d4r_ngx_set_float(void*, const char*, float);
void d4r_ngx_set_uint(void*, const char*, unsigned);
void d4r_ngx_set_int(void*, const char*, int);
void d4r_ngx_set_void(void*, const char*, void*);
void d4r_ngx_set_d3d12_resource(void*, const char*, ID3D12Resource*);
unsigned d4r_ngx_get_ull(void*, const char*, unsigned long long*);
unsigned d4r_ngx_get_float(void*, const char*, float*);
unsigned d4r_ngx_get_uint(void*, const char*, unsigned*);
unsigned d4r_ngx_get_int(void*, const char*, int*);
unsigned d4r_ngx_get_void(void*, const char*, void**);
unsigned d4r_ngx_get_d3d12_resource(void*, const char*, ID3D12Resource**);
void* d4r_ngx_parameters_create();
void d4r_ngx_parameters_destroy(void*);
}
namespace d4r::ngx {
inline unsigned uint_value(void* p, const char* name, unsigned fallback = 0) {
    unsigned value = fallback; (void)d4r_ngx_get_uint(p, name, &value); return value;
}
inline int int_value(void* p, const char* name, int fallback = 0) {
    int value = fallback; (void)d4r_ngx_get_int(p, name, &value); return value;
}
inline float float_value(void* p, const char* name, float fallback = 0) {
    float value = fallback; (void)d4r_ngx_get_float(p, name, &value); return value;
}
inline void copy_create(void* source, void* dest) {
    for (const char* name : {"Width", "Height", "OutWidth", "OutHeight", "CreationNodeMask", "VisibilityNodeMask",
         "DLSS.Hint.Render.Preset.DLAA", "DLSS.Hint.Render.Preset.Quality", "DLSS.Hint.Render.Preset.Balanced",
         "DLSS.Hint.Render.Preset.Performance", "DLSS.Hint.Render.Preset.UltraPerformance", "DLSS.Hint.Render.Preset.UltraQuality"}) {
        unsigned value = 0;
        if (d4r_ngx_get_uint(source, name, &value) == 1) d4r_ngx_set_uint(dest, name, value);
    }
    for (const char* name : {"PerfQualityValue", "DLSS.Feature.Create.Flags", "DLSS.Enable.Output.Subrects"}) {
        int value = 0;
        if (d4r_ngx_get_int(source, name, &value) == 1) d4r_ngx_set_int(dest, name, value);
    }
}
inline void copy_frame(void* source, void* dest) {
    for (const char* name : {"Jitter.Offset.X", "Jitter.Offset.Y", "Sharpness", "MV.Scale.X", "MV.Scale.Y",
         "DLSS.Pre.Exposure", "DLSS.Exposure.Scale", "FrameTimeDeltaInMsec"}) {
        const float fallback = name[0] == 'M' || name[0] == 'D' ? 1.f : name[0] == 'F' ? 16.666667f : 0.f;
        d4r_ngx_set_float(dest, name, float_value(source, name, fallback));
    }
    for (const char* name : {"Reset", "Disable.Watermark", "DLSS.Indicator.Invert.X.Axis", "DLSS.Indicator.Invert.Y.Axis"})
        d4r_ngx_set_int(dest, name, int_value(source, name));
    for (const char* name : {"DLSS.Render.Subrect.Dimensions.Width", "DLSS.Render.Subrect.Dimensions.Height",
         "DLSS.Input.Color.Subrect.Base.X", "DLSS.Input.Color.Subrect.Base.Y", "DLSS.Input.Depth.Subrect.Base.X",
         "DLSS.Input.Depth.Subrect.Base.Y", "DLSS.Input.MV.Subrect.Base.X", "DLSS.Input.MV.Subrect.Base.Y",
         "DLSS.Output.Subrect.Base.X", "DLSS.Output.Subrect.Base.Y"})
        d4r_ngx_set_uint(dest, name, uint_value(source, name));
}
}
