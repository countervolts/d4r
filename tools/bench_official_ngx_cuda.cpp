// Official NVIDIA NGX CUDA timing probe; see docs/nvidia-blackwell-presets.md.
// Build with NVIDIA DLSS SDK includes/static library and CUDA driver headers.
#include "fsr_rr_scene.h"
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_defs_dlssd.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <string>
#include <vector>
void ck(CUresult r) {
  if (r) {
    const char *s;
    cuGetErrorString(r, &s);
    fprintf(stderr, "CUDA %d %s\n", r, s);
    exit(2);
  }
}
void ng(NVSDK_NGX_Result r, const char *s) {
  if (r != 1) {
    fprintf(stderr, "NGX %s 0x%x\n", s, r);
    exit(3);
  }
}
struct Tex {
  CUarray a;
  CUtexObject t;
};
Tex tex(int w, int h, int channels, CUarray_format fmt, const void *data,
        int bytes) {
  Tex t{};
  CUDA_ARRAY_DESCRIPTOR d{(size_t)w, (size_t)h, fmt, (unsigned)channels};
  ck(cuArrayCreate(&t.a, &d));
  CUDA_MEMCPY2D c{};
  c.srcMemoryType = CU_MEMORYTYPE_HOST;
  c.srcHost = data;
  c.srcPitch = w * channels * bytes;
  c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  c.dstArray = t.a;
  c.WidthInBytes = c.srcPitch;
  c.Height = h;
  ck(cuMemcpy2D(&c));
  CUDA_RESOURCE_DESC r{};
  r.resType = CU_RESOURCE_TYPE_ARRAY;
  r.res.array.hArray = t.a;
  CUDA_TEXTURE_DESC s{};
  s.addressMode[0] = s.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  s.filterMode =
      channels == 4 ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;
  s.flags = CU_TRSF_NORMALIZED_COORDINATES;
  ck(cuTexObjectCreate(&t.t, &r, &s, nullptr));
  return t;
}
int main(int argc, char **argv) {
  if (argc < 7) {
    fprintf(stderr, "bench BINARY_DIR PRESET IW IH OW OH [frames]\n");
    return 1;
  }
  int preset = atoi(argv[2]), iw = atoi(argv[3]), ih = atoi(argv[4]),
      ow = atoi(argv[5]), oh = atoi(argv[6]),
      frames = argc > 7 ? atoi(argv[7]) : 100;
  if (iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0 || frames <= 0) {
    fprintf(stderr, "Dimensions and frame count must be positive\n");
    return 1;
  }
  const bool rr = std::getenv("D4R_BENCH_RR") != nullptr;
  const bool legacy = std::getenv("D4R_BENCH_LEGACY") != nullptr;
  const int flags =
      std::getenv("D4R_BENCH_FLAGS")
          ? std::strtol(std::getenv("D4R_BENCH_FLAGS"), nullptr, 0)
          : (rr ? 3 : 0x4b);
  const auto feature = rr ? NVSDK_NGX_Feature_RayReconstruction
                          : NVSDK_NGX_Feature_SuperSampling;
  ck(cuInit(0));
  CUdevice dev;
  ck(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  ck(cuDevicePrimaryCtxRetain(&ctx, dev));
  ck(cuCtxSetCurrent(ctx));
  char name[256];
  ck(cuDeviceGetName(name, 256, dev));
  printf("GPU %s preset=%d %dx%d -> %dx%d\n", name, preset, iw, ih, ow, oh);
  auto eval = &NVSDK_NGX_CUDA_EvaluateFeature;
  auto release = &NVSDK_NGX_CUDA_ReleaseFeature;
  NVSDK_NGX_CUDADevice cd{ctx, nullptr};
  std::string directory = argv[1];
  std::wstring wideDirectory(directory.begin(), directory.end());
  const wchar_t *paths[] = {wideDirectory.c_str()};
  NVSDK_NGX_FeatureCommonInfo info{};
  info.PathListInfo.Path = paths;
  info.PathListInfo.Length = 1;
  // Older snippets (including RR 310.3.0) do not export
  // Init_Ext1/CreateFeature1.
  ng(legacy ? NVSDK_NGX_CUDA_Init(0x1337, L"/tmp", &info)
            : NVSDK_NGX_CUDA_Init1(0x1337, L"/tmp", &cd, &info),
     "init");
  NVSDK_NGX_Parameter *pp = nullptr;
  ng(NVSDK_NGX_CUDA_AllocateParameters(&pp), "params");
  auto &p = *pp;
  p.Set("Width", (unsigned)iw);
  p.Set("Height", (unsigned)ih);
  p.Set("OutWidth", (unsigned)ow);
  p.Set("OutHeight", (unsigned)oh);
  p.Set("PerfQualityValue", (int)NVSDK_NGX_PerfQuality_Value_MaxPerf);
  p.Set("DLSS.Feature.Create.Flags", flags);
  printf("CONFIG feature=%d flags=0x%x quality=MaxPerf warmup=40 measured=%d "
         "directory=%s\n",
         int(feature), flags, frames, argv[1]);
  for (auto tier : {"DLAA", "Quality", "Balanced", "Performance",
                    "UltraPerformance", "UltraQuality"})
    p.Set((std::string(rr ? "RayReconstruction.Hint.Render.Preset."
                          : "DLSS.Hint.Render.Preset.") +
           tier)
              .c_str(),
          (unsigned)preset);
  CUdeviceptr scratch = 0;
  if (rr) {
    p.Set("DLSS.Denoise.Mode", 1);
    p.Set("DLSS.Roughness.Mode", 1);
    p.Set("DLSS.Use.HW.Depth", 0u);
    p.Set("Input1", (void *)ctx);
    p.Set("Input2", (void *)nullptr);
    ck(cuMemAlloc(&scratch, 64 * 1024 * 1024));
    p.Set("Scratch", (void *)scratch);
    p.Set("Scratch.SizeInBytes", (unsigned long long)(64 * 1024 * 1024));
  }
  NVSDK_NGX_Handle *h = nullptr;
  ng(legacy ? NVSDK_NGX_CUDA_CreateFeature(feature, &p, &h)
            : NVSDK_NGX_CUDA_CreateFeature1(&cd, feature, &p, &h),
     "create");
  // NGX can load snippets into a separate linker namespace; /proc maps includes
  // those.
  std::ifstream mappings("/proc/self/maps");
  std::string mapping, previousLibrary;
  while (std::getline(mappings, mapping)) {
    const auto name = mapping.find('/');
    if (name != std::string::npos &&
        mapping.find("libnvidia-ngx") != std::string::npos) {
      const auto library = mapping.substr(name);
      if (library != previousLibrary)
        printf("LIBRARY %s\n", library.c_str());
      previousLibrary = library;
    }
  }
  std::vector<unsigned short> color((size_t)iw * ih * 4),
      motion((size_t)iw * ih * 2);
  std::vector<float> depth((size_t)iw * ih, 0.5f);
  for (int y = 0; y < ih; y++)
    for (int x = 0; x < iw; x++) {
      size_t i = ((size_t)y * iw + x) * 4;
      color[i] = 0x3400 + ((x / 8 + y / 8) % 8) * 128;
      color[i + 1] = 0x3800;
      color[i + 2] = 0x3000;
      color[i + 3] = 0x3c00;
    }
  fsr_scene::Frame scene;
  const fsr_scene::Camera camera{{-0.6f, 0.6f, -1.5f}, {0.f, -0.1f, 6.5f}};
  auto projection =
      fsr_scene::projection_matrix(1.f, float(iw) / ih, 0.1f, 1000.f);
  auto view = fsr_scene::view_matrix(camera.eye, camera.target);
  if (rr) {
    fsr_scene::render(scene, iw, ih, camera, camera, projection, 0);
    color = scene.color;
    motion = scene.motionPixels;
    depth = scene.depth;
    for (size_t i = 0; i < depth.size(); ++i)
      if (!scene.valid[i])
        depth[i] = 1.e7f;
  }
  float one = 1;
  auto ct = tex(iw, ih, 4, CU_AD_FORMAT_HALF, color.data(), 2);
  auto mt = tex(iw, ih, 2, CU_AD_FORMAT_HALF, motion.data(), 2);
  auto dt = tex(iw, ih, 1, CU_AD_FORMAT_FLOAT, depth.data(), 4);
  auto et = tex(1, 1, 1, CU_AD_FORMAT_FLOAT, &one, 4);
  Tex normal{}, diffuse{}, specular{}, hit{};
  if (rr) {
    normal =
        tex(iw, ih, 4, CU_AD_FORMAT_HALF, scene.normalsRoughness.data(), 2);
    diffuse = tex(iw, ih, 4, CU_AD_FORMAT_UNSIGNED_INT8,
                  scene.diffuseAlbedoLinear.data(), 1);
    specular = tex(iw, ih, 4, CU_AD_FORMAT_UNSIGNED_INT8,
                   scene.specularAlbedoLinear.data(), 1);
    std::vector<float> distances((size_t)iw * ih, 4.f);
    hit = tex(iw, ih, 1, CU_AD_FORMAT_FLOAT, distances.data(), 4);
    p.Set("GBuffer.Normals", (void *)&normal.t);
    p.Set("DLSS.Input.DiffuseAlbedo", (void *)&diffuse.t);
    p.Set("DLSS.Input.SpecularAlbedo", (void *)&specular.t);
    p.Set("DLSSD.SpecularHitDistance", (void *)&hit.t);
    p.Set("WorldToViewMatrix", (void *)view.m);
    p.Set("ViewToClipMatrix", (void *)projection.m);
  }
  CUDA_ARRAY3D_DESCRIPTOR od{};
  od.Width = ow;
  od.Height = oh;
  od.Format = CU_AD_FORMAT_HALF;
  od.NumChannels = 4;
  od.Flags = CUDA_ARRAY3D_SURFACE_LDST;
  CUarray oa;
  ck(cuArray3DCreate(&oa, &od));
  CUDA_RESOURCE_DESC rd{};
  rd.resType = CU_RESOURCE_TYPE_ARRAY;
  rd.res.array.hArray = oa;
  CUsurfObject out;
  ck(cuSurfObjectCreate(&out, &rd));
  p.Set("Color", (void *)&ct.t);
  p.Set("Depth", (void *)&dt.t);
  p.Set("MotionVectors", (void *)&mt.t);
  p.Set("ExposureTexture", (void *)&et.t);
  p.Set("Output", (void *)&out);
  p.Set("MV.Scale.X", 1.f);
  p.Set("MV.Scale.Y", 1.f);
  p.Set("DLSS.Pre.Exposure", 1.f);
  p.Set("DLSS.Exposure.Scale", 1.f);
  p.Set("FrameTimeDeltaInMsec", 16.6667f);
  p.Set("Sharpness", 0.f);
  p.Set("DLSS.Render.Subrect.Dimensions.Width", (unsigned)iw);
  p.Set("DLSS.Render.Subrect.Dimensions.Height", (unsigned)ih);
  std::vector<unsigned short> sentinel((size_t)ow * oh * 4, 0x7e00);
  CUDA_MEMCPY2D sentinelCopy{};
  sentinelCopy.srcMemoryType = CU_MEMORYTYPE_HOST;
  sentinelCopy.srcHost = sentinel.data();
  sentinelCopy.srcPitch = ow * 8;
  sentinelCopy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  sentinelCopy.dstArray = oa;
  sentinelCopy.WidthInBytes = ow * 8;
  sentinelCopy.Height = oh;
  ck(cuMemcpy2D(&sentinelCopy));
  sentinel.clear();
  sentinel.shrink_to_fit();
  CUevent a, b;
  ck(cuEventCreate(&a, 0));
  ck(cuEventCreate(&b, 0));
  std::vector<float> times;
  for (int f = 0; f < frames + 40; f++) {
    p.Set("Reset", f == 0 ? 1 : 0);
    p.Set("Jitter.Offset.X", (f % 2 ? 0.25f : -0.25f));
    p.Set("Jitter.Offset.Y", (f % 4 < 2 ? 0.25f : -0.25f));
    ck(cuEventRecord(a, 0));
    ng(eval(h, &p, nullptr), "eval");
    ck(cuEventRecord(b, 0));
    ck(cuEventSynchronize(b));
    float ms;
    ck(cuEventElapsedTime(&ms, a, b));
    if (f >= 40) {
      times.push_back(ms);
      printf("FRAME %d %.6f\n", f - 40, ms);
    }
  }
  std::vector<unsigned short> output((size_t)ow * oh * 4);
  CUDA_MEMCPY2D cp{};
  cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  cp.srcArray = oa;
  cp.dstMemoryType = CU_MEMORYTYPE_HOST;
  cp.dstHost = output.data();
  cp.dstPitch = ow * 8;
  cp.WidthInBytes = ow * 8;
  cp.Height = oh;
  ck(cuMemcpy2D(&cp));
  unsigned long long hash = 1469598103934665603ULL;
  size_t nonzero = 0, nonfinite = 0;
  for (auto v : output) {
    hash = (hash ^ v) * 1099511628211ULL;
    nonzero += v != 0;
    nonfinite += (v & 0x7c00) == 0x7c00;
  }
  printf("OUTPUT hash=%llx nonzero=%zu/%zu nonfinite=%zu\n", hash, nonzero,
         output.size(), nonfinite);
  if (nonfinite || !nonzero) {
    fprintf(stderr, "Invalid output\n");
    return 5;
  }
  std::sort(times.begin(), times.end());
  printf("RESULT preset=%d input=%dx%d output=%dx%d median_ms=%.6f p10_ms=%.6f "
         "p90_ms=%.6f frames=%d\n",
         preset, iw, ih, ow, oh, times[times.size() / 2],
         times[times.size() / 10], times[times.size() * 9 / 10], frames);
  ng(release(h), "release");
  return 0;
}
