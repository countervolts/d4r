# RDNA 4 optimization candidates (2026-10-17)

These changes expose target-specific paths for gfx1200 and gfx1201. The available
GPU is an RX 7700 XT (gfx1101), so **no RDNA 4 speedup or output equivalence has
been measured**. Compilation and instruction checks do not establish either.
Existing packages and games were not updated.

## Paths

- **K:** `build-native-k-experiment.ps1 -F16Wmma` exposes the existing gfx12 native
  FP16-accumulator WMMA implementation. It requires the strict K baseline, rejects
  gfx11 and incompatible packed-accumulator storage, and records the selection in
  candidate metadata. It remains experimental pending layer and temporal-frame
  equality checks. The default kernel arithmetic is unchanged.
- **M:** `build-native-m-fp8.ps1 -GpuArch gfx1200|gfx1201` builds the existing native
  FP8 path for either target, including optional hardware packing and token lanes.
  The ISA gate requires FP8 WMMA instructions. The separate strict FP16 candidate
  builder now supports both gfx12 targets with native lane exchange; gfx11-only
  LDS/DS flags are excluded. Neither candidate is selected automatically.
- **DLSS RR D/E:** the Windows model builder now accepts `--gpu-arch gfx1200|gfx1201`.
  It uses the existing native FP8 cooperative-matrix shaders, omits gfx11's widened
  weight copies and records `arithmetic gfx12`. Every native network shader must
  declare Float8EXT, Float8CooperativeMatrixEXT and CooperativeMatrixKHR.
  The Windows Vulkan bridge enables and requires the float8 extension/features on
  gfx12. Model deployment requires matching runtime `gpu-target.json`; the bridge
  rejects a model from the other arithmetic family. Exact conversion decorations
  remain enabled; relaxed arithmetic is not introduced. The NGX regression helper
  accepts the same `--gpu-arch` to select its format module and native directory.
- **FSR RR:** gfx12 no longer receives `wmma_rdna3_workaround` by default. The local
  dxil-spirv implementation applies gfx11-specific column mapping and FP16 RTZ
  under that setting. gfx11 retains it. Explicit process overrides survive; lookup
  uses the Windows environment rather than the C runtime's stale snapshot.
  This removes an inappropriate target setting; native FSR execution and its
  performance still depend on the target Vulkan driver's capabilities.
  A subsequent change now selects direct D3D12 on gfx12 by default, bypassing
  this compatibility route entirely. See [native route status](windows-fsr-rr.md).

AMD documents RDNA 4's native FP8 matrix support in its
[matrix-core guide](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/).
That hardware capability does not establish Windows Vulkan driver support.

## Build examples

Use a fresh ignored output directory for each candidate. Replace DLL paths with
the locally supplied pinned binaries. Do not install candidates before validating
their outputs and paired timings on the target GPU.

```powershell
scripts/windows/build-native-k-experiment.ps1 -GpuArch gfx1201 -Layer enc1 -F16Wmma -OutputDirectory build/my-k-gfx1201
scripts/windows/build-native-m-fp8.ps1 -GpuArch gfx1201 -DlssDll PATH/nvngx_dlss.dll -HardwarePacking -TokenLanes -OutputDirectory build/my-m-gfx1201
python tools/windows/build_rr_model.py --gpu-arch gfx1201 --preset D --dll PATH/nvngx_dlssd.dll --output build/my-rr-d-gfx1201
```

Use gfx1200 consistently for that target. Build the shim/probe with the matching
`D4R_GPU_ARCH`. Deployment can stage a verified private model with
`deploy_rr_model.py --model MODEL --runtime RUNTIME`; selection still requires
`--enable` or an explicit model path. No relaxed model is built by this recipe.

## Checks performed

- K enc1 FP16 WMMA candidates compile for both targets. All eleven baseline K and
  five baseline M objects additionally compile for gfx1201.
- All five M native FP8 modules compile for both targets, with native WMMA and
  hardware packing instructions verified by the builder. Strict FP16 M candidate
  sets also compile for both targets.
- Complete D/gfx1201 and E/gfx1200 RR models compile. All eleven native network
  stages per model pass the SPIR-V native FP8 capability gate.
- The shim and standalone RR runner build for gfx1201; the gfx1101 shim, probe and
  runner also rebuild successfully.
- 13 RR Python tests and 16 existing CTest checks pass on gfx1101. Some legacy
  CTest names contain gfx1201/gfx12; these runs used the configured gfx1101 target.
- Native DLSS RR passes six synthetic camera/reset frames against the existing
  reference, byte-for-byte. Evidence: `build/rdna4-opt-rr-rdna3-regression-v3/`.
- FSR RR passes 24 moving-camera synthetic frames with exact same-frame output
  widening. Tail RMSE is approximately 0.1066 versus noisy input 0.4862. The log
  confirms gfx1101 still selects `wmma_rdna3_workaround`. Evidence:
  `build/rdna4-opt-fsr-rdna3-regression-v3/`.

The additional NGX `--contracts --without-sr` run fails at
`RR advertised on unsupported WARP adapter`, before inference. That capability
contract is unresolved; the successful six-frame run does not clear it.

Candidate artifacts remain under `build/rdna4-opt-*`. RDNA 4 layer/frame equality,
real-game quality, GPU timings and paired performance comparisons remain open.
