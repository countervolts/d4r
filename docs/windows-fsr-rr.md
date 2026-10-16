# Native Windows FSR Ray Regeneration investigation

The native D3D12 provider check currently fails on RX 7700 XT. This does not
implement a working Windows FSR RR backend or DLSS RR replacement.

AMD SDK 2.3's unmodified denoiser 1.2.0.2740 loads successfully and accepts the
provider query, but returns an empty provider list. The installed driver reports
shader model 6.6. No context creation or denoiser dispatch follows this result.
The DLL's Authenticode signature was valid, signed by Advanced Micro Devices;
its SHA256 matches the Linux reference's tested binary:
`48f1e5888ba6a0a3d59a98b9751e37c392b0f7b8c223d0082d5c1f40642879d3`.

Linux's RDNA3 route uses vkd3d-proton, dxil-spirv scalar FP8 emulation, and
`wmma_rdna3_workaround`. Those are not native Windows driver features. Native
parity requires a compatible shader execution route as well as the DLSS guide
conversion and internal SR adapter. Merely importing the Linux backend or
selecting `Backend=fsr` would not supply those pieces.

## Reproduce without a game

```powershell
./scripts/windows/setup-fsr-rr.ps1
cmake --build build/native-engine-windows-gfx1101 --target d4r_fsr_rr_probe
./scripts/windows/test-fsr-rr.ps1 `
  -Probe build/native-engine-windows-gfx1101/d4r_fsr_rr_probe.exe `
  -DenoiserDll .tools/fsr-rr-1.2/amd_fidelityfx_denoiser_dx12.dll `
  -OutputDirectory build/fsr-rr-new-query -QueryOnly
```

Use a new output folder under ignored `build/`. The wrapper pins the DLL hash,
captures stdout/stderr, limits execution to 120 seconds, and propagates failure.
Without `-QueryOnly`, the harness proceeds only if a provider exists: it measures
48 synthetic frames with static/moving cameras and a reset against noise-free
signals. A successful query alone is not execution or image-quality evidence.
The Windows adapter helper queries the device's DXGI identity. It does not load
Proton driver shims or modify drivers, GPU settings, or security policy.

## Source provenance

`tools/fsr_rr_probe.cpp`, `tools/fsr_rr_scene.h`, `tools/d4r_fsr_rr.h`, and the
MIT-licensed FidelityFX API headers were imported from Linux reference
`34e6b0fe71d770729000a553f79b784e9b2dc279`. The probe includes the native Windows
identity helper instead of Linux's sysfs/vkd3d identity helper. AMD's DLL stays
under ignored `.tools/` and is not included in source or release packages.

Evidence: `build/fsr-rr-native-wrapper-query/probe.json`. No game was touched.
