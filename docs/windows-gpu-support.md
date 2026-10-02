# Windows GPU targets

The native Windows build accepts **gfx1200** and **gfx1201**. It builds a
separate shim, diagnostics and code objects for each target. The default stays
gfx1201. GPU selection uses HIP's real `gcnArchName`, followed by D3D12/HIP
adapter LUID matching; it does not depend on a marketing name or overridden
DXGI vendor ID.

| Target | Examples in LLVM's processor table | This project's validation |
| --- | --- | --- |
| gfx1201 | RX 9070, RX 9070 XT | RX 9070 XT: hardware, K/M and game tests. Other cards untested. |
| gfx1200 | RX 9060, RX 9060 XT | Host/device compilation and software tests only. No physical gfx1200 available. |

Sources: [LLVM AMDGPU processors](https://llvm.org/docs/AMDGPUUsage.html#processors)
and [AMD's Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html).
AMD's matrix is dependency information, not evidence that d4r's K/M or external
memory path works on an untested card. Other RDNA4 SKUs are eligible only when
HIP reports one of these exact targets and exposes the required Windows APIs.
Device-specific drivers, performance and memory capacity still need testing.
gfx1250/gfx1251 are a separate ISA group and are not enabled by this change.

Do not rename a gfx1201 package to gfx1200. `gpu-target.json`, the compiled
host target and each ELF code-object target must agree. Runtime selection
rejects another architecture before launching kernels. Staging, packaging and
installation inspect the documented AMDGPU ELF machine ID and reject mixed
objects. They never rewrite a binary target or set `HSA_OVERRIDE_GFX_VERSION`.
Legacy packages without target metadata remain gfx1201 only.

## Build and package

Prepare the pinned dependencies as described in `windows-rdna4-port.md`.
Select a target explicitly; this example creates the experimental gfx1200
package. Use gfx1201 for a 9070/9070 XT. Keep separate output directories when
building both targets.

```powershell
$arch = 'gfx1200'
$diag = "$PWD\dist\windows-$arch-diagnostics"
.\scripts\windows\build-windows-rdna4.ps1 -RuntimeProfile therock -GpuArch $arch -ZludaRoot "$PWD\dist\zluda-windows-final" -BuildDirectory "$PWD\build\windows-$arch" -InstallDirectory $diag
.\scripts\windows\stage-native-k.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -PackageRoot $diag
.\scripts\windows\stage-native-m.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -PackageRoot $diag
.\scripts\windows\package-windows-game.ps1 -DiagnosticRoot $diag -GpuArch $arch -PackageRoot "$PWD\dist\windows-$arch-game" -ArchivePath "$PWD\dist\windows-$arch-game.zip"
```

ZLUDA, HIP and OptiScaler must already be built/prepared. The package contains
the strict K FP16 and M FP16-equivalent baseline; native FP8 stays disabled.
The architecture does not change arithmetic or the existing WMMA layout.
Private NVIDIA-derived texture overrides remain optional local experiments;
their builder accepts `-GpuArch`, and their validation must match the target
package before installation. Existing gfx1201 performance measurements are
not measurements of gfx1200.
The separate experimental FP8/F16 arithmetic and tuning recipes remain
gfx1201-specific; this change enables the conservative production baseline
on both build targets.

Equivalent CMake selection is `-DD4R_GPU_ARCH=gfx1200`. Unknown targets fail
configuration. Some diagnostic executable names and source filenames retain
`gfx1201` for compatibility; the logged architecture and compiled code objects
use the selected target.

## First hardware gate on another card

From the repository root, with the matching diagnostic package above and the
pinned dependencies, run this single command:

```powershell
.\scripts\windows\test-windows-rdna4.ps1 -RuntimeProfile therock -PackageRoot "$PWD\dist\windows-gfx1200-diagnostics" -ZludaRoot "$PWD\dist\zluda-windows-final"
```

Success requires exit 0, all selected tests passed in `summary.json`, the
correct actual GPU/target in stdout, exact HIP/PTX/WMMA/pixel-format checks and
stable interop lifetime. On failure, retain the single printed diagnostic ZIP;
it includes stdout/stderr, exit code, runtime/driver inventory and DLL identity.
No NVIDIA DLL is required for this first gate. Full K/M reference, temporal,
standalone DLSS and game validation are subsequent gates on that hardware;
passing the first gate alone does not close them.

Without that GPU, run `ctest --test-dir build/windows-gfx1200 -L software
--output-on-failure`. These CPU/WARP checks do not execute gfx1200 kernels.

## Fork review

Reviewed [realdody's windows-rdna3 branch](https://github.com/realdody/d4r/tree/windows-rdna3)
at `ae530b0f53ed5c0bb87558dd7c20f619e014e4fd`, on 2026-10-02. Adapted:

- Per-target build configuration and actual-target diagnostics, narrowed here
  to the two documented RDNA4 subtargets, with package/ELF guards added.
- MSVC binary16 conversion in the synthetic CUDA MMA reference. An exhaustive
  CPU test covers 63,490 finite/infinity encodings and 31,743 rounding
  boundaries; LLVM also checks it against native `_Float16` conversion.
- `WINBOOL` to `BOOL` normalization in generated D3D12 hooks, so the generated
  header works with the Microsoft SDK as well as MinGW headers.

The fork's RDNA3 implementation forces `D4R_WMMA_LAYOUT=12` for gfx11. That
kernel-layout change, the raw gfx11 probe and the expanded gfx11 runtime
allowlist are not imported. Its reported RX 7800 XT/K result has not been
independently reproduced here; its M/translated-path limitations still require
separate checks. This change does not advertise native Windows RDNA3 readiness.
The existing upstream Linux/RDNA3 implementation is unchanged. Credit for
the adapted portability fixes: realdody <dodobozicek@gmail.com>.

## Local results, 2026-10-02

- Both targets: complete MinGW host build and 19 device code objects using
  TheRock `10.2.0a20260929`, HIP `7.17.26386`, Clang 24. The runtime, driver,
  patched ZLUDA and OptiScaler versions are unchanged from the validated port.
- gfx1201/RX 9070 XT: 20/20 CTest gates, including 16 physical-GPU gates.
- gfx1200: 4/4 software gates; no gfx1200 kernel execution.
- MSVC v143 `14.44.35207`: binary16 CPU test passes; CUDA MMA probe and D3D12
  hook translation units compile. This is not a complete MSVC build claim.
- K: 12 finite 4K frames at 2259x1271 input, burst and feature recreation;
  all six async RGB images exactly match their synchronous controls.
- M: eight finite 4K control/candidate frames at the same input dimensions;
  all four new-build RGB images exactly match the prior validated build.
  Both presets log every required native layer, zero CPU image copies and
  current-frame output.
- Original gfx1201 objects remain byte-identical on disk. In fresh builds,
  `.text` and `.rodata*` match the prior gfx1201 objects for all 19 kernels;
  fresh ELF files have different overall hashes. gfx1200 and gfx1201 have
  distinct correct ELF targets. Matching instruction sections are compile
  evidence, not a substitute for testing gfx1200 hardware.
- A gfx1200 HIP probe on the physical gfx1201 fails cleanly with exit 4 before
  any GPU allocation/kernel execution. This tests refusal, not gfx1200 support.
- Both target packages pass install/restore checks in isolated workspace
  fixtures; no game is launched. Only manifest-listed runtime files install.
  Packaging rejects a mismatched destination or native kernel directory before
  writing files. Every packaged code object has the expected ELF target.

Logs are local in `test-results/rdna4-targets-*` and
`test-results/rdna4-target-objects.json`; they contain no public NVIDIA DLLs.
