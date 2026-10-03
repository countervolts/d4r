# Experimental Windows RDNA3 / RDNA4 coverage

The shim, diagnostics, native K/M kernels and game installer accept the
targets below, with one package per actual ISA. **Only RX 9070 XT has physical
validation here.** Community RX 9060 XT / gfx1200 logs now show K/M backend
execution in Cyberpunk and Dawnwalker with all required native layers and
finite output. Visual issues remain reported and are not resolved by those
checks. All other targets are experimental and compile-tested; their DLSS,
interop and performance remain unverified.

| Family | Target | Examples / scope |
| --- | --- | --- |
| RDNA3 | gfx1100 | RX 7900 family; same-target workstation GPUs |
| RDNA3 | gfx1101 | RX 7700 / 7700 XT / 7800 XT; same-target mobile/workstation GPUs |
| RDNA3 | gfx1102 | RX 7600 / 7600 XT; same-target mobile/workstation GPUs |
| RDNA3 | gfx1103 | RDNA3 integrated graphics |
| RDNA3.5 | gfx1150, gfx1151, gfx1152, gfx1153, gfx1154 | Corresponding integrated graphics targets |
| RDNA4 | gfx1200 | RX 9060 / 9060 XT; same-target GPUs |
| RDNA4 | gfx1201 | RX 9070 / 9070 XT / 9070 GRE; same-target GPUs |

Names and IDs follow [LLVM processors](https://llvm.org/docs/AMDGPUUsage.html#processors)
and [ELF ABI](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-elf-header-e-flags).
`tools/windows/gpu-targets.json` supplies CMake-generated C++, PowerShell and
the private texture builder. Selection uses HIP's real `gcnArchName` and the
D3D12 adapter LUID, including systems with several GPUs sharing a target.
Marketing names do not control eligibility.

All listed RDNA3/RDNA3.5/RDNA4 targets with existing gfx11/gfx12 WMMA contracts
are included. TBD RDNA4m gfx1170..gfx1172 and gfx12.5 gfx1250/gfx1251 are
different ISA groups and excluded. Unknown future targets need explicit review.
Check [AMD's Windows matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
for driver/runtime eligibility. Compiler support cannot supply missing Windows
driver support or external-memory APIs, especially on APUs.

## Installation and diagnostics

Use the actual target's archive: **gfx1200 for RX 9060 XT**, for example.
Metadata, host target and every ELF target must agree. Runtime/scripts reject
mixed targets and never rename targets, rewrite ELF IDs or set
`HSA_OVERRIDE_GFX_VERSION`. Legacy metadata-free packages remain gfx1201 only.

Before changing game files, `windows-game.ps1` now runs a read-only HIP
inventory and checks the target and wave32. Inventory allocates no GPU buffers
and runs no kernels; success is discovery, not validation. Failure prints one
ZIP containing stdout/stderr, exit code, runtime/DLL and driver details. A local
`d3d12.dll` identifying itself as vkd3d/Wine is rejected before install. Restore
the game's native D3D12 installation; the script never loads or removes that
proxy itself. Begin synchronously on an untested GPU:

```powershell
.\windows-game.ps1 -GameExe "D:\Games\Example\Game.exe" -NgxCore "C:\LocalDLLs\_nvngx.dll" -DlssDll "C:\LocalDLLs\nvngx_dlss.dll" -Preset 11 -ValidateOutput -CaptureExceptions
```

The local DLLs must match the documented NGX version and DLSS 310.9.1 SHA256
in `package.json`. NVIDIA DLLs are never distributed. Retain the printed ZIP
on failure. Async K and performance are subsequent gates; gfx1201-specific
FP8, arithmetic and tuning experiments remain separate.

## Issue #10 review, 2026-10-02

Reviewed [comment 5952822115](https://github.com/countervolts/d4r/issues/10#issuecomment-5952822115),
including all 32 attached text logs from eight attempts. HIP discovers
**RX 9060 XT / gfx1200** successfully. Every attempt fails with the old shim's
**"No selected gfx1201 device"** rejection; OptiScaler reports `support=false`
/ `BAD00002`. None reaches `D4R_RUNTIME` initialization or native transformer
execution. These logs do not demonstrate a K/M kernel crash. Use the gfx1200
package instead of overriding architecture. Three attempts also load
vkd3d-proton 3.1.0 from a local D3D12 proxy, outside this Windows backend.

The new packages/preflight address the identified failures. They do not prove
those games work on RX 9060 XT; a new native-D3D12 run on that GPU is required.
Raw community logs and personal directory paths stay local.

## Community retest, 2026-10-03

[Comment 5962444244](https://github.com/countervolts/d4r/issues/10#issuecomment-5962444244)
uses the corrected `52d9872` package / `3d62b98` binaries on RX 9060 XT
(`gfx1200`, 16 GB). Cyberpunk K/M complete 4,365 / 1,751 session frames;
Dawnwalker K/M complete 6,713 / 1,690. Every frame has a finite-output GPU
check, all required native layers execute and all four sessions exit normally.
The logs record zero backend failures, previous-frame outputs and CPU image
copies. This is community backend coverage, not a reference-image comparison:
the tester reports texture/LOD and motion-related artifacts. All four runs use
the HDR network input/output path, unlike the local Cyberpunk LDR validation.

Spider-Man 2 crashes with `0xc0000005` before any completed DLSS frame;
its nonstandard launch is not evidence of a transformer kernel failure.
Assassin's Creed exits with zero frames and Streamline logs repeated internal
minidumps; its clean process exit is still inconclusive. Their causes remain
open without a stack trace. The Witcher 3 log in
[comment 5962808629](https://github.com/countervolts/d4r/issues/10#issuecomment-5962808629)
selects the gfx1200 d4r shim instead of the original NGX core (its SHA256
matches the shipped shim exactly), and the installer correctly refuses
before modifying the game. It does not establish a Witcher runtime failure.

## Build

Prepare dependencies following `windows-rdna4-port.md`. Default: gfx1201.
Select another target explicitly and keep separate directories:

```powershell
$arch = 'gfx1101'  # for example RX 7800 XT
$diag = "$PWD\dist\windows-$arch-diagnostics"
.\scripts\windows\build-windows-rdna4.ps1 -RuntimeProfile therock -GpuArch $arch -ZludaRoot "$PWD\dist\zluda-windows-final" -BuildDirectory "$PWD\build\windows-$arch" -InstallDirectory $diag
.\scripts\windows\stage-native-k.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -PackageRoot $diag
.\scripts\windows\stage-native-m.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -PackageRoot $diag
.\scripts\windows\package-windows-game.ps1 -DiagnosticRoot $diag -GpuArch $arch -PackageRoot "$PWD\dist\windows-$arch-game" -ArchivePath "$PWD\dist\windows-$arch-game.zip"
```

Rebuild diagnostics for the new inventory executable. The package keeps strict
K FP16 and M FP16-equivalent baselines; native M FP8 remains disabled. CMake:
`-DD4R_GPU_ARCH=gfx1101`. Some source/executable names retain gfx1201/gfx12 for
compatibility; actual targets follow the selected architecture.

One command compiles all eleven targets without GPU workloads:

```powershell
.\scripts\windows\build-windows-gpu-matrix.ps1 -ZludaRoot "$PWD\dist\zluda-windows-final"
```

Add `-PackageGames -DlssDll "$PWD\nvngx_dlss.dll" -ArchiveDirectory "$PWD\dist\windows-gpu-archives"`
to stage/archive all packages after committing source and preparing OptiScaler.
Use `-GpuArchitectures gfx1100,gfx1101,gfx1102,gfx1200,gfx1201` for a subset.
Per-target logs and `matrix-build.json` distinguish compilation from execution.
The supplied DLL generates manifests locally and is never copied into archives.

Rebuild LLVM/ZLUDA when updating from an older package: the old LLVM 22 backend
does not recognize gfx1154 even though Clang 24 compiles native objects for it.
`build-zluda-llvm.ps1` now applies the source backport in `patches/zluda-llvm`
automatically; relink with `build-zluda-windows.ps1` afterward. Compiler base
and patch hashes are recorded in the ZLUDA build metadata and included in game
packages. The expected patched LLVM submodule appears modified in its parent
checkout; its complete source delta is exported, not a binary target override.

For a compilation gate requiring neither a GPU nor NVIDIA DLLs, run:

```powershell
.\scripts\windows\test-zluda-target-compilation.ps1 -ZludaRoot "$PWD\dist\zluda-windows-final"
```

All 33 public PTX compilations pass for the eleven targets: Driver API pattern,
FP16 MMA strict reference and explicit WMMA modes, with correct ELF targets
and wave32. The gfx12 strict F16 reference intentionally retains its existing
double-precision helper; the separate WMMA mode checks code generation only.
It does not enable relaxed arithmetic in game packages or validate numerical
results on other GPUs. The original gfx1154 compiler abort is preserved locally
in `build/llvm-gfx1154/before.log`; the regression results are in
`test-results/windows-gpu-coverage/zluda-offline`.

## Fork adaptation

Reviewed [realdody's branch](https://github.com/realdody/d4r/tree/windows-rdna3)
at `ae530b0f53ed5c0bb87558dd7c20f619e014e4fd` and the newer
[`1897b4a` correction](https://github.com/realdody/d4r/commit/1897b4afa79832bd879234cfb37e7c1468db5cde).
Earlier MSVC binary16 and generated `BOOL` fixes remain. The earlier forced
gfx12 fragment layout is not used.

The newer fork reports Windows gfx1101 `permlanex16` returning its own lane,
duplicating even K values into odd slots. This port adapts that finding into
`kernels/common/wave32_exchange.h`: a Windows/gfx11-only per-wave LDS exchange,
with native gfx11 layout, generic x/y/z indexing, 2 KiB scratch (512 threads
maximum), volatile accesses and wave fences/barriers protecting scratch reuse.
It covers K/M fragment conversions and other K half-wave exchanges. gfx12 and
Linux keep the original instruction path. The raw gfx11 probe uses its actual
intrinsic/layout, without a forced gfx12 shim.

The fork reports RX 7800 XT enc1/enc2 identity/nonzero/reference checks. Those
are **the author's results**, not validation of this adaptation or full K/M
here. LDS exchange, full network, translated CUDA, accuracy, interop and speed
still need physical gfx11 testing. Credit: realdody <dodobozicek@gmail.com>.

## Results and next hardware gate

All eleven targets compile the complete MinGW host build and 19 device objects
with TheRock `10.2.0a20260929`, HIP `7.17.26386`, Clang 24. Software target
contracts inspect actual ELF objects without other-target GPU launches.
RX 9070 XT passes all 20 CTest gates after these changes. K/M full-frame and
package checks are recorded in `windows-rdna4-port.md`. Local logs are in
`build/windows-gpu-coverage` and `test-results/windows-gpu-coverage`.

The first hardware gate on another GPU needs no NVIDIA DLL:

```powershell
.\scripts\windows\test-windows-rdna4.ps1 -RuntimeProfile therock -PackageRoot "$PWD\dist\windows-gfx1101-diagnostics" -ZludaRoot "$PWD\dist\zluda-windows-final"
```

Success requires exit 0, correct actual target, exact HIP/PTX/WMMA/pixel checks
and stable interop lifetime. Failure prints one ZIP. Full K/M reference,
temporal, DLSS and game validation remain subsequent gates.
