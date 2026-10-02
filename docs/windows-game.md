# Native Windows RDNA3 / RDNA4 game development package

Validated hardware: Windows 11 x64 / Radeon RX 9070 XT / gfx1201. Native K and
M transformer kernels pass standalone D3D12 + OptiScaler checks. Silent Hill 2
K now renders real menu frames and M renders a saved gameplay level with
GPU NaN/Inf checks. Both K and M pass 3840x2160 output on this machine.
Cyberpunk 2077 2.31 also reaches saved gameplay with K and M at 3840x2160:
all required native transformer layers execute and output checks pass.
The [port notes](windows-rdna4-port.md) record session counts and limitations.
Performance measurements and their controls are recorded in
[windows-performance.md](windows-performance.md). This is a
development package; no NVIDIA proprietary DLL is included.

Experimental builds also accept gfx1100..gfx1103 (RDNA3), gfx1150..gfx1154
(RDNA3.5) and gfx1200 (for example RX 9060/9060 XT). These targets are
**compile-tested only**; no physical validation here has been performed.
RX 9070 and other gfx1201 cards also remain untested. Use a package built for
the GPU's actual HIP architecture: its `gpu-target.json`, host shim and ELF
code objects must agree. Installation/runtime reject mixed targets. See
[windows-gpu-support.md](windows-gpu-support.md) for separate-target build
commands, the fork review and the first hardware diagnostic gate.

Before changing game files, the runner inventories HIP devices without
executing kernels. It rejects another architecture's package and a local
D3D12 proxy identifying itself as vkd3d/Wine, producing one diagnostic ZIP.
RX 9060 XT needs gfx1200: the old gfx1201-only package fails before DLSS
initialization on it. Use native D3D12. On an untested GPU, start with
synchronous K and output diagnostics. The ordinary launcher does not attach
a debugger. `-CaptureExceptions` is an explicit developer option; games with
anti-debug protection can refuse it.

Numeric CUDA driver NGX initialization searches for `nvngx_dlss.dll` beside
the calling d4r shim. The installer stages your local feature DLL in `d4r/`
and the original NVIDIA core in `d4r/vendor/`. `Libraries.NvngxDlssPath` is
the feature directory. Keeping the feature DLL only in `vendor/` causes
`BAD00004` (FeatureNotFound) in numeric-ID games such as Cyberpunk.

Build the shim/diagnostics, corrected ZLUDA, K/M native objects and patched
OptiScaler using the scripts in `scripts/windows`, then run:

```powershell
.\scripts\windows\package-windows-game.ps1 -ArchivePath "$PWD\dist\windows-rdna4-game.zip"
```

The package pins TheRock 10.2.0a20260929 because its Windows external-memory
mapping lifetime passes the repeated import/release test. Stable HIP 7.2 has
a reproduced mapped-view leak; see `docs/windows-rdna4-port.md` for the source
fix, exact dependency versions, validation and complete build commands.
The ZIP contains only manifest-listed files, including a snapshot of committed
d4r source. Private NVIDIA-derived objects, caches and diagnostic captures are
excluded. Original FP8 M and the optional hardware-packing variant pass strict
replay/network tests but do not establish a full-network speedup, so the game
package retains its FP16-equivalent M baseline.

Run a D3D12 game without anti-cheat using your locally supplied NVIDIA DLLs:

```powershell
.\dist\windows-rdna4-game\windows-game.ps1 -GameExe "D:\Games\SILENT HILL 2\SHProto\Binaries\Win64\SHProto-Win64-Shipping.exe" -NgxCore "C:\Users\Administrator\d4r\_nvngx.dll" -DlssDll "C:\Users\Administrator\d4r\nvngx_dlss.dll" -Preset 11
```

Use `-Preset 13` for M. The validated manifest requires DLSS 310.9.1 with the
recorded SHA256; other feature DLLs require regenerating and validating the
native manifest. Select an upscaler supported by OptiScaler in the game.
The script installs `dxgi.dll`, an explicit DLSS configuration and a private
`d4r` runtime directory, and captures stdout/stderr, exit code, loaded DLL paths,
driver information and OptiScaler logs. Ordinary runs use a process job without
an attached debugger. Add `-CaptureExceptions` to capture exception events and
minidumps when reproducing a crash. A local
ZIP bundle is printed at exit. `-RunSeconds 120` bounds a diagnostic run;
the normal default allows you to play until you close the game.

For the Silent Hill 2 startup black-screen investigation, add
`-DiagnosticResolution 1280x720`. This backs up its two graphics configuration
files, temporarily selects a window at that resolution, and restores the
original bytes after the test. It does not change progression saves. First-use
PTX compilation can be slow; `-CacheDirectory <existing-ZLUDA-cache>` reuses
modules with matching GPU, codegen switches and ZLUDA binary fingerprint.

Add `-ValidateOutput` to scan the entire RGBA16F DLSS output for NaN/Inf on
the GPU before returning it to D3D12. Only two diagnostic counters (8 bytes)
are read by the CPU; image data remains in VRAM. Validation failures are
reported explicitly. Disable this diagnostic when measuring production speed.

`-ProfileStages` records CPU completion time for seven interop/NGX stages,
command-list split/replay, submitted state publication and suffix completion.
The recording interval measures NGX calls on the recording thread, not Present.
`-ProfileKernels` enables synchronizing HIP-event kernel profiling and changes
frame scheduling; use it separately from a production FPS measurement.
`-ProfileCommandHooks` samples one in 64 D3D12 interceptions to distinguish
tracking/lock waits from driver calls. It is disabled in ordinary runs.
`-ProfileCudaApi` aggregates existing CUDA API host durations without adding
HIP events or synchronization. `-UncachedInteropLists` restores per-frame
copy-list allocation for comparison with the normal safely reused lists.
`-ProfileGpuBoundary` requires `-AsyncInterop` and measures D3D12 GPU intervals
around the input/output copies and external fence. It reads 32 timestamp bytes
only after existing completion, with no extra CPU wait. Its external interval
includes HIP and scheduling; it is not an isolated transformer kernel time.
The runner drains stdout/stderr concurrently as raw bytes; the earlier
line-oriented PowerShell collector imposed large per-frame delays. The optional
ZLUDA kernel profiler is for diagnostics only. Measurements and reproduction
commands are in [windows-performance.md](windows-performance.md).

`-AsyncInterop` enables the experimental K scheduling path. Submission queues
the current frame's input/output dependencies and returns while a CUDA worker
executes DLSS. Owned command memory is retained until GPU completion; feature
release and resource reconfiguration wait for pending consumers. It passes
the standalone queued-frame and Release/CreateFeature exact-output regression,
but remains opt-in while game stability and performance are checked. It
currently requires one D3D12 command queue. This switch is independent of
profiling and output scans; omit those diagnostics for a production FPS run.

Ordinary runs use OptiScaler info logging and suppress successful HIP/CUDA API
messages. Errors, GPU identity, imports, frame records and native/translated
kernel launches remain logged. `-VerboseRuntime` enables all API and frontend
trace messages when investigating a failure. Numerical relaxation flags are
cleared in the child process; FP16 rounding and subnormals stay enabled.

The optional K output-store optimization is built locally from your NVIDIA DLL
with `build-native-texture.ps1` and checked by `test-native-texture.ps1`;
commands and measurements are in [windows-performance.md](windows-performance.md).
Add `-LocalTextureKernels "C:\Users\Administrator\d4r\build\private-textures-gfx1201"`
to the game command to install it. The object/manifest hashes and DLSS identity
must match validation. These generated objects contain NVIDIA code, remain
private, and are excluded from the distributable package. Ordinary launches
inherit the game's resolution/window settings; the default argument is `-dx12`.

Original files are backed up before replacement. Restore with:

```powershell
.\dist\windows-rdna4-game\windows-game.ps1 -Action restore -GameExe "D:\Games\SILENT HILL 2\SHProto\Binaries\Win64\SHProto-Win64-Shipping.exe"
```

Restore refuses to overwrite files changed since installation. Backups and
diagnostic results are retained. The script does not change progression saves,
system DLLs, driver settings or security settings. All runtime work is native
Windows. Colour, depth, motion, exposure and output cross APIs through VRAM
and shared D3D12 fences; the backend waits for the current frame and never
uses a previous-frame fallback.

OptiScaler source modifications are GPL-3.0 and exported under
`source-patches/optiscaler`. ZLUDA source modifications are exported under
`source-patches/zluda`; the LLVM gfx1154 backport is under
`source-patches/zluda-llvm`. Dependency license texts and build hashes are included.
