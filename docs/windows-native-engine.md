# Native Windows inference engine

Windows builds use the Linux branch's standalone Vulkan runtime. It loads Vulkan through pinned Volk, not through the
D3D12 NGX shim. This keeps game integration separate while the inference and frame path is validated.

## Setup and build

Run `scripts/windows/setup-windows-tools.ps1 -RuntimeProfile therock`, then build the standalone runners:

```powershell
pwsh -NoProfile -File scripts/windows/build-native-engine.ps1 `
  -HipRoot .tools/therock-10.2.0a20260929/_rocm_sdk_core -GpuArch gfx1101
```

The script verifies pinned Vulkan-Headers and Volk revisions. It downloads Khronos glslang 16.6.0 from its official
release and checks its SHA256 before extraction. Tools remain under ignored `.tools/`; runners remain under ignored
`build/`. Rebuild a different architecture with its exact `gfx11*` or `gfx12*` target. The runner checks its required
Vulkan features at startup and reports missing extensions or matrix types.

## Models and local frames

K, M and L model weights are local, derived from the user's own DLSS DLL or capture. Never copy them into source control
or packages. Linux-compatible preparation tools are in `engine/`; K uses `extract_k.py`, `prepare_k.py` and
`compile_k.py`. M and L use `compile_m.py`. A complete model needs a real captured NGX upload. The current native
Windows runner does not provide a model-capture front end or D3D12 game adapter.

Prepare a frame case from tightly packed linear HDR color, motion, and inverse-depth images, then run K:

```powershell
python engine/prepare_frames.py frames.json build/case.txt
build/native-engine-windows-gfx1101/d4r-k.exe build/engine/k build/case.txt build/results 1
```

To compile a local capture into K model files, expose the provisioned NumPy and shader compiler:

```powershell
$env:PYTHONPATH = "$PWD/.tools/python/vendor"
$env:PATH = "$PWD/.tools/glslang-16.6.0/bin;$env:PATH"
python engine/extract_k.py CAPTURE/replay build/model-k/model
python engine/prepare_k.py CAPTURE build/model-k/case --lut-only
python engine/compile_k.py build/model-k/model build/model-k/case/lut.bin build/engine/k
```

The runner writes full-resolution output and intermediate history files. Set `D4R_ENGINE_PROFILE=1` to report per-stage
GPU timings. Timings are Vulkan replay measurements; they do not prove game-queue latency or DLSS numerical parity.

M uses `d4r-m.exe MODEL_DIR RENDER_W RENDER_H OUT_W OUT_H OUTPUT_PREFIX [REPEATS] -- COLOR MOTION DEPTH JITTER_X JITTER_Y ...`.
L uses the same runner and a preset-L model produced from an L capture. `d4r-rr.exe` uses the Linux RR model/input
formats documented in `engine/README.md`; it is not connected to Ghost of Tsushima.

## Current Windows evidence

K, M and RR runners compile as native Windows x64 executables. RX 7700 XT Vulkan feature checks pass for the K f16
engine. A seeded finite synthetic K model completed a 256x256 full-frame dispatch and readback on that GPU in about
4.14 ms. It has synthetic weights and is not evidence of DLSS model quality. NVIDIA NGX model extraction, game
integration, presets M/L runtime feature support, and preset timings remain unverified.

## NVIDIA runtime files

The release includes `install.bat` plus `install-dlss.ps1`. It downloads SR and RR 310.9.1 from the pinned official
NVIDIA/DLSS source, and the NGX core from NVIDIA driver 617.14. It verifies archive and DLL SHA256 values before
changing the local d4r folder. The 991 MiB driver archive is extracted with pinned 7-Zip `7zr`; the driver installer
never runs. Existing target DLLs are copied to `%LOCALAPPDATA%\d4r\backups` with a manifest of original paths and
hashes. Verified files are cached under `%LOCALAPPDATA%\d4r\cache\nvidia`. The RR DLL is preparatory; RR inference is
not connected to the Windows game shim.
