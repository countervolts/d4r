# Native Windows RDNA4 game development package

Validated hardware: Windows 11 x64 / Radeon RX 9070 XT / gfx1201. Native K and
M transformer kernels pass standalone D3D12 + OptiScaler checks. Actual game
compatibility and performance are still being validated. This is a local
development package; no NVIDIA proprietary DLL is included.

Build the shim/diagnostics, corrected ZLUDA, K/M native objects and patched
OptiScaler using the scripts in `scripts/windows`, then run:

```powershell
.\scripts\windows\package-windows-game.ps1
```

The package pins TheRock 10.2.0a20260929 because its Windows external-memory
mapping lifetime passes the repeated import/release test. Stable HIP 7.2 has
a reproduced mapped-view leak; see `docs/windows-rdna4-port.md` for the source
fix, exact dependency versions, validation and complete build commands.

Run a D3D12 game without anti-cheat using your locally supplied NVIDIA DLLs:

```powershell
.\dist\windows-rdna4-game\windows-game.ps1 -GameExe "D:\Games\SILENT HILL 2\SHProto\Binaries\Win64\SHProto-Win64-Shipping.exe" -NgxCore "C:\Users\Administrator\d4r\_nvngx.dll" -DlssDll "C:\Users\Administrator\d4r\nvngx_dlss.dll" -Preset 11
```

Use `-Preset 13` for M. The validated manifest requires DLSS 310.9.1 with the
recorded SHA256; other feature DLLs require regenerating and validating the
native manifest. Select an upscaler supported by OptiScaler in the game.
The script installs `dxgi.dll`, an explicit DLSS configuration and a private
`d4r` runtime directory, and captures stdout/stderr, an external debugger log,
crash dump, loaded DLL paths, driver information and OptiScaler logs. A local
ZIP bundle is printed at exit. `-RunSeconds 120` bounds a diagnostic run;
the normal default allows you to play until you close the game.

For the Silent Hill 2 startup black-screen investigation, add
`-DiagnosticResolution 1280x720`. This backs up its two graphics configuration
files, temporarily selects a window at that resolution, and restores the
original bytes after the test. It does not change progression saves. First-use
PTX compilation can be slow; `-CacheDirectory <existing-ZLUDA-cache>` reuses
modules with matching GPU, codegen switches and ZLUDA binary fingerprint.

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
`source-patches/zluda`. Dependency license texts and build hashes are included.
