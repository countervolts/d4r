# Local gfx1101 drop-in package

## Current local development package: v4

`build/d4r-windows-gfx1101-local-v4.zip` is 90,720,715 bytes; SHA256
`74fbb4c78dba3e616b9bbba7dc1fe4cddf0a17855968b174f8ea69968d873dfc`.
It refreshes the verified v3 package with the current native-RR shim, player INI,
README, checker and installer sources. Every manifest file hash verifies against
the final ZIP. Exact local modified-build provenance and base ZIP hash are
recorded; no clean source commit is asserted. Existing ZIPs remain unchanged.

The package contains no NVIDIA binaries, private optimized kernels, or RR
models. RR defaults off. A locally generated D/E model can be selected through
`[RayReconstruction] Backend = native` and `ModelDir` in d4r.ini. Auto selects
`d4r/engine/rr-d`; relative paths resolve beside d4r.ini. The installer does not
generate that model yet, and FSR RR remains unavailable. This is a gfx1101-only
development package with synthetic RR coverage, not verified game RR support.

The refreshed target shim passed six exact RR hardware-depth frames through the
packaged architecture loader using player INI selection without launcher path
overrides. The staged test supplied the pinned SR/core dependencies separately
and used a separate local RR model. It covered capability/negative contracts,
optional guides, enhanced/inherited state, inverted D32S8, unchanged stencil and
Release-before-submit. Evidence: `build/rr-ngx-local-v4-loader-v1/summary.json`.
ZIP verification: `build/d4r-windows-gfx1101-local-v4.validation.json`.
The installer itself was not rerun for v4 under the current script policy; no
policy bypass or system change was used. No game was touched or release published.

The refresh helper is `tools/windows/refresh_local_package.py`. It accepts only
the pinned verified v3 base and fresh destinations under ignored build/, verifies
every base/output manifest hash, and marks output as local modified provenance.

## Preserved v3 package evidence

Local development ZIP: `build/d4r-windows-gfx1101-local-v3.zip` (92,158,725 bytes).
This package targets gfx1101, including this machine's RX 7700 XT. It is not a
multi-GPU public release, and it does not implement Windows game RR or FSR RR.

Extract its contents beside the game's main executable, then run `install.bat`
and accept the dependency download prompt. Start the game normally afterward.
The ZIP contains the shim, architecture loader, OptiScaler, native K/M objects,
HIP/ZLUDA runtime, configuration, installer and licenses. NVIDIA binaries are
excluded; the installer fetches pinned official SR/RR/core DLLs locally. It does
not install the NVIDIA driver. See the ZIP's `WINDOWS_README.txt` for placement
and existing-install handling.

Validation completed without touching a game:

- Every ZIP entry listed in the manifest matches its recorded SHA256.
- No NVIDIA SR, RR or core binary is bundled; `d4r/_nvngx.dll` is d4r's loader.
- Extracted the real ZIP into an ignored mock game folder and ran its installer
  with the verified local download cache. All three installed DLL hashes match
  the current pins. The installed checker passed on RX 7700 XT.
- The packaged architecture loader and target shim passed 32 synthetic K Quality
  frames at 1280x720 to 1920x1080. Outputs were finite, nonuniform and identical
  between control/batched-input runs. This is synthetic execution, not a game or
  image-quality claim.
- V3 adds `NativeDir` configuration for locally generated complete native
  directories. The exact new target shim passed eight additional synthetic
  player-mode frames through the architecture loader at 256x144 to 512x288.
  Its log confirms selection of the generated private K output override.
  Checker fixtures pass 20/20; release package fixtures pass 6/6.

The input package is explicitly marked as a local modified build. Its clean
binary source commit is not asserted; exact file hashes are authoritative. The
dependency commits are inherited from the earlier package's provenance. No
source commit or public upload was created.

Evidence: `build/windows-drop-in-v3-validation.json` and
`build/player-private-native-output/validation.json`. Earlier package validation
remains in `build/windows-drop-in-validation.json`.

ZIP SHA256:
`e93be288f4013616cc2229be0132b40395d5031d8733bfa63886ba75ce1389ca`.

## Optional private K output optimization

Generate a complete private directory in this source workspace:

```powershell
./scripts/windows/prepare-k-native-cache.ps1 `
  -DlssDll build/got-k-test/isolated-sr/sr/nvngx_dlss.dll `
  -NativeDirectory build/got-k-test/work/package-gfx1101-current/d4r/native `
  -OutputDirectory build/my-private-k-cache
```

This uses the locally built emitter and TheRock compiler. It verifies the input
target and pinned DLL, builds the selected output variant, combines the baseline
K/M objects, regenerates the native manifest and records hashes. It never edits
a game or configuration. Select the complete `native` subfolder with `[Paths]
`NativeDir` in the new shim's `d4r.ini` after validation. Relative paths resolve
beside `d4r.ini`; `auto` uses packaged kernels. Missing or wrong-target private
objects are rejected. The default variant currently optimized is LDR, regular
depth and high-resolution motion vectors; other variants retain the normal path.

The workflow output under `build/k-native-cache-workflow/native` is byte-identical
to the previously validated 17-object candidate. The new player-mode smoke loaded
that directory. These NVIDIA-derived objects stay local and are absent from V3.
