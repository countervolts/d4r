# Local gfx1101 drop-in package

## Current local development package: v14 (2026-10-17)

`build/d4r-windows-gfx1101-local-v14.zip`, 133,240,107 bytes, SHA256
`2f88dec3b4086371efefa8fbfa316c3f7b86fb7858ca1c1bdee63146056f7edf`.
Every manifest hash and the exact ZIP entry set verify. It adds strict DS lane exchange in all eleven K objects
and FSR-only RGBA32F output with temporal format switching. The original FSR denoiser payload is unchanged;
compiler/subgroup/rounding trials produced no reliable gain. v13 and older packages remain preserved.

Actual fresh extraction, architecture loader and player INI: K control/candidate/candidate/control passes 64 exact
frames per size. Warmed CPU interop medians are 3.900 -> 3.3915 ms at 1080p and 5.4015 -> 4.820 ms at 1440p,
about 13%/11% faster. These include CPU synchronization/copies/NGX and are separate from standalone HIP GPU
timestamps (1.812/2.885 ms). The HIP runner/private model is not in the ZIP.

FSR-only test through the fresh loader passes 24 moving-camera frames with alternating float guides and half/float
output, typeless D32S8 and preserved stencil. Every output component equals the same-frame composed half result,
widened independently where needed. FSR+SR dynamic regression passes another 24 frames, retained resources and
minimum-bound rejections. Native RR passes six exact camera/reset frames with contracts and enhanced/inherited
states. Evidence: `build/k-ds-local-v14-{1080,1440}-v1/`, `build/fsr-local-v14-format-switch-v1/`,
`build/fsr-local-v14-dynamic-sr-v1/`, `build/rr-native-local-v14-regression-v1/`.

All execution is synthetic on gfx1101. Full install.bat was not rerun, and no game, driver/security/GPU setting,
commit, branch, PR or publication was involved. Linux targets, FSR denoiser speed, broader encodings/alpha/reference
quality and the original remaining parity gaps are not declared complete. This ends the user's narrowed pass.

## Preserved v13 package evidence

`build/d4r-windows-gfx1101-local-v13.zip`, 133,239,594 bytes, SHA256
`b2ff1157e27b9353afcf287fa8df011c7efa119ac4e26ab7746166ff87b05b34`.
Every manifest hash verifies. This is v12 plus FSR/SR dynamic render-size support within the loaded SR library's
queried bounds. Below-minimum sizes reject before command recording. The three fixed K objects remain included;
the HIP standalone K engine/model is still outside the package. All older packages remain preserved.

Fresh-extracted architecture loader/INI passes 24 moving-camera dynamic FSR+SR frames, including both below-minimum
axis rejections, retained input resources and all SR submissions (`build/fsr-rr-local-v13-dynamic-bounds-v1/`).
Native RR regression passes six exact camera/reset frames with contracts and enhanced/inherited states
(`build/rr-native-local-v13-dynamic-regression-v1/`). These are synthetic tests with separately staged dependencies.

Full `install.bat` was not rerun: the NGX core is absent from the download cache and would require the 945 MiB
driver archive. No driver runs, game interaction, commit or publication occurred. See `windows-fsr-rr.md` and the
newest checklist entry for exact synthetic coverage and remaining limits. Real-game/reference-quality and Linux
performance parity remain unverified.

## Preserved v8 package evidence

`build/d4r-windows-gfx1101-local-v8.zip` is 130,975,742 bytes; SHA256
`962530d1a914268cc6f9d2ec908185f8e4d9e5baafca6800d77ebc3281a647fe`.
It adds GPU RGBA32F RR output conversion and temporal RGBA16F/RGBA32F output
switching to v7 coverage. Packaged loader/INI E passes 12 exact evaluations
across two live features, inverted D32S8 and unchanged stencil. Float32 results
match widened standalone half output; this is synthetic engine consistency,
not NVIDIA reference quality or game/timing parity.

Installer still installs pinned SR, RR DLL and NGX core through bundled Python,
with optional D/E model generation/deployment. No NVIDIA DLL, private K override
or generated model is bundled. Every ZIP manifest hash verifies. Evidence:
`build/d4r-windows-gfx1101-local-v8.validation.json` and
`build/rr-ngx-local-v8-output-switch-e-v1/summary.json`. Other output encodings,
external exposure, FSR execution, K timing parity and original engine gaps remain.

## Preserved v7 package evidence

`build/d4r-windows-gfx1101-local-v7.zip` is 130,973,094 bytes; SHA256
`b7c68e6f6148c648b6e8e6f47362e0d5f0f6c6796d651d1e762a80d8f07e8d54`.
It retains the v6 native Python installer/clean portable tools and adds native
RR dynamic render-size handling. NVIDIA SR/RR/core DLLs still download locally;
no NVIDIA binary, private K override or RR model is bundled. All final manifest
hashes verify. Older ZIPs are preserved.

Native RR supports per-frame render dimensions from 64 pixels through creation
size, with fixed output size and implicit history reset on dimension changes.
Model weights/pipelines and the shared packet/fence remain loaded. Packaged
loader/INI E tests pass 16 exact evaluations across two live features and changing
sizes, both with resized input textures and retained creation-sized textures.
Six static D depth/format-switch frames also pass. These are synthetic engine
consistency checks, not NVIDIA reference quality, real-game RR or timing parity.
See `windows-rr-interop.md` for coverage and limitations. Evidence:
`build/d4r-windows-gfx1101-local-v7.validation.json` and
`build/rr-ngx-local-v7-dynamic-two-e-v1/`,
`build/rr-ngx-local-v7-max-dynamic-two-e-v1/`, and
`build/rr-ngx-local-v7-static-regression-v2/`.

## Preserved v6 package evidence

`build/d4r-windows-gfx1101-local-v6.zip` is 130,970,436 bytes; SHA256
`a144185823af334483543d4989584d65cfa0418049e56fb8e58410b91ae8700b`.
It adds the pristine portable-tools v2 inputs and native Python dependency
installer to the verified runtime. Every ZIP manifest hash verifies. No NVIDIA
DLL, private K override or generated RR model is bundled. All earlier ZIPs,
including intermediate v5, remain preserved. V5 lacks the final README update;
use v6 for the documented installer flow.

Extract beside the game executable and run `install.bat`. The batch wrapper
uses bundled isolated Python, without PowerShell, system Python or Git Bash.
It installs pinned SR 310.9.1 at `d4r/dlss/nvngx_dlss.dll`, RR 310.9.1 at
`d4r/rr/nvngx_dlssd.dll`, and 617.14 core at `d4r/ngx/_nvngx.dll`. Driver archive
is only extracted, never executed. The Python installer retains the exact
existing PowerShell installer URLs and SHA256 pins. Hash identity excludes old
596.36 core fixtures.

Choose N to retain existing RR settings. Optional D/E selection builds and
enables a private exact gfx1101 model. Tool inputs copy by manifest into the
external cache; generated products remain there and in a versioned private
runtime model directory. Existing models remain intact. Configuration selection
preserves comments/BOM/newlines and creates an exact external INI backup.
Model failure leaves verified DLLs installed and does not select an incomplete
model. Automatic RR building currently requires a gfx1101-only package.

DLL replacement first stages and verifies all changes and external original
backups. Replacement failure rolls back applied entries; concurrent edits are
preserved. Reruns skip matching DLLs and reuse only verified model products.
`install.bat --yes --offline --cache-root PATH --backup-root PATH` supports
preverified DLL caches. Add `--rr-preset D` or `E` and `--enable-rr` for model
generation/selection without the interactive prompts. Cache and backups must
stay outside the game directory.

Validation uses ignored mock game folders only:

- Eight CPU tests cover preservation, rollback including newly created files,
  concurrent edits, cache rejection, pin parity and manifest-only tool copying.
- Actual v5 batch wrapper builds/deploys D in 48 steps, verifies 66 model
  products, exact DLL/INI backups, unrelated-file preservation and an unchanged
  repeat. E also completes its actual 48-step installer path.
- Actual v6 ZIP interactive default installs all three pinned DLLs while
  declining model generation leaves INI unchanged. E cache reuse/deployment,
  repeat safety and tampered-tool rejection with unchanged runtime hashes pass.
- Native downloader fetches verified official 7-Zip, reuses the hash-verified
  617.14 driver archive and extracts a core matching its exact pin. No driver runs.
- Installer-deployed E selected solely through INI passes six exact synthetic
  NGX hardware-depth frames through the packaged architecture loader, with
  optional guides, enhanced/inherited access and stencil preservation. This is
  engine consistency, not NVIDIA quality reference or real-game RR evidence.

Evidence: `build/native-installer-v1/validation.json`,
`build/native-installer-v2/validation.json`,
`build/native-installer-download-v1/validation.json`,
`build/rr-ngx-native-installer-e-v1/summary.json`, and
`build/d4r-windows-gfx1101-local-v6.validation.json`. The final v6 fixture also
passes the same six exact E frames:
`build/rr-ngx-local-v6-installer-e-v1/summary.json`. Modified PowerShell release
packager passed static parser validation; its fixture suite was not rerun under
the recorded script policy. No policy bypass, game access, commit, publication
or system setting change occurred. FSR execution, broad game/quality validation,
K timing parity and remaining original engine requirements remain open.

## Preserved v4 package evidence

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

## Local v9: experimental FSR RR payload (2026-10-17)

`build/d4r-windows-gfx1101-local-v9.zip`, 133,200,126 bytes, SHA256
`332719443a9b050e6a5b479816a81fb4d3e482a8956e2c20f2596904f7f6152d`. Built by `tools/windows/refresh_local_package.py`
from the verified v8 ZIP (`--fsr-payload build/fsr-payload-v1`); all manifest hashes verified; v2-v8 preserved.

Added over v8: the shim with the FSR RR backend and its Super Sampling stage (`d4r/_nvngx_gfx1101.dll`);
`d4r/fsr/d4r_vkd3d_core.dll` (vkd3d-proton built by `tools/windows/build_fsr_payload.py`, LGPL-2.1, source offer
and patches in `d4r/fsr/SOURCE.txt` and `d4r/fsr/patches/`) and `d4r/fsr/amdxc64.dll` (d4r's capability shim, not AMD
software); vkd3d-proton, dxil-spirv and dxbc-spirv licence texts in `d4r/licenses/`; `install.bat` with an [F]SR
choice and `install-dependencies.py --enable-fsr` / `--install-fsr` (AMD's denoiser, SHA256 `48f1e588...`, is
downloaded hash-verified, backed up and rolled back like the NVIDIA DLLs, and `Backend = fsr` is written to
`d4r.ini` with an external backup); `d4r.ini` key `[RayReconstruction] FsrDenoiser`; README text.

Still not bundled: NVIDIA DLLs, any RR/K model, AMD's denoiser. FSR RR is off by default.

Verification (synthetic only, no game): ZIP extracted fresh and `test_fsr_rr_ngx.py` run against the extracted files
(`build/fsr-rr-ngx-upscale-v9-v1`: PASS, SR used, RMSE 0.289 vs 0.581 noisy input); release-package test PASS
cases=7 including FSR payload accept/reject cases (`build/release-package-fsr-v1`); installer unit tests 12 OK;
player-config test PASS cases=2; installer FSR path run for real on the extracted copy (AMD DLL downloaded,
hash matched, installed, single `Backend` line changed, backup equals original, second run changes nothing).
A full `install.bat` run was not repeated: the NGX core is not in the local download cache and would pull the
945 MiB driver archive. The new K work (HIP network, authentic K model) is NOT in v9.

## Local v11: v9 plus FSR tuning, GPU timing and dynamic-size refusal (2026-10-17)

`build/d4r-windows-gfx1101-local-v11.zip`, 133,236,207 bytes, SHA256
`ba91dd6cc178865244c9903a461f213cdcacf38bcf6905e919ba5e8ee652b228`; built the same way as v9 (verified v8 base,
`--fsr-payload build/fsr-payload-v1`), all hashes verified. v9 and an intermediate v10 are preserved but superseded.
Differences from v9: `d4r.ini` keys `FsrStabilityBias` (default auto) and `FsrDetail`; the shim's GPU-timestamp log
(`D4R_FSR_RR_LOG`), motion/depth dump diagnostic, tuning keys, exposure handled by the SR stage, and a clean refusal
of render sizes below the creation size. Validated from a fresh extraction with `test_fsr_rr_ngx.py`: static
same-size 0.127 (noisy 0.485), moving 640x360 -> 1280x720 0.2135 (noisy 0.554), moving same-size 0.1069 (noisy 0.486);
all use SR. Synthetic only, no game. The HIP K network and authentic K model are NOT in this package.
