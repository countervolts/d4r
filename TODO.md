# TODO

## 1. Windows RDNA2: native K kernels never load (about 12 fps)

The RX 6650 XT log shows K at about 77 ms per frame: every kernel is translated by ZLUDA, and none of the gfx1032
native layers are used.

Cause: in the native Windows branch of `tools/d4r_nvngx_shim_config.cpp` (around line 239), the shim never sets
`D4R_ZLUDA_CACHE_HOME`. The bridge then has no cache root, because there is no `XDG_CACHE_HOME` or `HOME` on
Windows, so it calls `prepare_native_kernels(NULL)`. That makes the manifest NULL, and the bridge unsets
`D4R_ZLUDA_NATIVE_DIR` without logging anything (`tools/wine_nvcuda_bridge.c`, lines 652 and 1103).

- [x] Set `D4R_ZLUDA_CACHE_HOME` on native Windows: use `[Paths] CacheDir` if it is set, otherwise
      `%LOCALAPPDATA%\d4r`. An explicit CacheDir also sets `ZLUDA_CACHE_DIR` (`<CacheDir>\zluda`); auto keeps
      ZLUDA's own cache so existing installs do not recompile.
- [x] Log a line on each silent return in `prepare_native_kernels` (no cache, no manifest, no set for this GPU).
- [x] Test the per-process served-kernel directory on Windows. It uses `kill`, `realpath` and `opendir`, and has
      probably never run there. Harness against the release's kernels on a gfx1101: create, serve, stale-pid
      cleanup and exit cleanup all work. Fixed: `make_directories` ignored `\` (nested CacheDir never created),
      and a read-only source made its copy undeletable (copies are now made writable; cleanup clears the flag).
- [x] Block or warn before L (and M) on RDNA2. In the same log, L took 8-12 s per frame and then crashed.
      CreateFeature on the CUDA route of a pre-RDNA3 GPU runs K instead (logged; `D4R_ALLOW_SLOW_PRESETS=1` keeps
      M/L). `d4r-check.ps1` flags `Model = M/L` on RDNA2 and no longer reports missing engine models there.
- [ ] VRAM interop fails on gfx1032 (`import 801`), so every frame is staged through host memory (about 10 ms).
      Check this once K is native.

## 2. Windows RDNA3: K has no HIP layers

Reported by a 7800 XT user and a 7900 XT user. Their log says `k has no HIP layers (hipnet.bin)`, so K runs the
slower DXIL network.

- [x] `vkhip_kernels()` in `packaging/build_engine_models.py` uses only `D4R_GPU_ARCH`, and returns `None` when the
      release holds more than one `gfx110x` folder. It should use the arch that `hip_device_only()` already detected,
      as the M/L branch does. (Regression test in `tests/test_model_builder.py`.)

## 3. Frame generation

- [ ] **#72 (RE Requiem, 9070 XT): trails on hair and hand edges that get worse with the multiplier.** Fix the
      frame dump first: it freezes the game right after `dump: triggered`, and only writes the params file. The params
      to check are placeholder camera vectors, identity `ClipToPrevClip`, MVs that are not dilated, and no HUD-less
      or UI inputs.
  - [x] Dump fix (untested in a game): the copies transitioned every input from `ALL_SHADER_RESOURCE`, which is wrong
        for a backbuffer in PRESENT/RENDER_TARGET or depth in DEPTH_*, and a GPU hang fits the symptom. The real state
        now comes from the game's own barriers (`cux::trackResourceStates`, only while a dump is armed); an input
        whose state is unknown is skipped and logged, never guessed. The params files log each input's state.
  - [ ] Get a dump from the reporter and check the params above.
- [x] **#71 (Phantom Blade Zero): only 2x works.** The game asks for `numFramesToGenerate: 1`, and the reported
      multi-frame capability flips between 1 and 5. Make the capability consistent, allow an override, and log what
      the game asked for next to what was applied.
      Likely cause: Model auto picked Lite vs full from the kernel cache's arch file, so an RDNA4 GPU without its
      cache yet got Lite (1 frame). Auto now goes by the GPU in the PnP tree (cache file as fallback), and so does
      KernelCache auto. The published `DLSSG.MultiFrameCountMax` stays at the highest value seen this session;
      `[FrameGeneration] MultiFrameCountMax = 1-5` forces it (OptiScaler clamps its own
      `DLSSGOverrideInterpolationCount` to this). Evaluate logs the caller's frame count next to what was published.
      Needs a retest by the reporter.
- [ ] **HUD/UI corruption** (#66 Cyberpunk, Subnautica 2). The HUD-input log from 77dfcf6 is in place; get a log
      from the next nightly.
- [ ] **Latency above 4x** (Subnautica 2, #66). Retest on the next nightly.
- [ ] **#66: `grid barrier timed out` cost.** Retest with the new FG optimizations.

## 4. Games

- [ ] **#74 (Mortal Shell II, 9060 XT): Ray Reconstruction is never created.** RR D/E/F load, the game creates only
      SR preset K, and nothing in d4r's log says why ("Upscaler failed to run!" toasts too). d4r's side of the log
      never shows a RR CreateFeature, so the game (or OptiScaler answering capabilities itself) chose SR.
  - [x] Logging: the RR capability answer is logged on the native D3D12 route too (it was silent there), with
        MinDriverVersion fields filled as the core does. The first SR CreateFeature with RR available logs once
        whether anyone asked d4r about RR and what it answered, so the next log shows which side dropped it.
  - [ ] Get a log (d4r and OptiScaler) from a nightly with that line; then fix the side that drops RR.
- [ ] **#73 (Spider-Man 2, 9060 XT): `Input: FFX` smears trails**, and the first launch crashes with an out-of-VRAM
      error. OptiPatcher with `Input: DLSS` works. Either support the FFX route, or default to OptiPatcher with DLSS
      inputs.
- [ ] **Crash on boot with spoofing on** (7800 XT, Windows 11). Needs logs.
- [ ] **Crimson Desert crash** (9060 XT). Needs logs; possibly related to #34.

## 5. Packaging and release

- [ ] Ship d4r as an OptiScaler ASI plugin or custom DLL with an external `d4r` folder, so it works with any v10
      build (#73 and many Discord requests).
- [ ] Stop shipping a custom `OptiScaler.ini`; let `setup_windows.bat` write it.
- [ ] Have the installer fetch or enable OptiPatcher.
- [x] Add a check for a stray `_nvngx.dll` in the game folder to `d4r-check.ps1` (it broke DLSS in Palworld).
      (`packaging/d4r-check.ps1`, which the nightly ships, and `scripts/windows/d4r-check.ps1`.)
- [ ] Release 0.1.6 and a fresh Windows nightly.

## 6. Performance

- [ ] RR preset F on RDNA4 Windows (about 11.8 ms on a 9060 XT). This is the most requested item.
- [ ] RDNA4 K/M/L (9060 XT: K about 2 ms, M about 4 ms, L about 6 ms at Balanced 1440p). A community write-up
      (`d4r-vs-wronggpu-analysis.md` on Discord) reports M about 1 ms faster when recorded fully in the game's command
      list. Needs an RDNA4 owner to profile.
- [x] Document that M is meant for Performance and L for Ultra Performance. (`packaging/d4r.ini` already had the
      table; added to `WINDOWS_README.txt` and the Windows `d4r.ini`, with the RDNA2 K-only note.)

## 7. GitHub replies

- [ ] #68: answer the question about whether the Windows ZLUDA patches should carry patch 0009's LLVM
      strict-FP-division hunks. Then close.
- [ ] #66: items 1 and 4 are fixed; split the HUD part into its own issue and close.
- [ ] #71: RR padded inputs and the absolute NvngxPath landed in dbbbd88; keep it open for FG.
- [ ] #73: the pause-menu error spam is fixed in 8f99494.

## Linux

- Nothing open. Star Citizen needs GE or CachyOS Proton's built-in OptiScaler (0.9 pre10) plus fake system32 DLLs
  to get past EAC; d4r hasn't been tried there.

## Done (verify and close)

- Pause-menu `Upscaler failed to run!` spam (8f99494).
- An SR feature no longer dies after an uncovered texture (77dfcf6).
- RR accepts padded UE5 inputs, and the DLSS-G NvngxPath is written as an absolute path (dbbbd88).
- RDNA2 installs no longer build the RR/M/L engine models (dbbbd88).
- Witcher 3 black screen on a 9060 XT (fixed in windows.11.10).
