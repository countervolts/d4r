# Hill-climb blockers (performance, native Windows, RX 9070 XT)

Things the climb cannot finish on its own. Each entry says what is blocked, what was measured, and what a human or
a tester should do.

## Needs a human

- **The release's model packages have no HIP layers on RDNA4** (2026-11-04). `packaging/build_engine_models.py`
  passes `--hip` only for K and only with a gfx110x/gfx103x kernel folder (`vkhip_kernels`); M and L never get
  `--hip`. On this RX 9070 XT the installed packages therefore send Vulkan games' K to the slow Vulkan network
  (3.7 ms at 720p -> 1440p), M to CUDA staging (29.6 ms, "missing hipnet.bin") and L / RR F to a crash. Every
  Vulkan number in the climb uses hand-built packages (`compile_m.py --hip kernels/m-vkhip-gfx1201`,
  `compile_k.py --hip kernels/k-gfx1201`, `kernels/rr/build_vkhip_f.sh` for RR F). The release needs gfx1201 /
  gfx1200 kernel folders (`D4R_NATIVE_FP8=1 D4R_VKHIP_LAYERS=1 kernels/build.sh m`, `kernels/build.sh k`) and the
  builder needs to pass them for K, M and L, plus the RR F rr2 code objects.
- **The release kit has no `spirv-opt`.** `compile_rr2.py` (RR F's `enc0_llpc.spv` / `dec0_1_llpc.spv`) and now
  `compile_m.py` (preset L's `*_llpc.spv` matrix stages) need SPIRV-Tools' `spirv-opt` (`D4R_SPIRV_OPT` or PATH);
  without it AMD's Windows driver crashes on RR F's Vulkan route and runs L's slow soft-dot2 stages
  (5.15 ms against 2.28 ms at 720p -> 1440p). The installed rr-f (2026-10-27 kit) has no `_llpc` files.
  The kit's `tools/models/bin` holds dxc and glslangValidator; `spirv-opt.exe` (Vulkan SDK, Apache-2.0) belongs
  next to them.
- **Game checks cannot use `windows-game.ps1` here** (2026-11-05). Cyberpunk 2077 (`bin\x64\d4r`) and Ghost of
  Tsushima (`d4r`) carry release installs of d4r (install.ps1, 2026-10-26/27) with no `.d4r-backup\manifest.json`;
  `windows-game.ps1 -Action install` refuses them ("An existing d4r directory has no installation manifest;
  preserved") and also expects a ZLUDA-style `package.json` package, which the climb's builds are not. The release
  shim has no redirect to another nvngx.dll. So no in-game check has run yet, including the Cyberpunk check lane B
  needs (d3dhip K on RDNA4, committed with the harness evidence). A human should pick one: allow a hash-verified
  manual swap of `d4r\nvngx.dll`, `d4r\d4r_hip_net.dll` and `d4r\engine\k*` (restored afterwards), or reinstall
  the games through a `windows-game.ps1`-managed install. Whichever it is, the check is: Cyberpunk 1440p preset K,
  Ghost Quality (1712x960 -> 2560x1440) preset K, Ghost 1080p -> 4K preset M, OptiScaler upscaler GPU time.
- **The K layers' cross-wave race is fixed only in Windows builds** (2026-11-05). `pos_core` (kernels/k/pwin_pos.h,
  enc0 / dec0) lacked a barrier between attention and residual; `-DPOS_HB_SYNC` adds it, but the climb may not change
  the shared flag lines, so only `kernels/windows-flags.txt` (gfx12 on Windows hosts) sets it. Linux builds, gfx11 /
  gfx10 builds and packages built before today (including every hand-built gfx1201 package that d3dhip K now uses by
  default) still race between waves: rarely through HIP (about 1 in 70 harness runs here), more where waves are skewed.
  A human should decide to add `-DPOS_HB_SYNC` to the sources' `d4r-build-flags*` lines of enc0 and dec0 (one
  `s_barrier` per window; on gfx1201 it measured no slower) and rebuild the packages.
- **The LLPC FP8-conversion changes need a Vulkan game look** (2026-11-05, 441a237 and the L commit after it). RR F's
  enc0 / dec0_1 and preset L's input / expansion stages on AMD's Windows driver now convert f32 sums to FP8 with a
  clamp instead of LLPC's guarded saturating conversion. Identical on every harness case, but a NaN accumulator now
  gives +-448 instead of the NaN code, and LLPC's leaked FP16_OVFL mode (which binary16 overflow follows) moved
  (journal). Portal RTX (Vulkan, RR) is the place to look: RR and preset L, highlights and disocclusions.
- **Indiana Jones and the Great Circle** (the verified Windows Vulkan game) is not installed; Portal RTX is the
  Vulkan game check until it is.

## Needs hardware

(none yet)

## Blocked measurements

(none yet)
