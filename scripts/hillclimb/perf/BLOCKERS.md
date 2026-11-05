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
- **Indiana Jones and the Great Circle** (the verified Windows Vulkan game) is not installed; Portal RTX is the
  Vulkan game check until it is.

## Needs hardware

(none yet)

## Blocked measurements

(none yet)
