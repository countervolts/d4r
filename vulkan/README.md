# Vulkan kernels (experimental)

Tools for running DLSS texture kernels as Vulkan compute shaders instead of through CUDA/HIP. Nothing here is
wired into the shim yet; it is a kernel-level proof with a byte-exact check against captured frames.

| File | Role |
|---|---|
| `ptx2glsl.py` | Translates one PTX kernel from your DLSS DLL into GLSL. Every instruction becomes the same operation `kernels/native/ptx2hip.py` emits, gotos become nested `if`/`else` and `do`/`while` (each branch rejoins at its immediate post-dominator), and registers that only hold half floats are typed `float16_t` / `f16vec2`. |
| `kvk.cpp` | Headless runner: binds captured textures, surfaces, buffers and the parameter block, dispatches, writes the surfaces back and times the dispatch. |
| `kcase.py` | Builds a `kvk` manifest from a capture and compares `kvk`'s output with the captured surfaces. |

## Reproducing the K output kernel check

```sh
# 1. capture (bridge options that already exist): every launch's resources after it ran, plus the arguments
D4R_CUDA_VERBOSE=1 D4R_CUDA_LAUNCH_STATS=1 D4R_CUDA_LAUNCH_DUMP_DIR=$CAP/dump \
D4R_CUDA_REPLAY_DUMP_DIR=$CAP/replay D4R_CUDA_REPLAY_DUMP_FILTER=hiluma_engine_output D4R_CUDA_REPLAY_DUMP_LIMIT=8 \
    <run the NGX CUDA benchmark, preset K>
# 2. translate and compile
K=hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel
python3 ptx2glsl.py MODULE.ptx $K > k.comp && glslangValidator -V --target-env vulkan1.3 k.comp -o k.spv
g++ -O2 -std=c++17 -o kvk kvk.cpp -lvulkan
# 3. run frame N and compare (the three linear-filtered textures are named by parameter offset)
python3 kcase.py make $CAP $K N case 200:linear 224:linear 232:linear
./kvk k.spv case/manifest 150 64        # 150 dispatches per submit, 64-wide subgroups
python3 kcase.py cmp case
```

Sampler settings come from the `cuTexObjectCreate` lines of the verbose log (all clamp, normalized coordinates;
filter per object).

## Result (RX 9070 XT, RADV / Mesa 26.2.4, DLSS 310.7, 2026-10-08)

Byte-identical to the native HIP kernel's four output surfaces on six consecutive frames at 640x360 -> 1280x720 and
one frame at 1920x1080 -> 3840x2160. At 4K: 1.18 ms per dispatch under sustained load (GPU clock 2.38 GHz),
1.05 ms with 0.3 ms gaps (2.7 GHz). The native kernel measures 1.16 ms in the live benchmark at about 3.1 GHz. The
two were not timed in the same run or clock state.

## Pitfalls found

- Shader-visible buffers must be allocated from a `DEVICE_LOCAL | HOST_VISIBLE` memory type. The first
  host-visible type on RADV is system memory; reading the network output from it made every variant take 2.9 ms.
- A program-counter loop around the whole shader (`while (pc != END) { if (pc == k) ... }`) keeps every value
  live across the back edge: 35 ms. Guards without the loop: 2.7 ms (VRAM buffers). Real nesting: 1.69 ms.
- Packing half floats into `uint` registers costs about 840 `v_mov_b16`; native half types: 1.35 -> 1.18 ms.
- A 64K-entry table for `ex2.approx.f16x2` is exact but slower than `v_exp_f32`; dropping `precise` gains under 1%
  and changes about 0.005% of the output values.
