# K plain-layer benchmark on Linux: Vulkan (RADV) and D3D12 WMMA (vkd3d-proton)

Date: 2026-10-22. Synthetic single-layer microbenchmark (K `dec5`-shaped plain
Swin block), **not** full-frame inference.

## Result

| Kernel (Linux, RX 9070 XT) | Median of 7 trial medians | Trial-median range | Stdev | vs CPU oracle |
|---|---:|---:|---:|---:|
| Existing Vulkan `engine/layer.comp`, native RADV | **42.570 µs** | 42.515–42.617 (0.24 %) | 0.040 µs | 92.04 dB |
| D3D12 `layer_k_wmma.hlsl`, Wine + vkd3d-proton → RADV | **91.520 µs** | 91.270–92.379 (1.21 %) | 0.399 µs | 93.59 dB |

Per dispatch, GPU timestamps, 100 dispatches per batch with a barrier after each.
On Linux, D3D12 WMMA takes **2.15×** as long as the existing Vulkan shader. That
reverses the native Windows ranking:

| | Windows reference (AMD 26.9.1) | Linux (this run) |
|---|---:|---:|
| D3D12 WMMA | 119.140 µs | 91.520 µs (vkd3d-proton) |
| Existing Vulkan matrix shader | 323.107 µs | 42.570 µs (RADV) |
| WMMA / Vulkan | 0.37 | 2.15 |
| WMMA vs Vulkan output agreement | 88.43 dB | 92.18 dB |

Five interleaved rounds (Vulkan, then D3D12, alternating) confirmed the
result: Vulkan 43.172–43.375 µs and D3D12 89.156–91.988 µs.

### Interpretation limits

- The GPU model and PCI ID match the Windows reference (RX 9070 XT, 1002:7550).
  This checkout is on an NTFS partition, so it is probably the same card, but
  that has not been verified. Even so, the Windows/Linux differences combine the
  OS, the user-mode driver and compiler (AMD Windows vs Mesa RADV/ACO), the
  DXIL→SPIR-V translation (vkd3d-proton/dxil-spirv), the DXC version, power and
  clock management, and runner details. **Do not attribute the difference
  solely to RADV.**
- The grid is 4×4 = 16 workgroups, a small fraction of the 64-CU GPU, and
  each dispatch is followed by a barrier. These numbers therefore measure
  per-workgroup latency, not throughput. Without barriers (evk wall-clock mode,
  overlapping dispatches), Vulkan takes 13.8 µs per dispatch. Full-frame
  behavior at real resolutions may rank differently.
- Batch structure differs by runner. Each evk trial runs 12 batches, discards 3
  as warmup and reports the median of 9. Each d4r-k-layer trial runs 7 batches,
  discards 2 and reports the median of 5. Both exclude upload and readback.
- Clocks were not pinned (`power_dpm_force_performance_level=auto`; no root).

## Correctness

The fixtures are byte-identical between the two paths (SHA-256 prefixes). The
WMMA backend uses the same row-major weights, so `weights.bin` is
`weights-row.bin`:

| File | SHA-256 (prefix) |
|---|---|
| params.bin | c6c8496b139916be |
| input.bin | a60713e72327acdc |
| weights-row.bin / weights.bin | 7d2dfee7c2f84ecc |
| cpu-reference.bin | d888bfaee2d010d3 |

| Comparison | PSNR | Max abs error | Bit-exact |
|---|---:|---:|---:|
| Vulkan vs NumPy oracle | 92.04 dB | 0.001953125 | 90.39 % |
| D3D12 WMMA vs NumPy oracle | 93.59 dB | 0.0009765625 | 92.81 % |
| D3D12 WMMA vs Vulkan | 92.18 dB | 0.0009765625 | 90.58 % |

All 7 Vulkan trial outputs are bit-identical, as are all 7 D3D12 outputs. The
threshold is 60 dB.

**WMMA was really used; nothing fell back to dot2.** The D3D12 run passes
`--wmma` with `layer_k_wmma.hlsl`. In the RADV ISA dump (`RADV_DEBUG=shaders`),
both final shaders contain 136 `v_wmma_f32_16x16x16_f16`:

| Shader ISA (ACO, GFX1201) | v_wmma | scratch load+store (static) | ds ops | instructions |
|---|---:|---:|---:|---:|
| Vulkan layer.comp | 136 | 118 (VGPR spills; 255 VGPRs) | 595 | 3816 |
| D3D12 WMMA via vkd3d-proton | 136 | 96 | 621 | 3551 |

Both shaders spill registers on RADV, which is worth investigating for either path.

## Environment

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT (RADV GFX1201), 1002:7550, VBIOS 115-G295BP00-100, PCIe 32 GT/s x16 |
| CPU | AMD Ryzen 7 9800X3D |
| Kernel | Linux 7.2.10-1-cachyos |
| Mesa / RADV | 26.2.4 (`Mesa 26.2.4-arch3.1`, `DRIVER_ID_MESA_RADV`, API 1.4.354); prints “radv is not a conformant Vulkan implementation” |
| Vulkan loader / headers | 1.4.363.0 |
| linux-firmware | 20260916 |
| glslang | 16.6.0 (`glslangValidator`, package 1.4.363.0) |
| Native compiler | g++ 16.2.1 |
| Cross compiler | x86_64-w64-mingw32-g++ 16.2.0 |
| DXC | v1.9.2609 Linux release (`linux_dxc_2026_09_28.x86_x64.tar.gz`, sha256 96faadc7…aa39a1) |
| Wine | wine-11.20 (system), fresh prefix |
| vkd3d-proton | from Proton Experimental `experimental-11.0-20261014`; reports applicationVersion 3.1.0, build 0bb64c56c8f0dcc |
| DXVK dxgi | v3.1.1-97-ga78a467cadb22be (same Proton) |
| Python / NumPy | 3.14.8 / 2.5.4 |

Shader defines:
- Vulkan: `-V --target-env vulkan1.3 -DC=160 -DNH=8 -DKIND=0 -DPOSATTN=0 -DEMBED=0 -DHEAD=0 -DXC=16 -DMERGE_N=16`
- D3D12: `-T cs_6_6 -E main -enable-16bit-types -O3 -DC=160 -DNH=8 layer_k_wmma.hlsl`

Fixture shape: C=160, NH=8, 32×32 tokens, shift (0,0), seed 42, dispatch grid
4×4.

Environment for timing runs:
- `VK_LOADER_LAYERS_DISABLE=~all~`. Several implicit layers are installed:
  MangoHud, vkBasalt, the Steam overlay and Fossilize, LSFG, MAKO, AFMF and the
  Mesa device-select/anti-lag layers. `VK_LOADER_DEBUG=layer,driver` confirmed
  that no layer was inserted and that the loader used
  `/usr/lib/libvulkan_radeon.so`.
- `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json`. This is the only
  ICD; vulkaninfo lists a single device.
- No validation or debug layers, no `RADV_DEBUG` and no `VKD3D_DEBUG` were set
  during timing. Those were used only in separate inspection runs.
- For D3D12 only: `WINEDEBUG=-all`,
  `WINEDLLOVERRIDES=d3d12,d3d12core,dxgi=n`,
  `VKD3D_SHADER_CACHE_PATH=<scratch>`, `D4R_EVK_GPU_TIMING=1` (Vulkan).
- `DXVK_HDR=1` was set in the session; it is irrelevant because no swapchain is
  used.

## Commands

`bench_vulkan.py` (alongside this report) generates the fixtures exactly as
`test_layer.py --synthetic` does: the same `synthetic()` call and RNG order,
with weights first and then input. It runs evk without any D3D12, DXC or Wine
dependency. Its checks are the same as the wrapper's: the NumPy oracle,
60 dB, finiteness and shape. It also requires every trial to be bit-identical.

```sh
# Vulkan (native)
g++ -O2 -std=c++17 engine/evk.cpp -lvulkan -o $B/d4r-vulkan-layer
export VK_LOADER_LAYERS_DISABLE='~all~' VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json
python3 engine/d3d12/bench_vulkan.py $W/vk-bench --vulkan-runner $B/d4r-vulkan-layer \
    --synthetic 160 --heads 8 --width 32 --height 32 --repeats 100 --trials 7
#   evk invocation per trial (cwd = work dir, D4R_EVK_GPU_TIMING=1):
#   d4r-vulkan-layer layer.spv 4 4 100 in:0=params.bin in:1=weights-row.bin in:2=input.bin \
#       in:3=params.bin out:4=vulkan-N.bin:327680 out:5=unused.bin:16

# D3D12 runner (Windows exe via mingw; CMakeLists only supports WIN32 targets)
x86_64-w64-mingw32-g++ -O2 -std=c++17 -Wall -Wextra -DNOMINMAX -DWIN32_LEAN_AND_MEAN \
    engine/d3d12/replay_layer.cpp -o $B/d4r-k-layer.exe -ld3d12 -ldxgi -ld3dcompiler \
    -static -static-libgcc -static-libstdc++
# Wine prefix with vkd3d-proton + DXVK dxgi from Proton Experimental
P="$HOME/.local/share/Steam/steamapps/common/Proton - Experimental/files/lib/wine"
export WINEPREFIX=$W/wineprefix WINEDEBUG=-all WINEDLLOVERRIDES='d3d12,d3d12core,dxgi=n'
wineboot -i
cp "$P"/vkd3d-proton/x86_64-windows/d3d12{,core}.dll "$P"/dxvk/x86_64-windows/dxgi.dll \
   $WINEPREFIX/drive_c/windows/system32/

# D3D12 WMMA, correctness + first timing (same seed, so fixtures are byte-identical)
python3 engine/d3d12/test_layer.py $W/d3d12-wmma --runner $B/d4r-k-layer.exe \
    --synthetic 160 --heads 8 --width 32 --height 32 --backend wmma --repeats 100 \
    --dxc $DXC/bin/dxc --launch-prefix wine
# Then 7 timing trials on those fixtures (Z: paths):
wine $B/d4r-k-layer.exe Z:\...\layer.dxil Z:\...\params.bin Z:\...\weights.bin \
    Z:\...\input.bin Z:\...\d3d12-N.bin 4 4 100 --wmma
```

`test_layer.py --launch-prefix wine --vulkan-runner …` would also launch evk
through Wine. For that reason, the cross-API comparison above uses the native
evk outputs together with an explicit `compare()` on the shared fixtures, not
the wrapper's built-in Vulkan step.

## Raw timings (µs per dispatch)

| Trial | Vulkan median | Vulkan min | D3D12 median | D3D12 min |
|---:|---:|---:|---:|---:|
| 0 | 42.570 | 42.472 | 92.046 | 91.207 |
| 1 | 42.515 | 42.414 | 91.406 | 91.287 |
| 2 | 42.515 | 42.484 | 91.270 | 88.747 |
| 3 | 42.534 | 42.433 | 91.477 | 91.300 |
| 4 | 42.586 | 42.561 | 91.520 | 91.482 |
| 5 | 42.590 | 42.537 | 91.854 | 91.796 |
| 6 | 42.617 | 42.564 | 92.379 | 92.349 |

Interleaved rounds (Vulkan / D3D12): 43.375 / 91.446, 43.239 / 91.988,
43.248 / 89.156, 43.172 / 91.622, 43.369 / 91.770.

Vulkan wall clock without barriers (3 runs): 13.810, 13.747, 13.920.
