# Preset K on native Windows through d3dhip (RDNA3, 2026-11-02)

RX 7700 XT (gfx1101), AMD Software 26.10.1 (driver 32.0.32015.7029), Ghost of Tsushima's Quality case
(1712x960 -> 2560x1440). Replay: `d4r-k-d3d12 MODEL case.txt - 100 hip` (`build/engine-hip`, built with the HIP SDK);
game: OptiScaler's upscaler GPU time.

## Why d3dhip for K

26.10.1's shader compiler schedules the AGS WMMA network of the DXIL route about 9% worse than 26.8.1 (more
`s_waitcnt` and `s_delay_alu` in 10 of the 11 layers; same WMMA count, same registers and spills; RGP captures and
the extracted ISA of both drivers in `nightly-work/rgp`). The native HIP kernels are compiled by our own LLVM and do
not change with the driver. d3dhip runs them between the D3D12 engine's front and back.

d3dhip's handoff is now GPU-side: the split's front fence is created shared, imported into HIP as an external
semaphore (`HipNetworkDx::useFence`), and `launchAfter(wait, signal)` queues the semaphore wait, the network and the
signal on HIP's stream; the game's queue waits for the signal before the back. No CPU wait between the parts. Output is
byte-identical to the CPU-serialized handoff (48 of 48 replay files). `D4R_D3DHIP_SYNC=cpu` restores the old handoff.
Preset K's d3dhip was opt-in on RDNA3, `D4R_D3DHIP_K=1` (DXIL remains faster on 26.8.1); from driver 26.10.1 on it is
now the default there too (section below). On RDNA4 it is the default
in the fast set when the package has HIP layers (`compile_k.py --hip` with a gfx1201 kernel folder; the release
packages have none yet); `D4R_D3DHIP_K=0` keeps the DXIL network, and `[Kernels] PreferAccuracy = true` keeps it too.
RX 9070 XT, 26.10.1, harness, interleaved: 1.371 -> 1.297 ms at 1280x720 -> 2560x1440, 2.799 -> 2.744 at 1080p ->
4K, 1.406 -> 1.331 at 1706x960 -> 2560x1440. The HIP network's arithmetic differs from the DXIL network's (f32
against f16 accumulation): the output is not byte-identical (57 dB against the DXIL route on the motion scene, the same
quality-scene PSNR, blinds and flicker).

## d3dhip K by default on RDNA3 from driver 26.10.1 (2026-11-10)

The default (`D4R_D3DHIP_K` unset) now takes d3dhip on RDNA3 when the render GPU's driver is 32.0.32015 (AMD Software
26.10.1) or newer and the package has HIP layers; older drivers keep the DXIL network, the faster one on 26.8.1 (above).
The driver version comes from the render GPU's PnP driver key (`DriverVersion`), not from DXGI, so OptiScaler's
spoofing cannot change it; the shim logs it next to the render GPU. RX 7700 XT, 26.10.1, harness,
`scripts/hillclimb/perf/matrix-rdna3.json`, K packages rebuilt with the current compile_k.py:

| Case | DXIL | d3dhip |
|---|---:|---:|
| 1280x720 -> 2560x1440 | 2.48 (another session 2.94) | 2.26 |
| 1706x960 -> 2560x1440 (Ghost's Quality) | 2.61 | 2.39 |
| 1920x1080 -> 3840x2160 | 4.87 | 4.76 |

The DXIL network's time moved between sessions (2.48-2.94 ms at 1440p); d3dhip's did not (2.25-2.27).
Not measured: 26.8.1 (not installed), gfx1100/1102/1103, the game.

The gfx1101 HIP layers are now built without scratch (`kernels/windows-flags.txt`), which also lets vkhip run K in
native Vulkan games on RDNA3 (docs/experimental-vkhip.md). For d3dhip that flag set is byte-identical and
time-neutral (dx-K-*-hip within +-1.3%).

## Presets M and L through d3dhip on RDNA3 (2026-11-10)

D3D12 M and L on RDNA3 run the RDNA 3 package (FP16 WMMA, software FP8 storage); its DXIL network was the slow part
(M 8.3 ms, L 11.4 ms at 720p -> 1440p on an RX 7700 XT). The package's native HIP layers now replace it there, as on
RDNA4, by default when the package has them (`hipnet.bin`, `hip/`, `rdna3/m_planes.dxil`). The FP8 package's plane
stages cannot run on RDNA3 (FP8 WMMA), and the windows branch's input / expansion stages have no non-FP8 plane
variant, so the RDNA 3 engine keeps its own stages and converts between its token-major buffers and the network's
plane layout: `tokens` ([tok][64] FP8) into the network's input, its head into `dec1`, a byte permutation
(`engine/d3d12/m_planes.hlsl`, two dispatches). Everything else is the RDNA4 d3dhip machinery: the split list, the
GPU-fence handoff, one network per render size. `D4R_D3DHIP_M=0` or `D4R_DX_NET=dxil` keeps the DXIL network.

Layers: `D4R_VKHIP_LAYERS=1 kernels/build.sh m DIR` for the gfx110x target (the dataflow set; the release ships it as
`kernels/<target>-swin-hip`), packaged with `compile_m.py --hip DIR`; `build-engine-models.py` does that on a gfx110x GPU.

RX 7700 XT, 26.10.1, HAGS off, harness, interleaved (bench.py, 3 rounds), same package both ways:

| Case | DXIL | d3dhip |
|---|---:|---:|
| M 1280x720 -> 2560x1440 | 8.33 | 5.53 (-33.6%) |
| M 1920x1080 -> 3840x2160 | 16.86 | 12.60 (-25.3%) |
| L 1280x720 -> 2560x1440 | 11.36 | 8.91 (-21.5%) |
| L 1920x1080 -> 3840x2160 | 23.49 | 20.00 (-14.9%) |
| Ghost of Tsushima, M, 1712x960 -> 2560x1440 LDR (game, gpu_eval) | 13.16 / 13.32 | 8.76 / 8.75 (-34%) |
| Ghost of Tsushima, L, same (game) | 19.28 | 15.09 (-22%) |

Lane B (the HIP layers' arithmetic): quality-scene truth metrics equal (M psnr 18.59 / 18.58, blinds 21.10 / 21.07,
flicker 0.01532 / 0.01531; L 18.58 / 18.58, 21.48 / 21.49, 0.01300 / 0.01297); 43.7 dB between the two after 100
motion frames, the same picture. Vulkan games on RDNA3 are unchanged (their M / L still take the staging path: these
layers spill and vkhip refuses spilling gfx11 kernels). Not measured: gfx1100 / 1102 / 1103, RDNA4 (untouched: its
FP8 package path is separate).

## Presets M and L through d3dhip: the layer chain (RDNA4, 2026-11-05)

D3D12 M and L take d3dhip by default when the package has HIP layers. The HIP network used to launch M's ten Swin
layers (L reuses them) as a graph of ordinary kernels, with a full barrier between consecutive layers. Isolated, that
graph took 956 us against vkhip's 0.80 ms for the same code objects, so d3dhip M was 3% slower than DXIL at 720p ->
1440p. vkhip chains the layers instead (`D4R_VKHIP_M_CHAIN`): a block sets a flag once its outputs are written, and a
block of the next layer waits only for the producer blocks it reads. On Windows, `engine/hip_net.cpp` now does the
same. It uses the same flag layout, launches each consumer layer with `hipExtModuleLaunchKernel(...,
hipExtAnyOrderLaunch)` (no AQL barrier), and clears the flags before each evaluation; the chained nodes launch
directly, not through a graph. It applies only to layers built with `SWIN_DATAFLOW` (`d4r_dataflow`);
`D4R_HIP_M_CHAIN=0` restores the barriers. The isolated graph drops from 956 to 787 us. The output is byte-identical.
RX 9070 XT, 26.10.1, harness, interleaved: M 1.689 -> 1.534 ms at 720p -> 1440p (DXIL 1.65), 3.431 -> 3.371 at
1080p -> 4K, 2.208 -> 2.095 at 720p -> 4K; L 2.450 -> 2.334 at 720p -> 1440p, 5.308 -> 5.251 at 1080p -> 4K.
Preset K's layers are not built with the flags and keep their graph.

## Results

| | Replay (ms) | Game, average (ms) | Game, range (ms) |
|---|---:|---:|---:|
| DXIL network | 2.71 | 2.55 | 2.51-2.89 |
| d3dhip, GPU fence, HAGS on | 2.50 | 2.40 | 2.32-2.70 |
| d3dhip, GPU fence, HAGS off | 2.50 | 2.44 | 2.40-2.51 |
| d3dhip, tuned kernels (below) | 2.446 | not measured yet | |

Hardware-accelerated GPU scheduling (HAGS) causes regular spikes at the two cross-queue waits; with it off the range
is a third as wide. Recommend HAGS off with d3dhip.

Frame budget (replay, tuned): front (setup, exposure, input) ~0.21, HIP network ~1.32, back (reconstruction) ~0.73,
cross-queue handoff ~0.2. The HIP network costs the same in all eight window phases (1.345-1.359 ms per graph).

## Kernel flags tuned for Windows HIP on gfx1101

Swept one flag at a time per layer from the shipped flags, two rounds, every variant checked for byte-identical
output; the set below is 55 us faster than the shipped one (2.501 -> 2.446 ms) and byte-identical. The biggest gains:
enc2 102 -> 80 us and dec2 94 -> 78 us without `PWIN_NO_BUFFER_IMG` (buffer images win on Windows' HIP runtime).
Measured on Windows only: the source flag lines also drive the Linux builds, where these choices were tuned the other
way, so they are not in `kernels/k/*.hip` yet.

| Layer | Flags |
|---|---|
| enc0 | `-mcumode -DWM_HALF_BPERMUTE -DPWIN_VEC_SCALAR -DWM_CVT_ASM -DPWIN_PK_ELEM -DPWIN_PACKED_RESIDUAL -DPOS_NORM_LANE -mllvm -amdgpu-set-wave-priority` |
| enc1 | `-DPOS_VSHARE1 -DPWIN_VEC_SCALAR -DPWIN_PK_ELEM -DPOS_DIRECT_NORM -DWM_HALF_BPERMUTE` |
| enc2 | `-DPWIN_VEC_FROM_WEIGHTS -DPWIN_NORM_OWNER -mllvm -amdgpu-set-wave-priority` |
| enc3 | `-mllvm -amdgpu-set-wave-priority -DWM_HALF_BPERMUTE` |
| enc4 | `-DWM_HALF_BPERMUTE` |
| dec5 | `-DPOS_NORM_LANE -DPWIN_PACKED_RESIDUAL` |
| dec4 | `-DPOS_NORM_LANE` |
| dec3 | `-DWM_HALF_BPERMUTE -mllvm -amdgpu-set-wave-priority -DPWIN_PK_ELEM` |
| dec2 | `-DPWIN_VEC_FROM_WEIGHTS -DPWIN_NORM_OWNER -mllvm -amdgpu-set-wave-priority` |
| dec1 | `-DPWIN_VEC_FROM_WEIGHTS -DPWIN_QKV_SPLIT -DPWIN_NORM_OWNER -mllvm -amdgpu-set-wave-priority -DWM_HALF_BPERMUTE` |
| dec0 | `-mcumode -DWM_HALF_BPERMUTE -DWM_CVT_ASM -DPWIN_PK_ELEM -DPWIN_PACKED_RESIDUAL -DPOS_NORM_LANE -mllvm -amdgpu-set-wave-priority -DPOS_VSHARE1` |

## Tried and rejected

- Memory-word handoff (`hipStreamWaitValue32` / `hipStreamWriteValue32` against a D3D12 spin shader, as the Linux
  path does): HIP's stream memory operations on imported D3D12 memory never meet D3D12's writes or reads on Windows,
  in either direction (the wait hangs; the written word stays 0).
- Splitting the reconstruction so its head-independent part runs during the HIP network: that part is 0.38 of
  0.73 ms, but the only expensive piece, the 5x5 Catmull-Rom history resample (0.124 ms), would hand 16 bytes per
  pixel to the second pass (~0.14-0.17 ms to read back at 1440p): no gain. The other head-independent parts are cheap
  but hand over much more (the staged tile alone ~99 MB per frame).
- Reconstruction as wave32: slower (2.541 vs 2.453 ms) and 7 of 48 outputs differ (the highlight quad exchange).
- Input stage as wave64: byte-identical, slightly slower (2.463 vs 2.457 ms).
- DXIL network prefetch depth 4 instead of all tiles: byte-identical, -0.056 ms on 26.10.1 (not taken over: the
  d3dhip route does not run the DXIL network).

## What is left

Within byte-identical output the remaining pieces are the cross-queue handoff (~0.2 ms, Windows' cost of two queues),
the exact reconstruction (0.73 ms) and the large HIP layers (enc0, enc1, dec1, dec0: ~0.92 ms of the network, already
tuned). Native-half Gaussian sums in the reconstruction (the reference's own arithmetic, -0.06 ms) differ from the
current Windows build's f32 sums, which were chosen because AMD's Cyberpunk application profile breaks native half
accumulation.
