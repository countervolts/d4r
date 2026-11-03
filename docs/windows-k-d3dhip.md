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
Preset K's d3dhip is opt-in, `D4R_D3DHIP_K=1` (DXIL remains faster on 26.8.1).

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
