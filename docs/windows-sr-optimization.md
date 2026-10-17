# Windows K/M optimization (2026-10-17)

The local gfx1101 v20 package reduces SR interop overhead and the cost of strict M attention. Linux performance parity remains unverified. Measurements below are synthetic CPU interop times, including NGX, copies and synchronization; they are not GPU kernel times or game frame times.

## Changes

- The output CUDA array download and optional HIP format conversion now share one ordered legacy-stream batch. One whole-context completion precedes the D3D12 output signal. NGX's all-stream completion still precedes the batch. Output diagnostics insert their own completion before inspecting the canonical image.
- M enc2 and tube layers first sum their 16 FP16 products in FP32. Each product is exact in FP32. An error bound larger than gamma(16), including magnitude-sum error, determines whether the result is safely inside one FP16 rounding interval. Ambiguous intervals, cancellation, overflow and nonfinite cases retain the original sequential FP64 dot. Quantization, weights, rounding and tensor layouts are unchanged. Other M layers retain FP64 because the candidate regressed their replay timings.
- The local v20 package enables the existing `BatchInputCopies` setting. The source INI and parser fallback remain false; older installations retain their selection. This local package targets the tested RX 7700 XT only.
- The standalone engine build script now requests Release. This did not improve measured K frame timing and is not counted as a speedup.

## Final package checks

Final candidates use the packaged loader, rebuilt optimized shim and the actual CMake-built M objects. Controls use v19's shim and native objects. Runs alternate control/candidate/candidate/control, with 24 temporal frames per process and four warmup frames excluded from timing. All output hashes match exactly across all four processes. All frames pass the harness's finite-output check.

K render sizes are 1280x720 and 1706x960; M render sizes are 960x540 and 1280x720.

| Preset and mode | Output | v19 CPU interop median | v20 CPU interop median | Reduction |
|---|---|---:|---:|---:|
| K Quality | 1920x1080 | 3.190 ms | 3.063 ms | 4.0% |
| K Quality | 2560x1440 | 4.651 ms | 4.606 ms | 1.0% |
| M Performance | 1920x1080 | 11.107 ms | 10.251 ms | 7.7% |
| M Performance | 2560x1440 | 18.372 ms | 17.186 ms | 6.5% |

The small K 1440p difference is within observed run variation; do not treat it as a demonstrated GPU speedup. Earlier isolated input-batching runs measured 3.187 -> 3.092 ms and 4.690 -> 4.496 ms, with 128 exact frames per size. CPU load and synchronization make these timings variable.

Packed-format checks add 64 exact K frames per profile at 1280x720. R11G11B10 output measured 2.705 -> 2.566 ms, and BGRA8 output measured 2.726 -> 2.341 ms. Both native packed bytes and decoded half-image bytes are compared. These exercise the deferred output conversion as well as input batching. These format-specific timings must not be projected to other formats or games.

Final package evidence:

- `build/sr-opt-final-K-1920x1080-v1/summary.json`
- `build/sr-opt-final-K-2560x1440-v1/summary.json`
- `build/sr-opt-final-M-1920x1080-v1/summary.json`
- `build/sr-opt-final-M-2560x1440-v1/summary.json`
- `build/sr-opt-final-K-packed-v1/summary.json`
- `build/sr-opt-final-K-unorm-v1/summary.json`

The final package passes 512 exact temporal frames across these cases. This is same-GPU regression evidence, not NVIDIA reference-quality validation. Other GPUs, M Quality, real games and RR behavior were not validated in this pass.

## Numerical and build checks

`build/m-opt-pv-stress-v2/summary.json` records 262,144 GPU dots with zero differing FP16 bits against the original FP64 sum. Inputs include signed products over the finite FP16 exponent range, cancellation, every finite FP16 magnitude, rounding midpoints and tiny signed perturbations. All five synthetic M layer replays also match every allocation byte. Only enc2/tube improved sufficiently to select the bounded path.

Reproduce the dot check:

```powershell
$sdk = '.tools/therock-10.2.0a20260929/_rocm_sdk_core'
& "$sdk/lib/llvm/bin/clang++.exe" -x hip -std=c++17 --offload-arch=gfx1101 `
  --offload-device-only --no-gpu-bundle-output -mno-wavefrontsize64 -nogpuinc -nogpulib `
  -O3 -DD4R_DEVICE_ONLY_MINIMAL tools/windows/exact_pv_probe.hip -o build/exact_pv_probe.hsaco
python tools/windows/test_exact_pv.py --probe build/windows-rdna4-therock-gfx1101/d4r_native_replay_probe.exe `
  --hip-root $sdk --module build/exact_pv_probe.hsaco --output-dir build/exact-pv-fresh-run
```

`scripts/windows/build-native-m-experiment.ps1` builds strict M candidates without installing them. `-ReferencePv` restores the FP64-only control. K's experimental builder also accepts `-LaneExchange native`; it matched the tested 1080p case but provided no gain. K DS exchange remains selected. M LDS/DS flag comparisons provided no gain. Release versus unoptimized standalone K runners matched all five frame products but showed no timing improvement. Standalone K remains roughly 1.85 ms at 1080p and 2.91 ms at 1440p, above the reported Linux targets.

`build/sr-opt-player-config-v1/summary.json` passes the INI/environment and loader-selection checks. Both relevant CMake builds pass. Python compilation and PowerShell parsing pass for the changed tools.

The texture builder additionally accepts M folded post variants and an experimental `--wave64` / `-Wave64` option. It verifies the emitted wave64 target feature and retains strict arithmetic. Such objects contain NVIDIA-derived code and stay in ignored local build directories; they are excluded from v20.

The tested M HDR/render-motion post variant passes another 96 exact frames per resolution (`build/m-post-wave64-{1920x1080,2560x1440}-v1/`). Its CPU interop difference is small and varies between runs; it is not promoted. Enc0/dec0/post/downsample still use the translated texture path in v20. Matching Linux's broader native texture coverage and reducing cross-API overhead remain open work.

Accepted transient raw readbacks were hashed before pruning; summaries, logs, hashes and first-frame previews remain. Inventories are `build/sr-opt-readback-cleanup-v1.json` and `build/sr-opt-readback-cleanup-v2.json`. Earlier work and user data were not pruned.
