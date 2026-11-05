# Hill-climb scoreboard (performance, native Windows, RX 9070 XT)

One line per kept commit: `<sha> score=<geomean ms D3D12 / Vulkan / headline> worst=<case ±%> note`.
Method and matrix: bench.py / matrix.json; journal: docs/hillclimb/journal.md.

## Summary (2026-11-05, updated after the post-plateau round; game checks still owed)

Stopped under performance.md section 8's plateau rule: the last 10 iterations, across K's network scheduling, the
d3dhip handoff, RR F's HIP stages, M's flags, L's front stages and the stage wave sizes, kept nothing (journal). The
stop was rejected (>5% items untried); the round after it kept three FP8-conversion changes (441a237, 92ec16d,
0f61b35). What is left above 5% of a case needs a kernel project (below). **No result is confirmed in game yet**: the game folders
hold release installs that `windows-game.ps1` refuses (BLOCKERS, Needs a human). Vulkan is confirmed in the harness
only: no verified Vulkan game is installed, and Portal RTX has not been run.

| Case | Baseline | Final | Target | Gap now |
|---|---:|---:|---:|---:|
| dx-K-1440 | 1.384 | 1.285 | 0.649 | 1.98x |
| dx-K-4k | 2.772 | 2.711 | 1.532 | 1.77x |
| dx-M-1440 (DXIL / d3dhip) | 1.674 | 1.664 / 1.534 | 1.081 | 1.42x (d3dhip) |
| dx-M-4k (DXIL / d3dhip) | 3.692 | 3.518 / 3.371 | 2.553 | 1.32x (d3dhip) |
| dx-L-1440 (DXIL / d3dhip) | 2.433 | 2.447 / 2.334 | 1.413 | 1.65x (d3dhip) |
| dx-L-4k (DXIL / d3dhip) | 5.376 | 5.407 / 5.251 | 3.161 | 1.66x (d3dhip) |
| dx-RRF-1440 | 4.126 | 4.061 | 2.100 | 1.93x |
| dx-RRF-4k | 9.089 | 8.942 | 4.658 | 1.92x |
| vk-K-1440 | 3.553 | 1.085 | 0.649 | 1.67x |
| vk-K-4k | 5.374 | 2.378 | 1.532 | 1.55x |
| vk-M-1440 | 1.422 | 1.431 | 1.081 | 1.32x |
| vk-M-4k | 3.201 | 3.216 | 2.553 | 1.26x |
| vk-L-1440 | 5.158 | 2.181 | 1.413 | 1.54x |
| vk-L-4k | 11.315 | 4.886 | 3.161 | 1.55x |
| vk-RRF-1440 | 4.650 | 3.359 | 2.100 | 1.60x |
| vk-RRF-4k | 9.600 | 7.378 | 4.658 | 1.58x |

Scores (geomean): D3D12 2.532 -> 2.480, Vulkan 4.089 -> 2.530. The matrix pins D3D12 M/L on DXIL (installed packages
have no HIP layers); d3dhip, the default with HIP packages, is the second figure.

What helped:
- RR F on vkhip by default (18ab21b).
- L's matrix stages on LLPC through spirv-opt (696ea5d).
- K's Vulkan network reads transposed weight tiles on LLPC (fd6c117).
- d3dhip K by default on RDNA4 (e4fd9a5).
- K on vkhip in the native Vulkan frontend (cdc0787), made safe by finding the cross-wave race in K's enc0 / dec0
  (79b9ee4, which also adds the Windows kernel flag file).
- d3dhip chaining M / L's layers like vkhip (ed503cc).
- LLPC's FP8 conversions without the per-element guard: RR F's enc0 / dec0_1 (441a237), L's front stages (92ec16d).
- RR F's D3D12 gather encodes FP8 with the hardware conversion in uniform flow (0f61b35).
- RR F's encoder 0 on vkhip (944bd20), and dec0_1 through descriptor set 0's table (ffde45b).

What didn't (journal):
- DXIL M two windows per wave; output_k as wave32.
- The gfx1201 K flag sweeps beyond the kept lines; the M flag sweep (combined -0.59%).
- A per-dispatch counter chain for K.
- Wave64 / spirv-opt variants for L's front stages; stage wave 32 for K, M and L.
- The CPU d3dhip handoff.

What's left, largest first:
- **RR F's Vulkan stages chained without barriers.** Removing the 15 barriers between the HIP stages bounds the gain
  at 3.33 -> 2.98 ms (replay; dependent stages overlap there, so only an upper bound). It needs per-block dataflow
  like M's, plus write-after-read waits for the reused arena regions and mirror-aware ranges at the image edges
  (journal).
- **RR F's translated stages.** D3D12 ablations at 720p -> 1440p (4.03 ms): enc0 0.66, post 0.56, dec0_0 0.48, dec0_1
  0.47, gather 0.40 ms. Their FP8 conversions are now one clamp + one conversion per element on both APIs; the rest is
  translated NVIDIA code (lane shuffles, selects). On Vulkan enc0 and dec0_1 now run as HIP (944bd20, ffde45b). The gather and post stay
  SPIR-V: they sample textures, which needs sampler descriptors in the same set-0 scheme and a port of their filtering.
- **RR F's D3D12 network.** The same stages as HIP run 1.61 ms on vkhip against 2.05 ms as DXIL (AMD's D3D12 compiler:
  per-element clamps the HIP kernels avoid, more registers). d3dhip would carry the HIP version into D3D12 games, but
  its two cross-API fence waits cost ~0.6 ms here (prototype: 4.13 against 3.87 ms end to end), more than it gains.
- **K's output stage.** 0.37 ms Vulkan / 0.47 DXIL, about a third of K. AMD's DXIL compiler scalarises half vectors.
- **K's network.** A per-block dataflow chain for the deep layers is worth at most ~5% end to end (measured bound).
  The deep layers' phase kernels make it a large job.
- **L's front stages on LLPC.** About 1.09 ms against 0.86 on D3D12; a HIP port.
- **The d3dhip handoff.** ~0.12 ms, two cross-API fence waits; possibly OS scheduling, which is off limits here.
- **For a human.** The release packages need gfx1201 HIP layers (BLOCKERS). With them, D3D12 M/L would take the
  faster d3dhip route.

## Phase 0 baseline (fe53aa0 shim, HIP packages in D:\d4r-hc\models, now C:\d4r-hc\models, frames back to back)

D3D12 (bench.py self-noise run, median of 6 run medians per side; control-vs-control deltas <= 0.72%):

| Case | ms | Target | Gap |
|---|---:|---:|---:|
| dx-K-1440 | 1.384 | 0.649 | 2.13x |
| dx-K-4k | 2.772 | 1.532 | 1.81x |
| dx-M-1440 | 1.674 | 1.081 | 1.55x |
| dx-M-4k | 3.692 | 2.553 | 1.45x |
| dx-L-1440 | 2.433 | 1.413 | 1.72x |
| dx-L-4k | 5.376 | 3.161 | 1.70x |
| dx-RRF-1440 | 4.126 | 2.100 | 1.96x |
| dx-RRF-4k | 9.089 | 4.658 | 1.95x |

Untargeted D3D12 cases: K 1706x960->1440p 1.427, K RGB10A2 1.385, M 720p->4K 2.267, M Quality 2.432, M accuracy
1.629 / 3.509 (byte-identical to the fast set: the D3D12 M engine has one set), M RGB10A2 1.662, L RGB10A2 2.373.
D3D12 score (geomean of all 16 D3D12 cases): 2.532 ms.

Vulkan (bench.py self-noise run 2026-11-05, idle GPU; control-vs-control deltas <= 0.35%, spreads 0.26-2.92%):

| Case | ms | Target | Gap |
|---|---:|---:|---:|
| vk-K-1440 | 3.553 | 0.649 | 5.47x |
| vk-K-4k | 5.374 | 1.532 | 3.51x |
| vk-M-1440 | 1.422 | 1.081 | 1.32x |
| vk-M-4k | 3.201 | 2.553 | 1.25x |
| vk-L-1440 | 5.158 | 1.413 | 3.65x |
| vk-L-4k | 11.315 | 3.161 | 3.58x |
| vk-RRF-1440 | 4.650 | 2.100 | 2.21x |
| vk-RRF-4k | 9.600 | 4.658 | 2.06x |

Untargeted: vk-M-1440-r11 1.412. Vulkan score (geomean of the 9 Vulkan cases): 4.089 ms. K runs the engine's Vulkan
network (the native Vulkan frontend has no K vkhip); L runs soft-dot2 front stages on LLPC.

## Kept commits
18ab21b score=- / 3.957 / - worst=vk-L-1440 +0.99% (spread 1.38%) RR F on vkhip by default on AMD's driver: vk-RRF-1440 -18.9%, vk-RRF-4k -13.5%, byte-identical (lane A)
696ea5d score=- / 3.292 / - worst=vk-RRF-1440 -0.25% L's matrix stages on LLPC through spirv-opt: vk-L-1440 -56.8%, vk-L-4k -55.5% (lane B, Vulkan L on AMD's driver only)
fd6c117 score=- / 2.810 / - worst=dx-K-1440 1.372 (unchanged) K's Vulkan network on LLPC reads transposed weight tiles: vk-K-1440 3.610 -> 1.653, vk-K-4k 5.461 -> 2.927 (lane A for the tiles, the LLPC load variant byte-identical too)
e4fd9a5 score=- / - / - (dx-K geomean 1.650 -> 1.577) worst=dx-K-4k -1.95% d3dhip K by default on RDNA4: dx-K-1440 -5.40%, dx-K-4k -1.95%, dx-K-q1440 -5.32%, dx-K-1440-rgb10 -5.11% (lane B; Cyberpunk check blocked, BLOCKERS)
cdc0787 score=- / 1.628 (K cases) / - worst=dx-K-1440 +0.02% K on vkhip in the native Vulkan frontend, scratch on gfx12: vk-K-1440 1.652 -> 1.107 (-32.97%), vk-K-4k 2.938 -> 2.392 (-18.57%) (lane B; 4K nondeterminism fixed in the next commit)
2291ccc score=- / - / - worst=vk-K-1440 +0.62% full grids for vkhip K on AMD's driver (noise-level cost; did not fix the race, reverted in the next commit)
ed58f6e score=- / - / - worst=- tool only: vk_pipeline_stats writes the pipeline binary (VKSTATS_ELF)
7c029f0 score=- / - / - worst=vk-K-4k +0.6% vkhip scratch opt-in: K's Vulkan route back on the Vulkan network, byte-identical to before cdc0787 (vk-K-4k 2.916 -> 2.922)
79b9ee4 score=2.654 -> 2.557 (d3d12 2.499 -> 2.489, vulkan 2.819 -> 2.628) worst=vk-L-1440 +0.54% K cross-wave barrier (POS_HB_SYNC), Windows flag file, vkhip K default on gfx12: vk-K-1440 -34.36%, vk-K-4k -19.53%, dx-K -1.4..-2.0% identical
ed503cc score=- / - / - (side matrix, D3D12 M/L on d3dhip: 2.523 -> 2.399) worst=dx-L-4k-hip -1.07% d3dhip chains M/L's layers (hipExtAnyOrderLaunch + flags): dx-M-1440-hip -9.19%, dx-M-4k-hip -1.75%, dx-L-1440-hip -4.74%, all identical (lane A); main matrix unchanged
441a237 score=- / 2.620 -> 2.584 / 2.551 -> 2.534 worst=vk-L-4k +0.18% RR F's LLPC enc0 / dec0_1 FP8 conversions without the per-element guard (RR_LLPC_SAT): vk-RRF-1440 -5.82%, vk-RRF-4k -6.21%, every case identical (lane A on the harness; NaN / FP16_OVFL caveat in the journal)
92ec16d score=- / 2.574 -> 2.561 (vulkan) worst=- preset L's LLPC input / expansion stages convert FP8 with a clamp (M_LLPC_SAT, normalised tokens stay guarded): vk-L-1440 -1.97%, vk-L-4k -2.82%, every case identical (FP16_OVFL caveat in the journal)
0f61b35 score=2.485 -> 2.480 (d3d12; dx cases only, 3 rounds) worst=dx-K-1440 +0.71% (spread 1.26%) RR F's D3D12 gather (enc0g) hardware FP8 encode in uniform flow: dx-RRF-1440 -1.50% (spread 0.15%), dx-RRF-4k -1.55% (spread 0.36%), every case identical
944bd20 score=- / 2.561 -> 2.537 (vulkan; vk-RRF cases benched, 3 rounds, the rest carried) worst=- RR F's encoder 0 as a vkhip HIP kernel (enc0_hip_prep.py + glsl2hip.py, Windows builds only): vk-RRF-1440 -3.09% (spread 0.73%), vk-RRF-4k -2.74% (spread 0.41%), identical
ffde45b score=- / 2.537 -> 2.530 (vulkan; vk-RRF cases benched, 3 rounds, the rest carried) worst=- RR F's dec0_1 as a vkhip HIP kernel reading descriptor set 0's table (no descriptor_buffer; Windows builds only): vk-RRF-1440 -1.73% (spread 0.8%), vk-RRF-4k -1.83% (spread 0.81%), lane B (49 dB pan / 59 dB still against control, finite)
947afda score=- / - / - (dx-RRF cases only, tools/ab.sh 3 rounds, same shim with D4R_RR_DX_NET=dxil as control) worst=- RR F on d3dhip in D3D12 games (HIP network between the split list's parts, shared-fence handoff): dx-RRF-1440 4.091 -> 3.841 (-6.11%, runs within 0.6%), dx-RRF-4k 9.035 -> 8.387 (-7.17%), lane B (35 dB against DXIL after 200 motion frames; quality-scene truth metrics equal; deterministic)
ab6aac7 score=- / - / - (dx-RRF cases, tools/ab.sh 3 rounds, control D4R_HIP_RR_ENC0=0) worst=- RR F's encoder 0 in the d3dhip HIP section: dx-RRF-1440 3.848 -> 3.798 (-1.29%), dx-RRF-4k 8.387 -> 8.235 (-1.82%), byte-identical to the previous d3dhip route
a953312 score=- / - / - (dx-L cases, tools/ab.sh 3 rounds, same shim) worst=- L's two d3dhip plane stages with two windows per wave (WPW=2, Windows DXIL only): dx-L-1440 2.354 -> 2.317 (-1.59%), dx-L-4k 5.237 -> 5.154 (-1.59%), byte-identical
