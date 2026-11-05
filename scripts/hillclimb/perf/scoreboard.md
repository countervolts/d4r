# Hill-climb scoreboard (performance, native Windows, RX 9070 XT)

One line per kept commit: `<sha> score=<geomean ms D3D12 / Vulkan / headline> worst=<case ±%> note`.
Method and matrix: bench.py / matrix.json; journal: docs/hillclimb/journal.md.

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
