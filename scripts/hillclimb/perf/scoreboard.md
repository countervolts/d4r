# Hill-climb scoreboard (performance, native Windows, RX 9070 XT)

One line per kept commit: `<sha> score=<geomean ms D3D12 / Vulkan / headline> worst=<case ±%> note`.
Method and matrix: bench.py / matrix.json; journal: docs/hillclimb/journal.md.

## Phase 0 baseline (fe53aa0 shim, HIP packages in D:\d4r-hc\models, frames back to back)

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

Vulkan: vk-K-1440 3.475 (Vulkan network: no vkhip for K in the native frontend). The other Vulkan cases of that run
overlapped another GPU workload (three whisper-cli jobs from 21:14) and are being re-measured.
