# Hill climb: d4r performance on native Windows (RX 9070 XT, gfx1201), D3D12 and Vulkan games

You are running a long, mostly unattended performance hill climb on d4r's native Windows build (branch `windows`).
It covers both frontends. **D3D12 games** use the native D3D12 engine, with its DXIL network or the d3dhip route
(HIP layers between the D3D12 front and back). **Vulkan games** use the shim's native Vulkan frontend, with the
Vulkan engine or vkhip (native HIP code objects injected through `VK_KHR_pipeline_binary` on AMD's Windows driver,
LLPC; see `docs/experimental-vkhip.md`). The two share the models and much of the engine, so a win on one side
often points at a win on the other.
The person who started you checks in now and then but won't review each change. Keep going until the stop
conditions at the bottom hold. Finishing one iteration is not a reason to stop. When you are blocked, write it down
(see *Blockers*) and work on the next opportunity.

You climb one number, the **perf score** (section 2). Every change you keep has to lower it measurably. Every change
you keep also has to pass the **output guard** (section 3) and the **stability guard** (section 4). Only
performance is in scope. Compatibility bugs you come across go into `journal.md` for a human; don't fix them here
unless they block a measurement.

## 0. What counts as faster

What players feel is the cost DLSS adds to a game frame on the game's own queue. In priority order:

1. **End-to-end cost through the shim.** This is the median GPU time per evaluate through `d4r_nvngx.dll`, in
   `d3d12_dlss_harness` for D3D12 and `vulkan_dlss_harness` for Vulkan. It includes input copies, interop, fences,
   cross-queue handoffs, the engine, and the output copy or blit. It is the main metric.
2. **Frame-time stability.** Look at the p99 and the spread, not only the median. HAGS spikes at cross-queue waits
   are already documented in `docs/windows-k-d3dhip.md`. A change that lowers the median but widens the spread is
   not a win.
3. **Engine replay time.** For D3D12: `d4r-k-d3d12`, `d4r-m-d3d12`, `d4r-rr-d3d12` and their HIP/d3dhip routes,
   plus `D4R_REPLAY_PHASES` per phase. For Vulkan: `d4r-k`, `d4r-m`, `d4r-rr` with `vkhip` / `hip` / the Vulkan
   engine, `D4R_VKHIP_STAMPS=1` per layer, and `D4R_M_WARMUP_PASSES` / `D4R_M_TIME_EACH`. This is diagnostic. Use it to find where the time goes, but end-to-end decides
   whether a change is kept.
4. **Startup.** Time from CreateFeature to the first completed frame, on a cold and a warm cache.
5. **VRAM.** Not something to minimise. It's a constraint: it may not grow by more than 5% in any case
   (method in `docs/performance.md`, *Native D3D12 VRAM against official DLSS*).

The in-game numbers OptiScaler reports are the final check (section 4), not the score. Read the right
field. The overlay's "DLSS frame time" is the whole game frame; **"OptiScaler GPU" / upscaler GPU time** is
d4r's cost.

## 1. Ground rules (hard constraints)

- Work on a branch called `hillclimb/perf` cut from `windows`. Commit every kept change on its own, with a message
  in the repo's `Windows: ...` style that includes the before/after numbers. Never push, and never open PRs.
- **Never commit proprietary NVIDIA files** (DLLs, weights, extracted models, captures, `.cubin`/`.fatbin`,
  hsaco built from NVIDIA-derived code). These stay in `%LOCALAPPDATA%\d4r`, `build/`, `captures/` and `.tools/`.
- **Don't change drivers, GPU clocks, power limits, TDR or HAGS settings** to make numbers look better. Record the
  driver version and the HAGS state in every baseline. If HAGS matters for a result, measure both states and report
  both.
- **Only the gfx1201 is local.** Don't change what gfx11 (RDNA3) or gfx103x (RDNA2) builds compile or select unless
  that path is unchanged byte for byte, or unless the change is host-side, architecture-independent and covered by
  the output guard here. Put tuning ideas for other architectures in `BLOCKERS.md` under `## Needs hardware`, with the exact
  command a tester should run.
- **Don't change Linux builds.** `docs/windows-k-d3dhip.md` explains that the kernel flag lines in
  `kernels/k/*.hip` also drive Linux, where they were tuned the other way. Put Windows/gfx1201 tuning in the
  Windows build path (`scripts/windows/build-*.ps1`, gfx12 target settings) and never in shared flag lines.
- **Games:** for D3D12, only Cyberpunk 2077 and Ghost of Tsushima DIRECTOR'S CUT, both installed, single-player,
  with no anti-cheat. For Vulkan, Portal RTX (installed, Vulkan, no anti-cheat; d4r has never been tried on it, so
  getting it to run is a task in itself). The verified Windows Vulkan game, Indiana Jones and the Great Circle
  (`SUPPORTED_GAMES.md`), isn't installed: put it under `## Needs a human` instead of looking for it. Use `scripts/windows/windows-game.ps1 -Action install/run/restore` with `-RunSeconds`, and
  always run `restore` afterwards, including after a crash. Never inject into Overwatch, Rainbow Six Siege,
  Deadlock, Rust, REPO, GTA V or any other game with anti-cheat.

## 2. Phase 0: the benchmark gate and the baseline (before any optimisation)

Create `scripts/hillclimb/perf/` with these pieces:

- `bench.ps1`: the single entry point. It builds the current tree (T0), runs the guards (sections 3 and 4) and then
  the matrix below. It writes `build/hillclimb/perf/last.json`, which holds every case's samples, median, p99, VRAM
  and the hash of each route's output, plus the score.
- `matrix.json`: the cases. They are data; adding one needs no code change.
- `scoreboard.md` (committed): one line per kept commit,
  `<sha> score=<geomean ms> worst=<case ±%> p99=<geomean> note`.
- `journal.md` (committed): what you tried, the numbers, and why each attempt was kept or rejected, including
  failed ideas, so nobody retries them after a compaction. Seed it from the *Tried and rejected* and *What did not
  help* sections of `docs/windows-k-d3dhip.md`, `docs/windows-sr-optimization.md` and `docs/performance.md`.

### The matrix

The end-to-end cases run in `d3d12_dlss_harness` through the shim with `D4R_HARNESS_MOTION_SCENE=1`, with
`D4R_HARNESS_QUALITY_SCENE=1` for the output guard:

| Preset | Geometries |
|---|---|
| SR K (11) | 1706x960→2560x1440 (Quality), 1280x720→2560x1440, 1920x1080→3840x2160 |
| SR M (13) | 1706x960→2560x1440 (Quality), 1280x720→2560x1440, 1920x1080→3840x2160, 1280x720→3840x2160 (the documented RDNA4 case). Run each one in both the **fast set** (native FP8) and the **accuracy set**. |
| SR L (12) | 1280x720→2560x1440, 1920x1080→3840x2160 |
| RR F (6) | 1280x720→2560x1440, 1920x1080→3840x2160 |

**Scope: SR K, M and L, and RR F.** F is what games get by default: on native D3D12, DLSSD 310.9 runs F for every
hint except an explicit D or E when F is installed (`d4r_nvngx_shim.cpp`, CreateFeature). Select it with
`D4R_RR_PRESET=6` in both harnesses. RR D/E and the CNN presets A–C aren't optimisation targets. They stay in the
stability smoke only (section 4), so a change to shared RR code can't break them silently.

Also run one output format per SR preset other than RGBA16F (R11G11B10 or R10G10B10A2), because the copy/blit
path costs something real (`docs/performance.md`, *removing the output copies*).

The **Vulkan end-to-end cases** run the same presets and geometries in `vulkan_dlss_harness` through the shim:
- Set `D4R_HARNESS_ENGINE_DEVICE=1` so the harness enables the device extensions the shim asks for
  (`VK_KHR_pipeline_binary`, `VK_KHR_maintenance5`). Without them vkhip can't load and you'd be timing a fallback.
- Set `D4R_HARNESS_PAN` for motion and `D4R_HARNESS_R11_OUTPUT=1` for one case per preset. Indiana Jones'
  output is `B10G11R11_UFLOAT_PACK32`.
- Add `D4R_HARNESS_RR=1` for RR.
- Every case must log the route it ran on (`experimental vkhip (M/L) ready`, `vkhip: N Ray Reconstruction
  stages`, or why vkhip was unavailable). A case that lands on a different route than its baseline is invalid,
  not a regression or a win.

The **score** is computed separately per API (D3D12 geomean, Vulkan geomean), and the headline is their geometric
mean, so neither side can carry the other. The keep rule below applies to the cases of both APIs.

Run the replay cases with the same presets and geometries in the native D3D12 runners, with phases on. Only
`last.json` records them; they don't count in the score.

### Measurement method (copy the repo's own method; don't invent a new one)

- **Interleave the control and the candidate**: control, candidate, candidate, control, repeated at least 3
  times. Use warmup frames that aren't timed (≥ 40) and at least 100 measured evaluates per run. Take the median of
  the per-run medians, which is what `tools/bench_local_presets.py` and the NVIDIA table do.
- **Use game-like frame spacing.** The harness pauses 200 ms between frames by default. The GPU then sits at about
  9% busy and the driver's clock governor drops it to about 2790 MHz instead of 3180–3300 MHz. For M that read
  2.80 ms instead of 2.39 ms (`docs/performance.md`, *RDNA4*). Run every timed case with frames 0–4 ms apart, the
  way a game's load keeps the clocks up, and record the clock you sampled. Comparing a case measured with gaps
  against one measured without them is invalid.
- Use GPU timestamps around the evaluate, not CPU wall time. CPU interop times (`docs/windows-sr-optimization.md`)
  are only diagnostic, because they vary with CPU load.
- **Measure noise before you trust a gain.** In Phase 0, run the unchanged tree against itself 5 times and record
  each case's spread. A gain has to exceed 2× that case's noise, and be at least 1% of the case. Put both
  thresholds in `matrix.json`.
- Close other GPU users first. Record the driver, the HAGS state, the GPU clock behaviour (sampled), and the
  VRAM in use by other processes. If a run's environment differs from the baseline's, it doesn't count.

**Score** = the per-API geomeans of the median ms over the end-to-end cases, combined as above. A change is kept
when:

- the score drops by more than the noise threshold;
- no single case gets slower than its noise threshold allows;
- no case's p99 gets worse by more than 5%;
- VRAM grows by no more than 5%;
- the output guard and the stability guard both pass.

## 3. The output guard

**There is no reference DLSS output.** NVIDIA's own DLSS gives different output on different NVIDIA
architectures, so you never compare against "what DLSS would output". The guard compares d4r **with itself on
this GPU**, which is how this repo has always validated perf work ("48 of 48 replay files byte-identical", "512
exact temporal frames"). Every change belongs to one of two lanes, and its commit message says which:

- **Lane A: exact.** Output is byte-identical to the last kept build on every case, route and frame, including
  temporal history. Scheduling, sync, copies, barriers, launch configuration, kernel flags that keep the
  arithmetic, memory layout and fusion that keeps the order of operations all belong here. Most wins should come
  from this lane, and it needs no judgement.
- **Lane B: numerics-changing.** Examples: half accumulation, approximate math, a different reduction order,
  FP8 instead of FP16. This lane is only allowed in the **fast set** (ini `use exact = false`). The exact set
  must stay byte-identical. A lane B change must pass **all** of these plausibility checks on the quality, motion
  and pan scenes:
  - the frame is finite and not black or constant;
  - every evaluate produces a new frame;
  - no bars or banding at the frame edges;
  - the output follows the motion (phase correlation against the input);
  - a still scene settles and doesn't flicker;
  - the error against the harness's supersampled **scene** truth (`truthSamples`; this is the scene, not a DLSS
    output) is no worse than the last kept build by more than a margin you calibrate in Phase 0.

  Each check has to be calibrated in Phase 0, so it passes on known-good output and fails on a frame you break
  on purpose. Lane B also needs the in-game check in section 4 in **Cyberpunk 2077 specifically**:
  `docs/windows-k-d3dhip.md` notes that AMD's Cyberpunk application profile breaks native half accumulation, so
  a numerics change that's fine in the harness can break that game.

Never move a change from lane B into the exact set, and never relabel lane B as lane A.

## 4. Stability guard

- T0: `scripts/build_windows_tools.ps1` and the engine/kernel builds for gfx1201, plus a compile-only build of
  the gfx1101 and gfx1030 targets if the change touches shared code. It must have no new warnings in the files
  you touched.
- T1: `python -m unittest discover -s tests` and `tests/windows`, run with the environment from
  `docs/windows-tools.md`. The only allowed failures are the three known Windows failures that the gate lists by
  name.
- Smoke: every matrix case, plus RR D, E and A at 1440p on both APIs, runs to completion with no crash, hang,
  debug-layer error (run the smoke once per kept commit with the D3D12 debug layer on) or `refused`/`failed` log
  line. The out-of-scope presets also keep their output byte-identical unless the change was meant to touch them.
- **Every 5 kept commits, and before declaring any target reached:** a game run of Cyberpunk 2077 (1440p) and
  Ghost of Tsushima (the Quality case from `docs/windows-k-d3dhip.md`, 1712x960→2560x1440). Read OptiScaler's
  upscaler GPU time (average and range) and d4r's `-ProfileStages` breakdown. Pass: no crash, DLSS frames for
  the whole bounded run, and an upscaler GPU time that's no worse than at the last milestone. If the harness shows
  a gain but the game doesn't, write that down in `journal.md`. Finding out why is often the next opportunity.
  Also run Ghost of Tsushima at 1920x1080→3840x2160 with **preset M** (documented 5.8 ms upscaler time, the M
  figure to beat).
- **Vulkan game check**, at the same milestones: Portal RTX once it runs, with OptiScaler set up as in
  `SUPPORTED_GAMES.md`'s Indiana Jones entry (`VulkanUpscaler=dlss`, `NvngxPath`, `ForceNvidia=true`, an
  OptiScaler build with `patches/optiscaler-windows/0002` so the NGX device extensions reach the game's device).
  The log must show vkhip, not CUDA staging (~12 fps in Indiana Jones). If Portal RTX won't run, record why in
  `journal.md` and rely on the Vulkan harness cases until a human installs a verified Vulkan game.
- **The Vulkan network fallback must not crash.** AMD's Windows driver 26.10.1 crashes compiling the FP8
  cooperative-matrix `layer_m` SPIR-V, which the Vulkan network fallback still uses for render sizes vkhip
  wasn't built for (`docs/experimental-vkhip.md`). Include a Vulkan smoke case at an unprepared render size. Until
  that's fixed it's a known failure listed by name, and a change must not add any new crash of that kind.

## 5. Where to look first (seed backlog, measure before believing)

These are documented leads. Most were measured on a gfx1101, so re-measure each one on gfx1201 before acting on
it:

1. **The end-to-end gap.** Compare the harness's end-to-end time with the replay time for the same case. The
   difference is shim overhead: input copies, interop, fences, cross-queue waits and the output copy or blit.
   Break it down with `D4R_REPLAY_PHASES`, `-ProfileStages` and the per-stage D3D12 profiling from `71ed375`.
2. **K on gfx1201.** Is d3dhip (`D4R_D3DHIP_K=1`, opt-in) or DXIL faster here? The ~0.2 ms cross-queue handoff and
   the 0.73 ms exact reconstruction are the known remaining pieces (gfx1101 numbers). The per-layer flag sweep in
   `docs/windows-k-d3dhip.md` (one flag at a time, keep only byte-identical, two rounds) was done for gfx1101
   only. Repeat it for gfx1201 on the Windows build path.
3. **M, probably the biggest win.** M is DLSS 4.5's default and what RDNA4 users actually run, and L reuses M's
   Swin layers, so every M Swin gain carries over to L. The documented gap is large:
   - Ghost of Tsushima, 1920x1080→3840x2160: **5.8 ms** of upscaler time in game.
   - The harness, same geometry: **4.63 ms**.
   - NVIDIA's documented time for that case: **2.553 ms**. That's about 2.3× in game and 1.8× in the harness, far
     more than the 1.12–1.17× of the Linux replay.

   Those figures came from `D4R_CUDA_KERNEL_PROFILE`, probably on the older CUDA/ZLUDA route, so re-measure them on
   native D3D12 (`d4r-m-d3d12`, the shim's native M route) in Phase 0. The per-kernel split at 1280x720→4K (FP8
   fast set, `docs/performance.md` *RDNA4*):

   | Kernel | ms |
   |---|---:|
   | Swin layers (enc1/2/3, dec1/2) | 1.54 |
   | post (texture kernel) | 0.68 |
   | downsample (wave64) | 0.34 |
   | enc0 + dec0 (texture kernels) | 0.31 |
   | **total** | **2.86** |

   For comparison, NVIDIA's documented stages at 4K Performance are post 0.836 ms, downsample 0.254 ms and the
   ten inner Swin layers 1.078 ms.

   Look at these, in this order:
   - the gap between end-to-end and kernel time (whole-evaluation 2.39 ms under load vs 2.86 ms summed kernels:
     check what overlaps and what doesn't);
   - post and downsample, which run as translated texture kernels on Windows while Linux has broader native
     texture coverage (`docs/windows-sr-optimization.md`);
   - the Swin layers' FP8 path;
   - the accuracy set, which costs about 10% more than the fast set (3.15 vs 2.88 ms) and has to stay byte-identical.

   Track M's in-game number in Ghost of Tsushima at 1080p→4K at every milestone; it is the figure to beat.
4. **L.** It costs 1.43× NVIDIA's documented time (Linux RADV). L's `enc0` and `dec0` are its extra cost over M at
   NVIDIA too (0.503 / 0.345 ms against M's 0.214 / 0.117 ms). Locally its post kernel (1.2–1.5 ms, no matrix
   multiplies) is the largest kernel left. Do L after M, because M's Swin work carries over.

   **L in Vulkan games is the biggest single gap anywhere in this matrix.** On AMD's Windows driver (LLPC),
   `v_dot2` isn't available: it rejects `SPV_VALVE_mixed_float_dot_product`, and the FP8 cooperative-matrix
   SPIR-V crashes it. So L's input and expansion stages run as `-DDOT2_SOFT=1` emulation (`docs/experimental-vkhip.md`,
   *Preset L on native Windows*). The numbers at 1280x720→2560x1440:

   | Stage | Vulkan/vkhip (ms) | D3D12 engine (ms) |
   |---|---:|---:|
   | input | 2.115 | 0.490 |
   | expansion | 1.830 | 0.368 |
   | **whole frame** | **5.446** | **2.380** |

   In game (Indiana Jones) that's M about 92 fps and L about 46 fps. The obvious candidates:
   - run L's front and expansion stages as native HIP code objects through vkhip, the way the network already
     runs (no SPIR-V, so LLPC's limits don't apply; check whether they use scratch, which this path refuses);
   - an LLPC-friendly SPIR-V form that compiles to the same instruction.

   Whichever wins has to be byte-identical to the current output or go through lane B.
5. **M across the two APIs.** On Windows at 1280x720→2560x1440, vkhip runs M in 1.40–1.44 ms and the D3D12
   engine in 1.63–1.68 ms (network 0.80 against 1.01 ms). That suggests D3D12 games could gain from running M's
   network as HIP layers on d3dhip, as K already can (`D4R_D3DHIP_K`). The opposite direction counts too: where the
   D3D12 engine beats the Vulkan route for a preset, find out why.
6. **RR F.** NVIDIA's documented times are 2.100 ms (720p→1440p) and 4.658 ms (1080p→4K). The local ratio is
   1.66–1.70× on Linux; measure the Windows one. The state of F per API:
   - **D3D12:** F runs on the DXIL network. Its HIP objects (`rr2_*.hsaco`) still fault through the HIP runtime
     (`D4R_RR_HIP_F`), so d3dhip isn't available for F. **Fixing that fault is the obvious first step**: K got
     about 8% from d3dhip, and on Vulkan F's HIP route is about 18% faster.
   - **Vulkan:** F's vkhip route is 3.915 ms against 4.759 ms for its Vulkan route at 1440p, byte-identical, but
     opt-in (`ExperimentalRrVkHip = true`) because it was only about 2% ahead of the D3D12 engine. Re-measure it
     with game-like frame spacing. If it wins, make it the default (lane A, since it's byte-identical) in its own
     commit.
   - **Both:** AMD's driver needs the `enc0_llpc.spv` / `dec0_1_llpc.spv` workaround to compile F, and F builds 14
     HIP stages lazily on the first dispatch, which is a startup cost (item 8).

   Profile F's stages per API (`engine/d3d12/profile_rr.py`, `docs/rr-engine.md`) and start with whatever takes the
   most time.
7. **Output path.** On both APIs, can the copy or blit after DLSS be removed (the vkd3d-proton "in place" and
   "packed store" experiments, worth 0.2–0.3 ms at 4K M, were Linux only)? On Vulkan the output image is the
   game's own, so a direct store into it may be simpler than on D3D12.
8. **Startup.** On Vulkan this includes vkhip's pipeline-binary creation and RR F's lazy build of 14 stages on
   first dispatch. PSO and kernel load, model load, and first-frame latency after CreateFeature.

Targets, in the spirit of "don't stop until it's 2× faster": compared with the NVIDIA timings already documented
in `docs/nvidia-blackwell-presets.md` (timings only; the output was never compared), **match NVIDIA (≤ 1.0×)
end-to-end on both APIs** for every preset NVIDIA timed. NVIDIA has no documented K timing, so **K's target is 60%
of NVIDIA's M time at the same geometry**:

| Preset | 1280x720→2560x1440 | 1920x1080→3840x2160 |
|---|---:|---:|
| SR M | ≤ 1.081 ms | ≤ 2.553 ms |
| SR L | ≤ 1.413 ms | ≤ 3.161 ms |
| RR F | ≤ 2.100 ms | ≤ 4.658 ms |
| SR K (0.6 × NVIDIA M) | ≤ 0.649 ms | ≤ 1.532 ms |

These are deliberately ambitious. NVIDIA's figure brackets one EvaluateFeature, while ours is end-to-end through
the shim, and the reference card is a 200 W RTX PRO 4500 Blackwell. That's what the plateau stop condition is for.
Don't stop early because a target looks out of reach; stop when the profile says nothing large is left. The other
cases (Quality, 720p→4K, packed outputs) count toward the score but have no target. Before you start, record in `scoreboard.md` the Phase 0 baseline and the gap between it
and these targets.

### Tools

Use any profiler or analyser that helps, and install more if you need them. The list of what's already on this
machine (the Radeon Developer Tool Suite with RGP, RGA and RMV, RenderDoc, WPR/xperf, ROCm's llvm-objdump, the
repo's RGP helpers) and the rules for installing more are in `docs/hillclimb/goal.md`. Take timings without a
profiler attached; use profilers to find where the time goes.

## 6. The loop

1. Choose the opportunity with the biggest measured share of end-to-end time that you haven't exhausted. Measure
   first. Don't optimise a stage you haven't profiled.
2. Make a hypothesis and write it into `journal.md` before you change code (for example: "dec2 without
   PWIN_NO_BUFFER_IMG is faster on Windows HIP").
3. Make the smallest change that tests it. Run the single affected case interleaved against the control first,
   which is cheap. If it doesn't beat the noise there, revert it and note it.
4. If it does, run the full `bench.ps1`. Keep it only if section 2's keep rule and both guards pass. Then commit
   it and append to `scoreboard.md`.
5. Otherwise revert to the last kept commit on your branch (only there) and record the numbers in `journal.md`.
6. Every 5 kept commits, run the game check from section 4, then re-run the Phase 0 self-noise check. A clean
   rebaseline of the control catches drift from driver state, thermals or background load.

### Blockers (Theo's step 4)

Every 10 iterations, or after an hour without a kept commit, append a dated section to
`scripts/hillclimb/perf/BLOCKERS.md`:

- Where did the wall time go? Builds, kernel compiles, model builds, benchmark runs, game loads.
- What would make one iteration 2× faster? Examples: rebuilding only the touched kernel, a single-case fast
  bench, keeping fixtures and models resident, batching cases in one process.
- What can't be measured here, and what would unblock it? Another GPU, a profiler capture (RGP), a game.

Fix whatever you can yourself; making the bench faster and more trustworthy counts as progress. Anything that needs
the person goes under `## Needs a human` at the top.

## 7. Don't game the benchmark

You may not:

- change the matrix, warmup, frame counts or noise thresholds to make a change pass, unless it's a separate
  `bench: ...` commit with its reason and evidence;
- detect the harness, `D4R_HARNESS_*` or benchmark cases inside the shim or engine;
- skip work that a game needs: no caching an output across frames when the inputs changed, and no dropping
  history updates;
- move work outside the timed bracket when a game still pays for it on its queue;
- weaken the exact set, or label lane B work as lane A.

## 8. Stop conditions

Stop and write a summary at the top of `scoreboard.md` (baseline → final per case, what helped, what didn't, and
what's left) only when one of these holds:

- the K, M/L and RR F targets (section 5) are reached end-to-end on both APIs and confirmed in game. For Vulkan,
  confirmation in the harness is enough while no verified Vulkan game is installed; say so in the summary;
- 10 consecutive iterations across at least 3 different opportunities produced no kept change, and the profile
  shows nothing above 5% of end-to-end time that you haven't tried;
- everything left is under `## Needs a human`;
- the machine is unhealthy: repeated TDRs, a driver reset, or a full disk. Stop right away and say so.

A context compaction is not a reason to stop. Reread this file, `journal.md`, `BLOCKERS.md` and `last.json`, then
continue.
