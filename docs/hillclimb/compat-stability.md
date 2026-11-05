# Hill climb: d4r on native Windows — compatibility and stability first, performance second

You are running a long, mostly unattended hill climb on d4r's native Windows build (branch `windows`). The
person who started you will check in now and then but will not review each change. You keep going until the stop
conditions at the bottom hold. Don't stop because one iteration is finished. When something blocks you, write it
down (see *Blockers*) and move on to the next failing scenario.

You work toward one number, the **gate score**. You build the gate before you change any product code. After that,
every change you keep has to raise the score or leave it unchanged, and it must never turn a passing scenario red.

## 0. What "better" means, in priority order

1. **No crash, no hang, no TDR.** This covers the game, the harness and the driver. A game that ran before your
   change still runs after it.
2. **No silent failure.** Any DLSS request a game or OptiScaler can make produces a frame (a correct remap is
   fine) or fails with a clear `d4r_nvngx.log` line that says what the user should change. A feature that
   creates fine and then shows black, banded or garbage output is the worst kind of failure here.
3. **Plausible output, not matching output.** There is no reference DLSS output at all. NVIDIA's own DLSS gives
   different output on different NVIDIA architectures, and d4r's output also differs by backend and math path (fp8,
   int8, f16, fast or exact). Never compare against "what DLSS would output", and never try to obtain such an
   output. Don't chase exactness. A frame is correct when it is
   finite and not black or frozen, has no edge bars or banding, follows the scene's motion, stays stable over time,
   and is clearly closer to the ground truth than a plain bilinear upscale (section 2, *Output checks*). Tight
   comparisons (`tools/windows/frame_compare.py`) are only for checking that a change which shouldn't touch pixels
   (shim, lifetime or logging) left them unchanged: same route, same GPU, before vs after.
4. **Honest diagnostics.** The log names the backend that really ran and any fallback that really happened. It
   doesn't print Linux/vkd3d-proton messages on native D3D12, and its lines stay in timestamp order even when two
   processes write to it.
5. **Performance (secondary).** Work on this only when every scenario is green, or when everything that is still
   red sits in the *needs hardware* queue. See section 5.

## 1. Ground rules (hard constraints)

- Work on a branch called `hillclimb/compat` cut from `windows`. Commit every change that gates green, and only
  those. Use small commits whose messages follow the repo's `Windows: ...` style. Never push and never open PRs.
  The person merges.
- **Never commit proprietary NVIDIA files** (DLLs, weights, captures, `.cubin`/`.fatbin`) or anything else that
  `.gitignore` excludes. Models and captures stay in `%LOCALAPPDATA%\d4r`, `build/`, `captures/` and `.tools/`.
- **Only launch games without anti-cheat.** These are allowed: Cyberpunk 2077, Hogwarts Legacy, Shadow of the
  Tomb Raider, The Witcher 3, Kingdom Come Deliverance II, Ghost of Tsushima DIRECTOR'S CUT, RESIDENT EVIL
  requiem, Assassin's Creed Shadows, Detroit Become Human, Stray, BioShock Infinite and Portal RTX.
  **Never inject into** Overwatch, Rainbow Six Siege, Deadlock, Rust, REPO or anything else with online
  anti-cheat. Doing so risks a ban on the person's account. Before you inject into any game not on the allowed
  list, check that it has no anti-cheat, then add it to the list in this file in a commit of its own that
  explains why it is safe.
- Use `scripts/windows/windows-game.ps1 -Action install/run/restore` for every game run, and always run
  `restore` afterwards, including after a crash. Never leave modified files in a game folder. Bound every run
  with `-RunSeconds`.
- Don't change drivers, TDR registry values, Windows security policy or GPU clocks/power. If a hang
  reproduces, shrink it down to a harness case instead of a game run.
- `send-game-key.ps1` only works for processes on its allowlist. Add a game there only if that game is on the
  allowed list above, and do it in its own commit.
- **Only the RX 9070 XT (gfx1201, RDNA4) is local.** A report from RDNA2/RDNA3 hardware can only be reproduced
  here if its cause doesn't depend on the architecture (preset arbitration, parameter parsing, lifetime,
  logging). Put anything that needs gfx103x/gfx110x on the *needs hardware* queue with the exact harness command
  a tester should run. Don't pretend you verified it.

## 2. Phase 0: build the gate (do this before any fix)

Create `scripts/hillclimb/` with these pieces:

- `gate.ps1`: the single entry point. It runs the tiers in order and stops at the first red tier, unless you
  pass `-Continue`. It writes `build/hillclimb/last-gate.json`, which holds each scenario's status, metrics and
  log excerpt plus the overall score, and prints a short summary at the end.
- `scenarios.json`: the scenario list (format below). Scenarios are data. Adding a scenario never needs a code
  change in the gate.
- `scoreboard.md` (committed): one line per kept commit, `<sha> score=<n>/<total> perf=<geomean ms> note`.
- `journal.md` (committed): what you tried, what failed, and why, so a later context window (after
  compaction) or another agent doesn't redo dead ends.

### Tiers

| Tier | What | Pass condition | Runs |
|---|---|---|---|
| T0 build | `scripts/build_windows_tools.ps1` (shim, harness, engine runners), plus the kernels that the touched code needs | it compiles with no new warnings in the files you touched | every change |
| T1 unit | `python -m unittest discover -s tests` with the environment from `docs/windows-tools.md`, plus `tests/windows`. Add an **exhaustive preset arbitration test** next to `tests/test_shim_config.py`, which already compiles `tools/d4r_render_presets.h`. It covers every combination of 6 slots × hints 0–15 and Latest × OptiScaler override on/off × `D4R_DLSS_PRESET`/`D4R_RR_PRESET` × SR/RR × which packages are installed. Invariants: it never picks a preset without a package; an explicit user choice is never silently replaced; the logged preset equals the chosen one. | no new failures; the three Windows failures known in `docs/windows-tools.md` are the only allowed ones, listed by name in the gate | every change |
| T1b call-sequence fuzzer | a seeded harness mode that issues random, legal NGX call sequences: create / evaluate / release / quality change / output resize / reset / parameter changes between frames / release with 0 frames / two features alive at once. Run it with the D3D12 debug layer, GPU-based validation and DRED on, and compare `QueryVideoMemoryInfo` before and after. A failing seed gets shrunk to its shortest failing sequence and saved as a T2 scenario. | no crash, hang, debug-layer error or VRAM growth over K seeds | every change: a few seeds; every 5 kept commits: a long run |
| T2 harness scenarios | `build/tools/d3d12_dlss_harness.exe` (and `vulkan_dlss_harness.exe` for Vulkan scenarios), driven by `scenarios.json` env vars, against the locally built shim and the models in `%LOCALAPPDATA%\d4r` | per scenario: exit 0, the expected number of completed frames, the expected log lines present, forbidden log lines absent, and the scenario's output checks pass | every change, ~minutes |
| T2b real OptiScaler in the loop | the same harness loaded through the pinned OptiScaler v10 build (`packaging/optiscaler-release.json`, `scripts/fetch_optiscaler.py`) the way a game loads it, as `dxgi.dll`/`winmm.dll` next to the harness. Port `scripts/run_d3d12_dlss_harness_optiscaler_proton.sh` to a PowerShell runner. Run it across an `OptiScaler.ini` matrix: `RenderPresetOverride` off / each preset / mixed per slot, ForceNvidia on/off, the upscaler and FG input/output pairs users actually use (FSR-FG inputs + DLSS output, OptiFG inputs + DLSS output), and quality changes from the menu, done as recreation | as T2. OptiScaler's own log is also checked: no errors, and the backend it reports matches d4r's log | every change to the shim's parameter, preset or lifetime handling; the full matrix every 5 kept commits |
| T2c recorded API traces | add `D4R_SHIM_API_TRACE=<file>` to the shim. It records every NGX/NVAPI entry point d4r receives, with its parameters (names, types, values), resource descriptions and states, thread ID, command queue, and create/evaluate/release order, but no pixel data. Add a harness mode that replays a trace with synthetic textures of the recorded formats and sizes. Record traces in every allowed game, in the menu, in play, and during a settings change, and keep them under `build/hillclimb/traces/` | each trace replays to the end without a crash, refusal, debug-layer error or VRAM growth, and every evaluate produces a frame that passes the output checks | every change; traces are small and fast |
| T3 capture replays | `D4R_HARNESS_REPLAY_DIR` replays of real game frames (`D4R_SHIM_CAPTURE_TRIGGER`, `engine/capture.py`, `scripts/windows/test-k-captures.ps1`, `test-m-captures.ps1`) | the output checks pass on real content; for a change that shouldn't touch pixels, the output is also unchanged on the same route | every change that touches the engine, models or input conversion |
| T4 real games | `windows-game.ps1 run` on the allowed games, with `-ValidateOutput`, plus `capture-game-window.ps1` screenshots | no crash; completed DLSS frames for the whole run; no `refused`/`failed` lines in `d4r_nvngx.log`; the screenshot judge panel passes (below) | at milestones: after every 5 kept commits, and before any perf phase |
| T5 perf | `tools/bench_local_presets.py` (1440p and 4K, sr and rr), `scripts/windows/benchmark-native-replay.ps1`, harness evaluate timing with `D4R_ENGINE_PROFILE=1` | no case regresses by more than 2% against the last kept baseline (noise is checked by rerunning) | perf phase, plus every 5 kept commits |

The **score** is the number of green T2, T2b, T2c and T3 scenarios plus the number of green T4 game runs. T2b
and T2c count the most: they test what games and OptiScaler actually do, not what d4r assumes they do. Most
field reports are integration bugs (preset arbitration, parameter conventions, lifetime, fallback, logging),
not model math. When a field report or a game run turns up a new call pattern, add the trace before you fix
the bug. T0/T1 are
preconditions: if either is red, the score is 0. T5 is reported as a separate geomean so performance never
substitutes for correctness.

### Scenario format

```json
{
  "id": "sr-optiscaler-override-preset-E",
  "source": "R1",
  "tool": "d3d12_dlss_harness",
  "args": ["1505", "847", "2560", "1440"],
  "frames": 30,
  "env": { "D4R_DLSS_PRESET": "11", "D4R_HARNESS_DLSS_PRESET": "5", "D4R_HARNESS_CREATE_FLAGS": "0x4b",
           "D4R_HARNESS_MOTION_SCENE": "1" },
  "expect_log": ["preset 5 has no native package"],
  "forbid_log": ["CreateFeature refused", "not vkd3d-proton"],
  "reference": "truth",
  "checks": ["finite", "frames_completed", "edges", "motion"]
}
```

`reference` is one of the following. Neither is a DLSS reference: `truth` is the *scene* itself, and
`same-route` is d4r's own earlier output, kept only to catch accidental changes.

- `truth`: the harness's quality scene ships a supersampled ground truth of the scene (`truthSamples` in
  `d3d12_dlss_harness.cpp`). The check is a loose bound, not a match. The error against the truth must be
  clearly lower than that of a bilinear upscale of the input, and no worse than the last kept build by more than
  a small margin you calibrate from run-to-run noise.
- `same-route:<path>`: the output of the last kept build on the same route and the same GPU, frozen under ignored
  `build/hillclimb/golden/`. Use it only for scenarios whose changes shouldn't move pixels; there,
  `frame_compare.py`'s tight thresholds apply.
- `none`: only the output checks apply.

Don't compare backends or GPUs with tight thresholds: fp8, int8, f16 and the fast or exact paths legitimately
differ. A large difference between backends (a visibly different image, not noise) is still worth a journal
entry.

### Output checks (plausibility, not exactness)

Each one has to be calibrated on known-good output, so it doesn't fire there, and shown to fire on a broken frame
you construct on purpose, before you trust it:

- `finite`: no NaN or Inf, and the frame isn't all zero or constant.
- `frames_completed`: every evaluate produced a new frame, and the output isn't stuck on an earlier one.
- `edges`: the failure R1 shows, dark bars or banding along the frame's edges. For each 8-pixel border strip,
  compare the mean luminance and the variance with the neighbouring interior strip. Flag strips that are
  constant, much darker than their neighbours, or periodic.
- `motion`: in the motion and pan scenes, the output's content moves the same way as the input (phase
  correlation between successive frames). This catches motion-vector scale or sign mistakes such as R3 without
  any reference image.
- `stable`: in a static scene with jitter, the output settles. Frame-to-frame change falls below a threshold
  after N frames and doesn't flicker.
- `invariant:<scenario>`: the same scene expressed two equivalent ways must give visibly the same result
  (loose bound): pixel-space vs normalised motion vectors with `mvScale`, R16G16 vs R32G32 motion vectors,
  R11G11B10 vs RGBA16F colour, inverted depth with the inverted-depth flag, an exposure texture vs
  auto-exposure at the same value.

### The screenshot judge panel (T4)

Theo's trick for things that don't have exact outputs. For each game run, take screenshots at fixed points (the
main menu, after load, after 30 s of camera movement if the game can be driven) in three versions: native
resolution with no upscaler, FSR/the game's own TAA upscaler at the same internal resolution, and d4r. Give them to
**three independent judge subagents**. Each judge gets only the images and this rubric: *edge bars/banding, black
or frozen regions, ghosting trails, shimmering, wrong exposure/colour, softness compared with FSR*. Each judge
returns pass/fail per item with one sentence per item. A run passes when at least 2 of the 3 judges pass every
item. Judges never see your code or your expectations, and you never edit a judge's verdict.

## 3. Initial backlog (field reports, nightly-20261102-windows)

These come from user reports gathered 2026-11-04. Raw logs and screenshots, if the person has copied them in, are
in `captures/field-reports/` (ignored by git). Read them when they exist. Some of these may already be fixed at
`HEAD`: for example, `b295f99` added a remap of unknown Super Resolution presets to K. **Always reproduce first.**
A scenario that is already green at `HEAD` gets added as a regression guard and nothing more.

| ID | Report | Turn it into |
|---|---|---|
| R1 | CP2077, RX 7600 XT (gfx1102), D3D12 1505x847→2560x1440. OptiScaler `RenderPresetOverride` = E (5) on every slot beat `D4R_DLSS_PRESET=11`. The log showed `native D3D12 CreateFeature refused: preset=5 flags=0x4b` and `CUDA fallback unavailable`, and the game showed dark bars on its left edge. | Scenarios for SR presets 0–15 × flags {0xb, 0x4b} × override on/off, all of which must produce frames (remap to K/L/M, logged once). Also: what is the left-edge banding when the feature *does* run? Reproduce it in CP2077 itself (it is installed) at 1505x847→1440p and add an `edges` scenario for it. |
| R2 | Double arbitration: OptiScaler's override "wins over `D4R_DLSS_PRESET` (which still fills Default slots)". A user saw K "not work" in GTA V Enhanced (RX 9070 XT) even though K created and recorded frames. | Scenarios that mix per-slot hints (some slots Default, some K, some M), with recreation when the quality mode changes. Check that the preset that ran is the one the log claims ran and the one the user asked for. Check the flags 0xb vs 0x4b difference (bit 0x40) between the K and M runs. |
| R3 | GTA V Enhanced uses normalised motion vectors (`mvScale 1.0,1.0`, mv fmt 34 = R16G16_FLOAT) and a 1x1 exposure fmt 2. CP2077 uses pixel-space `mvScale 1505,847`. | Pairs of motion-scene scenarios, normalised vs pixel MVs with `D4R_HARNESS_MV_SCALE`, that must give the same output, for K, M and L. |
| R4 | Feature churn: `ReleaseFeature(...) after 0 completed frames` on every in-game quality or preset switch, and many create/release cycles within seconds (GTA V, Dawnwalker). | A harness churn mode (add it if missing): create → release with 0 frames, ×N; create → 3 frames → release; quality switch mid-run; `D4R_HARNESS_RECREATE_OUTPUT`. It must not crash, leak VRAM (check committed memory before/after), or wait on a model load that never finishes. |
| R5 | Dawnwalker (UE 5.5, RX 9070 XT), preset L, 1279x720→3840x2160 (3x per axis), colour fmt 26 (R11G11B10), depth fmt 19, mv fmt 34: the image looked soft and odd. | Extreme-ratio scenarios (720p→4K, 3x and 3.33x) for K/M/L with `truth` references and the edge check. Add an informational softness comparison against FSR's output at the same ratio (`tools/fsr_upscale_probe.cpp`); it's reported but doesn't gate, since how soft is too soft is partly taste. R11G11B10 colour input as an `invariant` pair. This one is the least integration-like, so give it lower priority than R1–R4. |
| R6 | Logging: `VRAM interop: ID3D12DXVKInteropDevice1 unavailable (not vkd3d-proton?)` printed on native Windows. The log tail was appended out of timestamp order after Shutdown (likely two processes appending; see `d4r_nvngx_shim.cpp` around `_wfopen(... d4r_nvngx.log ...)`). | `forbid_log` for the vkd3d message on native D3D12. A two-process test: the harness run twice at the same time with the same `D4R_SHIM_LOG`. The lines must stay whole, each must be tagged with its PID, and they must be in order (or each process gets its own log file). |
| R7 | DRG crashes when frame generation is enabled (Linux side; a decoder fix is pending from another dev). DLSS FG wasn't in the 11-02 nightly; `45a6d1e` added it (RDNA3). | If an FG harness path exists in `fg/`, add FG scenarios (FSR-FG inputs + DLSS output, OptiFG inputs + DLSS output) that check for no crash on enable/disable/toggle. DRG is not installed, so it goes on the *needs hardware/game* queue. |
| R8 | `[Interop] VramInterop = true` reportedly crashes on the first frame (release notes; no user has confirmed it). | A harness scenario with VramInterop on. Reproduce it, fix it, or turn it into a clear refusal with a log line. |
| R9 | Install friction: `install.cmd` and the per-game model build are seen as tedious. | Not gated. Only look at this after T2–T4 are green: measure the time from a clean install to the first frame, and record ideas in `journal.md`. Don't redesign the installer without the person. |

## 4. The loop

Repeat:

1. Run `gate.ps1 -Continue` and read `last-gate.json`.
2. Pick the failing scenario with the highest priority (section 0 order). Within a priority level, pick the one
   you judge cheapest. If a scenario has been red for 3 attempts with no progress, park it in `journal.md` with
   what you learned and pick another.
3. Reproduce it with the smallest command possible, and write that command into the scenario so the scenario is
   red for the right reason.
4. Find the root cause in the shim/engine/kernels. Read the code around it and the docs in `docs/` that cover
   that path before you edit it. This codebase has many hard-won, commented workarounds, so respect them.
5. Fix it. Rerun the scenario, then the full gate.
6. Keep the change only if T0/T1 are green, no previously green scenario is red, and the score went up or stayed
   the same (the same score is fine for a pure diagnostics or refactor change that adds a guard scenario). Then
   commit, and append to `scoreboard.md` and `journal.md`.
7. Otherwise revert it (`git restore`/`git reset --hard` to the last kept commit on your branch only) and note
   the reason in `journal.md`.
8. Every 5 kept commits: run T4 and T5. If T5 regressed by more than 2%, find the commit responsible
   (`git bisect` over your kept commits) and fix or revert it before you continue.

### Blockers (Theo's step 4, which matters most over a long run)

Every 10 iterations, or whenever an hour passes without a kept commit, append a dated section to
`scripts/hillclimb/BLOCKERS.md` that answers these questions:

- Where did the last N iterations spend their wall time (build, kernel compiles, ZLUDA cache, game load,
  waiting on the GPU)?
- What would make the gate 2x faster? Examples: a narrower T0 that rebuilds only the shim, a precompiled-kernel
  cache, harness scenarios batched in one process, smaller resolutions for logic-only scenarios.
- What can't you verify locally, and what hardware, game or file would unblock it?

Then **act on whatever you can fix yourself**. Making the gate faster and more trustworthy is in scope and is
often the best use of an iteration. Anything that needs the person (another GPU, a game install, NVIDIA DLLs, a
decision) goes at the top of `BLOCKERS.md` under `## Needs a human`, and you carry on with something else.

## 5. Performance phase (secondary)

Enter this phase only when every scenario is green, or when everything still red is parked under
*needs hardware*. Targets, in order:

1. **Game-queue cost, not replay cost.** The field report shows 17.98 ms of DLSS frame time at 1440p on a
   7600 XT, which is far above the replay numbers. Use `windows-game.ps1 -ProfileStages` on CP2077 at 1440p
   to break down where the in-game cost goes (input copies, interop waits, fences, engine dispatch, output). Add
   `-AsyncInterop -ProfileGpuBoundary` only after the synchronous path is understood; the script requires the
   two together. Shrink the gap between game-queue time and replay time.
2. **The replay ratios** in `docs/local-gpu-preset-comparison.md` (local/NVIDIA): SR L 1.43x, RR D/E ~1.98x,
   RR F 1.66x are the largest gaps. The goal is ≤ 1.3x for SR and ≤ 1.6x for RR without loosening the
   accuracy gates.
3. Startup cost: first-frame latency after CreateFeature, and kernel/model cache misses.

Perf rules: interleave runs and use the median of medians (the method `bench_local_presets.py` already uses). A
win has to show up in three trials. It never ships with any T1–T3 regression. Changes to exact or approximate math
(the ini `use exact` / fast path) are allowed to change pixels. They still have to pass every output check, and
their `truth` error can't get worse than the calibrated margin.

## 6. Don't game the gate

The gate is only worth anything if it stays honest. You may not:

- loosen a threshold, delete or skip a scenario or trace, add a test to the allowed-failures list, or refreeze a
  `same-route` golden to match new output. The only exception is a separate commit titled `gate: ...` that
  explains why the old expectation was wrong, with evidence. Refreezing a golden after an intentional pixel
  change is fine when that change passed every output check and its `truth` bound.
- detect the harness, any `D4R_HARNESS_*` variable, or the scenario names inside the shim or engine.
- turn a crash into a silent no-op. A refusal is allowed only with a clear log line and a scenario that
  `expect_log`s it, and only when no correct output is possible (for example, RR with no RR package at all).
- claim a fix for hardware you don't have.

If you notice that you want to do one of these, write that in `journal.md`. It usually means the scenario is
measuring the wrong thing, which is itself a finding.

## 7. Stop conditions

Stop and write a final summary at the top of `scoreboard.md` (what's fixed, what's parked, and why) only when one
of these holds:

- every scenario is green, T4 is green on every allowed installed game, and two consecutive perf iterations
  produced no win of 2% or more;
- every remaining red item is under `## Needs a human` in `BLOCKERS.md`;
- the machine is unhealthy (repeated TDRs, the driver gets reset, the disk is full). Stop right away and say so.

A context reset or compaction is not a reason to stop. Reread this file, `journal.md`, `BLOCKERS.md` and
`last-gate.json`, then continue.
