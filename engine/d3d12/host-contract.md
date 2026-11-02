# Native D3D12 host contract (milestone 1)

This is the contract between a D3D12 host (the game, through NGX/OptiScaler) and
d4r's native backend, as implemented by `host.h` and tested by `d4r-host-test`.
The spike runs a diagnostic temporal shader in place of inference. The contract
is what later milestones keep; the shader is not.

## Device and resources

- d4r borrows the application's `ID3D12Device`. It creates no device or queue,
  and does not use AGS to create or replace one.
- Capabilities are checked on that device when the backend is created. If they
  are missing, creation fails with a reason; nothing falls back silently.
- All per-feature memory is allocated when the feature is created, sized for
  its maximum resolution: history, a constant ring and a shader-visible
  descriptor heap with a fixed number of frame slots. Recording a frame
  allocates nothing.
- Pipelines and the root signature belong to the backend and are shared by
  its features. These are where model weights will live later.

## Recording

`Feature::record(list, frame, &ticket)` records into the caller's open direct
command list:

- Entry states: color `NON_PIXEL_SHADER_RESOURCE`, output `UNORDERED_ACCESS`.
  Both are left in those states.
- d4r sets its own descriptor heap, root signature and pipeline, and does not
  restore the caller's. The caller rebinds its own state after the call.
- UAV barriers surround the work, so earlier writes in the list are visible
  to it and its result is visible to later commands.
- Only the output rectangle is written.
- d4r keeps references to the borrowed color and output resources until the
  recording retires. The caller may release its own references immediately.

## Submission and completion

Every successful `record` must be followed by exactly one of these calls:

- **`submitted(ticket, fence, value)`**: the list was executed on the queue,
  and `fence` is signaled to `value` after it on that queue. A feature uses
  one fence, with strictly increasing values, and one queue. This call commits
  the frame's temporal history.
- **`discarded(ticket)`**: the list was closed or reset without being
  executed. History stays at the last submitted frame.

Until then, a further `record` on that feature returns `pending`. A frame's
history has exactly one successor, so a host cannot record frame N+1 before
reporting frame N.

A frame slot, its constants, descriptors and retained resources are reused or
released only once `fence->GetCompletedValue() >= value`. There are no
fixed-delay assumptions and no CPU waits. If every slot is in flight, `record`
returns `busy`, and the host decides whether to wait or skip the frame.

## Release, shutdown and device loss

- `Backend::release(feature)` is safe while work is queued. The feature is
  destroyed by a later `collect()` once all its recordings have retired.
- The backend destructor is the only place that waits on the CPU. The wait is
  bounded, and it covers submitted work that is still outstanding.
- After device removal, every call returns `deviceLost`. Released features
  are freed without waiting, because a removed device never executes queued
  work, and shutdown does not hang.

## Open questions for the NGX/OptiScaler integration

- **Where the fence value comes from.** NGX's evaluate call passes only a
  command list. Something must supply `(fence, value)` after the host's
  `ExecuteCommandLists`, or say that the list was dropped. The candidates are:
  - an OptiScaler extension that reports it explicitly;
  - hooking `ID3D12CommandQueue::ExecuteCommandLists` for lists that hold a
    d4r recording, then signaling a d4r fence on that queue;
  - a command-list close/reset hook for discards.

  Hooking the queue is the most host-independent option, and the most
  intrusive one. Choose one before milestone 3.
- **Recording ahead.** Hosts that record frame N+1 before submitting frame N
  hit `pending`. If that turns out to be common, history needs a deferred
  commit keyed on submission order, not on recording order.
- **Descriptor heaps.** Hosts that cannot tolerate d4r replacing the bound
  descriptor heaps would need d4r to use descriptors allocated from the host's
  heap instead.

## Results (RX 9070 XT, AMD 26.9.1, Windows 11)

`d4r-host-test`, `--debug` and `--gpu-validation` all pass 8 of 8 checks:

- recording embedded in the application's list, with the output rectangle
  (pixels outside it unchanged);
- a frame-paced 24-frame sequence with a cut;
- retirement-gated slot reuse behind a queue gate;
- a discarded recording, plus rejected misuse (double resolve, non-increasing
  fence value, a second fence, out-of-range rectangle and size);
- release with queued work;
- two features sharing one backend;
- no NVIDIA, ZLUDA or vkd3d modules loaded, and `d3d12.dll` from System32;
- device removal.

Planted bugs fail the matching checks:

- Committing history at record time fails the discarded-recording check.
- Reusing slots without checking the fence fails the retirement and release
  checks.
- Not retaining borrowed resources removes the device.

## D3D12 -> Vulkan interop cost

`d4r-interop-bench` runs this on the game's D3D12 queue:

1. Copy the color (RGBA16F), depth (R32F) and motion (RG16F) textures into
   shared buffers.
2. Signal a shared fence. Vulkan imports it as a timeline semaphore, waits,
   runs, and signals.
3. The D3D12 queue waits, then copies the shared output buffer into the
   game's output texture.

A probe copy checks the data path end to end. The baseline runs the same
copies without the Vulkan round trip. These are medians of 600 frames per
mode, with modes alternating (latency) or in blocks (three frames in flight).
Clocks were not pinned. The first 200-frame run was distorted by clock
ramp-up, so use `--frames 600` or more.

| Render -> output | Copies (GPU) | Round trip, latency | Round trip, 3 in flight |
|---|---:|---:|---:|
| 2560x1440 -> 3840x2160 | 0.42 ms | +0.09–0.10 ms | +0.09–0.11 ms |
| 1920x1080 -> 3840x2160 | 0.33 ms | +0.09 ms | +0.12 ms |
| 1708x960 -> 2560x1440 | 0.19 ms | +0.09 ms | +0.10 ms |

A native D3D12 backend reads the game's textures directly, so it would avoid
both columns. The interop path therefore costs about 0.3–0.5 ms per frame on
top of the Vulkan engine's own K frame time. That frame time is still to be
measured natively on Windows (`d4r-k`, which needs the model and capture).
These numbers do not yet include:

- format conversion the engine would need beyond raw copies;
- moving the copies into the host's command list under the lifetime contract
  above;
- contention with game work on the same queue.
