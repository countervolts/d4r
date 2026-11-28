# Presentation

Minecraft: `acquireNextTexture` (frame start) -> render -> `blitFromTexture` (records blit main target -> swapchain image, waits the acquire semaphore,
signals the present semaphore) -> `encoder.submit()` -> `present()`. 3 swapchain images, one VkQueue (graphics) shared with our work.

Frame k with factor N (N-1 generated frames between real frames k-1 and k):

| step | what | where |
|---|---|---|
| 1 | generation recorded in front of the blit, in the same submission; **G1** blitted into Minecraft's acquired image (instead of real k) | `VulkanGpuSurfaceMixin.blitFromTexture` (cancel + replacement) |
| 2 | Minecraft presents G1 | unchanged |
| 3 | `afterPresent` queues a job for the **presenter thread**, which presents G2..G(N-1) and the real frame (a snapshot taken by `generate`, so the main target can be overwritten) at `period*j/N` (sleeping when vsync is off): own acquire -> own command buffer (blit, flipped) -> submit -> present | `FrameGen.presenterLoop` |

Display order is G1..G(N-1), real k: the real frame is shown one frame late (the usual interpolation latency). The main target still holds real k at step 3
because Minecraft only starts frame k+1 after `present()` returns.

**Pacing.** The render thread never sleeps: the presenter schedules from the measured render period. With vsync (FIFO) acquire blocks per refresh instead of sleeping. If the presenter falls behind it drops the rest of that frame's output.

**Sync.** Minecraft and the presenter thread share one VkQueue and swapchain, so every Minecraft queue call (submit, present, waitIdle, acquire) takes `FrameGen.QUEUE` through mixin redirects; acquires poll with 1 ms timeouts so a blocked acquire never holds the lock. Our submissions go to the same queue after Minecraft's frame submission, so queue order gives execution order; every recorded block starts with an all-commands
barrier. Generated images of variant v are rewritten two frames later; their blits are queue-ordered before that. Command buffers use a ring of 8 with fences.

**Resets.** First frame, level change, camera jump > 24 blocks, a frame without a world (menu) -> script variant 2 (init) again.

## Pacer (vsync off)
Minecraft's own `present()` is cancelled (`VulkanGpuSurfaceMixin`, `FrameGen.deferPresent`); the image it blitted is presented by the pacer thread with the other N-1 images of the cycle on one clock:
interval `d = smoothed frame period / N * scale`; `scale` (0.85..1.25) is nudged so the schedule stays about N/2 intervals ahead of the clock (absorbs a late frame), a long idle restarts the clock with a one-interval lead,
and if another frame is already queued the in-between images of the older one are dropped. Waits sleep to 150 us before the due time and spin the rest.
Cost: about one extra cycle of latency. Statistics (present interval avg/sigma/max, late presents, lead, scale) are logged every 2 s (`[D4R] pacing:`) and shown in the overlay. Not yet measured in game.

## Output ring (11-28 fix)
The first pacer let the render thread wait for the presenter (images were rewritten two frames later) and the schedule's slowdown fed back into the frame rate (scale 1.25, bunched presents). Now `generateCompute` copies each generated frame into a 4-deep ring
(and `generateGfx` the real frame into a 4-deep snapshot ring), so the presenter reads stable copies and the render thread only waits if the ring is full. The pacer's scale is limited to 0.92..1.08, and a presenter far behind restarts its clock instead of presenting a burst.
Changing the factor at run time used to leak the images of queued frames (Minecraft then timed out acquiring: crash); `destroy` now returns them to the swapchain first. Async compute is disabled (it does not copy into the ring).

## Rate cap and swapchain depth
`cap=true` (default): the render thread sleeps in `deferPresent` so that `factor * real fps <= refresh rate` (display mode refresh, or `maxShownFps`). Above the refresh rate frames only tear and the pacer's work is wasted
(the 5x/6x logs showed sigma ~ interval, 20% late). The swapchain gets min(10, surface max) images so the pacer's N acquires per cycle never wait for the display to release one.
The overlay's "shown (real)" fps comes from the pacer's present counter.
