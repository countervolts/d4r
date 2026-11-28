# Minecraft 26.3 rendering lifecycle (verified against the 26.3 jar, decompiled by Loom)

Names below are real 26.3 (unobfuscated) classes.

**Backend selection.** `Minecraft` tries `PreferredGraphicsApi.getBackendsToTry()` (`VulkanBackend`, `GlBackend`); `--graphicsBackend vulkan`
forces it. `VulkanBackend.createDevice()` builds a Vulkan **1.2** instance (`VulkanInstance`), picks a physical device
(discrete preferred), enables `requiredFeatureSets()` + every supported `optionalFeatureSets()` and calls `vkCreateDevice`
in `VulkanBackend.createDevice(FeatureSet, VulkanPhysicalDevice)`. Queues: `VulkanDevice.graphicsQueue/computeQueue/transferQueue`
(`VulkanQueue`: VkQueue + family). The mod detects Vulkan with `RenderSystem.tryGetDevice()` -> `FrontendGpuDevice.backend()` (accessor mixin) `instanceof VulkanDevice`.

**Command buffers.** `VulkanCommandEncoder` records into a transient `VkCommandBuffer` from a 2-deep pool ring and collects
them in a `VulkanQueue.Submission`; `submit()` sends one `vkQueueSubmit` per frame with a timeline semaphore.
`allocateAndBeginTransientCommandBuffer()` and `execute(VkCommandBuffer)` are public: `execute` ends Minecraft's current buffer
and appends an external one, so D4R records into its **own** buffer and ownership is never shared.

**Images.** All textures are created with `VK_IMAGE_LAYOUT_GENERAL` as their steady layout (render passes use dynamic rendering with
GENERAL attachments) and full memory barriers between buffers. Texture usage maps only to SAMPLED / COLOR / DEPTH / TRANSFER bits
(`VulkanConst.textureUsageToVk`): **no STORAGE**, hence D4R writes its own RGBA16F storage image and the bridge blits it
into Minecraft's RGBA8 target. `MainTarget`: `RGBA8_UNORM` colour + `D32_FLOAT` depth. Depth is **reversed-Z**, zero-to-one
(cleared to 0.0; `Projection` swaps near/far). Framebuffer rows are in GL orientation (row 0 = NDC y -1); no Y flip exists in the shaders.

**Frame (`GameRenderer.render`).** clear main (window size) -> `renderLevel()` (projection -> `LevelRenderer.render` frame graph: sky, terrain,
entities, particles, weather, translucency; then `render3dHud`: hand, which clears depth first unless `consistentDepthRequired`) ->
`blitEntityOutline` -> `applyPostEffects` (PostChain) -> `fogRenderer.endFrame()` -> clear depth -> `guiRenderer.render()` (HUD, windowed size) ->
`Minecraft` submits and presents (blit main -> swapchain). All world targets are sized from `gameRenderer.mainRenderTarget()`.

## Hooks implemented (`GameRendererMixin`, `VulkanGpuSurfaceMixin`, `VulkanFeatureSetsMixin`, accessors)
1. `VulkanFeatureSets.optionalFeatureSets()` RETURN: adds `D4rFeatureSet.FEATURE_SET` (extensions only; before `vkCreateDevice`).
2. `GameRenderer.renderLevel` `@ModifyVariable consistentDepthRequired` (ordinal 2): forced true while frame generation is active so the hand does not replace the world depth.
3. `renderLevel` `@ModifyArg ProjectionMatrixBuffer.getBuffer(Matrix4f)`: capture the final projection (bobbing included), `viewRotationMatrix`, camera position. Nothing is jittered.
4. `GameRenderer.render` before `FogRenderer.endFrame()`: the HUD-less point; record world colour copy + depth/motion-vector preparation into an own command buffer.
5. `VulkanGpuSurface.blitFromTexture` (cancellable) and `present` (RETURN): generation + presentation, docs/presentation.md.
