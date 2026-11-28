# Vulkan requirements

- Instance: Minecraft's Vulkan 1.2. Layer: `VK_LAYER_NVXRT_nvx` (d4r `fg/dlssg/nvxrt`, built by `scripts/build-native.sh`), enabled through the environment before instance creation.
  It stays inert unless the GPU is an RDNA3 RADV device and `NVXRT_CACHE` holds a gfx11 kernel cache.
- Device extensions requested through Minecraft's `FeatureSet` mechanism (`VulkanFeatureSetsMixin` adds `D4rFeatureSet` to `optionalFeatureSets()`; enabled only if supported):
  `VK_NVX_binary_import`, `VK_NVX_image_view_handle` (both supplied by the layer).
- Features: added by the layer itself in `vkCreateDevice` (shaderFloat16/Int8/Int16/Int64, storageBuffer8/16Bit, bufferDeviceAddress, descriptorIndexing + partially bound/update-after-bind,
  scalarBlockLayout, vulkanMemoryModel, shaderBufferInt64Atomics, subgroupSizeControl, computeFullSubgroups, integer dot product, cooperative matrix, workgroup memory explicit layout, storage image formatless access).
  Read the enabled set back with `VulkanDeviceAccessor` (`d4r$enabledFeatures()` contains `FEATURE_SET` when the extensions were enabled).
- Minecraft images are always in `VK_IMAGE_LAYOUT_GENERAL`; colour RGBA8_UNORM, depth D32_FLOAT reversed-Z. The engine's own images are all GENERAL too.
- Memory: engine buffers (about 150 MB at 2560x1440) and images (about 200 MB) are separate allocations on the device; weights are uploaded once at creation.
