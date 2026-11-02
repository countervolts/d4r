# Reading the DLSS inputs in place: findings

The colour, depth and motion inputs are copied from the game's textures into buffers shared with HIP on every frame. This page records an attempt to remove those copies by letting DLSS read the game's textures directly, why it does not work, and what was ruled out. None of the code for it is in the tree; the output-side work it grew out of is described in [performance.md](performance.md).

All of this was measured in the D3D12 harness on an RX 9070 XT (gfx1201) with ROCm 7.2, Mesa 26.2 (RADV) and Linux 7.2, preset M at 1280×720 → 3840×2160.

## The fault

With vkd3d-proton patch 0003 a game texture can be created linear on exportable memory and imported into HIP, exactly as the shim's own buffers are. A pointer to its texels is then available to CUDA. Three ways of reading through that pointer give different answers:

| Read | Result |
|---|---|
| `cuMemcpy2D` | current contents, every frame |
| A compute kernel using plain loads | current contents, 0 of 7,372,800 bytes differ |
| A compute kernel using a pitch-linear texture object | wrong from the first frame |

DLSS's kernels fetch their inputs through texture objects, so DLSS ran on wrong inputs. A probe kernel showed what the texture fetch returns: the image the texture held at its first upload in 81% of texels, with 32 bytes of unrelated data at every 256-byte step. That is the look of memory that has been freed and reused, not of an older version of the texture.

Writing through the same kind of mapping is not affected: the output kernel writes the game's output texture with plain stores, and Vulkan sees the result.

## What was ruled out

- **NGX and ZLUDA.** The probe is wrong before NGX evaluates anything, and ZLUDA passes texture creation straight to HIP.
- **A copy inside ROCm.** ROCm 7.2 creates the texture's image view directly on the imported pointer (`rocmemory.cpp`, `Image::createView`); the workaround path that would allocate a copy is disabled for HIP.
- **A wrong descriptor.** The hardware image descriptor stored in a bad texture object is identical to that of a working one apart from its address, and the address is the imported pointer.
- **The texture having moved.** The kernel driver's debug files (`amdgpu_gem_info`, `amdgpu_vm_info`) show the texture's memory in VRAM, exported, with an up-to-date mapping from before the import to the end of the run. Vulkan and HIP share one GPU address space in the process.
- **Stale handles or mappings.** Recreating the texture object every frame, and releasing and re-importing the memory at a new address every frame, change nothing.
- **Memory type and allocation style.** CPU-visible and CPU-invisible memory types behave the same, and a non-dedicated exportable allocation behaves the same.
- **A general ROCm, kernel or RADV bug.** `tools/hip_vk_texture_stale_repro.cpp` performs the same sequence natively (Vulkan object on exportable memory, upload, import into HIP, texture object, further uploads) and texture sampling follows Vulkan's writes for a linear image and a buffer, every memory type, one to four uploads before the import, game-like usage flags and zero-initialised memory.

So the same address in the same address space gives current data to a plain load and other data to a hardware texture fetch, only for textures created by vkd3d-proton inside the Wine process. The cause is not known. The untested differences from the native program are vkd3d-proton's image creation chain (alignment control, compression control, format lists), its allocation call as issued through Wine, and the path its uploads take in the driver.

A warning for anyone repeating this: copying texels out and back with `cuMemcpy2D` makes exactly the rewritten rows visible to the texture fetch, which means that copy writes into whatever memory the fetch is reading. It was done a few times during this investigation without visible effect, but it should not be repeated.

## The workaround that works, and why it was dropped

Since plain loads are correct, the fetches themselves can be replaced. For preset M the inputs are read by three kernels only: enc0 (colour, motion), post (colour, motion) and downsample (colour); depth is passed to enc0 but not fetched in the `mvhi_ldr` variant. All 20 of those fetches are integer texel fetches (`tex.base.2d.v4.f32.s32`), so no filtering is involved. `make_ptx.py` was extended to trace each fetch's texture handle back to the kernel's parameter block (enc0 offsets 80, 96, 136; post 0, 176; downsample 56) and to replace the input fetches with a load through the pointer HIP keeps in the texture object (`__hip_texture`: resource type at byte 96, pointer at 104, channel description at 112, width 136, height 144, pitch 152). An integer fetch outside the image returns zeros, not the edge texel; motion depends on that.

With colour, depth and motion read in place this way and the output written in place, no frame-sized copy was left and the output of a 48-frame panning scene was byte-identical. It was not faster:

| Configuration | Frame interval | DLSS kernel time |
|---|---|---|
| Copies, original kernels | 5.91 ms | 3.09 ms |
| Output in place only | 5.76–5.89 ms | 3.15–3.16 ms |
| Inputs and output in place | 5.89–5.93 ms | 3.27–3.30 ms |

The rewritten fetches add about 0.11 ms of kernel time, and about 0.07 ms even when unused because each fetch first checks whether the texture is tagged. The three input copies cost roughly 0.05–0.1 ms. The two cancel, so the code was removed.

## If this is picked up again

- The zero-cost fix is a working hardware fetch. The next step is to make the native reproducer fail by adding vkd3d-proton's behaviour to it piece by piece, or by running it through Wine's Vulkan layer.
- A cheaper workaround would need the per-fetch check removed (a separate kernel set chosen when the inputs are in place); a plain load plus conversion is still unlikely to beat a hardware fetch.
- Real games pass depth in a depth format, which RADV is not expected to offer with linear tiling (not checked here; the harness uses R32F), so depth would probably keep its copy either way.
