# Native kernels

The native kernels are RDNA3 (gfx11) and RDNA4 (gfx12) code objects that ZLUDA runs in place of specific DLSS kernels. Each one has the same name, launch shape and parameter block as the PTX kernel it replaces, and reads the same buffers, including the network weights inside NVIDIA's DLL. No weights or NVIDIA code are stored here.

## The override hook (patches/zluda/0004)

With `D4R_ZLUDA_NATIVE_DIR=DIR`, whenever NGX loads a PTX module and asks for kernel `NAME`, ZLUDA looks for `DIR/NAME.hsaco` and uses its kernel `NAME` instead. The code object can export:

| Symbol | Meaning |
|---|---|
| `NAME_prep` kernel | launched first on the same stream, with the same parameters (typically re-lays out weights) |
| `u32 d4r_prep_blocks` | grid size of the prep kernel (128 threads per block) |
| `u32 d4r_prep_key_offset` | byte offset of a u64 in the parameter block; the prep is skipped while that value (the weights pointer) repeats (0 = no key) |
| `u32 d4r_prep_key_at` | the same as offset + 1, so a key at offset 0 can be named (M's layers keep their weights pointer there) |
| `u32 d4r_prep_key_slots` | N > 0: the kernel keeps one prepared image per key for up to N keys; keys beyond N share an overflow image. This requires stable weight-address identity. Enc3 disables key caching and prepares every launch instead. |
| `u32 d4r_block_z` | replaces the launch's block z dimension (more waves per window) |
| `u32 d4r_grid_x` | replaces the grid (persistent kernels) |

Kernels without a file in `DIR` are compiled from PTX as usual, so a partial set works.

**Release directories.** A directory that holds `d4r-kernels.txt` (or `<gfx target>/d4r-kernels.txt`) is checked before use:
- `kernels/tools/kernel_manifest.py DIR nvngx_dlss.dll...` writes the manifest. For each `NAME.hsaco` it records the FNV-1a 64 hash of the PTX module that defines `.entry NAME`, taken from each DLL given.
- The CUDA bridge serves a kernel only after NGX has loaded a module with one of the listed hashes.
- It picks the KFD GPU with the most SIMDs, then selects that GPU's target folder. `D4R_GPU_ARCH` overrides the choice. On gfx12, `D4R_ZLUDA_WMMA_FP8_NATIVE=1` selects the `<arch>-fp8` folder.

The release is built this way, with hashes from DLSS 310.7.0 and 310.9.1. The PTX of every replaced kernel is identical in those two versions.

## Families

**DLSS 4, preset K (`kernels/k`, `dltss_pwin_*`).**
- **Network:** a Swin-style U-Net of eleven layers over 8×8 token windows. There are encoders enc0–enc4 with 2×2 patch merging, a bottleneck dec5, and decoders dec4–dec0 with patch expansion and skip inputs. The window attention is either learned (query × key plus a bias table) or position-only (a fixed table times the values).
- **Templates:**
  - `pwin_layer.h`: four waves per window, each owning a 16-token tile.
  - `pwin_pos.h`: position-only layers with two token tiles per wave, which halves weight traffic.
  - `pwin_wide.h`: the deep layers, which have only 6–77 windows per frame. It uses 4·NG waves per window, split by token tile and channel group, so the GPU stays busy.
- **Weights:** prep kernels expand the weights into WMMA operand images once.

**DLSS 4.5, presets L/M (`kernels/m`, `rrlite_*`).**
- **Network:** the Swin blocks of DLSS 4.5, whose weights are FP8. Their prep kernels expand the weights to f16 WMMA operands. The enc3 tube layer prepares on every launch: NGX can recycle a weights address for a different tube block after feature recreation, so persistent address-only reuse can return stale weights and corrupt the image after quality changes. Other layers retain their existing preparation policy.
- **Template:** `swin_block.h` covers encoders, the tube-shaped enc3 and decoders.
- **Output encoding:** the FP8 output is encoded two values at a time with packed 16-bit operations (`enc8x2`, exhaustively equal to the scalar encoder), and the 2×2 patch merge reads the rounded f16 values the codes decode to, from a row layout without LDS bank conflicts, instead of decoding the bytes again in every wave.
- **Weight loads:** enc1 forms each weight tile's address in scalar registers (`SWIN_SCALAR_BLOAD`); the 8-wave layers are faster without it.

**Preset L texture variants.** L uses unfolded `rrlite_enc0_4x4_*`, `rrlite_dec0_4x4`, and `rrlite_post_3_*` kernels. The L build keeps their full neural arithmetic and replaces surface stores with the existing native format-conversion functions. M's folded enc0 tail and dec0 head cannot be used for these variants. All four input flag combinations and all eight post variants are built, along with the shared downsample kernels. `all` and `tex` include these variants; `l` builds the shared Swin layers plus only the L texture set. L remains experimental; these replacements do not establish RTX parity or game performance.

**Motion mitigation.** `[Kernels] NativeSwinEncoders = false` (`D4R_NATIVE_SWIN_ENCODERS=0`) bypasses only `rrlite_enc1_4x4` and `rrlite_enc2_4x4` in the native manifest. L/M then use the original translated PTX for those encoders, retaining native tube, decoder and texture kernels. Keep `PreferAccuracy = true` for the tested conservative path. It reduced the extra building-edge trails in a 64-frame moving Townfall capture at 853×480 → 2560×1440 on gfx1101; it does not recover detail absent from the low-resolution input or establish RTX parity. The default remains `true` for speed. Flat developer sets without manifests fall back entirely when this switch is off; add a manifest to keep selective acceleration.

Initial validation: a gfx1101 fast L build completed two Ultra Performance harness frames (192×108 → 576×324, DLSS 310.7). Its finite, nonzero RGBA16F output matched the original translated L texture kernels byte for byte with the same native Swin layers. A gfx1201 FP8 accuracy L set compiled successfully; RDNA4 hardware and 4K game output have not been tested. Full packaging rebuilds missing L variants when given an older texture bundle.

**Ray Reconstruction, the denoiser (`kernels/tex` + `cuda_dldn_engine_*`).**
- **Network:** `nvngx_dlssd.dll` defines the DLSS-D denoiser: six Swin encoders (`cuda_dldn_engine_swin_enc0_kernel` … `enc5`), five decoders (`dec0` … `dec4`), a high-fidelity kernel-prediction network whose output kernel is `cuda_dldn_engine_hkpn_output_kernel_transformer`, and the luma/auto-exposure/`reduce_sum` helpers. It shares no kernel with the Super Resolution libraries.
- **Native today:** the hkpn output kernel (and its `_diamond_wallaby` twin), built by `kernels/build.sh rr` as a wave64 surface-store replacement. `make_ptx.py` replaces `sust.b.2d.v4.b16.zero` — a whole RGBA16F texel — with `d4r_sust_v4b16`; the neural body and the texture reads are unchanged.
- **Still translated:** all eleven Swin layers. ZLUDA compiles NVIDIA's PTX at load time, and that compile is what costs the time (see the table below). A native replacement has to recover each layer's attention, norm and merge stages, the way `kernels/m/rrlite_*` does for the upscaler.
- **Manifest:** the denoiser DLL is passed to `kernel_manifest.py` as `--rr-dll`, because its hashes may only authorize `cuda_dldn_engine_*` replacements. Without that split, a name both libraries define — `dl4rt_input_kernel` — would let a super-resolution binary be served for the denoiser's different implementation of it.

Measured on the RX 9070 XT (gfx1201, RDNA4), Ray Reconstruction preset E, 1280×720 → 3840×2160, `D4R_CUDA_KERNEL_PROFILE` in the D3D12 harness, 16 frames after 6 warm-up, with the shipped `[Kernels]` settings (`NativeFp8`, `IgnoreDenormals`; `IgnoreDenormals` alone is worth 11% here, and `FastMath`, the f32 accumulator shadows, a 1024- or 128-thread bound, both scheduler strategies and the wave64 switch are all neutral or worse):

| kernel | ms/frame | kernel | ms/frame |
|---|---|---|---|
| dec0 | 7.71 | dec4 | 1.16 |
| enc0 | 6.49 | dec3 | 1.17 |
| dec1 | 3.66 | enc4 | 1.02 |
| enc1 | 3.48 | enc3 | 0.92 |
| dec2 | 2.18 | enc5 | 0.65 |
| enc2 | 1.74 | luma, exposure, reduce_sum | 0.05 |
| hkpn output (translated → native) | 1.44 → 1.00 | **total** | **31.6 → 31.1** |

The denoiser's Swin block is the same block as presets L/M's: `cuda_dldn_engine_swin_enc0_kernel` computes position-only attention `S = bias + Q*Q^T` with no K projection, a 16x16 f16 bias table at 512 bytes per head, and one fused Q/V projection - the score `mma` at line 8406 takes both of its operands from the same accumulator registers, `%r1900`/`%r1920`, through their `cvt.rn.satfinite.e4m3x2.f16x2` at 8126/8129 and 8294/8297. That is `kernels/m/swin_block.h`'s block (`T16 S = bias; k32(S, a0, a0, a1, a1);`), not `kernels/k`'s learned-attention one. So the native denoiser layers can be built by instantiating that template - a prep that permutes the denoiser's own weight image into its slot layout, a parameter struct for the 144-byte block, and the surface stores through `tex_common.h` - rather than by writing a Swin block from scratch.

The eleven Swin layers are 30.2 of those 31.6 ms. They are instruction-bound, not MMA-bound: the compiled `enc0` kernel is 83 373 instructions per thread for 536 FP8 WMMAs (0.6% of the stream), and the frame reaches about 6% of the card's FP8 WMMA peak. The stream is 13 k operand-gather shuffles (`ds_bpermute`/`v_perm`/`permlane`), 15 k packed 16-bit ops from the software `cvt.rn.satfinite.e4m3x2.f16x2` requantization, 9 k selects and 13 k ALU-dependency waits. `D4R_ZLUDA_IMPLICIT_MAX_BLOCK` (256 in the shipped config), `D4R_ZLUDA_SCHED_STRATEGY` and a 128-thread bound move the total by under 3%; a 128-thread bound additionally fails the launches of kernels that use larger blocks, and the whole-module wave64 switch produces no launches at all. Removing the gather and the codec — what a native layer does — is the remaining work.

## RDNA4

`kernels/common/wmma_layout.h` selects the WMMA operand and accumulator layout. gfx12 uses eight f16 values per lane in each K half; the accumulator rows are `i + 8·half`. The gfx11 path retains its original layout. `D4R_WMMA_LAYOUT=12` on gfx11 is a test shim: it exercises gfx12 indexing but executes gfx11 WMMA, so its replay output can be compared byte for byte with the existing gfx11 build.

The gfx12 M variant can use native e4m3 FP8 WMMA with `D4R_NATIVE_FP8=1`. Its activations are requantized to e4m3 at operand load, and its prepared weights remain bytes. The matching ZLUDA and texture-tail variant uses `D4R_ZLUDA_WMMA_FP8_NATIVE=1` / `D4R_TEX_FP8=1`; the release bridge chooses the `-fp8` kernel folder. `NativeFp8` in d4r.ini controls that choice and defaults to on. The f16 widening path remains available.

The gfx11 refactor and gfx12-layout shim matched the recorded gfx11 replays and 88-frame K/M harness images byte for byte on the RX 7700 XT. rocjitsu gfx1201 replay comparisons put the native K layers within 96–109 dB PSNR of gfx11, and M differed in 0.001–0.3% of e4m3 bytes. These are emulator results, not RDNA4 hardware results. rocjitsu's FP8 arithmetic is not hardware-checked; bit-exactness, speed and driver stability on an RDNA4 GPU remain unverified.

For ZLUDA-compiled code, compare rocjitsu runs with other rocjitsu runs: the emulator does not reproduce the denormal-flush mode of those code objects, so a GPU byte comparison is not a valid arithmetic check. Captures with texture or surface objects must not be sent to the replay tools; the standalone `tail_check.hip` and the full D3D12 harness cover those tails.

**Texture kernels (`kernels/tex`).**
- **The problem:** some DLSS kernels are mostly texture sampling and scalar maths that ZLUDA already compiles well, apart from a few slow parts.
- **The approach:** for those, `make_ptx.py` edits NVIDIA's PTX (extracted from your DLL at build time):
  - surface stores become calls into `tex_common.h`;
  - the `roundf` idiom that forces ZLUDA into strict-FP mode is rewritten exactly;
  - for M's enc0 and dec0, a range of the kernel is replaced by a native tail or head.
- **Build:** ZLUDA's `d4r_emit` compiles the edited PTX for `D4R_GPU_ARCH` with the HIP bitcode linked in (`D4R_ZLUDA_EXTRA_BC`), and the resulting code object is saved. This is an offline per-target build.
- **Every flag combination:** DLSS picks kernel variants by the game's settings (motion vectors at render or display resolution, HDR or LDR input, inverted or regular depth, ...). `kernels/build.sh` builds all of them: 24 K output kernels, 8 M post, 4 M downsample and 4 M enc0 kernels. enc0's native tail is spliced into each variant at its own line, with the registers read off the variant's PTX (`make_ptx.py` checks that the code after the cut matches the reference variant line for line).
- **Wave64:** kernels without MMAs can be compiled as wave64 (`D4R_ZLUDA_WAVE64=1`, ZLUDA patch 0006): each wave runs two CUDA warps, and RDNA3 issues FP32 work for all 64 lanes at once. The post and K output kernels are built this way (post 0.89 → 0.77 ms), with bit-identical results.
- **Output redirect:** the kernel that writes a preset's result can write it straight into the shim's output buffer instead of a CUDA array: `hiluma_engine_output_*` for K, `rrlite_downsample_kernel_*` for M (M's post kernel works at 3840×2160; the downsample produces the output). See `D4R_SHIM_OUTPUT_DIRECT` in [architecture.md](architecture.md).

## Numerics

The conservative kernels round their f16 accumulator after each MMA step (k16 for K, k32 for M). The fast kernels keep f32 accumulators through a chain (`PWIN_F32ACC`, `SWIN_F32ACC`); RDNA3 M also skips intermediate FP8 re-quantization (`SWIN_NO_Q8`). Wider arithmetic changes the network's intermediate values; it does not establish better image quality. Recorded fast-versus-conservative replays were close (about 66 dB PSNR), but that is not an RTX reference comparison or a guarantee for all scenes.

The installed K set uses f32 accumulation for enc0–enc2 and dec0–dec2 and exact rounding for the deep layers. The M set uses the fast numerics throughout.

**`[Kernels] PreferAccuracy = true`** (default `false`, environment `D4R_PREFER_ACCURACY=1`) selects a separate accuracy set for all K/M network layers and all texture/output variants, on every supported target, including gfx12 FP8. K disables `PWIN_F32ACC`; M restores FP8 quantization and rounds each k32 accumulator to f16. Texture builds use wave32 and retain the original denormal requirements; the existing exact round-half-away rewrite is retained. Native texture heads/tails already round their accumulators per k32 step. The bridge also disables `IgnoreDenormals`, translated f32 accumulator shadows, wave64 translation and NGX synchronization elision before ZLUDA loads, including when native kernels are off or fail their PTX hash check. Other interop and latency settings remain independent.

The bridge selects `kernels/accuracy/<target>` (or `<target>-fp8`). Accuracy directories carry a `d4r-accuracy.txt` marker written only after a successful build. A marked flat developer directory also works. Missing accuracy variants fall back to translated PTX and are logged; the bridge never substitutes fast native binaries. Restart the game when changing the setting. The mode uses a separate `d4r-accuracy` JIT cache under the configured cache directory so older runtimes cannot reuse experimental fast-math results. The first accuracy run may compile new kernels.

This mode aims to preserve NVIDIA's arithmetic, not to promise bit-identical RTX output: matrix accumulation order differs across hardware, and the shim still canonicalizes input formats and does not forward the optional transparency/current-color-bias masks. RTX parity requires identical captured inputs, library version, preset and temporal sequence on NVIDIA hardware.

## Building

```sh
kernels/build.sh k          # DLSS 4 layers   (ROCm clang only)
kernels/build.sh m          # DLSS 4.5 layers
kernels/build.sh l          # L shared layers + unfolded textures (needs DLL and D4R_ZLUDA_EMIT)
kernels/build.sh tex        # texture kernels (needs D4R_DLSS_DLL and D4R_ZLUDA_BUILD)
kernels/build.sh all DIR    # everything into DIR (default kernels/out/native)
D4R_PREFER_ACCURACY=1 kernels/build.sh all DIR/accuracy  # conservative versions of every family
```

`D4R_GPU_ARCH` selects one target for this standalone build (default gfx1101). The release script builds the network layers for gfx1100–gfx1103 and gfx1200–gfx1201 by default and puts each set in its own directory. The network kernels need gfx11 or gfx12 WMMA. Texture-kernel builds use `D4R_ZLUDA_EMIT` to generate code objects for the selected target without that GPU. Set `D4R_NATIVE_FP8=1` on gfx12 for the matching FP8 M and texture variants.

The release includes both fast and accuracy network sets. Full releases also include accuracy texture sets: supply them in `D4R_BUNDLE_TEX/accuracy/<target>` or set `D4R_ZLUDA_EMIT` so packaging builds them. Build accuracy sets into a separate directory; the builder refuses to mark a directory containing unmarked older binaries as accurate.

## Validating a kernel

1. **Capture launches.** Set `D4R_CUDA_REPLAY_DUMP_DIR` (with `D4R_CUDA_REPLAY_DUMP_FILTER=<kernel name>`) while running the D3D12 harness or a game. The bridge then writes each matching launch as `manifest.txt`, `args.bin` and one `alloc-N.bin` per buffer.
2. **Run the native kernel on a capture.** Build `kernels/tools/dump_runner.cpp` with hipcc. Then run:

   ```sh
   PREP_GRID=<d4r_prep_blocks> dump_runner NAME.hsaco NAME <capture> <out> 30
   ```

   It runs the prep and main kernels, saves every buffer after the first launch, and times 30 launches. The replay tool refuses captures with texture or surface objects; use the full harness for those kernels.
3. **A per-layer oracle for kernels with texture and surface objects.** The replay capture above cannot describe those kernels (the replayers refuse them), but the bridge's other instrumentation can: with `D4R_CUDA_LAUNCH_STATS=1` and `D4R_CUDA_LAUNCH_DUMP_DIR=<dir>` (with `D4R_CUDA_LAUNCH_STATS_FILTER=<substring>` to keep only the kernels being worked on, checked before the synchronize) every `cuLaunchKernel` is followed by a synchronize and a raw dump of every allocation, surface object and texture object named in its packed argument buffer, one file per launch and argument (`launch-NNN-<kernel>-argNNN-{buffer,surface,texture}-<shape>-fmtN.bin`). Run the harness once with the translated kernel and once with the native one and compare that kernel's files byte for byte: the surface files are the layer's own output, and a mismatch localises to that layer rather than to the frame. `cuda_dldn_engine_swin_enc0_kernel` dumps its weight image at arg040, its activation buffer at arg048 and four surfaces (1280x736x4 and 640x368x4 RGBA16F) as its outputs. The dump is large - 3 frames cost 3.7 GB - so keep the frame count small.

4. **Compare against a reference.** `kernels/tools/pwin_model.py` (K) and `swin_model.py` (M) are numpy models of the layers, written stage by stage against `ptxsim.py`, a small vectorised PTX interpreter that runs NVIDIA's kernels on the CPU. A bit-exact variant should match the model within f16 rounding; after any change, compare the new build's buffers with the previous build's.
5. **Check the whole pipeline.** Replay captured frames through the harness with both kernel sets and compare the outputs with `kernels/tools/psnr.py`. `D4R_CUDA_KERNEL_PROFILE=1` makes the bridge time every kernel; `kernels/tools/kprof.py` summarises the log per frame.

## Per-kernel cost

Median GPU time per frame at Quality (1706×960 → 2560×1440, RX 7700 XT), from `D4R_CUDA_KERNEL_PROFILE` in the D3D12 harness (flags as in Townfall; frames 21–40). The profiler's per-launch events add a little to every kernel, so totals are slightly above the in-game cost. "Before" is release 0.1.1.

| DLSS 4 (K) | before | now | DLSS 4.5 (M) | before | now |
|---|---|---|---|---|---|
| hiluma output (texture kernel) | 0.92 | 0.89 | enc3 tube (6 launches) | 1.62 | 1.57 |
| dec0 | 0.33 | 0.32 | enc1 | 1.41 | 1.27 |
| enc0 | 0.31 | 0.31 | dec1 | 1.44 | 1.43 |
| dec1 | 0.26 | 0.26 | enc2 | 0.87 | 0.77 |
| hiluma input (ZLUDA) | 0.25 | 0.24 | post (texture kernel) | 0.88 | 0.77 |
| enc1 | 0.23 | 0.22 | dec2 | 0.84 | 0.82 |
| enc2, dec2 | 0.12, 0.10 | 0.12, 0.10 | enc0 (texture kernel) | 0.64 | 0.63 |
| enc3, dec3, enc4, dec4, dec5 | 0.07–0.09 each | 0.07–0.09 each | dec0 (texture kernel) | 0.64 | 0.63 |
| NGX exposure and misc | 0.1 | 0.1 | downsample, NGX misc | 0.32 | 0.33 |
| **total** | **3.10** | **2.97** | **total** | **9.04** | **8.23** |

Runs of the same build vary by about ±3% per kernel. M additionally skips its output copy now (direct output through the downsample kernel), which is not a kernel and not in the table. Games whose flags differ from Townfall's (motion vectors at render resolution, LDR input, regular depth) gain more: those kernel variants used to run as plain ZLUDA compiles (K 3.6–4.6 ms → 2.8–3.0 ms, M −0.5 to −0.7 ms per frame).

## Experimental thin-feature motion coverage

`[Interop] MotionVectorDilation = 1` or `2` (`D4R_MOTION_DILATION`) extends nearby foreground motion into pixels whose depth otherwise belongs to the background. The value selects the opaque-surface radius in render pixels. Both enabled values also fill sky gaps using the nearest foreground pixel within an eight-render-pixel circle (far depth 0 for reversed Z, 1 for normal Z). This is the wider sky fill from the comparison; it can strengthen trails in some frames and remains opt-in through `MotionVectorDilation`. This is a depth-guided approximation to better thin-feature motion coverage, not engine-side conservative rasterization. It improved the captured Townfall rooftop where the earlier native-encoder fallback did not resolve the artifact.

The d4r-owned GPU kernel preserves the original depth/motion buffers and writes a separate RG16F velocity texture. The worker waits for its completion before NGX evaluates. No per-frame CPU readback is needed. It requires same-frame linear VRAM inputs, full-resolution unjittered motion vectors and zero subrect bases. Unsupported features or unavailable kernels keep the original vectors. The default is `0` (off); larger radii can affect disocclusion and unrelated foreground edges. The method does not recreate geometry missing from every input frame or establish RTX parity.

`scripts/build_wine_nvcuda_bridge.sh` embeds the support kernels when `D4R_ROCM_DIR` points to ROCm clang. All six gfx110x/gfx120x targets compile; physical validation is currently gfx1101 only. A bridge built without the support kernels reports the feature unavailable rather than changing inputs. Optional GPU regressions cover forward/reversed depth, two radii, jitter, padded rows and preservation of input buffers; see `tests/test_motion_kernel_gpu.py`.
