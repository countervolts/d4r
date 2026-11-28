# Testing

## Offline, no game (what was actually run)
```sh
scripts/build-native.sh
python3 -I tools/mkcapture.py CAPDIR --w 2560 --h 1440 --frames 8 --mfc 3       # synthetic Minecraft-convention capture (RGBA8, D32 reversed-Z, RG16F pixel MV)
export VK_ADD_LAYER_PATH=$PWD/build/d4r/layer VK_INSTANCE_LAYERS=VK_LAYER_NVXRT_nvx NVXRT_ARCH=gfx11 NVXRT_CACHE=$HOME/.cache/nvxrt-gfx11
build/native/fg_replay_test ~/.cache/d4r-mc-fg/2560x1440/mfc3 $NVXRT_CACHE CAPDIR OUTDIR 8     # writes rRR_gK.ppm + GPU ms per real frame
```
`fg_replay_test` also reads d4r's own harness captures (`~/.cache/dlssg-capture-got`). **Bit-exactness check:** run the real DLL on the same capture
(`fg/dlssg/harness/run.sh DLL OUT --mfc 3 --replay CAPDIR`, exact mode) and compare the PPMs byte for byte (done: 0 differing bytes in 8 frames x 3 outputs, both captures).
Fast mode: `NVXRT_TEX=hw NVXRT_ACC=fast NVXRT_SHADOW=b,c,r` (not bit-exact, d4r's default).

## Recording / tooling
- `tools/mkscript.py`, `scripts/build-fg-scripts.sh`: script generation from the real DLL (needs Proton + the patched vkd3d-proton + the traced layer, `patches/nvxrt-resource-trace.patch`).
- `tools/param_diff.py`: which parameter bytes of each kernel vary between launches.
- Findings the design rests on: only handles/pointers vary (ping-pong, period 2) and `ClipToPrevClip` (+80 of `Kernel_EstimatePrev2CurrScatter`); camera pos/FOV/near/far do not matter for Lite.

## In game (not yet run)
`./gradlew runClient` with `config/d4r-minecraft.properties` `overlay=true`; check the log for `[D4R] frame generation initialised`, `d4r-nvxrt.log` for layer messages, and the overlay for shown vs real fps.
