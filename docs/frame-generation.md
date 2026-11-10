# Frame generation and latency reduction (Linux/Proton)

d4r's full proxy package includes OptiScaler **10.0.0-pre1**, pinned to
[nightly-20261024](https://github.com/optiscaler/OptiScaler-nightly/releases/tag/nightly-20261024).
This is a prerelease. Its complete `OptiScaler/` dependency folder ships with the
package, including XeFG, XeLL, AMD FSR frame generation and Nukem's
`dlssg_to_fsr3_amd_is_better.dll`. Upstream license files are preserved in
`d4r/licenses/OptiScaler/`. The archive digest and source revision are pinned in
`packaging/optiscaler-release.json`; packaged provenance is in
`d4r/source/optiscaler.json`.

FG is disabled by default. Open OptiScaler's Insert menu, select the FG input
provided by the game and choose **XeFG** or **FSR FG** as output. Enable the matching
FG setting in-game. For games without native FG, use **Upscaler / OptiFG** input;
HUD handling may be less reliable. Restart when requested by OptiScaler.

Alternatively, close the game and run this from its executable folder:

```sh
python3 d4r/d4r-features.py . --fg xefg --input dlssg
```

Use `--input fsrfg` for native FSR 3.1 FG, `fsrfg30` for FSR 3.0, or `upscaler`
for OptiFG. To select FSR FG output, replace `--fg xefg` with `--fg fsrfg`.
The helper checks dependencies before changing anything, preserves SR/RR choices,
and creates an exact INI backup. It also supports older 0.9 ASI installations.

XeFG needs borderless mode and SDR/HDR10 rather than scRGB/FP16 HDR. Treat D3D12
under Proton as requiring per-game testing. d4r's native Vulkan route has not been
validated with FG. NVIDIA's own DLSS FG runs through d4r's DLSS-G backend (see below). The bundled
Nukem DLL is an alternate DLSSG replacement; v10's DLSSG-output routes additionally
require matching Streamline files, which d4r does not bundle. The direct XeFG/FSR FG
routes above are the starting point for testing.

## NVIDIA DLSS Frame Generation (RDNA3/RDNA4)

`d4r/fg/dlssg` runs the game's own `nvngx_dlssg.dll` (2x/3x/4x) next to d4r's DLSS SR/RR, ported from
[dlssg-amd](https://github.com/itchyfeetleech/dlssg-amd). With the game closed, run this from its executable folder:

```sh
python3 d4r/d4r-features.py . --fg dlssg       # --exact, --lite, --dll PATH; --fg off undoes it
```

It translates the DLL's CUDA kernels for this GPU (RDNA4: `~/.cache/nvxrt`, with hand-written network kernels;
RDNA3: `~/.cache/nvxrt-gfx11`, with the same hand-written kernels in RDNA3's layout), routes OptiScaler's NGX path through the DLSS-G host, sets
`HwSchMode=2` in the Proton prefix and `[FrameGeneration] Backend = dlssg` in `d4r.ini`. `d4r_boot.dll` enables the
Vulkan layer at launch: no launch options. Enable DLSS Frame Generation in the game. DLSS-G 310.7.0 is the tested
version; 3.x DLLs (encrypted PTX) are refused, as are newer 310.x builds the translator cannot handle yet, without
changing anything (`--dll` can point at a 310.7.0 copy). Cost at 2560x1440: RDNA4 2.2 ms per real frame for 2x
(upstream, RX 9070); RDNA3 5.4 ms for 2x and 11.7 ms for 4x (RX 7700 XT, harness; Ghost of Tsushima 4x: 10.1 ms). Details: [fg/dlssg/README.md](../fg/dlssg/README.md).

## Latency and GPU identity

v10 integrates Fakenvapi and hooks latency interfaces on the system NVAPI. Keep
`[fakenvapi] UseFakenvapi=false`: Proton must supply GPU identity, architecture and
topology for NGX. A separate `fakenvapi.dll` and d4r NVAPI router are unnecessary
with v10. OptiScaler selects its latency output automatically when it receives
supported latency markers. Enable Reflex in-game and check the active output in
OptiScaler. Host/vkd3d support is still needed for Anti-Lag 2. XeFG owns its XeLL
context. These settings do not change d4r's `[Latency] FrameAge`.

To explicitly restore automatic latency selection with FG disabled:

```sh
python3 d4r/d4r-features.py . --fg off --latency auto
```

To disable FG and request Reflex output through Proton:

```sh
python3 d4r/d4r-features.py . --fg off --latency proton
```

In v10, `--latency proton` selects `LowLatencyOutput=6` (Reflex); OptiScaler may
fall back if that output is unavailable. It does not remove OptiScaler's hooks.
For exact rollback, restore the printed `OptiScaler.ini.d4r-features-backup-*` file
while the game is closed.

## Existing ASI installations

ASI add-ons keep the existing OptiScaler version, dependencies and FG choices.
Use **Save INI** normally. Later setup runs and runtime updates leave saved
upscaler, FG, spoofing and overlay preferences untouched. Moving the plugin
refreshes only paths that still point to its previous location.
For v10, retain its complete dependency folder and `Libraries.OptiDllPath`; the
helper uses that directory even if d4r itself is in a custom plugin folder.
For OptiScaler 0.9, pass `--runtime /path/to/plugins/d4r` and place matching
`fakenvapi.dll`, `libxell.dll`, `libxess_fg.dll` and (for FSR FG)
`amd_fidelityfx_framegeneration_dx12.dll` in that runtime folder with their licenses.
The legacy helper uses d4r's latency-only NVAPI router and directory overrides to
work around 0.9's XeLL/XeFG path parsing. Keep `Hooks.UseNtdllHooks=false` with it.

## Testing

Enable OptiScaler file logging. First verify that DLSS/RR still loads d4r's shim,
then select FG and confirm that its provider/context initializes. Use the FG debug
view to inspect interpolated frames, HUD handling and frame pacing. Check the
reported latency method. An FPS overlay alone does not prove interpolation or
reduced latency. Package tests establish staging and configuration only; game
image quality, timing and compatibility require actual game sessions.

Upstream references:

- [OptiScaler v10 changes](https://github.com/optiscaler/OptiScaler/releases/tag/nightly)
- [OptiScaler FG setup](https://github.com/optiscaler/OptiScaler/wiki/Frame-Generation-Options)
- [Pinned source configuration](https://github.com/optiscaler/OptiScaler/blob/fd3a45a6cac6d610a35c570479f80dae95ca4eac/OptiScaler.ini)
