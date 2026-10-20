# Frame generation and latency reduction (Linux/Proton)

d4r supplies OptiScaler's XeFG and FSR FG implementations and a latency-only
NVAPI router. These features are experimental and opt-in. A fresh full package
keeps FG off and uses Proton's original NVAPI. Neither DLSS model selection nor
RR backend selection changes when enabling them. NVIDIA's DLSS Frame Generation
network is not implemented: a game's DLSSG input can feed XeFG or FSR FG instead.

Close the game before changing its configuration. From the game's main executable
folder, enable XeFG using its native Streamline 2 DLSSG input, for example Cyberpunk:

```sh
python3 d4r/d4r-features.py . --fg xefg --input dlssg
```

For native FSR 3.1 FG use `--input fsrfg`, or `fsrfg30` for FSR 3.0.
For a game without native FG use `--input upscaler` (OptiFG); this has less reliable
HUD handling. To use FSR FG output replace `--fg xefg` with `--fg fsrfg`.
The command validates dependencies, preserves other settings and backs up the
original INI before replacing it. Errors leave the config untouched.

Restart, enable the corresponding FG option in-game (except for `upscaler`),
and activate FG in OptiScaler's Insert menu. Use borderless mode for XeFG;
it supports SDR/HDR10, not scRGB/FP16 HDR. Native Vulkan games cannot use this
D3D12 XeFG route. D3D12 games under Proton still require game testing.
Streamline 1 DLSSG inputs are not supported by this setup.

For latency conversion alone, leaving FG disabled:

```sh
python3 d4r/d4r-features.py . --fg off --latency auto
```

Enable Reflex in-game. Fakenvapi selects Anti-Lag 2 when its D3D12 extension is
available, then XeLL, then LatencyFlex. On Proton, AL2 depends on vkd3d-proton
and the host's `VK_AMD_anti_lag` support, including any enabled latency layer;
installing the DLL does not create that support. XeFG uses its XeLL context and
requires the router so the game's Reflex markers reach that context.
This is separate from d4r's `[Latency] FrameAge`, which remains unchanged.

The router sends only six D3D latency interfaces and Fakenvapi's context/FG
interfaces to `fakenvapi.dll`. GPU handles, architecture, LUID/topology, driver
information and other NVAPI interfaces stay with Proton's original provider.
Initialize/Unload are forwarded to both providers, preserving the original
provider's status. Missing latency exports fall back to Proton. The router
is selected through OptiScaler's `Libraries.NvapiPath`; no system32 DLL is replaced.
`Hooks.UseNtdllHooks=false` prevents OptiScaler from redirecting the router's
own load of the real NVAPI back to itself. Keep it false with this router.
OptiScaler 0.9.4 accidentally reads XeLLPath/XeFGPath into its XeSS override.
The helper uses directory overrides and sets OptiDllPath to the private runtime
folder, so startup discovery finds the right DLLs. Existing explicit NGX/SR library
paths remain unchanged. The router preloads its colocated XeLL DLL without changing
the process's DLL search path.

Return to the default routes:

```sh
python3 d4r/d4r-features.py . --fg off --latency proton
```

For exact rollback, copy the printed `OptiScaler.ini.d4r-features-backup-*` file
over `OptiScaler.ini` while the game is closed. The disable command leaves inactive
library paths in place; it disables their NVAPI override and FG selection.

For ASI installations, pass `--runtime /path/to/plugins/d4r` to the helper in
the actual plugin's `d4r/` folder. Copy the needed `fakenvapi.dll`, `libxell.dll`,
`libxess_fg.dll` and (for FSR FG) `amd_fidelityfx_framegeneration_dx12.dll` from
the existing matching OptiScaler release into that folder. Keep their licenses.
ASI setup preserves FG and NVAPI selections.

Enable OptiScaler file logging for verification. Check that NGX still loads d4r's
shim, the router loads through `NvapiPath`, and the chosen FG DLL/context initializes.
Check the reported latency method rather than assuming AL2 was selected. Enable
the FG debug view and inspect generated frames; an FPS overlay alone is not proof
of interpolation or lower latency. These checks need an actual game session;
package/ABI tests cannot establish image quality or latency.

Upstream references:

- [OptiScaler FG setup and limitations](https://github.com/optiscaler/OptiScaler/wiki/Frame-Generation-Options)
- [OptiScaler v0.9.4 configuration](https://github.com/optiscaler/OptiScaler/blob/v0.9.4/OptiScaler.ini)
- [Proton Reflex/AL2 layer setup](https://github.com/Korthos-Software/low_latency_layer#usage-and-configuration)
