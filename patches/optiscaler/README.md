# Explicit Windows d4r backend

Base: OptiScaler `45a2001303ddff632e279f77aef85ceede5832cb` (2026-09-30).

`0004` serializes D3D12 device hook installation/removal and guards recursive
device discovery while the d4r command backend installs. The previous frontend
can start two installations, fail a Detours transaction with `0x10dd`, clear
live original-function pointers and call null. This was reproduced in the
numeric-ID standalone harness; no game was required.

`0005` propagates numeric Init/Init_Ext failures for the explicitly selected
external backend, matching the existing ProjectID behavior. FeatureNotFound
must fail initialization instead of being hidden until repeated CreateFeature
attempts. `Libraries.NvngxDlssPath` takes a directory, not the DLL filename;
the Windows installer and standalone runner now use that directory.
The modifications to OptiScaler are GPL-3.0, matching its upstream license.
No NVIDIA DLL is supplied or patched.

`0001` introduces `[DLSS] AllowExternalBackend=true`, default false. Set
`[Libraries] NvngxPath` to the absolute directory containing the Windows d4r
shim as `_nvngx.dll`. Discovery requires its GPU-free ABI-1 marker. Physical
adapter support is queried through the bridge's D3D12 requirements API after
DLL startup; vendor, LUID and HIP architecture are retained.

The configured `D4R_NGX_CORE` absolute path bypasses game-facing NGX DLL
redirection. The external backend loads its own NVAPI provider. An explicitly
selected external DLSS backend reports initialization/load errors instead of
silently switching to FSR. The patch fixes the upstream loader's filesystem
exception by making IsSubpath a lexical check, and retains a PDB for diagnosis.

`0002` installs the command backend at D3D12 device discovery, before NGX Init
and before OptiScaler's own method hooks. Command signatures created during
game startup therefore retain their public argument descriptions for replay.
The standalone runner's `-EarlyIndirectProbe` exercises this startup order.

`0003` preserves the Project ID and engine identity through OptiScaler's
older `Init_with_ProjectID` entry point used by its XeSS/FSR inputs. It also
propagates an explicit external backend initialization failure. Reproduce
with the D3D12 runner's `-NgxAbi project-legacy`.

Build with Microsoft Build Tools 2022, v143 and Windows SDK 10.0.26100:

```powershell
.\scripts\windows\build-optiscaler-windows.ps1
```

The script clones the pinned source if absent, checks/applies the patch,
initializes pinned submodules, invokes MSBuild and writes the DLL, PDB, license
and build metadata into `dist/optiscaler-windows-d4r`. Existing source is not
reset. The upstream packaging post-build event is disabled for this build.
The official bootstrapper is https://aka.ms/vs/17/release/vs_buildtools.exe;
install the `Microsoft.VisualStudio.Workload.VCTools`,
`Microsoft.VisualStudio.Component.VC.Tools.x86.x64` and
`Microsoft.VisualStudio.Component.Windows11SDK.26100` components. The audited
local build uses MSVC 14.44.35207 / Build Tools 17.14.

For standalone verification add
`-OptiScalerDll <absolute-dist-path>/OptiScaler.dll -CaptureExceptions`
to the documented K/M runner. Its private INI selects DLSS, disables frame
generation and keeps RestoreComputeSignature, RestoreGraphicSignature and
ExtendedStateRestore false. Use `-RequireNativeNetwork`; finite output alone
cannot distinguish DLSS from an unwanted FSR replacement.
