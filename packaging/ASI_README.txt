d4r ASI add-on for Linux / Proton
================================

Optional FG/latency: d4r/d4r-features.py configures OptiScaler with an exact INI
backup. Pass the game folder and --runtime /path/to/plugins/d4r. Copy the matching
OptiScaler release's fakenvapi.dll, libxell.dll, libxess_fg.dll and (for FSR FG)
amd_fidelityfx_framegeneration_dx12.dll into that d4r folder, retaining licenses.
Example: python3 plugins/d4r/d4r-features.py . --runtime plugins/d4r --fg xefg --input dlssg
Restart, enable native DLSS FG in-game and activate FG in OptiScaler. For a game
without native FG use --input upscaler. Latency-only: --fg off --latency auto,
then enable Reflex in-game. AL2 requires host/vkd3d support; XeLL/LFX are fallbacks.
Disable: --fg off --latency proton. Setup preserves these choices.

1. Drop d4r.asi and the d4r folder into OptiScaler's existing plugins folder.
   For example: <game>/plugins/d4r.asi and <game>/plugins/d4r/.
   Custom plugin folders work too: keep the ASI and its companion folder together.
2. Run bash install.sh inside that d4r folder once to download and verify the
   official NVIDIA DLSS/NGX DLLs. No Python or configuration command is needed.
3. Launch the game. At the first setup message, close the game. Setup installs
   the bundled D3D12 wrapper and private core and updates OptiScaler's settings after the game exits,
   with backups. Wait for the ready message, then launch the game again.

Requires OptiScaler 0.9.4 through a game-local proxy (dxgi.dll, winmm.dll, etc.)
or GE-Proton's built-in loader, with [Plugins] LoadAsiPlugins=true already enabled. A plugin cannot enable
its own loader before it has been loaded. OptiScaler's nvngx.dll mode does not
load ASIs. OptiScaler installed as d3d12.dll conflicts with the patched D3D12 pair;
use another supported proxy name first. GE-Proton's built-in OptiScaler also
works with its prefix-local plugins folder. Setup edits OptiScaler.ini beside
the active loader, including windows/system32/umu/OptiScaler.ini. Explicit
PROTON_OPTISCALER_CONFIG values are applied again by Proton at launch; keep
those consistent with d4r's route if you use that environment option.

The runtime, kernels, NVIDIA files, logs and d4r.ini stay inside plugins/d4r/.
The patched d3d12.dll and d4r12core.dll are kept there as an installation payload;
the helper installs d3d12.dll and d4r12core.dll beside the main game EXE.
The game's original D3D12Core.dll remains untouched. d4r's forwarder uses the
private core name to avoid Agility SDK loader interception.
Setup uses OptiScaler's regular Windows library hooks and disables its Agility
SDK upgrade so the supplied D3D12 implementation stays selected.
It waits for the running game to exit before changing either DLL or the INI.
Changed files get unique .d4r-backup.* copies. Unrelated settings, your existing
OptiScaler proxy and other plugins are preserved. Matching files are skipped on
later launches. Background utilities such as xalia.exe, launchers and crash
reporters do not run setup; one helper per plugin installation owns the setup. The helper never changes the Proton installation or system packages.

install.sh requires Bash, curl, sha256sum and 7zz, 7z or 7za; no sudo is needed.
The NVIDIA driver archive (~915 MiB on first use) is extracted without being
executed, and temporary downloads are removed afterward. For a nonvidia archive,
supply the NVIDIA DLLs inside this d4r folder before launching instead; full
archives already include them. Internet access and NVIDIA's license terms apply:
  https://github.com/NVIDIA/DLSS/blob/v310.7.0/LICENSE.txt
  https://www.nvidia.com/en-us/drivers/nvidia-license/

Use GE-Proton 11 with these Steam launch options:
  PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %command%
Keep PROTON_USE_OPTISCALER for Proton's built-in OptiScaler. Remove it only when
using a separate game-local OptiScaler install.
Select DLSS in OptiScaler; choose your model in plugins/d4r/d4r.ini.

Errors and completion status are recorded in d4r/installation-status.txt.
Setup retries on the next launch if it fails. Moving the game/plugin folder
refreshes the paths automatically after another close-and-relaunch cycle.
The ASI is the NGX backend itself; NvngxPath selects it, so only one backend is
active. HookOriginalNvngxOnly=true keeps the internal NVIDIA core load separate.
CUDA/GPU initialization stays lazy until NGX Init. Installation and loader tests
do not prove successful rendering in a game; native Vulkan remains experimental.

Uninstall: close the game and setup helper, restore your OptiScaler.ini backup beside its loader and
the D3D12 backups beside the EXE (or delete this setup's D3D12 files if none existed
before), then remove only d4r.asi and this d4r folder from the plugins directory.
Keep your existing OptiScaler loader and other plugins.

d4r is GPL-3.0-only. Third-party notices and provenance are in d4r/licenses/ and
d4r/source/SOURCES.txt. NVIDIA files and derived kernels have separate terms.
https://github.com/countervolts/d4r
