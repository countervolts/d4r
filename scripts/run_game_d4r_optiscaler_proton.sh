#!/usr/bin/env bash
set -euo pipefail

# Launches a Windows game through a GE-Proton build's integrated OptiScaler
# (PROTON_USE_OPTISCALER) with OptiScaler's DLSS backend routed into the d4r
# shim, i.e. official NVIDIA DLSS executed through the Wine CUDA bridge and
# ZLUDA on the AMD GPU.
#
# usage: run_game_d4r_optiscaler_proton.sh PROTON_DIR COMPAT_DATA_DIR GAME_EXE [ARGS...]
#
# The Proton installation and game directory are not modified. The prefix's
# system32/umu/OptiScaler.ini (rewritten by GE-Proton from
# PROTON_OPTISCALER_CONFIG) is restored when the game exits, and the real
# DXVK-NVAPI is placed next to the identity bridge as d4r_nvapi64_real.dll.
# With D4R_VKD3D_DIR (a directory holding d3d12.dll and d3d12core.dll, e.g. the
# d4r-patched vkd3d-proton), those replace Proton's vkd3d-proton in the prefix
# for this launch; the prefix's previous copies are backed up to the log
# directory and restored when the game exits.
# With D4R_UE_ENGINE_INI (the game's Saved/Config/Windows/Engine.ini) and
# D4R_UE_CVARS ("r.A=0;r.B=1"), those console variables are added to its
# [SystemSettings] for this launch only: an existing file is backed up to the
# log directory and restored on exit, a new one is removed again. Unreal games
# apply tonemapper sharpening and chromatic fringe after the upscaler; e.g.
# Townfall's r.Tonemapper.Sharpen and r.SceneColorFringeQuality turn DLSS's
# clean thin lines into haloed, stippled and beaded ones.
# Logs go to D4R_GAME_LOG_DIR (default: a timestamped capture directory).
if [[ $# -lt 3 ]]; then
  printf 'usage: %s PROTON_DIR COMPAT_DATA_DIR GAME_EXE [ARGS...]\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTON_DIR="$(realpath "$1")"
COMPAT_DIR="$(realpath "$2")"
GAME_EXE="$(realpath "$3")"
shift 3
# User settings from d4r.ini (config/d4r.ini.default documents them); explicit environment wins.
# D4R_NO_CONFIG=1 ignores the file (automated tests); D4R_CONFIG picks another file.
if [[ "${D4R_NO_CONFIG:-0}" != 1 ]]; then
  python3 "$ROOT/scripts/d4r_config.py" --print >&2 || true
  D4R_CONFIG_EXPORTS="$(python3 "$ROOT/scripts/d4r_config.py")" || exit 2
  eval "$D4R_CONFIG_EXPORTS"
fi
source "$ROOT/scripts/d4r_proton_env.sh"
# [Engine] ModelDir = auto: the model scripts/install_d4r_runtime.sh staged
export D4R_ENGINE_MODEL_DIR="${D4R_ENGINE_MODEL_DIR:-$D4R_RUNTIME_DIR/engine/k}"

PREFIX="$COMPAT_DIR/pfx"
UMU="$PREFIX/drive_c/windows/system32/umu"
LOG_DIR="${D4R_GAME_LOG_DIR:-$HOME/.cache/d4r-dlss-captures/game-$(basename "$GAME_EXE" .exe)-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"
[[ -d "$PREFIX" ]] || { printf 'No Proton prefix at %s\n' "$PREFIX" >&2; exit 2; }
# A second launch would join the running wineserver, and whichever run exits
# first would restore its OptiScaler.ini backup underneath the other.
SERVER_DIR="/tmp/.wine-$(id -u)/server-$(stat -c '%D' "$PREFIX" | sed 's/^0*//')-$(printf '%x' "$(stat -c '%i' "$PREFIX")")"
if [[ -S "$SERVER_DIR/socket" ]] && fuser "$SERVER_DIR/socket" >/dev/null 2>&1; then
  printf 'A wineserver is already running for %s; close that game first\n' "$PREFIX" >&2
  exit 2
fi

"$ROOT/scripts/build_d4r_nvapi_identity.sh" >/dev/null
NVAPI_REAL="$PROTON_DIR/files/lib/wine/nvapi/x86_64-windows/nvapi64.dll"
[[ -f "$NVAPI_REAL" ]] || { printf 'Cannot find DXVK-NVAPI at %s\n' "$NVAPI_REAL" >&2; exit 2; }
cp -f "$NVAPI_REAL" "$PREFIX/drive_c/windows/system32/d4r_nvapi64_real.dll"

# Temporary launcher: identical to the Proton build except that the NVAPI copy
# installs the identity bridge (so OptiScaler enables its DLSS backend).
LAUNCHER_DIR="$(mktemp -d "${TMPDIR:-/tmp}/d4r-game-proton.XXXXXXXX")"
for entry in "$PROTON_DIR"/*; do
  [[ "$(basename "$entry")" == proton ]] || ln -s "$entry" "$LAUNCHER_DIR/$(basename "$entry")"
done
cp -f "$PROTON_DIR/proton" "$LAUNCHER_DIR/proton"
LAUNCHER="$LAUNCHER_DIR/proton" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["LAUNCHER"])
source = path.read_text()
variants = [
    'g_proton.arch_pe_dir("wine/nvapi", False) + "nvapi64.dll"',
    'g_proton.arch_pe_dir("wine/" + nvapi_path, False) + "nvapi64.dll"',
]
for old in variants:
    target = f'try_copy({old}, "drive_c/windows/system32",'
    if source.count(target) == 1:
        new = f'try_copy(os.environ.get("D4R_NVAPI_IDENTITY_DLL", {old}), "drive_c/windows/system32",'
        path.write_text(source.replace(target, new))
        break
else:
    raise SystemExit("Proton NVAPI copy point not found; cannot stage identity bridge")
if os.environ.get("D4R_VKD3D_DIR"):
    source = path.read_text()
    old = 'try_copy(g_proton.arch_pe_dir("wine/vkd3d-proton", False) + f + ".dll", "drive_c/windows/system32",'
    new = ('try_copy(os.path.join(os.environ["D4R_VKD3D_DIR"], f + ".dll"), "drive_c/windows/system32",')
    if source.count(old) != 1:
        raise SystemExit("Proton vkd3d-proton copy point not found; cannot stage D4R_VKD3D_DIR")
    path.write_text(source.replace(old, new))
PY

# GE-Proton rewrites OptiScaler.ini from PROTON_OPTISCALER_CONFIG; keep the
# user's configuration and put it back afterwards.
INI="$UMU/OptiScaler.ini"
[[ -f "$INI" ]] && cp -f "$INI" "$LOG_DIR/OptiScaler.ini.before"
VKD3D_BACKUP="$LOG_DIR/vkd3d-prefix-backup"
if [[ -n "${D4R_VKD3D_DIR:-}" ]]; then
  export D4R_VKD3D_DIR="$(realpath "$D4R_VKD3D_DIR")"
  for dll in d3d12.dll d3d12core.dll; do
    [[ -f "$D4R_VKD3D_DIR/$dll" ]] || { printf 'D4R_VKD3D_DIR lacks %s\n' "$dll" >&2; exit 2; }
  done
  mkdir -p "$VKD3D_BACKUP"
  for dll in d3d12.dll d3d12core.dll; do
    [[ -f "$PREFIX/drive_c/windows/system32/$dll" ]] && cp -f "$PREFIX/drive_c/windows/system32/$dll" "$VKD3D_BACKUP/$dll"
  done
fi
ENGINE_INI="${D4R_UE_ENGINE_INI:-}"
ENGINE_INI_STATE=""
restore_prefix() {
  [[ -f "$LOG_DIR/OptiScaler.ini.before" ]] && cp -f "$LOG_DIR/OptiScaler.ini.before" "$INI"
  case "$ENGINE_INI_STATE" in
    restore) cp -f "$LOG_DIR/Engine.ini.before" "$ENGINE_INI" ;;
    remove) rm -f "$ENGINE_INI" ;;
  esac
  if [[ -n "${D4R_VKD3D_DIR:-}" ]]; then
    for dll in d3d12.dll d3d12core.dll; do
      [[ -f "$VKD3D_BACKUP/$dll" ]] && cp -f "$VKD3D_BACKUP/$dll" "$PREFIX/drive_c/windows/system32/$dll"
    done
  fi
  rm -rf "$LAUNCHER_DIR"
}
trap restore_prefix EXIT
if [[ -n "$ENGINE_INI" && -n "${D4R_UE_CVARS:-}" ]]; then
  if [[ -f "$ENGINE_INI" ]]; then
    cp -f "$ENGINE_INI" "$LOG_DIR/Engine.ini.before"
    ENGINE_INI_STATE=restore
  else
    mkdir -p "$(dirname "$ENGINE_INI")"
    ENGINE_INI_STATE=remove
  fi
  { printf '\n[SystemSettings]\n'; tr ';' '\n' <<<"$D4R_UE_CVARS"; printf '\n'; } >> "$ENGINE_INI"
fi

OPTI_CONFIG="Upscalers.Dx12Upscaler=dlss;DLSS.Enabled=true"
OPTI_CONFIG+=";Libraries.NvngxPath=$(d4r_winpath "$D4R_RUNTIME_DIR/bin/d4r_nvngx.dll")"
OPTI_CONFIG+=";Libraries.NvngxDlssPath=$(d4r_winpath "$D4R_RUNTIME_DIR/bin")"
# Game-side DLSS inputs stay hooked; only NGX loads outside the game directory
# are redirected, which lets the shim request the official core by an
# exe-relative path.
OPTI_CONFIG+=";Inputs.EnableDlssInputs=true;Hooks.HookOriginalNvngxOnly=true"
OPTI_CONFIG+=";NvApi.OverrideNvapiDll=false;FrameGen.Enabled=false;FrameGen.FGInput=nofg;FrameGen.FGOutput=nofg"
OPTI_CONFIG+=";Log.LogToFile=true;Log.LogLevel=${D4R_OPTISCALER_LOG_LEVEL:-2}"
# Detailed FPS overlay with frame-time graph (the user's own INI default too).
OPTI_CONFIG+=";Menu.ShowFps=${D4R_OPTISCALER_SHOW_FPS:-true};Menu.FpsOverlayType=${D4R_OPTISCALER_FPS_OVERLAY_TYPE:-3}"
OPTI_CONFIG+="${D4R_EXTRA_OPTISCALER_CONFIG:+;$D4R_EXTRA_OPTISCALER_CONFIG}"

export PROTON_USE_OPTISCALER=1
export PROTON_OPTISCALER_CONFIG="$OPTI_CONFIG"
export D4R_NVAPI_IDENTITY_DLL="$ROOT/build/nvapi64.dll"
export DXVK_CONFIG="${DXVK_CONFIG:-dxgi.customVendorId = 10de}"
export D4R_SHIM_LOG="$(d4r_winpath "$LOG_DIR/d4r_nvngx.log")"
export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="$COMPAT_DIR"
export PROTON_LOG=1 PROTON_LOG_DIR="$LOG_DIR"
export WINEDEBUG="${WINEDEBUG:--all}"
# Games load dozens of modules per launch; do not capture them unless asked.
export D4R_CUDA_CAPTURE="${D4R_CUDA_CAPTURE:-0}"

printf 'd4r game launch: %s\nlogs: %s\n' "$GAME_EXE" "$LOG_DIR"
status=0
(cd "$(dirname "$GAME_EXE")" && "$LAUNCHER_DIR/proton" waitforexitandrun "$GAME_EXE" "$@") \
  >"$LOG_DIR/launcher.log" 2>&1 || status=$?
cp -f "$UMU/OptiScaler.log" "$LOG_DIR/OptiScaler.log" 2>/dev/null || true
printf 'game exited with status %s\n' "$status"
exit "$status"
