#!/usr/bin/env bash
set -euo pipefail
umask 077

# Runs the synthetic D3D12 harness through a locally supplied OptiScaler
# dxgi.dll and the d4r shim. All proxy files and the NVAPI identity bridge are
# staged in disposable /tmp paths; the Proton installation is not modified.
# usage: run_d3d12_dlss_harness_optiscaler_proton.sh PROTON_DIR OPTISCALER_DLL OUTPUT_RAW [FRAMES [IN_W IN_H OUT_W OUT_H]]
if [[ $# -lt 3 ]]; then
  printf 'usage: %s PROTON_DIR OPTISCALER_DLL OUTPUT_RAW [FRAMES [IN_W IN_H OUT_W OUT_H]]\n' "$0" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTON_DIR="$(realpath "$1")"
OPTISCALER_DLL="$(realpath "$2")"
OUTPUT_PATH="$(realpath -m "$3")"
shift 3
source "$ROOT/scripts/d4r_proton_env.sh"

for path in "$D4R_RUNTIME_DIR/bin/d4r_nvngx.dll" "$D4R_RUNTIME_DIR/bin/nvcuda.dll" \
            "$D4R_RUNTIME_DIR/bin/nvngx_dlss.dll" "$D4R_RUNTIME_DIR/ngx/_nvngx.dll"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing staged d4r runtime file: %s\n' "$path" >&2
    exit 2
  fi
done

# DLSS Enabler has been found installed under the name OptiScaler.dll on this
# machine. Reject it here: its proxy rejects dxgi/NGX harness naming and can
# open a modal error dialog during LoadLibrary.
EXPORT_HEADER="$(x86_64-w64-mingw32-objdump -p "$OPTISCALER_DLL" | sed -n '/The Export Tables/,+6p')"
if [[ "$EXPORT_HEADER" != *OptiScaler.dll* ]]; then
  printf 'Expected a real OptiScaler DLL (PE export name OptiScaler.dll): %s\n' "$OPTISCALER_DLL" >&2
  exit 2
fi

"$ROOT/scripts/build_d4r_nvapi_identity.sh" >/dev/null
APP_DIR="${D4R_OPTISCALER_APP_DIR:-${TMPDIR:-/tmp}/d4r-optiscaler-harness-${UID}-$(date +%s%N)}"
mkdir -p "$APP_DIR"
LAUNCHER_DIR="$(mktemp -d "${TMPDIR:-/tmp}/d4r-optiscaler-proton.XXXXXXXX")"
COMPAT_DIR="${D4R_OPTISCALER_COMPAT_DIR:-${TMPDIR:-/tmp}/d4r-optiscaler-compat-$UID}"
mkdir -p "$COMPAT_DIR"

cp -f "$OPTISCALER_DLL" "$APP_DIR/dxgi.dll"
INPUT_HOOK="${D4R_OPTISCALER_ENABLE_DLSS_INPUTS:-false}"
ORIGINAL_HOOK="${D4R_OPTISCALER_HOOK_ORIGINAL_NVNGX_ONLY:-$INPUT_HOOK}"
if [[ "$INPUT_HOOK" != true && "$INPUT_HOOK" != false ]] ||
   [[ "$ORIGINAL_HOOK" != true && "$ORIGINAL_HOOK" != false ]]; then
  printf 'OptiScaler hook settings must be true or false\n' >&2
  exit 2
fi
# D4R_OPTISCALER_STAGE_CORE=0 keeps the official core in the runtime directory,
# as in a game install; the shim then loads it below OptiScaler's hooks.
STAGE_CORE="${D4R_OPTISCALER_STAGE_CORE:-1}"
if [[ "$INPUT_HOOK" == true && "$STAGE_CORE" == 1 ]]; then
  if [[ "$ORIGINAL_HOOK" != true || "$APP_DIR" != "${APP_DIR,,}" ]]; then
    printf 'The DLSS input hook requires HookOriginalNvngxOnly=true and a lowercase app path\n' >&2
    exit 2
  fi
  # OptiScaler hooks game-side nvngx loads. Keep the official core's required
  # filename inside the harness EXE directory and pass a lowercase z: path;
  # its hook then leaves this internal core load alone.
  cp -f "$D4R_RUNTIME_DIR/ngx/_nvngx.dll" "$APP_DIR/_nvngx.dll"
  cp -f "$D4R_RUNTIME_DIR/dlss/nvngx_dlss.dll" "$APP_DIR/nvngx_dlss.dll"
  core_windows_path="$(d4r_winpath "$APP_DIR/_nvngx.dll")"
  export D4R_NGX_CORE="z:${core_windows_path#Z:}"
fi
cat > "$APP_DIR/OptiScaler.ini" <<EOF
[Upscalers]
Dx12Upscaler=dlss
[FrameGen]
Enabled=false
FGInput=nofg
FGOutput=nofg
[DLSS]
Enabled=true
[Inputs]
EnableDlssInputs=$INPUT_HOOK
[Hooks]
HookOriginalNvngxOnly=$ORIGINAL_HOOK
[FSR]
FsrAgilitySDKUpgrade=false
[Libraries]
NvngxPath=$(d4r_winpath "$D4R_RUNTIME_DIR/bin/d4r_nvngx.dll")
NvngxDlssPath=$(d4r_winpath "$D4R_RUNTIME_DIR/bin")
[NvApi]
OverrideNvapiDll=false
[Spoofing]
Dxgi=false
[Plugins]
LoadAsiPlugins=false
[Log]
LogToFile=true
LogLevel=1
EOF

# Proton refreshes nvapi64.dll in the prefix on every run. Make a temporary
# launcher that copies our identity bridge instead. Its other files are links
# to the unmodified Proton installation.
cp -f "$PROTON_DIR/proton" "$PROTON_DIR/filelock.py" "$PROTON_DIR/version" "$LAUNCHER_DIR/"
ln -s "$PROTON_DIR/files" "$LAUNCHER_DIR/files"
ln -s "$PROTON_DIR/protonfixes" "$LAUNCHER_DIR/protonfixes"
LAUNCHER="$LAUNCHER_DIR/proton" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["LAUNCHER"])
source = path.read_text()
old = 'try_copy(g_proton.arch_pe_dir("wine/nvapi", False) + "nvapi64.dll", "drive_c/windows/system32",'
new = 'try_copy(os.environ.get("D4R_NVAPI_IDENTITY_DLL", g_proton.arch_pe_dir("wine/nvapi", False) + "nvapi64.dll"), "drive_c/windows/system32",'
if source.count(old) != 1:
    raise SystemExit("Proton NVAPI copy point changed; cannot stage identity bridge")
path.write_text(source.replace(old, new))
PY
chmod +x "$LAUNCHER_DIR/proton"

export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="$COMPAT_DIR"
PROTON_ENABLE_NVAPI=1 WINEDEBUG=-all "$PROTON_DIR/proton" run cmd /c exit 0 >/dev/null
NVAPI_REAL="$PROTON_DIR/files/lib/wine/nvapi/x86_64-windows/nvapi64.dll"
if [[ ! -f "$NVAPI_REAL" ]]; then
  printf 'Cannot find Proton DXVK-NVAPI at %s\n' "$NVAPI_REAL" >&2
  exit 2
fi
cp -f "$NVAPI_REAL" "$COMPAT_DIR/pfx/drive_c/windows/system32/d4r_nvapi64_real.dll"

export D4R_NVAPI_IDENTITY_DLL="$ROOT/build/nvapi64.dll"
export D4R_HARNESS_APP_DIR="$APP_DIR"
export D4R_HARNESS_NGX_DLL="$APP_DIR/dxgi.dll"
export D4R_HARNESS_LOAD_NGX_EARLY=1
export D4R_HARNESS_TRACE="$(d4r_winpath "$APP_DIR/harness.trace")"
export D4R_SHIM_LOG="$(d4r_winpath "$APP_DIR/d4r_nvngx.log")"
export D4R_PROTON_COMPAT_DATA="$COMPAT_DIR"
export DXVK_CONFIG="${DXVK_CONFIG:-dxgi.customVendorId = 10de}"

"$ROOT/scripts/run_d3d12_dlss_harness_proton.sh" "$LAUNCHER_DIR" "$OUTPUT_PATH" "$@"

if [[ ! -s "$OUTPUT_PATH" || ! -f "$APP_DIR/OptiScaler.log" || ! -f "$APP_DIR/d4r_nvngx.log" ]] ||
   ! rg -q 'DLSSFeatureDx12::Init _CreateFeature result: NVSDK_NGX_Result_Success' "$APP_DIR/OptiScaler.log" ||
   ! rg -q 'NVSDK_NGX_CUDA_CreateFeature -> 0x00000001' "$APP_DIR/d4r_nvngx.log" ||
   ! rg -q 'frame [0-9]+ done:' "$APP_DIR/d4r_nvngx.log"; then
  printf 'OptiScaler did not complete the official DLSS path; inspect %s\n' "$APP_DIR" >&2
  exit 1
fi

printf 'OptiScaler DLSS harness output: %s\n' "$OUTPUT_PATH"
printf 'OptiScaler, shim, and stage logs: %s\n' "$APP_DIR"
sha256sum "$OUTPUT_PATH"
