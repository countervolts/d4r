#!/usr/bin/env bash
# Self-test outside any game: runs d3d12_dlssg.exe (a synthetic scene through NVIDIA's DLSS-G DLL via the NGX host)
# under Proton in a private prefix, with the patched vkd3d-proton, dxvk-nvapi and the nvxrt layer.
# usage: run.sh DLSSG_DLL OUTDIR [harness options, e.g. --mfc 3 --frames 6 --bench]
# Env: PROTON (proton script; default: newest GE-Proton in compatibilitytools.d), VKD3D_DIR (patched d3d12.dll +
#      d3d12core.dll; required), NVXRT_CACHE (default ~/.cache/nvxrt), ARCH (default 0x1b0), HOST_INI_EXTRA,
#      plus any NVXRT_* option (NVXRT_TEX=hw, NVXRT_ACC=fast, ...).
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
DLL=$(readlink -f "${1:?usage: run.sh DLSSG_DLL OUTDIR [options]}")
OUT=$(mkdir -p "${2:?usage: run.sh DLSSG_DLL OUTDIR [options]}" && cd "$2" && pwd); shift 2
: "${VKD3D_DIR:?set VKD3D_DIR to the patched vkd3d-proton build (vkd3d-proton/build.sh)}"
STEAM=${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}
PROTON=${PROTON:-$(ls -d "$STEAM"/compatibilitytools.d/GE-Proton*/proton 2>/dev/null | sort -V | tail -1)}
[[ -x "$PROTON" ]] || { echo "run.sh: no Proton found; set PROTON=/path/to/proton" >&2; exit 1; }
winpath() { printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"; }

WORK=${HARNESS_WORK:-$HOME/.cache/dlssg-amd-harness}
mkdir -p "$WORK/compat" "$WORK/bin"
export STEAM_COMPAT_CLIENT_INSTALL_PATH=$STEAM STEAM_COMPAT_DATA_PATH=$WORK/compat
cp -f "$HERE/d3d12_dlssg.exe" "$ROOT/host/build/nvngx.dll_dlssg_host.dll" "$VKD3D_DIR/d3d12.dll" \
  "$VKD3D_DIR/d3d12core.dll" "$WORK/bin/"
export WINEDLLOVERRIDES="d3d12,d3d12core=n,b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export PROTON_FORCE_NVAPI=1 PROTON_ENABLE_NVAPI=1 DXVK_NVAPI_ALLOW_OTHER_DRIVERS=1 DXVK_NVAPI_GPU_ARCH=${DXVK_NVAPI_GPU_ARCH:-AD100}
export VK_ADD_LAYER_PATH=${NVXRT_LAYER_DIR:-$ROOT/nvxrt/build}
export VK_INSTANCE_LAYERS=VK_LAYER_NVXRT_nvx
export NVXRT_CACHE=${NVXRT_CACHE:-$HOME/.cache/nvxrt}
export NVXRT_LOG=${NVXRT_LOG:-$OUT/nvxrt.log}
export VKD3D_DEBUG=${VKD3D_DEBUG:-warn}
# HARNESS_NO_SETUP=1: the cache is prepared already (e.g. T1 variants applied by hand)
[[ "${HARNESS_NO_SETUP:-0}" == 1 ]] || "$ROOT/tools/dlssg-setup" "$DLL" "$NVXRT_CACHE"
cat > "$WORK/bin/dlssg_host.ini" <<INI
[Host]
Next=none
FG=$(winpath "$DLL")
ArchID=${ARCH:-0x1b0}
LogLevel=${LOGLEVEL:-1}
${HOST_INI_EXTRA:-}
INI
rm -f "$WORK/bin/dlssg_host.log" "$OUT"/*.ppm
"$PROTON" run "$WORK/bin/d3d12_dlssg.exe" "$(winpath "$WORK/bin/nvngx.dll_dlssg_host.dll")" "$(winpath "$OUT")" "$@" 2>&1 |
  grep -v -E "^(fixme|err:(ole|setupapi))" || true
cp -f "$WORK/bin/dlssg_host.log" "$OUT/host.log" 2>/dev/null || true
