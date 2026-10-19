#!/usr/bin/env bash
# Builds the native engine's models ([Engine] in d4r.ini) on this machine, from the NVIDIA DLLs install.sh put in
# the d4r folder. Shipped as d4r/tools/build-engine-models.sh; install.sh runs it.
#
# usage: bash d4r/tools/build-engine-models.sh [--rr-only | --sr-only] [--exact] [D4R_FOLDER]
#        bash d4r/tools/build-engine-models.sh --find-proton      prints the GE-Proton it would use (or nothing)
#
#   Ray Reconstruction  engine/rr-d, rr-e, rr-f   from d4r/nvngx_dlssd.dll 310.9.1 alone (about half a minute)
#   Super Resolution    engine/k, k-ldr, m, m-ldr, l, l-ldr
#                       NVIDIA's DLSS keeps these weights packed in nvngx_dlss.dll; they are read where NGX unpacks
#                       them: the bundled D3D12 test app renders one frame per model through this d4r folder under
#                       GE-Proton, and the uploads are saved (a minute or two each; the first run also compiles
#                       DLSS's kernels and creates a temporary Proton prefix)
# Needs python3 with numpy, and for Super Resolution a GE-Proton (D4R_PROTON=/path/to/GE-Proton11-3 to choose one).
# The models are the fastest builds (this GPU's own arithmetic; engine/README.md). --exact builds NVIDIA's roundings
# instead: bit-identical to the CUDA path, about 20% slower (K is the same either way).
# Everything written is derived from NVIDIA's DLLs: it stays in this folder and must not be redistributed.
set -euo pipefail

fail() { printf 'build-engine-models: %s\n' "$*" >&2; exit 1; }
DO_RR=1 DO_SR=1 FIND_PROTON=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rr-only) DO_SR=0; shift ;;
    --sr-only) DO_RR=0; shift ;;
    --exact) export D4R_RR_RELAXED=0 M_EXACT=1; shift ;;
    --find-proton) FIND_PROTON=1; shift ;;
    -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) break ;;
  esac
done
find_proton() {
  # a GE-Proton: D4R_PROTON, else GE-Proton11-3 (the tested one), else the newest stock GE-Proton11-N in Steam's
  # compatibility tools. Renamed or rebuilt variants (2026-10-14: one with a suffix after the version number) can
  # fail NGX's CUDA initialisation, so they are only used when named.
  PROTON="${D4R_PROTON:-}"
  if [[ -z "$PROTON" ]]; then
    best=-1
    for d in "$HOME/.local/share/Steam" "$HOME/.steam/root" "$HOME/.steam/steam" \
             "$HOME/.var/app/com.valvesoftware.Steam/data/Steam"; do
      for p in "$d"/compatibilitytools.d/GE-Proton11-*; do
        n="${p##*/GE-Proton11-}"
        [[ -x "$p/proton" && "$n" =~ ^[0-9]+$ ]] || continue
        if [[ "$n" == 3 ]]; then best=1000; PROTON="$p"; elif (( n > best )); then best=$n; PROTON="$p"; fi
      done
    done
  fi
}
if (( FIND_PROTON )); then find_proton; [[ -z "$PROTON" ]] || printf '%s\n' "$PROTON"; exit 0; fi
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KIT="$TOOLS/models"
RUNTIME="$(cd -- "${1:-$TOOLS/..}" && pwd)"
[[ -f "$RUNTIME/d4r.ini" && -d "$KIT/engine" ]] || fail "no d4r folder with the model tools at $RUNTIME"
GAME="$(dirname -- "$RUNTIME")"
DLSSD_SHA256=4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80   # nvngx_dlssd.dll 310.9.1
DLSS_SHA256=be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e    # nvngx_dlss.dll 310.7.0

command -v python3 > /dev/null || fail 'python3 is missing (install Python 3)'
python3 -c 'import numpy' 2> /dev/null || fail 'numpy is missing (install python-numpy / python3-numpy)'
export PATH="$KIT/bin:$PATH"   # the bundled glslangValidator
glslangValidator --version > /dev/null 2>&1 || fail "glslangValidator ($KIT/bin) does not run on this system"
sha() { sha256sum -- "$1" | cut -d' ' -f1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/d4r-models.XXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# replace RUNTIME/engine/NAME with the files of a built model that the runtime reads (a rename: safe while a game runs)
place() {   # place NAME DIR FILE...
  local name=$1 src=$2 dst="$RUNTIME/engine/$1" f
  shift 2
  mkdir -p "$RUNTIME/engine" "$WORK/place/$name"
  for f in "$@"; do cp -- "$src/$f" "$WORK/place/$name/"; done
  rm -rf -- "$dst.new"; mv -- "$WORK/place/$name" "$dst.new"
  rm -rf -- "$dst.old"; [[ ! -e "$dst" ]] || mv -- "$dst" "$dst.old"
  mv -- "$dst.new" "$dst"; rm -rf -- "$dst.old"
  printf '  engine/%s (%s files)\n' "$name" "$(ls "$dst" | wc -l)"
}
model_files() {   # rr.txt and the files it names, for model directory $1
  echo rr.txt
  awk '$1 == "stage" { print $3; if ($5 != "-") print $5; if ($6 ~ /\.args$/) print $6 } $1 == "template" { print $3 }' "$1/rr.txt" | sort -u
}
run() {   # run LOG COMMAND...: the command's output in WORK/LOG, shown when it fails
  local log="$WORK/$1"; shift
  "$@" > "$log" 2>&1 || { tail -n 20 "$log" >&2; fail "failed: $(basename -- "$2") (log above)"; }
}

if (( DO_RR )); then
  DLL="$RUNTIME/nvngx_dlssd.dll"
  [[ -f "$DLL" && "$(sha "$DLL")" == "$DLSSD_SHA256" ]] || fail "Ray Reconstruction models need nvngx_dlssd.dll 310.9.1 in $RUNTIME (run install.sh)"
  echo 'Building the Ray Reconstruction models (presets D, E and F) from nvngx_dlssd.dll...'
  for p in D E; do
    l=$(tr DE de <<< "$p")
    bash "$KIT/engine/rr/build_native_model.sh" --dll "$DLL" "$p" "$WORK/rr-$l" > "$WORK/rr-$l.log" 2>&1 ||
      { tail -n 20 "$WORK/rr-$l.log" >&2; fail "preset $p model failed"; }
    place "rr-$l" "$WORK/rr-$l" $(model_files "$WORK/rr-$l") exposure_rr0.spv exposure_rr1.spv
  done
  run ptx-f.log python3 "$KIT/kernels/tools/extract_dlss_ptx.py" "$DLL" "$WORK/ptx-f"
  run rr-f.log python3 "$KIT/engine/compile_rr2.py" --dll "$DLL" "$WORK/rr-f" --ptx-dir "$WORK/ptx-f"
  place rr-f "$WORK/rr-f" $(model_files "$WORK/rr-f") enc0-aux.bin exposure_f0.spv exposure_f1.spv exposure_f2.spv
fi

if (( DO_SR )); then
  DLL="$RUNTIME/nvngx_dlss.dll"
  [[ -f "$DLL" && "$(sha "$DLL")" == "$DLSS_SHA256" ]] || fail "Super Resolution models need nvngx_dlss.dll 310.7.0 in $RUNTIME (run install.sh)"
  [[ -f "$RUNTIME/ngx/_nvngx.dll" ]] || fail "missing $RUNTIME/ngx/_nvngx.dll (run install.sh)"
  find_proton
  [[ -n "$PROTON" && -x "$PROTON/proton" ]] ||
    fail 'no GE-Proton found: set D4R_PROTON=/path/to/GE-Proton11-3 (the folder holding the proton script)'
  STEAM_ROOT="$(dirname -- "$(dirname -- "$PROTON")")"
  [[ -d "$STEAM_ROOT/steamapps" || ! -d "$HOME/.local/share/Steam" ]] || STEAM_ROOT="$HOME/.local/share/Steam"
  # the test app beside the D3D12 pair the game uses (proxy release: the game folder; ASI release: the d4r folder)
  APP="$WORK/app"; mkdir -p "$APP"
  cp -- "$KIT/bin/d3d12_dlss_harness.exe" "$APP/"
  if [[ -f "$RUNTIME/d4r12core.dll" ]]; then
    cp -- "$RUNTIME/d3d12.dll" "$RUNTIME/d4r12core.dll" "$APP/"
  elif [[ -f "$GAME/d3d12core.dll" ]]; then
    cp -- "$GAME/d3d12.dll" "$GAME/d3d12core.dll" "$APP/"
  else
    fail "d4r's d3d12.dll / d3d12core.dll not found next to $RUNTIME"
  fi
  winpath() { printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"; }
  dump() {   # dump NAME PRESET: the uploads of one frame of preset PRESET in WORK/NAME/replay
    local w="$WORK/$1"
    mkdir -p "$w/replay"
    printf '  rendering one frame with model %s through d4r (GE-Proton %s)...\n' "$1" "$(basename "$PROTON")"
    (cd "$APP" && env STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM_ROOT" STEAM_COMPAT_DATA_PATH="$WORK/prefix" \
      PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 WINEDLLOVERRIDES='d3d12=n,b;d3d12core=n,b;d4r12core=n,b' \
      D4R_ENGINE=0 D4R_DLSS_PRESET="$2" D4R_HARNESS_CREATE_FLAGS=0x4B D4R_SHIM_LOG="$(winpath "$w/shim.log")" \
      D4R_CUDA_REPLAY_DUMP_DIR="$w/replay" D4R_CUDA_REPLAY_DUMP_LIMIT=60 \
      "$PROTON/proton" run "$APP/d3d12_dlss_harness.exe" "$(winpath "$RUNTIME/nvngx.dll")" "$(winpath "$w/out.raw")" \
      1 640 360 1280 720) > "$w/run.log" 2>&1 || true
    if ! ls "$w/replay" | grep -q "$3"; then
      tail -n 15 "$w/run.log" >&2
      [[ ! -f "$w/shim.log" ]] || tail -n 15 "$w/shim.log" >&2
      fail "model $1: the test frame did not run through DLSS (logs above)"
    fi
  }
  echo 'Building the Super Resolution models (K, M and L) from what NGX uploads...'
  mkdir -p "$WORK/prefix"
  dump k 11 dltss_pwin_dec0_layer
  run k-extract.log python3 "$KIT/engine/extract_k.py" "$WORK/k/replay" "$WORK/k/model"
  run k-lut.log python3 "$KIT/engine/prepare_k.py" "$WORK/k" "$WORK/k/case" --lut-only
  for v in k k-ldr; do
    run "$v.log" python3 "$KIT/engine/compile_k.py" "$WORK/k/model" "$WORK/k/case/lut.bin" "$WORK/$v-out" \
      $([[ $v == k-ldr ]] && echo --ldr)
    place "$v" "$WORK/$v-out" $(ls "$WORK/$v-out")
  done
  rm -rf -- "$WORK/k"
  for c in "m 13 rrlite_enc0_4x4_mvlo_hdr_folded" "l 12 rrlite_enc0_4x4_mvlo_hdr$"; do
    set -- $c
    dump "$1" "$2" "$3"
    for v in "$1" "$1-ldr"; do
      run "$v.log" python3 "$KIT/engine/compile_m.py" "$WORK/$1" "$WORK/$v-out" $([[ $v == *-ldr ]] && echo --ldr)
      place "$v" "$WORK/$v-out" $(ls "$WORK/$v-out")
    done
    rm -rf -- "${WORK:?}/$1"
  done
fi
printf 'Engine models are in %s/engine.\n' "$RUNTIME"
