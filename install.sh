#!/usr/bin/env bash
# NVIDIA installer: beside the proxy release's d4r/ or inside the ASI's d4r/.
# Installs NVIDIA's DLLs and then builds the native engine's models from them on this machine. Both are kept in
# ~/.cache/d4r, so the next game's install copies them instead of downloading or building again.
set -euo pipefail
DLSS_URL=https://raw.githubusercontent.com/NVIDIA/DLSS/a291cc7d2cc642a51566f3dfd5376f635cd1b284/lib/Windows_x86_64/rel/nvngx_dlss.dll
DLSS_SHA256=be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e
DLSSD_URL=https://raw.githubusercontent.com/NVIDIA/DLSS/374959484e79a640feaba44c93ac8cfb0a03f5b5/lib/Windows_x86_64/rel/nvngx_dlssd.dll
DLSSD_SHA256=4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80
DRIVER_URL=https://us.download.nvidia.com/Windows/596.36/596.36-desktop-win10-win11-64bit-international-dch-whql.exe
DRIVER_SHA256=4a1793d2bc7792f43270fe8286ec381ae24409fe29049dbe1eb8530ed89e19cf
NGX_SHA256=0e3423b3b7afa1019c7ecac053a8408df8e5d9d9d904434705c6aaeacec02fef

fail() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }
usage() {
  printf '%s\n' 'Usage: bash install.sh [options] [extracted-release-directory]' \
    'Default directory: the one containing this script, regardless of your working directory.' \
    'Run it from a terminal and it asks what to do. The options answer those questions in advance:' \
    '  --yes             accept every default (also the behaviour without a terminal)' \
    '  --no-engine       only the NVIDIA DLLs, no engine models' \
    '  --engine-only     only the engine models (the DLLs are already installed)' \
    '  --exact           engine models with NVIDIA'"'"'s exact roundings (slower) instead of the fastest builds' \
    '  --rebuild-engine  build the models again even when a cached build fits' \
    '1. Downloads official NVIDIA DLSS 310.7.0 (Super Resolution), DLSS Ray Reconstruction 310.9.1 and NGX from' \
    '   driver 596.36 (~1 GiB, once: verified copies are kept in ~/.cache/d4r/nvidia).' \
    '2. Builds the native engine'"'"'s models from those DLLs into d4r/engine (once per DLL set and builder version:' \
    '   ~/.cache/d4r/engine-models). Faster on RDNA4; d4r.ini [Engine] Enabled = auto uses it.' \
    'Requires Bash, curl, sha256sum, 7zz/7z/7za, python3 with numpy, and GE-Proton 11 in Steam (or' \
    'D4R_PROTON=/path/to/GE-Proton11-3). No sudo needed. NVIDIA license terms apply. D4R_CACHE_DIR moves the cache.'
}
DO_DLLS=1 DO_ENGINE=ask EXACT=ask REBUILD=0 YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -y|--yes) YES=1; shift ;;
    --no-engine) DO_ENGINE=0; shift ;;
    --engine-only) DO_DLLS=0; DO_ENGINE=1; shift ;;
    --exact) EXACT=1; shift ;;
    --rebuild-engine) REBUILD=1; shift ;;
    -*) usage >&2; exit 2 ;;
    *) break ;;
  esac
done
[[ $# -le 1 ]] || { usage >&2; exit 2; }
ROOT=$(cd -- "${1:-$(dirname -- "${BASH_SOURCE[0]}")}" && pwd)
RUNTIME="$ROOT/d4r"
[[ ! -f "$ROOT/d4r.ini" ]] || RUNTIME="$ROOT"
[[ -f "$RUNTIME/nvngx.dll" && -f "$RUNTIME/d4r.ini" ]] || fail "Extract the release ZIP first; no d4r runtime found in $ROOT"
command -v sha256sum >/dev/null || fail 'Missing sha256sum (install coreutils).'
valid() { [[ -f "$1" ]] && [[ $(sha256sum -- "$1" | cut -d' ' -f1) == "$2" ]]; }
CACHE="${D4R_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/d4r}"

# Questions only on a terminal; otherwise (or with --yes) every default applies.
interactive() { (( YES == 0 )) && [[ -t 0 && -t 1 ]]; }
ask() {   # ask "QUESTION" DEFAULT(y|n): 0 for yes. Enter takes the default; any answer starting with y or n counts.
  local answer
  if ! interactive; then [[ $2 == y ]]; return; fi
  while true; do
    read -r -p "$1 [$([[ $2 == y ]] && echo Y/n || echo y/N)] " answer || { echo; answer=''; }
    answer="${answer//[[:space:]]/}"
    case "${answer,,}" in
      '') [[ $2 == y ]]; return ;;
      y*) return 0 ;;
      n*) return 1 ;;
      *) echo '  Please answer y or n (or press Enter for the default).' ;;
    esac
  done
}

TMP=$(mktemp -d)
STAGED=''
trap '[[ -z "$STAGED" ]] || rm -f -- "$STAGED"; rm -rf -- "$TMP"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
download() {
  curl --fail --location --show-error --retry 3 --connect-timeout 30 --progress-bar \
    --proto '=https' --proto-redir '=https' --output "$2" "$1" || fail "Download failed: $1. Run the installer again to retry."
}
verify() { valid "$1" "$2" || fail "SHA-256 mismatch for $1; no DLLs installed."; }
# Stage on the destination filesystem for atomic replacement and retain a unique backup of a differing file.
place() {
  local src=$1 dst=$2 backup
  mkdir -p -- "$(dirname -- "$dst")"
  [[ ! -d "$dst" ]] || fail "Destination is a directory: $dst"
  STAGED=$(mktemp "${dst}.install.XXXXXX")
  cp -- "$src" "$STAGED"
  chmod 644 "$STAGED"
  if [[ -e "$dst" || -L "$dst" ]]; then
    backup=$(mktemp "${dst}.backup.XXXXXX")
    cp -p -- "$dst" "$backup"
    printf 'Saved previous DLL: %s\n' "$backup"
  fi
  mv -f -- "$STAGED" "$dst"
  STAGED=''
  printf 'Installed: %s\n' "$dst"
}
# A verified copy in the cache, or none. Cached files are checked again before use.
cached() { local f="$CACHE/nvidia/$1"; valid "$f" "$2" && printf '%s\n' "$f"; }
keep() { mkdir -p -- "$CACHE/nvidia" && cp -- "$1" "$CACHE/nvidia/$2.tmp" && mv -f -- "$CACHE/nvidia/$2.tmp" "$CACHE/nvidia/$2"; }

install_dlls() {
  local need_dlss=1 need_dlssd=1 need_ngx=1 tool src
  valid "$RUNTIME/nvngx_dlss.dll" "$DLSS_SHA256" && need_dlss=0
  valid "$RUNTIME/nvngx_dlssd.dll" "$DLSSD_SHA256" && need_dlssd=0
  valid "$RUNTIME/ngx/_nvngx.dll" "$NGX_SHA256" && need_ngx=0
  if (( need_dlss == 0 && need_dlssd == 0 && need_ngx == 0 )); then
    echo 'NVIDIA DLLs are already installed and verified.'
    return
  fi
  # earlier installs' verified copies first
  if (( need_dlss )) && src=$(cached nvngx_dlss.dll "$DLSS_SHA256"); then cp -- "$src" "$TMP/nvngx_dlss.dll"; need_dlss=2; fi
  if (( need_dlssd )) && src=$(cached nvngx_dlssd.dll "$DLSSD_SHA256"); then cp -- "$src" "$TMP/nvngx_dlssd.dll"; need_dlssd=2; fi
  if (( need_ngx )) && src=$(cached _nvngx.dll "$NGX_SHA256"); then mkdir -p "$TMP/ngx"; cp -- "$src" "$TMP/ngx/_nvngx.dll"; need_ngx=2; fi
  if (( need_dlss == 1 || need_dlssd == 1 || need_ngx == 1 )); then
    local size=0
    (( need_dlss != 1 )) || size=$((size + 57))
    (( need_dlssd != 1 )) || size=$((size + 47))
    (( need_ngx != 1 )) || size=$((size + 915))
    echo "d4r needs NVIDIA's DLSS files, downloaded from NVIDIA under NVIDIA's license terms:"
    echo '  DLSS: https://github.com/NVIDIA/DLSS/blob/v310.9.1/LICENSE.txt'
    echo '  Driver: https://www.nvidia.com/en-us/drivers/nvidia-license/'
    ask "Download them now (~$size MiB, once: later installs use $CACHE/nvidia)?" y ||
      fail 'Not downloaded. d4r needs these files; run install.sh again when you want to install them.'
    command -v curl >/dev/null || fail 'Missing curl. Install it and run this script again.'
    SEVENZIP=''
    if (( need_ngx == 1 )); then
      for tool in 7zz 7z 7za; do
        if command -v "$tool" >/dev/null; then SEVENZIP=$tool; break; fi
      done
      [[ -n "$SEVENZIP" ]] || fail 'Missing 7-Zip. Install 7zip (or p7zip) and run this script again.'
    fi
  fi
  if (( need_dlss == 1 )); then
    echo 'Downloading DLSS 310.7.0...'
    download "$DLSS_URL" "$TMP/nvngx_dlss.dll"
    verify "$TMP/nvngx_dlss.dll" "$DLSS_SHA256"
    keep "$TMP/nvngx_dlss.dll" nvngx_dlss.dll
  fi
  if (( need_dlssd == 1 )); then
    echo 'Downloading DLSS Ray Reconstruction 310.9.1...'
    download "$DLSSD_URL" "$TMP/nvngx_dlssd.dll"
    verify "$TMP/nvngx_dlssd.dll" "$DLSSD_SHA256"
    keep "$TMP/nvngx_dlssd.dll" nvngx_dlssd.dll
  fi
  if (( need_ngx == 1 )); then
    echo 'Downloading NVIDIA driver 596.36 (~915 MiB); extracting NGX without running the installer...'
    download "$DRIVER_URL" "$TMP/driver.exe"
    verify "$TMP/driver.exe" "$DRIVER_SHA256"
    "$SEVENZIP" e -y -bso0 -bsp0 "-o$TMP/ngx" "$TMP/driver.exe" 'Display.Driver/_nvngx.dll' || fail 'Could not extract NGX; no DLLs installed.'
    rm -f -- "$TMP/driver.exe"
    verify "$TMP/ngx/_nvngx.dll" "$NGX_SHA256"
    keep "$TMP/ngx/_nvngx.dll" _nvngx.dll
  fi
  # Everything is verified before any destination changes.
  (( need_dlss == 0 )) || place "$TMP/nvngx_dlss.dll" "$RUNTIME/nvngx_dlss.dll"
  (( need_dlssd == 0 )) || place "$TMP/nvngx_dlssd.dll" "$RUNTIME/nvngx_dlssd.dll"
  (( need_ngx == 0 )) || place "$TMP/ngx/_nvngx.dll" "$RUNTIME/ngx/_nvngx.dll"
  echo 'NVIDIA DLLs installed and verified.'
}

(( DO_DLLS == 0 )) || install_dlls

# --- the engine's models ---------------------------------------------------------------------------------------
BUILDER="$RUNTIME/tools/build-engine-models.sh"
done_msg() { echo 'Done. Follow the included README for Steam launch options.'; }
if [[ ! -f "$BUILDER" ]]; then
  [[ "$DO_ENGINE" != 1 ]] || fail "missing $BUILDER (this release has no engine-model builder)"
  done_msg; exit 0
fi
# the Super Resolution models depend on the GPU (RDNA2 builds no L and M), so the builder lists them
models=(rr-d rr-e rr-f rr-d-gfx11 rr-e-gfx11 $(bash "$BUILDER" --list-sr-models))
# One build serves every game with the same DLLs, builder and model set: the key covers them and the accuracy choice.
key() {   # key EXACT(0|1)
  { echo "exact=$1"
    echo "models=${models[*]}"
    # Model compilers accept per-layer K_DEFS_*, M_* arithmetic/tile options, and D4R_RR_*, D4R_RR2_* and
    # D4R_PTX2RR_* choices. A cache hit must not silently replace a model built with different compiler options.
    env | LC_ALL=C sort | sed -n -e '/^K_/p' -e '/^M_/p' -e '/^D4R_RR_/p' -e '/^D4R_RR2_/p' -e '/^D4R_PTX2RR_/p'
    sha256sum -- "$RUNTIME/nvngx_dlss.dll" "$RUNTIME/nvngx_dlssd.dll" 2>/dev/null | cut -d' ' -f1
    (cd "$RUNTIME/tools" && find . -type f ! -name '*.pyc' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum); } | sha256sum | cut -c1-16
}
have() { python3 "$RUNTIME/tools/models/model_tools.py" validate "$1" "${models[@]}" 2>/dev/null; }
STAMP="$RUNTIME/engine/.d4r-models"
if [[ "$EXACT" == ask ]]; then
  EXACT=0
  for e in 0 1; do [[ -f "$STAMP" && "$(cat "$STAMP")" == "$(key $e)" ]] && EXACT=$e; done
fi
if (( REBUILD == 0 )) && [[ "$DO_ENGINE" != 0 && -f "$STAMP" && "$(cat "$STAMP")" == "$(key "$EXACT")" ]] && have "$RUNTIME/engine"; then
  echo 'The engine models in this folder are up to date.'
  done_msg; exit 0
fi
if [[ "$DO_ENGINE" == ask ]] && ! interactive; then DO_ENGINE=1; fi
if [[ "$DO_ENGINE" == ask ]]; then
  echo
  echo "Set up d4r's own DLSS engine (recommended)?"
  echo '  + Faster on RDNA4: DLSS 4 / 4.5 and Ray Reconstruction run on d4r'"'"'s Vulkan engine instead of NVIDIA'"'"'s kernels'
  echo '    through CUDA emulation (for example Ray Reconstruction at 720p -> 4K in about 3.6 ms per frame).'
  echo '  + On RDNA2 (Steam Deck) it is the route for DLSS 3'"'"'s CNN presets E and F and for DLSS 4 (K): those run without cooperative matrices.'
  echo '  + Lower latency: the game shows the upscaled result of the frame it just rendered, not an older one.'
  echo '  + No minute-long freeze while DLSS'"'"'s CUDA kernels compile the first time a model is used.'
  echo '  - Longer setup, once: building its models takes a few minutes (it renders a test frame per model'
  echo '    through GE-Proton) and needs ~300 MB of temporary space. Later installs reuse the models.'
  echo '  - Slower for K and M on RDNA3; there d4r keeps those two models on its normal path.'
  echo '  Without it, d4r uses its normal path; anything the engine does not cover uses that path anyway.'
  if ask 'Set up the engine now?' y; then DO_ENGINE=1; else DO_ENGINE=0; fi
fi
if [[ "$DO_ENGINE" == 0 ]]; then
  echo 'No engine models: d4r uses its normal path. Run install.sh again to build them later.'
  done_msg; exit 0
fi
KEY="$(key "$EXACT")"
SHARED="$CACHE/engine-models/$KEY"
copy_models() {   # copy_models FROM TO: whole model folders, each replaced by a rename
  local m
  mkdir -p -- "$2"
  for m in "${models[@]}"; do
    rm -rf -- "$2/$m.new"; cp -r -- "$1/$m" "$2/$m.new"
    rm -rf -- "$2/$m.old"; [[ ! -e "$2/$m" ]] || mv -- "$2/$m" "$2/$m.old"
    mv -- "$2/$m.new" "$2/$m"; rm -rf -- "$2/$m.old"
  done
}
if (( REBUILD == 0 )) && have "$SHARED"; then
  echo "Using the engine models built earlier ($SHARED)."
  copy_models "$SHARED" "$RUNTIME/engine"
else
  failed=()
  flags=(); (( EXACT == 0 )) || flags=(--exact)
  bash "$BUILDER" --rr-only "${flags[@]}" "$RUNTIME" || failed+=('Ray Reconstruction')
  if [[ -z "${D4R_PROTON:-}" && -z "$(bash "$BUILDER" --find-proton)" ]] && interactive; then
    echo 'The Super Resolution models render one test frame through GE-Proton 11, which was not found in Steam.'
    read -r -p 'Path to a GE-Proton folder (Enter to skip): ' D4R_PROTON || D4R_PROTON=''
    export D4R_PROTON
  fi
  if [[ -n "${D4R_PROTON:-}" || -n "$(bash "$BUILDER" --find-proton)" ]]; then
    bash "$BUILDER" --sr-only "${flags[@]}" "$RUNTIME" || failed+=('Super Resolution')
  else
    failed+=('Super Resolution')
    echo 'install.sh: no GE-Proton 11 found for the Super Resolution models.' >&2
  fi
  if (( ${#failed[@]} )); then
    printf 'install.sh: the engine models for %s were not built (reason above). d4r runs those on its normal path;\n' \
      "$(IFS=/; echo "${failed[*]}")" >&2
    printf 'fix the cause and run install.sh again.\n' >&2
    exit 1
  fi
  # keep a copy for the next game
  rm -rf -- "$SHARED.tmp"; mkdir -p -- "$SHARED.tmp"
  for m in "${models[@]}"; do cp -r -- "$RUNTIME/engine/$m" "$SHARED.tmp/"; done
  rm -rf -- "$SHARED"; mv -- "$SHARED.tmp" "$SHARED"
fi
printf '%s\n' "$KEY" > "$STAMP"
if grep -qiE '^[[:space:]]*Enabled[[:space:]]*=[[:space:]]*(false|0|no|off)[[:space:]]*$' <(sed -n '/^\[Engine\]/,/^\[/p' "$RUNTIME/d4r.ini"); then
  echo 'Note: d4r/d4r.ini turns the engine off ([Engine] Enabled = false); set it to auto to use these models.'
fi
done_msg
