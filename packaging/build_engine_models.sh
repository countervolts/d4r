#!/usr/bin/env bash
# Builds the native engine's models ([Engine] in d4r.ini) on this machine, from the NVIDIA DLLs install.sh put in
# the d4r folder. Shipped as d4r/tools/build-engine-models.sh; install.sh runs it.
#
# usage: bash d4r/tools/build-engine-models.sh [--rr-only | --sr-only] [--exact] [D4R_FOLDER]
#        bash d4r/tools/build-engine-models.sh --find-proton      prints the Proton it would use (or nothing)
#        bash d4r/tools/build-engine-models.sh --list-sr-models   prints the Super Resolution models it builds here
#   Ray Reconstruction  engine/rr-d, rr-e, rr-f   from d4r/nvngx_dlssd.dll 310.9.1 alone (about half a minute)
#   Super Resolution    engine/e, f, k, k-ldr, l, l-ldr, m, m-ldr
#                       NVIDIA's DLSS keeps these weights packed in nvngx_dlss.dll; they are read where NGX unpacks
#                       them: the bundled D3D12 test app renders one frame per model through this d4r folder under
#                       Proton, and the uploads are saved (the frames after the first render at the same time;
#                       the first run also compiles DLSS's kernels and creates a temporary Proton prefix)
# Needs python3 with numpy, and for Super Resolution a Proton (D4R_PROTON=/path/to/proton-folder to choose one).
# The models are the fastest builds (this GPU's own arithmetic; engine/README.md). --exact builds NVIDIA's roundings
# instead: bit-identical to the CUDA path, about 20% slower (K is the same either way).
# Everything written is derived from NVIDIA's DLLs: it stays in this folder and must not be redistributed.
set -euo pipefail

fail() { printf 'build-engine-models: %s\n' "$*" >&2; exit 1; }
DO_RR=1 DO_SR=1 FIND_PROTON=0 LIST_SR=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rr-only) DO_SR=0; shift ;;
    --sr-only) DO_RR=0; shift ;;
    --exact) export D4R_RR_RELAXED=0 D4R_RR2_POOLED=rgba32f M_EXACT=1; shift ;;
    --find-proton) FIND_PROTON=1; shift ;;
    --list-sr-models) LIST_SR=1; shift ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) break ;;
  esac
done
find_proton() {
  # a Proton for the test frames: D4R_PROTON, else GE-Proton11-3 (the tested one), else the newest GE-Proton, else
  # the newest other Proton in Steam's compatibility tools or libraries (most Proton versions run d4r fine)
  PROTON="${D4R_PROTON:-}" PROTON_STEAM=''
  [[ -z "$PROTON" ]] || return 0
  local d p lib ge=() other=()
  declare -A root=()
  for d in "$HOME/.local/share/Steam" "$HOME/.steam/root" "$HOME/.steam/steam" \
           "$HOME/.var/app/com.valvesoftware.Steam/data/Steam"; do
    [[ -d "$d" ]] || continue
    for p in "$d"/compatibilitytools.d/*; do
      [[ -x "$p/proton" ]] || continue
      root[$p]=$d
      if [[ "${p##*/}" == GE-Proton* ]]; then ge+=("$p"); else other+=("$p"); fi
    done
    # Valve's Proton, in every Steam library
    for lib in "$d" $(sed -n 's/^[[:space:]]*"path"[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' \
                      "$d/steamapps/libraryfolders.vdf" 2>/dev/null | tr ' ' '\001'); do
      lib=$(tr '\001' ' ' <<< "$lib")
      for p in "$lib"/steamapps/common/Proton*; do
        [[ -x "$p/proton" ]] || continue
        root[$p]=$d; other+=("$p")
      done
    done
  done
  for p in "${ge[@]}"; do [[ "${p##*/}" != GE-Proton11-3 ]] || { PROTON=$p; PROTON_STEAM=${root[$p]}; return 0; }; done
  for list in ge other; do
    local -n candidates=$list
    (( ${#candidates[@]} )) || continue
    PROTON=$(for p in "${candidates[@]}"; do printf '%s\t%s\n' "${p##*/}" "$p"; done | sort -V | tail -n 1 | cut -f2)
    PROTON_STEAM=${root[$PROTON]}
    return 0
  done
  return 0
}
gpu_arch() {   # the GPU the native kernels are chosen for: D4R_GPU_ARCH, else the KFD node with the most SIMDs
  if [[ -n "${D4R_GPU_ARCH:-}" ]]; then printf '%s\n' "$D4R_GPU_ARCH"; return; fi
  local f simds version best=0 found=''
  for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [[ -r "$f" ]] || continue
    simds=$(awk '$1 == "simd_count" { print $2 }' "$f")
    version=$(awk '$1 == "gfx_target_version" { print $2 }' "$f")   # gfx1033 = 100303
    if (( ${simds:-0} > best )); then best=$simds; found=$version; fi
  done
  [[ -z "$found" ]] || printf 'gfx%d%d%x\n' $((found / 10000)) $((found / 100 % 100)) $((found % 100))
}
sr_models() {
  # every GPU: on RDNA2 M and L run on the engine's v_dot2 stages (their network kernels stubbed, stub_network)
  echo e f k k-ldr l l-ldr m m-ldr
}
stub_network() {
  # The L/M test frames only provide their network layers' launches (arguments and weight uploads): on RDNA2 those
  # layers run as empty kernels (model_tools.py stub-ptx), so ZLUDA does not compile them with emulated matrix
  # products for minutes each. D4R_ENGINE_STUB_NETWORK=0 / 1 forces either.
  case "${D4R_ENGINE_STUB_NETWORK:-auto}" in
    0) return 1 ;;
    1) return 0 ;;
    *) [[ "$(gpu_arch)" == gfx103* ]] ;;
  esac
}
if (( LIST_SR )); then sr_models; exit 0; fi
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

LOG_ROOT="${D4R_MODEL_LOG_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/d4r/model-builds}"
mkdir -p -- "$LOG_ROOT"
BUILD_LOGS="$(mktemp -d "$LOG_ROOT/build.XXXXXX")"
WORK="$BUILD_LOGS/work"; mkdir -p "$WORK"
MODEL_TOOLS="$KIT/model_tools.py"
[[ -f "$MODEL_TOOLS" ]] || fail "missing $MODEL_TOOLS; update the model builder tools"
cleanup() {
  local status=$?
  kill $(jobs -p) 2> /dev/null || true
  if (( status == 0 )); then
    while IFS= read -r -d '' log; do
      relative="${log#"$WORK/"}"
      mkdir -p -- "$BUILD_LOGS/$(dirname -- "$relative")"
      cp -- "$log" "$BUILD_LOGS/$relative"
    done < <(find "$WORK" -name '*.log' -type f -print0)
    rm -rf -- "$WORK"
  else
    printf 'build-engine-models: failed build and captures preserved at %s\n' "$WORK" >&2
  fi
  printf 'Model build diagnostics: %s\n' "$BUILD_LOGS"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# replace RUNTIME/engine/NAME with the files of a built model that the runtime reads (a rename: safe while a game runs)
place() {   # place NAME DIR FILE...
  local name=$1 src=$2 dst="$RUNTIME/engine/$1" f
  shift 2
  mkdir -p "$RUNTIME/engine" "$WORK/place/$name"
  for f in "$@"; do cp -r -- "$src/$f" "$WORK/place/$name/"; done
  python3 "$MODEL_TOOLS" manifest "$WORK/place/$name"
  rm -rf -- "$dst.new"; mv -- "$WORK/place/$name" "$dst.new"
  rm -rf -- "$dst.old"; [[ ! -e "$dst" ]] || mv -- "$dst" "$dst.old"
  mv -- "$dst.new" "$dst"; rm -rf -- "$dst.old"
  printf '  engine/%s (%s files)\n' "$name" "$(ls "$dst" | wc -l)"
}
model_files() {   # rr.txt and the files it names, for model directory $1
  echo rr.txt
  awk '$1 == "stage" { print $3; if ($5 != "-") print $5; if ($6 ~ /\.args$/) print $6 } $1 == "template" { print $3 }
       $1 == "phases" { print $3; print $4; print $5; print $6 }' "$1/rr.txt" | sort -u
}
run() {   # run LOG COMMAND...: the command's output in WORK/LOG, shown when it fails
  local log="$WORK/$1"; shift
  python3 "$MODEL_TOOLS" run --log "$log" --timeout "${D4R_MODEL_TIMEOUT:-900}" -- "$@" ||
    fail "failed: $1 (see $log)"
}

if (( DO_RR )); then
  DLL="$RUNTIME/nvngx_dlssd.dll"
  [[ -f "$DLL" && "$(sha "$DLL")" == "$DLSSD_SHA256" ]] || fail "Ray Reconstruction models need nvngx_dlssd.dll 310.9.1 in $RUNTIME (run install.sh)"
  echo 'Building the Ray Reconstruction models (presets D, E and F) from nvngx_dlssd.dll...'
  # Build both portable arithmetic variants; runtime selects by device capabilities. Do not let the build
  # host (or an inherited developer switch) silently determine what another game/GPU receives from the cache.
  for p in D E; do
    for arithmetic in fp8 gfx11; do
      l=$(tr DE de <<< "$p"); name="rr-$l"; gfx11=0
      if [[ "$arithmetic" == gfx11 ]]; then name+="-gfx11"; gfx11=1; fi
      run "$name.log" env D4R_RR_GFX11="$gfx11" bash "$KIT/engine/rr/build_native_model.sh" \
        --dll "$DLL" "$p" "$WORK/$name"
      place "$name" "$WORK/$name" $(model_files "$WORK/$name") exposure_rr0.spv exposure_rr1.spv
    done
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
    fail 'no Proton found: set D4R_PROTON=/path/to/proton-folder (the folder holding the proton script)'
  STEAM_ROOT="${PROTON_STEAM:-$(dirname -- "$(dirname -- "$PROTON")")}"
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
  PTX="$WORK/ptx"
  winpath() { printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"; }
  STUBS="$WORK/stubs"
  dump() {   # dump NAME PRESET PATTERN [FLAGS]: the uploads of one frame of preset PRESET in WORK/NAME/replay
    local w="$WORK/$1" stub=()
    mkdir -p "$w/replay"
    # L and M with empty network kernels (stub_network): their frame's pixels are not checked
    [[ ! -d "${STUBS:-}" || ( "$1" != l && "$1" != m ) ]] || stub=(D4R_CUDA_REPLACE_DIR="$STUBS")
    # CPU input polling: the replay dump synchronizes and reads device memory at every launch, which the GPU-side
    # input wait does not allow (NGX's output came out empty)
    printf '  rendering one frame with model %s through d4r (%s)...\n' "$1" "$(basename "$PROTON")"
    (cd "$APP" && python3 "$MODEL_TOOLS" run --log "$w/run.log" --timeout "${D4R_MODEL_TIMEOUT:-900}" -- \
      env STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM_ROOT" STEAM_COMPAT_DATA_PATH="$WORK/prefix" \
      PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 WINEDLLOVERRIDES='d3d12=n,b;d3d12core=n,b;d4r12core=n,b' \
      D4R_ENGINE=0 D4R_SHIM_GPU_INPUT_WAIT=0 "${stub[@]}" D4R_DLSS_PRESET="$2" D4R_HARNESS_CREATE_FLAGS="${4:-0x4B}" D4R_SHIM_LOG="$(winpath "$w/shim.log")" \
      D4R_CUDA_TRACE="$(winpath "$w/cuda.log")" D4R_CUDA_VERBOSE=1 \
      D4R_CUDA_REPLAY_DUMP_DIR="$w/replay" D4R_CUDA_REPLAY_DUMP_LIMIT=60 \
      "$PROTON/proton" run "$APP/d3d12_dlss_harness.exe" "$(winpath "$RUNTIME/nvngx.dll")" "$(winpath "$w/out.raw")" \
      1 640 360 1280 720) || fail "model $1: capture process failed; diagnostics in $w"
    if ! python3 "$MODEL_TOOLS" capture "$w" "$1" ${stub[@]:+--stubbed}; then
      [[ ! -f "$w/shim.log" ]] || tail -n 15 "$w/shim.log" >&2
      fail "model $1: incomplete or invalid test frame (diagnostics in $w)"
    fi
  }
  mkdir -p "$WORK/prefix"
  models=" $(sr_models) "
  echo "Building the Super Resolution models ($(sr_models)) from what NGX uploads..."
  run ptx.log python3 "$KIT/kernels/tools/extract_dlss_ptx.py" "$DLL" "$PTX"
  if [[ "$models" == *" m "* || "$models" == *" l "* ]] && stub_network; then
    run stubs.log python3 "$MODEL_TOOLS" stub-ptx "$DLL" "$STUBS"
  fi
  # Each test frame compiles its DLSS kernels in one thread, so the frames render at the same time; the first
  # renders alone because it creates the Proton prefix. A model is built as soon as its frame is done.
  declare -A JOBS
  start() { dump "$@" & JOBS[$1]=$!; }
  finish() { wait "${JOBS[$1]}" || exit 1; }   # dump has said why
  # presets E and F (DLSS 3's CNN): LDR with render-size motion (the model serves every flag set)
  dump e 5 cuda_engine_input_kernel_rel 0xa
  start f 6 cuda_engine_input_kernel_rel 0xa
  start k 11 dltss_pwin_dec0_layer
  [[ "$models" != *" l "* ]] || start l 12 'rrlite_enc0_4x4_mvlo_hdr$'
  [[ "$models" != *" m "* ]] || start m 13 rrlite_enc0_4x4_mvlo_hdr_folded
  for c in "e E" "f F"; do
    set -- $c
    [[ $1 == e ]] || finish "$1"
    run "$1.log" python3 "$KIT/engine/compile_sr_e.py" "$WORK/$1/replay" "$DLL" "$WORK/$1-out" --preset "$2" --ptx-dir "$PTX"
    place "$1" "$WORK/$1-out" $(ls "$WORK/$1-out")
    rm -rf -- "${WORK:?}/$1"
  done
  finish k
  run k-extract.log python3 "$KIT/engine/extract_k.py" "$WORK/k/replay" "$WORK/k/model"
  run k-lut.log python3 "$KIT/engine/prepare_k.py" "$WORK/k" "$WORK/k/case" --lut-only
  # RDNA3 vkhip uses these same captured weights and the shipped native kernels: the fast set by default and,
  # when the release has it, the accuracy set too ([Kernels] PreferAccuracy; byte-identical to the CUDA backend).
  hip_args=()
  arch=$(gpu_arch)
  if [[ "$arch" == gfx110[0-3] && -f "$RUNTIME/kernels/$arch/dltss_pwin_enc0_layer.hsaco" ]]; then
    hip_args=(--hip "$RUNTIME/kernels/$arch" --capture "$WORK/k")
    if [[ -f "$RUNTIME/kernels/accuracy/$arch/dltss_pwin_enc0_layer.hsaco" ]]; then
      hip_args+=(--hip-accuracy "$RUNTIME/kernels/accuracy/$arch")
    fi
  fi
  for v in k k-ldr; do
    run "$v.log" python3 "$KIT/engine/compile_k.py" "$WORK/k/model" "$WORK/k/case/lut.bin" "$WORK/$v-out" \
      "${hip_args[@]}" $([[ $v == k-ldr ]] && echo --ldr)
    place "$v" "$WORK/$v-out" $(ls "$WORK/$v-out")
  done
  rm -rf -- "$WORK/k"
  for m in l m; do
    [[ "$models" == *" $m "* ]] || continue
    finish "$m"
    for v in "$m" "$m-ldr"; do
      run "$v.log" python3 "$KIT/engine/compile_m.py" "$WORK/$m" "$WORK/$v-out" $([[ $v == *-ldr ]] && echo --ldr)
      place "$v" "$WORK/$v-out" $(ls "$WORK/$v-out")
    done
    rm -rf -- "${WORK:?}/$m"
  done
fi
printf 'Engine models are in %s/engine.\n' "$RUNTIME"
