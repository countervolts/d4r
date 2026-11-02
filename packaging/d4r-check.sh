#!/bin/sh
# Checks an d4r install: run it from the game folder (the one holding the game's .exe and d4r/),
# or give that folder as the argument. It only reads files and prints what it finds.
# usage: sh plugins/d4r/d4r-check.sh GAME_FOLDER [ROCM_DIR] (or sh d4r/d4r-check.sh for proxy installs)
GAME="${1:-.}"
D4R="$GAME/d4r"
INI="${D4R_OPTISCALER_INI:-$GAME/OptiScaler.ini}"
script_dir=$(cd "$(dirname "$0")" 2>/dev/null && pwd -P)
if [ -z "${D4R_OPTISCALER_INI:-}" ] && [ ! -f "$INI" ] && [ "${script_dir##*/}" = d4r ]; then
  loader_dir=$(cd "$script_dir/../.." 2>/dev/null && pwd -P)
  [ ! -f "$loader_dir/OptiScaler.ini" ] || INI="$loader_dir/OptiScaler.ini"
fi
wine_file() {
  windows_path=$(printf '%s' "$1" | tr '\\' '/')
  case "$windows_path" in
    [zZ]:*) printf '%s\n' "${windows_path#??}" ;;
    [a-zA-Z]:*)
      case "$INI" in
        */drive_c/*)
          prefix="${INI%/drive_c/*}"
          drive=$(printf '%s' "${windows_path%%:*}" | tr '[:upper:]' '[:lower:]')
          drive_root=$(readlink -f "$prefix/dosdevices/$drive:" 2>/dev/null)
          if [ -n "$drive_root" ]; then printf '%s%s\n' "$drive_root" "${windows_path#??}"; else return 1; fi ;;
        *) return 1 ;;
      esac ;;
    *) printf '%s/%s\n' "$GAME" "$windows_path" ;;
  esac
}
problems=0
ok() { printf '  ok       %s\n' "$1"; }
bad() { printf '  MISSING  %s\n' "$1"; problems=$((problems + 1)); }
note() { printf '  note     %s\n' "$1"; }
misconfigured() { printf '  CONFIG   %s\n' "$1"; problems=$((problems + 1)); }
ini_value() {
  awk -v section="$1" -v key="$2" '
    { sub(/\r$/, ""); line = $0; sub(/^[ \t]+/, "", line); sub(/[ \t]+$/, "", line) }
    line ~ /^\[/ { active = (tolower(line) == "[" section "]"); next }
    active && tolower(line) ~ "^" key "[ \t]*=" {
      count++; sub(/^[^=]*=[ \t]*/, "", line); value = line
    }
    END { if (count != 1) exit 1; print value }
  ' "$INI"
}
nvngx_path=$(ini_value libraries nvngxpath 2>/dev/null) && unique_path=yes || unique_path=no
normalized_path=$(printf '%s' "$nvngx_path" | tr '\\' '/' | tr '[:upper:]' '[:lower:]')
asi_backend=no
case "$normalized_path" in */d4r.asi) asi_backend=yes ;; esac
if [ "$asi_backend" = yes ]; then
  asi_file=$(wine_file "$nvngx_path") || asi_file=
  D4R="$(dirname "$asi_file")/d4r"
fi
# ROCm: the argument, else d4r.ini's RocmDir, else the bundled d4r/rocm, else D4R_ROCM_DIR, else /opt/rocm
ini_rocm=$(sed -n 's/^[[:space:]]*RocmDir[[:space:]]*=[[:space:]]*\([^;#]*\).*/\1/p' "$D4R/d4r.ini" 2>/dev/null | tail -1 | sed 's/[[:space:]]*$//')
[ "$ini_rocm" = auto ] && ini_rocm=
case "$ini_rocm" in "~/"*) ini_rocm="$HOME/${ini_rocm#\~/}" ;; esac
[ -z "$ini_rocm" ] && [ -d "$D4R/rocm/lib" ] && ini_rocm="$D4R/rocm"
ROCM="${2:-${ini_rocm:-${D4R_ROCM_DIR:-/opt/rocm}}}"
external_backend=$(awk '
  { sub(/\r$/, ""); line = $0; sub(/^[ \t]+/, "", line); sub(/[ \t]+$/, "", line) }
  line ~ /^\[/ { dlss = (tolower(line) == "[dlss]"); next }
  dlss && tolower(line) ~ /^allowexternalbackend[ \t]*=/ {
    count++; sub(/^[^=]*=[ \t]*/, "", line); value = tolower(line)
  }
  END { if (count == 1 && value == "true") print "true" }
' "$INI" 2>/dev/null)

printf 'd4r install in %s\n' "$(cd "$GAME" 2>/dev/null && pwd || echo "$GAME")"
[ -d "$D4R" ] || { printf 'no d4r folder here; run this from the folder with the game .exe\n'; exit 1; }
ls "$GAME"/*.exe >/dev/null 2>&1 && ok "game executable next to d4r/" || note "no .exe next to d4r/ (it must be the game's main executable folder)"
if [ "$asi_backend" = yes ]; then
  loader=no
  loader_dir=$(dirname "$INI")
  for folder in "$GAME" "$loader_dir"; do
    for f in dxgi.dll winmm.dll version.dll winhttp.dll wininet.dll dbghelp.dll OptiScaler.dll; do
      [ ! -f "$folder/$f" ] || { ok "$folder/$f (verify this is your existing OptiScaler loader)"; loader=yes; }
    done
  done
  [ "$loader" = yes ] || bad 'OptiScaler proxy loader (dxgi.dll, winmm.dll, etc.; nvngx.dll mode does not load ASIs)'
else
  [ -f "$GAME/dxgi.dll" ] && ok dxgi.dll || bad 'dxgi.dll (re-extract the d4r zip)'
fi
[ -f "$INI" ] && ok "OptiScaler config: $INI" || bad "OptiScaler config: $INI"
core_name=d3d12core.dll
[ "$asi_backend" != yes ] || [ ! -f "$D4R/d4r12core.dll" ] || core_name=d4r12core.dll
for f in d3d12.dll "$core_name"; do
  [ -f "$GAME/$f" ] && ok "$f" || bad "$f (re-extract the d4r zip)"
done
for f in nvcuda.dll zluda/libcuda.so d4r.ini; do
  [ -f "$D4R/$f" ] && ok "$D4R/$f" || bad "$D4R/$f (re-extract the d4r zip)"
done
[ "$asi_backend" = yes ] || { [ -f "$GAME/d4r/nvngx.dll" ] && ok d4r/nvngx.dll || bad 'd4r/nvngx.dll (re-extract the d4r zip)'; }
# File presence alone does not establish NGX routing: OptiDllPath=d4r can still
# select system32/_nvngx.dll before d4r/nvngx.dll. Check the per-DLL override.
if [ -f "$INI" ]; then
  # Also recognize an absolute Wine Z: path to this install's shim (as used
  # by developer launches); do not accept an arbitrary nvngx.dll elsewhere.
  game_path=$(cd "$GAME" 2>/dev/null && pwd -P)
  absolute_path=$(printf 'z:%s/d4r/nvngx.dll' "$game_path" | tr '[:upper:]' '[:lower:]')
  if [ "$asi_backend" = yes ]; then
    plugins_path=$(ini_value plugins path 2>/dev/null) || plugins_path=auto
    case "$plugins_path" in auto|'') plugins_path=plugins ;; esac
    plugin_route=$(printf '%s/d4r.asi' "$plugins_path" | tr '\\' '/' | tr '[:upper:]' '[:lower:]')
    if [ "$unique_path" = yes ] && [ "$normalized_path" = "$plugin_route" ]; then
      ok "OptiScaler NGX routing: NvngxPath=$nvngx_path (d4r ASI)"
    else
      misconfigured 'NvngxPath must select d4r.asi in the configured [Plugins] Path; relaunch to retry automatic setup'
    fi
    [ -f "$asi_file" ] && ok "ASI backend: $asi_file" || bad "ASI backend: $asi_file"
    load_asi=$(ini_value plugins loadasiplugins 2>/dev/null | tr '[:upper:]' '[:lower:]')
    [ "$load_asi" = true ] || misconfigured 'Set [Plugins] LoadAsiPlugins=true'
    original_hook=$(ini_value hooks hookoriginalnvngxonly 2>/dev/null | tr '[:upper:]' '[:lower:]')
    [ "$original_hook" = true ] || misconfigured 'Set [Hooks] HookOriginalNvngxOnly=true'
  elif [ "$unique_path" = yes ] && { [ "$normalized_path" = 'd4r/nvngx.dll' ] ||
       [ "$normalized_path" = './d4r/nvngx.dll' ] || [ "$normalized_path" = "$absolute_path" ]; }; then
    ok "OptiScaler NGX routing: NvngxPath=$nvngx_path (d4r shim)"
    case "$normalized_path" in
      z:*) ;;
      *) [ "$external_backend" = true ] || note "relative NvngxPath requires the game's working directory to be this folder" ;;
    esac
  else
    misconfigured 'OptiScaler NGX routing: set [Libraries] NvngxPath=d4r\nvngx.dll (exactly once); OptiDllPath alone can load system _nvngx.dll'
    note 'd4r/ngx/_nvngx.dll is the NVIDIA core for the shim, not the OptiScaler target'
  fi
fi
[ -f "$D4R/nvngx_dlss.dll" ] && ok "d4r/nvngx_dlss.dll (NVIDIA DLSS library)" || bad "d4r/nvngx_dlss.dll: copy NVIDIA's DLSS library here"
[ -f "$D4R/ngx/_nvngx.dll" ] && ok "d4r/ngx/_nvngx.dll (NVIDIA NGX runtime)" || bad "d4r/ngx/_nvngx.dll: copy NVIDIA's NGX runtime here"
if [ -f "$D4R/nvngx_dlss.dll" ] && command -v strings >/dev/null 2>&1; then
  v=$(strings -el "$D4R/nvngx_dlss.dll" | grep -A1 '^FileVersion$' | sed -n 2p | tr ',' '.')
  case "$v" in
    310.7.*|310.9.*) ok "DLSS version $v (native kernels verified for 310.7 and 310.9)" ;;
    "") note "cannot read the DLSS version" ;;
    *) note "DLSS version $v: kernels whose code changed run without native kernels (slower)" ;;
  esac
fi

# Optional features are only requirements when selected. Their absence must not
# make an ordinary SR/RR installation fail this checker.
use_fake=$(ini_value fakenvapi usefakenvapi 2>/dev/null | tr '[:upper:]' '[:lower:]')
fg_enabled=$(ini_value framegen enabled 2>/dev/null | tr '[:upper:]' '[:lower:]')
fg_output=$(ini_value framegen fgoutput 2>/dev/null | tr '[:upper:]' '[:lower:]')
fg_input=$(ini_value framegen fginput 2>/dev/null | tr '[:upper:]' '[:lower:]')
feature_file() {
  feature_path=$(ini_value libraries "$1" 2>/dev/null) || feature_path=auto
  case "$feature_path" in auto|'')
    if [ -n "$use_fake" ]; then
      feature_base=$(ini_value libraries optidllpath 2>/dev/null) || feature_base=auto
      case "$feature_base" in auto|'') feature_base="$GAME/OptiScaler" ;;
        *) feature_base=$(wine_file "$feature_base") || feature_base= ;; esac
      feature_path="$feature_base/$2"
    else feature_path="$D4R/$2"; fi ;;
    *) feature_path=$(wine_file "$feature_path") || feature_path= ;; esac
  [ ! -d "$feature_path" ] || feature_path="$feature_path/$2"
  [ -f "$feature_path" ] && ok "$2 (optional feature)" || bad "$2 (selected optional feature; check its library override)"
}
if [ "$fg_enabled" = true ]; then
  case "$fg_input" in ''|auto|nofg) misconfigured 'FG is enabled but no FGInput is selected' ;; esac
  case "$fg_output" in
    xefg) feature_file xefgpath libxess_fg.dll; feature_file xellpath libxell.dll
          note 'XeFG requires borderless SDR/HDR10, Reflex markers and FG activation after restart' ;;
    fsrfg) feature_file ffxdx12fgpath amd_fidelityfx_framegeneration_dx12.dll
           if [ -n "$use_fake" ]; then feature_file ffxdx12path amd_fidelityfx_loader_dx12.dll; fi ;;
    ''|auto|nofg) misconfigured 'FG is enabled but no FGOutput is selected' ;;
    *) note "custom FG output: $fg_output (not checked)" ;;
  esac
fi
if [ -n "$use_fake" ]; then
  [ "$use_fake" = false ] || misconfigured 'd4r requires fakenvapi.UseFakenvapi=false with OptiScaler v10 to retain Proton GPU topology'
  note 'OptiScaler v10 uses integrated latency hooks on Proton NVAPI; no external fakenvapi.dll is required'
fi
nvapi_path=$(ini_value libraries nvapipath 2>/dev/null | tr '\\' '/' | tr '[:upper:]' '[:lower:]')
nvapi_override=$(ini_value nvapi overridenvapidll 2>/dev/null | tr '[:upper:]' '[:lower:]')
if [ "$nvapi_override" = true ]; then
  case "$nvapi_path" in
    */d4r_nvapi64.dll)
      feature_file nvapipath d4r_nvapi64.dll
      router_path=$(ini_value libraries nvapipath 2>/dev/null)
      router_path=$(wine_file "$router_path") || router_path=
      [ -f "$(dirname "$router_path")/fakenvapi.dll" ] && ok 'fakenvapi.dll (latency provider)' || bad 'fakenvapi.dll beside d4r_nvapi64.dll'
      ntdll_hooks=$(ini_value hooks usentdllhooks 2>/dev/null | tr '[:upper:]' '[:lower:]')
      [ "$ntdll_hooks" = false ] || misconfigured 'd4r latency router requires Hooks.UseNtdllHooks=false'
      note 'Latency routing configured; confirm the active AL2/XeLL/LFX method in OptiScaler at runtime'
      ;;
    *) note 'custom NVAPI override; d4r topology compatibility is not checked' ;;
  esac
fi

found=
for dir in "$ROCM/lib" /opt/rocm/lib /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu; do
  [ -e "$dir/libamdhip64.so.7" ] && { found="$dir"; break; }
done
[ -n "$found" ] && ok "ROCm HIP runtime ($found/libamdhip64.so.7)" || bad "ROCm HIP runtime 7.x (libamdhip64.so.7); re-extract the d4r zip, which includes it in d4r/rocm"
[ -e /dev/kfd ] && ok "/dev/kfd (ROCm compute device)" || bad "/dev/kfd: the amdgpu compute interface is not available"
target="" best_simds=0
for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
  s=$(sed -n 's/^simd_count //p' "$props" 2>/dev/null); t=$(sed -n 's/^gfx_target_version //p' "$props" 2>/dev/null)
  [ -n "$s" ] && [ "$s" -gt "$best_simds" ] && [ -n "$t" ] && [ "$t" != 0 ] && { target="$t"; best_simds="$s"; }
done
if [ -n "$target" ]; then
  arch=$(printf 'gfx%d%d%x' $((target / 10000)) $(((target / 100) % 100)) $((target % 100)))
  if [ -d "$D4R/kernels/$arch" ]; then ok "GPU $arch: native kernels present"
  else note "GPU $arch: no native kernels for it in this release (DLSS runs, much slower)"; fi
fi

if [ "$external_backend" = true ]; then
  note "AllowExternalBackend requires the patched OptiScaler DLL; keep native AMD identity visible for FSR4"
  printf '\nSteam launch options for this patched setup:\n  PROTON_ENABLE_NVAPI=1 DXVK_NVAPI_ALLOW_OTHER_DRIVERS=1 DXVK_NVAPI_GPU_ARCH=AD100 WINE_HIDE_AMD_GPU=0 %%command%%\n'
else
  printf '\nSteam launch options for this game:\n  PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %%command%%\n'
fi
[ "$problems" -eq 0 ] && printf '\nFiles and OptiScaler NGX routing checks passed (runtime initialization is not tested).\n' || printf '\n%d thing(s) to fix above.\n' "$problems"
[ "$problems" -eq 0 ]
