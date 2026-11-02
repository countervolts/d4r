#!/usr/bin/env bash
# Removes d4r from a game: beside the proxy release's d4r/ or inside the ASI's d4r/ (like install.sh).
# Only files d4r installed are removed; a game's own DLLs that d4r's ASI setup replaced are restored from their
# .d4r-backup copies. The shared cache (~/.cache/d4r) is kept unless asked: other games with d4r reuse it.
# The whole script is read before anything is deleted (it may delete itself).
{
set -euo pipefail

fail() { printf 'uninstall.sh: %s\n' "$*" >&2; exit 1; }
usage() {
  printf '%s\n' 'Usage: bash uninstall.sh [options] [extracted-release-directory]' \
    'Default directory: the one containing this script, regardless of your working directory.' \
    'Close the game first. Run it from a terminal and it asks before removing anything.' \
    '  --yes          remove without asking (also the behaviour without a terminal); keeps the cache' \
    '  --purge-cache  also delete the shared cache: NVIDIA DLLs, engine models and compiled kernels' \
    '                 (~/.cache/d4r, or D4R_CACHE_DIR), which other games with d4r reuse' \
    '  --dry-run      only list what would be removed'
}
YES=0 PURGE=ask DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -y|--yes) YES=1; shift ;;
    --purge-cache) PURGE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -*) usage >&2; exit 2 ;;
    *) break ;;
  esac
done
[[ $# -le 1 ]] || { usage >&2; exit 2; }
ROOT=$(cd -- "${1:-$(dirname -- "${BASH_SOURCE[0]}")}" && pwd)
RUNTIME="$ROOT/d4r"
[[ ! -f "$ROOT/d4r.ini" ]] || RUNTIME="$ROOT"
[[ -f "$RUNTIME/nvngx.dll" && -f "$RUNTIME/d4r.ini" ]] || fail "no d4r runtime found in $ROOT"
CACHE="${D4R_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/d4r}"

interactive() { (( YES == 0 )) && [[ -t 0 && -t 1 ]]; }
ask() {   # ask "QUESTION" DEFAULT(y|n): 0 for yes
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
has() { grep -aqi -- "$2" "$1" 2>/dev/null; }   # has FILE TEXT: the file contains TEXT (any case)
sha() { sha256sum -- "$1" | cut -d' ' -f1; }

# The plan: paths to remove, files to restore (backup -> destination), INIs whose d4r route is reset, and notes.
REMOVE=() RESTORE_FROM=() RESTORE_TO=() INIS=() NOTES=()
remove() { [[ -e "$1" || -L "$1" ]] && REMOVE+=("$1"); return 0; }
note() { NOTES+=("$*"); }

if [[ -f "$RUNTIME/d4r12core.dll" || -f "$(dirname -- "$RUNTIME")/d4r.asi" ]]; then
  # --- ASI: d4r.asi + d4r/ in OptiScaler's plugin folder; d4r-setup put the D3D12 pair beside the game -----------
  MODE=asi
  PLUGIN_DIR=$(dirname -- "$RUNTIME")
  remove "$PLUGIN_DIR/d4r.asi"
  GAME=''
  d=$PLUGIN_DIR
  for _ in 1 2 3 4 5; do
    if [[ -f "$d/d4r12core.dll" ]]; then GAME=$d; break; fi
    [[ "$d" != / ]] || break
    d=$(dirname -- "$d")
  done
  if [[ -n "$GAME" ]]; then
    for name in d3d12.dll d4r12core.dll; do
      dst="$GAME/$name" original=''
      # the oldest backup that is not one of d4r's own (an update backs up the previous d4r version too)
      while IFS= read -r backup; do
        [[ -n "$backup" ]] || continue
        if [[ -z "$original" && "$name" == d3d12.dll ]] && ! has "$backup" d4r12core; then original=$backup
        else remove "$backup"; fi
      done < <(ls -1tr -- "$dst".d4r-backup* 2>/dev/null)
      if [[ -n "$original" ]]; then
        RESTORE_FROM+=("$original"); RESTORE_TO+=("$dst")
      elif [[ "$name" == d4r12core.dll ]] || has "$dst" d4r12core; then
        remove "$dst"
      elif [[ -f "$dst" ]]; then
        note "kept $dst: it is not d4r's"
      fi
    done
  else
    note "the game folder with d4r12core.dll was not found near $PLUGIN_DIR: setup never completed, nothing to restore"
  fi
  # OptiScaler.ini routes NvngxPath to d4r.asi: back to auto, where it still points at this plugin. The INI can be
  # beside the game, beside the loader, or in a Proton prefix (GE-Proton's system32/umu).
  PLUGIN_WIN=''
  [[ ! -f "$RUNTIME/installation-plugin.txt" ]] || PLUGIN_WIN=$(tr -d '\r\n' < "$RUNTIME/installation-plugin.txt")
  candidates=("$PLUGIN_DIR/OptiScaler.ini" "$(dirname -- "$PLUGIN_DIR")/OptiScaler.ini")
  [[ -z "$GAME" ]] || candidates+=("$GAME/OptiScaler.ini")
  for steam in "$HOME/.local/share/Steam" "$HOME/.steam/steam" "$HOME/.var/app/com.valvesoftware.Steam/data/Steam"; do
    for f in "$steam"/steamapps/compatdata/*/pfx/drive_c/windows/system32/umu/OptiScaler.ini; do candidates+=("$f"); done
  done
  declare -A seen=()
  for f in "${candidates[@]}"; do
    [[ -f "$f" ]] || continue
    key=$(realpath -- "$f"); [[ -z "${seen[$key]:-}" ]] || continue; seen[$key]=1
    value=$(sed -n 's/^[[:space:]]*NvngxPath[[:space:]]*=[[:space:]]*//Ip' "$f" | tr -d '\r' | tail -n 1)
    if [[ -n "$PLUGIN_WIN" && "${value,,}" == "${PLUGIN_WIN,,}" ]] ||
       [[ -z "$PLUGIN_WIN" && "${value,,}" == *d4r.asi && ( "$f" != */compatdata/* ) ]]; then
      INIS+=("$f")
    fi
  done
  (( ${#INIS[@]} )) || note 'no OptiScaler.ini routes to this d4r.asi (if OptiScaler fails to find DLSS, set [Libraries] NvngxPath=auto)'
  remove "$RUNTIME"
else
  # --- proxy: OptiScaler as dxgi.dll, d4r's vkd3d-proton pair and d4r/ in the game folder ----------------------
  MODE=proxy
  GAME=$(dirname -- "$RUNTIME")
  manifest="$RUNTIME/source/optiscaler.json"
  ours=0
  if [[ -f "$GAME/dxgi.dll" ]]; then
    if [[ -f "$manifest" ]] && grep -q "\"$(sha "$GAME/dxgi.dll")\"" "$manifest"; then ours=1
    elif has "$GAME/dxgi.dll" optiscaler; then ours=1
    fi
  fi
  if (( ours )); then
    remove "$GAME/dxgi.dll"; remove "$GAME/OptiScaler.ini"; remove "$GAME/OptiScaler.log"; remove "$GAME/OptiScaler"
  elif [[ -f "$GAME/dxgi.dll" ]]; then
    note "kept $GAME/dxgi.dll, OptiScaler.ini and OptiScaler/: that dxgi.dll is not OptiScaler"
  fi
  for name in d3d12.dll d3d12core.dll; do
    if has "$GAME/$name" vkd3d; then remove "$GAME/$name"
    elif [[ -f "$GAME/$name" ]]; then note "kept $GAME/$name: it is not d4r's vkd3d-proton"; fi
  done
  for name in amd_fidelityfx_dx12.dll amd_fidelityfx_upscaler_dx12.dll; do
    [[ ! -f "$GAME/$name" ]] ||
      note "kept $GAME/$name (older d4r releases put it there, but games ship it too; verify the game's files in Steam if FSR breaks)"
  done
  remove "$GAME/D4R_README.txt"; remove "$GAME/install.sh"
  remove "$RUNTIME"
  [[ "$(realpath -- "${BASH_SOURCE[0]}")" != "$GAME/uninstall.sh" ]] || remove "$GAME/uninstall.sh"
fi

if [[ $MODE == asi ]]; then echo "d4r (ASI plugin in $PLUGIN_DIR${GAME:+, game folder $GAME}):"
else echo "d4r (in $GAME):"; fi
(( ${#REMOVE[@]} )) && { echo 'Remove:'; printf '  %s\n' "${REMOVE[@]}"; }
for i in "${!RESTORE_TO[@]}"; do printf 'Restore the game'"'"'s own %s\n  from %s\n' "${RESTORE_TO[$i]}" "${RESTORE_FROM[$i]}"; done
(( ${#INIS[@]} )) && { echo 'Set [Libraries] NvngxPath=auto (a backup is kept beside it) in:'; printf '  %s\n' "${INIS[@]}"; }
(( ${#NOTES[@]} )) && printf 'Note: %s\n' "${NOTES[@]}"
if [[ "$PURGE" == ask ]]; then
  PURGE=0
  if [[ -d "$CACHE" ]] && interactive; then
    size=$(du -sh -- "$CACHE" 2>/dev/null | cut -f1)
    ask "Also delete the shared cache $CACHE (${size:-?}: NVIDIA DLLs, engine models, kernels; other games with d4r reuse it)?" n && PURGE=1
  fi
fi
(( PURGE == 0 )) || [[ ! -d "$CACHE" ]] || echo "Remove the shared cache: $CACHE"
if (( DRY )); then echo 'Dry run: nothing changed.'; exit 0; fi
ask 'Uninstall d4r now?' y || { echo 'Nothing changed.'; exit 0; }

for i in "${!INIS[@]}"; do
  f=${INIS[$i]}
  cp -p -- "$f" "$f.d4r-uninstall-backup"
  cr=''; grep -q $'\r$' "$f" && cr=$'\r'   # keep the file's own line endings
  sed -i -E "s/^([[:space:]]*NvngxPath[[:space:]]*=).*\$/\\1auto$cr/I" "$f"
  echo "Reset NvngxPath: $f"
done
for i in "${!RESTORE_TO[@]}"; do
  mv -f -- "${RESTORE_FROM[$i]}" "${RESTORE_TO[$i]}"
  echo "Restored: ${RESTORE_TO[$i]}"
done
for p in "${REMOVE[@]}"; do
  rm -rf -- "$p"
done
(( ${#REMOVE[@]} == 0 )) || echo "Removed ${#REMOVE[@]} d4r item(s)."
if (( PURGE )) && [[ -d "$CACHE" ]]; then rm -rf -- "$CACHE"; echo "Removed the shared cache: $CACHE"; fi
echo
echo 'd4r is uninstalled. In Steam, remove PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 (and any D4R_ variables)'
echo 'from the game'"'"'s launch options. If the game shipped its own dxgi.dll or d3d12.dll, use'
echo 'Properties > Installed Files > Verify integrity of game files to get them back.'
exit 0
}
