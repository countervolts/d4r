#!/usr/bin/env bash
# Release installer: copy this beside D4R_README.txt and the extracted d4r/ folder.
set -euo pipefail
DLSS_URL=https://raw.githubusercontent.com/NVIDIA/DLSS/a291cc7d2cc642a51566f3dfd5376f635cd1b284/lib/Windows_x86_64/rel/nvngx_dlss.dll
DLSS_SHA256=be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e
DRIVER_URL=https://us.download.nvidia.com/Windows/596.36/596.36-desktop-win10-win11-64bit-international-dch-whql.exe
DRIVER_SHA256=4a1793d2bc7792f43270fe8286ec381ae24409fe29049dbe1eb8530ed89e19cf
NGX_SHA256=0e3423b3b7afa1019c7ecac053a8408df8e5d9d9d904434705c6aaeacec02fef

fail() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  printf '%s\n' 'Usage: bash install.sh [extracted-release-directory]' \
    'Default: the directory containing this script, regardless of your working directory.' \
    'Downloads official NVIDIA DLSS 310.7.0 and NGX from driver 596.36 (~915 MiB).' \
    'Requires Bash, curl, sha256sum, and 7zz, 7z or 7za. No sudo needed.' \
    'Existing matching DLLs are kept; differing DLLs are backed up. NVIDIA license terms apply.'
  exit 0
fi
[[ $# -le 1 ]] || fail 'Usage: bash install.sh [extracted-release-directory]'
ROOT=$(cd -- "${1:-$(dirname -- "${BASH_SOURCE[0]}")}" && pwd)
[[ -f "$ROOT/d4r/nvngx.dll" && -f "$ROOT/d4r/d4r.ini" ]] || fail "Extract the release ZIP first; no d4r runtime found in $ROOT"
command -v sha256sum >/dev/null || fail 'Missing sha256sum (install coreutils).'
valid() { [[ -f "$1" ]] && [[ $(sha256sum -- "$1" | cut -d' ' -f1) == "$2" ]]; }
need_dlss=1; need_ngx=1
valid "$ROOT/d4r/nvngx_dlss.dll" "$DLSS_SHA256" && need_dlss=0
valid "$ROOT/d4r/ngx/_nvngx.dll" "$NGX_SHA256" && need_ngx=0
if (( need_dlss == 0 && need_ngx == 0 )); then
  echo 'NVIDIA DLLs are already installed and verified.'
  exit 0
fi
command -v curl >/dev/null || fail 'Missing curl. Install it and run this script again.'
if (( need_ngx )); then
  SEVENZIP=''
  for tool in 7zz 7z 7za; do
    if command -v "$tool" >/dev/null; then SEVENZIP=$tool; break; fi
  done
  [[ -n "$SEVENZIP" ]] || fail 'Missing 7-Zip. Install 7zip (or p7zip) and run this script again.'
fi
TMP=$(mktemp -d)
STAGED=''
trap '[[ -z "$STAGED" ]] || rm -f -- "$STAGED"; rm -rf -- "$TMP"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
download() {
  curl --fail --location --show-error --retry 3 --connect-timeout 30 \
    --proto '=https' --proto-redir '=https' --output "$2" "$1" || fail "Download failed: $1. Run the installer again to retry."
}
verify() { valid "$1" "$2" || fail "SHA-256 mismatch for $1; no DLLs installed."; }
echo 'Downloading NVIDIA files under NVIDIA license terms:'
echo 'DLSS: https://github.com/NVIDIA/DLSS/blob/v310.7.0/LICENSE.txt'
echo 'Driver: https://www.nvidia.com/en-us/drivers/nvidia-license/'
if (( need_dlss )); then
  echo 'Downloading DLSS 310.7.0...'
  download "$DLSS_URL" "$TMP/nvngx_dlss.dll"
  verify "$TMP/nvngx_dlss.dll" "$DLSS_SHA256"
fi
if (( need_ngx )); then
  echo 'Downloading NVIDIA driver 596.36 (~915 MiB); extracting NGX without running the installer...'
  download "$DRIVER_URL" "$TMP/driver.exe"
  verify "$TMP/driver.exe" "$DRIVER_SHA256"
  "$SEVENZIP" e -y -bso0 -bsp0 "-o$TMP/ngx" "$TMP/driver.exe" 'Display.Driver/_nvngx.dll' || fail 'Could not extract NGX; no DLLs installed.'
  verify "$TMP/ngx/_nvngx.dll" "$NGX_SHA256"
fi
# Verify all downloads before modifying either destination. Stage on the destination
# filesystem for atomic replacement and retain a unique backup of a differing file.
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
(( need_dlss == 0 )) || place "$TMP/nvngx_dlss.dll" "$ROOT/d4r/nvngx_dlss.dll"
(( need_ngx == 0 )) || place "$TMP/ngx/_nvngx.dll" "$ROOT/d4r/ngx/_nvngx.dll"
echo 'NVIDIA DLLs installed and verified. Follow D4R_README.txt for Steam launch options.'
