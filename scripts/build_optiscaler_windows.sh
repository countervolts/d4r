#!/usr/bin/env bash
set -euo pipefail

# Builds OptiScaler for the native Windows release: the source commit pinned in
# packaging/optiscaler-release.json with patches/optiscaler-windows/ applied (Spoofing.ForceNvidia: upstream v10
# turns DLSS off on every non-NVIDIA GPU under native Windows). Prints the output folder, which holds
# OptiScaler.dll, the patched OptiScaler.ini and the applied patches; it is reused while the commit and patches
# are unchanged. Needs Git for Windows and Visual Studio 2022 with the C++ workload (toolset v143).
# Env: OPTISCALER_SRC (clone, default build/optiscaler-src), MSBUILD (default: found with vswhere)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
SRC="${OPTISCALER_SRC:-$ROOT/build/optiscaler-src}"
COMMIT="$("$PYTHON" -c "import json, sys; print(json.load(open(sys.argv[1]))['source_commit'])" \
  "$ROOT/packaging/optiscaler-release.json")"
PATCHES=("$ROOT"/patches/optiscaler-windows/*.patch)
KEY="$( (echo "$COMMIT"; cat "${PATCHES[@]}") | sha256sum | cut -c1-16)"
OUT="$ROOT/build/optiscaler-d4r/$KEY"

if [[ -f "$OUT/OptiScaler.dll" && -f "$OUT/OptiScaler.ini" ]]; then
  echo "$OUT"
  exit 0
fi

if [[ -z "${MSBUILD:-}" ]]; then
  VSWHERE="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
  MSBUILD="$("$VSWHERE" -version '[17.0,18.0)' -products '*' -requires Microsoft.Component.MSBuild \
    -find 'MSBuild\**\Bin\amd64\MSBuild.exe' | head -1 | tr -d '\r')"
  [[ -n "$MSBUILD" ]] || { echo "Visual Studio 2022 (MSBuild, toolset v143) not found; set MSBUILD" >&2; exit 1; }
fi

# The FidelityFX SDK submodules have paths past MAX_PATH.
[[ -d "$SRC/.git" ]] || git -c core.longpaths=true clone -q https://github.com/optiscaler/OptiScaler.git "$SRC"
git -C "$SRC" config core.longpaths true
git -C "$SRC" cat-file -e "$COMMIT^{commit}" 2>/dev/null || git -C "$SRC" fetch -q origin "$COMMIT"
# -f: the clone is a build cache; the patches are reapplied below.
git -C "$SRC" checkout -q -f "$COMMIT"
git -C "$SRC" clean -q -fdx -e x64/
git -C "$SRC" submodule update --init --recursive --depth 1 -q
git -C "$SRC" submodule foreach -q --recursive 'git config core.longpaths true; git checkout -q -f -- .'
git -C "$SRC" apply "${PATCHES[@]}"

"$MSBUILD" -m -nologo -verbosity:minimal -p:Configuration=Release -p:Platform=x64 "$(cygpath -w "$SRC/OptiScaler.sln")" >&2

mkdir -p "$OUT.tmp"
cp "$SRC/x64/Release/a/OptiScaler.dll" "$SRC/OptiScaler.ini" "$OUT.tmp/"
cp "${PATCHES[@]}" "$OUT.tmp/"
rm -rf "$OUT"
mv "$OUT.tmp" "$OUT"
echo "$OUT"
