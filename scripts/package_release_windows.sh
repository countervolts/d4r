#!/usr/bin/env bash
# Builds the native Windows release: a zip whose contents are extracted into the folder that holds a game's main
# .exe, like an OptiScaler release (packaging/WINDOWS_README.txt describes the result). The Windows counterpart of
# scripts/package_release.sh's installer variant: NVIDIA's DLLs are not included, the native kernels (with the
# texture kernels built from NVIDIA's PTX) are. install.cmd (install.ps1) downloads NVIDIA's pinned DLLs and builds
# the native D3D12 engine's models from them on the user's machine (packaging/build_engine_models.py): Ray
# Reconstruction D/E/F and Super Resolution K, M and L. Runs in Git Bash (or MSYS2) on Windows.
#
# usage: scripts/package_release_windows.sh [OUT_DIR]          (default: dist/)
#
# Inputs (environment):
#   D4R_OPTISCALER   OptiScaler v10 archive (.7z/.zip) or extracted folder;
#                    default: checksum-pinned packaging/optiscaler-release.json
#   D4R_OPTISCALER_LICENSE  OptiScaler's LICENSE (GPL-3.0) text                     (required)
#   D4R_OPTISCALER_BUILD  its OptiScaler.dll and INI rebuilt with patches/optiscaler-windows (Visual Studio 2022)
#                    default: scripts/build_optiscaler_windows.sh's output, built when missing
#   D4R_ZLUDA_DIR    Windows ZLUDA build with patches/zluda applied (nvcuda.dll)     (required)
#   D4R_ZLUDA_SRC    ZLUDA source checkout, for its license files      default: D4R_ZLUDA_DIR/../..
#   D4R_GLSLANG      glslang Windows release archive or folder (bin/glslangValidator.exe), and
#   D4R_GLSLANG_LICENSE  its LICENSE.txt                                             (scripts/stage_engine_kit.sh)
#   D4R_DXC          DirectX Shader Compiler release archive or folder              (scripts/stage_engine_kit.sh)
#   D4R_VCRUNTIME_DIR  Visual C++ runtime for glslangValidator.exe                  (scripts/stage_engine_kit.sh)
# The native kernels (scripts/stage_native_kernels.sh):
#   D4R_ROCM_DIR     AMD HIP SDK with clang                             default: the newest C:/ROCm/<version>
#   D4R_GPU_ARCHS    GPU targets                     default gfx1030-gfx1036, gfx1100-gfx1103, gfx1200, gfx1201
#   D4R_BUNDLE_DLSS  nvngx_dlss.dll 310.7.0 (install.ps1's pin) for the texture kernels and the kernel manifest;
#   D4R_BUNDLE_DLSSD nvngx_dlssd.dll 310.9.1 (install.ps1's pin) for the Ray Reconstruction texture kernels;
#                    default: install.ps1's verified copies in %LOCALAPPDATA%\d4r\nvidia. Never packaged.
#   D4R_BUNDLE_TEX   texture kernels, <target folder>/ and accuracy/<target folder>/: those it lacks are built into it
#                    (default: a temporary folder; keep one to reuse the slow part between releases)
#   D4R_ZLUDA_EMIT   ZLUDA's d4r_emit.exe (`cargo build --release -p ptx --example d4r_emit` in the patched ZLUDA)
#                    default: D4R_ZLUDA_DIR/examples/d4r_emit.exe
#   D4R_LLVM_AS      the llvm-as of ZLUDA's LLVM (the HIP SDK's clang is newer; kernels/tex/build_tex.sh)
#                    default: the newest one in D4R_ZLUDA_SRC/target/release/build/llvm_zluda-*/out/build/bin
#   D4R_PACKAGE_KERNEL_JOBS  independent target builds at once (default 1)
# Other:
#   D4R_SKIP_BUILD=1 packages the shim, CUDA bridge, NVAPI and Vulkan test app already in build/
#   D4R_STRIP        strip (binutils), for the shim, bridge and NVAPI's debug information   default: MSYS2's
#   D4R_BUILD_INFO   build-toolchain/provenance text for d4r/source/BUILD_INFO.txt   default: the tools found here
#   SOURCE_DATE_EPOCH  timestamp given to every packaged file                default: the last commit's
#   D4R_VERSION      the name's version (a nightly: nightly-YYYYMMDD)       default: packaging/VERSION
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${1:-$ROOT/dist}")"
VERSION="${D4R_VERSION:-$(cat "$ROOT/packaging/VERSION")}"
NAME="d4r-$VERSION-windows"
STAGE="$OUT/$NAME"
: "${D4R_ZLUDA_DIR:?set D4R_ZLUDA_DIR to the Windows ZLUDA build (target/release with nvcuda.dll)}"
: "${D4R_OPTISCALER_LICENSE:?set D4R_OPTISCALER_LICENSE to OptiScaler\'s GPL-3.0 license text}"
ZLUDA_SRC="${D4R_ZLUDA_SRC:-$D4R_ZLUDA_DIR/../..}"
PYTHON="$(command -v python3 || command -v python)"
export PYTHON
D4R_OPTISCALER="${D4R_OPTISCALER:-$("$PYTHON" "$ROOT/scripts/fetch_optiscaler.py")}"
newest() { ls -d $1 2>/dev/null | sort -V | tail -1; }   # newest PATTERN: the last match by version order
# MSYS2's MinGW toolchain, which the build scripts and BUILD_INFO use, when this shell does not have it
command -v "${MINGW_CXX:-x86_64-w64-mingw32-g++}" >/dev/null || PATH="/c/msys64/mingw64/bin:$PATH"
ROCM="${D4R_ROCM_DIR:-$(newest '/c/ROCm/*')}"
STRIP="${D4R_STRIP:-$(command -v strip || newest '/c/msys64/mingw64/bin/strip.exe')}"
export D4R_ZLUDA_EMIT="${D4R_ZLUDA_EMIT:-$D4R_ZLUDA_DIR/examples/d4r_emit.exe}"
export D4R_LLVM_AS="${D4R_LLVM_AS:-$(newest "$ZLUDA_SRC/target/release/build/llvm_zluda-*/out/build/bin/llvm-as.exe")}"
NVIDIA_CACHE="$(cygpath -u "${LOCALAPPDATA:-$HOME/AppData/Local}")/d4r/nvidia"
export D4R_BUNDLE_DLSS="${D4R_BUNDLE_DLSS:-$NVIDIA_CACHE/nvngx_dlss.dll}"
export D4R_BUNDLE_DLSSD="${D4R_BUNDLE_DLSSD:-$NVIDIA_CACHE/nvngx_dlssd.dll}"
export D4R_DLSS_DLLS="$(cygpath -u "$D4R_BUNDLE_DLSS")"   # a colon-separated list: no drive letter

if [[ "${D4R_SKIP_BUILD:-0}" != 1 ]]; then
  "$ROOT/scripts/build_d4r_nvngx_shim.sh" >/dev/null
  "$ROOT/scripts/build_windows_nvcuda_bridge.sh" >/dev/null
  "$ROOT/scripts/build_windows_nvapi.sh" >/dev/null
  "$ROOT/scripts/build_vulkan_dlss_harness.sh" >/dev/null
fi
for f in "$ROOT/build/d4r_nvngx.dll" "$ROOT/build/windows-nvcuda/nvcuda.dll" "$ROOT/build/windows-nvapi/nvapi64.dll" \
         "$ROOT/build/vulkan_dlss_harness.exe" "$D4R_ZLUDA_DIR/nvcuda.dll" "$D4R_OPTISCALER_LICENSE" "$D4R_ZLUDA_EMIT" \
         "$D4R_LLVM_AS" "$ROCM/lib/llvm/bin/clang++.exe" "$STRIP" "$D4R_BUNDLE_DLSS" "$D4R_BUNDLE_DLSSD"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
# The kernels are built from, and their manifest accepts, the DLLs install.ps1 downloads: refuse others.
pinned() { sed -n "s/^\$$1 = '\([0-9a-f]*\)'.*/\1/p" "$ROOT/install.ps1"; }
# hashed from stdin: sha256sum escapes a file name with a backslash (a Windows path) by prefixing the hash with "\"
[[ "$(sha256sum < "$D4R_BUNDLE_DLSS" | cut -d' ' -f1)" == "$(pinned DlssSha256)" ]] ||
  { echo "D4R_BUNDLE_DLSS must be the DLSS DLL pinned in install.ps1" >&2; exit 2; }
[[ "$(sha256sum < "$D4R_BUNDLE_DLSSD" | cut -d' ' -f1)" == "$(pinned DlssdSha256)" ]] ||
  { echo "D4R_BUNDLE_DLSSD must be the DLSS Ray Reconstruction DLL pinned in install.ps1" >&2; exit 2; }

rm -rf "$STAGE"
mkdir -p "$STAGE/d4r/zluda" "$STAGE/d4r/nvapi" "$STAGE/d4r/ngx" "$STAGE/d4r/licenses" "$STAGE/d4r/source"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# OptiScaler, as dxgi.dll with its INI preconfigured for d4r
if [[ -d "$D4R_OPTISCALER" ]]; then
  OPTI="$D4R_OPTISCALER"
else
  OPTI="$TMP/optiscaler"
  mkdir -p "$OPTI"
  case "$D4R_OPTISCALER" in
    *.7z) if command -v 7z >/dev/null; then 7z x -y -o"$OPTI" "$D4R_OPTISCALER" >/dev/null
          else "$SYSTEMROOT/System32/tar.exe" -x -f "$(cygpath -w "$D4R_OPTISCALER")" -C "$(cygpath -w "$OPTI")"; fi ;;
    *.zip) unzip -q -o "$D4R_OPTISCALER" -d "$OPTI" ;;
    *) echo "D4R_OPTISCALER must be a .7z, a .zip or a folder" >&2; exit 2 ;;
  esac
fi
# Upstream v10 turns DLSS off on every non-NVIDIA GPU under native Windows: dxgi.dll and the INI come from the
# release's source commit with patches/optiscaler-windows (Spoofing.ForceNvidia), the rest from the release.
OPTI_BUILD="${D4R_OPTISCALER_BUILD:-$("$ROOT/scripts/build_optiscaler_windows.sh")}"
"$PYTHON" "$ROOT/scripts/stage_optiscaler.py" --patched "$OPTI_BUILD" "$OPTI" "$STAGE"   "$ROOT/packaging/optiscaler-windows.settings"

# d4r itself: the shim, which runs the native D3D12 engine in the game; the CUDA bridge, ZLUDA and NVAPI, which
# run NGX's CUDA backend for what the engine does not cover and for the model builder's test frames. Their debug
# information stays out of the release (the shim's is most of its size).
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" log -1 --format=%ct)}"   # strip stamps PE files with it
"$STRIP" -o "$STAGE/d4r/nvngx.dll" "$ROOT/build/d4r_nvngx.dll"
# d3dhip's network (built with the shim when a HIP SDK is present); the shim loads it on first use
if [[ -f "$ROOT/build/d4r_hip_net.dll" ]]; then
  "$STRIP" -o "$STAGE/d4r/d4r_hip_net.dll" "$ROOT/build/d4r_hip_net.dll"
fi
"$STRIP" -o "$STAGE/d4r/nvcuda.dll" "$ROOT/build/windows-nvcuda/nvcuda.dll"
"$STRIP" -o "$STAGE/d4r/nvapi/nvapi64.dll" "$ROOT/build/windows-nvapi/nvapi64.dll"
cp "$D4R_ZLUDA_DIR/nvcuda.dll" "$STAGE/d4r/zluda/nvcuda.dll"
# a ZLUDA built with llvm-mingw (scripts/windows/build-zluda-windows.ps1) imports its C++ runtime from beside it
for runtime in libc++.dll libunwind.dll; do
  [[ -f "$D4R_ZLUDA_DIR/$runtime" ]] && cp "$D4R_ZLUDA_DIR/$runtime" "$STAGE/d4r/zluda/$runtime"
done
cp "$ROOT/packaging/d4r.ini" "$STAGE/d4r/d4r.ini"
cp "$ROOT/packaging/d4r-check.ps1" "$ROOT/packaging/d4r-check.cmd" "$STAGE/d4r/"
printf 'install.cmd puts NVIDIA'"'"'s NGX runtime, _nvngx.dll, in this folder.\r\n' > "$STAGE/d4r/ngx/README.txt"
cp "$ROOT/install.ps1" "$ROOT/install.cmd" "$STAGE/"
D4R_KIT_PLATFORM=windows "$ROOT/scripts/stage_engine_kit.sh" "$STAGE/d4r"

# The native kernels for the CUDA path, as in the Linux release. kernels/build.sh extracts the DLLs' PTX into the
# repository for each target unless given it: once here, which also lets targets build in parallel.
export D4R_DLSS_PTX_DIR="$TMP/ptx" D4R_DLSSD_PTX_DIR="$TMP/ptx-rr"
"$PYTHON" "$ROOT/kernels/tools/extract_dlss_ptx.py" "$D4R_BUNDLE_DLSS" "$D4R_DLSS_PTX_DIR" >/dev/null
"$PYTHON" "$ROOT/kernels/tools/extract_dlss_ptx.py" "$D4R_BUNDLE_DLSSD" "$D4R_DLSSD_PTX_DIR" >/dev/null
ARCHS="${D4R_GPU_ARCHS:-gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201}"
JOBS="${D4R_PACKAGE_KERNEL_JOBS:-1}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "D4R_PACKAGE_KERNEL_JOBS must be positive" >&2; exit 2; }
# The texture kernels of every set (the Linux release takes them prebuilt): each target folder D4R_BUNDLE_TEX lacks
# is built there, so a kept D4R_BUNDLE_TEX makes the next release quick. RDNA2's are built with its layers.
export D4R_BUNDLE_TEX="${D4R_BUNDLE_TEX:-$TMP/tex}"
pending=() failed=0
for arch in $ARCHS; do
  [[ "$arch" == gfx103* ]] && continue
  folders=("$arch"); [[ "$arch" == gfx12* ]] && folders+=("$arch-fp8")
  for folder in "${folders[@]}"; do
    for accuracy in 0 1; do
      dir="$D4R_BUNDLE_TEX/$folder"; [[ "$accuracy" == 1 ]] && dir="$D4R_BUNDLE_TEX/accuracy/$folder"
      [[ -f "$dir/.complete" ]] && continue
      fp8=0; [[ "$folder" == *-fp8 ]] && fp8=1
      log="$TMP/tex-$folder-$accuracy.log"
      ( rm -rf "$dir" && mkdir -p "$dir" &&
        D4R_PREFER_ACCURACY="$accuracy" D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" \
          D4R_DLSS_DLL="$D4R_BUNDLE_DLSS" "$ROOT/kernels/build.sh" tex "$dir" > "$log" 2>&1 && touch "$dir/.complete" ) ||
        { echo "texture kernels for $dir failed:" >&2; tail -20 "$log" >&2; exit 1; } &
      pending+=("$!")
      if (( ${#pending[@]} >= JOBS )); then wait "${pending[0]}" || failed=1; pending=("${pending[@]:1}"); fi
    done
  done
done
for pid in "${pending[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo "texture kernel build failed; no release ZIP created" >&2; exit 1; }
D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCHS="$ARCHS" D4R_PACKAGE_KERNEL_JOBS="$JOBS" \
  "$ROOT/scripts/stage_native_kernels.sh" "$STAGE/d4r" installer

# licenses and sources
L="$STAGE/d4r/licenses"
cp "$ROOT/LICENSE" "$L/d4r-LICENSE.txt"
cp "$ROOT/NOTICE" "$L/d4r-NOTICE.txt"
cp "$ZLUDA_SRC/LICENSE-APACHE" "$L/ZLUDA-LICENSE-APACHE.txt"
cp "$ZLUDA_SRC/LICENSE-MIT" "$L/ZLUDA-LICENSE-MIT.txt"
cp "$ZLUDA_SRC/ext/llvm-project/llvm/LICENSE.TXT" "$L/LLVM-LICENSE.txt"
cp "$D4R_OPTISCALER_LICENSE" "$L/OptiScaler-LICENSE-GPL-3.0.txt"
mkdir -p "$STAGE/d4r/source/patches"
cp -r "$ROOT/patches/zluda" "$ROOT/patches/zluda-windows" "$ROOT/patches/optiscaler-windows" "$STAGE/d4r/source/patches/"
ZLUDA_COMMIT="$(git -C "$ZLUDA_SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
patches=("$ROOT"/patches/zluda/*.patch)
first="$(basename "${patches[0]}")" last="$(basename "${patches[-1]}")"
version_of() { command -v "$1" >/dev/null || return 0; "$@" 2>&1 | head -1 | tr -d '\r'; }
optiscaler() { "$PYTHON" -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$ROOT/packaging/optiscaler-release.json" "$1"; }
CRT_VERSION="$(basename "$(dirname "$(dirname "$(newest "/c/Program Files/Microsoft Visual Studio/*/*/VC/Redist/MSVC/*/x64/Microsoft.VC14*.CRT")")")")"
sed -e "s/@VERSION@/$VERSION/g" -e "s/@D4R_COMMIT@/$(git -C "$ROOT" rev-parse HEAD)$(git -C "$ROOT" diff --quiet HEAD -- || echo ' with uncommitted changes')/g" \
  -e "s/@ZLUDA_COMMIT@/$ZLUDA_COMMIT/g" -e "s/@ZLUDA_PATCHES@/${first%%-*} to ${last%%-*}/g" \
  -e "s|@ROCM_VERSION@|$(basename "$ROCM")|g" \
  -e "s|@ROCM_CLANG@|$(version_of "$ROCM/lib/llvm/bin/clang++.exe" --version | sed 's/.*clang version \([^ ]*\).*/\1/')|g" \
  -e "s|@GLSLANG_VERSION@|$(version_of "$STAGE/d4r/tools/models/bin/glslangValidator.exe" --version | sed 's/.*:\([0-9.]*\)$/\1/')|g" \
  -e "s|@DXC_VERSION@|$(version_of "$STAGE/d4r/tools/models/bin/dxc.exe" --version | sed 's/.*- \([0-9.]*\) .*/\1/')|g" \
  -e "s|@VCRUNTIME_VERSION@|${D4R_VCRUNTIME_DIR:+from $D4R_VCRUNTIME_DIR}${D4R_VCRUNTIME_DIR:-$CRT_VERSION}|g" \
  -e "s|@OPTISCALER_VERSION@|$(optiscaler version)|g" -e "s|@OPTISCALER_TAG@|$(optiscaler tag)|g"   -e "s|@OPTISCALER_COMMIT@|$(optiscaler source_commit)|g" \
  -e "s|@OPTISCALER_RELEASE@|https://github.com/optiscaler/OptiScaler-nightly/releases/tag/$(optiscaler tag)|g" \
  "$ROOT/packaging/WINDOWS_SOURCES.txt" | sed 's/$/\r/' > "$STAGE/d4r/source/SOURCES.txt"
if [[ -n "${D4R_BUILD_INFO:-}" ]]; then
  cp "$D4R_BUILD_INFO" "$STAGE/d4r/source/BUILD_INFO.txt"
else
  # what this machine builds with (D4R_SKIP_BUILD=1 packages whatever build/ holds: pass D4R_BUILD_INFO for that)
  {
    version_of "${MINGW_CXX:-x86_64-w64-mingw32-g++}" --version
    version_of "${CLANG_CL:-clang-cl}" --version
    version_of rustc --version
    version_of cargo --version
    version_of "$ROCM/lib/llvm/bin/clang++.exe" --version
    version_of "$PYTHON" --version
    for msvc in "/c/Program Files/Microsoft Visual Studio/2022/"*/VC/Tools/MSVC/*; do
      [[ -d "$msvc" ]] && echo "MSVC $(basename "$msvc") (Visual Studio 2022, OptiScaler)"
    done | sort -V | tail -1
  } | sed 's/$/\r/' > "$STAGE/d4r/source/BUILD_INFO.txt"
fi
sed -e "s/@VERSION@/$VERSION/g" -e 's/$/\r/' "$ROOT/packaging/WINDOWS_README.txt" > "$STAGE/D4R_README.txt"

# Last line of defence for the published zip: what the installer variant may contain, never values read from DLSS.
"$PYTHON" "$ROOT/scripts/check_redistributable.py" release installer "$STAGE" ||
  { echo "refusing to build $NAME.zip" >&2; exit 2; }

# One timestamp for every file (SOURCE_DATE_EPOCH, default the last commit); the zip holds the folder's contents, so
# it can be extracted straight into the game folder.
EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" log -1 --format=%ct)}"
rm -f "$OUT/$NAME.zip"
"$PYTHON" - "$STAGE" "$OUT/$NAME.zip" "$EPOCH" <<'PY'
import sys, time, zipfile
from pathlib import Path
stage, out, epoch = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
stamp = time.gmtime(max(epoch, 315532800))[:6]
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in sorted(p for p in stage.rglob('*') if p.is_file()):
        info = zipfile.ZipInfo(f.relative_to(stage).as_posix(), stamp)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, f.read_bytes())
PY
(cd "$OUT" && sha256sum "$NAME.zip" > "$NAME.zip.sha256")
printf 'Built %s (%s)\n' "$OUT/$NAME.zip" "$(du -h "$OUT/$NAME.zip" | cut -f1)"
