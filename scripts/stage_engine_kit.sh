#!/usr/bin/env bash
# Puts the engine-model builder into a release's d4r folder: d4r/tools/build-engine-models.sh (install.sh runs it)
# and d4r/tools/models/ with d4r's model compilers and shader sources, the D3D12 test app and glslangValidator.
# Only d4r's own code and Khronos' glslang go in; the models are built on the user's machine from their DLLs.
#
# usage: scripts/stage_engine_kit.sh D4R_FOLDER
#   D4R_GLSLANG      glslang release archive or extracted folder (bin/glslangValidator; Khronos 16.6.0 tested)
#   D4R_GLSLANG_LICENSE  its LICENSE.txt (the release archives have none; the tag's LICENSE.txt in KhronosGroup/glslang)
#   D4R_HARNESS_EXE  the D3D12 test app (default build/d3d12_dlss_harness.exe, scripts/build_d3d12_dlss_harness.sh)
#   D4R_KIT_PLATFORM linux (default), or windows: the native Windows builder instead (d4r/tools/build-engine-models.py,
#                    which install.ps1 runs), the Vulkan test app (default build/vulkan_dlss_harness.exe),
#                    glslangValidator.exe, and dxc for the D3D12 shaders every Windows model gets:
#   D4R_DXC          (windows) DirectX Shader Compiler release archive or extracted folder (bin/x64/dxc.exe,
#                    dxcompiler.dll, dxil.dll and its LICENSE-*.txt; 1.9.2609 tested)
#   D4R_VCRUNTIME_DIR  (windows) Visual Studio's x64 Microsoft.VC14*.CRT redistributable folder: glslangValidator.exe
#                    needs its C++ runtime, put beside it (app-local, as Visual Studio's redistribution terms allow)
#                    default: the newest one in a Visual Studio installation
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D4R="$(realpath "${1:?usage: stage_engine_kit.sh D4R_FOLDER}")"
PLATFORM="${D4R_KIT_PLATFORM:-linux}"
case "$PLATFORM" in
  linux) HARNESS="${D4R_HARNESS_EXE:-$ROOT/build/d3d12_dlss_harness.exe}" ;;
  windows) HARNESS="${D4R_HARNESS_EXE:-$ROOT/build/vulkan_dlss_harness.exe}"
           : "${D4R_DXC:?set D4R_DXC to a DirectX Shader Compiler release (dxc_2026_09_29.zip)}" ;;
  *) echo "D4R_KIT_PLATFORM must be linux or windows" >&2; exit 2 ;;
esac
: "${D4R_GLSLANG:?set D4R_GLSLANG to a glslang release (glslang-16.6.0-linux-x86_64-release.tar.gz)}"
: "${D4R_GLSLANG_LICENSE:?set D4R_GLSLANG_LICENSE to the LICENSE.txt of glslang}"
[[ -f "$HARNESS" ]] || { echo "missing $HARNESS (scripts/build_d3d12_dlss_harness.sh, build_vulkan_dlss_harness.sh)" >&2; exit 2; }
KIT="$D4R/tools/models"
rm -rf "$KIT"
mkdir -p "$KIT/bin" "$KIT/engine/rr" "$KIT/engine/sr" "$KIT/engine/cnn_sr" "$KIT/kernels/tools" "$KIT/kernels/rr" "$KIT/vulkan" "$D4R/licenses"
copy() {   # copy SRC_DIR DST_DIR PATTERN...: never tests
  local src=$1 dst=$2 pat f
  shift 2
  for pat in "$@"; do
    for f in "$ROOT/$src"/$pat; do
      [[ -f "$f" ]] || continue
      case "$(basename "$f")" in test_*) continue ;; esac
      cp "$f" "$dst/"
    done
  done
}
copy engine "$KIT/engine" '*.py' '*.comp' '*.glsl'
copy engine/rr "$KIT/engine/rr" '*.py' '*.comp' '*.glsl' '*.txt' '*.npy' '*.sh'
copy engine/cnn_sr "$KIT/engine/cnn_sr" '*.glsl'
copy engine/sr "$KIT/engine/sr" '*.comp' '*.glsl'
copy kernels/tools "$KIT/kernels/tools" '*.py'
copy kernels/rr "$KIT/kernels/rr" '*.py'
copy vulkan "$KIT/vulkan" '*.py'   # ptx2glsl.py: engine/rr/ptx2rr.py builds on it
# the model compilers import these test modules' packing helpers
cp "$ROOT/engine/test_k.py" "$ROOT/engine/test_m.py" "$KIT/engine/"
cp "$ROOT/engine/rr/test_layer.py" "$KIT/engine/rr/"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
GL="$D4R_GLSLANG"
if [[ -f "$GL" ]]; then
  mkdir -p "$TMP/glslang"
  case "$GL" in *.zip) unzip -q "$GL" -d "$TMP/glslang" ;; *) tar xf "$GL" -C "$TMP/glslang" ;; esac
  GL="$TMP/glslang"
fi
cp "$D4R_GLSLANG_LICENSE" "$D4R/licenses/glslang-LICENSE.txt"
cp "$ROOT/packaging/model_tools.py" "$KIT/model_tools.py"
if [[ "$PLATFORM" == windows ]]; then
  # the D3D12 shaders' sources: d4r's HLSL, its headers and the HLSL builders (hlsl_build.py includes engine/
  # and engine/d3d12)
  mkdir -p "$KIT/engine/d3d12"
  copy engine "$KIT/engine" '*.h'
  copy engine/d3d12 "$KIT/engine/d3d12" '*.py' '*.hlsl' '*.hlsli' '*.h' '*.comp'
  cp "$HARNESS" "$KIT/bin/vulkan_dlss_harness.exe"
  cp "$GL/bin/glslangValidator.exe" "$KIT/bin/glslangValidator.exe"
  # the newest runtime (by its MSVC version): newer runtimes serve programs built with older toolsets
  CRT="${D4R_VCRUNTIME_DIR:-$(ls -d "/c/Program Files/Microsoft Visual Studio"/*/*/VC/Redist/MSVC/*/x64/Microsoft.VC14*.CRT 2>/dev/null |
    awk -F/ '{ print $(NF-2) "\t" $0 }' | sort -V | tail -1 | cut -f2)}"
  [[ -f "$CRT/vcruntime140.dll" ]] || { echo "no Visual C++ runtime folder (set D4R_VCRUNTIME_DIR)" >&2; exit 2; }
  for f in msvcp140.dll vcruntime140.dll vcruntime140_1.dll; do cp "$CRT/$f" "$KIT/bin/$f"; done
  DXC="$D4R_DXC"
  if [[ -f "$DXC" ]]; then
    # Microsoft's archive names its entries with backslashes, which unzip only warns about (exit status 1)
    "$(command -v python3 || command -v python)" -c '
import sys, zipfile
from pathlib import Path
with zipfile.ZipFile(sys.argv[1]) as z:
    for i in z.infolist():
        if not i.is_dir():
            p = Path(sys.argv[2], *i.filename.replace("\\", "/").split("/"))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(i))' "$DXC" "$TMP/dxc"
    DXC="$TMP/dxc"
  fi
  for f in dxc.exe dxcompiler.dll dxil.dll; do cp "$DXC/bin/x64/$f" "$KIT/bin/$f"; done
  for f in LICENSE-LLVM.txt LICENSE-MS.txt LICENCE-MIT.txt; do cp "$DXC/$f" "$D4R/licenses/dxc-$f"; done
  cp "$ROOT/packaging/build_engine_models.py" "$D4R/tools/build-engine-models.py"
else
  cp "$HARNESS" "$KIT/bin/d3d12_dlss_harness.exe"
  cp "$GL/bin/glslangValidator" "$KIT/bin/glslangValidator"
  chmod +x "$KIT/bin/glslangValidator"
  cp "$ROOT/packaging/build_engine_models.sh" "$D4R/tools/build-engine-models.sh"
  chmod +x "$D4R/tools/build-engine-models.sh"
fi
printf 'Staged the engine-model builder in %s (%s)\n' "$D4R/tools" "$(du -sh "$D4R/tools" | cut -f1)"
